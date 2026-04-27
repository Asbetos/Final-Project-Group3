"""Unified side-by-side chat app for baseline vs acceleration methods.

Run with:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0

This app uses the shared `eagle3-gemma3-12B` codebase so it can compare:
1. Baseline autoregressive decoding on `google/gemma-3-12b-it`
2. One selected acceleration method on the same prompt:
   - Standard speculative decoding with pair `F`
   - EAGLE-3 draft-head decoding with pair `I`

Models are cached in two ways:
1. Hugging Face weights are stored on local disk under `.hf-cache/`
2. Loaded model objects are kept warm in Streamlit process memory via `st.cache_resource`
"""

from __future__ import annotations

import gc
import os
from dataclasses import replace
from pathlib import Path
from typing import Dict

_APP_DIR = Path(__file__).resolve().parent
_HF_CACHE_DIR = _APP_DIR / ".hf-cache"
os.environ.setdefault("HF_HOME", str(_HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE_DIR / "transformers"))

import torch
import streamlit as st

from baseline import autoregressive_decode
from config import EAGLE3_PAIR_I, PAIR_F
from data import format_prompt_for_chat
from eagle3 import Eagle3Config, Eagle3DraftHead, eagle3_decode
from eagle3_train import load_checkpoint
from models import get_device, load_model, load_tokenizer
from speculative import speculative_decode


st.set_page_config(
    page_title="Gemma Speedup Chat Comparison",
    page_icon=":zap:",
    layout="wide",
)


METHODS = {
    "pair_f": {
        "label": "Standard speculative decoding (Pair F)",
        "short": "Pair F",
        "description": "Gemma-3-12B target + Gemma-3-1B draft model",
    },
    "eagle3_i": {
        "label": "EAGLE-3 draft head (Pair I)",
        "short": "EAGLE-3",
        "description": "Gemma-3-12B target + trained EAGLE-3 draft head",
    },
}


@st.cache_resource(show_spinner=False)
def load_runtime_bundle() -> Dict[str, object]:
    """Load the shared target model plus both acceleration resources once.

    This keeps a single warm target model in memory, avoiding duplicate 12B
    loads when users switch between pair `F` and EAGLE-3.
    """
    tokenizer = load_tokenizer(PAIR_F.target_model_id)
    target_model = load_model(
        PAIR_F.target_model_id,
        quantize_4bit=PAIR_F.target_quantize_4bit,
        compile_model=False,
    )
    draft_model = load_model(
        PAIR_F.draft_model_id,
        quantize_4bit=PAIR_F.draft_quantize_4bit,
        compile_model=False,
    )

    eagle3_base_config = Eagle3Config.from_model(
        target_model,
        tree_budget=EAGLE3_PAIR_I.tree_budget,
        max_depth=EAGLE3_PAIR_I.max_depth,
        top_k=EAGLE3_PAIR_I.top_k,
    )
    draft_head = Eagle3DraftHead(eagle3_base_config, target_model)
    device = get_device()
    draft_head = draft_head.to(device=device, dtype=torch.bfloat16)
    load_checkpoint(draft_head, EAGLE3_PAIR_I.checkpoint_path)
    draft_head.eval()

    return {
        "target_model": target_model,
        "tokenizer": tokenizer,
        "draft_model": draft_model,
        "draft_head": draft_head,
        "eagle3_base_config": eagle3_base_config,
    }


def clear_runtime_bundle() -> None:
    load_runtime_bundle.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def ensure_state(selected_method: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    prev = st.session_state.get("selected_method")
    if prev is not None and prev != selected_method:
        st.session_state.messages = []
    st.session_state.selected_method = selected_method


def build_inputs(user_prompt: str, system_prompt: str, tokenizer, model_device: torch.device):
    formatted = format_prompt_for_chat(user_prompt, system_prompt, tokenizer)
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    return {
        "input_ids": encoded["input_ids"].to(model_device),
        "attention_mask": encoded["attention_mask"].to(model_device),
    }


def build_eagle_chat_config(base_config: Eagle3Config, tree_budget: int) -> Eagle3Config:
    return replace(
        base_config,
        tree_budget=tree_budget,
        max_depth=EAGLE3_PAIR_I.max_depth,
        top_k=EAGLE3_PAIR_I.top_k,
    )


def run_baseline(runtime: Dict[str, object], inputs: Dict[str, torch.Tensor], temperature: float, max_new_tokens: int, seed: int):
    model_device = next(runtime["target_model"].parameters()).device
    generator = torch.Generator(device=model_device)
    generator.manual_seed(seed)
    return autoregressive_decode(
        model=runtime["target_model"],
        input_ids=inputs["input_ids"].clone(),
        attention_mask=inputs["attention_mask"].clone(),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        tokenizer=runtime["tokenizer"],
        generator=generator,
    )


def run_accelerated(
    runtime: Dict[str, object],
    method_key: str,
    inputs: Dict[str, torch.Tensor],
    temperature: float,
    max_new_tokens: int,
    seed: int,
    gamma: int,
    tree_budget: int,
):
    model_device = next(runtime["target_model"].parameters()).device
    generator = torch.Generator(device=model_device)
    generator.manual_seed(seed)

    if method_key == "pair_f":
        return speculative_decode(
            target_model=runtime["target_model"],
            draft_model=runtime["draft_model"],
            input_ids=inputs["input_ids"].clone(),
            attention_mask=inputs["attention_mask"].clone(),
            gamma=gamma,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tokenizer=runtime["tokenizer"],
            generator=generator,
        )

    eagle_cfg = build_eagle_chat_config(runtime["eagle3_base_config"], tree_budget)
    return eagle3_decode(
        target_model=runtime["target_model"],
        draft_head=runtime["draft_head"],
        eagle3_config=eagle_cfg,
        input_ids=inputs["input_ids"].clone(),
        attention_mask=inputs["attention_mask"].clone(),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        tokenizer=runtime["tokenizer"],
        generator=generator,
    )


def render_baseline_metrics(metrics) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tokens/sec", f"{metrics.tokens_per_second:.1f}")
    c2.metric("Wall Clock", f"{metrics.wall_clock_ms:.0f} ms")
    c3.metric("TTFT", f"{metrics.ttft_ms:.1f} ms")
    c4.metric("Tokens", f"{metrics.total_tokens_generated}")


def render_accelerated_metrics(metrics, baseline_metrics, method_label: str) -> None:
    speedup = (
        metrics.tokens_per_second / baseline_metrics.tokens_per_second
        if baseline_metrics.tokens_per_second > 0
        else 0.0
    )
    st.caption(method_label)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Tokens/sec",
        f"{metrics.tokens_per_second:.1f}",
        delta=f"{metrics.tokens_per_second - baseline_metrics.tokens_per_second:+.1f}",
    )
    c2.metric(
        "Wall Clock",
        f"{metrics.wall_clock_ms:.0f} ms",
        delta=f"{metrics.wall_clock_ms - baseline_metrics.wall_clock_ms:+.0f} ms",
        delta_color="inverse",
    )
    c3.metric("TTFT", f"{metrics.ttft_ms:.1f} ms")
    c4.metric("Speedup", f"{speedup:.2f}x")
    c5, c6 = st.columns(2)
    c5.metric("Acceptance Rate", f"{metrics.acceptance_rate:.1%}")
    c6.metric("Accepted / Round", f"{metrics.acceptance_length:.2f}")


with st.sidebar:
    st.header("Comparison Settings")

    selected_method = st.radio(
        "Right-hand chat method",
        options=list(METHODS.keys()),
        format_func=lambda key: METHODS[key]["label"],
    )

    gamma = 5
    tree_budget = 20
    if selected_method == "pair_f":
        gamma = st.slider("Speculation Length (gamma)", min_value=1, max_value=10, value=5, step=1)
    else:
        tree_budget = st.select_slider("EAGLE Tree Budget", options=[1, 20, 60], value=20)

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    max_new_tokens = st.slider("Max New Tokens", min_value=32, max_value=512, value=128, step=32)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Cached Models"):
        clear_runtime_bundle()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Local model cache: `{_HF_CACHE_DIR}`")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        st.progress(min(vram_used / vram_total, 1.0), text=f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB")
    else:
        st.error("CUDA GPU required for this app")

    with st.expander("Deployment Note"):
        st.write(
            "This app is written in Streamlit, but truly fast warm-model deployment requires a "
            "persistent GPU process. Vercel serverless functions do not provide a persistent GPU or "
            "reliable in-memory model warm state, so Vercel is better used as a thin frontend or proxy "
            "rather than as the actual model host."
        )


if not torch.cuda.is_available():
    st.error("This comparison app requires a CUDA GPU with enough VRAM to hold Gemma-3-12B.")
    st.stop()


ensure_state(selected_method)

st.title("Baseline vs Accelerated Gemma Chat")
st.caption(
    "The same prompt is sent to both chats. The left side always runs baseline autoregressive decoding, "
    "and the right side runs the selected acceleration method on the same Gemma-3-12B target model."
)

with st.spinner("Preloading target model, draft model, and EAGLE draft head..."):
    runtime = load_runtime_bundle()

target_model = runtime["target_model"]
tokenizer = runtime["tokenizer"]
model_device = next(target_model.parameters()).device

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### Baseline Chat")
with col_right:
    st.markdown(f"#### {METHODS[selected_method]['short']} Chat")
    st.caption(METHODS[selected_method]["description"])

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with col_left:
            with st.chat_message("user"):
                st.write(msg["content"])
        with col_right:
            with st.chat_message("user"):
                st.write(msg["content"])
        continue

    baseline = msg["baseline"]
    accelerated = msg["accelerated"]

    with col_left:
        with st.chat_message("assistant"):
            st.markdown(baseline["text"])
            st.divider()
            render_baseline_metrics(baseline["metrics"])

    with col_right:
        with st.chat_message("assistant"):
            st.markdown(accelerated["text"])
            st.divider()
            render_accelerated_metrics(accelerated["metrics"], baseline["metrics"], accelerated["method_label"])


user_input = st.chat_input("Type a prompt to compare both chats...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with col_left:
        with st.chat_message("user"):
            st.write(user_input)
    with col_right:
        with st.chat_message("user"):
            st.write(user_input)

    shared_inputs = build_inputs(user_input, system_prompt, tokenizer, model_device)
    seed = 42

    if model_device.type == "cuda":
        torch.cuda.synchronize()

    with col_left:
        with st.chat_message("assistant"):
            with st.spinner("Generating baseline response..."):
                baseline_result = run_baseline(runtime, shared_inputs, temperature, max_new_tokens, seed)
                baseline_metrics = baseline_result["metrics"]
            st.markdown(baseline_result["output_text"])
            st.divider()
            render_baseline_metrics(baseline_metrics)

    with col_right:
        with st.chat_message("assistant"):
            with st.spinner(f"Generating {METHODS[selected_method]['short']} response..."):
                accelerated_result = run_accelerated(
                    runtime=runtime,
                    method_key=selected_method,
                    inputs=shared_inputs,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    gamma=gamma,
                    tree_budget=tree_budget,
                )
                accelerated_metrics = accelerated_result["metrics"]
            st.markdown(accelerated_result["output_text"])
            st.divider()
            render_accelerated_metrics(
                accelerated_metrics,
                baseline_metrics,
                METHODS[selected_method]["label"],
            )

    if model_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "baseline": {
                "text": baseline_result["output_text"],
                "metrics": baseline_metrics,
            },
            "accelerated": {
                "text": accelerated_result["output_text"],
                "metrics": accelerated_metrics,
                "method_key": selected_method,
                "method_label": METHODS[selected_method]["label"],
            },
        }
    )
