"""
Streamlit demo: side-by-side comparison of baseline autoregressive decoding
vs speculative decoding on Qwen3 models.

Run with:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import gc

import torch
import streamlit as st

from baseline import autoregressive_decode
from config import PAIR_MAP
from data import format_prompt_for_chat
from models import load_model_pair
from speculative import speculative_decode

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Speculative Decoding Demo",
    page_icon=":zap:",
    layout="wide",
)

# ── Pair descriptions for the sidebar ────────────────────────────────────

PAIR_LABELS = {
    "A": "Pair A  —  Qwen3-8B (fp16) + Qwen3-0.6B  (~20.6 GB)",
    "B": "Pair B  —  Qwen3-8B (fp16) + Qwen3-1.7B  (~22.8 GB)",
    "C": "Pair C  —  Qwen3-8B (4-bit) + Qwen3-0.6B  (~8.2 GB)",
}

# ── Model caching ────────────────────────────────────────────────────────


@st.cache_resource
def load_models(pair_id: str):
    """Load and cache a model pair. Keyed by pair_id so switching pairs
    triggers a fresh load."""
    pair_config = PAIR_MAP[pair_id]
    target_model, draft_model, tokenizer = load_model_pair(pair_config)
    return target_model, draft_model, tokenizer


def ensure_models(pair_id: str):
    """Handle pair switching: clear old models before loading new ones."""
    prev = st.session_state.get("loaded_pair_id")
    if prev != pair_id:
        if prev is not None:
            load_models.clear()
            gc.collect()
            torch.cuda.empty_cache()
        st.session_state.loaded_pair_id = pair_id
        st.session_state.messages = []

    return load_models(pair_id)


# ── Session state defaults ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    pair_id = st.selectbox(
        "Model Pair",
        options=list(PAIR_LABELS.keys()),
        format_func=lambda k: PAIR_LABELS[k],
        index=0,
    )

    gamma = st.slider(
        "Speculation Length (gamma)",
        min_value=1, max_value=10, value=5, step=1,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.5, value=0.0, step=0.1,
    )
    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=32, max_value=512, value=128, step=32,
    )

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=80,
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # GPU status
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        st.progress(
            vram_used / vram_total,
            text=f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB",
        )
    else:
        st.error("No CUDA GPU detected")

# ── Load models ──────────────────────────────────────────────────────────

with st.spinner(f"Loading model pair {pair_id} — this may take a minute on first run..."):
    target_model, draft_model, tokenizer = ensure_models(pair_id)

# ── Title ────────────────────────────────────────────────────────────────

st.title("Speculative Decoding vs Baseline")
st.caption(
    "Type a message below. Both decoding methods run on the **same prompt** "
    "with the **same target model** — compare the speed in real time."
)

# ── Two-column chat windows ─────────────────────────────────────────────

col_bl, col_sp = st.columns(2)

with col_bl:
    st.markdown("#### Baseline (Autoregressive)")
with col_sp:
    st.markdown("#### Speculative Decoding")

# ── Render chat history ──────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with col_bl:
            with st.chat_message("user"):
                st.write(msg["content"])
        with col_sp:
            with st.chat_message("user"):
                st.write(msg["content"])
    else:
        bl = msg["baseline"]
        sp = msg["speculative"]

        with col_bl:
            with st.chat_message("assistant"):
                st.markdown(bl["text"])
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Tokens/sec", f"{bl['tps']:.1f}")
                c2.metric("Wall Clock", f"{bl['wall_ms']:.0f} ms")
                c3.metric("TTFT", f"{bl['ttft_ms']:.1f} ms")

        with col_sp:
            with st.chat_message("assistant"):
                st.markdown(sp["text"])
                st.divider()
                speedup = sp["tps"] / bl["tps"] if bl["tps"] > 0 else 0
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Tokens/sec",
                    f"{sp['tps']:.1f}",
                    delta=f"{sp['tps'] - bl['tps']:+.1f}",
                )
                c2.metric(
                    "Wall Clock",
                    f"{sp['wall_ms']:.0f} ms",
                    delta=f"{sp['wall_ms'] - bl['wall_ms']:+.0f} ms",
                    delta_color="inverse",
                )
                c3.metric("Speedup", f"{speedup:.2f}x")
                c4, c5 = st.columns(2)
                c4.metric("Acceptance Rate", f"{sp['acceptance_rate']:.1%}")
                c5.metric("Tokens Accepted/Round", f"{sp['acceptance_length']:.2f}")

# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Type a message...")

if user_input:
    # Show user message in both columns
    st.session_state.messages.append({"role": "user", "content": user_input})

    with col_bl:
        with st.chat_message("user"):
            st.write(user_input)
    with col_sp:
        with st.chat_message("user"):
            st.write(user_input)

    # Tokenize the prompt once, clone for each run
    formatted = format_prompt_for_chat(user_input, system_prompt, tokenizer)
    encoded = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=2048,
    )
    # Use the device the model's parameters are actually on
    model_device = next(target_model.parameters()).device
    input_ids = encoded["input_ids"].to(model_device)
    attention_mask = encoded["attention_mask"].to(model_device)

    seed = 42

    # ── Run baseline (left column) ───────────────────────────────────
    if model_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    with col_bl:
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                gen_bl = torch.Generator(device=model_device)
                gen_bl.manual_seed(seed)

                bl_result = autoregressive_decode(
                    model=target_model,
                    input_ids=input_ids.clone(),
                    attention_mask=attention_mask.clone(),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    generator=gen_bl,
                )
                bl_m = bl_result["metrics"]

            st.markdown(bl_result["output_text"])
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Tokens/sec", f"{bl_m.tokens_per_second:.1f}")
            c2.metric("Wall Clock", f"{bl_m.wall_clock_ms:.0f} ms")
            c3.metric("TTFT", f"{bl_m.ttft_ms:.1f} ms")

    # ── Run speculative (right column) ───────────────────────────────
    if model_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    with col_sp:
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                gen_sp = torch.Generator(device=model_device)
                gen_sp.manual_seed(seed)

                sp_result = speculative_decode(
                    target_model=target_model,
                    draft_model=draft_model,
                    input_ids=input_ids.clone(),
                    attention_mask=attention_mask.clone(),
                    gamma=gamma,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    generator=gen_sp,
                )
                sp_m = sp_result["metrics"]

            st.markdown(sp_result["output_text"])
            st.divider()
            speedup = (
                sp_m.tokens_per_second / bl_m.tokens_per_second
                if bl_m.tokens_per_second > 0
                else 0
            )
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Tokens/sec",
                f"{sp_m.tokens_per_second:.1f}",
                delta=f"{sp_m.tokens_per_second - bl_m.tokens_per_second:+.1f}",
            )
            c2.metric(
                "Wall Clock",
                f"{sp_m.wall_clock_ms:.0f} ms",
                delta=f"{sp_m.wall_clock_ms - bl_m.wall_clock_ms:+.0f} ms",
                delta_color="inverse",
            )
            c3.metric("Speedup", f"{speedup:.2f}x")
            c4, c5 = st.columns(2)
            c4.metric("Acceptance Rate", f"{sp_m.acceptance_rate:.1%}")
            c5.metric("Tokens Accepted/Round", f"{sp_m.acceptance_length:.2f}")

    if model_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ── Save to history ──────────────────────────────────────────────
    st.session_state.messages.append(
        {
            "role": "assistant",
            "baseline": {
                "text": bl_result["output_text"],
                "tps": bl_m.tokens_per_second,
                "wall_ms": bl_m.wall_clock_ms,
                "ttft_ms": bl_m.ttft_ms,
                "total_tokens": bl_m.total_tokens_generated,
            },
            "speculative": {
                "text": sp_result["output_text"],
                "tps": sp_m.tokens_per_second,
                "wall_ms": sp_m.wall_clock_ms,
                "ttft_ms": sp_m.ttft_ms,
                "total_tokens": sp_m.total_tokens_generated,
                "acceptance_rate": sp_m.acceptance_rate,
                "acceptance_length": sp_m.acceptance_length,
            },
        }
    )
