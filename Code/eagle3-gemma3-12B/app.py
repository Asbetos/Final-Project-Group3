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
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict

_APP_DIR = Path(__file__).resolve().parent
# Use the default HF cache at ~/.cache/huggingface/ where models are pre-downloaded.
# Set HF_HOME only if not already set by the user.
_HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"

import torch
import streamlit as st

from baseline import autoregressive_decode
from config import EAGLE3_PAIR_I, PAIR_F
from data import format_prompt_for_chat
from eagle3 import Eagle3Config, Eagle3DraftHead, eagle3_decode
from eagle3_train import load_checkpoint
from models import get_device, load_model, load_tokenizer
from speculative import speculative_decode


# ── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Inference Acceleration — Speculative Decoding Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global overrides */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2rem;
    margin: 0 0 0.3rem 0;
    animation: gradient-shift 4s ease infinite;
}
@keyframes gradient-shift {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.main-header p {
    color: #a0aec0;
    font-size: 0.95rem;
    margin: 0;
}

/* Column headers */
.col-header {
    background: linear-gradient(135deg, rgba(15,12,41,0.6), rgba(36,36,62,0.6));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.col-header h3 {
    margin: 0;
    font-weight: 600;
    font-size: 1.1rem;
}
.col-header-baseline h3 { color: #a0aec0; }
.col-header-accel h3 { color: #00d2ff; }
.col-header p {
    color: #718096;
    font-size: 0.8rem;
    margin: 0.3rem 0 0 0;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.metric-card {
    background: rgba(15,12,41,0.5);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    flex: 1;
    min-width: 100px;
    backdrop-filter: blur(10px);
}
.metric-card .label {
    color: #718096;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
}
.metric-card .value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
}
.metric-card .delta {
    font-size: 0.75rem;
    margin-top: 0.1rem;
}
.delta-positive { color: #48bb78; }
.delta-negative { color: #fc8181; }

/* Speedup badge */
.speedup-badge {
    background: linear-gradient(135deg, #00d2ff, #3a7bd5);
    color: white;
    font-weight: 700;
    font-size: 1.4rem;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    display: inline-block;
    box-shadow: 0 4px 15px rgba(0,210,255,0.3);
    animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 4px 15px rgba(0,210,255,0.3); }
    50% { box-shadow: 0 4px 25px rgba(0,210,255,0.5); }
}

/* GPU status */
.gpu-status {
    background: rgba(15,12,41,0.4);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem;
    margin-top: 1rem;
}
.gpu-status .label {
    color: #718096;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.gpu-bar {
    background: rgba(255,255,255,0.1);
    border-radius: 6px;
    height: 8px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.gpu-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #48bb78, #38b2ac);
    transition: width 0.5s ease;
}

/* Method selector pills */
div[data-testid="stRadio"] > label {
    font-weight: 500;
}

/* About section */
.about-card {
    background: linear-gradient(135deg, rgba(15,12,41,0.6), rgba(36,36,62,0.6));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.about-card h4 {
    color: #00d2ff;
    margin-top: 0;
}
.about-card p, .about-card li {
    color: #a0aec0;
    font-size: 0.85rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Method definitions ───────────────────────────────────────────────────

METHODS = {
    "pair_f": {
        "label": "Standard Speculative Decoding (Pair F)",
        "short": "Speculative (Pair F)",
        "description": "Gemma-3-12B target + Gemma-3-1B draft model",
        "icon": "🔀",
    },
    "eagle3_i": {
        "label": "EAGLE-3 Draft Head (Pair I)",
        "short": "EAGLE-3",
        "description": "Gemma-3-12B target + trained EAGLE-3 draft head",
        "icon": "🦅",
    },
}


# ── Model loading ────────────────────────────────────────────────────────

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


# ── Helper functions ─────────────────────────────────────────────────────

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


def run_baseline(runtime, inputs, temperature, max_new_tokens, seed):
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


def run_accelerated(runtime, method_key, inputs, temperature, max_new_tokens, seed, gamma, tree_budget):
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


# ── Metric rendering ────────────────────────────────────────────────────

def render_baseline_metrics_html(metrics) -> str:
    return f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Tokens/sec</div>
            <div class="value">{metrics.tokens_per_second:.1f}</div>
        </div>
        <div class="metric-card">
            <div class="label">Wall Clock</div>
            <div class="value">{metrics.wall_clock_ms:.0f} ms</div>
        </div>
        <div class="metric-card">
            <div class="label">TTFT</div>
            <div class="value">{metrics.ttft_ms:.1f} ms</div>
        </div>
        <div class="metric-card">
            <div class="label">Tokens</div>
            <div class="value">{metrics.total_tokens_generated}</div>
        </div>
    </div>
    """


def render_accelerated_metrics_html(metrics, baseline_metrics, method_label: str) -> str:
    speedup = (
        metrics.tokens_per_second / baseline_metrics.tokens_per_second
        if baseline_metrics.tokens_per_second > 0
        else 0.0
    )
    tps_delta = metrics.tokens_per_second - baseline_metrics.tokens_per_second
    wall_delta = metrics.wall_clock_ms - baseline_metrics.wall_clock_ms
    tps_delta_class = "delta-positive" if tps_delta > 0 else "delta-negative"
    wall_delta_class = "delta-positive" if wall_delta < 0 else "delta-negative"

    return f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Tokens/sec</div>
            <div class="value">{metrics.tokens_per_second:.1f}</div>
            <div class="delta {tps_delta_class}">{tps_delta:+.1f} vs baseline</div>
        </div>
        <div class="metric-card">
            <div class="label">Wall Clock</div>
            <div class="value">{metrics.wall_clock_ms:.0f} ms</div>
            <div class="delta {wall_delta_class}">{wall_delta:+.0f} ms</div>
        </div>
        <div class="metric-card">
            <div class="label">Speedup</div>
            <div class="value" style="color: {'#48bb78' if speedup > 1.0 else '#fc8181'}">{speedup:.2f}×</div>
        </div>
    </div>
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Acceptance Rate</div>
            <div class="value">{metrics.acceptance_rate:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="label">Accepted / Round</div>
            <div class="value">{metrics.acceptance_length:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="label">TTFT</div>
            <div class="value">{metrics.ttft_ms:.1f} ms</div>
        </div>
    </div>
    """


def render_speedup_badge(speedup: float) -> str:
    if speedup > 1.0:
        return f'<div class="speedup-badge">⚡ {speedup:.2f}× Faster</div>'
    else:
        return f'<div class="speedup-badge" style="background: linear-gradient(135deg, #fc8181, #e53e3e); box-shadow: 0 4px 15px rgba(252,129,129,0.3);">{speedup:.2f}×</div>'


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    selected_method = st.radio(
        "Acceleration Method",
        options=list(METHODS.keys()),
        format_func=lambda key: f"{METHODS[key]['icon']} {METHODS[key]['label']}",
    )

    st.markdown("---")

    gamma = 5
    tree_budget = 20
    if selected_method == "pair_f":
        gamma = st.slider("Speculation Length (γ)", min_value=1, max_value=10, value=5, step=1,
                          help="Number of draft tokens generated per round")
    else:
        tree_budget = st.select_slider("EAGLE Tree Budget", options=[1, 20, 60], value=20,
                                       help="Number of candidate tokens in the draft tree")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1,
                            help="0.0 = greedy (deterministic), higher = more random")
    max_new_tokens = st.slider("Max New Tokens", min_value=32, max_value=512, value=128, step=32)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reload Models", use_container_width=True):
            clear_runtime_bundle()
            st.session_state.messages = []
            st.rerun()

    # GPU status
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        pct = min(vram_used / vram_total, 1.0)
        color = "#48bb78" if pct < 0.7 else ("#ecc94b" if pct < 0.9 else "#fc8181")
        st.markdown(f"""
        <div class="gpu-status">
            <div class="label">🖥️ {torch.cuda.get_device_name(0)}</div>
            <div style="color: #a0aec0; font-size: 0.8rem; margin-top: 0.3rem;">
                VRAM: {vram_used:.1f} / {vram_total:.1f} GB ({pct:.0%})
            </div>
            <div class="gpu-bar">
                <div class="gpu-bar-fill" style="width: {pct*100:.0f}%; background: {color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ CUDA GPU required")

    # About section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **LLM Inference Acceleration** using Speculative Decoding.

        This app demonstrates two acceleration methods:

        - **Pair F**: Standard speculative decoding with a small draft model
          (Gemma-3-1B) that proposes tokens verified by the large target
          (Gemma-3-12B).

        - **EAGLE-3**: A lightweight trained draft *head* attached to the
          target model's hidden states, using tree-structured candidate
          verification for higher throughput.

        Both methods produce outputs **identical in distribution** to the
        target model — they accelerate generation without quality loss.
        """)


# ── GPU gate ─────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    st.error("🚫 This app requires a CUDA GPU with enough VRAM to hold Gemma-3-12B.")
    st.stop()


# ── State setup ──────────────────────────────────────────────────────────

ensure_state(selected_method)


# ── Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>⚡ Speculative Decoding Live Demo</h1>
    <p>Compare baseline autoregressive decoding vs accelerated methods on Gemma-3-12B — same model, same prompt, real-time speed comparison.</p>
</div>
""", unsafe_allow_html=True)


# ── Load models ──────────────────────────────────────────────────────────

with st.spinner("🔧 Loading models (target + draft + EAGLE-3 head)... This takes ~2 minutes on first run."):
    runtime = load_runtime_bundle()

target_model = runtime["target_model"]
tokenizer = runtime["tokenizer"]
model_device = next(target_model.parameters()).device


# ── Column headers ───────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div class="col-header col-header-baseline">
        <h3>🐢 Baseline (Autoregressive)</h3>
        <p>Standard token-by-token generation — Gemma-3-12B (4-bit)</p>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    method_info = METHODS[selected_method]
    st.markdown(f"""
    <div class="col-header col-header-accel">
        <h3>{method_info['icon']} {method_info['short']}</h3>
        <p>{method_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Render chat history ──────────────────────────────────────────────────

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
            st.markdown("---")
            st.markdown(render_baseline_metrics_html(baseline["metrics"]), unsafe_allow_html=True)

    with col_right:
        with st.chat_message("assistant"):
            st.markdown(accelerated["text"])
            st.markdown("---")
            speedup = (
                accelerated["metrics"].tokens_per_second / baseline["metrics"].tokens_per_second
                if baseline["metrics"].tokens_per_second > 0 else 0.0
            )
            st.markdown(render_speedup_badge(speedup), unsafe_allow_html=True)
            st.markdown(
                render_accelerated_metrics_html(
                    accelerated["metrics"], baseline["metrics"], accelerated["method_label"]
                ),
                unsafe_allow_html=True,
            )


# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Type a prompt to compare both methods...")

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

    # ── Baseline generation ──────────────────────────────────────────
    with col_left:
        with st.chat_message("assistant"):
            with st.spinner("🐢 Generating baseline response..."):
                baseline_result = run_baseline(runtime, shared_inputs, temperature, max_new_tokens, seed)
                baseline_metrics = baseline_result["metrics"]
            st.markdown(baseline_result["output_text"])
            st.markdown("---")
            st.markdown(render_baseline_metrics_html(baseline_metrics), unsafe_allow_html=True)

    # ── Accelerated generation ───────────────────────────────────────
    with col_right:
        with st.chat_message("assistant"):
            with st.spinner(f"⚡ Generating {METHODS[selected_method]['short']} response..."):
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
            st.markdown("---")

            speedup = (
                accelerated_metrics.tokens_per_second / baseline_metrics.tokens_per_second
                if baseline_metrics.tokens_per_second > 0 else 0.0
            )
            st.markdown(render_speedup_badge(speedup), unsafe_allow_html=True)
            st.markdown(
                render_accelerated_metrics_html(
                    accelerated_metrics, baseline_metrics, METHODS[selected_method]["label"]
                ),
                unsafe_allow_html=True,
            )

    if model_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ── Save to history ──────────────────────────────────────────────
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
