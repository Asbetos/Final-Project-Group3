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
        st.session_state.chat_history = []

    return load_models(pair_id)


# ── Session state defaults ───────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    pair_id = st.selectbox(
        "Model Pair",
        options=list(PAIR_LABELS.keys()),
        format_func=lambda k: PAIR_LABELS[k],
        index=0,
    )

    gamma = st.slider("Speculation Length (gamma)", min_value=1, max_value=10, value=5, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    max_new_tokens = st.slider("Max New Tokens", min_value=32, max_value=512, value=128, step=32)

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=80,
    )

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Model status
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
        st.progress(vram_used / vram_total, text=f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB")
    else:
        st.error("No CUDA GPU detected")

# ── Load models ──────────────────────────────────────────────────────────

with st.spinner(f"Loading model pair {pair_id} — this may take a minute on first run..."):
    target_model, draft_model, tokenizer = ensure_models(pair_id)

# ── Title ────────────────────────────────────────────────────────────────

st.title("Speculative Decoding vs Baseline")
st.caption(
    "Send a message to compare standard autoregressive decoding against "
    "speculative decoding. Both methods use the **same target model** and "
    "produce the **same output distribution** — speculative decoding just "
    "gets there faster."
)

# ── Render chat history ──────────────────────────────────────────────────

for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.write(entry["content"])
    else:
        with st.chat_message("assistant"):
            bl = entry["baseline"]
            sp = entry["speculative"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Baseline (Autoregressive)**")
                st.markdown(bl["text"])
                bm = bl["metrics"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Tokens/sec", f"{bm['tps']:.1f}")
                c2.metric("Wall Clock", f"{bm['wall_ms']:.0f} ms")
                c3.metric("TTFT", f"{bm['ttft_ms']:.1f} ms")

            with col2:
                st.markdown("**Speculative Decoding**")
                st.markdown(sp["text"])
                sm = sp["metrics"]
                speedup = sm["tps"] / bm["tps"] if bm["tps"] > 0 else 0
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Tokens/sec",
                    f"{sm['tps']:.1f}",
                    delta=f"{sm['tps'] - bm['tps']:+.1f}",
                )
                c2.metric(
                    "Wall Clock",
                    f"{sm['wall_ms']:.0f} ms",
                    delta=f"{sm['wall_ms'] - bm['wall_ms']:+.0f} ms",
                    delta_color="inverse",
                )
                c3.metric(
                    "TTFT",
                    f"{sm['ttft_ms']:.1f} ms",
                    delta=f"{sm['ttft_ms'] - bm['ttft_ms']:+.1f} ms",
                    delta_color="inverse",
                )

            # Summary bar
            if bm["tps"] > 0:
                speedup = sm["tps"] / bm["tps"]
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Speedup", f"{speedup:.2f}x")
                sc2.metric("Acceptance Rate", f"{sm['acceptance_rate']:.1%}")
                sc3.metric("Acceptance Length", f"{sm['acceptance_length']:.2f}")
                sc4.metric("Tokens Generated", f"{sm['total_tokens']}")

# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Type a message...")

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # Tokenize the user's message
        formatted = format_prompt_for_chat(user_input, system_prompt, tokenizer)
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        seed = 42

        # ── Run baseline ─────────────────────────────────────────────
        with st.status("Running baseline (autoregressive)...", expanded=True) as status:
            st.write("Generating tokens one at a time with the target model...")
            gen_bl = torch.Generator(device="cuda")
            gen_bl.manual_seed(seed)

            bl_result = autoregressive_decode(
                model=target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                generator=gen_bl,
            )
            bl_m = bl_result["metrics"]
            status.update(
                label=f"Baseline done — {bl_m.tokens_per_second:.1f} tok/s",
                state="complete",
            )

        torch.cuda.empty_cache()

        # ── Run speculative ──────────────────────────────────────────
        with st.status("Running speculative decoding...", expanded=True) as status:
            st.write(
                f"Draft model proposes {gamma} tokens per round, "
                "target model verifies in one pass..."
            )
            gen_sp = torch.Generator(device="cuda")
            gen_sp.manual_seed(seed)

            sp_result = speculative_decode(
                target_model=target_model,
                draft_model=draft_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                gamma=gamma,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                generator=gen_sp,
            )
            sp_m = sp_result["metrics"]
            status.update(
                label=f"Speculative done — {sp_m.tokens_per_second:.1f} tok/s",
                state="complete",
            )

        torch.cuda.empty_cache()

        # ── Display results side-by-side ─────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline (Autoregressive)**")
            st.markdown(bl_result["output_text"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Tokens/sec", f"{bl_m.tokens_per_second:.1f}")
            c2.metric("Wall Clock", f"{bl_m.wall_clock_ms:.0f} ms")
            c3.metric("TTFT", f"{bl_m.ttft_ms:.1f} ms")

        with col2:
            st.markdown("**Speculative Decoding**")
            st.markdown(sp_result["output_text"])
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
            c3.metric(
                "TTFT",
                f"{sp_m.ttft_ms:.1f} ms",
                delta=f"{sp_m.ttft_ms - bl_m.ttft_ms:+.1f} ms",
                delta_color="inverse",
            )

        # Summary bar
        speedup = (
            sp_m.tokens_per_second / bl_m.tokens_per_second
            if bl_m.tokens_per_second > 0
            else 0
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Speedup", f"{speedup:.2f}x")
        sc2.metric("Acceptance Rate", f"{sp_m.acceptance_rate:.1%}")
        sc3.metric("Acceptance Length", f"{sp_m.acceptance_length:.2f}")
        sc4.metric("Tokens Generated", f"{sp_m.total_tokens_generated}")

        # ── Save to history (plain dicts for session state) ──────────
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "baseline": {
                    "text": bl_result["output_text"],
                    "metrics": {
                        "tps": bl_m.tokens_per_second,
                        "wall_ms": bl_m.wall_clock_ms,
                        "ttft_ms": bl_m.ttft_ms,
                        "total_tokens": bl_m.total_tokens_generated,
                    },
                },
                "speculative": {
                    "text": sp_result["output_text"],
                    "metrics": {
                        "tps": sp_m.tokens_per_second,
                        "wall_ms": sp_m.wall_clock_ms,
                        "ttft_ms": sp_m.ttft_ms,
                        "total_tokens": sp_m.total_tokens_generated,
                        "acceptance_rate": sp_m.acceptance_rate,
                        "acceptance_length": sp_m.acceptance_length,
                    },
                },
            }
        )
