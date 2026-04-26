# app.py — Streamlit Interactive Demo

## Purpose

This script provides a web-based chat interface for interactively comparing baseline autoregressive decoding against speculative decoding. Users type a message and see both methods' outputs side-by-side with real-time performance metrics (tokens/sec, wall clock, TTFT, acceptance rate, speedup). It serves as a demonstration tool for presentations and as an intuitive way to explore how speculative decoding behaves on different types of prompts.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `gc` | `gc` | `gc.collect()` for forcing garbage collection when switching model pairs to free GPU memory. |
| `torch` | `torch` | `torch.cuda.empty_cache()` for VRAM cleanup, `torch.cuda.memory_allocated()` / `torch.cuda.get_device_properties()` for the VRAM usage bar, `torch.Generator` for seeded sampling. |
| `streamlit` | `st` | The web framework. Provides page config, sidebar, chat UI, columns, metrics, spinners, caching, and session state. |
| `baseline` (local) | `autoregressive_decode` | Standard autoregressive decoder for the baseline comparison. |
| `config` (local) | `PAIR_MAP` | Maps pair IDs to `ModelPairConfig` objects. |
| `data` (local) | `format_prompt_for_chat` | Applies the Qwen3 chat template to user input. |
| `models` (local) | `load_model_pair` | Loads target model, draft model, and tokenizer. |
| `speculative` (local) | `speculative_decode` | Speculative decoding algorithm. |

## Inputs and Outputs

- **Inputs**: User text input via the Streamlit chat widget, sidebar parameter controls.
- **Outputs**: Side-by-side generated text with performance metrics displayed in the browser.

## Detailed Line-by-Line Explanation

### Page Configuration (lines 19-23)

```python
st.set_page_config(
    page_title="Speculative Decoding Demo",
    page_icon=":zap:",
    layout="wide",
)
```
- **`layout="wide"`**: Uses the full browser width. Essential for the side-by-side comparison columns to have enough space.
- **`page_icon=":zap:"`**: Lightning bolt emoji, thematically appropriate for a speed comparison demo.

This must be the first Streamlit command in the script (Streamlit requirement).

### PAIR_LABELS (lines 27-31)

```python
PAIR_LABELS = {
    "A": "Pair A  —  Qwen3-8B (fp16) + Qwen3-0.6B  (~20.6 GB)",
    "B": "Pair B  —  Qwen3-8B (fp16) + Qwen3-1.7B  (~22.8 GB)",
    "C": "Pair C  —  Qwen3-8B (4-bit) + Qwen3-0.6B  (~8.2 GB)",
}
```
Human-readable labels for the pair selector dropdown. Includes VRAM estimates so users can choose based on their available memory.

### Model Caching (lines 36-56)

**load_models() (lines 36-42):**
```python
@st.cache_resource
def load_models(pair_id: str):
    pair_config = PAIR_MAP[pair_id]
    target_model, draft_model, tokenizer = load_model_pair(pair_config)
    return target_model, draft_model, tokenizer
```
`@st.cache_resource` is Streamlit's decorator for caching objects that should persist across reruns (user interactions). Without this, models would be reloaded (~30-60 seconds) every time the user sends a message, because Streamlit reruns the entire script on each interaction.

The cache is keyed by `pair_id`, so:
- Same pair: Returns cached models instantly.
- Different pair: Cache miss, loads new models.

**ensure_models() (lines 45-56):**
```python
def ensure_models(pair_id: str):
    prev = st.session_state.get("loaded_pair_id")
    if prev != pair_id:
        if prev is not None:
            load_models.clear()
            gc.collect()
            torch.cuda.empty_cache()
        st.session_state.loaded_pair_id = pair_id
        st.session_state.chat_history = []
    return load_models(pair_id)
```

Handles model pair switching. When the user selects a different pair:
1. **`load_models.clear()`**: Invalidates the Streamlit cache, releasing references to the old models.
2. **`gc.collect()`**: Forces Python garbage collection to delete the model objects.
3. **`torch.cuda.empty_cache()`**: Returns freed CUDA memory to the allocator pool.
4. **Chat history cleared**: Previous conversation was with a different model — keeping it would be confusing.

Without this cleanup, both old and new models would coexist in VRAM, causing an OOM error on the A10G.

### Session State (lines 60-62)

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```
Initializes chat history on first load. `st.session_state` persists across Streamlit reruns (which happen on every user interaction). The conditional prevents resetting history on every rerun.

### Sidebar (lines 66-99)

```python
with st.sidebar:
    pair_id = st.selectbox("Model Pair", options=list(PAIR_LABELS.keys()),
                           format_func=lambda k: PAIR_LABELS[k], index=0)
    gamma = st.slider("Speculation Length (gamma)", min_value=1, max_value=10, value=5, step=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    max_new_tokens = st.slider("Max New Tokens", min_value=32, max_value=512, value=128, step=32)
```

All configurable parameters are in the sidebar:
- **Pair selector**: Dropdown with human-readable labels via `format_func`.
- **Gamma slider**: 1-10 range. Default 5 matches the commonly used speculation length.
- **Temperature slider**: 0.0-1.5 range. Default 0.0 (greedy) for deterministic comparison.
- **Max tokens slider**: 32-512 range in steps of 32.

```python
system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)
```
Editable system prompt for the chat template. Users can experiment with different instructions.

```python
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
```
Clears the conversation and triggers a full page rerun.

**Lines 92-99 — VRAM status bar:**
```python
if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    st.progress(vram_used / vram_total, text=f"VRAM: {vram_used:.1f} / {vram_total:.1f} GB")
```
Shows a real-time VRAM usage bar using `st.progress`. `get_device_properties(0).total_memory` returns the GPU's total memory in bytes (24 GB for A10G). The progress bar fills proportionally.

### Chat History Rendering (lines 111-157)

```python
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            st.write(entry["content"])
    else:
        with st.chat_message("assistant"):
            ...
```

Re-renders the full chat history on each Streamlit rerun. Streamlit's execution model re-runs the entire script top-to-bottom on every interaction, so all UI elements must be recreated. The chat history in session state provides the persistence.

**Assistant messages** are rendered as side-by-side columns:
```python
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Baseline (Autoregressive)**")
    st.markdown(bl["text"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens/sec", f"{bm['tps']:.1f}")
```

Each assistant entry stores both baseline and speculative results as plain dicts (not dataclass instances, because Streamlit session state requires JSON-serializable objects).

### Chat Input and Generation (lines 161-265)

**Lines 161-165 — User input handling:**
```python
user_input = st.chat_input("Type a message...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
```
`st.chat_input` renders a text input at the bottom of the page. When the user submits, the message is appended to history and the script re-runs.

**Lines 169-177 — Tokenization:**
```python
formatted = format_prompt_for_chat(user_input, system_prompt, tokenizer)
encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
input_ids = encoded["input_ids"].to("cuda")
attention_mask = encoded["attention_mask"].to("cuda")
```
Applies the Qwen3 chat template with the user-configured system prompt, then tokenizes. The max length (2048) is higher than the benchmark default (512) to allow longer interactive conversations.

**Lines 181-197 — Baseline execution:**
```python
with st.status("Running baseline (autoregressive)...", expanded=True) as status:
    st.write("Generating tokens one at a time with the target model...")
    bl_result = autoregressive_decode(
        model=target_model,
        input_ids=input_ids, attention_mask=attention_mask,
        temperature=temperature, max_new_tokens=max_new_tokens,
        tokenizer=tokenizer, generator=gen_bl,
    )
    status.update(label=f"Baseline done — {bl_m.tokens_per_second:.1f} tok/s", state="complete")
```

`st.status` shows an expandable progress indicator while the baseline runs. The label updates to show the result when complete.

**Line 199:**
```python
torch.cuda.empty_cache()
```
Clears the CUDA cache between baseline and speculative runs to ensure the speculative run doesn't benefit from cached memory allocations.

**Lines 202-221 — Speculative execution:**
Same pattern as baseline, but calls `speculative_decode()` with the user-configured gamma.

**Lines 225-249 — Results display:**
```python
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Baseline (Autoregressive)**")
    st.markdown(bl_result["output_text"])
    c1.metric("Tokens/sec", f"{bl_m.tokens_per_second:.1f}")
with col2:
    st.markdown("**Speculative Decoding**")
    st.markdown(sp_result["output_text"])
    c1.metric("Tokens/sec", f"{sp_m.tokens_per_second:.1f}",
              delta=f"{sp_m.tokens_per_second - bl_m.tokens_per_second:+.1f}")
```

The speculative column's metrics include **deltas** showing the difference from baseline:
- `delta` on Tokens/sec: Positive = speculative is faster (shown green).
- `delta` on Wall Clock and TTFT: Uses `delta_color="inverse"` so that negative values (speculative is faster/lower) appear green.

**Lines 251-256 — Summary metrics bar:**
```python
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Speedup", f"{speedup:.2f}x")
sc2.metric("Acceptance Rate", f"{sp_m.acceptance_rate:.1%}")
sc3.metric("Acceptance Length", f"{sp_m.acceptance_length:.2f}")
sc4.metric("Tokens Generated", f"{sp_m.total_tokens_generated}")
```
A four-column summary bar below the side-by-side output, highlighting the key speculative decoding metrics.

**Lines 259-277 — History persistence:**
```python
st.session_state.chat_history.append({
    "role": "assistant",
    "baseline": {
        "text": bl_result["output_text"],
        "metrics": {"tps": bl_m.tokens_per_second, "wall_ms": bl_m.wall_clock_ms, ...},
    },
    "speculative": {
        "text": sp_result["output_text"],
        "metrics": {"tps": sp_m.tokens_per_second, ...},
    },
})
```
Saves both results as plain dicts. On the next rerun, the chat history rendering loop (lines 111-157) will display them.

## Design Decisions

- **`@st.cache_resource` for models**: Models are 4-20 GB in VRAM. Reloading them on every Streamlit rerun (which happens on every user interaction) would make the app unusable. Caching persists them in memory across reruns.
- **Plain dict metrics in session state**: Streamlit serializes session state objects. Dataclass instances can fail serialization in some configurations, so extracting scalar values into plain dicts ensures reliability.
- **Sequential baseline-then-speculative execution**: Running both simultaneously would require double the KV-cache memory. Sequential execution with `torch.cuda.empty_cache()` between them keeps VRAM usage manageable.
- **Fixed seed (42) for both methods**: Both generators use the same seed, so at temperature=0 both produce identical output (demonstrating the correctness guarantee). At temperature>0, the same seed means any output differences are due to the speculative algorithm's sampling path, not different random draws.
- **VRAM bar in sidebar**: Provides at-a-glance memory monitoring. Users can see if they're approaching the 24 GB limit before an OOM crash occurs.
- **`delta_color="inverse"` for latency metrics**: For wall clock and TTFT, lower is better. Streamlit's default colors positive deltas green, but for latency a positive delta (speculative is slower) should be red. `delta_color="inverse"` flips this convention.
