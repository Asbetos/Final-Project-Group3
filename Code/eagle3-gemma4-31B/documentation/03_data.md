# data.py — Dataset Loading, Prompt Formatting, and Tokenization

## Purpose

This script manages the entire data pipeline: loading raw text prompts from HuggingFace Datasets, formatting them with the Qwen3 chat template, and tokenizing them into GPU-ready tensors. It abstracts away all dataset-specific details behind a declarative task registry.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `logging` | `logging` | Logs dataset loading progress and prompt counts. |
| `typing` | `Dict`, `List` | Type annotations for function signatures. |
| `torch` | `torch` | Tensor creation and device placement (`.to(device)`). |
| `datasets` | `load_dataset` | HuggingFace Datasets library for downloading and accessing NLP datasets from the Hub. Handles caching, streaming, and split management. |
| `transformers` | `AutoTokenizer` | Tokenizer for converting formatted text to token IDs and attention masks. |

## Inputs and Outputs

- **Inputs**: Task name (string), tokenizer, number of prompts, random seed.
- **Outputs**: List of dicts, each containing `"input_ids"` and `"attention_mask"` tensors on GPU.

## Detailed Line-by-Line Explanation

### TASK_REGISTRY (lines 16-49)

```python
TASK_REGISTRY: Dict[str, dict] = {
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {},
        "split": "test",
        "field": "prompt",
        "system_prompt": "Complete the following Python function.",
        "max_prompt_tokens": 512,
    },
    ...
}
```

A declarative configuration for all four tasks. Each entry specifies:

- `load_args` / `load_kwargs`: Positional and keyword arguments for `load_dataset()`. For example, TriviaQA uses `("mandarjoshi/trivia_qa", "rc")` where `"rc"` is the Reading Comprehension configuration subset.
- `split`: Which dataset split to use. HumanEval uses `"test"` (164 samples); TriviaQA and WritingPrompts use `"validation"` to avoid contamination with potential training data.
- `field`: The column name containing the raw prompt text. Different datasets use different field names (`"prompt"`, `"question"`, `"article"`).
- `system_prompt`: A task-specific instruction prepended as the system message in the chat template. This guides the model toward the expected output format.
- `max_prompt_tokens`: Maximum tokenized prompt length. Varies by task: code prompts (512) are longer than questions (256), and articles (1024) are longest. Truncation prevents excessive KV-cache memory usage during generation.

**Why these four tasks:**
- **HumanEval** (code): Highly structured, predictable output. Expected highest acceptance rates because code follows rigid syntax patterns.
- **TriviaQA** (factual QA): Short, factual answers. Moderate predictability.
- **CNN/DailyMail** (summarization): Medium-length, semi-structured output.
- **WritingPrompts** (creative writing): Open-ended, high-entropy output. Expected lowest acceptance rates because creative text is least predictable.

### load_prompts() (lines 52-75)

```python
def load_prompts(task: str, num_prompts: int = 50, seed: int = 42) -> List[str]:
```

**Line 58-59 — Validation:**
```python
if task not in TASK_REGISTRY:
    raise ValueError(...)
```
Fail-fast with a clear error message listing valid task names.

**Line 64 — Dataset loading:**
```python
ds = load_dataset(*info["load_args"], **info["load_kwargs"])
```
Uses unpacking to forward the registry's arguments to `load_dataset()`. This design lets the registry handle datasets that need positional args (like the TriviaQA subset name) without any conditional logic.

**Line 66 — Deterministic shuffle:**
```python
split = split.shuffle(seed=seed)
```
Shuffles with a fixed seed so every run selects the same prompts. This is critical for reproducibility — re-running an experiment with the same seed produces identical prompts.

**Lines 68-69 — Subset selection:**
```python
n = min(num_prompts, len(split))
subset = split.select(range(n))
```
`min()` guards against datasets smaller than `num_prompts` (HumanEval has only 164 test samples). `.select(range(n))` takes the first `n` from the shuffled split — efficient because it uses index-based access without copying.

### format_prompt_for_chat() (lines 78-98)

```python
def format_prompt_for_chat(raw_prompt, system_prompt, tokenizer) -> str:
```

Applies the Qwen3 chat template to produce the expected input format:

**Lines 88-91 — Message construction:**
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": raw_prompt},
]
```
Constructs a two-turn conversation: one system message (task instruction) and one user message (the actual prompt).

**Lines 92-97 — Template application:**
```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```
- `tokenize=False`: Returns the formatted string, not token IDs. Tokenization happens separately to allow truncation control.
- `add_generation_prompt=True`: Appends the assistant turn prefix so the model knows to start generating. For Qwen3, this adds the `<|im_start|>assistant\n` marker.
- `enable_thinking=False`: Qwen3 has a "thinking" mode that outputs chain-of-thought reasoning in `<think>` tags. Disabling it ensures standard autoregressive behavior, which is what the speculative decoding experiments require.

### tokenize_prompts() (lines 101-139)

```python
def tokenize_prompts(task, tokenizer, num_prompts=50, seed=42, device="cuda"):
```

Full pipeline function that chains load -> format -> tokenize.

**Lines 120-124 — Tokenization:**
```python
encoded = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=info["max_prompt_tokens"],
)
```
- `return_tensors="pt"`: Returns PyTorch tensors (not lists), ready for model input.
- `truncation=True`: Truncates prompts exceeding `max_prompt_tokens`. Without this, long CNN/DailyMail articles could consume all available KV-cache memory.
- `max_length`: Per-task token limit from the registry.

**Lines 128-129 — GPU placement:**
```python
"input_ids": encoded["input_ids"].to(device),
"attention_mask": encoded["attention_mask"].to(device),
```
Moves tensors to GPU immediately. Each prompt is a separate dict because prompts have variable lengths and cannot be batched without padding (which would waste compute on pad tokens in an inference benchmark).

## Design Decisions

- **Registry pattern over if/elif chains**: Adding a new task requires adding one dict entry — no code changes. The registry separates "what" (dataset details) from "how" (loading/formatting logic).
- **Per-prompt dicts instead of batched tensors**: Speculative decoding processes one prompt at a time (no batch dimension > 1). Variable-length prompts would require padding in a batch, adding wasted computation that would skew benchmark results.
- **Prompt truncation, not generation truncation**: Limiting prompt length (not output length) ensures all prompts fit comfortably in VRAM while letting the model generate its full `max_new_tokens` output.
