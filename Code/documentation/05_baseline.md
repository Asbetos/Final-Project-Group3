# baseline.py — Standard Autoregressive Decoder

## Purpose

This script implements standard token-by-token autoregressive decoding — the conventional way language models generate text. It serves as the **performance baseline** against which speculative decoding is measured. The implementation uses the same sampling logic and timing instrumentation as the speculative path so that comparisons are fair.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `logging` | `logging` | Logs generation progress. |
| `time` | `time` | `time.perf_counter()` for wall-clock TTFT and total time. |
| `torch` | `torch` | Tensor operations, `@torch.inference_mode()` decorator. |
| `torch.nn.functional` | `F` | Softmax (imported but used indirectly via sampling). |
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer` | Type annotations for function signature. |
| `metrics` (local) | `CudaTimer`, `GenerationMetrics`, `RoundMetrics`, `record_peak_vram`, `reset_peak_vram` | GPU timing and metric tracking. |
| `sampling` (local) | `sample_from_logits` | Shared sampling function for temperature-controlled token selection. |

## Inputs and Outputs

- **Inputs**: Target model, tokenized prompt (input_ids + attention_mask on GPU), temperature, max_new_tokens, tokenizer, optional generator.
- **Outputs**: Dict with `"output_ids"` (list of ints), `"output_text"` (decoded string), and `"metrics"` (GenerationMetrics dataclass).

## Detailed Line-by-Line Explanation

### autoregressive_decode() (lines 22-123)

```python
@torch.inference_mode()
def autoregressive_decode(model, input_ids, attention_mask, temperature, max_new_tokens, tokenizer, generator=None):
```

**Line 22 — `@torch.inference_mode()`**: This decorator disables PyTorch's autograd engine entirely. Unlike `torch.no_grad()`, `inference_mode()` also disables version counting on tensors, providing a small additional speedup. Since we are only doing inference (no backward pass), autograd tracking would be pure overhead.

**Lines 50-53 — State initialization:**
```python
generated_ids: list[int] = []
past_key_values = None
current_input = input_ids
current_mask = attention_mask
```
- `generated_ids`: Accumulates the generated token IDs as plain Python ints.
- `past_key_values`: The KV-cache. Starts as `None` (first step processes the full prompt). After the first step, it caches key/value tensors from all transformer layers so previous tokens don't need recomputation.
- `current_input` / `current_mask`: The full input sequence, growing by one token each step.

**Lines 56-59 — Metric initialization:**
```python
reset_peak_vram()
wall_start = time.perf_counter()
ttft_ms = 0.0
ttft_recorded = False
```
`reset_peak_vram()` zeroes the peak VRAM tracker so this generation's peak is measured in isolation. `wall_start` marks the beginning for total wall-clock time.

**Lines 61-75 — Main generation loop:**
```python
for step in range(max_new_tokens):
    with CudaTimer() as timer:
        if past_key_values is not None:
            outputs = model(
                input_ids=current_input[:, -1:],
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            outputs = model(
                input_ids=current_input,
                attention_mask=current_mask,
                use_cache=True,
            )
```

Two code paths:
1. **First step** (`past_key_values is None`): Feed the entire prompt. The model processes all prompt tokens and outputs logits for the next token position.
2. **Subsequent steps** (`past_key_values is not None`): Feed only the last token (`current_input[:, -1:]`). The KV-cache already contains key/value tensors for all previous positions. This reduces each step from O(sequence_length) to O(1) in the attention computation.

`use_cache=True` tells the model to return updated `past_key_values` with the new token's key/value appended.

Each step is wrapped in `CudaTimer` for GPU-accurate per-step timing.

**Lines 77-83 — Sampling:**
```python
past_key_values = outputs.past_key_values
logits = outputs.logits[:, -1, :]
token_id, _ = sample_from_logits(logits.squeeze(0), temperature, generator)
generated_ids.append(token_id)
```
- `outputs.logits[:, -1, :]`: Takes logits from the last position only — this is the prediction for the next token.
- `logits.squeeze(0)`: Removes the batch dimension (goes from `(1, vocab_size)` to `(vocab_size,)`) to match `sample_from_logits()`'s expected input shape.
- The probability distribution (second return value) is discarded (`_`) since the baseline doesn't need it for verification.

**Lines 85-87 — Time to First Token:**
```python
if not ttft_recorded:
    ttft_ms = (time.perf_counter() - wall_start) * 1000.0
    ttft_recorded = True
```
TTFT is measured after the first token is produced. For the baseline, this includes the time to process the entire prompt (the "prefill" phase). For speculative decoding, TTFT includes both draft and verify phases for the first round.

**Lines 89-95 — Sequence extension:**
```python
next_token = torch.tensor([[token_id]], device=device, dtype=input_ids.dtype)
current_input = torch.cat([current_input, next_token], dim=1)
current_mask = torch.cat(
    [current_mask, torch.ones(1, 1, device=device, dtype=current_mask.dtype)],
    dim=1,
)
```
Appends the new token to the input sequence and extends the attention mask with a 1. The full sequence is maintained even though only the last token is fed to the model — this is needed because the attention mask must cover all positions.

**Lines 97-98 — EOS check:**
```python
if token_id == tokenizer.eos_token_id:
    break
```
Stops generation early if the model produces the end-of-sequence token.

**Lines 100-115 — Metric computation:**
```python
wall_ms = (time.perf_counter() - wall_start) * 1000.0
total_tokens = len(generated_ids)
tps = (total_tokens / wall_ms * 1000.0) if wall_ms > 0 else 0.0
```
Tokens-per-second is the primary throughput metric. The baseline always has `acceptance_rate = 1.0` and `acceptance_length = 1.0` (one token per step) since these concepts only apply to speculative decoding.

## Design Decisions

- **Same sampling function as speculative path**: Both `baseline.py` and `speculative.py` call `sample_from_logits()` from `sampling.py`. This ensures any sampling behavior differences between the two methods are due to the speculative algorithm, not implementation divergence.
- **CudaTimer per step**: While not used for per-step reporting in the baseline, this keeps the instrumentation overhead consistent with the speculative path.
- **No batch dimension > 1**: The baseline processes one prompt at a time, matching the speculative path. Batching would improve baseline throughput but would make the comparison unfair.
- **Fixed acceptance metrics**: `acceptance_rate = 1.0` and `acceptance_length = 1.0` are hardcoded since these fields exist in `GenerationMetrics` but have no meaning for autoregressive decoding. This allows the same CSV schema for both methods.
