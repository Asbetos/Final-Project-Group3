# speculative.py — Core Speculative Decoding Loop

## Purpose

This is the most important script in the project. It implements the speculative sampling algorithm from Leviathan et al. (ICML 2023) from scratch, with full per-token instrumentation, KV-cache management, and CUDA timing. The algorithm pairs a small draft model with a large target model to generate multiple tokens per target-model forward pass.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `logging` | `logging` | Logs warnings and errors during decoding. |
| `time` | `time` | Wall-clock timing for TTFT and per-round duration. |
| `torch` | `torch` | Tensor operations, `@torch.inference_mode()` decorator, CUDA device management. |
| `torch.nn.functional` | `F` | `F.softmax()` for converting target logits to probability distributions during verification. |
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer` | Type annotations for function parameters. |
| `metrics` (local) | `CudaTimer`, `GenerationMetrics`, `RoundMetrics`, `record_peak_vram`, `reset_peak_vram` | GPU timing, VRAM tracking, and metric dataclasses. |
| `sampling` (local) | `rejection_sample_token`, `sample_bonus_token`, `sample_from_logits` | Core sampling operations for drafting and verification. |

## Inputs and Outputs

- **Inputs**: Target model, draft model, tokenized prompt on GPU, gamma, temperature, max_new_tokens, tokenizer, optional generator.
- **Outputs**: Dict with `"output_ids"`, `"output_text"`, and `"metrics"` (GenerationMetrics with per-round details).

## Algorithm Overview

Each round of speculative decoding has three phases:

```
1. DRAFT:  Small model generates gamma candidate tokens autoregressively
2. VERIFY: Large model processes ALL draft tokens in ONE forward pass
3. ACCEPT: Rejection sampling decides how many draft tokens to keep
```

If all gamma tokens are accepted, a bonus token is sampled. If any token is rejected, a correction token replaces it and the round ends. The next round starts from the last accepted position.

## Detailed Line-by-Line Explanation

### KV-Cache Helpers (lines 38-88)

**`_get_cache_seq_len()`** (lines 38-51):
```python
def _get_cache_seq_len(past_key_values) -> int:
```
Returns the number of positions stored in the KV-cache. Supports three formats:
1. **Transformers >= 5.0** (`get_seq_length()` method): The newest API uses a `.layers` list with a `get_seq_length()` method.
2. **Transformers 4.36-4.x** (`key_cache` attribute): Older DynamicCache stores `key_cache` as a list of tensors per layer.
3. **Legacy tuple format**: Earliest format where cache is a tuple of (key, value) tensor pairs.

The three-way dispatch ensures compatibility across transformers versions.

**`_trim_kv_cache()`** (lines 54-88):
```python
def _trim_kv_cache(past_key_values, target_seq_len: int):
```
Trims the cache to keep only the first `target_seq_len` positions. This is necessary after verification because:
- If some draft tokens were rejected, their KV entries in the cache are invalid (computed from wrong tokens).
- The cache must align exactly with the accepted prefix for the next round.

Three code paths matching the three cache formats:
1. **`crop()` method** (transformers >= 5.0): Single method call, modifies in-place.
2. **`key_cache` slicing** (transformers 4.x): Manually slices each layer's key and value tensors.
3. **Tuple reconstruction**: Creates a new tuple of sliced (key, value) pairs.

### _draft_step() (lines 96-166)

```python
@torch.inference_mode()
def _draft_step(draft_model, input_ids, attention_mask, gamma, temperature, generator=None):
```

Generates `gamma` tokens autoregressively from the draft model.

**Lines 120-138 — Autoregressive loop:**
```python
for _ in range(gamma):
    if draft_cache is not None:
        out = draft_model(input_ids=current_input[:, -1:], ...)
    else:
        out = draft_model(input_ids=current_input, ...)
```
Same two-path pattern as `baseline.py`: full input on first iteration, single token with cache on subsequent iterations. The draft model uses its own local KV-cache (`draft_cache`) that is rebuilt from scratch each round.

**Why rebuild the draft cache each round?** The draft cache from the previous round becomes invalid when tokens are rejected or corrected. Rather than implementing complex cache trimming for the draft model (which would save little time since the draft is small), the code simply reprocesses the full accepted prefix. For a 0.6B model, this is acceptable overhead.

**Lines 143-147 — Sampling and recording:**
```python
token_id, token_probs = sample_from_logits(logits.squeeze(0), temperature, generator)
tokens.append(token_id)
probs.append(token_probs)
```
Both the token ID and the full probability distribution are saved. The distribution `q(x)` is needed later by the rejection sampler to compute the acceptance ratio `p(x)/q(x)`.

**Lines 150-160 — Sequence extension:**
The draft token is appended to the input and the attention mask is extended, preparing for the next iteration of the draft loop.

### _verify_step() (lines 174-312)

```python
@torch.inference_mode()
def _verify_step(target_model, prefix_ids, prefix_mask, draft_tokens, draft_probs, gamma, temperature, target_cache, prefix_len_in_cache, generator=None):
```

This function is the heart of speculative decoding. It runs **one** forward pass of the target model over all draft tokens and performs rejection sampling.

**Lines 214-227 — Target model forward pass:**
```python
new_start = prefix_len_in_cache
if target_cache is not None and new_start > 0:
    out = target_model(input_ids=prefix_ids[:, new_start:], ...)
else:
    out = target_model(input_ids=prefix_ids, ...)
```
The target model processes only the NEW tokens (those not already in its KV-cache). On the first round (`target_cache is None`), it processes the entire prompt + draft tokens. On subsequent rounds, it processes only the tokens added since the last round: the bonus/correction token from the previous round + the new draft tokens.

This is the key efficiency insight: the target model does one forward pass for gamma+1 tokens instead of gamma+1 separate passes.

**Lines 232-247 — Logit offset calculation:**
```python
prefix_without_draft = prefix_ids.shape[1] - gamma
logit_offset = prefix_without_draft - new_start - 1
```

In a causal language model, the logits at position `j` predict the token at position `j+1`. To verify draft token `i` at position `prefix_without_draft + i`, we need the logits at position `prefix_without_draft + i - 1`. Since the model only outputs logits for the new tokens (starting at index 0 for position `new_start`), the logits index is `(prefix_without_draft + i - 1) - new_start`.

This unified formula works for both cached and uncached cases:
- **No cache** (new_start=0): `logit_offset = prefix_without_draft - 1` (matches the standard LM logit indexing)
- **With cache**: Adjusts for the offset of the new input within the full sequence

**Lines 249-279 — Rejection sampling loop:**
```python
for i in range(gamma):
    idx = logit_offset + i
    target_logits_i = all_logits[0, idx, :]
```

For each draft token:
1. Extract the target model's logits at the position that predicts this draft token.
2. Convert to probabilities (one-hot for greedy, softmax for stochastic).
3. Call `rejection_sample_token()` to decide accept/reject.
4. If accepted, continue to the next draft token.
5. If rejected, append the correction token and break out of the loop.

**Lines 282-294 — Bonus token:**
```python
if num_accepted == gamma:
    bonus_idx = logit_offset + gamma
    if bonus_idx < all_logits.shape[1]:
        ...
        bonus_tok = sample_bonus_token(bonus_probs, temperature, generator)
        accepted_tokens.append(bonus_tok)
```
If all gamma tokens passed verification, the target model's logits at position `bonus_idx` predict the token after all draft tokens. Sampling from this gives a free extra token, meaning the best-case output per round is `gamma + 1` tokens.

**Lines 296-303 — Cache trimming:**
```python
keep_len = prefix_without_draft + num_accepted
target_cache = _trim_kv_cache(target_cache, keep_len)
```
The cache is trimmed to keep only positions with valid KV entries. The `num_accepted` draft tokens have correct KV in the cache (they were part of the model input). The bonus token and correction token were never in the model input, so their KV will be computed in the next round when they are fed as part of the new input.

### speculative_decode() (lines 320-513)

```python
@torch.inference_mode()
def speculative_decode(target_model, draft_model, input_ids, attention_mask, gamma, temperature, max_new_tokens, tokenizer, generator=None):
```

The main entry point that orchestrates the draft-verify loop.

**Lines 364-371 — Budget management:**
```python
remaining = max_new_tokens - len(generated_ids)
effective_gamma = min(gamma, remaining)
```
Near the end of generation, fewer than gamma tokens may be needed. `effective_gamma` prevents over-generation.

**Lines 374-391 — Sequence reconstruction:**
```python
if generated_ids:
    gen_tensor = torch.tensor([generated_ids], device=device, dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, gen_tensor], dim=1)
```
The full sequence (prompt + all accepted tokens so far) is rebuilt each round. The draft model needs the complete sequence because its cache is not maintained across rounds.

**Lines 439-442 — Cache length tracking:**
```python
target_cache = verify_result["target_cache"]
target_cache_len = current_len + verify_result["num_accepted"]
```
`target_cache_len` tracks how many positions have valid KV entries in the target cache. Only `num_accepted` (not `len(accepted_tokens)`) is added because bonus/correction tokens are not in the cache.

**Lines 476-488 — Metric aggregation:**
```python
acceptance_rate = sum(all_decisions) / len(all_decisions)
acceptance_length = sum(r.tokens_accepted for r in all_rounds) / len(all_rounds)
```
- `acceptance_rate`: Fraction of all proposed draft tokens that were accepted (across all rounds).
- `acceptance_length`: Average number of draft tokens accepted per round (not counting bonus tokens).

**Line 491:**
```python
draft_overhead = total_draft_ms / wall_ms
```
The fraction of total time spent on draft generation. If this is too high, the draft model is too slow relative to the time saved by batching target verification.

## Design Decisions

- **Draft cache rebuilt each round**: Simpler and correct. The draft model is small enough (0.6B-1.7B) that reprocessing the prefix is acceptable. Maintaining an incremental draft cache would add complexity with marginal benefit.
- **Target cache maintained across rounds**: The target model (8B) is much larger, so reprocessing is expensive. The cache is carefully trimmed to exactly the accepted prefix length, and the next round feeds only the new tokens.
- **Unified logit offset formula**: A single formula `prefix_without_draft - new_start - 1` replaces separate cached/uncached branches, reducing the risk of indexing bugs.
- **`num_accepted` for cache tracking, not `len(accepted_tokens)`**: Bonus and correction tokens are never part of the model input, so their KV is not in the cache. Using `len(accepted_tokens)` would over-count the cache length, causing incorrect logit alignment in subsequent rounds.
