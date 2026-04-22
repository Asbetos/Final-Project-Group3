# sampling.py — Probability-Level Sampling Utilities

## Purpose

This script implements the core mathematical operations of speculative decoding verification: converting logits to probability distributions, performing rejection sampling to decide whether to accept or reject a draft token, resampling correction tokens from a residual distribution, and sampling bonus tokens. These are pure functions with no model or tokenizer dependencies.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `torch` | `torch` | Tensor operations: argmax, multinomial sampling, random number generation. |
| `torch.nn.functional` | `F` | `F.softmax()` for converting logits to normalized probability distributions. |
| `typing` | `Tuple` | Return type annotations. |

## Inputs and Outputs

- **Inputs**: Raw logit tensors (shape `(vocab_size,)`), probability tensors, temperature floats, optional `torch.Generator` for reproducibility.
- **Outputs**: Token IDs (int) and probability distributions (tensors).

## Detailed Line-by-Line Explanation

### sample_from_logits() (lines 13-40)

```python
def sample_from_logits(logits, temperature, generator=None) -> Tuple[int, torch.Tensor]:
```

Converts raw model logits into a probability distribution and samples one token. Used by both the draft phase (to generate candidate tokens) and the baseline decoder.

**Lines 29-34 — Greedy path (temperature=0):**
```python
if temperature == 0.0:
    probs = torch.zeros_like(logits)
    token_id = logits.argmax(dim=-1).item()
    probs[token_id] = 1.0
    return token_id, probs
```
At temperature 0, always pick the highest-scoring token (argmax). The function constructs a one-hot probability distribution and returns it alongside the token. This one-hot distribution is needed by the verification step — even in greedy mode, the rejection sampler compares probability vectors.

**Lines 37-40 — Stochastic path (temperature > 0):**
```python
scaled_logits = logits / temperature
probs = F.softmax(scaled_logits, dim=-1)
token_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
```
- **Temperature scaling** (line 37): Dividing logits by temperature before softmax controls randomness. Temperature < 1 sharpens the distribution (more deterministic); temperature > 1 flattens it (more random).
- **Softmax** (line 38): Converts scaled logits to a valid probability distribution (non-negative, sums to 1).
- **Multinomial sampling** (line 39): Draws one token from the categorical distribution. The `generator` parameter ensures reproducibility — given the same seed, the same sequence of random draws occurs.

### rejection_sample_token() (lines 43-95)

```python
def rejection_sample_token(target_probs, draft_probs, draft_token, temperature, generator=None):
```

This is the core verification step of speculative sampling. It decides whether to accept a draft token based on the target model's probability distribution.

**Lines 65-70 — Greedy verification:**
```python
if temperature == 0.0:
    target_argmax = target_probs.argmax(dim=-1).item()
    if target_argmax == draft_token:
        return True, draft_token
    return False, target_argmax
```
At temperature 0, acceptance is binary: accept if and only if the target model's argmax matches the draft token. If rejected, the correction token is simply the target's argmax. This guarantees that greedy speculative decoding produces exactly the same output as greedy autoregressive decoding.

**Lines 72-74 — Stochastic acceptance probability:**
```python
p = target_probs[draft_token].item()
q = draft_probs[draft_token].item()
```
Extract the target probability `p(x)` and draft probability `q(x)` for the specific draft token `x`.

**Lines 76-83 — Edge case: zero draft probability:**
```python
if q == 0.0:
    if p == 0.0:
        return True, draft_token
    return False, sample_residual_distribution(target_probs, draft_probs, generator)
```
If the draft model assigned zero probability to the token it somehow produced (can happen with numerical edge cases), and the target also assigns zero, accept it (both agree it's impossible — vacuously correct). If only the draft assigns zero but the target doesn't, reject and resample.

**Lines 85-88 — Main acceptance logic:**
```python
acceptance_prob = min(1.0, p / q)
u = torch.rand(1, generator=generator, device=target_probs.device).item()
if u < acceptance_prob:
    return True, draft_token
```
This is the key equation from Leviathan et al.: accept with probability `min(1, p(x)/q(x))`. If the target assigns higher probability to the draft token than the draft model did (`p >= q`), the token is always accepted. If the target assigns lower probability (`p < q`), the token is accepted with probability `p/q`.

The uniform random draw `u` is generated on the same device as the tensors and uses the seeded generator for reproducibility.

**Lines 92-95 — Rejection and correction:**
```python
correction_token = sample_residual_distribution(target_probs, draft_probs, generator)
return False, correction_token
```
If rejected, sample a correction token from the residual distribution instead.

### sample_residual_distribution() (lines 98-119)

```python
def sample_residual_distribution(target_probs, draft_probs, generator=None) -> int:
```

Computes and samples from `max(0, p - q) / Z`, where `Z` is the normalization constant.

**Line 109:**
```python
residual = torch.clamp(target_probs - draft_probs, min=0.0)
```
Element-wise `p(x) - q(x)`, clamped to non-negative values. Where `p > q`, the target model "wanted more" of that token than the draft provided. Where `p <= q`, the residual is zero — the draft already over-represented that token.

**Lines 110-118 — Normalization and sampling:**
```python
total = residual.sum()
if total <= 0.0:
    return torch.multinomial(target_probs, num_samples=1, generator=generator).item()
residual = residual / total
return torch.multinomial(residual, num_samples=1, generator=generator).item()
```
The residual is normalized to sum to 1, creating a valid probability distribution. If the residual is all zeros (degenerate case where `q >= p` everywhere), fall back to sampling from the target distribution directly.

**Mathematical guarantee**: The combination of the acceptance/rejection decision and the residual sampling ensures that the marginal distribution of each output token exactly matches the target model's distribution. This is the fundamental correctness property of speculative sampling.

### sample_bonus_token() (lines 122-138)

```python
def sample_bonus_token(target_probs, temperature, generator=None) -> int:
```

When all gamma draft tokens are accepted, one additional "bonus" token is sampled from the target model's distribution at the next position. This is why speculative decoding can produce up to `gamma + 1` tokens per round.

- **Greedy** (line 133-134): Return the argmax.
- **Stochastic** (lines 136-138): Multinomial sample from the target distribution.

## Design Decisions

- **Pure functions, no model dependencies**: This module only operates on probability tensors. This makes unit testing straightforward — `test_correctness.py` level 1 tests these functions without any GPU or model loading.
- **Generator parameter everywhere**: Every sampling function accepts an optional `torch.Generator`. This enables exact reproducibility: given the same seed, the entire speculative decoding process produces identical results.
- **Explicit temperature=0 branches**: Rather than using a very small temperature (e.g., 1e-8), the code explicitly checks `temperature == 0.0` and uses argmax. This avoids numerical instability from dividing logits by near-zero values and makes the greedy behavior mathematically exact.
