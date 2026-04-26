# test_correctness.py — Correctness Validation Suite

## Purpose

This script validates that the speculative decoding implementation is mathematically correct. It provides three progressive testing levels: Level 1 tests the pure sampling functions without any GPU or model dependencies, Level 2 verifies the critical guarantee that greedy speculative decoding produces identical output to greedy autoregressive decoding, and Level 3 runs a quick smoke test to ensure nothing crashes under various configurations. This layered approach allows rapid iteration (Level 1 runs in milliseconds) while still providing comprehensive end-to-end validation (Levels 2-3).

## Packages Used

| Package | Import | Why |
|---|---|---|
| `argparse` | `argparse` | CLI argument parsing for `--level` and `--pair` flags. |
| `logging` | `logging` | Structured test output with timestamps and pass/fail status. |
| `sys` | `sys` | `sys.stdout` as the logging handler target. |
| `torch` | `torch` | Tensor creation for test inputs, `torch.Generator` for reproducibility, `torch.nn.functional.F` imported for potential use. |
| `sampling` (local) | `sample_from_logits`, `rejection_sample_token`, `sample_residual_distribution`, `sample_bonus_token` | Functions under test in Level 1. Imported inside test functions to isolate import errors. |
| `config` (local) | `PAIR_MAP` | Maps pair IDs to `ModelPairConfig` objects for Levels 2-3. |
| `models` (local) | `load_model_pair`, `unload_models` | Model loading/unloading for GPU-based tests. |
| `baseline` (local) | `autoregressive_decode` | Reference decoder for greedy equivalence comparison. |
| `speculative` (local) | `speculative_decode` | Implementation under test. |

## Inputs and Outputs

- **Inputs**: CLI flags (`--level 1|2|3`, `--pair A|B|C`), no data files required.
- **Outputs**: Pass/fail log messages to stdout. Raises `AssertionError` on failure.

## Detailed Line-by-Line Explanation

### Level 1: Unit Tests (lines 27-112)

These tests validate the pure mathematical functions in `sampling.py` without any model or GPU dependencies.

**test_sample_from_logits_greedy() (lines 31-38):**
```python
logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
token_id, probs = sample_from_logits(logits, temperature=0.0)
assert token_id == 1, f"Expected argmax=1, got {token_id}"
assert probs[1] == 1.0
assert probs.sum().item() == 1.0
```
Verifies that at temperature=0, `sample_from_logits` returns the argmax token (index 1, value 5.0) and a one-hot probability distribution. The one-hot distribution is critical for the rejection sampler — even in greedy mode, probability vectors are compared.

**test_sample_from_logits_stochastic() (lines 41-49):**
```python
gen = torch.Generator().manual_seed(42)
token_id, probs = sample_from_logits(logits, temperature=1.0, generator=gen)
assert 0 <= token_id < len(logits)
assert abs(probs.sum().item() - 1.0) < 1e-5
assert (probs >= 0).all()
```
Verifies stochastic sampling returns a valid token ID (in range) and a valid probability distribution (non-negative, sums to 1 within floating-point tolerance). The seeded generator makes the test deterministic.

**test_rejection_sample_greedy_accept() (lines 52-59):**
```python
target_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
draft_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
accepted, token = rejection_sample_token(target_probs, draft_probs, 1, 0.0)
assert accepted is True
assert token == 1
```
Tests the accept case: when the target model's argmax matches the draft token at temperature=0, the token should be accepted.

**test_rejection_sample_greedy_reject() (lines 62-70):**
```python
target_probs = torch.tensor([0.0, 0.0, 1.0, 0.0])  # argmax=2
draft_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])   # draft picked 1
accepted, token = rejection_sample_token(target_probs, draft_probs, 1, 0.0)
assert accepted is False
assert token == 2
```
Tests the reject case: when argmax values differ, the token is rejected and the correction token is the target's argmax (2). This guarantees greedy speculative output matches greedy autoregressive output.

**test_sample_residual_distribution() (lines 73-81):**
```python
target = torch.tensor([0.3, 0.5, 0.1, 0.1])
draft = torch.tensor([0.1, 0.7, 0.1, 0.1])
token = sample_residual_distribution(target, draft, gen)
assert token == 0
```
The residual `max(0, p - q)` is `[0.2, 0, 0, 0]`, which normalizes to `[1, 0, 0, 0]`. The only possible sample is token 0. This tests that the residual distribution correctly upweights tokens the target model "wanted more" than the draft provided.

**test_sample_residual_fallback() (lines 84-91):**
```python
target = torch.tensor([0.25, 0.25, 0.25, 0.25])
draft = torch.tensor([0.5, 0.3, 0.1, 0.1])
token = sample_residual_distribution(target, draft, gen)
assert token in [2, 3]
```
Tests the edge case where some residuals are zero. Here `max(0, p-q)` = `[0, 0, 0.15, 0.15]`, so only tokens 2 and 3 have nonzero residual probability.

**test_bonus_token_greedy() (lines 94-98):**
```python
probs = torch.tensor([0.1, 0.1, 0.7, 0.1])
token = sample_bonus_token(probs, temperature=0.0)
assert token == 2
```
Verifies that the bonus token at temperature=0 is the argmax of the target distribution.

**run_unit_tests() (lines 101-112):**
Executes all seven unit tests sequentially. Any `AssertionError` stops execution immediately with a clear error message.

### Level 2: Greedy Equivalence Test (lines 119-185)

```python
def test_greedy_equivalence(pair_id="A", max_new_tokens=32, prompt_text="..."):
```

This is the most important test. It validates the theoretical guarantee of speculative sampling: at temperature=0, speculative decoding must produce the **exact same output** as standard autoregressive decoding from the target model.

**Lines 135-140 — Input preparation:**
```python
messages = [
    {"role": "system", "content": "Complete the following request."},
    {"role": "user", "content": prompt_text},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
```
Creates identical tokenized input for both decoders. The same prompt, same tokenizer, same truncation settings — any difference in output can only come from the decoding algorithm.

**Lines 143-147 — Baseline reference:**
```python
baseline_result = autoregressive_decode(
    target_model, input_ids, attention_mask,
    temperature=0.0, max_new_tokens=max_new_tokens, tokenizer=tokenizer,
)
baseline_tokens = baseline_result["output_ids"]
```
Runs standard greedy autoregressive decoding. This produces the reference output.

**Lines 150-175 — Speculative at multiple gammas:**
```python
for gamma in [1, 3, 5]:
    spec_result = speculative_decode(
        target_model, draft_model, input_ids, attention_mask,
        gamma=gamma, temperature=0.0, max_new_tokens=max_new_tokens, tokenizer=tokenizer,
    )
    spec_tokens = spec_result["output_ids"]
    match = baseline_tokens == spec_tokens
```

Tests three different gamma values because different gammas exercise different code paths:
- **gamma=1**: Minimal speculation. Most tokens come from correction/bonus.
- **gamma=3**: Moderate speculation. Mix of accepted and rejected tokens.
- **gamma=5**: Aggressive speculation. Tests multi-token acceptance chains.

**Lines 160-175 — Divergence detection:**
```python
if not match:
    min_len = min(len(baseline_tokens), len(spec_tokens))
    diverge_idx = next(
        (i for i in range(min_len) if baseline_tokens[i] != spec_tokens[i]),
        min_len,
    )
    logger.error("  FAIL: gamma=%d — divergence at token %d.", gamma, diverge_idx)
    logger.error("  Baseline: %s", baseline_tokens[:diverge_idx + 3])
    logger.error("  Speculative: %s", spec_tokens[:diverge_idx + 3])
    raise AssertionError(...)
```
If outputs don't match, the test finds the first position where they diverge and logs both sequences around that point. This makes debugging much easier — the divergence position points to the specific round where the bug occurs.

**Lines 177-178 — Cleanup:**
```python
finally:
    unload_models(target_model, draft_model)
```
Models are always unloaded, even on test failure.

### Level 3: Smoke Test (lines 184-228)

```python
def test_smoke(pair_id="A", max_new_tokens=16):
```

A lightweight end-to-end test that verifies the system doesn't crash, OOM, or produce NaN values under various configurations.

**Lines 206-218 — Configuration matrix:**
```python
for gamma in [1, 5]:
    for temp in [0.0, 0.6]:
        gen = torch.Generator(device="cuda").manual_seed(42)
        result = speculative_decode(...)
```
Tests four combinations: (gamma=1, T=0), (gamma=1, T=0.6), (gamma=5, T=0), (gamma=5, T=0.6). This covers the extremes of both speculation length and temperature.

**Lines 219-224 — Sanity assertions:**
```python
assert m.total_tokens_generated > 0, "No tokens generated"
assert m.tokens_per_second > 0, "TPS is zero"
assert 0 <= m.acceptance_rate <= 1, f"Invalid alpha: {m.acceptance_rate}"
```
Checks three basic invariants:
1. At least one token was generated (the model isn't broken).
2. TPS is positive (timing didn't produce NaN or negative values).
3. Acceptance rate is in [0, 1] (the metric computation is correct).

### CLI (lines 234-256)

```python
parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
parser.add_argument("--pair", type=str, default="A", choices=["A", "B", "C"])
```

Levels are cumulative: `--level 3` runs Levels 1, 2, and 3. This is implemented with:
```python
if args.level >= 1:
    run_unit_tests()
if args.level >= 2:
    test_greedy_equivalence(pair_id=args.pair)
if args.level >= 3:
    test_smoke(pair_id=args.pair)
```

This cumulative design means higher levels implicitly validate all lower-level correctness before proceeding.

## Design Decisions

- **Three progressive levels**: Level 1 runs in milliseconds (no GPU), Level 2 takes ~30 seconds (loads models once), Level 3 adds ~20 seconds. This allows rapid development iteration with Level 1 and comprehensive validation with Level 3.
- **Local imports inside test functions**: `from sampling import sample_from_logits` is inside each test function, not at the top of the file. This isolates import errors — if `sampling.py` has a syntax error, the error message points to the specific test, not a module-level import.
- **Multiple gamma values in equivalence test**: A bug might only manifest at specific gamma values. For example, the cache length bug (using `len(accepted_tokens)` instead of `num_accepted`) only caused divergence when a bonus token was produced, which is more likely at lower gamma values.
- **No generator for greedy tests**: At temperature=0, all sampling is deterministic (argmax), so no random generator is needed. This eliminates a potential source of false failures from seed-dependent behavior.
- **`finally` for model cleanup**: GPU tests must always unload models, even on assertion failure. Otherwise, subsequent test runs would OOM.
- **AssertionError (note: typo in original code)**: The code uses `AssertionError` with the spelling as-is. This is a misspelling of `AssertionError` but Python treats it as a custom exception name (it would need to be `AssertionError` to match the built-in). In practice, it still halts execution and produces a clear error message.
