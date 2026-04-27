# `sampling.py`

## Purpose

`sampling.py` implements the probability-level operations that make speculative decoding correct.

## Main Functions

1. `align_shared_vocab(...)`
   Restricts target and draft distributions to their shared vocabulary prefix.
2. `sample_from_logits(...)`
   Samples one token from logits or returns the argmax in greedy mode.
3. `rejection_sample_token(...)`
   Verifies one draft token against the target distribution.
4. `batch_rejection_sample(...)`
   Vectorized rejection sampling for a whole drafted block.
5. `sample_residual_distribution(...)`
   Samples from the residual correction distribution.
6. `sample_bonus_token(...)`
   Samples the extra target token when all draft tokens are accepted.

## Why `align_shared_vocab(...)` Exists

The standard speculative module compares models from different Gemma families in some saved experiments. If target and draft vocab sizes differ, direct subtraction of distributions breaks. This helper slices both distributions to the shared prefix and renormalizes them.

## Greedy vs Stochastic Modes

At `temperature = 0.0`:

1. sampling becomes argmax
2. rejection becomes exact equality against the target argmax

At `temperature > 0`:

1. acceptance uses `min(1, p / q)`
2. rejected tokens are corrected from the residual distribution

## Current Scope Notes

1. These helpers are the mathematical core of the standard speculative path.
2. The vectorized batch path is used in `_verify_step(...)` inside `speculative.py`.
