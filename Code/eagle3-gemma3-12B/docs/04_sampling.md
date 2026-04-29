# `core/sampling.py`

## Purpose

`core/sampling.py` contains the pure probability and rejection-sampling helpers used by both standard speculative decoding and EAGLE verification.

This file has no model-loading logic. It only operates on logits and probability tensors.

## Core Functions

1. `_safe_multinomial(...)`
   Samples while handling generator and tensor device mismatches.
2. `_safe_rand(...)`
   Creates random tensors on the generator's device.
3. `sample_from_logits(logits, temperature, generator=None)`
   Converts logits to a sampled token and distribution.
4. `rejection_sample_token(...)`
   Verifies a single draft token against the target distribution.
5. `batch_rejection_sample(...)`
   Vectorized version of rejection sampling for a whole drafted block.
6. `sample_residual_distribution(...)`
   Samples from the residual correction distribution `max(0, p - q)`.
7. `sample_bonus_token(...)`
   Samples the extra target token when all drafted tokens are accepted.

## Greedy vs Stochastic Behavior

At `temperature = 0.0`:

1. Sampling becomes argmax.
2. Rejection sampling becomes exact token equality against the target argmax.

At `temperature > 0`:

1. The draft token is accepted with `min(1, p / q)`.
2. Rejected tokens are corrected from the residual distribution.

## Why This File Matters

This is the mathematical core that preserves the target model distribution while allowing faster candidate generation.

## Current Scope Notes

1. The same functions are reused by standard speculative decoding and EAGLE verification.
2. The vectorized batch path is especially important for the standard speculative implementation in `core/speculative.py`.
