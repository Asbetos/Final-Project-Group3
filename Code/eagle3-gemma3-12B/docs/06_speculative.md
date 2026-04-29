# `core/speculative.py`

## Purpose

`core/speculative.py` contains the standard speculative decoding implementation used for target-plus-draft model pairs.

In the active project scope, this is not the main EAGLE evaluation path, but it remains part of the shared module because:

1. the repo still supports pair-based speculative experiments
2. the Streamlit app uses it
3. many shared timing and comparison patterns were developed here first

## Main Components

1. `_get_cache_seq_len(...)`
   Reads the sequence length from multiple KV-cache formats.
2. `_trim_kv_cache(...)`
   Crops caches after partial acceptance.
3. `_draft_step(...)`
   Runs the small draft model autoregressively for `gamma` tokens.
4. `_verify_step(...)`
   Runs one target forward pass over the drafted span and applies vectorized rejection sampling.
5. `speculative_decode(...)`
   Orchestrates the full loop for one prompt.

## What The Decoder Tracks

Each generation records:

1. total generated tokens
2. total rounds
3. TTFT
4. tokens per second
5. acceptance rate
6. acceptance length
7. draft overhead ratio
8. peak VRAM

## Implementation Notes

The implementation is optimized around:

1. pre-allocated token buffers
2. KV-cache reuse
3. vectorized verification
4. avoiding repeated `torch.cat` inside the hot loop

## Current Scope Notes

1. This file documents the standard speculative path only.
2. EAGLE generation does not call `speculative_decode(...)`; it uses `eagle3_decode(...)` from `core/eagle3.py`.
3. Pair `F` and pair `G` support still exists in this code, even though the active EAGLE project write-up focuses on pair `I`.
