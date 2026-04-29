# `speculative.py`

## Purpose

`speculative.py` contains the full standard speculative decoding implementation used by this module.

## Main Components

1. `_get_cache_seq_len(...)`
   Reads sequence length from multiple KV-cache representations.
2. `_trim_kv_cache(...)`
   Crops KV cache after partial acceptance.
3. `_draft_step(...)`
   Generates `gamma` draft tokens from the small model.
4. `_verify_step(...)`
   Runs a single target-model verification pass over the drafted block.
5. `speculative_decode(...)`
   Executes the full prompt-level decode loop.

## Implementation Features

The decoder is optimized around:

1. pre-allocated sequence buffers
2. in-place token writes
3. KV-cache reuse for target and draft models
4. vectorized batch rejection sampling

## Metrics Recorded Per Prompt

1. total generated tokens
2. total rounds
3. TTFT
4. tokens per second
5. acceptance rate
6. acceptance length
7. draft overhead ratio
8. peak VRAM

## How The Loop Works

For each round:

1. the draft model proposes up to `gamma` tokens
2. those tokens are written into a pre-allocated buffer
3. the target model verifies them in one forward pass
4. accepted tokens and any correction token are appended to the output
5. both caches are trimmed to the accepted prefix

## Current Scope Notes

1. This is the core file for pair `F` benchmarking.
2. The code still supports pair `G` as well.
3. The active project report focuses on the results of pair `F`.
