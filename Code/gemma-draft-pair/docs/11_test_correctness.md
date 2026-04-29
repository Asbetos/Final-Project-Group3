# `scripts/test_correctness.py`

## Purpose

`scripts/test_correctness.py` validates the standard speculative decoding implementation. It delegates to `core/test_correctness.py`.

## Main Test Levels Used By This Module

1. Level 1
   Unit tests for the sampling math.
2. Level 2
   Greedy equivalence against baseline.
3. Level 3
   Short smoke tests.

## Typical Validation Sequence For Pair `F`

```bash
python3 scripts/test_correctness.py --level 1
python3 scripts/test_correctness.py --level 3 --pair F
```

## What The Tests Check

1. sampling correctness
2. residual distribution behavior
3. greedy equivalence between baseline and speculative decoding
4. short end-to-end runs that check for crashes, invalid metrics, or NaNs

## Current Scope Notes

1. The cleaned module exposes only the standard speculative test levels.
2. Pair `F` is the active validation target.
3. Pair `G` remains selectable.
4. The root project README documents the recommended run order for this module.
