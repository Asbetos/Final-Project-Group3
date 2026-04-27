# `test_correctness.py`

## Purpose

`test_correctness.py` validates the standard speculative decoding implementation.

## Main Test Levels Used By This Module

1. Level 1
   Unit tests for the sampling math.
2. Level 2
   Greedy equivalence against baseline.
3. Level 3
   Short smoke tests.

## Typical Validation Sequence For Pair `F`

```bash
python3 test_correctness.py --level 1
python3 test_correctness.py --level 3 --pair F
```

## What The Tests Check

1. sampling correctness
2. residual distribution behavior
3. greedy equivalence between baseline and speculative decoding
4. short end-to-end runs that check for crashes, invalid metrics, or NaNs

## Shared Legacy Content

This file still contains EAGLE-related tests because it was originally shared with the larger combined codebase. Those EAGLE test levels are not part of the active `gemma-draft-pair` validation workflow.

## Current Scope Notes

1. Pair `F` is the active validation target.
2. Pair `G` remains selectable.
3. The root project README documents the recommended run order for this module.
