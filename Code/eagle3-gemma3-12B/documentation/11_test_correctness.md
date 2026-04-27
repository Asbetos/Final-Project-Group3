# `test_correctness.py`

## Purpose

`test_correctness.py` is the validation script for both decoding paths in this module.

It provides six test levels.

## Test Levels

1. Level 1
   Pure unit tests for `sampling.py`.
2. Level 2
   Greedy equivalence test for standard speculative decoding.
3. Level 3
   Standard speculative smoke test.
4. Level 4
   EAGLE-3 unit tests for tree logic.
5. Level 5
   EAGLE-3 greedy equivalence test with `tree_budget=1`.
6. Level 6
   EAGLE-3 smoke test.

## Active EAGLE Pair

The active EAGLE pair for the current project is `I`.

The CLI now accepts:

```bash
python3 test_correctness.py --level 6 --pair I
```

## Typical Validation Sequence

For the EAGLE workflow:

```bash
python3 test_correctness.py --level 1
python3 test_correctness.py --level 4
python3 test_correctness.py --level 6 --pair I
```

For the standard path:

```bash
python3 test_correctness.py --level 1
python3 test_correctness.py --level 3 --pair F
```

## What The EAGLE Tests Check

1. tree attention mask validity
2. path extraction from the draft tree
3. greedy equivalence against baseline when `tree_budget=1`
4. basic non-crashing generation for small EAGLE runs

## Current Scope Notes

1. The script still supports the standard pair IDs `F` and `G`.
2. For the current project scope, only pair `I` matters on the EAGLE side.
