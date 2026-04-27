# Gemma Draft Pair Module

This module benchmarks standard speculative decoding for the active project pair:

1. Pair `F`
2. Target: `google/gemma-3-12b-it`
3. Draft: `google/gemma-3-1b-it`

The main entrypoint is `sweep.py`.

## Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Code/gemma-draft-pair/requirements.txt
```

Then switch into this module:

```bash
cd Code/gemma-draft-pair
```

## Quick Validation

```bash
python3 test_correctness.py --level 1
python3 test_correctness.py --level 3 --pair F
python3 sweep.py --dry-run --pairs F
```

## Full Evaluation Run

```bash
python3 sweep.py --pairs F --output-dir gemma_runs/outputs/F_rerun
```

## Full Project Grid For Pair F

The saved final pair `F` results were produced across:

1. `gamma`: `1, 3, 5, 7, 10`
2. `temperature`: `0.0, 0.6, 1.0`
3. `task`: `humaneval`, `triviaqa`, `cnn_dailymail`, `writingprompts`
4. `num_prompts`: `50`
5. `max_new_tokens`: `128`

Equivalent command:

```bash
python3 sweep.py \
  --pairs F \
  --gammas 1 3 5 7 10 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir gemma_runs/outputs/F_full
```

## Outputs

```text
gemma_runs/outputs/<run_name>/
  baseline/
  speculative/
  summary.csv
```

Saved project artifacts:

1. `gemma_runs/outputs/F_final/summary.csv`
2. `figures/FG_final/`

## Notes

1. Some historical comparison artifacts for pair `G` are still present in this folder, but pair `F` is the active project scope.
2. Use the repository root `README.md` for the full two-module project overview.
