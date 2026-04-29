# Gemma Draft Pair Module

This module benchmarks standard speculative decoding for the active project pair:

1. Pair `F`
2. Target: `google/gemma-3-12b-it`
3. Draft: `google/gemma-3-1b-it`

The main entrypoint is `scripts/sweep.py`.

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
python3 scripts/test_correctness.py --level 1
python3 scripts/test_correctness.py --level 3 --pair F
python3 scripts/sweep.py --dry-run --pairs F
```

## Full Evaluation Run

```bash
python3 scripts/sweep.py --pairs F --output-dir artifacts/results/F_rerun
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
python3 scripts/sweep.py \
  --pairs F \
  --gammas 1 3 5 7 10 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir artifacts/results/F_full
```

## Streamlit Demo

```bash
streamlit run apps/app.py --server.port 8501 --server.address 0.0.0.0
```

## Outputs

```text
artifacts/results/<run_name>/
  baseline/
  speculative/
  summary.csv
```

Saved project artifacts:

1. `artifacts/results/F_final/summary.csv`
2. `artifacts/results/G_final/summary.csv`
3. `artifacts/figures/final/`

## Notes

1. Pair `F` is the active project scope used in the final report.
2. Smoke runs and older residual artifacts were moved into `archive/`.
3. Pair `G` remains supported in the codebase as a larger comparison path.
4. Use the repository root `README.md` for the full two-module project overview.
