## EAGLE-3 Gemma-3-12B Module

This folder contains my individual EAGLE-3 implementation work for the project, including the active code, run outputs, figures, and raw training/evaluation logs.

Layout overview:

1. `core/` - reusable implementation modules
2. `scripts/` - CLI entrypoints
3. `apps/` - Streamlit UI
4. `docs/` - module-by-module written references
5. `artifacts/` - logs, results, and figures
6. `archive/` - legacy material kept for reference

Active EAGLE pair:

1. Pair `I`
2. Target: `google/gemma-3-12b-it`
3. Draft component: trained EAGLE-3 draft head

## Active Configuration

| Pair | Method | Target Model | Draft Component | Quantization | Estimated VRAM |
| --- | --- | --- | --- | --- | --- |
| `I` | EAGLE-3 | `google/gemma-3-12b-it` | trained EAGLE-3 draft head | 4-bit target | ~6.6 GB target + head overhead |

## Hardware

1. Evaluation target: AWS `g5.2xlarge` (`A10G`, 24 GB VRAM) is sufficient.
2. System RAM: `32 GB+`
3. Disk: `60 GB+` recommended for caches, checkpoints, and results.

## Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r venkatesh-nagarjuna-individual-project/Code/requirements.txt
cd venkatesh-nagarjuna-individual-project/Code
```

Gemma downloads may require Hugging Face access approval and authentication.

## Final Checkpoint

The completed training run wrote the final checkpoint locally at:

```text
checkpoints/eagle3/gemma3_12b/eagle3_gemma3_12b_final.pt
```

The raw proof for that run is preserved in `artifacts/logs/training.log`. The checkpoint binary itself is not included in this folder snapshot.

Set the checkpoint path before running evaluation if you have the local checkpoint available:

```bash
export EAGLE3_GEMMA3_CHECKPOINT="$PWD/checkpoints/eagle3/gemma3_12b/eagle3_gemma3_12b_final.pt"
```

## Quick Validation

```bash
# Unit tests
python3 scripts/test_correctness.py --level 1

# EAGLE-3 unit tests
python3 scripts/test_correctness.py --level 4

# EAGLE-3 smoke test on the active Gemma-3 pair
python3 scripts/test_correctness.py --level 6 --pair I

# Preview the EAGLE-3 evaluation grid
python3 scripts/sweep.py --dry-run --eagle3 --eagle3-only --eagle3-pairs I
```

## Run EAGLE-3 Benchmarks

```bash
python3 scripts/sweep.py \
  --eagle3 --eagle3-only --eagle3-pairs I \
  --tree-budgets 20 60 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir artifacts/results/eagle3_gemma3_full
```

## Optional Smaller Calibration Run

```bash
python3 scripts/sweep.py \
  --eagle3 --eagle3-only --eagle3-pairs I \
  --tree-budgets 20 \
  --temps 0.0 \
  --tasks humaneval triviaqa \
  --num-prompts 10 \
  --max-tokens 128 \
  --output-dir artifacts/results/eagle3_gemma3_calibration
```

## Unified Chat Comparison App

This module also contains a Streamlit app that compares:

1. baseline autoregressive decoding on the left
2. one selected acceleration method on the right

The right-hand chat can switch between:

1. standard speculative decoding with pair `F`
2. EAGLE-3 decoding with pair `I`

Run it with:

```bash
streamlit run apps/app.py --server.port 8501 --server.address 0.0.0.0
```

The app keeps models warm in memory with `st.cache_resource` and stores downloaded Hugging Face artifacts under a local `.hf-cache/` directory so they do not need to be fetched again on every restart.

Deployment note:

1. the app is Streamlit-compatible
2. truly fast warm-model serving requires a persistent GPU process
3. Vercel is better used as a thin frontend or proxy than as the actual model host for Gemma-12B inference

## Outputs

```text
artifacts/results/<run_name>/
  baseline/
  eagle3/
  summary.csv

checkpoints/
  eagle3/
    gemma3_12b/
      eagle3_gemma3_12b_final.pt
```

## Notes

1. Historical or archived artifacts may still be present in this folder.
2. Use the repository root `README.md` for the two-module project overview.
3. `artifacts/logs/training.log` is the source-of-truth record for the completed training run.

## Push-Ready Artifact Guide

For a compact inventory of the code I worked on plus the training/evaluation proof artifacts, see `ARTIFACTS.md`.

Raw proof logs included in this folder:

1. `artifacts/logs/training.log`
2. `artifacts/logs/eval_eagle3_gemma3_full.log`
3. `artifacts/logs/cnn_rerun_fix.log`
