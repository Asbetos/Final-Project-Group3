# LLM Inference Acceleration using Speculative Decoding and EAGLE-3

This repository contains two active decoding-acceleration modules for Gemma models:

1. `Code/gemma-draft-pair/`
   Standard speculative decoding for pair `F`.
   Target: `google/gemma-3-12b-it`
   Draft: `google/gemma-3-1b-it`

2. `Code/eagle3-gemma3-12B/`
   EAGLE-3 draft-head training and inference for a Gemma-3-12B target.
   Active EAGLE pair ID: `I`
   Target: `google/gemma-3-12b-it`
   Draft component: trained EAGLE-3 draft head

The goal of the project is to measure how much decoding can be accelerated while keeping the target model responsible for final token verification.

## What Order To Run The Code In

For a fresh setup, the code should be run in this order:

1. Create the environment and install dependencies.
2. Ensure Hugging Face authentication is available for Gemma checkpoints.
3. Run correctness checks for the module you want to use.
4. Run the evaluation sweep for `gemma-draft-pair` or `eagle3-gemma3-12B`.
5. Inspect the generated `summary.csv` and per-config JSON outputs.
6. Optionally generate plots or use the Streamlit app.

If you want to reproduce the full EAGLE workflow from scratch, the order is:

1. Setup environment.
2. Train the EAGLE-3 head with `eagle3_train.py`.
3. Export the final checkpoint path.
4. Run EAGLE-3 correctness checks.
5. Run the EAGLE-3 evaluation sweep.

## Common Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Code/gemma-draft-pair/requirements.txt
```

The two module `requirements.txt` files are aligned, so a single environment is enough.

You also need:

1. Hugging Face access for Gemma models.
2. A logged-in Hugging Face CLI or equivalent token setup.
3. A GPU machine with enough VRAM.

Recommended hardware:

1. `gemma-draft-pair`: A10G-class GPU is sufficient.
2. `eagle3-gemma3-12B` evaluation: A10G-class GPU is sufficient.
3. System RAM: `32 GB+`
4. Disk: `60 GB+` recommended.

## Repository Layout

```text
Code/
  gemma-draft-pair/
  eagle3-gemma3-12B/

Final-Group-Project-Report/
Final-Group-Presentation/
docs/
```

## Module 1: `Code/gemma-draft-pair`

### What This Module Does

This module benchmarks standard speculative decoding for pair `F`.

### Run Order

1. Enter the module folder.
2. Run validation tests.
3. Preview the grid with `--dry-run`.
4. Run the full sweep.
5. Inspect `gemma_runs/outputs/<run_name>/summary.csv`.

### Commands

```bash
cd Code/gemma-draft-pair
source ../../venv/bin/activate

# Validation
python3 test_correctness.py --level 1
python3 test_correctness.py --level 3 --pair F
python3 sweep.py --dry-run --pairs F

# Full evaluation
python3 sweep.py \
  --pairs F \
  --gammas 1 3 5 7 10 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir gemma_runs/outputs/F_full
```

### Important Scripts

1. `config.py`
   Defines pair `F`, the evaluation grid, and runtime defaults.
2. `data.py`
   Loads prompts from Hugging Face datasets and formats them for Gemma chat generation.
3. `models.py`
   Loads the target and draft models.
4. `baseline.py`
   Runs standard autoregressive decoding for baseline measurements.
5. `speculative.py`
   Implements standard speculative decoding.
6. `runner.py`
   Executes one full pair sweep and writes results.
7. `sweep.py`
   Main CLI entrypoint for the full benchmark grid.
8. `test_correctness.py`
   Unit tests, greedy equivalence checks, and smoke tests.
9. `visualize.py`
   Generates plots from saved result summaries.
10. `app.py`
   Optional Streamlit demo.

### Outputs

```text
gemma_runs/outputs/<run_name>/
  baseline/
  speculative/
  summary.csv
```

## Module 2: `Code/eagle3-gemma3-12B`

### What This Module Does

This module contains the EAGLE-3 implementation, training pipeline, and evaluation sweep for the active Gemma-3-12B draft-head run.

### Run Order

If the model is already trained:

1. Enter the module folder.
2. Export the checkpoint path.
3. Run correctness checks.
4. Preview the grid.
5. Run the full EAGLE-3 evaluation.

If training from scratch:

1. Run `eagle3_train.py` first.
2. Confirm the final checkpoint exists.
3. Export `EAGLE3_GEMMA3_CHECKPOINT`.
4. Run the EAGLE-3 validation and sweep commands below.

### Commands

```bash
cd Code/eagle3-gemma3-12B
source ../../venv/bin/activate

export EAGLE3_GEMMA3_CHECKPOINT="$PWD/checkpoints/eagle3/gemma3_12b/eagle3_gemma3_12b_final.pt"

# Validation
python3 test_correctness.py --level 1
python3 test_correctness.py --level 4
python3 test_correctness.py --level 6 --pair I
python3 sweep.py --dry-run --eagle3 --eagle3-only --eagle3-pairs I

# Full evaluation
python3 sweep.py \
  --eagle3 --eagle3-only --eagle3-pairs I \
  --tree-budgets 20 60 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir results/eagle3_gemma3_full
```

### Optional Training Command

```bash
python3 eagle3_train.py \
  --target-model google/gemma-3-12b-it \
  --target-4bit \
  --checkpoint-dir checkpoints/eagle3/gemma3_12b \
  --final-checkpoint-name eagle3_gemma3_12b_final.pt
```

### Important Scripts

1. `config.py`
   Defines the active EAGLE pair `I`, checkpoint environment variables, and the EAGLE grid.
2. `data.py`
   Loads benchmark prompts and EAGLE training data.
3. `models.py`
   Loads the target model and EAGLE draft head.
4. `eagle3.py`
   Implements the EAGLE-3 draft head and decode logic.
5. `eagle3_train.py`
   Trains the draft head against the frozen target model.
6. `baseline.py`
   Runs baseline autoregressive decoding for comparison.
7. `runner.py`
   Executes the EAGLE-3 evaluation sweep and writes results.
8. `sweep.py`
   Main CLI entrypoint for EAGLE-3 evaluation.
9. `test_correctness.py`
   Unit tests, EAGLE unit tests, equivalence tests, and smoke tests.
10. `visualize.py`
   Generates figures from saved summaries.
11. `app.py`
   Optional Streamlit app.

### Outputs

```text
results/<run_name>/
  baseline/
  eagle3/
  summary.csv

checkpoints/eagle3/gemma3_12b/
  eagle3_gemma3_12b_final.pt
```

## Model Release

The final trained EAGLE-3 draft head for `Gemma-3-12b-it` is also available on Hugging Face:

[planethunter98/eagle3-head-gemma3-12b-it](https://huggingface.co/planethunter98/eagle3-head-gemma3-12b-it)

## Where To Look Next

1. Module-specific instructions:
   `Code/gemma-draft-pair/README.md`
   `Code/eagle3-gemma3-12B/README.md`
2. Final report:
   `Final-Group-Project-Report/`
3. Final presentation:
   `Final-Group-Presentation/`
