# Gemma Draft Pair Benchmark Module

This folder contains my individual Gemma draft-pair benchmark work for the final project.

It includes the active code, benchmark outputs, and final figures used to evaluate standard speculative decoding for Gemma Pair F and Pair G.

## Layout Overview

1. `core/` - main source code for the Gemma draft-pair benchmark pipeline
2. `artifacts/` - saved outputs, figures, and archived run artifacts
3. `requirements.txt` - Python package requirements for running this module
4. `README.md` - setup, validation, and run instructions for this module
5. `ARTIFACTS.md` - detailed inventory of included code files and result artifacts

## Gemma Pairs

Main benchmark pairs:

1. Pair `F`
   - Target: `google/gemma-3-12b-it`
   - Draft: `google/gemma-3-1b-it`

2. Pair `G`
   - Target: `google/gemma-4-31b-it`
   - Draft: `google/gemma-3-1b-it`

The main entrypoint is `core/sweep.py`.

For the detailed file-by-file inventory, see `ARTIFACTS.md`.

## Setup

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r jeongmin-an-individual-project/Code/requirements.txt
cd jeongmin-an-individual-project/Code
```

## Quick Validation

```bash
python3 core/test_correctness.py --level 1
python3 core/test_correctness.py --level 3 --pair F
python3 core/sweep.py --dry-run --pairs F
```
These commands check that the validation script runs, Pair F can be selected, and the sweep configuration can be generated without launching the full benchmark.

To check both Pair F and Pair G configurations:

```bash
python3 core/sweep.py --dry-run --pairs F G
```


## Full Project Grid

The saved final results were produced across:

1. `gamma`: `1, 3, 5, 7, 10`
2. `temperature`: `0.0, 0.6, 1.0`
3. `task`: `humaneval`, `triviaqa`, `cnn_dailymail`, `writingprompts`
4. `num_prompts`: `50`
5. `max_new_tokens`: `128`

Equivalent full Pair F command:

```bash
python3 core/sweep.py \
  --pairs F \
  --gammas 1 3 5 7 10 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir artifaacts/outputs/F_final
```

Equivalent full Pair G command:

```bash
python3 core/sweep.py \
  --pairs G \
  --gammas 1 3 5 7 10 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir artifacts/outputs/archive/G_final
```

## Outputs

```text
artifacts/outputs/<run_name>/
  baseline/
  speculative/
  summary.csv
```

Saved project artifacts:

1. `artifacts/outputs/F_final/summary.csv`
2. `artifacts/outputs/archive/G_final/summary.csv`
3. `figures/FG_final/`


## Figures
Final figures are stored in:
 `artifacts/figures/FG_final/`

Older or backup figures are stored in:
 `artifacts/figures/previous_figures/`


## Notes

1. Pair F is the main final benchmark output included in this folder.
2. Pair G is included in the analysis as a comparison pair for speedup, TTFT, acceptance behavior, and VRAM usage.
3. Historical or backup outputs are kept under `artifacts/outputs/archive/`.
4. Large model weights, Hugging Face cache files, virtual environments, and temporary AWS logs are not included.
5. For a detailed inventory of the included code files, outputs, and figures, see `ARTIFACTS.md`.
