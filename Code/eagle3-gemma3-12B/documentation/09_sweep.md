# `sweep.py`

## Purpose

`sweep.py` is the main CLI entrypoint for benchmark execution.

It supports:

1. standard speculative sweeps
2. EAGLE sweeps
3. `--dry-run` grid previews
4. `--eagle3-only` execution

## Active EAGLE Command

The current project uses the EAGLE path with pair `I`:

```bash
export EAGLE3_GEMMA3_CHECKPOINT="$PWD/checkpoints/eagle3/gemma3_12b/eagle3_gemma3_12b_final.pt"
python3 sweep.py \
  --eagle3 --eagle3-only --eagle3-pairs I \
  --tree-budgets 20 60 \
  --temps 0.0 0.6 1.0 \
  --tasks humaneval triviaqa cnn_dailymail writingprompts \
  --num-prompts 50 \
  --max-tokens 128 \
  --output-dir results/eagle3_gemma3_full
```

## Important Flags

### Shared flags

1. `--tasks`
2. `--temps`
3. `--num-prompts`
4. `--max-tokens`
5. `--output-dir`
6. `--seed`
7. `--dry-run`

### Standard speculative flags

1. `--pairs`
2. `--gammas`

### EAGLE flags

1. `--eagle3`
2. `--eagle3-only`
3. `--eagle3-pairs`
4. `--tree-budgets`

## What `main()` Does

1. Configures CUDA-friendly math settings.
2. Parses CLI arguments.
3. Computes and logs the size of the requested grid.
4. Prints the full grid if `--dry-run` is set.
5. Creates output folders.
6. Dispatches to `run_pair_sweep(...)` and or `run_eagle3_pair_sweep(...)`.

## Current Scope Notes

1. The dry-run and grid-count logging were updated to handle `--eagle3-only` correctly.
2. `--eagle3-pairs` still allows both `H` and `I` because legacy support remains in code.
3. For the current project, use pair `I` only.
