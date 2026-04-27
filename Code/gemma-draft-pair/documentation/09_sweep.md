# `sweep.py`

## Purpose

`sweep.py` is the CLI entrypoint for benchmark execution.

For the active project scope, it is used to run pair `F` speculative decoding experiments.

## Main CLI Flow

1. parse arguments
2. compute and log the grid size
3. optionally print the full run matrix with `--dry-run`
4. create the output directory structure
5. dispatch each selected pair to `run_pair_sweep(...)`

## Active Pair `F` Command

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

## Important Flags

1. `--pairs`
2. `--gammas`
3. `--temps`
4. `--tasks`
5. `--num-prompts`
6. `--max-tokens`
7. `--output-dir`
8. `--seed`
9. `--dry-run`

## Important Scope Note

The current `config.py` defaults are narrower than the saved final pair `F` run in the repository. So if you want to reproduce the final project outputs, explicitly pass the broader gamma and temperature lists.

## Shared Legacy Content

The CLI still includes EAGLE-related flags because this file was shared with the broader decoding codebase. Those flags are not part of the active `gemma-draft-pair` run instructions.
