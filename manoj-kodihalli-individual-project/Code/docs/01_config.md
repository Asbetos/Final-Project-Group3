# `core/config.py`

## Purpose

`core/config.py` is the central configuration module for both decoding paths that still exist in this folder:

1. Standard speculative decoding pairs.
2. EAGLE-3 pairs.

For the active project scope, the important EAGLE entry is pair `I`:

1. Pair `I`
2. Target: `google/gemma-3-12b-it`
3. Checkpoint environment variable: `EAGLE3_GEMMA3_CHECKPOINT`

## What It Defines

### Standard speculative types

1. `ModelPairConfig`
   Describes a target model, draft model, and VRAM estimates.
2. `ExperimentConfig`
   Describes one standard speculative benchmark cell.

### EAGLE-3 types

1. `Eagle3PairConfig`
   Describes an EAGLE target model plus draft-head checkpoint path.
2. `Eagle3ExperimentConfig`
   Describes one EAGLE evaluation cell.

## Active Constants

### Standard path

1. `PAIR_F`
   Gemma-3-12B target plus Gemma-3-1B draft.
2. `PAIR_G`
   Gemma-4-31B target plus Gemma-3-1B draft.

These remain in code because the folder still supports the shared standard speculative path, but they are not the main focus of the current EAGLE module.

### EAGLE path

1. `EAGLE3_PAIR_H`
   Legacy Gemma-4 entry still present in code.
2. `EAGLE3_PAIR_I`
   Active Gemma-3-12B EAGLE pair.

## Active Evaluation Grid

Standard speculative grid defaults:

1. `ALL_GAMMAS = [1, 5, 10]`
2. `ALL_TEMPERATURES = [0.0, 0.6, 1.0]`
3. `ALL_TASKS = ["humaneval", "triviaqa", "cnn_dailymail", "writingprompts"]`

EAGLE grid defaults:

1. `ALL_TREE_BUDGETS = [20, 60]`
2. Same temperature list as above.
3. Same task list as above.

## Important Functions

1. `build_grid(...)`
   Builds the standard speculative grid.
2. `build_eagle3_grid(...)`
   Builds the EAGLE grid.

Both functions are simple Cartesian-product builders used by the CLI and runner logic.

## Current Scope Notes

1. The documentation and project report treat pair `I` as the active EAGLE workflow.
2. Pair `H` is still defined in code for compatibility but is not the active project target.
3. The checkpoint path for pair `I` is automatically read from `EAGLE3_GEMMA3_CHECKPOINT`, with a local default under `checkpoints/eagle3/gemma3_12b/`.
