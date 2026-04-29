# `config.py`

## Purpose

`config.py` defines the benchmark grid and pair metadata for the standard speculative decoding module.

## Active Scope

The current project scope is pair `F`:

1. target: `google/gemma-3-12b-it`
2. draft: `google/gemma-3-1b-it`

Pair `G` still exists in code and saved outputs, but it is no longer the main focus of the project write-up.

## Main Types

1. `ModelPairConfig`
   Stores the pair ID, target model, draft model, quantization flags, and VRAM estimates.
2. `ExperimentConfig`
   Stores one benchmark cell: pair, gamma, temperature, task, and runtime defaults.

## Important Constants

1. `PAIR_F`
   Gemma-3-12B target plus Gemma-3-1B draft.
2. `PAIR_G`
   Gemma-4-31B target plus Gemma-3-1B draft.
3. `PAIR_MAP`
   Lookup table from pair ID to config.

## Grid Defaults In This File

Current code defaults are:

1. `ALL_GAMMAS = [1, 5, 10]`
2. `ALL_TEMPERATURES = [0.0]`
3. `ALL_TASKS = ["humaneval", "triviaqa", "cnn_dailymail", "writingprompts"]`

## Important Scope Note

The saved final pair `F` results in the repository were produced with a broader sweep than the current default constants:

1. `gamma = 1, 3, 5, 7, 10`
2. `temperature = 0.0, 0.6, 1.0`

So this file reflects the current default CLI behavior, while the archived final outputs reflect a larger completed run.

## Helper Function

1. `build_grid(...)`
   Builds the Cartesian product of selected pairs, gammas, temperatures, and tasks.

## Shared Legacy Content

This file still includes EAGLE config dataclasses and a legacy pair `H` definition because code was shared with the EAGLE module at one point. Those definitions are not part of the active `gemma-draft-pair` workflow.
