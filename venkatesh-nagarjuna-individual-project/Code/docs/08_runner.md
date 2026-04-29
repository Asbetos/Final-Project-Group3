# `core/runner.py`

## Purpose

`core/runner.py` is the execution layer for experiment runs. It loads prompts, performs warmups, executes decoding, aggregates metrics, saves JSON, and appends to the master CSV.

## Standard Path Functions

1. `run_single_config(...)`
   Runs one standard speculative configuration.
2. `run_baseline_for_pair(...)`
   Runs one baseline pass for each `(task, temperature)` combination.
3. `run_pair_sweep(...)`
   Loads one target-draft pair and executes the full standard sweep.

## EAGLE Path Functions

1. `run_single_eagle3_config(...)`
   Runs one EAGLE configuration.
2. `run_eagle3_pair_sweep(...)`
   Loads one EAGLE pair, runs baselines, then runs the EAGLE grid.

## Resume Behavior

The runner supports resume-by-results-file:

1. `_result_json_path(...)` builds the expected JSON filename.
2. `_load_existing_summary(...)` checks whether a config has already completed.
3. If a JSON result exists, that configuration is skipped.

This is the main mechanism that lets long sweeps continue safely after interruption.

## Warmup Behavior

Before measuring each config, the runner performs a few warmup generations with reduced token count:

1. default warmup count: `3`
2. warmup max tokens: `min(32, max_new_tokens)`

## CSV Conventions

Standard speculative rows include:

1. `gamma`
2. `is_baseline`
3. `speedup`

EAGLE rows additionally include:

1. `tree_budget`
2. `is_eagle3=True`

## Current Scope Notes

1. `run_eagle3_pair_sweep(...)` is the active function for the current project evaluation.
2. The active EAGLE pair is `I`.
3. Baselines are always run first so speedup can be computed immediately after each accelerated config finishes.
