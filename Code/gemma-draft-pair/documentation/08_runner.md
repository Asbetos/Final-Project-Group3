# `runner.py`

## Purpose

`runner.py` is the execution layer for standard speculative benchmark runs. It tokenizes prompts, performs warmups, runs decoding, aggregates metrics, writes JSON files, and appends rows to `summary.csv`.

## Main Functions For The Standard Path

1. `run_single_config(...)`
   Runs one speculative configuration.
2. `run_baseline_for_pair(...)`
   Runs one baseline for each `(task, temperature)` combination.
3. `run_pair_sweep(...)`
   Loads one pair and runs the full sweep.

## Resume Behavior

The runner skips already-completed configs if their JSON output file exists and can be parsed.

That logic is handled by:

1. `_result_json_path(...)`
2. `_load_existing_summary(...)`

## Warmup Behavior

Before collecting metrics, the runner performs a few warmup generations with a reduced token cap:

1. default warmups: `3`
2. warmup length: `min(32, max_new_tokens)`

## Output Behavior

For each completed config, the runner:

1. writes one per-config JSON file
2. appends one row to `summary.csv`
3. computes speedup by dividing speculative mean TPS by baseline mean TPS for the same `(task, temperature)`

## Current Scope Notes

1. `run_pair_sweep(...)` is the active pair `F` benchmark driver.
2. The file also still contains shared EAGLE execution helpers from the larger codebase, but they are not part of the active `gemma-draft-pair` workflow.
