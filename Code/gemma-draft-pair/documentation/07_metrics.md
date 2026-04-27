# `metrics.py`

## Purpose

`metrics.py` defines the metric containers, timing helpers, and result serialization helpers used by the module.

## Main Data Structures

1. `RoundMetrics`
   Stores one draft-and-verify round.
2. `GenerationMetrics`
   Stores one full prompt generation.

`GenerationMetrics.aggregate(...)` converts many prompt-level generations into mean, standard deviation, and p95 summaries.

## Timing Helpers

1. `CudaTimer`
   Measures GPU time using CUDA events.
2. `WallTimer`
   Measures end-to-end wall-clock time.

## VRAM Helpers

1. `reset_peak_vram()`
2. `record_peak_vram()`

These are used to report peak GPU memory per generation.

## Result Writers

1. `save_results_json(...)`
   Writes config, summary, and per-prompt details to JSON.
2. `save_summary_csv(...)`
   Appends one result row to the master CSV.

## Output Structure

Typical outputs look like:

```text
gemma_runs/outputs/<run_name>/
  baseline/
  speculative/
  summary.csv
```

## Current Scope Notes

1. The CSV writer can extend the header if new fields appear.
2. That behavior matters because the broader shared codebase also supports EAGLE rows with extra columns.
