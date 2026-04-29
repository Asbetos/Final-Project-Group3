# `core/metrics.py`

## Purpose

`core/metrics.py` defines the metric containers, timing helpers, and output writers used across the whole module.

## Main Data Structures

1. `RoundMetrics`
   Stores one draft-and-verify round.
2. `GenerationMetrics`
   Stores one full prompt generation.

`GenerationMetrics.aggregate(...)` converts a list of prompt-level metrics into mean, standard deviation, and p95 summaries.

## Timing Helpers

1. `CudaTimer`
   Uses CUDA events for GPU-accurate timing.
2. `WallTimer`
   Uses wall-clock timing for end-to-end latency.

## VRAM Helpers

1. `reset_peak_vram()`
2. `record_peak_vram()`

These are called before and after generation runs so the JSON and CSV outputs contain peak memory usage.

## Serialization Helpers

1. `save_results_json(...)`
   Writes config, summary, and per-prompt metrics to a JSON file.
2. `save_summary_csv(...)`
   Appends one configuration summary row to the master CSV.

## Output Layout

Typical outputs look like:

```text
artifacts/results/<run_name>/
  baseline/*.json
  speculative/*.json
  eagle3/*.json
  summary.csv
```

## Current Scope Notes

1. The CSV writer supports column growth over time by rewriting the header if new fields appear.
2. EAGLE rows add `tree_budget` and `is_eagle3`, while standard speculative rows use `gamma`.
