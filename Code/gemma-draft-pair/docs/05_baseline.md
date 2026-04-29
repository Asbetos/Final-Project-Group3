# `baseline.py`

## Purpose

`baseline.py` provides the plain autoregressive decoding path used as the reference point for every speculative benchmark.

## Main Function

1. `autoregressive_decode(...)`

This function:

1. generates one token at a time
2. reuses the target model KV cache
3. uses the same temperature-driven sampling helper as the speculative path
4. records TTFT, throughput, wall time, and peak VRAM

## Output

The function returns:

1. `output_ids`
2. `output_text`
3. `metrics`

The `metrics` field is a `GenerationMetrics` object.

## Baseline Metric Conventions

Because there is no draft model:

1. `acceptance_rate = 1.0`
2. `acceptance_length = 1.0`
3. `draft_overhead_ratio = 0.0`

That makes the baseline rows easy to compare against speculative rows in `summary.csv`.

## Current Scope Notes

1. Baseline runs are executed first for each `(task, temperature)` combination.
2. Their mean TPS values are used to compute speedup for each speculative configuration.
