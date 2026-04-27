# `baseline.py`

## Purpose

`baseline.py` implements plain autoregressive generation from the target model. It exists so every accelerated method can be compared against a consistent baseline.

## Main Function

1. `autoregressive_decode(...)`

This function:

1. Reuses the target model KV cache.
2. Generates one token at a time.
3. Uses the same sampling helper as the accelerated paths.
4. Records TTFT, throughput, total latency, and peak VRAM.

## Inputs

The function expects:

1. a loaded target model
2. `input_ids`
3. `attention_mask`
4. `temperature`
5. `max_new_tokens`
6. a tokenizer
7. an optional seeded `torch.Generator`

## Output

It returns a dictionary with:

1. `output_ids`
2. `output_text`
3. `metrics`

The `metrics` value is a `GenerationMetrics` object.

## Metric Conventions

Because this is the baseline path:

1. `acceptance_rate = 1.0`
2. `acceptance_length = 1.0`
3. `draft_overhead_ratio = 0.0`

That makes it easy to compare CSV rows across baseline and accelerated runs.

## Current Scope Notes

1. Baseline runs are used by both `run_pair_sweep(...)` and `run_eagle3_pair_sweep(...)`.
2. In EAGLE experiments, the baseline is still the same target-only decoding path.
