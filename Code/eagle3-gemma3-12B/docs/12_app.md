# `apps/app.py`

## Purpose

`apps/app.py` is a Streamlit demo for side-by-side generation comparisons.

It is intended for interactive inspection, not for benchmark collection.

## What It Shows

The app compares:

1. baseline autoregressive decoding
2. standard speculative decoding
3. EAGLE-3 decoding

for the same prompt and the same target model.

## Current Modes

The app exposes:

1. baseline autoregressive decoding
2. standard speculative decoding with pair `F`
3. EAGLE-3 decoding with pair `I`

## Main Flow

1. Load and cache the selected model stack.
2. Format the user prompt with the chat template.
3. Run baseline decoding.
4. Run speculative or EAGLE-3 decoding.
5. Display speed, TTFT, speedup, and acceptance metrics.

## Run Command

```bash
streamlit run apps/app.py --server.port 8501 --server.address 0.0.0.0
```

## Current Scope Notes

1. This is a convenience demo only.
2. It should not be treated as the benchmark runner.
3. The reusable decoding logic lives under `core/` and this file is just the UI layer.
