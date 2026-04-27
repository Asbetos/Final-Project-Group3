# `app.py`

## Purpose

`app.py` is a Streamlit demo for side-by-side generation comparisons.

It is intended for interactive inspection, not for benchmark collection.

## What It Shows

The app compares:

1. baseline autoregressive decoding
2. standard speculative decoding

for the same prompt and the same target model.

## Current Limitation

The app is not wired to the active EAGLE path.

It currently exposes standard speculative model pairs:

1. `F`
2. `G`

and uses:

1. `autoregressive_decode(...)`
2. `speculative_decode(...)`

## Main Flow

1. Load and cache one model pair.
2. Format the user prompt with the chat template.
3. Run baseline decoding.
4. Run speculative decoding.
5. Display speed, TTFT, speedup, and acceptance metrics.

## Run Command

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Current Scope Notes

1. This is a convenience demo only.
2. It should not be treated as the EAGLE evaluation UI.
3. If an interactive EAGLE demo is needed later, this file is the natural place to extend.
