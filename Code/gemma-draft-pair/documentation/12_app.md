# `app.py`

## Purpose

`app.py` is a Streamlit demo for interactive side-by-side comparison of:

1. baseline autoregressive decoding
2. standard speculative decoding

## Supported Pairs In The UI

The app exposes:

1. `F`
2. `G`

For the active project scope, pair `F` is the important one.

## Main Flow

1. load and cache the selected pair
2. format the user prompt using the tokenizer's chat template
3. run baseline decoding
4. run speculative decoding on the same prompt
5. display TPS, TTFT, wall time, speedup, and acceptance metrics

## Run Command

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Current Scope Notes

1. The app is for demonstration only.
2. It is not the source of benchmark CSVs or final project figures.
3. Use `sweep.py` for repeatable evaluation runs.
