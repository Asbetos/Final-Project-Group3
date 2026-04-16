# Gemma Runs Notes

## Current status
This folder tracks my experimental progress on Gemma speculative decoding under the `jeongmin` branch.

### Pair F
- Target: `google/gemma-3-12b-it`
- Draft: `google/gemma-3-1b-it`

## What has been completed
- Environment setup on AWS `g5.2xlarge` (A10G, 24GB VRAM)
- Hugging Face access/authentication for Gemma models
- Smoke test for Pair F completed successfully
- Full greedy speculative sweep completed for:
  - tasks: `humaneval`, `triviaqa`, `cnn_dailymail`, `writingprompts`
  - gammas: `1, 3, 5, 7, 10`
  - temperature: `0.0`

## Current issue
For stochastic temperatures (`0.6`, `1.0`), speculative decoding failed due to a generator/device mismatch in `sampling.py`.

## Code changes
- `Code/models.py`
  - Updated Gemma model loading to use `torch_dtype` for compatibility.
- `Code/sampling.py`
  - Fixed generator/device mismatch in stochastic rejection sampling.
- `Code/sweep.py`
  - Updated output path handling to organize runs under `Code/gemma_runs/outputs/`.


## Files included here
- `F_pair_partial_summary.csv`: partial summary of completed runs
- code fixes in `models.py`,`sampling.py` and `sweep.py`

## Next step
- rerun missing speculative configs for `temp=0.6` and `temp=1.0`
- update final summary after rerun