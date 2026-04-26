# Gemma Runs Notes

## Current status
This folder tracks experimental progress on active Gemma speculative decoding 

### Active Pairs
#### Pair F
- Target: `google/gemma-3-12b-it`
- Draft: `google/gemma-3-1b-it`

#### Pair G
- Target: `google/gemma-4-31B`
- Draft: `google/gemma-3-1b-it`

## What has been completed
- Environment setup on AWS `g5.2xlarge` (A10G, 24GB VRAM)
- Hugging Face access/authentication for Gemma models
- Standard speculative benchmark runs completed for active Gemma pairs `F` and `G`
- Sweeps completed across:
  - tasks: `humaneval`, `triviaqa`, `cnn_dailymail`, `writingprompts`
  - gammas: `1, 3, 5, 7, 10`
  - temperatures: `0.0, 0.6, 1.0`
- Final summaries generated under:
  - `Code/gemma_runs/outputs/F_final/summary.csv`
  - `Code/gemma_runs/outputs/G_final/summary.csv`
- F vs G comparison visualizations generated under:
  - `Code/figures/FG_final/`

## Why gamma and temperature were swept
- `gamma` controls speculation length, or how many draft tokens are proposed before target verification.
- Increasing `gamma` tests whether longer draft proposals improve throughput enough to offset extra verification overhead.
- `temperature` controls sampling randomness.
- Sweeping `temperature` tests whether speculative decoding remains effective beyond greedy decoding, especially under more stochastic generation settings.

## Main observations so far
- Pair `F` is generally more stable than Pair `G` across tasks and temperatures.
- Pair `F` achieves higher speedup across all evaluated tasks in the current F vs G comparison plots.
- Pair `G` is more sensitive to higher temperatures, especially at `T=1.0`, where acceptance and speedup drop more noticeably.
- Pair `G` uses substantially more VRAM than Pair `F`, although both still fit within the 24GB A10G memory limit.

## Code changes
- `Code/models.py`
  - Added a Gemma-4 tokenizer fallback for compatibility.
  - Switched model loading to an auto/offload-based flow so large Gemma models could be loaded more reliably in our AWS GPU environment, while keeping the existing 4-bit quantization path.
- `Code/sampling.py`
  - Updated stochastic batch rejection sampling to generate random values on the same device as the target logits, avoiding device mismatch during non-zero temperature runs.
  - Added shared-vocabulary alignment for target and draft distributions to handle vocab-size mismatch during speculative comparison and residual sampling.
- `Code/sweep.py`
  - Updated output path handling to organize runs under `Code/gemma_runs/outputs/`.
- `Code/visualize.py`
  - Updated visualization flow to compare active Gemma pairs `F` and `G` from their final summary files.

## Files included here
- `Code/gemma_runs/outputs/F_final/`
- `Code/gemma_runs/outputs/G_final/`
- `Code/figures/FG_final/`
- supporting code changes in:
  - `models.py`
  - `sampling.py`
  - `sweep.py`
  - `visualize.py`

## Next step
- Integrate Pair `H` (EAGLE-3)
- Update comparison plots to include `H`
- Use the final figures in the project report and presentation