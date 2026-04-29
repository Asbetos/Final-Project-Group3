# Artifact Index

This folder is a self-contained snapshot of my Gemma draft-pair benchmark work.
It includes the code, outputs, and figures used to test standard speculative decoding for the Gemma target-draft pair.

## Code Inventory

Core experiment and runtime code:

1. `core/config.py` - defines model pairs, benchmark tasks, gamma values, temperatures, and generation settings.
2. `core/models.py` - loads models and tokenizers with compatibility and memory-efficient loading support.
3. `core/data.py` - loads and preprocesses benchmark prompts.
4. `core/baseline.py` - implements baseline autoregressive decoding.
5. `core/speculative.py` - implements standard speculative decoding.
6. `core/sampling.py` - provides sampling and rejection-sampling utilities.
7. `core/metrics.py` - collects speed, TTFT, VRAM, acceptance, and throughput metrics.
8. `core/runner.py` - runs each benchmark configuration and connects models, data, decoding, and metrics.
9. `core/sweep.py` -  launches benchmark sweeps across tasks, gamma values, and temperatures.
10. `core/visualize.py` - generates figures from benchmark summary files.


## Code Changes

### `models.py`
- Added a Gemma-4 tokenizer fallback for compatibility.
- Updated model loading to use an auto/offload-based flow so large Gemma models could be loaded more reliably in the AWS GPU environment.
- Kept the existing 4-bit quantization path for memory-efficient large model loading.

### `sampling.py`
- Updated stochastic batch rejection sampling to generate random values on the same device as the target logits.
- Fixed device mismatch issues during non-zero temperature runs.
- Added shared-vocabulary alignment for target and draft distributions.
- Handled vocabulary-size mismatch during speculative comparison and residual sampling.

### `sweep.py`
- Updated output path handling to organize benchmark runs under `Code/gemma_runs/outputs/`.
- Used the sweep script to run Gemma speculative decoding experiments across model pairs, tasks, gamma values, and temperatures.

### `visualize.py`
- Updated the visualization flow to compare active Gemma pairs F and G from their final summary files.
- Generated final plots used in the individual report and group presentation.


## Result Artifacts

Primary result summaries:
1. `artifacts/outputs/F_final/summary.csv`

These summary files were used to analyze speedup, TTFT, acceptance behavior, VRAM usage, and draft overhead.

Additional historical or comparison outputs may also be present under:
1. `artifacts/outputs/ar`


Final figures are organized under:
1. `artifacts/figures/FG_final/acceptance_heatmap.png`
2. `artifacts/figures/FG_final/draft_size_comparison.png`
3. `artifacts/figures/FG_final/pareto_frontier.png`
4. `artifacts/figures/FG_final/quantization_impact.png`
5. `artifacts/figures/FG_final/speedup_vs_gamma.png`
6. `artifacts/figures/FG_final/temperature_effect.png`
7. `artifacts/figures/FG_final/ttft_comparison.png`
8. `artifacts/figures/FG_final/vram_usage.png`


## Notes

1. Pair F is the active project scope for this module.
2. Some historical comparison artifacts for Pair G or earlier Qwen experiments may still be present, but they are not the main final contribution.
3. Large model weights, Hugging Face cache files, virtual environments, and temporary AWS logs are not included.
