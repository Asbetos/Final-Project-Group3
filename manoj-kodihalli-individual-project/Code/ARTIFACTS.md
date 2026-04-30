# Artifact Index

This folder is organized so the code, outputs, and training evidence can be pushed together as one self-contained snapshot of my individual work.

## Code Inventory

Core experiment and runtime code:

1. `core/config.py` - experiment grid definitions for baseline, standard speculative decoding, and EAGLE-3.
2. `core/models.py` - model and tokenizer loading helpers.
3. `core/data.py` - benchmark prompt loading and preprocessing.
4. `core/baseline.py` - baseline autoregressive decoding implementation.
5. `core/speculative.py` - standard speculative decoding implementation.
6. `core/eagle3.py` - EAGLE-3 draft-head model and decode loop.
7. `core/eagle3_train.py` - EAGLE-3 draft-head training pipeline and checkpoint helpers.
8. `core/sampling.py` - token sampling and rejection-sampling helpers.
9. `core/metrics.py` - timing, VRAM, and run metrics collection.
10. `core/runner.py` - per-configuration execution flow.
11. `core/visualize.py` - figure generation from result summaries.
12. `core/safetensors_nommap.py` - local safetensors loading helper.

CLI and app entrypoints:

1. `scripts/sweep.py` - experiment grid launcher.
2. `scripts/test_correctness.py` - validation and smoke-test entrypoint.
3. `scripts/visualize.py` - plot-generation entrypoint.
4. `scripts/eagle3_train.py` - training entrypoint.
5. `apps/app.py` - Streamlit comparison app.

Supporting documentation:

1. `docs/01_config.md` through `docs/14_eagle3_train.md`
2. `archive/qwen_legacy/README.md`

## Result Artifacts

Primary result summaries:

1. `artifacts/results/eagle3_gemma3_full/summary.csv`
2. `artifacts/results/eagle3_cnn_fix_validation/summary.csv`

Final figures:

1. `artifacts/figures/final/eagle3_final_speedup_heatmap.png`
2. `artifacts/figures/final/eagle3_final_acceptance_heatmap.png`
3. `artifacts/figures/final/eagle3_best_vs_baseline_tps.png`
4. `artifacts/figures/final/eagle3_best_ttft_vram.png`
5. `artifacts/figures/final/acceptance_heatmap.png`
6. `artifacts/figures/final/draft_size_comparison.png`
7. `artifacts/figures/final/pareto_frontier.png`
8. `artifacts/figures/final/quantization_impact.png`
9. `artifacts/figures/final/speedup_vs_gamma.png`
10. `artifacts/figures/final/temperature_effect.png`
11. `artifacts/figures/final/ttft_comparison.png`
12. `artifacts/figures/final/vram_usage.png`

## Training And Evaluation Proof

Raw logs included for proof:

1. `artifacts/logs/training.log`
2. `artifacts/logs/eval_eagle3_gemma3_full.log`
3. `artifacts/logs/cnn_rerun_fix.log`

Key training evidence from `artifacts/logs/training.log`:

```text
2026-04-26 04:43:42,746 [INFO] __main__: Training on: NVIDIA A10G
2026-04-26 04:45:28,265 [INFO] __main__: Prepared 52002 training examples
2026-04-27 18:13:37,078 [INFO] __main__: Checkpoint saved: checkpoints/eagle3/gemma3_12b/eagle3_gemma3_12b_final.pt
2026-04-27 18:13:37,079 [INFO] __main__: Training complete. Checkpoints in: checkpoints/eagle3/gemma3_12b
```

Key full-evaluation evidence from `artifacts/logs/eval_eagle3_gemma3_full.log`:

```text
2026-04-27 18:55:22,707 [INFO] __main__: Experiment grid: 2 pairs x 3 gammas x 3 temps x 4 tasks = 72 speculative + 24 baseline = 96 total configs
2026-04-28 01:24:16,874 [INFO] __main__: All experiments complete. Results in: results/eagle3_gemma3_full
```

Key rerun evidence from `artifacts/logs/cnn_rerun_fix.log`:

```text
2026-04-28 02:21:24,475 [INFO] __main__: Experiment grid: standard=0 baseline + 0 speculative, eagle3=3 baseline + 6 configs, total=9
2026-04-28 04:20:36,385 [INFO] __main__: All experiments complete. Results in: results/eagle3_gemma3_full
```

The rerun metrics snapshot to review alongside that log is:

1. `artifacts/results/eagle3_cnn_fix_validation/summary.csv`

## Notes

1. The local `.gitignore` in this folder explicitly re-includes the three proof `.log` files so they can be committed without changing repository-wide ignore rules.
2. The log excerpts preserve the original runtime output paths from before this cleanup, so they may still mention `results/...`.
3. The final checkpoint path is preserved in the logs and README, but the checkpoint binary itself is not part of this folder snapshot.
