# `visualize.py`

## Purpose

`visualize.py` generates plots from saved summary CSVs.

## What It Loads

By default, `load_master_csv(...)` combines:

1. `gemma_runs/outputs/F_final/summary.csv`
2. `gemma_runs/outputs/G_final/summary.csv`

and adds `is_eagle3=False` when that column is missing.

## Plot Functions

1. `plot_speedup_vs_gamma(...)`
2. `plot_acceptance_rate_heatmap(...)`
3. `plot_pareto_frontier(...)`
4. `plot_draft_size_comparison(...)`
5. `plot_quantization_impact(...)`
6. `plot_ttft_comparison(...)`
7. `plot_temperature_effect(...)`
8. `plot_vram_usage(...)`
9. `generate_all_plots(...)`

## Important Scope Note

This script still reflects the older broader comparison story:

1. many plots compare `F` vs `G`
2. one plot still references `H`

So the script is still useful for inspecting saved data, but it is not perfectly aligned with the narrowed pair `F` project scope.

## Practical Use Today

Use it when you want to:

1. inspect existing `F_final` and `G_final` results
2. regenerate comparison plots from the saved summary files

Treat it as a plotting utility rather than the final authoritative reporting layer.
