# `core/visualize.py`

## Purpose

`core/visualize.py` generates plots from the master `summary.csv` file.

## Main Entry Points

1. `load_master_csv(...)`
   Loads the CSV into a DataFrame and normalizes dtypes.
2. `generate_all_plots(...)`
   Calls every plotting function and writes PNG files.

## Plot Functions

1. `plot_speedup_vs_gamma(...)`
2. `plot_acceptance_rate_heatmap(...)`
3. `plot_pareto_frontier(...)`
4. `plot_draft_size_comparison(...)`
5. `plot_quantization_impact(...)`
6. `plot_ttft_comparison(...)`
7. `plot_temperature_effect(...)`
8. `plot_vram_usage(...)`

## Important Scope Caveat

This file still contains several plotting assumptions from older multi-pair experiments.

Examples:

1. some plots focus on pairs `F` and `G`
2. one plot still references `H`
3. some charts are more useful for the standard speculative path than for the active EAGLE-only evaluation

## Practical Use Today

`core/visualize.py` is still useful for:

1. loading and inspecting `summary.csv`
2. generating generic metrics plots such as VRAM usage and temperature trends

But if you want polished final figures for the current project scope, treat this script as a starting point rather than the final reporting layer.
