# visualize.py — Plot Generation from Experiment Results

## Purpose

This script generates eight publication-quality plots from the master CSV produced by the experiment sweep. Each plot addresses a specific research question: how speculation length affects speedup, how acceptance rates vary across tasks, what the Pareto frontier looks like, how draft model size and quantization affect performance, and more. The script can be run standalone after the sweep completes.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `os` | `os` | `os.makedirs` for creating the output figure directory, `os.path.join` for constructing file paths. |
| `matplotlib.pyplot` | `plt` | Core plotting library. Creates figures, axes, subplots; saves to PNG files. |
| `numpy` | `np` | Imported for potential numerical operations (used indirectly by pandas/seaborn). |
| `pandas` | `pd` | `pd.read_csv` for loading the master CSV, DataFrame operations for filtering, pivoting, and grouping. |
| `seaborn` | `sns` | High-level statistical plotting built on matplotlib. Provides `lineplot`, `heatmap`, `scatterplot`, and `barplot` with automatic legend/color/style handling. |

## Inputs and Outputs

- **Inputs**: `results/summary.csv` — the master CSV file with one row per experiment configuration (both baseline and speculative).
- **Outputs**: Eight PNG files in the `figures/` directory, each at 300 DPI.

## Detailed Line-by-Line Explanation

### load_master_csv() (lines 11-15)

```python
def load_master_csv(path: str = "results/summary.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["temperature"] = df["temperature"].astype(float)
    df["gamma"] = df["gamma"].astype(int)
    return df
```

Loads the CSV and explicitly casts `temperature` and `gamma` to their correct types. CSV parsing sometimes reads these as strings or mixed types, which would break grouping and filtering operations downstream. The explicit casts ensure consistent behavior.

### Plot 1: plot_speedup_vs_gamma() (lines 22-52)

**Research question**: How does speculation length affect wall-clock speedup?

```python
spec = df[df["is_baseline"] == False].copy()
```
Filters to speculative-only rows (excludes baseline, which has `gamma=0`).

```python
fig, axes = plt.subplots(1, len(temps), figsize=(6 * len(temps), 5), squeeze=False)
```
Creates one subplot per temperature. `squeeze=False` ensures `axes` is always a 2D array, even if there's only one temperature — this prevents indexing errors.

```python
sns.lineplot(data=subset, x="gamma", y="speedup", hue="pair_id", style="task", markers=True, ax=ax)
```
- **`hue="pair_id"`**: Colors lines by model pair (A, B, C).
- **`style="task"`**: Uses different line styles (solid, dashed, dotted, dash-dot) for each task.
- **`markers=True`**: Adds data point markers for readability.

```python
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
```
Draws a reference line at speedup=1.0. Points above this line indicate speculative decoding is faster than the baseline; below means it's slower.

**Saved as**: `figures/speedup_vs_gamma.png`

### Plot 2: plot_acceptance_rate_heatmap() (lines 58-100)

**Research question**: How does acceptance rate vary across tasks and gamma values?

```python
fig, axes = plt.subplots(len(pairs), len(temps), figsize=(5 * len(temps), 4 * len(pairs)), squeeze=False)
```
Creates a grid of subplots: one row per model pair, one column per temperature.

```python
pivot = subset.pivot_table(
    values="mean_acceptance_rate", index="gamma", columns="task", aggfunc="mean",
)
```
Reshapes the data into a gamma-by-task matrix suitable for a heatmap. `aggfunc="mean"` handles any duplicate entries by averaging.

```python
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", vmin=0, vmax=1, ax=ax)
```
- **`annot=True, fmt=".3f"`**: Prints acceptance rate values (3 decimal places) inside each cell.
- **`cmap="YlGn"`**: Yellow-to-green colormap where darker green = higher acceptance rate.
- **`vmin=0, vmax=1`**: Fixes the color scale from 0 to 1 across all subplots for consistent comparison.

**Saved as**: `figures/acceptance_heatmap.png`

### Plot 3: plot_pareto_frontier() (lines 106-150)

**Research question**: What is the optimal tradeoff between acceptance rate and speedup?

```python
sns.scatterplot(data=spec, x="mean_acceptance_rate", y="speedup",
                hue="pair_id", style="task", s=100, alpha=0.7, ax=ax)
```
Each point is one (pair, gamma, temperature, task) configuration. The scatter reveals clusters and tradeoffs.

**Lines 138-145 — Pareto frontier computation:**
```python
sorted_df = spec.sort_values("mean_acceptance_rate")
frontier_x, frontier_y = [], []
max_speedup = -float("inf")
for _, row in sorted_df.iterrows():
    if row["speedup"] > max_speedup:
        frontier_x.append(row["mean_acceptance_rate"])
        frontier_y.append(row["speedup"])
        max_speedup = row["speedup"]
```
Computes the Pareto frontier by iterating through configurations sorted by acceptance rate. A configuration is on the frontier if it has the highest speedup seen so far — no other configuration achieves both higher acceptance rate and higher speedup. The frontier is drawn as a red dashed line.

**Saved as**: `figures/pareto_frontier.png`

### Plot 4: plot_draft_size_comparison() (lines 156-191)

**Research question**: How does draft model size (0.6B vs 1.7B) affect acceptance rate, speedup, and overhead?

```python
spec = df[(df["is_baseline"] == False) & (df["pair_id"].isin(["A", "B"]))].copy()
```
Filters to Pairs A and B only — these share the same target model (Qwen3-8B fp16) but differ in draft model size (0.6B vs 1.7B).

Three grouped bar charts are created side-by-side:
1. **Acceptance Rate**: Larger draft model should have higher acceptance rates.
2. **Speedup**: Net effect — higher acceptance but slower drafting.
3. **Draft Overhead**: Fraction of time spent on draft generation. Larger draft = higher overhead.

**Saved as**: `figures/draft_size_comparison.png`

### Plot 5: plot_quantization_impact() (lines 197-232)

**Research question**: How does 4-bit quantization of the target model affect performance and VRAM?

```python
spec = df[(df["is_baseline"] == False) & (df["pair_id"].isin(["A", "C"]))].copy()
```
Compares Pairs A and C — same draft model (0.6B), same target model (8B), but Pair A uses fp16 and Pair C uses 4-bit NF4 quantization.

Three bar charts:
1. **Acceptance Rate**: Quantization may reduce target model quality, affecting acceptance.
2. **Speedup**: Net throughput effect.
3. **Peak VRAM**: The primary motivation for quantization — Pair C should use significantly less VRAM.

**Saved as**: `figures/quantization_impact.png`

### Plot 6: plot_ttft_comparison() (lines 238-269)

**Research question**: How does speculative decoding affect time to first token?

```python
baseline = df[df["is_baseline"] == True].copy()
spec_g3 = df[(df["is_baseline"] == False) & (df["gamma"] == 3)].copy()
baseline["method"] = "Baseline"
spec_g3["method"] = "Speculative (gamma=3)"
combined = pd.concat([baseline, spec_g3], ignore_index=True)
```
Combines baseline and speculative (gamma=3) rows into one DataFrame with a `method` column for grouping. Gamma=3 is chosen as a representative middle value.

Speculative decoding typically has higher TTFT because the first token requires both a draft pass and a verify pass.

**Saved as**: `figures/ttft_comparison.png`

### Plot 7: plot_temperature_effect() (lines 275-308)

**Research question**: How does sampling temperature affect acceptance rate?

```python
sns.lineplot(data=subset, x="temperature", y="mean_acceptance_rate",
             hue="task", markers=True, style="task", ax=ax)
```
One subplot per model pair, with lines for each task. At temperature=0 (greedy), acceptance is all-or-nothing. At higher temperatures, the distributions become more spread out, typically reducing acceptance rates because the draft and target models diverge more.

**Saved as**: `figures/temperature_effect.png`

### Plot 8: plot_vram_usage() (lines 314-337)

**Research question**: How much GPU memory does each model pair configuration use?

```python
vram_by_pair = spec.groupby("pair_id")["mean_peak_vram_gb"].max().reset_index()
```
Groups by pair and takes the maximum peak VRAM across all configurations for that pair.

```python
ax.axhline(y=24.0, color="red", linestyle="--", alpha=0.5, label="A10G Limit (24 GB)")
```
Draws the A10G's 24 GB VRAM limit as a reference line. Configurations close to this line risk OOM errors.

**Saved as**: `figures/vram_usage.png`

### generate_all_plots() (lines 343-370)

```python
def generate_all_plots(csv_path="results/summary.csv", output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_master_csv(csv_path)
    ...
```

Convenience function that runs all eight plot functions in sequence. Each plot call is followed by a progress print statement. This is the function invoked when the script is run as `__main__`.

## Design Decisions

- **Seaborn over raw matplotlib**: Seaborn handles grouping, coloring, and legends automatically from DataFrame columns. This reduces boilerplate and ensures consistent styling across plots.
- **Each plot is a separate function**: Allows running individual plots during development without regenerating all eight. Also makes the code easier to modify if a specific plot needs adjustment.
- **300 DPI output**: Publication-quality resolution suitable for academic papers and presentations.
- **`squeeze=False` on subplots**: Defensive coding against single-value axes. Without this, a grid with one row or one column would return a 1D array instead of 2D, causing `axes[0, col]` to fail.
- **Empty data guards**: Each plot function checks `if spec.empty: return` before attempting to create plots, preventing crashes when running with partial results.
- **Fixed color scale on heatmaps**: `vmin=0, vmax=1` ensures acceptance rate colors are comparable across subplots. Without this, each subplot would auto-scale independently.
