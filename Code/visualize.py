"""Plot generation from saved experiment results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_master_csv(path: str = "results/summary.csv") -> pd.DataFrame:
    """Load the aggregated results CSV into a DataFrame."""
    df = pd.read_csv(path)
    df["temperature"] = df["temperature"].astype(float)
    df["gamma"] = df["gamma"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Plot 1: Speedup vs Gamma
# ---------------------------------------------------------------------------


def plot_speedup_vs_gamma(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Line plot: speedup vs gamma, faceted by task, hued by pair."""
    spec = df[df["is_baseline"] == False].copy()
    if spec.empty:
        return

    temps = sorted(spec["temperature"].unique())
    fig, axes = plt.subplots(1, len(temps), figsize=(6 * len(temps), 5), squeeze=False)

    for col, temp in enumerate(temps):
        ax = axes[0, col]
        subset = spec[spec["temperature"] == temp]
        sns.lineplot(
            data=subset,
            x="gamma",
            y="speedup",
            hue="pair_id",
            style="task",
            markers=True,
            ax=ax,
        )
        ax.set_title(f"Temperature = {temp}")
        ax.set_xlabel("Speculation Length (γ)")
        ax.set_ylabel("Speedup (×)")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Wall-Clock Speedup vs Speculation Length", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_gamma.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: Acceptance Rate Heatmap
# ---------------------------------------------------------------------------


def plot_acceptance_rate_heatmap(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Heatmap: gamma x task, one subplot per (pair, temperature)."""
    spec = df[df["is_baseline"] == False].copy()
    if spec.empty:
        return

    pairs = sorted(spec["pair_id"].unique())
    temps = sorted(spec["temperature"].unique())
    n_plots = len(pairs) * len(temps)

    fig, axes = plt.subplots(
        len(pairs), len(temps), figsize=(5 * len(temps), 4 * len(pairs)), squeeze=False
    )

    for row, pair in enumerate(pairs):
        for col, temp in enumerate(temps):
            ax = axes[row, col]
            subset = spec[(spec["pair_id"] == pair) & (spec["temperature"] == temp)]
            if subset.empty:
                ax.set_visible(False)
                continue

            pivot = subset.pivot_table(
                values="mean_acceptance_rate",
                index="gamma",
                columns="task",
                aggfunc="mean",
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlGn",
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "α"},
            )
            ax.set_title(f"Pair {pair}, T={temp}")
            ax.set_xlabel("Task")
            ax.set_ylabel("γ")

    fig.suptitle("Acceptance Rate (α) Heatmap", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "acceptance_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Pareto Frontier
# ---------------------------------------------------------------------------


def plot_pareto_frontier(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Scatter: acceptance rate vs speedup, colored by pair, shaped by task."""
    spec = df[df["is_baseline"] == False].copy()
    if spec.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=spec,
        x="mean_acceptance_rate",
        y="speedup",
        hue="pair_id",
        style="task",
        s=100,
        alpha=0.7,
        ax=ax,
    )

    # Draw Pareto frontier
    sorted_df = spec.sort_values("mean_acceptance_rate")
    frontier_x, frontier_y = [], []
    max_speedup = -float("inf")
    for _, row in sorted_df.iterrows():
        if row["speedup"] > max_speedup:
            frontier_x.append(row["mean_acceptance_rate"])
            frontier_y.append(row["speedup"])
            max_speedup = row["speedup"]
    if frontier_x:
        ax.plot(frontier_x, frontier_y, "r--", linewidth=2, label="Pareto Frontier")

    ax.set_xlabel("Acceptance Rate (α)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Pareto Frontier: Speedup vs Acceptance Rate", fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pareto_frontier.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Draft Size Comparison (Pair A vs B)
# ---------------------------------------------------------------------------


def plot_draft_size_comparison(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Grouped bar chart comparing Pair A (0.6B draft) vs Pair B (1.7B draft)."""
    spec = df[(df["is_baseline"] == False) & (df["pair_id"].isin(["A", "B"]))].copy()
    if spec.empty:
        return

    metrics = ["mean_acceptance_rate", "speedup", "mean_draft_overhead"]
    labels = ["Acceptance Rate (α)", "Speedup (×)", "Draft Overhead"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        sns.barplot(data=spec, x="task", y=metric, hue="pair_id", ax=ax)
        ax.set_ylabel(label)
        ax.set_xlabel("Task")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Draft Size Comparison: 0.6B (Pair A) vs 1.7B (Pair B)", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "draft_size_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5: Quantization Impact (Pair A vs C)
# ---------------------------------------------------------------------------


def plot_quantization_impact(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Grouped bar chart comparing Pair A (fp16) vs Pair C (4-bit target)."""
    spec = df[(df["is_baseline"] == False) & (df["pair_id"].isin(["A", "C"]))].copy()
    if spec.empty:
        return

    metrics = ["mean_acceptance_rate", "speedup", "mean_peak_vram_gb"]
    labels = ["Acceptance Rate (α)", "Speedup (×)", "Peak VRAM (GB)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        sns.barplot(data=spec, x="task", y=metric, hue="pair_id", ax=ax)
        ax.set_ylabel(label)
        ax.set_xlabel("Task")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Quantization Impact: fp16 (Pair A) vs 4-bit NF4 (Pair C)", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "quantization_impact.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 6: TTFT Comparison
# ---------------------------------------------------------------------------


def plot_ttft_comparison(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Bar chart: TTFT for baseline vs speculative across pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine baseline and best speculative (gamma=3) for comparison
    baseline = df[df["is_baseline"] == True].copy()
    spec_g3 = df[(df["is_baseline"] == False) & (df["gamma"] == 3)].copy()

    baseline["method"] = "Baseline"
    spec_g3["method"] = "Speculative (γ=3)"
    combined = pd.concat([baseline, spec_g3], ignore_index=True)

    if combined.empty:
        return

    sns.barplot(
        data=combined, x="pair_id", y="mean_ttft_ms", hue="method", ax=ax
    )
    ax.set_xlabel("Model Pair")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token: Baseline vs Speculative", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ttft_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 7: Temperature Effect
# ---------------------------------------------------------------------------


def plot_temperature_effect(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Line plot: temperature vs acceptance rate, hued by task, faceted by pair."""
    spec = df[df["is_baseline"] == False].copy()
    if spec.empty:
        return

    pairs = sorted(spec["pair_id"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), squeeze=False)

    for col, pair in enumerate(pairs):
        ax = axes[0, col]
        subset = spec[spec["pair_id"] == pair]
        sns.lineplot(
            data=subset,
            x="temperature",
            y="mean_acceptance_rate",
            hue="task",
            markers=True,
            style="task",
            ax=ax,
        )
        ax.set_title(f"Pair {pair}")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Acceptance Rate (α)")
        ax.legend(fontsize=8)

    fig.suptitle("Temperature Effect on Acceptance Rate", fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "temperature_effect.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------------
# Plot 8: VRAM Usage
# ---------------------------------------------------------------------------


def plot_vram_usage(df: pd.DataFrame, output_dir: str = "figures") -> None:
    """Bar chart: peak VRAM per pair configuration."""
    spec = df[df["is_baseline"] == False].copy()
    if spec.empty:
        return

    vram_by_pair = spec.groupby("pair_id")["mean_peak_vram_gb"].max().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=vram_by_pair, x="pair_id", y="mean_peak_vram_gb", ax=ax)
    ax.set_xlabel("Model Pair")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak GPU Memory Usage by Model Pair", fontweight="bold")
    ax.axhline(y=80.0, color="red", linestyle="--", alpha=0.5, label="A100 Limit (80 GB)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "vram_usage.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------


def generate_all_plots(
    csv_path: str = "results/summary.csv",
    output_dir: str = "figures",
) -> None:
    """Run all plot functions from the master CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {csv_path}")
    df = load_master_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    plot_speedup_vs_gamma(df, output_dir)
    print("  [1/8] speedup_vs_gamma.png")

    plot_acceptance_rate_heatmap(df, output_dir)
    print("  [2/8] acceptance_heatmap.png")

    plot_pareto_frontier(df, output_dir)
    print("  [3/8] pareto_frontier.png")

    plot_draft_size_comparison(df, output_dir)
    print("  [4/8] draft_size_comparison.png")

    plot_quantization_impact(df, output_dir)
    print("  [5/8] quantization_impact.png")

    plot_ttft_comparison(df, output_dir)
    print("  [6/8] ttft_comparison.png")

    plot_temperature_effect(df, output_dir)
    print("  [7/8] temperature_effect.png")

    plot_vram_usage(df, output_dir)
    print("  [8/8] vram_usage.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
