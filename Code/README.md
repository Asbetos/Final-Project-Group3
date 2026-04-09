# Speculative Decoding for LLM Inference Acceleration on Qwen3

This project benchmarks **speculative decoding** (Leviathan et al., ICML 2023) on the Qwen3 model family. A small draft model proposes candidate tokens and a large target model verifies them in a single forward pass, producing multiple tokens per step without changing output quality.

This is an **inference benchmarking** project — no model training or fine-tuning occurs. All scripts run generation experiments and collect performance metrics.

---

## Hardware Requirements

- **GPU**: NVIDIA A10G (24 GB VRAM) or equivalent. All three model pair configurations fit within 24 GB.
- **RAM**: 32 GB system memory recommended.
- **Instance**: AWS `g5.2xlarge` or similar.

---

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.1.0 | Model inference, CUDA timing |
| `transformers` | >= 4.51.0 | Qwen3 model loading, tokenization |
| `bitsandbytes` | >= 0.43.0 | 4-bit NF4 quantization (Pair C) |
| `accelerate` | >= 0.26.0 | GPU device placement |
| `datasets` | >= 2.14.0 | HuggingFace dataset loading |
| `matplotlib` | >= 3.7.0 | Plotting |
| `seaborn` | >= 0.13.0 | Statistical visualization |
| `numpy` | >= 1.24.0 | Numerical operations |

**Note**: The first run will download Qwen3 model weights from HuggingFace (~20 GB total). These are cached locally for subsequent runs.

---

## Quick Start

```bash
source venv/bin/activate

# 1. Validate correctness (no GPU needed)
python3 test_correctness.py --level 1

# 2. Validate correctness on GPU
python3 test_correctness.py --level 3 --pair A

# 3. Preview the full experiment grid (no GPU needed)
python3 sweep.py --dry-run

# 4. Run a small experiment to verify everything works
python3 sweep.py --pairs A --gammas 3 --temps 0.0 --tasks humaneval --num-prompts 5

# 5. Run the full sweep for one model pair
python3 sweep.py --pairs A

# 6. Generate plots from results
python3 visualize.py
```

---

## File Descriptions

Each file serves a specific role in the pipeline. They are listed here in the order you would encounter them during execution.

| File | Description |
|---|---|
| `config.py` | Defines the 3 model pair configurations (A, B, C), experiment grid constants (gamma, temperature, tasks), and the `ExperimentConfig` dataclass. |
| `models.py` | Loads Qwen3 models onto the GPU with optional 4-bit quantization. Handles tokenizer setup, VRAM tracking, and model cleanup. |
| `data.py` | Loads prompts from HuggingFace Datasets (HumanEval, TriviaQA, CNN/DailyMail, WritingPrompts), applies the Qwen3 chat template, and tokenizes them for inference. |
| `sampling.py` | Pure sampling functions: temperature-scaled softmax, rejection sampling, residual distribution resampling, and bonus token sampling. These implement the core math of speculative verification. |
| `baseline.py` | Standard token-by-token autoregressive decoder. Serves as the performance baseline that speculative decoding is compared against. |
| `speculative.py` | Core speculative decoding loop. The draft model proposes gamma tokens, the target model verifies them in one forward pass via rejection sampling. Includes KV-cache management and per-round instrumentation. |
| `metrics.py` | Metric dataclasses (`GenerationMetrics`, `RoundMetrics`), GPU-accurate timing via CUDA events, VRAM tracking, and result serialization to JSON/CSV. |
| `runner.py` | Executes a single experiment configuration: tokenizes prompts, runs warm-up passes, collects per-prompt metrics, aggregates results, and computes speedup vs baseline. |
| `sweep.py` | **Main entry point.** CLI that orchestrates the full experiment grid. Loads models once per pair, then sweeps all gamma/temperature/task combinations. |
| `visualize.py` | Reads `results/summary.csv` and generates 8 publication-quality plots (speedup curves, heatmaps, Pareto frontiers, VRAM usage, etc.) into a `figures/` directory. |
| `test_correctness.py` | Three-level correctness validation: unit tests (level 1), greedy equivalence (level 2), and smoke tests (level 3). |

---

## Step-by-Step Execution Guide

### Step 1: Run Correctness Tests

Before running any experiments, validate that the speculative decoding implementation is correct.

```bash
# Level 1 — Unit tests for sampling logic (no GPU, runs in seconds)
python3 test_correctness.py --level 1

# Level 2 — Greedy equivalence: verifies speculative output exactly matches
#           baseline autoregressive output at temperature=0 (requires GPU)
python3 test_correctness.py --level 2 --pair A

# Level 3 — Smoke test: runs all of the above plus a quick generation test
#           across multiple gamma/temperature combos (requires GPU)
python3 test_correctness.py --level 3 --pair A
```

| Argument | Values | Default | Description |
|---|---|---|---|
| `--level` | `1`, `2`, `3` | `1` | Test depth. Each level includes all previous levels. |
| `--pair` | `A`, `B`, `C` | `A` | Which model pair to use for GPU tests (levels 2-3). |

### Step 2: Preview the Experiment Grid

See all configurations that will be run without using the GPU.

```bash
python3 sweep.py --dry-run
```

This prints every (pair, task, gamma, temperature) combination. The full grid is 180 speculative configurations + 36 baseline configurations = **216 total**.

### Step 3: Run Experiments

`sweep.py` is the main entry point. It loads models, runs baselines, then sweeps all speculative configurations.

```bash
# Full sweep — all 3 pairs, all configurations
python3 sweep.py

# Single pair (recommended — run pairs one at a time)
python3 sweep.py --pairs A

# Subset of configurations
python3 sweep.py --pairs A --gammas 1 3 5 --temps 0.0 0.6 --tasks humaneval triviaqa

# Quick test run
python3 sweep.py --pairs A --gammas 3 --temps 0.0 --tasks humaneval --num-prompts 5
```

**Recommended approach**: Run one pair at a time. This way, if Pair B (22.8 GB) hits VRAM limits, you don't lose results from other pairs.

```bash
python3 sweep.py --pairs A 2>&1 | tee sweep_pair_A.txt
python3 sweep.py --pairs B 2>&1 | tee sweep_pair_B.txt
python3 sweep.py --pairs C 2>&1 | tee sweep_pair_C.txt
```

#### Time Estimates

Measured on an AWS `g5.2xlarge` (NVIDIA A10G, 24 GB VRAM) with 50 prompts per config and 128 max new tokens.

| Scope | Configs | Estimated Time |
|---|---|---|
| Quick test (`--pairs A --gammas 3 --temps 0.0 --tasks humaneval --num-prompts 5`) | 2 | ~2 min |
| Single task, all gammas (`--pairs A --temps 0.0 --tasks humaneval`) | 6 | ~30 min |
| Single pair, all configs (`--pairs A`) | 72 | ~6.5 hr |
| Two pairs (`--pairs A C`) | 144 | ~13 hr |
| Full sweep, all 3 pairs (`--pairs A B C`) | 216 | ~19 hr |

Breakdown per pair (72 configs = 12 baseline + 60 speculative):

| Phase | Configs | Time per Config | Subtotal |
|---|---|---|---|
| Baselines (all tasks and temps) | 12 | ~5 min | ~1 hr |
| Speculative (low gamma: 1, 3, 5) | 36 | ~5 min | ~3 hr |
| Speculative (high gamma: 7, 10) | 24 | ~6.5 min | ~2.5 hr |
| **Total per pair** | **72** | | **~6.5 hr** |

Model loading adds ~5-10 seconds per pair (cached weights) or several minutes on first download.

#### sweep.py Arguments

| Argument | Values | Default | Description |
|---|---|---|---|
| `--pairs` | `A` `B` `C` | `A B C` | Model pairs to evaluate (see table below). |
| `--gammas` | integers | `1 3 5 7 10` | Speculation lengths. Higher gamma = more draft tokens per round. |
| `--temps` | floats | `0.0 0.6 1.0` | Sampling temperatures. `0.0` = greedy (deterministic), `1.0` = maximum randomness. |
| `--tasks` | task names | all 4 | Evaluation tasks (see table below). |
| `--max-tokens` | integer | `128` | Maximum new tokens to generate per prompt. |
| `--num-prompts` | integer | `50` | Number of prompts to evaluate per task. |
| `--output-dir` | path | `results` | Directory for output JSON and CSV files. |
| `--seed` | integer | `42` | Random seed for reproducibility. |
| `--dry-run` | flag | off | Print the configuration grid and exit without running. |

#### Model Pairs

| Pair | Target Model | Draft Model | Target Quantization | Estimated VRAM |
|---|---|---|---|---|
| **A** | Qwen3-8B | Qwen3-0.6B | fp16 | ~20.6 GB |
| **B** | Qwen3-8B | Qwen3-1.7B | fp16 | ~22.8 GB |
| **C** | Qwen3-8B | Qwen3-0.6B | 4-bit NF4 | ~8.2 GB |

- **Pair A**: Baseline configuration. Small, fast draft model with high alignment.
- **Pair B**: Tests whether a larger draft model's higher accuracy compensates for its slower speed.
- **Pair C**: Tests whether 4-bit quantization of the target degrades verification quality.

#### Evaluation Tasks

| Task Name | Dataset | Type | Max Prompt Tokens |
|---|---|---|---|
| `humaneval` | OpenAI HumanEval | Code completion | 512 |
| `triviaqa` | TriviaQA (rc) | Factual QA | 256 |
| `cnn_dailymail` | CNN/DailyMail 3.0.0 | Summarization | 1024 |
| `writingprompts` | WritingPrompts | Creative writing | 256 |

These span highly structured output (code) to high-entropy output (creative writing), enabling analysis of when speculative decoding helps vs hurts.

### Step 4: Generate Visualizations

After experiments complete, generate all 8 plots from the results.

```bash
python3 visualize.py
```

This reads `results/summary.csv` and writes plots to `figures/`:

| Plot | Filename | What It Shows |
|---|---|---|
| 1 | `speedup_vs_gamma.png` | Speedup over baseline as gamma increases, faceted by temperature |
| 2 | `acceptance_heatmap.png` | Acceptance rate heatmap (gamma x task) per pair and temperature |
| 3 | `pareto_frontier.png` | Pareto-optimal configs on acceptance rate vs speedup |
| 4 | `draft_size_comparison.png` | Pair A (0.6B draft) vs Pair B (1.7B draft) |
| 5 | `quantization_impact.png` | Pair A (fp16 target) vs Pair C (4-bit target) |
| 6 | `ttft_comparison.png` | Time-to-first-token: baseline vs speculative |
| 7 | `temperature_effect.png` | How temperature affects acceptance rate |
| 8 | `vram_usage.png` | Peak VRAM per pair with 24 GB A10G limit line |

---

## Output Structure

After running experiments, the output directory will contain:

```
results/
  baseline/          Per-config JSON files for baseline runs
  speculative/       Per-config JSON files for speculative runs
  summary.csv        Master CSV with one row per configuration

figures/             Generated plots (after running visualize.py)
```

Each JSON file contains the full experiment config, aggregated summary statistics (mean, std, p95), and per-prompt metrics.

The `summary.csv` has one row per configuration with columns: `pair_id`, `task`, `gamma`, `temperature`, `is_baseline`, `mean_tps`, `std_tps`, `p95_tps`, `mean_acceptance_rate`, `mean_acceptance_length`, `mean_ttft_ms`, `mean_peak_vram_gb`, `mean_draft_overhead`, `speedup`.

---

## Metrics Collected

| Metric | Description |
|---|---|
| **Acceptance Rate** | Fraction of draft tokens accepted by the target model. |
| **Acceptance Length** | Mean tokens produced per verification round. |
| **Tokens/Second** | Wall-clock throughput (higher is better). |
| **Speedup** | Speculative tokens/sec divided by baseline tokens/sec. |
| **Time to First Token (TTFT)** | Latency before the first token is produced. |
| **Peak VRAM** | Maximum GPU memory used during generation. |
| **Draft Overhead Ratio** | Fraction of wall time spent on draft generation vs target verification. |

---

## Example Runs

```bash
# Minimal: 1 pair, 1 gamma, 1 temperature, 1 task, 5 prompts
python3 sweep.py --pairs A --gammas 3 --temps 0.0 --tasks humaneval --num-prompts 5

# Medium: 1 pair, all gammas, greedy only, 2 tasks
python3 sweep.py --pairs A --gammas 1 3 5 7 10 --temps 0.0 --tasks humaneval triviaqa

# Full pair sweep (60 speculative + 12 baseline configs)
python3 sweep.py --pairs A

# Full experiment (180 speculative + 36 baseline configs)
python3 sweep.py
```
