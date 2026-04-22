# runner.py — Single-Config Executor and Pair-Level Sweep Orchestrator

## Purpose

This script is the execution engine. It connects the decoding algorithms (baseline and speculative) with the data pipeline, metric collection, and result serialization. It defines three levels of orchestration: running a single experiment configuration, running all baseline configurations for a model pair, and sweeping all speculative configurations for a pair. The sweep module (`sweep.py`) calls into this module to execute the full experiment grid.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `logging` | `logging` | Progress logging for long-running sweeps (per-config status, prompt counts, speedup). |
| `os` | `os` | `os.path.join` for constructing output file paths. |
| `time` | `time` | Imported but used indirectly via metric collection. |
| `typing` | `Dict`, `List`, `Optional` | Type annotations for function signatures. |
| `torch` | `torch` | `torch.Generator` creation for seeded random number generation. |
| `baseline` (local) | `autoregressive_decode` | Standard autoregressive decoder for baseline measurements. |
| `config` (local) | `ALL_GAMMAS`, `ALL_TASKS`, `ALL_TEMPERATURES`, `ExperimentConfig`, `ModelPairConfig` | Default parameter lists and configuration dataclasses. |
| `data` (local) | `tokenize_prompts` | Loads and tokenizes prompts for a given task. |
| `metrics` (local) | `GenerationMetrics`, `save_results_json`, `save_summary_csv` | Metric aggregation and result persistence. |
| `models` (local) | `load_model_pair`, `unload_models` | Model loading and VRAM cleanup. |
| `speculative` (local) | `speculative_decode` | Speculative decoding algorithm. |

## Inputs and Outputs

- **Inputs**: `ExperimentConfig` or `ModelPairConfig` objects, loaded model/tokenizer objects, hyperparameter lists.
- **Outputs**: Summary dicts with aggregated statistics; side effects include JSON files per configuration and CSV rows appended to the master summary.

## Detailed Line-by-Line Explanation

### _make_generator() (lines 24-27)

```python
def _make_generator(seed: int, device: str = "cuda") -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen
```

Creates a seeded `torch.Generator` on the specified device. This is a factory function used throughout the runner to ensure reproducible random sampling. Each prompt gets a unique seed derived from the base seed (`seed + 1000 + idx`), so:
1. Adding new prompts doesn't change the seeds of existing ones.
2. Warm-up passes use different seeds (`seed + i`) from measurement passes (`seed + 1000 + idx`), preventing seed reuse.

The generator is created on `"cuda"` because `torch.multinomial` (used in sampling) requires the generator and input tensor to be on the same device.

### run_single_config() (lines 30-100)

```python
def run_single_config(config, target_model, draft_model, tokenizer, output_dir="results"):
```

Runs one complete experiment configuration — one cell in the 180-configuration grid. This function handles a single (pair, gamma, temperature, task) combination.

**Lines 51-53 — Prompt tokenization:**
```python
prompts = tokenize_prompts(config.task, tokenizer, config.num_prompts, config.seed)
```
Loads and tokenizes prompts for the specified task. Prompts are cached by the `tokenize_prompts` function, so repeated calls with the same task don't reload the dataset.

**Lines 56-68 — Warm-up passes:**
```python
for i in range(min(config.num_warmup, len(prompts))):
    gen = _make_generator(config.seed + i)
    speculative_decode(
        target_model, draft_model,
        prompts[i]["input_ids"], prompts[i]["attention_mask"],
        gamma=config.gamma, temperature=config.temperature,
        max_new_tokens=min(32, config.max_new_tokens),
        tokenizer=tokenizer, generator=gen,
    )
```

Warm-up runs are critical for accurate GPU benchmarking. On the first execution, CUDA performs several one-time operations:
- **JIT compilation** of CUDA kernels for specific tensor shapes.
- **cuBLAS handle initialization** and workspace allocation for matrix multiplications.
- **Memory pool creation** in PyTorch's caching allocator.

Without warm-up, the first few prompts would have inflated timings that don't represent steady-state performance. The warm-up uses `min(32, config.max_new_tokens)` to limit generation length — only enough to trigger kernel compilation, not enough to waste time.

**Lines 71-86 — Main measurement loop:**
```python
for idx, prompt_data in enumerate(prompts):
    gen = _make_generator(config.seed + 1000 + idx)
    result = speculative_decode(...)
    m = result["metrics"]
    m.prompt_index = idx
    all_metrics.append(m)
```

Each prompt gets a unique generator seeded with `config.seed + 1000 + idx`. The offset of 1000 ensures measurement seeds never collide with warm-up seeds (which use `config.seed + i` where `i < num_warmup`, typically 3).

Progress is logged every 10 prompts to provide visibility during long runs without flooding the log.

**Lines 89-97 — Aggregation and persistence:**
```python
summary = GenerationMetrics.aggregate(all_metrics)
json_path = os.path.join(output_dir, "speculative", f"{run_id}.json")
save_results_json(config.to_dict(), summary, all_metrics, json_path)
```
Results are aggregated into summary statistics and saved to a JSON file named after the `run_id` (e.g., `A_humaneval_g5_t0.0.json`). The JSON contains the full config, summary, and per-prompt details.

### run_baseline_for_pair() (lines 103-172)

```python
def run_baseline_for_pair(pair, target_model, tokenizer, tasks, temperatures, ...):
```

Runs baseline autoregressive decoding for all (task, temperature) combinations for a single model pair.

**Why baseline is separate from speculative:** Baseline decoding is independent of gamma — the autoregressive decoder doesn't use a draft model. Running it separately avoids redundant computation: for 5 gamma values, the baseline would otherwise be repeated 5 times per (task, temperature) combination.

**Lines 119-125 — Per-task prompt loading:**
```python
for task in tasks:
    prompts = tokenize_prompts(task, tokenizer, num_prompts, seed)
    for temp in temperatures:
```
Prompts are loaded once per task (outer loop) and reused across temperatures (inner loop). This is an efficiency optimization — tokenization is the same regardless of temperature.

**Lines 127-137 — Warm-up:**
Same pattern as `run_single_config()`: short generations on the first few prompts to prime the CUDA runtime.

**Lines 139-153 — Measurement loop:**
Identical structure to the speculative measurement loop, but calls `autoregressive_decode()` instead.

**Lines 155-172 — CSV row construction:**
```python
csv_row = {
    "pair_id": pair.pair_id,
    "task": task,
    "gamma": 0,
    "temperature": temp,
    "is_baseline": True,
    "mean_tps": round(summary["tokens_per_second"]["mean"], 2),
    ...
    "mean_acceptance_rate": 1.0,
    "mean_acceptance_length": 1.0,
    "speedup": 1.0,
}
```

Baseline rows use `gamma=0` as a sentinel value and set `is_baseline=True`. Acceptance metrics are hardcoded to 1.0 (every token is "accepted" in autoregressive decoding) and speedup is 1.0 (baseline is the reference). This allows both methods to share the same CSV schema.

### run_pair_sweep() (lines 175-263)

```python
def run_pair_sweep(pair, gammas=None, temperatures=None, tasks=None, ...):
```

The top-level orchestrator for one model pair. Loads models, runs all baselines, then sweeps all speculative configurations, and finally unloads models.

**Lines 195-197 — Default parameter handling:**
```python
gammas = gammas or ALL_GAMMAS
temperatures = temperatures or ALL_TEMPERATURES
tasks = tasks or ALL_TASKS
```
Defaults to the full experiment grid if no subsets are specified. The `or` pattern works because empty lists are falsy, falling back to the global defaults from `config.py`.

**Lines 205-207 — Model loading:**
```python
target_model, draft_model, tokenizer = load_model_pair(pair)
```
Models are loaded once per pair and reused for all configurations. This is the most expensive operation (30-60 seconds per pair), so batching all configs per pair is critical.

**Lines 209-263 — try/finally with unload:**
```python
try:
    baselines = run_baseline_for_pair(...)
    # ... speculative sweep ...
finally:
    unload_models(target_model, draft_model)
```

The `try/finally` ensures models are unloaded even if an error occurs mid-sweep. Without this, a crash would leave ~20 GB of VRAM allocated, causing the next pair to OOM.

**Lines 226-237 — Speculative sweep loop:**
```python
for gamma in gammas:
    for temp in temperatures:
        for task in tasks:
            config = ExperimentConfig(pair=pair, gamma=gamma, temperature=temp, task=task, ...)
            summary = run_single_config(config, target_model, draft_model, tokenizer, output_dir)
```

The triple-nested loop iterates over all (gamma, temperature, task) combinations. The loop order (gamma outermost) is arbitrary — all orderings produce the same results because each config is independent.

**Lines 240-247 — Speedup computation:**
```python
baseline_key = (task, temp)
baseline_tps = baselines[baseline_key]["tokens_per_second"]["mean"]
spec_tps = summary["tokens_per_second"]["mean"]
speedup = spec_tps / baseline_tps if baseline_tps > 0 else 0.0
```

Speedup is computed as the ratio of speculative TPS to baseline TPS for the same (task, temperature) combination. This is the primary metric for evaluating whether speculative decoding provides a wall-clock benefit.

**Lines 249-262 — CSV row with speedup:**
The speculative CSV row includes all the same fields as the baseline row, plus meaningful values for `mean_acceptance_rate`, `mean_acceptance_length`, `mean_draft_overhead`, and the computed `speedup`.

## Design Decisions

- **Models loaded once per pair**: Loading Qwen3-8B takes 30-60 seconds. Reloading for each of the 60 configs per pair would add 30-60 minutes of overhead.
- **Baselines run first**: Baseline TPS is needed to compute speedup for each speculative config. Running baselines first ensures the speedup is available immediately when each speculative config completes.
- **Warm-up with reduced tokens**: Using `min(32, max_new_tokens)` instead of the full `max_new_tokens` saves time while still triggering all necessary CUDA kernel compilations. The kernel shapes depend on the model architecture, not the generation length.
- **Unique seeds per prompt**: `seed + 1000 + idx` ensures reproducibility at the prompt level — re-running the same experiment produces identical outputs, but each prompt within a run gets different random draws.
- **Incremental CSV writes**: Results are appended after each config rather than at the end. A crash at hour 15 of a 19-hour sweep preserves all completed results.
- **try/finally for model cleanup**: GPU memory leaks across pairs would cascade into OOM errors. Guaranteed cleanup prevents this.
