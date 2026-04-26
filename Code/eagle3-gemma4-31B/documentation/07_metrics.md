# metrics.py — Metric Dataclasses, CUDA Timing, and Result Serialization

## Purpose

This script defines the data structures for capturing benchmark measurements, provides GPU-accurate and wall-clock timing utilities, and handles serialization of results to JSON and CSV. It is the measurement backbone of the project — every other module that records performance data depends on the dataclasses and helpers defined here.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `csv` | `csv` | `csv.DictWriter` for appending summary rows to the master CSV file with automatic header management. |
| `json` | `json` | `json.dump` for writing detailed per-prompt results as structured JSON. |
| `os` | `os` | `os.makedirs` for creating output directories, `os.path.isfile` for checking if the CSV already exists. |
| `time` | `time` | `time.perf_counter()` for high-resolution wall-clock timing in `WallTimer`. |
| `dataclasses` | `dataclass`, `field`, `asdict` | Declarative definition of metric containers with automatic `__init__`, default factories for mutable fields. |
| `typing` | `Dict`, `List`, `Optional` | Type annotations for function signatures and dataclass fields. |
| `numpy` | `np` | `np.mean`, `np.std`, `np.percentile` for computing summary statistics across prompts. |
| `torch` | `torch` | `torch.cuda.Event` for GPU timing, `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_peak_memory_stats()` for VRAM tracking. |

## Inputs and Outputs

- **Inputs**: Raw timing measurements from decoding loops, lists of `GenerationMetrics` for aggregation, dicts and metric objects for serialization.
- **Outputs**: `RoundMetrics` and `GenerationMetrics` dataclass instances, JSON files with per-prompt details, CSV rows appended to the master summary file.

## Detailed Line-by-Line Explanation

### RoundMetrics (lines 19-31)

```python
@dataclass
class RoundMetrics:
    round_index: int
    draft_tokens_proposed: int
    tokens_accepted: int
    bonus_token_generated: bool
    total_tokens_produced: int
    per_token_accepted: List[bool]
    draft_time_ms: float
    verify_time_ms: float
    round_time_ms: float
```

Captures metrics for a single draft-verify round in speculative decoding:

- `round_index`: Zero-based index of this round within the generation.
- `draft_tokens_proposed`: Number of tokens the draft model generated this round (equals `effective_gamma`).
- `tokens_accepted`: How many of those draft tokens passed rejection sampling.
- `bonus_token_generated`: Whether all draft tokens were accepted and a bonus token was sampled.
- `total_tokens_produced`: Total tokens added to the output this round (`tokens_accepted` + 1 if bonus or correction).
- `per_token_accepted`: A list of booleans, one per draft token, recording individual accept/reject decisions. Used to compute per-position acceptance statistics.
- `draft_time_ms` / `verify_time_ms`: GPU-timed durations for the draft and verify phases.
- `round_time_ms`: Total wall-clock time for the round (draft + verify + overhead).

### GenerationMetrics (lines 34-82)

```python
@dataclass
class GenerationMetrics:
    prompt_index: int
    total_tokens_generated: int
    total_rounds: int
    wall_clock_ms: float
    ttft_ms: float
    tokens_per_second: float
    acceptance_rate: float
    acceptance_length: float
    draft_overhead_ratio: float
    peak_vram_bytes: int
    rounds: List[RoundMetrics] = field(default_factory=list)
```

Aggregated metrics for one complete generation (one prompt). Key fields:

- `tokens_per_second`: The primary throughput metric (`total_tokens_generated / wall_clock_ms * 1000`).
- `ttft_ms`: Time to first token — measures the latency a user would perceive before output starts.
- `acceptance_rate`: Fraction of all proposed draft tokens that were accepted across all rounds.
- `acceptance_length`: Average number of draft tokens accepted per round (not counting bonus tokens).
- `draft_overhead_ratio`: Fraction of total wall time spent on draft generation. High values indicate the draft model is too slow relative to the verification savings.
- `peak_vram_bytes`: Maximum GPU memory allocated during this generation.
- `rounds`: List of per-round details. Uses `field(default_factory=list)` because mutable defaults in dataclasses must use factories to avoid shared state between instances.

**Lines 47-82 — `aggregate()` static method:**

```python
@staticmethod
def aggregate(metrics_list: List["GenerationMetrics"]) -> Dict:
```

Computes summary statistics (mean, standard deviation, 95th percentile) across all prompts for a single experiment configuration.

```python
def _stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
```

The inner `_stats` helper converts a list of values to a NumPy array for efficient computation. `dtype=np.float64` prevents precision loss when averaging. Three statistics are computed:
- **mean**: Average performance across prompts.
- **std**: Variability — high std indicates performance depends heavily on prompt content.
- **p95**: 95th percentile — captures worst-case behavior without being as sensitive to outliers as max.

The method computes these statistics for eight metrics: `tokens_per_second`, `acceptance_rate`, `acceptance_length`, `ttft_ms`, `wall_clock_ms`, `draft_overhead_ratio`, `peak_vram_gb` (converted from bytes), and `total_tokens`.

### CudaTimer (lines 89-102)

```python
class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float = 0.0
```

A context manager that uses CUDA events for GPU-accurate timing.

**Why CUDA events instead of `time.perf_counter()`?** CPU wall-clock timers measure when the CPU submits work, not when the GPU finishes. GPU operations are asynchronous — `time.perf_counter()` would undercount because it returns before the GPU kernel completes. CUDA events are recorded in the GPU command stream and measure the actual GPU execution time.

**`__enter__` (line 97):** Records the start event in the CUDA stream.

**`__exit__` (lines 100-102):**
```python
self.end_event.record()
torch.cuda.synchronize()
self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
```
1. Records the end event.
2. `torch.cuda.synchronize()`: Blocks the CPU until all GPU operations complete. This ensures the end event has been processed before querying the elapsed time.
3. `elapsed_time()`: Returns the time in milliseconds between the two events, measured on the GPU clock.

### WallTimer (lines 105-114)

```python
class WallTimer:
    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0
```

A simpler context manager using `time.perf_counter()` for wall-clock timing. Used where user-perceived latency matters (total generation time, TTFT) rather than GPU kernel timing.

`time.perf_counter()` provides the highest-resolution timer available on the platform (nanosecond resolution on Linux). The result is multiplied by 1000 to convert from seconds to milliseconds.

### VRAM Helpers (lines 117-126)

**`record_peak_vram()` (lines 117-120):**
```python
def record_peak_vram() -> int:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0
```
Returns the peak GPU memory allocated since the last reset. `max_memory_allocated()` tracks the high-water mark of `torch.cuda.memory_allocated()`, which measures the total memory occupied by tensors (not the CUDA cache, which can be larger).

**`reset_peak_vram()` (lines 123-126):**
```python
def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
```
Resets the peak memory counter to zero. Called at the start of each generation so that the peak measurement captures only that generation's allocation, not carry-over from previous runs.

Both functions guard against non-CUDA environments to prevent crashes during unit testing.

### Result Serialization (lines 132-185)

**`_round_to_json()` (lines 132-141):**
Converts a `RoundMetrics` dataclass to a JSON-friendly dict with abbreviated keys and rounded floats. The `per_token_accepted` list is deliberately excluded to keep JSON files compact — it can be reconstructed from `tokens_accepted` and `draft_tokens_proposed` for aggregate analysis.

**`_gen_to_json()` (lines 144-157):**
Converts a `GenerationMetrics` to a JSON dict, including the nested list of round details. Numeric values are rounded (`round(value, 2)` or `round(value, 4)`) to reduce file size without losing meaningful precision. VRAM is converted from bytes to GB for human readability.

**`save_results_json()` (lines 160-172):**
```python
def save_results_json(config_dict, summary, per_prompt, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "config": config_dict,
        "summary": summary,
        "per_prompt": [_gen_to_json(g) for g in per_prompt],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```
Writes a complete result file containing three sections:
1. **config**: The experiment parameters (pair, gamma, temperature, task).
2. **summary**: Aggregated statistics from `GenerationMetrics.aggregate()`.
3. **per_prompt**: Detailed metrics for each individual prompt.

`os.makedirs(..., exist_ok=True)` ensures the output directory exists. The `or "."` handles the edge case where `os.path.dirname` returns an empty string (file in current directory).

**`save_summary_csv()` (lines 175-185):**
```python
def save_summary_csv(row, path="results/summary.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
```
Appends one row to the master CSV file. On the first call (file doesn't exist), it writes the header row. Subsequent calls append data rows only. This incremental approach means results are persisted as soon as each configuration finishes — if the sweep crashes midway, completed results are preserved.

The `newline=""` argument prevents blank lines between rows on Windows (a `csv` module requirement).

## Design Decisions

- **Dataclasses over plain dicts**: Provides type safety, autocompletion, and clear documentation of what metrics are collected. The `@dataclass` decorator auto-generates `__init__`, `__repr__`, and `__eq__`.
- **Separate CudaTimer and WallTimer**: GPU kernel time and user-perceived latency are fundamentally different measurements. Using the wrong timer type would produce misleading benchmarks.
- **Incremental CSV writes**: Each configuration's results are appended immediately rather than collected in memory and written at the end. This is crash-resilient — a 19-hour sweep that fails at hour 15 still has all completed results.
- **NumPy for statistics**: While Python's `statistics` module could compute mean and stdev, NumPy's `percentile()` function handles p95 efficiently and the array operations are cleaner for batch computation.
- **VRAM in bytes internally, GB in output**: Internal tracking uses bytes for precision. Conversion to GB happens only at serialization time, avoiding rounding errors in intermediate calculations.
