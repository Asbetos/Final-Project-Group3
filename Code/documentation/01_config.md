# config.py — Experiment Configuration and Grid Definitions

## Purpose

This script defines the entire experiment parameter space as Python dataclasses and constants. It serves as the single source of truth for model pair configurations, speculation lengths, temperatures, and task names. Every other script in the pipeline imports from here rather than hardcoding values.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `dataclasses` | `dataclass`, `asdict` | Provides structured, type-annotated configuration objects with automatic `__init__`, `__repr__`, and `__eq__` methods. Chosen over plain dicts for IDE autocompletion and field validation. |
| `json` | `json` | Serializes experiment configs to JSON files for reproducibility. |
| `os` | `os` | Creates directories when saving config files. |

No external (pip-installed) packages are required — this module is pure Python standard library.

## Inputs and Outputs

- **Inputs**: None at import time. `build_grid()` accepts optional filter arguments.
- **Outputs**: Dataclass instances and lists of `ExperimentConfig` objects used by `runner.py` and `sweep.py`.

## Detailed Line-by-Line Explanation

### ModelPairConfig (lines 10-22)

```python
@dataclass
class ModelPairConfig:
    pair_id: str
    target_model_id: str
    draft_model_id: str
    target_quantize_4bit: bool
    target_vram_estimate_gb: float
    draft_vram_estimate_gb: float
```

Defines one of the three model pair configurations (A, B, or C). Fields:

- `pair_id`: Single-letter identifier ("A", "B", or "C") used in filenames and logs.
- `target_model_id`: HuggingFace model hub ID for the large target model (e.g., `"Qwen/Qwen3-8B"`).
- `draft_model_id`: Hub ID for the small draft model (e.g., `"Qwen/Qwen3-0.6B"`).
- `target_quantize_4bit`: Whether to apply 4-bit NF4 quantization to the target. Only `True` for Pair C.
- `target_vram_estimate_gb` / `draft_vram_estimate_gb`: Pre-measured VRAM estimates. Used for logging and the `--dry-run` display; not enforced at runtime.

The `total_vram_estimate_gb` property (line 21-22) sums target + draft estimates, giving users a quick way to check if a pair fits in GPU memory.

### ExperimentConfig (lines 25-50)

```python
@dataclass
class ExperimentConfig:
    pair: ModelPairConfig
    gamma: int
    temperature: float
    task: str
    max_new_tokens: int = 128
    num_prompts: int = 50
    num_warmup: int = 3
    seed: int = 42
```

Represents a single cell in the 180-configuration experiment grid. Each unique combination of (pair, gamma, temperature, task) is one `ExperimentConfig`.

- `gamma`: Speculation length — how many tokens the draft model proposes per round. Higher gamma means more speculative work per round.
- `temperature`: Sampling temperature. `0.0` produces deterministic (greedy) output; `1.0` produces maximum-entropy stochastic sampling.
- `task`: Which dataset/task to use for evaluation.
- `max_new_tokens`: Generation length cap. Default 128 balances enough tokens for meaningful metrics with reasonable runtime.
- `num_prompts`: How many prompts to evaluate per config. Default 50 gives statistically meaningful averages.
- `num_warmup`: Number of warmup prompts (results discarded). Default 3 ensures GPU caches, JIT compilation, and memory allocators are warmed before measurement.
- `seed`: Random seed for deterministic shuffling and sampling.

**`run_id()`** (line 38-40) generates a unique filename-safe string like `"A_humaneval_g5_t0.6"`. This is used as the JSON output filename stem.

**`to_dict()`** (line 42-44) converts the config to a plain dict for JSON serialization. It calls `asdict()` on both the config and the nested `ModelPairConfig`.

**`to_json()`** (line 47-50) writes the config to a JSON file, creating parent directories as needed.

### Pre-built Pair Definitions (lines 57-82)

```python
PAIR_A = ModelPairConfig(
    pair_id="A",
    target_model_id="Qwen/Qwen3-8B",
    draft_model_id="Qwen/Qwen3-0.6B",
    target_quantize_4bit=False,
    target_vram_estimate_gb=16.7,
    draft_vram_estimate_gb=3.9,
)
```

Three concrete pair configurations:

- **Pair A** (~20.6 GB): fp16 target + smallest draft. The baseline configuration testing maximum draft-target alignment within the Qwen3 family.
- **Pair B** (~22.8 GB): fp16 target + larger 1.7B draft. Tests whether a more capable draft model produces higher acceptance rates that offset its slower speed.
- **Pair C** (~8.2 GB): 4-bit quantized target + smallest draft. Tests whether quantizing the target degrades verification quality while dramatically reducing memory.

### Grid Constants (lines 88-93)

```python
ALL_PAIRS = [PAIR_A, PAIR_B, PAIR_C]
PAIR_MAP = {p.pair_id: p for p in ALL_PAIRS}
ALL_GAMMAS = [1, 3, 5, 7, 10]
ALL_TEMPERATURES = [0.0, 0.6, 1.0]
ALL_TASKS = ["humaneval", "triviaqa", "cnn_dailymail", "writingprompts"]
```

- `PAIR_MAP`: Dict for O(1) lookup by pair ID string. Used by CLI argument parsing in `sweep.py`.
- `ALL_GAMMAS`: Five speculation lengths spanning minimal (1, equivalent to standard decoding overhead) to aggressive (10 tokens per draft round).
- `ALL_TEMPERATURES`: Three temperatures covering greedy, moderate, and high-entropy sampling.
- `ALL_TASKS`: Four NLP tasks from structured (code) to unstructured (creative writing).

Total grid size: 3 x 5 x 3 x 4 = **180 speculative configurations**.

### build_grid() (lines 96-123)

```python
def build_grid(pairs=None, gammas=None, temperatures=None, tasks=None, **kwargs):
```

Generates a list of all `ExperimentConfig` combinations. Accepts optional filter arguments — pass a subset to run partial sweeps. The `**kwargs` are forwarded to `ExperimentConfig`, allowing overrides like `max_new_tokens=256`.

The four nested loops iterate in pair > gamma > temperature > task order. This ordering matters for `sweep.py` because all configs within a pair share the same loaded models — grouping by pair minimizes model load/unload cycles.

## Design Decisions

- **Dataclasses over dicts**: Provides IDE autocompletion, type checking, and immutability signals. `asdict()` gives free JSON serialization.
- **VRAM estimates as fields, not runtime checks**: Estimates are informational. Runtime VRAM depends on batch size, sequence length, and KV cache growth, making pre-checks unreliable.
- **Constants as module-level lists**: Imported directly by other modules (`from config import ALL_GAMMAS`), avoiding indirection through config files or environment variables.
