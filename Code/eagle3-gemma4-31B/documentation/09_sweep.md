# sweep.py — CLI Entry Point for the Full Experiment Grid

## Purpose

This is the command-line entry point for running the entire experiment grid. It parses arguments, computes the experiment matrix, supports a dry-run preview mode, and dispatches execution to `runner.py` one model pair at a time. This is the script users invoke directly to start benchmarking.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `argparse` | `argparse` | Standard library argument parser for CLI flags (`--pairs`, `--gammas`, `--temps`, etc.). |
| `logging` | `logging` | Configures root logger with timestamped format and stdout handler. |
| `os` | `os` | `os.makedirs` for creating output subdirectories before the sweep starts. |
| `sys` | `sys` | `sys.stdout` as the logging stream handler target. |
| `config` (local) | `ALL_GAMMAS`, `ALL_TASKS`, `ALL_TEMPERATURES`, `PAIR_MAP`, `ALL_PAIRS` | Default hyperparameter lists and the pair ID-to-config mapping. |

Note: `runner` is imported **lazily** inside `main()` (line 103) to allow `--dry-run` to work without a GPU or heavy dependencies like PyTorch.

## Inputs and Outputs

- **Inputs**: Command-line arguments controlling which pairs, gammas, temperatures, and tasks to run.
- **Outputs**: No return value. Side effects are the result files written by `runner.py` (JSON per config, master CSV).

## Detailed Line-by-Line Explanation

### Logging Configuration (lines 11-18)

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
```

Configures the root logger at module import time. Key choices:
- **`level=logging.INFO`**: Shows progress messages but not debug-level noise.
- **Format string**: Includes timestamp, log level, and logger name. The logger name (e.g., `runner`, `models`) helps identify which module produced each message during multi-hour sweeps.
- **`sys.stdout`** (not `sys.stderr`): When running in a terminal or redirecting output to a file, stdout is easier to capture. Some logging setups use stderr by default, which can cause confusion when piping output.

### parse_args() (lines 22-78)

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Experiment Sweep on Qwen3"
    )
```

Defines all CLI arguments:

**`--pairs` (lines 27-32):**
```python
parser.add_argument(
    "--pairs", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"],
    help="Model pair IDs to run (default: A B C)",
)
```
- `nargs="+"`: Accepts one or more values (e.g., `--pairs A B`).
- `choices`: Restricts input to valid pair IDs. Argparse raises an error for invalid values before any models are loaded, providing fail-fast validation.
- Default: All three pairs.

**`--gammas` (lines 33-38):**
```python
parser.add_argument(
    "--gammas", nargs="+", type=int, default=ALL_GAMMAS,
)
```
- `type=int`: Automatically converts string inputs to integers. Invalid values (e.g., `--gammas abc`) cause an argparse error.
- Default: `[1, 2, 3, 5, 10]` from `config.py`.

**`--temps` (lines 39-44):**
```python
parser.add_argument(
    "--temps", nargs="+", type=float, default=ALL_TEMPERATURES,
)
```
- `type=float`: Handles both `0` and `0.6` inputs.
- Default: `[0.0, 0.6, 1.0]` from `config.py`.

**`--tasks` (lines 45-50):**
```python
parser.add_argument(
    "--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS,
)
```
- `choices=ALL_TASKS`: Validates against the four registered tasks (humaneval, triviaqa, cnn_dailymail, writing_prompts).

**`--max-tokens` (lines 51-55):**
Default 128 tokens per prompt. This controls output length, not prompt length (prompt truncation is handled by `data.py`).

**`--num-prompts` (lines 56-60):**
Default 50 prompts per task. With 50 prompts and 4 tasks, each model pair processes 200 baseline prompts + 200 speculative prompts per (gamma, temperature) combination.

**`--output-dir` (lines 61-65):**
Default `"results"`. All JSON and CSV files are written under this directory.

**`--seed` (lines 66-70):**
Default 42. Controls dataset shuffling and random sampling. Same seed = same results.

**`--dry-run` (lines 71-74):**
```python
parser.add_argument("--dry-run", action="store_true")
```
`action="store_true"` makes this a boolean flag (no value needed). When set, the script prints the experiment grid and exits without loading models or running anything.

### main() (lines 77-129)

**Lines 78-79 — Pair resolution:**
```python
args = parse_args()
pairs = [PAIR_MAP[pid] for pid in args.pairs]
```
Converts pair ID strings (`"A"`, `"B"`, `"C"`) to `ModelPairConfig` objects via the `PAIR_MAP` dictionary.

**Lines 81-91 — Grid size computation:**
```python
total_baseline = len(args.tasks) * len(args.temps) * len(pairs)
total_spec = len(pairs) * len(args.gammas) * len(args.temps) * len(args.tasks)
total = total_baseline + total_spec
```
Computes the total number of configurations for the log message. For the full grid: 3 pairs x 5 gammas x 3 temps x 4 tasks = 180 speculative + 36 baseline = 216 total.

**Lines 93-114 — Dry-run mode:**
```python
if args.dry_run:
    print(f"\n{'='*70}")
    print("DRY RUN — would execute the following configurations:")
    ...
    for pair in pairs:
        print(f"Pair {pair.pair_id}: {pair.target_model_id} + {pair.draft_model_id}")
        print(f"  4-bit target: {pair.target_quantize_4bit}")
        print(f"  VRAM estimate: {pair.total_vram_estimate_gb:.1f} GB\n")
        for task in args.tasks:
            for temp in args.temps:
                print(f"  [BASELINE] {task} t={temp}")
        for gamma in args.gammas:
            for temp in args.temps:
                for task in args.tasks:
                    print(f"  [SPEC] {task} gamma={gamma} t={temp}")
```

Prints every configuration that would be executed. This is valuable for:
1. **Verification**: Users can confirm the grid is correct before starting a multi-hour run.
2. **Time estimation**: Knowing the total config count helps estimate runtime.
3. **No GPU required**: Because heavy imports are deferred, `--dry-run` works on any machine.

**Lines 116-117 — Lazy import:**
```python
from runner import run_pair_sweep
```
The `runner` module imports `torch`, model loading, and decoding — all heavy dependencies. Deferring this import to after the dry-run check means `python sweep.py --dry-run` works instantly without loading PyTorch.

**Lines 120-121 — Output directory creation:**
```python
os.makedirs(os.path.join(args.output_dir, "baseline"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "speculative"), exist_ok=True)
```
Creates the directory structure upfront. `exist_ok=True` prevents errors if directories already exist from a previous run.

**Lines 124-134 — Per-pair execution:**
```python
for pair in pairs:
    logger.info("=" * 70)
    logger.info("Starting pair %s", pair.pair_id)
    logger.info("=" * 70)
    run_pair_sweep(
        pair,
        gammas=args.gammas,
        temperatures=args.temps,
        tasks=args.tasks,
        max_new_tokens=args.max_tokens,
        num_prompts=args.num_prompts,
        seed=args.seed,
        output_dir=args.output_dir,
    )
```

Pairs are processed sequentially because only one pair's models fit in VRAM at a time. `run_pair_sweep()` handles model loading, all baseline and speculative runs, and model unloading for each pair.

**Line 136:**
```python
logger.info("All experiments complete. Results in: %s", args.output_dir)
```
Final log message confirming the sweep completed successfully.

## Design Decisions

- **Lazy import of `runner`**: Enables instant `--dry-run` without loading PyTorch or initializing CUDA. This makes the script useful for grid planning even on machines without GPUs.
- **Sequential pair execution**: Models are loaded per-pair because the A10G (24 GB) cannot fit two pairs simultaneously. The runner handles model lifecycle within each pair.
- **Argparse with choices validation**: Invalid pair IDs, task names, etc. are caught before any expensive computation starts. The error messages are auto-generated by argparse and include the valid options.
- **Flat directory structure for output**: `results/baseline/` and `results/speculative/` with one JSON per config. This makes individual results easy to inspect and is simpler than deeply nested directory hierarchies.
- **Logging to stdout**: Allows easy redirection (`python sweep.py > sweep.log 2>&1`) for monitoring long-running experiments. The timestamp format makes it easy to compute elapsed time between log entries.
