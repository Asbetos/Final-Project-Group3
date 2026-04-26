## Gemma Speculative Decoding

This project benchmarks speculative decoding on active **Gemma** model pairs and trains an **EAGLE-3** draft head for the Gemma-4-31B target.

Active scope:
- Standard speculative decoding: `F`, `G`
- EAGLE-3: `H`
- Qwen artifacts and historical results are archived under `Code/archive/qwen_legacy/`

### Active Pairs

| Pair | Method | Target Model | Draft Model / Head | Quantization | Estimated VRAM |
|---|---|---|---|---|---|
| `F` | Standard speculative | `google/gemma-3-12b-it` | `google/gemma-3-1b-it` | 4-bit target | ~8.6 GB |
| `G` | Standard speculative | `google/gemma-4-31B` | `google/gemma-3-1b-it` | 4-bit target | ~17.5 GB |
| `H` | EAGLE-3 | `google/gemma-4-31B` | trained EAGLE-3 draft head | 4-bit target | ~16.3 GB inference |

### Hardware

- Local benchmarking target: AWS `g5.2xlarge` (`A10G`, 24 GB VRAM)
- Recommended EAGLE-3 training target for pair `H`: `H100` on Lightning.ai
- System RAM: `32 GB+`
- Disk: enough for Gemma model caches and checkpoints (`60 GB+` practical minimum)

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Gemma model downloads may require Hugging Face access approval and authentication.

### Quick Start

```bash
source venv/bin/activate

# Unit tests (no GPU)
python3 test_correctness.py --level 1

# Standard speculative smoke test on Gemma pair F
python3 test_correctness.py --level 3 --pair F

# EAGLE-3 smoke test on Gemma pair H
python3 test_correctness.py --level 6 --pair H

# Preview the active Gemma sweep
python3 sweep.py --dry-run --eagle3
```

### Run Standard Gemma Benchmarks

```bash
# Pair F only
python3 sweep.py --pairs F

# Pair G only
python3 sweep.py --pairs G

# Both active standard pairs
python3 sweep.py --pairs F G
```

### Train The Active EAGLE-3 Head

Pair `H` is the only active EAGLE-3 training target.

```bash
python3 eagle3_train.py \
  --target-model google/gemma-4-31B \
  --target-4bit \
  --checkpoint-dir checkpoints/eagle3/gemma4_31b \
  --final-checkpoint-name eagle3_gemma4_31b_final.pt
```

Recommended training environment:
- `H100` on Lightning.ai
- keep `--batch-size 1`
- keep `--grad-accum 8` unless profiling suggests otherwise

Quick startup validation on a smaller box:

```bash
python3 eagle3_train.py \
  --num-samples 8 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-seq-len 128 \
  --checkpoint-dir checkpoints/eagle3/gemma4_31b_smoke
```

### Run EAGLE-3 Benchmarks

```bash
export EAGLE3_GEMMA4_CHECKPOINT="$PWD/checkpoints/eagle3/gemma4_31b/eagle3_gemma4_31b_final.pt"
python3 sweep.py --eagle3 --eagle3-only --eagle3-pairs H
```

### App

The Streamlit demo covers the active standard Gemma pairs only (`F`, `G`).

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Outputs

```text
results/
  baseline/
  speculative/
  eagle3/
  summary.csv

checkpoints/
  eagle3/
    gemma4_31b/
      eagle3_gemma4_31b_final.pt

archive/
  qwen_legacy/
```

### Validation Checklist

```bash
python3 test_correctness.py --level 1
python3 test_correctness.py --level 3 --pair F
python3 test_correctness.py --level 3 --pair G
python3 test_correctness.py --level 6 --pair H
python3 sweep.py --dry-run --pairs F G --eagle3 --eagle3-pairs H
```

### Notes

- Active development is Gemma-only.
- Qwen checkpoints, logs, and results are retained only as archived legacy artifacts.
- If you need to revisit Qwen comparisons, use the archived data rather than the active configs and scripts.
