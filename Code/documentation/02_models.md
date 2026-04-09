# models.py — Model Loading, Tokenizer Setup, and VRAM Management

## Purpose

This script handles all interactions with GPU model lifecycle: loading Qwen3 models from HuggingFace Hub, configuring precision and quantization, placing models on the GPU, and freeing VRAM when models are no longer needed. It is the only module that calls `AutoModelForCausalLM.from_pretrained()`.

## Packages Used

| Package | Import | Why |
|---|---|---|
| `gc` | `gc` | Python's garbage collector. Called explicitly during model unloading to ensure Python objects referencing GPU tensors are freed before calling CUDA cache cleanup. |
| `logging` | `logging` | Logs VRAM usage after each model load and unload for monitoring memory pressure. |
| `torch` | `torch` | Checks CUDA availability, queries allocated VRAM (`torch.cuda.memory_allocated()`), and clears GPU caches. |
| `transformers` | `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig` | HuggingFace Transformers library for model and tokenizer loading. `BitsAndBytesConfig` configures 4-bit quantization. |
| `config` (local) | `ModelPairConfig` | Dataclass defining which models to load and whether to quantize. |

## Inputs and Outputs

- **Inputs**: `ModelPairConfig` from `config.py`, HuggingFace model IDs (strings).
- **Outputs**: PyTorch model objects on CUDA and a tokenizer instance.

## Detailed Line-by-Line Explanation

### get_device() (lines 14-18)

```python
def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    return torch.device("cuda:0")
```

Returns `cuda:0` or raises immediately. The project is designed exclusively for single-GPU inference — there is no CPU fallback. Failing fast here prevents confusing errors later when tensors or models end up on the wrong device.

### load_tokenizer() (lines 21-26)

```python
def load_tokenizer(model_id: str = "Qwen/Qwen3-0.6B") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
```

- `trust_remote_code=True`: Required for Qwen3 models which include custom tokenizer code in their HuggingFace repository.
- **Pad token fallback**: Many causal LMs (including Qwen3) do not define a pad token since they are trained without padding. Setting `pad_token = eos_token` prevents errors when batching or when the tokenizer expects a pad token for attention mask generation.
- The tokenizer is loaded from the **draft** model ID by default. All Qwen3 sizes share the same tokenizer (vocab_size=151,936), so the choice of model ID is arbitrary — the draft model is used simply because it is smaller to download.

### load_model() (lines 29-71)

```python
def load_model(model_id: str, quantize_4bit: bool = False, device: torch.device = None):
```

This is the core model loading function. Key decisions explained:

**Lines 47-50 — Base loading kwargs:**
```python
load_kwargs = dict(
    trust_remote_code=True,
    dtype=torch.float16,
)
```
- `trust_remote_code=True`: Required for Qwen3's custom attention implementation.
- `dtype=torch.float16`: Loads weights in half-precision (16-bit floating point). This halves VRAM usage compared to fp32 and is the standard precision for inference on consumer/mid-tier GPUs. The A10G supports fp16 natively with no performance penalty.

**Lines 52-62 — Quantization config:**
```python
if quantize_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs["quantization_config"] = bnb_config
```
- `load_in_4bit=True`: Enables 4-bit weight quantization.
- `bnb_4bit_quant_type="nf4"`: Uses NormalFloat4 quantization, which is optimized for normally-distributed neural network weights. NF4 provides better accuracy than uniform 4-bit quantization.
- `bnb_4bit_compute_dtype=torch.float16`: Even though weights are stored in 4-bit, computation happens in fp16. Weights are dequantized on-the-fly during matrix multiplications.
- `bnb_4bit_use_double_quant=True`: Quantizes the quantization constants themselves, saving an additional ~0.4 bits per parameter. Reduces the 8B model from ~16.7 GB to ~4.3 GB.

**Line 62 — `device_map="auto"`**: Tells the `accelerate` library to automatically place model layers on available GPUs (or split across GPU + CPU if needed). For single-GPU setups, this places everything on `cuda:0`.

**Line 66 — `model.eval()`**: Sets the model to evaluation mode. Disables dropout and batch normalization training behavior. Essential for deterministic inference.

**Lines 68-69 — VRAM logging:**
```python
vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
```
Logs the cumulative GPU memory after loading. This reports the actual allocated memory (not the peak), giving accurate per-model VRAM accounting.

### load_model_pair() (lines 74-102)

```python
def load_model_pair(pair: ModelPairConfig):
```

Orchestrates loading all three components for an experiment pair:
1. **Tokenizer first** (line 83): Lightweight, goes to CPU. Loaded from the draft model ID.
2. **Target model second** (lines 86-89): The largest model. Loaded first so that if it causes an OOM error, we fail fast before wasting time loading the draft.
3. **Draft model third** (line 92): Always loaded in fp16 (never quantized). Even for Pair C where the target is quantized, the draft stays in fp16 — it is small enough that quantization would save negligible memory while potentially degrading acceptance rates.

VRAM is logged after each model load to help diagnose memory issues.

### unload_models() (lines 105-113)

```python
def unload_models(*models) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
```

Three-step cleanup:
1. **`del model`**: Removes the Python reference. Without this, the garbage collector cannot reclaim the GPU tensors.
2. **`gc.collect()`**: Forces garbage collection immediately rather than waiting for Python's periodic GC cycle. This is critical because PyTorch tensors have custom deleters that free CUDA memory.
3. **`torch.cuda.empty_cache()`**: Returns freed CUDA memory to the CUDA memory pool so it can be reused. Without this call, PyTorch's caching allocator holds onto freed memory blocks.
4. **`torch.cuda.reset_peak_memory_stats()`**: Resets the peak VRAM counter so the next pair's peak measurement starts from zero.

## Design Decisions

- **No model caching across pairs**: Models are loaded and unloaded per-pair. Caching would exceed the 24 GB VRAM budget for most pair combinations.
- **Target loaded before draft**: Fail-fast strategy. If the 8B target OOMs, we don't waste time loading the smaller draft model first.
- **Draft never quantized**: The 0.6B and 1.7B draft models are small enough in fp16 (~1.2 GB and ~3.4 GB). Quantizing them would save negligible VRAM while potentially reducing draft quality and acceptance rates.
