# `models.py`

## Purpose

`models.py` loads tokenizers, target models, standard draft models, and EAGLE draft heads. It is also responsible for GPU-only safeguards and cleanup.

## Main Responsibilities

1. Refuse to run without CUDA.
2. Load tokenizers with a valid pad token.
3. Load target models in BF16, optionally with 4-bit NF4 quantization.
4. Load standard speculative model pairs.
5. Load EAGLE target model plus trained draft head from checkpoint.
6. Free VRAM cleanly after a run.

## Important Functions

1. `get_device()`
   Returns `cuda:0` and raises if no GPU is available.
2. `load_tokenizer(model_id)`
   Loads the tokenizer and fills in `pad_token` if missing.
3. `load_model(model_id, quantize_4bit=False, ...)`
   Loads one target or draft model.
4. `load_model_pair(pair)`
   Loads a standard target-draft pair.
5. `load_eagle3_pair(pair)`
   Loads the target model, derives `Eagle3Config`, instantiates `Eagle3DraftHead`, and loads the saved checkpoint.
6. `unload_models(*models)`
   Deletes model references, runs garbage collection, and clears CUDA cache.

## Quantization Behavior

When `quantize_4bit=True`, the loader uses `BitsAndBytesConfig` with:

1. `load_in_4bit=True`
2. `bnb_4bit_quant_type="nf4"`
3. `bnb_4bit_compute_dtype=torch.bfloat16`
4. `bnb_4bit_use_double_quant=True`

## Special Loading Detail

The file imports and applies `patch_transformers_safetensors_loader()` from `safetensors_nommap.py` before model loading. This avoids the default mmap-based safetensors path on memory-constrained machines.

## EAGLE-Specific Behavior

`load_eagle3_pair(pair)` does the following:

1. Loads the target model.
2. Derives architecture dimensions from the loaded target via `Eagle3Config.from_model(...)`.
3. Instantiates the draft head against the target backbone.
4. Loads only the trainable draft-head weights from the checkpoint.
5. Returns:
   target model, draft head, derived config, tokenizer.

## Current Scope Notes

1. Pair `I` is the active EAGLE load path.
2. If `pair.checkpoint_path` is missing, `load_eagle3_pair` raises immediately.
3. The error text still mentions older environment variable names in one branch, so the root README should be treated as the runbook for the correct variable name.
