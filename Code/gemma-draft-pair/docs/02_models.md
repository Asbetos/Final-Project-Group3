# `models.py`

## Purpose

`models.py` loads the target and draft models for standard speculative decoding and handles tokenizer setup plus cleanup.

## Main Responsibilities

1. Load tokenizers safely.
2. Load target and draft models.
3. Apply 4-bit quantization when requested.
4. Handle Gemma-4 tokenizer fallback logic.
5. Free model memory after a run.

## Important Functions

1. `get_device()`
   Returns `cuda:0` and raises if no GPU is available.
2. `load_tokenizer(model_id)`
   Loads the tokenizer and fills in `pad_token` if needed.
3. `load_model(model_id, quantize_4bit=False, ...)`
   Loads a single model with `device_map="auto"` and BF16 weights.
4. `load_model_pair(pair)`
   Loads the target model, the draft model, and the shared tokenizer.
5. `unload_models(*models)`
   Deletes model references, runs garbage collection, and clears CUDA memory.

## Quantization Path

When `quantize_4bit=True`, the file uses `BitsAndBytesConfig` with NF4 quantization and BF16 compute.

## Gemma-4 Tokenizer Fallback

`load_tokenizer(...)` contains a specific retry path for Gemma-4 tokenizer loading if Hugging Face returns problematic `extra_special_tokens` metadata. This is one of the leftovers that keeps pair `G` functioning.

## Current Scope Notes

1. For the active project workflow, the important pair is `F`.
2. Pair `G` support remains in this module.
3. The file also still contains `load_eagle3_pair(...)` from the shared codebase, but that path is not part of the active `gemma-draft-pair` run instructions.
