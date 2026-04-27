# `data.py`

## Purpose

`data.py` provides the evaluation prompt pipeline used by both standard speculative decoding and EAGLE evaluation.

This file handles:

1. Dataset selection.
2. Prompt formatting.
3. Tokenization.
4. Device placement.

Training data for EAGLE is not handled here. That logic lives in `eagle3_train.py`.

## Supported Tasks

The evaluation tasks are stored in `TASK_REGISTRY`:

1. `humaneval`
2. `triviaqa`
3. `cnn_dailymail`
4. `writingprompts`

Each registry entry defines:

1. Hugging Face dataset source.
2. Split name.
3. Text field to read.
4. System prompt.
5. Maximum prompt length.

## Important Functions

1. `load_prompts(task, num_prompts=50, seed=42)`
   Loads raw text prompts from the configured dataset and shuffles deterministically.
2. `format_prompt_for_chat(raw_prompt, system_prompt, tokenizer)`
   Applies the tokenizer chat template when available.
3. `tokenize_prompts(task, tokenizer, ...)`
   Runs the full pipeline and returns tensors already moved to the target device.

## Template Handling

The module tries to use the model tokenizer's native chat template first. If that fails, it falls back to plain text sections:

1. `System:`
2. `User:`
3. `Assistant:`

The helper `_is_qwen_tokenizer(...)` remains because the shared codebase still contains generic handling for tokenizer families that expose `enable_thinking=False`.

## Output Format

`tokenize_prompts(...)` returns a list of dictionaries, each containing:

1. `input_ids`
2. `attention_mask`

Both tensors are moved to the requested device, which defaults to `cuda`.

## Current Scope Notes

1. This file is evaluation-only.
2. EAGLE training examples come from Alpaca GPT-4 in `eagle3_train.py`.
3. The prompt preparation logic is shared across baseline, speculative, and EAGLE evaluation paths.
