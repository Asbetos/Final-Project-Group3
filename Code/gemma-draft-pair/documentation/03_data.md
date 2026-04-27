# `data.py`

## Purpose

`data.py` loads benchmark prompts, formats them for chat-style Gemma models, and tokenizes them onto the target device.

## Supported Tasks

The tasks are stored in `TASK_REGISTRY`:

1. `humaneval`
2. `triviaqa`
3. `cnn_dailymail`
4. `writingprompts`

Each task entry contains:

1. dataset name and split
2. prompt field name
3. system prompt
4. max prompt token length

## Important Functions

1. `load_prompts(task, num_prompts=50, seed=42)`
   Loads and deterministically shuffles prompts.
2. `format_prompt_for_chat(raw_prompt, system_prompt, tokenizer)`
   Applies the tokenizer's chat template when available.
3. `tokenize_prompts(task, tokenizer, ...)`
   Returns `input_ids` and `attention_mask` tensors on the chosen device.

## Template Handling

The file uses the tokenizer's native chat template whenever possible. If the tokenizer has no template or throws an exception, it falls back to a plain text `System/User/Assistant` structure.

## Current Scope Notes

1. This file is used by both the baseline and speculative decoding paths.
2. It still contains generic tokenizer-family helpers such as `_is_qwen_tokenizer(...)` because the shared codebase supports multiple model families.
