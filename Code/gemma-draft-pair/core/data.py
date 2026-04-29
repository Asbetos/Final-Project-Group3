"""Dataset loading, prompt formatting, and tokenization."""

import logging
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _fallback_chat_text(system_prompt: str, user_prompt: str, assistant_text: str = "") -> str:
    """Format a simple text prompt when the tokenizer has no chat template."""
    sections = [
        f"System: {system_prompt}",
        f"User: {user_prompt}",
    ]
    if assistant_text:
        sections.append(f"Assistant: {assistant_text}")
    else:
        sections.append("Assistant:")
    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# Task registry — all dataset-specific details in one declarative structure
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, dict] = {
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {},
        "split": "test",
        "field": "prompt",
        "system_prompt": "Complete the following Python function.",
        "max_prompt_tokens": 512,
    },
    "triviaqa": {
        "load_args": ("mandarjoshi/trivia_qa", "rc"),
        "load_kwargs": {},
        "split": "validation",
        "field": "question",
        "system_prompt": "Answer the following trivia question concisely.",
        "max_prompt_tokens": 256,
    },
    "cnn_dailymail": {
        "load_args": ("abisee/cnn_dailymail", "3.0.0"),
        "load_kwargs": {},
        "split": "test",
        "field": "article",
        "system_prompt": "Summarize the following article in a few sentences.",
        "max_prompt_tokens": 1024,
    },
    "writingprompts": {
        "load_args": ("euclaise/writingprompts",),
        "load_kwargs": {},
        "split": "validation",
        "field": "prompt",
        "system_prompt": "Write a creative story continuation based on this prompt.",
        "max_prompt_tokens": 256,
    },
}


def load_prompts(task: str, num_prompts: int = 50, seed: int = 42) -> List[str]:
    """
    Load raw text prompts from a HuggingFace dataset.

    Shuffles deterministically with *seed*, then takes the first *num_prompts*.
    """
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_REGISTRY)}")

    info = TASK_REGISTRY[task]
    logger.info("Loading dataset for task '%s' ...", task)

    ds = load_dataset(*info["load_args"], **info["load_kwargs"])
    split = ds[info["split"]]
    split = split.shuffle(seed=seed)

    n = min(num_prompts, len(split))
    subset = split.select(range(n))

    field = info["field"]
    prompts = [row[field] for row in subset]

    logger.info("Loaded %d prompts for task '%s'", len(prompts), task)
    return prompts


def _is_qwen_tokenizer(tokenizer: AutoTokenizer) -> bool:
    """Detect if tokenizer is from the Qwen family (supports enable_thinking)."""
    name = getattr(tokenizer, "name_or_path", "") or ""
    cls_name = type(tokenizer).__name__
    return "qwen" in name.lower() or "qwen" in cls_name.lower()


def format_prompt_for_chat(
    raw_prompt: str,
    system_prompt: str,
    tokenizer: AutoTokenizer,
) -> str:
    """
    Apply the appropriate chat template for the given tokenizer family.

    Qwen3 tokenizers support `enable_thinking=False` to suppress chain-of-thought
    tokens. Gemma and other tokenizers do not accept this argument.

    Returns the formatted string (not yet tokenized).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_prompt},
    ]

    if not getattr(tokenizer, "chat_template", None):
        return _fallback_chat_text(system_prompt, raw_prompt)

    try:
        if _is_qwen_tokenizer(tokenizer):
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        formatted = _fallback_chat_text(system_prompt, raw_prompt)
    return formatted


def tokenize_prompts(
    task: str,
    tokenizer: AutoTokenizer,
    num_prompts: int = 50,
    seed: int = 42,
    device: str = "cuda",
) -> List[Dict[str, torch.Tensor]]:
    """
    Full pipeline: load -> format -> tokenize.

    Returns a list of dicts, each with "input_ids" and "attention_mask" tensors
    on the specified device. Prompts are truncated to max_prompt_tokens.
    """
    info = TASK_REGISTRY[task]
    raw_prompts = load_prompts(task, num_prompts, seed)

    tokenized = []
    for raw in raw_prompts:
        text = format_prompt_for_chat(raw, info["system_prompt"], tokenizer)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=info["max_prompt_tokens"],
        )
        tokenized.append(
            {
                "input_ids": encoded["input_ids"].to(device),
                "attention_mask": encoded["attention_mask"].to(device),
            }
        )

    logger.info(
        "Tokenized %d prompts for '%s' (max_tokens=%d)",
        len(tokenized),
        task,
        info["max_prompt_tokens"],
    )
    return tokenized
