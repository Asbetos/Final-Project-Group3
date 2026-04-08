"""Model loading, tokenizer setup, and VRAM management."""

import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ModelPairConfig

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the CUDA device, raising if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    return torch.device("cuda:0")


def load_tokenizer(model_id: str = "Qwen/Qwen3-0.6B") -> AutoTokenizer:
    """Load the shared Qwen3 tokenizer (vocab_size=151936, identical across sizes)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    quantize_4bit: bool = False,
    device: torch.device = None,
) -> AutoModelForCausalLM:
    """
    Load a single Qwen3 model for inference.

    Args:
        model_id: HuggingFace model identifier (e.g. "Qwen/Qwen3-8B").
        quantize_4bit: If True, apply NF4 4-bit quantization via bitsandbytes.
        device: Target device (defaults to cuda:0).

    Returns:
        The loaded model in eval mode with gradients disabled.
    """
    device = device or get_device()

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    logger.info("Loading model %s (4-bit=%s) ...", model_id, quantize_4bit)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info("Model %s loaded. VRAM allocated: %.2f GB", model_id, vram_gb)

    return model


def load_model_pair(pair: ModelPairConfig):
    """
    Load target model, draft model, and shared tokenizer for a given pair.

    Returns:
        (target_model, draft_model, tokenizer)
    """
    logger.info("Loading model pair %s ...", pair.pair_id)

    tokenizer = load_tokenizer(pair.draft_model_id)

    # Load target first (larger) — fail fast on OOM
    target_model = load_model(
        pair.target_model_id,
        quantize_4bit=pair.target_quantize_4bit,
    )
    target_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    draft_model = load_model(pair.draft_model_id, quantize_4bit=False)
    total_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    logger.info(
        "Pair %s loaded. Target VRAM: %.2f GB, Total VRAM: %.2f GB",
        pair.pair_id,
        target_vram,
        total_vram,
    )

    return target_model, draft_model, tokenizer


def unload_models(*models) -> None:
    """Delete models and free GPU memory."""
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.info("Models unloaded. VRAM freed.")
