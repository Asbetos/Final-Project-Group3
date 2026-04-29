"""Model loading, tokenizer setup, and VRAM management."""

import gc
import logging
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - older transformers fallback
    AutoModelForImageTextToText = None

from core.config import ModelPairConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA GPU optimizations — enable TF32 for any stray FP32 matmuls
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    """Return the CUDA device, raising if no GPU is available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available")
    return torch.device("cuda:0")


def load_tokenizer(model_id: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,)
    except AttributeError as e:
        # Gemma 4 tokenizer workaround:
        # HF tokenizer_config can expose extra_special_tokens as a list,
        # but this loading path expects a mapping-like object.
        if "has no attribute 'keys'" in str(e) and "gemma-4" in model_id.lower():
            print(f"[WARN] Retrying tokenizer load for {model_id} with sanitized extra_special_tokens")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                extra_special_tokens={},
            )
        else:
            raise

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    quantize_4bit: bool = False,
    device: torch.device = None,
    compile_model: bool = False,
) -> torch.nn.Module:
    """
    Load a single target model for inference or EAGLE-3 training.

    Args:
        model_id: HuggingFace model identifier.
        quantize_4bit: If True, apply NF4 4-bit quantization via bitsandbytes.
        compile_model: Whether to apply torch.compile (kept False for stability).

    Returns:
        The loaded model in eval mode with gradients disabled.
    """
    logger.info("Loading model %s (4-bit=%s) ...", model_id, quantize_4bit)
    os.makedirs("offload", exist_ok=True)

    #1)  Inspect model config and choose the right HF model class
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    architectures = getattr(model_config, "architectures", []) or []
    is_conditional_generation = any(
        "ConditionalGeneration" in architecture for architecture in architectures
    )
    model_cls = AutoModelForCausalLM
    if is_conditional_generation and AutoModelForImageTextToText is not None:
        model_cls = AutoModelForImageTextToText

    # 2) Build loading kwargs
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "offload_folder": "offload",
        "offload_state_dict": True,
        "torch_dtype": torch.bfloat16,
    }

    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 3) Load model
    model = model_cls.from_pretrained(model_id, **load_kwargs)
    model.eval()

    # torch.compile disabled — CUDA graphs conflict with dynamic KV cache
    # in transformers 5.x + PyTorch 2.4. SDPA on modern GPUs is already fast.
    if False and compile_model:
        logger.info("Compiling model %s with torch.compile ...", model_id)
        model = torch.compile(model, mode="reduce-overhead")

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

    tokenizer = load_tokenizer(pair.target_model_id)

    # Load target first (larger) — fail fast on OOM
    target_model = load_model(
        pair.target_model_id,
        quantize_4bit=pair.target_quantize_4bit,
        compile_model=False,
    )
    target_vram = torch.cuda.memory_allocated() / (1024 ** 3)

    draft_model = load_model(
        pair.draft_model_id,
        quantize_4bit=pair.draft_quantize_4bit,
        compile_model=False,
    )
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
