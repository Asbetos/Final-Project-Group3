"""Standard autoregressive decoder for baseline comparison."""

import logging
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import (
    CudaTimer,
    GenerationMetrics,
    RoundMetrics,
    record_peak_vram,
    reset_peak_vram,
)
from sampling import sample_from_logits

logger = logging.getLogger(__name__)


@torch.inference_mode()
def autoregressive_decode(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float,
    max_new_tokens: int,
    tokenizer: AutoTokenizer,
    generator: torch.Generator = None,
) -> dict:
    """
    Standard token-by-token autoregressive decoding with KV-cache.

    Uses the same sampling logic and timing instrumentation as the speculative
    path so comparisons are fair.

    Args:
        model: The target model.
        input_ids: (1, prompt_len) input token tensor.
        attention_mask: (1, prompt_len) mask tensor.
        temperature: Sampling temperature (0.0 = greedy).
        max_new_tokens: Maximum tokens to generate.
        tokenizer: For EOS detection.
        generator: Optional torch generator for reproducibility.

    Returns:
        dict with "output_ids", "output_text", and "metrics".
    """
    device = input_ids.device
    generated_ids: list[int] = []
    past_key_values = None
    current_input = input_ids
    current_mask = attention_mask

    reset_peak_vram()
    wall_start = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False

    for step in range(max_new_tokens):
        with CudaTimer() as timer:
            if past_key_values is not None:
                outputs = model(
                    input_ids=current_input[:, -1:],
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=current_input,
                    attention_mask=current_mask,
                    use_cache=True,
                )

        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # (1, vocab_size)

        token_id, _ = sample_from_logits(
            logits.squeeze(0), temperature, generator
        )
        generated_ids.append(token_id)

        if not ttft_recorded:
            ttft_ms = (time.perf_counter() - wall_start) * 1000.0
            ttft_recorded = True

        # Prepare next step
        next_token = torch.tensor([[token_id]], device=device, dtype=input_ids.dtype)
        current_input = torch.cat([current_input, next_token], dim=1)
        current_mask = torch.cat(
            [current_mask, torch.ones(1, 1, device=device, dtype=current_mask.dtype)],
            dim=1,
        )

        if token_id == tokenizer.eos_token_id:
            break

    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    total_tokens = len(generated_ids)
    tps = (total_tokens / wall_ms * 1000.0) if wall_ms > 0 else 0.0

    metrics = GenerationMetrics(
        prompt_index=-1,  # set by caller
        total_tokens_generated=total_tokens,
        total_rounds=total_tokens,
        wall_clock_ms=wall_ms,
        ttft_ms=ttft_ms,
        tokens_per_second=tps,
        acceptance_rate=1.0,       # N/A for baseline
        acceptance_length=1.0,     # always 1 token per step
        draft_overhead_ratio=0.0,  # no draft model
        peak_vram_bytes=record_peak_vram(),
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "output_ids": generated_ids,
        "output_text": output_text,
        "metrics": metrics,
    }
