"""
Core speculative decoding loop with full instrumentation.

Implements the speculative sampling algorithm (Leviathan et al., ICML 2023)
from scratch with per-token acceptance tracking, CUDA timing, and KV-cache
management.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

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
from sampling import (
    rejection_sample_token,
    sample_bonus_token,
    sample_from_logits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------


def _get_cache_seq_len(past_key_values) -> int:
    """Return the current sequence length stored in the KV-cache."""
    if past_key_values is None:
        return 0
    # DynamicCache with get_seq_length method (transformers >= 5.0)
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # DynamicCache with key_cache attribute (transformers 4.36–4.x)
    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return 0
        return past_key_values.key_cache[0].shape[2]
    # Legacy tuple format
    return past_key_values[0][0].shape[2]


def _trim_kv_cache(past_key_values, target_seq_len: int):
    """
    Trim KV-cache to keep only the first *target_seq_len* positions.

    Supports DynamicCache (transformers 5.x and 4.x) and legacy tuple-of-tuples.
    Returns the trimmed cache (modifies in-place for DynamicCache).
    """
    if past_key_values is None:
        return None

    current_len = _get_cache_seq_len(past_key_values)
    if current_len <= target_seq_len:
        return past_key_values

    # DynamicCache with crop method (transformers >= 5.0)
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(target_seq_len)
        return past_key_values

    # DynamicCache with key_cache attribute (transformers 4.36–4.x)
    if hasattr(past_key_values, "key_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = (
                past_key_values.key_cache[layer_idx][:, :, :target_seq_len, :]
            )
            past_key_values.value_cache[layer_idx] = (
                past_key_values.value_cache[layer_idx][:, :, :target_seq_len, :]
            )
        return past_key_values

    # Legacy tuple format
    return tuple(
        (k[:, :, :target_seq_len, :], v[:, :, :target_seq_len, :])
        for k, v in past_key_values
    )


# ---------------------------------------------------------------------------
# Draft step
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _draft_step(
    draft_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: int,
    temperature: float,
    draft_cache=None,
    draft_cache_len: int = 0,
    generator: torch.Generator = None,
) -> Dict:
    """
    Generate *gamma* tokens autoregressively from the draft model.

    Reuses the draft KV-cache across rounds to avoid redundant prefill.
    Only new tokens (those beyond draft_cache_len) are fed to the model.

    Returns dict with:
        "tokens": List[int]               — drafted token ids
        "probs":  List[torch.Tensor]      — full vocab distribution per token
        "draft_cache":                     — updated KV-cache (trimmed by caller)
        "elapsed_ms": float               — CUDA-timed duration
    """
    device = input_ids.device
    tokens: List[int] = []
    probs: List[torch.Tensor] = []

    current_input = input_ids
    current_mask = attention_mask

    with CudaTimer() as timer, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for step in range(gamma):
            if draft_cache is not None and draft_cache_len > 0:
                # Feed only tokens not yet in the cache
                new_start = draft_cache_len + step
                out = draft_model(
                    input_ids=current_input[:, new_start:],
                    attention_mask=current_mask,
                    past_key_values=draft_cache,
                    use_cache=True,
                )
            else:
                out = draft_model(
                    input_ids=current_input,
                    attention_mask=current_mask,
                    use_cache=True,
                )

            draft_cache = out.past_key_values
            logits = out.logits[:, -1, :]  # (1, vocab_size)

            token_id, token_probs = sample_from_logits(
                logits.squeeze(0), temperature, generator
            )
            tokens.append(token_id)
            probs.append(token_probs)

            # Extend for next iteration
            next_tok = torch.tensor(
                [[token_id]], device=device, dtype=input_ids.dtype
            )
            current_input = torch.cat([current_input, next_tok], dim=1)
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(1, 1, device=device, dtype=current_mask.dtype),
                ],
                dim=1,
            )

    return {
        "tokens": tokens,
        "probs": probs,
        "draft_cache": draft_cache,
        "elapsed_ms": timer.elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Verify step
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _verify_step(
    target_model: AutoModelForCausalLM,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    draft_tokens: List[int],
    draft_probs: List[torch.Tensor],
    gamma: int,
    temperature: float,
    target_cache,
    prefix_len_in_cache: int,
    generator: torch.Generator = None,
) -> Dict:
    """
    Run ONE target-model forward pass over all draft tokens and perform
    rejection sampling.

    Args:
        target_model: The large target model.
        prefix_ids: (1, total_len) — prompt + all previously accepted tokens + draft tokens.
        prefix_mask: (1, total_len) — corresponding attention mask.
        draft_tokens: List of gamma drafted token ids.
        draft_probs: List of gamma probability tensors (one per draft token).
        gamma: Speculation length.
        temperature: Sampling temperature.
        target_cache: Existing target KV-cache (or None for first round).
        prefix_len_in_cache: Number of positions already in the target cache.
        generator: Optional torch generator.

    Returns dict with:
        "accepted_tokens": List[int] — accepted + possibly correction/bonus token
        "num_accepted": int
        "per_token_accepted": List[bool]
        "target_cache": updated and trimmed target KV-cache
        "elapsed_ms": float
    """
    device = prefix_ids.device

    with CudaTimer() as timer, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Feed only new tokens (those after the cached prefix) to the target
        new_start = prefix_len_in_cache
        if target_cache is not None and new_start > 0:
            out = target_model(
                input_ids=prefix_ids[:, new_start:],
                attention_mask=prefix_mask,
                past_key_values=target_cache,
                use_cache=True,
            )
        else:
            out = target_model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )

        target_cache = out.past_key_values
        all_logits = out.logits  # (1, num_new_tokens, vocab_size)

        # The logits correspond to predictions for the NEXT token at each position.
        # With cache of length L, new input tokens are at positions L, L+1, ...
        # logits[j] (at position L+j) predicts the token at position L+j+1.
        # Without cache (L=0), logits[j] at position j predicts token at j+1.
        #
        # To verify draft token i at position (prefix_without_draft + i),
        # we need logits that predict that position, i.e. logits at position
        # (prefix_without_draft + i - 1). The logits index is:
        #   (prefix_without_draft + i - 1) - new_start

        # Number of tokens before the draft tokens
        prefix_without_draft = prefix_ids.shape[1] - gamma

        # Unified formula: works for both cached (new_start > 0) and
        # uncached (new_start == 0) cases.
        logit_offset = prefix_without_draft - new_start - 1

        # --- Rejection sampling ---
        accepted_tokens: List[int] = []
        per_token_accepted: List[bool] = []
        num_accepted = 0

        for i in range(gamma):
            idx = logit_offset + i
            target_logits_i = all_logits[0, idx, :]  # (vocab_size,)

            if temperature == 0.0:
                target_probs_i = torch.zeros_like(target_logits_i)
                target_probs_i[target_logits_i.argmax()] = 1.0
            else:
                target_probs_i = F.softmax(target_logits_i / temperature, dim=-1)

            accepted, token = rejection_sample_token(
                target_probs_i,
                draft_probs[i],
                draft_tokens[i],
                temperature,
                generator,
            )

            if accepted:
                accepted_tokens.append(draft_tokens[i])
                per_token_accepted.append(True)
                num_accepted += 1
            else:
                accepted_tokens.append(token)
                per_token_accepted.append(False)
                break

        # Bonus token: if all gamma accepted, sample one more
        bonus = False
        if num_accepted == gamma:
            bonus_idx = logit_offset + gamma
            if bonus_idx < all_logits.shape[1]:
                bonus_logits = all_logits[0, bonus_idx, :]
                if temperature == 0.0:
                    bonus_probs = torch.zeros_like(bonus_logits)
                    bonus_probs[bonus_logits.argmax()] = 1.0
                else:
                    bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                bonus_tok = sample_bonus_token(bonus_probs, temperature, generator)
                accepted_tokens.append(bonus_tok)
                bonus = True

        # Trim target cache to keep only positions with valid KV entries.
        # The model processed prefix_without_draft + gamma input tokens,
        # so the cache has that many entries. We keep only the prefix plus
        # the num_accepted draft tokens (whose KV is correct). Bonus and
        # correction tokens were never in the model input, so their KV
        # will be computed in the next round.
        keep_len = prefix_without_draft + num_accepted
        target_cache = _trim_kv_cache(target_cache, keep_len)

    return {
        "accepted_tokens": accepted_tokens,
        "num_accepted": num_accepted,
        "bonus": bonus,
        "per_token_accepted": per_token_accepted,
        "target_cache": target_cache,
        "elapsed_ms": timer.elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Main speculative decoding entry point
# ---------------------------------------------------------------------------


@torch.inference_mode()
def speculative_decode(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: int,
    temperature: float,
    max_new_tokens: int,
    tokenizer: AutoTokenizer,
    generator: torch.Generator = None,
) -> Dict:
    """
    Full speculative decoding for one prompt.

    Args:
        target_model: Large target model.
        draft_model: Small draft model.
        input_ids: (1, prompt_len) input token IDs.
        attention_mask: (1, prompt_len) attention mask.
        gamma: Number of tokens to draft per round.
        temperature: Sampling temperature (0.0 = greedy).
        max_new_tokens: Maximum number of new tokens to generate.
        tokenizer: For EOS detection and decoding.
        generator: Optional torch generator for reproducibility.

    Returns:
        dict with "output_ids", "output_text", and "metrics".
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    generated_ids: List[int] = []
    target_cache = None
    target_cache_len = 0  # positions currently in target cache
    draft_cache = None
    draft_cache_len = 0   # positions currently in draft cache

    all_rounds: List[RoundMetrics] = []
    total_draft_ms = 0.0

    reset_peak_vram()
    wall_start = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False

    while len(generated_ids) < max_new_tokens:
        round_start = time.perf_counter()
        current_len = prompt_len + len(generated_ids)

        # Remaining budget
        remaining = max_new_tokens - len(generated_ids)
        effective_gamma = min(gamma, remaining)
        if effective_gamma <= 0:
            break

        # Build the current full sequence for the draft model
        if generated_ids:
            gen_tensor = torch.tensor(
                [generated_ids], device=device, dtype=input_ids.dtype
            )
            full_ids = torch.cat([input_ids, gen_tensor], dim=1)
            full_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        1, len(generated_ids), device=device, dtype=attention_mask.dtype
                    ),
                ],
                dim=1,
            )
        else:
            full_ids = input_ids
            full_mask = attention_mask

        # ---- Draft phase ----
        draft_result = _draft_step(
            draft_model,
            full_ids,
            full_mask,
            effective_gamma,
            temperature,
            draft_cache,
            draft_cache_len,
            generator,
        )
        draft_tokens = draft_result["tokens"]
        draft_probs = draft_result["probs"]
        draft_cache = draft_result["draft_cache"]
        draft_ms = draft_result["elapsed_ms"]
        total_draft_ms += draft_ms

        # ---- Build verification input (prefix + draft tokens) ----
        draft_tensor = torch.tensor(
            [draft_tokens], device=device, dtype=input_ids.dtype
        )
        verify_ids = torch.cat([full_ids, draft_tensor], dim=1)
        verify_mask = torch.cat(
            [
                full_mask,
                torch.ones(
                    1,
                    len(draft_tokens),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        # ---- Verify phase ----
        verify_result = _verify_step(
            target_model,
            verify_ids,
            verify_mask,
            draft_tokens,
            draft_probs,
            effective_gamma,
            temperature,
            target_cache,
            target_cache_len,
            generator,
        )

        accepted_tokens = verify_result["accepted_tokens"]
        num_accepted = verify_result["num_accepted"]
        target_cache = verify_result["target_cache"]
        target_cache_len = current_len + verify_result["num_accepted"]
        verify_ms = verify_result["elapsed_ms"]

        # Trim draft cache to match accepted prefix length.
        # The draft cache currently covers: prefix + gamma draft tokens.
        # We keep only prefix + num_accepted (the accepted draft tokens).
        # The correction/bonus token is NOT in the draft cache and will be
        # processed as new input in the next round's draft step.
        draft_keep_len = current_len + num_accepted
        draft_cache = _trim_kv_cache(draft_cache, draft_keep_len)
        draft_cache_len = draft_keep_len

        # Append accepted tokens, clamping to max_new_tokens
        remaining = max_new_tokens - len(generated_ids)
        generated_ids.extend(accepted_tokens[:remaining])

        if not ttft_recorded and accepted_tokens:
            ttft_ms = (time.perf_counter() - wall_start) * 1000.0
            ttft_recorded = True

        round_ms = (time.perf_counter() - round_start) * 1000.0
        all_rounds.append(
            RoundMetrics(
                round_index=len(all_rounds),
                draft_tokens_proposed=effective_gamma,
                tokens_accepted=num_accepted,
                bonus_token_generated=verify_result["bonus"],
                total_tokens_produced=len(accepted_tokens),
                per_token_accepted=verify_result["per_token_accepted"],
                draft_time_ms=draft_ms,
                verify_time_ms=verify_ms,
                round_time_ms=round_ms,
            )
        )

        # Check for EOS
        if tokenizer.eos_token_id in accepted_tokens:
            break

    # ---- Finalize metrics ----
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    total_tokens = len(generated_ids)
    tps = (total_tokens / wall_ms * 1000.0) if wall_ms > 0 else 0.0

    # Acceptance rate: fraction of draft tokens that were accepted
    all_decisions = []
    for r in all_rounds:
        all_decisions.extend(r.per_token_accepted)
    acceptance_rate = (
        sum(all_decisions) / len(all_decisions) if all_decisions else 0.0
    )

    # Acceptance length: mean tokens accepted per round
    acceptance_length = (
        sum(r.tokens_accepted for r in all_rounds) / len(all_rounds)
        if all_rounds
        else 0.0
    )

    draft_overhead = total_draft_ms / wall_ms if wall_ms > 0 else 0.0

    metrics = GenerationMetrics(
        prompt_index=-1,  # set by caller
        total_tokens_generated=total_tokens,
        total_rounds=len(all_rounds),
        wall_clock_ms=wall_ms,
        ttft_ms=ttft_ms,
        tokens_per_second=tps,
        acceptance_rate=acceptance_rate,
        acceptance_length=acceptance_length,
        draft_overhead_ratio=draft_overhead,
        peak_vram_bytes=record_peak_vram(),
        rounds=all_rounds,
    )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "output_ids": generated_ids,
        "output_text": output_text,
        "metrics": metrics,
    }
