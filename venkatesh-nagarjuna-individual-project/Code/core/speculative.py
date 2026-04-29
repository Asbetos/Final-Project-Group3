"""
Core speculative decoding loop with full instrumentation.

Implements the speculative sampling algorithm (Leviathan et al., ICML 2023)
from scratch with per-token acceptance tracking, CUDA timing, and KV-cache
management.

Key optimizations over naive implementation:
  - Pre-allocated sequence buffer — no torch.cat inside the generation loop
  - In-place token writes inside _draft_step — no inner-loop concatenation
  - Vectorized _verify_step — batch softmax + single torch.rand(gamma) call
  - CudaTimer uses end_event.synchronize() — no global pipeline drain
"""

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.metrics import (
    CudaTimer,
    GenerationMetrics,
    RoundMetrics,
    record_peak_vram,
    reset_peak_vram,
)
from core.sampling import (
    batch_rejection_sample,
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
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "key_cache"):
        if len(past_key_values.key_cache) == 0:
            return 0
        return past_key_values.key_cache[0].shape[2]
    return past_key_values[0][0].shape[2]


def _trim_kv_cache(past_key_values, target_seq_len: int):
    """
    Trim KV-cache to keep only the first *target_seq_len* positions.

    Supports DynamicCache (transformers 5.x and 4.x) and legacy tuple-of-tuples.
    Returns the trimmed cache (modifies in-place for DynamicCache).

    Some Gemma sliding-window cache layers cannot be cropped once they have
    advanced past the sliding window. In that case we return None so the caller
    can rebuild the cache from the accepted prefix on the next round.
    """
    if past_key_values is None:
        return None

    current_len = _get_cache_seq_len(past_key_values)
    if current_len <= target_seq_len:
        return past_key_values

    if hasattr(past_key_values, "crop"):
        try:
            past_key_values.crop(target_seq_len)
        except ValueError as exc:
            if "DynamicSlidingWindowLayer" not in str(exc):
                raise
            logger.warning(
                "KV cache crop unsupported for sliding-window layer; rebuilding cache next round"
            )
            return None
        return past_key_values

    if hasattr(past_key_values, "key_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx] = (
                past_key_values.key_cache[layer_idx][:, :, :target_seq_len, :]
            )
            past_key_values.value_cache[layer_idx] = (
                past_key_values.value_cache[layer_idx][:, :, :target_seq_len, :]
            )
        return past_key_values

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
    Uses a pre-allocated in-place buffer instead of torch.cat inside the loop.

    Returns dict with:
        "tokens": List[int]                    — drafted token ids
        "probs":  List[Optional[torch.Tensor]] — full vocab dist per token (None if greedy)
        "draft_cache":                          — updated KV-cache (trimmed by caller)
        "elapsed_ms": float                    — CUDA-timed duration
    """
    device = input_ids.device
    cur_len = input_ids.shape[1]

    tokens: List[int] = []
    probs: List[Optional[torch.Tensor]] = []

    # Pre-allocate a buffer large enough to hold current input + all gamma draft tokens.
    # This avoids torch.cat inside the loop entirely.
    buf_ids = torch.empty(1, cur_len + gamma, dtype=input_ids.dtype, device=device)
    buf_ids[0, :cur_len] = input_ids[0]
    buf_mask = torch.ones(1, cur_len + gamma, dtype=attention_mask.dtype, device=device)
    buf_mask[0, :cur_len] = attention_mask[0]
    pos = cur_len  # next write position in the buffer

    with CudaTimer() as timer:
        for step in range(gamma):
            if draft_cache is not None and draft_cache_len > 0:
                new_start = draft_cache_len + step
                feed_ids = buf_ids[:, new_start:pos]
                if feed_ids.shape[1] == 0:
                    feed_ids = buf_ids[:, pos - 1:pos]
                out = draft_model(
                    input_ids=feed_ids,
                    attention_mask=buf_mask[:, :pos],
                    past_key_values=draft_cache,
                    use_cache=True,
                )
            else:
                out = draft_model(
                    input_ids=buf_ids[:, :pos],
                    attention_mask=buf_mask[:, :pos],
                    use_cache=True,
                )

            draft_cache = out.past_key_values
            logits = out.logits[:, -1, :]  # (1, vocab_size)

            token_id, token_probs = sample_from_logits(
                logits.squeeze(0), temperature, generator
            )
            tokens.append(token_id)
            probs.append(token_probs)

            # In-place write — no torch.cat
            buf_ids[0, pos] = token_id
            pos += 1

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
    draft_probs: List[Optional[torch.Tensor]],
    gamma: int,
    temperature: float,
    target_cache,
    prefix_len_in_cache: int,
    generator: torch.Generator = None,
) -> Dict:
    """
    Run ONE target-model forward pass over all draft tokens and perform
    vectorized rejection sampling.

    Args:
        target_model: The large target model.
        prefix_ids: (1, total_len) — prompt + all previously accepted tokens + draft tokens.
        prefix_mask: (1, total_len) — corresponding attention mask.
        draft_tokens: List of gamma drafted token ids.
        draft_probs: List of gamma probability tensors (None entries for greedy drafts).
        gamma: Speculation length (effective, may be < config gamma near end).
        temperature: Sampling temperature.
        target_cache: Existing target KV-cache (or None for first round).
        prefix_len_in_cache: Number of positions already in the target cache.
        generator: Optional torch generator.

    Returns dict with:
        "accepted_tokens": List[int]
        "num_accepted": int
        "per_token_accepted": List[bool]
        "target_cache": updated and trimmed target KV-cache
        "elapsed_ms": float
    """
    device = prefix_ids.device

    with CudaTimer() as timer:
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

        # Compute logit index offset.
        # logits[j] at position (new_start + j) predicts the token at (new_start + j + 1).
        # To verify draft token i at position (prefix_without_draft + i), we need
        # logits at position (prefix_without_draft + i - 1), i.e., index:
        #   (prefix_without_draft - 1) - new_start + i  =  logit_offset + i
        prefix_without_draft = prefix_ids.shape[1] - gamma
        logit_offset = prefix_without_draft - new_start - 1

        # Extract target logits for all gamma positions in one slice: (gamma, vocab_size)
        target_logits_batch = all_logits[0, logit_offset:logit_offset + gamma, :]

        # Build draft probs batch tensor if any probs are available (stochastic mode)
        draft_tokens_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device)
        draft_probs_batch: Optional[torch.Tensor] = None
        if temperature > 0.0 and any(p is not None for p in draft_probs):
            draft_probs_batch = torch.stack(
                [p if p is not None else torch.zeros(target_logits_batch.shape[-1], device=device)
                 for p in draft_probs],
                dim=0,
            )  # (gamma, vocab_size)

        # Vectorized rejection sampling across all gamma tokens
        num_accepted, accepted_tokens = batch_rejection_sample(
            target_logits_batch,
            draft_probs_batch,
            draft_tokens_tensor,
            gamma,
            temperature,
            generator,
        )

        # Reconstruct per_token_accepted flags
        per_token_accepted: List[bool] = [True] * num_accepted
        if num_accepted < gamma:
            per_token_accepted.append(False)

        # Bonus token: if all gamma accepted, sample one more from position logit_offset + gamma
        bonus = False
        if num_accepted == gamma:
            bonus_idx = logit_offset + gamma
            if bonus_idx < all_logits.shape[1]:
                bonus_logits = all_logits[0, bonus_idx, :]
                if temperature == 0.0:
                    bonus_tok = int(bonus_logits.argmax().item())
                else:
                    bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
                    bonus_tok = sample_bonus_token(bonus_probs, bonus_logits, temperature, generator)
                accepted_tokens.append(bonus_tok)
                bonus = True

        # Trim target cache: keep only prefix + accepted draft positions.
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
    draft_model: Union[AutoModelForCausalLM, Callable],
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
        draft_model: Small draft model (torch.nn.Module) OR any callable with signature
            (input_ids, attention_mask, gamma, temperature, draft_cache, draft_cache_len,
             generator) -> {"tokens", "probs", "draft_cache", "elapsed_ms"}.
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

    # Set up draft function — supports both nn.Module and arbitrary callables
    if isinstance(draft_model, torch.nn.Module):
        def draft_fn(fids, fmask, eg, t, dc, dcl, gen):
            return _draft_step(draft_model, fids, fmask, eg, t, dc, dcl, gen)
    else:
        draft_fn = draft_model  # already a callable with the right signature

    # Pre-allocate sequence buffer: prompt + all generated + gamma draft tokens + bonus
    max_buf = prompt_len + max_new_tokens + gamma + 1
    buf_ids = torch.empty(1, max_buf, dtype=input_ids.dtype, device=device)
    buf_mask = torch.ones(1, max_buf, dtype=attention_mask.dtype, device=device)
    buf_ids[0, :prompt_len] = input_ids[0]
    buf_mask[0, :prompt_len] = attention_mask[0]
    cur_pos = prompt_len  # exclusive end of valid content in buf_ids / buf_mask

    generated_ids: List[int] = []
    target_cache = None
    target_cache_len = 0
    draft_cache = None
    draft_cache_len = 0

    all_rounds: List[RoundMetrics] = []
    total_draft_ms = 0.0

    reset_peak_vram()
    wall_start = time.perf_counter()
    ttft_ms = 0.0
    ttft_recorded = False

    while len(generated_ids) < max_new_tokens:
        round_start = time.perf_counter()

        remaining = max_new_tokens - len(generated_ids)
        effective_gamma = min(gamma, remaining)
        if effective_gamma <= 0:
            break

        # Slice the current valid prefix from the pre-allocated buffer
        full_ids = buf_ids[:, :cur_pos]
        full_mask = buf_mask[:, :cur_pos]

        # ---- Draft phase ----
        draft_result = draft_fn(
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

        # Write draft tokens into the buffer in-place (no torch.cat needed)
        draft_len = len(draft_tokens)
        buf_ids[0, cur_pos:cur_pos + draft_len] = torch.tensor(
            draft_tokens, dtype=input_ids.dtype, device=device
        )
        verify_ids = buf_ids[:, :cur_pos + draft_len]
        verify_mask = buf_mask[:, :cur_pos + draft_len]

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
        target_cache_len = cur_pos + num_accepted if target_cache is not None else 0
        verify_ms = verify_result["elapsed_ms"]

        # Trim draft cache to match accepted prefix
        draft_keep_len = cur_pos + num_accepted
        draft_cache = _trim_kv_cache(draft_cache, draft_keep_len)
        draft_cache_len = draft_keep_len if draft_cache is not None else 0

        # Write accepted tokens into the buffer and advance cur_pos
        n_to_add = min(len(accepted_tokens), remaining)
        for i, tok in enumerate(accepted_tokens[:n_to_add]):
            buf_ids[0, cur_pos + i] = tok
        cur_pos += n_to_add
        generated_ids.extend(accepted_tokens[:n_to_add])

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
                total_tokens_produced=len(accepted_tokens[:n_to_add]),
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

    all_decisions: List[bool] = []
    for r in all_rounds:
        all_decisions.extend(r.per_token_accepted)
    acceptance_rate = (
        sum(all_decisions) / len(all_decisions) if all_decisions else 0.0
    )

    acceptance_length = (
        sum(r.tokens_accepted for r in all_rounds) / len(all_rounds)
        if all_rounds
        else 0.0
    )

    draft_overhead = total_draft_ms / wall_ms if wall_ms > 0 else 0.0

    metrics = GenerationMetrics(
        prompt_index=-1,
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
