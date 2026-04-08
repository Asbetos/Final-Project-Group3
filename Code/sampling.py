"""
Probability-level sampling utilities for speculative decoding.

Pure functions operating on logit/probability tensors — no model or
tokenizer dependencies.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator = None,
) -> Tuple[int, torch.Tensor]:
    """
    Convert logits to a probability distribution and sample one token.

    Args:
        logits: Raw logits of shape (vocab_size,).
        temperature: Sampling temperature. 0.0 = greedy (argmax).
        generator: Optional torch generator for reproducibility.

    Returns:
        (token_id, probs) where probs is the full distribution over the vocab.
    """
    if temperature == 0.0:
        # Greedy: deterministic argmax
        probs = torch.zeros_like(logits)
        token_id = logits.argmax(dim=-1).item()
        probs[token_id] = 1.0
        return token_id, probs

    # Temperature-scaled sampling
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return token_id, probs


def rejection_sample_token(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token: int,
    temperature: float,
    generator: torch.Generator = None,
) -> Tuple[bool, int]:
    """
    Perform one step of speculative sampling verification.

    Args:
        target_probs: p(x) over the full vocab from the target model.
        draft_probs: q(x) over the full vocab from the draft model.
        draft_token: The token proposed by the draft model.
        temperature: Sampling temperature (affects acceptance logic at T=0).
        generator: Optional generator for reproducibility.

    Returns:
        (accepted, token):
            If accepted is True, token == draft_token.
            If accepted is False, token is the resampled correction token.
    """
    if temperature == 0.0:
        # Greedy: accept iff the target's argmax matches the draft token
        target_argmax = target_probs.argmax(dim=-1).item()
        if target_argmax == draft_token:
            return True, draft_token
        return False, target_argmax

    # Stochastic acceptance: accept with probability min(1, p(x)/q(x))
    p = target_probs[draft_token].item()
    q = draft_probs[draft_token].item()

    if q == 0.0:
        # Draft assigned zero probability — accept unconditionally if target
        # also assigns zero; otherwise reject and resample
        if p == 0.0:
            return True, draft_token
        return False, sample_residual_distribution(
            target_probs, draft_probs, generator
        )

    acceptance_prob = min(1.0, p / q)
    u = torch.rand(1, generator=generator, device=target_probs.device).item()

    if u < acceptance_prob:
        return True, draft_token

    # Rejected — resample from the corrected residual distribution
    correction_token = sample_residual_distribution(
        target_probs, draft_probs, generator
    )
    return False, correction_token


def sample_residual_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    generator: torch.Generator = None,
) -> int:
    """
    Sample from the normalized residual distribution: max(0, p - q) / Z.

    This is the correction distribution used when a draft token is rejected.
    Guarantees that the marginal output distribution matches the target.
    """
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum()

    if total <= 0.0:
        # Degenerate case (q >= p everywhere): fall back to target distribution
        return torch.multinomial(
            target_probs, num_samples=1, generator=generator
        ).item()

    residual = residual / total
    return torch.multinomial(residual, num_samples=1, generator=generator).item()


def sample_bonus_token(
    target_probs: torch.Tensor,
    temperature: float,
    generator: torch.Generator = None,
) -> int:
    """
    Sample the bonus token from the target distribution.

    Called when all gamma draft tokens are accepted. This extra token ensures
    at least gamma+1 tokens are produced per round in the best case.
    """
    if temperature == 0.0:
        return target_probs.argmax(dim=-1).item()

    return torch.multinomial(
        target_probs, num_samples=1, generator=generator
    ).item()
