"""
Probability-level sampling utilities for speculative decoding.

Pure functions operating on logit/probability tensors — no model or
tokenizer dependencies.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def align_shared_vocab(
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
):
    """
    Restrict target/draft distributions to the shared vocab prefix.
    """
    if draft_probs is None:
        return target_probs, None, target_probs.shape[-1]

    common_vocab = min(target_probs.shape[-1], draft_probs.shape[-1])

    target_probs = target_probs[..., :common_vocab]
    draft_probs = draft_probs[..., :common_vocab]

    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return target_probs, draft_probs, common_vocab


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator = None,
) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Convert logits to a probability distribution and sample one token.

    Args:
        logits: Raw logits of shape (vocab_size,).
        temperature: Sampling temperature. 0.0 = greedy (argmax).
        generator: Optional torch generator for reproducibility.

    Returns:
        (token_id, probs) where probs is the full distribution. In greedy mode we
        return a one-hot distribution so callers and tests can treat both paths
        uniformly.
    """
    if temperature == 0.0:
        token_id = int(logits.argmax(dim=-1).item())
        probs = torch.zeros_like(logits)
        probs[token_id] = 1.0
        return token_id, probs

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return token_id, probs


def rejection_sample_token(
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    draft_token: int,
    temperature: float,
    generator: torch.Generator = None,
) -> Tuple[bool, int]:
    """
    Perform one step of speculative sampling verification (single-token path).

    Args:
        target_probs: p(x) over the full vocab from the target model, or None if greedy.
        draft_probs: q(x) over the full vocab from the draft model, or None if greedy.
        draft_token: The token proposed by the draft model.
        temperature: Sampling temperature (affects acceptance logic at T=0).
        generator: Optional generator for reproducibility.

    Returns:
        (accepted, token):
            If accepted is True, token == draft_token.
            If accepted is False, token is the resampled correction token.
    """
    if temperature == 0.0:
        # Greedy: accept iff the target's argmax matches the draft token.
        target_argmax = int(target_probs.argmax(dim=-1).item())
        if target_argmax == draft_token:
            return True, draft_token
        return False, target_argmax

    # Stochastic: accept with probability min(1, p(x)/q(x)).
    if draft_probs is None:
        # Draft has no probability distribution (e.g. n-gram or future zero-cost draft).
        # Treat as q(x)=1 for the proposed token: accept with probability p(x).
        p = float(target_probs[draft_token].item())
        u = float(torch.rand(1, generator=generator).item())
        if u < p:
            return True, draft_token
        return False, int(sample_residual_distribution(target_probs, None, generator))

    p = float(target_probs[draft_token].item())
    q = float(draft_probs[draft_token].item())

    if q == 0.0:
        if p == 0.0:
            return True, draft_token
        return False, int(sample_residual_distribution(target_probs, draft_probs, generator))

    acceptance_prob = min(1.0, p / q)
    u = float(torch.rand(1, generator=generator).item())

    if u < acceptance_prob:
        return True, draft_token

    correction_token = sample_residual_distribution(target_probs, draft_probs, generator)
    return False, int(correction_token)


def batch_rejection_sample(
    target_logits_batch: torch.Tensor,
    draft_probs_batch: Optional[torch.Tensor],
    draft_tokens: torch.Tensor,
    gamma: int,
    temperature: float,
    generator: torch.Generator = None,
) -> Tuple[int, List[int]]:
    """
    Vectorized rejection sampling across all gamma draft tokens in one pass.

    This replaces the Python for-loop in _verify_step, eliminating per-token
    .item() calls and generating all random numbers in a single GPU call.

    Args:
        target_logits_batch: (gamma, vocab_size) raw target logits for each position.
        draft_probs_batch: (gamma, vocab_size) draft probabilities, or None for greedy.
        draft_tokens: (gamma,) int64 tensor of drafted token ids.
        gamma: Number of draft tokens.
        temperature: Sampling temperature.
        generator: Optional torch generator.

    Returns:
        (num_accepted, accepted_tokens_list) where accepted_tokens_list includes
        the first rejected/correction token (or bonus indicator if all accepted).
        Callers must separately sample the bonus token if num_accepted == gamma.
    """
    device = target_logits_batch.device

    if temperature == 0.0:
        # Greedy: compare argmax of target logits against each draft token.
        target_argmaxes = target_logits_batch.argmax(dim=-1)  # (gamma,)
        accepted_mask = target_argmaxes == draft_tokens         # (gamma,) bool

        if accepted_mask.all():
            num_accepted = gamma
            return num_accepted, draft_tokens.tolist()

        # Find first rejection via argmin on the integer representation
        first_reject = int(accepted_mask.logical_not().to(torch.uint8).argmax().item())
        num_accepted = first_reject
        correction = int(target_argmaxes[first_reject].item())

        result = draft_tokens[:num_accepted].tolist()
        result.append(correction)
        return num_accepted, result

    # Stochastic path — batch softmax over all gamma positions at once
    target_probs_batch = F.softmax(target_logits_batch / temperature, dim=-1)  # (gamma, vocab)

    common_vocab = target_probs_batch.shape[-1]
    if draft_probs_batch is not None:
        target_probs_batch, draft_probs_batch, common_vocab = align_shared_vocab(
            target_probs_batch, draft_probs_batch
        )


    idx = torch.arange(gamma, device=device)

    if draft_probs_batch is None:
        # No draft probability distribution — accept with probability p(proposed token)
        p_vals = target_probs_batch[idx, draft_tokens]   # (gamma,)
        accept_probs = p_vals.clamp(max=1.0)
    else:
        p_vals = target_probs_batch[idx, draft_tokens]   # (gamma,)
        q_vals = draft_probs_batch[idx, draft_tokens]    # (gamma,)
        # Avoid division by zero; where q==0 and p==0, accept; where q==0 and p>0, reject
        safe_q = q_vals.clamp(min=1e-9)
        accept_probs = (p_vals / safe_q).clamp(max=1.0)
        # Force-reject where q≈0 regardless of p: if draft never proposes a
        # token (q→0), the residual distribution handles resampling correctly.
        accept_probs = torch.where(q_vals < 1e-9, torch.zeros_like(accept_probs), accept_probs)

    # Generate all random numbers on the same device as the target logits
    # to avoid device-mismatch issues during stochastic verification.
    u = torch.rand(gamma,device=device, generator=generator)  # (gamma,) on CPU
    accepted_mask = u < accept_probs      # (gamma,) bool

    if accepted_mask.all():
        num_accepted = gamma
        return num_accepted, draft_tokens.tolist()

    first_reject = int(accepted_mask.logical_not().to(torch.uint8).argmax().item())
    num_accepted = first_reject

    # Sample correction token from residual distribution at the first rejection position
    t_probs = target_probs_batch[first_reject]
    if draft_probs_batch is not None:
        d_probs = draft_probs_batch[first_reject]
        correction = int(sample_residual_distribution(t_probs, d_probs, generator))
    else:
        correction = int(sample_residual_distribution(t_probs, None, generator))

    result = draft_tokens[:num_accepted].tolist()
    result.append(correction)
    return num_accepted, result


def sample_residual_distribution(
    target_probs: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    generator: torch.Generator = None,
) -> int:
    """
    Sample from the normalized residual distribution: max(0, p - q) / Z.

    This is the correction distribution used when a draft token is rejected.
    Guarantees that the marginal output distribution matches the target.
    If draft_probs is None (no distribution available), samples directly from target.
    """
    if draft_probs is None:
        return int(torch.multinomial(target_probs, num_samples=1, generator=generator).item())

    # -------------------------------------------------------------------
    # Workaround:
    # Align target and draft probabilities to the shared vocab space before
    # computing the residual distribution. Otherwise, vocab-size mismatch
    # can break the subtraction target_probs - draft_probs.
    # -------------------------------------------------------------------
    target_probs, draft_probs, _ = align_shared_vocab(target_probs, draft_probs)

    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum()

    if total <= 0.0:
        # Degenerate case (q >= p everywhere): fall back to target distribution
        return int(torch.multinomial(target_probs, num_samples=1, generator=generator).item())

    residual = residual / total
    return int(torch.multinomial(residual, num_samples=1, generator=generator).item())


def sample_bonus_token(
    target_probs: Optional[torch.Tensor],
    target_logits: Optional[torch.Tensor],
    temperature: float,
    generator: torch.Generator = None,
) -> int:
    """
    Sample the bonus token from the target distribution.

    Called when all gamma draft tokens are accepted. This extra token ensures
    at least gamma+1 tokens are produced per round in the best case.

    Args:
        target_probs: Pre-computed softmax probs, or None if greedy.
        target_logits: Raw logits (used only if temperature==0.0 and probs is None).
        temperature: Sampling temperature.
        generator: Optional generator.
    """
    if temperature == 0.0:
        if target_logits is not None:
            return int(target_logits.argmax(dim=-1).item())
        return int(target_probs.argmax(dim=-1).item())

    return int(torch.multinomial(target_probs, num_samples=1, generator=generator).item())
