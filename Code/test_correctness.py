"""
Correctness validation for the speculative decoding implementation.

Three levels of testing:
  1. Unit tests for sampling.py (pure math, no GPU required).
  2. Greedy equivalence test: speculative output at T=0 must exactly match
     baseline autoregressive output — the gold-standard correctness check.
  3. Smoke test: run every pair briefly, check no crashes / OOM / NaN.
"""

import argparse
import logging
import sys

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===================================================================
# Level 1: Unit tests for sampling.py
# ===================================================================


def test_sample_from_logits_greedy():
    """At temperature=0, sample_from_logits should return argmax."""
    from sampling import sample_from_logits

    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    token_id, probs = sample_from_logits(logits, temperature=0.0)
    assert token_id == 1, f"Expected argmax=1, got {token_id}"
    assert probs[1] == 1.0, f"Expected one-hot at 1, got {probs}"
    assert probs.sum().item() == 1.0
    logger.info("  PASS: test_sample_from_logits_greedy")


def test_sample_from_logits_stochastic():
    """At temperature>0, sample should return a valid token id and valid probs."""
    from sampling import sample_from_logits

    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    gen = torch.Generator().manual_seed(42)
    token_id, probs = sample_from_logits(logits, temperature=1.0, generator=gen)
    assert 0 <= token_id < len(logits), f"Token out of range: {token_id}"
    assert abs(probs.sum().item() - 1.0) < 1e-5, f"Probs don't sum to 1: {probs.sum()}"
    assert (probs >= 0).all(), "Negative probabilities"
    logger.info("  PASS: test_sample_from_logits_stochastic")


def test_rejection_sample_greedy_accept():
    """At T=0, rejection should accept when argmax(target) == draft_token."""
    from sampling import rejection_sample_token

    target_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
    draft_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])
    accepted, token = rejection_sample_token(target_probs, draft_probs, 1, 0.0)
    assert accepted is True, "Should accept when argmax matches"
    assert token == 1
    logger.info("  PASS: test_rejection_sample_greedy_accept")


def test_rejection_sample_greedy_reject():
    """At T=0, rejection should reject when argmax(target) != draft_token."""
    from sampling import rejection_sample_token

    target_probs = torch.tensor([0.0, 0.0, 1.0, 0.0])  # argmax=2
    draft_probs = torch.tensor([0.0, 1.0, 0.0, 0.0])   # draft picked 1
    accepted, token = rejection_sample_token(target_probs, draft_probs, 1, 0.0)
    assert accepted is False, "Should reject when argmax differs"
    assert token == 2, f"Should return target argmax=2, got {token}"
    logger.info("  PASS: test_rejection_sample_greedy_reject")


def test_sample_residual_distribution():
    """Residual distribution should be normalized and non-negative."""
    from sampling import sample_residual_distribution

    target = torch.tensor([0.3, 0.5, 0.1, 0.1])
    draft = torch.tensor([0.1, 0.7, 0.1, 0.1])
    # Residual: max(0, [0.2, -0.2, 0.0, 0.0]) = [0.2, 0, 0, 0] -> normalized = [1,0,0,0]
    gen = torch.Generator().manual_seed(0)
    token = sample_residual_distribution(target, draft, gen)
    assert token == 0, f"Expected token 0 (only nonzero residual), got {token}"
    logger.info("  PASS: test_sample_residual_distribution")


def test_sample_residual_fallback():
    """When q >= p everywhere, should fall back to target distribution."""
    from sampling import sample_residual_distribution

    target = torch.tensor([0.25, 0.25, 0.25, 0.25])
    draft = torch.tensor([0.5, 0.3, 0.1, 0.1])
    # Residual: max(0, [-0.25, -0.05, 0.15, 0.15]) = [0, 0, 0.15, 0.15]
    gen = torch.Generator().manual_seed(0)
    token = sample_residual_distribution(target, draft, gen)
    assert token in [2, 3], f"Expected token 2 or 3, got {token}"
    logger.info("  PASS: test_sample_residual_fallback")


def test_bonus_token_greedy():
    """Bonus token at T=0 should be argmax."""
    from sampling import sample_bonus_token

    probs = torch.tensor([0.1, 0.1, 0.7, 0.1])
    token = sample_bonus_token(probs, temperature=0.0)
    assert token == 2, f"Expected argmax=2, got {token}"
    logger.info("  PASS: test_bonus_token_greedy")


def run_unit_tests():
    """Run all sampling unit tests."""
    logger.info("=" * 60)
    logger.info("Level 1: Sampling Unit Tests")
    logger.info("=" * 60)
    test_sample_from_logits_greedy()
    test_sample_from_logits_stochastic()
    test_rejection_sample_greedy_accept()
    test_rejection_sample_greedy_reject()
    test_sample_residual_distribution()
    test_sample_residual_fallback()
    test_bonus_token_greedy()
    logger.info("All unit tests passed.\n")


# ===================================================================
# Level 2: Greedy Equivalence Test (requires GPU + models)
# ===================================================================


def test_greedy_equivalence(
    pair_id: str = "A",
    max_new_tokens: int = 32,
    prompt_text: str = "Write a Python function that computes the factorial of a number.",
):
    """
    At temperature=0, speculative decoding MUST produce the exact same
    output as standard autoregressive decoding from the target model.

    This is the theoretical guarantee of speculative sampling.
    """
    from config import PAIR_MAP
    from models import load_model_pair, unload_models
    from baseline import autoregressive_decode
    from speculative import speculative_decode

    logger.info("=" * 60)
    logger.info("Level 2: Greedy Equivalence Test (pair %s)", pair_id)
    logger.info("=" * 60)

    pair = PAIR_MAP[pair_id]
    target_model, draft_model, tokenizer = load_model_pair(pair)

    try:
        # Prepare input
        messages = [
            {"role": "system", "content": "Complete the following request."},
            {"role": "user", "content": prompt_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        # Baseline autoregressive (greedy)
        logger.info("Running baseline autoregressive (greedy) ...")
        baseline_result = autoregressive_decode(
            target_model,
            input_ids,
            attention_mask,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
        )
        baseline_tokens = baseline_result["output_ids"]

        # Speculative (greedy, gamma=5)
        for gamma in [1, 3, 5]:
            logger.info("Running speculative (greedy, γ=%d) ...", gamma)
            spec_result = speculative_decode(
                target_model,
                draft_model,
                input_ids,
                attention_mask,
                gamma=gamma,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
            )
            spec_tokens = spec_result["output_ids"]

            # Compare
            match = baseline_tokens == spec_tokens
            if match:
                logger.info("  PASS: γ=%d — exact match (%d tokens)", gamma, len(spec_tokens))
            else:
                # Find first divergence
                min_len = min(len(baseline_tokens), len(spec_tokens))
                diverge_idx = next(
                    (i for i in range(min_len) if baseline_tokens[i] != spec_tokens[i]),
                    min_len,
                )
                logger.error(
                    "  FAIL: γ=%d — divergence at token %d. "
                    "Baseline len=%d, Spec len=%d",
                    gamma,
                    diverge_idx,
                    len(baseline_tokens),
                    len(spec_tokens),
                )
                logger.error("  Baseline: %s", baseline_tokens[:diverge_idx + 3])
                logger.error("  Speculative: %s", spec_tokens[:diverge_idx + 3])
                raise AssertionError(
                    f"Greedy equivalence failed at γ={gamma}, position {diverge_idx}"
                )

        logger.info("Greedy equivalence test passed for all gamma values.\n")

    finally:
        unload_models(target_model, draft_model)


# ===================================================================
# Level 3: Smoke Test (requires GPU + models)
# ===================================================================


def test_smoke(pair_id: str = "A", max_new_tokens: int = 16):
    """Quick smoke test: 1 prompt, few tokens, check no crash/NaN."""
    from config import PAIR_MAP
    from models import load_model_pair, unload_models
    from speculative import speculative_decode

    logger.info("=" * 60)
    logger.info("Level 3: Smoke Test (pair %s)", pair_id)
    logger.info("=" * 60)

    pair = PAIR_MAP[pair_id]
    target_model, draft_model, tokenizer = load_model_pair(pair)

    try:
        messages = [
            {"role": "user", "content": "Hello, what is 2+2?"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        for gamma in [1, 5]:
            for temp in [0.0, 0.6]:
                gen = torch.Generator(device="cuda").manual_seed(42)
                result = speculative_decode(
                    target_model,
                    draft_model,
                    input_ids,
                    attention_mask,
                    gamma=gamma,
                    temperature=temp,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    generator=gen,
                )
                m = result["metrics"]
                assert m.total_tokens_generated > 0, "No tokens generated"
                assert m.tokens_per_second > 0, "TPS is zero"
                assert 0 <= m.acceptance_rate <= 1, f"Invalid α: {m.acceptance_rate}"
                logger.info(
                    "  PASS: γ=%d t=%.1f — %d tokens, %.1f TPS, α=%.3f",
                    gamma,
                    temp,
                    m.total_tokens_generated,
                    m.tokens_per_second,
                    m.acceptance_rate,
                )

        logger.info("Smoke test passed.\n")

    finally:
        unload_models(target_model, draft_model)


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Correctness tests for speculative decoding")
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Test level: 1=unit, 2=greedy equivalence (GPU), 3=smoke (GPU)",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="A",
        choices=["A", "B", "C"],
        help="Model pair for GPU tests (default: A)",
    )
    args = parser.parse_args()

    if args.level >= 1:
        run_unit_tests()
    if args.level >= 2:
        test_greedy_equivalence(pair_id=args.pair)
    if args.level >= 3:
        test_smoke(pair_id=args.pair)

    logger.info("All tests at level %d passed!", args.level)


if __name__ == "__main__":
    main()
