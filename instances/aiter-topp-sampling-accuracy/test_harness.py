#!/usr/bin/env python3
"""
Test harness for aiter top-p sampling accuracy eval.

Validates correctness of top-p sampling implementation through
behavioral checks.
"""

import sys

RESULTS = {"passed": [], "failed": []}


def record(name: str, passed: bool, detail: str = ""):
    bucket = "passed" if passed else "failed"
    RESULTS[bucket].append((name, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


# ===================================================================
# 1. Behavioral test: top-p sampling distribution correctness
# ===================================================================
def test_behavioral_sampling():
    """
    Run top-p sampling with a known probability vector on a single GPU and
    verify that the sampled token distribution approximately matches the
    expected distribution (tokens within the nucleus).
    """
    print("\n=== Behavioral: top-p sampling distribution ===")
    try:
        import torch

        if not torch.cuda.is_available():
            record("behavioral_sampling", False, "No GPU available")
            return

        # Try to import aiter sampling utilities
        try:
            from aiter import top_p_sampling_from_probs
        except ImportError:
            try:
                # Alternative import path
                from aiter.ops.triton import top_p_sampling_from_probs
            except ImportError:
                record(
                    "behavioral_sampling",
                    False,
                    "Cannot import top_p_sampling_from_probs from aiter",
                )
                return

        device = torch.device("cuda:0")
        batch_size = 1024
        vocab_size = 128
        top_p = 0.9
        num_trials = 10

        # Create a known distribution: a few high-prob tokens, rest near zero.
        # Tokens 0..4 get most of the mass.
        probs = torch.zeros(batch_size, vocab_size, device=device)
        high_prob_tokens = 5
        for i in range(high_prob_tokens):
            probs[:, i] = (high_prob_tokens - i) / (
                high_prob_tokens * (high_prob_tokens + 1) / 2
            )
        # sprinkle a tiny amount on remaining tokens
        probs[:, high_prob_tokens:] = 0.001
        # renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        all_samples = []
        for trial in range(num_trials):
            try:
                result = top_p_sampling_from_probs(probs, top_p)
                if isinstance(result, tuple):
                    samples = result[0]
                else:
                    samples = result
                all_samples.append(samples.cpu())
            except Exception as e:
                record("behavioral_sampling", False, f"Sampling call failed: {e}")
                return

        all_samples = torch.cat(all_samples, dim=0)  # (num_trials*batch, )

        # Check 1: vast majority of samples should be in [0, high_prob_tokens)
        in_nucleus = (all_samples < high_prob_tokens).float().mean().item()
        record(
            "behavioral_high_prob_concentration",
            in_nucleus > 0.85,
            f"fraction in nucleus = {in_nucleus:.3f} (expect > 0.85)",
        )

        # Check 2: we should see more than one unique token (not all identical)
        unique_tokens = all_samples.unique().numel()
        record(
            "behavioral_diversity",
            unique_tokens >= 3,
            f"unique tokens sampled = {unique_tokens} (expect >= 3)",
        )

        # Check 3: across trials, results should not be identical (RNG advancing)
        first_trial = all_samples[: batch_size]
        second_trial = all_samples[batch_size : 2 * batch_size]
        identical_frac = (first_trial == second_trial).float().mean().item()
        record(
            "behavioral_rng_variation",
            identical_frac < 0.95,
            f"identical fraction between trials = {identical_frac:.3f} (expect < 0.95)",
        )

    except Exception as e:
        record("behavioral_sampling", False, f"Unexpected error: {e}")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  AIter Top-P Sampling Accuracy -- Test Harness")
    print("=" * 60)

    test_behavioral_sampling()

    print("\n" + "=" * 60)
    print(f"  SUMMARY: {len(RESULTS['passed'])} passed, {len(RESULTS['failed'])} failed")
    print("=" * 60)

    if RESULTS["failed"]:
        print("\nFailed tests:")
        for name, detail in RESULTS["failed"]:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
