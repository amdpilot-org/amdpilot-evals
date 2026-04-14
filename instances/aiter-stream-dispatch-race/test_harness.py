#!/usr/bin/env python3
"""Test harness for aiter kernel stream dispatch correctness.

Tests (behavioral):
  1. Run aiter sampling kernels on a non-default stream and verify output
     correctness against a stream-isolated reference.
  2. Run concurrent work on default stream while aiter kernels execute on
     a non-default stream, checking for data corruption from races.
"""

import os
import subprocess
import sys

_PY = "/opt/venv/bin/python3"

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}: {detail}")


def check_aiter_sampling_correctness():
    """Run aiter sampling on non-default stream and verify output correctness."""
    test_code = r'''
import torch
import sys
sys.path.insert(0, "/sgl-workspace/aiter")

torch.cuda.set_device(0)
device = "cuda:0"

passed = 0
total = 0
details = []

try:
    from aiter.ops import sampling as aiter_sampling
except ImportError:
    print("SAMPLING_RESULT: 0/1 import_failed")
    sys.exit(0)

# Test top_p_sampling_from_probs on non-default stream
if hasattr(aiter_sampling, "top_p_sampling_from_probs"):
    for trial in range(5):
        total += 1
        torch.manual_seed(100 + trial)
        batch_size = 16
        vocab_size = 32000

        probs = torch.softmax(
            torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device),
            dim=-1
        )
        # Compute descending sort indices (required by the API)
        indices = torch.argsort(probs, dim=-1, descending=True)
        top_p_arr = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

        # Reference: run on default stream with full sync
        torch.cuda.synchronize()
        try:
            ref_result = aiter_sampling.top_p_sampling_from_probs(
                probs.clone(), indices.clone(), top_p_arr.clone(), 0.9
            )
            torch.cuda.synchronize()
        except Exception as e:
            details.append(f"ref_failed_trial{trial}:{e}")
            continue

        ref_indices = ref_result

        # Test: run on non-default stream with concurrent default-stream work
        test_stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize()

        # Start conflicting work on default stream
        dummy_a = torch.randn(2048, 2048, device=device)
        dummy_b = torch.randn(2048, 2048, device=device)
        torch.mm(dummy_a, dummy_b)  # keep default stream busy

        with torch.cuda.stream(test_stream):
            try:
                test_result = aiter_sampling.top_p_sampling_from_probs(
                    probs.clone(), indices.clone(), top_p_arr.clone(), 0.9
                )
            except Exception as e:
                details.append(f"test_failed_trial{trial}:{e}")
                continue

        torch.cuda.synchronize()

        test_indices = test_result

        # Verify: indices should be valid token IDs
        valid_range = (test_indices >= 0).all() and (test_indices < vocab_size).all()
        # Verify: same inputs should give same output
        matches_ref = torch.equal(test_indices, ref_indices)

        if valid_range and matches_ref:
            passed += 1
        else:
            details.append(
                f"trial{trial}:valid={valid_range.item()},match_ref={matches_ref}"
            )

# Test top_k_top_p_sampling_from_probs similarly
if hasattr(aiter_sampling, "top_k_top_p_sampling_from_probs"):
    for trial in range(5):
        total += 1
        torch.manual_seed(200 + trial)
        batch_size = 16
        vocab_size = 32000

        probs = torch.softmax(
            torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device),
            dim=-1
        )
        indices = torch.argsort(probs, dim=-1, descending=True)
        top_p_arr = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
        top_k_arr = torch.full((batch_size,), 50, dtype=torch.int32, device=device)

        # Reference on default stream
        torch.cuda.synchronize()
        try:
            ref_result = aiter_sampling.top_k_top_p_sampling_from_probs(
                probs.clone(), indices.clone(), top_k_arr.clone(), 50,
                top_p_arr.clone(), 0.9
            )
            torch.cuda.synchronize()
        except Exception as e:
            details.append(f"topk_ref_failed_trial{trial}:{e}")
            continue

        ref_indices = ref_result

        # Test on non-default stream with concurrent work
        test_stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize()

        dummy = torch.randn(2048, 2048, device=device)
        torch.mm(dummy, dummy)

        with torch.cuda.stream(test_stream):
            try:
                test_result = aiter_sampling.top_k_top_p_sampling_from_probs(
                    probs.clone(), indices.clone(), top_k_arr.clone(), 50,
                    top_p_arr.clone(), 0.9
                )
            except Exception as e:
                details.append(f"topk_test_failed_trial{trial}:{e}")
                continue

        torch.cuda.synchronize()

        test_indices = test_result

        valid_range = (test_indices >= 0).all() and (test_indices < vocab_size).all()
        matches_ref = torch.equal(test_indices, ref_indices)

        if valid_range and matches_ref:
            passed += 1
        else:
            details.append(
                f"topk_trial{trial}:valid={valid_range.item()},match_ref={matches_ref}"
            )

detail_str = "; ".join(details[:5]) if details else ""
print(f"SAMPLING_RESULT: {passed}/{total} {detail_str}")
'''

    result = subprocess.run(
        [_PY, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": "/sgl-workspace/aiter",
             "HIP_VISIBLE_DEVICES": "0"},
    )

    if result.returncode != 0:
        check("aiter_sampling_stream_correctness", False,
              f"Test crashed: {result.stderr[-500:]}")
        return

    for line in result.stdout.split("\n"):
        if "SAMPLING_RESULT:" in line:
            parts = line.split(":")[1].strip().split(" ", 1)
            ratio = parts[0].split("/")
            passed_n = int(ratio[0])
            total_n = int(ratio[1])
            detail = parts[1] if len(parts) > 1 else ""
            check("aiter_sampling_stream_correctness",
                  total_n > 0 and passed_n == total_n,
                  f"{passed_n}/{total_n} {detail}")
            return

    check("aiter_sampling_stream_correctness", False, "No result line found")


def check_concurrent_stream_corruption():
    """Stress test: interleave aiter ops across multiple streams."""
    test_code = r'''
import torch
import sys
sys.path.insert(0, "/sgl-workspace/aiter")

torch.cuda.set_device(0)
device = "cuda:0"

passed = 0
total = 0

try:
    from aiter.ops import sampling as aiter_sampling
except ImportError:
    print("CONCURRENT_RESULT: 0/1 import_failed")
    sys.exit(0)

if not hasattr(aiter_sampling, "top_p_sampling_from_probs"):
    print("CONCURRENT_RESULT: 0/1 no_top_p")
    sys.exit(0)

# Create multiple streams
streams = [torch.cuda.Stream(device=device) for _ in range(4)]

for trial in range(3):
    total += 1
    torch.manual_seed(300 + trial)

    batch_size = 8
    vocab_size = 32000

    probs = torch.softmax(
        torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device),
        dim=-1
    )
    indices = torch.argsort(probs, dim=-1, descending=True)
    top_p_arr = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)

    # Run same operation on all 4 streams concurrently
    results = [None] * 4
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            # Do some dummy work to offset timing
            d = torch.randn(512 * (i + 1), 512, device=device)
            torch.mm(d, d.t())
            try:
                r = aiter_sampling.top_p_sampling_from_probs(
                    probs.clone(), indices.clone(), top_p_arr.clone(), 0.9
                )
                results[i] = r
            except Exception:
                pass

    torch.cuda.synchronize()

    # All streams should produce the same result (same inputs)
    valid_results = [r for r in results if r is not None]
    if len(valid_results) >= 2:
        all_match = all(torch.equal(valid_results[0], r) for r in valid_results[1:])
        all_valid = all(
            (r >= 0).all() and (r < vocab_size).all() for r in valid_results
        )
        if all_match and all_valid:
            passed += 1

print(f"CONCURRENT_RESULT: {passed}/{total}")
'''

    result = subprocess.run(
        [_PY, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": "/sgl-workspace/aiter",
             "HIP_VISIBLE_DEVICES": "0"},
    )

    if result.returncode != 0:
        check("concurrent_stream_corruption", False,
              f"Test crashed: {result.stderr[-500:]}")
        return

    for line in result.stdout.split("\n"):
        if "CONCURRENT_RESULT:" in line:
            parts = line.split(":")[1].strip().split(" ", 1)
            ratio = parts[0].split("/")
            passed_n = int(ratio[0])
            total_n = int(ratio[1])
            check("concurrent_stream_corruption",
                  total_n > 0 and passed_n == total_n,
                  f"{passed_n}/{total_n}")
            return

    check("concurrent_stream_corruption", False, "No result line found")


def main():
    print("=" * 60)
    print("Aiter Kernel Stream Dispatch Test")
    print("=" * 60)

    print("\n--- Behavioral: Sampling Correctness on Non-Default Stream ---")
    check_aiter_sampling_correctness()

    print("\n--- Behavioral: Concurrent Multi-Stream Corruption ---")
    check_concurrent_stream_corruption()

    print(f"\n--- Results ---")
    print(f"  {checks_passed}/{checks_total} checks passed")

    score = checks_passed / checks_total * 100.0 if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
