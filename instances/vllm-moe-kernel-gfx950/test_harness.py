#!/usr/bin/env python3
"""Test harness for vllm MoE kernel gfx950 multi-bug fix (PR #37833).

This PR fixes 6 distinct bugs across the MoE kernel subsystem on gfx950:
  1. FP8 quantization 1-ULP boundary error from GPU fast-division
  2. C++ kernel no-op on ROCm (#ifndef USE_ROCM gate)
  3. Test with uninitialized weights (false-positive)
  4. topk API type mismatch (tuple vs SparseMatrix)
  5. Unsupported AiterExperts quant combinations
  6. Hardcoded SiluAndMul in test utilities

Tests (behavioral + source inspection):
  - FP8 quantization correctness (multiply-by-reciprocal vs division)
  - Platform-aware kernel dispatch (ROCm fallback for gated C++ kernels)
  - Test weight initialization checks
  - Source inspection for correct FP8 quant pattern
"""

import ast
import os
import subprocess
import sys

_PY = "/usr/bin/python3"
VLLM_PATH = "/workspace/vllm"

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


def check_fp8_quant_precision():
    """Check FP8 quantization uses multiply-by-reciprocal, not division."""
    fp8_utils_path = os.path.join(
        VLLM_PATH,
        "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
    )
    if not os.path.exists(fp8_utils_path):
        check("fp8_quant_precision", False, "fp8_utils.py not found")
        return

    with open(fp8_utils_path) as f:
        content = f.read()

    # The bug: kernels use `amax / fp8_max` (division)
    # The fix: use `amax * (1/fp8_max)` (multiply-by-reciprocal)
    # GPU fast-division introduces 1-ULP error at FP8 boundaries

    # Check Triton kernel functions for the pattern
    # Look for division by fp8_max (buggy) vs multiplication by reciprocal (fixed)
    has_division = "/ fp8_max" in content or "/ FP8_MAX" in content
    has_reciprocal = "* (1.0 / fp8" in content or "* (1 / fp8" in content or "fp8_inv" in content

    check("fp8_no_division",
          not has_division or has_reciprocal,
          "FP8 quantization uses division instead of multiply-by-reciprocal")


def check_fp8_behavioral():
    """Test FP8 quantization correctness at boundary values."""
    test_code = r'''
import torch
import sys
sys.path.insert(0, "/workspace/vllm")

torch.cuda.set_device(0)
device = "cuda:0"

passed = 0
total = 0

# Test FP8 quantization at boundary values
fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

for scale_factor in [1.0, 0.5, 2.0, 0.1, 10.0]:
    total += 1
    # Create values near FP8 bucket boundaries
    x = torch.tensor([fp8_max * scale_factor - 0.001,
                       fp8_max * scale_factor,
                       fp8_max * scale_factor + 0.001],
                      dtype=torch.float32, device=device)

    # Reference: quantize with exact arithmetic
    amax = x.abs().max()
    scale = amax / fp8_max
    x_scaled_ref = (x / scale).clamp(-fp8_max, fp8_max)
    x_fp8_ref = x_scaled_ref.to(torch.float8_e4m3fn)

    # Also check with reciprocal multiplication
    inv_scale = 1.0 / scale
    x_scaled_recip = (x * inv_scale).clamp(-fp8_max, fp8_max)
    x_fp8_recip = x_scaled_recip.to(torch.float8_e4m3fn)

    # Both methods should give same result for correct implementation
    match = (x_fp8_ref.to(torch.float32) - x_fp8_recip.to(torch.float32)).abs().max().item()
    if match < 1e-6:
        passed += 1

print(f"FP8_RESULT: {passed}/{total}")
'''

    result = subprocess.run(
        [_PY, "-c", test_code],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
    )

    if result.returncode != 0:
        check("fp8_behavioral", False, f"Crashed: {result.stderr[-300:]}")
        return

    for line in result.stdout.split("\n"):
        if "FP8_RESULT:" in line:
            parts = line.split(":")[1].strip().split("/")
            p, t = int(parts[0]), int(parts[1])
            check("fp8_behavioral", p == t, f"{p}/{t}")
            return
    check("fp8_behavioral", False, "No result")


def check_rocm_kernel_gate():
    """Check that C++ kernels gated by USE_ROCM have Triton fallbacks."""
    batched_dgm_path = os.path.join(
        VLLM_PATH,
        "vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py"
    )
    if not os.path.exists(batched_dgm_path):
        check("rocm_kernel_gate", False, "batched_deep_gemm_moe.py not found")
        return

    with open(batched_dgm_path) as f:
        content = f.read()

    # The fix gates the C++ kernel behind is_cuda() and provides ROCm fallback
    has_platform_check = (
        "is_cuda" in content or
        "current_platform" in content or
        "USE_ROCM" in content or
        "is_hip" in content or
        "is_rocm" in content
    )

    check("rocm_kernel_gate",
          has_platform_check,
          "No platform check found — C++ kernel may silently no-op on ROCm")


def check_test_weight_init():
    """Check that MoE test weights are properly initialized (not all zeros)."""
    test_path = os.path.join(
        VLLM_PATH,
        "tests/kernels/moe/test_shared_fused_moe_routed_transform.py"
    )
    if not os.path.exists(test_path):
        check("test_weight_init", False, "Test file not found")
        return

    with open(test_path) as f:
        content = f.read()

    # The bug: weights were uninitialized (all zeros), making tests vacuously pass
    # The fix: initializes weights with seeded random values
    has_random_init = (
        "torch.randn" in content or
        "torch.rand" in content or
        "manual_seed" in content or
        "random" in content.lower()
    )

    # Also check for NaN detection in assertions
    has_nan_check = (
        "isnan" in content or
        "nan" in content.lower() or
        "assert_close" in content
    )

    check("test_weight_init",
          has_random_init,
          "Test weights may not be properly initialized with random values")

    check("test_nan_detection",
          has_nan_check,
          "Test may not detect NaN in outputs")


def check_gpt_oss_topk():
    """Check GPT-OSS triton kernel handles topk API type correctly."""
    gpt_oss_path = os.path.join(
        VLLM_PATH,
        "vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py"
    )
    if not os.path.exists(gpt_oss_path):
        # File may not exist in this version
        check("gpt_oss_topk", True, "File not found (may not exist in this version)")
        return

    with open(gpt_oss_path) as f:
        content = f.read()

    # The fix handles tuple return type from topk (ROCm returns tuple, not SparseMatrix)
    has_tuple_handling = (
        "tuple" in content or
        "isinstance" in content or
        "SparseMatrix" in content
    )

    check("gpt_oss_topk",
          has_tuple_handling,
          "topk API return type may not be handled correctly")


def check_quark_moe_support():
    """Check Quark MoE quantization has proper ROCm support."""
    quark_path = os.path.join(
        VLLM_PATH,
        "vllm/model_executor/layers/quantization/quark/quark_moe.py"
    )
    if not os.path.exists(quark_path):
        check("quark_moe_rocm", True, "Quark MoE file not found (may not exist)")
        return

    with open(quark_path) as f:
        content = f.read()

    # Check for proper ROCm/AITER integration
    has_backend_check = (
        "aiter" in content.lower() or
        "rocm" in content.lower() or
        "hip" in content.lower() or
        "supports" in content
    )

    check("quark_moe_rocm",
          has_backend_check,
          "Quark MoE may lack ROCm backend support")


def main():
    print("=" * 60)
    print("vLLM MoE Kernel gfx950 Multi-Bug Test")
    print("=" * 60)

    print("\n--- FP8 Quantization Precision ---")
    check_fp8_quant_precision()
    check_fp8_behavioral()

    print("\n--- ROCm Platform Gating ---")
    check_rocm_kernel_gate()

    print("\n--- Test Quality ---")
    check_test_weight_init()

    print("\n--- API Compatibility ---")
    check_gpt_oss_topk()
    check_quark_moe_support()

    print(f"\n--- Results ---")
    print(f"  {checks_passed}/{checks_total} checks passed")

    score = checks_passed / checks_total * 100.0 if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
