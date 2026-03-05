#!/usr/bin/env python3
"""Test harness for vllm-ck-mxfp4-moe eval instance.

Bug: No CK backend for MXFP4 MoE quantization on ROCm. The mxfp4.py has no
fused MoE support. Requires _aiter_ops.py MoE functions and mxfp4.py CK backend.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import os
import sys
from pathlib import Path

VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/workspace/vllm"))
AITER_OPS = VLLM_ROOT / "vllm/_aiter_ops.py"
MXFP4 = VLLM_ROOT / "vllm/model_executor/layers/quantization/mxfp4.py"

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def check_aiter_ops_moe():
    """Verify _aiter_ops.py has MoE-related CK/MXFP4 functions."""
    if not check("vllm/_aiter_ops.py exists", AITER_OPS.is_file()):
        return

    source = AITER_OPS.read_text()
    # Fix adds: fused_topk, fused_moe, shuffle_weight_a16w4, shuffle_scale_a16w4
    has_fused_moe = "fused_moe" in source or "fused_topk" in source
    check("_aiter_ops.py: MoE-related fused ops (fused_moe/fused_topk)",
          has_fused_moe,
          "No fused MoE or fused_topk found")

    has_weight_shuffle = (
        "shuffle_weight" in source or "shuffle_scale" in source or
        "a16w4" in source.lower() or "A16W4" in source
    )
    check("_aiter_ops.py: weight shuffle / A16W4 layout helpers",
          has_weight_shuffle,
          "No shuffle_weight, shuffle_scale, or A16W4 layout helpers")


def check_mxfp4_ck_backend():
    """Verify mxfp4.py has CK backend and fused MoE forward pass."""
    if not check("mxfp4.py exists", MXFP4.is_file()):
        return

    source = MXFP4.read_text()
    # Fix adds CK backend enum and CK-specific forward
    has_ck_backend = "CK" in source and ("backend" in source.lower() or "Backend" in source)
    check("mxfp4.py: CK backend variant",
          has_ck_backend,
          "No CK backend in Mxfp4Backend")

    has_fused_moe_apply = (
        "fused_moe" in source or "fused_topk" in source or
        ("moe" in source.lower() and "rocm" in source.lower())
    )
    check("mxfp4.py: fused MoE apply / CK forward path",
          has_fused_moe_apply,
          "No fused MoE or CK-specific MoE forward")

    has_weight_prep = (
        "shuffle" in source or "interleave" in source or
        "gate_up" in source or "hidden_pad" in source or "intermediate_pad" in source
    )
    check("mxfp4.py: CK weight preparation (shuffle/interleave/pad)",
          has_weight_prep,
          "No weight shuffle, interleave, or pad handling for CK")


def run_checks():
    print("=" * 60)
    print("vllm-ck-mxfp4-moe test harness")
    print("=" * 60)

    print("\n--- vllm/_aiter_ops.py ---")
    check_aiter_ops_moe()

    print("\n--- mxfp4.py ---")
    check_mxfp4_ck_backend()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
