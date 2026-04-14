#!/usr/bin/env python3
"""Test harness for vllm-rocm-fused-moe-fix.

Verify that the ROCm fused MoE custom op and grouped top-k path
handle all required parameters correctly.
"""
from __future__ import annotations

import inspect
import sys

sys.path.insert(0, "/workspace/vllm")

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
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


def signatures_match(a, b) -> tuple[bool, str]:
    pa = list(inspect.signature(a).parameters.items())
    pb = list(inspect.signature(b).parameters.items())
    if len(pa) != len(pb):
        return False, f"param count {len(pa)} vs {len(pb)}"
    for (na, _), (nb, _) in zip(pa, pb):
        if na != nb:
            return False, f"param name mismatch at same position: {na!r} vs {nb!r}"
    return True, ""


print("=" * 60)
print("vllm-rocm-fused-moe-fix test harness")
print("=" * 60)

# --- Check 1: import fused_moe-related modules ---
try:
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod  # noqa: F401
    import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe as rocm_aiter_fmoe  # noqa: F401

    check("Import fused_moe and rocm_aiter_fused_moe modules", True)
except Exception as e:
    check("Import fused_moe and rocm_aiter_fused_moe modules", False, str(e)[:200])

# --- Check 2: fake vs impl signature for ROCm AITER fused_moe custom op ---
try:
    import vllm._aiter_ops as aiter_ops

    fake = aiter_ops._rocm_aiter_fused_moe_fake
    impl = aiter_ops._rocm_aiter_fused_moe_impl
    ok, detail = signatures_match(fake, impl)
    check(
        "_rocm_aiter_fused_moe_fake matches _rocm_aiter_fused_moe_impl signature",
        ok,
        detail,
    )
except Exception as e:
    check(
        "_rocm_aiter_fused_moe_fake matches _rocm_aiter_fused_moe_impl signature",
        False,
        str(e)[:200],
    )

# --- Check 3: routed_scaling_factor in ROCm AITER grouped top-k path ---
try:
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        rocm_aiter_grouped_topk,
    )

    sig = inspect.signature(rocm_aiter_grouped_topk)
    has_param = "routed_scaling_factor" in sig.parameters
    check(
        "rocm_aiter_grouped_topk accepts routed_scaling_factor",
        has_param,
        "missing parameter" if not has_param else "",
    )
except Exception as e:
    check("rocm_aiter_grouped_topk accepts routed_scaling_factor", False, str(e)[:200])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
