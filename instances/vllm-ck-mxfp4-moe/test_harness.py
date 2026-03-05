#!/usr/bin/env python3
"""Test harness for vllm-ck-mxfp4-moe.

Checks for SPECIFIC new functions/classes added by the CK MXFP4 MoE PR.
The pre-fix code already has generic fused_moe -- we must check for the
MXFP4/CK-specific additions that don't exist before the fix.
"""
import sys
sys.path.insert(0, "/workspace/vllm")

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

print("=" * 60)
print("vllm-ck-mxfp4-moe test harness")
print("=" * 60)

import importlib.util

# Check 1: _aiter_ops has shuffle_weight_a16w4 (NEW function from the PR)
try:
    spec = importlib.util.spec_from_file_location(
        "_aiter_ops", "/workspace/vllm/vllm/_aiter_ops.py")
    aiter_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aiter_mod)

    # These functions are SPECIFIC to the CK MXFP4 PR -- don't exist before
    has_shuffle_weight = False
    has_shuffle_scale = False
    has_fused_topk = False
    for name in dir(aiter_mod):
        obj = getattr(aiter_mod, name, None)
        if isinstance(obj, type):
            if hasattr(obj, "shuffle_weight_a16w4"):
                has_shuffle_weight = True
            if hasattr(obj, "shuffle_scale_a16w4"):
                has_shuffle_scale = True
            if hasattr(obj, "fused_topk"):
                has_fused_topk = True

    check("shuffle_weight_a16w4 exists in _aiter_ops", has_shuffle_weight,
          "PR-specific function not found")
    check("shuffle_scale_a16w4 exists in _aiter_ops", has_shuffle_scale,
          "PR-specific function not found")
    check("fused_topk method exists in _aiter_ops", has_fused_topk,
          "PR-specific function not found")
except Exception as e:
    check("Import _aiter_ops", False, str(e))

# Check 2: mxfp4.py has Mxfp4Backend.CK enum value (doesn't exist before fix)
try:
    spec2 = importlib.util.spec_from_file_location(
        "mxfp4", "/workspace/vllm/vllm/model_executor/layers/quantization/mxfp4.py")
    mxfp4_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mxfp4_mod)

    backend_cls = getattr(mxfp4_mod, "Mxfp4Backend", None)
    has_ck = backend_cls is not None and hasattr(backend_cls, "CK")
    check("Mxfp4Backend.CK enum value exists", has_ck,
          "CK backend enum not found in Mxfp4Backend")
except Exception as e:
    check("Import mxfp4 module", False, str(e)[:200])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
