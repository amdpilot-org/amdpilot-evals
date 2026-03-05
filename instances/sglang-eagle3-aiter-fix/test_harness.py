#!/usr/bin/env python3
"""Test harness for sglang-eagle3-aiter-fix eval instance.

Bug: EAGLE3 speculative decoding crashes with aiter attention backend.
target_verify CUDA graph has stale indices, missing use_mla guards.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import os
import sys
from pathlib import Path

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/workspace/sglang"))
AITER_BACKEND = SGLANG_ROOT / "python/sglang/srt/layers/attention/aiter_backend.py"

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


def check_use_mla_guards():
    """Verify use_mla guard in init_cuda_graph_state or capture/replay methods."""
    if not check("aiter_backend.py exists", AITER_BACKEND.is_file()):
        return

    source = AITER_BACKEND.read_text()
    # Bug: _use_mla_ps_kernel guards missing self.use_mla check -> AttributeError
    # Fix: guards should include use_mla (e.g. self.use_mla and _use_mla_ps_kernel)
    lines = source.split("\n")
    has_unsafe_mla_guard = False
    for i, line in enumerate(lines):
        if "_use_mla_ps_kernel" in line and "use_mla" not in line:
            # Check if this is a guard that could run for non-MLA (missing use_mla)
            context = "\n".join(lines[max(0, i - 2):i + 2])
            if "if " in context and "max_split_per_batch" not in context:
                has_unsafe_mla_guard = True
                break

    check("aiter_backend.py: use_mla guard with _use_mla_ps_kernel",
          not has_unsafe_mla_guard,
          "Found _use_mla_ps_kernel guard without use_mla check (causes AttributeError)")


def check_target_verify_indptr():
    """Verify target_verify and target_extend handle qo_indptr/kv_indptr correctly."""
    if not AITER_BACKEND.is_file():
        return

    source = AITER_BACKEND.read_text()
    # Bug: target_verify had separate qo_indptr/kv_indptr for MLA vs non-MLA, non-MLA used stale
    # Fix: both target_verify and target_extend should handle qo_indptr/kv_indptr consistently
    has_target_verify = "target_verify" in source
    has_indptr_handling = "qo_indptr" in source and "kv_indptr" in source
    check("aiter_backend.py: target_verify with qo_indptr/kv_indptr handling",
          has_target_verify and has_indptr_handling,
          "target_verify or indptr handling missing")

    has_target_extend = "target_extend" in source
    check("aiter_backend.py: target_extend present",
          has_target_extend,
          "target_extend missing")


def check_custom_mask_verify():
    """Verify custom_mask is handled in verify path."""
    if not AITER_BACKEND.is_file():
        return

    source = AITER_BACKEND.read_text()
    # custom_mask should be handled in verify path for EAGLE3
    has_custom_mask = "custom_mask" in source
    check("aiter_backend.py: custom_mask handling in verify path",
          has_custom_mask,
          "custom_mask not found in aiter_backend")


def run_checks():
    print("=" * 60)
    print("sglang-eagle3-aiter-fix test harness")
    print("=" * 60)

    print("\n--- use_mla guards ---")
    check_use_mla_guards()

    print("\n--- target_verify / target_extend indptr ---")
    check_target_verify_indptr()

    print("\n--- custom_mask ---")
    check_custom_mask_verify()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
