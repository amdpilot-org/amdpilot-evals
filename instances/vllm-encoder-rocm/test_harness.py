#!/usr/bin/env python3
"""Test harness for vllm-encoder-rocm eval instance.

Bug: RocmAttentionImpl raises NotImplementedError for encoder self-attention
(AttentionType.ENCODER). Two backend files need changes: rocm_attn.py and
rocm_aiter_unified_attn.py.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import os
import sys
from pathlib import Path

VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/workspace/vllm"))
ROCM_ATTN = VLLM_ROOT / "vllm/v1/attention/backends/rocm_attn.py"
ROCM_AITER_UNIFIED = VLLM_ROOT / "vllm/v1/attention/backends/rocm_aiter_unified_attn.py"

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


def check_rocm_attn():
    """Verify rocm_attn.py has encoder attention support (no NotImplementedError for ENCODER)."""
    if not check("rocm_attn.py exists", ROCM_ATTN.is_file()):
        return

    source = ROCM_ATTN.read_text()
    # Bug: only DECODER handled, ENCODER raises NotImplementedError
    # Fix: ENCODER should be handled (no NotImplementedError for AttentionType.ENCODER)
    lines = source.split("\n")
    encoder_raises_not_impl = False
    for i, line in enumerate(lines):
        if "NotImplementedError" in line:
            # Check if this raise is in a path that catches ENCODER (e.g. else after DECODER)
            context = "\n".join(lines[max(0, i - 15):i + 1])
            if ("attention_type" in context or "AttentionType" in context) and "ENCODER" not in context:
                # Raise exists in attention_type block but ENCODER has no explicit branch
                encoder_raises_not_impl = True
                break

    check("rocm_attn.py: no NotImplementedError for ENCODER attention type",
          not encoder_raises_not_impl,
          "ENCODER falls through to NotImplementedError")


def check_rocm_attn_encoder_handling():
    """Verify rocm_attn.py handles ENCODER in AttentionType dispatch."""
    if not ROCM_ATTN.is_file():
        return

    source = ROCM_ATTN.read_text()
    # Fix: should have ENCODER in the handling (not just DECODER)
    has_encoder_handling = (
        "ENCODER" in source and
        ("AttentionType.ENCODER" in source or "attention_type" in source)
    )
    check("rocm_attn.py: ENCODER in AttentionType handling",
          has_encoder_handling,
          "No ENCODER handling found; only DECODER may be supported")


def check_rocm_aiter_unified():
    """Verify rocm_aiter_unified_attn.py has encoder attention support."""
    if not check("rocm_aiter_unified_attn.py exists", ROCM_AITER_UNIFIED.is_file()):
        return

    source = ROCM_AITER_UNIFIED.read_text()
    # Bug: ENCODER raises NotImplementedError; fix adds explicit ENCODER handling
    lines = source.split("\n")
    encoder_raises_not_impl = False
    for i, line in enumerate(lines):
        if "NotImplementedError" in line:
            context = "\n".join(lines[max(0, i - 15):i + 1])
            if ("attention_type" in context or "AttentionType" in context) and "ENCODER" not in context:
                encoder_raises_not_impl = True
                break

    check("rocm_aiter_unified_attn.py: no NotImplementedError for ENCODER",
          not encoder_raises_not_impl,
          "ENCODER falls through to NotImplementedError")

    has_encoder_handling = "ENCODER" in source
    check("rocm_aiter_unified_attn.py: ENCODER in handling",
          has_encoder_handling,
          "No ENCODER handling found")


def run_checks():
    print("=" * 60)
    print("vllm-encoder-rocm test harness")
    print("=" * 60)

    print("\n--- rocm_attn.py ---")
    check_rocm_attn()
    check_rocm_attn_encoder_handling()

    print("\n--- rocm_aiter_unified_attn.py ---")
    check_rocm_aiter_unified()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
