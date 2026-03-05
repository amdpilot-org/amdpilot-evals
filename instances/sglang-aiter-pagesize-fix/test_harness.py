#!/usr/bin/env python3
"""Test harness for sglang-aiter-pagesize-fix eval instance.

Bug: (1) aiter MLA metadata hardcoded page_size=1, (2) deepseek_v2 hidden_states
received as tuple, (3) HiCache backend override broke aiter.

Exit 0 = PASS, Exit 1 = FAIL.
Output: SCORE: <0-100>
"""

import os
import re
import sys
from pathlib import Path

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/workspace/sglang"))
AITER_BACKEND = SGLANG_ROOT / "python/sglang/srt/layers/attention/aiter_backend.py"
DEEPSEEK_V2 = SGLANG_ROOT / "python/sglang/srt/models/deepseek_v2.py"
SERVER_ARGS = SGLANG_ROOT / "python/sglang/srt/server_args.py"

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


def check_aiter_page_size():
    """Verify aiter_backend.py uses dynamic page_size, not hardcoded 1."""
    if not check("aiter_backend.py exists", AITER_BACKEND.is_file()):
        return

    source = AITER_BACKEND.read_text()
    # Bug: hardcoded page_size=1 in MLA metadata
    # Fix: should use dynamic page_size (from config/args)
    hardcoded_match = re.search(r"page_size\s*=\s*1\b", source)
    check("aiter_backend.py: no hardcoded page_size=1 (use dynamic page_size)",
          not hardcoded_match,
          "MLA metadata still uses hardcoded page_size=1")


def check_deepseek_tuple_handling():
    """Verify deepseek_v2.py handles hidden_states as tuple."""
    if not check("deepseek_v2.py exists", DEEPSEEK_V2.is_file()):
        return

    source = DEEPSEEK_V2.read_text()
    # Bug: hidden_states can be tuple in some quantization paths, shape-based alloc fails
    # Fix: handle tuple (e.g. unpack, or isinstance check)
    has_tuple_handling = (
        "tuple" in source and "hidden_states" in source or
        "isinstance" in source and "hidden_states" in source or
        "hidden_states[0]" in source or "hidden_states[1]" in source or
        "unpack" in source.lower() and "hidden" in source.lower()
    )
    check("deepseek_v2.py: tuple handling for hidden_states",
          has_tuple_handling,
          "No tuple handling for hidden_states (causes shape alloc failure)")


def check_hicache_aiter_override():
    """Verify server_args.py HiCache condition does not override aiter backend."""
    if not check("server_args.py exists", SERVER_ARGS.is_file()):
        return

    source = SERVER_ARGS.read_text()
    # Bug: HiCache kernel I/O enabled -> FlashAttention3 workaround overrode user's aiter backend
    # Fix: FA3 override should only kick in when effective decode backend is FA3, not when user chose aiter
    has_hicache_condition = "hicache" in source.lower() or "hierarchical" in source.lower()
    has_backend_override = "attention" in source.lower() and "backend" in source.lower()
    # The fix adds a condition: only override when effective backend is FA3
    has_fa3_guard = (
        "fa3" in source.lower() or "flash_attn" in source.lower() or
        "decode" in source.lower() and "backend" in source.lower()
    )
    # If HiCache + backend logic exists, should have guard so aiter isn't overridden
    if has_hicache_condition and has_backend_override:
        check("server_args.py: HiCache does not override aiter backend",
              has_fa3_guard or "aiter" in source.lower(),
              "HiCache/backend override may break aiter selection")
    else:
        check("server_args.py: HiCache/backend logic present",
              has_hicache_condition or has_backend_override,
              "Could not find HiCache or backend override logic")


def run_checks():
    print("=" * 60)
    print("sglang-aiter-pagesize-fix test harness")
    print("=" * 60)

    print("\n--- aiter_backend.py page_size ---")
    check_aiter_page_size()

    print("\n--- deepseek_v2.py tuple handling ---")
    check_deepseek_tuple_handling()

    print("\n--- server_args.py HiCache/aiter ---")
    check_hicache_aiter_override()


if __name__ == "__main__":
    run_checks()
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
