#!/usr/bin/env python3
"""Test harness for aiter-flash-attn-overhead eval instance.

Measures flash_attn_func launch overhead and verifies it's reduced.
Exit 0 = PASS, Exit 1 = FAIL.

Output: SCORE: <0-100>
"""

import sys
import time
from pathlib import Path

AITER_ROOT = Path("/sgl-workspace/aiter")
sys.path.insert(0, str(AITER_ROOT))

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        checks_passed += 1
    msg = f"  [{status}] {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return condition


def check_static():
    """Verify the JIT code has been modified to reduce overhead."""
    core_file = AITER_ROOT / "aiter" / "jit" / "core.py"
    if not check("aiter/jit/core.py exists", core_file.is_file()):
        return False

    source = core_file.read_text()
    lines = source.split("\n")
    has_original_try_except = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "get_module(md_name)" in stripped and i + 3 < len(lines):
            next_lines = "\n".join(lines[i:i+4])
            if "except" in next_lines and "get_module(" in next_lines:
                has_original_try_except = True

    check("Original try-except overhead pattern removed or optimized",
          not has_original_try_except,
          "The get_module(md_name) -> exception -> get_module(md) pattern still exists")
    return True


def check_runtime():
    """Measure actual launch overhead of flash_attn_func."""
    try:
        import torch
    except ImportError:
        check("torch available", False)
        return False

    if not torch.cuda.is_available():
        check("GPU available", False)
        return False
    check("GPU available", True)
    device = torch.device("cuda:0")

    try:
        import aiter
        from aiter import flash_attn_func
    except ImportError as e:
        check("aiter.flash_attn_func importable", False, str(e))
        return False
    check("aiter.flash_attn_func importable", True)

    batch_size, seqlen, num_heads, head_dim = 1, 128, 8, 64
    q = torch.randn(batch_size, seqlen, num_heads, head_dim,
                     device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for _ in range(5):
        try:
            out = flash_attn_func(q, k, v)
        except Exception as e:
            check("flash_attn_func warmup", False, str(e))
            return False
    check("flash_attn_func warmup", True)

    out = flash_attn_func(q, k, v)
    has_nan = torch.isnan(out).any().item()
    check("flash_attn_func output valid (no NaN)", not has_nan)

    torch.cuda.synchronize()
    n_iters = 100
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)

    times.sort()
    p50 = times[len(times) // 2]
    p10 = times[len(times) // 10]
    avg = sum(times) / len(times)

    print(f"\n  Launch overhead measurements ({n_iters} iterations):")
    print(f"    P10: {p10:.1f}us | P50: {p50:.1f}us | Avg: {avg:.1f}us")

    OVERHEAD_TARGET_US = 150.0
    check(f"Launch overhead P50 < {OVERHEAD_TARGET_US}us",
          p50 < OVERHEAD_TARGET_US,
          f"P50={p50:.1f}us (target <{OVERHEAD_TARGET_US}us)")
    return True


def run_checks():
    print("=" * 60)
    print("aiter-flash-attn-overhead test harness")
    print("=" * 60)
    check_static()
    check_runtime()


if __name__ == "__main__":
    run_checks()
    print()
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
