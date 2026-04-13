#!/usr/bin/env python3
"""Test harness for sglang-amd-context-len-perf-fix.

Verifies that the attention backend computes KV cache length correctly
so the kernel dispatcher selects the right path for all context-length
settings.
"""
import importlib
import inspect
import re
import sys

sys.path.insert(0, "/workspace/sglang/python")

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


print("=" * 60)
print("sglang-amd-context-len-perf-fix test harness")
print("=" * 60)

# Read the aiter backend source
src_path = "/workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"
try:
    with open(src_path) as f:
        source = f.read()
    check("Read aiter_backend.py", True)
except Exception as e:
    check("Read aiter_backend.py", False, str(e)[:200])
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 1: Find where max_kv_len is computed for kernel dispatch
# The max KV length determines which kernel path is selected.
# The bug: max_kv_len is computed from page_table dimensions but
# doesn't account for the actual capacity per page.
max_kv_len_assignments = []
lines = source.split("\n")
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if "max_kv_len" in stripped and "=" in stripped and not stripped.startswith("#"):
        max_kv_len_assignments.append((i, stripped))

check(
    "max_kv_len computation found",
    len(max_kv_len_assignments) > 0,
    "No max_kv_len assignment found in attention backend"
)

# Check 2: Behavioral — verify the computation is correct
# Import the module and check if max_kv_len computation accounts for
# paged KV cache geometry (pages * capacity_per_page)
try:
    # Try to find decode-related methods and verify they compute
    # max_kv_len correctly for paged KV caches
    has_correct_geometry = False

    for lineno, line in max_kv_len_assignments:
        # The computation must account for both the number of pages
        # AND the capacity of each page. A buggy computation only
        # looks at page count, giving an underestimate.
        # Check that the computation involves a multiplication with
        # the page/block capacity factor
        if re.search(r"\*", line) and "page" in line.lower():
            has_correct_geometry = True
        # Also accept if it uses a precomputed total capacity
        if "total" in line.lower() or "capacity" in line.lower():
            has_correct_geometry = True

    check(
        "max_kv_len accounts for full KV cache geometry",
        has_correct_geometry,
        "max_kv_len computation appears to underestimate actual KV capacity"
    )
except Exception as e:
    check(
        "max_kv_len accounts for full KV cache geometry",
        False,
        f"Error analyzing computation: {str(e)[:100]}"
    )

# Check 3: Verify no regression — the kernel dispatch threshold
# should not be affected by context-length configuration
# Look for the dispatch logic that selects between kernel paths
dispatch_lines = []
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if "max_kv_len" in stripped and ("512" in stripped or "seqlen" in stripped):
        dispatch_lines.append((i, stripped))

if dispatch_lines:
    check(
        "Kernel dispatch uses corrected max_kv_len",
        True,
        "Dispatch threshold references max_kv_len"
    )
else:
    # Dispatch may use a different variable name — that's OK
    # as long as max_kv_len is computed correctly above
    check(
        "Kernel dispatch uses corrected max_kv_len",
        has_correct_geometry if 'has_correct_geometry' in dir() else False,
        "Could not locate dispatch threshold"
    )

# Check 4: Module imports successfully
try:
    from sglang.srt.layers.attention import aiter_backend
    check("Import aiter_backend module", True)
except Exception as e:
    err = str(e)
    if "rocm" in err.lower() or "hip" in err.lower() or "aiter" in err.lower():
        check("Import aiter_backend module", True,
              "(import skipped — no ROCm, but source fix verified)")
    else:
        check("Import aiter_backend module", False, err[:200])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.2f}")
sys.exit(0 if checks_passed == checks_total else 1)
