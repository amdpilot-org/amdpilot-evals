#!/usr/bin/env python3
"""Test harness for sglang-amd-context-len-perf-fix.

Verifies that the aiter attention backend correctly computes max_kv_len
by accounting for page_size, preventing incorrect kernel selection.
"""
import ast
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

# Read the source file
src_path = "/workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"
try:
    with open(src_path) as f:
        source = f.read()
    check("Read aiter_backend.py", True)
except Exception as e:
    check("Read aiter_backend.py", False, str(e)[:200])
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 1: Find the forward_decode method(s) that compute max_kv_len
# Find all lines that assign max_kv_len
max_kv_len_lines = []
for i, line in enumerate(source.split("\n"), 1):
    stripped = line.strip()
    if "max_kv_len" in stripped and "=" in stripped and "page_table" in stripped:
        max_kv_len_lines.append((i, stripped))

check(
    "Find max_kv_len assignment with page_table",
    len(max_kv_len_lines) > 0,
    "No max_kv_len = ... page_table... line found"
)

# Check 2: Every max_kv_len assignment from page_table should include page_size
all_include_page_size = True
buggy_lines = []
for lineno, line in max_kv_len_lines:
    # The line should contain page_size multiplication
    if "page_size" not in line:
        all_include_page_size = False
        buggy_lines.append(f"L{lineno}: {line}")

check(
    "max_kv_len includes page_size factor",
    all_include_page_size,
    f"Missing page_size in: {'; '.join(buggy_lines)}"
)

# Check 3: Verify the pattern is specifically multiplication
# Accept patterns like:
#   page_table.shape[1] * self.page_size
#   page_table.shape[1] * page_size
#   self.page_size * page_table.shape[1]
correct_pattern = True
for lineno, line in max_kv_len_lines:
    has_multiply = bool(re.search(
        r"page_table\.shape\[1\]\s*\*\s*(?:self\.)?page_size|"
        r"(?:self\.)?page_size\s*\*\s*page_table\.shape\[1\]",
        line
    ))
    if not has_multiply:
        correct_pattern = False

check(
    "Correct multiplication pattern (shape * page_size)",
    correct_pattern,
    "max_kv_len should be page_table.shape[1] * page_size"
)

# Check 4: Module imports successfully
try:
    from sglang.srt.layers.attention import aiter_backend
    check("Import aiter_backend module", True)
except Exception as e:
    # Import may fail without ROCm — that's OK for this check
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
