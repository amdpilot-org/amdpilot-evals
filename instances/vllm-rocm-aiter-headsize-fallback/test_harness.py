#!/usr/bin/env python3
"""Test harness for vllm PR #34570: AITER paged_attention_v1 decode fallback
for small head sizes.

Bug: The ll4mi kernel used in AITER's paged_attention_v1 has a minimum
head size requirement. Models with smaller head sizes (e.g., head_size=32)
crash with an obscure kernel error.

Tests:
  1. A head size threshold constant exists to guard the kernel path.
  2. Small head sizes are dispatched to a fallback attention path.
  3. The fallback attention kernel is imported and callable.
  4. Fallback dispatch occurs before other specialized paths.
  5. Fallback is called with the correct attention arguments.
"""
import ast
import os
import sys

checks_passed = 0
checks_total = 0

AITER_FA_PATH = "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_fa.py"


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
print("vllm-rocm-aiter-headsize-fallback test harness (PR #34570)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm_aiter_fa.py exists", os.path.isfile(AITER_FA_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

try:
    with open(AITER_FA_PATH) as f:
        source = f.read()
    tree = ast.parse(source)
except SyntaxError as e:
    check("rocm_aiter_fa.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: _MIN_HEAD_SIZE_FOR_LL4MI constant exists with value 64.
#
# Before fix: no head_size check — ll4mi kernel crashes on head_size < 64.
# After fix: explicit constant defines the minimum head_size threshold.
# ---------------------------------------------------------------------------
print("\n--- Check 1: Head size threshold constant ---")

has_min_head_constant = False
min_head_value_64 = False

# Check for the constant in the source (could be module-level or method-local)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and "MIN_HEAD_SIZE" in target.id:
                has_min_head_constant = True
                if isinstance(node.value, ast.Constant) and node.value.value == 64:
                    min_head_value_64 = True

check(
    "_MIN_HEAD_SIZE_FOR_LL4MI constant exists",
    has_min_head_constant,
    "No head_size threshold constant found — ll4mi kernel will crash on small head sizes",
)

check(
    "Head size threshold is 64 (16 * NWARPS=4)",
    min_head_value_64,
    "Expected threshold of 64 for ll4mi kernel compatibility",
)

# ---------------------------------------------------------------------------
# Check 2: forward method has head_size comparison that routes to fallback.
#
# Before fix: decode path goes directly to paged_attention_v1 or shuffle path.
# After fix: head_size < threshold check routes to unified_attention first.
# ---------------------------------------------------------------------------
print("\n--- Check 2: Head size dispatch in forward ---")

# Find the forward method in the impl class and look for head_size comparison
has_headsize_check = False
has_unified_import = False
has_unified_call = False

# Search for the decode section of forward that has the head_size check
source_lines = source.splitlines()

# Look for the pattern: use_unified_attention = self.head_size < _MIN_HEAD_SIZE
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and "use_unified_attention" in target.id:
                has_headsize_check = True
    # Also check for direct if-comparison
    if isinstance(node, ast.If):
        test_src = ast.dump(node.test)
        if "head_size" in test_src and "MIN_HEAD_SIZE" in test_src:
            has_headsize_check = True

check(
    "Forward method has head_size < threshold dispatch",
    has_headsize_check,
    "No head_size comparison found — small head_size models will crash",
)

# ---------------------------------------------------------------------------
# Check 3: unified_attention import exists in the fallback path.
#
# The fix adds a local import of unified_attention from
# aiter.ops.triton.unified_attention when head_size < 64.
# ---------------------------------------------------------------------------
print("\n--- Check 3: unified_attention fallback import ---")

for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and "unified_attention" in node.module:
            for alias in node.names:
                if alias.name == "unified_attention":
                    has_unified_import = True

check(
    "unified_attention imported from aiter.ops.triton",
    has_unified_import,
    "Missing unified_attention import — no fallback path for small head_size",
)

# ---------------------------------------------------------------------------
# Check 4: unified_attention is actually called in the fallback branch.
#
# Verify that unified_attention() is called with the expected keyword args
# (softmax_scale, causal, block_table, etc.)
# ---------------------------------------------------------------------------
print("\n--- Check 4: unified_attention call with correct args ---")

for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id == "unified_attention":
            has_unified_call = True
        elif isinstance(func, ast.Attribute) and func.attr == "unified_attention":
            has_unified_call = True

# Also check key args in the source around the unified_attention call
ua_call_section = ""
for i, line in enumerate(source_lines):
    if "unified_attention(" in line and "import" not in line:
        # Grab surrounding context (the call with keyword args)
        ua_call_section = "\n".join(source_lines[max(0, i):min(len(source_lines), i + 20)])
        break

has_key_args = (
    "softmax_scale" in ua_call_section
    and "block_table" in ua_call_section
    and "causal" in ua_call_section
)

check(
    "unified_attention called in fallback branch",
    has_unified_call,
    "unified_attention not called — fallback path incomplete",
)

check(
    "unified_attention called with required args (softmax_scale, causal, block_table)",
    has_key_args,
    "Missing key arguments in unified_attention call",
)

# ---------------------------------------------------------------------------
# Check 5: Fallback dispatches BEFORE the shuffle_kv_cache path.
#
# The fix adds the head_size check as an if/elif before the existing
# is_shuffle_kv_cache_enabled() branch. This ensures small head_size
# models take the fallback path instead of the ll4mi kernel.
# ---------------------------------------------------------------------------
print("\n--- Check 5: Fallback ordering ---")

# Find lines with use_unified_attention and is_shuffle_kv_cache_enabled
ua_line = -1
shuffle_line = -1
for i, line in enumerate(source_lines):
    stripped = line.strip()
    if "use_unified_attention" in stripped and ("if " in stripped or "elif " in stripped):
        ua_line = i
    if "is_shuffle_kv_cache_enabled" in stripped and ("if " in stripped or "elif " in stripped):
        if ua_line > 0:  # Only check after we found the ua check
            shuffle_line = i
            break

check(
    "Head size fallback dispatches before shuffle_kv_cache path",
    ua_line > 0 and shuffle_line > ua_line,
    "Fallback path not ordered before shuffle_kv_cache — wrong dispatch order",
)

# ---------------------------------------------------------------------------
# Check 6: The fallback path asserts shuffle KV cache is not enabled.
#
# unified_attention fallback doesn't support shuffle layout, so the fix
# adds an assertion for safety.
# ---------------------------------------------------------------------------
print("\n--- Check 6: Safety assertion ---")

has_shuffle_assert = False
for i, line in enumerate(source_lines):
    if ("assert" in line and "shuffle" in line.lower()
            and "unified_attention" in "\n".join(source_lines[max(0, i-5):i+5]).lower()):
        has_shuffle_assert = True
        break

check(
    "Fallback path asserts shuffle KV cache is not enabled",
    has_shuffle_assert,
    "Missing safety assertion — unified_attention with shuffle layout is unsupported",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
