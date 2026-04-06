#!/usr/bin/env python3
"""Test harness for vllm PR #38615: nhead<16 persistent MLA buffer mismatch on ROCm.

Bug: AiterMLADecodeMetadata.__init__ sets self._num_attention_heads from
vllm_config.model_config.get_num_attention_heads(parallel_config), which can
be <16 with enough TP (e.g., kimi-k2.5 with TP8 → 8 heads). But aiter's
get_mla_metadata_info_v1 / get_mla_metadata_v1 and mla_decode_fwd require
the head count to be at least 16 for correct persistent buffer sizing.

Tests (behavioral, not source-pattern matching):
  1. AST extraction — locate _num_attention_heads assignment in __init__,
     compile the RHS expression, evaluate with self.num_heads=8 → expect 16.
  2. Regression guard — evaluate with self.num_heads=32 → expect 32 (not
     hardcoded to 16).
  3. Anti-hack guard — evaluate with self.num_heads=1 → expect 16 (not 1).
  4. GPU behavioral (optional) — import the class, check _num_attention_heads
     value with a mock config where num_heads<16.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

ROCM_AITER_MLA_PATH = "/workspace/vllm/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
VENV_PYTHON = "/opt/venv/bin/python3"


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


def run_subprocess(script, timeout=120):
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-mla-nhead-fix test harness (PR #38615)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm_aiter_mla.py exists", os.path.isfile(ROCM_AITER_MLA_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(ROCM_AITER_MLA_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("rocm_aiter_mla.py is valid Python", True)
except SyntaxError as e:
    check("rocm_aiter_mla.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Checks 2-4 (static behavioral): extract _num_attention_heads assignment
# from __init__ and evaluate it with controlled self.num_heads values.
#
# Strategy: walk the AST to find assignments to self._num_attention_heads
# inside any __init__ method. Extract the RHS expression, compile it, and
# evaluate with a mock `self` object that has num_heads set to test values.
#
# Before fix: RHS is vllm_config.model_config.get_num_attention_heads(...)
#   → NameError on vllm_config (not in our namespace), or returns actual
#     head count (8) if we provide a mock vllm_config → FAIL
#
# After fix: RHS is max(16, self.num_heads)
#   → evaluates cleanly with just self.num_heads in namespace → 16 for
#     num_heads=8, 32 for num_heads=32 → PASS
# ---------------------------------------------------------------------------
print("\n--- Checks 2-4: _num_attention_heads assignment behavior ---")

# Find the assignment to self._num_attention_heads inside an __init__
nhead_assign_node = None
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "__init__":
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (isinstance(target, ast.Attribute)
                            and target.attr == "_num_attention_heads"
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        nhead_assign_node = child
                        break
            if nhead_assign_node:
                break
    if nhead_assign_node:
        break

if not check(
    "self._num_attention_heads assignment found in __init__",
    nhead_assign_node is not None,
    "assignment not found — code structure may have changed",
):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Extract the RHS expression source
rhs_src = ast.get_source_segment(source_text, nhead_assign_node.value)
if rhs_src is None:
    # Fallback: reconstruct from line range
    src_lines = source_text.splitlines()
    assign_lines = src_lines[nhead_assign_node.lineno - 1:nhead_assign_node.end_lineno]
    full_assign = "\n".join(assign_lines)
    # Extract RHS after the '='
    eq_idx = full_assign.index("=")
    rhs_src = full_assign[eq_idx + 1:].strip()


def eval_nhead_expr(num_heads_val):
    """Evaluate the _num_attention_heads RHS with a mock self object."""

    class MockSelf:
        pass

    mock_self = MockSelf()
    mock_self.num_heads = num_heads_val
    ns = {"self": mock_self, "max": max, "min": min}
    try:
        return eval(compile(rhs_src, ROCM_AITER_MLA_PATH, "eval"), ns)
    except NameError as e:
        # Before fix: references vllm_config which isn't in namespace
        return f"NameError:{e}"
    except Exception as e:
        return f"Error:{type(e).__name__}:{e}"


# Check 2: num_heads=8 → should return 16 (the fix: max(16, 8) = 16)
result_8 = eval_nhead_expr(8)
check(
    "_num_attention_heads with num_heads=8 → 16 (fix for kimi-k2.5 TP8)",
    result_8 == 16,
    f"got {result_8!r} — heads<16 not clamped to minimum 16",
)

# Check 3: num_heads=32 → should return 32 (no clamping for large values)
result_32 = eval_nhead_expr(32)
check(
    "_num_attention_heads with num_heads=32 → 32 (no regression)",
    result_32 == 32,
    f"got {result_32!r} — large head count should pass through unchanged",
)

# Check 4: num_heads=1 → should return 16 (anti-hack: not hardcoded to
# any specific small value, but correctly uses max(16, ...) for all <16)
result_1 = eval_nhead_expr(1)
check(
    "_num_attention_heads with num_heads=1 → 16 (anti-hack guard)",
    result_1 == 16,
    f"got {result_1!r} — minimum head count must be 16",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
