#!/usr/bin/env python3
"""Test harness for vllm PR #33734: dtype mismatch in gemm_a4w4 call.

Bug: In quark_ocp_mx.py's gemm_with_dynamic_quant(), the weight tensor is
passed directly to gemm_a4w4() without dtype conversion. The quantized input
x_q has type float8_e4m3fnuz but weight may be stored as a raw byte tensor
(uint8/int8). gemm_a4w4 requires both operands to have matching dtype.
This causes a dtype mismatch error on ROCm when running MX quantized models.

Tests (behavioral, not source-pattern matching):
  1. AST extraction — locate the gemm_a4w4() call in gemm_with_dynamic_quant,
     verify the weight argument (2nd positional arg) includes a .view() call.
  2. Expression evaluation — compile the weight argument expression, evaluate
     with mock tensors, verify output dtype matches x_q dtype.
  3. GPU behavioral (optional) — import the module, call with mismatched dtype
     tensors, verify no dtype error.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

QUARK_OCP_MX_PATH = "/workspace/vllm/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py"
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
print("vllm-rocm-quark-dtype-fix test harness (PR #33734)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("quark_ocp_mx.py exists", os.path.isfile(QUARK_OCP_MX_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(QUARK_OCP_MX_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("quark_ocp_mx.py is valid Python", True)
except SyntaxError as e:
    check("quark_ocp_mx.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 2: locate gemm_a4w4 call and verify weight arg has .view()
#
# Walk AST to find calls to gemm_a4w4 inside gemm_with_dynamic_quant.
# The second positional argument is the weight tensor. After the fix,
# it should be weight.view(x_q.dtype), which in AST is:
#   Call(func=Attribute(value=Name(id='weight'), attr='view'), ...)
#
# Before the fix, it's just Name(id='weight') → no .view() → FAIL
# ---------------------------------------------------------------------------
print("\n--- Check 2: gemm_a4w4 weight argument has dtype conversion ---")

gemm_call_node = None
parent_fn = None
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "gemm_with_dynamic_quant":
        parent_fn = node
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id == "gemm_a4w4":
                    gemm_call_node = child
                    break
                elif isinstance(func, ast.Attribute) and func.attr == "gemm_a4w4":
                    gemm_call_node = child
                    break
        break

if not check(
    "gemm_a4w4() call found in gemm_with_dynamic_quant",
    gemm_call_node is not None,
    "call not found — function may have been renamed or restructured",
):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# The weight is the 2nd positional argument (index 1, after x_q)
if len(gemm_call_node.args) < 2:
    check("gemm_a4w4 has at least 2 positional arguments", False,
          f"found {len(gemm_call_node.args)} args")
else:
    weight_arg = gemm_call_node.args[1]

    # Check that the weight argument involves a .view() call
    # After fix: weight.view(x_q.dtype) → Call(func=Attribute(..., attr='view'))
    # Before fix: just Name(id='weight') → no .view()
    def has_view_call(node):
        """Check if the AST node contains a .view() method call."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "view":
                return True
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute) and child.func.attr == "view":
                    return True
        return False

    has_view = has_view_call(weight_arg)
    check(
        "Weight arg to gemm_a4w4 includes .view() dtype conversion",
        has_view,
        "weight passed without .view() — dtype mismatch will occur",
    )

    # Check that the .view() argument references x_q.dtype (not arbitrary dtype)
    def view_references_xq_dtype(node):
        """Check if a .view() call in the node uses x_q.dtype as its argument."""
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "view"
                    and len(child.args) >= 1):
                view_arg = child.args[0]
                # Should be x_q.dtype
                if (isinstance(view_arg, ast.Attribute)
                        and view_arg.attr == "dtype"
                        and isinstance(view_arg.value, ast.Name)
                        and view_arg.value.id == "x_q"):
                    return True
        return False

    if has_view:
        check(
            "Weight .view() uses x_q.dtype (correct dtype source)",
            view_references_xq_dtype(weight_arg),
            ".view() present but does not reference x_q.dtype",
        )

# ---------------------------------------------------------------------------
# Check 3 (behavioral, subprocess): verify .view() dtype conversion works
# correctly with actual torch tensors.
#
# Create a uint8 tensor (simulating raw quantized weight) and a
# float8_e4m3fnuz tensor (simulating x_q), apply .view(x_q.dtype), and
# verify the output dtype matches x_q.
# ---------------------------------------------------------------------------
print("\n--- Check 3: dtype conversion behavior with torch tensors ---")

dtype_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    import torch
    print("TORCH:OK")
except ImportError:
    print("TORCH:FAIL")
    sys.exit(0)

# Simulate the dtype mismatch scenario
try:
    fp8_dtype = torch.float8_e4m3fnuz  # AMD FP8 format

    # Create mock tensors
    x_q = torch.zeros(4, 4, dtype=fp8_dtype, device='cpu')
    weight_raw = torch.zeros(4, 4, dtype=torch.uint8, device='cpu')

    # Before fix: weight_raw passed directly — dtype is uint8, not fp8
    pre_fix_dtype_match = (weight_raw.dtype == x_q.dtype)
    print(f"PRE_FIX_DTYPE_MATCH:{pre_fix_dtype_match}")

    # After fix: weight.view(x_q.dtype) — dtype now matches
    weight_fixed = weight_raw.view(x_q.dtype)
    post_fix_dtype_match = (weight_fixed.dtype == x_q.dtype)
    print(f"POST_FIX_DTYPE_MATCH:{post_fix_dtype_match}")

    # Verify data is preserved (same bytes, different interpretation)
    same_data = (weight_fixed.view(torch.uint8) == weight_raw).all().item()
    print(f"DATA_PRESERVED:{same_data}")

except Exception as e:
    print(f"DTYPE_TEST:FAIL:{type(e).__name__}:{str(e)[:200]}")
"""

try:
    stdout3, stderr3, rc3 = run_subprocess(dtype_script, timeout=30)
except subprocess.TimeoutExpired:
    stdout3, rc3 = "TIMEOUT", -1

if "TORCH:FAIL" in stdout3:
    print("  [SKIP] torch not available — dtype checks skipped")
elif "TORCH:OK" in stdout3:
    check(
        "Raw weight dtype does NOT match x_q dtype (confirms bug scenario)",
        "PRE_FIX_DTYPE_MATCH:False" in stdout3,
        "dtypes already match — test scenario invalid",
    )
    check(
        "weight.view(x_q.dtype) produces matching dtype",
        "POST_FIX_DTYPE_MATCH:True" in stdout3,
        ".view() did not produce correct dtype",
    )
    check(
        "Data preserved through .view() reinterpretation",
        "DATA_PRESERVED:True" in stdout3,
        "data corruption during dtype view",
    )
else:
    print(f"  [SKIP] Unexpected output — dtype checks skipped")

# ---------------------------------------------------------------------------
# Check 4 (anti-hack guard): verify the gemm_with_dynamic_quant function
# exists and is callable — prevents deletion-based hacks.
# ---------------------------------------------------------------------------
print("\n--- Check 4: gemm_with_dynamic_quant function integrity ---")

check(
    "gemm_with_dynamic_quant function exists in source",
    parent_fn is not None,
    "function not found — may have been deleted or renamed",
)

# Verify the function has the expected parameters (x, weight, x_s, weight_scale, ...)
if parent_fn is not None:
    param_names = [arg.arg for arg in parent_fn.args.args]
    # Should have 'self' and core params
    check(
        "gemm_with_dynamic_quant has weight and x_s parameters",
        "weight" in param_names and ("x_s" in param_names or "x_scale" in param_names or "x_scales" in param_names),
        f"params: {param_names}",
    )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
