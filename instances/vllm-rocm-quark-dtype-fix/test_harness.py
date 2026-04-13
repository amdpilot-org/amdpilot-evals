#!/usr/bin/env python3
"""Test harness for vllm-rocm-quark-dtype-fix.

Verifies that gemm_with_dynamic_quant converts weight dtype before
passing to the GEMM kernel (gemm_a4w4).

Checks:
  0. Target file exists
  1. gemm_with_dynamic_quant function found in source
  2. Weight dtype converted before gemm_a4w4 call (source analysis)
"""
import ast
import os
import re
import sys

checks_passed = 0
checks_total = 0

QUARK_OCP_MX_PATH = "/workspace/vllm/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py"

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
print("vllm-rocm-quark-dtype-fix test harness")
print("=" * 60)

# --- Check 0: target file exists ---
if not check("quark_ocp_mx.py exists", os.path.isfile(QUARK_OCP_MX_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Read the source file
with open(QUARK_OCP_MX_PATH) as f:
    source = f.read()

# --- Check 1: find gemm_with_dynamic_quant function ---
print("\n--- Check 1: Function discovery ---")

# Parse AST to find the function
try:
    tree = ast.parse(source)
except SyntaxError as e:
    check("gemm_with_dynamic_quant found in source", False, f"SyntaxError: {e}")
    print(f"\nSCORE: {checks_passed / checks_total * 100:.1f}")
    sys.exit(1)

gemm_func_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "gemm_with_dynamic_quant":
        gemm_func_node = node
        break

func_found = gemm_func_node is not None
check("gemm_with_dynamic_quant found in source", func_found,
      "function not found in AST")

if not func_found:
    print(f"\nSCORE: {checks_passed / checks_total * 100:.1f}")
    sys.exit(1)

# --- Check 2: Weight dtype converted before gemm_a4w4 call ---
print("\n--- Check 2: Dtype conversion analysis ---")

# Extract the function source lines
func_start = gemm_func_node.lineno
func_end = gemm_func_node.end_lineno
func_lines = source.split("\n")[func_start - 1:func_end]
func_source = "\n".join(func_lines)

# Find the gemm_a4w4 call and check if weight has .view() applied
# The bug: weight passed directly to gemm_a4w4 without dtype conversion
# The fix: weight.view(x_q.dtype) or similar dtype reinterpretation

# Look for gemm_a4w4 call context
gemm_a4w4_match = re.search(
    r'gemm_a4w4\s*\((.*?)\)',
    func_source,
    re.DOTALL
)

if gemm_a4w4_match:
    call_args = gemm_a4w4_match.group(1)
    # Check if weight argument has dtype conversion applied
    # In the call, weight is the second positional arg after x_q
    # Valid conversion patterns: weight.view(...), weight.to(...), weight.type(...)
    has_weight_conversion = bool(re.search(
        r'weight\s*\.\s*(?:view|to|type)\s*\(',
        call_args
    ))

    check("Weight dtype converted in gemm_a4w4 call",
          has_weight_conversion,
          "weight passed to gemm_a4w4 without dtype conversion")
else:
    # gemm_a4w4 not called in this function — check for alternative GEMM paths
    has_any_gemm = "gemm_a" in func_source.lower()
    check("Weight dtype converted in gemm_a4w4 call", False,
          "gemm_a4w4 call not found in function" if not has_any_gemm
          else "gemm call found but could not parse arguments")

# --- Summary ---
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
