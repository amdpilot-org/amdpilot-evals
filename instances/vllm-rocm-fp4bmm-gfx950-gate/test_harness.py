#!/usr/bin/env python3
"""Test harness for vLLM issue #34641 / PR #35250: FP4BMM gfx950 hardware gate.

Bug: VLLM_ROCM_USE_AITER_FP4BMM defaults to True, which crashes on MI300X
(gfx942) because FP4 instructions are gfx950-only (MI325X/MI350X/MI355X).

The fix adds an on_gfx950() hardware check to:
  1. is_fp4bmm_enabled()
  2. is_asm_fp4_gemm_dynamic_quant_enabled()

Tests (AST-based, no GPU required):
  1. is_fp4bmm_enabled() must call on_gfx950()
  2. is_asm_fp4_gemm_dynamic_quant_enabled() must call on_gfx950()
  3. on_gfx950 must be imported from vllm.platforms.rocm
  4. Both functions must still check _AITER_ENABLED (no over-removal)
  5. Both functions must still check their respective env var flags
"""
import ast
import os
import sys

checks_passed = 0
checks_total = 0

AITER_OPS_PATH = "/workspace/vllm/vllm/_aiter_ops.py"
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


print("=" * 60)
print("vllm-rocm-fp4bmm-gfx950-gate test harness (issue #34641)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Pre-check: target file exists and parses
# ---------------------------------------------------------------------------
if not check("_aiter_ops.py exists", os.path.isfile(AITER_OPS_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

try:
    with open(AITER_OPS_PATH) as fh:
        source = fh.read()
    tree = ast.parse(source)
    check("_aiter_ops.py is valid Python", True)
except SyntaxError as e:
    check("_aiter_ops.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: find a function or method by name in the AST
# ---------------------------------------------------------------------------
def find_function(tree, name):
    """Find a FunctionDef node by name anywhere in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def get_function_source(fn_node):
    """Extract the source lines for a function node."""
    lines = source.splitlines()[fn_node.lineno - 1 : fn_node.end_lineno]
    return "\n".join(lines)


def function_references(fn_node, name):
    """Check if a function body references a given name (attribute or identifier)."""
    for child in ast.walk(fn_node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


# =========================================================================
# CHECK 1: is_fp4bmm_enabled() must gate on on_gfx950()
# =========================================================================
print("\n--- Check 1: is_fp4bmm_enabled() hardware gate ---")

fp4bmm_fn = find_function(tree, "is_fp4bmm_enabled")

if not check(
    "is_fp4bmm_enabled function found",
    fp4bmm_fn is not None,
    "function not found in _aiter_ops.py",
):
    pass
else:
    fp4bmm_source = get_function_source(fp4bmm_fn)

    # Must reference on_gfx950
    has_gfx950_check = "on_gfx950" in fp4bmm_source
    check(
        "is_fp4bmm_enabled checks on_gfx950()",
        has_gfx950_check,
        "missing hardware gate -- FP4 will crash on MI300X (gfx942)",
    )

    # Must still check _AITER_ENABLED or _FP4BMM_ENABLED (not over-removed)
    has_aiter_check = (
        "_AITER_ENABLED" in fp4bmm_source or "_FP4BMM_ENABLED" in fp4bmm_source
    )
    check(
        "is_fp4bmm_enabled still checks AITER/FP4BMM enabled flags",
        has_aiter_check,
        "function should still verify AITER is enabled and FP4BMM env var is set",
    )


# =========================================================================
# CHECK 2: is_asm_fp4_gemm_dynamic_quant_enabled() must gate on on_gfx950()
# =========================================================================
print("\n--- Check 2: is_asm_fp4_gemm_dynamic_quant_enabled() hardware gate ---")

fp4_asm_fn = find_function(tree, "is_asm_fp4_gemm_dynamic_quant_enabled")

if not check(
    "is_asm_fp4_gemm_dynamic_quant_enabled function found",
    fp4_asm_fn is not None,
    "function not found in _aiter_ops.py",
):
    pass
else:
    fp4_asm_source = get_function_source(fp4_asm_fn)

    # Must reference on_gfx950
    has_gfx950_check = "on_gfx950" in fp4_asm_source
    check(
        "is_asm_fp4_gemm_dynamic_quant_enabled checks on_gfx950()",
        has_gfx950_check,
        "missing hardware gate -- FP4 ASM GEMM will crash on MI300X (gfx942)",
    )

    # Must still check _AITER_ENABLED
    has_aiter_check = "_AITER_ENABLED" in fp4_asm_source
    check(
        "is_asm_fp4_gemm_dynamic_quant_enabled still checks AITER enabled flag",
        has_aiter_check,
        "function should still verify AITER is enabled",
    )


# =========================================================================
# CHECK 3: on_gfx950 is imported from vllm.platforms.rocm
# =========================================================================
print("\n--- Check 3: on_gfx950 import ---")

# Check for import of on_gfx950 anywhere in the file.
# It could be a module-level import or a lazy import inside the functions.
# Either is acceptable.
has_on_gfx950_import = False

# Check module-level imports
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        if node.module and "rocm" in node.module:
            for alias in node.names:
                if alias.name == "on_gfx950":
                    has_on_gfx950_import = True
                    break

# Also check inline/lazy imports inside function bodies
if not has_on_gfx950_import:
    # Check if "from vllm.platforms.rocm import on_gfx950" appears in source
    has_on_gfx950_import = "from vllm.platforms.rocm import on_gfx950" in source

check(
    "on_gfx950 imported from vllm.platforms.rocm",
    has_on_gfx950_import,
    "on_gfx950 not found -- fix should import it from vllm.platforms.rocm",
)

# Verify on_gfx950 is used consistently with the existing codebase pattern
# (other gfx950-only features also use this function)
check(
    "on_gfx950 referenced in source (used for gating)",
    "on_gfx950" in source,
    "on_gfx950 not referenced anywhere in _aiter_ops.py",
)


# =========================================================================
# CHECK 4: Return statement structure
#
# The canonical fix pattern is:
#   return cls._AITER_ENABLED and cls._FP4BMM_ENABLED and on_gfx950()
#
# We verify the return value includes all three conditions.
# =========================================================================
print("\n--- Check 4: Return statement includes all conditions ---")

if fp4bmm_fn is not None:
    # Find return statements in is_fp4bmm_enabled
    return_nodes = [
        n for n in ast.walk(fp4bmm_fn) if isinstance(n, ast.Return)
    ]
    check(
        "is_fp4bmm_enabled has a return statement",
        len(return_nodes) > 0,
        "no return statement found",
    )

    if return_nodes:
        # Get the return source
        ret_node = return_nodes[0]
        ret_line = source.splitlines()[ret_node.lineno - 1].strip()

        # All three conditions should be in the return
        has_all_conditions = (
            "on_gfx950" in ret_line
            and ("_FP4BMM_ENABLED" in ret_line or "_AITER_ENABLED" in ret_line)
        )
        check(
            "is_fp4bmm_enabled return combines hw gate with env var check",
            has_all_conditions,
            f"return statement: {ret_line}",
        )

if fp4_asm_fn is not None:
    return_nodes = [
        n for n in ast.walk(fp4_asm_fn) if isinstance(n, ast.Return)
    ]
    check(
        "is_asm_fp4_gemm_dynamic_quant_enabled has a return statement",
        len(return_nodes) > 0,
        "no return statement found",
    )

    if return_nodes:
        ret_node = return_nodes[0]
        ret_line = source.splitlines()[ret_node.lineno - 1].strip()
        has_all_conditions = (
            "on_gfx950" in ret_line and "_AITER_ENABLED" in ret_line
        )
        check(
            "is_asm_fp4_gemm_dynamic_quant_enabled return combines hw gate with env var check",
            has_all_conditions,
            f"return statement: {ret_line}",
        )


# =========================================================================
# CHECK 5: No other FP4-related functions missing the gate
#
# Scan for any other method that references _FP4BMM_ENABLED or
# FP4_GEMM_DYNAMIC_QUANT without also referencing on_gfx950.
# =========================================================================
print("\n--- Check 5: No other FP4 functions missing hardware gate ---")

# Only flag functions that *read* FP4 flags for gating decisions (is_* pattern).
# Config/init functions that merely *assign* these flags (e.g. refresh_env_variables,
# __init__) should NOT need a hardware gate.
GATE_CHECK_EXCLUDE = {"refresh_env_variables", "__init__", "__init_subclass__"}

ungated_fp4_functions = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name not in GATE_CHECK_EXCLUDE:
        fn_src = get_function_source(node)
        has_fp4_ref = (
            "_FP4BMM_ENABLED" in fn_src
            or "_FP4_GEMM_DYNAMIC_QUANT" in fn_src
        )
        has_hw_gate = "on_gfx950" in fn_src
        if has_fp4_ref and not has_hw_gate:
            ungated_fp4_functions.append(node.name)

check(
    "All FP4-related functions have on_gfx950() hardware gate",
    len(ungated_fp4_functions) == 0,
    f"ungated functions: {ungated_fp4_functions}",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
