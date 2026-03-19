#!/usr/bin/env python3
"""Verification harness for SGLang #20187 — FP8 prefill + radix cache.

Uses AST analysis and import checks to verify that FP8 prefill attention
was integrated into the radix-cache code path. Checks for actual function
calls (not just string presence) and code path structure.
"""

import ast
import sys

AITER_BACKEND = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"


def _find_function_calls(tree, func_name):
    """Find all call sites of a function by name in the AST."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == func_name:
            calls.append(node)
    return calls


def _find_class_methods(tree, class_name, method_name):
    """Find method definitions within a class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == method_name:
                        return item
    return None


def _method_contains_call(method_node, func_name):
    """Check if a method body contains a call to func_name."""
    for node in ast.walk(method_node):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == func_name:
                return True
    return False


def _method_has_fp8_dtype_usage(method_node):
    """Check if a method references fp8-related dtype variables/attributes."""
    for node in ast.walk(method_node):
        if isinstance(node, ast.Name) and "fp8" in node.id.lower():
            return True
        if isinstance(node, ast.Attribute) and "fp8" in node.attr.lower():
            return True
    return False


def _source_has_env_check(source, env_var):
    """Check the source for os.environ/getenv usage of a specific variable."""
    return env_var in source


def _find_radix_cache_branch(source_lines, method_node):
    """Check if within forward_extend there's a branch handling the
    radix-cache case (prefix tokens) that contains FP8 code."""
    if method_node is None:
        return False
    start = method_node.lineno - 1
    end = method_node.end_lineno if hasattr(method_node, 'end_lineno') and method_node.end_lineno else start + 500
    method_lines = source_lines[start:end]
    method_text = "\n".join(method_lines)

    in_else_branch = False
    fp8_in_branch = False
    for line in method_lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "extend_no_prefix" in stripped and ("else" in stripped or "elif" in stripped or "not " in stripped):
            in_else_branch = True
        if in_else_branch:
            if any(kw in stripped for kw in [
                "fp8_dtype", "_use_fp8_prefill", "mla_fp8", "fp8_prefill",
                "float8", "fused_gemm",
            ]):
                fp8_in_branch = True
                break
            if stripped.startswith("def ") or (stripped.startswith("elif ") and "extend_no_prefix" not in stripped):
                break

    return fp8_in_branch


def verify():
    try:
        with open(AITER_BACKEND) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"ERROR: {AITER_BACKEND} not found")
        return 0

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"ERROR: Syntax error in {AITER_BACKEND}: {e}")
        return 0

    source_lines = source.splitlines()
    checks_passed = 0
    total_checks = 4

    # CHECK 1: SGLANG_AITER_FP8_PREFILL_ATTN env var is referenced (not in a comment)
    code_no_comments = "\n".join(
        line for line in source_lines if not line.strip().startswith("#")
    )
    if "SGLANG_AITER_FP8_PREFILL_ATTN" in code_no_comments:
        checks_passed += 1
        print("CHECK 1 PASS: FP8 prefill env var referenced in code (not just comments)")
    else:
        print("CHECK 1 FAIL: SGLANG_AITER_FP8_PREFILL_ATTN not found in code")

    # CHECK 2: fused_gemm_afp4wfp4_split_cat is actually called (AST call node, not just string)
    fused_calls = _find_function_calls(tree, "fused_gemm_afp4wfp4_split_cat")
    if fused_calls:
        checks_passed += 1
        print(f"CHECK 2 PASS: fused_gemm_afp4wfp4_split_cat called ({len(fused_calls)} call site(s))")
    else:
        print("CHECK 2 FAIL: fused_gemm_afp4wfp4_split_cat not called (AST check)")

    # CHECK 3: forward_extend method contains FP8 dtype usage
    forward_extend = _find_class_methods(tree, "AiterAttnBackend", "forward_extend")
    if forward_extend and _method_has_fp8_dtype_usage(forward_extend):
        checks_passed += 1
        print("CHECK 3 PASS: forward_extend uses fp8 dtype references")
    else:
        if not forward_extend:
            print("CHECK 3 FAIL: forward_extend method not found")
        else:
            print("CHECK 3 FAIL: forward_extend has no fp8 dtype usage")

    # CHECK 4: radix-cache branch in forward_extend has FP8 code
    if _find_radix_cache_branch(source_lines, forward_extend):
        checks_passed += 1
        print("CHECK 4 PASS: FP8 code found in radix-cache (prefix) branch")
    else:
        print("CHECK 4 FAIL: No FP8 code in radix-cache branch of forward_extend")

    print(f"\nChecks passed: {checks_passed}/{total_checks}")

    return int(100 * checks_passed / total_checks)


def main():
    print("=== Verification: FP8 prefill + radix cache integration ===\n")
    score = verify()
    print(f"\nSCORE: {score}")


if __name__ == "__main__":
    main()
