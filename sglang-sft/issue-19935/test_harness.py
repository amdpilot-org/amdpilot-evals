#!/usr/bin/env python3
"""Verification harness for SGLang #19935 — FP8 MLA decode k_scale fix.

Uses AST analysis to verify the fix was applied correctly at all
mla_decode_fwd call sites. Checks that kv_scale arguments are not
the raw `layer.k_scale` attribute (which can be None), but instead
use a fallback or conditional expression.
"""

import ast
import sys

AITER_BACKEND = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"


def _is_raw_layer_k_scale(node):
    """Check if an AST node is exactly `layer.k_scale` (the buggy pattern)."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "k_scale"
        and isinstance(node.value, ast.Name)
        and node.value.id == "layer"
    )


def _has_self_k_scale_ref(node):
    """Check if an AST node references `self.k_scale` anywhere."""
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Attribute)
            and child.attr == "k_scale"
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
        ):
            return True
    return False


def _find_mla_decode_calls(tree):
    """Find all calls to mla_decode_fwd in the AST."""
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
        if name == "mla_decode_fwd":
            calls.append(node)
    return calls


def _check_call_site(call):
    """Check if a mla_decode_fwd call has proper kv_scale handling.

    Returns (q_ok, kv_ok) — True if the argument is NOT raw layer.k_scale.
    """
    q_ok = True
    kv_ok = True
    for kw in call.keywords:
        if kw.arg == "q_scale" and _is_raw_layer_k_scale(kw.value):
            q_ok = False
        if kw.arg == "kv_scale" and _is_raw_layer_k_scale(kw.value):
            kv_ok = False
    return q_ok, kv_ok


def _check_surrounding_scope_has_fallback(source_lines, call_lineno):
    """Check that within ~80 lines before the call, self.k_scale is referenced
    in a conditional or assignment context (not just in a comment)."""
    start = max(0, call_lineno - 80)
    end = call_lineno
    region = source_lines[start:end]
    for line in region:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "self.k_scale" in stripped:
            return True
    return False


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

    calls = _find_mla_decode_calls(tree)
    if not calls:
        print("ERROR: No mla_decode_fwd calls found")
        return 0

    source_lines = source.splitlines()
    total = len(calls)
    fixed = 0

    for i, call in enumerate(calls):
        q_ok, kv_ok = _check_call_site(call)
        has_fallback = _check_surrounding_scope_has_fallback(source_lines, call.lineno)

        site_ok = (q_ok and kv_ok) or has_fallback
        if site_ok:
            fixed += 1
            print(f"  SITE {i+1} (line {call.lineno}): FIXED")
        else:
            detail = []
            if not q_ok:
                detail.append("q_scale=layer.k_scale (raw)")
            if not kv_ok:
                detail.append("kv_scale=layer.k_scale (raw)")
            if not has_fallback:
                detail.append("no self.k_scale fallback nearby")
            print(f"  SITE {i+1} (line {call.lineno}): UNFIXED — {', '.join(detail)}")

    print(f"\nmla_decode_fwd sites: {fixed}/{total} fixed")

    if total < 4:
        print(f"WARNING: Expected >= 4 call sites, found {total}")

    if fixed == total and total >= 4:
        return 100
    elif fixed == total and total > 0:
        return 80
    elif fixed > 0:
        return int(25 * fixed)
    return 0


def main():
    print("=== Verification: FP8 MLA decode k_scale fallback ===\n")
    score = verify()
    print(f"\nSCORE: {score}")


if __name__ == "__main__":
    main()
