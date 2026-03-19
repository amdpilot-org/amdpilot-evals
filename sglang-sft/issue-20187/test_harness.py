#!/usr/bin/env python3
"""Verification harness for SGLang #20187 — FP8 prefill + radix cache.

Uses AST analysis and source inspection to verify all 6 structural
changes required for integrating FP8 prefill into the radix-cache path.
"""

import ast
import sys

AITER_BACKEND = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"


def _find_class_methods(tree, class_name):
    """Return {method_name: method_node} for a class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name: item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return {}


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


def _find_function_calls(tree, func_name):
    """Find all AST Call nodes for a function by name."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == func_name:
                calls.append(node)
    return calls


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
        print(f"ERROR: Syntax error: {e}")
        return 0

    source_lines = source.splitlines()
    code_no_comments = "\n".join(
        line for line in source_lines if not line.strip().startswith("#")
    )
    methods = _find_class_methods(tree, "AiterAttnBackend")
    checks_passed = 0
    total_checks = 6

    # ── CHECK 1: mla_fp8_prefill_attn helper method exists ──
    # The human PR extracts shared FP8 prefill logic into a reusable method.
    # Without it, the code is duplicated (bad) or missing from the radix path.
    if "mla_fp8_prefill_attn" in methods:
        helper = methods["mla_fp8_prefill_attn"]
        has_mla_prefill = _method_contains_call(helper, "mla_prefill_ps_asm_fwd")
        has_reduce = _method_contains_call(helper, "mla_reduce_v1")
        if has_mla_prefill and has_reduce:
            checks_passed += 1
            print("CHECK 1 PASS: mla_fp8_prefill_attn helper method exists with correct kernel calls")
        else:
            print("CHECK 1 PARTIAL: mla_fp8_prefill_attn exists but missing kernel calls")
    else:
        print("CHECK 1 FAIL: mla_fp8_prefill_attn helper method not found")

    # ── CHECK 2: init_forward_metadata uses kv_indptr (not qo_indptr) ──
    # The buggy code passes qo_indptr as the kv_indptr argument to
    # make_mla_prefill_ps_meta_data. The fix assigns kv_indptr from the
    # mla_indices_updater and passes it instead.
    init_meta = methods.get("init_forward_metadata")
    if init_meta:
        init_lines = source_lines[init_meta.lineno - 1 : (init_meta.end_lineno or init_meta.lineno + 200)]
        init_text = "\n".join(init_lines)

        has_kv_indptr_assign = any(
            "kv_indptr" in line and "mla_indices_updater" in line
            and "=" in line.split("mla_indices_updater")[0]
            for line in init_lines if not line.strip().startswith("#")
        )
        buggy_double_qo = False
        for i, line in enumerate(init_lines):
            if "make_mla_prefill_ps_meta_data" in line and "buffer" not in line:
                nearby = "\n".join(init_lines[i:i+5])
                args = nearby.count("qo_indptr")
                if args >= 2:
                    buggy_double_qo = True

        if has_kv_indptr_assign and not buggy_double_qo:
            checks_passed += 1
            print("CHECK 2 PASS: init_forward_metadata uses kv_indptr from mla_indices_updater")
        elif has_kv_indptr_assign:
            print("CHECK 2 PARTIAL: kv_indptr assigned but still passed as qo_indptr to make_mla_prefill_ps_meta_data")
        else:
            print("CHECK 2 FAIL: init_forward_metadata does not assign kv_indptr from mla_indices_updater")
    else:
        print("CHECK 2 FAIL: init_forward_metadata method not found")

    # ── CHECK 3: total_s uses seq_lens_sum (not extend_seq_lens.sum()) ──
    # The buggy code computes total_s = int(forward_batch.extend_seq_lens.sum())
    # which only covers new tokens. The fix uses forward_batch.seq_lens_sum
    # which includes prefix tokens.
    if init_meta:
        has_buggy_total_s = any(
            "extend_seq_lens" in line and "total_s" in line
            for line in init_lines if not line.strip().startswith("#")
        )
        has_fixed_total_s = any(
            "seq_lens_sum" in line and "total_s" in line
            for line in init_lines if not line.strip().startswith("#")
        )
        if has_fixed_total_s and not has_buggy_total_s:
            checks_passed += 1
            print("CHECK 3 PASS: total_s uses forward_batch.seq_lens_sum")
        elif has_fixed_total_s:
            print("CHECK 3 PARTIAL: seq_lens_sum present but extend_seq_lens.sum still exists")
        else:
            print("CHECK 3 FAIL: total_s still uses extend_seq_lens.sum() instead of seq_lens_sum")
    else:
        print("CHECK 3 FAIL: init_forward_metadata not found")

    # ── CHECK 4: fused GEMM via kv_b_proj tuple-dispatch or direct call ──
    # The human PR uses layer.kv_b_proj((kvc, k_pe, dim1, dim2, fp8_dtype))
    # which triggers fused_gemm_afp4wfp4_split_cat inside the quantization
    # scheme. Check for either a direct call OR the tuple-dispatch pattern
    # (kv_b_proj called with a Tuple argument containing fp8_dtype).
    fused_calls = _find_function_calls(tree, "fused_gemm_afp4wfp4_split_cat")
    if fused_calls:
        checks_passed += 1
        print(f"CHECK 4 PASS: fused_gemm_afp4wfp4_split_cat called directly ({len(fused_calls)} site(s))")
    else:
        fwd_extend = methods.get("forward_extend")
        has_tuple_dispatch = False
        if fwd_extend:
            fwd_lines = source_lines[fwd_extend.lineno - 1 : (fwd_extend.end_lineno or fwd_extend.lineno + 500)]
            fwd_text = "\n".join(l for l in fwd_lines if not l.strip().startswith("#"))
            has_tuple_dispatch = (
                "kv_b_proj" in fwd_text
                and "fp8_dtype" in fwd_text
                and ("squeeze" in fwd_text or "expand" in fwd_text)
                and "torch.uint8" in fwd_text
            )
        if has_tuple_dispatch:
            checks_passed += 1
            print("CHECK 4 PASS: kv_b_proj tuple-dispatch with fp8_dtype found (triggers fused_gemm)")
        elif fwd_extend and any("fused_gemm" in l for l in fwd_lines if not l.strip().startswith("#")):
            print("CHECK 4 PARTIAL: fused_gemm reference but not in correct dispatch pattern")
        else:
            print("CHECK 4 FAIL: no fused GEMM dispatch (neither direct call nor kv_b_proj tuple pattern)")

    # ── CHECK 5: forward_extend radix-cache branch has FP8 dispatch ──
    # The radix-cache branch is the `elif layer.qk_head_dim != ...` block.
    # It must dispatch to FP8 prefill attention (mla_fp8_prefill_attn or similar).
    fwd_extend = methods.get("forward_extend")
    if fwd_extend:
        fwd_lines = source_lines[fwd_extend.lineno - 1 : (fwd_extend.end_lineno or fwd_extend.lineno + 500)]
        in_radix_branch = False
        fp8_dispatch_in_radix = False
        for line in fwd_lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "qk_head_dim" in stripped and "elif" in stripped:
                in_radix_branch = True
                continue
            if in_radix_branch:
                if any(kw in stripped for kw in [
                    "mla_fp8_prefill_attn", "mla_prefill_ps_asm_fwd",
                    "_use_fp8_prefill_attn",
                ]):
                    fp8_dispatch_in_radix = True
                    break
                if stripped.startswith("def ") or (stripped.startswith("elif ") and "qk_head_dim" not in stripped):
                    break

        if fp8_dispatch_in_radix:
            checks_passed += 1
            print("CHECK 5 PASS: radix-cache branch dispatches to FP8 prefill")
        else:
            print("CHECK 5 FAIL: radix-cache branch does not dispatch to FP8 prefill")
    else:
        print("CHECK 5 FAIL: forward_extend method not found")

    # ── CHECK 6: no-prefix path calls the helper (not inline FP8 code) ──
    # The human PR refactors the no-prefix path to use mla_fp8_prefill_attn
    # instead of inlining the FP8 kernel calls. Check that the no-prefix FP8
    # path is clean (calls the helper, not inlining mla_prefill_ps_asm_fwd).
    if fwd_extend:
        fwd_lines = source_lines[fwd_extend.lineno - 1 : (fwd_extend.end_lineno or fwd_extend.lineno + 500)]
        in_no_prefix = False
        calls_helper = False
        inlines_kernel = False
        for line in fwd_lines:
            stripped = line.strip()
            if "extend_no_prefix" in stripped and ("if" in stripped) and "not" not in stripped:
                in_no_prefix = True
            if in_no_prefix:
                if "mla_fp8_prefill_attn" in stripped:
                    calls_helper = True
                if "mla_prefill_ps_asm_fwd" in stripped:
                    inlines_kernel = True
                if "extend_no_prefix" in stripped and ("else" in stripped or "elif" in stripped):
                    break
                if stripped.startswith("def "):
                    break

        if calls_helper and not inlines_kernel:
            checks_passed += 1
            print("CHECK 6 PASS: no-prefix path calls mla_fp8_prefill_attn helper (not inlined)")
        elif calls_helper:
            print("CHECK 6 PARTIAL: helper called but kernel also inlined")
            checks_passed += 1
        elif not inlines_kernel:
            print("CHECK 6 PASS: no inline FP8 kernel in no-prefix path (may use different structure)")
            checks_passed += 1
        else:
            print("CHECK 6 FAIL: no-prefix path still inlines mla_prefill_ps_asm_fwd instead of using helper")
    else:
        print("CHECK 6 FAIL: forward_extend not found")

    print(f"\nChecks passed: {checks_passed}/{total_checks}")
    return int(100 * checks_passed / total_checks)


def main():
    print("=== Verification: FP8 prefill + radix cache integration ===\n")
    score = verify()
    print(f"\nSCORE: {score}")


if __name__ == "__main__":
    main()
