#!/usr/bin/env python3
"""Test harness for sglang-tp4-accuracy-cross-repo.

Verify that the aiter attention backend handles TP=4 (32-head)
configurations correctly for MLA decode.
"""

import ast
import os
import re
import sys

SGLANG_ROOT = os.environ.get("SGLANG_ROOT", "/sgl-workspace/sglang")
AITER_BACKEND = os.path.join(
    SGLANG_ROOT,
    "python", "sglang", "srt", "layers", "attention", "aiter_backend.py"
)
ROCM_DOCKERFILE = os.path.join(SGLANG_ROOT, "docker", "rocm.Dockerfile")

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}: {detail}")


# ---------------------------------------------------------------------------
# Test 1: num_head == 32 handling
# ---------------------------------------------------------------------------

def check_num_head_32_handling():
    """Check that __init__ handles num_head == 32 (TP=4 case)."""
    if not os.path.exists(AITER_BACKEND):
        check("num_head_32", False, "aiter_backend.py not found")
        return

    with open(AITER_BACKEND) as f:
        source = f.read()

    # The fix adds a branch for num_head == 32 that sets fast_mode and
    # intra_batch_mode appropriately
    has_32_check = "num_head == 32" in source or "self.num_head == 32" in source

    check("num_head_32_check_exists",
          has_32_check,
          "No num_head == 32 branch found -- TP=4 not handled")

    if has_32_check:
        # Check that it sets fast_mode = True
        lines = source.split("\n")
        in_32_block = False
        found_fast_mode = False
        found_intra_batch = False

        for i, line in enumerate(lines):
            if "num_head == 32" in line:
                in_32_block = True
                block_indent = len(line) - len(line.lstrip())
                continue
            if in_32_block:
                line_indent = len(line) - len(line.lstrip()) if line.strip() else block_indent + 1
                if line.strip() and line_indent <= block_indent:
                    in_32_block = False
                    continue
                if "fast_mode" in line and ("True" in line or "= True" in line):
                    found_fast_mode = True
                if "intra_batch_mode" in line and ("False" in line or "= False" in line):
                    found_intra_batch = True

        check("num_head_32_fast_mode",
              found_fast_mode,
              "num_head==32 block should set fast_mode = True")

        check("num_head_32_intra_batch",
              found_intra_batch,
              "num_head==32 block should set intra_batch_mode = False")


# ---------------------------------------------------------------------------
# Test 2: Non-persist fallback scoped to num_head == 16
# ---------------------------------------------------------------------------

def check_non_persist_fallback_scoped():
    """
    The non-persist fallback (setting _use_mla_ps_kernel = False) should be
    scoped to num_head == 16, not applied broadly to all non-FP8 cases.
    """
    if not os.path.exists(AITER_BACKEND):
        check("non_persist_scoped", False, "aiter_backend.py not found")
        return

    with open(AITER_BACKEND) as f:
        source = f.read()

    # Look for the pattern: if self.num_head == 16 and ... not fp8
    # The buggy version doesn't have the num_head == 16 guard.
    # Use a single-line regex (no DOTALL) to match the condition on one line.
    has_16_guard = re.search(
        r"if\s+.*num_head\s*==\s*16.*kv_cache_dtype|if\s+.*kv_cache_dtype.*num_head\s*==\s*16",
        source,
    )

    if has_16_guard:
        # Check that _use_mla_ps_kernel = False follows within a few lines
        match_pos = has_16_guard.start()
        nearby = source[match_pos:match_pos + 400]
        has_disable = "_use_mla_ps_kernel" in nearby and "False" in nearby
        check("non_persist_scoped_to_16",
              has_disable,
              "Found num_head==16 guard but _use_mla_ps_kernel disable not nearby")
    else:
        # Alternative: check that the disable line's guarding if-statement
        # contains num_head == 16 (not just the string "16")
        lines = source.split("\n")
        found_guarded = False
        for i, line in enumerate(lines):
            if "_use_mla_ps_kernel" in line and "False" in line and "=" in line:
                # Look backwards for the if-condition
                for j in range(i - 1, max(0, i - 5), -1):
                    stripped = lines[j].strip()
                    if stripped.startswith("if ") and "num_head" in stripped:
                        found_guarded = True
                        break
                    if stripped.startswith("if "):
                        break

        check("non_persist_scoped_to_16",
              found_guarded,
              "Non-persist fallback not scoped to num_head==16")


# ---------------------------------------------------------------------------
# Test 3: get_mla_metadata_v1 boolean argument
# ---------------------------------------------------------------------------

def check_metadata_boolean():
    """
    The get_mla_metadata_v1 call in make_mla_meta_data should pass False
    (not True) as the boolean argument after nhead_kv.
    """
    if not os.path.exists(AITER_BACKEND):
        check("metadata_boolean", False, "aiter_backend.py not found")
        return

    with open(AITER_BACKEND) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=AITER_BACKEND)
    except SyntaxError:
        check("metadata_boolean", False, "Failed to parse aiter_backend.py")
        return

    # Find make_mla_meta_data function
    class MetadataCallFinder(ast.NodeVisitor):
        def __init__(self):
            self.found_call = False
            self.bool_arg_value = None

        def visit_Call(self, node):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name == "get_mla_metadata_v1":
                self.found_call = True
                # The boolean arg is the 6th positional argument (index 5)
                if len(node.args) > 5:
                    arg = node.args[5]
                    if isinstance(arg, ast.Constant):
                        self.bool_arg_value = arg.value

            self.generic_visit(node)

    finder = MetadataCallFinder()
    finder.visit(tree)

    if not finder.found_call:
        check("metadata_boolean_call_found", False,
              "get_mla_metadata_v1 call not found")
        return

    check("metadata_boolean_call_found", True)

    if finder.bool_arg_value is not None:
        check("metadata_boolean_is_false",
              finder.bool_arg_value == False,
              f"6th arg to get_mla_metadata_v1 is {finder.bool_arg_value}, expected False")
    else:
        # Might be a variable, not a literal -- check source text
        meta_call = re.search(
            r"get_mla_metadata_v1\s*\([^)]*\)",
            source, re.DOTALL
        )
        if meta_call:
            call_text = meta_call.group(0)
            # Split args
            args = call_text.split(",")
            if len(args) > 5:
                sixth_arg = args[5].strip()
                check("metadata_boolean_is_false",
                      "False" in sixth_arg,
                      f"6th arg is '{sixth_arg}', expected False")
            else:
                check("metadata_boolean_is_false", False,
                      "Could not extract 6th argument")
        else:
            check("metadata_boolean_is_false", False,
                  "Could not find get_mla_metadata_v1 call text")


# ---------------------------------------------------------------------------
# Test 4: aiter version pin in Dockerfile
# ---------------------------------------------------------------------------

def check_aiter_version():
    """
    The rocm.Dockerfile should pin aiter to a version >= v0.1.10.post3
    (the version that includes the TP=4 fix).
    """
    if not os.path.exists(ROCM_DOCKERFILE):
        check("aiter_version", False, "rocm.Dockerfile not found")
        return

    with open(ROCM_DOCKERFILE) as f:
        content = f.read()

    # Look for AITER_COMMIT version string
    version_match = re.search(r'AITER_COMMIT\s*=\s*"([^"]*)"', content)
    if not version_match:
        version_match = re.search(r"AITER_COMMIT\s*=\s*'([^']*)'", content)

    if not version_match:
        check("aiter_version", False, "No AITER_COMMIT found in rocm.Dockerfile")
        return

    version = version_match.group(1)

    # The buggy version is v0.1.10.post2, fix needs >= v0.1.10.post3
    is_fixed = version != "v0.1.10.post2"

    check("aiter_version_updated",
          is_fixed,
          f"AITER_COMMIT is '{version}', should be updated from v0.1.10.post2")


# ---------------------------------------------------------------------------
# Test 5: Overall code quality -- use_mla guard presence
# ---------------------------------------------------------------------------

def check_use_mla_in_init():
    """
    The __init__ method should have self.use_mla checks gating the
    MLA-specific kernel configuration (fast_mode, intra_batch_mode, etc).
    """
    if not os.path.exists(AITER_BACKEND):
        check("use_mla_init", False, "aiter_backend.py not found")
        return

    with open(AITER_BACKEND) as f:
        source = f.read()

    # Check that use_mla is referenced in __init__
    has_use_mla = "self.use_mla" in source

    check("use_mla_init_present",
          has_use_mla,
          "self.use_mla not found in aiter_backend.py")

    # Check that the MLA kernel config is inside a use_mla block
    # (fast_mode, intra_batch_mode, max_split_per_batch should be under use_mla)
    mla_config_keywords = ["fast_mode", "intra_batch_mode", "max_split_per_batch"]
    found_under_mla = 0

    lines = source.split("\n")
    in_mla_block = False
    mla_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if "if self.use_mla" in line or "if self.use_mla:" in line:
            in_mla_block = True
            mla_indent = indent
            continue

        if in_mla_block:
            if stripped and indent <= mla_indent:
                in_mla_block = False
            elif any(kw in line for kw in mla_config_keywords):
                found_under_mla += 1

    check("mla_config_under_guard",
          found_under_mla >= 2,
          f"Found {found_under_mla} MLA config keywords under use_mla guard (expect >= 2)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SGLang TP=4 Accuracy Cross-Repo Fix Test")
    print("=" * 60)

    print("\n--- num_head == 32 Handling ---")
    check_num_head_32_handling()

    print("\n--- Non-Persist Fallback Scoping ---")
    check_non_persist_fallback_scoped()

    print("\n--- get_mla_metadata_v1 Boolean ---")
    check_metadata_boolean()

    print("\n--- aiter Version Pin ---")
    check_aiter_version()

    print("\n--- use_mla Init Guard ---")
    check_use_mla_in_init()

    print(f"\n--- Results ---")
    print(f"  {checks_passed}/{checks_total} checks passed")

    score = checks_passed / checks_total * 100.0 if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
