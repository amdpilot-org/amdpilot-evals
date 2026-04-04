#!/usr/bin/env python3
"""Test harness for vllm-mxfp4-moe-fallback (PR #35893).

Bug: CK MXFP4 MoE GEMM kernels crash with RuntimeError when intermediate_size
per partition is not a multiple of 256 (e.g. MiniMax-M2.1 TP=4 → 384).
Test: Verify dimension validation and fallback logic exist in the quantization
code to handle incompatible dimensions gracefully.
"""
import ast
import sys
import subprocess
from pathlib import Path

checks_passed = 0
checks_total = 0

VLLM_PATH = "/workspace/vllm"
MXFP4_UTILS_PATH = f"{VLLM_PATH}/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py"
MXFP4_PATH = f"{VLLM_PATH}/vllm/model_executor/layers/quantization/mxfp4.py"
QUARK_MOE_PATH = f"{VLLM_PATH}/vllm/model_executor/layers/quantization/quark/quark_moe.py"


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


def _contains_name(node, name):
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


print("=" * 60)
print("vllm-mxfp4-moe-fallback test harness")
print("=" * 60)

# Check 1: CK_MXFP4_MOE_DIM_ALIGNMENT constant exists with value 256 in mxfp4_utils.py
# Uses AST to avoid circular import issues
if not Path(MXFP4_UTILS_PATH).is_file():
    check("mxfp4_utils.py exists", False, "file not found")
    check("CK_MXFP4_MOE_DIM_ALIGNMENT = 256", False, "file missing")
else:
    check("mxfp4_utils.py exists", True)
    utils_src = Path(MXFP4_UTILS_PATH).read_text()
    utils_tree = ast.parse(utils_src)
    found_const = False
    const_value = None
    for node in ast.walk(utils_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CK_MXFP4_MOE_DIM_ALIGNMENT":
                    found_const = True
                    if isinstance(node.value, ast.Constant):
                        const_value = node.value.value
    check("CK_MXFP4_MOE_DIM_ALIGNMENT = 256",
          found_const and const_value == 256,
          f"found={found_const}, value={const_value}")

# Check 2: mxfp4.py imports the constant and uses it for modulo alignment check
if not Path(MXFP4_PATH).is_file():
    check("mxfp4.py imports CK_MXFP4_MOE_DIM_ALIGNMENT and uses modulo check", False, "file not found")
else:
    mxfp4_src = Path(MXFP4_PATH).read_text()
    mxfp4_tree = ast.parse(mxfp4_src)
    has_import = False
    for node in ast.iter_child_nodes(mxfp4_tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            for alias in node.names:
                if alias.name == "CK_MXFP4_MOE_DIM_ALIGNMENT":
                    has_import = True
    has_modulo = False
    for node in ast.walk(mxfp4_tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if _contains_name(node, "CK_MXFP4_MOE_DIM_ALIGNMENT"):
                has_modulo = True
    check("mxfp4.py imports CK_MXFP4_MOE_DIM_ALIGNMENT and uses modulo check",
          has_import and has_modulo,
          f"import={has_import}, modulo={has_modulo}")

# Check 3: mxfp4.py has Triton fallback when CK alignment fails
if Path(MXFP4_PATH).is_file():
    has_triton_fallback = False
    for node in ast.walk(mxfp4_tree):
        if isinstance(node, ast.Attribute) and node.attr == "TRITON":
            has_triton_fallback = True
    check("mxfp4.py has Triton backend fallback",
          has_triton_fallback,
          "no Mxfp4Backend.TRITON reference found")
else:
    check("mxfp4.py has Triton backend fallback", False, "file not found")

# Check 4: quark_moe.py validates dims and falls back to emulation mode
if not Path(QUARK_MOE_PATH).is_file():
    check("quark_moe.py validates dims and falls back to emulation", False, "file not found")
else:
    quark_src = Path(QUARK_MOE_PATH).read_text()
    quark_tree = ast.parse(quark_src)
    quark_has_import = False
    for node in ast.iter_child_nodes(quark_tree):
        if isinstance(node, ast.ImportFrom) and node.names:
            for alias in node.names:
                if alias.name == "CK_MXFP4_MOE_DIM_ALIGNMENT":
                    quark_has_import = True
    quark_has_modulo = False
    for node in ast.walk(quark_tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if _contains_name(node, "CK_MXFP4_MOE_DIM_ALIGNMENT"):
                quark_has_modulo = True
    quark_has_emulate = False
    for node in ast.walk(quark_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "emulate":
                    if isinstance(node.value, ast.Constant) and node.value.value is True:
                        quark_has_emulate = True
    check("quark_moe.py validates dims and falls back to emulation",
          quark_has_import and quark_has_modulo and quark_has_emulate,
          f"import={quark_has_import}, modulo={quark_has_modulo}, emulate={quark_has_emulate}")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
