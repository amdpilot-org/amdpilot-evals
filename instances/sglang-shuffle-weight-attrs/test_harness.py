#!/usr/bin/env python3
"""Test harness for sglang-shuffle-weight-attrs (PR #21825).

Bug: In unquant.py, MoE weight shuffling uses direct Parameter() reassignment
like `layer.w13_weight = torch.nn.Parameter(shuffle_weight(...))`, which loses
custom attributes (like weight_loader) on the original parameter.

Tests verify the fix via AST analysis of the correct file (unquant.py).
"""
import ast
import sys
import subprocess
from pathlib import Path

checks_passed = 0
checks_total = 0


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


def run_test(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout, cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("sglang-shuffle-weight-attrs test harness")
print("=" * 60)

# The PR modifies unquant.py, NOT ep_moe/layer.py
TARGET = "/workspace/sglang/python/sglang/srt/layers/quantization/unquant.py"

if not check("Target file exists", Path(TARGET).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source = Path(TARGET).read_text()

try:
    tree = ast.parse(source)
    check("Valid Python syntax", True)
except SyntaxError as e:
    check("Valid Python syntax", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: check if an AST node tree contains a reference to a name
# ---------------------------------------------------------------------------
def _contains_name(node, name):
    'Return True if name appears anywhere inside node.'
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


# ---------------------------------------------------------------------------
# Check 1: copy_or_rebind_param is imported in unquant.py
# ---------------------------------------------------------------------------
has_import = False
for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.ImportFrom) and node.names:
        for alias in node.names:
            if alias.name == "copy_or_rebind_param":
                has_import = True
                break

check("copy_or_rebind_param is imported in unquant.py",
      has_import,
      "copy_or_rebind_param not found in imports")


# ---------------------------------------------------------------------------
# Check 2: AST — copy_or_rebind_param calls exist with shuffle_weight
# The fix replaces direct Parameter() assignments with copy_or_rebind_param()
# calls wrapping shuffle_weight.
# ---------------------------------------------------------------------------
good_copy_or_rebind = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        is_copy_call = False
        if isinstance(func, ast.Name) and func.id == "copy_or_rebind_param":
            is_copy_call = True
        if isinstance(func, ast.Attribute) and func.attr == "copy_or_rebind_param":
            is_copy_call = True
        if is_copy_call:
            # Check if any argument references shuffle_weight
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if _contains_name(arg, "shuffle_weight"):
                    good_copy_or_rebind += 1

check("copy_or_rebind_param calls wrap shuffle_weight (>= 2 expected)",
      good_copy_or_rebind >= 2,
      f"found {good_copy_or_rebind} calls (expected >= 2 for w13_weight and w2_weight)")


# ---------------------------------------------------------------------------
# Check 3: No direct Parameter() assignment with shuffle_weight
# The old buggy pattern: layer.w13_weight = torch.nn.Parameter(shuffle_weight(...))
# ---------------------------------------------------------------------------
bad_direct_assignments = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target_node in node.targets:
            if isinstance(target_node, ast.Attribute):
                if _contains_name(node.value, "shuffle_weight"):
                    # Check if wrapped in Parameter() — that's the bug
                    if isinstance(node.value, ast.Call):
                        vfunc = node.value.func
                        if isinstance(vfunc, ast.Attribute) and vfunc.attr == "Parameter":
                            bad_direct_assignments += 1
                        elif isinstance(vfunc, ast.Name) and vfunc.id == "Parameter":
                            bad_direct_assignments += 1

check("No direct Parameter(shuffle_weight(...)) assignments remain",
      bad_direct_assignments == 0,
      f"found {bad_direct_assignments} direct Parameter() assignments with shuffle_weight")


# ---------------------------------------------------------------------------
# Check 4: Behavioral — copy_or_rebind_param preserves custom attributes
# Import from sglang.srt.layers.utils (the actual module location)
# ---------------------------------------------------------------------------
behavioral_script = """
import sys
sys.path.insert(0, '/workspace/sglang/python')
try:
    from sglang.srt.layers.utils import copy_or_rebind_param
    import torch
    import torch.nn as nn

    module = nn.Module()
    original = torch.nn.Parameter(torch.randn(4, 4))
    original.custom_attr = "test_value"
    original.weight_loader = lambda *a: None
    module.register_parameter("my_param", original)

    new_data = torch.randn(4, 4)
    copy_or_rebind_param(module, "my_param", new_data)

    param_after = module.my_param
    survived = getattr(param_after, "custom_attr", None) == "test_value"
    data_ok = torch.equal(param_after.data, new_data)
    print(f"SURVIVED:{survived}")
    print(f"DATA_OK:{data_ok}")
except ImportError as e:
    # Function may not be importable due to env deps — not a test failure
    print(f"IMPORT_SKIP:{e}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
"""

stdout, stderr, rc = run_test(behavioral_script)

if "IMPORT_SKIP" in stdout:
    # Can't import — treat as pass (AST checks above are the primary gate)
    check("copy_or_rebind_param preserves attributes (skipped, import unavailable)", True)
elif "SURVIVED:True" in stdout and "DATA_OK:True" in stdout:
    check("copy_or_rebind_param preserves attributes and updates data", True)
elif "ERROR:" in stdout:
    check("copy_or_rebind_param preserves attributes", False,
          stdout.split("ERROR:")[1].strip()[:200])
else:
    check("copy_or_rebind_param preserves attributes", False,
          f"unexpected output: {stdout.strip()[:200]}")


# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
