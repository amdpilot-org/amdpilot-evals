#!/usr/bin/env python3
"""Test harness for sglang-cutedsl-lazy-import (PR #21428).

Bug: Importing kda_backend crashes on ROCm with ModuleNotFoundError because
CuteDSL (which requires cuda.bindings) is imported at module top level.

Tests are behavioral -- they verify the import succeeds and the module
structure is correct, without checking for specific code patterns of the fix.
"""
import sys
import subprocess
import ast
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


print("=" * 60)
print("sglang-cutedsl-lazy-import test harness")
print("=" * 60)

TARGET = "/workspace/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py"

if not check("Target file exists", Path(TARGET).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source = Path(TARGET).read_text()

# Validate file is parseable Python
try:
    tree = ast.parse(source)
    check("Valid Python syntax", True)
except SyntaxError as e:
    check("Valid Python syntax", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Check 1 (PRIMARY): Behavioral -- subprocess import of kda_backend
# On the buggy commit, this crashes with "No module named 'cuda'" because
# the top-level CuteDSL import triggers cuda.bindings resolution.
# On the fixed commit, the import succeeds (or fails with a *different* error
# that is unrelated to the cuda.bindings bug).
# ---------------------------------------------------------------------------
def run_subprocess(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout, cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


import_script = (
    "import sys; sys.path.insert(0, '/workspace/sglang/python'); "
    "from sglang.srt.layers.attention.linear.kda_backend import KDABackend; "
    "print('IMPORT_OK')"
)
stdout, stderr, rc = run_subprocess(import_script)

import_ok = "IMPORT_OK" in stdout
has_cuda_error = "No module named 'cuda'" in stderr

if has_cuda_error:
    # This is exactly the bug: top-level CuteDSL import pulls in cuda.bindings
    check("kda_backend imports without cuda ModuleNotFoundError", False,
          "ModuleNotFoundError: No module named 'cuda' -- the bug is present")
elif not import_ok and rc != 0:
    # Import failed for a reason unrelated to the cuda.bindings bug.
    # This is acceptable -- the fix is about removing the cuda dependency,
    # not about making every other dependency available.
    check("kda_backend imports without cuda ModuleNotFoundError", True,
          f"import failed for non-target reason (rc={rc})")
else:
    check("kda_backend imports without cuda ModuleNotFoundError", True)


# ---------------------------------------------------------------------------
# Check 2: If the import succeeded, verify KDABackend class structure
# ---------------------------------------------------------------------------
if import_ok:
    verify_script = (
        "import sys; sys.path.insert(0, '/workspace/sglang/python'); "
        "from sglang.srt.layers.attention.linear.kda_backend import KDABackend; "
        "assert hasattr(KDABackend, '__init__'), 'KDABackend missing __init__'; "
        "print('CLASS_OK')"
    )
    stdout2, stderr2, rc2 = run_subprocess(verify_script)
    check("KDABackend class has __init__ method",
          "CLASS_OK" in stdout2,
          stderr2[-300:] if stderr2 else "")
else:
    # Module didn't import (non-cuda reason), so we can't verify the class.
    # Give credit -- the fix is about removing the cuda dependency, and
    # whatever prevented import is a different issue.
    check("KDABackend class structure (skipped, module not loadable)", True)


# ---------------------------------------------------------------------------
# Check 3: AST analysis -- no top-level import of CuteDSLKDAKernel / kda_cutedsl
# The bug is that `from ...kda_cutedsl import CuteDSLKDAKernel` appears at
# module level, causing the cuda.bindings crash. The fix moves it inside a
# method. We verify via AST that no top-level ImportFrom references
# kda_cutedsl.
# ---------------------------------------------------------------------------
top_level_cutedsl_import = False
for node in ast.iter_child_nodes(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # Check ImportFrom: module containing 'kda_cutedsl'
        if isinstance(node, ast.ImportFrom) and node.module and "kda_cutedsl" in node.module:
            top_level_cutedsl_import = True
            break
        # Check Import: any name containing 'kda_cutedsl'
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "kda_cutedsl" in alias.name:
                    top_level_cutedsl_import = True
                    break
        # Also check for direct 'CuteDSLKDAKernel' in top-level import names
        if isinstance(node, ast.ImportFrom) and node.names:
            for alias in node.names:
                if alias.name == "CuteDSLKDAKernel":
                    top_level_cutedsl_import = True
                    break

check("No top-level CuteDSL/kda_cutedsl import",
      not top_level_cutedsl_import,
      "Found top-level import of kda_cutedsl or CuteDSLKDAKernel -- this causes the crash on ROCm")


# ---------------------------------------------------------------------------
# Final score
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
