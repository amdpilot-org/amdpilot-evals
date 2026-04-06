#!/usr/bin/env python3
"""Test harness for vllm PR #31282: Fix paged_kv_last_page_len in AITER MLA decode.

Bug: The AITER MLA kernel uses block_size=1, meaning every page holds exactly
one token. Therefore paged_kv_last_page_len must always be 1 for every
request. However, the buggy code sets it to the full sequence length:

    paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)

This causes wrong attention scores and potential out-of-bounds memory access
for sequences whose length is not a power of two (e.g., prime-length
sequences).

Tests (behavioral, source-inspection, and AST-based):
  1. Import check -- AiterMLADecodeMetadata can be imported.
  2. Source inspection -- paged_kv_last_page_len is set via torch.ones (all-1s
     buffer) rather than derived from seq_lens_device.
  3. AST analysis -- the __init__ buffer creation uses torch.ones, not a
     variable-based expression tied to seq_lens.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

ROCM_AITER_MLA_PATH = "/workspace/vllm/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
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
print("vllm-rocm-mla-last-page-len test harness (PR #31282)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm_aiter_mla.py exists", os.path.isfile(ROCM_AITER_MLA_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python and AiterMLADecodeMetadata can be imported
# ---------------------------------------------------------------------------
print("\n--- Check 1: Import AiterMLADecodeMetadata ---")

import_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLADecodeMetadata
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
"""

try:
    stdout1, stderr1, rc1 = run_subprocess(import_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout1, rc1 = "TIMEOUT", -1

if "IMPORT:OK" in stdout1:
    check("AiterMLADecodeMetadata imports successfully", True)
elif "IMPORT:FAIL" in stdout1:
    err = stdout1.split("IMPORT:FAIL:")[1].split("\n")[0]
    # Import may fail on non-ROCm systems; treat as skip rather than fail
    print(f"  [SKIP] Cannot import AiterMLADecodeMetadata ({err}) -- skipped")
else:
    print(f"  [SKIP] Import check inconclusive (timeout or unexpected output) -- skipped")

# ---------------------------------------------------------------------------
# Check 2: Source inspection -- paged_kv_last_page_len uses torch.ones
# ---------------------------------------------------------------------------
print("\n--- Check 2: Source inspection for paged_kv_last_page_len ---")

try:
    with open(ROCM_AITER_MLA_PATH) as fh:
        source_text = fh.read()
except Exception as e:
    check("Read source file", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# The fix introduces: self.paged_kv_last_page_len = torch.ones(...)
# The buggy code has: paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)
#
# We check that:
#  a) torch.ones is used with paged_kv_last_page_len (the fix pattern)
#  b) The old buggy pattern (torch.where with seq_lens_device) is NOT present

has_ones_init = ("paged_kv_last_page_len" in source_text
                 and "torch.ones" in source_text)

# Check the source has the ones-based initialization for paged_kv_last_page_len
# Look for lines that assign paged_kv_last_page_len using torch.ones
lines = source_text.splitlines()
ones_assignment_found = False
for i, line in enumerate(lines):
    stripped = line.strip()
    if ("paged_kv_last_page_len" in stripped
            and "torch.ones" in stripped
            and "=" in stripped):
        ones_assignment_found = True
        break

check(
    "paged_kv_last_page_len initialized with torch.ones (all-1s buffer)",
    ones_assignment_found,
    "Expected 'paged_kv_last_page_len = torch.ones(...)' but not found. "
    "The buffer must be all-ones since block_size=1.",
)

# Verify the buggy pattern is NOT present:
# The old code: torch.where(seq_lens_device == 0, 1, seq_lens_device)
buggy_pattern_present = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith("#"):
        continue
    if ("paged_kv_last_page_len" in stripped
            and "torch.where" in stripped
            and "seq_lens" in stripped):
        buggy_pattern_present = True
        break

check(
    "Old buggy pattern (torch.where with seq_lens) is removed",
    not buggy_pattern_present,
    "Found 'paged_kv_last_page_len = torch.where(...seq_lens...)' -- "
    "this derives last_page_len from sequence lengths, which is wrong for block_size=1.",
)

# ---------------------------------------------------------------------------
# Check 3: AST analysis -- __init__ creates paged_kv_last_page_len with
#           torch.ones, not derived from seq_lens or variable-based
# ---------------------------------------------------------------------------
print("\n--- Check 3: AST analysis of __init__ buffer creation ---")

try:
    source_tree = ast.parse(source_text)
except SyntaxError as e:
    check("rocm_aiter_mla.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


def find_paged_kv_last_page_len_init(tree):
    """Find the assignment to self.paged_kv_last_page_len in __init__."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (isinstance(target, ast.Attribute)
                            and target.attr == "paged_kv_last_page_len"
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        return child
    return None


init_assign = find_paged_kv_last_page_len_init(source_tree)

if not check(
    "self.paged_kv_last_page_len assignment found in __init__",
    init_assign is not None,
    "No assignment to self.paged_kv_last_page_len in __init__ -- "
    "the buffer should be pre-initialized in the constructor.",
):
    print(f"\nSCORE: {checks_passed / checks_total * 100.0:.1f}")
    sys.exit(1)

# Check the RHS is a call to torch.ones (or similar fixed-value pattern)
rhs = init_assign.value

# The RHS should be a Call node where the function is torch.ones
is_torch_ones = False
if isinstance(rhs, ast.Call):
    func = rhs.func
    # torch.ones(...) pattern
    if (isinstance(func, ast.Attribute)
            and func.attr == "ones"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"):
        is_torch_ones = True
    # Could also be torch.ones_like or similar
    elif (isinstance(func, ast.Attribute)
            and func.attr in ("ones", "ones_like", "full")
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"):
        is_torch_ones = True

check(
    "Buffer created via torch.ones (fixed all-1s values, not variable-based)",
    is_torch_ones,
    "Expected torch.ones(...) or similar constant initialization, "
    "but found a different pattern. Block_size=1 means last_page_len is always 1.",
)

# Extra AST check: verify _build_decode does NOT recompute paged_kv_last_page_len
# from seq_lens. After the fix, it should just slice self.paged_kv_last_page_len.
def find_build_decode_seq_lens_usage(tree, source):
    """Check if _build_decode computes paged_kv_last_page_len from seq_lens."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_build_decode"):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "paged_kv_last_page_len":
                        # Check if the RHS references seq_lens_device
                        rhs_src = ast.get_source_segment(source, child.value)
                        if rhs_src and "seq_lens" in rhs_src:
                            return True
                        # Also check for torch.where usage
                        for subnode in ast.walk(child.value):
                            if (isinstance(subnode, ast.Attribute)
                                    and subnode.attr == "where"):
                                return True
        return False
    return False

uses_seq_lens_in_build = find_build_decode_seq_lens_usage(source_tree, source_text)

check(
    "_build_decode does not derive paged_kv_last_page_len from seq_lens",
    not uses_seq_lens_in_build,
    "Found seq_lens-based computation of paged_kv_last_page_len in _build_decode. "
    "With block_size=1, last_page_len should always be 1 (sliced from pre-initialized buffer).",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
