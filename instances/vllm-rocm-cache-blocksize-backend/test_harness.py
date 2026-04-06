#!/usr/bin/env python3
"""Test harness for vllm PR #37606: cache block size mismatch for aiter unified attention.

Bug: Block size for the AITER unified attention backend was hardcoded in
RocmPlatform.check_and_update_config() — setting cache_config.block_size = 64
for aiter unified attention and block_size = 16 otherwise. This platform-level
hardcoding caused block_size mismatches: the backend expected 64, but if the
config flags didn't match exactly, the platform set 16 (or vice versa).

Tests:
  1. Backend has get_preferred_block_size() returning 64.
  2. check_and_update_config does NOT hardcode cache_config.block_size.
  3. update_block_size_for_backend stub is removed.
  4. get_supported_kernel_block_sizes still present on backend.
  5. Behavioral: calling get_preferred_block_size returns 64.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
UNIFIED_ATTN_PATH = "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_unified_attn.py"
ROCM_PY_PATH = "/workspace/vllm/vllm/platforms/rocm.py"


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
print("vllm-rocm-cache-blocksize-backend test harness (PR #37606)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target files exist
# ---------------------------------------------------------------------------
if not check("rocm_aiter_unified_attn.py exists", os.path.isfile(UNIFIED_ATTN_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

if not check("rocm.py exists", os.path.isfile(ROCM_PY_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Read and parse sources
try:
    with open(UNIFIED_ATTN_PATH) as f:
        unified_source = f.read()
    unified_tree = ast.parse(unified_source)
except SyntaxError as e:
    check("unified_attn.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

try:
    with open(ROCM_PY_PATH) as f:
        rocm_source = f.read()
    rocm_tree = ast.parse(rocm_source)
except SyntaxError as e:
    check("rocm.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: RocmAiterUnifiedAttentionBackend has get_preferred_block_size
# that returns 64.
#
# Before fix: method does not exist — block_size is set by the platform.
# After fix: backend reports its own preferred block_size (64).
# ---------------------------------------------------------------------------
print("\n--- Check 1: Backend block size method ---")

found_get_preferred = False
get_preferred_returns_64 = False

for node in ast.walk(unified_tree):
    if isinstance(node, ast.ClassDef) and "UnifiedAttention" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_preferred_block_size":
                found_get_preferred = True
                # Check if it returns 64
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                        if sub.value.value == 64:
                            get_preferred_returns_64 = True

check(
    "Backend has get_preferred_block_size() method",
    found_get_preferred,
    "RocmAiterUnifiedAttentionBackend missing get_preferred_block_size — block size not backend-controlled",
)

check(
    "get_preferred_block_size() returns 64",
    get_preferred_returns_64,
    "Expected return value of 64 for AITER unified attention backend",
)

# ---------------------------------------------------------------------------
# Check 2: check_and_update_config does NOT set cache_config.block_size.
#
# Before fix: hardcodes cache_config.block_size = 64 or 16 based on env vars.
# After fix: block_size logic removed — delegated to backend.
# ---------------------------------------------------------------------------
print("\n--- Check 2: Platform config cleanup ---")

config_sets_blocksize = False
for node in ast.walk(rocm_tree):
    if isinstance(node, ast.ClassDef) and "Rocm" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "check_and_update_config":
                fn_source = "\n".join(
                    rocm_source.splitlines()[item.lineno - 1:item.end_lineno]
                )
                # Check for actual assignment to cache_config.block_size
                if "cache_config.block_size =" in fn_source or "cache_config.block_size=" in fn_source:
                    config_sets_blocksize = True

check(
    "check_and_update_config does NOT hardcode block_size",
    not config_sets_blocksize,
    "Platform still hardcodes cache_config.block_size — should be delegated to backend",
)

# ---------------------------------------------------------------------------
# Check 3: update_block_size_for_backend stub removed.
#
# Before fix: empty stub method exists with TODO comment.
# After fix: removed entirely — backend handles its own block_size.
# ---------------------------------------------------------------------------
print("\n--- Check 3: Stub removal ---")

has_update_stub = False
for node in ast.walk(rocm_tree):
    if isinstance(node, ast.ClassDef) and "Rocm" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "update_block_size_for_backend":
                has_update_stub = True

check(
    "update_block_size_for_backend stub removed from RocmPlatform",
    not has_update_stub,
    "Stub method still exists — should be removed since backend handles block_size",
)

# ---------------------------------------------------------------------------
# Check 4: cache_config no longer referenced in check_and_update_config.
#
# Before fix: cache_config = vllm_config.cache_config at top of method.
# After fix: cache_config reference removed (no longer needed).
# ---------------------------------------------------------------------------
print("\n--- Check 4: cache_config reference removed ---")

has_cache_config_ref = False
for node in ast.walk(rocm_tree):
    if isinstance(node, ast.ClassDef) and "Rocm" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "check_and_update_config":
                fn_source = "\n".join(
                    rocm_source.splitlines()[item.lineno - 1:item.end_lineno]
                )
                # Check for cache_config local variable assignment
                if "cache_config = vllm_config.cache_config" in fn_source:
                    has_cache_config_ref = True

check(
    "check_and_update_config does not reference cache_config",
    not has_cache_config_ref,
    "cache_config still referenced — block_size logic not fully removed",
)

# ---------------------------------------------------------------------------
# Check 5 (behavioral): call get_preferred_block_size via subprocess.
#
# This verifies the method is actually callable and returns the expected value.
# Uses AST extraction to avoid full vllm import.
# ---------------------------------------------------------------------------
print("\n--- Check 5: Behavioral block size test ---")

behavioral_script = """
import ast, sys, types

TARGET = "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_unified_attn.py"

with open(TARGET) as f:
    source = f.read()
tree = ast.parse(source)

# Extract the get_preferred_block_size method
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and "UnifiedAttention" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_preferred_block_size":
                # Extract function source
                fn_lines = source.splitlines()[item.lineno - 1:item.end_lineno]
                fn_src = "\\n".join(fn_lines)
                # Remove decorator lines and class context, make standalone
                # Find the def line
                for i, line in enumerate(fn_lines):
                    if line.strip().startswith("def "):
                        fn_lines = fn_lines[i:]
                        break
                # Dedent
                indent = len(fn_lines[0]) - len(fn_lines[0].lstrip())
                fn_lines = [l[indent:] for l in fn_lines]
                fn_src = "\\n".join(fn_lines)
                break

try:
    import logging
    _logger = logging.getLogger("test")
    # vllm adds warning_once/info_once methods to loggers
    if not hasattr(_logger, 'warning_once'):
        _logger.warning_once = _logger.warning
    if not hasattr(_logger, 'info_once'):
        _logger.info_once = _logger.info
    ns = {"logger": _logger, "logging": logging}
    exec(fn_src, ns)
    fn = ns["get_preferred_block_size"]
    # Call with cls=None and default_block_size=16
    result = fn(None, 16)
    print(f"RESULT:{result}")
    print(f"IS_64:{result == 64}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
"""

try:
    stdout, stderr, rc = run_subprocess(behavioral_script, timeout=30)
except subprocess.TimeoutExpired:
    stdout, rc = "TIMEOUT", -1

if "ERROR:" in stdout:
    err = stdout.split("ERROR:")[1].split("\n")[0]
    check("get_preferred_block_size(default=16) returns 64", False, err)
elif "IS_64:True" in stdout:
    check("get_preferred_block_size(default=16) returns 64", True)
elif "IS_64:False" in stdout:
    result_val = stdout.split("RESULT:")[1].split("\n")[0] if "RESULT:" in stdout else "?"
    check("get_preferred_block_size(default=16) returns 64", False, f"returned {result_val}")
else:
    check("get_preferred_block_size(default=16) returns 64", False, f"unexpected output: {stdout[:200]}")

# ---------------------------------------------------------------------------
# Check 6: get_supported_kernel_block_sizes still present on backend.
# This method was NOT removed — only the platform hardcoding was.
# ---------------------------------------------------------------------------
print("\n--- Check 6: Supported block sizes method ---")

has_supported_sizes = False
for node in ast.walk(unified_tree):
    if isinstance(node, ast.ClassDef) and "UnifiedAttention" in node.name:
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "get_supported_kernel_block_sizes":
                has_supported_sizes = True

check(
    "Backend retains get_supported_kernel_block_sizes() method",
    has_supported_sizes,
    "Missing get_supported_kernel_block_sizes — backend can't report valid block sizes",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
