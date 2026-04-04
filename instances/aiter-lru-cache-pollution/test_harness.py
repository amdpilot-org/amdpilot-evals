#!/usr/bin/env python3
"""Test harness for aiter PR #2169: LRU cache pollution in get_gemm_config.

Bug: get_gemm_config() returns a reference to the LRU-cached config dict.
Callers (like fused_gemm_afp4wfp4_split_cat) mutate this dict in-place by
adding/removing keys. The next call to get_gemm_config() with the same
(M, N, K) returns the mutated dict, which may be missing expected keys
(e.g., BLOCK_SIZE_S3) → KeyError crash.

Fix: Renamed the cached function to _get_gemm_config_cached() and added a
public get_gemm_config() wrapper that returns copy.deepcopy(config), so
callers always get a fresh copy they can freely mutate.

Tests (behavioral, not source-pattern matching):
  1. Call get_gemm_config() twice with same args, verify independent dicts.
  2. Mutate first result, verify second result is NOT polluted.
  3. Verify cache still works (same args → same base values).
  4. Anti-hack: direct mutation detection.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
AITER_PATH = "/workspace/aiter"


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
print("aiter-lru-cache-pollution test harness (PR #2169)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: aiter source exists
# ---------------------------------------------------------------------------
config_utils_path = os.path.join(
    AITER_PATH, "aiter/ops/triton/utils/gemm_config_utils.py"
)
if not check("gemm_config_utils.py exists", os.path.isfile(config_utils_path)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Checks 1-4: behavioral cache pollution test via subprocess.
#
# The core test: call get_gemm_config() → mutate result → call again →
# verify the second result is NOT polluted by the mutation.
#
# Before fix: get_gemm_config returns a reference to the cached dict.
#   Mutation pollutes the cache → second call returns the mutated dict → FAIL.
#
# After fix: get_gemm_config returns a deep copy.
#   Mutation only affects the copy → second call returns a clean dict → PASS.
# ---------------------------------------------------------------------------
print("\n--- Checks 1-4: cache isolation behavior ---")

cache_test_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, '/workspace/aiter')

try:
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

# Use a common config name that should exist. Try a few fallbacks.
config_name = None
test_M = 16
for candidate in ["GEMM-A16W16", "GEMM-FP8", "GEMM-A8W8", "GEMM-A16W4"]:
    try:
        cfg, _ = get_gemm_config(candidate, test_M)
        if cfg:
            config_name = candidate
            break
    except Exception:
        continue

if config_name is None:
    print("CONFIG:NONE")
    sys.exit(0)

print(f"CONFIG:{config_name}")

# --- Test 1: Two calls return equal but independent dicts ---
cfg1, tuned1 = get_gemm_config(config_name, test_M)
cfg2, tuned2 = get_gemm_config(config_name, test_M)

# They should have the same content
same_content = (cfg1 == cfg2)
print(f"SAME_CONTENT:{same_content}")

# But they should NOT be the exact same object (id)
different_objects = (cfg1 is not cfg2)
print(f"DIFFERENT_OBJECTS:{different_objects}")

# --- Test 2: Mutating cfg1 does NOT affect cfg2 ---
# Add a sentinel key
sentinel_key = "__test_pollution_sentinel__"
cfg1[sentinel_key] = "POLLUTED"

cfg3, _ = get_gemm_config(config_name, test_M)
is_polluted = sentinel_key in cfg3
print(f"CACHE_POLLUTED:{is_polluted}")

# --- Test 3: Delete a key from cfg1, verify it still exists in new calls ---
if len(cfg1) > 1:
    # Pick a real key (not our sentinel)
    real_keys = [k for k in cfg1 if k != sentinel_key]
    if real_keys:
        deleted_key = real_keys[0]
        del cfg1[deleted_key]
        cfg4, _ = get_gemm_config(config_name, test_M)
        key_survived = deleted_key in cfg4
        print(f"KEY_SURVIVED_DELETION:{key_survived}")
    else:
        print("KEY_SURVIVED_DELETION:SKIP")
else:
    print("KEY_SURVIVED_DELETION:SKIP")

# --- Test 4: Verify original values are preserved ---
cfg5, _ = get_gemm_config(config_name, test_M)
original_match = (cfg5 == cfg2)  # cfg2 was fetched before any mutations
print(f"ORIGINAL_PRESERVED:{original_match}")
"""

try:
    stdout, stderr, rc = run_subprocess(cache_test_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout, rc = "TIMEOUT", -1

if "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import get_gemm_config", False, err)
elif "CONFIG:NONE" in stdout:
    print("  [SKIP] No GEMM config files found — cannot test cache behavior")
elif "IMPORT:OK" in stdout:
    check("Import get_gemm_config", True)

    check(
        "Two get_gemm_config() calls return equal content",
        "SAME_CONTENT:True" in stdout,
        "configs differ — function is non-deterministic",
    )

    check(
        "Two get_gemm_config() calls return DIFFERENT objects (deep copy)",
        "DIFFERENT_OBJECTS:True" in stdout,
        "same object returned — cache returns reference, not copy",
    )

    check(
        "Mutating returned config does NOT pollute cache",
        "CACHE_POLLUTED:False" in stdout,
        "sentinel key found in subsequent call — cache is polluted!",
    )

    if "KEY_SURVIVED_DELETION:SKIP" not in stdout:
        check(
            "Deleting key from returned config does NOT affect cache",
            "KEY_SURVIVED_DELETION:True" in stdout,
            "deleted key missing from subsequent call — cache is polluted!",
        )

    check(
        "Original config values preserved after mutations to copies",
        "ORIGINAL_PRESERVED:True" in stdout,
        "values changed — deep copy not working correctly",
    )
else:
    check("Subprocess ran successfully", False, f"stderr: {stderr[:200]}")

# ---------------------------------------------------------------------------
# Check 5: Verify the fix mechanism — get_gemm_config should use deep copy.
# Parse the source to confirm copy.deepcopy or dict() wrapping is present
# in the public function. This is a supplementary structural check.
# ---------------------------------------------------------------------------
print("\n--- Check 5: fix mechanism verification ---")

import ast

try:
    with open(config_utils_path) as f:
        source = f.read()
    tree = ast.parse(source)

    # Find the public get_gemm_config function
    public_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_gemm_config":
            public_fn = node
            break

    if public_fn is not None:
        # Check if it contains a call to copy.deepcopy or dict()
        fn_source = "\n".join(
            source.splitlines()[public_fn.lineno - 1:public_fn.end_lineno]
        )
        has_deepcopy = "deepcopy" in fn_source
        has_dict_copy = "dict(" in fn_source
        has_copy_call = ".copy()" in fn_source

        uses_copy = has_deepcopy or has_dict_copy or has_copy_call
        check(
            "get_gemm_config uses copy mechanism (deepcopy/dict()/copy())",
            uses_copy,
            "no copy mechanism found — returns raw cached reference",
        )
    else:
        check("get_gemm_config function found", False, "function not in source")

except SyntaxError as e:
    check("gemm_config_utils.py is valid Python", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
