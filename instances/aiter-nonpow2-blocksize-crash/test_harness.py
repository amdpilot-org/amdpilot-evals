#!/usr/bin/env python3
"""Test harness for aiter-nonpow2-blocksize-crash eval instance.

Validates that attention config generation handles non-power-of-2 block sizes
without crashing Triton kernel compilation.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
AITER_PATH = "/workspace/aiter"
UNIFIED_ATTN_PATH = os.path.join(
    AITER_PATH, "aiter/ops/triton/attention/unified_attention.py"
)


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


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


print("=" * 60)
print("aiter-nonpow2-blocksize-crash test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("unified_attention.py exists", os.path.isfile(UNIFIED_ATTN_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(UNIFIED_ATTN_PATH) as fh:
        source_text = fh.read()
    compile(source_text, UNIFIED_ATTN_PATH, "exec")
    check("unified_attention.py is valid Python", True)
except SyntaxError as e:
    check("unified_attention.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Checks 2-6: behavioral test via subprocess.
# ---------------------------------------------------------------------------
print("\n--- Checks 2-6: config generation behavior ---")

config_test_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, '/workspace/aiter')

try:
    from aiter.ops.triton.attention.unified_attention import (
        select_2d_config,
        select_3d_config,
    )
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0

# --- Test with non-pow2 block_size = 48 (Qwen3-Next) ---
# Signature: select_2d_config(block_size, head_size, sliding_window,
#     all_decode, max_seqlen_q, max_seqlen_k, num_queries_per_kv, num_2d_prgms)
# Returns: dict with keys BLOCK_M, BLOCK_Q, TILE_SIZE, num_warps, num_stages, waves_per_eu
try:
    cfg_2d = select_2d_config(48, 128, -1, True, 1, 2048, 4, 1)
    tile_size_2d = cfg_2d.get("TILE_SIZE")
    print(f"SELECT_2D_48:OK:TILE_SIZE={tile_size_2d}")
    print(f"SELECT_2D_48_POW2:{is_pow2(tile_size_2d) if tile_size_2d is not None else 'UNKNOWN'}")
    print(f"SELECT_2D_48_LE64:{tile_size_2d <= 64 if tile_size_2d is not None else 'UNKNOWN'}")
except TypeError as e:
    print(f"SELECT_2D_48:FAIL:TypeError:{str(e)[:200]}")
except Exception as e:
    print(f"SELECT_2D_48:FAIL:{type(e).__name__}:{str(e)[:200]}")

# --- Test with non-pow2 block_size = 48, 3D config ---
# Signature: select_3d_config(head_size, block_size, element_size,
#     max_seqlen_k, target_num_prgms, num_2d_prgms)
# Returns: tuple of 2 dicts (attn_config, reduce_config), each with TILE_SIZE
try:
    cfg_3d = select_3d_config(128, 48, 2, 2048, 8, 1)
    attn_cfg, reduce_cfg = cfg_3d
    tile_size_3d = attn_cfg.get("TILE_SIZE")
    print(f"SELECT_3D_48:OK:TILE_SIZE={tile_size_3d}")
    print(f"SELECT_3D_48_POW2:{is_pow2(tile_size_3d) if tile_size_3d is not None else 'UNKNOWN'}")
except TypeError as e:
    print(f"SELECT_3D_48:FAIL:TypeError:{str(e)[:200]}")
except Exception as e:
    print(f"SELECT_3D_48:FAIL:{type(e).__name__}:{str(e)[:200]}")

# --- Test with pow2 block_size = 16 (should still work) ---
try:
    cfg_pow2 = select_2d_config(16, 128, -1, True, 1, 2048, 4, 1)
    tile_size_16 = cfg_pow2.get("TILE_SIZE")
    print(f"SELECT_2D_16:OK:TILE_SIZE={tile_size_16}")
    print(f"SELECT_2D_16_POW2:{is_pow2(tile_size_16) if tile_size_16 is not None else 'UNKNOWN'}")
except Exception as e:
    print(f"SELECT_2D_16:FAIL:{type(e).__name__}:{str(e)[:200]}")

# --- Test with pow2 block_size = 64 (should still work) ---
try:
    cfg_64 = select_2d_config(64, 128, -1, True, 1, 2048, 4, 1)
    tile_size_64 = cfg_64.get("TILE_SIZE")
    print(f"SELECT_2D_64:OK:TILE_SIZE={tile_size_64}")
    print(f"SELECT_2D_64_POW2:{is_pow2(tile_size_64) if tile_size_64 is not None else 'UNKNOWN'}")
except Exception as e:
    print(f"SELECT_2D_64:FAIL:{type(e).__name__}:{str(e)[:200]}")
"""

try:
    stdout, stderr, rc = run_subprocess(config_test_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout, rc = "TIMEOUT", -1

if "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import select_2d_config/select_3d_config", False, err)
elif "IMPORT:OK" in stdout:
    check("Import select_2d_config/select_3d_config", True)

    # Check 2: select_2d_config with block_size=48 succeeds
    if "SELECT_2D_48:OK" in stdout:
        check("select_2d_config(block_size=48) succeeds", True)

        # Check 3: TILE_SIZE is power of 2
        check(
            "select_2d_config(block_size=48) TILE_SIZE is power-of-2",
            "SELECT_2D_48_POW2:True" in stdout,
            "TILE_SIZE is not power-of-2 — Triton kernel will crash",
        )

        # Check 4: TILE_SIZE ≤ 64
        check(
            "select_2d_config(block_size=48) TILE_SIZE ≤ 64",
            "SELECT_2D_48_LE64:True" in stdout,
            "TILE_SIZE > 64 — wasted shared memory",
        )
    else:
        err = stdout.split("SELECT_2D_48:FAIL:")[1].split("\n")[0] if "SELECT_2D_48:FAIL" in stdout else "unknown"
        check("select_2d_config(block_size=48) succeeds", False, err)

    # Check 5: select_3d_config with block_size=48 succeeds
    if "SELECT_3D_48:OK" in stdout:
        check("select_3d_config(block_size=48) succeeds", True)
        check(
            "select_3d_config(block_size=48) TILE_SIZE is power-of-2",
            "SELECT_3D_48_POW2:True" in stdout,
            "TILE_SIZE is not power-of-2",
        )
    elif "SELECT_3D_48:FAIL" in stdout:
        err = stdout.split("SELECT_3D_48:FAIL:")[1].split("\n")[0]
        check("select_3d_config(block_size=48) succeeds", False, err)

    # Check 6: pow2 block_size=16 regression
    if "SELECT_2D_16:OK" in stdout:
        check(
            "select_2d_config(block_size=16) still works (regression guard)",
            "SELECT_2D_16_POW2:True" in stdout,
            "pow2 block_size broke",
        )

    # Check 7: pow2 block_size=64 regression
    if "SELECT_2D_64:OK" in stdout:
        check(
            "select_2d_config(block_size=64) still works (regression guard)",
            "SELECT_2D_64_POW2:True" in stdout,
            "pow2 block_size broke",
        )
else:
    check("Subprocess ran successfully", False, f"stderr: {stderr[:200]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
