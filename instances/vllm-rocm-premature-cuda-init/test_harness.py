#!/usr/bin/env python3
"""Test harness for vllm PR #33941: premature CUDA initialization in platform detection.

Bug: Importing vllm.platforms triggers torch.cuda.get_device_properties() at
module load time, initializing CUDA before Ray workers can set
CUDA_VISIBLE_DEVICES. All workers end up using GPU 0 instead of their assigned
GPUs.

Fix: Use amdsmi for GPU arch detection instead of torch.cuda, and cache the
result at module level. The on_gfx*() functions no longer call into torch.cuda.

Tests (behavioral, subprocess-isolated):
  1. Import vllm.platforms in subprocess, verify torch.cuda is NOT initialized.
  2. After import, set CUDA_VISIBLE_DEVICES=0, verify device_count respects it.
  3. Verify on_gfx*() functions don't reference torch.cuda.get_device_properties.
  4. AST: verify rocm.py uses amdsmi-based arch detection, not torch.cuda.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
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


def run_subprocess(script, timeout=120, env_override=None):
    env = os.environ.copy()
    # Clear CUDA visibility env vars for clean test
    for key in ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]:
        env.pop(key, None)
    if env_override:
        env.update(env_override)
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/workspace",
        env=env,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-premature-cuda-init test harness (PR #33941)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm.py exists", os.path.isfile(ROCM_PY_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1 (primary behavioral): importing vllm.platforms must NOT initialize
# CUDA. This is the core bug — the fix ensures amdsmi is used instead.
#
# Before fix: torch.cuda.get_device_properties() called at import time → CUDA
#   initialized → torch.cuda.is_initialized() returns True → FAIL.
#
# After fix: amdsmi used for arch detection → CUDA not initialized → PASS.
# ---------------------------------------------------------------------------
print("\n--- Check 1: platform import does not initialize CUDA ---")

cuda_init_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import torch

# Verify CUDA not yet initialized
pre_init = torch.cuda.is_initialized()
print(f"PRE_IMPORT_CUDA_INIT:{pre_init}")

if pre_init:
    print("SKIP:CUDA already initialized before import")
    sys.exit(0)

# This is the import that should NOT initialize CUDA
try:
    from vllm.platforms import current_platform
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

# Check if CUDA was initialized by the import
post_init = torch.cuda.is_initialized()
print(f"POST_IMPORT_CUDA_INIT:{post_init}")
print(f"PLATFORM:{current_platform}")
"""

try:
    stdout1, stderr1, rc1 = run_subprocess(cuda_init_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout1, rc1 = "TIMEOUT", -1

if "SKIP:CUDA already initialized" in stdout1:
    print("  [SKIP] CUDA was initialized before vllm import — test invalid")
elif "IMPORT:FAIL" in stdout1:
    err = stdout1.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import vllm.platforms", False, err)
elif "IMPORT:OK" in stdout1:
    check("Import vllm.platforms succeeds", True)

    check(
        "CUDA NOT initialized after vllm.platforms import",
        "POST_IMPORT_CUDA_INIT:False" in stdout1,
        "torch.cuda.is_initialized() is True — CUDA was prematurely initialized during import",
    )

# ---------------------------------------------------------------------------
# Check 2 (behavioral): after platform import, CUDA_VISIBLE_DEVICES should
# still be respected by torch.cuda.device_count().
#
# Before fix: CUDA initialized at import → device_count locked → env var ignored.
# After fix: CUDA not initialized → env var respected → device_count = 1.
# ---------------------------------------------------------------------------
print("\n--- Check 2: CUDA_VISIBLE_DEVICES respected after import ---")

device_count_script = """
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import torch

from vllm.platforms import current_platform  # should NOT init CUDA

# Now set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count = torch.cuda.device_count()

if count == 0:
    print("NO_GPU")
else:
    print(f"DEVICE_COUNT:{count}")
    print(f"COUNT_IS_ONE:{count == 1}")
"""

try:
    stdout2, stderr2, rc2 = run_subprocess(device_count_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout2, rc2 = "TIMEOUT", -1

if "NO_GPU" in stdout2:
    print("  [SKIP] No GPU available — device count check skipped")
elif "DEVICE_COUNT:" in stdout2:
    check(
        "device_count=1 after setting CUDA_VISIBLE_DEVICES=0 post-import",
        "COUNT_IS_ONE:True" in stdout2,
        "device_count not 1 — CUDA may have been initialized before env var was set",
    )

# ---------------------------------------------------------------------------
# Checks 3-4 (AST): verify arch detection uses amdsmi, not torch.cuda.
#
# The fix moves from:
#   torch.cuda.get_device_properties("cuda").gcnArchName
# to:
#   amdsmi-based detection (via _get_gcn_arch_via_amdsmi or _get_gcn_arch)
#
# We check that on_gfx*() functions do NOT contain torch.cuda references.
# ---------------------------------------------------------------------------
print("\n--- Checks 3-4: arch detection mechanism ---")

try:
    with open(ROCM_PY_PATH) as f:
        source = f.read()
    tree = ast.parse(source)

    # Check 3: on_gfx9() function should NOT call torch.cuda.get_device_properties
    gfx_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "on_gfx9", "on_gfx942", "on_gfx950", "on_mi3xx", "on_gfx1x"
        ):
            gfx_funcs[node.name] = node

    if gfx_funcs:
        has_torch_cuda_in_gfx = False
        for name, fn_node in gfx_funcs.items():
            fn_src = "\n".join(source.splitlines()[fn_node.lineno - 1:fn_node.end_lineno])
            if "torch.cuda.get_device_properties" in fn_src:
                has_torch_cuda_in_gfx = True
                break

        check(
            "on_gfx*() functions do NOT call torch.cuda.get_device_properties",
            not has_torch_cuda_in_gfx,
            "torch.cuda.get_device_properties found in on_gfx* — causes premature CUDA init",
        )
    else:
        # Functions might have been replaced with module-level constants
        # (which is even better — the #34108 fix)
        check(
            "on_gfx*() functions or equivalent arch constants exist",
            "_GCN_ARCH" in source or "_ON_GFX9" in source or "on_gfx9" in source,
            "no arch detection mechanism found",
        )

    # Check 4: module should have amdsmi-based arch detection
    has_amdsmi_detection = (
        "_get_gcn_arch_via_amdsmi" in source
        or "_query_gcn_arch_from_amdsmi" in source
        or "_get_gcn_arch" in source
        or "amdsmi_get_gpu_asic_info" in source
    )
    check(
        "Module uses amdsmi-based GPU arch detection",
        has_amdsmi_detection,
        "no amdsmi arch detection found — may use torch.cuda instead",
    )

except SyntaxError as e:
    check("rocm.py is valid Python", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
