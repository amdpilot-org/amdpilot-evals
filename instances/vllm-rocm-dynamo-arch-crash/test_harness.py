#!/usr/bin/env python3
"""Test harness for vllm-rocm-dynamo-arch-crash.

Behavioral test: verifies that ROCm GPU architecture detection functions
(on_gfx9, on_gfx942, etc.) work correctly inside torch.compile regions
without crashing TorchDynamo.
"""
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

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
print("vllm-rocm-dynamo-arch-crash test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Test 1: Import and call arch detection functions.
# Verify they return plain Python bools (not traced/wrapped objects).
#
# Pre-fix: These functions call amdsmi FFI at runtime via @cache. The
# return type may be a cached result object, not a plain bool.
# Post-fix: Functions return pre-computed module-level bool constants.
# ---------------------------------------------------------------------------
print("\n--- Test 1: Arch detection returns plain bools ---")

import_test_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    import torch
    print("TORCH:OK")
except ImportError:
    print("TORCH:FAIL")
    sys.exit(0)

try:
    from vllm.platforms.rocm import on_gfx9, on_gfx942, on_gfx950, on_mi3xx
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

r_gfx9 = on_gfx9()
r_942 = on_gfx942()
r_950 = on_gfx950()
r_mi3xx = on_mi3xx()

print(f"ON_GFX9:{r_gfx9}")
print(f"ON_GFX9_TYPE:{type(r_gfx9).__name__}")
print(f"ON_GFX942:{r_942}")
print(f"ON_GFX950:{r_950}")
print(f"ON_MI3XX:{r_mi3xx}")

all_bool = all(isinstance(v, bool) for v in [r_gfx9, r_942, r_950, r_mi3xx])
print(f"ALL_PLAIN_BOOL:{all_bool}")
"""

try:
    stdout1, stderr1, rc1 = run_subprocess(import_test_script, timeout=60)
except subprocess.TimeoutExpired:
    stdout1, rc1 = "TIMEOUT", -1

if "TORCH:FAIL" in stdout1:
    print("  [SKIP] torch not available")
elif "IMPORT:FAIL" in stdout1:
    err = stdout1.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import arch detection functions", False, err)
elif "IMPORT:OK" in stdout1:
    check("Import arch detection functions", True)
    check(
        "on_gfx*() functions return plain Python bools",
        "ALL_PLAIN_BOOL:True" in stdout1,
        f"got non-bool type: {[l for l in stdout1.splitlines() if 'TYPE' in l]}",
    )

# ---------------------------------------------------------------------------
# Test 2: Call on_gfx9() inside a torch.compile region.
# This is the core bug test.
#
# Pre-fix: on_gfx9() calls amdsmi_init() via @cache → Dynamo cannot trace
# the FFI call → torch._dynamo.exc.Unsupported crash.
# Post-fix: on_gfx9() returns a pre-computed bool constant → Dynamo-safe.
# ---------------------------------------------------------------------------
print("\n--- Test 2: torch.compile with arch detection (Dynamo safety) ---")

dynamo_test_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    import torch
    import torch._dynamo
except ImportError:
    print("TORCH:FAIL")
    sys.exit(0)

try:
    from vllm.platforms.rocm import on_gfx9
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

try:
    # Clear functools.cache so Dynamo must re-trace through the function body.
    # Pre-fix: on_gfx9() is @cache-wrapped and calls amdsmi FFI — cache gets
    # pre-populated during import, hiding the FFI from Dynamo. Clearing it
    # forces Dynamo to trace through the actual FFI call path.
    # Post-fix: on_gfx9() returns a module-level bool constant — no FFI.
    if hasattr(on_gfx9, 'cache_clear'):
        on_gfx9.cache_clear()
    torch._dynamo.reset()

    @torch.compile(backend="eager", fullgraph=True)
    def test_fn(x):
        if on_gfx9():
            return x + 1
        return x + 2

    result = test_fn(torch.tensor(1.0))
    print(f"TORCH_COMPILE:OK:result={result.item()}")
except Exception as e:
    ename = type(e).__name__
    is_dynamo_crash = "Unsupported" in ename or "dynamo" in str(e).lower()
    print(f"TORCH_COMPILE:FAIL:{ename}:{str(e)[:200]}")
    print(f"IS_DYNAMO_CRASH:{is_dynamo_crash}")
"""

try:
    stdout2, stderr2, rc2 = run_subprocess(dynamo_test_script, timeout=120)
except subprocess.TimeoutExpired:
    stdout2, rc2 = "TIMEOUT", -1

if "TORCH:FAIL" in stdout2:
    print("  [SKIP] torch not available")
elif "IMPORT:FAIL" in stdout2:
    err = stdout2.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import on_gfx9 for Dynamo test", False, err)
elif "TORCH_COMPILE:OK" in stdout2:
    check("on_gfx9() inside torch.compile does not crash (Dynamo-safe)", True)
elif "TORCH_COMPILE:FAIL" in stdout2:
    is_dynamo = "IS_DYNAMO_CRASH:True" in stdout2
    err = stdout2.split("TORCH_COMPILE:FAIL:")[1].split("\n")[0]
    check(
        "on_gfx9() inside torch.compile does not crash (Dynamo-safe)",
        False,
        f"{'Dynamo tracing crash — ' if is_dynamo else ''}{err}",
    )

# ---------------------------------------------------------------------------
# Test 3: Repeated calls return consistent results.
# Pre-fix: @cache + FFI could have race conditions or inconsistent state.
# Post-fix: Module-level constants are immutable.
# ---------------------------------------------------------------------------
print("\n--- Test 3: Consistency of arch detection ---")

consistency_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    from vllm.platforms.rocm import on_gfx9, on_gfx942
except Exception as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)

results_9 = [on_gfx9() for _ in range(100)]
results_942 = [on_gfx942() for _ in range(100)]

consistent_9 = len(set(results_9)) == 1
consistent_942 = len(set(results_942)) == 1

print(f"CONSISTENT_GFX9:{consistent_9}")
print(f"CONSISTENT_GFX942:{consistent_942}")
print(f"ALL_CONSISTENT:{consistent_9 and consistent_942}")
"""

try:
    stdout3, stderr3, rc3 = run_subprocess(consistency_script, timeout=30)
except subprocess.TimeoutExpired:
    stdout3, rc3 = "TIMEOUT", -1

if "IMPORT:FAIL" in stdout3:
    check("Arch detection is consistent across calls", False,
          stdout3.split("IMPORT:FAIL:")[1].split("\n")[0])
else:
    check("Arch detection is consistent across 100 repeated calls",
          "ALL_CONSISTENT:True" in stdout3,
          "inconsistent results from repeated calls")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
