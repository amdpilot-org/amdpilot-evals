#!/usr/bin/env python3
"""Test harness for vllm PR #34108: Dynamo tracing crash from amdsmi calls.

Bug: on_gfx9(), on_gfx942(), etc. are @cache-decorated functions that call
_get_gcn_arch_via_amdsmi() → amdsmi_init() at runtime. When called inside
a torch.compile region, Dynamo can't trace through the amdsmi FFI call →
torch._dynamo.exc.Unsupported crash.

Tests (behavioral):
  1. on_gfx*() functions are Dynamo-safe — calling inside torch.compile works.
  2. on_gfx*() returns plain bool (not wrapped/traced object).
  3. Module-level constants (_ON_GFX9, etc.) exist.
  4. AST: on_gfx*() functions don't call amdsmi or torch.cuda internally.
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
print("vllm-rocm-dynamo-arch-crash test harness (PR #34108)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("rocm.py exists", os.path.isfile(ROCM_PY_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(ROCM_PY_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("rocm.py is valid Python", True)
except SyntaxError as e:
    check("rocm.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Checks 2-3 (behavioral, subprocess): on_gfx*() must be Dynamo-safe.
#
# Before fix: on_gfx9() calls amdsmi_init() via @cache function → Dynamo
#   can't trace the FFI call → Unsupported crash.
#
# After fix: on_gfx9() returns a plain Python bool constant → Dynamo-safe.
# ---------------------------------------------------------------------------
print("\n--- Checks 2-3: Dynamo compatibility ---")

dynamo_test_script = """
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    import torch
    print("TORCH:OK")
except ImportError:
    print("TORCH:FAIL")
    sys.exit(0)

# Import the arch detection functions
try:
    from vllm.platforms.rocm import on_gfx9, on_gfx942, on_gfx950, on_mi3xx
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT:FAIL:{type(e).__name__}:{str(e)[:200]}")
    sys.exit(0)

# Check that the functions return plain bools
r_gfx9 = on_gfx9()
r_942 = on_gfx942()
r_950 = on_gfx950()
r_mi3xx = on_mi3xx()

print(f"ON_GFX9:{r_gfx9}")
print(f"ON_GFX9_TYPE:{type(r_gfx9).__name__}")
print(f"ON_GFX942:{r_942}")
print(f"ON_GFX950:{r_950}")
print(f"ON_MI3XX:{r_mi3xx}")

# Verify they return plain Python bools (not traced/wrapped objects)
all_bool = all(isinstance(v, bool) for v in [r_gfx9, r_942, r_950, r_mi3xx])
print(f"ALL_PLAIN_BOOL:{all_bool}")

# Try calling inside torch.compile — this is the actual bug test
try:
    @torch.compile(backend="eager")
    def test_fn(x):
        if on_gfx9():
            return x + 1
        return x + 2

    result = test_fn(torch.tensor(1.0))
    print(f"TORCH_COMPILE:OK:result={result.item()}")
except Exception as e:
    ename = type(e).__name__
    # The specific crash is torch._dynamo.exc.Unsupported
    is_dynamo_crash = "Unsupported" in ename or "dynamo" in str(e).lower()
    print(f"TORCH_COMPILE:FAIL:{ename}:{str(e)[:200]}")
    print(f"IS_DYNAMO_CRASH:{is_dynamo_crash}")
"""

try:
    stdout, stderr, rc = run_subprocess(dynamo_test_script, timeout=120)
except subprocess.TimeoutExpired:
    stdout, rc = "TIMEOUT", -1

if "TORCH:FAIL" in stdout:
    print("  [SKIP] torch not available")
elif "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("Import arch detection functions", False, err)
elif "IMPORT:OK" in stdout:
    check("Import arch detection functions", True)

    # Check 2: functions return plain bools
    check(
        "on_gfx*() functions return plain Python bools",
        "ALL_PLAIN_BOOL:True" in stdout,
        f"got type: {[l for l in stdout.splitlines() if 'TYPE' in l]}",
    )

    # Check 3: torch.compile with on_gfx9() doesn't crash
    if "TORCH_COMPILE:OK" in stdout:
        check("on_gfx9() inside torch.compile does not crash (Dynamo-safe)", True)
    elif "TORCH_COMPILE:FAIL" in stdout:
        is_dynamo = "IS_DYNAMO_CRASH:True" in stdout
        err = stdout.split("TORCH_COMPILE:FAIL:")[1].split("\n")[0]
        check(
            "on_gfx9() inside torch.compile does not crash (Dynamo-safe)",
            False,
            f"{'Dynamo tracing crash — ' if is_dynamo else ''}{err}",
        )

# ---------------------------------------------------------------------------
# Checks 4-5 (AST): on_gfx*() functions should be trivial (return constant).
#
# After fix: each on_gfx*() is a 1-line function returning a module-level bool.
# Before fix: each calls _get_gcn_arch_via_amdsmi() → amdsmi FFI.
# ---------------------------------------------------------------------------
print("\n--- Checks 4-5: function body analysis ---")

gfx_funcs = {}
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name in (
        "on_gfx9", "on_gfx942", "on_gfx950", "on_mi3xx", "on_gfx1x"
    ):
        gfx_funcs[node.name] = node

if gfx_funcs:
    # Check 4: none of the on_gfx*() functions call amdsmi or _get_gcn_arch
    has_runtime_call = False
    for name, fn_node in gfx_funcs.items():
        fn_src = "\n".join(
            source_text.splitlines()[fn_node.lineno - 1:fn_node.end_lineno]
        )
        if any(pattern in fn_src for pattern in [
            "_get_gcn_arch", "amdsmi", "torch.cuda.get_device_properties",
            "amdsmi_get_gpu_asic_info",
        ]):
            has_runtime_call = True
            break

    check(
        "on_gfx*() functions have NO runtime arch detection calls",
        not has_runtime_call,
        "function body calls amdsmi or torch.cuda — not Dynamo-safe",
    )

    # Check 5: functions are simple (≤3 lines body — just return a constant)
    all_simple = True
    for name, fn_node in gfx_funcs.items():
        body_lines = fn_node.end_lineno - fn_node.lineno
        if body_lines > 5:  # generous: decorated function might have extra lines
            all_simple = False
            break

    check(
        "on_gfx*() functions are trivial (return pre-computed constant)",
        all_simple,
        "function body is too complex — likely does runtime computation",
    )
else:
    # Functions may have been completely removed and replaced with constants
    has_constants = (
        "_ON_GFX9" in source_text
        and "_ON_GFX942" in source_text
        and "_ON_GFX950" in source_text
    )
    check(
        "Module-level arch constants exist (_ON_GFX9, _ON_GFX942, etc.)",
        has_constants,
        "no on_gfx*() functions or module-level constants found",
    )
    # Functions might be inlined, which is fine — the torch.compile test above
    # is the authoritative check.
    check("Arch detection is Dynamo-safe (module-level constants)", has_constants)

# ---------------------------------------------------------------------------
# Check 6: module-level _GCN_ARCH or equivalent constant exists.
# This is the arch string resolved once at import time.
# ---------------------------------------------------------------------------
print("\n--- Check 6: module-level arch resolution ---")

has_module_arch = (
    "_GCN_ARCH" in source_text
    or "_get_gcn_arch()" in source_text  # called at module level
    or "_get_gcn_arch_via_amdsmi" in source_text  # cached version
)
check(
    "Module has module-level arch resolution (_GCN_ARCH or equivalent)",
    has_module_arch,
    "no module-level arch detection — arch resolved at function call time",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
