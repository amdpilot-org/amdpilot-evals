#!/usr/bin/env python3
"""Test harness for aiter-asm-pa-headsize-fix eval instance.

Validates that paged attention kernel dispatch correctly handles
different head sizes and does not route unsupported configurations
to the ASM kernel.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

VENV_PYTHON = "/opt/venv/bin/python3"
AITER_PATH = "/workspace/aiter"
ATTENTION_PATH = os.path.join(AITER_PATH, "aiter/ops/attention.py")


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
print("aiter-asm-pa-headsize-fix test harness")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("attention.py exists", os.path.isfile(ATTENTION_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(ATTENTION_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("attention.py is valid Python", True)
except SyntaxError as e:
    check("attention.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 2: _should_use_asm_kernel signature includes head_size parameter.
# ---------------------------------------------------------------------------
print("\n--- Check 2: function signature ---")

dispatch_fn = None
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_should_use_asm_kernel":
        dispatch_fn = node
        break

if not check(
    "_should_use_asm_kernel function found",
    dispatch_fn is not None,
    "function not found — may have been renamed",
):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

param_names = [arg.arg for arg in dispatch_fn.args.args]
check(
    "_should_use_asm_kernel has head_size parameter",
    "head_size" in param_names,
    f"params: {param_names} — head_size missing, ASM kernel will be used for all head sizes",
)

# ---------------------------------------------------------------------------
# Checks 3-5: dispatch behavior verification.
# ---------------------------------------------------------------------------
print("\n--- Checks 3-5: dispatch behavior ---")

fn_src_lines = source_text.splitlines()[dispatch_fn.lineno - 1:dispatch_fn.end_lineno]
fn_src = "\n".join(fn_src_lines)

# The function references torch.int8, torch.float16, etc.
# We need torch in the namespace for the comparison.
dispatch_test_script = f"""
import sys, os
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

try:
    import torch
    print("TORCH:OK")
except ImportError:
    print("TORCH:FAIL")
    sys.exit(0)

# Compile and execute the extracted function
fn_source = {fn_src!r}
ns = {{"torch": torch, "__builtins__": __builtins__}}
try:
    exec(compile(fn_source, "attention.py", "exec"), ns)
    fn = ns["_should_use_asm_kernel"]
    print("COMPILE:OK")
except Exception as e:
    print(f"COMPILE:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

import inspect
sig = inspect.signature(fn)
num_params = len(sig.parameters)
print(f"NUM_PARAMS:{{num_params}}")

# Test with different head_sizes
if num_params >= 5:
    # head_size=64 + int8 cache → should return False
    r64 = fn(1, 32, 64, torch.int8, 0)
    print(f"HEAD64_INT8:{{r64}}")

    # head_size=128 + int8 cache → should return True (supported)
    r128 = fn(1, 32, 128, torch.int8, 0)
    print(f"HEAD128_INT8:{{r128}}")

    # head_size=256 + int8 cache → should return False
    r256 = fn(1, 32, 256, torch.int8, 0)
    print(f"HEAD256_INT8:{{r256}}")

    # head_size=128 + fp16 cache + high_precision=2 → should return True
    r128_hp = fn(1, 32, 128, torch.float16, 2)
    print(f"HEAD128_HP2:{{r128_hp}}")

    # head_size=64 + fp16 cache + high_precision=2 → should return False
    r64_hp = fn(1, 32, 64, torch.float16, 2)
    print(f"HEAD64_HP2:{{r64_hp}}")

elif num_params == 3:
    r_int8 = fn(1, 32, torch.int8)
    print(f"OLD_INT8:{{r_int8}}")
    # Flag that the function lacks head_size parameter
    print("HEAD_SIZE_CHECK:MISSING")
else:
    print(f"UNEXPECTED_PARAMS:{{num_params}}")
"""

try:
    stdout, stderr, rc = run_subprocess(dispatch_test_script, timeout=30)
except subprocess.TimeoutExpired:
    stdout, rc = "TIMEOUT", -1

if "TORCH:FAIL" in stdout:
    print("  [SKIP] torch not available")
elif "COMPILE:FAIL" in stdout:
    err = stdout.split("COMPILE:FAIL:")[1].split("\n")[0]
    check("Dispatch function compiles", False, err)
elif "COMPILE:OK" in stdout:
    check("Dispatch function compiles", True)

    if "HEAD_SIZE_CHECK:MISSING" in stdout:
        check("head_size=64 → rejects ASM kernel", False,
              "function has no head_size parameter — all head sizes route to ASM")
        check("head_size=128 → accepts ASM kernel", False,
              "function has no head_size parameter")
        check("head_size=256 → rejects ASM kernel", False,
              "function has no head_size parameter")
    else:
        # Check 3: head_size=64 must return False
        check(
            "head_size=64 + int8 cache → rejects ASM kernel",
            "HEAD64_INT8:False" in stdout,
            "ASM kernel incorrectly selected for unsupported head_size=64",
        )

        # Check 4: head_size=128 must return True
        check(
            "head_size=128 + int8 cache → accepts ASM kernel",
            "HEAD128_INT8:True" in stdout,
            "ASM kernel incorrectly rejected for supported head_size=128",
        )

        # Check 5: head_size=256 must return False
        check(
            "head_size=256 + int8 cache → rejects ASM kernel",
            "HEAD256_INT8:False" in stdout,
            "ASM kernel incorrectly selected for unsupported head_size=256",
        )

        # Anti-hack: head_size=64 + high_precision=2 must still return False
        if "HEAD64_HP2:" in stdout:
            check(
                "head_size=64 + high_precision=2 → still rejects (head_size check takes priority)",
                "HEAD64_HP2:False" in stdout,
                "high_precision=2 bypasses head_size check — incorrect ordering",
            )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
