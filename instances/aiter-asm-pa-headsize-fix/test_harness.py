#!/usr/bin/env python3
"""Test harness for aiter PR #2273: ASM paged attention called for unsupported head_size.

Bug: _should_use_asm_kernel() in attention.py does not check head_size.
The ASM paged attention kernel only supports head_size=128, but the
dispatch function routes any head_size to the ASM kernel when other
conditions match (e.g., int8 kv cache, or high_precision=2). Models with
head_size=64 or 256 produce incorrect results because the ASM kernel
reads/writes memory at the wrong offsets.

Tests (behavioral, not source-pattern matching):
  1. Call _should_use_asm_kernel with head_size=64 → must return False
  2. Call _should_use_asm_kernel with head_size=128 → must return True (for int8)
  3. Call _should_use_asm_kernel with head_size=256 → must return False
  4. Verify function signature includes head_size parameter
  5. Verify paged_attention_common passes head_size to the dispatch function
"""
import ast
import inspect
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
print("aiter-asm-pa-headsize-fix test harness (PR #2273)")
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
#
# Before fix: _should_use_asm_kernel(num_seqs, num_heads, kv_cache_tensor_dtype)
# After fix:  _should_use_asm_kernel(num_seqs, num_heads, head_size, kv_cache_tensor_dtype, high_precision)
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
# Checks 3-5 (behavioral): extract and call _should_use_asm_kernel.
#
# We extract just the function source, compile it in isolation, and call
# it with controlled arguments. This avoids importing the full module
# (which needs torch, aiter C++ extensions, etc.) while still testing
# the actual compiled dispatch logic.
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

# Determine function arity — before fix has 3 params, after fix has 5
import inspect
sig = inspect.signature(fn)
num_params = len(sig.parameters)
print(f"NUM_PARAMS:{{num_params}}")

# Test with different head_sizes
if num_params >= 5:
    # After fix: fn(num_seqs, num_heads, head_size, kv_cache_tensor_dtype, high_precision)

    # head_size=64 + int8 cache → should return False (ASM doesn't support head_size!=128)
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

    # head_size=64 + fp16 cache + high_precision=2 → should return False (head_size check first)
    r64_hp = fn(1, 32, 64, torch.float16, 2)
    print(f"HEAD64_HP2:{{r64_hp}}")

elif num_params == 3:
    # Before fix: fn(num_seqs, num_heads, kv_cache_tensor_dtype)
    # No head_size parameter → always routes to ASM for int8
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
# Check 6: paged_attention_common passes head_size to dispatch function.
#
# After fix, the call should be:
#   _should_use_asm_kernel(num_seqs, num_heads, head_size, ...)
# Before fix:
#   _should_use_asm_kernel(num_seqs, num_heads, kv_cache_tensor_dtype) or high_precision == 2
# ---------------------------------------------------------------------------
print("\n--- Check 6: caller passes head_size ---")

pa_common_fn = None
for node in ast.walk(source_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "paged_attention_common":
        pa_common_fn = node
        break

if pa_common_fn is not None:
    # Find calls to _should_use_asm_kernel in paged_attention_common
    dispatch_calls = []
    for child in ast.walk(pa_common_fn):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == "_should_use_asm_kernel":
                dispatch_calls.append(child)

    if dispatch_calls:
        call = dispatch_calls[0]
        num_args = len(call.args) + len(call.keywords)
        # After fix: should pass 5 args (num_seqs, num_heads, head_size, dtype, high_precision)
        # Before fix: passes 3 args (num_seqs, num_heads, dtype)
        check(
            "paged_attention_common passes ≥5 args to _should_use_asm_kernel",
            num_args >= 5,
            f"only passes {num_args} args — head_size/high_precision likely missing",
        )

        # Check that one arg references head_size
        arg_names = set()
        for arg in call.args:
            if isinstance(arg, ast.Name):
                arg_names.add(arg.id)
        check(
            "Dispatch call includes head_size argument",
            "head_size" in arg_names,
            f"args reference: {arg_names}",
        )
    else:
        check("_should_use_asm_kernel called from paged_attention_common", False,
              "no call found")
else:
    check("paged_attention_common found", False, "function not in source")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
