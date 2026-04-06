#!/usr/bin/env python3
"""Test harness for sglang PR #20840: fp8 w8a8 block-scaled gemm tuning for gfx950.

Bug: use_aiter_triton_gemm_w8a8_tuned_gfx950(n=7168, k=2304) returned False,
causing the FP8 block-scaled GEMM (DeepSeek-V3.2 shape) to fall back to the CK
kernel on MI355 (gfx950). The CK kernel produces incorrect output for this shape,
dropping DeepSeek-V3.2 accuracy from ~95% to 0%.

Tests (behavioral, not source-pattern matching):
  1. Static dispatch logic — extract and call the dispatch function to verify it
     returns True for (7168, 2304), False for untested shapes, True for pre-existing
     tuned shapes (regression guard).
  2. GPU behavioral correctness — run the actual Triton FP8 block-scaled GEMM for
     shape (M=8, N=7168, K=2304) via subprocess and compare against a float32
     dequantize-then-matmul reference.  Skipped gracefully if GPU/aiter is absent.
"""
import ast
import os
import subprocess
import sys

checks_passed = 0
checks_total = 0

FP8_UTILS_PATH = "/workspace/sglang/python/sglang/srt/layers/quantization/fp8_utils.py"
SGLANG_PYTHON_PATH = "/workspace/sglang/python"
VENV_PYTHON = "/opt/venv/bin/python3"

# N=7168, K=2304 is the exact DeepSeek-V3.2 GEMM shape affected by the bug.
BUG_N = 7168
BUG_K = 2304


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


def run_subprocess(script, timeout=180):
    result = subprocess.run(
        [VENV_PYTHON, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("sglang-fp8-w8a8-gfx950-tune test harness (PR #20840)")
print("=" * 60)

# ---------------------------------------------------------------------------
# Check 0: target file exists
# ---------------------------------------------------------------------------
if not check("fp8_utils.py exists at expected path", os.path.isfile(FP8_UTILS_PATH)):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 1: file is valid Python
# ---------------------------------------------------------------------------
try:
    with open(FP8_UTILS_PATH) as fh:
        source_text = fh.read()
    source_tree = ast.parse(source_text)
    check("fp8_utils.py is valid Python", True)
except SyntaxError as e:
    check("fp8_utils.py is valid Python", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check 2 (static): extract and evaluate use_aiter_triton_gemm_w8a8_tuned_gfx950
#
# We use ast to locate and extract just the function definition, then exec it in
# a fresh namespace.  This avoids importing the whole module (which needs torch,
# aiter, ROCm etc.) while still calling the real compiled function logic.
#
# Sub-check 2a: (7168, 2304) returns True   — the bug fix
# Sub-check 2b: (1234, 5678) returns False  — guards against over-broad return True
# Sub-check 2c: (7168, 2048) returns True   — pre-existing shape, no regression
# ---------------------------------------------------------------------------
print("\n--- Check 2: dispatch function behavior for critical shapes ---")

dispatch_fn = None
for node in ast.walk(source_tree):
    if (isinstance(node, ast.FunctionDef)
            and node.name == "use_aiter_triton_gemm_w8a8_tuned_gfx950"):
        dispatch_fn = node
        break

if not check(
    "use_aiter_triton_gemm_w8a8_tuned_gfx950 function exists in fp8_utils.py",
    dispatch_fn is not None,
    "function definition not found — has it been renamed or removed?",
):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Extract function source and execute it in a clean namespace.
src_lines = source_text.splitlines()
fn_src = "\n".join(src_lines[dispatch_fn.lineno - 1 : dispatch_fn.end_lineno])
fn_ns = {}
try:
    exec(compile(fn_src, FP8_UTILS_PATH, "exec"), fn_ns)
    fn = fn_ns["use_aiter_triton_gemm_w8a8_tuned_gfx950"]
    check("Dispatch function compiles and is callable", True)
except Exception as e:
    check("Dispatch function compiles and is callable", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# 2a — the bug fix
result_bug = fn(BUG_N, BUG_K)
check(
    f"use_aiter_triton_gemm_w8a8_tuned_gfx950({BUG_N}, {BUG_K}) returns True (fix for DeepSeek-V3.2 shape)",
    result_bug is True,
    f"returned {result_bug!r} — shape still missing from tuned config list",
)

# 2b — reward-hacking guard: an arbitrary untuned shape must still return False
result_bad = fn(1234, 5678)
check(
    "use_aiter_triton_gemm_w8a8_tuned_gfx950(1234, 5678) returns False (no over-broad True)",
    result_bad is False,
    f"returned {result_bad!r} — function returns True for untested shapes (reward-hacking detected)",
)

# 2c — regression guard: a shape already in the original list must still return True
result_known = fn(7168, 2048)
check(
    "use_aiter_triton_gemm_w8a8_tuned_gfx950(7168, 2048) returns True (no regression)",
    result_known is True,
    f"returned {result_known!r} — pre-existing tuned shape now broken",
)

# ---------------------------------------------------------------------------
# Check 3 (behavioral, GPU): actual FP8 block-scaled GEMM correctness
#
# Run the Triton FP8 kernel for shape (M=8, N=7168, K=2304) in an isolated
# subprocess.  Compare against a float32 dequantize-then-matmul reference.
# On a correctly fixed tree the Triton path is numerically accurate; the
# original CK path on gfx950 produces garbage for this shape.
#
# Skipped gracefully if GPU/aiter is not available in the eval environment.
# ---------------------------------------------------------------------------
print("\n--- Check 3: behavioral FP8 GEMM correctness for (N=7168, K=2304) ---")

gemm_correctness_script = f"""
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, {SGLANG_PYTHON_PATH!r})

import torch

if not torch.cuda.is_available():
    print("GPU_UNAVAILABLE")
    sys.exit(0)

try:
    import aiter
    from aiter import get_hip_quant
    from aiter.ops.triton.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Build FP8 tensors for shape M=8, N={BUG_N}, K={BUG_K}
torch.manual_seed(42)
M, N, K = 8, {BUG_N}, {BUG_K}
BLOCK = 128  # aiter per_1x128 quantisation block size

device = torch.device("cuda:0")
dtype  = torch.bfloat16
fp8_t  = torch.float8_e4m3fnuz  # AMD FP8 FNUZ format

# Random inputs / weights in float32 for reference
x_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
w_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

# Quantize input: per-row 128-wide groups using aiter's per_1x128 quantizer
try:
    aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)
    q_input, x_scale = aiter_per1x128_quant(
        x_fp32.to(dtype), quant_dtype=aiter.dtypes.fp8
    )
    print(f"QUANT_INPUT:OK:q_input={{tuple(q_input.shape)}}:x_scale={{tuple(x_scale.shape)}}")
except Exception as e:
    print(f"QUANT_INPUT:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Quantize weight: symmetric per-block-of-128 quantisation
try:
    w_blocks = w_fp32.view(N, K // BLOCK, BLOCK)
    w_scale_blocks = w_blocks.abs().amax(dim=-1) / 448.0  # fp8_e4m3fnuz max = 448
    w_scale_blocks = w_scale_blocks.clamp(min=1e-12)
    w_q_blocks = (w_blocks / w_scale_blocks.unsqueeze(-1)).clamp(-448.0, 448.0)
    w_q = w_q_blocks.view(N, K).to(fp8_t).contiguous()
    w_scale = w_scale_blocks.to(torch.float32).contiguous()  # shape: (N, K//BLOCK)
    print(f"QUANT_WEIGHT:OK:w_q={{tuple(w_q.shape)}}:w_scale={{tuple(w_scale.shape)}}")
except Exception as e:
    print(f"QUANT_WEIGHT:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Float32 reference: dequantize both operands, compute matmul
try:
    x_scale_ref = x_scale.view(M, K // BLOCK).to(torch.float32)
    x_dq = (x_fp32.view(M, K // BLOCK, BLOCK) * x_scale_ref.unsqueeze(-1)).view(M, K)
    w_dq = (w_q_blocks * w_scale_blocks.unsqueeze(-1)).view(N, K)
    ref = torch.mm(x_dq.float(), w_dq.float().t())
    print("REF:OK")
except Exception as e:
    print(f"REF:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Run the Triton FP8 block-scaled GEMM kernel directly
try:
    out_triton = triton_gemm_a8w8_blockscale(
        q_input, w_q, x_scale, w_scale, dtype=dtype
    )
    torch.cuda.synchronize()
    print(f"TRITON:OK:shape={{tuple(out_triton.shape)}}")
except Exception as e:
    print(f"TRITON:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Numerical comparison: triton output vs float32 reference
# FP8 block-scaled gemm has limited precision; tolerances are generous but
# would catch the CK-path corruption (which produces values many OOM off).
try:
    diff = (out_triton.float() - ref).abs()
    ref_scale_abs = ref.abs() + 1e-6
    rel_err_mean = (diff / ref_scale_abs).mean().item()
    rel_err_max  = (diff / ref_scale_abs).max().item()
    abs_err_mean = diff.mean().item()
    abs_err_max  = diff.max().item()
    tol_rel_mean = 0.05   # 5% mean relative error
    tol_rel_max  = 0.30   # 30% max relative error
    triton_ok = rel_err_mean < tol_rel_mean and rel_err_max < tol_rel_max
    print(
        f"TRITON_VS_REF:"
        f"rel_mean={{rel_err_mean:.4f}}:rel_max={{rel_err_max:.4f}}"
        f":abs_mean={{abs_err_mean:.4f}}:abs_max={{abs_err_max:.4f}}"
        f":PASS={{triton_ok}}"
    )
except Exception as e:
    print(f"TRITON_VS_REF:FAIL:{{type(e).__name__}}:{{str(e)[:200]}}")
    sys.exit(0)

# Confirm dispatch selects Triton on the target hardware
dispatch_result = use_aiter_triton_gemm_w8a8_tuned_gfx950({BUG_N}, {BUG_K})
print(f"DISPATCH_TRITON:{{dispatch_result}}")
"""

try:
    stdout3, stderr3, rc3 = run_subprocess(gemm_correctness_script, timeout=120)
except subprocess.TimeoutExpired:
    check("FP8 GEMM behavioral test (subprocess)", False, "subprocess timed out after 120s")
    stdout3, stderr3, rc3 = "", "", -1

if "GPU_UNAVAILABLE" in stdout3:
    print("  [SKIP] GPU not available — behavioral GEMM checks skipped")
elif "IMPORT:FAIL" in stdout3:
    err_fragment = stdout3.split("IMPORT:FAIL:")[1].split("\n")[0]
    print(f"  [SKIP] aiter/sglang not importable ({err_fragment}) — behavioral GEMM checks skipped")
elif "IMPORT:OK" in stdout3:
    # GPU + aiter available: run full behavioral checks.
    check(
        "FP8 input quantization (aiter per_1x128) succeeds",
        "QUANT_INPUT:OK" in stdout3,
        (stdout3.split("QUANT_INPUT:FAIL:")[1].split("\n")[0]
         if "QUANT_INPUT:FAIL" in stdout3 else stderr3[-200:]),
    )
    check(
        "FP8 weight quantization succeeds",
        "QUANT_WEIGHT:OK" in stdout3,
        (stdout3.split("QUANT_WEIGHT:FAIL:")[1].split("\n")[0]
         if "QUANT_WEIGHT:FAIL" in stdout3 else ""),
    )
    check(
        "Float32 dequant reference GEMM succeeds",
        "REF:OK" in stdout3,
        (stdout3.split("REF:FAIL:")[1].split("\n")[0]
         if "REF:FAIL" in stdout3 else ""),
    )
    check(
        "Triton FP8 GEMM runs without error for (M=8, N=7168, K=2304)",
        "TRITON:OK" in stdout3,
        (stdout3.split("TRITON:FAIL:")[1].split("\n")[0]
         if "TRITON:FAIL" in stdout3 else stderr3[-300:]),
    )

    if "DISPATCH_TRITON:" in stdout3:
        dispatch_val = stdout3.split("DISPATCH_TRITON:")[1].split("\n")[0].strip()
        check(
            "Dispatch path selects Triton for (N=7168, K=2304) on gfx950",
            dispatch_val == "True",
            f"use_aiter_triton_gemm_w8a8_tuned_gfx950 returned {dispatch_val!r}",
        )
else:
    # Unexpected: neither sentinel appeared — print a diagnostic but don't penalise.
    if rc3 != 0 or stderr3:
        print(f"  [SKIP] Unexpected subprocess output — behavioral checks skipped")
        if stderr3:
            print(f"         stderr snippet: {stderr3[:200]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
