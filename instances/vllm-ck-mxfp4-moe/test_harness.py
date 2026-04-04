#!/usr/bin/env python3
"""Test harness for vllm-ck-mxfp4-moe (PR #34301). Behavioral tests only.

Feature: MXFP4 quantization lacks fused MoE support on ROCm via CK backend.
Test: Verify CK backend enum, utility functions, and new methods work correctly
by calling real functions with real inputs and checking return values.
"""
import sys
import subprocess

checks_passed = 0
checks_total = 0


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


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-ck-mxfp4-moe test harness")
print("=" * 60)

# Test 1: Mxfp4Backend enum has CK variant (import + attribute access, not source grep)
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Backend
    if hasattr(Mxfp4Backend, 'CK'):
        ck = Mxfp4Backend.CK
        triton = Mxfp4Backend.TRITON
        print(f"CK_VALUE:{ck.value}")
        print(f"TRITON_VALUE:{triton.value}")
        print(f"CK_NAME:{ck.name}")
    else:
        print("CK_MISSING")
except Exception as e:
    print(f"IMPORT_ERROR:{type(e).__name__}:{e}")
""")

if "CK_VALUE:" in stdout:
    ck_val = int(stdout.split("CK_VALUE:")[1].split("\n")[0])
    triton_val = int(stdout.split("TRITON_VALUE:")[1].split("\n")[0])
    check("Mxfp4Backend.CK enum exists", True)
    check("CK backend has higher value than TRITON", ck_val > triton_val,
          f"CK={ck_val}, TRITON={triton_val}")
elif "CK_MISSING" in stdout:
    check("Mxfp4Backend.CK enum exists", False, "CK variant not in enum")
    check("CK backend has higher value than TRITON", False, "CK missing")
else:
    err = (stderr or stdout)[:200]
    check("Mxfp4Backend.CK enum exists", False, f"Import error: {err}")
    check("CK backend has higher value than TRITON", False, "Import failed")

# Test 2: get_aiter_activation_type returns correct mappings (behavioral)
# This function maps string activation names to aiter ActivationType enums.
# Correct implementation requires importing from aiter and mapping known names.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm._aiter_ops import rocm_aiter_ops

    fn = getattr(rocm_aiter_ops, 'get_aiter_activation_type', None)
    if fn is None:
        print("FUNC_MISSING")
    else:
        for name in ["silu", "gelu", "swiglu", "none", "no"]:
            val = fn(name)
            print(f"ACT_{name}:{val}")
        # Invalid inputs must return None
        print(f"ACT_invalid:{fn('invalid_activation')}")
        print(f"ACT_nonstr:{fn(42)}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
""")

if "FUNC_MISSING" in stdout:
    check("get_aiter_activation_type exists", False, "function not found on rocm_aiter_ops")
    check("Activation type mappings are correct", False, "function missing")
elif "ERROR:" in stdout:
    check("get_aiter_activation_type exists", False, stdout.split("ERROR:")[1][:200])
    check("Activation type mappings are correct", False, "error occurred")
else:
    check("get_aiter_activation_type exists", True)
    # Valid names must return non-None (actual aiter enum values)
    silu_ok = "ACT_silu:" in stdout and "ACT_silu:None" not in stdout
    gelu_ok = "ACT_gelu:" in stdout and "ACT_gelu:None" not in stdout
    swiglu_ok = "ACT_swiglu:" in stdout and "ACT_swiglu:None" not in stdout
    # Invalid inputs must return None
    invalid_none = "ACT_invalid:None" in stdout
    nonstr_none = "ACT_nonstr:None" in stdout
    all_ok = silu_ok and gelu_ok and swiglu_ok and invalid_none and nonstr_none
    check("Activation type mappings are correct",
          all_ok,
          f"silu={silu_ok}, gelu={gelu_ok}, swiglu={swiglu_ok}, "
          f"invalid_none={invalid_none}, nonstr_none={nonstr_none}")

# Test 3: get_aiter_quant_type returns correct mappings (behavioral)
# Maps string quant type names to aiter QuantType enums.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm._aiter_ops import rocm_aiter_ops

    fn = getattr(rocm_aiter_ops, 'get_aiter_quant_type', None)
    if fn is None:
        print("FUNC_MISSING")
    else:
        for name in ["no", "per_tensor", "per_token", "per_1x32", "per_1x128", "per_128x128"]:
            val = fn(name)
            print(f"QT_{name}:{val}")
        print(f"QT_invalid:{fn('invalid_quant')}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
""")

if "FUNC_MISSING" in stdout:
    check("get_aiter_quant_type exists", False, "function not found on rocm_aiter_ops")
    check("Quant type mappings are correct", False, "function missing")
elif "ERROR:" in stdout:
    check("get_aiter_quant_type exists", False, stdout.split("ERROR:")[1][:200])
    check("Quant type mappings are correct", False, "error occurred")
else:
    check("get_aiter_quant_type exists", True)
    per_1x32_ok = "QT_per_1x32:" in stdout and "QT_per_1x32:None" not in stdout
    per_tensor_ok = "QT_per_tensor:" in stdout and "QT_per_tensor:None" not in stdout
    per_128x128_ok = "QT_per_128x128:" in stdout and "QT_per_128x128:None" not in stdout
    invalid_none = "QT_invalid:None" in stdout
    all_ok = per_1x32_ok and per_tensor_ok and per_128x128_ok and invalid_none
    check("Quant type mappings are correct",
          all_ok,
          f"per_1x32={per_1x32_ok}, per_tensor={per_tensor_ok}, "
          f"per_128x128={per_128x128_ok}, invalid_none={invalid_none}")

# Test 4: fused_topk method exists on rocm_aiter_ops and is callable
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm._aiter_ops import rocm_aiter_ops
    ft = getattr(rocm_aiter_ops, 'fused_topk', None)
    sw = getattr(rocm_aiter_ops, 'shuffle_weight_a16w4', None)
    ss = getattr(rocm_aiter_ops, 'shuffle_scale_a16w4', None)
    print(f"FUSED_TOPK:{'OK' if ft and callable(ft) else 'MISSING'}")
    print(f"SHUFFLE_WEIGHT:{'OK' if sw and callable(sw) else 'MISSING'}")
    print(f"SHUFFLE_SCALE:{'OK' if ss and callable(ss) else 'MISSING'}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{e}")
""")

check("fused_topk method callable on rocm_aiter_ops",
      "FUSED_TOPK:OK" in stdout,
      "fused_topk not found or not callable")
check("shuffle_weight_a16w4 method callable",
      "SHUFFLE_WEIGHT:OK" in stdout,
      "shuffle_weight_a16w4 not found or not callable")
check("shuffle_scale_a16w4 method callable",
      "SHUFFLE_SCALE:OK" in stdout,
      "shuffle_scale_a16w4 not found or not callable")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
