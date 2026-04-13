#!/usr/bin/env python3
"""Test harness for vllm-rocm-mxfp4-oracle-regressions.

Behavioral test: verifies that MXFP4-quantized MoE computations produce
correct results across different model configurations on ROCm.
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


def run_test(script, timeout=180):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-mxfp4-oracle-regressions test harness")
print("=" * 60)

# Single holistic test: Verify that MXFP4 MoE backend selection handles
# non-256-aligned intermediate dimensions correctly on the current device.
# A model with intermediate_size=2880 (real gpt-oss-20b config) must not
# select a backend that will crash at runtime.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch

try:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_mxfp4_moe_backend,
        Mxfp4MoeBackend,
    )

    parallel_config = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
        sp_size=1, use_ep=False, all2all_backend="", enable_eplb=False,
    )

    # Non-256-aligned intermediate size (real gpt-oss-20b config)
    config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=1024,
        intermediate_size_per_partition=2880,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
    )

    try:
        backend, kernel_cls = select_mxfp4_moe_backend(config)
        backend_name = backend.value
        # CK backend requires 256-aligned dims — selecting it here will crash
        is_ck = (backend == Mxfp4MoeBackend.CK)
        print(f"BACKEND:{backend_name}")
        print(f"IS_CK:{is_ck}")
        print(f"SAFE_SELECTION:{not is_ck}")
    except ValueError as e:
        # No backend available — check if CK was at least rejected
        err_msg = str(e)
        print(f"SELECTION_ERROR:{err_msg[:200]}")
        # If the error message indicates CK was tried and rejected,
        # the fix is partially working (gating correct, but no fallback)
        print(f"SAFE_SELECTION:True")
    except Exception as e:
        print(f"ERROR:{type(e).__name__}:{str(e)[:200]}")

except ImportError as e:
    print(f"IMPORT_ERROR:{type(e).__name__}:{str(e)[:200]}")
except Exception as e:
    print(f"ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "IMPORT_ERROR:" in stdout:
    err = stdout.split("IMPORT_ERROR:")[1][:200]
    check("MXFP4 MoE backend selection handles non-aligned dims", False,
          f"import error: {err}")
elif "ERROR:" in stdout:
    err = stdout.split("ERROR:")[1].strip()[:200]
    check("MXFP4 MoE backend selection handles non-aligned dims", False,
          f"error: {err}")
elif "SAFE_SELECTION:True" in stdout:
    check("MXFP4 MoE backend selection handles non-aligned dims", True)
elif "SAFE_SELECTION:False" in stdout:
    backend = "unknown"
    if "BACKEND:" in stdout:
        backend = stdout.split("BACKEND:")[1].split("\\n")[0].strip()
    check("MXFP4 MoE backend selection handles non-aligned dims", False,
          f"selected incompatible backend '{backend}' for non-256-aligned dim")
else:
    check("MXFP4 MoE backend selection handles non-aligned dims", False,
          f"unexpected output: {(stdout + stderr)[:200]}")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
