#!/usr/bin/env python3
"""Test harness for vllm-rocm-fused-moe-fix.

Verify that the ROCm fused MoE custom op and grouped top-k path
handle all required parameters correctly.

All checks run in isolated subprocesses to prevent monkey-patching.
"""
from __future__ import annotations

import subprocess
import sys

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
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


def run_test(script: str, timeout: int = 60) -> tuple[str, str, int]:
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-fused-moe-fix test harness")
print("=" * 60)

# --- Check 1: import fused_moe-related modules ---
stdout, stderr, rc = run_test("""
import sys; sys.path.insert(0, "/workspace/vllm")
import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
import vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe as rocm_aiter_fmoe
print("IMPORT:OK")
""")
check("Import fused_moe and rocm_aiter_fused_moe modules",
      "IMPORT:OK" in stdout,
      (stderr[:200] if rc != 0 else ""))

# --- Check 2: fake vs impl signature for ROCm AITER fused_moe custom op ---
stdout, stderr, rc = run_test("""
import sys; sys.path.insert(0, "/workspace/vllm")
import inspect
import vllm._aiter_ops as aiter_ops

fake = aiter_ops._rocm_aiter_fused_moe_fake
impl = aiter_ops._rocm_aiter_fused_moe_impl

pa = list(inspect.signature(fake).parameters.items())
pb = list(inspect.signature(impl).parameters.items())

if len(pa) != len(pb):
    print(f"SIG_MATCH:FAIL:param count {len(pa)} vs {len(pb)}")
else:
    mismatch = None
    for (na, _), (nb, _) in zip(pa, pb):
        if na != nb:
            mismatch = f"param name mismatch: {na!r} vs {nb!r}"
            break
    if mismatch:
        print(f"SIG_MATCH:FAIL:{mismatch}")
    else:
        print(f"SIG_MATCH:OK:{len(pa)} params")
""")
if "SIG_MATCH:OK" in stdout:
    check("_rocm_aiter_fused_moe_fake matches _rocm_aiter_fused_moe_impl signature", True)
else:
    detail = ""
    for line in stdout.splitlines():
        if "SIG_MATCH:FAIL:" in line:
            detail = line.split("SIG_MATCH:FAIL:", 1)[1]
    if not detail and rc != 0:
        detail = stderr[:200]
    check("_rocm_aiter_fused_moe_fake matches _rocm_aiter_fused_moe_impl signature",
          False, detail)

# --- Check 3: routed_scaling_factor in ROCm AITER grouped top-k path ---
stdout, stderr, rc = run_test("""
import sys; sys.path.insert(0, "/workspace/vllm")
import inspect
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_grouped_topk,
)
sig = inspect.signature(rocm_aiter_grouped_topk)
has_param = "routed_scaling_factor" in sig.parameters
print(f"HAS_ROUTED_SCALING_FACTOR:{has_param}")
""")
check("rocm_aiter_grouped_topk accepts routed_scaling_factor",
      "HAS_ROUTED_SCALING_FACTOR:True" in stdout,
      stderr[:200] if rc != 0 else "missing parameter")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
