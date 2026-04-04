#!/usr/bin/env python3
"""Test harness for vllm-mxfp4-moe-fallback (PR #35893).

Bug: CK MXFP4 MoE GEMM kernels crash with RuntimeError when intermediate_size
per partition is not a multiple of 256 (e.g. MiniMax-M2.1 TP=4 → 384).
Test: Verify dimension validation and fallback logic exist in the quantization
code to handle incompatible dimensions gracefully.
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


def run_test(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-mxfp4-moe-fallback test harness")
print("=" * 60)

# Check 1: vllm mxfp4 config can be imported (basic sanity)
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
print("IMPORT:OK")
""")
check("MXFP4 quantization config imports", "IMPORT:OK" in stdout, stderr[:200])

# Check 2: CK_MXFP4_MOE_DIM_ALIGNMENT constant exists in the utils source.
# The fix introduces this constant; without it, incompatible dims crash CK.
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
path = "/workspace/vllm/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py"
with open(path) as f:
    src = f.read()
has_const = "CK_MXFP4_MOE_DIM_ALIGNMENT" in src and "= 256" in src
print(f"HAS_ALIGNMENT_CONST:{has_const}")
""")
check("CK_MXFP4_MOE_DIM_ALIGNMENT=256 defined in mxfp4_utils.py",
      "HAS_ALIGNMENT_CONST:True" in stdout,
      "missing dimension validation constant — CK kernel will crash on misaligned dims")

# Check 3: mxfp4.py validates dimensions and has Triton fallback
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
path = "/workspace/vllm/vllm/model_executor/layers/quantization/mxfp4.py"
with open(path) as f:
    src = f.read()
has_import = "CK_MXFP4_MOE_DIM_ALIGNMENT" in src
has_check = "% CK_MXFP4_MOE_DIM_ALIGNMENT" in src
has_fallback = "Mxfp4Backend.TRITON" in src and "alignment" in src.lower()
print(f"MXFP4_IMPORT:{has_import}")
print(f"MXFP4_MODCHECK:{has_check}")
print(f"MXFP4_FALLBACK:{has_fallback}")
""")
check("mxfp4.py validates CK dimension alignment and falls back to Triton",
      all(x in stdout for x in ["MXFP4_IMPORT:True", "MXFP4_MODCHECK:True", "MXFP4_FALLBACK:True"]),
      "missing dimension check or Triton fallback in Mxfp4MoEMethod")

# Check 4: quark_moe.py has emulation fallback for incompatible dims
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
path = "/workspace/vllm/vllm/model_executor/layers/quantization/quark/quark_moe.py"
with open(path) as f:
    src = f.read()
has_import = "CK_MXFP4_MOE_DIM_ALIGNMENT" in src
has_check = "% CK_MXFP4_MOE_DIM_ALIGNMENT" in src
has_emulate = "self.emulate = True" in src
print(f"QUARK_IMPORT:{has_import}")
print(f"QUARK_MODCHECK:{has_check}")
print(f"QUARK_EMULATE:{has_emulate}")
""")
check("quark_moe.py validates dims and falls back to emulation mode",
      all(x in stdout for x in ["QUARK_IMPORT:True", "QUARK_MODCHECK:True", "QUARK_EMULATE:True"]),
      "missing dimension check or emulation fallback in QuarkOCP_MX_MoEMethod")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
