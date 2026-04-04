#!/usr/bin/env python3
"""Test harness for aiter-mxfp4-rounding-fix. Behavioral tests only.

Bug: The Triton kernel _dynamic_mxfp4_quant_kernel_asm_layout in fp4_utils.py
uses incorrect rounding logic for FP4 E2M1 quantization. The old rounding
(tie-breaking up) produces values that differ from the correct round-to-nearest-
even algorithm, especially for denormals and boundary values.
Test: Compare dynamic_mxfp4_quant output against a known-correct reference.
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
    return condition


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("aiter-mxfp4-rounding-fix test harness")
print("=" * 60)

# Test: Compare fp4_utils.dynamic_mxfp4_quant against reference implementation
# The reference uses correct round-to-nearest-even; the buggy kernel doesn't.
test_script = """
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, "/sgl-workspace/aiter")
import torch

# Reference implementation (correct round-to-nearest-even)
def ref_dynamic_mxfp4_quant(x):
    BLOCK = 32; BIAS32 = 127; BIAS4 = 1; MB32 = 23; MB4 = 1; max_n = 6; min_n = 1
    sign_mask = 1 << 3
    shape = x.shape
    if shape[-1] % BLOCK != 0:
        s = list(shape); s[-1] = ((s[-1]-1+BLOCK)//BLOCK)*BLOCK
        xp = torch.zeros(s, device=x.device, dtype=x.dtype); xp[...,:shape[-1]] = x
    else:
        xp = x
    xp = xp.reshape(-1, xp.shape[-1]//BLOCK, BLOCK).to(torch.float32)
    amax, _ = torch.max(torch.abs(xp), dim=-1)
    amax = amax.view(torch.int32); amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    se = torch.log2(amax).floor() - 2
    se = torch.clamp(se, min=-127, max=127)
    qs = torch.exp2(-se)
    qx = xp * qs.unsqueeze(-1)
    bs = se.to(torch.uint8) + 127
    qx = qx.view(torch.int32)
    s = qx & 0x80000000; qx = qx ^ s
    qf = qx.view(torch.float32)
    sat = qf >= max_n; den = (~sat) & (qf < min_n); nor = ~(sat | den)
    de = (BIAS32 - BIAS4) + (MB32 - MB4) + 1
    dmi = de << MB32; dmf = torch.tensor(dmi, dtype=torch.int32).view(torch.float32)
    dx = qf + dmf; dx = dx.view(torch.int32); dx -= dmi; dx = dx.to(torch.uint8)
    nx = qx; mo = (nx >> (MB32 - MB4)) & 1
    va = ((BIAS4 - BIAS32) << MB32) + (1 << 21) - 1
    nx += va; nx += mo; nx = nx >> (MB32 - MB4); nx = nx.to(torch.uint8)
    e2 = torch.full_like(qx, 0x7, dtype=torch.uint8)
    e2 = torch.where(nor, nx, e2); e2 = torch.where(den, dx, e2)
    sl = s >> (MB32 + 8 - MB4 - 2); sl = sl.to(torch.uint8); sl = sl & sign_mask
    e2 = e2 | sl
    fp4 = e2[..., ::2] | (e2[..., 1::2] << 4)
    fp4 = torch.flatten(fp4, -2, -1)
    if shape[-1] % BLOCK != 0:
        fp4 = fp4[..., :shape[-1]//2]
    ms = list(shape); ms[-1] = ms[-1]//2
    return fp4.reshape(ms), bs

# Import the function under test
try:
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant as fp4_quant
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)

shapes = [(1, 32), (2, 64), (128, 32), (1, 68), (256, 32)]
mismatches = 0
total = 0

for M, N in shapes:
    torch.manual_seed(20)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    try:
        fp4_out, fp4_scale = fp4_quant(x)
        ref_out, ref_scale = ref_dynamic_mxfp4_quant(x)
        fp4_bytes = fp4_out.view(torch.uint8).cpu()
        ref_bytes = ref_out.view(torch.uint8).cpu()
        scale_match = torch.equal(fp4_scale.view(torch.uint8).cpu(), ref_scale.cpu())
        data_match = torch.equal(fp4_bytes, ref_bytes)
        total += 1
        if not (scale_match and data_match):
            mismatches += 1
            ndiff = (fp4_bytes != ref_bytes).sum().item()
            print(f"SHAPE_{M}x{N}:MISMATCH:ndiff={ndiff}")
        else:
            print(f"SHAPE_{M}x{N}:MATCH")
    except Exception as e:
        print(f"SHAPE_{M}x{N}:ERROR:{type(e).__name__}:{str(e)[:200]}")
        total += 1
        mismatches += 1

print(f"TOTAL:{total}")
print(f"MISMATCHES:{mismatches}")
"""

stdout, stderr, rc = run_test(test_script, timeout=180)

if "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].split("\n")[0]
    check("dynamic_mxfp4_quant import", False, err)
    print(f"\nSCORE: 0.0")
    sys.exit(1)

check("dynamic_mxfp4_quant import", "IMPORT:OK" in stdout)

if rc < 0:
    check("Quantization runs without crash", False, f"Signal {-rc}")
else:
    check("Quantization runs without crash", True)

shapes_tested = 0
shapes_matched = 0
for line in stdout.splitlines():
    if line.startswith("SHAPE_"):
        shapes_tested += 1
        if ":MATCH" in line:
            shapes_matched += 1

check("All shapes tested", shapes_tested >= 4,
      f"Only {shapes_tested} shapes tested")

check("All shapes match reference (correct rounding)",
      shapes_tested > 0 and shapes_matched == shapes_tested,
      f"{shapes_matched}/{shapes_tested} matched")

for line in stdout.splitlines():
    if ":MISMATCH:" in line:
        check(f"{line.split(':')[0]} matches reference", False,
              line.split("MISMATCH:")[1])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
