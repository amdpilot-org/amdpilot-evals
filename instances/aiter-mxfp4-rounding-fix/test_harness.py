#!/usr/bin/env python3
"""Test harness for aiter MXFP4 quantization accuracy.

Tests (behavioral):
  Compare dynamic_mxfp4_quant kernel output against precomputed reference
  fixtures for multiple input shapes.
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

# Test: Compare fp4_utils.dynamic_mxfp4_quant against precomputed reference fixtures
test_script = """
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
sys.path.insert(0, "/sgl-workspace/aiter")
import torch

# Load precomputed reference fixtures
fixtures = torch.load("/workspace/fixtures/mxfp4_fixtures.pt", weights_only=False)

# Import the function under test
try:
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant as fp4_quant
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
    sys.exit(0)

mismatches = 0
total = 0

for key, data in sorted(fixtures.items()):
    x_cpu = data["input"]
    expected_fp4 = data["fp4"]
    expected_scale = data["scale"]

    x = x_cpu.to("cuda")
    try:
        fp4_out, fp4_scale = fp4_quant(x)
        fp4_bytes = fp4_out.view(torch.uint8).cpu()
        scale_match = torch.equal(fp4_scale.view(torch.uint8).cpu(),
                                  expected_scale.view(torch.uint8))
        data_match = torch.equal(fp4_bytes, expected_fp4)
        total += 1
        if not (scale_match and data_match):
            mismatches += 1
            ndiff = (fp4_bytes != expected_fp4).sum().item()
            print(f"SHAPE_{key}:MISMATCH:ndiff={ndiff}")
        else:
            print(f"SHAPE_{key}:MATCH")
    except Exception as e:
        print(f"SHAPE_{key}:ERROR:{type(e).__name__}:{str(e)[:200]}")
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
