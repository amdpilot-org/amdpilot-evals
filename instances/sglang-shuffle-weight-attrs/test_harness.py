#!/usr/bin/env python3
"""Test harness for sglang-shuffle-weight-attrs (PR #21825).

Bug: In the MoE weight loading module, weight shuffling uses direct
     torch.nn.Parameter() reassignment, which silently drops any custom
     attributes (e.g., weight_loader) that were set on the original parameter.

Expected behavior after fix: Custom parameter attributes survive through the
     weight shuffling step. The parameter data is updated, but the attributes
     are preserved on the parameter object.
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


def run_test(script, timeout=90):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("sglang-shuffle-weight-attrs test harness")
print("=" * 60)

# Test: BEHAVIORAL — after MoE weight shuffling, custom parameter attributes
# (such as weight_loader) must be preserved on w13_weight and w2_weight.
#
# Strategy:
# 1. Import the MoE weight loading module and the relevant class.
# 2. Inject a CPU-compatible mock for shuffle_weight (no GPU required).
# 3. Patch the module to force the aiter shuffle code path.
# 4. Create a fake MoE layer with w13_weight and w2_weight, each carrying
#    a sentinel custom attribute.
# 5. Call process_weights_after_loading and verify the attribute survives.
#
# Pre-fix: torch.nn.Parameter() reassignment drops custom attributes → FAIL.
# Post-fix: attribute-preserving rebind keeps custom attributes → PASS.
# IMPORT_SKIP → explicit FAIL (fix must be importable).
stdout, stderr, rc = run_test("""
import sys
sys.path.insert(0, '/workspace/sglang/python')
import torch
import torch.nn as nn

try:
    import sglang.srt.layers.quantization.unquant as unquant_mod
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoELinearMethod
except ImportError as e:
    print(f"IMPORT_SKIP:{e}")
    sys.exit(0)
except Exception as e:
    print(f"IMPORT_SKIP:{type(e).__name__}:{e}")
    sys.exit(0)

# Inject a CPU-compatible mock shuffle — identity transform, no GPU needed.
def _mock_shuffle(weight, config):
    return weight.clone()

# Force the aiter shuffle code path regardless of env vars.
unquant_mod._use_aiter = True
unquant_mod.shuffle_weight = _mock_shuffle

class _MockMoEBackend:
    def is_auto(self):
        return True

unquant_mod.get_moe_runner_backend = lambda: _MockMoEBackend()

# Create a fake MoE layer with custom attributes on both weight parameters.
class _FakeMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        w13 = nn.Parameter(torch.zeros(8, 4), requires_grad=False)
        w13.weight_loader = "sentinel_w13"
        self.w13_weight = w13

        w2 = nn.Parameter(torch.zeros(4, 8), requires_grad=False)
        w2.weight_loader = "sentinel_w2"
        self.w2_weight = w2

layer = _FakeMoELayer()

# Run the actual weight loading / shuffle code path.
try:
    method = UnquantizedFusedMoELinearMethod()
    method.process_weights_after_loading(layer)
    print("SHUFFLE_OK")
except Exception as e:
    print(f"SHUFFLE_ERROR:{type(e).__name__}:{e}")
    sys.exit(0)

# Behavioral check: custom attributes must survive on both parameters.
w13_attr = getattr(layer.w13_weight, "weight_loader", None)
w2_attr  = getattr(layer.w2_weight,  "weight_loader", None)

print(f"W13_ATTR:{w13_attr}")
print(f"W2_ATTR:{w2_attr}")
""")

if "IMPORT_SKIP" in stdout:
    err = stdout.split("IMPORT_SKIP:")[1].split("\\n")[0].strip()
    check("MoE weight shuffling preserves w13_weight custom attributes", False,
          f"Import failed: {err}")
    check("MoE weight shuffling preserves w2_weight custom attributes", False,
          "Import failed")
elif "SHUFFLE_ERROR" in stdout:
    err = stdout.split("SHUFFLE_ERROR:")[1].split("\n")[0].strip()
    check("MoE weight shuffling preserves w13_weight custom attributes", False,
          f"process_weights_after_loading raised: {err}")
    check("MoE weight shuffling preserves w2_weight custom attributes", False,
          f"process_weights_after_loading raised: {err}")
else:
    w13_ok = "W13_ATTR:sentinel_w13" in stdout
    w2_ok  = "W2_ATTR:sentinel_w2"  in stdout

    check("MoE weight shuffling preserves w13_weight custom attributes", w13_ok,
          f"w13_weight.weight_loader lost after shuffle: {stdout.strip()[:200]}")
    check("MoE weight shuffling preserves w2_weight custom attributes", w2_ok,
          f"w2_weight.weight_loader lost after shuffle: {stdout.strip()[:200]}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
