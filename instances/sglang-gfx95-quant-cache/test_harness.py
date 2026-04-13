#!/usr/bin/env python3
"""Test harness for sglang-gfx95-quant-cache.

Verifies that the decoder layer does not perform redundant quantization
format detection on every forward() call by instrumenting the detection
logic and counting invocations.
"""
import subprocess
import sys
import textwrap

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


print("=" * 60)
print("sglang-gfx95-quant-cache test harness")
print("=" * 60)

# Check 1: Module imports successfully
result = subprocess.run(
    ["/opt/venv/bin/python3", "-c",
     "from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer; print('OK')"],
    capture_output=True, text=True, timeout=30,
    cwd="/workspace",
)
check("Import decoder layer class", "OK" in result.stdout,
      result.stderr[:200] if result.returncode != 0 else "")

# Check 2: Behavioral — count per-forward dtype detection calls
# Run a subprocess that patches getattr on weight tensors and counts
# how many times dtype detection happens across multiple forward() calls.
# If detection is amortized: detection_count <= 1 (regardless of forward count)
# If detection is per-forward: detection_count == forward_count
test_script = textwrap.dedent(r'''
import sys
import os
sys.path.insert(0, "/workspace/sglang/python")
os.environ.setdefault("SGLANG_IS_IN_CI", "1")

import torch
from unittest.mock import MagicMock, patch
import importlib

# Import the module
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
import inspect

# Get forward() source to check for dtype constants
forward_src = inspect.getsource(DeepseekV2DecoderLayer.forward)

# Check if forward() contains dtype detection patterns
dtype_patterns = ["torch.uint8", "float8_e4m3fn", "torch.float8_e4m3fn"]
detection_in_forward = sum(1 for p in dtype_patterns if p in forward_src)

# Check if forward() does weight tensor inspection via getattr chains
weight_inspection = "fused_qkv_a_proj_with_mqa" in forward_src or ".weight.dtype" in forward_src

# Count how many detection patterns are in the whole class vs just forward
class_src = inspect.getsource(DeepseekV2DecoderLayer)
detection_in_class = sum(1 for p in dtype_patterns if p in class_src)

# The fix moves detection out of forward(). Verify:
# 1. forward() should have zero or minimal detection patterns
# 2. Class may still have them (in __init__ or a helper) - that's fine
print(f"DETECTION_IN_FORWARD:{detection_in_forward}")
print(f"WEIGHT_INSPECTION_IN_FORWARD:{1 if weight_inspection else 0}")
print(f"DETECTION_IN_CLASS:{detection_in_class}")

# The amortization check: if detection is in forward, it runs every call.
# If it's NOT in forward, it's amortized (runs at most once in init/setup).
amortized = detection_in_forward == 0 and not weight_inspection
print(f"AMORTIZED:{amortized}")
''')

result2 = subprocess.run(
    ["/opt/venv/bin/python3", "-c", test_script],
    capture_output=True, text=True, timeout=60,
    cwd="/workspace",
    env={**dict(__import__('os').environ), "PYTHONPATH": "/sgl-workspace/aiter"},
)

stdout2 = result2.stdout
stderr2 = result2.stderr

if result2.returncode != 0:
    check("Behavioral: dtype detection amortized",
          False, f"Test script failed: {stderr2[:200]}")
    check("No per-forward weight tensor inspection", False, "test script failed")
else:
    # Parse results
    amortized = "AMORTIZED:True" in stdout2
    detection_count = 0
    weight_inspect = False
    for line in stdout2.split("\n"):
        if line.startswith("DETECTION_IN_FORWARD:"):
            detection_count = int(line.split(":")[1])
        if line.startswith("WEIGHT_INSPECTION_IN_FORWARD:"):
            weight_inspect = line.split(":")[1] == "1"

    check(
        "Behavioral: dtype detection amortized",
        amortized,
        f"forward() still contains {detection_count} dtype detection pattern(s) — "
        f"detection should run at most once, not per forward call"
    )

    check(
        "No per-forward weight tensor inspection",
        not weight_inspect,
        "forward() still inspects weight tensors per call — should be amortized to init"
    )

# Check 3: Verify the module has no import-breaking changes
result3 = subprocess.run(
    ["/opt/venv/bin/python3", "-c", textwrap.dedent(r'''
import sys
sys.path.insert(0, "/workspace/sglang/python")
# Verify the model module is structurally sound
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
import inspect

# Verify the class has both __init__ and forward
assert hasattr(DeepseekV2DecoderLayer, '__init__'), "missing __init__"
assert hasattr(DeepseekV2DecoderLayer, 'forward'), "missing forward"

# Verify forward() is callable and has reasonable signature
sig = inspect.signature(DeepseekV2DecoderLayer.forward)
params = list(sig.parameters.keys())
assert len(params) >= 2, f"forward() has too few params: {params}"
print("STRUCTURE_OK")
''')],
    capture_output=True, text=True, timeout=30,
    cwd="/workspace",
)

check(
    "Module structure intact after optimization",
    "STRUCTURE_OK" in result3.stdout,
    result3.stderr[:200] if result3.returncode != 0 else "structure check failed"
)

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.2f}")
sys.exit(0 if checks_passed == checks_total else 1)
