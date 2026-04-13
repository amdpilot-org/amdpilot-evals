#!/usr/bin/env python3
"""Test harness for sglang-gfx95-quant-cache.

Verifies that the decoder layer does not perform redundant quantization
format detection on every forward() call.
"""
import inspect
import re
import sys
import time

sys.path.insert(0, "/workspace/sglang/python")

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
try:
    from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
    check("Import decoder layer class", True)
except Exception as e:
    check("Import decoder layer class", False, str(e)[:200])
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Check 2: Get forward() source and verify no per-call dtype detection
try:
    forward_src = inspect.getsource(DeepseekV2DecoderLayer.forward)
except Exception as e:
    check("Get forward() source", False, str(e)[:200])
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# The expensive detection pattern: checking weight tensor dtypes inside forward()
# These dtype constants appear when detection runs per-call
dtype_detection_patterns = [
    r"torch\.uint8",
    r"float8_e4m3fn",
    r"torch\.float8_e4m3fn",
]

detection_in_forward = any(
    re.search(pat, forward_src) for pat in dtype_detection_patterns
)

check(
    "No per-forward dtype detection",
    not detection_in_forward,
    "forward() still contains dtype constant checks — detection should be amortized"
)

# Check 3: forward() should not perform weight tensor inspection per call
# getattr chains walking into weight tensors are the expensive detection mechanism
weight_inspection = bool(re.search(
    r"getattr.*proj.*weight|\.weight\.dtype",
    forward_src,
))

check(
    "No per-forward weight tensor inspection",
    not weight_inspection,
    "forward() still walks into weight tensors — should be amortized"
)

# Check 4: Behavioral — instrument the class to verify detection is amortized
# If we can instantiate enough of the module to test, do so
behavioral_ok = False
try:
    init_src = inspect.getsource(DeepseekV2DecoderLayer.__init__)

    # Verify that __init__ or another setup path handles the detection
    # We check that SOME amortization mechanism exists in the class
    # (could be in __init__, a property, a post-init hook, etc.)
    class_src = inspect.getsource(DeepseekV2DecoderLayer)

    # The class should reference the detection logic somewhere outside forward()
    # This is a weak structural check — the behavioral proof is checks 2-3 above
    has_setup_detection = any(
        re.search(pat, class_src) and not re.search(pat, forward_src)
        for pat in dtype_detection_patterns
    )

    # Alternative: the detection was moved to a one-time helper/property
    has_cached_attr = bool(re.search(r"self\.\w+format\w*\s*=|self\.\w+quant\w*\s*=|self\.\w+dtype_cache\w*\s*=", class_src))

    behavioral_ok = has_setup_detection or has_cached_attr or not detection_in_forward
    check(
        "Detection amortized outside forward path",
        behavioral_ok,
        "Could not verify that detection is amortized to init/setup"
    )
except Exception as e:
    check(
        "Detection amortized outside forward path",
        not detection_in_forward,  # If forward is clean, accept
        f"Could not inspect class: {str(e)[:100]}"
    )

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.2f}")
sys.exit(0 if checks_passed == checks_total else 1)
