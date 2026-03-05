#!/usr/bin/env python3
"""Test harness for vllm-encoder-rocm.

Bug: RocmAttentionImpl raises NotImplementedError for ENCODER attention.
Test: Try to call the init with ENCODER type -- unfixed code rejects it.
"""
import sys
sys.path.insert(0, "/workspace/vllm")

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

print("=" * 60)
print("vllm-encoder-rocm test harness")
print("=" * 60)

import importlib.util

# Load modules
attn_type_mod = None
rocm_mod = None
aiter_mod = None

try:
    spec = importlib.util.spec_from_file_location(
        "backend", "/workspace/vllm/vllm/v1/attention/backend.py")
    attn_type_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attn_type_mod)
    AttentionType = attn_type_mod.AttentionType
    check("Import AttentionType", True)
except Exception as e:
    check("Import AttentionType", False, str(e)[:150])

try:
    spec2 = importlib.util.spec_from_file_location(
        "rocm_attn", "/workspace/vllm/vllm/v1/attention/backends/rocm_attn.py")
    rocm_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(rocm_mod)
    check("Import rocm_attn", True)
except Exception as e:
    check("Import rocm_attn", False, str(e)[:150])

try:
    spec3 = importlib.util.spec_from_file_location(
        "rocm_aiter",
        "/workspace/vllm/vllm/v1/attention/backends/rocm_aiter_unified_attn.py")
    aiter_mod = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(aiter_mod)
    check("Import rocm_aiter_unified_attn", True)
except Exception as e:
    check("Import rocm_aiter_unified_attn", False, str(e)[:150])

# Check: RocmAttentionImpl.__init__ accepts ENCODER type
# The unfixed code raises NotImplementedError in __init__ for non-DECODER types.
if rocm_mod and attn_type_mod:
    try:
        # The __init__ checks attn_type and raises NotImplementedError
        # We can test this by reading the source of __init__ to see if ENCODER
        # is in the allowed list. This is more robust than just checking forward().
        import inspect
        init_src = inspect.getsource(rocm_mod.RocmAttentionImpl.__init__)
        # In the unfixed code: if attn_type not in [DECODER, ENCODER_DECODER]: raise
        # In the fixed code: ENCODER should be in the allowed list
        if "NotImplementedError" in init_src:
            # Find the list of allowed types before NotImplementedError
            # The pattern is: if attn_type not in [...]: raise NotImplementedError
            lines = init_src.split("\n")
            for i, line in enumerate(lines):
                if "NotImplementedError" in line:
                    # Look at the preceding lines for the allowed list
                    context = "\n".join(lines[max(0, i-5):i+1])
                    encoder_allowed = "ENCODER" in context and "ENCODER_DECODER" != context.strip()
                    # More precisely: check if AttentionType.ENCODER is in the not-in list
                    # or if the check has been removed/modified to accept ENCODER
                    check("RocmAttentionImpl accepts ENCODER type",
                          "ENCODER" in context and context.count("ENCODER") >= 2,
                          "ENCODER not in allowed attention types")
                    break
            else:
                # No NotImplementedError found at all -- the check was removed entirely
                check("RocmAttentionImpl accepts ENCODER type", True)
        else:
            check("RocmAttentionImpl accepts ENCODER type", True)
    except Exception as e:
        check("RocmAttentionImpl accepts ENCODER type", False, str(e)[:150])

# Same check for AITER unified backend
if aiter_mod and attn_type_mod:
    try:
        # Find the impl class
        impl_cls = None
        for name in dir(aiter_mod):
            obj = getattr(aiter_mod, name)
            if isinstance(obj, type) and "Impl" in name and "Attn" in name:
                impl_cls = obj
                break

        if impl_cls:
            init_src = inspect.getsource(impl_cls.__init__)
            if "NotImplementedError" in init_src:
                lines = init_src.split("\n")
                for i, line in enumerate(lines):
                    if "NotImplementedError" in line:
                        context = "\n".join(lines[max(0, i-5):i+1])
                        check(f"{impl_cls.__name__} accepts ENCODER type",
                              "ENCODER" in context and context.count("ENCODER") >= 2,
                              "ENCODER not in allowed attention types")
                        break
                else:
                    check(f"{impl_cls.__name__} accepts ENCODER type", True)
            else:
                check(f"{impl_cls.__name__} accepts ENCODER type", True)
        else:
            check("AITER unified impl accepts ENCODER", False, "No Impl class found")
    except Exception as e:
        check("AITER unified impl accepts ENCODER", False, str(e)[:150])

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
