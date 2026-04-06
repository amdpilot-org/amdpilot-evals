#!/usr/bin/env python3
import sys
import os
import re

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

BACKEND_FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"

def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return None

def get_forward_extend_body(content):
    """Extract the forward_extend method body from the file content."""
    # Find forward_extend method
    pattern = r'def forward_extend\('
    match = re.search(pattern, content)
    if not match:
        return None
    start = match.start()
    # Find the next method at the same or lower indentation level
    # We'll just grab a large chunk after the method definition
    lines = content[start:].split('\n')
    method_lines = [lines[0]]
    if len(lines) > 1:
        # Determine the indentation of the def line
        def_indent = len(lines[0]) - len(lines[0].lstrip())
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped == '' or stripped.startswith('#'):
                method_lines.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= def_indent and stripped and not stripped.startswith('#') and not stripped.startswith('@'):
                break
            method_lines.append(line)
    return '\n'.join(method_lines)

if __name__ == "__main__":
    print("=" * 60)
    print("FP8 Prefill + Radix Cache Integration Test Harness")
    print("=" * 60)

    # Check 1: File exists
    file_exists = os.path.isfile(BACKEND_FILE)
    check("aiter_backend.py exists", file_exists, f"File not found at {BACKEND_FILE}")

    content = read_file(BACKEND_FILE) if file_exists else None
    has_content = content is not None and len(content) > 100
    check("aiter_backend.py is readable and non-empty", has_content)

    if not has_content:
        print("\nCannot proceed without readable file content.")
        score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
        print(f"\nResults: {checks_passed}/{checks_total} checks passed")
        print(f"SCORE: {score:.1f}")
        sys.exit(0 if checks_passed == checks_total else 1)

    # Extract forward_extend body
    forward_extend_body = get_forward_extend_body(content)
    has_forward_extend = forward_extend_body is not None and len(forward_extend_body) > 50
    check("forward_extend method found", has_forward_extend,
          "Could not find forward_extend method in aiter_backend.py")

    if not has_forward_extend:
        forward_extend_body = ""

    # Check 2: fused_gemm_afp4wfp4_split_cat is referenced in the file
    has_fused_gemm_import_or_ref = "fused_gemm_afp4wfp4_split_cat" in content
    check("fused_gemm_afp4wfp4_split_cat referenced in file",
          has_fused_gemm_import_or_ref,
          "fused_gemm_afp4wfp4_split_cat not found anywhere in the file")

    # Check 3: fused_gemm_afp4wfp4_split_cat is used in forward_extend
    has_fused_gemm_in_extend = "fused_gemm_afp4wfp4_split_cat" in forward_extend_body
    check("fused_gemm_afp4wfp4_split_cat used in forward_extend",
          has_fused_gemm_in_extend,
          "fused_gemm_afp4wfp4_split_cat not found in forward_extend method body")

    # Check 4: FP8 dtype handling in forward_extend
    # Look for fp8/float8 references in forward_extend
    fp8_patterns = [
        r'float8_e4m3',
        r'fp8',
        r'FP8',
        r'float8',
    ]
    has_fp8_in_extend = any(re.search(p, forward_extend_body, re.IGNORECASE) for p in fp8_patterns)
    check("FP8 dtype handling in forward_extend",
          has_fp8_in_extend,
          "No FP8 dtype references found in forward_extend")

    # Check 5: Look for radix-cache related code path with FP8
    # The radix cache path typically involves prefix/cached tokens, paged KV, etc.
    # Key indicators: use_ragged, prefix_lens, paged attention patterns, or explicit radix references
    # In forward_extend, there should be TWO paths or a unified path handling both cases
    
    # Look for patterns that indicate radix cache handling with FP8
    # Common patterns: prefix handling, cached seq handling, or explicit cache path
    radix_indicators = [
        r'prefix',
        r'cached',
        r'paged',
        r'extend_prefix',
        r'radix',
        r'use_ragged',
    ]
    has_radix_path = any(re.search(p, forward_extend_body, re.IGNORECASE) for p in radix_indicators)
    check("Radix/cache path indicators in forward_extend",
          has_radix_path,
          "No radix cache path indicators found in forward_extend")

    # Check 6: Verify FP8 prefill flag/env handling exists
    fp8_prefill_env_patterns = [
        r'FP8_PREFILL',
        r'fp8_prefill',
        r'SGLANG_AITER_FP8_PREFILL',
    ]
    has_fp8_prefill_flag = any(re.search(p, content) for p in fp8_prefill_env_patterns)
    check("FP8 prefill flag/env var handling exists in file",
          has_fp8_prefill_flag,
          "No FP8 prefill flag or env var handling found")

    # Check 7: The forward_extend method should have FP8 prefill in a code path
    # that is NOT exclusively for non-radix (e.g., not guarded by "if not use_ragged" only)
    # This is the key check - FP8 prefill should be available in the radix cache path too
    
    # Count occurrences of fused_gemm_afp4wfp4_split_cat in forward_extend
    # If the fix is applied, there should be usage in the radix cache path
    fused_gemm_count = forward_extend_body.count("fused_gemm_afp4wfp4_split_cat")
    check("fused_gemm_afp4wfp4_split_cat used sufficiently in forward_extend",
          fused_gemm_count >= 1,
          f"Found {fused_gemm_count} occurrences, expected at least 1 for radix cache path coverage")

    # Check 8: Look for .to(dtype) or cast operations near FP8 handling
    # The fix should reduce extra element-wise casts by using fused_gemm
    cast_patterns = [r'\.to\(', r'\.cast', r'type_as']
    has_cast_handling = any(re.search(p, forward_extend_body) for p in cast_patterns)
    check("Type casting/conversion present in forward_extend",
          has_cast_handling,
          "No type casting found - may indicate missing FP8 conversion logic")

    # Check 9: Verify the file imports or references aiter-related modules
    aiter_import_patterns = [
        r'from\s+aiter',
        r'import\s+aiter',
        r'aiter_ops',
    ]
    has_aiter_imports = any(re.search(p, content) for p in aiter_import_patterns)
    check("aiter module imports present",
          has_aiter_imports,
          "No aiter module imports found")

    # Check 10: Verify forward_extend handles the case with both prefix and new tokens
    # This is the hallmark of radix cache support - handling cached prefixes + new extends
    has_prefix_and_extend = (
        ('prefix' in forward_extend_body.lower() or 'cached' in forward_extend_body.lower()) and
        ('extend' in forward_extend_body.lower() or 'new' in forward_extend_body.lower())
    )
    check("forward_extend handles both prefix and extend tokens",
          has_prefix_and_extend,
          "forward_extend doesn't appear to handle both prefix (cached) and extend (new) tokens")

    # Check 11: Verify there's no exclusive guard preventing FP8 in radix cache path
    # The bug was that FP8 was only in the non-radix path. After fix, it should be in both.
    # Look for the FP8 prefill flag being checked
    fp8_prefill_in_extend = any(re.search(p, forward_extend_body) for p in fp8_prefill_env_patterns)
    check("FP8 prefill flag checked within forward_extend",
          fp8_prefill_in_extend,
          "FP8 prefill flag not checked in forward_extend - may not be integrated")

    # Check 12: Runtime - try to import the module
    runtime_import_ok = False
    try:
        sys.path.insert(0, "/sgl-workspace/sglang/python")
        # Just check syntax by compiling
        compile(content, BACKEND_FILE, 'exec')
        runtime_import_ok = True
    except SyntaxError as e:
        runtime_import_ok = False
        check("aiter_backend.py has valid Python syntax", False, str(e))
    except Exception as e:
        # Other errors during compile are ok (just checking syntax)
        runtime_import_ok = True

    if runtime_import_ok:
        check("aiter_backend.py has valid Python syntax", True)

    # Check 13: Verify the class structure - should have an AiterAttnBackend or similar class
    class_pattern = r'class\s+\w*[Aa]iter\w*Backend'
    has_backend_class = bool(re.search(class_pattern, content))
    check("Aiter attention backend class defined",
          has_backend_class,
          "No AiterAttnBackend class found")

    # Check 14: Check that forward_extend has attention computation calls
    attn_patterns = [
        r'flash_attn',
        r'paged_attention',
        r'attn_fwd',
        r'flash_mha',
        r'triton_attention',
        r'context_attention',
    ]
    has_attn_compute = any(re.search(p, forward_extend_body, re.IGNORECASE) for p in attn_patterns)
    check("forward_extend contains attention computation",
          has_attn_compute,
          "No attention computation calls found in forward_extend")

    # Check 15: Verify that the fp8 prefill path includes scale factors
    # FP8 operations need scale/scaling factors
    scale_patterns = [r'scale', r'quant', r'dequant']
    has_scale_in_extend = any(re.search(p, forward_extend_body, re.IGNORECASE) for p in scale_patterns)
    check("Scale/quantization handling in forward_extend",
          has_scale_in_extend,
          "No scale/quantization handling found - FP8 requires scale factors")

    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)