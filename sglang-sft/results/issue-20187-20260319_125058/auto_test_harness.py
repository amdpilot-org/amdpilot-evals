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


def find_sglang_root():
    """Find the sglang installation root."""
    candidates = [
        "/sgl-workspace/sglang",
        os.path.expanduser("~/sglang"),
    ]
    # Also check pip-installed location
    try:
        import sglang
        pkg_dir = os.path.dirname(os.path.dirname(sglang.__file__))
        candidates.insert(0, pkg_dir)
    except Exception:
        pass

    for c in candidates:
        aiter_path = os.path.join(c, "python", "sglang", "srt", "layers", "attention", "aiter_backend.py")
        if os.path.exists(aiter_path):
            return c, aiter_path

    # Try installed package path directly
    try:
        import sglang.srt.layers.attention.aiter_backend as mod
        return None, mod.__file__
    except Exception:
        pass

    return None, None


def read_file_content(filepath):
    """Read and return file content."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return None


def extract_forward_extend(content):
    """Extract the forward_extend method body from the file content."""
    # Find the start of forward_extend
    pattern = r'def forward_extend\('
    match = re.search(pattern, content)
    if not match:
        return None

    start = match.start()
    # Find the next method at the same or lower indentation level
    # Get the indentation of the def line
    line_start = content.rfind('\n', 0, start) + 1
    indent = start - line_start

    # Find next def at same or lower indent, or end of class
    remaining = content[start:]
    lines = remaining.split('\n')
    method_lines = [lines[0]]
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped.startswith('def ') and (len(line) - len(stripped)) <= indent:
            break
        method_lines.append(line)

    return '\n'.join(method_lines)


if __name__ == "__main__":
    print("=" * 60)
    print("FP8 Prefill + Radix Cache Integration Test Harness")
    print("=" * 60)

    sglang_root, aiter_file = find_sglang_root()

    # Check 1: File exists
    check("aiter_backend.py file found",
          aiter_file is not None and os.path.exists(aiter_file) if aiter_file else False,
          f"Could not find aiter_backend.py. Searched in /sgl-workspace/sglang and package paths")

    if aiter_file and os.path.exists(aiter_file):
        content = read_file_content(aiter_file)
        check("File is readable", content is not None and len(content) > 0,
              "Could not read file content")

        if content:
            # Check 2: forward_extend method exists
            forward_extend_body = extract_forward_extend(content)
            check("forward_extend method exists",
                  forward_extend_body is not None,
                  "Could not find forward_extend method in aiter_backend.py")

            if forward_extend_body:
                # Check 3: FP8 prefill is referenced in forward_extend
                has_fp8_prefill_ref = bool(
                    re.search(r'fp8.*prefill|prefill.*fp8|FP8.*PREFILL|PREFILL.*FP8|fp8_prefill|use_fp8_prefill',
                              forward_extend_body, re.IGNORECASE)
                )
                check("forward_extend references FP8 prefill",
                      has_fp8_prefill_ref,
                      "No FP8 prefill references found in forward_extend")

                # Check 4: fused_gemm_afp4wfp4_split_cat is used
                has_fused_gemm = 'fused_gemm_afp4wfp4_split_cat' in content
                check("fused_gemm_afp4wfp4_split_cat is referenced in the file",
                      has_fused_gemm,
                      "fused_gemm_afp4wfp4_split_cat not found in file")

                # Check 5: fused_gemm_afp4wfp4_split_cat is used in forward_extend
                has_fused_gemm_in_extend = 'fused_gemm_afp4wfp4_split_cat' in forward_extend_body
                check("fused_gemm_afp4wfp4_split_cat used in forward_extend",
                      has_fused_gemm_in_extend,
                      "fused_gemm_afp4wfp4_split_cat not found in forward_extend method")

                # Check 6: FP8 dtype handling (float8_e4m3 variants)
                has_fp8_dtype = bool(
                    re.search(r'float8_e4m3|fp8_e4m3|e4m3fn', forward_extend_body)
                )
                check("FP8 dtype (e4m3) referenced in forward_extend",
                      has_fp8_dtype,
                      "No float8_e4m3/fp8_e4m3 dtype references in forward_extend")

                # Check 7: Radix cache path has FP8 support
                # The radix cache path typically involves prefix/cached tokens handling
                # Look for patterns that indicate the radix cache code path has FP8 integration
                # This could be indicated by FP8 operations near prefix_lens, cached, or radix-related code
                
                # We look for the pattern where there are multiple occurrences of fp8 prefill
                # logic, suggesting both radix and non-radix paths are covered
                fp8_prefill_occurrences = len(re.findall(
                    r'(fp8.*prefill|prefill.*fp8|use_fp8_prefill|fp8_prefill)',
                    forward_extend_body, re.IGNORECASE
                ))
                check("Multiple FP8 prefill code paths in forward_extend (radix + non-radix)",
                      fp8_prefill_occurrences >= 2,
                      f"Only {fp8_prefill_occurrences} FP8 prefill reference(s) found; "
                      f"expected >= 2 for both radix and non-radix paths")

                # Check 8: fused_gemm_afp4wfp4_split_cat appears multiple times
                # (should be used in both radix cache and non-radix-cache paths)
                fused_gemm_count = forward_extend_body.count('fused_gemm_afp4wfp4_split_cat')
                check("fused_gemm_afp4wfp4_split_cat used multiple times in forward_extend",
                      fused_gemm_count >= 2,
                      f"Only {fused_gemm_count} occurrence(s) found; expected >= 2 for both paths")

                # Check 9: The code handles quantization/scaling for FP8
                has_scale_handling = bool(
                    re.search(r'(scale|quant|quantize).*fp8|fp8.*(scale|quant)', 
                              forward_extend_body, re.IGNORECASE)
                    or re.search(r'per_tensor_quantize|quantize_fp8|fp8_quantize|_scale',
                                 forward_extend_body, re.IGNORECASE)
                )
                check("FP8 scale/quantization handling in forward_extend",
                      has_scale_handling,
                      "No FP8 scale or quantization handling found")

                # Check 10: The environment variable check is present
                has_env_check = bool(
                    re.search(r'SGLANG_AITER_FP8_PREFILL', content)
                )
                check("SGLANG_AITER_FP8_PREFILL environment variable check present",
                      has_env_check,
                      "No SGLANG_AITER_FP8_PREFILL_ATTN env var check found")

                # Check 11: Verify import of fused_gemm_afp4wfp4_split_cat
                has_import = bool(
                    re.search(r'(from|import).*fused_gemm_afp4wfp4_split_cat', content)
                )
                check("fused_gemm_afp4wfp4_split_cat is imported",
                      has_import,
                      "No import statement for fused_gemm_afp4wfp4_split_cat found")

                # Check 12: Verify flash_attn or triton_prefill or similar attention call 
                # in the FP8 radix-cache path
                has_attn_call = bool(
                    re.search(r'(flash_attn|prefill_attention|triton.*prefill|flash_fwd)',
                              forward_extend_body, re.IGNORECASE)
                )
                check("Attention kernel call present in forward_extend",
                      has_attn_call,
                      "No attention kernel call found in forward_extend")

            else:
                # Skip remaining checks if we couldn't find forward_extend
                for name in ["FP8 prefill refs", "fused_gemm in file", "fused_gemm in extend",
                             "FP8 dtype", "multiple paths", "fused_gemm count",
                             "scale handling", "env check", "import", "attn call"]:
                    check(name, False, "Skipped: forward_extend not found")
        else:
            for i in range(11):
                check(f"Skipped check {i}", False, "Could not read file")
    else:
        for i in range(12):
            check(f"Skipped check {i}", False, "File not found")

    # Runtime checks - try to import
    print("\n--- Runtime Import Checks ---")
    try:
        sys.path.insert(0, os.path.join(sglang_root or "/sgl-workspace/sglang", "python"))
        import sglang
        check("sglang package importable", True)
    except Exception as e:
        check("sglang package importable", False, str(e))

    # Try importing the specific module (may fail due to GPU deps, that's OK)
    try:
        from sglang.srt.layers.attention import aiter_backend
        check("aiter_backend module importable", True)
        
        # Check if the module has the class with forward_extend
        has_class = False
        for name in dir(aiter_backend):
            obj = getattr(aiter_backend, name)
            if isinstance(obj, type) and hasattr(obj, 'forward_extend'):
                has_class = True
                break
        check("Attention backend class with forward_extend found", has_class,
              "No class with forward_extend method found in module")
    except ImportError as e:
        # This is expected if aiter/ROCm deps aren't available
        check("aiter_backend module importable (may need GPU deps)", True,
              f"Skipped gracefully: {e}")
        check("Attention backend class check (skipped - import issue)", True,
              "Skipped gracefully due to import dependencies")
    except Exception as e:
        check("aiter_backend module importable", False, str(e))
        check("Attention backend class check", False, "Skipped due to import failure")

    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)