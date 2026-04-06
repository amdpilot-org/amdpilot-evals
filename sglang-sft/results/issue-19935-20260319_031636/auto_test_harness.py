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


def find_aiter_backend():
    """Find the aiter_backend.py file."""
    candidates = [
        "/sgl-workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py",
    ]
    # Also try to find it via python path
    try:
        import sglang
        pkg_dir = os.path.dirname(sglang.__file__)
        candidates.append(os.path.join(pkg_dir, "srt", "layers", "attention", "aiter_backend.py"))
    except ImportError:
        pass

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def extract_mla_decode_fwd_calls(content):
    """Extract all mla_decode_fwd call sites with their surrounding context."""
    calls = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'mla_decode_fwd' in line and 'import' not in line and 'def ' not in line:
            # Get surrounding context (20 lines before and after)
            start = max(0, i - 20)
            end = min(len(lines), i + 20)
            context = '\n'.join(lines[start:end])
            calls.append((i + 1, line.strip(), context))
    return calls


def check_kv_scale_fallback_in_context(context, call_line_num):
    """Check if the context around a mla_decode_fwd call has proper k_scale fallback."""
    # Look for patterns that indicate proper fallback:
    # 1. Direct inline fallback: `layer.k_scale if layer.k_scale is not None else self.k_scale`
    # 2. Or similar: `(layer.k_scale or self.k_scale)`
    # 3. Variable assignment with fallback before the call
    # 4. `self.k_scale` used directly instead of `layer.k_scale`
    
    # Check if raw `layer.k_scale` is NOT being passed directly to mla_decode_fwd
    # without any fallback
    
    # Pattern: the call uses self.k_scale or a fallback expression
    has_fallback = False
    
    # Check for inline fallback patterns
    fallback_patterns = [
        r'layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None\s+else\s+self\.k_scale',
        r'layer\.k_scale\s+or\s+self\.k_scale',
        r'self\.k_scale\s+if\s+layer\.k_scale\s+is\s+None\s+else\s+layer\.k_scale',
        r'kv_scale\s*=\s*.*self\.k_scale',
        r'k_scale\s*=\s*.*self\.k_scale',
    ]
    
    for pat in fallback_patterns:
        if re.search(pat, context):
            has_fallback = True
            break
    
    # Also check if `self.k_scale` appears near the mla_decode_fwd call
    # AND the call doesn't pass raw `layer.k_scale` without a guard
    if not has_fallback:
        # Check if self.k_scale is referenced in the context
        if 'self.k_scale' in context:
            has_fallback = True
    
    return has_fallback


def check_no_raw_layer_k_scale_in_call(content):
    """
    Check that mla_decode_fwd is NOT called with raw layer.k_scale without a fallback.
    Returns True if the fix is applied (no raw layer.k_scale without fallback).
    """
    lines = content.split('\n')
    
    # Find all mla_decode_fwd calls and check their arguments
    # We need to handle multi-line calls
    in_mla_call = False
    call_text = ""
    call_start = -1
    paren_depth = 0
    mla_calls = []
    
    for i, line in enumerate(lines):
        if 'mla_decode_fwd' in line and 'import' not in line and 'def ' not in line and '#' not in line.split('mla_decode_fwd')[0]:
            in_mla_call = True
            call_text = ""
            call_start = i
            paren_depth = 0
        
        if in_mla_call:
            call_text += line + "\n"
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0 and '(' in call_text:
                mla_calls.append((call_start + 1, call_text))
                in_mla_call = False
    
    if not mla_calls:
        return False, "No mla_decode_fwd calls found"
    
    problematic_calls = []
    for line_num, call_text in mla_calls:
        # Check if the call contains raw `layer.k_scale` without a fallback guard
        # A raw usage would be: `layer.k_scale` appearing as an argument
        # without `if layer.k_scale is not None else` or `or self.k_scale`
        
        has_layer_k_scale = 'layer.k_scale' in call_text
        has_fallback = bool(re.search(
            r'layer\.k_scale\s+(if|or)\s+',
            call_text
        )) or bool(re.search(
            r'if\s+layer\.k_scale\s+is\s+not\s+None',
            call_text
        )) or bool(re.search(
            r'self\.k_scale\s+if\s+layer\.k_scale\s+is\s+None',
            call_text
        ))
        
        # Also check if self.k_scale is used instead
        has_self_k_scale = 'self.k_scale' in call_text
        
        # Also check if a variable like kv_scale is used (which was set with fallback earlier)
        uses_kv_scale_var = bool(re.search(r'(?:kv_scale|k_scale_val)', call_text)) and 'layer.k_scale' not in call_text
        
        if has_layer_k_scale and not has_fallback:
            problematic_calls.append(line_num)
    
    if problematic_calls:
        return False, f"Raw layer.k_scale without fallback found at lines: {problematic_calls}"
    
    return True, f"All {len(mla_calls)} mla_decode_fwd calls have proper fallback"


if __name__ == "__main__":
    print("=" * 60)
    print("Test: FP8 k_scale fallback in aiter MLA decode")
    print("=" * 60)

    filepath = find_aiter_backend()
    
    check("aiter_backend.py file exists", filepath is not None,
          "Could not find aiter_backend.py")
    
    if filepath is None:
        print()
        score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
        print(f"Results: {checks_passed}/{checks_total} checks passed")
        print(f"SCORE: {score:.1f}")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check 1: File contains mla_decode_fwd calls
    mla_calls = extract_mla_decode_fwd_calls(content)
    check("mla_decode_fwd calls found in file",
          len(mla_calls) >= 4,
          f"Expected at least 4 call sites, found {len(mla_calls)}")
    
    # Check 2: self.k_scale initialization exists
    has_k_scale_init = bool(re.search(
        r'self\.k_scale\s*=\s*torch\.tensor\(\[1\.0\]\)',
        content
    ))
    check("self.k_scale initialized with torch.tensor([1.0])",
          has_k_scale_init,
          "self.k_scale initialization not found")
    
    # Check 3: No raw layer.k_scale passed to mla_decode_fwd without fallback
    no_raw, detail = check_no_raw_layer_k_scale_in_call(content)
    check("No raw layer.k_scale without fallback in mla_decode_fwd calls",
          no_raw, detail)
    
    # Check 4: Verify each call site has proper fallback
    # Look for the fallback pattern in the broader context around each call
    all_have_fallback = True
    missing_fallback_lines = []
    for line_num, line_text, context in mla_calls:
        has_fb = check_kv_scale_fallback_in_context(context, line_num)
        if not has_fb:
            all_have_fallback = False
            missing_fallback_lines.append(line_num)
    
    check("All mla_decode_fwd call sites have k_scale fallback",
          all_have_fallback,
          f"Missing fallback at line(s): {missing_fallback_lines}")
    
    # Check 5: Verify that the same fallback is applied for q_scale (often same as k_scale)
    # The q_scale should also have a fallback or use self.k_scale
    q_scale_mentions = []
    for line_num, line_text, context in mla_calls:
        if 'q_scale' in context or 'kv_scale' in context:
            q_scale_mentions.append(line_num)
    
    check("q_scale/kv_scale handling present in mla_decode_fwd contexts",
          len(q_scale_mentions) >= len(mla_calls),
          f"Only {len(q_scale_mentions)} of {len(mla_calls)} call sites mention q_scale/kv_scale")
    
    # Check 6: Verify the forward_decode method has the fix
    # Find forward_decode method and check its mla_decode_fwd call
    decode_match = re.search(
        r'def\s+forward_decode\s*\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
        content, re.DOTALL
    )
    if decode_match:
        decode_body = decode_match.group(1)
        has_decode_fix = ('self.k_scale' in decode_body) or (
            'layer.k_scale if layer.k_scale is not None else' in decode_body
        ) or (
            'layer.k_scale or self.k_scale' in decode_body
        )
        # Also check that raw layer.k_scale is not the only thing passed
        raw_in_decode = ('layer.k_scale' in decode_body and 
                        'self.k_scale' not in decode_body)
        check("forward_decode has k_scale fallback", 
              has_decode_fix and not raw_in_decode,
              "forward_decode still uses raw layer.k_scale without fallback")
    else:
        check("forward_decode method found", False, "Could not find forward_decode method")
    
    # Check 7: Verify the forward_extend method has the fix
    extend_match = re.search(
        r'def\s+forward_extend\s*\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
        content, re.DOTALL
    )
    if extend_match:
        extend_body = extend_match.group(1)
        has_extend_fix = ('self.k_scale' in extend_body) or (
            'layer.k_scale if layer.k_scale is not None else' in extend_body
        ) or (
            'layer.k_scale or self.k_scale' in extend_body
        )
        raw_in_extend = ('layer.k_scale' in extend_body and 
                        'self.k_scale' not in extend_body)
        check("forward_extend has k_scale fallback",
              has_extend_fix and not raw_in_extend,
              "forward_extend still uses raw layer.k_scale without fallback")
    else:
        check("forward_extend method found", False, "Could not find forward_extend method")
    
    # Check 8: Verify the fix matches the flashmla_backend pattern
    # Check if flashmla_backend.py also has this pattern (as reference)
    flashmla_path = filepath.replace('aiter_backend.py', 'flashmla_backend.py')
    if os.path.exists(flashmla_path):
        with open(flashmla_path, 'r') as f:
            flashmla_content = f.read()
        has_flashmla_pattern = ('self.k_scale' in flashmla_content)
        check("flashmla_backend.py has self.k_scale fallback pattern (reference)",
              has_flashmla_pattern,
              "flashmla_backend.py doesn't have the reference pattern")
    else:
        check("flashmla_backend.py exists for reference", True, 
              "Skipped - file not found but not critical")

    # Check 9: Count the number of fallback patterns - should be at least 4
    # (one for each call site)
    fallback_count = len(re.findall(
        r'(?:layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None\s+else\s+self\.k_scale|'
        r'layer\.k_scale\s+or\s+self\.k_scale|'
        r'self\.k_scale\s+if\s+layer\.k_scale\s+is\s+None\s+else\s+layer\.k_scale)',
        content
    ))
    
    # If not inline fallback, check for variable-based fallback
    if fallback_count < 4:
        # Check for pattern like: kv_scale = layer.k_scale if ... else self.k_scale
        # followed by usage in mla_decode_fwd
        var_fallback = re.findall(
            r'(?:kv_scale|k_scale_val|_k_scale)\s*=\s*(?:layer\.k_scale\s+if|self\.k_scale\s+if)',
            content
        )
        fallback_count += len(var_fallback) * 2  # each variable could serve multiple calls
    
    # Also count direct self.k_scale usage in mla_decode_fwd calls
    if fallback_count < 4:
        # Count mla_decode_fwd calls that use self.k_scale directly
        self_k_in_mla = 0
        for _, _, ctx in mla_calls:
            if 'self.k_scale' in ctx:
                self_k_in_mla += 1
        fallback_count = max(fallback_count, self_k_in_mla)
    
    check("Sufficient fallback patterns for all 4 call sites",
          fallback_count >= 4,
          f"Found {fallback_count} fallback patterns, expected at least 4")

    # Runtime check: try importing the module
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(filepath), '..', '..', '..', '..'))
        # Don't actually import as it may need GPU, just verify syntax
        import py_compile
        result = py_compile.compile(filepath, doraise=True)
        check("aiter_backend.py compiles without syntax errors", True)
    except py_compile.PyCompileError as e:
        check("aiter_backend.py compiles without syntax errors", False, str(e))
    except Exception as e:
        check("aiter_backend.py compiles without syntax errors", False, str(e))

    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)