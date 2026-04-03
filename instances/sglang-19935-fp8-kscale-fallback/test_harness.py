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
    # Also try to find via Python path
    try:
        import sglang
        if sglang.__file__ is not None:
            pkg_dir = os.path.dirname(os.path.abspath(sglang.__file__))
            # go up to find the repo root
            candidates.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir))))
    except (ImportError, TypeError, AttributeError):
        pass

    for c in candidates:
        aiter_path = os.path.join(c, "python", "sglang", "srt", "layers", "attention", "aiter_backend.py")
        if os.path.exists(aiter_path):
            return c
    return None

def get_aiter_backend_path(root):
    return os.path.join(root, "python", "sglang", "srt", "layers", "attention", "aiter_backend.py")

def extract_mla_decode_fwd_calls(source):
    """Extract all mla_decode_fwd call sites with surrounding context."""
    lines = source.split('\n')
    call_sites = []
    i = 0
    while i < len(lines):
        if 'mla_decode_fwd' in lines[i]:
            # Grab context: 10 lines before and 10 lines after
            start = max(0, i - 10)
            end = min(len(lines), i + 15)
            context = '\n'.join(lines[start:end])
            call_sites.append({
                'line_num': i + 1,
                'line': lines[i],
                'context': context,
            })
        i += 1
    return call_sites

def check_call_site_has_fallback(context, source_full):
    """
    Check that a mla_decode_fwd call site does NOT pass raw layer.k_scale
    without a fallback. The fix should use something like:
      layer.k_scale if layer.k_scale is not None else self.k_scale
    or:
      kv_scale = layer.k_scale or self.k_scale
    or equivalent.
    
    We check that the call does NOT have a bare `layer.k_scale` without
    a fallback pattern nearby.
    """
    # Check for patterns that indicate the fix is applied
    # Pattern 1: ternary/conditional expression
    has_fallback_ternary = bool(re.search(
        r'layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None\s+else\s+self\.k_scale',
        context
    ))
    # Pattern 2: or-based fallback
    has_fallback_or = bool(re.search(
        r'layer\.k_scale\s+or\s+self\.k_scale',
        context
    ))
    # Pattern 3: pre-computed variable with fallback
    has_precomputed_fallback = bool(re.search(
        r'k_scale\s*=\s*.*layer\.k_scale.*(?:if|or).*self\.k_scale',
        context
    ))
    # Pattern 4: variable assigned with fallback used in the call
    # e.g., kv_scale = layer.k_scale if layer.k_scale is not None else self.k_scale
    has_variable_fallback = bool(re.search(
        r'(?:kv_scale|k_scale|_k_scale)\s*=\s*(?:layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None\s+else\s+self\.k_scale|layer\.k_scale\s+or\s+self\.k_scale)',
        context
    ))
    # Pattern 5: getattr-based fallback
    has_getattr_fallback = bool(re.search(
        r'getattr\s*\(\s*layer\s*,\s*["\']k_scale["\']\s*,\s*self\.k_scale\s*\)',
        context
    ))
    # Pattern 6: Maybe the fallback is done earlier in the function and a local variable is used
    # Check if the call uses self.k_scale directly or a variable that was assigned with fallback
    uses_self_k_scale = 'self.k_scale' in context
    
    # Check for the BAD pattern: raw layer.k_scale passed to mla_decode_fwd
    # without any fallback nearby
    has_raw_layer_k_scale_in_call = False
    for line in context.split('\n'):
        if 'mla_decode_fwd' in line and 'layer.k_scale' in line:
            # Check if there's a fallback on the same line
            if not re.search(r'layer\.k_scale\s+if\s+layer\.k_scale', line) and \
               not re.search(r'layer\.k_scale\s+or\s+self\.k_scale', line):
                has_raw_layer_k_scale_in_call = True

    any_fallback = (has_fallback_ternary or has_fallback_or or has_precomputed_fallback or 
                    has_variable_fallback or has_getattr_fallback or uses_self_k_scale)
    
    return any_fallback or not has_raw_layer_k_scale_in_call

def check_no_bare_layer_k_scale_in_mla_calls(source):
    """
    Verify that no mla_decode_fwd call passes layer.k_scale without a fallback.
    Returns (all_ok, details)
    """
    lines = source.split('\n')
    problems = []
    
    # Find all mla_decode_fwd calls and check their arguments
    # These calls can span multiple lines, so we need to handle that
    in_call = False
    call_start = -1
    paren_depth = 0
    call_text = ""
    
    for i, line in enumerate(lines):
        if not in_call:
            if 'mla_decode_fwd' in line:
                in_call = True
                call_start = i
                call_text = line
                paren_depth = line.count('(') - line.count(')')
                if paren_depth <= 0:
                    # Single line call
                    in_call = False
                    # Check this call
                    if 'layer.k_scale' in call_text:
                        if not re.search(r'layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None', call_text) and \
                           not re.search(r'layer\.k_scale\s+or\s+self\.k_scale', call_text):
                            problems.append(f"Line {i+1}: bare layer.k_scale in mla_decode_fwd call")
                    call_text = ""
        else:
            call_text += " " + line
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                in_call = False
                if 'layer.k_scale' in call_text:
                    if not re.search(r'layer\.k_scale\s+if\s+layer\.k_scale\s+is\s+not\s+None', call_text) and \
                       not re.search(r'layer\.k_scale\s+or\s+self\.k_scale', call_text):
                        problems.append(f"Lines {call_start+1}-{i+1}: bare layer.k_scale in mla_decode_fwd call")
                call_text = ""
    
    return len(problems) == 0, problems

def check_self_k_scale_init(source):
    """Check that self.k_scale is initialized (e.g., torch.tensor([1.0]))."""
    return bool(re.search(r'self\.k_scale\s*=\s*.*(?:torch\.tensor|torch\.ones|1\.0)', source))

if __name__ == "__main__":
    print("=" * 60)
    print("Test: FP8 MLA decode k_scale fallback in aiter_backend.py")
    print("=" * 60)

    root = find_sglang_root()
    
    check("sglang repo found", root is not None, 
          "Could not find sglang repo at /sgl-workspace/sglang or via import")
    
    if root is None:
        print()
        score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
        print(f"Results: {checks_passed}/{checks_total} checks passed")
        print(f"SCORE: {score:.1f}")
        sys.exit(1)
    
    aiter_path = get_aiter_backend_path(root)
    
    check("aiter_backend.py exists", os.path.exists(aiter_path),
          f"File not found at {aiter_path}")
    
    if not os.path.exists(aiter_path):
        print()
        score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
        print(f"Results: {checks_passed}/{checks_total} checks passed")
        print(f"SCORE: {score:.1f}")
        sys.exit(1)
    
    with open(aiter_path, 'r') as f:
        source = f.read()
    
    # Check 1: self.k_scale is initialized in __init__
    check("self.k_scale initialized in __init__", 
          check_self_k_scale_init(source),
          "self.k_scale should be initialized with a default tensor (e.g., torch.tensor([1.0]))")
    
    # Check 2: Find all mla_decode_fwd call sites
    call_sites = extract_mla_decode_fwd_calls(source)
    check("mla_decode_fwd call sites found", len(call_sites) >= 4,
          f"Expected at least 4 call sites, found {len(call_sites)}")
    
    print(f"\n  Found {len(call_sites)} mla_decode_fwd call sites:")
    for cs in call_sites:
        print(f"    - Line {cs['line_num']}: {cs['line'].strip()[:80]}")
    print()
    
    # Check 3: No bare layer.k_scale in mla_decode_fwd calls
    all_ok, problems = check_no_bare_layer_k_scale_in_mla_calls(source)
    check("no bare layer.k_scale in mla_decode_fwd calls", all_ok,
          f"Found problematic call sites: {'; '.join(problems)}")
    
    # Check 4: Each call site has a fallback mechanism
    all_have_fallback = True
    for cs in call_sites:
        has_fb = check_call_site_has_fallback(cs['context'], source)
        if not has_fb:
            all_have_fallback = False
            print(f"    WARNING: No fallback found near line {cs['line_num']}")
    
    check("all mla_decode_fwd call sites have k_scale fallback", all_have_fallback,
          "Some call sites pass layer.k_scale without a fallback to self.k_scale")
    
    # Check 5: Verify the forward_decode method handles k_scale
    # Find forward_decode method and check it
    forward_decode_match = re.search(r'def\s+forward_decode\s*\(.*?\n(?=\n\s*def\s|\Z)', source, re.DOTALL)
    if forward_decode_match:
        fd_source = forward_decode_match.group(0)
        has_decode_fallback = ('self.k_scale' in fd_source or 
                               (not 'layer.k_scale' in fd_source) or
                               bool(re.search(r'layer\.k_scale.*(?:if|or).*self\.k_scale', fd_source)))
        check("forward_decode uses k_scale fallback", has_decode_fallback,
              "forward_decode should use self.k_scale fallback when layer.k_scale is None")
    else:
        check("forward_decode method found", False, "Could not find forward_decode method")
    
    print()
    score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)
