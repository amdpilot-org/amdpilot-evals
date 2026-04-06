#!/usr/bin/env python3
"""Test harness for sglang-mla-ps-kernel-guard.

Bug: Non-MLA models crash with AttributeError on `max_split_per_batch`
because a MLA-only code path is entered unconditionally.

Tests use file-based AST analysis to verify the guard condition logic in each method.
No module import required -- reads the source file directly.
"""
import ast
import sys
from pathlib import Path

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
print("sglang-mla-ps-kernel-guard test harness")
print("=" * 60)

# Read source file directly — no import needed
SOURCE_PATH = "/workspace/sglang/python/sglang/srt/layers/attention/aiter_backend.py"

if not check("aiter_backend.py exists", Path(SOURCE_PATH).is_file()):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

source = Path(SOURCE_PATH).read_text()

try:
    tree = ast.parse(source)
    check("Valid Python syntax", True)
except SyntaxError as e:
    check("Valid Python syntax", False, str(e))
    print(f"\nSCORE: 0.0")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: check if an AST node tree contains a reference to a name
# ---------------------------------------------------------------------------
def contains_name(node, name):
    'Check if an AST node tree contains a reference to name.'
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
        if isinstance(child, ast.Attribute) and child.attr == name:
            return True
    return False


def contains_mla_ps_kernel_ref(node):
    'Check if an AST node references _use_mla_ps_kernel.'
    return contains_name(node, "_use_mla_ps_kernel")


def contains_use_mla_ref(node):
    'Check if an AST node references use_mla (typically self.use_mla).'
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "use_mla":
            return True
    return False


def check_method_guard(method_name):
    'Find method and check that at least one _use_mla_ps_kernel guard includes use_mla.'
    # Find the method definition in the AST
    method_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            method_def = node
            break

    if method_def is None:
        return None, f"method {method_name} not found in source"

    # Count _use_mla_ps_kernel If nodes that also include use_mla in the test.
    guarded_count = 0
    total_refs = 0

    for node in ast.walk(method_def):
        if isinstance(node, ast.If):
            test = node.test
            if contains_mla_ps_kernel_ref(test):
                total_refs += 1
                if contains_use_mla_ref(test):
                    guarded_count += 1

    if total_refs == 0:
        # No if-nodes reference _use_mla_ps_kernel at all — method may have
        # been restructured; treat as guarded (the bug is gone).
        return True, "no _use_mla_ps_kernel guard found in method"
    elif guarded_count >= 1:
        return True, f"{guarded_count}/{total_refs} _use_mla_ps_kernel guards include use_mla"
    else:
        return False, f"0/{total_refs} _use_mla_ps_kernel guards include use_mla — missing fix"


# ---------------------------------------------------------------------------
# Test 1: Check AiterAttnBackend class exists in the file
# ---------------------------------------------------------------------------
class_found = False
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AiterAttnBackend":
        class_found = True
        break

check("AiterAttnBackend class found in source", class_found)

# ---------------------------------------------------------------------------
# Test 2: AST guard check on init_forward_metadata_capture_cuda_graph
# ---------------------------------------------------------------------------
result, detail = check_method_guard("init_forward_metadata_capture_cuda_graph")
if result is None:
    check("init_forward_metadata_capture_cuda_graph guard (AST)", False, detail)
else:
    check("init_forward_metadata_capture_cuda_graph guard (AST)", result, detail)

# ---------------------------------------------------------------------------
# Test 3: AST guard check on init_forward_metadata_replay_cuda_graph
# ---------------------------------------------------------------------------
result, detail = check_method_guard("init_forward_metadata_replay_cuda_graph")
if result is None:
    check("init_forward_metadata_replay_cuda_graph guard (AST)", False, detail)
else:
    check("init_forward_metadata_replay_cuda_graph guard (AST)", result, detail)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
