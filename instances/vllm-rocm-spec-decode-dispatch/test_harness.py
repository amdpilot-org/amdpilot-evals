#!/usr/bin/env python3
"""Test harness for vllm-rocm-spec-decode-dispatch (PR #32877).

Bug: AITER FA decode path hardcodes max_seqlen_q=1 in paged_attention_v1,
producing incorrect results with speculative decoding where max_query_len > 1.

Tests:
1. Import the AITER FA backend module
2. Source inspection: decode method has conditional check for max_query_len > 1
3. AST analysis: decode method has branching logic based on query length
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


def run_test(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/workspace",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-spec-decode-dispatch test harness")
print("=" * 60)

# ---- Check 1: Import the AITER FA backend ----
print("\n--- Check 1: Import AITER FA backend ---")
stdout1, stderr1, rc1 = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend
    print("IMPORT_OK:True")
except Exception as e:
    print(f"IMPORT_OK:False")
    print(f"IMPORT_ERR:{e}")
""")

if "IMPORT_OK:True" in stdout1:
    check("Import AiterFlashAttentionBackend", True)
else:
    err_detail = ""
    for line in (stdout1 + stderr1).splitlines():
        if "IMPORT_ERR:" in line:
            err_detail = line.split("IMPORT_ERR:", 1)[1]
            break
    if not err_detail:
        err_detail = stderr1[:200]
    check("Import AiterFlashAttentionBackend", False, err_detail)

# ---- Check 2: Source inspection for multi-token decode routing ----
print("\n--- Check 2: Source inspection for decode dispatch ---")
stdout2, stderr2, rc2 = run_test("""
import sys, inspect
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl

# Get the source of the forward method (the decode path lives here)
src = inspect.getsource(AiterFlashAttentionImpl.forward)

# Check for conditional routing based on query length
has_query_len_check = (
    "max_query_len" in src and (
        "max_query_len > 1" in src
        or "max_query_len>" in src
        or "decode_max_query_len > 1" in src
        or "decode_max_query_len>" in src
    )
)

# Check that unified_attention is referenced (the correct kernel for multi-token)
has_unified_attention = "unified_attention" in src

print(f"HAS_QUERY_LEN_CHECK:{has_query_len_check}")
print(f"HAS_UNIFIED_ATTENTION:{has_unified_attention}")
""")

if rc2 != 0:
    check("Decode method source accessible", False, stderr2[:200])
    check("Decode routes multi-token to different kernel", False, "source not accessible")
else:
    has_qlen = "HAS_QUERY_LEN_CHECK:True" in stdout2
    has_unified = "HAS_UNIFIED_ATTENTION:True" in stdout2
    check(
        "Decode has max_query_len > 1 conditional",
        has_qlen,
        "No conditional check for multi-token decode query length found"
    )
    check(
        "Decode references unified_attention for multi-token path",
        has_unified,
        "unified_attention not found; multi-token decode still uses paged_attention_v1"
    )

# ---- Check 3: AST analysis for branching logic ----
print("\n--- Check 3: AST analysis for decode branching ---")
stdout3, stderr3, rc3 = run_test("""
import sys, ast, inspect, textwrap
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl

src = inspect.getsource(AiterFlashAttentionImpl.forward)
src = textwrap.dedent(src)
tree = ast.parse(src)

class DecodeDispatchVisitor(ast.NodeVisitor):
    \"\"\"Walk the AST looking for an If node whose test references
    a query-length variable (max_query_len / decode_max_query_len)
    with a comparison > 1, and whose body contains a call to
    unified_attention (or any function other than paged_attention_v1).\"\"\"

    def __init__(self):
        self.has_query_len_branch = False
        self.branch_calls_alternative = False

    def _test_mentions_query_len(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and "query_len" in child.id:
                return True
            if isinstance(child, ast.Attribute) and "query_len" in child.attr:
                return True
        return False

    def _body_calls_alternative(self, body):
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name and name != "paged_attention_v1" and "attention" in name.lower():
                    return True
        return False

    def visit_If(self, node):
        if self._test_mentions_query_len(node.test):
            self.has_query_len_branch = True
            if self._body_calls_alternative(node.body):
                self.branch_calls_alternative = True
        self.generic_visit(node)

visitor = DecodeDispatchVisitor()
visitor.visit(tree)

print(f"HAS_QUERY_LEN_BRANCH:{visitor.has_query_len_branch}")
print(f"BRANCH_CALLS_ALTERNATIVE:{visitor.branch_calls_alternative}")
""")

if rc3 != 0:
    check("AST parse of forward method", False, stderr3[:300])
    check("AST: decode branches on query length", False, "AST parse failed")
else:
    has_branch = "HAS_QUERY_LEN_BRANCH:True" in stdout3
    calls_alt = "BRANCH_CALLS_ALTERNATIVE:True" in stdout3
    check(
        "AST: decode branches on query length",
        has_branch,
        "No If-node found with query_len in test condition"
    )
    check(
        "AST: branch body calls alternative attention kernel (not paged_attention_v1)",
        calls_alt,
        "Branch body does not call an alternative attention function"
    )

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
