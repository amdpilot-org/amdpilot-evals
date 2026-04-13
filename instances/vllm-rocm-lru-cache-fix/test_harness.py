#!/usr/bin/env python3
"""Test harness for vllm-rocm-lru-cache-fix.

Behavioral test: verifies that the paged MQA logits helper function
is properly cached at module scope for ROCm sparse MLA.
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
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("vllm-rocm-lru-cache-fix test harness")
print("=" * 60)

# Test 1: Module imports
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mla
    print("IMPORT:OK")
except Exception as e:
    print(f"IMPORT:FAIL:{e}")
""")
check("Import rocm_aiter_mla_sparse", "IMPORT:OK" in stdout,
      stdout.strip() if "IMPORT:OK" not in stdout else "")

# Test 2: paged_mqa_logits_module at module scope
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mla
# If the function is at module scope, it's accessible as a module attribute
has_attr = hasattr(mla, "paged_mqa_logits_module")
is_callable = callable(getattr(mla, "paged_mqa_logits_module", None))
print(f"HAS_ATTR:{has_attr}")
print(f"IS_CALLABLE:{is_callable}")
""")
has_attr = "HAS_ATTR:True" in stdout
check("paged_mqa_logits_module is module-level attribute",
      has_attr,
      "function is nested inside another function (unfixed)")

is_callable = "IS_CALLABLE:True" in stdout
check("paged_mqa_logits_module is callable",
      is_callable,
      "not callable or doesn't exist at module level")

# Test 3: The function has lru_cache
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as mla
fn = getattr(mla, "paged_mqa_logits_module", None)
if fn is not None:
    has_cache = hasattr(fn, "cache_info") or hasattr(fn, "__wrapped__")
    print(f"HAS_CACHE:{has_cache}")
else:
    print("NO_FUNC")
""")
has_cache = "HAS_CACHE:True" in stdout
check("paged_mqa_logits_module has lru_cache",
      has_cache,
      "no cache_info attribute (not wrapped with lru_cache at module scope)")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
