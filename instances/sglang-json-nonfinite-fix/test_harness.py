#!/usr/bin/env python3
"""Test harness for sglang-json-nonfinite-fix. Behavioral tests only.

Bug: /generate endpoint crashes with ORJSON serialization error when
response contains non-finite float values (NaN, -inf, inf) in top_logprobs.
Test: Verify JSON serialization handles non-finite floats gracefully.
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
    return condition


def run_test(script, timeout=60):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("sglang-json-nonfinite-fix test harness")
print("=" * 60)

# Test 1: Check json_response utility module exists
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
try:
    from sglang.srt.utils.json_response import dumps_json, orjson_response
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
""")
has_module = "IMPORT:OK" in stdout
check("json_response utility module exists (dumps_json, orjson_response)",
      has_module, stdout.strip() if not has_module else "")

# Test 2: Check dumps_json handles non-finite floats
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import json, math
try:
    from sglang.srt.utils.json_response import dumps_json
    data = {"logprobs": [float("nan"), float("-inf"), float("inf"), 0.5]}
    result = dumps_json(data)
    decoded = json.loads(result)
    print(f"SERIALIZE:OK")
    logprobs = decoded.get("logprobs", [])
    # NaN/Inf should be serialized to null in JSON
    non_finite_ok = all(v is None for v in logprobs[:3])
    print(f"NULL_CONVERT:{non_finite_ok}")
    finite_ok = logprobs[3] == 0.5
    print(f"FINITE_OK:{finite_ok}")
except ImportError:
    print("SERIALIZE:NO_MODULE")
except Exception as e:
    print(f"SERIALIZE:ERROR:{type(e).__name__}:{str(e)[:200]}")
""")

if "SERIALIZE:NO_MODULE" in stdout:
    check("dumps_json serializes non-finite floats", False, "Module not found")
elif "SERIALIZE:ERROR" in stdout:
    err = stdout.split("SERIALIZE:ERROR:")[1].split("\n")[0]
    check("dumps_json serializes non-finite floats", False, err)
elif "SERIALIZE:OK" in stdout:
    check("dumps_json serializes non-finite floats", True)
else:
    check("dumps_json serializes non-finite floats", False, f"Unexpected: {stdout[:200]}")

# Test 3: Check http_server uses the utility
stdout, stderr, rc = run_test("""
import sys; sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import inspect
try:
    from sglang.srt.entrypoints import http_server
    source = inspect.getsource(http_server)
    uses_util = "json_response" in source or "orjson_response" in source or "dumps_json" in source
    print(f"USES_UTIL:{uses_util}")
except Exception as e:
    print(f"CHECK_ERROR:{e}")
""")
uses_util = "USES_UTIL:True" in stdout
check("http_server uses json_response utility",
      uses_util, "http_server does not import from json_response")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
