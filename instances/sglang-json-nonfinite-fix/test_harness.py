#!/usr/bin/env python3
"""Test harness for sglang-json-nonfinite-fix. Behavioral tests only.

Bug: /generate endpoint crashes with a serialization error when the response
     contains non-finite float values (NaN, -Inf, Inf) in top_logprobs.
     ORJSON's default mode rejects non-finite floats.

Expected behavior after fix: Non-finite floats are serialized as JSON null
     without raising an exception. Finite floats are preserved exactly.
"""
import sys
import json
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


def run_test(script, timeout=90):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("sglang-json-nonfinite-fix test harness")
print("=" * 60)

# Test 1: Baseline — verify that raw orjson.dumps raises on NaN without options.
# This confirms the bug scenario is present in the test environment.
stdout, stderr, rc = run_test("""
import sys
sys.path.insert(0, '/sgl-workspace/sglang/python')
try:
    import orjson
    data = {"logprobs": [float("nan"), float("-inf"), float("inf")]}
    try:
        orjson.dumps(data)
        print("BASELINE:NO_ERROR")
    except orjson.JSONEncodeError:
        print("BASELINE:RAISES_AS_EXPECTED")
    except Exception as e:
        print(f"BASELINE:OTHER:{type(e).__name__}")
except ImportError as e:
    print(f"IMPORT_FAIL:{e}")
""")
baseline_raises = "BASELINE:RAISES_AS_EXPECTED" in stdout
check(
    "Raw orjson.dumps raises on non-finite floats (bug scenario confirmed)",
    baseline_raises,
    f"Unexpected: {stdout.strip()[:120]}",
)

# Test 2: BEHAVIORAL — sglang provides a JSON serialization mechanism that
# handles NaN/Inf → null without crashing.
#
# Strategy: import http_server (loading all serialization dependencies), then
# scan sglang.srt.utils for any callable that converts a dict with NaN/Inf
# to valid JSON bytes, with null for non-finite and preserved finite values.
# Does NOT prescribe any specific module or function name.
# IMPORT_SKIP (if sglang unavailable) → explicit FAIL.
stdout, stderr, rc = run_test("""
import sys
sys.path.insert(0, '/sgl-workspace/sglang/python')
import json, importlib, pkgutil

test_data = {"logprobs": [float("nan"), float("-inf"), float("inf"), 0.5]}

try:
    import sglang.srt.entrypoints.http_server as _hs
except ImportError as e:
    print(f"IMPORT_SKIP:{e}")
    sys.exit(0)

import sglang.srt.utils
found_name = None
null_ok = False
finite_ok = False

for finder, modname, ispkg in pkgutil.iter_modules(sglang.srt.utils.__path__):
    try:
        mod = importlib.import_module(f"sglang.srt.utils.{modname}")
    except Exception:
        continue
    for attr_name in dir(mod):
        fn = getattr(mod, attr_name, None)
        if not callable(fn) or attr_name.startswith("_"):
            continue
        try:
            result = fn(test_data)
            if not isinstance(result, (bytes, bytearray)) or len(result) < 5:
                continue
            parsed = json.loads(result)
            logprobs = parsed.get("logprobs", [])
            if (
                len(logprobs) >= 4
                and all(v is None for v in logprobs[:3])
                and logprobs[3] == 0.5
            ):
                found_name = f"{modname}.{attr_name}"
                null_ok = True
                finite_ok = True
                break
        except Exception:
            pass
    if found_name:
        break

if found_name:
    print(f"SERIALIZER_FOUND:{found_name}")
else:
    print("SERIALIZER_NOT_FOUND")
print(f"NULL_OK:{null_ok}")
print(f"FINITE_OK:{finite_ok}")
""")

if "IMPORT_SKIP" in stdout:
    err = stdout.split("IMPORT_SKIP:")[1].split("\n")[0]
    check("sglang JSON utility serializes NaN/Inf to null", False,
          f"sglang import failed: {err}")
    check("Serialized output: null for NaN/Inf, value preserved for finite", False,
          "Import failed")
elif "SERIALIZER_NOT_FOUND" in stdout:
    check("sglang JSON utility serializes NaN/Inf to null", False,
          "No utility in sglang.srt.utils handles NaN/Inf → null — fix not applied")
    check("Serialized output: null for NaN/Inf, value preserved for finite", False,
          "No serializer found")
else:
    found = "SERIALIZER_FOUND:" in stdout
    null_ok = "NULL_OK:True" in stdout
    finite_ok = "FINITE_OK:True" in stdout
    check("sglang JSON utility serializes NaN/Inf to null", found and null_ok,
          stdout.strip()[:200] if not (found and null_ok) else "")
    check("Serialized output: null for NaN/Inf, value preserved for finite",
          null_ok and finite_ok,
          f"null_ok={null_ok}, finite_ok={finite_ok}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
