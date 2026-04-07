#!/usr/bin/env python3
"""Validation harness for sglang-kimi-w4a16-moe-optimize.

Runs the W4A16 MoE benchmark and validates the output.
If the benchmark crashes (model fails to load), score is 0.
If it runs successfully and produces a valid metric, score is 100.

The continuous optimization metric (decode_median_ms) is tracked by
the orchestrator via task.yaml's metric_pattern.

SCORE: 0 or 100
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

BENCH = "/workspace/bench_kimi_w4a16.sh"
TIMEOUT_S = 1800
METRIC_RE = re.compile(r"Decode median \(ms\):\s+([\d.]+)")
CONTEXT_RE = re.compile(r"tp=8 batch=1 in=8192 out=2048")

checks_passed = 0
checks_total = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
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
print("sglang-kimi-w4a16-moe-optimize test harness")
print("=" * 60)

check("Benchmark script exists", os.path.isfile(BENCH), f"Missing {BENCH}")

if not os.path.isfile(BENCH):
    print("\nSCORE: 0.0")
    sys.exit(1)

try:
    result = subprocess.run(
        ["bash", BENCH],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env=os.environ.copy(),
    )
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
except subprocess.TimeoutExpired:
    check("Benchmark completed within timeout", False, f"Timed out after {TIMEOUT_S}s")
    print("\nSCORE: 0.0")
    sys.exit(1)

print(output)

check("Benchmark exited successfully", result.returncode == 0, f"exit={result.returncode}")

metric_match = METRIC_RE.search(output)
check("Metric line present", metric_match is not None, "Could not find decode median line")

context_ok = CONTEXT_RE.search(output) is not None
check("Benchmark context matches expected workload", context_ok, "Missing 'tp=8 batch=1 in=8192 out=2048' context")

metric_value = None
if metric_match:
    try:
        metric_value = float(metric_match.group(1))
    except ValueError:
        metric_value = None

check("Metric is parseable and positive", metric_value is not None and metric_value > 0, f"value={metric_value!r}")

score = 100.0 if checks_passed == checks_total else 0.0
print()
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
