#!/usr/bin/env python3
"""Validation harness for sglang-kimi-w4a16-moe-optimize.

Two-phase validation:
  Phase 1 (smoke gate): --output-len 2 validates the model loads and the
    MoE dispatch path executes a decode step. Fast (~3-5 min).
    If this fails, SCORE: 0.0 immediately — no need to run the full benchmark.
  Phase 2 (full benchmark): --output-len 2048 measures decode latency.
    Only runs if phase 1 passes.

The continuous optimization metric (decode_median_ms) is tracked by
the orchestrator via task.yaml's metric_pattern.

SCORE: 0 or 100
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

SMOKE = "/workspace/bench_kimi_w4a16_smoke.sh"
BENCH = "/workspace/bench_kimi_w4a16.sh"
TIMEOUT_SMOKE = 600   # 10 min for smoke test
TIMEOUT_BENCH = 1800  # 30 min for full benchmark
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

# ---- Phase 1: Smoke gate ----
print("\n--- Phase 1: Smoke gate (--output-len 1) ---")

check("Smoke script exists", os.path.isfile(SMOKE), f"Missing {SMOKE}")
check("Benchmark script exists", os.path.isfile(BENCH), f"Missing {BENCH}")

if not os.path.isfile(SMOKE) or not os.path.isfile(BENCH):
    print("\nSCORE: 0.0")
    sys.exit(1)

try:
    smoke_result = subprocess.run(
        ["bash", SMOKE],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SMOKE,
        env=os.environ.copy(),
    )
    smoke_output = (smoke_result.stdout or "") + (
        "\n" + smoke_result.stderr if smoke_result.stderr else ""
    )
except subprocess.TimeoutExpired:
    check(
        "Smoke test completed within timeout",
        False,
        f"Timed out after {TIMEOUT_SMOKE}s",
    )
    print("\nSCORE: 0.0")
    sys.exit(1)

# Print truncated smoke output for diagnostics
if len(smoke_output) > 3000:
    print(smoke_output[-3000:])
else:
    print(smoke_output)

smoke_pass = "SMOKE_PASS" in smoke_output
check(
    "Smoke test passed (model loads + single decode)",
    smoke_pass,
    "Model failed to load or MoE decode failed",
)

if not smoke_pass:
    # Print last 50 lines for debugging
    lines = smoke_output.strip().split("\n")
    print("\n--- Last 50 lines of smoke output ---")
    for line in lines[-50:]:
        print(line)
    print("\nSCORE: 0.0")
    sys.exit(1)

# ---- Phase 2: Full benchmark ----
print("\n--- Phase 2: Full benchmark (--output-len 2048) ---")

try:
    result = subprocess.run(
        ["bash", BENCH],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_BENCH,
        env=os.environ.copy(),
    )
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
except subprocess.TimeoutExpired:
    check(
        "Benchmark completed within timeout",
        False,
        f"Timed out after {TIMEOUT_BENCH}s",
    )
    print("\nSCORE: 0.0")
    sys.exit(1)

print(output)

check(
    "Benchmark exited successfully",
    result.returncode == 0,
    f"exit={result.returncode}",
)

metric_match = METRIC_RE.search(output)
check(
    "Metric line present",
    metric_match is not None,
    "Could not find decode median line",
)

context_ok = CONTEXT_RE.search(output) is not None
check(
    "Benchmark context matches expected workload",
    context_ok,
    "Missing 'tp=8 batch=1 in=8192 out=2048' context",
)

metric_value = None
if metric_match:
    try:
        metric_value = float(metric_match.group(1))
    except ValueError:
        metric_value = None

check(
    "Metric is parseable and positive",
    metric_value is not None and metric_value > 0,
    f"value={metric_value!r}",
)

score = 100.0 if checks_passed == checks_total else 0.0
print()
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
