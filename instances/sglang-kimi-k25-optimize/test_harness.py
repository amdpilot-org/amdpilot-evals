#!/usr/bin/env python3
"""Test harness for sglang-kimi-k25-optimize.

Runs the Kimi-K2.5 (1T MoE) decode benchmark and scores based on latency
improvement. The benchmark runs sglang.bench_one_batch with fixed workload
params (TP=8, batch=1, input=8192, output=2048) and reports decode median
latency in ms.

Scoring (continuous, not binary):
  - 0.0: benchmark fails to run or extract metric
  - 33.3: benchmark runs but no improvement over baseline
  - 33.3–100.0: linear interpolation based on improvement toward target
  - 100.0: latency at or below target
"""
import re
import subprocess
import sys

# Baseline and target latencies (ms) — established during Docker validation.
# BASELINE_MS: unoptimized decode latency (what you get out of the box)
# TARGET_MS: latency that earns 100.0 (meaningful optimization achieved)
# These values are calibrated for Kimi-K2.5 on 8x MI355X with TP=8, batch=1,
# input=8192, output=2048.
BASELINE_MS = 999.0   # placeholder — set during Docker baseline measurement
TARGET_MS = 999.0      # placeholder — set to meaningful improvement threshold


def score_latency(measured, baseline, target):
    """Score latency on a 0-100 scale with partial credit.

    - At or above baseline: 33.3 (benchmark ran successfully, no improvement)
    - Between baseline and target: linear interpolation 33.3 -> 100.0
    - At or below target: 100.0
    """
    if measured >= baseline:
        return 33.3
    if measured <= target:
        return 100.0
    # Linear interpolation between baseline (33.3) and target (100.0)
    improvement_ratio = (baseline - measured) / (baseline - target)
    return 33.3 + improvement_ratio * (100.0 - 33.3)


print("=" * 60)
print("sglang-kimi-k25-optimize test harness")
print("=" * 60)

# --- Run benchmark ---
print("\n--- Running Kimi-K2.5 decode benchmark ---")
print("(First run takes a long time for model loading + CUDA graph compilation)")

try:
    result = subprocess.run(
        ["bash", "/workspace/bench_kimi_k25.sh"],
        capture_output=True, text=True, timeout=1200,
        cwd="/workspace",
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
except subprocess.TimeoutExpired:
    print("  [FAIL] bench_kimi_k25.sh timed out after 1200s")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

# Print tail of benchmark output for debugging
print("--- Benchmark output (last 20 lines) ---")
for line in stdout.splitlines()[-20:]:
    print(f"  {line}")
if result.returncode != 0:
    print("--- stderr (last 10 lines) ---")
    for line in stderr.splitlines()[-10:]:
        print(f"  {line}")
print("---")

# --- Extract decode median ---
print("\n--- Latency extraction ---")
match = re.search(r'Decode median \(ms\):\s+([\d.]+)', stdout)

if not match:
    print("  [FAIL] Could not find 'Decode median (ms):' in benchmark output")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

decode_ms = float(match.group(1))
print(f"  [PASS] Decode median: {decode_ms:.1f} ms")

# --- Check CUDA graphs ---
cuda_graph_disabled = "--disable-cuda-graph" in stdout or "--disable-cuda-graph" in stderr
if cuda_graph_disabled:
    print("  [FAIL] Benchmark ran with --disable-cuda-graph (not allowed)")
    print(f"\nSCORE: 0.0")
    sys.exit(1)
print("  [PASS] CUDA graphs enabled")

# --- Score based on improvement ---
print(f"\n--- Performance scoring ---")
print(f"  Baseline (unoptimized): {BASELINE_MS:.1f} ms")
print(f"  Target (100.0 score):   {TARGET_MS:.1f} ms")
print(f"  Measured:               {decode_ms:.1f} ms")

score = score_latency(decode_ms, BASELINE_MS, TARGET_MS)

if decode_ms >= BASELINE_MS:
    print(f"  No improvement over baseline")
elif decode_ms <= TARGET_MS:
    improvement_pct = (BASELINE_MS - decode_ms) / BASELINE_MS * 100
    print(f"  Target reached! {improvement_pct:.1f}% improvement over baseline")
else:
    improvement_pct = (BASELINE_MS - decode_ms) / BASELINE_MS * 100
    target_pct = (BASELINE_MS - TARGET_MS) / BASELINE_MS * 100
    print(f"  Partial improvement: {improvement_pct:.1f}% (need {target_pct:.1f}% for full score)")

print(f"\nSCORE: {score:.1f}")
