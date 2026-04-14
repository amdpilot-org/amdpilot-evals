#!/usr/bin/env python3
"""Test harness for sglang-glm5-optimize.

Runs the GLM-5.1-FP8 decode benchmark and scores based on latency improvement.
The benchmark runs sglang.bench_one_batch with fixed workload params (TP=8,
batch=1, input=1024, output=128) and reports decode median latency in ms.

Scoring (continuous, open-ended):
  - 0.0: benchmark fails to run or extract metric
  - 33.3: benchmark runs but no improvement over baseline
  - 33.3+: score increases with % improvement (no cap — lower latency is always better)
    e.g. 10% improvement = 43.3, 30% = 63.3, 50% = 83.3
"""
import re
import subprocess
import sys

# Baseline latency (ms) — established during Docker validation.
# BASELINE_MS: unoptimized decode latency (what you get out of the box)
# This is an open-ended optimization task: lower latency is always better.
# These values are calibrated for GLM-5.1-FP8 on 8x MI355X with TP=8, batch=1.
BASELINE_MS = 23.6    # decode median on 8x MI355X, TP=8, batch=1 (unoptimized)


def score_latency(measured, baseline):
    """Score latency on a continuous scale based on % improvement.

    - At or above baseline: 33.3 (benchmark ran successfully, no improvement)
    - Below baseline: 33.3 + improvement_pct (no cap — lower is always better)
      e.g. 10% improvement = 43.3, 30% improvement = 63.3, 50% = 83.3
    """
    if measured >= baseline:
        return 33.3
    improvement_pct = (baseline - measured) / baseline * 100.0
    return 33.3 + improvement_pct


print("=" * 60)
print("sglang-glm5-optimize test harness")
print("=" * 60)

# --- Run benchmark ---
print("\n--- Running GLM-5.1-FP8 decode benchmark ---")
print("(First run takes ~5 min for model loading + CUDA graph compilation)")

try:
    result = subprocess.run(
        ["bash", "/workspace/bench_glm5.sh"],
        capture_output=True, text=True, timeout=600,
        cwd="/workspace",
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
except subprocess.TimeoutExpired:
    print("  [FAIL] bench_glm5.sh timed out after 600s")
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
print(f"  Measured:               {decode_ms:.1f} ms")

score = score_latency(decode_ms, BASELINE_MS)

if decode_ms >= BASELINE_MS:
    print(f"  No improvement over baseline")
else:
    improvement_pct = (BASELINE_MS - decode_ms) / BASELINE_MS * 100
    print(f"  Improvement: {improvement_pct:.1f}% over baseline")

print(f"\nSCORE: {score:.1f}")
