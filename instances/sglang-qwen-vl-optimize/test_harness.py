#!/usr/bin/env python3
"""Test harness for sglang-qwen-vl-optimize.

Runs the Qwen3-VL-8B triton-attention benchmark and scores based on throughput
improvement. The benchmark runs a serving workload with image prompts and
reports output throughput in tokens per second.

Scoring (continuous, open-ended):
  - 0.0: benchmark fails to run or extract metric
  - 33.3: benchmark runs but no improvement over baseline
  - 33.3+: score increases with % improvement (no cap)
"""
import re
import subprocess
import sys

# Baseline throughput (tok/s) — established during Docker validation.
# BASELINE_TOKS: unoptimized output throughput (what you get out of the box)
# Calibrated for Qwen3-VL-8B-Instruct on MI355X with --attention-backend triton,
# bench_serving image workload (128 prompts, concurrency=16).
BASELINE_TOKS = 1235.0


def score_throughput(measured, baseline):
    """Score throughput on a continuous scale based on % improvement.

    - At or below baseline: 33.3 (benchmark ran successfully, no improvement)
    - Above baseline: 33.3 + improvement_pct (no cap — higher is always better)
      e.g. 10% improvement = 43.3, 30% improvement = 63.3
    """
    if measured <= baseline:
        return 33.3
    improvement_pct = (measured - baseline) / baseline * 100.0
    return 33.3 + improvement_pct


print("=" * 60)
print("sglang-qwen-vl-optimize test harness")
print("=" * 60)

# --- Run benchmark ---
print("\n--- Running Qwen3-VL triton attention benchmark ---")
print("(First run takes 15-25 min for model loading + CUDA graphs + warmup)")

try:
    result = subprocess.run(
        ["bash", "/workspace/bench_qwen_vl.sh"],
        capture_output=True, text=True, timeout=2400,
        cwd="/workspace",
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
except subprocess.TimeoutExpired:
    print("  [FAIL] bench_qwen_vl.sh timed out after 2400s")
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

# --- Extract output throughput ---
print("\n--- Throughput extraction ---")
match = re.search(r'Output throughput \(tok/s\):\s+([\d.]+)', stdout)

if not match:
    print("  [FAIL] Could not find 'Output throughput (tok/s):' in benchmark output")
    print(f"\nSCORE: 0.0")
    sys.exit(1)

throughput = float(match.group(1))
print(f"  [PASS] Output throughput: {throughput:.1f} tok/s")

# --- Verify triton backend ---
triton_used = "--attention-backend triton" in stdout or "attention_backend=triton" in stdout
if not triton_used:
    print("  [WARN] Could not confirm triton backend in output")
print("  [PASS] Benchmark completed")

# --- Score based on improvement ---
print(f"\n--- Performance scoring ---")
print(f"  Baseline (unoptimized): {BASELINE_TOKS:.1f} tok/s")
print(f"  Measured:               {throughput:.1f} tok/s")

score = score_throughput(throughput, BASELINE_TOKS)

if throughput <= BASELINE_TOKS:
    print(f"  No improvement over baseline")
else:
    improvement_pct = (throughput - BASELINE_TOKS) / BASELINE_TOKS * 100
    print(f"  Improvement: {improvement_pct:.1f}% over baseline")

print(f"\nSCORE: {score:.1f}")
