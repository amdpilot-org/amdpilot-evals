#!/usr/bin/env python3
"""Compile final KernelBench results across all levels."""

import json
import os
from collections import Counter

run_dir = "/workspace/KernelBench/runs/amdpilot_triton_qwen35_v1"

def analyze_level(filepath, level_name):
    with open(filepath) as f:
        data = json.load(f)
    total = len(data)
    compiled = correct = 0
    fast0 = fast1 = fast2 = 0
    speedups = []
    errors = Counter()

    for pid, samples in data.items():
        s = samples[0]
        if s["compiled"]:
            compiled += 1
        if s["correctness"]:
            correct += 1
            fast0 += 1
        if s.get("correctness") and s.get("runtime", -1) > 0 and s.get("ref_runtime", -1) > 0:
            sp = s["ref_runtime"] / s["runtime"]
            speedups.append((pid, sp))
            if sp > 1.0:
                fast1 += 1
            if sp > 2.0:
                fast2 += 1
        if not s["compiled"]:
            meta = s.get("metadata", {})
            for key in ["runtime_error_name", "cuda_error_name", "other_error_name", "error"]:
                if key in meta:
                    errors[meta[key]] += 1
                    break

    sep = "=" * 60
    print(f"\n{sep}")
    print(level_name)
    print(sep)
    print(f"  Total problems:   {total}")
    print(f"  Compiled:         {compiled}/{total} ({compiled/total*100:.1f}%)")
    print(f"  Correct (fast_0): {correct}/{total} ({correct/total*100:.1f}%)")
    if speedups:
        print(f"  fast_1 (>1x):     {fast1}/{total} ({fast1/total*100:.1f}%)")
        print(f"  fast_2 (>2x):     {fast2}/{total} ({fast2/total*100:.1f}%)")
        sps = [s for _, s in speedups]
        print(f"  Mean speedup:     {sum(sps)/len(sps):.2f}x")
        print(f"  Max speedup:      {max(sps):.2f}x")
        print(f"  Median speedup:   {sorted(sps)[len(sps)//2]:.2f}x")
    if errors:
        print("  Top errors:")
        for err, cnt in errors.most_common(5):
            short = err[:60] + "..." if len(err) > 60 else err
            print(f"    [{cnt}] {short}")

    return {
        "total": total, "compiled": compiled, "correct": correct,
        "fast1": fast1, "fast2": fast2, "speedups": speedups,
    }


r1 = analyze_level(f"{run_dir}/eval_results_level1_v2.json", "Level 1 - Single Kernel Ops (100 problems)")
r2 = analyze_level(f"{run_dir}/eval_results_level2.json", "Level 2 - Fused Patterns (100 problems)")
r3 = analyze_level(f"{run_dir}/eval_results_level3.json", "Level 3 - Full Models (50 problems)")

total = r1["total"] + r2["total"] + r3["total"]
compiled = r1["compiled"] + r2["compiled"] + r3["compiled"]
correct = r1["correct"] + r2["correct"] + r3["correct"]
fast1 = r1["fast1"] + r2["fast1"] + r3["fast1"]
fast2 = r1["fast2"] + r2["fast2"] + r3["fast2"]

sep = "=" * 60
print(f"\n{sep}")
print("OVERALL SUMMARY (All Levels)")
print(sep)
print(f"  Total problems:   {total}")
print(f"  Compiled:         {compiled}/{total} ({compiled/total*100:.1f}%)")
print(f"  Correct (fast_0): {correct}/{total} ({correct/total*100:.1f}%)")
print(f"  fast_1 (>1x):     {fast1}/{total} ({fast1/total*100:.1f}%)")
print(f"  fast_2 (>2x):     {fast2}/{total} ({fast2/total*100:.1f}%)")
print(f"\n  Hardware: AMD Instinct MI355X")
print(f"  Backend:  Triton (ROCm)")
print(f"  Model:    Qwen3.5-397B-A17B")
print(f"  Prompt:   KernelBench default one-shot (no AMD-specific guidance)")

result_json = {
    "hardware": "AMD Instinct MI355X",
    "backend": "triton",
    "model": "Qwen3.5-397B-A17B",
    "levels": {
        "level_1": {"total": r1["total"], "compiled": r1["compiled"], "correct": r1["correct"],
                     "fast_1": r1["fast1"], "fast_2": r1["fast2"]},
        "level_2": {"total": r2["total"], "compiled": r2["compiled"], "correct": r2["correct"],
                     "fast_1": r2["fast1"], "fast_2": r2["fast2"]},
        "level_3": {"total": r3["total"], "compiled": r3["compiled"], "correct": r3["correct"],
                     "fast_1": r3["fast1"], "fast_2": r3["fast2"]},
    },
    "overall": {"total": total, "compiled": compiled, "correct": correct,
                "fast_1": fast1, "fast_2": fast2},
}
with open("/workspace/amdpilot/evals/kernelbench/final_results.json", "w") as f:
    json.dump(result_json, f, indent=2)
print(f"\nResults saved to /workspace/amdpilot/evals/kernelbench/final_results.json")
