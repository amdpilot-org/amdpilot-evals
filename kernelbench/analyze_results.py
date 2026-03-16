#!/usr/bin/env python3
"""Analyze KernelBench evaluation results and compute fast_p metrics."""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_eval_results(runs_dir: str, run_name: str, level: int) -> dict | None:
    eval_path = os.path.join(runs_dir, run_name, "eval_results.json")
    if not os.path.exists(eval_path):
        return None
    with open(eval_path) as f:
        return json.load(f)


def load_baseline_times(runs_dir: str, run_name: str) -> dict | None:
    baseline_path = os.path.join(runs_dir, run_name, "baseline_times.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            return json.load(f)
    return None


def compute_fast_p(eval_results: dict, p_values: list[float] = [0, 1, 2]) -> dict:
    """Compute fast_p metrics: fraction of problems that are correct and have speedup > p."""
    total = len(eval_results)
    if total == 0:
        return {f"fast_{p}": 0.0 for p in p_values}

    metrics = {}
    compiled = 0
    correct = 0
    speedups = []

    for problem_id, samples in eval_results.items():
        best = samples[0] if samples else None
        if best is None:
            continue
        if best.get("compiled"):
            compiled += 1
        if best.get("correctness"):
            correct += 1
            runtime = best.get("runtime", -1)
            ref_time = best.get("ref_runtime", -1)
            if ref_time <= 0:
                ref_time = best.get("runtime_stats", {}).get("ref_runtime", -1)
            if runtime > 0 and ref_time > 0:
                speedup = ref_time / runtime
                speedups.append((problem_id, speedup))

    metrics["total_problems"] = total
    metrics["compiled"] = compiled
    metrics["compiled_rate"] = compiled / total
    metrics["correct"] = correct
    metrics["correct_rate"] = correct / total

    for p in p_values:
        fast_count = sum(1 for _, s in speedups if s > p)
        metrics[f"fast_{p}"] = fast_count / total
        metrics[f"fast_{p}_count"] = fast_count

    if speedups:
        all_speedups = [s for _, s in speedups]
        metrics["mean_speedup"] = sum(all_speedups) / len(all_speedups)
        metrics["max_speedup"] = max(all_speedups)
        metrics["min_speedup"] = min(all_speedups)
        metrics["median_speedup"] = sorted(all_speedups)[len(all_speedups) // 2]

    error_types = defaultdict(int)
    for problem_id, samples in eval_results.items():
        for sample in samples:
            if not sample.get("compiled"):
                meta = sample.get("metadata", {})
                for key in ["cuda_error_name", "other_error_name", "error"]:
                    if key in meta:
                        error_types[meta[key]] += 1
                        break
                else:
                    error_types["unknown"] += 1

    metrics["error_distribution"] = dict(
        sorted(error_types.items(), key=lambda x: -x[1])[:20]
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze KernelBench results")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_results = {}
    summary = {"run_name": args.run_name, "levels": {}}

    for level in [1, 2, 3]:
        eval_results = load_eval_results(args.runs_dir, args.run_name, level)
        if eval_results is None:
            print(f"Level {level}: No eval_results.json found, skipping.")
            continue

        metrics = compute_fast_p(eval_results)
        summary["levels"][f"level_{level}"] = metrics

        print(f"\n{'='*60}")
        print(f"Level {level} Results")
        print(f"{'='*60}")
        print(f"  Total problems:  {metrics['total_problems']}")
        print(f"  Compiled:        {metrics['compiled']} ({metrics['compiled_rate']:.1%})")
        print(f"  Correct:         {metrics['correct']} ({metrics['correct_rate']:.1%})")
        print(f"  fast_0 (correct): {metrics['fast_0']:.1%}")
        print(f"  fast_1 (>1x):     {metrics.get('fast_1', 0):.1%}")
        print(f"  fast_2 (>2x):     {metrics.get('fast_2', 0):.1%}")
        if "mean_speedup" in metrics:
            print(f"  Mean speedup:    {metrics['mean_speedup']:.2f}x")
            print(f"  Max speedup:     {metrics['max_speedup']:.2f}x")
        if metrics.get("error_distribution"):
            print(f"  Top errors:")
            for err, count in list(metrics["error_distribution"].items())[:5]:
                print(f"    {err}: {count}")

    totals = {"total": 0, "compiled": 0, "correct": 0}
    for level_data in summary["levels"].values():
        totals["total"] += level_data["total_problems"]
        totals["compiled"] += level_data["compiled"]
        totals["correct"] += level_data["correct"]

    if totals["total"] > 0:
        summary["overall"] = {
            "total_problems": totals["total"],
            "compiled_rate": totals["compiled"] / totals["total"],
            "correct_rate": totals["correct"] / totals["total"],
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
