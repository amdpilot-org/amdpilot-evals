#!/usr/bin/env python3
"""
Generic KernelBench test harness for amdpilot integration.

Evaluates a generated Triton kernel against the PyTorch reference for a
specific KernelBench problem. Prints SCORE for amdpilot's benchmark_hook.

Usage:
    python test_harness.py --level 1 --problem-id 42
    python test_harness.py --level 2 --problem-id 5 --kernel-path /workspace/solution.py

The harness looks for the generated kernel at:
    /workspace/generated_kernel.py  (default)

SCORE mapping:
    0   = compile failure
    25  = compiles but incorrect
    50  = correct but slower than baseline
    50 + 50 * min(speedup/5, 1) = correct and fast (max 100 at 5x speedup)
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.utils import set_gpu_arch, read_file

set_gpu_arch(["gfx950"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem-id", type=int, required=True)
    parser.add_argument("--kernel-path", default="/workspace/generated_kernel.py")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    dataset = construct_kernelbench_dataset(level=args.level, source="local")
    problem = dataset.get_problem_by_id(args.problem_id)
    ref_src = problem.code

    if not os.path.exists(args.kernel_path):
        print(f"Kernel file not found: {args.kernel_path}")
        print("SCORE: 0")
        sys.exit(1)

    kernel_src = read_file(args.kernel_path)
    device = torch.device(f"cuda:{args.device}")

    try:
        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=True,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            build_dir=None,
            device=device,
            backend="triton",
            precision=get_torch_dtype_from_string("fp32"),
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        print("SCORE: 0")
        sys.exit(1)

    if not result.compiled:
        print(f"Compilation failed: {result.metadata}")
        print("SCORE: 0")
        sys.exit(1)

    if not result.correctness:
        print(f"Correctness check failed: {result.metadata}")
        print("SCORE: 25")
        sys.exit(0)

    speedup = 1.0
    if result.ref_runtime > 0 and result.runtime > 0:
        speedup = result.ref_runtime / result.runtime

    if speedup >= 1.0:
        score = 50.0 + 50.0 * min(speedup / 5.0, 1.0)
    else:
        score = 50.0

    print(f"RUNTIME_MS: {result.runtime:.3f}")
    print(f"REF_RUNTIME_MS: {result.ref_runtime:.3f}")
    print(f"SPEEDUP: {speedup:.3f}")
    print(f"SCORE: {score:.1f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
