#!/usr/bin/env python3
"""Test harness for aiter blockscale GEMM correctness.

Tests (behavioral):
  1. Run blockscale GEMM under memory pressure and check output.
  2. Run multiple GEMM calls with intervening alloc/free cycles.
  3. Compare against torch reference matmul.
"""

import os
import subprocess
import sys

_PY = "/opt/venv/bin/python3"
AITER_PATH = "/workspace/aiter"

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}: {detail}")


def check_gemm_correctness():
    """Run blockscale GEMM under memory pressure and check output."""
    test_code = r'''
import torch
import sys
import gc
sys.path.insert(0, "/sgl-workspace/aiter")

torch.cuda.set_device(0)
device = "cuda:0"

passed = 0
total = 0
max_errors = []

try:
    from aiter import gemm_a8w8_blockscale
except ImportError:
    try:
        from aiter.ops.gemm import gemm_a8w8_blockscale_cktile as gemm_a8w8_blockscale
    except ImportError:
        try:
            from aiter import tuned_gemm
            gemm_a8w8_blockscale = None
        except ImportError:
            print("GEMM_RESULT: 0/1 import_failed")
            sys.exit(0)

# Test dimensions (typical MoE shapes)
test_cases = [
    (1, 7168, 2048),   # decode batch=1
    (4, 7168, 2048),   # decode batch=4
    (1, 2048, 7168),   # down_proj
    (8, 7168, 2048),   # decode batch=8
]

for M, K, N in test_cases:
    for trial in range(3):
        total += 1
        try:
            torch.manual_seed(42 + trial)

            # Create quantized inputs
            A = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
            B = torch.randint(-127, 127, (N, K), dtype=torch.int8, device=device)

            block_size = 128
            x_scale = torch.rand(M, (K + block_size - 1) // block_size,
                                 dtype=torch.float32, device=device) * 0.01 + 0.001
            w_scale = torch.rand(N, (K + block_size - 1) // block_size,
                                 dtype=torch.float32, device=device) * 0.01 + 0.001

            # Create memory pressure: alloc and free many tensors to
            # force the caching allocator to recycle memory blocks
            pressure_tensors = []
            for _ in range(50):
                t = torch.randn(1024, 1024, device=device)
                pressure_tensors.append(t)
            del pressure_tensors
            gc.collect()
            torch.cuda.empty_cache()

            # More targeted pressure: alloc/free tensors of similar size
            for _ in range(20):
                dummy = torch.randn_like(x_scale)
                del dummy

            # Run blockscale GEMM
            try:
                if gemm_a8w8_blockscale is not None:
                    out = gemm_a8w8_blockscale(A, B, x_scale, w_scale)
                else:
                    # Try alternate API
                    tg = tuned_gemm.TunedGemm()
                    out = tg.gemm_a8w8_blockscale(A, B, x_scale, w_scale)
            except Exception as e:
                # Try yet another API path
                try:
                    from aiter.jit.core import compile_ops
                    from aiter.ops.gemm import gemm_a8w8_blockscale_cktile_impl
                    out = gemm_a8w8_blockscale_cktile_impl(A, B, x_scale, w_scale)
                except:
                    check(f"gemm_{M}x{K}x{N}_t{trial}", False, f"API not found: {e}")
                    continue

            torch.cuda.synchronize()

            # Reference computation in float
            A_f = A.float()
            B_f = B.float()
            # Dequantize: each block of block_size elements uses its own scale
            A_deq = torch.zeros(M, K, device=device)
            B_deq = torch.zeros(N, K, device=device)
            for i in range(0, K, block_size):
                end = min(i + block_size, K)
                block_idx = i // block_size
                A_deq[:, i:end] = A_f[:, i:end] * x_scale[:, block_idx:block_idx+1]
                B_deq[:, i:end] = B_f[:, i:end] * w_scale[:, block_idx:block_idx+1]

            ref = torch.mm(A_deq, B_deq.t()).to(out.dtype)

            # Compare
            max_err = (out.float() - ref.float()).abs().max().item()
            max_errors.append(max_err)

            # Tolerance: correct output should have max_err < 1.0
            is_pass = max_err < 1.0
            if is_pass:
                passed += 1

        except Exception as e:
            max_errors.append(float('inf'))

errors_str = ",".join([f"{e:.4f}" for e in max_errors[:10]])
print(f"GEMM_RESULT: {passed}/{total} max_errors=[{errors_str}]")
'''

    result = subprocess.run(
        [_PY, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": "/sgl-workspace/aiter",
             "HIP_VISIBLE_DEVICES": "0"},
    )

    if result.returncode != 0:
        check("gemm_correctness", False,
              f"Test crashed: {result.stderr[-500:]}")
        return

    for line in result.stdout.split("\n"):
        if "GEMM_RESULT:" in line:
            parts = line.split(":")[1].strip().split(" ")
            ratio = parts[0].split("/")
            passed = int(ratio[0])
            total = int(ratio[1])
            check("gemm_correctness",
                  passed == total,
                  f"{passed}/{total} GEMM checks passed. {' '.join(parts[1:])}")
            return

    check("gemm_correctness", False, "No result line found")


def main():
    print("=" * 60)
    print("Aiter Blockscale GEMM Correctness Test")
    print("=" * 60)

    print("\n--- Behavioral: GEMM Under Memory Pressure ---")
    check_gemm_correctness()

    print(f"\n--- Results ---")
    print(f"  {checks_passed}/{checks_total} checks passed")

    score = checks_passed / checks_total * 100.0 if checks_total > 0 else 0.0
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
