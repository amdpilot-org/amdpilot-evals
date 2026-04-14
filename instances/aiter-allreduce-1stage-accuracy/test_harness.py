#!/usr/bin/env python3
"""Test harness for aiter fused allreduce+RMSNorm 1-stage accuracy.

Tests (behavioral):
  1. Compare fused 1-stage output vs unfused reference for small inputs.
  2. Repeat across multiple random seeds.
  3. Check that max absolute error and divergent element percentage are within tolerance.
"""

import os
import subprocess
import sys
import tempfile
import json

_PY = "/opt/venv/bin/python3"

# The actual test runs as a distributed subprocess on 2 GPUs
_WORKER_SCRIPT = r'''
import os
import sys
import json
import torch
import torch.distributed as dist

def run_test(rank, world_size, result_file):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Create a CPU (gloo) group for CustomAllreduce — it requires non-NCCL
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    # Create NCCL group for reference allreduce
    nccl_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")

    results = {"checks": []}

    try:
        # Import aiter custom allreduce
        sys.path.insert(0, "/sgl-workspace/aiter")
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        # The 1-stage kernel is used when total bytes <= 128KB
        # For bf16 hidden_dim=4096: 4096 * 2 bytes = 8KB per row
        hidden_dim = 4096
        max_m = 16
        eps = 1e-6

        ca = CustomAllreduce(
            group=cpu_group,
            device=torch.device(device),
        )

        num_seeds = 10
        total_checks = 0
        passed_checks = 0

        for seed in range(num_seeds):
            torch.manual_seed(42 + seed + rank * 1000)

            for m in [1, 2, 4]:
                inp = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
                residual = torch.randn(m, hidden_dim, dtype=torch.bfloat16, device=device)
                weight = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device).abs() + 0.1

                # Keep copies for unfused reference
                inp_ref = inp.clone()
                residual_ref = residual.clone()

                # --- Fused path (1-stage) ---
                res_out_fused = torch.empty_like(residual)
                out_fused = torch.empty_like(inp)

                try:
                    ca.fused_ar_rms(
                        inp, residual,
                        res_out=res_out_fused,
                        out=out_fused,
                        w=weight,
                        eps=eps,
                        use_1stage=True,
                    )
                    torch.cuda.synchronize()
                except Exception as e:
                    # If fused path fails, try alternate API
                    try:
                        result = ca.custom_fused_ar_rms(
                            inp, residual, weight, eps, use_1stage=True
                        )
                        if result is not None:
                            out_fused, res_out_fused = result
                        else:
                            results["checks"].append({
                                "name": f"fused_api_seed{seed}_m{m}",
                                "pass": False,
                                "detail": f"Fused path returned None: {e}"
                            })
                            total_checks += 1
                            continue
                    except Exception as e2:
                        results["checks"].append({
                            "name": f"fused_api_seed{seed}_m{m}",
                            "pass": False,
                            "detail": f"Fused path failed: {e2}"
                        })
                        total_checks += 1
                        continue

                # --- Unfused reference path ---
                # Step 1: allreduce
                dist.all_reduce(inp_ref, group=nccl_group)
                torch.cuda.synchronize()

                # Step 2: add residual (in bf16, matching unfused numerical sequence)
                res_out_ref = inp_ref.to(torch.bfloat16) + residual_ref

                # Step 3: RMSNorm
                variance = res_out_ref.float().pow(2).mean(-1, keepdim=True)
                out_ref = (res_out_ref.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16)
                out_ref = out_ref * weight

                # --- Compare ---
                total_checks += 1

                # Primary check: residual output (directly tests allreduce precision)
                res_max_err = (res_out_fused - res_out_ref).abs().max().item()
                n_res_elements = res_out_fused.numel()
                n_res_divergent = (res_out_fused != res_out_ref).sum().item()
                res_divergent_pct = n_res_divergent / n_res_elements * 100

                # Secondary check: RMSNorm output (relaxed — reduction order
                # differences between fused kernel block-reduce and PyTorch
                # cause expected divergence even when residual is exact)
                out_max_err = (out_fused - out_ref).abs().max().item()

                # Pass if residual output matches closely AND rmsnorm error is bounded
                is_pass = (res_divergent_pct < 1.0 and out_max_err < 0.1)
                if is_pass:
                    passed_checks += 1

                results["checks"].append({
                    "name": f"seed{seed}_m{m}",
                    "pass": is_pass,
                    "out_max_err": out_max_err,
                    "res_max_err": res_max_err,
                    "res_divergent_pct": res_divergent_pct,
                })

        results["total"] = total_checks
        results["passed"] = passed_checks

    except Exception as e:
        results["error"] = str(e)
        results["total"] = 1
        results["passed"] = 0

    finally:
        dist.destroy_process_group()

    # Only rank 0 writes results
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    result_file = sys.argv[1]
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    run_test(rank, world_size, result_file)
'''


def main():
    print("=" * 60)
    print("Aiter Fused Allreduce+RMSNorm 1-Stage Accuracy Test")
    print("=" * 60)

    # Write worker script to temp file
    worker_path = "/tmp/allreduce_test_worker.py"
    with open(worker_path, "w") as f:
        f.write(_WORKER_SCRIPT)

    result_file = "/tmp/allreduce_test_results.json"

    # Run with torchrun on 2 GPUs
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0,1"
    env["PYTHONPATH"] = "/sgl-workspace/aiter"

    print("Launching distributed test on 2 GPUs...")
    proc = subprocess.run(
        [
            _PY, "-m", "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29501",
            worker_path,
            result_file,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    print("STDOUT:", proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        print("\nDistributed test failed to run.")
        print("SCORE: 0.0")
        return

    # Parse results
    try:
        with open(result_file) as f:
            results = json.load(f)
    except Exception as e:
        print(f"Failed to read results: {e}")
        print("SCORE: 0.0")
        return

    if "error" in results:
        print(f"Test error: {results['error']}")
        print("SCORE: 0.0")
        return

    total = results.get("total", 0)
    passed = results.get("passed", 0)

    print(f"\n--- Results ---")
    for check in results.get("checks", []):
        status = "PASS" if check["pass"] else "FAIL"
        detail = ""
        if "res_divergent_pct" in check:
            detail = f" (res_div={check['res_divergent_pct']:.1f}%, out_err={check['out_max_err']:.6f})"
        elif "detail" in check:
            detail = f" ({check['detail']})"
        print(f"  [{status}] {check['name']}{detail}")

    if total == 0:
        print("\nNo checks completed.")
        print("SCORE: 0.0")
        return

    score = passed / total * 100.0
    print(f"\n{passed}/{total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
