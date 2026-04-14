#!/usr/bin/env python3
"""Test harness for aiter-topp-sampling-accuracy.

Bug: Top-p sampling kernels in AIter produce incorrect results under tensor
     parallelism (TP >= 2). Different TP ranks diverge in their sampling
     decisions due to inconsistent random state, causing repetitive/corrupted
     text generation and accuracy regressions on benchmarks like GSM8K.

Expected behavior after fix: Sampling produces correct nucleus-concentrated
     distributions on single GPU, and multi-GPU TP ranks produce consistent
     coordinated sampling decisions.
"""
import sys
import subprocess
import os

checks_passed = 0
checks_total = 0


def check(name, condition, detail=""):
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


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/sgl-workspace/aiter",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("aiter-topp-sampling-accuracy test harness")
print("=" * 60)

# -----------------------------------------------------------------------
# Check 1: Import top_p_sampling_from_probs
# -----------------------------------------------------------------------
stdout, stderr, rc = run_test("""
import sys
import torch

if not torch.cuda.is_available():
    print("NO_GPU")
    sys.exit(0)

try:
    from aiter import top_p_sampling_from_probs
    print("IMPORT:OK")
except ImportError:
    try:
        from aiter.ops.triton import top_p_sampling_from_probs
        print("IMPORT:OK")
    except ImportError as e:
        print(f"IMPORT:FAIL:{e}")
""")

if "NO_GPU" in stdout:
    check("GPU available", False, "No GPU detected")
    check("Import top_p_sampling_from_probs", False, "No GPU")
    check("Nucleus concentration (single GPU)", False, "No GPU")
    check("Sampling diversity", False, "No GPU")
    check("RNG variation across trials", False, "No GPU")
    check("Distribution fidelity (frequency ratio)", False, "No GPU")
    check("TP=2 rank consistency", False, "No GPU")
elif "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].strip()[:200]
    check("Import top_p_sampling_from_probs", False, f"import error: {err}")
    check("Nucleus concentration (single GPU)", False, "import failed")
    check("Sampling diversity", False, "import failed")
    check("RNG variation across trials", False, "import failed")
    check("Distribution fidelity (frequency ratio)", False, "import failed")
    check("TP=2 rank consistency", False, "import failed")
else:
    check("Import top_p_sampling_from_probs",
          "IMPORT:OK" in stdout,
          f"unexpected: {stdout.strip()[:200]}")

    # -------------------------------------------------------------------
    # Checks 2-4: Single-GPU behavioral correctness (subprocess-isolated)
    # -------------------------------------------------------------------
    stdout2, stderr2, rc2 = run_test("""
import sys, torch

try:
    from aiter import top_p_sampling_from_probs
except ImportError:
    from aiter.ops.triton import top_p_sampling_from_probs

device = torch.device("cuda:0")
batch_size = 1024
vocab_size = 128
top_p = 0.9
num_trials = 10

# Known distribution: tokens 0..4 get most mass
probs = torch.zeros(batch_size, vocab_size, device=device)
high_prob_tokens = 5
for i in range(high_prob_tokens):
    probs[:, i] = (high_prob_tokens - i) / (
        high_prob_tokens * (high_prob_tokens + 1) / 2
    )
probs[:, high_prob_tokens:] = 0.001
probs = probs / probs.sum(dim=-1, keepdim=True)

all_samples = []
for trial in range(num_trials):
    result = top_p_sampling_from_probs(probs, top_p)
    if isinstance(result, tuple):
        samples = result[0]
    else:
        samples = result
    all_samples.append(samples.cpu())

all_samples = torch.cat(all_samples, dim=0)

# Check: vast majority of samples in [0, high_prob_tokens)
in_nucleus = (all_samples < high_prob_tokens).float().mean().item()
print(f"IN_NUCLEUS:{in_nucleus:.4f}")

# Check: more than one unique token (not all identical)
unique_tokens = all_samples.unique().numel()
print(f"UNIQUE_TOKENS:{unique_tokens}")

# Check: across trials, results should differ (RNG advancing)
first_trial = all_samples[:batch_size]
second_trial = all_samples[batch_size:2*batch_size]
identical_frac = (first_trial == second_trial).float().mean().item()
print(f"IDENTICAL_FRAC:{identical_frac:.4f}")

# Check: frequency ratio — token 0 (~33% mass) should be sampled much
# more than token 4 (~7% mass). A torch.randint(0,5,...) stub gives ~20%
# each → ratio ~1.0 → FAIL. Real top-p sampling respects the distribution.
freq_0 = (all_samples == 0).float().mean().item()
freq_4 = (all_samples == 4).float().mean().item()
print(f"FREQ_RATIO:{freq_0 / max(freq_4, 1e-6):.2f}")
""")

    if rc2 != 0 and not stdout2.strip():
        check("Nucleus concentration (single GPU)", False,
              f"subprocess crashed: {stderr2.strip()[:200]}")
        check("Sampling diversity", False, "subprocess crashed")
        check("RNG variation across trials", False, "subprocess crashed")
        check("Distribution fidelity (frequency ratio)", False,
              "subprocess crashed")
    else:
        # Parse results
        in_nucleus = None
        unique_tokens = None
        identical_frac = None
        freq_ratio = None
        for line in stdout2.strip().split("\n"):
            if line.startswith("IN_NUCLEUS:"):
                try: in_nucleus = float(line.split(":")[1])
                except: pass
            elif line.startswith("UNIQUE_TOKENS:"):
                try: unique_tokens = int(line.split(":")[1])
                except: pass
            elif line.startswith("IDENTICAL_FRAC:"):
                try: identical_frac = float(line.split(":")[1])
                except: pass
            elif line.startswith("FREQ_RATIO:"):
                try: freq_ratio = float(line.split(":")[1])
                except: pass

        check("Nucleus concentration (single GPU)",
              in_nucleus is not None and in_nucleus > 0.85,
              f"in_nucleus={in_nucleus} (expect > 0.85)")

        check("Sampling diversity",
              unique_tokens is not None and unique_tokens >= 3,
              f"unique_tokens={unique_tokens} (expect >= 3)")

        check("RNG variation across trials",
              identical_frac is not None and identical_frac < 0.95,
              f"identical_frac={identical_frac} (expect < 0.95)")

        check("Distribution fidelity (frequency ratio)",
              freq_ratio is not None and freq_ratio > 2.0,
              f"freq_ratio={freq_ratio} (expect > 2.0 — token 0 should "
              f"be sampled much more than token 4)")

    # -------------------------------------------------------------------
    # Check 5: Multi-GPU TP=2 rank consistency (subprocess-isolated)
    #
    # The core bug: under TP >= 2, different ranks produce divergent
    # sampling decisions. We launch 2 processes via torchrun, each
    # sampling from the same probability distribution with the same seed.
    # Post-fix, both ranks should produce identical samples. Pre-fix,
    # the ranks diverge due to inconsistent RNG state.
    # -------------------------------------------------------------------
    import tempfile, os, json

    tp_script = '''
import sys, os, json, torch, torch.distributed as dist

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")

try:
    from aiter import top_p_sampling_from_probs
except ImportError:
    from aiter.ops.triton import top_p_sampling_from_probs

device = torch.device(f"cuda:{local_rank}")

# All ranks use the same seed and same probability distribution
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

batch_size = 512
vocab_size = 128
top_p = 0.9

probs = torch.zeros(batch_size, vocab_size, device=device)
high_prob_tokens = 5
for i in range(high_prob_tokens):
    probs[:, i] = (high_prob_tokens - i) / (
        high_prob_tokens * (high_prob_tokens + 1) / 2
    )
probs[:, high_prob_tokens:] = 0.001
probs = probs / probs.sum(dim=-1, keepdim=True)

# Sample multiple rounds
all_samples = []
for _ in range(5):
    result = top_p_sampling_from_probs(probs, top_p)
    if isinstance(result, tuple):
        samples = result[0]
    else:
        samples = result
    all_samples.append(samples)

all_samples = torch.cat(all_samples, dim=0)  # (5*batch,)

# Gather all samples to rank 0 for comparison
gathered = [torch.zeros_like(all_samples) for _ in range(world_size)]
dist.all_gather(gathered, all_samples)

if rank == 0:
    # Compare rank 0 vs rank 1 samples
    r0 = gathered[0].cpu()
    r1 = gathered[1].cpu()
    match_frac = (r0 == r1).float().mean().item()
    # Also check nucleus concentration on rank 0
    in_nucleus = (r0 < high_prob_tokens).float().mean().item()
    print(f"TP_MATCH_FRAC:{match_frac:.4f}")
    print(f"TP_IN_NUCLEUS:{in_nucleus:.4f}")

dist.destroy_process_group()
'''

    # Check if we have 2+ GPUs available
    gpu_count_out, _, _ = run_test("import torch; print(f'GPU_COUNT:{torch.cuda.device_count()}')")
    gpu_count = 0
    for line in gpu_count_out.strip().split("\n"):
        if line.startswith("GPU_COUNT:"):
            try: gpu_count = int(line.split(":")[1])
            except: pass

    if gpu_count < 2:
        check("TP=2 rank consistency",
              False,
              f"Only {gpu_count} GPU(s) available, need 2 for TP test")
    else:
        # Write the TP script to a temp file and run via torchrun
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                          dir='/tmp') as f:
            f.write(tp_script)
            tp_script_path = f.name

        try:
            tp_result = subprocess.run(
                ["/opt/venv/bin/torchrun",
                 "--nproc_per_node=2",
                 "--master_port=29501",
                 tp_script_path],
                capture_output=True, text=True, timeout=180,
                cwd="/sgl-workspace/aiter",
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
            )
            tp_stdout = tp_result.stdout or ""
            tp_stderr = tp_result.stderr or ""

            match_frac = None
            for line in tp_stdout.strip().split("\n"):
                if line.startswith("TP_MATCH_FRAC:"):
                    try: match_frac = float(line.split(":")[1])
                    except: pass

            if match_frac is not None:
                # Post-fix: ranks should produce identical samples (match ~1.0)
                # Pre-fix: ranks diverge significantly (match << 1.0)
                check("TP=2 rank consistency",
                      match_frac > 0.95,
                      f"rank match fraction={match_frac:.4f} (expect > 0.95 — "
                      f"ranks are diverging, indicates inconsistent RNG state)")
            else:
                check("TP=2 rank consistency",
                      False,
                      f"torchrun failed or no output: {(tp_stdout + tp_stderr).strip()[:300]}")
        except subprocess.TimeoutExpired:
            check("TP=2 rank consistency", False, "torchrun timed out after 180s")
        finally:
            try: os.unlink(tp_script_path)
            except: pass


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
