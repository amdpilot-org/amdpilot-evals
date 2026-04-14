#!/usr/bin/env python3
"""Test harness for aiter-sampling-optimize. Runtime correctness + performance.

Task: Optimize the TopK/TopP sampling kernel to reduce latency below 0.5ms
      for batch_size=1, vocab_size=128256.

Correctness: Sampled tokens must respect top-k/top-p constraints — high-prob
     tokens should dominate, and results must be valid indices. A stub that
     returns random indices would fail the distribution check.

Performance: Average latency must be below 0.5ms (matching task description).
"""
import sys
import subprocess

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


def run_test(script, timeout=120):
    result = subprocess.run(
        ["/opt/venv/bin/python3", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd="/sgl-workspace/aiter",
    )
    return result.stdout or "", result.stderr or "", result.returncode


print("=" * 60)
print("aiter-sampling-optimize test harness")
print("=" * 60)

# -----------------------------------------------------------------------
# Check 1: Import sampling ops (subprocess-isolated)
# -----------------------------------------------------------------------
stdout, stderr, rc = run_test("""
import sys, torch
if not torch.cuda.is_available():
    print("NO_GPU")
    sys.exit(1)
try:
    from aiter.ops.sampling import top_k_top_p_sampling_from_probs
    print("IMPORT:OK")
except ImportError as e:
    print(f"IMPORT:FAIL:{e}")
""")

if "NO_GPU" in stdout:
    check("GPU available", False, "No GPU detected")
    check("Import sampling ops", False, "No GPU")
    check("Valid sample indices", False, "No GPU")
    check("Top-k distribution correctness", False, "No GPU")
    check("Latency < 0.5ms", False, "No GPU")
elif "IMPORT:FAIL" in stdout:
    err = stdout.split("IMPORT:FAIL:")[1].strip()[:200]
    check("Import sampling ops", False, f"import error: {err}")
    check("Valid sample indices", False, "import failed")
    check("Top-k distribution correctness", False, "import failed")
    check("Latency < 0.5ms", False, "import failed")
else:
    check("Import sampling ops", "IMPORT:OK" in stdout,
          f"unexpected: {stdout.strip()[:200]}")

    # -------------------------------------------------------------------
    # Checks 2-3: Correctness (subprocess-isolated)
    # -------------------------------------------------------------------
    stdout2, stderr2, rc2 = run_test("""
import sys, torch
torch.manual_seed(42)

from aiter.ops.sampling import top_k_top_p_sampling_from_probs

device = torch.device("cuda:0")

# Test 1: Basic validity — indices in [0, vocab)
batch, vocab = 64, 32000
probs = torch.softmax(torch.randn(batch, vocab, device=device), dim=-1)
result = top_k_top_p_sampling_from_probs(probs, None, None, 5, None, 0.9, True)
if isinstance(result, tuple):
    result = result[0]
valid = torch.all(result >= 0).item() and torch.all(result < vocab).item()
print(f"VALID_INDICES:{valid}")

# Test 2: Distribution correctness — with a skewed distribution,
# top-k=5 sampling should strongly favor the top 5 tokens.
# A torch.randint stub would sample uniformly across vocab and fail this.
batch2 = 2048
vocab2 = 1000
# Create a distribution where tokens 0..4 have 95% of the mass
probs2 = torch.ones(batch2, vocab2, device=device) * 0.001
probs2[:, :5] = torch.tensor([0.4, 0.3, 0.15, 0.08, 0.02], device=device)
probs2 = probs2 / probs2.sum(dim=-1, keepdim=True)

samples = []
for _ in range(10):
    r = top_k_top_p_sampling_from_probs(probs2, None, None, 5, None, 1.0, True)
    if isinstance(r, tuple):
        r = r[0]
    samples.append(r.cpu())
all_samples = torch.cat(samples)

# With top_k=5 and 95% mass in top 5, >90% of samples should be in [0,5)
in_top_k = (all_samples < 5).float().mean().item()
print(f"IN_TOP_K:{in_top_k:.4f}")

# Token 0 (prob ~0.4) should be sampled more often than token 4 (prob ~0.02)
# Verify monotonic-ish ordering of frequencies
counts = torch.zeros(5)
for i in range(5):
    counts[i] = (all_samples == i).float().sum().item()
total = counts.sum().item()
if total > 0:
    freq_0 = counts[0].item() / total
    freq_4 = counts[4].item() / total
    print(f"FREQ_RATIO:{freq_0 / max(freq_4, 1e-6):.2f}")
else:
    print("FREQ_RATIO:0.0")
""")

    if rc2 != 0 and not stdout2.strip():
        check("Valid sample indices", False,
              f"subprocess crashed: {stderr2.strip()[:200]}")
        check("Top-k distribution correctness", False, "subprocess crashed")
    else:
        valid = "VALID_INDICES:True" in stdout2
        check("Valid sample indices", valid,
              f"got invalid indices: {stdout2.strip()[:200]}")

        in_top_k = None
        freq_ratio = None
        for line in stdout2.strip().split("\n"):
            if line.startswith("IN_TOP_K:"):
                try: in_top_k = float(line.split(":")[1])
                except: pass
            elif line.startswith("FREQ_RATIO:"):
                try: freq_ratio = float(line.split(":")[1])
                except: pass

        # >90% in top-k AND token 0 sampled at least 3x more than token 4
        dist_ok = (in_top_k is not None and in_top_k > 0.90 and
                   freq_ratio is not None and freq_ratio > 3.0)
        check("Top-k distribution correctness", dist_ok,
              f"in_top_k={in_top_k} (expect > 0.90), "
              f"freq_ratio={freq_ratio} (expect > 3.0)")

    # -------------------------------------------------------------------
    # Check 4: Performance — latency < 0.5ms (subprocess-isolated)
    # -------------------------------------------------------------------
    stdout3, stderr3, rc3 = run_test("""
import sys, time, torch

from aiter.ops.sampling import top_k_top_p_sampling_from_probs

device = torch.device("cuda:0")
probs = torch.softmax(torch.randn(1, 128256, device=device), dim=-1)

# Warmup
for _ in range(30):
    top_k_top_p_sampling_from_probs(probs, None, None, 1, None, 0.9, True)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(200):
    top_k_top_p_sampling_from_probs(probs, None, None, 1, None, 0.9, True)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) * 1000 / 200
print(f"LATENCY_MS:{ms:.4f}")
""")

    latency_ms = None
    for line in stdout3.strip().split("\n"):
        if line.startswith("LATENCY_MS:"):
            try: latency_ms = float(line.split(":")[1])
            except: pass

    if latency_ms is not None:
        check(f"Latency < 0.5ms (got {latency_ms:.3f}ms)",
              latency_ms < 0.5,
              f"latency {latency_ms:.3f}ms exceeds 0.5ms target")
    else:
        err = (stdout3 + stderr3).strip()[:200]
        check("Latency < 0.5ms", False, f"benchmark failed: {err}")


print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total}")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
