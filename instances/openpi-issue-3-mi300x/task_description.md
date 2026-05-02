# Task: Optimize pi0 LIBERO Inference on MI300X

## Objective

Optimize pi0 LIBERO inference throughput on a single AMD MI300X GPU using the upstream
`amdpilot-org/openpi` codebase (commit `e4429ad`) and the canonical benchmark script
`benchmark_pi0_libero_rocm.py`.

**Current MI300X baseline: 7.0 inf/s vs 23.05 inf/s on H100 (3.3× gap).**

Your goal is to close or eliminate this gap through ROCm-compatible code and environment
changes, without altering any of the fixed test conditions below.

---

## Environment

| Property | Value |
|----------|-------|
| Base image | `rocm/sgl-dev:v0.5.10rc0-rocm720-mi30x-20260420` |
| GPU | Single MI300X (`cuda:0`) |
| Repo | `amdpilot-org/openpi` @ `e4429ad` at `/workspace/openpi` |
| Python | `/workspace/openpi/.venv/bin/python` (managed by `uv`) |
| Model | `pi0_libero` (Pi0 3.5B params, bf16) |
| Checkpoint | `~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch` |
| Benchmark script | `/workspace/openpi/benchmark_pi0_libero_rocm.py` |

---

## Fixed Test Conditions (do NOT change)

| Parameter | Value |
|-----------|-------|
| Model | `pi0_libero` (Pi0 3.5B params) |
| Backend | PyTorch (`--backend pytorch`) |
| Precision | bf16 |
| Batch size | 1 |
| Denoising steps | 10 |
| Image size | 224×224 |
| Cameras | 2 (`image` + `wrist_image`) |
| State dim | 8 |
| Action horizon | 10 |
| Warmup runs | 3 (`--num-warmup 3`) |
| Timed runs | 20 (`--num-runs 20`) |
| Device | `cuda:0` |
| torch.compile mode | `max-autotune` (model default) |
| GPU shape | 1×MI300X |

---

## Canonical Benchmark Command (do NOT modify)

```bash
cd /workspace/openpi && .venv/bin/python benchmark_pi0_libero_rocm.py \
    --config pi0_libero \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
    --device cuda:0 \
    --num-warmup 3 \
    --num-runs 20
```

The scored benchmark is invoked via:
```bash
bash /workspace/bench_openpi.sh
```

This script calls the exact command above and emits a `THROUGHPUT: <N> inf/s` line that
the amdpilot harness captures.

---

## Expected Baseline / Reference Setup

| Setup | Throughput | Warmup time |
|-------|-----------|-------------|
| MI300X (ROCm, `e4429ad`) | 7.0 inf/s | ~212 s |
| H100 (CUDA, reference) | 23.05 inf/s | ~12.9 s |

**First action**: run the canonical benchmark command before making any changes and confirm
your measured baseline is ≥ 5.6 inf/s (80% of 7.0). If it falls below this floor, treat
it as an environment/reproduction mismatch and resolve it before proceeding.

Known root causes reported in the issue:
- **212 s warmup**: `torch.compile` `max-autotune` mode is very slow on ROCm — tunable-op
  cache, lower compile modes, or AOT compilation strategies may help.
- **Triton shared-memory OOM**: Triton kernels request 131072 bytes but MI300X exposes
  65536 — custom kernels or backend substitutions may be needed.

---

## Optimization Scope

**Allowed** (executor may modify these):
- `torch.compile` mode and backend (e.g. switch from `max-autotune` to `reduce-overhead`,
  `inductor`, `aot_eager`, `cudagraphs`, etc.)
- Compilation cache setup (`TORCH_COMPILE_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`)
- `PYTORCH_TUNABLEOP_*` environment variables and tuning databases
- Custom Triton kernels or kernel replacements for attention / matmul
- CUDAGraph capture and replay (`torch.cuda.CUDAGraph`)
- Attention backend selection (e.g. Flash-Attention, SDPA, Triton custom)
- Operator fusion or memory layout changes
- ROCm-specific environment variables (`HIP_*`, `ROCM_*`, `HSA_*`, etc.)
- Any patch to openpi source files under `/workspace/openpi/src/`

**Out of scope** (do NOT touch):
- Batch size, warmup/iteration counts, image resolution
- Model config (`pi0_libero`) or checkpoint path
- Denoising steps, action horizon, state dim, camera count
- `benchmark_pi0_libero_rocm.py` — this file is **immutable**
- `bench_openpi.sh` — this file is **immutable**
- `src/openpi/models/model.py` — this file is **immutable**
- The single-MI300X GPU shape

---

## Optimization Targets

| Target | Throughput |
|--------|-----------|
| Baseline | 7.0 inf/s |
| Parity (goal) | 23.0 inf/s |
| Stretch goal | 50.0 inf/s |

---

## Verification / Scoring

The benchmark is scored by running:
```bash
bash /workspace/bench_openpi.sh
```

The script must emit a line matching:
```
THROUGHPUT: <number> inf/s
```

Higher throughput = higher score. The amdpilot pipeline extracts the numeric value via:
```
THROUGHPUT:\s+([\d.]+)\s+inf/s
```

---

## Rules

- **Python**: always use `/workspace/openpi/.venv/bin/python` — NOT system Python or
  `/opt/venv/bin/python3`. This is a `uv`-managed venv inside the repo checkout.
- **Do NOT run `pip install -e .` on sglang or aiter** — the base image is `rocm/sgl-dev`;
  reinstalling would clobber the ROCm PyTorch wheel.
- **Do NOT modify** `benchmark_pi0_libero_rocm.py`, `bench_openpi.sh`, or
  `src/openpi/models/model.py` — these are immutable scored artifacts.
- **Do NOT change** any fixed test condition (batch size, denoising steps, image size,
  cameras, state dim, action horizon, model config, checkpoint).
- Before each benchmark run, kill leftover GPU processes safely:
  ```bash
  pgrep -f 'python.*benchmark' | xargs -r kill -9; sleep 2
  ```
  NEVER use `pkill -f python` broadly — it may kill the agent shell.
- Verify the checkpoint is accessible at
  `~/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch` before running.
  The host volume `/home/amd/openpi_cache` is mounted to `/root/.cache/openpi`.
- Confirm `uv sync` completed successfully and `.venv/bin/python` exists before running
  the benchmark. If the venv is missing, run `cd /workspace/openpi && uv sync`.
- After the ROCm PyTorch wheel is forced in (Dockerfile step), verify with:
  ```bash
  .venv/bin/python -c "import torch; print(torch.version.hip)"
  ```
  If this returns `None` or raises, the CUDA wheel was picked up instead — fix it before
  continuing.
- Keep a log of each experiment: note what you changed, the measured throughput, and
  whether it improved or regressed.
- Confirm the `THROUGHPUT:` line appears in bench output before claiming an improvement.
