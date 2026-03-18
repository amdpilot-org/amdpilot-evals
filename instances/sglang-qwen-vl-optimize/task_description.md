# Qwen3-VL Serving Throughput Optimization

Optimize SGLang serving throughput for Qwen3-VL-8B-Instruct on AMD MI355X to eliminate a regression vs vLLM.

## Problem — SGLang Qwen-VL Regression

SGLang v0.5.9 (ROCm 7.2.0, MI355X) has a **33% throughput regression** compared to vLLM when serving the Qwen3-VL-8B-Instruct vision-language model. SGLang should match or exceed vLLM on AMD GPUs — the vLLM numbers below represent the **pre-regression target** that SGLang must recover:

| Backend | Output Throughput (tok/s) | TPOT (ms) | E2E Latency p50 (ms) | TTFT p50 (ms) |
|---------|--------------------------|-----------|----------------------|---------------|
| SGLang (native API) | 1235.85 | 12.21 | 25786 | 1134 |
| SGLang (OAI chat) | 1115.81 | 13.29 | 29223 | 1267 |
| **vLLM (target)** | **1648.09** | **9.09** | **19385** | **1275** |

Key observations:
- **Decode is the bottleneck**: TPOT 12.21ms (SGLang) vs 9.09ms (vLLM) — a 34% gap. Prefill (TTFT ~1100-1275ms) is comparable across all backends.
- **SGLang native API > OAI chat**: 1235 vs 1115 tok/s. Focus optimization on the native decode path.
- **Vision tokens are ~17% of input**: 112,896 vision tokens out of 652,598 total input tokens. Vision processing overhead may compound during batched decode.

## Deliverable

**The final deliverable is a clean git commit on a new branch in the sglang fork at `/workspace/sglang/`.** After you achieve the target throughput, you MUST create the branch and commit your changes (see "Creating the Fix Branch" section below). The committed Docker image will be used to extract this branch.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang fork checkout**: `/workspace/sglang-fork/` — clone of `github.com/Arist12/sglang` for creating the fix branch after optimization.
- **Model weights**: `Qwen/Qwen3-VL-8B-Instruct` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_qwen_vl.sh` — starts server, runs warmup + benchmark, reports output throughput.

## Benchmark

The benchmark runs a full serving workload (self-contained):
1. Starts SGLang server with the model (attention backend configurable)
2. Waits for server health
3. Runs 128 image-prompt requests as warmup (results discarded)
4. Runs 128 image-prompt requests as the actual measurement
5. Reports: `Output throughput (tok/s): <value> | concurrency=16 model=Qwen3-VL-8B`

First run takes 10–15 minutes (model loading + warmup + benchmark). Set `timeout: 1200` when running it.

### Configuring the benchmark

The server launch can be configured via `/workspace/bench_config.env`. Write environment variables to this file so that the verification run uses the same configuration:

```bash
# Switch attention backend
echo 'export ATTENTION_BACKEND=aiter' > /workspace/bench_config.env

# Add extra server arguments
echo 'export EXTRA_SERVER_ARGS="--chunked-prefill-size 4096"' >> /workspace/bench_config.env
```

### Quick iteration

For faster feedback during development, you can start the server manually and run `bench_serving` directly with fewer prompts:

```bash
# Start server (background)
SGLANG_DISABLE_CUDNN_CHECK=1 /opt/venv/bin/python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 --port 30000 --trust-remote-code \
    --attention-backend triton &

# Quick test with 32 prompts
/opt/venv/bin/python3 -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen3-VL-8B-Instruct --dataset-name image \
    --num-prompts 32 --random-input-len 4000 --random-output-len 2000 \
    --random-range-ratio 1.0 --image-count 1 --image-resolution 720p \
    --image-content random --max-concurrency 16 --seed 123 --warmup-requests 0
```

Use the full benchmark script (`bash /workspace/bench_qwen_vl.sh`) for the final measurement.

## Target

Match or exceed the pre-regression (vLLM) output throughput: **≥1600 tok/s** (vLLM achieves 1648 tok/s on the identical workload). Current SGLang baseline is ~1235 tok/s — this is a 33% gap to close.

## Repro Commands (Reference)

These commands were used to reproduce the regression on `lmsysorg/sglang:v0.5.9-rocm720-mi35x`:

**SGLang server:**
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 sglang serve \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 --port 30000 --trust-remote-code \
    --attention-backend triton
```

**Benchmark:**
```bash
python3 -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen3-VL-8B-Instruct --dataset-name image \
    --num-prompts 128 --random-input-len 4000 --random-output-len 2000 \
    --random-range-ratio 1.0 --image-count 1 --image-resolution 720p \
    --image-content random --max-concurrency 16 --seed 123 --warmup-requests 0
```

**vLLM reference (1648 tok/s):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --trust-remote-code
```

## Suggested Investigation Areas

1. **Attention backend**: The repro uses `--attention-backend triton`. Try `aiter` (AMD's native high-performance attention backend) which provides significantly better decode throughput on MI355X. Check `sglang/srt/layers/attention/` for backend dispatch logic.

2. **Vision encoder pipeline**: Profile the vision encoder and cross-attention layers. Look for unnecessary data copies, synchronization points, or suboptimal kernel dispatch in the Qwen3-VL model implementation under `sglang/srt/models/`.

3. **Multimodal scheduler efficiency**: Image requests generate vision tokens that may cause scheduling inefficiency during batched decode. Check `sglang/srt/managers/schedule_batch.py` and related files for multimodal token handling.

4. **Decode kernel performance**: Since the regression is entirely in decode (TPOT), examine the decode attention kernel path. Check if there are VL-specific code paths that are less optimized than text-only paths.

5. **Memory management**: Multimodal tokens may trigger extra memory operations (allocation, defragmentation) that slow the decode loop. Check `sglang/srt/mem_cache/`.

6. **CUDA graph compatibility**: Ensure CUDA graphs are enabled and working correctly for VL models. VL models sometimes disable CUDA graphs due to variable input shapes.

## Creating the Fix Branch (REQUIRED)

After achieving the target throughput, you **MUST** create a clean git commit on a new branch. This is the primary deliverable.

The runtime SGLang at `/sgl-workspace/sglang/` may not be a git repo. Use `diff` to identify changes, then apply them to the fork checkout:

```bash
# 1. Identify what you changed in the runtime (adjust paths as needed)
diff -ruN /workspace/sglang-fork/python/sglang/ /sgl-workspace/sglang/python/sglang/ > /workspace/changes.patch

# 2. Create the fix branch in the fork
cd /workspace/sglang-fork
git checkout -b fix/qwen-vl-throughput

# 3. Apply the patch
git apply /workspace/changes.patch || patch -p0 < /workspace/changes.patch

# 4. Commit with a descriptive message
git add -A
git commit -m "fix: optimize Qwen3-VL serving throughput on MI355X

Closes the ~33% throughput gap vs vLLM on bench_serving image workload.
Baseline: ~1235 tok/s -> Target: >=1600 tok/s (vLLM: 1648 tok/s)

Changes:
- <describe what you changed and why>
"
```

If the diff approach fails (paths differ), manually copy each changed file:
```bash
cd /workspace/sglang-fork
git checkout -b fix/qwen-vl-throughput
cp /sgl-workspace/sglang/python/sglang/<changed_file> python/sglang/<changed_file>
git add -A
git commit -m "fix: optimize Qwen3-VL serving throughput on MI355X"
```

The branch will be preserved in the committed Docker image.

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Do NOT modify the benchmark script or its parameters (128 prompts, concurrency 16, etc.).
- **Run `bash /workspace/bench_qwen_vl.sh` as your LAST command before creating the git branch.**
- **Create the fix branch as the very last step** (after confirming the benchmark passes).
- Kill leftover sglang server processes before starting a new one:
  `ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`
