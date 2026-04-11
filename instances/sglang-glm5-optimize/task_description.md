# GLM-5 Kernel Optimization on AMD MI355X

Maximize the decode throughput for **zai-org/GLM-5** (744B MoE, 40B active) on AMD MI355X using SGLang. There is NO fixed target — always push for higher throughput.

## Proxy Model Approach

The full GLM-5 model is 744B parameters and requires TP=4+ to serve. To iterate quickly, **you are running a 2-layer proxy** with dummy weights (`--load-format dummy --json-model-override-args '{"num_hidden_layers": 2}'`). This proxy:

- Uses the **exact same kernels** as the full model (attention, MoE dispatch, GEMMs, RMSNorm)
- Runs on a **single GPU** (TP=1) in ~30-60 seconds per benchmark
- Any kernel-level improvement on the proxy transfers directly to the full model

## Architecture Details

GLM-5 uses the `glm_moe_dsa` architecture:
- **Attention**: DeepSeek Sparse Attention (DSA) — similar to MLA (Multi-head Latent Attention)
- **MoE**: Mixture of Experts with gated routing (744B total, 40B active per token)
- **Precision**: BF16 weights (FP8 for quantized deployment)
- **Key dims**: hidden_size likely ~6144-8192, num_attention_heads ~64, num_experts ~256

## Benchmark

The benchmark script `bench_glm5_proxy.sh`:
1. Starts SGLang with `--load-format dummy --json-model-override-args '{"num_hidden_layers": 2}' --tp 1`
2. Runs 32 random text requests at concurrency=16
3. Reports: `Output throughput (tok/s): <value> | tp=1 layers=2 proxy model=GLM-5`

Run: `bash /workspace/bench_glm5_proxy.sh`

Each run takes ~30-60 seconds (vs 10+ minutes for full model). Use this speed advantage to iterate rapidly.

## Target

**There is no fixed target. Always maximize throughput.** Higher is better. After each optimization, re-run the benchmark and profile the next bottleneck.

## Optimization Areas

Focus on what affects kernel-level performance (these transfer to the full model):

1. **MoE dispatch kernels**: Expert routing, token-to-expert assignment, fused MoE GEMM. Check `aiter` MoE ops and tuning configs. MI355X may need different GEMM tile sizes.
2. **DSA / MLA attention**: The attention kernel may use a different path than standard MHA/GQA. Check how SGLang dispatches attention for this architecture. Triton kernel tuning (BLOCK, waves_per_eu, num_warps, num_stages) applies here.
3. **torch.compile / Inductor**: Check if `torch.compile` reduces overhead. Try `--enable-torch-compile`. Watch for hangs with `TORCHINDUCTOR_MAX_AUTOTUNE`.
4. **CUDA graph capture**: Graph replay overhead, padding, alignment. Check `--disable-cuda-graph` vs default.
5. **Fused operators**: RMSNorm+attention, SwiGLU, fused MoE topk+gate. Check if custom fused kernels exist for MI355X.
6. **Memory bandwidth**: MI355X has 8TB/s HBM. MoE models are bandwidth-hungry due to expert weight loading. Check if expert caching or prefetching helps.
7. **FP8 quantization**: Try `--quantization fp8` to halve memory bandwidth for weights. This can significantly improve MoE throughput.

## Rules

- Do NOT modify `bench_glm5_proxy.sh` or its parameters.
- Kill leftover sglang server processes before starting a new one:
  `ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`
- The installed SGLang runtime is at `/sgl-workspace/sglang/` — edit files HERE.
- Commit improvements to `/workspace/sglang-fork/` periodically.
- The proxy model uses dummy weights — do NOT try to evaluate output quality. Focus purely on throughput.
- Extra server args can be set via `/workspace/bench_config.env` (sourced by the benchmark):
  ```
  EXTRA_SERVER_ARGS="--enable-torch-compile"
  PROXY_NUM_LAYERS=4
  ```
