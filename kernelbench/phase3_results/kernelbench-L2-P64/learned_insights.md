# Learned Insights

- **Trial 1**: MI355X (gfx950, CDNA4) has wavefront size 64, not 32 like NVIDIA
- **Trial 1**: tl.math.tanh and tl.libdevice.* are unavailable on ROCm Triton - must use manual tanh: (exp(2x) - 1) / (exp(2x) + 1)
- **Trial 1**: torch.compile with custom Triton kernels fails in this environment due to ImportError: cannot import name 'specialize_impl' from 'triton.runtime.jit'
- **Trial 1**: torch.set_float32_matmul_precision('high') improves both reference and optimized models significantly on MI355X
- **Trial 1**: For problem 64 (1024x8192 GEMM + LogSumExp + activations), GEMM dominates at ~95% of runtime; elementwise fusion gives ~5% improvement
- **Trial 1**: Baseline score 61.90 corresponds to 0.311ms optimized vs 0.366ms reference (1.177x speedup)
- **Trial 2**: Trial 2 produced no output — agent may have gotten stuck without running the benchmark; always ensure the benchmark command is executed
- **Trial 2**: For GEMM-dominated workloads (95% of runtime), precision reduction (fp16/bf16) is the highest-impact optimization since it roughly doubles FLOPS throughput on MI355X
- **Trial 3**: Agent has failed to produce output in 2 consecutive optimization trials - needs extremely specific step-by-step instructions
- **Trial 3**: For GEMM-dominated workloads, casting to fp16/bf16 before torch.nn.functional.linear is the simplest high-impact optimization
- **Trial 4**: Agent has failed to produce output in 3 consecutive optimization trials (trials 2-4) - needs extremely prescriptive, copy-paste-ready instructions
- **Trial 4**: For GEMM-dominated workloads on MI355X, fp16 casting (x.half()) before torch.nn.functional.linear is the simplest high-impact optimization
- **Trial 4**: Always ensure the benchmark command /opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 64 is executed regardless of optimization success
- **Trial 5**: Agent has failed to produce output in 4 consecutive optimization trials (trials 2-5) - the core problem is the agent getting stuck before benchmark execution
- **Trial 5**: For agents that repeatedly fail to produce output, the most important instruction is to run the benchmark regardless of whether optimization succeeds
- **Trial 5**: When providing copy-paste instructions, keep them to absolute minimum steps (3 or fewer) to prevent the agent from getting lost
