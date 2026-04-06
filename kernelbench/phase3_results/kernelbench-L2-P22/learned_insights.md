# Learned Insights

- **Trial 1**: KernelBench L2P22: GEMM (1024x8192 @ 8192x8192) dominates at 97.9% of GPU time; elementwise ops are only 2.1%
- **Trial 1**: KernelBench L2P22: The reference x*scale_factor then x+x equals x*(scale_factor*2), not x*(scale_factor+1)
- **Trial 1**: ROCm Triton: tl.math.tanh is unavailable; must implement tanh manually as (exp(2x)-1)/(exp(2x)+1)
- **Trial 1**: KernelBench L2P22: Fusing 15+ separate elementwise kernels into 2 Triton kernels saves ~9ms (11ms->2ms)
- **Trial 1**: KernelBench L2P22: Further gains require GEMM optimization — epilogue fusion, precision reduction, or custom Triton GEMM
- **Trial 2**: KernelBench L2P22: Agent may fail silently if it doesn't actually execute the benchmark — always verify benchmark runs first
- **Trial 2**: KernelBench L2P22: GEMM epilogue fusion (fusing scale+add+clamp into GEMM) is the key remaining optimization since GEMM is 97.9% of time
- **Trial 3**: KernelBench L2P22: Two consecutive trials produced no output — agent needs explicit execution commands and step-by-step workflow
- **Trial 3**: KernelBench L2P22: GEMM epilogue fusion (fusing scale*2+clamp into matmul epilogue) avoids a memory round-trip and is the main remaining optimization opportunity
- **Trial 4**: KernelBench L2P22: Agent has failed 3 consecutive trials with no output — needs extremely explicit step-by-step commands with copy-paste code
- **Trial 4**: KernelBench L2P22: A Triton GEMM for 1024x8192x8192 with fused epilogue (scale*2+clamp) could eliminate the memory round-trip but may be slower than hipBLAS; fallback is fusing scale+clamp into the logsumexp reduction kernel
- **Trial 5**: KernelBench L2P22: Agent has failed 4 consecutive trials with no output — the blocker is execution, not algorithmic
- **Trial 5**: KernelBench L2P22: With 42min remaining, must prioritize running the benchmark over planning optimizations
