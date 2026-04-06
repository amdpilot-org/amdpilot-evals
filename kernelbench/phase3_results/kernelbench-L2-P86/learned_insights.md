# Learned Insights

- **Trial 1**: For KernelBench problem 86 (Matmul+Div+GELU, 1024x8192 @ 8192x8192): GEMM dominates at 85%, elementwise at 15%
- **Trial 1**: Fusing a full Triton matmul for large GEMM (8192x8192) fails with pointer shape mismatch on ROCm Triton — avoid this approach
- **Trial 1**: GELU on ROCm Triton must use manual tanh via exp: tanh(x) = (exp(2x)-1)/(exp(2x)+1) with clamping to [-10,10]
- **Trial 1**: PyTorch F.linear is already highly optimized for large GEMM on ROCm; focus on epilogue fusion instead
- **Trial 1**: Score of 60.1 corresponds to ~1.01x speedup — the fused div+GELU elementwise kernel alone provides minimal benefit
- **Trial 2**: Trial 2 produced no output — likely the agent timed out or crashed without running any code. Need explicit step-by-step instructions.
- **Trial 2**: For fused matmul on ROCm Triton, use manual pointer arithmetic with offs_m[:, None] and offs_k[None, :] patterns — do NOT use tl.make_block_ptr which causes shape mismatch errors on ROCm
- **Trial 2**: nn.Linear weight shape is (output_size, input_size); to compute X @ W^T, load W with stride_wk=W.stride(1), stride_wn=W.stride(0)
- **Trial 3**: Agent crashed/timed out in trials 2 and 3 — needs paste-ready code with minimal agent decision-making
- **Trial 3**: Fusing bias+div+GELU into one Triton kernel (with torch.mm for GEMM) saves one full read+write of the 1024x8192 output tensor vs F.linear + separate elementwise kernel
- **Trial 3**: For KernelBench, ModelNew class must be defined in generated_kernel.py with same __init__ signature as Model
- **Trial 4**: Agent crashed/timed out in trials 2-4 — needs paste-ready code with zero decisions required
- **Trial 4**: For addmm trick: torch.addmm(bias, x, W_T, beta=1/div, alpha=1/div) absorbs division into GEMM epilogue for free
- **Trial 4**: Pre-transposing nn.Linear weight to contiguous may help GEMM: self._weight_t = self.linear.weight.t().contiguous()
- **Trial 5**: Agent has crashed 4 consecutive trials (2-5) — needs absolute minimal paste-ready code with no decision points
- **Trial 5**: For N=8192 (power of 2), tl.arange works directly — no padding needed
- **Trial 5**: torch.addmm with alpha=1/div, beta=1/div fuses linear+bias+division into single BLAS call
