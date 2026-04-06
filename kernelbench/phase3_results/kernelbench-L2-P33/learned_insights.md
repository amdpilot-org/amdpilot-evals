# Learned Insights

- **Trial 1**: For GEMM 8192x8192 on MI355X, rocBLAS (via nn.Linear/F.linear) is unbeatable—custom Triton GEMM is 72% slower
- **Trial 1**: PyTorch's BatchNorm1d is highly optimized on ROCm—naive Triton replacement adds ~8% overhead
- **Trial 1**: Scale folding into GEMM weights in __init__ (weight *= scale.unsqueeze(1), bias *= scale) is a valid algebraic optimization that eliminates one elementwise kernel
- **Trial 1**: Problem 33 dimensions: batch=1024, in=8192, out=8192. GEMM dominates at 85% of runtime, BN at 15%
- **Trial 1**: Score of 60.1 corresponds to ~1.008x speedup (0.953ms vs 0.961ms)
- **Trial 2**: Trial 2 produced no output — agent may have been stuck on complex kernel implementation without testing incrementally
- **Trial 2**: For Problem 33, the only remaining optimization target is the BN (15% of runtime) since GEMM (85%) uses optimal rocBLAS
- **Trial 2**: torch.compile should be tried before manual kernel surgery as per optimization playbook
- **Trial 3**: Agent crashed in trials 2 and 3 with no output — likely stuck on complex implementations. Must enforce incremental testing.
- **Trial 3**: For Problem 33 (GEMM+Scale+BN, 8192x8192), the optimization ceiling is very low since rocBLAS GEMM (85%) and PyTorch BN (15%) are already near-optimal
- **Trial 3**: torch.compile has not been tried yet — it's the next logical step per the optimization playbook
- **Trial 4**: Agent crashes (no output) 3 consecutive trials when attempting complex kernel implementations for Problem 33 — must enforce minimal incremental changes
- **Trial 4**: For Problem 33, the theoretical optimization ceiling is very low (~1-5% max) since rocBLAS GEMM dominates at 85% of runtime
- **Trial 5**: Agent crashes 4 consecutive trials attempting complex kernels for Problem 33 — must provide near-complete solution code
- **Trial 5**: torch.compile has STILL not been tried after 5 trials — this is the last viable optimization lever
