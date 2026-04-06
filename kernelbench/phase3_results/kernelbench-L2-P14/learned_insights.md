# Learned Insights

- **Trial 1**: Problem 14 (Gemm_Divide_Sum_Scaling): GEMM is 95% of runtime. Mathematical rearrangement sum_j(x @ W^T) = x @ W.sum(dim=0) reduces (1024,8192)×(8192,8192) GEMM to (1024,8192)×(8192,1) matvec — 8000x fewer FLOPs
- **Trial 1**: Agent incorrectly assumed 1e-4 absolute tolerance and abandoned the mathematical rearrangement. Must read test_harness.py to find actual torch.allclose parameters before giving up on an optimization
- **Trial 1**: Float32 accumulation order differences cause ~0.02 absolute error in rearranged computation, but this may be within rtol for large output values (~2000-5000 magnitude)
- **Trial 1**: Triton kernels with different accumulation order than PyTorch BLAS will not match bit-for-bit in fp32 — tolerance must be checked before pursuing custom GEMM kernels
- **Trial 2**: Trial 2 produced no output - agent may have timed out or crashed without producing any code changes
- **Trial 2**: The mathematical rearrangement sum_j(x @ W^T) = x @ W.sum(dim=0) is the key optimization for this problem - converting O(M*N*K) GEMM to O(M*K) matvec
- **Trial 2**: For supervisor_tightens stages, if the first trial fails with no metric, must retry before tightening
- **Trial 3**: Two consecutive trials with no output suggest agent timeout or crash — provide copy-paste-ready code in hints to minimize agent work
- **Trial 3**: Mathematical rearrangement: sum_j(x @ W^T / 2) * s = (s/2) * x @ W.sum(dim=0) converts GEMM to matvec, reducing FLOPs by ~8000x
- **Trial 3**: Weight sum precomputation cache must handle device placement and potential weight updates during benchmarking
- **Trial 4**: Three consecutive no-output trials suggest the agent is running out of time reading files or compiling — must provide complete copy-paste code
- **Trial 4**: Mathematical rearrangement sum_j(x @ W^T) = x @ W.sum(dim=0) converts (1024,8192)x(8192,8192) GEMM to (1024,8192)x(8192,1) matvec
- **Trial 4**: Cache W.sum(dim=0) to avoid recomputation on every forward pass — check device match for cache invalidation
- **Trial 4**: If Triton matvec fails tolerance, fall back to PyTorch torch.matmul for the matvec — still massive speedup from reduced problem size
- **Trial 5**: Agent has failed 4 consecutive trials with no output on this problem — likely timing out during file reading/analysis phase
- **Trial 5**: Must provide complete copy-paste code and forbid any analysis/reading steps to prevent timeout
- **Trial 5**: Mathematical rearrangement converts O(M*N*K) GEMM to O(M*K) matvec — key optimization for sum-after-GEMM patterns
