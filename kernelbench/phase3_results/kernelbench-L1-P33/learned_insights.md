# Learned Insights

- **Trial 1**: For BatchNorm on MI355X (64x64x512x512): PyTorch reference is 5.11ms, hybrid PyTorch-reduction + Triton-elementwise achieves 4.19ms (1.22x speedup)
- **Trial 1**: Pure Triton with nested loops and atomic reduction is much slower (29.5ms) on ROCm — avoid atomics for reductions
- **Trial 1**: 2D blocking patterns with BLOCK_N x BLOCK_HW cause shape compatibility errors in ROCm Triton
- **Trial 1**: Flat indexing with channel extraction via (offsets // (H*W)) % C is efficient for BatchNorm in Triton
- **Trial 1**: Hybrid approach uses 256 programs with BLOCK_SIZE=2048 — there may be room to tune these parameters
- **Trial 2**: Trial 2 produced no output - agent may have attempted a rewrite that failed to compile or run, always verify incrementally
- **Trial 2**: torch.var_mean can compute both mean and variance in a single pass, potentially faster than separate mean()+var() calls
- **Trial 2**: MI355X has 304 CUs - setting num_programs to match CU count (304 or 2*304=608) may improve occupancy
- **Trial 3**: Trials 2 and 3 both produced no output - agent gets stuck when attempting full rewrites. Must enforce incremental changes only.
- **Trial 3**: Working solution exists at /workspace/generated_kernel.py achieving score 62.20 (4.19ms). Always verify existing solution works before modifying.
- **Trial 3**: For BatchNorm optimization: try torch.var_mean(), num_programs=608 (2x304 CUs), BLOCK_SIZE=4096, and precomputing scale=weight*rsqrt(var+eps)
- **Trial 4**: Agent has failed 3 consecutive trials (2,3,4) with no output when attempting optimization of BatchNorm — always caused by full rewrites that break compilation
- **Trial 4**: Must enforce incremental single-line changes with backup/restore pattern to prevent losing working solution
- **Trial 4**: Working BatchNorm solution uses hybrid PyTorch-reduction + Triton-elementwise at score 62.20 (4.19ms vs 5.11ms reference)
- **Trial 5**: Agent has failed 4 consecutive trials (2-5) when attempting BatchNorm optimization - all due to full rewrites breaking compilation
- **Trial 5**: Must enforce verify-first, backup, single-change workflow to prevent losing working solutions
- **Trial 5**: Score 62.20 (4.19ms) is the stable baseline from hybrid PyTorch-reduction + Triton-elementwise approach
