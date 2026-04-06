# Learned Insights

- **Trial 1**: MSE loss on 1B elements: fusing sub+square+partial_reduce into one Triton kernel gives 2.69x speedup over PyTorch's 3 separate kernels (sub 2.2ms + pow 1.4ms + mean 1.1ms = 4.7ms vs 1.77ms)
- **Trial 1**: BLOCK_SIZE=65536 with 16384 blocks works but creates a large intermediate block_sums array; persistent kernel with fewer CTAs could reduce this
- **Trial 1**: Triton kernel compilation for each new BLOCK_SIZE takes significant time on ROCm — use TRITON_CACHE_DIR and limit block size exploration
- **Trial 1**: tl.atomic_add doesn't work correctly for multi-block float32 reduction on ROCm — use two-level reduction (partial sums array + host-side sum) instead
- **Trial 1**: Explicit .to(tl.float32) casts are required in ROCm Triton even when inputs are already float32
- **Trial 2**: MSE loss on 1B elements: fusing sub+square+partial_reduce into one Triton kernel gives 2.69x speedup over PyTorch's 3 separate kernels (sub 2.2ms + pow 1.4ms + mean 1.1ms = 4.7ms vs 1.77ms)
- **Trial 2**: BLOCK_SIZE=65536 with 16384 blocks works but creates a large intermediate block_sums array; persistent kernel with fewer CTAs could reduce this
- **Trial 2**: tl.atomic_add doesn't work correctly for multi-block float32 reduction on ROCm — use two-level reduction (partial sums array + host-side sum) instead
- **Trial 2**: Explicit .to(tl.float32) casts are required in ROCm Triton even when inputs are already float32
- **Trial 2**: Trial 2 produced no output — agent may need explicit instruction to verify existing kernel works before attempting changes
- **Trial 3**: Trial 2 and Trial 3 both produced no output — agent may be crashing during kernel rewrite. Give minimal edit instructions instead of full rewrites.
- **Trial 3**: Current best score 76.90 corresponds to 1.77ms total (1.69ms kernel + overhead) vs 4.77ms baseline
- **Trial 4**: Agent crashes when attempting full kernel rewrites — give minimal, incremental edit instructions
- **Trial 4**: Trials 2-4 all produced no output despite working kernel existing from trial 1
- **Trial 4**: Current best: BLOCK_SIZE=65536 with 16384 blocks, 1.69ms kernel time, 76.90 score
- **Trial 5**: Agent crashes 4 consecutive times when attempting kernel rewrites — must give copy-paste minimal single-line changes only
- **Trial 5**: Existing kernel at BLOCK_SIZE=65536 scores 76.90 (1.77ms total) — this is the safe fallback
- **Trial 5**: For MSE on 1B elements, try BLOCK_SIZE=131072 to reduce block count from 16384 to 8192
