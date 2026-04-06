# Learned Insights

- **Trial 1**: LayerNorm problem 40: input shape (16, 64, 256, 256), normalized_shape=(64,256,256), so M=16 rows with N=4,194,304 elements each — very large N, low M parallelism
- **Trial 1**: 2-pass Triton LayerNorm with BLOCK_SIZE=8192 achieves 2.82x speedup (1.46ms vs 4.12ms baseline) as a starting point
- **Trial 1**: With only M=16 rows, kernel parallelism is severely limited — consider splitting rows across multiple blocks with parallel reduction
- **Trial 1**: Score metric is higher-is-better, computed from speedup ratio
- **Trial 2**: Trial 2 produced no output — agent may need explicit instructions to run benchmark first before optimizing
- **Trial 2**: With M=16 rows and N=4,194,304 elements, splitting rows across multiple blocks (e.g., 128 per row) could increase occupancy from 16 to 2048 blocks on MI355X
- **Trial 2**: Current 2-pass Triton kernel with BLOCK_SIZE=8192 achieves score ~78.70 (2.82x speedup over PyTorch nn.LayerNorm)
- **Trial 3**: Agent crashed/hung in trials 2 and 3 — likely due to infinite loop, OOM from too-large BLOCK_SIZE, or compilation error in generated Triton kernel
- **Trial 3**: For M=16 rows with N=4,194,304 elements, multi-block-per-row decomposition is needed for occupancy but requires inter-block synchronization (atomic ops or multi-kernel approach)
- **Trial 3**: When agent produces no output, provide step-by-step instructions starting with 'run the benchmark first' to ensure at least partial progress
- **Trial 4**: Agent crashes repeatedly (trials 2-4) when attempting complex multi-block-per-row LayerNorm decomposition on M=16 rows — avoid this approach entirely
- **Trial 4**: For LayerNorm with M=16 and N=4,194,304: multi-block-per-row requires inter-block sync (atomics or multi-kernel) which is error-prone and causes agent hangs
- **Trial 4**: When agent produces no output 3+ times, provide step-by-step instructions starting with 'run benchmark AS-IS first' and emphasize incremental changes only
- **Trial 4**: Safe optimization levers for existing 2-pass Triton LayerNorm: num_warps tuning, BLOCK_SIZE tuning, single-pass Welford algorithm
- **Trial 5**: Agent crashes 4 consecutive times when attempting multi-block-per-row LayerNorm decomposition — this approach is infeasible within the agent's capabilities for this problem
- **Trial 5**: For LayerNorm with M=16, N=4,194,304: safe optimization levers are num_warps tuning, BLOCK_SIZE tuning, Welford single-pass algorithm, and num_stages tuning
- **Trial 5**: When agent produces no output repeatedly, must provide step-by-step instructions that start with verifying the existing kernel works before any changes
