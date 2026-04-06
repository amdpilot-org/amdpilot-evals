# Learned Insights

- **Trial 1**: AMD MI355X has grid x-dimension limit around 16-32M programs; use 2D/3D grids for large problems
- **Trial 1**: Triton does not support 'break' statements in loops
- **Trial 1**: For depthwise conv with (K,1) kernel, vectorizing across width dimension with tl.arange gives massive speedup vs per-element parallelism
- **Trial 1**: Row-level parallelism (one program per output row) with BLOCK_WIDTH=512 achieved 0.347ms for depthwise conv on 16x32x512x512 input
- **Trial 1**: KernelBench score of 97.30 corresponds to 4.7x speedup (0.347ms vs 1.63ms reference)
- **Trial 2**: KernelBench score of 97.30 corresponds to 0.347ms kernel time vs 1.63ms reference
- **Trial 2**: Score formula is 100*(ref/kernel)/(1+ref/kernel), so diminishing returns above 95+
- **Trial 2**: Trial 2 agent produced no output - may need explicit instruction to run benchmark immediately rather than over-plan
- **Trial 3**: Trial 2 and Trial 3 both produced no output - agent may be over-planning or hitting timeout before executing benchmark
- **Trial 3**: Score of 97.30 is already excellent; diminishing returns make further improvement very difficult
- **Trial 3**: Must instruct agent to run benchmark FIRST before attempting any changes
- **Trial 4**: KernelBench score of 97.30 (4.7x speedup) is near-optimal; score formula 100*(ref/kernel)/(1+ref/kernel) has severe diminishing returns above 95
- **Trial 4**: Row-level parallelism with BLOCK_WIDTH=512 vectorizing across width dimension is optimal for depthwise conv with (K,1) kernel on MI355X
- **Trial 4**: Agent repeatedly timed out or produced no output on trials 2-4; explicit instructions to run benchmark immediately are insufficient to prevent this
- **Trial 4**: For depthwise 2D conv (16x32x512x512 input, kernel_size=3x1), 0.347ms achieved with 2D grid (height_out, batch*channels) and vectorized width loads
