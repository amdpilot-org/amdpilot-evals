# Learned Insights

- **Trial 1**: KernelBench L2-P25: conv2d dominates at 85% of baseline time; only min+tanh+tanh fusion is targetable
- **Trial 1**: Launching one Triton program per spatial position (8.4M launches for batch=128, H=254, W=254) causes catastrophic overhead
- **Trial 1**: tl.static_range for spatial tiling causes code bloat due to compile-time unrolling; use tl.arange for runtime vectorized access instead
- **Trial 1**: The correct pattern for channel-reduction kernels: 2D load with c_offs[:, None] and spatial_offs[None, :], then tl.min(vals, axis=0)
- **Trial 1**: torch.compile on ModelNew with custom Triton kernels can produce incorrect output on ROCm
- **Trial 1**: BLOCK_C must be constexpr and power-of-2 >= out_channels for the 2D tl.load to work
- **Trial 2**: For channel-reduction Triton kernels, use 1D grid over flattened (B*H*W) spatial positions with BLOCK>=256 to keep grid size manageable
- **Trial 2**: Python range() in Triton kernels unrolls at compile time and works on ROCm; tl.static_range causes code bloat and Break AST errors
- **Trial 2**: Agent produced no output in trial 2 — may need explicit file-writing and benchmark-running instructions
- **Trial 3**: Agent stuck for 2 consecutive trials producing no output — needs very explicit code templates and step-by-step execution instructions
- **Trial 3**: For KernelBench scoring, using identical PyTorch ops as baseline should yield score ~100; Triton kernel must beat native PyTorch to score >100
- **Trial 3**: Channel-reduction Triton kernel: use 1D grid over flattened (B*H*W), BLOCK=256, loop over C channels with tl.minimum accumulation
- **Trial 4**: Agent has been stuck for 3 consecutive trials with no output — extremely explicit step-by-step instructions with complete code templates are required
- **Trial 4**: For KernelBench L2-P25: BLOCK_SPATIAL=1024 with 1D grid over B*H*W should reduce grid from 8.4M to ~8K programs
- **Trial 4**: The C-channel loop (64 iterations) in the Triton kernel should use Python range() which unrolls at compile time on ROCm
- **Trial 5**: Agent has been stuck producing no output for 4 trials — explicit complete code templates with copy-paste instructions are essential
- **Trial 5**: For vectorized spatial processing: use offs = pid * BLOCK + tl.arange(0, BLOCK) to process BLOCK positions per program, with mask = offs < total for boundary handling
- **Trial 5**: Stride computation for contiguous (B,C,H,W) tensor: stride_b = C*H*W, stride_c = H*W, and base pointer = b_idx * stride_b + spatial_idx
