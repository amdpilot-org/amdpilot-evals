# Learned Insights

- **Trial 1**: On ROCm Triton, tl.libdevice.tanh and tl.libdevice.exp are unavailable. Use tl.math.exp instead and implement tanh manually as (exp(2x)-1)/(exp(2x)+1)
- **Trial 1**: MIOpen grouped conv3d accounts for ~34% of runtime and cannot be optimized via Triton
- **Trial 1**: The Conv3d layer may add its own bias separately even when a fused activation kernel handles the model bias — check for duplicate bias additions in profiling
- **Trial 1**: GELU can be approximated as x * sigmoid(1.702 * x) to avoid expensive tanh computation, reusing the sigmoid path
- **Trial 1**: BLOCK_SIZE=256 aligned with wavefront size 64 works but autotuning with larger sizes (512-4096) may improve throughput on MI355X
- **Trial 2**: Trial 2 produced no output - likely the agent timed out or got stuck before running the benchmark. Always verify the benchmark runs first before attempting optimizations.
- **Trial 2**: Conv3d's own bias adds a separate 3.40ms elementwise add kernel (10.1% of runtime). Fusing this into the Triton activation kernel by setting conv bias=None can save significant time.
- **Trial 2**: GELU approximation x * sigmoid(1.702 * x) avoids the expensive manual tanh implementation needed on ROCm Triton
- **Trial 3**: Agent got stuck in trials 2 and 3 with zero output - need explicit step-by-step instructions starting with running the benchmark first
- **Trial 3**: Two main optimization targets remain: (1) fuse conv bias into activation kernel to save ~3.4ms, (2) use GELU sigmoid approximation x*sigmoid(1.702*x) instead of tanh approximation
- **Trial 4**: Agent has failed to produce output 3 trials in a row - needs extremely explicit step-by-step instructions starting with running the existing benchmark
- **Trial 4**: Two concrete optimizations remain: fuse conv bias into activation kernel (save ~3.4ms/10%), use GELU sigmoid approximation x*sigmoid(1.702*x) (faster than manual tanh)
- **Trial 4**: When agent is stuck, first instruction must be to run the benchmark to confirm working state before making any changes
- **Trial 5**: Agent has been stuck 4 consecutive trials - needs complete copy-paste code with no ambiguity
- **Trial 5**: Safer to keep conv bias=True and only fuse the model bias into activation kernel, avoiding state_dict loading issues
- **Trial 5**: GELU sigmoid approximation x*sigmoid(1.702*x) uses tl.math.exp which is available on ROCm, avoids tanh entirely
