# Learned Insights

- **Trial 1**: torch.compile(mode='max-autotune') fails on ROCm MI355X with MLIR async_copy_global_to_local legalization errors — avoid it
- **Trial 1**: torch.compile(mode='default') works on ROCm and matches MIOpen baseline performance for Conv3d but doesn't beat it
- **Trial 1**: Custom Triton 3D convolution kernels have persistent correctness issues with large grid indexing — integer division/modulo behavior differs
- **Trial 1**: For 3D convolution on ROCm, MIOpen/rocBLAS GEMM is extremely well optimized — channels_last_3d memory format or FP16 may be needed to beat it
- **Trial 1**: KernelBench score of 50 means correct but no speedup; score > 50 means speedup achieved
- **Trial 2**: Trial 2 produced no agent output — possibly due to agent getting stuck in planning without executing anything
- **Trial 2**: Score of 60 was achieved with torch.compile(mode='default') on Conv3d in trial 1 — this is the working baseline to build on
- **Trial 2**: channels_last_3d memory format is a promising optimization for 3D convolution on ROCm as it aligns with MIOpen's preferred layout
- **Trial 3**: Agent got stuck with no output on 2 consecutive trials — need extremely specific step-by-step instructions with exact code to copy
- **Trial 3**: channels_last_3d memory format is a key optimization to try for 3D convolution on ROCm
- **Trial 3**: When combining FP16 with Conv3d, BOTH input and weights must be converted to half precision to avoid type mismatch
- **Trial 4**: Agent gets stuck with no output when given open-ended optimization instructions — need exact copy-paste code blocks
- **Trial 4**: channels_last_3d memory format has not been tested yet for 3D convolution optimization
- **Trial 4**: torch.compile(mode='default') + channels_last_3d is the most promising untried combination for Conv3d on ROCm
- **Trial 4**: FP16 Conv3d requires .half() on BOTH the module AND the input tensor, and .float() on output for correctness
- **Trial 5**: Agent has been completely stuck for 4 consecutive trials with no output — this is likely a prompt overload issue
- **Trial 5**: For stuck agents, reduce instructions to absolute minimum: cat file, write file, run benchmark — nothing else
- **Trial 5**: Score of 60 was achieved with torch.compile(mode='default') alone; channels_last_3d has never been successfully tested
