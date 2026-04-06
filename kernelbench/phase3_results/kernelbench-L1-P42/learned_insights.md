# Learned Insights

- **Trial 1**: For MaxPool2D on AMD MI355X with large tensors (32x64x512x512), 1D grid indexing overflows or produces incorrect results beyond ~33M elements - must use 3D grid
- **Trial 1**: Rectangular blocks (16x256) significantly outperform square blocks for MaxPool2D: 2.63ms vs 3.08ms (64x64) vs 3.92ms (32x32) vs 4.17ms (128x128)
- **Trial 1**: MaxPool2D kernel is memory bandwidth bound - each output element requires kernel_size^2=16 loads for 4x4 kernel, making input reuse the key optimization opportunity
- **Trial 1**: PyTorch MaxPool2d baseline for 32x64x512x512 with kernel_size=4, stride=1, padding=1 is 6.46ms on MI355X
- **Trial 2**: For MaxPool2D with stride=1 and kernel_size=4, adjacent output elements share 75% of input data — shared memory tiling should yield significant speedup
- **Trial 2**: Trial 2 produced no output — agent may have spent too long modifying code without running benchmark. Must prioritize getting a metric quickly.
- **Trial 2**: Current best Triton MaxPool2D kernel uses BLOCK_H=16, BLOCK_W=256 with 3D grid, achieving 2.63ms vs 6.46ms PyTorch baseline (score 74.50)
- **Trial 3**: Agent has failed 2 consecutive trials without producing output on MaxPool2D optimization — needs very explicit step-by-step instructions and time constraints
- **Trial 3**: Current best MaxPool2D Triton kernel: BLOCK_H=16, BLOCK_W=256, 3D grid, 2.63ms latency, score 74.50
- **Trial 4**: Agent has failed 3 consecutive trials on MaxPool2D optimization without producing output — needs extremely explicit step-by-step instructions with time constraints
- **Trial 4**: Current best MaxPool2D Triton kernel: BLOCK_H=16, BLOCK_W=256, 3D grid, 2.63ms latency, score 74.50
- **Trial 4**: For MaxPool2D optimization, shared memory tiling is the next logical step since stride=1 means 75% input data overlap between adjacent outputs
- **Trial 5**: Agent has failed 4 consecutive trials on MaxPool2D optimization without producing output — extreme time management issues
- **Trial 5**: Current best MaxPool2D Triton kernel: BLOCK_H=16, BLOCK_W=256, 3D grid, 2.63ms latency, score 74.50
