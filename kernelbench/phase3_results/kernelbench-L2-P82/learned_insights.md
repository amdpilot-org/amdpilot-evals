# Learned Insights

- **Trial 1**: ROCm Triton does not have tl.libdevice.tanh; must use manual formula: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
- **Trial 1**: BLOCK_SIZE=256 (multiple of 64) works well for AMD MI355X wavefront alignment
- **Trial 1**: For problem 82 (Conv2d+Tanh+Scale+Bias+MaxPool), fusing tanh+scale+bias gives ~1.16x speedup; conv2d and max_pool remain as PyTorch ops
- **Trial 1**: Problem 82 dimensions: batch=128, in_ch=8, out_ch=64, H=W=256, conv_kernel=3, pool_kernel=4 — output after conv is 254x254, after pool is 63x63
- **Trial 1**: Score formula appears to be: score = 100 * (1 - new_time/ref_time) or similar; 4.49ms vs 5.21ms gives score ~61.7
- **Trial 2**: Trial 2 produced no output — possibly agent crashed before executing any code; need to ensure agent starts by reading existing files
- **Trial 2**: Fusing max_pool into the tanh+scale+bias kernel is the next logical optimization — eliminates a full memory pass over (128,64,254,254) tensor
- **Trial 3**: Trial 2 and 3 both produced no output — agent may be stuck on startup/environment rather than coding; need extremely explicit step-by-step instructions
- **Trial 3**: Fusing max_pool(4x4) into tanh+scale+bias kernel eliminates a full memory roundtrip over (128,64,254,254) tensor — each output program handles one (b,c,oh,ow) and loads 16 elements
- **Trial 4**: Agent has crashed/stalled 3 consecutive trials (2,3,4) with no output — may need extremely explicit code snippets to avoid getting stuck
- **Trial 4**: Fusing max_pool(4x4) into tanh+scale+bias eliminates memory roundtrip over (128,64,254,254) tensor; output is (128,64,63,63) = 32.5M elements
- **Trial 4**: For the fused maxpool kernel, use 1D grid with each program handling BLOCK_SIZE output elements, decompose linear index to (b,c,oh,ow), iterate 4x4 pool window
- **Trial 5**: Agent has crashed 4 consecutive trials (2-5) with no output — must provide complete copy-paste code
- **Trial 5**: tl.static_range should be used instead of range() for compile-time-known pool loops in Triton
- **Trial 5**: Clamping input to [-20, 20] before exp(2x) prevents float32 overflow in manual tanh
