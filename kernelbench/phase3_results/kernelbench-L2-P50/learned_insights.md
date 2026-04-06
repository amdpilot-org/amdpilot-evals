# Learned Insights

- **Trial 1**: KernelBench Problem 50: ConvTranspose3d dominates at 64.4% - uses MIOpen/CK, hard to optimize further
- **Trial 1**: torch.compile(mode='default') gives 7% speedup (3.42->3.17ms) by fusing elementwise ops
- **Trial 1**: Manual Triton kernels on AMD MI355X only support 3D grids (program_id 0,1,2)
- **Trial 1**: Integer division for 5D flat index decoding causes correctness issues at large batch sizes (>=64) - decompose step by step
- **Trial 1**: Fine-grained Triton grids (one program per output element) cause excessive launch overhead on AMD - use coarser blocks with tl.arange vectorization
- **Trial 1**: ConvTranspose3d(3,16,3,stride=2,padding=1) on input (128,3,16,32,32) produces output (128,16,31,63,63)
- **Trial 2**: Trial 2 agent produced no output - possibly got stuck trying to write complex Triton kernel. Need to instruct agent to start from working solution first.
- **Trial 2**: Fusing scale1 into ConvTranspose3d weights at init time eliminates one full elementwise pass over the output tensor
- **Trial 3**: Agent has stalled twice on this problem when trying to write complex code from scratch - give explicit code snippets
- **Trial 3**: For KernelBench scoring, incremental improvements to a working solution are safer than rewriting from scratch
- **Trial 4**: Agent repeatedly stalls when given open-ended optimization tasks on hard problems - provide explicit code snippets
- **Trial 4**: For KernelBench problems dominated by vendor-optimized ops (MIOpen ConvTranspose3d), focus on init-time weight fusion and torch.compile rather than manual kernels
- **Trial 5**: Agent stalls repeatedly when given open-ended optimization tasks - must provide complete copy-paste code
- **Trial 5**: For KernelBench Problem 50, fusing scale1 into conv weights saves one elementwise pass over (128,16,31,63,63) tensor
