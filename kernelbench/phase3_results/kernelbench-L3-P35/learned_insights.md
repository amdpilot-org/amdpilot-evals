# Learned Insights

- **Trial 1**: nn.LSTM on ROCm uses rocDNN which is extremely optimized - custom Triton LSTM cell with Python loop is 2.7x slower due to 3072 kernel launches (512 seq × 6 layers)
- **Trial 1**: nn.LSTM output tensor is non-contiguous (stride=(256, 2560, 1)), must call .contiguous() before passing to Triton kernels
- **Trial 1**: Fused Triton fc+extract kernel failed correctness (max_diff 0.000158 > 1e-4 tolerance) due to accumulation order differences - need explicit fp32 accumulation
- **Trial 1**: KernelBench scoring: score=50 means matching baseline speed, score=60 means slight improvement, higher=better
- **Trial 1**: CDNA4 wavefront size is 64, so BLOCK_SIZE should be multiples of 64 for Triton kernels on MI355X
- **Trial 2**: Agent trial 2 produced no output - likely got stuck on complex optimization without producing a runnable result
- **Trial 2**: torch.compile on nn.LSTM is untried and should be the first optimization attempt
- **Trial 3**: Agent has failed 2 consecutive trials with no output on LSTM optimization - needs extremely specific step-by-step instructions
- **Trial 3**: torch.compile on nn.LSTM has not been tested yet and should be the first thing to try
- **Trial 3**: The working baseline solution scores 50 and is in /workspace/generated_kernel.py
- **Trial 4**: Agent has failed 3 consecutive trials with no output on LSTM optimization - extremely specific copy-paste instructions are needed
- **Trial 4**: torch.compile on nn.LSTM module is the highest-value untested optimization
- **Trial 4**: Fused Triton fc kernel needs explicit fp32 accumulation to pass correctness (max_diff was 0.000158 > 1e-4 with default accumulation)
- **Trial 5**: Agent has failed 4 consecutive trials with no output on LSTM optimization - something is fundamentally blocking execution
- **Trial 5**: The simplest untested optimization is torch.compile on nn.LSTM - just one line change to existing working solution
- **Trial 5**: When agent gets stuck repeatedly, instructions must be reduced to absolute minimum viable change
