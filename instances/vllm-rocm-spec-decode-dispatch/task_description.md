# AITER FA decode produces incorrect results during speculative decoding

The ROCM_AITER_FA attention backend (the ROCm AITER FA attention backend) produces incorrect attention output when speculative decoding is enabled (e.g., ngram speculation with `num_speculative_tokens > 0`).

## Symptom

During speculative decoding, the decode batch contains multi-token queries (`max_query_len > 1`), but the AITER FA decode path unconditionally calls `paged_attention_v1` with `max_seqlen_q=1`. This hardcoded assumption is only correct for standard single-token decode steps. When multiple speculative tokens are verified in a single decode step, the query length exceeds 1 and the attention computation produces wrong results, leading to garbled or incorrect text output.

## Reproduction

On MI300X with AITER installed:
```bash
VLLM_ATTENTION_BACKEND=ROCM_AITER_FA python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='meta-llama/Llama-3.1-8B-Instruct',
          speculative_config={'method': 'ngram', 'num_speculative_tokens': 3})
out = llm.generate(['What is 2+2?'], SamplingParams(max_tokens=20))
print(out[0].outputs[0].text)
"
```

The output will be incorrect or garbled when using the AITER FA backend with speculative decoding.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
