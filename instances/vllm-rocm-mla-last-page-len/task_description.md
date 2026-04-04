# Bug: AITER MLA decode produces wrong attention scores due to incorrect paged_kv_last_page_len

## Symptom

The AITER MLA decode backend on ROCm computes incorrect attention scores for certain sequence lengths. The `paged_kv_last_page_len` buffer, which tells the kernel how many tokens reside in the last page of each sequence's KV cache, contains the full sequence length instead of the correct value. Since the AITER MLA kernel always uses a block size of 1 (each page holds exactly one token), the last page always contains exactly 1 token -- but the buffer reports otherwise.

This mismatch causes:
- Wrong attention score calculations for sequences with non-standard lengths
- Potential out-of-bounds memory access in the MLA decode kernel
- Unpredictable behavior especially for sequences with prime-number lengths (e.g., 127, 131)

## Affected file

`vllm/v1/attention/backends/mla/rocm_aiter_mla.py`

## How to reproduce

Run any MLA-architecture model (e.g., DeepSeek-V2) through the AITER MLA decode path on a ROCm GPU with sequences whose token count is a prime number. The incorrect `paged_kv_last_page_len` values will cause the kernel to read beyond valid page boundaries or compute attention with incorrect masking.
