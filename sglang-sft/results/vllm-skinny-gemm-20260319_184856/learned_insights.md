# Learned Insights

- **Trial 1**: The FP8 wvSplitKQ variant already had Kap/Kbp stride parameters and served as a reference pattern for adding stride support to the FP16/BF16 variants
- **Trial 1**: Three kernel variants needed stride changes: wvSplitK_hf_sml_ (small A fits LDS), wvSplitK_hf_ (medium A), wvSplitK_hf_big_ (large A with PCML chunking)
- **Trial 1**: Key insight for A loading to LDS: flat A[k_in] indexing must be decomposed into row/col with stride-aware source addressing: A[row * Kap + col], while LDS-internal reads stay unchanged since LDS is packed densely using K
- **Trial 1**: B (weight) stride change: B[(m+y) * K + k_] becomes B[(m+y) * Kbp + k_]
- **Trial 1**: The wvSplitKrc (reduce-counting) path was NOT modified — is_contiguous guard kept for that path only
- **Trial 1**: Strides extracted via in_a.stride(0) and in_b.stride(0) in the host function
- **Trial 2**: UV build environment pollutes PATH causing python resolution failures; fix with explicit PATH export before running benchmarks
- **Trial 2**: The complete implementation was done in stage 1 and persisted across stages - no rebuild needed when .so timestamp is newer than .cu source
- **Trial 2**: All four test tiers (profiling 15pts, padded correctness 40pts, non-padded regression 20pts, integration 25pts) pass with the stride parameter approach
- **Trial 3**: The FP8 wvSplitKQ variant already had Kap/Kbp stride parameters and served as a reference pattern for adding stride support to the FP16/BF16 variants
- **Trial 3**: Three kernel variants needed stride changes: wvSplitK_hf_sml_ (small A fits LDS), wvSplitK_hf_ (medium A), wvSplitK_hf_big_ (large A with PCML chunking)
- **Trial 3**: Key insight for A loading to LDS: flat A[k_in] indexing must be decomposed into row/col with stride-aware source addressing: A[row * Kap + col], while LDS-internal reads stay unchanged since LDS is packed densely using K
- **Trial 3**: B (weight) stride change: B[(m+y) * K + k_] becomes B[(m+y) * Kbp + k_]
- **Trial 3**: The wvSplitKrc (reduce-counting) path was NOT modified — is_contiguous guard kept for that path only
- **Trial 3**: Strides extracted via in_a.stride(0) and in_b.stride(0) in the host function
- **Trial 3**: Complete implementation can be done in a single stage if the reference pattern (FP8 variant) is identified early
- **Trial 3**: Test harness has four tiers: profiling evidence (15pts), padded tensor correctness (40pts), non-padded regression (20pts), integration checks (25pts)
