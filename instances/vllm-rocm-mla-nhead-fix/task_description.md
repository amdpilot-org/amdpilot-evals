# Bug: AiterMLA decode backend crashes for models with fewer than 16 attention heads

## Symptom

Models with fewer than 16 attention heads crash when using the AiterMLA decode backend on ROCm. The persistent buffer used during decode is allocated with the raw `num_heads` value, but the underlying aiter kernel requires a minimum of 16 heads for its buffer. When `num_heads < 16`, this causes a buffer size mismatch that leads to a runtime crash.

## How to reproduce

Run any model that has fewer than 16 attention heads (e.g., a smaller variant) through the AiterMLA decode path on a ROCm GPU. The crash occurs during the persistent buffer allocation in `AiterMLADecodeMetadata`.
