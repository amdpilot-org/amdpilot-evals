# Learned Insights

- **Trial 1**: Exit code 137 means the agent process was killed (OOM or container timeout). Keep the trial lightweight.
- **Trial 1**: The fix is in aiter_backend.py: replace `layer.k_scale` with a fallback to `self.k_scale` at all 4 `mla_decode_fwd` call sites.
- **Trial 1**: self.k_scale is initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__
- **Trial 2**: Exit code 137 means the agent process was killed (OOM or container timeout). Keep the trial lightweight.
- **Trial 2**: The fix is in aiter_backend.py: replace `layer.k_scale` with a fallback to `self.k_scale` at all 4 `mla_decode_fwd` call sites.
- **Trial 2**: self.k_scale is initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__
- **Trial 2**: Agent got killed twice with no output - must give direct executable commands, no file reading or planning phase.
- **Trial 2**: Use python3 -c with string replace to apply the fix in one shot rather than reading the file first.
- **Trial 3**: Exit code 137 means the agent process was killed (OOM or container timeout). Keep the trial lightweight.
- **Trial 3**: The fix is in aiter_backend.py: replace `layer.k_scale` with a fallback to `self.k_scale` at all 4 `mla_decode_fwd` call sites.
- **Trial 3**: self.k_scale is initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__
- **Trial 3**: Agent got killed 3 times with no output - must give direct executable commands as first action, no file reading or planning phase.
- **Trial 3**: Use sed -i 's/layer\.k_scale,/layer.k_scale if layer.k_scale is not None else self.k_scale,/g' aiter_backend.py to apply the fix in one shot.
