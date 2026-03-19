# Learned Insights

- **Trial 1**: Docker container pausing can block all exec commands — must unpause before any work can proceed
- **Trial 1**: The fix requires changing 4 mla_decode_fwd call sites in aiter_backend.py to fall back to self.k_scale when layer.k_scale is None
- **Trial 2**: Docker container pausing can block all exec commands — must unpause before any work can proceed
- **Trial 2**: The fix requires changing 4 mla_decode_fwd call sites in aiter_backend.py to fall back to self.k_scale when layer.k_scale is None
- **Trial 2**: self.k_scale should be initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__ if not already present
- **Trial 3**: Docker container pausing can block all exec commands — must unpause before any work can proceed
- **Trial 3**: The fix requires changing 4 mla_decode_fwd call sites in aiter_backend.py to fall back to self.k_scale when layer.k_scale is None
- **Trial 3**: self.k_scale should be initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__ if not already present
- **Trial 3**: LLM provider errors can cause complete trial failure with no agent action taken — retry is appropriate
- **Trial 4**: Exit code 137 indicates OOM or container kill — agent must act quickly and avoid unnecessary file reads
- **Trial 4**: The fix requires changing 4 mla_decode_fwd call sites in aiter_backend.py to fall back to self.k_scale when layer.k_scale is None
- **Trial 4**: A simple string replace of 'layer.k_scale,' with '(layer.k_scale if layer.k_scale is not None else self.k_scale),' in the file can patch all 4 call sites at once
- **Trial 4**: self.k_scale should be initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__ if not already present
- **Trial 5**: Exit code 137 indicates OOM or container kill — agent must act quickly and avoid unnecessary file reads
- **Trial 5**: The fix requires changing 4 mla_decode_fwd call sites in aiter_backend.py to fall back to self.k_scale when layer.k_scale is None
- **Trial 5**: A simple sed replace of 'layer\.k_scale,' with '(layer.k_scale if layer.k_scale is not None else self.k_scale),' can patch all call sites at once
- **Trial 5**: Trial 5 applied the sed fix successfully but was killed before running the test — changes may have persisted on disk
- **Trial 5**: self.k_scale should be initialized to torch.tensor([1.0]) in AiterAttnBackend.__init__ if not already present
