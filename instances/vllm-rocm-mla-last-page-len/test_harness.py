#!/usr/bin/env python3
"""Behavioral test: AITER MLA decode paged_kv_last_page_len correctness.

The AITER MLA kernel uses block_size=1 internally, meaning each page holds exactly
one token. The paged_kv_last_page_len buffer must therefore be all-ones regardless
of sequence length. Pre-fix code incorrectly set this to the full sequence length,
causing wrong attention scores for non-power-of-2 sequence lengths.
"""
import subprocess
import sys
import os
import textwrap

NUM_CHECKS = 3
results = {}


def run_subprocess(test_code: str, env_overrides: dict = None) -> tuple:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True, text=True, timeout=120, env=env
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


# CHECK 1: Import succeeds on ROCm
check1_code = textwrap.dedent("""
import torch
if not torch.cuda.is_available() or 'gfx9' not in torch.cuda.get_device_properties(0).gcnArchName:
    print("IMPORT_SKIP: not ROCm gfx9")
    exit(1)
try:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLADecodeMetadata
    print("IMPORT_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {e}")
    exit(1)
""")
ok, out = run_subprocess(check1_code)
if "IMPORT_SKIP" in out:
    print("SCORE: 0 (IMPORT_SKIP — not ROCm gfx9, auto-FAIL)")
    sys.exit(0)
results[1] = ok and "IMPORT_OK" in out
print(f"CHECK 1: {'PASS' if results[1] else 'FAIL'} — AiterMLADecodeMetadata import")

# CHECK 2: Construct AiterMLADecodeMetadata, verify paged_kv_last_page_len is all-ones
check2_code = textwrap.dedent("""
import torch, sys, inspect

from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLADecodeMetadata

device = torch.device("cuda:0")
test_seq_lens = [1, 7, 127, 131, 251, 509, 1024, 2048]
batch = len(test_seq_lens)
block_size = 1

seq_lens_t = torch.tensor(test_seq_lens, dtype=torch.int32, device=device)
max_blocks = max(test_seq_lens)
block_tables = torch.zeros(batch, max_blocks, dtype=torch.int32, device=device)
for i, sl in enumerate(test_seq_lens):
    block_tables[i, :sl] = torch.arange(sl, dtype=torch.int32, device=device)

# Introspect constructor to build with correct field names
sig = inspect.signature(AiterMLADecodeMetadata)
kwargs = {}
for name, param in sig.parameters.items():
    if name == 'self':
        continue
    nl = name.lower()
    if 'seq_len' in nl:
        kwargs[name] = seq_lens_t
    elif name == 'block_tables':
        kwargs[name] = block_tables
    elif name == 'block_size':
        kwargs[name] = block_size
    elif 'max' in nl and 'seq' in nl:
        kwargs[name] = max(test_seq_lens)
    elif 'num_decode' in nl or 'num_actual' in nl:
        kwargs[name] = batch
    elif param.default is not inspect.Parameter.empty:
        continue
    else:
        kwargs[name] = None

try:
    metadata = AiterMLADecodeMetadata(**kwargs)
except Exception as e:
    print(f"CONSTRUCT_FAIL: {e}")
    sys.exit(0)

# The critical invariant: with block_size=1, every page holds exactly 1 token,
# so paged_kv_last_page_len must be 1 for ALL sequences regardless of seq_len.
# The bug: code set paged_kv_last_page_len = seq_len (e.g., [1,7,127,...,2048])
# instead of [1,1,1,...,1].
last_page = getattr(metadata, 'paged_kv_last_page_len', None)
if last_page is None:
    print("FIELD_NONE: paged_kv_last_page_len not set by constructor")
    sys.exit(0)

expected = torch.ones(batch, dtype=last_page.dtype, device=last_page.device)
if torch.all(last_page[:batch] == expected):
    print("ALL_ONES_OK")
else:
    actual = last_page[:batch].tolist()
    print(f"ALL_ONES_FAIL: expected all 1s, got {actual}")
""")
ok, out = run_subprocess(check2_code)
if "CONSTRUCT_FAIL" in out:
    results[2] = False
    detail = [l for l in out.strip().split('\n') if 'CONSTRUCT_FAIL' in l]
    print(f"CHECK 2: FAIL — could not construct metadata: {detail[0] if detail else 'unknown'}")
elif "FIELD_NONE" in out:
    results[2] = False
    print("CHECK 2: FAIL — paged_kv_last_page_len not computed by constructor")
else:
    results[2] = ok and "ALL_ONES_OK" in out
    print(f"CHECK 2: {'PASS' if results[2] else 'FAIL'} — paged_kv_last_page_len all-ones verification")

# CHECK 3: Cross-validate with different prime-length sequences
check3_code = textwrap.dedent("""
import torch, sys, inspect

try:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLADecodeMetadata
except ImportError:
    print("IMPORT_FAIL")
    exit(1)

device = torch.device("cuda:0")

# Cross-validate with a different set of prime sequence lengths
seq_lens = torch.tensor([131, 251, 509], dtype=torch.int32, device=device)
block_tables = torch.zeros(3, 509, dtype=torch.int32, device=device)
for i, sl in enumerate([131, 251, 509]):
    block_tables[i, :sl] = torch.arange(sl, dtype=torch.int32, device=device)

sig = inspect.signature(AiterMLADecodeMetadata)
kwargs = {}
for name, param in sig.parameters.items():
    if name == 'self':
        continue
    nl = name.lower()
    if 'seq_len' in nl:
        kwargs[name] = seq_lens
    elif name == 'block_tables':
        kwargs[name] = block_tables
    elif name == 'block_size':
        kwargs[name] = 1
    elif 'max' in nl and 'seq' in nl:
        kwargs[name] = 509
    elif 'num_decode' in nl or 'num_actual' in nl:
        kwargs[name] = 3
    elif param.default is not inspect.Parameter.empty:
        continue
    else:
        kwargs[name] = None

try:
    metadata = AiterMLADecodeMetadata(**kwargs)
    last_page = getattr(metadata, 'paged_kv_last_page_len', None)
    if last_page is not None and torch.all(last_page[:3] == 1):
        print("FORWARD_OK")
    else:
        vals = last_page[:3].tolist() if last_page is not None else "None"
        print(f"FORWARD_FAIL: last_page_len={vals}")
except Exception as e:
    print(f"FORWARD_FAIL: {e}")
""")
ok, out = run_subprocess(check3_code)
results[3] = ok and "FORWARD_OK" in out
print(f"CHECK 3: {'PASS' if results[3] else 'FAIL'} — MLA decode metadata cross-validation with prime seq lengths")

passed = sum(1 for v in results.values() if v)
score = int(100 * passed / NUM_CHECKS)
print(f"SCORE: {score}")
