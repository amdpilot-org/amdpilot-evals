#!/usr/bin/env python3
"""Test harness for sglang NSA tensor aliasing crash on ROCm.

Tests (behavioral):
  1. Import NSA indexer module and verify Indexer class and methods exist.
  2. Source inspect _get_q_k_bf16 — must have .clone() on both q_rope and
     k_rope write-back lines to prevent self-aliased tensor writes.
  3. Source inspect _get_k_bf16 (nsa_indexer.py) AND _set_mla_kv_buffer
     (forward_mha.py) — must have .clone() on all remaining aliased sites.
  4. Behavioral platform test — verify ROCm enforces self-aliased write
     rejection, confirming the bug is real and .clone() is necessary.
"""

import os
import subprocess
import sys
import json

_PY = "/opt/venv/bin/python3"

# Known file paths in the container
_NSA_INDEXER = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py"
_FORWARD_MHA_GLOB = "/sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py"


def run_check(name, script, timeout=120):
    """Run a subprocess check and return (pass, detail)."""
    proc = subprocess.run(
        [_PY, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": "/sgl-workspace/sglang/python"},
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if proc.returncode != 0:
        if "same memory location" in stderr or "same memory location" in stdout:
            return False, "Tensor aliasing crash detected"
        return False, f"Exit code {proc.returncode}: {stderr[-500:]}"

    try:
        result = json.loads(stdout.split("\n")[-1])
        return result.get("pass", False), result.get("detail", stdout[-200:])
    except (json.JSONDecodeError, IndexError):
        return True, stdout[-200:]


def main():
    print("=" * 60)
    print("SGLang NSA Tensor Aliasing Test")
    print("=" * 60)

    checks = []

    # Check 1: Import NSA indexer and verify Indexer class exists
    print("\n[Check 1] Import NSA indexer module...")
    script1 = """
import sys, json
sys.path.insert(0, "/sgl-workspace/sglang/python")
try:
    from sglang.srt.layers.attention.nsa import nsa_indexer
    has_indexer = hasattr(nsa_indexer, 'Indexer')
    has_get_q_k = hasattr(nsa_indexer.Indexer, '_get_q_k_bf16') if has_indexer else False
    has_get_k = hasattr(nsa_indexer.Indexer, '_get_k_bf16') if has_indexer else False

    ok = has_indexer and has_get_q_k and has_get_k
    detail = f"Indexer={has_indexer}, _get_q_k_bf16={has_get_q_k}, _get_k_bf16={has_get_k}"
    print(json.dumps({"pass": ok, "detail": detail}))
except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Import failed: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("import_nsa", script1)
    checks.append({"name": "import_nsa_indexer", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 2: _get_q_k_bf16 must have .clone() on BOTH q_rope and k_rope write-backs
    # These are the two aliased write-back sites in this function.
    print("\n[Check 2] _get_q_k_bf16 has .clone() on aliased write-backs...")
    script2 = f"""
import sys, json
sys.path.insert(0, "/sgl-workspace/sglang/python")

try:
    # Try inspect.getsource first, fall back to file reading
    src = ""
    try:
        import inspect
        from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
        src = inspect.getsource(Indexer._get_q_k_bf16)
    except Exception:
        with open("{_NSA_INDEXER}") as f:
            content = f.read()
        # Extract _get_q_k_bf16 function body
        start = content.find("def _get_q_k_bf16")
        if start >= 0:
            # Find next def at same or lower indent
            rest = content[start:]
            lines = rest.split("\\n")
            func_lines = [lines[0]]
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines[1:]:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                if stripped and indent <= base_indent and stripped.startswith("def "):
                    break
                func_lines.append(line)
            src = "\\n".join(func_lines)

    if not src:
        print(json.dumps({{"pass": False, "detail": "_get_q_k_bf16 source not found"}}))
        sys.exit(1)

    lines = src.split("\\n")

    # Find write-back lines and check for .clone()
    # Pattern: query[..., : self.rope_head_dim] = q_rope.clone()
    #          key[..., : self.rope_head_dim] = k_rope.clone()
    q_rope_clone = False
    k_rope_clone = False

    for line in lines:
        stripped = line.strip()
        # Match the q_rope write-back
        if "rope_head_dim" in stripped and "= q_rope" in stripped:
            q_rope_clone = ".clone()" in stripped
        # Match the k_rope write-back
        if "rope_head_dim" in stripped and "= k_rope" in stripped:
            k_rope_clone = ".clone()" in stripped

    ok = q_rope_clone and k_rope_clone
    detail = f"q_rope.clone()={{q_rope_clone}}, k_rope.clone()={{k_rope_clone}}"
    print(json.dumps({{"pass": ok, "detail": detail}}))

except Exception as e:
    print(json.dumps({{"pass": False, "detail": f"Error: {{e}}"}}))
    sys.exit(1)
"""
    passed, detail = run_check("get_q_k_clone", script2)
    checks.append({"name": "get_q_k_bf16_cloned", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 3: _get_k_bf16 AND _set_mla_kv_buffer must have .clone()
    # This ensures the fix spans BOTH files (nsa_indexer.py and forward_mha.py).
    print("\n[Check 3] _get_k_bf16 and _set_mla_kv_buffer have .clone()...")
    script3 = f"""
import sys, json
sys.path.insert(0, "/sgl-workspace/sglang/python")

try:
    # --- Check _get_k_bf16 in nsa_indexer.py ---
    k_bf16_src = ""
    try:
        import inspect
        from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
        k_bf16_src = inspect.getsource(Indexer._get_k_bf16)
    except Exception:
        with open("{_NSA_INDEXER}") as f:
            content = f.read()
        start = content.find("def _get_k_bf16")
        if start >= 0:
            rest = content[start:]
            lines = rest.split("\\n")
            func_lines = [lines[0]]
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            for line in lines[1:]:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                if stripped and indent <= base_indent and stripped.startswith("def "):
                    break
                func_lines.append(line)
            k_bf16_src = "\\n".join(func_lines)

    k_rope_clone = False
    if k_bf16_src:
        for line in k_bf16_src.split("\\n"):
            stripped = line.strip()
            if "rope_head_dim" in stripped and "= k_rope" in stripped:
                k_rope_clone = ".clone()" in stripped

    # --- Check _set_mla_kv_buffer in forward_mha.py ---
    k_pe_clone = False
    import glob
    mha_files = glob.glob("/sgl-workspace/sglang/python/**/forward_mha.py", recursive=True)
    mha_src = ""
    for mf in mha_files:
        with open(mf) as f:
            content = f.read()
        if "_set_mla_kv_buffer" in content:
            # Extract the function
            start = content.find("def _set_mla_kv_buffer")
            if start >= 0:
                rest = content[start:]
                lines = rest.split("\\n")
                func_lines = [lines[0]]
                base_indent = len(lines[0]) - len(lines[0].lstrip())
                for line in lines[1:]:
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    if stripped and indent <= base_indent and (stripped.startswith("def ") or not line.startswith(" ")):
                        break
                    func_lines.append(line)
                mha_src = "\\n".join(func_lines)
            break

    if mha_src:
        for line in mha_src.split("\\n"):
            stripped = line.strip()
            if "kv_lora_rank" in stripped and "= k_pe" in stripped:
                k_pe_clone = ".clone()" in stripped

    ok = k_rope_clone and k_pe_clone
    detail = f"_get_k_bf16 k_rope.clone()={{k_rope_clone}}, _set_mla_kv_buffer k_pe.clone()={{k_pe_clone}}"
    print(json.dumps({{"pass": ok, "detail": detail}}))

except Exception as e:
    print(json.dumps({{"pass": False, "detail": f"Error: {{e}}"}}))
    sys.exit(1)
"""
    passed, detail = run_check("remaining_clones", script3)
    checks.append({"name": "remaining_sites_cloned", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Check 4: Behavioral — ROCm enforces self-aliased write rejection
    # Replicates the exact aliasing pattern from _get_q_k_bf16 and confirms
    # that .clone() is the correct fix on this platform.
    print("\n[Check 4] ROCm self-aliased write enforcement...")
    script4 = """
import torch, sys, json
sys.path.insert(0, "/sgl-workspace/sglang/python")

torch.cuda.set_device(0)
device = "cuda:0"

try:
    # Replicate the aliasing pattern from _get_q_k_bf16:
    # 1. query tensor, q_rope = query[..., :rope_dim] (aliased view)
    # 2. In-place modification (simulates rotary_emb forward_cuda)
    # 3. Write-back: query[..., :rope_dim] = q_rope (self-aliased write)

    query = torch.randn(32, 8, 128, dtype=torch.bfloat16, device=device)
    rope_dim = 64
    q_rope = query[..., :rope_dim]  # aliased view of query

    # Simulate forward_cuda in-place modification
    q_rope.mul_(2.0)

    # Self-aliased write-back (the buggy pattern)
    alias_rejected = False
    try:
        query[..., :rope_dim] = q_rope
    except RuntimeError as e:
        if "same memory" in str(e).lower() or "alias" in str(e).lower():
            alias_rejected = True

    # Verify .clone() fixes it
    query2 = torch.randn(32, 8, 128, dtype=torch.bfloat16, device=device)
    q_rope2 = query2[..., :rope_dim]
    q_rope2.mul_(2.0)

    clone_works = False
    try:
        query2[..., :rope_dim] = q_rope2.clone()
        clone_works = True
    except RuntimeError:
        pass

    ok = alias_rejected and clone_works
    detail = f"alias_rejected={alias_rejected}, clone_works={clone_works}"
    print(json.dumps({"pass": ok, "detail": detail}))

except Exception as e:
    print(json.dumps({"pass": False, "detail": f"Error: {e}"}))
    sys.exit(1)
"""
    passed, detail = run_check("platform_check", script4, timeout=180)
    checks.append({"name": "rocm_aliasing_enforced", "pass": passed, "detail": detail})
    print(f"  {'PASS' if passed else 'FAIL'}: {detail}")

    # Score
    total = len(checks)
    passed_count = sum(1 for c in checks if c["pass"])

    print(f"\n--- Results ---")
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['detail'][:100]}")

    if total == 0:
        print("\nNo checks completed.")
        print("SCORE: 0.0")
        return

    score = passed_count / total * 100.0
    print(f"\n{passed_count}/{total} checks passed")
    print(f"SCORE: {score:.1f}")


if __name__ == "__main__":
    main()
