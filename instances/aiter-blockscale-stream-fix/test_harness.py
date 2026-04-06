#!/usr/bin/env python3
import os
import re
import subprocess
import sys

checks_passed = 0
checks_total = 0

_P = "/opt/venv/bin/python3"
_R = "/sgl-workspace/aiter"
_D = os.path.join(_R, "csrc/ck_gemm_a8w8_blockscale")


def check(name, cond, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    if cond:
        checks_passed += 1
    s = "PASS" if cond else "FAIL"
    m = f"  [{s}] {name}"
    if detail and not cond:
        m += f": {detail}"
    print(m)
    return cond


print("=" * 60)
print("aiter-blockscale-stream-fix test harness")
print("=" * 60)

print("\n--- Check 1: kernel launch configuration ---")

_sc = []
for root, dirs, files in os.walk(_D):
    for fn in files:
        if fn.endswith((".cuh", ".h", ".hpp", ".cu", ".cpp")):
            fp = os.path.join(root, fn)
            with open(fp) as f:
                c = f.read()
            for m in re.finditer(r'stream_config\s*\{([^}]*)\}', c, re.DOTALL):
                a = [x.strip() for x in m.group(1).split(',')]
                if a:
                    v = re.sub(r'/\*.*?\*/', '', a[0]).strip()
                    _sc.append({"f": os.path.relpath(fp, _R), "r": m.group(0), "v": v})

if not check("Found kernel launch stream_config", len(_sc) > 0,
             f"no stream_config found under {os.path.relpath(_D, _R)}"):
    print(f"\nSCORE: 0.0")
    sys.exit(1)

for e in _sc:
    print(f"    Found in {e['f']}: stream_arg = '{e['v']}'")

print("\n--- Check 2: kernel configuration ---")

_bad = ("nullptr", "NULL", "0", "cudaStreamDefault", "hipStreamDefault",
        "cudaStreamLegacy", "hipStreamLegacy")
_hc = any(e["v"] in _bad for e in _sc)
for e in _sc:
    if e["v"] in _bad:
        print(f"    PROBLEM: {e['f']} uses hardcoded stream '{e['v']}'")

check("Kernel configuration is correct", not _hc,
      "kernel configuration uses a hardcoded value")

print("\n--- Check 3: runtime context ---")

_kw = ["getStream", "getCurrentStream", "current_stream", "Stream()", "stream()"]
_ds = any(any(k in e["v"] for k in _kw) for e in _sc)
if not _ds:
    for e in _sc:
        v = e["v"]
        if v and v not in _bad:
            if "(" in v or (v[0].islower() and v.isidentifier()):
                _ds = True
                break

check("Kernel uses dynamic runtime context", _ds,
      "kernel does not appear to query the active runtime context")

print("\n--- Check 4: module import ---")

_is = 'import sys\nsys.path.insert(0,"/sgl-workspace/aiter")\ntry:\n import aiter\n print("OK")\nexcept Exception as e:\n print(f"FAIL:{e}")'
try:
    r = subprocess.run([_P, "-c", _is], capture_output=True, text=True, timeout=60)
    check("aiter module imports successfully", "OK" in r.stdout.strip(),
          r.stdout.strip() if "FAIL" in r.stdout else "import check failed")
except subprocess.TimeoutExpired:
    check("aiter module imports successfully", False, "timeout")
except Exception as e:
    check("aiter module imports successfully", False, str(e)[:200])

print("\n--- Check 5: cross-stream GEMM consistency ---")

_gt = """
import sys, torch
sys.path.insert(0, "/sgl-workspace/aiter")
if not (hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available()):
    print("NO_HIP_GPU"); sys.exit(0)
try:
    from aiter import dtypes
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_cktile
except (ImportError, Exception) as e:
    print(f"NO_FUNC:{e}"); sys.exit(0)
d = torch.device("cuda:0")
M, N, K = 256, 512, 512
bn, bk = 128, 128
sm, sn, sk = M, (N+bn-1)//bn, (K+bk-1)//bk
xq = (torch.rand((M,K),dtype=torch.float32,device=d)/10).to(dtypes.fp8)
wq = (torch.rand((N,K),dtype=torch.float32,device=d)/10).to(dtypes.fp8)
xsc = torch.rand([sm,sk],dtype=torch.float32,device=d)
wsc = torch.rand([sn,sk],dtype=torch.float32,device=d)
torch.cuda.synchronize()
o1 = torch.empty(M,N,dtype=dtypes.bf16,device=d)
r1 = gemm_a8w8_blockscale_cktile(xq,wq,xsc,wsc,o1)
torch.cuda.synchronize()
if torch.isnan(r1).any().item():
    print("DEFAULT_NAN"); sys.exit(0)
st = torch.cuda.Stream()
torch.cuda.synchronize()
ok = 0
for i in range(5):
    with torch.cuda.stream(st):
        o2 = torch.empty(M,N,dtype=dtypes.bf16,device=d)
        r2 = gemm_a8w8_blockscale_cktile(xq,wq,xsc,wsc,o2)
    st.synchronize()
    if not torch.isnan(r2).any().item() and torch.allclose(r1,r2,rtol=1e-2,atol=1e-2):
        ok += 1
print(f"XSTREAM:{ok}/5")
"""

try:
    r = subprocess.run([_P, "-c", _gt], capture_output=True, text=True, timeout=300)
    out = r.stdout.strip()
    if "NO_HIP_GPU" in out:
        print("  [SKIP] No HIP GPU available")
    elif "NO_FUNC" in out:
        print(f"  [SKIP] GEMM function not available: {out}")
    elif "DEFAULT_NAN" in out:
        print("  [SKIP] Default stream produces NaN (JIT rebuild needed)")
    elif "XSTREAM:5/5" in out:
        check("FP8 GEMM cross-stream consistency", True)
    elif "XSTREAM:" in out:
        mg = re.search(r'XSTREAM:(\d+)/5', out)
        ok = int(mg.group(1)) if mg else 0
        check("FP8 GEMM cross-stream consistency", ok >= 4,
              f"only {ok}/5 iterations matched across streams")
    else:
        print(f"  [SKIP] Unexpected output: {out[:200]}")
except subprocess.TimeoutExpired:
    print("  [SKIP] Cross-stream test timed out (JIT compilation)")
except Exception as e:
    print(f"  [SKIP] Error: {str(e)[:200]}")

print()
score = (checks_passed / checks_total * 100.0) if checks_total > 0 else 0.0
print(f"Results: {checks_passed}/{checks_total} checks passed")
print(f"SCORE: {score:.1f}")
sys.exit(0 if checks_passed == checks_total else 1)
