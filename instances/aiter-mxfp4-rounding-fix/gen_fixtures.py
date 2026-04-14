#!/usr/bin/env python3
"""Generate test fixtures for MXFP4 quantization correctness checks.

Runs on CPU during Docker build. Produces precomputed expected outputs
for known inputs so the test harness can compare kernel output without
embedding reference code.
"""
import sys
import os
import torch


def ref_dynamic_mxfp4_quant(x):
    BLOCK = 32; BIAS32 = 127; BIAS4 = 1; MB32 = 23; MB4 = 1; max_n = 6; min_n = 1
    sign_mask = 1 << 3
    shape = x.shape
    if shape[-1] % BLOCK != 0:
        s = list(shape); s[-1] = ((s[-1]-1+BLOCK)//BLOCK)*BLOCK
        xp = torch.zeros(s, device=x.device, dtype=x.dtype); xp[...,:shape[-1]] = x
    else:
        xp = x
    xp = xp.reshape(-1, xp.shape[-1]//BLOCK, BLOCK).to(torch.float32)
    amax, _ = torch.max(torch.abs(xp), dim=-1)
    amax = amax.view(torch.int32); amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    se = torch.log2(amax).floor() - 2
    se = torch.clamp(se, min=-127, max=127)
    qs = torch.exp2(-se)
    qx = xp * qs.unsqueeze(-1)
    bs = se.to(torch.uint8) + 127
    qx = qx.view(torch.int32)
    s = qx & 0x80000000; qx = qx ^ s
    qf = qx.view(torch.float32)
    sat = qf >= max_n; den = (~sat) & (qf < min_n); nor = ~(sat | den)
    de = (BIAS32 - BIAS4) + (MB32 - MB4) + 1
    dmi = de << MB32; dmf = torch.tensor(dmi, dtype=torch.int32).view(torch.float32)
    dx = qf + dmf; dx = dx.view(torch.int32); dx -= dmi; dx = dx.to(torch.uint8)
    nx = qx; mo = (nx >> (MB32 - MB4)) & 1
    va = ((BIAS4 - BIAS32) << MB32) + (1 << 21) - 1
    nx += va; nx += mo; nx = nx >> (MB32 - MB4); nx = nx.to(torch.uint8)
    e2 = torch.full_like(qx, 0x7, dtype=torch.uint8)
    e2 = torch.where(nor, nx, e2); e2 = torch.where(den, dx, e2)
    sl = s >> (MB32 + 8 - MB4 - 2); sl = sl.to(torch.uint8); sl = sl & sign_mask
    e2 = e2 | sl
    fp4 = e2[..., ::2] | (e2[..., 1::2] << 4)
    fp4 = torch.flatten(fp4, -2, -1)
    if shape[-1] % BLOCK != 0:
        fp4 = fp4[..., :shape[-1]//2]
    ms = list(shape); ms[-1] = ms[-1]//2
    return fp4.reshape(ms), bs


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)

    shapes = [(1, 32), (2, 64), (128, 32), (1, 68), (256, 32)]
    fixtures = {}

    for M, N in shapes:
        torch.manual_seed(20)
        x = torch.randn((M, N), dtype=torch.bfloat16)
        fp4_out, fp4_scale = ref_dynamic_mxfp4_quant(x)
        fixtures[f"{M}x{N}"] = {
            "input": x,
            "fp4": fp4_out.view(torch.uint8),
            "scale": fp4_scale,
        }

    torch.save(fixtures, os.path.join(outdir, "mxfp4_fixtures.pt"))
    print(f"Generated fixtures for {len(shapes)} shapes in {outdir}")


if __name__ == "__main__":
    main()
