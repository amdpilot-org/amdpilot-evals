#!/usr/bin/env python3
"""Test harness for sglang-kimi-w4a16-moe-dispatch.

Behavioral test: verifies W4A16-quantized MoE models can load and run on ROCm.

Scoring:
  Check 1 (25 pts): Production dispatch returns a ROCm-compatible method
  Check 2 (25 pts): Weight format conversion (int32-packed -> compute layout)
  Check 3 (25 pts): method.apply() forward pass on GPU
  Check 4 (25 pts): Numerical correctness against dequantized reference

SCORE: 0-100
"""

import os
import sys
import traceback

SGLANG_DIR = "/workspace/sglang"
sglang_python = os.path.join(SGLANG_DIR, "python")
if sglang_python not in sys.path:
    sys.path.insert(0, sglang_python)

os.environ.setdefault("SGLANG_USE_AITER", "0")

scores = {}


def check(name, condition, points, detail=""):
    status = "PASS" if condition else "FAIL"
    awarded = points if condition else 0
    scores[name] = awarded
    msg = f"  [{status}] {name} ({awarded}/{points} pts)"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


# ── Test dimensions (small for speed) ────────────────────────────
E = 8           # experts
K = 256         # hidden size
N = 128         # intermediate size per expert
GROUP_SIZE = 32
TOPK = 2
PACK_FACTOR = 8  # 8 int4 values per int32


def make_mock_layer(device):
    """Create a mock layer with int32-packed W4A16 weights."""
    import torch

    class MockLayer(torch.nn.Module):
        pass

    layer = MockLayer()

    # Deterministic seed for reproducible weights
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)

    # CompressedTensors stores int4 weights packed in int32: [E, K//8, 2N]
    layer.w13_weight_packed = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (E, K // PACK_FACTOR, 2 * N),
                       dtype=torch.int32, generator=gen).to(device),
        requires_grad=False,
    )
    layer.w2_weight_packed = torch.nn.Parameter(
        torch.randint(0, 2**31 - 1, (E, N // PACK_FACTOR, K),
                       dtype=torch.int32, generator=gen).to(device),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.randn(E, K // GROUP_SIZE, 2 * N, dtype=torch.float16,
                     generator=gen).abs().to(device) * 0.01,
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.randn(E, N // GROUP_SIZE, K, dtype=torch.float16,
                     generator=gen).abs().to(device) * 0.01,
        requires_grad=False,
    )
    # g_idx and sort_indices (empty for non-actorder, as in production)
    layer.w13_weight_g_idx = torch.nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )
    layer.w2_weight_g_idx = torch.nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )
    layer.w13_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )
    layer.w2_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device),
        requires_grad=False,
    )
    # Weight shape params
    layer.w13_weight_shape = torch.nn.Parameter(
        torch.tensor([[K, 2 * N]] * E, dtype=torch.float32, device=device),
        requires_grad=False,
    )
    layer.w2_weight_shape = torch.nn.Parameter(
        torch.tensor([[N, K]] * E, dtype=torch.float32, device=device),
        requires_grad=False,
    )
    layer._original_shapes = {
        "w13_weight_packed": (E, K // PACK_FACTOR, 2 * N),
        "w2_weight_packed": (E, N // PACK_FACTOR, K),
        "w13_weight_scale": (E, K // GROUP_SIZE, 2 * N),
        "w2_weight_scale": (E, N // GROUP_SIZE, K),
    }
    return layer


def build_quant_config():
    """Build a CompressedTensorsConfig matching W4A16 group-quantized MoE."""
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )
    from compressed_tensors.config import CompressionFormat
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    weight_quant = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=GROUP_SIZE,
        dynamic=False,
    )

    config = CompressedTensorsConfig(
        target_scheme_map={
            "Linear": {"weights": weight_quant, "input_activations": None},
        },
        ignore=[],
        quant_format=CompressionFormat.pack_quantized.value,
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )
    return config


def dequantize_int4(packed_uint8, scales, group_size):
    """Dequantize W4A16 packed weights to float32.

    packed_uint8: [rows, cols_packed] uint8, each byte holds 2 int4 nibbles.
                  Low nibble = even k-index, high nibble = odd k-index.
    scales: [rows, cols_full // group_size] float
    Returns: [rows, cols_full] float32 where cols_full = cols_packed * 2
    """
    import torch

    rows, cols_packed = packed_uint8.shape
    cols_full = cols_packed * 2

    low = (packed_uint8 & 0xF).to(torch.float32)
    high = ((packed_uint8 >> 4) & 0xF).to(torch.float32)

    # Interleave: [rows, cols_packed, 2] -> [rows, cols_full]
    w = torch.stack([low, high], dim=-1).reshape(rows, cols_full)

    # Symmetric int4: subtract zero point (8)
    w = w - 8.0

    # Apply per-group scales
    scale_expanded = scales.float().repeat_interleave(group_size, dim=1)
    scale_expanded = scale_expanded[:, :cols_full]

    return w * scale_expanded


def reference_moe_forward(x, w13_packed, w2_packed, w13_scale, w2_scale,
                          topk_ids, topk_weights, group_size):
    """Reference MoE forward pass using manual dequantization.

    All inputs are the POST-conversion weights (uint8 packed, transposed).
    w13_packed: [E, 2N, K//2] uint8
    w2_packed: [E, K, N//2] uint8
    w13_scale: [E, 2N, K//group_size] float16
    w2_scale: [E, K, N//group_size] float16
    """
    import torch
    import torch.nn.functional as F

    M, hidden = x.shape
    output = torch.zeros(M, hidden, dtype=torch.float32, device=x.device)

    for i in range(M):
        token_out = torch.zeros(hidden, dtype=torch.float32, device=x.device)
        for j in range(topk_ids.shape[1]):
            e = topk_ids[i, j].item()
            w = topk_weights[i, j].item()

            # Dequantize w13 for expert e: [2N, K//2] uint8 -> [2N, K] float
            w13_e = dequantize_int4(w13_packed[e], w13_scale[e], group_size)
            n = w13_e.shape[0] // 2

            # gate + up projection: [K] -> [2N]
            x_f32 = x[i].float()
            gate_up = w13_e @ x_f32  # [2N, K] @ [K] -> [2N]

            gate = gate_up[:n]
            up = gate_up[n:]
            hidden_act = F.silu(gate) * up  # [N]

            # Dequantize w2 for expert e: [K, N//2] uint8 -> [K, N] float
            w2_e = dequantize_int4(w2_packed[e], w2_scale[e], group_size)

            # down projection: [K, N] @ [N] -> [K]
            down = w2_e @ hidden_act  # [K]

            token_out += w * down

        output[i] = token_out

    return output


def make_fixed_dispatch(x, device):
    """Create a StandardDispatchOutput with fixed (deterministic) routing.

    Uses a fixed seed for gating to ensure identical expert assignments
    across repeated calls with the same input.
    """
    import torch
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput

    M = x.shape[0]

    # Fixed gating with explicit seed
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    router_logits = torch.randn(M, E, dtype=torch.float32, device=device,
                                generator=gen)

    topk_config = TopKConfig(top_k=TOPK, renormalize=True)
    topk_output = select_experts(x, router_logits, topk_config)

    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=topk_output,
    )
    return dispatch_output, topk_output


def main():
    print("=" * 60)
    print("W4A16 MoE ROCm Dispatch Test Harness")
    print("=" * 60)

    import torch

    method = None
    layer = None

    # ═══════════════════════════════════════════════════════════════
    # Check 1: Production dispatch routing (25 pts)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Check 1: Production Dispatch Routing (25 pts) ---")
    print("  Calling CompressedTensorsMoEMethod.get_moe_method() ...")

    try:
        from sglang.srt.utils import is_hip
        is_rocm = is_hip()
        if not is_rocm:
            print("  WARNING: Not on ROCm — dispatch routing may differ.")

        quant_config = build_quant_config()

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            CompressedTensorsMoEMethod,
        )

        mock_layer = torch.nn.Module()
        method = CompressedTensorsMoEMethod.get_moe_method(
            quant_config, mock_layer, "model.layers.0.mlp.experts"
        )

        method_exists = method is not None
        check("get_moe_method() returns a method", method_exists, 10,
              "get_moe_method() returned None")

        if method is not None:
            method_cls = type(method).__name__

            # Inspect the method's apply() for Marlin usage
            import inspect
            try:
                src = inspect.getsource(type(method).apply)
                uses_marlin = "fused_marlin_moe" in src or "marlin" in src.lower()
            except (TypeError, OSError):
                uses_marlin = True

            check("Method does not use Marlin kernels", not uses_marlin, 10,
                  f"{method_cls}.apply() references Marlin — "
                  "will crash on ROCm")

            has_methods = (
                hasattr(method, "process_weights_after_loading")
                and hasattr(method, "apply")
                and hasattr(method, "create_moe_runner")
            )
            check("Method has required interface", has_methods, 5,
                  f"{method_cls} missing process_weights/apply/create_moe_runner")
        else:
            check("Method does not use Marlin kernels", False, 10, "No method")
            check("Method has required interface", False, 5, "No method")

    except Exception as e:
        check("Production dispatch", False, 25, str(e))
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # Check 2: Weight format conversion (25 pts)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Check 2: Weight Format Conversion (25 pts) ---")
    print("  Verifying int32-packed weights -> compute-ready format...")

    weight_ok = False
    w13_orig = None
    w2_orig = None
    w13_scale_orig = None
    w2_scale_orig = None

    try:
        if not torch.cuda.is_available():
            check("GPU available", False, 25, "No GPU")
        elif method is None:
            check("Weight conversion (no method)", False, 25,
                  "Skipped — dispatch check failed")
        else:
            device = torch.device("cuda:0")
            layer = make_mock_layer(device)

            # Save pre-conversion copies for reference oracle (Check 4)
            w13_orig = layer.w13_weight_packed.data.clone()
            w2_orig = layer.w2_weight_packed.data.clone()
            w13_scale_orig = layer.w13_weight_scale.data.clone()
            w2_scale_orig = layer.w2_weight_scale.data.clone()

            orig_w13_dtype = layer.w13_weight_packed.data.dtype
            orig_w13_shape = layer.w13_weight_packed.data.shape

            method.process_weights_after_loading(layer)

            w13 = layer.w13_weight_packed.data
            w2 = layer.w2_weight_packed.data

            # Weights must leave int32 format
            converted = w13.dtype != torch.int32
            check("Weights converted from int32", converted, 10,
                  f"Still {w13.dtype}")

            # Shape should change (transpose for compute layout)
            shape_changed = w13.shape != orig_w13_shape
            check("Weight shape transformed for compute", shape_changed, 5,
                  f"Shape unchanged: {w13.shape}")

            # Scales should be transposed too
            s13 = layer.w13_weight_scale.data
            # After conversion: [E, 2N, K//group_size]
            scale_ok = s13.shape[1] == 2 * N
            check("Scales transposed for compute layout", scale_ok, 10,
                  f"Scale shape {s13.shape}, expected dim1 = {2 * N}")

            weight_ok = converted and shape_changed and scale_ok

    except Exception as e:
        check("Weight conversion", False, 25, str(e))
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # Check 3: method.apply() forward pass (25 pts)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Check 3: method.apply() Forward Pass (25 pts) ---")
    print("  Running full dispatch->runner pipeline on GPU...")

    forward_ok = False
    fwd_output = None

    # Initialize mock server args — needed outside a serving context
    # because kernel config lookup calls get_global_server_args().
    try:
        import sglang.srt.server_args as _sa
        if not hasattr(_sa, '_global_server_args') or _sa._global_server_args is None:
            class _MockServerArgs:
                enable_deterministic_inference = False
            _sa._global_server_args = _MockServerArgs()
    except Exception:
        pass

    try:
        if not torch.cuda.is_available():
            check("GPU available", False, 25, "No GPU")
        elif method is None or layer is None or not weight_ok:
            check("Forward pass (prerequisites failed)", False, 25,
                  "Skipped — earlier checks failed")
        else:
            device = torch.device("cuda:0")

            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

            moe_cfg = MoeRunnerConfig(
                num_experts=E,
                num_local_experts=E,
                top_k=TOPK,
                hidden_size=K,
                intermediate_size_per_partition=N,
                inplace=False,
            )
            method.create_moe_runner(layer, moe_cfg)

            M = 32
            torch.manual_seed(999)
            x = torch.randn(M, K, dtype=torch.float16, device=device)

            dispatch_output, _ = make_fixed_dispatch(x, device)
            combine_input = method.apply(layer, dispatch_output)
            fwd_output = combine_input.hidden_states

            check("method.apply() completes", True, 10)

            shape_ok = fwd_output.shape == (M, K)
            check(f"Output shape {tuple(fwd_output.shape)} == ({M}, {K})",
                  shape_ok, 5, f"Got {tuple(fwd_output.shape)}")

            no_nan = not torch.isnan(fwd_output).any().item()
            no_inf = not torch.isinf(fwd_output).any().item()
            check("Output has no NaN/Inf", no_nan and no_inf, 5,
                  f"NaN={not no_nan}, Inf={not no_inf}")

            # Determinism: same fixed dispatch -> same output
            dispatch_output2, _ = make_fixed_dispatch(x, device)
            combine_input2 = method.apply(layer, dispatch_output2)
            fwd_output2 = combine_input2.hidden_states
            match = torch.allclose(fwd_output, fwd_output2, rtol=1e-3, atol=1e-3)
            check("Deterministic (fixed routing -> same output)", match, 5,
                  f"Max diff {(fwd_output - fwd_output2).abs().max().item():.6f}")

            forward_ok = shape_ok and no_nan and no_inf

    except Exception as e:
        check("method.apply() forward pass", False, 25, str(e))
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # Check 4: Numerical correctness oracle (25 pts)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Check 4: Correctness Oracle (25 pts) ---")
    print("  Comparing method output against dequantized reference...")

    try:
        if not forward_ok or fwd_output is None:
            check("Correctness (forward failed)", False, 25, "Skipped")
        elif w13_orig is None:
            check("Correctness (no original weights)", False, 25, "Skipped")
        else:
            device = torch.device("cuda:0")

            # Manually convert original weights: transpose + view as uint8
            w13_conv = w13_orig.transpose(1, 2).contiguous().view(torch.uint8)
            w2_conv = w2_orig.transpose(1, 2).contiguous().view(torch.uint8)
            w13_s_conv = w13_scale_orig.transpose(1, 2).contiguous()
            w2_s_conv = w2_scale_orig.transpose(1, 2).contiguous()

            # Reproduce the fixed routing used in Check 3
            M = 32
            torch.manual_seed(999)
            x = torch.randn(M, K, dtype=torch.float16, device=device)
            _, topk_out = make_fixed_dispatch(x, device)
            topk_weights, topk_ids, _ = topk_out

            # Compute reference output via manual dequantization
            ref_output = reference_moe_forward(
                x, w13_conv, w2_conv, w13_s_conv, w2_s_conv,
                topk_ids, topk_weights, GROUP_SIZE,
            )

            # Compare: quantized kernel vs dequantized reference
            fwd_f32 = fwd_output.float()
            ref_f32 = ref_output.float()

            # 4a: Cosine similarity — outputs should be highly correlated
            fwd_flat = fwd_f32.flatten()
            ref_flat = ref_f32.flatten()
            nonzero_mask = (ref_flat.abs() > 1e-8) | (fwd_flat.abs() > 1e-8)
            if nonzero_mask.sum() > 0:
                fwd_nz = fwd_flat[nonzero_mask]
                ref_nz = ref_flat[nonzero_mask]
                cosine_sim = torch.nn.functional.cosine_similarity(
                    fwd_nz.unsqueeze(0), ref_nz.unsqueeze(0)
                ).item()
            else:
                cosine_sim = 0.0

            check(f"Cosine similarity with reference ({cosine_sim:.4f} >= 0.90)",
                  cosine_sim >= 0.90, 10,
                  f"cosine_sim={cosine_sim:.4f}, outputs diverge from reference")

            # 4a-2: Relative L2 error — catches scale-invariant gaming
            # (e.g., returning 2*correct_output would get cosine=1.0 but fail here)
            ref_norm = ref_f32.norm()
            if ref_norm > 1e-6:
                rel_l2 = (fwd_f32 - ref_f32).norm() / ref_norm
                rel_l2_val = rel_l2.item()
            else:
                rel_l2_val = float("inf")

            check(f"Relative L2 error ({rel_l2_val:.4f} <= 0.50)",
                  rel_l2_val <= 0.50, 5,
                  f"rel_l2={rel_l2_val:.4f}, magnitude diverges from reference")

            # 4b: Output is input-dependent (not a constant function)
            torch.manual_seed(777)
            x2 = torch.randn(M, K, dtype=torch.float16, device=device)
            dispatch_out2, _ = make_fixed_dispatch(x2, device)
            combine2 = method.apply(layer, dispatch_out2)
            out2 = combine2.hidden_states

            differs = not torch.allclose(fwd_output, out2, rtol=1e-3, atol=1e-3)
            check("Output is input-dependent", differs, 5,
                  "Same output for different inputs — ignoring input data")

            # 4c: Different batch size works and matches reference
            M2 = 16
            torch.manual_seed(888)
            x3 = torch.randn(M2, K, dtype=torch.float16, device=device)
            dispatch_out3, topk_out3 = make_fixed_dispatch(x3, device)
            combine3 = method.apply(layer, dispatch_out3)
            out3 = combine3.hidden_states

            topk_w3, topk_i3, _ = topk_out3
            ref3 = reference_moe_forward(
                x3, w13_conv, w2_conv, w13_s_conv, w2_s_conv,
                topk_i3, topk_w3, GROUP_SIZE,
            )
            cos3 = torch.nn.functional.cosine_similarity(
                out3.float().flatten().unsqueeze(0),
                ref3.float().flatten().unsqueeze(0),
            ).item()
            ref3_norm = ref3.float().norm()
            rel3 = ((out3.float() - ref3.float()).norm() / ref3_norm).item() if ref3_norm > 1e-6 else float("inf")
            ok3 = cos3 >= 0.90 and rel3 <= 0.50
            check(f"M={M2} reference match (cos={cos3:.4f}, relL2={rel3:.4f})",
                  ok3, 5,
                  f"cosine={cos3:.4f}, rel_l2={rel3:.4f}")

    except Exception as e:
        check("Correctness oracle", False, 25, str(e))
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    total = sum(scores.values())
    print()
    print("=" * 60)
    c1 = sum(v for k, v in scores.items()
             if k.startswith(("get_moe", "Method")))
    c2 = sum(v for k, v in scores.items()
             if k.startswith(("Weights", "Weight", "Scales")))
    c3_keys = ["method.apply() completes",
               "Deterministic (fixed routing -> same output)",
               "Output has no NaN/Inf"]
    c3_keys += [k for k in scores if k.startswith("Output shape")]
    c3 = sum(scores.get(k, 0) for k in c3_keys)
    c4_keys = [k for k in scores
               if k.startswith(("Cosine", "Relative L2", "Output is input", "M="))]
    c4 = sum(scores.get(k, 0) for k in c4_keys)
    print(f"  Check 1 (Dispatch):    {c1:>3} / 25")
    print(f"  Check 2 (Weights):     {c2:>3} / 25")
    print(f"  Check 3 (Forward):     {c3:>3} / 25")
    print(f"  Check 4 (Correct.):    {c4:>3} / 25")
    print(f"  {'=' * 30}")
    print(f"  Total:                 {total:>3} / 100")
    print()
    print(f"SCORE: {total:.1f}")
    sys.exit(0 if total == 100 else 1)


if __name__ == "__main__":
    main()
