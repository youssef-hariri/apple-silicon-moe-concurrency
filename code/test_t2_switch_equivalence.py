#!/usr/bin/env python3
"""
T2 (Qwen3-30B-A3B) switch_mlp equivalence gate — $0 synthetic numeric check.

Purpose (pre-rental, paper-grade): prove that OUR pinning machinery
(`PinnedSwitchMLP` fully-pinned fast path → `ResidentSwitch`) reproduces the
STOCK mlx_lm `SwitchGLU` bit-for-bit at Qwen3's *exact* expert dimensions,
before we spend money on the 24 GB rental. This is the same style of strict
numeric test that caught the rope-position bug on T1.

Method
------
Generate ONE set of synthetic 4-bit affine-quantized expert weights (per
projection) and feed the *identical* tensors + identical token activations +
identical router-selected expert indices through both arms:

  (A) stock reference : mlx_lm SwitchGLU whose three SwitchLinear projections
                        are replaced by QuantizedSwitchLinear holding our
                        exact weight/scales/biases. Called exactly as stock
                        (__call__ incl. _gather_sort/_scatter_unsort and
                        sorted_indices=do_sort).
  (B) our pinned arm  : PinnedSwitchMLP.build_resident_from_slabs(...) then the
                        fully-pinned fast path (self._P == num_experts).

Both arms get expert ids that ARE the resident slots (LUT is identity because
we pin all E experts), so slot_idx == indices and the only variable is the
kernel/stack plumbing.

Dimensions are Qwen3-30B-A3B verbatim (fetched from HF config 2026-09-23):
  hidden_size=2048, moe_intermediate_size=768, num_experts=128, top_k=8.
`moe_intermediate_size != intermediate_size` is the exact Qwen3-vs-OLMoE
parameterization difference this test targets.

Regimes covered (the whole point of a strict numeric test):
  decode  : T*top_k <  64  -> SwitchGLU do_sort == False
  prefill : T*top_k >= 64  -> do_sort == True  (exercises the sorted_indices
                            asymmetry between our _StackProj and stock).

Exit code 0 only if EVERY case is bit-identical. Also reports max |diff| so a
failure is diagnosable, not just a red light.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Make the pinned-arm module importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinned_arm import PinnedSwitchMLP, ExpertPool  # noqa: E402

from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear  # noqa: E402


# ── T2 Qwen3-30B-A3B exact dimensions ──────────────────────────────────────
H = 2048            # hidden_size
I = 768             # moe_intermediate_size  (expert FFN width; != intermediate 6144)
E = 128             # num_experts
TOPK = 8            # num_experts_per_tok
G = 64              # group_size (both arms share it; only internal to gather_qmm)
BITS = 4


def quantize_experts(out_f: int, in_f: int) -> tuple:
    """Deterministic synthetic 4-bit affine weights for E experts.

    Uses QuantizedSwitchLinear.__init__'s init scale sqrt(1/in_f) so the
    magnitudes are model-realistic. The exact values do not matter — both arms
    consume the SAME returned (weight, scales, biases) tensors, so this is a
    pure equivalence test of the plumbing, not of the weights."""
    scale = (1.0 / in_f) ** 0.5
    fp = mx.random.uniform(low=-scale, high=scale, shape=(E, out_f, in_f))
    mx.eval(fp)
    out = mx.quantize(fp, group_size=G, bits=BITS, mode="affine")
    w, s = out[0], out[1]
    biases = out[2] if len(out) > 2 else None
    mx.eval(w, s, *( [biases] if biases is not None else [] ))
    return w, s, biases  # (E,out,in/8) uint32, (E,out,in/G), (E,out,in/G)


def build_stock_reference(quant: dict) -> SwitchGLU:
    """Stock SwitchGLU with quantized projections carrying OUR exact tensors."""
    stock = SwitchGLU(input_dims=H, hidden_dims=I, num_experts=E, bias=False)
    for name, out_f, in_f in (("up_proj", I, H),
                              ("gate_proj", I, H),
                              ("down_proj", H, I)):
        q = QuantizedSwitchLinear(in_f, out_f, E, bias=False,
                                  group_size=G, bits=BITS, mode="affine")
        w, s, bi = quant[name]
        q.weight = w
        q.scales = s
        if bi is not None:
            q.biases = bi
        mx.eval(q.weight, q.scales, q.biases)
        setattr(stock, name, q)
    return stock


def build_pinned_reference(quant: dict) -> PinnedSwitchMLP:
    """Our arm: fully-pinned PinnedSwitchMLP assembled directly from slabs."""
    pool = ExpertPool(Path("/nonexistent"), H, I, cold_capacity=1)  # unused hot path
    mlp = PinnedSwitchMLP(pool=pool, layer_id=0, hidden=H, inter=I,
                          num_experts=E, group=G)
    slab_data = {}
    for e in range(E):
        slab_data[e] = {
            "up_proj.weight": quant["up_proj"][0][e],
            "up_proj.scales": quant["up_proj"][1][e],
            "up_proj.biases": quant["up_proj"][2][e],
            "gate_proj.weight": quant["gate_proj"][0][e],
            "gate_proj.scales": quant["gate_proj"][1][e],
            "gate_proj.biases": quant["gate_proj"][2][e],
            "down_proj.weight": quant["down_proj"][0][e],
            "down_proj.scales": quant["down_proj"][1][e],
            "down_proj.biases": quant["down_proj"][2][e],
        }
    mlp.build_resident_from_slabs(slab_data)
    assert mlp._P == mlp.num_experts, "fully-pinned fast path not engaged"
    return mlp


def run_case(name: str, T: int, stock: SwitchGLU, ours: PinnedSwitchMLP) -> bool:
    mx.random.seed(1234 + T)  # identical x and identical indices for both arms
    x = mx.random.normal(shape=(T, H)).astype(mx.bfloat16)
    # per-token: TOPK distinct experts in [0,E)
    idx = mx.argsort(mx.random.uniform(shape=(T, E)), axis=-1)[:, :TOPK].astype(mx.int32)
    mx.eval(x, idx)

    do_sort = idx.size >= 64
    a = stock(x, idx)                 # stock SwitchGLU
    mx.eval(a)
    b = ours(x, idx)                  # our fully-pinned path
    mx.eval(b)

    assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
    equal = mx.array_equal(a, b).item()
    diff = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
    status = "BIT-EXACT" if equal else f"DIVERGENT (max|Δ|={diff:.3e})"
    print(f"  [{name:<16}] T={T:<4} k={TOPK} sort={str(do_sort):<5} "
          f"size={idx.size:<5} -> {status}")
    return bool(equal)


def main() -> int:
    print("=" * 72)
    print("T2 (Qwen3-30B-A3B) switch_mlp equivalence gate — $0 numeric test")
    print(f"dims: hidden={H} moe_intermediate={I} num_experts={E} "
          f"top_k={TOPK} group={G} bits={BITS}")
    print("=" * 72)

    mx.random.seed(7)
    quant = {
        "up_proj":   quantize_experts(I, H),
        "gate_proj": quantize_experts(I, H),
        "down_proj": quantize_experts(H, I),
    }

    stock = build_stock_reference(quant)
    ours = build_pinned_reference(quant)

    results = []
    # decode regime (T*top_k < 64) and prefill regime (>= 64)
    for T in (1, 4, 8, 16, 64):
        results.append(run_case(f"T{T}{'-decode' if T*TOPK < 64 else '-prefill'}",
                                T, stock, ours))

    print("-" * 72)
    if all(results):
        print(f"RESULT: PASS — {sum(results)}/{len(results)} cases bit-identical.")
        print("Pinned fast-path reproduces stock SwitchGLU exactly at T2 dims.")
        return 0
    else:
        print(f"RESULT: FAIL — {sum(results)}/{len(results)} bit-identical; "
              f"{results.count(False)} case(s) diverge (see DIVERGENT lines).")
        print("Do NOT proceed to rental until the switch_mlp path is bit-exact.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
