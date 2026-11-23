#!/usr/bin/env python3
"""
mmap kernel-tier arm (P2, Phase 1, N2) — "Who should own the expert cache?"
(arXiv:2608.12103) tested on Apple unified memory.

Design: NO explicit expert cache. Each MoE layer reads the routed experts'
slabs via MLX's mmap-backed lazy loader per step; the kernel page cache
decides what stays hot. This is also llama.cpp's implicit strategy
(mmap'd GGUF) and the OD-MoE "cacheless" point — so one arm stands in for
three literature positions.

Exactness: same D-031 gather_qmm math per expert; page cache changes WHEN
bytes are present, never WHAT is computed.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pinned_arm import Expert, moe_submodule, assert_attention_compatible

PROJS = ("up_proj", "gate_proj", "down_proj")


class MmapSwitchMLP(nn.Module):
    """Cacheless MoE block: per-step mmap reads, kernel-managed residency."""

    def __init__(self, experts_dir: Path, layer_id: int, hidden: int,
                 inter: int, num_experts: int):
        super().__init__()
        self.dir = Path(experts_dir)
        self.layer_id = layer_id
        self.hidden, self.inter = hidden, inter
        self.num_experts = num_experts
        self.stats = {"reads": 0, "read_seconds": 0.0}

    def _path(self, e: int) -> Path:
        p = self.dir / f"layer{self.layer_id:02d}_expert{e:03d}.safetensors"
        return p if p.exists() else self.dir / f"layer{self.layer_id:02d}_expert{e:02d}.safetensors"

    def __call__(self, x, indices):
        if x.ndim != 2:
            lead = indices.shape[:-1]
            y = self.__call__(x.reshape(-1, x.shape[-1]),
                              indices.reshape(-1, indices.shape[-1]))
            return y.reshape(*lead, y.shape[-2], y.shape[-1])
        T, k = indices.shape
        idx_h = indices.tolist()
        groups: dict[int, list[tuple[int, int]]] = {}
        for t in range(T):
            for kk in range(k):
                groups.setdefault(idx_h[t][kk], []).append((t, kk))

        out = mx.zeros((T * k, self.hidden), dtype=x.dtype)
        for e, pos in groups.items():
            t0 = time.perf_counter()
            slab = mx.load(str(self._path(e)))   # mmap-backed, lazy
            self.stats["reads"] += 1
            self.stats["read_seconds"] += time.perf_counter() - t0
            ex = Expert.from_slab(slab, self.hidden, self.inter)
            rows = mx.array([p[0] for p in pos])
            slots = mx.array([p[0] * k + p[1] for p in pos])
            xr = mx.take(x, rows, axis=0)

            def _qmm(proj, a):
                return mx.gather_qmm(
                    mx.expand_dims(a, (-2, -3)),
                    mx.expand_dims(proj.weight, 0),
                    mx.expand_dims(proj.scales, 0),
                    mx.expand_dims(proj.biases, 0),
                    rhs_indices=mx.zeros((a.shape[0], 1), dtype=mx.int32),
                    transpose=True, group_size=proj.group_size,
                    bits=proj.bits, mode=getattr(proj, "mode", "affine"),
                    sorted_indices=(a.shape[0] >= 64)).reshape(a.shape[0], -1)
            y = _qmm(ex.down_proj,
                     nn.silu(_qmm(ex.gate_proj, xr)) * _qmm(ex.up_proj, xr))
            out = out.at[slots].add(y)
        return out.reshape(T, k, self.hidden)


def make_mmap_model(ref_dir, experts_dir, memory_limit_gb: float = 6.0):
    import json as _json
    import mlx_lm
    mx.set_memory_limit(int(memory_limit_gb * 1e9))
    model, tokenizer = mlx_lm.load(str(ref_dir))
    assert_attention_compatible(model)
    cfg = model.args
    num_experts = (getattr(cfg, "num_experts", None)
                   or getattr(cfg, "n_routed_experts", None)
                   or getattr(cfg, "num_local_experts", None))
    moe_inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    layer_ids = []
    for l, blk in enumerate(model.layers):
        sub = moe_submodule(blk)
        if sub is None:
            continue
        sub.switch_mlp = MmapSwitchMLP(Path(experts_dir), l, cfg.hidden_size,
                                       moe_inter, num_experts)
        layer_ids.append(l)

    class MmapModel:
        def __init__(self):
            self.model = model
            self.tokenizer = tokenizer
            self.n_layers = len(model.layers)
            self.num_experts = num_experts
            self.moe_layer_ids = layer_ids

        def __call__(self, inputs, cache):
            return self.model(inputs, cache)

        def stats(self):
            agg = {"reads": 0, "read_seconds": 0.0}
            for l in self.moe_layer_ids:
                s = moe_submodule(self.model.layers[l]).switch_mlp.stats
                for kk in agg:
                    agg[kk] += s[kk]
            return agg

    return MmapModel()
