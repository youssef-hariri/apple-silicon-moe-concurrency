#!/usr/bin/env python3
"""
LRU expert cache arm (P2, Phase 1, M1) — FreeToken-style dynamic expert
residency for the pinned-engine codebase.

Design contract (D2-001 exactness): the LRU cache changes WHICH experts are
resident, never the math. Every projection still runs through the validated
mx.gather_qmm path with the same accumulation rules as ResidentSwitch
(T2 bit-exact gate), so per-step outputs equal the static pinned arm up to
the documented near-tie class (P1 fidelity, D2-007).

Mechanism (FreeToken §3.2, adapted to unified memory):
- Per MoE layer, a fixed pool of S slots (same RAM budget as the static
  arm's pinned set). Slots hold stacked projection tensors.
- Each decode step: router picks top-k experts (native router, unchanged);
  hits refresh recency; misses are loaded from SSD slabs into slots,
  evicting the least-recently-used slot NOT needed by the current step.
- Slot tensors are updated in place (sticky slots, Edge0 incr_stack-style):
  only the slots that changed are rewritten, never a full rebuild.

v1 scope: serial slab reads on miss (same as static arm's cold path);
parallel prefetch and mmap tier are separate arms (Phase 1 items 4-5).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pinned_arm import (  # reuse validated pieces verbatim
    ResidentSwitch, _gather_sort, _scatter_unsort,
)

PROJS = ("up_proj", "gate_proj", "down_proj")


class PrefillStreamer:
    """Double-buffered prefill (FreeToken §3.1, adapted): while layer l
    computes, worker threads read layer l+1's expert slabs off the SSD.
    Workers do pure file I/O (thread-safe, no MLX); the main thread converts
    bytes -> mx arrays zero-copy when the layer actually needs them.

    FreeToken loads the FULL next layer because prefill routing covers
    nearly all experts; we stage all experts that are not already resident.
    Exactness: staging changes WHEN bytes are read, never WHAT is computed.
    """

    def __init__(self, experts_dir: Path, workers: int = 8):
        from concurrent.futures import ThreadPoolExecutor
        self.dir = Path(experts_dir)
        self.ex = ThreadPoolExecutor(max_workers=workers)
        self.staged: dict[tuple[int, int], tuple] = {}  # (l,e) -> (buf, header, start)
        self.futs: dict[tuple[int, int], object] = {}
        self._lock = __import__("threading").Lock()
        self.stats = {"staged": 0, "consumed": 0}

    @staticmethod
    def _path(d: Path, l: int, e: int) -> Path:
        p = d / f"layer{l:02d}_expert{e:03d}.safetensors"
        return p if p.exists() else d / f"layer{l:02d}_expert{e:02d}.safetensors"

    @staticmethod
    def _read(path: Path):
        import json as _json, struct
        with open(str(path), "rb") as fh:
            buf = fh.read()
        hlen = struct.unpack("<Q", buf[:8])[0]
        header = _json.loads(buf[8:8 + hlen])
        return buf, header, 8 + hlen

    def submit_layer(self, layer_id: int, experts: list[int],
                     max_in_flight: int = 16):
        """Stage reads ahead; cap in-flight count so big models (235B:
        9.4 MB/expert) cannot pile whole layers into RAM (D2-022)."""
        with self._lock:
            if len(self.futs) >= max_in_flight:
                return
        for e in experts:
            key = (layer_id, e)
            with self._lock:
                if key in self.staged or key in self.futs:
                    continue
                if len(self.futs) >= max_in_flight:
                    break
                self.futs[key] = self.ex.submit(self._read, self._path(self.dir, layer_id, e))

    def fetch(self, layer_id: int, expert: int):
        """Return staged (buf, header, start) or None. Blocks only if the
        read was already submitted and is still in flight."""
        key = (layer_id, expert)
        with self._lock:
            fut = self.futs.pop(key, None)
        if fut is not None:
            try:
                data = fut.result(timeout=30)
                self.stats["staged"] += 1
                return data
            except Exception:
                return None
        return None

    def note_consumed(self):
        self.stats["consumed"] += 1


# safetensors dtype -> (numpy view dtype, mlx dtype) — same zero-copy rule
# as pinned_arm.ExpertPool._ST_DTYPE (byte-identical, no arithmetic).
import numpy as _np
_ST_DTYPE = {
    "BF16": (_np.uint16, mx.bfloat16), "F16": (_np.float16, mx.float16),
    "F32": (_np.float32, mx.float32), "F64": (_np.float64, mx.float64),
    "U32": (_np.uint32, mx.uint32), "I32": (_np.int32, mx.int32),
    "I64": (_np.int64, mx.int64), "U8": (_np.uint8, mx.uint8),
    "I8": (_np.int8, mx.int8), "BOOL": (_np.uint8, mx.bool_),
}


def staged_to_mx(buf, header, data_start):
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        npdt, mxdt = _ST_DTYPE[info["dtype"]]
        shape = info["shape"]
        s, e = info["data_offsets"]
        cnt = 1
        for d in shape:
            cnt *= d
        arr = _np.frombuffer(buf, dtype=npdt, count=cnt,
                             offset=data_start + s).reshape(shape)
        a = mx.array(arr)
        if a.dtype != mxdt:
            a = a.view(mxdt)
        out[name] = a
    return out


class LRUSlotPool:
    """Per-layer pool of S resident expert slots with LRU eviction.

    The stacked tensors have leading dim S; slot i holds one expert's
    (weight, scales, biases) per projection. lut maps global expert id ->
    slot or -1. recency is an OrderedDict slot -> last-used step counter.
    """

    def __init__(self, experts_dir: Path, layer_id: int, n_slots: int,
                 num_experts: int, group: int = 64,
                 streamer: "PrefillStreamer | None" = None):
        self.dir = Path(experts_dir)
        self.layer_id = layer_id
        self.S = n_slots
        self.num_experts = num_experts
        self.group = group
        self.streamer = streamer
        self.resident = ResidentSwitch()
        self.lut_host: list[int] = [-1] * num_experts
        self.slot_of: dict[int, int] = {}          # expert -> slot
        self.expert_at: list[int] = [-1] * n_slots  # slot -> expert
        self.protected: set[int] = set()            # pinned slots (hybrid arm)
        self.recency: OrderedDict[int, int] = OrderedDict()  # slot -> tick
        self.tick = 0
        self._stacked = False
        self.stats = {"hits": 0, "misses": 0, "evictions": 0,
                      "slot_writes": 0, "load_seconds": 0.0, "load_bytes": 0}

    # ---- slot stack management --------------------------------------------

    def _init_stack(self, slab):
        """Create zero-filled stacked tensors, each projection from its OWN
        slab tensors (up/gate are (1024,256), down is (2048,128) on OLMoE)."""
        for proj in PROJS:
            p = getattr(self.resident, proj)
            w, s, b = (slab[f"{proj}.weight"], slab[f"{proj}.scales"],
                       slab[f"{proj}.biases"])
            p.weight = mx.zeros((self.S,) + tuple(w.shape), dtype=w.dtype)
            p.scales = mx.zeros((self.S,) + tuple(s.shape), dtype=s.dtype)
            p.biases = mx.zeros((self.S,) + tuple(b.shape), dtype=b.dtype)
            p.group_size = self.group
        mx.eval(*[t for proj in PROJS for t in
                  (getattr(self.resident, proj).weight,
                   getattr(self.resident, proj).scales,
                   getattr(self.resident, proj).biases)])
        self._stacked = True

    def _load_slab(self, expert: int):
        """Staged-bytes first (prefill double buffer), disk read otherwise.
        Byte-identical either way (staged_to_mx is the same zero-copy rule
        as the P1 prefetch path)."""
        if self.streamer is not None:
            data = self.streamer.fetch(self.layer_id, expert)
            if data is not None:
                self.streamer.note_consumed()
                self.stats["staged_hits"] = self.stats.get("staged_hits", 0) + 1
                return staged_to_mx(*data)
        path = self.dir / f"layer{self.layer_id:02d}_expert{expert:03d}.safetensors"
        if not path.exists():
            path = self.dir / f"layer{self.layer_id:02d}_expert{expert:02d}.safetensors"
        slab = mx.load(str(path))
        self.stats["load_bytes"] += path.stat().st_size
        return slab

    def _write_slot(self, slot: int, expert: int):
        slab = self._load_slab(expert)
        if not self._stacked:
            self._init_stack(slab)
        for proj in PROJS:
            p = getattr(self.resident, proj)
            p.weight[slot] = slab[f"{proj}.weight"]
            p.scales[slot] = slab[f"{proj}.scales"]
            p.biases[slot] = slab[f"{proj}.biases"]
        mx.eval(*[t for proj in PROJS for t in
                  (getattr(self.resident, proj).weight,
                   getattr(self.resident, proj).scales,
                   getattr(self.resident, proj).biases)])
        old = self.expert_at[slot]
        if old >= 0:
            self.lut_host[old] = -1
            self.slot_of.pop(old, None)
        self.expert_at[slot] = expert
        self.slot_of[expert] = slot
        self.lut_host[expert] = slot
        self.stats["slot_writes"] += 1

    def reset_residency(self):
        """Clear all residency bookkeeping (keeps allocated stacks). Used by
        the hybrid arm between calibration (which fills the cache with
        warmup traffic) and pinning the calibrated hot set."""
        self.slot_of.clear()
        self.expert_at = [-1] * self.S
        self.protected.clear()
        self.recency.clear()
        self.lut_host = [-1] * self.num_experts

    def pin_expert(self, expert: int):
        """HYBRID arm: load expert into a reserved slot that eviction never
        touches. The pinned hot set (static part) is chosen by calibration;
        remaining slots stay dynamic (LRU part)."""
        if expert in self.slot_of:
            return
        free = [s for s in range(self.S) if self.expert_at[s] < 0]
        if not free:
            raise RuntimeError("pin_expert: no free slots (pin set too large)")
        slot = free[0]
        self._write_slot(slot, expert)
        self.protected.add(slot)
        self.recency[slot] = self.tick
        self.stats["pinned"] = self.stats.get("pinned", 0) + 1

    def _victim(self, needed: set[int]) -> int:
        """Least-recently-used UNPROTECTED slot whose expert is not needed
        this step."""
        for slot in self.recency:  # OrderedDict: oldest first
            if slot in self.protected:
                continue
            if self.expert_at[slot] not in needed:
                return slot
        # fallback (should not happen when S > k): oldest unprotected
        for slot in self.recency:
            if slot not in self.protected:
                return slot
        return next(iter(self.recency))

    def ensure(self, experts: list[int]) -> list[int]:
        """Make `experts` resident where capacity allows; return the list of
        experts that could NOT be cached this step (unique > S in prefill),
        to be served by the uncached exact fallback path."""
        self.tick += 1
        needed = set(experts)
        overflow: list[int] = []
        overflow_set: set[int] = set()
        for e in experts:
            slot = self.slot_of.get(e)
            if slot is not None:
                self.stats["hits"] += 1
                self.recency.move_to_end(slot)
                continue
            self.stats["misses"] += 1
            free = [s for s in range(self.S) if self.expert_at[s] < 0]
            if free:
                victim = free[0]
            else:
                victim = self._victim(needed)
                if self.expert_at[victim] in needed:
                    # every slot holds an expert needed this step: capacity
                    # exhausted -> serve e uncached (exact fallback), do NOT
                    # thrash a slot we need moments later.
                    overflow.append(e)
                    overflow_set.add(e)
                    self.stats["overflow"] = self.stats.get("overflow", 0) + 1
                    continue
                self.stats["evictions"] += 1
            self.recency.pop(victim, None)
            t0 = time.perf_counter()
            self._write_slot(victim, e)
            self.stats["load_seconds"] += time.perf_counter() - t0
            self.recency[victim] = self.tick
        return overflow


class LRUSwitchMLP(nn.Module):
    """Drop-in replacement for stock switch_mlp using a dynamic LRU slot pool
    instead of a statically pinned resident stack."""

    def __init__(self, experts_dir: Path, layer_id: int, hidden: int,
                 inter: int, num_experts: int, group: int, n_slots: int,
                 streamer: "PrefillStreamer | None" = None,
                 next_moe_layer: "int | None" = None):
        super().__init__()
        self.pool = LRUSlotPool(experts_dir, layer_id, n_slots,
                                num_experts, group, streamer=streamer)
        self.streamer = streamer
        self.next_moe_layer = next_moe_layer
        self.num_experts = num_experts
        self.hidden = hidden
        self.inter = inter

    def __call__(self, x, indices):
        if x.ndim != 2:
            lead = indices.shape[:-1]
            y = self.__call__(x.reshape(-1, x.shape[-1]),
                              indices.reshape(-1, indices.shape[-1]))
            return y.reshape(*lead, y.shape[-2], y.shape[-1])
        T, k = indices.shape
        idx_h = indices.tolist()  # same single sync as static partial path
        rec = getattr(self, "router_recorder", None)
        if rec is not None:
            rec(self.pool.layer_id, idx_h)

        # Double-buffered prefill: while THIS layer ensures + computes, start
        # reading the NEXT MoE layer's non-resident experts off the SSD.
        # Prefill covers nearly all experts (FreeToken §2.1), so routing of
        # the next layer does not need to be known.
        if T > 1 and self.streamer is not None and self.next_moe_layer is not None:
            nxt_pool = self._peer_pools.get(self.next_moe_layer) if hasattr(self, "_peer_pools") else None
            resident_next = set(nxt_pool.slot_of) if nxt_pool is not None else set()
            todo = [e for e in range(self.num_experts) if e not in resident_next]
            self.streamer.submit_layer(self.next_moe_layer, todo)

        uniq = sorted({e for row in idx_h for e in row})
        overflow = set(self.pool.ensure(uniq))
        slot_rows = [[self.pool.lut_host[e] for e in row] for row in idx_h]
        cached_mask = [[e not in overflow for e in row] for row in idx_h]

        out = mx.zeros((T * k, self.hidden), dtype=x.dtype)

        # Cached experts: sticky-slot stack gather (validated kernel path)
        if any(any(r) for r in cached_mask):
            hot = mx.array([[s if c else 0 for s, c in zip(sr, cr)]
                            for sr, cr in zip(slot_rows, cached_mask)],
                           dtype=mx.int32)
            sel = mx.array(cached_mask, dtype=x.dtype).reshape(-1)
            y = self.pool.resident(x, hot).reshape(T * k, self.hidden)
            out = out + y * sel[:, None]

        # Overflow experts (prefill working set > slot capacity): exact
        # per-expert gather_qmm, same D-031 cold math as the static arm.
        # NOT cached — thrashing a slot to serve it would corrupt the step.
        if overflow:
            from pinned_arm import Expert
            groups: dict[int, list[tuple[int, int]]] = {}
            for t in range(T):
                for kk in range(k):
                    e = idx_h[t][kk]
                    if e in overflow:
                        groups.setdefault(e, []).append((t, kk))
            for e, pos in groups.items():
                slab = self.pool._load_slab(e)
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
                        bits=proj.bits,
                        mode=getattr(proj, "mode", "affine"),
                        sorted_indices=(a.shape[0] >= 64)).reshape(a.shape[0], -1)
                y = _qmm(ex.down_proj,
                         nn.silu(_qmm(ex.gate_proj, xr)) * _qmm(ex.up_proj, xr))
                out = out.at[slots].add(y)
        return out.reshape(T, k, self.hidden)


def make_lru_model(ref_dir, experts_dir, n_slots_per_layer: int,
                   cold_capacity: int = 1, memory_limit_gb: float = 6.0,
                   prefill_stream: bool = False, stream_workers: int = 8,
                   lazy: bool = False):
    """Build the stock mlx_lm model with every MoE switch_mlp replaced by an
    LRU-cached block. Same backbone-verbatim rule as PinnedOlmoe.
    prefill_stream=True enables the double-buffered prefill streamer (M2).
    lazy=True loads weights lazily (REQUIRED for >RAM models like the 235B:
    expert tensors are never materialized because the LRU blocks replace
    switch_mlp before the first forward)."""
    import json as _json
    import mlx_lm
    from pinned_arm import moe_submodule, assert_attention_compatible
    mx.set_memory_limit(int(memory_limit_gb * 1e9))
    model, tokenizer = mlx_lm.load(str(ref_dir), lazy=lazy)
    assert_attention_compatible(model)
    cfg = model.args
    num_experts = (getattr(cfg, "num_experts", None)
                   or getattr(cfg, "n_routed_experts", None)
                   or getattr(cfg, "num_local_experts", None))
    moe_inter = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
    group = 64
    try:
        cfgj = _json.loads((Path(ref_dir) / "config.json").read_text())
        q = cfgj.get("quantization") or cfgj.get("quantization_config") or {}
        group = int(q.get("group_size", 64))
    except Exception:
        pass
    # which layers are MoE (needed for next-layer mapping before construction)
    moe_ids = [l for l, blk in enumerate(model.layers)
               if moe_submodule(blk) is not None]
    next_of = {l: (moe_ids[i + 1] if i + 1 < len(moe_ids) else None)
               for i, l in enumerate(moe_ids)}
    streamer = PrefillStreamer(Path(experts_dir),
                               workers=stream_workers) if prefill_stream else None
    layer_ids = []
    blocks = {}
    for l in moe_ids:
        sub = moe_submodule(model.layers[l])
        blk = LRUSwitchMLP(Path(experts_dir), l, cfg.hidden_size,
                           moe_inter, num_experts, group,
                           n_slots_per_layer,
                           streamer=streamer,
                           next_moe_layer=next_of[l])
        sub.switch_mlp = blk
        blocks[l] = blk
        layer_ids.append(l)
    # peer pool registry so a layer can see what the next layer already holds
    peers = {l: blk.pool for l, blk in blocks.items()}
    for blk in blocks.values():
        blk._peer_pools = peers

    class LRUModel:
        def __init__(self):
            self.model = model
            self.tokenizer = tokenizer
            self.n_layers = len(model.layers)
            self.num_experts = num_experts
            self.moe_layer_ids = layer_ids

        def __call__(self, inputs, cache):
            return self.model(inputs, cache)

        def pools(self):
            return [moe_submodule(self.model.layers[l]).switch_mlp.pool
                    for l in self.moe_layer_ids]

        def stats(self):
            agg = {"hits": 0, "misses": 0, "evictions": 0, "slot_writes": 0,
                   "load_seconds": 0.0, "load_bytes": 0}
            for p in self.pools():
                for k_ in agg:
                    agg[k_] += p.stats[k_]
            return agg

    return LRUModel()


def make_hybrid_model(ref_dir, experts_dir, n_slots_per_layer: int,
                      hot_per_layer: int, calib_prompt_ids: list,
                      memory_limit_gb: float = 11.0, prefill_stream: bool = False):
    """HYBRID arm (D2-016 follow-up): permanently pin the traffic-ranked hot
    set (`hot_per_layer` experts per layer, chosen by a calibration pass over
    calib_prompt_ids) + LRU-manage the remaining slots. Synthesis of P1
    (static pinning survives concurrency) and FreeToken (LRU wins single-user
    streaming). Exactness: same gather path; pinning changes residency only."""
    from collections import Counter
    m = make_lru_model(ref_dir, experts_dir, n_slots_per_layer,
                       memory_limit_gb=memory_limit_gb,
                       prefill_stream=prefill_stream)
    from mlx_lm.models.cache import KVCache
    from pinned_arm import moe_submodule

    # calibration: record routing selections per layer over the warmup prompts
    counts: dict[int, Counter] = {l: Counter() for l in m.moe_layer_ids}
    def mk_rec(l):
        def rec(layer_id, rows):
            for row in rows:
                counts[l].update(row)
        return rec
    for l in m.moe_layer_ids:
        moe_submodule(m.model.layers[l]).switch_mlp.router_recorder = mk_rec(l)
    for ids in calib_prompt_ids:
        cache = [KVCache() for _ in range(m.n_layers)]
        logits = m(mx.array([ids]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        for _ in range(30):
            logits = m(mx.array([[t]]), cache)
            t = mx.argmax(logits[:, -1], axis=-1).item()
    for l in m.moe_layer_ids:
        moe_submodule(m.model.layers[l]).switch_mlp.router_recorder = None

    # pin the hot set into protected slots (after clearing calibration traffic)
    for l in m.moe_layer_ids:
        blk = moe_submodule(m.model.layers[l]).switch_mlp
        blk.pool.reset_residency()
        for e, _ in counts[l].most_common(hot_per_layer):
            blk.pool.pin_expert(e)
    return m
