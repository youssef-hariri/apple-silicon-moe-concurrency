#!/usr/bin/env python3
"""P2 C-103 sweep: LRU vs static pinning across capacity S (slots/layer).
3 prompts × 150 tokens, warm start (1 warmup prompt), OLMoE-0125.
Frozen to results_P2/lru_capacity_sweep.json"""
import sys, os, json, ast, statistics, time
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))

import mlx.core as mx
from pinned_arm import PinnedOlmoe, generate_greedy
from lru_arm import make_lru_model
from mlx_lm.models.cache import KVCache

REF = ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit"
SPLIT = ROOT / "models" / "olmoe-0125-split-4bit"
OUT = HERE / "results_P2" / "lru_capacity_sweep.json"
OUT.parent.mkdir(exist_ok=True)

MAX_NEW = 150
S_SWEEP = [16, 32, 48, 64]

tree = ast.parse((ROOT / "code" / "config.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "TEST_PROMPTS":
                PROMPTS = ast.literal_eval(node.value)
PROMPTS = PROMPTS[:3]

import mlx_lm
tok = mlx_lm.load(str(REF))[1]
ids_all = [tok.encode(p) for p in PROMPTS]
eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
report = {"model": "OLMoE-1B-7B-0125-Instruct-4bit", "max_new": MAX_NEW,
          "sweep": S_SWEEP, "note": "warm start: 1 warmup prompt untimed"}

def lru_decode(m, ids):
    cache = [KVCache() for _ in range(m.n_layers)]
    logits = m(mx.array([ids]), cache)
    t = mx.argmax(logits[:, -1], axis=-1).item()
    seq = [t]
    mx.synchronize()
    t1 = time.perf_counter()
    for _ in range(MAX_NEW - 1):
        logits = m(mx.array([[t]]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        seq.append(t)
        if t in eos_ids:
            break
    mx.synchronize()
    return seq, (len(seq) - 1) / (time.perf_counter() - t1)

def static_arm(S):
    arm = PinnedOlmoe(REF, SPLIT / "experts", cold_capacity=256)
    for ids in ids_all:
        generate_greedy(arm, ids, max_new=MAX_NEW, eos_ids=eos_ids)
    per_layer = {l: {} for l in range(len(arm.model.layers))}
    for (l, e), c in arm.pool.usage.items():
        per_layer[l][e] = c
    pin_spec = {l: [e for e, _ in sorted(u.items(), key=lambda kv: (-kv[1], kv[0]))[:S]]
                for l, u in per_layer.items() if u}
    del arm
    mx.clear_cache()
    arm = PinnedOlmoe(REF, SPLIT / "experts", cold_capacity=256)
    arm.pin_experts(pin_spec)
    tgs = []
    for ids in ids_all:
        r = generate_greedy(arm, ids, max_new=MAX_NEW, eos_ids=eos_ids)
        tgs.append(r["tg_tok_s"])
    st = {kk: arm.pool.stats[kk] for kk in ("cold_hits", "loads", "stack_hits")}
    del arm
    mx.clear_cache()
    return statistics.median(tgs), st

for S in S_SWEEP:
    # LRU
    m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=S)
    lru_decode(m, ids_all[0])  # warmup, untimed
    tgs = [lru_decode(m, ids)[1] for ids in ids_all]
    st_l = m.stats()
    del m
    mx.clear_cache()
    # static
    tg_s, st_s = static_arm(S)
    report[f"S{S}"] = {
        "lru": {"tg_median": statistics.median(tgs), "tg_per_prompt": tgs,
                "hit_rate": st_l["hits"] / max(1, st_l["hits"] + st_l["misses"]),
                "stats": st_l},
        "static": {"tg_median": tg_s, "stats": st_s,
                   "hit_rate": st_s["stack_hits"] / max(1, st_s["stack_hits"] + st_s["cold_hits"])},
    }
    print(f"S={S}: LRU {statistics.median(tgs):.1f} tok/s (hit {st_l['hits']/(st_l['hits']+st_l['misses']):.2%}) "
          f"vs static {tg_s:.1f} tok/s (hit {st_s['stack_hits']/(st_s['stack_hits']+st_s['cold_hits']):.2%})",
          flush=True)

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("saved", OUT)
