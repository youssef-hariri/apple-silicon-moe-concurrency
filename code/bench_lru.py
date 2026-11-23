#!/usr/bin/env python3
"""P2 bench C-103: static pinning vs LRU expert cache, equal RAM budget.
Single-user, OLMoE-0125, greedy. Reports decode tok/s + cache stats +
output equivalence spot-check vs stock. Frozen to results_P2/lru_vs_static.json"""
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
OUT = HERE / "results_P2" / "lru_vs_static.json"
OUT.parent.mkdir(exist_ok=True)

PIN_FRAC = 0.25          # 16 of 64 experts per layer, both arms
MAX_NEW = 200
WARMUP_PROMPTS = 2       # prompts used to warm LRU before timing (separate arm reported too)

tree = ast.parse((ROOT / "code" / "config.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "TEST_PROMPTS":
                PROMPTS = ast.literal_eval(node.value)
PROMPTS = PROMPTS[:5]

import mlx_lm
tok = mlx_lm.load(str(REF))[1]
ids_all = [tok.encode(p) for p in PROMPTS]
eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
report = {"model": "OLMoE-1B-7B-0125-Instruct-4bit", "pin_frac": PIN_FRAC,
          "slots_per_layer": 16, "max_new": MAX_NEW}

# ---------- stock reference (for equivalence + speed context) --------------
def stock_run():
    stock, _ = mlx_lm.load(str(REF))
    outs, tgs = [], []
    for ids in ids_all:
        cache = [KVCache() for _ in range(len(stock.layers))]
        x = mx.array([ids])
        t0 = time.perf_counter()
        logits = stock(x, cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        seq = [t]
        mx.synchronize()
        t1 = time.perf_counter()
        for _ in range(MAX_NEW - 1):
            logits = stock(mx.array([[t]]), cache)
            t = mx.argmax(logits[:, -1], axis=-1).item()
            seq.append(t)
            if t in eos_ids:
                break
        mx.synchronize()
        dt = time.perf_counter() - t1
        outs.append(tok.decode(seq, skip_special_tokens=True))
        tgs.append((len(seq) - 1) / dt)
    return outs, tgs

ref_texts, ref_tg = stock_run()
report["stock"] = {"tg_median": statistics.median(ref_tg)}
del_stock = None
mx.clear_cache()

# ---------- static pinned arm (25% traffic-ranked pins) ---------------------
arm = PinnedOlmoe(REF, SPLIT / "experts", cold_capacity=256)
# calibration pass to get usage ranking (same as Stage-0)
for ids in ids_all:
    generate_greedy(arm, ids, max_new=MAX_NEW, eos_ids=eos_ids)
n_layers = len(arm.model.layers)
per_layer = {l: {} for l in range(n_layers)}
for (l, e), c in arm.pool.usage.items():
    per_layer[l][e] = c
k = int(PIN_FRAC * arm.num_experts)
pin_spec = {l: [e for e, _ in sorted(u.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
            for l, u in per_layer.items() if u}
del arm
mx.clear_cache()

arm = PinnedOlmoe(REF, SPLIT / "experts", cold_capacity=256)
arm.pin_experts(pin_spec)
static_texts, static_tg = [], []
for ids in ids_all:
    r = generate_greedy(arm, ids, max_new=MAX_NEW, eos_ids=eos_ids)
    static_texts.append(tok.decode(r["tokens"], skip_special_tokens=True))
    static_tg.append(r["tg_tok_s"])
report["static_pin"] = {"tg_median": statistics.median(static_tg),
                        "tg_per_prompt": static_tg,
                        "stats": {kk: arm.pool.stats[kk] for kk in
                                  ("pin_hits", "cold_hits", "loads", "stack_hits")}}
del arm
mx.clear_cache()

# ---------- LRU arm (16 slots/layer), cold start ----------------------------
def lru_run(warm_prompts=0, label="lru"):
    m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=k)
    texts, tgs = [], []
    # optional warmup (not timed)
    for ids in ids_all[:warm_prompts]:
        cache = [KVCache() for _ in range(m.n_layers)]
        x = mx.array([ids])
        logits = m(x, cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        for _ in range(MAX_NEW - 1):
            logits = m(mx.array([[t]]), cache)
            t = mx.argmax(logits[:, -1], axis=-1).item()
            if t in eos_ids:
                break
    for ids in ids_all:
        cache = [KVCache() for _ in range(m.n_layers)]
        x = mx.array([ids])
        logits = m(x, cache)
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
        dt = time.perf_counter() - t1
        texts.append(tok.decode(seq, skip_special_tokens=True))
        tgs.append((len(seq) - 1) / dt)
    st = m.stats()
    del m
    mx.clear_cache()
    report[label] = {"tg_median": statistics.median(tgs),
                     "tg_per_prompt": tgs, "stats": st,
                     "hit_rate": st["hits"] / max(1, st["hits"] + st["misses"])}
    return texts

lru_cold_texts = lru_run(0, "lru_cold_start")
lru_warm_texts = lru_run(WARMUP_PROMPTS, "lru_warm")

# ---------- equivalence ------------------------------------------------------
def parity(a, b):
    return [" ".join(x.split()) == " ".join(y.split()) for x, y in zip(a, b)]
report["parity"] = {
    "static_vs_stock": parity(static_texts, ref_texts),
    "lru_cold_vs_stock": parity(lru_cold_texts, ref_texts),
    "lru_warm_vs_stock": parity(lru_warm_texts, ref_texts),
}

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({kk: (v if kk != "parity" else report["parity"])
                  for kk, v in report.items() if kk in
                  ("stock", "static_pin", "lru_cold_start", "lru_warm", "parity")},
                 indent=2)[:2500])
