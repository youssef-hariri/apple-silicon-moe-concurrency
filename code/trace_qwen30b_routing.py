#!/usr/bin/env python3
"""P2 C-105 (N3 oracle quality, part A): per-token routing trace on
Qwen3-30B-A3B + predictability analysis. Replicates P1's OLMoE study
(D-044 family) on the crossover's big model, answering: is there exploitable
next-step routing structure for a training-free prefetch oracle?

Trace: LRU arm (native router — exactness contract makes routing identical),
S=32/128 per layer, 10 WildChat prompts × 60 tokens.
Analysis: Markov-1 temporal recall, adjacent-step Jaccard, popularity baseline,
same definitions as P1 sequence_predictability.py.
Frozen to results_P2/qwen30b_routing_predictability.json"""
import sys, os, json, statistics, time
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))

import mlx.core as mx
from lru_arm import make_lru_model
from mlx_lm.models.cache import KVCache

REF = ROOT / "models" / "qwen3-30b-a3b-mlx-4bit"
SPLIT = ROOT / "models" / "qwen3-30b-split-4bit"
OUT = HERE / "results_P2" / "qwen30b_routing_predictability.json"
OUT.parent.mkdir(exist_ok=True)

N_PROMPTS = 10
MAX_NEW = 60
K = 8  # Qwen3-30B top-k

# frozen corpus prompts (same file P1 used; first N lines' first 2000 chars)
corpus = (ROOT / "results" / "corpus" / "wildchat3k.txt").read_text().splitlines()
PROMPTS = [l[:2000] for l in corpus if l.strip()][:N_PROMPTS]

import mlx_lm
tok = mlx_lm.load(str(REF))[1]
eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})

m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=32,
                   memory_limit_gb=11.0)

# recorder: per layer, list over token positions of expert sets
trace: dict[int, list[list[int]]] = {l: [] for l in m.moe_layer_ids}
def rec(layer_id, rows):
    for row in rows:
        trace[layer_id].append(row)

from lru_arm import LRUSwitchMLP
from pinned_arm import moe_submodule
for l in m.moe_layer_ids:
    moe_submodule(m.model.layers[l]).switch_mlp.router_recorder = rec

t_start = time.perf_counter()
for pi, p in enumerate(PROMPTS):
    ids = tok.encode(p)
    cache = [KVCache() for _ in range(m.n_layers)]
    logits = m(mx.array([ids]), cache)
    t = mx.argmax(logits[:, -1], axis=-1).item()
    for _ in range(MAX_NEW - 1):
        logits = m(mx.array([[t]]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        if t in eos_ids:
            break
    print(f"prompt {pi} traced, elapsed {time.perf_counter()-t_start:.0f}s", flush=True)

# ---- analysis -------------------------------------------------------------
report = {"model": "Qwen3-30B-A3B-4bit", "n_prompts": N_PROMPTS,
          "max_new": MAX_NEW, "layers": len(m.moe_layer_ids), "k": K}
per_layer = {"temporal": [], "jaccard": [], "popularity": []}
for l in m.moe_layer_ids:
    seq = trace[l]
    if len(seq) < 3:
        continue
    sets = [set(r) for r in seq]
    # temporal recall: |E_t ∩ E_{t+1}| / k
    temp = [len(sets[i] & sets[i + 1]) / K for i in range(len(sets) - 1)]
    jac = [len(sets[i] & sets[i + 1]) / len(sets[i] | sets[i + 1])
           for i in range(len(sets) - 1)]
    # popularity baseline: top-K global experts recalled per step
    from collections import Counter
    pop = Counter(e for s in sets for e in s)
    top_pop = {e for e, _ in pop.most_common(K)}
    pop_rec = [len(s & top_pop) / K for s in sets]
    per_layer["temporal"].append(statistics.mean(temp))
    per_layer["jaccard"].append(statistics.mean(jac))
    per_layer["popularity"].append(statistics.mean(pop_rec))

report["temporal_recall_mean"] = statistics.mean(per_layer["temporal"])
report["jaccard_adjacent_mean"] = statistics.mean(per_layer["jaccard"])
report["popularity_recall_mean"] = statistics.mean(per_layer["popularity"])
report["per_layer_temporal"] = per_layer["temporal"]
report["interpretation"] = ("Temporal recall > popularity => exploitable "
    "next-step structure (prefetch oracle viable); ~= popularity => flat.")

OUT.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items()
                  if not k.startswith("per_layer")}, indent=2))
