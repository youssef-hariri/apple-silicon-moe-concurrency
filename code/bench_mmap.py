#!/usr/bin/env python3
"""P2 C-104: mmap kernel-tier arm vs explicit LRU (uses sweep protocol).
3 prompts × 150 tokens, 1 warmup prompt, OLMoE-0125.
Frozen to results_P2/mmap_vs_lru.json"""
import sys, os, json, ast, statistics, time
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))

import mlx.core as mx
from mmap_arm import make_mmap_model
from mlx_lm.models.cache import KVCache

REF = ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit"
SPLIT = ROOT / "models" / "olmoe-0125-split-4bit"
OUT = HERE / "results_P2" / "mmap_vs_lru.json"
OUT.parent.mkdir(exist_ok=True)
MAX_NEW = 150

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

def rss_gb():
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1e6

def decode(m, ids):
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

report = {"model": "OLMoE-1B-7B-0125-Instruct-4bit", "max_new": MAX_NEW,
          "note": "mmap arm: no explicit cache; kernel page cache is the tier. "
                  "Warm start (1 warmup prompt). RSS includes ~1.2 GB backbone."}
m = make_mmap_model(REF, SPLIT / "experts")
rss0 = rss_gb()
decode(m, ids_all[0])  # warmup
res = [decode(m, ids) for ids in ids_all]
texts = [tok.decode(s, skip_special_tokens=True) for s, _ in res]
tgs = [tg for _, tg in res]
report["mmap"] = {"tg_median": statistics.median(tgs), "tg_per_prompt": tgs,
                  "stats": m.stats(), "rss_gb_before": rss0,
                  "rss_gb_after": rss_gb()}

# parity vs stock (same near-tie rule as C-103)
stock, _ = mlx_lm.load(str(REF))
ref_texts = []
for ids in ids_all:
    cache = [KVCache() for _ in range(len(stock.layers))]
    logits = stock(mx.array([ids]), cache)
    t = mx.argmax(logits[:, -1], axis=-1).item()
    seq = [t]
    for _ in range(MAX_NEW - 1):
        logits = stock(mx.array([[t]]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        if t in eos_ids:
            break
    ref_texts.append(tok.decode(seq, skip_special_tokens=True))
report["parity_vs_stock"] = [" ".join(a.split()) == " ".join(b.split())
                             for a, b in zip(texts, ref_texts)]
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report["mmap"], indent=2)[:900])
print("parity:", report["parity_vs_stock"])
