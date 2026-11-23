#!/usr/bin/env python3
"""P2 E3 smoke test: Qwen3-235B-A22B on 16 GB M1 Pro via LRU streaming.
Lazy backbone load + S=8 slots/layer. Feasibility probe: does it run,
what's the per-token time, what's peak memory. Frozen to
results_P2/qwen235b_smoke.json"""
import sys, os, json, time
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))

import mlx.core as mx
from lru_arm import make_lru_model
from mlx_lm.models.cache import KVCache

REF = ROOT / "models" / "qwen3-235b-a22b-mlx-4bit"
SPLIT = ROOT / "models" / "qwen3-235b-split-4bit"
OUT = HERE / "results_P2" / "qwen235b_smoke.json"
OUT.parent.mkdir(exist_ok=True)

PROMPT = "The capital of France is"
N_TOKENS = 5
SLOTS = 8

def rss_gb():
    import subprocess
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                              capture_output=True, text=True).stdout.strip()) / 1e6

report = {"model": "Qwen3-235B-A22B-4bit", "machine": "16 GB M1 Pro",
          "slots_per_layer": SLOTS, "prompt": PROMPT, "n_tokens": N_TOKENS}

t0 = time.perf_counter()
m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=SLOTS,
                   memory_limit_gb=13.0, lazy=True)
report["load_s"] = time.perf_counter() - t0
report["rss_after_load_gb"] = rss_gb()
print(f"loaded in {report['load_s']:.0f}s, RSS {report['rss_after_load_gb']:.1f} GB", flush=True)

tok = m.tokenizer
ids = tok.encode(PROMPT)
cache = [KVCache() for _ in range(m.n_layers)]

t1 = time.perf_counter()
logits = m(mx.array([ids]), cache)
t = mx.argmax(logits[:, -1], axis=-1).item()
mx.synchronize()
report["prefill_s"] = time.perf_counter() - t1
print(f"prefill {report['prefill_s']:.0f}s, first token {tok.decode([t])!r}", flush=True)

seq = [t]
per_token = []
for _ in range(N_TOKENS - 1):
    ts = time.perf_counter()
    logits = m(mx.array([[t]]), cache)
    t = mx.argmax(logits[:, -1], axis=-1).item()
    mx.synchronize()
    per_token.append(time.perf_counter() - ts)
    seq.append(t)
    print(f"  token {len(seq)}: {tok.decode([t])!r} ({per_token[-1]:.1f}s)", flush=True)

report["per_token_s"] = per_token
report["text"] = tok.decode(seq, skip_special_tokens=True)
report["peak_rss_gb"] = rss_gb()
report["engine_stats"] = m.stats()
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k != "engine_stats"}, indent=2))
