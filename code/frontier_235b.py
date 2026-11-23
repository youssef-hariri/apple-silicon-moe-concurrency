#!/usr/bin/env python3
"""P2 E3 frontier run (final): Qwen3-235B-A22B on 16 GB M1 Pro, IMPROVED engine
(prefill streamer on, S=8, lazy load). Records proper MLX memory stats.
30 tokens for a quality sample. Frozen to results_P2/qwen235b_frontier.json"""
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
OUT = HERE / "results_P2" / "qwen235b_frontier.json"

PROMPT = ("Explain in a few sentences what a mixture-of-experts model is "
          "and why it can run on a laptop.")
N_TOKENS = 30
SLOTS = 4

report = {"model": "Qwen3-235B-A22B-4bit (130 GB on disk)",
          "machine": "16 GB M1 Pro", "slots_per_layer": SLOTS,
          "engine": "LRU + double-buffered prefill streamer, lazy load, exact",
          "n_tokens": N_TOKENS}

t0 = time.perf_counter()
m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=SLOTS,
                   memory_limit_gb=10.0, lazy=True, prefill_stream=True,
                   stream_workers=8)
report["load_s"] = time.perf_counter() - t0
print(f"loaded in {report['load_s']:.1f}s", flush=True)

tok = m.tokenizer
ids = tok.encode(PROMPT)
cache = [KVCache() for _ in range(m.n_layers)]

t1 = time.perf_counter()
logits = m(mx.array([ids]), cache)
t = mx.argmax(logits[:, -1], axis=-1).item()
mx.synchronize()
report["prefill_s"] = time.perf_counter() - t1
print(f"prefill {report['prefill_s']:.1f}s", flush=True)

seq = [t]
per_token = []
for i in range(N_TOKENS - 1):
    ts = time.perf_counter()
    logits = m(mx.array([[t]]), cache)
    t = mx.argmax(logits[:, -1], axis=-1).item()
    mx.synchronize()
    per_token.append(time.perf_counter() - ts)
    seq.append(t)
    print(f"  tok {i+2}: {tok.decode([t])!r} ({per_token[-1]:.1f}s)", flush=True)

report["per_token_s"] = per_token
report["decode_tok_s"] = (N_TOKENS - 1) / sum(per_token)
report["text"] = tok.decode(seq, skip_special_tokens=True)
report["mlx_peak_gb"] = mx.get_peak_memory() / 1e9
st = m.stats()
report["engine_stats"] = st
OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items() if k not in ("engine_stats", "per_token_s")}, indent=2))
