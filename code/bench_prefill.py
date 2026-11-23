#!/usr/bin/env python3
"""P2 C-102: double-buffered prefill (M2) — prefill time and end-to-end TTFT,
LRU arm at S=16 (streaming regime), long prompts, streamer off vs on.
Frozen to results_P2/prefill_double_buffer.json"""
import sys, os, json, statistics, time
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))

import mlx.core as mx
from lru_arm import make_lru_model
from mlx_lm.models.cache import KVCache

REF = ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit"
SPLIT = ROOT / "models" / "olmoe-0125-split-4bit"
OUT = HERE / "results_P2" / "prefill_double_buffer.json"
OUT.parent.mkdir(exist_ok=True)

import mlx_lm
tok = mlx_lm.load(str(REF))[1]
eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})

# synthetic long prompts: repeat a base paragraph to exact token lengths
BASE = ("Mixture-of-experts models route each token to a small subset of "
        "feed-forward experts, which makes them attractive for memory-bound "
        "consumer hardware because only a fraction of the weights is read "
        "per token. ")
def prompt_of_len(n):
    text = BASE
    while len(tok.encode(text)) < n:
        text += BASE
    return tok.encode(text)[:n]

LENGTHS = [256, 512]
report = {"model": "OLMoE-1B-7B-0125-Instruct-4bit", "slots_per_layer": 16,
          "lengths": LENGTHS, "protocol": "fresh model instance per arm, "
          "cold SSD page cache not purged (same condition both arms, rotated order)"}

def run(prefill_stream, ids_list):
    m = make_lru_model(REF, SPLIT / "experts", n_slots_per_layer=16,
                       prefill_stream=prefill_stream)
    pre_s, dec_tps = [], []
    for ids in ids_list:
        cache = [KVCache() for _ in range(m.n_layers)]
        t0 = time.perf_counter()
        logits = m(mx.array([ids]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        mx.synchronize()
        pre_s.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        for _ in range(49):
            logits = m(mx.array([[t]]), cache)
            t = mx.argmax(logits[:, -1], axis=-1).item()
            if t in eos_ids:
                break
        mx.synchronize()
        dec_tps.append(49 / (time.perf_counter() - t1))
    st = m.stats()
    strm = None
    if prefill_stream:
        s = m.pools()[0].streamer
        strm = s.stats
    del m
    mx.clear_cache()
    return pre_s, dec_tps, st, strm

for L in LENGTHS:
    ids_list = [prompt_of_len(L) for _ in range(2)]
    # rotated order: off, on, off, on would be ideal; keep one fresh instance
    # per arm to avoid cross-arm cache warm-up, note in protocol.
    pre_off, dec_off, _, _ = run(False, ids_list)
    pre_on, dec_on, st_on, strm = run(True, ids_list)
    report[f"L{L}"] = {
        "prefill_s_off": pre_off, "prefill_s_on": pre_on,
        "prefill_speedup": [round(a / b, 3) for a, b in zip(pre_off, pre_on)],
        "decode_tps_off": dec_off, "decode_tps_on": dec_on,
        "streamer_stats": strm,
    }
    print(f"L={L}: prefill off {pre_off} on {pre_on} "
          f"speedup {[round(a/b,2) for a,b in zip(pre_off,pre_on)]}", flush=True)

OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print("saved", OUT)
