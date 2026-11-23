#!/usr/bin/env python3
"""
P2 Phase 2 runner — resumable, pausable, crash-safe.

Protocol: E1 crossover re-run with the P2 engine (LRU arm vs static arm) at
concurrency levels 1,2,4,8,12,16 on the continuity models. Each (model, arm,
level) is one run UNIT producing one frozen JSON. The runner:
  - skips units whose JSON already exists (resume for free)
  - checks PAUSE / STOP between units (see p2ctl.sh)
  - stamps every unit with machine-state metadata (P1 habit)

Run units are single-user-equivalent sweeps: N concurrent greedy requests
against the engine, per-user decode tok/s recorded, p05 computed, ceiling =
largest N with p05 >= 10 tok/s (uncensored: ramp stops after 2 consecutive
failures above the last pass).

Usage: ./p2ctl.sh resume && ../code/venv_stage0/bin/python p2_runner.py
"""
import sys, os, json, time, subprocess, threading, queue
os.environ["DISABLE_METAL_KERNEL"] = "1"
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code" / "stage0"))
PAUSE = HERE / "PAUSE"
STOP = HERE / "STOP"
OUTDIR = HERE / "results_P2" / "phase2"
OUTDIR.mkdir(parents=True, exist_ok=True)

FLOOR = 10.0          # tok/s per-user SLO floor (P1 statistic)
MAX_NEW = 100         # tokens per request
LEVELS = [1, 2, 4, 8, 12, 16, 24, 32]  # uncensored: run past failure

# Frozen WildChat corpus (P1): DISTINCT prompt per user within a unit —
# reusing the 5 TEST_PROMPTS let later users freeride on a warm cache
# (caught in the first dry run, D2-014).
_CORPUS_CACHE = None
def corpus_prompts(n, tok, offset=0):
    global _CORPUS_CACHE
    if _CORPUS_CACHE is None:
        lines = (ROOT / "results" / "corpus" / "wildchat3k.txt").read_text().splitlines()
        _CORPUS_CACHE = [l[:1500] for l in lines if l.strip()]
    return [tok.encode(_CORPUS_CACHE[(offset + i) % len(_CORPUS_CACHE)]) for i in range(n)]

MODELS = {
    "olmoe": dict(
        ref=ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit",
        split=ROOT / "models" / "olmoe-0125-split-4bit",
        slots=48,          # 75% capacity: LRU's winning band from C-103
        mem_gb=11.0,
    ),
    "olmoeS56": dict(
        ref=ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit",
        split=ROOT / "models" / "olmoe-0125-split-4bit",
        slots=56,
        mem_gb=11.0,
    ),
    "olmoeS64": dict(
        ref=ROOT / "models" / "olmoe-1b-7b-0125-instruct-mlx-4bit",
        split=ROOT / "models" / "olmoe-0125-split-4bit",
        slots=64,
        mem_gb=11.0,
    ),
    "qwen15": dict(
        ref=ROOT / "models" / "qwen15-moe-a27b-mlx-4bit",
        split=ROOT / "models" / "qwen15-split-4bit",
        slots=45,          # 75% of 60 experts (1.8x ratio point)
        mem_gb=11.0,
        static_mem_gb=10.0,  # full residency for the static arm
        levels=[1, 2, 4, 8, 12, 16],
    ),
    "qwen30b": dict(
        ref=ROOT / "models" / "qwen3-30b-a3b-mlx-4bit",
        split=ROOT / "models" / "qwen3-30b-split-4bit",
        slots=64,          # 50% of 128 experts (0.9x ratio, >RAM)
        mem_gb=11.0,
        levels=[1, 2, 4, 8],
    ),
}

def machine_stamp():
    try:
        thermal = subprocess.run(["pmset", "-g", "thermstate"], capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    except Exception:
        thermal = "n/a"
    return {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "thermal": thermal}

REPS = [2, 3]  # repeat pass: existing singles count as r1; median of 3 at paper time
def unit_path(model, arm, level, rep=None):
    suffix = f"_r{rep}" if rep else ""
    return OUTDIR / f"{model}_{arm}_N{level}{suffix}.json"

def interrupted():
    if STOP.exists():
        return "stop"
    if PAUSE.exists():
        return "pause"
    return None

# ---------------------------------------------------------------- engine ---
import mlx.core as mx
from lru_arm import make_lru_model
from mlx_lm.models.cache import KVCache

def run_lru_unit(cfg, level):
    """N concurrent greedy requests on one engine instance, per-user tok/s.
    Requests share the process; decode steps interleave per request (simple
    round-robin continuous batching equivalent for the measurement purpose:
    each request decodes 1 token per round)."""
    m = make_lru_model(cfg["ref"], cfg["split"] / "experts",
                       n_slots_per_layer=cfg["slots"],
                       memory_limit_gb=cfg["mem_gb"], lazy=False)
    tok = m.tokenizer
    eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
    prompts = corpus_prompts(level, tok)

    # prefill all requests
    states = []
    for p in prompts:
        cache = [KVCache() for _ in range(m.n_layers)]
        logits = m(mx.array([p]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        states.append(dict(cache=cache, next=t, done=False, n=0, times=[]))
    mx.synchronize()

    # round-robin decode: per-user wall time per token
    t_all = time.perf_counter()   # decode phase starts: all prefills done
    active = [s for s in states]
    while active:
        for s in list(active):
            logits = m(mx.array([[s["next"]]]), s["cache"])
            t = mx.argmax(logits[:, -1], axis=-1).item()
            mx.synchronize()
            s["next"] = t
            s["n"] += 1
            s["times"].append(time.perf_counter())
            if t in eos_ids or s["n"] >= MAX_NEW:
                s["end"] = time.perf_counter()
                active.remove(s)
    total = time.perf_counter() - t_all
    # unfinished users count their time to the end of the phase
    for s in states:
        s.setdefault("end", t_all + total)

    per_user = [{"n_tokens": s["n"],
                 "mean_tok_s": s["n"] / (s["end"] - t_all) if s["end"] > t_all else 0}
                for s in states]
    # p05 across users: 5th percentile of per-user mean tok/s (P1 statistic,
    # numpy-equivalent linear interpolation)
    means = sorted(u["mean_tok_s"] for u in per_user)
    import numpy as np
    p05 = float(np.percentile(means, 5, method="linear")) if means else 0
    st = m.stats()
    del m
    mx.clear_cache()
    return {"level": level, "per_user": per_user,
            "per_user_mean_tok_s": means, "p05_tok_s": p05,
            "above_floor": p05 >= FLOOR, "total_s": total,
            "engine_stats": {k: v for k, v in st.items() if k in
                             ("hits", "misses", "evictions", "load_seconds")}}

def run_hybrid_unit(cfg, level):
    """Hybrid arm: pinned hot set (calibrated on a disjoint corpus slice) +
    LRU tail. slots=48 with hot=24 unless overridden in cfg."""
    from lru_arm import make_hybrid_model
    calib = corpus_prompts(4, None if False else __import__("mlx_lm").load(str(cfg["ref"]))[1], offset=1000)
    m = make_hybrid_model(cfg["ref"], cfg["split"] / "experts",
                          n_slots_per_layer=cfg["slots"],
                          hot_per_layer=cfg.get("hot", 24),
                          calib_prompt_ids=calib,
                          memory_limit_gb=cfg["mem_gb"])
    return _shared_round_robin(m, cfg, level)


def _shared_round_robin(m, cfg, level):
    """Common round-robin measurement used by all arms."""
    tok = m.tokenizer
    eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
    prompts = corpus_prompts(level, tok)
    states = []
    for p in prompts:
        cache = [KVCache() for _ in range(m.n_layers)]
        logits = m(mx.array([p]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        states.append(dict(cache=cache, next=t, done=False, n=0, times=[]))
    mx.synchronize()
    t_all = time.perf_counter()
    active = list(states)
    while active:
        for s_ in list(active):
            logits = m(mx.array([[s_["next"]]]), s_["cache"])
            t = mx.argmax(logits[:, -1], axis=-1).item()
            mx.synchronize()
            s_["next"] = t
            s_["n"] += 1
            if t in eos_ids or s_["n"] >= MAX_NEW:
                s_["end"] = time.perf_counter()
                active.remove(s_)
    total = time.perf_counter() - t_all
    for s_ in states:
        s_.setdefault("end", t_all + total)
    per_user = [{"n_tokens": s_["n"],
                 "mean_tok_s": s_["n"] / (s_["end"] - t_all) if s_["end"] > t_all else 0}
                for s_ in states]
    means = sorted(u["mean_tok_s"] for u in per_user)
    import numpy as np
    p05 = float(np.percentile(means, 5, method="linear")) if means else 0
    st = m.stats()
    del m
    mx.clear_cache()
    return {"level": level, "per_user": per_user,
            "per_user_mean_tok_s": means, "p05_tok_s": p05,
            "above_floor": p05 >= FLOOR, "total_s": total,
            "engine_stats": {k: v for k, v in st.items() if k in
                             ("hits", "misses", "evictions", "load_seconds", "pinned")}}


def run_static_unit(cfg, level):
    """Static full-residency arm (P1's fits-RAM winner): PinnedOlmoe with
    pin_all — zero misses, fully-pinned fast path. Only valid for models
    that fit RAM (OLMoE on 16 GB). Same round-robin protocol."""
    from pinned_arm import PinnedOlmoe
    arm = PinnedOlmoe(cfg["ref"], cfg["split"] / "experts", cold_capacity=1,
                      memory_limit_gb=cfg.get("static_mem_gb", 6.0))
    arm.pin_all(arm.num_experts)
    tok = arm.tokenizer
    eos_ids = tuple(getattr(tok, "eos_token_ids", None) or {tok.eos_token_id})
    prompts = corpus_prompts(level, tok)

    states = []
    for p in prompts:
        cache = [KVCache() for _ in range(arm.n_layers)]
        logits = arm(mx.array([p]), cache)
        t = mx.argmax(logits[:, -1], axis=-1).item()
        states.append(dict(cache=cache, next=t, done=False, n=0, times=[]))
    mx.synchronize()

    t_all = time.perf_counter()
    active = list(states)
    while active:
        for s in list(active):
            logits = arm(mx.array([[s["next"]]]), s["cache"])
            t = mx.argmax(logits[:, -1], axis=-1).item()
            mx.synchronize()
            s["next"] = t
            s["n"] += 1
            if t in eos_ids or s["n"] >= MAX_NEW:
                s["end"] = time.perf_counter()
                active.remove(s)
    total = time.perf_counter() - t_all
    for s in states:
        s.setdefault("end", t_all + total)

    per_user = [{"n_tokens": s["n"],
                 "mean_tok_s": s["n"] / (s["end"] - t_all) if s["end"] > t_all else 0}
                for s in states]
    means = sorted(u["mean_tok_s"] for u in per_user)
    import numpy as np
    p05 = float(np.percentile(means, 5, method="linear")) if means else 0
    st = {k: arm.pool.stats[k] for k in ("loads", "stack_hits")}
    del arm
    mx.clear_cache()
    return {"level": level, "per_user": per_user,
            "per_user_mean_tok_s": means, "p05_tok_s": p05,
            "above_floor": p05 >= FLOOR, "total_s": total,
            "engine_stats": st}


def worker_main(argv):
    """Fresh-process unit execution (D2-019: OS reclaims all memory on exit)."""
    model, arm, level, rep = argv[0], argv[1], int(argv[2]), (None if argv[3] == "None" else int(argv[3]))
    cfg = MODELS[model]
    up = unit_path(model, arm, level, rep)
    if up.exists():
        return 0
    res = (run_lru_unit(cfg, level) if arm == "lru"
           else run_static_unit(cfg, level) if arm == "static"
           else run_hybrid_unit(cfg, level))
    doc = {"model": model, "arm": arm, **res, **machine_stamp(),
           "protocol": {"floor": FLOOR, "max_new": MAX_NEW,
                        "note": "round-robin shared-engine concurrency"}}
    tmp = up.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.rename(up)
    print(f"  done: p05 {res['p05_tok_s']:.1f} tok/s "
          f"({'PASS' if res['above_floor'] else 'FAIL'})", flush=True)
    return 0

def main():
    plan = []
    reps = [None] + (REPS if os.environ.get("REPEAT_PASS") == "1" else [])
    for rep in reps:
        for model, cfg in MODELS.items():
            if model == "olmoe":
                arms = ("lru", "static", "hybrid")
            elif model == "qwen15":
                arms = ("lru", "static")
            else:
                arms = ("lru",)
            for arm in arms:
                for level in cfg.get("levels", LEVELS):
                    plan.append((model, arm, level, cfg, rep))
    print(f"queue: {len(plan)} units", flush=True)
    for model, arm, level, cfg, rep in plan:
        up = unit_path(model, arm, level, rep)
        if up.exists():
            print(f"skip {up.name} (frozen)", flush=True)
            continue
        why = interrupted()
        if why:
            print(f"{why}: exiting before {up.name}; rerun to resume", flush=True)
            return 2 if why == "pause" else 3
        print(f"RUN {up.name} ...", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             model, arm, str(level), str(rep)],
            cwd=str(HERE), env={**os.environ, "DISABLE_METAL_KERNEL": "1"}).returncode
        if rc != 0:
            err = {"error": f"worker exit {rc}", **machine_stamp()}
            (OUTDIR / f"ERROR_{up.name}").write_text(json.dumps(err, indent=2))
            print(f"ERROR in {up.name}: worker exit {rc}", flush=True)
            continue
        print(f"  done in {time.perf_counter()-t0:.0f}s", flush=True)
    print("queue complete", flush=True)
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(worker_main(sys.argv[2:]))
    sys.exit(main())
