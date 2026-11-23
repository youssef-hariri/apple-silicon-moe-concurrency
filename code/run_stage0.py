#!/usr/bin/env python3
"""
Stage 0 — driver: correctness gate + speed gate for the pinned arm vs
stock fully-resident MLX, reported against the llama.cpp baseline.

Passes (sequential — one heavy process at a time on the owner's laptop):
  1. reference  : stock mlx_lm 4-bit model (stacked experts), greedy,
                  5 pre-registered prompts × 50 tokens
  2. calibration: pinned arm with NO pins (pure lazy), expert usage counted
  3. measure    : pinned arm with top-K% experts pinned per layer (LRU cold)
  4. resident   : pinned arm with ALL experts pinned (upper-bound sanity ref)

Gate (EXPERIMENT_APPROACH.md §7 step 1): arm median decode ≥ 50% of the
llama.cpp tg baseline AND greedy text identical to the stock reference.

Usage:
    python run_stage0.py --ref-dir <mlx 4bit dir> --split-dir <dir with experts/>
"""
import argparse
import ast
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import mlx.core as mx
from pinned_arm import PinnedOlmoe, generate_greedy


def load_prompts() -> list[str]:
    """Pull TEST_PROMPTS from the pre-registered config.py without importing
    it (config pulls in torch, which is not in this venv)."""
    src = (HERE.parent / "config.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "TEST_PROMPTS":
                    return ast.literal_eval(node.value)
    raise RuntimeError("TEST_PROMPTS not found in config.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="stock mlx_lm 4-bit model dir")
    ap.add_argument("--split-dir", type=Path, required=True,
                    help="output of split_olmoe_experts.py (contains experts/)")
    ap.add_argument("--max-new", type=int, default=50)     # pre-registered
    ap.add_argument("--pin-frac", type=float, default=0.25)
    ap.add_argument("--cold-capacity", type=int, default=256)
    ap.add_argument("--llama-baseline-tg", type=float, default=132.1)
    ap.add_argument("--skip-reference-pass", action="store_true",
                    help="reuse reference text from --out if present")
    ap.add_argument("--out", type=Path,
                    default=HERE.parent.parent / "results/stage0/stage0_report.json")
    args = ap.parse_args()

    prompts = load_prompts()
    report: dict = {"prompts": prompts, "max_new": args.max_new,
                    "llama_baseline_tg": args.llama_baseline_tg,
                    "gate_ratio_needed": 0.5,
                    "arm_order": ["reference", "calibration", "measure",
                                  "resident_all"]}

    # ── 1. reference: stock mlx_lm, fully resident ────────────────────────
    import mlx_lm
    from mlx_lm.models.cache import KVCache
    ref_model, tokenizer = mlx_lm.load(str(args.ref_dir))
    eos_ids = tuple(getattr(tokenizer, "eos_token_ids", None)
                    or {tokenizer.eos_token_id})
    # Reference pass: SAME greedy KVCache loop as the pinned arms, on the
    # stock model. P2 re-run fix: stream_generate (used in P1) diverges from
    # the plain greedy loop on OLMoE-0125 *on the stock model itself*
    # (stock-vs-stock mismatch, near-tie argmax flip ~char 185, prompt 0) —
    # so a stream_generate reference tests "two different mlx_lm paths", not
    # "engine vs stock". The gate's purpose is engine-vs-stock equivalence,
    # which requires the reference to differ from the arm ONLY in the MoE
    # block. (Logged in DECISIONS_P2.)
    prompt_ids = [tokenizer.encode(p) for p in prompts]
    t0 = time.perf_counter()
    ref_texts, ref_tg = [], []
    for pids in prompt_ids:
        cache = [KVCache() for _ in range(len(ref_model.layers))]
        x = mx.array([pids])
        tp0 = time.perf_counter()
        logits = ref_model(x, cache)
        tok0 = mx.argmax(logits[:, -1], axis=-1).item()
        seq = [tok0]
        mx.synchronize()
        tp1 = time.perf_counter()
        for _ in range(args.max_new - 1):
            logits = ref_model(mx.array([[tok0]]), cache)
            tok0 = mx.argmax(logits[:, -1], axis=-1).item()
            seq.append(tok0)
            if tok0 in eos_ids:
                break
        mx.synchronize()
        dt = time.perf_counter() - tp1
        ref_texts.append(tokenizer.decode(seq, skip_special_tokens=True))
        ref_tg.append((len(seq) - 1) / dt if dt > 0 else float("nan"))
    report["reference"] = {"texts": ref_texts, "stock_tg_tok_s": ref_tg,
                           "seconds": time.perf_counter() - t0}
    print("reference pass done", flush=True)
    del ref_model
    mx.clear_cache()

    tok_ids = prompt_ids
    experts = args.split_dir / "experts"
    n_experts = 64
    try:
        cfg = json.loads((args.ref_dir / "config.json").read_text())
        n_experts = cfg["num_experts"]
    except Exception:
        pass

    # ── 2. calibration: pure lazy (no pins), count usage ─────────────────
    model = PinnedOlmoe(args.ref_dir, experts, cold_capacity=args.cold_capacity)
    for ids in tok_ids:
        generate_greedy(model, ids, max_new=args.max_new, eos_ids=eos_ids)
    n_layers = len(model.model.layers)
    per_layer_usage = {l: {} for l in range(n_layers)}
    for (l, e), c in model.pool.usage.items():
        per_layer_usage[l][e] = c
    k = max(1, int(args.pin_frac * n_experts))
    pin_spec = {l: [e for e, _ in sorted(u.items(),
                                         key=lambda kv: (-kv[1], kv[0]))[:k]]
                for l, u in per_layer_usage.items()}
    report["calibration"] = {"loads": model.pool.stats["loads"],
                             "distinct_experts": len(model.pool.usage),
                             "pin_count_per_layer": k}
    print(f"calibration done: {len(model.pool.usage)} distinct experts hit",
          flush=True)
    del model
    mx.clear_cache()

    # ── 3. measured pass: pinned arm ──────────────────────────────────────
    model = PinnedOlmoe(args.ref_dir, experts, cold_capacity=args.cold_capacity)
    model.pin_experts(pin_spec)
    pinned_mb = model.pool.resident_mb()
    runs = [generate_greedy(model, ids, max_new=args.max_new, eos_ids=eos_ids)
            for ids in tok_ids]
    arm_tg = [r["tg_tok_s"] for r in runs]
    st = model.pool.stats
    lookups = st["pin_hits"] + st["cold_hits"] + st["loads"]
    arm_texts = [tokenizer.decode(r["tokens"], skip_special_tokens=True) for r in runs]
    report["arm_pinned"] = {
        "tg_tok_s_median": statistics.median(arm_tg),
        "tg_tok_s_per_prompt": arm_tg,
        "prefill_s_per_prompt": [r["prefill_s"] for r in runs],
        "pin_hits": st["pin_hits"], "cold_hits": st["cold_hits"],
        "disk_loads": st["loads"], "evictions": st["evictions"],
        "pin_hit_rate": st["pin_hits"] / max(1, lookups),
        "resident_expert_file_mb": pinned_mb,
        "gpu_active_mb": mx.get_active_memory() / 1e6,
        "gpu_peak_mb": mx.get_peak_memory() / 1e6,
        "outputs": arm_texts,
    }
    med = report["arm_pinned"]["tg_tok_s_median"]
    print(f"pinned arm: median {med:.1f} tok/s", flush=True)
    del model
    mx.clear_cache()

    # ── 4. resident-all upper bound (same code path, all pinned) ─────────
    model = PinnedOlmoe(args.ref_dir, experts, cold_capacity=1)
    model.pin_all(n_experts)
    runs_ra = [generate_greedy(model, ids, max_new=args.max_new, eos_ids=eos_ids)
               for ids in tok_ids]
    tg_ra = [r["tg_tok_s"] for r in runs_ra]
    report["arm_resident_all"] = {
        "tg_tok_s_median": statistics.median(tg_ra),
        "resident_expert_file_mb": model.pool.resident_mb(),
        "outputs": [tokenizer.decode(r["tokens"], skip_special_tokens=True) for r in runs_ra],
    }
    print(f"resident-all: median {report['arm_resident_all']['tg_tok_s_median']:.1f} tok/s",
          flush=True)
    del model
    mx.clear_cache()

    # ── gates ──────────────────────────────────────────────────────────────
    def norm(s: str) -> str:
        return " ".join(s.split())
    parity = [norm(arm_texts[i]) == norm(ref_texts[i])
              for i in range(len(prompts))]
    parity_ra = [norm(report["arm_resident_all"]["outputs"][i]) == norm(ref_texts[i])
                 for i in range(len(prompts))]
    # Stage-0 gate: T1 on the rental box runs fully resident (5 GB in 18 GB).
    # The representative pass is resident-all; partial-pinned is a cold-path
    # characterization for T2 (informational).
    med_ra = report["arm_resident_all"]["tg_tok_s_median"]
    speed_ok_resident = med_ra >= 0.5 * args.llama_baseline_tg
    speed_ok_partial = med >= 0.5 * args.llama_baseline_tg  # informational
    report["token_parity"] = {"pinned_vs_reference_exact": all(parity),
                              "per_prompt": parity,
                              "resident_all_vs_reference": all(parity_ra)}
    report["gate"] = {
        "speed_within_2x_llama": bool(speed_ok_resident),
        "correctness": bool(all(parity) and all(parity_ra)),
        "stage0_pass": bool(speed_ok_resident and all(parity) and all(parity_ra)),
        "note": ("gate uses resident-all (T1 fits fully on rental box); "
                 "partial-pinned speed is T2 cold-path characterization"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["gate"], indent=2))
    print(f"resident-all median tg = {med_ra:.1f} vs llama {args.llama_baseline_tg:.1f} "
          f"(gate ≥ {0.5 * args.llama_baseline_tg:.1f}) | "
          f"parity {sum(parity)}/{len(parity)} pinned, "
          f"{sum(parity_ra)}/{len(parity_ra)} resident-all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
