# P2 Claim Map — skeleton (created 2026-11-21, Phase 0)

Convention (from P1): every claim gets an ID, prose text, frozen source, extraction
script, and verification status (pending / verified / refuted / corrected).
Semantic map (what the number MEANS) lives in `semantic_map_P2.md` — paired by claim ID.

## Phase 0 claims

| ID | Claim (draft) | Frozen source | Script | Status |
|---|---|---|---|---|
| P2-C001 | Engine re-validated on 16 GB M1 Pro: Stage-0 gate passes (resident-all ≥66 tok/s proxy vs llama 132.1 baseline; greedy parity 5/5) | results/stage0/stage0_report_P2_rerun.json | code/stage0/run_stage0.py | verified 2026-11-21 (resident-all 5/5 exact; pinned 4/5 + 1 measured near-tie, margin 0.078, D2-007) |
| P2-C002 | P2 uses OLMoE-1B-7B-**0125**-Instruct-4bit (mlx-community); the 0924 checkpoint used in P1 is no longer available (repo 401). P2 numbers are internally consistent; cross-paper token-level comparisons to P1 frozen logs are void, ratio-level comparisons stand | HF API response log; this file | manual | verified 2026-11-21 |
| P2-C003 | Edge0 and SlotStream build and run on the 16 GB M1 Pro (or documented failure) | competitors/ build logs | manual | pending |
| P2-C004 | Qwen3.5-122B-A10B config is hybrid linear-attention (full_attention_interval 4) → fails P1 Appendix-B pre-flight for the continuous-batching path | config.json fetched 2026-11-21 | curl log | verified 2026-11-21 |

## Phase 1 claims (mechanism ports — numbers TBD)

| ID | Claim (draft) | Status |
|---|---|---|
| P2-C101 | incr_stack-style sticky slots change single-user decode by X% (exact-output) | pending |
| P2-C102 | Double-buffered prefill: 2.1–2.6× prefill speedup on streamed OLMoE (S=16), decode unchanged | verified 2026-11-21 (results_P2/prefill_double_buffer.json, D2-010) |
| P2-C103 | LRU vs static at equal RAM: LRU wins mid-band (S=48: 52.9 vs 37.1 tok/s, +43%), ties low, loses at full residency (57.1 vs 84.8); knee at 75% capacity | verified 2026-11-21 (results_P2/lru_capacity_sweep.json, D2-009) |
| P2-C104 | Kernel-owned tier (mmap) loses: 5.7 tok/s vs LRU 12.8–52.9; per-expert dispatch dominates, kernel reclaims working set | verified 2026-11-21, negative result (results_P2/mmap_vs_lru.json, D2-011) |
| P2-C105 | Temporal recall 0.465 vs popularity 0.371 on Qwen3-30B; moderate structure, below payoff threshold on single SSD — oracle deferred (bounded pre-negative) | verified 2026-11-21 (results_P2/qwen30b_routing_predictability.json, D2-012) |

## Phase 2 claims (concurrency survival)

| ID | Claim (draft) | Status |
|---|---|---|
| P2-C201 | P2 crossover: p05 ceilings on 3 continuity models, static vs LRU vs mmap, overlaid on P1 | pending |
| P2-C202 | Mechanism decay: [mechanism] retains/loses X% of its single-user gain at N=8 | pending |
| P2-C203 | Frontier: [122B-class model] serves N users on 16 GB at p05 ≥ floor / feasibility | pending |
| P2-C204 | Price of exactness: approximation arms (prerouter, low-bit) gain X tok/s at Y quality cost | pending |
