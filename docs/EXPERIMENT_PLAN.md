# FreeToken on Apple — Experiment Plan (P2) — v0.2

**Status:** approved direction, 2026-11-21 (supersedes v0.1)
**Working title:** *"When Caching Pays: Edge-MoE Serving Mechanisms Under Concurrency and Exactness on Unified Memory"*
**Inputs folded in:** EXPERIMENT_PLAN v0.1, LITERATURE_CHECK, E0_THREAT_VERIFICATION, owner decisions 2026-11-21.

---

## 1. Thesis

The single-user SSD-streaming speed race on Apple Silicon is over (Edge0 20 tok/s @35B,
SlotStream ~12 tok/s @125B, SwiftLM 10×). The open scientific problem is **concurrent,
output-exact serving**: every published mechanism — dynamic caching, prefetch, streaming —
is measured single-request, and Edge0 explicitly states batching breaks their profiles.
P2 ports the field's best mechanisms into our exact MLX engine and measures **which survive
4→16 concurrent users on unified memory**, on one frozen protocol (p05 ceiling, 10 tok/s
SLO floor, WildChat corpus, fidelity gates).

## 2. The Exactness Contract (owner decision, binding)

The served model must answer as the original model: **token-exact output, native router
authoritative, no retraining, no quality-recovery LoRA.** All P1 fidelity gates apply
(bit-exact synthetic equivalence, greedy match, perplexity ≤0.5% band). Speed must come
from systems mechanisms, not changed math.

**Consequence:** anything that changes outputs — Edge0's prediction-as-routing prerouter,
low-bit (HOBBIT/PagedMoE-style) expert replicas — is tested only as a **clearly labeled
approximation arm**, reported in the "price of exactness" comparison, never the default.

## 3. Model roster (owner-approved)

| Model | 4-bit size | Role |
|---|---|---|
| OLMoE-1B-7B | 4.2 GB | P1 continuity point, fits-RAM (3.8× ratio) |
| Qwen1.5-MoE-A2.7B | 8.8 GB | P1 continuity point (1.8×) |
| Qwen3-30B-A3B | 18 GB | P1 continuity point (0.9×, >RAM on 16 GB) |
| **Qwen3.5-122B-A10B** (new) | ~65 GB | New frontier point — 122B served concurrently on a 16 GB laptop; SwiftLM proves streamable; must pass our pre-flight (NeoX-style rope check) |

Hardware: 16 GB M1 Pro laptop (primary). Optional 24 GB M4 rental for gate-scale checks.
Prerequisite: ~70 GB free SSD for the 122B checkpoint (owner to confirm disk space).

## 4. Phases

### Phase 0 — Harness & baselines (days)
- Stage-0 re-validation of current engine (parity gate as in P1).
- **E0.5 ecosystem benchmark:** run Edge0 and SlotStream on our 16 GB M1 Pro under OUR
  protocol (p05, WildChat) → same-machine comparison table for related work. Both are
  open source. (Also validates that our speed gaps are mechanism gaps, not measurement gaps.)
- Freeze P2 corpus/manifests; open `DECISIONS_P2.md`.
- **Start claim map + semantic map NOW** (see §6) — not at writing time.

### Phase 1 — Adopt-and-match (build the fast exact engine)
Port the proven single-user mechanisms, keeping exactness:
1. **incr_stack-style sticky slots** (Edge0, +34%): persistent per-layer expert stacks
   updated in place; kill per-step assembly tax.
2. **Double-buffered prefill** (FreeToken): stream layer l+1 during layer l compute.
3. **LRU expert cache** (FreeToken) replacing static pinning — same RAM budget.
4. **mmap/kernel-tier arm** (arXiv:2608.12103): let the OS page cache be the expert tier
   vs explicit LRU — the Apple-specific three-way A/B (static / explicit-LRU / mmap).
5. **Spec-decode as expert-prefetch oracle (N3):** draft model's routing predicts next-step
   expert sets (not tokens). Untested anywhere; SwiftLM showed token-level spec-decode fails
   under streaming — the expert-level oracle is the open version.

Gate after each mechanism: exactness gates green, then single-user speed delta logged.

**Success bar:** match or beat the ecosystem's single-user numbers on our machine
(Edge0-class: several tok/s on 18 GB model; target ≥5 tok/s on Qwen3-30B streaming)
while staying token-exact.

### Phase 2 — Concurrency-survival study (the paper's core)
Re-run the P1 protocols with the Phase-1 engine, sweeping concurrency:
- **E1 — Crossover re-run:** p05 ceilings on OLMoE / Qwen1.5 / Qwen3-30B with
  static vs LRU vs mmap arms. Does LRU lift the ~2× cap? Overlay on P1 curves.
- **E2 — Mechanism decay under load:** per-mechanism ablation at N = 1, 2, 4, 8, 16 users.
  Hypothesis (from P1 mechanism data): prefetch accuracy and cache hit-rate decay as
  interleaved users widen the working set (36→56 experts for 95% coverage). Which mechanism
  degrades gracefully, which collapses?
- **E3 — 122B frontier:** Qwen3.5-122B streamed on 16 GB: feasibility + p05 ceiling +
  per-user speeds at N = 1, 2, 4. The headline table.
- **E4 — Labeled approximation arms (not default):** Edge0-style prerouter and low-bit
  replica streaming, run under our protocol, published as the "price of exactness" figure
  (speed gained vs quality/token-exactness lost, including their AIME −6 to −10 pts class
  effects vs our exact engine).
- **E5 — Agentic arm (if time):** semantic anchors + Leyline-style KV splice on a scripted
  tool-calling trace; TTFT per turn. Positions P2 in the agentic conversation.

### Phase 3 — Analysis & write-up
- Paper skeleton already implied: crossover overlays (E1), decay curves (E2), frontier
  table (E3), price-of-exactness figure (E4), ecosystem comparison (E0.5).
- P1 consistency actions: scope the predictive-prefetch negative (training-free + exact +
  single-SSD), update Edge0 citation to arXiv:2609.18063.

## 5. Deliverables per experiment (rigor, unchanged from P1)

Frozen raw logs + hash manifest → p05 re-score script → derived JSON → claim-map entry →
semantic-map entry (what the number MEANS vs prose). Negative results are first-class.
No number enters the paper without a frozen artifact.

## 6. Claim & semantic maps (owner requirement, from day one)

- `claim_map_P2/`: one YAML per claim — claim text, frozen source, extraction script,
  verification status. Created when the experiment is designed, updated when numbers land.
- `semantic_map_P2/`: per-claim interpretation check — direction, baseline, conditions,
  what would falsify the prose. Reviewed at each phase boundary (P1 lesson: prose drifts
  from data silently).

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Qwen3.5-122B fails pre-flight (rope/arch) | Pre-flight check FIRST (cheap); fallback GLM-4-MoE or Qwen3.5-27B |
| SSD space for 65 GB checkpoint | Owner confirms free space before download |
| LRU doesn't lift the ~2× cap | Publishable negative — bounds FreeToken's claims; plan survives |
| Edge0/SlotStream don't build on M1 Pro | Document; substitute their published numbers with clear cross-machine caveats |
| mmap arm uncontrollable (page cache opaque) | Measure via RSS/fault counters; if untestable, report as limitation |

## 8. Immediate next actions

1. Owner: confirm ~70 GB free SSD.
2. Phase 0: Stage-0 re-validation; clone Edge0 + SlotStream; pre-flight Qwen3.5-122B.
3. Begin claim map skeleton for Phase 1 mechanisms.
