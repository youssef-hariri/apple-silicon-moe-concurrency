# E0 — Threat Verification & Novelty Re-Check

**Date:** 2026-11-21 · **Status:** COMPLETE (reading phase) · feeds EXPERIMENT_PLAN.md revision
**Inputs:** arXiv:2609.18063 full text; TinyTitan metadata; web verification of Apple SSD-streaming ecosystem.

---

## 1. TL;DR

- **The biggest "threat" is Edge0 — and it is the SAME Edge0 we already cited in P1** as
  "a project rather than a paper." It now has a full paper (AutoArk, Sep 2026, arXiv:2609.18063,
  "The Other Half of the Memory Wall"). It runs on **Apple Silicon (MLX)** — our platform.
- **TinyTitan is not alone:** there is now a visible ecosystem of Apple SSD-streaming MoE
  engines (Edge0, SlotStream, SwiftLM). Our P1 >RAM feasibility claim ("llama.cpp OOMs, ours
  streams") has been independently overtaken by faster systems.
- **What survives as P2 novelty:** multi-user concurrency under tail SLO, the crossover law,
  exactness gates, and the mechanism-transfer question. Nobody in this ecosystem does
  multi-tenant serving. P2's center of gravity must shift from "streaming speed" to
  "**concurrent** serving + mechanism transfer."

---

## 2. Edge0 (arXiv:2609.18063) — detailed assessment

**System:** MLX-based streaming engine; experts live on SSD, mmap-streamed; OS page cache +
LRU + fixed-slot double buffer.

**Key mechanisms:**
1. **Prerouter** — trained per-layer head predicts next layer's routing *one full token ahead*;
   the prediction **is consumed as the routing** ("prediction as routing" — staged set = routed
   set, nothing dropped). They price the alternative: same-token pre-gating (Pre-gated MoE
   style) costs 30–100 ms/step of GPU pipeline drain and *fell below a plain LRU baseline* in
   their measurements.
2. **Recovery LoRA, served unmerged** — pays back int4 + routing-replacement quality loss;
   merging+requantizing destroys 82–98% of the delta. Trained: head distillation → SFT on
   student path (~2M rows) → on-policy reverse-KL distillation (~200k rows).
3. **incr_stack** — sticky-slot tensors updated in place; +34% decode by removing per-step
   assembly.

**Numbers (their protocol, same-session rotated A/B, medians):**
- 35B MoE (Qwen3.6-35B-A3B int4, 19.5 GB) on 24 GB Mac mini M4 Pro: **20.4 tok/s in 2.9 GiB**
  active memory; vanilla mlx-lm fully-resident: 3.9 tok/s in 18.2 GiB (pages badly).
- 16 GB MacBook M2, 18.4 GiB checkpoint (does not fit): prerouter +80–84% decode over
  on-demand streaming (4.8→8.6, 3.5→6.4, 1.8→3.3 tok/s at K=2,4,8).
- Quality: within 3.9 (35B) / 2.8 (8B) mean OpenCompass points of fp16 teacher; **weakness:
  long-chain reasoning (−6.1 AIME on 35B, −10.0 on 8B)**.

**Their own stated limitations (gold for us):**
- **"Edge0 serves one request at a time, FIFO-serialized. Concurrency belongs to a serving
  layer… batching changes the expert working set in ways our per-request profiles do not
  model."** ← This is exactly our P1/p05 home turf.
- Decode floor is **CPU-side graph building (44 ms/step)**, not storage.
- Adjacent tokens share only ~25% of a layer's expert set → prefetch pays once, reuse limited.

**Relation to our P1 negative (predictive prefetch doesn't pay):**
- Edge0 does NOT refute us — it *reframes* us. Our negative was **training-free, cross-step,
  output-exact** prediction on a single SSD with the native router authoritative. Edge0 pays
  for prediction with **training** and gives up output exactness (routing is replaced, quality
  recovered by LoRA). Two different contracts. P2 must scope the P1 claim explicitly:
  *"training-free prediction under an exact-output contract doesn't pay on single-SSD
  consumer hardware"* — and then test whether Edge0-style trained prediction is worth its
  exactness price under OUR gates (perplexity band, greedy match will fail by construction).

---

## 3. The wider Apple SSD-streaming ecosystem (new since P1)

| System | Claim | Hardware | Notes |
|---|---|---|---|
| **Edge0** (arXiv:2609.18063) | 35B @ 20.4 tok/s, 2.9 GiB | 24 GB M4 Pro | Paper-verified; trained prerouter; single-request |
| **SlotStream** (v0.2.3, Sep 2026) | 125B (Qwen3.8-Flash-Next, 103.8 GB) @ ~12 tok/s warm; 6.22 tok/s on 32 GB M5 Air; 1.41 tok/s on 16 GB M2 | Swift/MLX | Bounded expert pool; 3.8 GB resident floor; SSD reads 17.3 GB/s sustained; **zero concurrency claims** |
| **SwiftLM** (SharpAI, Nov 2026) | SSD streaming 10× (0.58→5.91 tok/s) on 122B+ with ~10 GB resident | Swift/MLX | Cross-projection batching, concurrent pread QD=24, asyncEval pipeline; +TurboQuant KV, +speculative decoding (auto-capped to 1 draft token when streaming!) |
| **TinyTitan** (Borchert 2026) | ~96 GB 125B @ 5.5 tok/s | 24 GB MBP | ResearchGate only, PDF 403 — could not verify; likely same design family as above |

**Pattern across all four:** mmap/page-cache + bounded expert pool + I/O overlap; speeds
5–20 tok/s on >RAM models; **all single-user, no SLO, no concurrency, no tail metrics,
exactness only in Edge0 (element-wise ≤0.24% L2, not token-exact)**.

**Notable detail:** SwiftLM *caps speculative decoding at 1 draft token when SSD streaming* —
SSD I/O fan-out kills spec-decode under streaming. This nuances our N3 idea (spec-decode as
prefetch oracle): the oracle may still work for *expert* prefetch (predict sets, not tokens)
even where token-level spec-decode is capped.

---

## 4. Novelty re-check for P2 (post-E0)

| P2 element | Status after E0 |
|---|---|
| LRU vs static pinning crossover on Apple | ✅ Still novel — no one reports multi-user ceilings or ratio-swept crossover |
| p05 multi-tenant SLO methodology | ✅ Unique — Edge0/SlotStream/SwiftLM explicitly single-request |
| >RAM streaming speed alone | ❌ Dead as a headline — ecosystem is at 5–20 tok/s vs our 2.1 |
| mmap/kernel-tier vs explicit cache (N2) | ✅ Novel as a *controlled A/B* — all four systems use mmap implicitly; nobody isolates it |
| Double-buffered prefill port | ⚠ Partially anticipated (Edge0 whole-layer prefill path; FreeToken) — keep as engineering arm, not headline |
| Semantic anchors / KV splice for agentic | ✅ Novel on Apple — EdgeAgent touches orchestration, not KV mechanisms |
| Trained prerouter under exactness gates | ✅ Novel comparison — Edge0 never measures token-exactness; our gates would quantify the exactness cost of prediction-as-routing |
| Spec-decode as expert-prefetch oracle (N3) | ✅ Novel — SwiftLM caps token spec-decode; nobody uses draft routing as expert oracle |

**Revised P2 thesis:** *"The streaming-speed race on Apple Silicon is over (5–20 tok/s, three
independent engines). The open problem is concurrent, quality-gated serving: how do the new
mechanisms — trained prediction, dynamic caching, kernel tiering — interact with multi-tenant
load and exactness requirements on unified memory?"*

---

## 5. Required P1/P2 consistency actions

1. **Scope the P1 predictive-prefetch negative** in P2 (and optionally a P1 erratum note):
   training-free + output-exact + single-SSD. Edge0 changes the contract (trained, non-exact);
   cite it as the trained alternative, not a contradiction.
2. **Update P1 related-work pointer:** Edge0 is no longer "a project" — cite arXiv:2609.18063.
3. Add SlotStream, SwiftLM, TinyTitan to P2 related work as concurrent systems; verify
   TinyTitan PDF if it becomes accessible (currently 403; flagged unverified).

## 6. Plan-of-attack changes recommended (for owner approval)

1. **New E0.5 benchmark arm (cheap, high value):** reproduce Edge0 and SlotStream on our
   16 GB M1 Pro with OUR harness (p05, WildChat corpus) → same-machine comparable numbers,
   instant related-work credibility, and a defensible "we benchmark everyone on one protocol"
   table. Edge0 is open source (framework + checkpoints + adapters).
2. **Shift P2 headline from streaming speed to concurrency:** E1 (LRU vs static vs mmap
   three-way, multi-user) becomes the centerpiece; the streaming arms become "we match the
   ecosystem's single-user regime and exceed it in multi-user."
3. **Keep N3 (spec-decode expert oracle)** — strengthened by SwiftLM's cap finding.
4. **Add exactness-vs-speed frontier arm:** our exact engine vs Edge0-style approximate
   (if license/code permits) under one quality protocol — the "price of exactness" figure.

## 7. Reading queue status

- ✅ arXiv:2609.18063 (Edge0) — read in full
- ✅ TinyTitan — attempted, 403, metadata only (unverified)
- ⏭ Remaining queue deprioritized except: arXiv:2610.31479 (Yemets, transfer-to-Apple
  evidence) and arXiv:2608.12103 (kernel-tier argument) — read during E1 design.
