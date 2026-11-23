# P2 Literature Check — "FreeToken on Apple"

**Date:** 2026-11-21 · **Method:** Scholar plugin search (3 queries, 50 hits, CSVs in `literature/`) + targeted web search (2 queries) · **Scope:** 2024–2026, MoE serving on edge/unified memory, expert caching/prefetch, agentic KV reuse, mixed-precision experts.
**Status:** DRAFT — feeds into EXPERIMENT_PLAN.md revision. Raw data: `literature/s1..s3_*.csv`; claims below trace to those rows or to the URLs cited inline.

---

## 0. Headline findings

1. **The field moved fast since P1's lit review (Sep 2026).** Expert caching is now a crowded
   cluster (10+ systems). Our P2 must be positioned as a *mechanism-transfer + unified-memory*
   study, not "yet another expert cache."
2. **⚠ Direct overlap found: TinyTitan** (Borchert 2026) — a Metal/SSD streaming engine on
   Apple Silicon decoding a ~96 GB, 125B MoE at **5.5 tok/s on a 24 GB MacBook Pro**. This is
   the same regime as our P1 >RAM result (18 GB on 16 GB at 2.1 tok/s) with a better number.
   Must obtain, verify, cite, and ideally benchmark against. **This is the biggest threat to
   P2's novelty and the first thing to check.**
3. **Three new adoptable idea clusters** beyond FreeToken's three mechanisms (§2–§4).

---

## 1. Cluster map (what exists)

### A. Expert caching & replacement (the crowded core)
| System | Year | Idea | Relevance to P2 |
|---|---|---|---|
| FreeToken (arXiv:2608.16157) | 2026 | Global LRU + q\* miss splitting + double-buffered prefill + semantic anchors | Base system we port from |
| MoE-Infinity (arXiv:2401.14361) | 2024 | Trace-driven sparsity-aware cache, personal machines | Baseline/citation |
| SliceMoE (DAC'26, arXiv:2512.12990) | 2026 | Bit-sliced caching: cache experts at *slice* granularity, mixed precision expands capacity; **Predictive Cache Warmup during prefill** | New idea N1 |
| MoE-CORE (arXiv:2610.01950) | 2026 | Layer-wise dynamic cache-budget allocation | Possible refinement of M1 |
| Diff-MoE (2025) | 2025 | Priority-driven differential caching, layer-specific locality | Citation |
| FlashMoE (arXiv:2601.17063) | 2026 | ML-based cache replacement to cut SSD I/O | New idea N2 (SSD-tier) |
| OD-MoE (arXiv:2512.03927) | 2025 | **Cacheless** on-demand loading — counterpoint: caches underutilized | Negative-test arm? |
| "Who Should Own the Expert Cache?" (arXiv:2608.12103) | 2026 | **OS page cache (kernel) as the expert tier**, router traces on 128–896-expert models | Very Apple-relevant: mmap path vs explicit cache — cheap experiment, ties to P1's llama.cpp mmap caveat |
| CFMoE (2026) | 2026 | Cache-friendly serving on edge unified memory | Close neighbor — read fully |
| ExpertFlow (DAC'26) / adaptive ExpertFlow (arXiv:2510.26730) | 2025–26 | Predictive caching + token scheduling | Already cited in P1 |

### B. Prediction & prefetch (extends P1's negative)
| System | Idea | Relevance |
|---|---|---|
| "The Other Half of the Memory Wall" (arXiv:2609.18063) | **35B MoE from SSD via trained routing prediction**; claims post-trained MoE tolerates *halved* expert count | Directly challenges P1's "prediction doesn't pay" — on SSD with a *trained* predictor. Must engage. |
| SP-MoE (arXiv:2510.10302) / MoE-SpeQ (arXiv:2511.14102) | Speculative decoding ↔ expert prefetch co-design; draft model predicts experts | New idea N3: speculative decode as prefetch oracle — draft token tells you next experts for free |
| PROBE (arXiv:2602.00509) | Real-time predictive prefetch, compute/comm co-balance | Citation |
| Yu et al. (arXiv:2611.04794) | **Interleave concurrent requests to hide offloading latency**; notes prefetch harmful under contention | Directly relevant to our concurrency framing — prediction hurts under contention (supports P1) |
| Prefetch-Steered Routing (arXiv:2610.34721) / CECAR (arXiv:2611.20119) / cache-aware router adaptation (arXiv:2609.04895) | Nudge routing toward resident experts (deterministic, cache-aware) | Optional idea N4 — changes outputs; conflicts with our exact-output gates, treat as "relaxation" arm only |

### C. Mixed-precision / low-bit experts (bandwidth lever)
| System | Idea | Relevance |
|---|---|---|
| HOBBIT (arXiv:2411.01433) | Fetch **low-precision replicas** of missed experts; mixed-precision miss-penalty cache | New idea N5 — on single-SSD Apple, fetching a 2-bit replica is 2× cheaper than 4-bit |
| PagedMoE (arXiv:2611.03173) | Fine-grained 1–8-bit experts, DRAM bitplane cache + CPU execution, quality safeguard under bit budget | Strong design point for N5 |
| MoE-APEX / GEMQ (ICML'26, arXiv:2605.23078) | Per-expert bit allocation by importance, down to 1.5-bit + router fine-tune | Offline complement; router fine-tune conflicts with exactness — cite |
| D2MoE, DyMoE | Token-adaptive bit-width | Citations |

### D. Apple Silicon / unified memory (our home turf)
| Work | Idea | Relevance |
|---|---|---|
| **TinyTitan (Borchert 2026, ResearchGate)** | Metal SSD/NVMe streaming, 125B at 5.5 tok/s on 24 GB MBP | ⚠ See §0.2 — verify claims, benchmark head-to-head |
| BaseRT (arXiv:2607.00501, 2607.19438) | Native Metal runtime, chip-specific kernel fusion; M5 Neural Accelerators | Hardware note: M5 has neural accelerators in GPU — changes the bandwidth/compute balance our q\*-degeneracy argument assumes |
| NPU MoE serving (arXiv:2604.18788) | MoE on Apple Neural Engine | Already cited in P1 |
| EdgeAgent (arXiv:2610.03394) | Multi-agent orchestration on M4 unified memory | Agentic regime on Apple — cite for E3 |
| SiliconBench (arXiv:2609.19169) / Silicon Showdown (arXiv:2605.00519) | Benchmarks of unified-memory serving incl. concurrency readiness | Supports our p05 metric choice |
| Benazir & Lin (arXiv:2508.08531; POMACS 2025) | Profiling LLM inference on Apple Silicon | Background citations |
| Yemets et al. (arXiv:2610.31479) | "LLM inference optimizations carry no [benefit?] on Apple" | Likely supports transfer-not-guaranteed thesis — read fully |

### E. Agentic KV / context reuse (for M3)
| Work | Idea | Relevance |
|---|---|---|
| **Leyline (arXiv:2606.01065)** | Agent issues (span, replacement) edits; engine **splices KV in place** with δ-rotation re-anchoring, no re-prefill | Stronger than FreeToken's anchors for full-attention models — candidate upgrade to M3 |
| Continuum (arXiv:2511.02230) | KV TTL across tool-call gaps | Simple arm for E3 |
| CacheScout (arXiv:2608.14624) | Proactive agent-aware KV management, transition learning | Citation |
| IntentKV (arXiv:2606.09916) | Position-stable eviction preserving radix-prefix reuse | Citation |
| InferCept, Autellix, KVFlow, Sutradhara | Tool-gap KV management, program-level scheduling | Background |

### F. Energy & cost axis (optional differentiator)
- Kyung et al. 2025: SSD offloading *harmful for energy efficiency* — pairs with our
  feasibility-vs-speed honesty; Apple laptop battery angle is untested by FreeToken.
- Greenbench 2026: energy/carbon on M4 Pro. Possible E5 (cheap to add: watts during streaming).

---

## 2. New adoptable ideas (candidates to add to the plan)

| # | Idea | Source | Fits | Cost | Value |
|---|---|---|---|---|---|
| **N1** | Cache warmup during prefill (seed decode cache from prefill routing) | SliceMoE PCW; FreeToken also seeds from prefill buffers | M1 | Low | Medium |
| **N2** | SSD-tier aware replacement: evict to RAM vs re-fetch from SSD by cost | FlashMoE; "Who Owns the Cache" (mmap/kernel tier) | M1 | Low–Med | High on Apple — llama.cpp gets this free via mmap; we should measure explicit-LRU vs mmap-page-cache |
| **N3** | Speculative decoding as expert-prefetch oracle (draft tokens predict next experts) | SP-MoE, MoE-SpeQ | E2/E4 | Med | High — could turn P1's predictive-prefetch negative into a win (prediction becomes free) |
| **N4** | Cache-aware routing (route to resident experts when scores close) | CECAR, Prefetch-Steered Routing | Separate arm | Med | Medium — conflicts with exactness gates; report with perplexity delta |
| **N5** | Low-precision replicas for misses (fetch 2-bit instead of 4-bit on a miss) | HOBBIT, PagedMoE | E2 | Med–High | High for >RAM streaming (halves SSD bytes) — but needs quality gate |
| **N6** | Interleave concurrent requests to hide offloading | Yu et al. | E1/E4 | Low (we have batching) | Medium |
| **N7** | KV splicing for edited contexts (Leyline-style) instead of pure checkpoints | Leyline | M3/E3 | Med | High — better than anchors for full-attention models |

## 3. Threats & must-engage items

1. **TinyTitan** — verify its 5.5 tok/s @ 125B claim; if real, P2 must explain the difference
   (subject model? activation size? measurement?) or adopt its techniques. *Action: obtain the
   PDF, check methodology, try to reproduce contact/author info.*
2. **"Other Half of the Memory Wall"** — trained predictor on SSD serving of 35B. If their
   trained predictor pays on SSD, our P1 negative must be scoped as "*training-free* prediction
   doesn't pay on single SSD" — a correction/scoping note, honestly disclosed.
3. **CFMoE + EdgeAgent** — closest Apple neighbors; ensure our novelty statement excludes them.
4. **Yemets et al. 2610.31479** — if it shows standard optimizations don't transfer to Apple,
   it's our best supporting citation for the whole P2 premise.

## 4. Proposed EXPERIMENT_PLAN updates (pending owner confirmation)

- Add **E0 (replication/threat check): TinyTitan + Other-Half verification** before building.
- M1 (LRU) gains arms: static-pin vs explicit-LRU vs **mmap kernel tier** (N2) — the Apple-specific three-way.
- E2 gains arms: **N3 speculative-decode oracle** and optionally **N5 low-precision replicas** (with perplexity gate).
- M3/E3: implement anchors **and** Leyline-style KV splice (N7); compare on agentic trace.
- Optional **E5**: energy-per-token during streaming (cheap, differentiating).
- Rigor unchanged: claim map + semantic map started at E0, every arm gated.

## 5. Reading queue (full text, priority order)

1. TinyTitan (ResearchGate PDF)
2. arXiv:2609.18063 (Other Half of the Memory Wall)
3. arXiv:2610.31479 (Yemets — optimizations on Apple)
4. arXiv:2608.12103 (Who Should Own the Expert Cache)
5. arXiv:2512.12990 (SliceMoE)
6. arXiv:2606.01065 (Leyline)
7. arXiv:2611.04794 (Interleaving concurrent requests)
8. CFMoE (dl.acm.org 10.1145/3795866.3844455)
