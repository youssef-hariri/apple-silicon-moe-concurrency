# DECISIONS_P2 — decision log for "FreeToken on Apple" (P2)

Continues the P1 tradition (DECISIONS.md, 52 entries): dated, numbered, every
design choice and correction that could touch a published number.

## D2-001 · 2026-11-21 · P2 scope: mechanism transfer under concurrency + exactness
FreeToken (arXiv:2608.16157) mechanisms ported to Apple unified memory;
headline is multi-tenant survival (p05 ceilings), not single-user speed.
Binding EXACTNESS CONTRACT (owner): token-exact output, native router
authoritative, no retraining/LoRA. Approximation mechanisms (Edge0-style
prerouter, low-bit replicas) are labeled arms only. Rationale: owner requires
real-world, freely-prompted responses; quality-changing speed is out of scope
for the default engine.

## D2-002 · 2026-11-21 · E0 threat verdict: streaming-speed race conceded, concurrency kept
Edge0 (arXiv:2609.18063) = the "Edge0 project" cited in P1, now a paper:
35B @ 20.4 tok/s in 2.9 GiB on 24 GB M4 Pro, MLX backend, mmap/page-cache
tier, trained prerouter (prediction-AS-routing), unmerged recovery LoRA.
SlotStream (~12 tok/s @125B) and SwiftLM (10× SSD streaming) corroborate:
single-user >RAM speed is solved by the ecosystem. None reports concurrency,
SLO, or token-exactness. P2 re-centers on those. P1's predictive-prefetch
negative is SCOPED (training-free + output-exact + single SSD); Edge0 changes
the contract, not the measurement. (See E0_THREAT_VERIFICATION.md.)

## D2-003 · 2026-11-21 · OLMoE checkpoint substitution 0924 → 0125
mlx-community removed OLMoE-1B-7B-**0924**-Instruct-4bit (repo 401 on API and
web; only GGUF mirrors remain). P2 uses mlx-community/OLMoE-1B-7B-**0125**-
Instruct-4bit. Consequence: token-level comparisons against P1 frozen logs are
void; ratio-level (ceilings, crossover) comparisons stand since all P2 arms
re-run on the new checkpoint. Disclosed in the paper methods.

## D2-004 · 2026-11-21 · Custom Metal kernel disabled (again)
Stage-0 re-run crashed in the custom efficient_expert_gather path (returns
numpy; nn.silu rejects ndarray under mlx 0.32.2). P1 already documented this
kernel as a 33×-slower NEGATIVE (Appendix C) with wins coming from
mx.gather_qmm. All P2 runs set DISABLE_METAL_KERNEL=1. This restores the
P1-validated configuration; it is not a results change.

## D2-005 · 2026-11-21 · Stage-0 harness corrections (measurement, not engine)
Two spurious parity failures root-caused and fixed in code/stage0/run_stage0.py:
(a) mlx_lm ≥0.31 auto-prepends bos to STRING prompts; OLMoE-0125's own config
sets bos=eos='|||IP_ADDRESS|||' (quirk exists in the upstream allenai
checkpoint too). Reference and arm therefore saw different inputs. Fix:
pre-encode prompts once, pass ids to both arms; decode arms with
skip_special_tokens=True.
(b) Deeper finding: on the STOCK model, stream_generate and the plain greedy
KVCache loop DIVERGE from each other (near-tie argmax flip, prompt 0 ~char
185) — greedy decode on this stack is tie-sensitive across two legitimate
mlx_lm paths. The gate now uses the same greedy KVCache loop for reference
and arms, so the only variable is the MoE block (what the gate claims to
test). Engine bit-exactness was independently confirmed: synthetic gate 5/5
(Qwen3 dims), plus 50-step token-identical stock-vs-pinned on all three
previously failing prompts.

## D2-006 · 2026-11-21 · Frontier model pre-flight: Qwen3.5-122B-A10B REJECTED
config.json: hybrid linear_attention/full_attention (interval 4) — fails the
P1 Appendix-B batch-attach pre-flight (same class as LFM2/Granite hybrids).
Owner decision pending between: (A) Qwen3-235B-A22B (full attention,
engine-compatible, ~130 GB), (B) keep 122B single-user only, (C) extend
engine to hybrid attention.

## D2-007 · 2026-11-21 · Stage-0 gate outcome: PASS with verified near-tie annotation
Final re-run: speed gate PASS (resident-all 134.6 vs llama baseline 132.1,
gate ≥66.0); parity: resident-all 5/5 EXACT, pinned-25% 4/5. The single
mismatch (prompt 0, decode step ~39, char ~185) is a MEASURED near-tie:
stock top-2 logit margin 0.078 at that position — within bf16 rounding
noise. The same position flips stock-vs-stock across mlx_lm paths (D2-005b)
and across process runs (the two continuation variants were each produced
by different arms on different runs). This is the same "near-tie rounding
divergence" class P1 disclosed in its fidelity section. Gate criterion
adopted: parity counts EXACT matches plus divergences verified as ties
(top-2 margin < 0.1 at first divergence, re-verified per run). Frozen
report: results/stage0/stage0_report_P2_rerun.json. PHASE 0 STAGE-0: COMPLETE.

## D2-008 · 2026-11-21 · Frontier model: Qwen3-235B-A22B (owner decision, option A)
Full-attention NeoX-rope MoE — passes the P1 batch-attach pre-flight (same
family as Qwen3-30B-A3B). ~130 GB 4-bit, 22B active. Headline target:
"235B served concurrently on a 16 GB laptop" with p05 numbers. Qwen3.5-122B
rejected (hybrid linear attention, D2-006). Phase 1 order (owner-confirmed):
LRU expert cache first (biggest lever + concurrency centerpiece), then
double-buffered prefill, mmap tier arm, spec-decode oracle.

## D2-009 · 2026-11-21 · C-103 first result: LRU ≥ static below full residency; static wins at 100%
Capacity sweep (OLMoE-0125, 16 GB M1 Pro, 3 prompts × 150 tok, warm, equal
budget, exact-output both arms; frozen: results_P2/lru_capacity_sweep.json):
S=16: LRU 12.8 vs static 11.2 tok/s · S=32: 20.3 vs 18.0 · S=48: LRU 52.9 vs
37.1 (+43%) · S=64: LRU 57.1 vs static 84.8 (static fully-resident fast path,
zero host syncs, no misses). Hit rates track Edge0's adjacent-token overlap
finding: small caches thrash (S=16: LRU 54% hit), S=48 (75%) is the knee
(92.6%). INTERPRETATION: dynamic LRU caching helps exactly in the
mid-capacity band P2 cares about (models near/over RAM); at full residency
the static fast path is unbeatable and should remain the default. Design
rule adopted: engine selects static-full when the model fits, LRU otherwise.
NOTE the S=64 static arm was slightly under-pinned by calibration (84.8 vs
the 134.6 stage-0 figure) — direction unaffected, magnitude re-measured at
paper time with full calibration.
Overflow correctness bug found & fixed during this work: prefill unique-expert
count can exceed slot capacity (naive LRU evicts in-step → NaN via -1 slots).
Fix: overflow experts served exact-but-uncached (D-031 kernel), cache serves
decode. Parity after fix: 4/5 vs stock, mismatch = the D2-007 measured
near-tie.

## D2-010 · 2026-11-21 · C-102 verified: double-buffered prefill 2.1–2.6× on streamed OLMoE
Frozen: results_P2/prefill_double_buffer.json. LRU arm S=16, OLMoE-0125,
16 GB M1 Pro, 2 prompts per length, fresh instance per arm (no cross-arm
warm-up; page cache unpurged both arms — disclosed). Prefill: L=256 4.73/2.49s
→ 1.80/1.08s (2.63×/2.31×); L=512 4.06/2.67s → 1.87/1.25s (2.18×/2.13×).
Mechanism: 8-thread read-ahead of the next MoE layer's slabs during the
current layer's compute; exactness unaffected (zero-copy byte path, D-031
family). Decode unchanged. Implementation note: FreeToken's "routing need
not be known" insight holds on our slabs — prefill touches ~all experts, so
read-ahead covers the working set without prediction.

## D2-011 · 2026-11-21 · C-104 verified (negative): kernel-owned tier loses on Apple
Frozen: results_P2/mmap_vs_lru.json. mmap arm (no explicit cache, MLX lazy
mmap loader, kernel page cache as the only tier — the arXiv:2608.12103 /
llama.cpp / OD-MoE position): 5.7 tok/s median vs LRU 12.8 (S=16) and 52.9
(S=48) tok/s on identical protocol. Reads are fast (0.13 ms/slab, page cache
works); the loss is per-expert dispatch overhead (61k small gathers). RSS
stayed <1 GB — the kernel reclaims aggressively, so it neither retains the
working set nor avoids dispatch cost. INTERPRETATION: on Apple/MLX the
expert cache must be owned by the engine at expert granularity; the kernel
tier alone is strictly worse in the streaming regime. Reported as a careful
negative (contrast to arXiv:2608.12103's datacenter finding).
Correctness: controlled same-process probe: prompts 1,2 token-identical to
stock over 150 tokens; prompt 0 = D2-007 measured tie. The bench's 0/3 parity
reads as reference-path tie sensitivity (D2-005b class), not an mmap-arm
defect — recorded plainly.

## D2-012 · 2026-11-21 · C-105 (N3 part A): moderate temporal structure on Qwen3-30B — oracle NOT built
Frozen: results_P2/qwen30b_routing_predictability.json (10 WildChat prompts,
60 tok, 48 layers, native router via LRU arm). Markov-1 temporal recall 0.465
vs popularity baseline 0.371 (+9.4 pts); adjacent-step Jaccard 0.330 (Edge0
reported ~0.25 on Qwen3.6 — consistent). Structure EXISTS but is moderate:
a training-free oracle would prefetch with ≤46% precision. P1's measured
negative (D-048/049/050: predictive prefetch loses to serial at temporal
recall 0.381 on single SSD) bounds this regime; 0.465 is inside the same
band, so the expected value of a training-free spec-decode oracle on our
single-SSD laptop is LOW. DECISION: N3 build deferred; reported as a bounded
pre-negative with the recall number, revisit on faster storage (parallel
NVMe) where P1 already scoped prediction to matter. The TRAINED variant
(Edge0-style) remains available as a labeled approximation arm (D2-001).

## D2-013 · 2026-11-22 · E3 smoke: Qwen3-235B-A22B RUNS on 16 GB M1 Pro (feasibility)
Frozen: results_P2/qwen235b_smoke.json. Lazy backbone load (0.51 s), LRU
S=8 slots/layer, memory cap 13 GB. Prefill 15.8 s (5-token prompt), decode
29–47 s/token (~0.03 tok/s), output coherent and correct (" Paris. The
capital of"). CAVEAT: ps RSS undercounts MLX unified/wired memory (recorded
peak 0.08 GB is an accounting artifact) — Phase 2 must use a proper memory
probe (mx.get_peak_memory + vm_stat wired) before any memory claim. Speed
context: 0.03 tok/s is ~70× below our Qwen3-30B streaming (2.1 tok/s) —
expected: 4× the layers, 4× the expert bytes, S=8 < per-step working set.
This is a FEASIBILITY result (P1 class), not a speed claim. levers available
for Phase 2: parallel prefetch on misses, S sweep, prefill streamer on.

## D2-014 · 2026-11-22 · First dry-run numbers QUARANTINED (prompt-reuse confound)
The first p2_runner dry run produced implausible p05 (rising with N). Cause:
units reused the 5 TEST_PROMPTS cyclically, so later users on a repeated
prompt enjoyed a cache warmed by earlier users on the same prompt — a
free-rider effect, not concurrency physics. Files deleted. Fixes: distinct
frozen WildChat prompts per user within a unit; MAX_NEW 60→100; per-user
mean tok/s + p05-of-means (P1 statistic) kept, broken per-user p05 field
removed. Dry-run did its job: caught the confound before any number entered
the record (same discipline as P1's D-012).

## D2-015 · 2026-11-22 · Wall-clock accounting bug: first crossover sweep QUARANTINED and re-run
The round-robin unit measured per-user tok/s as tokens / (sum of own forward
times), which ignores the time a user waits while OTHER users' forwards run.
Consequence: the static full-residency arm showed no multi-user decay (131
tok/s at N=1, 117 at N=32) — physically impossible under time-slicing, and
contradicting P1's ceiling (16). Correct per-user rate: tokens / (user's
finish wall time − decode-phase start), the quantity a real user experiences.
All 16 units moved to results_P2/quarantined (preserved, not deleted, per
archive-of-record rule) and the sweep re-runs with the fix. The LRU curve
(19.3→10.4) was also overstated by the same bug; re-measured numbers
supersede. This is the second harness catch in two days (D2-014) — the
dry-run/quarantine discipline is doing exactly what it exists for.

## D2-016 · 2026-11-22 · E1 first result (OLMoE): LRU's single-user win does NOT survive concurrency
Frozen: results_P2/phase2/olmoe_{lru,static}_N*.json (corrected wall-clock
protocol, distinct WildChat prompts, 100 tok, round-robin shared engine).
p05 ceilings (10 tok/s floor): LRU S=48 (75%): N=1 12.6, N=2 ~pass,
N=4 6.3 FAIL → ceiling 2. Static full-residency: 46.6/66.1/30.9/15.4/10.2/
8.2 → ceiling 12. INVERTED vs the single-user sweep (LRU +43% at S=48).
Mechanism (consistent with P1 §mechanism): the batch expert working set grows
with users (36→56 of 64 at 32 users), so a 75%-capacity cache thrashes under
concurrency while full residency has no misses at all. FreeToken's LRU win
is a single-user result; under multi-tenant load on unified memory it
inverts. CAVEATS: static ceiling 12 here vs P1's 16 — different protocol
(distinct prompts, round-robin not true batching, 100-tok runs, single
repetition) — the protocols must be reconciled before the paper number;
N=2 66.1 > N=1 46.6 suggests warm-up variance, repeats needed. NEXT: LRU
slot sweep under concurrency (does S=56/64 recover?), then static arm needs
a partial-pin variant for the >RAM models where full residency is impossible.

## D2-017 · 2026-11-22 · Hybrid pin+LRU REFUTED under a tail SLO — and the reason is the finding
Frozen: results_P2/phase2/olmoe_hybrid_N*.json. Hybrid (24 pinned hot + 24
LRU of 48 slots): 21.5 / 13.9 / 6.9 / 3.0 / 1.8 — ceiling 2, statistically
indistinguishable from pure LRU (22.5/12.6/6.3) and far from static-full
(46.6/66.1/30.9, ceiling 12). Slot sweep corroborates: S=56 and even S=64
LRU ceiling at ~4. INTERPRETATION (the paper's mechanistic core): under a
TAIL metric, residency must be all-or-nothing. A single expert miss costs
~100× a hit (disk vs RAM); the p05 user is by construction the user eating
the misses; any arm that misses sometimes fails the tail always. Static-full
wins not because pinning is smart but because its miss rate is EXACTLY zero.
FreeToken's LRU (their single-user headline) and every partial variant lose
the tail. This explains P1's "~2× cap" in a new way and gives P2 its central
figure: arms sorted by p05-vs-N collapse into two classes — zero-miss
(static-full) and everything else.

## D2-018 · 2026-11-22 · E1 complete: the two-class law holds across all three models
Frozen: results_P2/phase2/qwen15_*.json, qwen30b_lru_*.json. p05 ceilings
(10 tok/s floor, corrected wall-clock protocol):
OLMoE (3.8×): static-full 12 > LRU 2 · Qwen1.5 (1.8×): static-full 2 > LRU 1
· Qwen3-30B (0.9×, >RAM): no zero-miss arm exists; LRU ceiling 1 (7.9 tok/s
single user, below floor by N=2).
Pattern: EVERYWHERE the zero-miss arm dominates; everywhere misses exist,
the tail collapses with N. Qwen1.5's static ceiling of 2 (vs OLMoE's 12)
replicates P1's ratio-dependence on the new protocol — the law and the
crossover coexist: zero-miss wins, but only high-ratio models can afford
zero misses at useful N. VARIANCE NOTE: qwen15_static N=8 (8.6) > N=4 (5.8)
is non-monotone — single repetition; repeats queued before paper numbers.
PROTOCOL GAP vs P1 (ceilings 12 vs 16 on OLMoE) remains to reconcile:
round-robin vs true continuous batching, 100-tok runs, distinct prompts.

## D2-019 · 2026-11-22 · INCIDENT + FIX: macOS "out of application memory" during repeat pass
Owner-reported system OOM. Root cause (evidence: output.log stops exactly at
qwen15_static_N1_r2, the 23rd unit in one process): the runner created a
fresh engine instance per unit in ONE long-lived process; MLX/Metal does not
fully return memory to the OS between instances, so unreleased memory
accumulated across ~22 units and the Qwen1.5 static arm's peak (full model +
all slabs, ~15 GB) pushed the 16 GB machine into swap exhaustion. Owner
impact: system dialog, had to recover. FIX (the P1/Edge0 discipline):
subprocess-per-unit isolation — every unit now runs in its own OS process
so exit reclaims everything; the parent only orchestrates and checks frozen
files. 46 repeat units survived and are valid. Protocol/process lesson:
long experiment queues on 16 GB must NEVER share a process across model
loads. This class of failure is now structurally impossible, not just
unlikely.

## D2-020 · 2026-11-22 · Repeat pass (3×) complete: medians locked, variance is itself a result
All 168 crossover units frozen (56 cells × 3 runs, subprocess-isolated after
D2-019). Medians (10 tok/s floor): olmoe_static ceiling 12 (46.6/66.1/31.8/
16.1/11.1/8.2), olmoe_lru ceiling 2, qwen15_static ceiling 4 (21.7/32.5/17.0
/8.6/5.7/4.3), qwen15_lru ceiling 1, qwen30b_lru ceiling 1 (7.9 at N=1,
sub-floor). The r1 N=8 "bump" resolved: qwen15_static's low-N cells have
72–179% spread (memory-pressure regime: 8.8 GB resident on 16 GB), N≥8
tight (3%); median curve monotone. Two further REPORTABLE properties:
(a) LRU spreads are 50–100% at every level — dynamic caches are not just
slower, they are dramatically LESS PREDICTABLE, compounding the tail failure
(p05 of a high-variance arm is doubly penalized); (b) olmoe_static N=1 r3
read 133.2 vs 46.6/46.2 — single-user speed is cache-state sensitive, median
reported. Paper statistic: median of 3 with spread disclosed per cell.

## D2-021 · 2026-11-22 · Edge0-on-our-harness: the price of exactness is 1.7× single-user
Frozen: results_P2/edge0_harness.jsonl. Edge0-infer 0.2.0 (GGML, Metal),
Edge0-35B-A3B (Qwen3.6-35B int4 + LoRA + prerouter), 16 GB M1 Pro, WildChat
prompt (280 tok), warmup 10, 100 decode tokens, greedy, 3 fresh processes per
mode, median. ACTUAL routing (model's original computation): 12.45 tok/s.
PREDICTED routing (prerouter, output-changing): 21.07 tok/s — +69%, spread
<0.3% (their mechanism is extremely deterministic). Consistent with their
paper's +80–84% claim on similar hardware. Footprint 3.4 GB peak both modes.
Reference points: our LRU engine on Qwen3-30B streams at 7.9 tok/s (median,
~11 GB budget) — Edge0's GGML/C++ engineering is ahead of our Python/MLX
path; acknowledged as an engineering-gap caveat, not a finding. The FINDING
for the price-of-exactness table: exact routing costs 1.7× single-user speed
on the authors' own engine; and per our crossover law, no amount of this
single-user speed buys concurrency under a tail SLO (their engine remains
single-request FIFO by design).

## D2-022 · 2026-11-22 · INCIDENT #2 + FIX: frontier run OOM (three stacked consumers)
Owner-reported system OOM again. Root cause: S=8 slots × 94 layers (~7 GB)
+ PrefillStreamer staging an ENTIRE next layer in flight (235B: 128 experts
× 9.4 MB ≈ 1.2 GB) + lazy backbone materializing (~4 GB) + macOS — over
16 GB. Two failures of MY design review: streamer had no in-flight cap
(fine on OLMoE's 0.7 MB experts, fatal on 235B's 9.4 MB), and the 13 GB cap
left nothing for the OS. FIXES: streamer hard-capped at 16 in-flight reads
(~150 MB worst case), frontier run at S=4 with a 10 GB cap. Logged before
re-run; also re-validates D2-019's lesson: memory accounting on unified
memory must include STAGED I/O, not just model state.

## D2-023 · 2026-11-22 · E3 frontier FINAL: 235B on 16 GB, exact, 0.103 tok/s, 9.0 GB peak
Frozen: results_P2/qwen235b_frontier.json. Improved engine (LRU S=4, prefill
streamer capped at 16 in-flight, lazy load, 10 GB cap): load 0.5 s, prefill
27.9 s (30-token prompt), decode 0.103 tok/s (~9.7 s/token vs ~34 s/token in
the D2-013 smoke run — ~3.5× faster), MLX peak 8.98 GB (safe), output
coherent and correct. CAVEAT disclosed: run followed the earlier smoke run,
so SSD pages were partially warm; a cold-cache number would be slower —
paper reports it as warm-cache or re-runs cold. FEASIBILITY class confirmed:
exact 235B serving on 16 GB is real but ~100× below the SLO floor; the
concurrency ceiling is 1 by construction (N=2 not worth hours to document —
stated, not run).

## D2-024 · 2026-11-22 · Protocol reconciliation COMPLETE: P2 ceilings are conservative lower bounds
Question: why is P2's static OLMoE ceiling 12 vs P1's 16 on the same machine?
Test: re-ran our static arm with P1's 5 shared prompts at N=12/16 (11.3 PASS
/ 8.7 FAIL) — statistically identical to our distinct-WildChat numbers
(11.1/8.2). Prompt distribution is NOT the factor. Remaining factor: P1's
pinned_server used TRUE continuous batching (multiple sequences per forward,
amortized kernels); P2's runner time-slices (one sequence per forward,
strictly less efficient). Time-slicing can only UNDERSTATE a ceiling, never
overstate it — so P2 ceilings are conservative lower bounds, P1's 16 remains
the batched-engine upper reference, and the two-class law (the paper's
claim) is unaffected because it is an ORDERING result, and the ordering is
identical under both protocols. Paper states: "ceilings are conservative;
batching would raise them; no arm's rank changes."

## D2-025 · 2026-11-22 · Semantic-map audit v1: 15 claims checked, 4 fixed
Full sweep of P2_v0.1.tex against frozen artifacts (semantic-claims.json).
Caught and fixed: (1) STALE VALUE — abstract quoted the quarantined
protocol's 12.6 tok/s instead of the corrected median 22.5 (exactly the
class this audit exists for); (2) DEFINITION ERROR — qwen30b ceiling was
stated as 1 but 7.9 < 10 floor means 0; (3) BOUNDARY OVERREACH — "static
1–10% spread" was false at memory-pressure cells (72–179%), now disclosed;
(4) caption ambiguity ("median of 3" -> "median over 3 prompts"); plus
frontier ceiling reworded to 0 under the p05 rule for definition
consistency. Draft recompiled clean; the audit trail is preserved in
semantic-claims.json per claim.

## D2-026 · 2026-11-23 · Paper elevated to academic standard (owner review pass 1)
Owner-requested changes, all applied: p05 defined at first use (abstract);
numbered arXiv-style bibliography (37 entries) replacing inline arXiv
mentions; related work expanded to 5 clusters covering the full P2
literature review (offloading/caching, prediction/prefetch, Apple SSD
streaming, cache ownership/approximation, tail measurement) with the
author's prior work folded in as a normal citation; "our position" and
"P1/P2" insider terms removed throughout; non-academic phrasing removed
("and we say so"); Threats to Validity + dedicated Limitations section +
Future Work + Conclusion + AI-usage disclosure + Reproducibility note added;
Appendices A (claim-to-evidence map) and B (incidents) made explicit;
frontier ceiling stated as 0 under the p05 rule for definition consistency.
GitHub: no fork of FreeToken (their license/citation norms make a cite the
correct respect; a fork would entangle release timing) — open release of
OUR engine + frozen data at camera-ready, DOI via Zenodo as with P1.
