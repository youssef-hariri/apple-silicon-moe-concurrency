# Zero-Miss or Nothing: The Concurrency Boundary of Mixture-of-Experts Serving on Apple Silicon

Artifacts for the preprint (DOI: 10.5281/zenodo.24466028), companion to
[*When Pinning Pays*](https://doi.org/10.5281/zenodo.23182002) (P1).

**TL;DR:** the 2025–26 edge-MoE mechanisms (LRU expert caches, prefetch,
SSD streaming) are measured single-user. Under a per-user tail SLO with
concurrent users on unified memory, every partial-residency configuration
collapses; only zero-miss full residency protects the tail. Expert
residency under a tail metric is all-or-nothing.

## Contents

- `paper/` — the paper (PDF + LaTeX source), figures, and the semantic
  claim map (`semantic-claims.json`)
- `code/` — the exact-output MLX engine (LRU / hybrid / mmap / static
  configurations), harnesses, benchmarks, and the resumable Phase-2 runner
- `results_P2/` — every frozen measurement (168 crossover units incl.
  3 repetitions per cell, capacity sweep, prefill, mmap, predictability,
  Edge0 harness, 235B frontier)
- `docs/` — the 26-entry decision log, experiment plan, literature check,
  threat verification, and the claim map

Model weights are NOT included (~300 GB); all experiments use public
mlx-community 4-bit checkpoints, re-downloadable via the scripts in `code/`.

## Headline numbers (16 GB M1 Pro)

| Result | Value | Frozen source |
|---|---|---|
| LRU vs static, single user, 75% capacity | +43% (52.9 vs 37.1 tok/s) | `results_P2/lru_capacity_sweep.json` |
| Double-buffered prefill | 2.1–2.6× faster | `results_P2/prefill_double_buffer.json` |
| p05 ceiling: static-full vs LRU (OLMoE) | 12 vs 2 users | `results_P2/phase2/` |
| Edge0 exact vs predicted routing | 12.45 vs 21.07 tok/s (1.7×) | `results_P2/edge0_harness.jsonl` |
| Qwen3-235B on 16 GB, exact | 0.103 tok/s, 9.0 GB peak | `results_P2/qwen235b_frontier.json` |

## License

Code and data: Apache 2.0. Third-party engines benchmarked (Edge0,
SlotStream) retain their own licenses and are credited in the paper.
