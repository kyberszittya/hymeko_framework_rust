# Report: Walks-Augmented HSiKAN on Epinions — 5-Seed Paired Validation

**Date:** 2026-05-11
**Branch:** refactor/extract-hymeko-hre
**Companion plan:** `docs/plans/2026-05-11-abb-global-fullness/`
**Queue logs:** `reports/overnight_2026_05_11_stage5/MASTER.log`

## Headline

5-seed paired comparison on Epinions (seed=0..4, 60 epochs):

| Variant | Mean AUC | SD | Per-seed |
|---|---|---|---|
| Baseline (`per_vertex` m=64, ABB OFF, h=16) | **0.7392** | 0.0094 | 0.7415, 0.7450, 0.7257, 0.7523, 0.7314 |
| Kitchen-sink (walks `c3,c4,w2,w3` + CPG-3 + g10 ABB + h=32) | **0.8145** | 0.0017 | 0.8141, 0.8171, 0.8126, 0.8130, 0.8155 |
| **Paired Δ** | **+0.0753** | 0.0105 | — |
| **Paired σ** | **+16.02** | — | — |
| Sign test | 5/5 positive | p≈0.0625 (max for n=5) | — |

**Paper-grade per `feedback_n_seed_before_paper_promotion.md`.** Treatment SD (0.0017) is 5× tighter than baseline SD (0.0094); the kitchen-sink config is also more *stable* across seeds, not just higher mean.

## Architecture ablation (Epinions seed=0, single-seed)

| Variant | AUC | Δ vs baseline 0.7415 | Verdict |
|---|---|---|---|
| baseline (`per_vertex` m=64 OFF h=16) | 0.7415 | — | reference |
| **uniform 128 + g10 ABB** | 0.7229 | −1.86 pp | bigger cap alone hurts (control) |
| CPG-soft-gentle h=16 (tiers `1.0:256,10.0:64,100.0:32`) | 0.6905 | −5.10 pp | step-tiered CPG alone hurts |
| CPG-soft-steep h=16 (tiers `0.1:1024,...,100.0:64`) | 0.6918 | −4.97 pp | steeper doesn't help |
| **walks `c3,c4,w2,w3` h=16 only** | 0.7962 | **+5.47 pp** | walks alone is a strong lever |
| **walks h=32 only** | 0.8012 | **+5.97 pp** | h=32 adds ~+0.5 pp on top of walks |
| **kitchen-sink** (walks + CPG-3 + g10 + h=32) | 0.8141 | **+7.26 pp** | + tier CPG ~ +1.3 pp over walks+h32 |
| **walks + CPG-soft-gentle + h=32** | 0.8165 | **+7.50 pp** | + soft CPG ~ +1.5 pp over walks+h32 (single-seed) |

## Conclusions

1. **Walks (`c3,c4,w2,w3`) are the primary lever** on Epinions. +5.5–6.0 pp alone over the cycles-only baseline. Matches the prior finding on Slashdot (`project_attention_cycle_batch_compose_2026_05_08.md` broke SGT at h=4 via the same walks mix).
2. **Wider hidden (h=32 vs h=16)** adds a further +0.5 pp on top of walks.
3. **CPG step-tiered cycle budgeting alone HURTS by ~−5 pp** on Epinions (and −2.2 pp on Bitcoin OTC). Root cause: zero or low cap on bottom-percentile vertices starves their `M_e` rows, removing learning signal for those vertices.
4. **CPG + walks ≥ walks alone by ~+1.3–1.5 pp single-seed.** Walks fill in the leaf-vertex signal that CPG removes; the composition is mildly synergistic. 5-seed-paired CPG-only-vs-walks comparison NOT YET run.
5. **Continuous degree-adaptive CPG (Stage 6) was cancelled** per user direction; archived as `project_cpg_idea_archive_2026_05_11.md` for future work.

## Wall-time

Stage 5 5-seed BA paired (warm cache after Stage 4): kitchen-sink ~142–147 s/seed, baseline ~41–46 s/seed. Total Stage 5 (5+5 paired + ablation): ~22 min. Cold-cache equivalent would have been ~10× longer; the cycle-cache fingerprint fix from this morning enabled the warm-cache speedup.

## Caveats and follow-ups

- **Still below SGT SOTA (~0.95):** our 0.8145 vs SGT ~0.95 is a −0.13 gap. Walks closed about half the gap from baseline 0.74. Further gap-closing levers (deeper walks `w4,w5`, attention variants, learned per-vertex weights) remain open.
- **CPG architecture remains parked.** The "CPG helps when paired with walks" single-seed signal is intriguing but requires 5-seed paired vs walks-only-h=32 to confirm.
- **Bitcoin OTC CPG-soft both OOM'd** in 9–10 s due to Chrome holding 5.98 GiB of the 7.6 GiB GPU. Environmental, not code. Bitcoin OTC CPG validation remains incomplete.
- **No paired 5-seed on Slashdot** at the kitchen-sink config; the prior Slashdot SOTA (`project_attention_cycle_batch_compose_2026_05_08.md`) was at a different config.

## Files staged this round

- `signedkan_wip/experiments/run_overnight_stage5_2026_05_11.sh`
- `signedkan_wip/experiments/run_overnight_stage6_2026_05_11.sh` (cancelled; left on disk as future work seed)
- `signedkan_wip/src/analyze_paired_5seed.py` (5-seed paired analyzer)
- `reports/overnight_2026_05_11_stage5/` (all per-seed JSONs)
- `signedkan_wip/src/cycle_cache.py` (fingerprint fix)
- `hymeko_graph/src/topk_cycles.rs` (fullness-gate normalisation; CPG doc rename)
- `hymeko_py/src/cycles.rs` (CPG-aware tiered_bb_global_batched binding)
- `signedkan_wip/src/n_tuples.py` (tiered+ABB dispatch)

## Provenance

- git SHA at queue start: `5f14ac08b85824ed82e4d97f8c010e089eda5b98` (working tree dirty)
- Host: Linux Amaterasu 6.17.0-23-generic, AMD Ryzen 7 3700X (16 threads), 32 GiB RAM
- GPU: NVIDIA RTX 2070 SUPER 8 GiB, driver 580.126.09
- Per-run cgroup cap: `MemoryMax=16G, MemorySwapMax=0` via `systemd-run --user --scope`
- Random seeds: 0..4 explicit per run
- Dataset hash: as packaged in `data/` (unchanged)
