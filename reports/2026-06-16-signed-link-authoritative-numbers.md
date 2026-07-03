# Signed-link results — authoritative numbers & provenance (2026-06-16)

Purpose: record the *verified* state of the signed-graph link-prediction numbers
for seminar slide 25 ("Signed-graph link prediction results"), so the deck row is
single-sourced and honest. No deck numbers were changed on the strength of this
note alone — the headline-row decision is pending the owner's confirmation (below).

Reproduce with: `python docs/seminar/table_from_results.py` (reads the JSONLs
named here; emits `figures/results_table_*.{md,csv}` with `[pending]` shown
explicitly, never guessed).

## Verified locally (this repo)

HSiKAN **edge_cr 5-seed paired** — computed from the on-disk JSONLs:

| Dataset | AUROC mean ± pstd | source JSONL |
|---|---|---|
| Slashdot | **0.907 ± 0.003** | `hymeko_neuro/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl` (5 seeds) |
| Epinions | **0.846 ± 0.009** | `hymeko_neuro/experiments/results/epinions_edge_cr_5seed_2026_05_09.jsonl` (5 seeds) |
| Bitcoin Alpha | **[pending]** | no `*bitcoin*edge_cr*` file in repo |
| Bitcoin OTC | **[pending]** | no `*otc*edge_cr*` file in repo |

Notes:
- These JSONLs report `val_auroc` per seed (held-out test key not present in-repo);
  treat as the available metric, label accordingly.
- The local edge_cr files contain **no shuffle-control rows** and **no baselines**
  (HSiKAN-only). The shuffle controls (Bitcoin-α 0.966, Slashdot 0.851) and the
  Bitcoin-α real number (0.987) cited in discussion come from **Komondor** runs
  (chains 13885739 / 13885723 etc.) that are **not** in this repo — verify on the
  cluster before quoting.
- A newer Komondor K-sweep (jobs 13885808/9/10) reportedly gives Epinions
  **0.883 ± 0.013** under the same protocol — a second candidate for that cell.

## Gömb-strict — does NOT exist as a clean test table

- The dedicated `gomb_strict_benchmark_*_2026_05_14` chain (step1_alpha →
  step4_epinions × 5 seeds + step0_shuffle) **OOM-crashed on the 7.6 GB local GPU**
  — orchestrator logs show `rc=1` on every step; step logs show
  `torch.OutOfMemoryError: CUDA out of memory`. No valid held-out-test Gömb-strict
  AUCs were produced.
- Surviving Gömb JSONLs (`gomb_bridge_gomb`, `outer_hsikan_gomb`, `stacked_gomb`)
  report **validation** metrics only (`val_auroc`, `val_auc_best ≈ 0.89` on
  Bitcoin-α) for **variant** configs — not a Gömb-strict held-out-test table.
  Filling the slide from `val_auroc` would be the val≠test inflation trap.
- No Komondor Gömb run exists; the Komondor jobs were the HSiKAN edge_cr cells.

**Conclusion:** the slide's "Gömb-strict" row should read **[pending — re-run on
Komondor / GCP]**, not be back-filled from OOM-failed or validation-only data.
Gömb is heavier than HSiKAN (which already took ~2.5 h/cell on an A100), so it is
precisely the model that needs the cluster — this doubles as justification for the
GPU-quota request.

## Decision needed before slide 25 is rewritten

1. Adopt **edge_cr 5-seed paired** as the authoritative protocol for the row (and
   relabel it from "Gömb-strict" to "HSiKAN edge_cr"), or keep the current row and
   wait for Gömb?
2. Epinions cell: **0.846** (2026-05-09 in-repo) or **0.883** (newer Komondor
   K-sweep)?
3. Baselines (SGCN / SiGAT / MLP): include as **"reported" (literature)** only —
   not "our run" — since they are not in the edge_cr JSONLs.

Until (1)–(3) are answered, slide 25 is unchanged.
