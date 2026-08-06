# Paper candidate 4 — Signed-link benchmark leakage audit (the "nature_hsikan" paper)

**Working title:** *Label-Shuffle Audits Reveal Where Signed-Link Benchmarks Leak — and Where They
Don't*
**Target venue:** Nature Communications / Nature MI (already framed for a Nature-family submission;
tables emitted in Springer LaTeX).
**Status:** the closest-to-submittable item in the repository. Harness, metrics, figures, and
tables are done; **one scientific gap** remains (the reddit_body residual), plus author-side
packaging.

## Abstract seed

We audit signed-link-prediction benchmarks with a train-label shuffle protocol and two scalar
metrics: **L = max(0, shuffle_AUROC − 0.5)** (residual leak surviving the shuffle) and
**Δ = real_AUROC − shuffle_AUROC** (the legitimately learnable part). Across 7 methods × 5 datasets
× 5 seeds, baselines carry small residual leaks (L = 0.015–0.045) with L ≈ L_topo — no structural
leak; most of the real score (Δ ≈ 0.33–0.37) is shuffle-destructible, i.e. legitimate. The residual
concentrates on specific datasets (reddit_body L = 0.06–0.09, flagged for 6/7 methods; epinions and
slashdot clean). A cycle-feature method leaks only through the label-reachable channel
(full-feature L = 0.16–0.24; topology-only L = 0), demonstrating that the audit localizes the
channel, not merely the presence, of leakage.

## Central claim

A cheap, method-agnostic shuffle audit with two scalar metrics localizes benchmark leakage per
(method, dataset, feature-channel) — and certifies the *absence* of structural leakage where it
holds.

## Evidence ledger

**Measured** (`reports/2026-06-21-nature-hsikan-leakage-metrics.md`,
`reports/2026-06-20-nature-hsikan-leakage-source-figures.md`; committed JSONLs
`no_leak_baselines.jsonl` (350 rows), `no_leak_baselines_topo.jsonl`, `cycle_reachability_grid.jsonl`):

- Baselines: L = 0.015–0.045; L ≈ L_topo (e.g. dadsgnn 0.022 vs 0.023) ⇒ no structural leak;
  Δ ≈ 0.33–0.37; sigat cleanest (L = 0.015). 5-seed mean ± 95 % CI (Student-t) throughout.
- Per-dataset localization: reddit_body L = 0.06–0.09 (6/7 methods flagged); bitcoin/sesgformer
  partial; epinions, slashdot ≈ 0.
- Channel localization: cycle/HSiKAN method — topo L = 0 (real 0.80–0.84, shuffle → chance);
  full-feature L = 0.16–0.24 (the label-reachable channel leaks, the topology does not).
- Builder tested: `hymeko_neuro/paperkit/build_leakage_metric_table.py`, 5 pytest passing,
  `--self-check` OK, ruff clean.

**Inferred:** the reddit_body residual has a per-node or degree-structure origin (plausible; the
discriminating control has not run — this is the gap).

**Still hypothesis:** nothing central. The paper's claims are within its measurements — its
strength.

## On-disk artifacts

- Metrics + tables: `hymeko_neuro/assets/paper/tables/leakage_metrics.md`,
  `leakage_metrics_baselines.tex` (Springer format).
- Figures: EPS set per `reports/2026-06-20-nature-hsikan-leakage-source-figures.md`.
- Harness: `hymeko_neuro/paperkit/` builders with self-checks; committed JSONLs (provenance-clean).
- Draft home: `01_analysis/.../nature_hsikan/` (outside this repo — wiring pending).
- Related methodological asset: the reachability-rules framing
  (memory `project-reachability-rules-article`) — audit protocols as reachability rules; a
  discussion-section candidate, not a dependency.

## Prior art and delineation (search debt: MODERATE)

Label-shuffle / permutation controls are classical statistics; the delineation is the **two-metric
decomposition (L, Δ) with channel localization (strict vs topo vs full) on signed-graph benchmarks
specifically**, plus the certified-clean findings. Owed before submission: a targeted search on
GNN-benchmark leakage audits (e.g. dataset-leakage literature post-2024, link-prediction evaluation
critiques) to position L/Δ against existing shuffle protocols.

## Missing work to reach submission

1. **The reddit_body residual (the one scientific gap).** Run the discriminating control:
   degree-preserving rewire and/or per-node-prior baseline. Outcome either explains the residual
   (degree/popularity prior) or leaves a flagged anomaly — both are publishable, but the run must
   happen.
2. Bibliography: resolve the `% VERIFY` items.
3. Wire EPS figures + LaTeX tables into the `01_analysis/.../nature_hsikan/` manuscript tree.
4. Reproduction-parity caption (declared pending in the metrics report).
5. Targeted related-work search (above).
6. §9 graphics check: figures exist; confirm every table in the paper has its plotted counterpart.

## Risks / falsifiers

- The rewire control could show the residual is a trivial degree artifact — that *weakens the
  drama* of the reddit_body finding but strengthens the audit story (the audit + one control fully
  explains the leak). No outcome kills the paper.
- Venue risk is fit, not validity: if Nature Communications judges scope too narrow, the fallback
  (TMLR / NeurIPS D&B) costs formatting only — the Springer tables are already built.
