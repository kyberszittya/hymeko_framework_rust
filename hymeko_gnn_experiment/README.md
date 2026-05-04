# HyMeKo vs. GNN — experiment scaffold

Implementation of `docs/plans/plans_20260429/hymeko_gnn_experiment_design.md`.
Tonight's deliverable: synthetic data generator, rule-based HyMeKo
classifier, and the directory shape that GNN baselines drop into.

## Status

| step (per §10 of the plan) | status |
|---|---|
| 1. Synthetic generator (.npz) | ✅ `src/synthetic.py` |
| 2. HIR→npz serializer | ⏸️ skipped tonight; npz emitted directly |
| 3. PyG baselines (HGNN, AllSetTransformer) | ⏸️ stubs only |
| 4. HyMeKo rule-based classifier | ✅ `src/hymeko_classifier.py` (Python; Rust binary is the long-term home) |
| 5. Benchmark harness | ⏸️ wires the above together; pending baselines |

## Generated dataset

`data/synth_n32_k5.npz` — 200 samples of |V|=32, |E|=32, k_max=5, signed.
Per-property positive fraction:

| property | WL-hard? | pos | balance |
|---|---|---:|---|
| `is_3_regular` | yes (k≥3) | 0 / 200 | trivially-rare; needs targeted generator |
| `is_5_regular` | yes | 0 / 200 | same |
| `has_triangle` | yes | 2 / 200 | very rare; targeted generator needed for usable baselines |
| `n_components_ge2` | yes | 118 / 200 | well-balanced ✓ — **start here** |

The two rare-positive properties need a *balanced* generator
variant before the GNN comparison is statistically meaningful.
Adding a `--balanced --property has_triangle` flag is the next
small change.

## Sanity check

The rule-based HyMeKo classifier hits 100% on every property — by
construction, since the predicates *are* the ground-truth definitions.
This validates the data-handling pipeline; the load-bearing test is
when GNN baselines plug in and produce sub-100% on the WL-hard
properties.

```bash
python3 -m src.synthetic --out data/synth_n32_k5.npz \
                          --n-samples 200 --n-vertices 32 \
                          --n-hyperedges 32 --k-max 5 --signed
python3 -m src.hymeko_classifier --in data/synth_n32_k5.npz \
                                  --property n_components_ge2
# → accuracy 1.0, F1 1.0
```

## Open work

1. **Balanced generator** for `is_k_regular` and `has_triangle`.
   ~30 LOC: rejection-sample structures matching the target.
2. **PyG baselines** as `baselines/{hgnn,gcn_clique,allset}.py`. Each
   is a small wrapper around the published reference impls; the
   `hymeko_hnn` crate already has signed_hgnn / gcn_clique on the Rust
   side, but the GNN comparison needs the PyTorch-Geometric versions
   for fair benchmarking.
3. **Benchmark harness** (`src/run_benchmark.py`): same shape as
   `python/benches/thesis_iv_hard/run_benchmark.py` — paired-seed,
   per-property table, latency vs. |V| plot.
4. **Rust port of `hymeko_classifier`** — drives a real `hymeko_query`
   pattern through the IR. Validates the "query IS the classifier"
   claim of §3.2.

## Cross-paper relevance

This experiment slots into the **arxiv_v1** framework paper at §VI.x
as a downstream-application demonstration, OR into a separate SISY /
SMC paper per the plan's §9 mapping. The signed-incidence row of the
expected-outcome map (§8) is the differentiator for SMC-flavoured
venues.
