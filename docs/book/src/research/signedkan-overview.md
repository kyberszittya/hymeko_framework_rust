# Research: signedkan_wip overview

`signedkan_wip/` is the research-grade workspace for the **HSiKAN family** — Hypergraph Signed Kolmogorov–Arnold Networks. It accreted fast during 2026-04 / 2026-05 and is mostly Python, separate from the Rust crates' production surface.

## What's stable

| component | status |
|---|---|
| `signedkan.py::SignedKANLayer` | Stable. Option-C signed-incidence aggregation per arity k. |
| `mixed_arity_signedkan.py::MixedAritySignedKAN` | Stable. αₖ-mixed multi-arity model. |
| `n_tuples.py::construct_k` | Stable. k-cycle / k-tuple enumeration (Python wrapper around Rust enumerator). |
| `hyperedges.py::construct` | Stable. k=3 cycle construction. |
| `splines.py` | Stable. Batched B-spline / Catmull-Rom / Kochanek-Bartels activations. |
| `datasets.py` | Stable. Bitcoin Alpha / OTC / Slashdot / Epinions loaders. |
| `run_final_cell.py::cell_signed_graph` | Stable. The training kernel everything else dispatches to. |

## What's WIP

| component | status |
|---|---|
| `hymeko_train_walker.py` | New (2026-05-06). Walks training.hymeko + dispatches via OPS dict. Inner forward delegated to cell_signed_graph; could be decomposed further. |
| `vision/hsikan_vision.py` | Negative result on MNIST/Fashion (HSiKAN-style does not transfer to vision). User redirected: a new vision-specific hypergraph conv operator is wanted, not an HSiKAN port. |
| `baselines/sgt.py` | Signed Graph Transformer baseline. Stable but not the primary line. |
| `baselines/sigat_model.py` | SiGAT baseline. Stable. |
| `baselines/sgcn_model.py` | SGCN baseline. Stable. |

## How to navigate

```
signedkan_wip/src/
├── hymeko_ir.py            # shared HyMeKo-IR parse helpers (driver + walker)
├── hymeko_driver.py        # legacy: parses .hymeko, extracts knobs, calls cell_signed_graph
├── hymeko_train_walker.py  # NEW: walks training.hymeko's dataflow, dispatches per-op
├── run_final_cell.py       # cell_signed_graph: the actual training kernel (~500 LOC)
├── signedkan.py            # SignedKANLayer (Option C)
├── mixed_arity_signedkan.py # MixedAritySignedKAN (αₖ mixer + sparse M_e)
├── splines.py              # Catmull-Rom / B-spline / KB activations
├── n_tuples.py             # k-cycle / k-tuple enumeration (calls Rust)
├── hyperedges.py           # k=3 triad construction
├── datasets.py             # signed-graph dataset loaders
├── baselines/              # SGCN, SiGAT, SGT for head-to-head comparisons
├── vision/                 # vision experiments (negative results so far)
├── benchmarks/             # k-cycle enumerator perf benches
└── test_harness/           # pipeline integrity tests
```

## Where memories / project-state docs live

- `docs/plans_*.md` — plans by date
- `docs/results/` — locked-in claims
- `signedkan_wip/HSIKAN_STATE_*.md` — research state per date
- `signedkan_wip/HSIKAN_FINAL_RESULTS_*.md` — the paper-ready numbers

## See also

- [HSiKAN architecture](./hsikan.md) — the core architecture
- [HyMeKo-driven training](./hymeko-driven.md) — how training.hymeko drives the walker
- [Quickstart: Build an HSiKAN architecture](../quickstart/08-hsikan-architecture.md)
