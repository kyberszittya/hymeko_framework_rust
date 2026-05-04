# SignedKAN WiP

**Status:** Phase 0 in progress (April 28–29, 2026).
**Target:** WiPI submission, deadline May 3.
**Backup:** SISY 2026, May 28.

## Tree

```
signedkan_wip/
├── DECISIONS.md            # Phase 0.1 — locked decisions
├── README.md               # this file
├── data/                   # Bitcoin Alpha + OTC raw + processed
├── src/
│   ├── __init__.py
│   ├── datasets.py         # Phase 1.1–1.4
│   ├── hyperedges.py       # Phase 1.5–1.9
│   ├── splines.py          # Phase 2.1–2.5
│   ├── signedkan.py        # Phase 2.6–2.10
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── sgcn.py         # SGCN wrapper / import
│   │   ├── sigat.py        # SiGAT wrapper / import
│   │   └── vanilla_kan.py  # vanilla KAN baseline
│   └── train.py            # Phase 3
├── experiments/
│   ├── configs/            # YAML per run
│   └── results/            # JSON output, one per run
└── paper/
    ├── signedkan_wip.tex   # IEEE conference template
    ├── refs.bib
    └── figures/
```

## Quick run (after Phase 1–3 complete)

```bash
cd signedkan_wip
python3 -m src.datasets --download bitcoin_alpha bitcoin_otc
python3 -m src.hyperedges --dataset bitcoin_alpha --construction triads
python3 -m src.train --config experiments/configs/bitcoin_alpha_signedkan.yaml
```

## Phase 0 — DONE (Apr 28–29 night)

| step | status | notes |
|------|--------|-------|
| 0.1 decisions | ✅ | `DECISIONS.md` |
| 0.2 repo scaffold | ✅ | this tree |
| 0.3 PyTorch + CUDA | ✅ | `torch 2.11.0+cu130`, RTX 2070 SUPER |
| 0.3 Bitcoin Alpha | ✅ | `data/bitcoin_alpha.csv` — 3,783 nodes, 24,186 edges, 93.6 % positive |
| 0.3 Bitcoin OTC | ✅ | `data/bitcoin_otc.csv` — 5,881 nodes, 35,592 edges, 90.0 % positive |
| 0.3 SGCN reference repo | ⚠️ | sandbox blocked auto-clone; clone manually: `git clone --depth 1 https://github.com/benedekrozemberczki/SGCN /tmp/sgcn_ref` |

## Phase 1 — DONE (out of order; wrapped Apr 29 night)

| step | status | notes |
|------|--------|-------|
| 1.1 Bitcoin Alpha loader | ✅ | `src/datasets.py` |
| 1.2 Bitcoin OTC loader | ✅ | same module, same interface |
| 1.3 80/10/10 split, seed 42 | ✅ | `src.datasets.split` |
| 1.4 graph statistics | ✅ | `SignedGraph.stats()` |
| 1.5 triangle enumeration | ✅ | `src/hyperedges.py::_enumerate_triangles` |
| 1.6 sign-balance classification | ✅ | Cartwright–Harary product test |
| 1.7 σ-assignment rule (apex+, base−) | ✅ | `_classify` |
| 1.8 hyperedge output format | ✅ | `SignedTriad` dataclass |
| 1.9 statistics: 22,153 triads on Bitcoin Alpha (84.6 % balanced) | ✅ | matches structural-balance literature |

## Phase 2 — DONE (out of order; wrapped Apr 29 night)

| step | status | notes |
|------|--------|-------|
| 2.1 Cox–de Boor recursion | ✅ | `src/splines.py::cox_de_boor` |
| 2.2 uniform G=5, k=3 knot grid | ✅ | `make_uniform_knots` |
| 2.3 learnable spline coefficients | ✅ | `BSplineActivation` |
| 2.4 partition-of-unity test | ✅ | max err 1.19e-7 |
| 2.5 gradient-flow test | ✅ | ‖∇c‖ = 13.96, finite |
| 2.6 Option C forward pass | ✅ | `src/signedkan.py::SignedKANLayer` |
| 2.7 inner spline per (vertex, edge, sign) | ✅ | per-sign branches |
| 2.8 outer spline per (edge, sign) | ✅ | per-sign branches |
| 2.9 full model | ✅ | `SignedKAN` (encode_triads + predict_edge_sign) |
| 2.10 smoke test | ✅ | 100-node × 50-triad forward+backward, 1,033 params at h=8 |

## Phase 3 — STARTED (smoke test)

5-epoch run on Bitcoin Alpha at hidden=8 yields **AUC 0.580** on test
(well above chance 0.5). 30,497 parameters at h=8; ~74 s for 5 epochs.

**Open performance issue carried into Phase 3 morning:**
`SignedKAN.encode_triads()` is currently called once per minibatch
inside the training loop. With ~22k triads × ~76 minibatches/epoch =
1.7 M triad evaluations/epoch, the throughput is ~15 s/epoch at h=8
and will scale to ~150–300 s/epoch at h=32. The 18-run hyperparameter
sweep would take ~30 h at this rate — too much.

**Fix in `src/train.py`:** move `encode_triads` outside the inner
batch loop, recompute once per epoch. Should give ~50× speedup on
the per-minibatch path. ~30 lines of refactoring; do it first thing
Phase 3.

## Quick run (after Phase 3 morning's optimization)

```bash
cd signedkan_wip
python3 -m src.datasets --download bitcoin_alpha bitcoin_otc
python3 -m src.hyperedges --dataset bitcoin_alpha
python3 -m src.train --dataset bitcoin_alpha --hidden 32 --n-epochs 100 --seed 0
```

## Anti-leak guard

Do not reference G-SPHF, HSMM, GGK internals beyond a single citation,
or any in-flight KAN variant. See DECISIONS.md §"Anti-leak checklist".

## How this connects to the rest of the HyMeKo programme

This paper is **standalone** — it does not depend on the HyMeKo Rust
codebase, the framework paper, or the regulariser paper. The
connection is mentioned as *future work* in §V Discussion: the
SignedKAN architecture maps cleanly to a Tier-2 emission target in
the `torch_dataflow` backend (parallel to `ResidualBlock` /
`HighwayBlock`). That extension is left to the journal version.
