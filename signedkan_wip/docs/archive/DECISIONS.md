# SignedKAN WiP — Locked Decisions (Phase 0.1)

**Submission target:** WiPI (Work-in-Progress + Industry track),
deadline May 3, 2026.
**Backup deadline:** SISY 2026, May 28.

## Architecture

| decision | value | reason |
|----------|-------|--------|
| KAN variant | **Option C** — three sub-aggregations (+, −, ~) per hyperedge with separate inner+outer spline pairs per sign | Cleanest separation of signed semantics; matches the framework paper's signed-incidence definition |
| Spline basis | **Cox–de Boor B-spline** | Standard KAN literature (Liu et al. 2024); well-understood gradient flow |
| Spline order | **k = 3 (cubic)**, fixed | Default; reviewers expect cubic unless otherwise stated |
| Spline grid | **G = 5**, fixed uniform on [−1, 1] | Same default as published KAN |
| Coefficients | **learnable per (vertex, edge, sign) triple** | Inner spline; outer spline learnable per (edge, sign) pair |
| Hyperedge construction | **Sign-balance triads** (Cartwright–Harary 1956) | Triads are the canonical signed-graph primitive; balanced/unbalanced classification is well-defined |
| σ assignment within triad | **apex = +, base vertices = −** (rule-based, deterministic) | Avoids per-triad sign-search; makes the construction reproducible |

## Datasets

| dataset | role | source | size |
|---------|------|--------|------|
| Bitcoin Alpha | **primary** | SNAP (https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | ~3.7K nodes, ~24K signed edges |
| Bitcoin OTC | **secondary** | SNAP (https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | ~5.9K nodes, ~35K edges |

**Out of scope:** Wikipedia RfA, Slashdot Zoo, Cora-CA, Citeseer-CA.
Add only if Phase 3 has > 4h of slack.

## Task

**Link sign prediction.** Standard formulation in the signed-graph
literature. Train on a fraction of edges with signs, predict signs of
held-out edges.

Split: **80 / 10 / 10** random edge split, fixed seed (42).

## Baselines

| baseline | role | source |
|----------|------|--------|
| SGCN (Derr 2018) | strongest published signed GCN | https://github.com/benedekrozemberczki/SGCN |
| SiGAT (Huang 2019) | attention-based signed GNN | author repo TBD; fallback to wrapper |
| Vanilla KAN | unsigned baseline; ingests aggregated edge features only | Liu et al. 2024 reference impl |

**Out of scope:** SDGNN, Signed-HGNN. Mention as related work; do not benchmark.

## Metrics

- **AUC** (binary classification of edge sign)
- **Binary F1** (positive class)
- **Macro F1** (class-balanced)

Reported as median ± std over **3 seeds** in Phase 3.

## Parameter budget

Match SGCN's parameter count to within 20%. Report parameter counts
explicitly in the results table — KAN's per-edge spline coefficients
make the count larger than dense MLPs by default; we want the
comparison to be honest.

## Implementation

| layer | language | reason |
|-------|----------|--------|
| Datasets / loaders | Python (NumPy + PyTorch) | SNAP TSVs are simple |
| Hyperedge construction | Python | Triangle enumeration; ~3.7K nodes is fast |
| Spline activation | PyTorch | autograd-friendly Cox-de Boor |
| SignedKAN model | PyTorch | matches baseline code conventions |
| Training loop | PyTorch | vanilla |
| HyMeKo integration | **out of scope for WiP** | mention in §V Discussion as future work; the SignedKAN architecture maps to a Tier-2 emission target alongside `residual_block` / `highway_block` in `transforms/torch_dataflow/template.py` |

## Hardware

NVIDIA RTX 2070 SUPER (the same machine that ran ph11–ph18).
CUDA toolchain already validated. Single-GPU, no distributed training.

## Authorship

**Solo** for the WiP submission. If institutional rules require a
co-author, add **Kovács** (already on the framework + regulariser
papers; affiliation overlaps with the Hungarian conference circuit).

If extended to a journal version: Csapó's cybernetic-framing
contribution slots into a §V "Lyapunov-stable training schedule for
SignedKAN" extension, mirroring the regulariser paper's treatment.

## Anti-leak checklist

The following must NOT appear in any text or figure:
- G-SPHF (graph-SPHF, in-flight arxiv submission)
- GGK kernel internals beyond a single citation
- HSMM hierarchy / level mechanism
- Any KAN variant from in-flight work other than Liu et al. 2024 baseline

## Decision-gate calendar

| date | gate | abort criterion |
|------|------|-----------------|
| Apr 29, 21:00 | Phase 1 done | Data layer + hyperedge construction validated |
| Apr 30, 21:00 | Phase 2 done | Forward + backward pass works on small data |
| May 1, 21:00 | Phase 3 done | Results table populated, even negative is acceptable |
| May 2, 21:00 | Phase 4 done | Draft complete, figures placed |
| May 3, 17:00 | Submission | PDF + PaperCept upload |

If any gate fails, **redirect to SISY (May 28)**. Deliverables to date
remain useful — none of this is wasted.
