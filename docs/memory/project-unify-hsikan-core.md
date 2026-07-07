---
name: project-unify-hsikan-core
description: "Unify the two HSiKAN impls (sparse/transductive signedkan_wip + dense/inductive hymeko_rl) under ONE signed-KAN core with a pluggable aggregation backend; 4-artifact plan on disk, full cross-package, OTC-regression-gated"
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

User chose (2026-06-24, "all these") **full cross-package unification** of HSiKAN. Plan:
`docs/plans/2026-06-24-unify-hsikan-signed-kan-core/` (tex/pdf/tikz/mmd, built + validated).

**The insight:** the two HSiKANs are one architecture in two costumes. Only deep difference = the **aggregation
backend** (sparse-scatter/transductive in `signedkan_wip` vs dense-einsum/inductive/batched in `hymeko_rl`). Spline,
skip/highway, incidence, pooling are shared config that is currently FORKED (CR re-implemented in both + parity-tested;
highway only in signedkan_wip; weighted incidence only in hymeko_rl).

**Design:** one `SignedKANLayer` core + `AggregationBackend` Strategy {Dense, Sparse, Triton} + shared spline /
skip / incidence / pool. Transductive-vs-inductive is the CONSUMER's input adapter (embedding lookup vs feature
tensor), NOT a layer axis. Recommended home: a new pure-torch `signed_kan/` package (keeps RL off the heavy vision
deps; Triton optional).

**Phasing (risk-ordered):** (1) extract core+backends, unit-test isolated; (2) migrate `hymeko_rl` (low risk — gets
highway + cr_highway + weighted incidence FOR FREE); (3) migrate `signedkan_wip` **gated on OTC regression
AUC≥0.8738 / F1m≥0.7651**, Triton kept as an optional registered backend.

**Key facts (don't re-derive):** signedkan_wip already dispatches `catmull_rom`/`kochanek_bartels` splines and has
`cr_highway` (highway gate threaded with arc weights) — the weighted+highway combo is validated there. The RL
backbone is a TRUNCATED HSiKAN (no highway, no residual, was binary incidence until `incidence="weighted"` added
2026-06-24). Neither package is CORE; parser/hymeko_core untouched. Both lines get the full HSiKAN once unified.

**STATUS 2026-06-24: phases 1+2 DONE; phase 3 HALTED (premise falsified).**
- **Phase 1 done:** `signed_kan/` core (splines/backends/layer/backbone), 11 tests, ruff/mypy clean.
- **Phase 2 done:** `hymeko_rl/policy.py` delegates to the core; dup deleted; HSiKANBackbone = thin hg_state adapter;
  `skip="highway"` reachable; 301 tests pass; **old checkpoints still load** (state-dict keys preserved).
- **Phase 3 HALTED — the plan's core assumption was WRONG.** Read `signedkan_wip/src/core/signedkan.py`
  `SignedKANLayer._forward_impl`: it is a **k-uniform hyperedge gather** (`triad_v (T,k)`, k≥3) with an INNER
  spline per arc + an OUTER spline per sign-branch, multiple sign branches S with per-sign masks, transductive
  node embeddings, edge-level output. That is a DIFFERENT ALGORITHM, not the RL pairwise signed-GCN with a
  different backend. The `aggregate(a_pos,a_neg,h)` abstraction does NOT fit it; forcing a merge = leaky
  abstraction or benchmark-breaking rewrite. So the "one layer, pluggable backend across both lines" goal is
  **not appropriate** for the aggregation/layer.

**RESOLUTION (user directive 2026-06-24): ONE general HSiKAN (CR body) + CHANGEABLE HEAD.** Not a merge of the
legacy triad layer — instead the general HSiKAN = `signed_kan.SignedKANBackbone` (CR signed message passing +
highway + weighted incidence) with a swappable head:
- RL convention → pooled actor/critic head (hymeko_rl.ActorCritic).
- Signed-graph convention → `signed_kan.EdgeSignHead` + `SignedGraphHSiKAN` (embeddings → CR body → edge-sign).
Built + tested (signed_kan/tests/test_graph_convention.py: edge head, dense↔sparse parity, signed-graph model
forwards+learns, highway+weighted+sparse path). Added `SparseSignedBackend` so it scales to large graphs (OTC).
**CR is now the single canonical spline** (`signed_kan.catmull_rom`, the verbatim gather form); signedkan_wip's
`_catmull_rom_eval` now imports it (keeping its torch.compile wrapper) and hymeko_rl re-exports it — three copies → one.
The legacy triad+B-spline `SignedKANLayer` stays for the current OTC numbers.

**OTC CR-vs-B-spline A/B — DONE 2026-06-24** (`reports/2026-06-24-otc-cr-ab.md`). Three settings, same verdict at
proper convergence: **CR ≈ B-spline on AUC (a TIE)**, CR modestly cheaper.
- Pairwise general HSiKAN (no-leakage): both 0.9054 — tie.
- Legacy bare-`train` (~0.6, under-fit): CR +0.033 — a REGIME ARTIFACT; did NOT survive convergence.
- **Full HighwaySignedKAN recipe, GPU (RTX 3070, the real ~0.87): CR 0.8689±0.009 vs B-spline 0.8639±0.013 — TIE
  (+0.005, within noise).** The bare-train edge washed out.
- Speed (setting-dependent, both REAL): **CPU ~6.5× cheaper** (clean 3-seed×200ep training-loop timing — CR
  closed-form vs B-spline's Cox–de Boor recursion, which is slow on CPU); **GPU ~1.3×** (bspline parallelizes, gap
  shrinks). The GPU 10ep-smoke "54×" was a first-call-CUDA-compile artifact. CR also marginally smaller (190,113 vs
  190,369 params). So: CR much cheaper on CPU (where these small models often run), modestly cheaper on GPU.
**"OTC should use CR" holds on EQUIVALENCE + simplicity + CR being HSiKAN's spline — NOT on accuracy.**
GPU: HSiKAN is GPU-ready (signed_kan device-agnostic); the full recipe runs on GPU but segfaults on the Windows-CPU
torch sparse-CSR beta (CUDA-oriented harness). The bspline/cox_de_boor dedup into signedkan_wip was **REVERTED**
(benchmark decoupled, ec98095) — signedkan_wip keeps its own spline copies. `build_signed_adjacency` lives in
signed_kan; redundancy audit `reports/2026-06-24-redundancy-audit.md` (flagged: link-sign AUC eval ~10×).

**Why:** phases 1-2 killed the RL-internal duplication + made RL "hsikan" a real Highway Signed KAN. **How to apply:**
phase 3 = (optional, narrow) re-home the spline evaluators into signed_kan, bit-parity-gated; NOT a layer merge.
Builds on [[project-kato-dual-discriminator-plan]]. HSiKAN = **Highway Signed KAN** (Schmidhuber) — never rename.
