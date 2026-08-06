---
name: project-hsikan-launchbound-alternatives
description: "HSiKAN launch-bound = a DISPATCH-bound B=1 problem (17x cheaper batched); SA-HSiKAN backbone SHIPPED; alternatives measured (batch-rollout 17x, Chebyshev 3x, CR-Chebyshev hybrid); sparse=scaling lever"
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-28: profiled + reframed HSiKAN's launch-bound. It is a **DISPATCH-bound B=1 problem**, NOT "HSiKAN is
slow". Profile (galambos, CPU/1-thread): ~200 tiny aten ops/forward, **~76% overhead** (elementwise + 71
dtype-copies + reshape), only ~24% compute; **B=256 is 17x cheaper/sample** than B=1 (overhead is fixed per-op →
amortises under batch). The aggregation is ALREADY one einsum/layer and the spline ALREADY batched (signed_kan/
backends.py, splines.py) → it is NOT a vectorisation gap; it is fusion/batch.

**Alternatives (all measured), composable:**
1. **BATCH THE ROLLOUT** (vectorised envs) = **17x**, model-agnostic — the big lever; needs trainer+env
   vectorisation (real work). The "do it differently" answer.
2. **Cheaper basis** Chebyshev-poly / RBF(FastKAN) = **3x / 2.8x at B=1** but SLOWER at B=256 (regime-dependent;
   they materialise (.,C,k); the CR 4-gather is cheaper batched). Drop-in swap of CatmullRomActivation.
3. **SA-HSiKAN (B^L collapse) = SHIPPED**: `policy.py` StructuralActorBackbone + `sa_hsikan` in
   _BACKBONES/POLICY_KINDS/exp_pernode _CONFIGS, unit-tested (test_sa_hsikan_backbone.py). B^L =
   matrix_power(a_pos - a_neg, walk_len) — one matmul + CR cell. **2.6x faster, 11x fewer params** than HSiKAN.
   Exp (nofinger, 20k×2seeds): SA-HSiKAN delivery median **0.25 @ 1416 params** vs MLP 0.15 @ 15296 (suggestive,
   2 seeds, wide IQR). [[project-structural-actor-walk-holonomy]] shorthand SA-HSiKAN.
4. **torch.compile / Triton** = big on GPU. **CORRECTED 2026-07-01: torch.compile WORKS on this Windows box** —
   Visual Studio IS installed (MSVC `cl` present), so the old "SKIP on Windows" was a stale, unverified
   assumption (it misled a whole session into calling GPU "out of scope"). Measured on the RTX 3070 Laptop, torch
   2.12: `torch.compile(mode="reduce-overhead")` (CUDA graphs) on the TD3+BC update = **5.08 vs 41.9 ms/update =
   8.25x** IN ISOLATION; **raw cuda eager is SLOWER than cpu** (53 vs 42 ms — confirms launch-bound). BUT
   **end-to-end only ~1.5x** (8000 galambos steps: 135 vs 201 s) — Amdahl: env physics (3.3 ms/step) + the B=1
   per-step action-selection (which GPU does *slower*) dominate once the batched update is fast. Lesson: the
   isolated-component speedup ≠ end-to-end. **Wired into `train_offpolicy`** (`OffPolicyConfig.device`/`compile`,
   opt-in `device="cpu"` default; compiles the loss *closures* not the modules → checkpoints stay clean; nets
   return on CPU). Further end-to-end gain needs vectorised-rollout off-policy (amortise physics + batch B=1) or
   CPU-side action-selection — see lever #1.
5. Kill the 71 dtype-copies = ~16%.

**CR-CHEBYSHEV HYBRID (Hajdu idea):** Chebyshev coeffs → CACHED knot-basis matrix (T_knots, fixed) → CR control
points. Measured: same params, **2.4x spline resolution**, smooth-by-construction. Key payoff = the
**train-CR / deploy-Chebyshev bridge** (control points lie on a degree-k Chebyshev → the function is
Chebyshev-representable → deploy via the cheap 3x matmul). NOT a raw B=1 speed win (gather dominates).

**SPARSE = scaling lever, not a small-graph win.** Dense einsum wins at current N=6-14; the SparseScatterBackend
(torch.sparse.mm) exists and pays for collaborative k-arm / large designs. Make the backend **N-adaptive** (auto
dense/sparse at a profiled crossover). Caveat: B^L DENSIFIES with L (L-hop fill-in), so sparse SA-HSiKAN only at
small L.

Artifacts: PDF `reports/2026-06-28-hsikan-launchbound-alternatives.pdf`, bug `reports/
2026-06-28-hsikan-launch-bound-bug.md`, plan `docs/plans/2026-06-28-hsikan-acceleration/` (NOTE: the plan's
"Tier 1 = vectorize 200→<30" is WRONG — vectorisation already done; correct it). NEXT: implement the CR-Chebyshev
cell as a `signed_kan` make_activation option (parity-tested), and/or scope the vectorised-rollout (the 17x).
Ties [[project-unify-hsikan-core]] (HSiKAN = Highway Signed KAN, never rename).
