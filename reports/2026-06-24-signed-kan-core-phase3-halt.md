# HSiKAN unification — phase 3 HALTED (plan premise falsified)

**Date:** 2026-06-24 · **Branch:** soma-vision · **Plan:** `docs/plans/2026-06-24-unify-hsikan-signed-kan-core/`

## Decision
Phase 3 (fold `signedkan_wip`'s layer onto the shared core) is **halted before any benchmark code was touched**.
Reading the actual implementation (`signedkan_wip/src/core/signedkan.py`, `SignedKANLayer._forward_impl`) falsifies
the plan's central assumption that *"the only deep difference is the aggregation backend."*

## Evidence
`SignedKANLayer.forward(x, triad_v (T,k), triad_sigma (T,k), arc_weights)` is a **different algorithm**, not the RL
layer with a different backend:
- **k-uniform hyperedge gather** (k≥3 triads / Davis n-tuples) — the RL line is pairwise `A_pos@h` (k=2).
- **two splines per hyperedge**: an *inner* spline per arc (with `residual`/`highway`/`cr_highway` skip, the latter
  threading `arc_weights`) **and** an *outer* spline per sign-branch — the RL line has one spline activation.
- **multiple sign branches** `S` with per-sign masks + count-normalised aggregation — not just ±.
- **transductive** node embeddings (`nn.Embedding(n_nodes, d)`); output is a **per-edge** representation for
  link-sign prediction — the RL line is inductive features → pooled per-vertex.

A pluggable `aggregate(a_pos, a_neg, h)` cannot express a k-uniform dual-spline sign-branch gather. Forcing the
merge would be a leaky abstraction or a rewrite of the validated OTC layer (AUC 0.8738) + its Triton kernel — exactly
the benchmark risk the phasing was meant to avoid. Per the operating contract, a measurement contradicting a plan
assumption is a stop-and-report condition.

## What stands
Phases 1–2 are independent of this and remain valid + verified (301 tests): they removed the **RL-internal**
duplication (the RL had its own `_SignedConv` + CR copy) and gave the RL line a real configurable Highway Signed KAN
(weighted incidence + highway, parity-preserving defaults, old checkpoints still load).

## Corrected scope (what is actually shareable)
- **Spline primitives only.** `catmull_rom` is bit-parity-confirmed across both lines; `bspline`/`kochanek_bartels`
  could be re-homed into `signed_kan` as the single source, both lines importing them — pure functions, bounded,
  gated on bit-parity (note: OTC uses `bspline`, so re-homing CR alone does not touch the OTC path).
- **The highway-gate formula** is conceptually shared but applied at different points (per-arc inner gate with
  `cr_highway` in the vision line; per-layer skip in the RL line) — keep both, do not force one module.
- **The two layers stay separate** — they are different architectures (pairwise signed-GCN vs k-uniform dual-spline
  transductive net). This is correct, not a shortfall.

## Recommendation
Do **not** merge the layers. If a narrow DRY win is wanted, re-home the spline evaluators into `signed_kan` behind a
bit-parity gate (separate, small task). Otherwise close the unification at phases 1–2. The plan doc should be amended
to record the falsified premise.
