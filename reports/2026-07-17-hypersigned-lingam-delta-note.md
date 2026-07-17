---
title: "HyperSignedLiNGAM handoff — delta note (already realized as SignedHyperLiNGAM; power-validated)"
date: 2026-07-17
branch: integration/hymeko-main
core_yaml_touched: none
status: discovery + validation (no new build) — the plan's Stages 1/3/5 already exist; Stages 2/4 are the real delta
---

# HyperSignedLiNGAM handoff — delta note

**Aiko · 2026-07-17 · branch `integration/hymeko-main`**

This responds to the *"HyperSignedLiNGAM extension"* handoff. Its Stage-0 ask was a spec + a tiny deterministic
prototype. The §6.1 discovery pass says something more useful: **the prototype already exists** — the handoff is
the design doc for code that landed as **`SignedHyperLiNGAM`** (commit `cc9cafa`, 2026-07-15). This note records
that, corrects the naming, presents a **power-validation run** done this session, and scopes the *genuine* remaining
delta so the next step is an extension, not a rebuild.

## 1. The plan is already realized (do not rebuild)

| Handoff element | Realized by | Where |
|---|---|---|
| `HyperSignedMechanism`, `HyperSignedResult` | `SignedHyperResult` (`order, mechanisms, adjacency`, `hyperedges()`, `to_causal_hypergraph()`) | `hymeko_rl/eval/causal/signed_hyper_lingam.py` (`cc9cafa`) |
| "group pairwise signed evidence into mechanisms" (Q1) | native SSG **tail-selection** — jointly-producing subset, interaction-aware R² | same |
| `hypersigned_lingam_from_b(vars, B, proposals)`; `B ≈ A_out·Σ·A_inᵀ`; `b_hat`/`residual`/explained-energy | `factorize_from_proposals` + `fit_loadings_least_squares` + `score_mechanism_set` | `mechanism_factorization.py` (steps 3A/4A) |
| signed split `A⁺/A⁻` (b55bfef bridge) | `signed_adjacency_split` | `hsikan_mechanism.py` |
| Stage-2 scramble control | `scramble_signed_operator` | `experiments/incidence_scramble.py` (used in `lingam_operator_harness.py`) |
| Stage-3 SEM head-to-head; Stage-5 HSiKAN consumer | `run_head_to_head`, `run_hsikan_modeling` | `experiments/exp_signed_hyper_lingam.py` |

**Name:** keep the committed **`SignedHyperLiNGAM`**. "HyperSignedLiNGAM" is its word-swap — a new build under that
name would mint a confusing parallel of existing code.

## 2. Power-validation (Mac, this session — the measured claim)

`python -m hymeko_rl.experiments.exp_signed_hyper_lingam` (pgraph Rust SSG present; artifacts in
`reports/figures/2026-07-17-shl-sweep/`).

**Stage 3 — structural recovery, 60 seeds** (`signed_hyper_lingam.{json,png}`):

| SEM regime | DirectLiNGAM recall | SignedHyperLiNGAM recall | verdict |
|---|---|---|---|
| linear (additive) | 1.000 | 1.000 | TIE (no headroom) |
| nonlinear (additive) | 1.000 | 1.000 | TIE (no headroom) |
| **joint-interaction** (`x = x_a·x_b + …`) | **0.732** | **1.000** | **SignedHyperLiNGAM WINS** |

A 3-seed smoke showed a spurious `0.60/0.60` TIE on the joint regime; at 60 seeds the win is clean — the claim is a
**power** claim, not a point estimate (a low-seed read would have mis-called it).

**Stage 5 — HSiKAN over the discovered operator, 8 seeds** (`..._hsikan.json`), joint-interaction sink:

| operator fed to HSiKAN | sink R² (median [IQR]) |
|---|---|
| **SignedHyperLiNGAM** hyperedge operator | **0.963** [−0.018, 0.970] |
| pairwise DirectLiNGAM operator | −0.001 [−0.002, −0.001] |
| linear model | −0.006 |

The *correct signed mechanism* structure lets a downstream KAN model the joint mechanism (R²≈0.96); the *pairwise*
structure and a linear model do not (R²≈0). IQR is wide — median-strong, not universal.

## 3. Allowed claim (now evidence-backed)

> SignedHyperLiNGAM provides a signed mechanism-level projection of pairwise LiNGAM evidence. It **ties**
> DirectLiNGAM on additive SEMs (both recover support ≈1.0 — no headroom) and **wins on joint-interaction
> mechanisms** (recall 0.73→1.00 at 60 seeds), where a parent's marginal effect vanishes and pairwise LiNGAM
> structurally misses it; and the recovered signed operator lets a downstream HSiKAN model that joint mechanism
> (sink R²≈0.96) where the pairwise operator cannot (≈0).

## 4. Non-claims (unchanged, from the handoff)

Not claimed: SignedHyperLiNGAM replaces DirectLiNGAM; proves causal discovery on its own; HSiKAN generally beats
MLP; mechanism grouping is identifiable without assumptions; any RL improvement; any real-MetaWorld validation
(none run). Additive regimes are a TIE, not a win.

## 5. The genuine remaining delta (the real next work — an extension, not a rebuild)

1. **Stage-2 scramble control inside the validation.** The head-to-head compares SHL vs DirectLiNGAM; the
   *sign/degree-preserving scramble* arm (SHL structure, scrambled grouping) is the tighter falsifier — "the win is
   from *correct* grouping, not from merely having a hypergraph." `scramble_signed_operator` exists; add it as an
   arm to `run_hsikan_modeling` (the `hsikan_over_directlingam` −0.001 arm already controls for structure-without-
   grouping; scramble sharpens it).
2. **Stage-4 real reward/monitor frames + Q5.** Run SHL over pick-place reward/monitor and coffee-push CIP frames —
   *does the mechanism hypergraph expose reward-monitor misalignment / proxy-farming more cleanly than a pairwise
   DAG?* This is the unrun, non-trivial question that matters for the Ito/Kato story.
3. **From-B factorization view unification.** Fold `mechanism_factorization`'s `b_hat`/`explained_energy` metrics
   into the native `SignedHyperResult`, so one result type surfaces both the native discovery and the `B ≈ A_out Σ
   A_inᵀ` view. Small, additive.

## Provenance
Git `integration/hymeko-main` (CIP baseline `8ec6b8f` + spec-reward merge `b2c2d73`). Env: Mac Apple-Silicon,
`.venv` py3.11, numpy 2, scipy 1.17, pgraph `target/debug/pgraph`. Seeds 0–59 (head-to-head), 0–7 (hsikan).
Compute note: this ran on the **Mac** — the SignedHyperLiNGAM sweep is numpy/CPU and needed no GPU; kato84 (intended
64-core host) wedged mid-setup and is left for a kick. Baseline CIP campaign unaffected on kato14/kato15.
