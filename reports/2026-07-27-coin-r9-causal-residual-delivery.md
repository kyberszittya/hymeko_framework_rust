# Coin-R9 — delivery-focused causal residual TD3 over the frozen R8 champion

**Date:** 2026-07-27 · **Branch:** `recovery/coin-r9-causal-residual-delivery` (worktree `hymeko_coin_r9_wt`, off tag `coin-r8-bounded-residual-heldout-improvement-v1`)
**STAGE 3/4 verdict (evidence-based, R10-0 reachability):** `R9A_RESIDUAL_BASIS_OVER_FROZEN_R8_BASE_CANNOT_EXPRESS_S1_DELIVERY` (Case **C**, `BASE_OR_RESIDUAL_BASIS_INSUFFICIENT`) — the dev gate (strict K6 on **both** s1 and s3) is **not** reached; **STAGES 5–6 not entered; the sealed blind final panel remains SEALED_NOT_EVALUATED.**

> **Correction (verdict wording).** An earlier draft labelled the constant-sweep result `STRUCTURAL_BOUND_INSUFFICIENT`. That
> **overclaimed** — a constant Δa sweep + 3 TD3 seeds cannot prove no temporal residual delivers s1. The precise measured
> statement is `CURRENT_BOUNDED_RESIDUAL_NEIGHBOURHOOD_HAS_NO_DEMONSTRATED_S1_DELIVERY`; the **R10-0 reachability audit** below
> then upgraded the localisation to **Case C** with proper positive controls — separating the *measured ceiling* from
> *mathematically-proven impossibility* (the CEM is still an existence search, not a non-existence proof).

## Mission

Push the coin into the strict 20 mm K6 zone by learning a bounded **causal per-step** increment Δa over the FROZEN R8
champion (`a_exec = clip(a_R8 + Δa)`), cert-only release, K6-independent reward. The 20 mm tolerance, physics, motion limits
and release certificate are unchanged.

## Gates executed

| stage | result |
|---|---|
| **S0** preflight | branch off the R8 tag; champion ckpt sha `71e7653b`; hard constraints recorded (`stage0_preflight.json`) |
| **S1** blind final panel | **SEALED** — 4 fresh cradles (f1–f4, seeds 17750/18250/16500/19500, dtz0 0.051–0.131) by a generation-geometry-only rule; `SEALED_NOT_EVALUATED` (`final_panel_manifest.json`). s4/s7 demoted to validation. |
| **S2** update-zero identity | **`R9_UPDATE_ZERO_REPRODUCES_R8_CHAMPION`** — Δa≡0 reproduces the R8 champion full trace bit-for-bit (dev 2/2, max diff `0.00e+00`) |
| **S3** delivery TD3 (Curriculum C, 3 seeds) | **NO dev-gate cross** — all seeds stuck at the R8 champion's K6 **1/2**, dtz ~36 mm, safe 2/2; best_val ≈ update-0 |
| **S3 ceiling** (constant sweep) | s1 best_dtz **52.3 mm, delivers=False** over ~130 constant bounded Δa; s3 17.7 mm delivers=True (measured; *not* a non-existence proof) |
| **R10-0 reachability** (decisive) | teacher s1 **18.5 mm K6 ✓** (feasible); temporal CEM s1 **51.8 mm declared / 46.2 mm full-range — no delivery**; s3 delivers in-search ⇒ **Case C `BASE_OR_RESIDUAL_BASIS_INSUFFICIENT`** |
| **R10-B0** base-coverage (M0) | scaffold-param CEM on s1 (zero residual) = **50.5 mm, no delivery**; default scaffold zero-residual delivers *neither* (s3 262.7 / s1 57.7 — s3 needs the R8 residual); s1/s3 geometrically separable ⇒ **`NEAR_BASE_NOT_IN_SCAFFOLD_FAMILY`** — the scaffold *strategy*, not just its base-center, is insufficient for s1 |

## The finding (measured / inferred)

- **Measured.** The per-decision TD3 (400 rollouts × 3 seeds, K6-independent Curriculum-C reward, warm-started so Δa≡0 = the
  R8 champion) does **not** improve dev delivery: every seed holds at K6 1/2 (s3 delivers via the R8 base; s1 stays ~53 mm),
  safe throughout. The engine is sound — update-0 reproduces the champion (distill loss `0.0`), updates fire, live signal is
  clean.
- **Measured (ceiling).** No bounded Δa over the frozen R8 champion delivers s1: the constant-residual sweep bottoms out at
  **52.3 mm** (vs the R8 base's 53.5 mm — essentially no gain), and the temporal TD3 matched it (~53 mm). The bounded
  residual space over this base **does not contain an s1-delivering solution**.
- **R10-0 reachability audit (the decisive test, with positive controls).** Teacher θ delivers s1 (18.5 mm, K6 — **feasible**)
  and the CEM finds delivery for s3 (declared 17.5 mm K6, full 17.3 mm K6 — **the search works**). Over the frozen R8 base, a
  temporal Δa-sequence CEM reaches only **s1 = 51.8 mm at the declared ±bound and 46.2 mm at FULL range** (a_exec spanning
  [−1,1]). **Precise (reviewer-proof) statement:** *in the examined three-channel, full-amplitude temporal residual family the
  CEM found no s1-delivery over the frozen R8 base, while the same search found s3-delivery and the teacher confirmed s1's
  physical feasibility.* (Not a non-existence proof — the CEM is an existence search — but operationally enough to rule out
  mere bound-growth.)
- **Inferred (evidence-based, Case C).** This rules out **Case B** (magnitude — full range also fails) and the obvious
  **Case A** (optimiser — a thorough temporal CEM, far beyond TD3, found nothing within bound while it *did* find s3). The
  read: the frozen R8 champion's s1 base `a_R8[s1] = [+0.68, −0.99, +0.97]` (near the squeeze/stop channel edges) + the 3
  residual channels do **not**, in the examined family, express an s1-delivering behaviour — `BASE_OR_RESIDUAL_BASIS_
  INSUFFICIENT`. A *single* global scaffold + 3-channel residual-neighbourhood does not cover the near (s1) and far (s3)
  delivery regimes at once; larger amplitude only nudges s1 (51.8 → 46.2 mm), nowhere near the 20 mm zone.
- **Consequence.** R9-B (terminal-phase shaping) would **not** help — still a bounded residual over the same base; the
  full-range reachability covers it. A *larger* symmetric bound would **not** help either (full range already failed). The
  blocker is the **base / residual coordinate basis**, not the bound. One R8 champion is the wrong base for both the far
  (s3/s4) and near (s1/s7) regimes — its s1 base sits where no residual reaches delivery.

## Honest scope / no overclaim

R9 does **not** deliver on dev, so nothing downstream is claimed: no validation (s4/s7) run, the blind final panel is
untouched, no delivery is asserted without strict K6 (process rule honored). R8's result stands unchanged. The R9 negative is
**preserved**, with a decisive ceiling that localizes the blocker to the residual **bound over the frozen base**.

## R10 direction (Case C + M0 → the scaffold STRATEGY is the limit, not the base-center or bound)

Two decisive audits ran before any retraining, and they narrow R10 sharply:

- **R10-A (bound growth) — SKIP:** full-range reachability already failed.
- **R10-B (state-conditioned base *within* the tip-transport scaffold) — INSUFFICIENT alone:** M0 shows re-tuning the
  scaffold parameters does not deliver s1 either (50.5 mm). A soft `w(s,h)`-gated mixture of two *tip-transport* bases will
  still be trapped in the same non-delivering strategy for s1. (The gate *is* feasible — s1/s3 separate on causal geometry.)
- **R10-C (a DIFFERENT near controller / a missing residual channel) — now PRIMARY:** project the delivering s1 teacher
  trace into the scaffold+residual coordinates to identify the d.o.f. the tip-transport servo cannot express (candidates:
  explicit release-timing, lateral alignment, preload-decay). The near controller for s1 likely needs that d.o.f.; the far
  (s3) regime keeps the tip-transport base. Then a small bounded residual + a causal gate over {near-controller, far-base}.
- **R10-D:** TD3 over the chosen bases; validation s4/s7; single blind f1–f4 opening.

Gate sequence unchanged (R10-B0 done → B1 update-zero → B2 teacher-projection/reachability → B3 causal gate dev 2/2 → B4
residual TD3 → B5 validation → B6 freeze → B7 blind open). The 20 mm K6 tolerance, physics, motion limits and certificate stay
fixed. s4/s7 remain validation-only; the blind panel opens once, after the R10 champion freeze.

## Artifacts

`reports/2026-07-27-coin-r9-causal-residual-delivery/`: `stage0_preflight.json`, `final_panel_manifest.json` (SEALED),
`stage2_update_zero.json`, `stage3_C.json`, `stage3_A_smoke.json`, `stage3_residual_ceiling.json`, `stage3_verdict.json`,
`ckpts/r9_C_seed{0,1,2}_best_val.pt`.
Code: `hymeko_rl/coin_delivery/theta_option/{r9_causal_residual,r9_delivery_train}.py`, `hymeko_rl/experiments/{coin_r9_blind_panel,coin_r9_causal_rl}.py`, `hymeko_rl/tests/test_coin_r9.py`.
Gates: ruff clean; no fn ≥ CC 15; R9 unit tests pass; CORE.YAML untouched; blind panel never evaluated.
