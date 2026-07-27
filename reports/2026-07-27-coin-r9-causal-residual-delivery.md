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
| **R10-C0** phase trace audit | teacher vs scaffold on s1 ⇒ **`APPROACH_TRANSPORT_PHASE_INSUFFICIENT`**: scaffold peak v_par **0.109 vs teacher 0.322** (1/3), min_dtz 53.5 vs 18.5; brake/settle *fine* (decel 1.0, term-speed 0). The monolithic distance-proportional velocity servo has **no distinct APPROACH/momentum-build phase** |
| **R10-C1** approach-effort projection | CEM *maximising* scaffold peak v_par over the full residual range = **0.154 vs teacher 0.322** (< half, gap 0.168) ⇒ **`r⊥` localised to APPROACH forward-effort** — the momentum-build is **structurally unexpressible** by the 3-channel basis ⇒ a dedicated APPROACH impulse primitive (C2) is justified |

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

## R10 direction — the real axis is a PHASE-STRUCTURED HYBRID PROGRAM, not near/far cradle modes

Three converging audits (reach 46.2 mm · M0 50.5 mm · C0 `APPROACH_TRANSPORT_PHASE_INSUFFICIENT`) reframe the problem. The
modes are **not** "near cradle" vs "far cradle" — they are the trajectory's physical **phases**, each with a different goal
and a different usable action basis. The current monolithic tip-transport servo squeezes them into one distance-proportional
control law + 3 residual channels, and C0 shows it lacks a distinct **APPROACH/momentum-build** phase (it under-transports s1
because `v_ref = k_d·d_remain` slows near the target instead of building momentum). Correct architecture:

```
structured state + response history
            ↓
      causal MODE gate  ── m_{t+1} = g(m_t, s_t, h_t, certificate)
   ↙        ↓        ↓        ↘
APPROACH   HOLD    BRAKE    RELEASE
(fwd effort (transport (decel demand (NO actor action —
 acquire    target,    non-reversing  R6 certificate
 squeeze)   squeeze,   stop, squeeze  guard only)
            balance)   decay)
   ↘        ↓        ↓        ↙
        small bounded residual per mode
            ↓
        release certificate (R6, sole authority)
```

- **The near/far cradle difference is guard TIMING**, not two worlds: how long in APPROACH/TRANSPORT, when to switch to BRAKE,
  from what state to start squeeze-decay. Same hybrid program, different guard schedule (s1 needs a longer/stronger APPROACH).
- **Per-mode local action bases:** APPROACH = forward effort + acquisition squeeze; HOLD = transport target + squeeze +
  balance/slip; BRAKE = deceleration demand + non-reversing stop + squeeze-decay; RELEASE = certificate guard only.
- **The gate** learns mode + control from structured state + history — *not* a single dtz threshold (C0/M0 show s1/s3 separate
  on dtz0, but that is 2 points; the gate input is initial/current distance, contact geometry, short-prefix response,
  tip–coin slip, momentum build-up, authority/preload history). Start with a hand dev gate, then learned mode inference.

This is the minimal instance of the program goal: **automatic discovery + learning of a hybrid dynamical system** (modes,
per-mode continuous dynamics/policy, and mode-transition guards learned jointly) from structured physical interaction —
HyMeKo as the representation, CIP as the execution protocol, learning the mode structure/guards/bases, not just the action.

Refined gates: **R10-C0 done** → C1 event-aligned per-phase teacher-vs-scaffold divergence + unexplained-component
localisation → C2 minimal per-mode primitives justified by the phase audit (APPROACH primitive first) → C3 near K6 s1 / far
K6 s3 with the mode program → C4 causal gate dev 2/2 → C5 bounded per-mode residual TD3 → C6 validation s4/s7 → C7 champion
freeze → **C8 single blind f1–f4 opening**. The 20 mm K6 tolerance, physics, motion limits and R6 certificate stay fixed;
s4/s7 remain validation-only; the blind panel opens once, after the R10 champion freeze.

## Artifacts

`reports/2026-07-27-coin-r9-causal-residual-delivery/`: `stage0_preflight.json`, `final_panel_manifest.json` (SEALED),
`stage2_update_zero.json`, `stage3_C.json`, `stage3_A_smoke.json`, `stage3_residual_ceiling.json`, `stage3_verdict.json`,
`ckpts/r9_C_seed{0,1,2}_best_val.pt`.
Code: `hymeko_rl/coin_delivery/theta_option/{r9_causal_residual,r9_delivery_train}.py`, `hymeko_rl/experiments/{coin_r9_blind_panel,coin_r9_causal_rl}.py`, `hymeko_rl/tests/test_coin_r9.py`.
Gates: ruff clean; no fn ≥ CC 15; R9 unit tests pass; CORE.YAML untouched; blind panel never evaluated.
