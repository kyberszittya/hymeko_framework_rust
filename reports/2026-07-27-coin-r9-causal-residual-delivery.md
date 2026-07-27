# Coin-R9 — delivery-focused causal residual TD3 over the frozen R8 champion

**Date:** 2026-07-27 · **Branch:** `recovery/coin-r9-causal-residual-delivery` (worktree `hymeko_coin_r9_wt`, off tag `coin-r8-bounded-residual-heldout-improvement-v1`)
**STAGE 3/4 verdict:** `R9A_BOUNDED_CAUSAL_RESIDUAL_STRUCTURALLY_CANNOT_DELIVER_S1` (a localized `CURRENT_CAUSAL_RESIDUAL_FORMULATION_INSUFFICIENT`) — the dev gate (strict K6 on **both** s1 and s3) is **not** reached; **STAGES 5–6 not entered; the sealed blind final panel remains SEALED_NOT_EVALUATED.**

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
| **S3 ceiling** (discriminating) | **`STRUCTURAL_BOUND_INSUFFICIENT`** — s1 best_dtz **52.3 mm, delivers=False** over ~130 bounded Δa; s3 17.7 mm delivers=True |

## The finding (measured / inferred)

- **Measured.** The per-decision TD3 (400 rollouts × 3 seeds, K6-independent Curriculum-C reward, warm-started so Δa≡0 = the
  R8 champion) does **not** improve dev delivery: every seed holds at K6 1/2 (s3 delivers via the R8 base; s1 stays ~53 mm),
  safe throughout. The engine is sound — update-0 reproduces the champion (distill loss `0.0`), updates fire, live signal is
  clean.
- **Measured (ceiling).** No bounded Δa over the frozen R8 champion delivers s1: the constant-residual sweep bottoms out at
  **52.3 mm** (vs the R8 base's 53.5 mm — essentially no gain), and the temporal TD3 matched it (~53 mm). The bounded
  residual space over this base **does not contain an s1-delivering solution**.
- **Inferred (localized, NOT a single-cause guess).** The stall is **structural, not optimizer/reward**: TD3 found nothing
  better because nothing better exists within the declared bound. The frozen R8 champion's s1 base
  (`a_R8[s1] = [+0.68, −0.99, +0.97]`) sits in a **non-delivering basin** that a bounded (±0.25 fwd / ±0.04 sqz / ±0.25 stop)
  increment cannot escape. Temporal freedom and per-step control (R9's addition) do not help here because the *reachable set*
  is bound-limited, not schedule-limited.
- **Consequence.** R9-B (terminal-phase shaping) would **not** help — it is still a bounded residual, and the ceiling covers
  all bounded residuals regardless of the reward. Per the R9 process rule, a larger correction space / a different base / a
  richer policy is **R10, not a silent R9 modification**. s1 is deliverable by *some* θ (the oracle witness), so the task is
  feasible — just not from the R8 champion's s1 basin under a bounded nudge.

## Honest scope / no overclaim

R9 does **not** deliver on dev, so nothing downstream is claimed: no validation (s4/s7) run, the blind final panel is
untouched, no delivery is asserted without strict K6 (process rule honored). R8's result stands unchanged. The R9 negative is
**preserved**, with a decisive ceiling that localizes the blocker to the residual **bound over the frozen base**.

## Next discriminating test (for R10, not run here)

Re-run the ceiling with **enlarged Δa bounds** (and/or a re-selected base) on s1: if a larger bound delivers s1, the limit is
purely the nudge magnitude (a bound choice); if not, s1 needs a different *strategy* than the R8 champion's tip-referenced
push — a base/architecture change (R10). Keep the 20 mm K6 tolerance fixed either way.

## Artifacts

`reports/2026-07-27-coin-r9-causal-residual-delivery/`: `stage0_preflight.json`, `final_panel_manifest.json` (SEALED),
`stage2_update_zero.json`, `stage3_C.json`, `stage3_A_smoke.json`, `stage3_residual_ceiling.json`, `stage3_verdict.json`,
`ckpts/r9_C_seed{0,1,2}_best_val.pt`.
Code: `hymeko_rl/coin_delivery/theta_option/{r9_causal_residual,r9_delivery_train}.py`, `hymeko_rl/experiments/{coin_r9_blind_panel,coin_r9_causal_rl}.py`, `hymeko_rl/tests/test_coin_r9.py`.
Gates: ruff clean; no fn ≥ CC 15; R9 unit tests pass; CORE.YAML untouched; blind panel never evaluated.
