# R10 Stage 2 — capture-residual exploration audit: IID per-step exploration is inadmissible on the slew-integrated capture interface (negative-first)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · parent `d6974b95` · dev s1 (14250) · downstream + transit + Stage-1B scaffold FROZEN · s4/s7 untouched · f1–f4 SEALED · no learned capture policy claimed · no tag moved**

## Summary

Stage 2 set out to turn the Stage-1B phase-shaped capture scaffold into a **reward-driven, state-dependent, robust**
residual policy (TD3), with the *learning* claim resting on improvement over the frozen scaffold and a BC-only baseline
on a pre-registered perturbation panel — not on canonical-start K6 (the scaffold already passes that). The first attempts
degraded the scaffold, and an initial over-strong diagnosis ("RL learning is the blocker" / "reactive-policy
unsolvable") was **retracted after review** as unsupported. This report records only what the data supports, and treats
the outcome as a genuine negative-first result, not a failure.

**Verified discovery.** On this capture interface the action is a *slew-normalised torque change*, and the deliverable
handoff is *wide on momentum direction but very narrow on preload*. Under those conditions, **per-step IID residual
exploration removes the delivery at every tested scale**, while a **temporally-coherent** residual delivers — a strong
signal that the *temporal structure* of the correction is load-bearing, and that vanilla TD3's IID Gaussian behaviour
policy explores in a physically wrong coordinate for a slew-integrated contact controller. This is **not** an RL failure
and **not** evidence that a reactive policy is impossible; it is a concrete statement about the exploration geometry.

Separately, the audit uncovered a **metric error I had been carrying**: the medoid scaffold delivers at terminal
`||prev_tau − tau_star|| ≈ 0.88`, i.e. **tau_star (the cradle) is not the medoid scaffold's terminal preload
reference**, so any cradle-centred preload metric (used in my discarded reward and in the first admissibility read) is
invalid for this scaffold.

## Supported verdicts

- `REAL_MEDOID_SCAFFOLD_IDENTITY_PASS` — the medoid `pi_0` with a genuinely-zero-output actor (verified, not mocked)
  composes **bit-exact** to the scaffold and delivers **strict K6 (min_dtz 2.79 mm)**.
- `IID_STEPWISE_RESIDUAL_EXPLORATION_INADMISSIBLE_UNDER_CURRENT_CAPTURE_INTERFACE`.
- `CRADLE_CENTERED_PRELOAD_METRIC_INVALID_FOR_MEDOID_SCAFFOLD` (terminal dτ-to-τ★ ≈ 0.88, not ≈ 0).
- `NO_REWARD_DRIVEN_CAPTURE_POLICY_YET`.

## Explicit non-claims

- RL is **not** blocked.
- Reactive policies are **not** shown impossible.
- Zero-integral noise is **not** sufficient (it is only marginally better than IID at these scales).
- The terminal-preload random-walk mechanism is **implied** by the action definition (`δτ_T ≈ α·Δτ·Σεₜ`) and the
  coherent-vs-IID contrast — it was **not directly isolated**, because the dτ-to-τ★ metric was mis-centred. Its exact
  characterisation needs rollout-level, *phase-conditioned* (scaffold-relative) coordinates.

## Provenance table (measured)

Medoid `pi_0`: s = 0.3793, preload_start = 0.4573, bmax = 0.4098, residual_norm = 0.892 (from the 3 Stage-1B solutions).

| probe | result |
|---|---|
| `pi_0` nominal, zero learned residual | **strict K6, min_dtz 2.79 mm**; terminal dτ-to-τ★ = 0.875 |
| real zero-init actor output on 160 panel states | **exactly 0.00e+00** (actor and its target copy) |
| exploration placement (reused `train_semi_mdp`) | noise added to the residual, then `clip[-1,1]`, then `×α` — **in residual space**, not the composed action |
| IID per-step, σ = 0.005 / 0.01 / 0.02 / 0.05 | K6 preserved **0/12** each |
| IID per-step, σ = 0.1 (rare) | 2/12 — **not** read as an improving trend |
| smooth-spline noise, σ ∈ {0.005…0.6} | K6 preserve 0.0–0.08 |
| zero-integral noise, σ ∈ {0.005…0.6} | K6 preserve 0.0–**0.25** (marginally better; not sufficient) |
| stepwise MPC teacher with **IID** candidates | recovers **0/7** |
| temporally-**coherent** episode-constant correction (CEM) | recovers **8/8** |

The IID vs structured ladder + the identity are reproduced by
`python -m hymeko_rl.experiments.coin_kinetic_capture_exploration_audit`
(`reports/2026-07-28-td3-capture-exploration-audit/exploration_audit.json`). The stepwise-IID-MPC (0/7) and coherent
(8/8) contrasts were established by inline diagnostic probes (commands in the session transcript); they are recorded here
as external results, not as production code.

## Files

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/moving_precapture.py` (+/− refactor) | expose the scaffold per-step action (`scaffold_action`/`apply_step`) — **bit-identical** (8 Stage-1B tests still pass) so a residual can wrap it action-preservingly |
| `hymeko_rl/coin_delivery/theta_option/capture_rl.py` (+180 new) | **verified reusable infra only**: `freeze_scaffold` (medoid), `capture_observation`, `ResidualCaptureRoll` (action-preserving), the deterministic panel + `evaluate_on_panel`, `make_zero_actor`. No training loop, no reward, no exploration policy |
| `hymeko_rl/experiments/coin_kinetic_capture_exploration_audit.py` (+150 new) | the diagnostic **negative-control** audit (identity + IID/smooth/zero-integral ladder + verdicts + non-claims). Imports no trainer |
| `hymeko_rl/tests/test_capture_rl.py` (+110 new) | 8 tests: medoid, panel, bit-exact identity, **real actor exactly-zero**, real-medoid strict-K6, obs schema, perturb-keeps-coin-canonical |
| `reports/2026-07-28-td3-capture-exploration-audit/exploration_audit.json` | machine-readable audit result |

**Quarantined / NOT committed:** the IID-exploration training path (`CaptureOptionEnv` over `train_semi_mdp`) and the
cradle-centred reward — the refuted approach. Nothing imports them.

**CORE.YAML items touched: none.** Frozen downstream, transit, and the Stage-1B scaffold machinery — `git status`
confirms untouched (the one modified committed file, `moving_precapture.py`, is the bit-identical scaffold-action
refactor). `ruff` clean; `radon cc` no block at C or worse; test suite **8 passed**. Env: Python 3.11.15 / mujoco 3.10.0
/ numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). Peak RSS < 1 GB.

## Fresh-session handoff (next investigation) — fix the *action coordinate*, keep TD3

The fix is not to abandon TD3 but to move the action out of the per-step slew-increment coordinate (where it explores
meaninglessly) into a **scaffold-relative desired-torque-path coordinate** (where the positive control already lives).
The learned decision is made **once, at READY**, as a bounded structured option vector; the per-step physical action is
a *deterministic tracker* of the resulting smooth torque path — so exploration stays temporally coherent by construction.

**Structured option action (emitted once at READY):**
`θ = (Δs, Δpreload_start, Δbmax, two 4-D transient torque-offset spline knots c₁,c₂, one optional 4-D terminal offset Δτ_T)`.

**Desired torque path** over the frozen medoid nominal `τ₀(φ)`:
`τ_des(φ) = τ₀(φ) + Σ_k c_k ψ_k(φ) + ψ_T(φ)·Δτ_T`, with transient bases `ψ_k(0)=ψ_k(1)=0` (modify the middle, **no
unnoticed endpoint drift**) and terminal basis `ψ_T(0)=0, ψ_T(1)=1` (preload change is **explicit and bounded**, never a
noise integral).

**Deterministic per-step tracker (the key):**
`a_t = clip((τ_des(φ_{t+1}) − prev_tau_t) / slew, −1, 1)` — a tracking form that cannot accumulate action error. **No
per-step residual noise** is ever added after this conversion.

**Exact identity (test on the same code path used for training + deployment):** zero actor output ⇒ θ=0 ⇒ offsets 0 ⇒
`τ_des ≡ τ₀` bit-identical ⇒ HOME_V1 → strict K6. Gates: `TORQUE_PATH_ZERO_DELTA_IDENTITY_PASS`,
`TORQUE_PATH_BIT_EXACT_SCAFFOLD_PASS`, `OPTION_ZERO_POLICY_K6_PASS`.

**Exploration** is sampled **once per episode** in the structured-θ space (≤3 pre-registered scales), never per-step.
Gate `STRUCTURED_THETA_EXPLORATION_ADMISSIBLE`: the smallest scale giving both preserved successes and safe
near-boundary failures; verify the executed terminal torque offset matches the requested `Δτ_T` (`NO_PRELOAD_RANDOM_WALK`).

**Reward:** dominant strict-K6 + bounded downstream min_dtz + safety + boundary correctness; shaping **only**
scaffold-phase-relative (`q−q₀(φ)`, `qvel−qvel₀(φ)`, `prev_tau−τ₀(φ)`). **Remove every cradle-centred τ★ term.**

**TD3 campaign:** structured option action, one decision from the causal READY observation, deterministic capture; 3
seeds, ≤400 option-episodes/seed, deterministic eval every 25 (no exploration), same frozen dev panel. Compare frozen
medoid / zero-output actor / post-TD3. Success `STRUCTURED_OPTION_TD3_IMPROVES_OVER_SCAFFOLD_PASS` = paired improvement in
≥2/3 seeds while preserving nominal HOME_V1 K6, safety, and all boundary contracts; else emit only
`STRUCTURED_OPTION_TD3_NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET` (never an RL-impossibility claim). No SAC/PPO until this
coordinate is frozen — the same interface then serves all three algorithms, keeping the later comparison fair.

## Status

Negative-first, honest, and bounded: the full **HOME→strict-K6 scaffold is proven and frozen** (Stage 1B), and Stage 2
precisely identified that **vanilla TD3's IID data-collection geometry is incompatible with a slew-integrated,
preload-history-dependent contact handoff**. The reusable infrastructure and the audit are committed; the reward-driven
policy is **not** claimed. **STOP** — the structure-preserving protocol is the next session's focused work.
