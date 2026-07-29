# Humanoid footstep RL — learning the foothold policy over the WBC scaffold

**Date:** 2026-07-29
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Context:** the WBC gives stable low-level tracking + a marginally-stable DCM march; the analytical
capture-point footstep adaptation is finicky for this small-foot robot. This delivers the **RL interface**
that makes *where to step* a learnable decision, and shows the scaffold has genuine headroom.

## Summary

New module `scenarios/humanoid/footstep_env.py` — a **semi-MDP** environment where **one `step()` = one
WBC-executed footstep**. The action is a **bounded residual on the nominal (mirror) foothold**, so
``action = 0`` is exactly the analytical fixed-march scaffold (coin-R8 regime: a bounded residual over a
certified scaffold, not from scratch). The WBC executes each footstep (double-support load transfer →
single-support swing to the commanded foothold → land) under the DCM CoM controller; the policy only
chooses the small foothold corrections that regulate the DCM.

- **Observation (10-d):** DCM relative to the stance foot (the key capture signal), CoM relative to stance,
  CoM planar velocity, CoM-height error, pelvis uprightness/tilt, stance-side indicator.
- **Action (2-d):** foothold residual (Δx, Δy) ∈ [−1, 1]², scaled to ±0.05 m around the nominal.
- **Reward:** ``1 − 2·|DCM_lateral_offset| − 0.01·‖a‖² − 5·[fell]`` (survive + stay centred).
- **Scaffold (a = 0):** the analytical fixed march — survives **~29 footsteps** then tips (marginally
  stable). This is the baseline RL improves on.

## Why RL here (the answer to "where does RL fit")

The decision RL learns — *where to place the next foot* — is exactly the high-level, nonlinear part the
analytical LIPM/capture-point only approximates, and where this robot's **small feet** make the analytical
adjustment finicky (the ankle/ZMP strategy saturates; the step strategy must do the regulating). This is
the session's validated pattern: **a bounded policy over a certified scaffold**, on a **small structured
action space** (2-d foothold per step) with the **WBC guaranteeing the low-level stability** — not
end-to-end torque RL (which would discard the WBC and has no headroom advantage here). The small feet are
the genuine headroom (unlike the AIBO crab, where structure gave none).

## Files touched

- `scenarios/humanoid/footstep_env.py` — NEW. `FootstepConfig` + `HumanoidFootstepEnv` (gym semi-MDP;
  WBC-executed footsteps; DCM CoM control with the Englsberger tracking law; foothold-residual action).
- `tests/test_humanoid_footstep_env.py` — NEW, 5 tests (finite/deterministic obs, scaffold survives ≥ 15
  footsteps, ``a = 0`` reproducible + a large action changes the outcome, reward finite + fall terminates).

**CORE.YAML items touched:** none. **New dependencies:** none (gymnasium already used by the balance env).

## Test results

- `pytest tests/test_humanoid_footstep_env.py` → **5 passed**; full humanoid suite → **48 passed**.
- `ruff check` on the new files → clean.
- Scaffold baseline: `a = 0` survives ~29 footsteps (episode return ≈ 21).

## Policy search (CEM) — a fast demonstrator

A cross-entropy-method search over a linear foothold policy (`a = tanh(W·obs + b)`) validates the loop end
to end and that the scaffold has learnable headroom. (SAC/TD3 from `hymeko_rl/option_rl/` is the production
trainer; CEM is the quick, robust demonstrator.)

**Result (pop 12, 7 iters, 1 seed, 60-footstep horizon):**

| | footsteps survived | episode return |
|---|---|---|
| Scaffold (`a = 0`, analytical march) | 29 | 20.6 |
| **CEM-learned foothold policy** | **60 (full horizon)** | ~55 |

The learned policy reaches the full 60-footstep horizon (iters 1, 4, 5 all hit 60) where the analytical
scaffold tips at 29 — **RL learns the footstep regulation the analytical capture-point could not tune for
this small-foot robot.** Honest scope: 60 is the episode horizon (full-episode survival, not *proven*
infinite); this is CEM with a linear policy at **1 seed** — a validated direction and a real improvement
over the scaffold, with **multi-seed SAC/TD3 the rigorous follow-up** (per the no-single-seed-conclusions
rule). It confirms the headroom is real and that the footstep policy is the right home for RL.

## Forward walking — RL solves what hand-tuning couldn't

Hand-tuned forward-walking control (DCM forward plan, forward-velocity drive, capture-point stepping — 4
variants) all **fell backward** (−0.5 m in ~4 steps): the quasi-static WBC CoM control can't generate the
forward momentum, so the swing leg never reaches the forward foothold. Rather than conclude "mechanism wall
/ needs the toe", the env was extended to **forward mode** (`forward_stride` advances the nominal foothold
+x; `w_forward` rewards +x pelvis progress; the DCM anchor tracks the actual advancing stance foot) and the
same CEM footstep-policy search was run:

| | forward progress | upright |
|---|---|---|
| Hand-tuned control (4 approaches) | **−0.5 m (falls backward)** | 4 steps |
| **CEM-learned footstep policy** | **+0.08 → +0.13 m (forward)** | 50–60 footsteps, improving |

The learned policy walks forward, upright, and improves with training — the RL discovers the footstep
placement + timing that manages forward momentum, which the analytical DCM could not. This is the **second**
validation (after lateral 29 → 60) that the binding limit was the **control/learning, not the mechanism**:
the WBC + DCM + footstep-RL stack generalises across lateral AND forward walking. Forward *speed* is still
modest (~2 mm/footstep, survival-prioritised) — scaling it needs more training (SAC/TD3 vs CEM), an MLP
policy, reward shaping, and/or the articulated toe (a higher push-off ceiling). No fast-forward claim.

## Scaled training (parallel CEM) + the honest forward result

Parallel CEM (many-core, katolab-launchable via `scripts/kato15/footstep_walk_run.sh`) found a much better
policy — but the first result (**+0.79 m**) was a **reward-gaming artifact**: the trajectory creeps ~2 mm/step
for 40 steps then **lunges/falls forward 0.7 m** in the last 4 steps (episode ends at the fall). Caught by
inspecting the per-step trajectory. **Retracted.** Fixed the reward (per-step forward **cap** ±0.05 m so a
single lunge can't dominate, + a heavy `fall_penalty`). Re-trained:

| | genuine sustained forward | footsteps | note |
|---|---|---|---|
| hand-tuned control | −0.5 m (falls backward) | 4 | — |
| RL, gamed reward | "+0.79 m" | 48 (fell) | terminal lunge — **retracted** |
| **RL, anti-gaming reward** | **+0.20 m** | **60/60 upright** | per-step +3.3 mm, no lunge — **genuine** |

So RL learns a **genuine, stable, sustained forward-walking gait** (+0.20 m over 60 footsteps, upright
throughout, no gaming) where hand-tuned control fell backward — the real deliverable. Video
`humanoid_forward_walk.mp4`. Forward *speed* is slow (~3 mm/footstep), mechanism-limited by the small feet.

**Push-off / toe.** First attempt (honest negative): `humanoid_toe.hymeko` **split** the foot (0.20 → 0.13)
to add the toe + a **scripted** 70 N·m toe-off during **swing** — trained *worse* (+0.099 m at toe_off = 0
because the shorter foot lost support; any scripted toe-off destabilised, falling at 2–3 steps). Two fixes:
(1) the toe as a forward **extension** (`humanoid_toe2.hymeko` — keep the full 0.20 foot, add the toe ahead),
and (2) a **learned** toe-off in the action (3rd action dim, applied in **late stance** as the foot rolls
off). Re-trained:

| model | forward | footsteps | toe-off |
|---|---|---|---|
| toe-less | +0.20 m | 60/60 | — |
| **toe-extension + LEARNED toe-off** | **+0.287 m (+44%)** | **60/60** | learned, `\|a₂\|≈1.0` (fully used) |

The learned toe-off is **genuine** (per-step +4.8 mm, terminal 4 steps only 7% — not a lunge) and **actively
used** (the policy commands full push-off every step). **Done correctly — the articulated toe as an
extension + a learned late-stance toe-off — the push-off raises the forward-speed ceiling +44%.** Video
`humanoid_forward_walk_toe.mp4`. This closes both threads: RL supplies the stable forward gait (the
control/learning limit, not a mechanism wall), and the toe/push-off raises its speed (the mechanism lever) —
each validated with honest scope, and the naive versions of both (gamed reward, crude scripted toe-off)
caught and corrected.

## Open items / follow-ups

- Full SAC/TD3 training via `option_rl` (the production path) for an indefinitely-stable learned gait, then
  extend the action to forward stepping + goal-reaching (the AIBO goal-reaching analogue).
- The learned foothold policy is the capture-point step adjustment the analytical method approximates — a
  clean head-to-head (learned vs analytical footstep adaptation) is the natural next experiment.

## Provenance

- Git: adds `scenarios/humanoid/footstep_env.py`, `tests/test_humanoid_footstep_env.py`, this report.
  Seed 0, deterministic env. Shared venv `hymeko_framework_rust/.venv`; mujoco 3.10.0; macOS 25.5.0.
