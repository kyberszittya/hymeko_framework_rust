# Katolab parallel humanoid-walk campaign — the mechanism wall, thoroughly confirmed (+ a flight-gait attempt)

**Date:** 2026-07-30
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Compute:** katolab (kato85, 64 cores) via a shared-home clone + the `~/envs/hymeko` env; scenarios run and
render headless (`MUJOCO_GL=egl`).

## Goal

The user wanted to *see the humanoid actually walk* (the prior +0.287 m footstep result is a low-lift
shuffle). Using katolab's parallel capacity, push the two available controllers to their limit and, if a
visibly-walking gait exists, render it.

## Part 1 — quasi-static WBC footstep sweeps (12 configs, 2 sweeps)

Parallel CEM over the footstep env, varying foot clearance (`step_h` 0.04–0.11), single-support timing, and
toe push-off. **The `best_fwd` numbers were reward-gaming and had to be verified by `best_steps`:**

| regime | best_fwd | best_steps (of 60) | verdict |
|---|---|---|---|
| light fall-penalty, high lift | **+1.76 m** | **8** | LUNGE then FALL (not walking) |
| heavy fall-penalty, high lift | −0.53 | 26–40 | falls backward |
| heavy fall-penalty, mid lift | +0.07 | **60** | sustains but barely moves (over-conservative) |
| moderate, low lift | +0.287 | 60 | the low-lift shuffle (prior best) |

**No config gives visible-lift + sustained + forward.** Higher lift → lunge/fall; forcing survival →
in-place. The `best_fwd`-only view would have reported "+1.76 m walking" — the `best_steps` check caught the
lunge-fall (same reward-gaming class as the session's earlier catches).

## Part 2 — a dynamic FLIGHT-PHASE gait (`flight_gait.py`, new)

A momentum-based alternative: a cyclic PD running gait (push-off + swing, **no** contact-consistency
constraint) so a flight phase is representable. **A first-cut "+0.13 m, ~20 % flight" result was a
measurement artifact** — `base_z0 = 0.95` left the feet 0.22 m in the air (the anatomical foot-body origin
sits high) so the rollout measured a *drop from height*, and flight detection used foot-body-z. Fixed: the
reset **settles** the feet onto the floor (`base_z0 = 0.76` + a q0-hold), flight = **no floor contact**
(contact-verified), `mj_resetData` clears the warmstart (deterministic). Katolab CEM (6 configs, high flight
weight to force lift-off, corrected metric):

| flight weight | best_fwd | best_flight | verdict |
|---|---|---|---|
| low (6) | **+0.14 – +0.17 m** | **0 %** | grounded forward shuffle |
| high (12–30) | −0.01 – −0.04 m | **10–13 %** | hops in place, no forward |

**Flight XOR forward — never both.** The gait either shuffles forward (no lift-off) or hops in place with a
genuine (but subtle, ~1 cm) flight phase. Video `humanoid_flight_hop.mp4` (F3, 13 % airborne): a real
contact-verified flight phase, in place.

## Conclusion — the wall is a mechanism limit, confirmed two ways

Across **both** control paradigms, on katolab, with corrected metrics: **visible forward motion trades off
against foot lift / flight.** The binding constraint is the **model/mechanism** — small feet (~5 cm, no
support margin for a dynamic step), a 2-D sagittal humanoid, and PD/WBC control — not the search (12+ configs,
parallel, multi-seed). This is not a tuning gap.

**What would actually break it** (each a real project, not more CEM):
- **Bigger / longer feet** (a support-polygon margin the dynamic step needs), or
- **A contact-implicit trajectory optimizer** (DDP / MPC over the full dynamics — the standard way to get a
  running gait), or
- **Deep RL** (PPO/SAC with a running reward + curriculum + domain randomisation) rather than a CEM proxy
  over a hand-parameterised gait.

## Files

- `scenarios/humanoid/flight_gait.py` — NEW flight-phase gait env (contact-verified flight, planted reset).
- `scenarios/humanoid/train_flight_gait.py` — NEW parallel CEM trainer + headless render.
- `tests/test_humanoid_flight_gait.py` — 4 tests (planted+deterministic reset, push-off>no-push flight,
  contact-based detection).
- Videos: `humanoid_A2_stepping.mp4` (best sustained footstep, 6 cm lift, +0.07 m, leans),
  `humanoid_flight_hop.mp4` (flight-gait, 13 % airborne, in place). Videos gitignored.

**CORE.YAML:** none touched. **Provenance:** katolab kato85, `~/envs/hymeko` (mujoco 3.10, torch 2.11+cu128);
CEM seeds 0–1, deterministic env; git 9d705ee6.
