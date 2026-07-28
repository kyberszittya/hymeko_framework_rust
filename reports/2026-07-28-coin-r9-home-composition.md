# R9 HOME-start composition — the frozen stack needs one learned upstream skill (HOME → precontact); everything downstream composes

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · immutable base `d55f5017` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO RL · no tag moved**

## Summary

Toward the stricter compositional benchmark — *from a collision-free home posture, the arms autonomously approach the coin, execute
the explicit hybrid handoff-reset, and use the learned R2 transport policy to deliver strict K6* — the decisive **no-learning** first
question is answered: **the existing frozen APPROACH does NOT reach the handoff-basin from HOME.**

- `HOME_STATE_V1` defined: both arms at a fixed retracted home q `[-0.9, -1.4, 0.0, 2.7]`, qdot = 0, prev_tau = 0, **no contact**
  (tips ~130–160 mm from the coin), fresh history/recurrent state, coin at the canonical s1 cradle (dtz 76.4 mm).
- Rolling the frozen chain (APPROACH → [HANDOFF_RESET] → R2) from HOME, teacher-free, no snapshot injection: **all 60 steps stay
  FROZEN/REGULATE** — the arms never establish contact, the phase machine never transitions to KINETIC, so **0 HANDOFF_RESET, R2
  never engages, the coin never moves (min_dtz stays 76.4 mm), no K6.** Confirmed robust across close *and* far home postures
  (57–160 mm tip distance) — it is not a reach-envelope artifact.

**Verdict: `HOME_START_COMPOSITION_NEEDS_UPSTREAM_REACH_SKILL`.** The frozen APPROACH is a *momentum-build-from-straddle* phase; it
has no reach-to-contact capability and requires the arms to already straddle the coin (as the certified cradle provides). The
compositional benchmark therefore needs exactly **one** new upstream skill — HOME → stable precontact / straddle-entry — with the
entire downstream stack (HANDOFF_RESET semantics, R2 transport, release/coast, K6 monitor, physics, safety) **frozen**, so the
learning is cleanly attributable to the reach alone.

## Gates

| gate | result |
|---|---|
| `HOME_REACH_PASS` (arms reach precontact / KINETIC entry) | ❌ — never reaches KINETIC (60/60 FROZEN) |
| `HOME_TO_HANDOFF_PASS` (exactly one HANDOFF_RESET) | ❌ — 0 HANDOFF_RESET |
| `HOME_TO_K6_PASS` (teacher-free strict K6) | ❌ — coin never moves, no K6 |

## What this does and doesn't change

- **The reproduced cradle-start result stands unchanged** — R2 under H1 = 22/24 verified K6 on the fixed cradle-start contract.
  HOME-start is a *stricter* composition on top of it, not a replacement.
- No downstream policy was modified; this is a read-only audit. `HOME_STATE_V1` is a start-state definition (like the cradle
  itself), not a mid-rollout state edit.

## Next (needs green) — learn only the reach, freeze everything else

A minimal upstream policy `HOME → stable precontact/straddle-entry` (its own gates: reach the legal precontact band, no forbidden
collision, no premature coin perturbation), composed with the frozen HANDOFF_RESET + R2 + coast/K6. The end-to-end target is
`HOME_START_END_TO_END_K6_COMPOSITION` (teacher-free, no snapshot injection). This is a new (small, well-scoped) RL sub-project and
is **deferred for review** — no RL was started.

## Files

| file | role |
|---|---|
| `hymeko_rl/experiments/coin_kinetic_home_composition.py` (+92) | `HOME_STATE_V1` (`build_home_state`) + no-learning composition audit + 3 gates |
| `reports/2026-07-28-coin-r9-home-composition/home_composition.json` | the audit result |

`ruff` clean; `radon cc -a` A/B. All `8a0c1c7b`/`41510cac` modules imported unchanged; frozen modules `git diff` empty. Python
3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU).

## Status

`HOME_START_COMPOSITION_NEEDS_UPSTREAM_REACH_SKILL`. `HOME_STATE_V1` is frozen and the no-learning composition is cleanly negative:
the reach is the single missing skill; the whole downstream hybrid stack is ready to compose. **STOP** — awaiting green to train the
HOME → precontact reach (downstream frozen), then the end-to-end home-start K6.
