# Cooperative two-contact launch (step 1) — reachability + mobile-coin block acquisition

**Date:** 2026-07-25
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4` + coast target + B1 barrier. Deterministic, no RL.
**One-line outcome:** step (1) — a two-contact reachability planner + decoupled synchronized close — is **not sufficient**
to exercise the cooperative launch: two-contact is reachable on only 3/8 states (the right arm cannot reach the coin on
the rest), and even on the reachable states simultaneous acquisition fails (0/8 both-tips) because the light coin escapes
the pinch. So I did **not** proceed to step (2). `TWO_CONTACT_REACHABILITY_LIMITED_BY_EMBODIMENT_AND_START_CONFIG`.

---

## What (1) built, and the two diagnostics that shaped it

- **DOF↔arm mapping:** DoF 0–1 = LEFT arm, 2–3 = RIGHT arm — independent. So each arm is driven with its own DoF
  (`_arm_dir`); the bimanual-V1 *coupled* 4-DoF gradient had let one arm dominate.
- **Reachability probe** (each arm alone → coin): the **RIGHT arm cannot reach the coin on 5/8 states** (min tip-coin
  0.04–0.07 m, contact needs ~0.02–0.03 m); the LEFT arm reaches on all 8. Two-contact is therefore **geometrically
  impossible** on those states — a relational fact of coin-position × arm-workspace, not a control failure.
- **Cooperative controller:** decoupled synchronized symmetric close (both arms → coin centre, self-correcting) → the
  coin-twist-Jacobian allocation (translate toward zone, zero spin).

## Benchmark (8 states)

| metric | value |
|---|---|
| two-contact reachable (both arms reach) | **0.375** (3/8: s1, s3, s6) |
| both-tips contact achieved | **0.0** (0/8) |
| both-tips on the *reachable* states | **0.0** |
| target-directed (full gate) | 0/8 |

Two independent blocks:
1. **Reachability / start-config:** on 5/8 states the coin sits where only the left arm reaches — the panel's start
   distribution + the two-arm workspace make two-contact impossible there.
2. **Mobile coin:** even on the reachable 3/8, the synchronized close never lands both tips at once — the first tip to
   touch pushes the light, low-friction coin away before the second arrives (the same squirt dynamic that made single-tip
   transport hard). ω_c stays ≈0 throughout (no spin) because no sustained two-sided load ever forms.

```
VERDICT: TWO_CONTACT_REACHABILITY_LIMITED_BY_EMBODIMENT_AND_START_CONFIG
(sub-finding: on the reachable subset, simultaneous acquisition is defeated by the mobile coin)
```

## Interpretation — this is where the scenario, not just the controller, is implicated

The cooperative force-allocation mechanism was proven sound earlier (ω_c ≈ 0 where both engage); the block is upstream, in
**getting two simultaneous contacts** — and it is blocked *both* by the start-config reachability distribution *and* by
the mobile-coin dynamics. Neither is solved by a better local controller alone:

- The reachability block is a property of **where the coin spawns relative to the two arms** — arguably a panel/scenario
  design choice (the coin often spawns on one side), not a fundamental embodiment limit.
- The mobile-coin block needs the coin **constrained** during the pinch (e.g. trap it against the B1 barrier, or an
  impulsive synchronized strike), not a slow symmetric close.

Both are **relational / dynamic facts** — exactly the structure a HyMeKo prior would represent (reachable contact pairs;
contact dynamics under a mobile object). The benchmark saves per-state teacher records (reachability + acquisition outcome)
for that study.

## Claims / non-claims

**Claimed (measured):** two-contact reachable on 3/8; both-tips acquisition 0/8 (blocked by reachability on 5/8 and by the
mobile coin on the reachable 3/8); the cooperative launch is therefore un-exercised across the panel.

**NOT claimed:** that two arms fundamentally cannot do it. The reachability block looks like a start-config/panel property
(re-centre the coin spawn and both arms may reach); the mobile-coin block needs a constrained or impulsive pinch. The
embodiment is not proven incapable — the current *panel + slow-close strategy* is.

## Exact next gate (step 1 not sufficient ⇒ step 2 deferred)

1. **Fix acquisition, two ways:** (a) a panel with the coin reachable by both arms (or a reposition phase that centres the
   coin first), and (b) a **constrained pinch** — trap the coin against the B1 barrier (already proven) while the two tips
   close, or an impulsive synchronized strike — so both-contact forms before the coin escapes.
2. Only once two-contact is reliably acquired is the cooperative launch (and step 2, the structural-prior study) worth
   running. **O3 stays paused.**

---

### Commits
- `306b2d30` — cooperative controller (reachability planner + decoupled synchronized close + twist allocation) + benchmark.
- (this report) — final.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
