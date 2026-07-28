# R10 Stage 0 — the K6-compatible dynamic handoff basin H_dyn (a set with tolerances, not the cradle point)

**2026-07-28 · immutable base `10aced90` · downstream FROZEN · NO RL · dev s1 · s4/s7 untouched · f1–f4 SEALED**

## Summary

Toward `HOME_TO_DYNAMIC_STRADDLE_CAPTURE`, this maps the K6-compatible **dynamic** handoff basin `H_dyn = { z : frozen APPROACH →
H1 → R2 gives strict K6 }` by targeted perturbation of the certified cradle handoff, each run through the FULL frozen downstream and
labelled. (State-editing is a characterization tool, as in the frozen-policy intervention; the planner/RL must reach the basin from
HOME with no state edit.)

**Verdict: `DYNAMIC_HANDOFF_BASIN_CHARACTERISED`.** The handoff is a dynamic SET, not the single cradle state:

| axis | K6 band | tolerance |
|---|---|---|
| arm qvel magnitude (× cradle momentum) | **[0.6, 1.25]** | wide — the momentum has real slack |
| prev_tau (× cradle) | [0.95, 1.1] | narrow (±5–10 %) |
| arm q (uniform offset, rad) | [0.0, 0.01] | narrow (config-sensitive) |
| coin position (m) | [0, 0] | ~exact — but the coin is FIXED at canonical (not a reach constraint) |
| coin velocity (m/s) | [−0.1, +0.1] | wide |

- **Factorial (fills the missing cell):** qvel×prev_tau are **jointly necessary** — qvel×1 + prev_tau×0 → **no K6** (35 mm),
  qvel×0 + prev_tau×1 → no K6; only both together deliver (1.0 mm, dwell 37). It is an interaction, not two separable factors.
- **Random box** (qvel 0.8–1.2, tau 0.9–1.1, q σ 0.01, coin σ 2 mm, qvel-dir noise 0.05): **25 % K6-compatible** — a non-trivial
  but reachable basin.

## Reading for the capture

The reach must arrive at ≈ the cradle configuration (narrow in q) with the right **torque pre-load** (prev_tau within ±5–10 %) and
**momentum** in a wide band (0.6–1.25× the cradle qvel), the coin held at its canonical position. Momentum tolerance is the good
news; the config + prev_tau are near-exact but controllable (the reach sets its own final q and torque). This is a velocity-matched
kinodynamic capture, not a static reach — a static-target planner would train against a bad target.

## Files

`hymeko_rl/experiments/coin_kinetic_dynamic_handoff_basin.py` (+~130) — the perturbation map (factorial + per-axis bands + random
basin). `reports/2026-07-28-coin-r9-dynamic-handoff-basin/dynamic_handoff_basin.json`. ruff clean; radon A/B; downstream frozen;
`git diff` empty on frozen modules.

## Next (needs green)

Stage-1 kinodynamic planner positive control: a short-horizon CEM/MPC on the upstream action that drives HOME → into `H_dyn` →
explicit `REACH_TO_APPROACH` → frozen downstream → strict K6, with lexicographic scoring (safety/collision ≻ no early coin sweep ≻
enter certified basin ≻ downstream K6 ≻ lower force/smoother), ≥ 3 planner seeds, NO state edit / snapshot injection. Gate
`HOME_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS`. Then the explicit boundary-equivalence tests, then the 3-seed TD3 smoke. **STOP.**
