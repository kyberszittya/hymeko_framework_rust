# CIP-PNP-01 — Acquire · Carry · Place (pick-and-place scenario)

**Date:** 2026-07-27 (JST)
**Branch:** `scenario/cip-pick-place-v0` (worktree `../hymeko_pick_place`)
**Base:** `hymeko-control-profile-v0` (`2210e4c9`)
**Highest gate passed: PNP-4 (complete externally certified trajectory).**

---

## Summary

Instantiated the frozen `hymeko-control-profile-v0` on contact-rich
pick-and-place and produced **one complete, natural-state, externally certified
trajectory**: APPROACH → GRASP → LIFT → CARRY → PLACE → RELEASE → SETTLE, ending
in `placed_stable` with the external certificate passing and no drop / no death.

The work **composes existing runnable assets** (audit-verified) rather than
rebuilding phases:

- `hymeko_rl/env/pick_place_env.PickPlaceEnv` via `viz.render_pick_place.fanuc_pick_env`
  (FANUC arm + 2-finger gripper + freejoint box + `target_bin`, `require_settle=True`);
- the built-in scripted expert **v3** (clean-transit → v1 grasp basin), the
  embodiment's authority-decoder / realizer;
- reward-independent physics signals (`both_contact, lifted, obj_to_target,
  placed_stable, settled, on_ground, death`) as certificate inputs.

New scenario code is only the CIP-0 **wrapping**: a HyMeKo contract, a CIP-0
adapter, and a reward-independent certificate suite.

## Semi-MDP mapping

One CIP-0 tick == one hybrid-mode **option**. `execute` runs the scripted expert
until the classified mode advances / the object drops / the episode ends; `decode`
is a pure deterministic function of `(intent, authority)`; the low-level joint
action is realized inside `execute`. **This is a baseline (scripted) trajectory,
not RL** — a learned decoder is future work (see RL-readiness).

## Gates (seed 1, expert v3)

| Gate | Requirement | Result |
|---|---|---|
| PNP-0 | schema + adapter conformance | ✅ (5/5 conformance tests) |
| PNP-1 | approach reaches legal acquisition | ✅ (bilateral contact, no death) |
| PNP-2 | bilateral grasp + lift | ✅ (`both_contact`, max_lifted **0.0603 m** > 0.035) |
| PNP-3 | grasp→carry handoff preserves object | ✅ (CARRY reached, contact retained, no drop) |
| PNP-4 | complete natural-state acquire-carry-place | ✅ (`SETTLE`, `placed_stable`, certificate passed) |

- **484 env steps**, 10 CIP-0 ticks, `final_mode=SETTLE`, `placed_stable_final=True`,
  `dropped=False`, `any_death=False`, `carry_contact_ok=True`.
- Mode classification is **Markov** and flickers at contact boundaries during
  place/release (visited-mode list oscillates PLACE↔LIFT↔GRASP↔RELEASE); every
  emitted transition remained **legal** in the declarative model (the runtime
  raised no `ModeError`), because `transition` walks the mode chain one legal step
  toward the classified target. This is honest physics, not a contract violation.

## Files touched (all NEW, non-core)

```
scenarios/__init__.py
scenarios/pick_place/__init__.py
scenarios/pick_place/cip_pnp_01.hymeko.yaml      (HyMeKo contract, schema v0)
scenarios/pick_place/adapter.py                  (PickPlaceCIPAdapter, 8-method CIP0Adapter)
scenarios/pick_place/certificate.py              (reward-independent suite)
scenarios/pick_place/run_pnp.py                  (gate runner + plot + gif)
tests/test_cip_pnp_conformance.py                (PNP-0, reuses hymeko_control.conformance.battery)
reports/2026-07-27-cip-pick-place/{pnp_gates.json, pnp_trajectory.png, pnp_trajectory.gif}
docs/plans/2026-07-27-cip-pick-place/ (plan.tex/pdf/tikz/mmd; gitignored)
```

## CORE.YAML items touched

**None.** The scenario depends on `hymeko_control` (core) and on `hymeko_rl` (the
embodiment); the core imports neither. Test `test_pnp0_core_import_isolation_preserved`
re-confirms `hymeko_control` imports no torch / hymeko_rl / scenario module.

## Test + lint results

- `pytest tests/test_cip_pnp_conformance.py` — **5 passed** in ~0.9 s
  (schema validates; malformed spec rejected; adapter is a `CIP0Adapter`;
  positive lifecycle conformance; core isolation preserved).
- `ruff check scenarios tests/...` — **all checks pass.**

## Graphical output (§9)

- Numerical: `reports/2026-07-27-cip-pick-place/pnp_gates.json`.
- Plotted: `pnp_trajectory.png` (lifted / obj_to_target / contact / placed / settled vs step).
- Animated: `pnp_trajectory.gif` (960×720 rollout of the certified trajectory).

## Performance

Full gate run (physics, no render): ~1–2 s wall for 484 steps; peak RSS well
under 2 GB, no GPU. GIF render adds a few seconds. Under §4's 16 GB cap.

## Reproducibility / provenance

- Deterministic given the seed (scripted expert, no RNG in the CIP-0 path). Seed 1.
- Requires the `hymeko` Rust CLI to emit the arm MJCF. The worktree has no `target/`;
  a symlink `target/release/hymeko → <main repo>/target/release/hymeko` was used
  (gitignored). A clean reproduction runs `cargo build -p hymeko_cli` in the worktree first.
- Python 3.11.15, mujoco 3.10.0, gymnasium 1.3.0, torch 2.12.0, numpy 2.4.6.

## RL-readiness (recorded, NOT started)

RL is **not** authorised yet and was not run. The baseline is the scripted expert.
For a learned decoder, the prerequisites (per campaign policy) are: update-zero
no-regression, exact action provenance, the unchanged external certificate above,
no hidden fallback, dev/held-out discipline. The audit found the strongest learned
arm pick-place checkpoints (DAgger 0.833, TD3+BC) **absent from this checkout**, so
a learned Stack-A policy would need retraining before any RL claim.

## Open issues / follow-ups

- Mode classifier could be hysteretic to suppress contact-flicker oscillation
  (cosmetic; the contract already holds).
- Success is demonstrated on a chosen passing seed (the gate is *one* certified
  trajectory). A multi-seed success-rate table is future work, not a PNP-4 gate.

**Verdict:** CIP-PNP-01 reaches **PNP-4** — a complete, externally certified,
natural-state pick-and-place trajectory under the CIP-0 profile. Tagging
`cip-pick-place-v0`.
