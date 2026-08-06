# CIP-HUM-01 — Stand · Shift-Support · Reach · Touch · Retract · Recover

**Date:** 2026-07-27 (JST)
**Branch:** `scenario/cip-humanoid-v0` (worktree `../hymeko_humanoid`)
**Base:** `hymeko-control-profile-v0` (`2210e4c9`)
**Highest GENUINE gate passed: HUM-1.** NOT tagged (tag only on a genuine complete pass).

---

## Summary

Built the complete CIP-HUM-01 scenario contract, a CIP-0 simulation-adapter
boundary over the HyMeKo humanoid, and HUM-0 conformance. A fixed-base
stand→reach→touch→retract→recover cycle runs end-to-end through the CIP-0 runtime
(informational). **The campaign's whole-body balance benchmark is NOT genuinely
met, and physical success is NOT claimed** — see the decisive audit finding.

## Decisive audit finding (why the ceiling is HUM-1)

`data/robotics/humanoid.hymeko` emits a **fixed-base** humanoid: `nq=nv=njnt=13`,
**no free joint**, the `base` joint range is `[0,0]` — the pelvis is welded to the
world. Consequences:

- The humanoid **cannot fall**. A hold-pose stays upright for 3000 steps trivially.
- Therefore **support margin, no-fall, support-shift and stance-recovery — the
  properties that DEFINE the humanoid benchmark — are vacuous** on it.
- `Humanoid-v5` (gymnasium) is a genuine floating base but the repo has **no
  balance controller / teacher** for it, and starting one is RL, which is not
  authorised without a certified baseline.

So a genuine whole-body stand-reach-recover-with-no-fall cannot be certified in
this checkout. This is recorded, not worked around.

## Gates (honest)

| Gate | Requirement | Result |
|---|---|---|
| HUM-0 | schema + adapter conformance | ✅ **genuine** (5/5 conformance tests) |
| HUM-1 | stable stand / support observation | ✅ **genuine** (stand stable, no divergence) — but balance is not *challenged* on a welded base |
| HUM-2 | bounded support shift without instability | ⛔ **BLOCKED** — no floating base; instability is impossible to induce or test |
| HUM-3 | reach + touch while support certificate holds | ⛔ reach is genuine (see below) but "support certificate holds" is **vacuous** — not a genuine pass |
| HUM-4 | complete stand→reach→retract→recover (no fall) | ⛔ **BLOCKED** — needs a floating-base humanoid + a balance/recovery controller |

**Informational (NOT a gate pass):** the fixed-base kinematic cycle
STAND→SHIFT_SUPPORT→REACH→TOUCH→RETRACT→RECOVER runs end-to-end; the left-arm
effector genuinely reaches a reachable target config (computed-torque PD with
gravity feedforward) and returns home, with joint/velocity limits preserved and
no divergence. This demonstrates the adapter drives a real CIP-0 lifecycle — it is
**not** evidence of balance competence.

## Files touched (all NEW, non-core)

```
scenarios/__init__.py
scenarios/humanoid/__init__.py
scenarios/humanoid/cip_hum_01.hymeko.yaml
scenarios/humanoid/adapter.py           (HumanoidCIPAdapter + HumanoidSim + computed-torque PD)
scenarios/humanoid/certificate.py       (joint-limit safety GENUINE; support-margin VACUOUS)
scenarios/humanoid/run_hum.py           (gate runner + plot + gif)
tests/test_cip_hum_conformance.py       (HUM-0, reuses hymeko_control.conformance.battery)
reports/2026-07-27-cip-humanoid/{hum_gates.json, hum_trajectory.png, hum_trajectory.gif}
docs/plans/2026-07-27-cip-humanoid/ (plan.tex/pdf/tikz/mmd; gitignored)
```

## CORE.YAML items touched

**None.** Scenario depends on `hymeko_control` + `hymeko_rl`/CLI; core imports neither
(re-checked by `test_hum0_core_import_isolation_preserved`).

## Test + lint

- `pytest tests/test_cip_hum_conformance.py` — **5 passed** (~0.25 s).
- `ruff check scenarios tests/...` — **all checks pass.**

## Graphical output (§9)

- Numerical `hum_gates.json`; plotted `hum_trajectory.png` (ee_to_target / ee_to_home
  / torso_z / max|qvel| vs step); animated `hum_trajectory.gif` (960×720, gitignored, reproducible).

## Performance

Full run ~a few seconds (computed-torque PD on a 13-DOF fixed-base sim), peak RSS < 1 GB, no GPU.

## RL-readiness (recorded, NOT started)

RL is **not** authorised. Prerequisite before any humanoid RL: a floating-base
humanoid model (add a free-joint root to `humanoid.hymeko` or adopt `Humanoid-v5`)
**and** a balance / support-shift controller (teacher) delivering a certified
stand-reach-recover baseline with update-zero no-regression and the genuine
support/no-fall certificate. None of these exist today.

## Exact remaining prerequisite (STOP condition)

> **A floating-base humanoid + a balance/support-shift controller.** Without a free
> root the no-fall/support-margin certificate cannot be genuinely evaluated; without
> a balance controller the floating-base humanoid falls immediately. HUM-2..4 are
> gated on these, not on more adapter code.

**Verdict:** CIP-HUM-01 reaches **HUM-1** genuinely and delivers the full contract +
adapter boundary + conformance the campaign's fallback requires. Balance is NOT
demonstrated. **No tag** is created.
