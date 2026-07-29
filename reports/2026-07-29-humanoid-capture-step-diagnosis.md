# Capture-point step: the target is formal, but the action space CAN'T lift a foot — 0/12 is a mechanism limit

**Date:** 2026-07-29 (JST) · **Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION. Diagnostic (no controller shipped).** · **Verdict: `STEPPING_IS_ACTION_SPACE_LIMITED_NOT_A_CONTROL_GAP`.**

---

## The plan (and why it stopped)

Part (2) gave the formal step target: when capturability = 1, place the swing foot at the capture point
``ξ`` so the new support contains it (level → 0). The next step was a capture-point step controller. Before
building it, the mechanical prerequisite: **can the action space lift a foot?**

## The blocking finding — the foot does not lift

The balance env is a **bounded position servo** (joint-target offset, `delta_scale=0.4`, `kp=60`, `kv=10`,
`tau_max=150 N·m`). Commanding every plausible lift — weight-shift (hip abduction toward the stance foot) +
swing-leg hip/knee flexion + ankle roll, at full authority, over 120 steps:

| attempt | max swing-foot rise |
|---|---|
| flex swing hip+knee | **+0.000 m** (foot goes *down* −0.022 quasi-statically) |
| shift-weight + flex | **+0.000 m** |
| ankle-roll + flex, all sign combos | **+0.000 m** |

**The swing foot never leaves the ground.** Leg flexion squats the pelvis (both feet planted) rather than
unloading and lifting one foot — because unloading a foot requires the CoM fully over the stance foot
(single support), which the bounded position servo cannot achieve on this ~symmetric stance. A real
protective step needs ~0.03–0.05 m clearance; the action space delivers 0.

## Conclusion — the same meta-lesson as the AIBO

The protective-stepping failure (**0/12**, across all prior RL residual / foot-clearance / step-shaping
attempts) is an **action-space / mechanism limit, not a control-policy gap**. The **capturability
certificate correctly identifies WHEN a step is required** (the 1-step region) and **WHERE it must land**
(``ξ`` + margin) — but the action space **cannot execute** the step. No controller (RL or model-based
capture-point) can produce a motion the actuation cannot.

This mirrors the AIBO exactly: the Vukobratović-ZMP / capturability framework is correct and unified, but
in **both** embodiments the **binding limit is the mechanism / action space**, not the policy — AIBO: the
rotational couple caps at ~47°/1000 (faster tips); humanoid: the position servo can't lift a foot to step.
The certificate says what is needed; the mechanism can't deliver it.

## What a genuine step would require (unaddressed, honest)

- A **torque/impulse action model** with the authority to unload one foot (dynamic weight transfer), or a
  model with an explicit **foot-clearance** phase — an *action-space / model* change, per CLAUDE.md a
  first-class change, not a control tweak.
- Then the capture-point target from part (2) becomes actionable: swing foot → ``ξ`` gated by the
  capturability level. The target is ready; the actuation is the gap.

## Files

```
reports/2026-07-29-humanoid-capture-step-diagnosis.md   NEW  (diagnostic; no controller shipped — the foot cannot lift)
```

CORE.YAML: none. SIMULATION. No code shipped: a capture-point step controller cannot function while the
action space yields 0 foot clearance — reporting the diagnosis instead (CLAUDE.md: no broken feature).
