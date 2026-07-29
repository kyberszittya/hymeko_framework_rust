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

## Addendum — the dynamic (Lyapunov-region) step, tested; root cause = the planted stance

The step is not static: it passes through a temporarily-unstable state (ZMP leaves support, the CoM
"falls" toward the stance foot to build momentum) but stays inside the **capturable / Lyapunov region**,
then the swing foot catches it. Tested this **dynamically** (a lateral sway pulse to build CoM momentum,
timed swing-foot lift, low posture stiffness) and with a direct Jacobian ``τ = Jᵀf``:

- Dynamic sway + lift (posture-Kp 2–10, sway/lift forces to 300/250 N): **swing-foot rise +0.000 m**.
- **Sanity check** — how far can the CoM even move laterally? A **220 N** lateral pelvis force shifts the
  CoM only **0.039 m** (0.046 m even with *zero* posture PD, free joints); neither foot unloads or lifts;
  the pelvis stays upright (the planted feet hold it). To unload one foot the CoM must reach ~0.09 m
  (over the stance foot) — it gets halfway, stiffly.

**Root cause:** the humanoid (34.6 kg, feet 0.18 m apart, 25 firm ground contacts, balance-tuned) is a
very *stably-planted* double-support stance — excellent for the certified in-place balance, but too stiff
to afford the **dynamic double→single-support weight transfer** a step requires. The double-support closed
kinematic chain resists the lateral CoM travel; the chicken-and-egg (unload a foot ⇐ CoM over the other ⇐
shift weight ⇐ unload) does not break under any tested excitation. The user's dynamic-Lyapunov framing is
correct; **this model cannot execute it.**

**Enabling a step is a first-class MODEL change** (per CLAUDE.md): a narrower initial stance (less CoM
travel to unload), a lighter/taller build, or an explicit swing-phase / lateral-sway mechanism in the
`.hymeko` — a deliberate design decision, not a control tweak. The capture-point target (part 2) is ready
to drive it once the model affords the maneuver.

## Addendum 2 — narrow-stance model change tested: confirms the direction, does not solve it

Per the model-change decision, tested the cheapest option: a **narrower initial stance** (adduct the hips
in the reset pose → foot spacing 0.171 m → 0.128 m at adduct 0.15, still upright; adduct 0.3 breaks the
pose to 0.54 m). From the narrow stance, a dynamic Jacobian sway + swing-foot lift gives **swing-foot rise
+0.004–0.005 m** (vs +0.000 at the wide stance) — the CoM still moves only ~0.01 m and the wrong way under
the ``Jᵀf`` sway. So narrowing the stance **marginally helps and confirms the direction** (less CoM travel
→ some foot clearance), but does **not** produce a step: ~0.005 m ≪ the ~0.03–0.05 m needed.

**Honest conclusion:** a working humanoid step needs **both** a model change (narrower stance and/or a
lighter/taller build with more sway authority) **and** a dedicated **whole-body stepping controller** (a
proper CoM trajectory + swing-foot placement + double→single→double phase timing — the core of bipedal
locomotion), not the quick Jacobian sway tried here. This is a substantial, deliberate design effort; the
capture-point target (part 2) + the ZMP/capturability certificates (parts 1-2) are the ready substrate for
it. No step controller shipped (the ~0.005 m clearance is not a step; CLAUDE.md: no broken feature).
