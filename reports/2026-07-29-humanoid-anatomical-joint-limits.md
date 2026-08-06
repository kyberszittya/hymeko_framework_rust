# Humanoid anatomical joint limits — fixing the wrong-direction / hyperextending joints

**Date:** 2026-07-29
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Trigger (user):** "the joints go the wrong direction" — building a gait on the uncorrected model is a
curiosity. Correct.

## The defect

The emitted model had **every revolute joint at `range="-3.1416 3.1416"` (±180°)** — the shared
`joint_rev_limit` default in `meta_kinematics.hymeko`. No anatomical limits: knees and elbows could
**hyperextend / bend the wrong way**, letting the controllers reach unphysical configurations.

## The fix

Per-joint anatomical limits in the model (bare limit nodes + `limit -> node` per joint — a model-level
override of the shared default, no `meta_kinematics` edit). **Flex directions were determined by
measurement, not by eye** (rendered left/right is easy to misread): commanding a joint and reading the
limb-tip's world-x displacement.

| joint | measured | anatomical range |
|---|---|---|
| hip flex (Y) | `+` = backward (extension) | forward-flex −120° … extension +25° |
| knee (Y) | `+` = foot back = flexion | −5° (no hyperext) … +150° flex |
| ankle pitch (Y) | — | ±40° dorsi/plantar |
| shoulder (Y) | `+` = arm back | −150° fwd … +60° back |
| elbow (Y) | flex forward | −150° flex … +5° |
| abdomen/neck (Y) | — | ±45° |
| hip abduction (X) | — | −30° … +45° |
| ankle roll (X) | — | ±25° |

`effort`/`velocity` were kept at the defaults (ctrlrange ±50 N·m unchanged) so **only the kinematic range
changes** — the actuator authority the balance stack was validated with is untouched.

## Verification

- **Left–right symmetry (the user's test): PASS, measured.** The same joint angle on both legs moves both
  the **same** direction (both feet Δx = −0.384 m at hip +0.6) — no mirror error. The pelvis local +x axis
  is world +x → **the robot faces forward (+x)**, and the foot geometry points +x = forward.
- A natural mid-stride pose (swing hip forward −0.5, knee flexed +0.6; counter-swinging arms) renders as a
  **real forward step with the knee bending the correct way** (no hyperextension). Figure `stride.png`.
- Stands (upright 1.000) and **all 48 tests pass** — the limits don't activate during normal balance.

## Files touched

- `data/robotics/humanoid.hymeko` — 8 anatomical limit nodes + `limit ->` on all 16 revolute joints.

**CORE.YAML:** none (`meta_kinematics` untouched — the override is model-level). **Deps:** none.

## Open items

- Restore the anthropomorphic + parametric redesign (discarded earlier by a `git checkout` that reverted
  the uncommitted model — my error; content is recoverable from the reports/history).
- **Real forward walking** (sagittal): advance the footstep plan in +x (heel-strike → stance → toe-off →
  forward swing) so the robot actually translates forward, replacing the lateral-marching "curiosity". The
  WBC + DCM + footstep-RL stack is reusable; the change is the footstep reference (forward, not in-place).
