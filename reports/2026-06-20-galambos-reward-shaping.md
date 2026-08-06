# Galambos planar grasp — why it wasn't moving (a chain of discriminating tests)

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-20-galambos-reward-shaping/](../docs/plans/2026-06-20-galambos-reward-shaping/)*

## Summary

"The RL scenarios are not moving." Rather than train harder, I ran a chain of
discriminating tests. Each ruled out a hypothesis and pointed to the next. The
endpoint is **not** a reward or training problem: the emitted planar arm overlaps its
base hub and first link, and the resulting self-contact **mechanically pins the
shoulder joint** — the arms could only flex their elbows, never sweep to the coin. Two
of the three things I changed were necessary cleanups; the third (a contact exclusion)
is the actual fix. Reported as found, including the dead end.

## The trail (each step measured, not assumed)

**0. The suspected bug was already fixed.** Memory listed the PPO truncation-bootstrap
bug as the prime suspect. Reading [ppo.py:108–118](../hymeko_rl/ppo.py) shows `_collect`
already separates `terminated`/`truncated` and bootstraps `γ·V(next)` on truncation.
Not the cause. Memory corrected.

**1. Where does the chain break? (diagnostic rollout).**
`hymeko_rl/diagnose_planar_grasp.py`, trained policy, 8 episodes:

| Metric (median) | Baseline | Reading |
|---|---|---|
| Both-finger contact | **0 / 160 steps** (all ep) | the grasp never happens |
| Min arm→coin (body origin) | 0.064 m | arms get *near* but never press |
| Coin displacement | 0.004 m | coin essentially never moves |
| Reward decomposition | ≈100 % `pull`, `contact`=0, `zone`=0 | only the near-flat term fires |

→ The chain breaks at **contact**. Hypothesis: the reward has no dense gradient from
"near" to "contact" (`both_contact` is a binary cliff; `ent_coef=0`).

**2. Add the missing gradient — and measure (it was not enough).** Added a dense,
declarative `grasp_approach` term (−½ of both arms' nearest-link distance to the coin)
+ `ent_coef=0.005`, retrained 150 it. Re-diagnosed: **contact still 0/160; min arm→coin
0.063 m — unchanged.** A measurement contradicting the plan → stop, do not stack another
reward hack. The approach term is sound but treats a symptom.

**3. Can the arm even reach? (constant-target test).** Held a fixed inward shoulder
target `[1.2, −1.6, −1.2, 1.6]`:

```
final qpos    : [ 0.01  -1.6   -0.01   1.6 ]
tracking error: [ 1.19  -0.0   -1.19  -0.0 ]
```

→ The **elbows (j2) track perfectly; the shoulders (j1) do not move at all** (1.19 rad
error, frozen at 0). Not weak actuation (gains are uniform); something *pins* the
shoulder.

**4. What pins it? (contact listing).** While holding the target, the only active
contacts are `base_left ↔ upper_left` and `base_right ↔ upper_right`, with **−0.022 m
penetration** — and both bodies share the *same* origin `[−0.14, −0.02, 0.04]`. The
emitter places the first link on top of the base hub; the overlap's contact force
resists shoulder rotation.

**5. Confirm by removing it.** Injecting `<contact><exclude base↔upper>` and re-holding
the target: shoulder tracking error **1.19 → 0.16 rad — the shoulders move.** Root cause
confirmed.

## Fix

- **Actual fix:** [planar_grasp_env.py](../hymeko_rl/env/planar_grasp_env.py) gains
  `adjacent_link_excludes()`, which compiles the arm alone, reads its exact parent→child
  body topology, and emits `<contact><exclude>` for every adjacent-link pair (robust to
  emitted `upper/lower` vs hand-authored `link1/link2` naming). Injected into the scene
  in `__init__`. This is the explicit form of the parent-child filtering MuJoCo's
  `filterparent` failed to apply here. Regression test
  `test_shoulder_joint_is_not_frozen_by_self_contact` holds the target and asserts the
  shoulders rotate.
- **Necessary cleanups (kept, but not the cause):** the dense `grasp_approach` reward
  term + `ent_coef` — correct shaping that will matter now that the arm can actually
  reach, but proven (step 2) insufficient on a frozen arm. Documented honestly, not as
  the fix.

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_rl/env/planar_grasp_env.py` | +~30 | `adjacent_link_excludes` (the fix) + `{left,right}_tip_dist` metrics |
| `hymeko_rl/env/reward.py` | +~12 | `_term_grasp_approach` + registry |
| `data/robotics/meta_reward.hymeko` | +4 | `@grasp_approach` term kind |
| `data/robotics/galambos_task.hymeko` | +3/−2 | `@approach` in `reward_spec` |
| `hymeko_rl/train_planar_grasp.py` | +3 | `--ent-coef` (default 0.005) |
| `hymeko_rl/tests/test_planar_grasp_env.py` | +~40 | shoulder-mobility regression + approach-term + metric tests |
| `hymeko_rl/diagnose_planar_grasp.py` | new | the discriminating diagnostic |

## CORE.YAML / dependencies

**None.** All under `hymeko_rl/` + `data/robotics/` (non-core). No new dependency. The
proper geometry fix (stop the emitter overlapping base hub + first link) is a follow-up
in `hymeko_formats` — same "link at body origin" defect class noted earlier in the RL
line; the env-level contact exclusion unblocks training now.

## Test results

- `pytest hymeko_rl/tests/test_planar_grasp_env.py` — **9 passed** (incl. the
  shoulder-mobility regression, the approach-term directed regression, the tip-distance
  metric test). Full `hymeko_rl` suite — **110 passed**, no regressions.
- `hymeko validate data/robotics/galambos_task.hymeko` — ✅.
- `ruff check` + `mypy --strict` on changed code — clean (pre-existing third-party
  `mujoco` import-untyped notes only).

## Validation (post-fix retrain — shoulder freed + approach reward)

_150 iters, 512 steps, ent_coef 0.005, seed 0 → `checkpoints/galambos/ppo_freed.pt`.
Training return **−45.6 → +21.2** (frozen-shoulder retrain with the same reward: −49 → −47)._

| Metric (median over 8 ep) | Baseline (frozen) | After fix | 
|---|---|---|
| Shoulder tracking error (held target) | 1.19 rad | **0.16 rad** |
| Coin displacement | 0.004 m | **0.055 m** (≈14×) |
| Disk→zone distance (init → final) | ~0.10 → ~0.10 | ~0.10 → **~0.05** (inside zone, half=0.055) |
| Zone steps | 0 / 160 | **5 / 160** (= success threshold) |
| **Goals reached** | **0 / 8 ep** | **5 / 8 ep** |
| Both-finger contact steps | 0 | 0 (see note) |

**The decisive variable is the shoulder unfreeze, not the reward.** The same approach
reward on the still-frozen arm (step 2) scored 0 goals; once the shoulder is freed it
scores 5/8. **Honest nuance:** `both_contact` stays 0 — the policy solves the task by
*pushing* the coin (asymmetric / single-arm nudging), not a two-sided pinch grasp. For
the stated Galambos task ("pull the coin into the zone") that is a valid solution; a
true two-sided grasp would need contact-specific shaping (a follow-up). The 3 non-goal
episodes push the coin hard (disp 0.04–0.08 m, 79–98 moved steps) but don't settle it.

## Open / follow-up

- **Proper geometry fix in the emitter** (`hymeko_formats`): give the first link a real
  offset from the base hub so the geoms don't overlap, removing the need for the
  exclusion workaround and fixing the visual.
- If contact emerges but goals stay rare after the shoulder is freed: a curriculum
  (coin spawned nearer the zone, annealed) is the next single lever.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing unrelated changes). CPU MuJoCo, no
GPU. Seeds fixed (env seed 0; diagnostic seeds 1000–1007). All measurements above are
from committed fixtures + the saved checkpoints.
