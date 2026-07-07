# v2 reach-envelope verification — the hover is REACHABLE (static); the blocker is the IK PATH

**Complement to** `reports/2026-07-06-pick-place-clean-expert-v2-attempt.md` (which owns the v2 status; edited
concurrently by the parallel track — this file is a non-conflicting add). Local MuJoCo 3.9 + built CLI; env
reverted to the parallel track's WIP v2 afterward (v1 restored, place 0.75, ruff clean). No BC/DAgger, no scene
change.

## What I checked and found

The v2 freeze attributed the block to a "geometrically unreachable hover." I verified the reach envelope at the
object's actual reset position (3 seeds, radii 0.29 / 0.36 / 0.38), with self/floor-collision checking:

| top-down target over the object | reachable (pos+down) | self/floor valid | lowest finger z (table 0.120) |
|---|---|---|---|
| grasp_z (0.255) | yes | ✓ | 0.135 (+1.5 cm) |
| **hover grasp_z+0.06 (0.315)** | yes | **✓** | **0.195 (+7.5 cm)** |
| hover grasp_z+0.10 (0.355) | yes | ✗ self-collide | 0.235 |
| hover grasp_z+0.14 (0.395) | yes | ✗ self-collide | 0.275 |

So **valid top-down clearance poses DO exist** (up to ≈ grasp_z+0.06); the old v2's grasp_z+0.14 hover
self-collides (that is the "unreachable" it hit). The reach envelope is **not a hard static wall**.

## But the shortcut fails — the blocker is the incremental IK PATH

Rewiring v2 to transit at the reachable +0.06 hover and rerunning the clearance gate (N=24) gave: **lift/place
0.0**, transit finger↔table **0.49**, **min clearance −0.11 m** (24/24) — *worse* than v1 (−0.026). Cause: the
**rate-limited DLS-IK path from `arm_home` to the hover dips ~11 cm below the table**; targeting a higher hover
just folds the arm harder en route. **Static reachability ≠ achievable rate-limited trajectory.** Reverted.

## Refined conclusion (input to the v2 waypoint redesign)

- The parallel track's freeze stands empirically (a hover-height tweak does not fix v2).
- But its *cause* is refined: **the IK PATH dips, not the static reach** — valid clearance poses exist.
- Therefore the envelope-following **waypoint plan** the v2 report already recommends is the right lever, with two
  measured constraints: (a) the valid clearance-hover ceiling is **~grasp_z+0.06** (above → self-collision);
  (b) the fix must control the **PATH** (rise → traverse → descend, staying in the valid envelope each step),
  **not** the direct DLS target or the hover height.
- **A scene/object change is a fallback, not yet forced** — because valid clearance poses exist, waypoint
  planning should be tried first.
