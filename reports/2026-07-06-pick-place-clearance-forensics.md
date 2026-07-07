# FANUC pick-place — bounded clearance forensics on the CURRENT (v1) expert (2026-07-06)

**Diagnosis only** — no env change, no BC/DAgger rerun. Current expert = `expert_version=1` (the default,
contact-exploiting baseline). N=32 episodes, seeds 50000–50031, local MuJoCo 3.9 + built CLI (kato15 offline).
Approach/transit phase := tool >3 cm horizontally from the object (i.e. not yet above it); box half = 0.02 m,
table top z = surf+height.

## ❄️ FROZEN CONCLUSION (2026-07-06) — classification C; do NOT overwrite v1

**The current FANUC pick-place expert is classified: C — physically dirty, requiring trajectory revision.**

**Evidence (frozen, exact):**
- 32/32 expert episodes strike the table edge during transit.
- The first finger↔table strike occurs around step 51.
- The gripper is not above the object until around step 275.
- Therefore the collision occurs far before the intended descent/grasp.
- 32/32 episodes have negative minimum clearance.
- The lowest finger geometry drops about 2.6 cm below the tabletop plane.
- There is no clean episode.

**Interpretation:** the v1 expert succeeds despite an invalid approach trajectory. It is **not** merely using
table contact as a grasp stop; the transit trajectory itself collides with the table edge/surface far from the
object.

**Versioning — the prior pick-place learning results remain valid ONLY as `v1_dirty_expert` results:**
`expert v1`, `BC v1`, `TD3+BC v1`, `DAgger v1`. They must **not** be described as clean pick-place learning
results. Do **not** overwrite the v1 numbers — **re-version** them.

**Next version — a separate `expert_version=2` branch (later, not now):** raised transit/approach height;
descend only above the object; clearance gate during approach; new expert ceiling; regenerated demos; new BC
baseline; new DAgger run; new report section.

**v2 acceptance criteria (all required, in order):**
1. lift/place stay high;
2. finger↔table contact during transit = 0 or near 0;
3. no table-edge strike before over-object;
4. min clearance positive during approach/transit;
5. only then: demo regen → BC → DAgger.

---

## 1. Task success

| metric | value |
|---|---|
| lift rate | **1.000** |
| place rate | **0.844** |
| episode length (median) | 492 / 620 |
| object→target dist (median) | 0.0744 m |

The expert **succeeds** — high lift, ~0.84 place.

## 2. Approach-phase collision evidence (before the intended descent)

| metric | value |
|---|---|
| finger↔table contact fraction (approach) | mean **0.321**, median 0.328 |
| palm↔table fraction | 0.000 |
| arm/link↔table fraction | 0.000 |
| **episodes with finger↔table BEFORE gripper above object** | **32 / 32** |
| first finger↔table step (median) | **51** |
| first over-object step (median) | 275 |
| **table-EDGE strikes** (contact ≤2 cm of table XY boundary) | **13 947 total; 32/32 episodes** |

The fingers strike the table ~220 steps *before* the gripper is ever above the object, and every episode strikes
the table **edge** during the transit. The arm/palm never touch — but the offender is the gripper's fingers, so
this does not make the approach clean.

## 3. Clearance evidence (lowest finger geometry vs table top, during approach)

| metric | value |
|---|---|
| min vertical clearance (median) | **−0.0262 m** |
| min vertical clearance (worst) | −0.0283 m |
| **episodes with NEGATIVE min clearance** (finger below the table-top plane) | **32 / 32** |

At minimum clearance (e.g. seed 50000, t=84): lowest finger xy ≈ **(0.546, −0.176)** — at the table's *far edge*
— while the object is at **(0.273, −0.140)** and the EE is at z≈0.216. So the transit swings the gripper out to
the table edge (x≈0.55, far from the object at x≈0.27) and drops the fingers ~2.6 cm below the table-top plane,
striking the edge, then returns over the object ~190 steps later to grasp.

## 4. Visual evidence (reports/gifs/)

- `expert_v1_success_early_table_contact_s50000.gif` — a *successful* episode (place=1) that strikes the table
  early (finger↔table from step 50, over-object at 252).
- `expert_v1_cleanest_still_dirty_s50005.gif` — the **cleanest** of 32 (lowest finger↔table frac 0.265) — still
  negative clearance (−0.026) and edge strikes. **There is no clean episode.**
- `expert_v1_failure_s50001.gif` — a failure (place=0, obj→target 0.144).

## 5. Acceptance check (clean requires ALL, per directive)

| requirement | met? |
|---|---|
| no gripper/finger↔table collision during transit | **NO** (finger↔table 32% of approach steps) |
| no table-edge strike before intended descent | **NO** (32/32 episodes, 13 947 strikes) |
| contact only during intended grasp/object interaction | **NO** (first strike step 51 ≪ over-object step 275) |
| high enough place/lift | yes (lift 1.0, place 0.844) |

## 6. Conclusion — classification

**C. Physically dirty, requiring trajectory revision.** The expert succeeds (place 0.844) **despite** an invalid
approach trajectory: every episode swings the gripper out to the table's far edge and drops the fingers below
the table-top plane, striking the edge ~220 steps before the intended grasp. This is not merely "exploiting the
table as a grasp stop" (which would be B) — the transit path itself collides with the table edge/surface far
from the object. **It is usable as a v1 baseline, but the trajectory must be revised (v2, already parameterised
via `expert_version`) for a clean pick-place; and the prior BC (~0.55) / DAgger (~0.79–0.83) numbers were
produced on this dirty trajectory and must be re-versioned, not carried over.**

Diagnosis complete. No env change made; v1 unchanged; no BC/DAgger rerun.
