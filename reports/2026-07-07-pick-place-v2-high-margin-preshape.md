# v2 higher-margin preshape route (cf_mid_retract → cf_hover) (2026-07-07)

## Summary

Routed `HOME_RETRACT_OR_PRESHAPE` through a **high, retracted `cf_mid_retract`** waypoint before `cf_hover`, so the
commanded tool stays high during the retract instead of a direct home→cf_hover joint move that dips mid-path.
**Isolated change — no gains, scene, arm_home, or reward touched.** It **improved** the physical result but did
**not** make physical clearance positive, and — critically — the joint-space route **still grazes 0 commanded
clearance**, so the several-cm commanded margin was *not* achieved. The dominant physical penetration is the
sag/lag during the catch-up to `cf_hover`, where the command is +6 cm.

## Changed files / v1

- Changed: `hymeko_rl/env/pick_place_env.py` only (two-stage preshape route + fields; `_V2_MID_RETRACT_*` consts).
- **`ik.py` not touched. v1 unchanged** — v1 re-smoke byte-identical (lift 1.0 / place 0.75 / min_clr −0.02577);
  17 pick/ik/probe tests green.
- **Selected route:** `cf_mid_retract` (multi-start, near-base `0.5·r`, `z_hover+0.08`) → `cf_hover` → hold-hover →
  descend.

## Results (v2 smoke, 4 ep, seeds 50000–50003, horizon 620) — reproduced bit-identical on kato15 (EGL)

| metric | v1 (ref) | v2 preshape (single) | **v2 high-margin (cf_mid→cf_hover)** |
|---|---|---|---|
| first strike step | ~51 | ~40 | **~100** (preshape no longer strikes early) |
| first-over-object | ~150–225 | 106–175 | 163–203 |
| transit finger↔table rate | 0.45 | 0.52 | **0.39** |
| **commanded** min clearance | — | ~+0.015 | **+0.000 (grazes)** |
| **physical** min clearance | −0.026 | −0.017 | **−0.019 min / −0.015 mean** |
| lift / place | 1.0 / 0.75 | 0.0 / 0.0 | **0.0 / 0.0** |
| gate | FAIL (dirty) | FAIL | **FAIL** |

## Decision-rule answer (traced commanded vs physical, seeds 50000/50001)

- **Commanded** min clearance = **+0.000** at `preshape[0/1]` — the joint-space interpolation between the
  multi-start configs **still swings the tool to touch the table** mid-route. Joint interpolation does not control
  the Cartesian tool height, so it cannot *guarantee* margin; the several-cm target was **not** met.
- **Physical** min clearance = **−0.019 at step ~109 during the catch-up to `cf_hover`**, where the **command is
  +6 cm** — i.e. the physical wrist sags/lags ~8 cm while crawling up to the held `cf_hover` command. This is the
  dominant penetration and it is a **tracking/sag** effect, not the commanded path.

**Verdict (per your rule):** physical clearance is still negative. It is **not** "commanded large everywhere" (the
route grazes 0), so path margin is *partly* — not fully — addressed. But the worst physical penetration sits under
a +6 cm command → the **gain/tracking fix is the justified next isolated step**. The residual commanded 0-graze is
a *secondary* path issue that joint-waypoints cannot fix — it needs **Cartesian-controlled** preshape (hold z high
while retracting, i.e. the waypoint-planner direction), not more joint interpolation.

## Is a gain/tracking fix still justified? — YES (next isolated step)

The higher-margin route hit the ceiling of what a joint-space preshape can do: it removed the early preshape strike
and cut transit contact, but the ~2–8 cm physical sag/lag under a clean command remains the dominant blocker. Per
your decision rule and the "one fix at a time" constraint, the **isolated gain/tracking fix** (stronger arm/wrist
position gain or wrist gravity-comp) is the justified next step. Optionally, a Cartesian-controlled preshape would
close the secondary commanded 0-graze — but that is the planner direction, larger, and separate.

## 32-episode gate

**Not justified** — physical clearance still negative, lift/place 0. Not run.

## Gates / provenance

ruff clean · 17 tests green · v1 byte-identical. Local and **kato15** (EGL, `~/envs/hymeko/bin/python`, mujoco 3.10)
smoke agree bit-for-bit. Stopped after the v2 smoke; the 32-ep gate was not run.
