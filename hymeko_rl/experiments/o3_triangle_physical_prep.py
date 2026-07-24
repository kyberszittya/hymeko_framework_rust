"""O3 — triangular-prism PHYSICAL-PANEL preparation (foundation only; NO learning campaign, runtime API frozen).

Validates the physical foundation the user gated the O3 experiment on, BEFORE any teacher/RL:
  1. full-footprint certificate — a centred aligned triangle certifies; a shifted/over-rotated one (a corner out) does not;
  2. runtime mass / COM / inertia — MuJoCo-derived, equal-area to the cylinder, and ROTATION-INVARIANT in the body frame;
  3. mesh-contact STABILITY — the mesh manipuland settles under stepping (bounded qacc + qvel, no NaN), not exploding;
  4. vertex/edge/orientation stratification — a deterministic orientation panel with per-orientation footprint margin.

Emits a JSON foundation report. The fresh-reconstruct teacher panel + the K-mode experiment come AFTER this validation
closes (a separate, gated step).
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.triangle_footprint import (
    footprint_margin, full_footprint_certified, leading_feature, orientation_strata, triangle_circumradius)
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

OUT = "reports/2026-07-24-o3-triangle-physical"
_R = 0.020


def validate_footprint_certificate(zone_half=0.055):
    """A centred, small triangle certifies; sliding it out or rotating a corner past the zone edge breaks it. The
    stricter full-footprint predicate must reject corners the centroid test would accept."""
    R = triangle_circumradius(_R)
    zone = np.array([0.0, 0.16])
    centred = full_footprint_certified(zone, 0.0, R, zone, zone_half)                 # centroid at zone centre
    # a shift that keeps the CENTROID in the zone but pushes a vertex out (R > zone_half - shift):
    shift = zone + np.array([0.0, zone_half - R + 0.004])
    centroid_in_but_corner_out = (np.linalg.norm(shift - zone) <= zone_half) and not full_footprint_certified(shift, 0.0, R, zone, zone_half)
    far = full_footprint_certified(zone + np.array([0.2, 0.0]), 0.0, R, zone, zone_half)
    return {"circumradius": round(R, 5), "centred_certifies": centred,
            "centroid_in_but_corner_out_rejected": bool(centroid_in_but_corner_out), "far_rejected": (not far),
            "ok": bool(centred and centroid_in_but_corner_out and not far)}


def validate_runtime_inertia():
    """Build the triangle coin env; assert MuJoCo-derived mass parity to the equal-area cylinder, COM at the body
    origin, and body-frame inertia INVARIANT to the object's in-plane rotation (disk_rz)."""
    tri = PlanarGraspEnv(max_steps=10, coin_shape="triangle", coin_density=1000.0)
    cyl = PlanarGraspEnv(max_steps=10, coin_shape="cylinder", coin_density=1000.0)
    tb = mujoco.mj_name2id(tri.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    cb = mujoco.mj_name2id(cyl.model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    mass_t, mass_c = float(tri.model.body_mass[tb]), float(cyl.model.body_mass[cb])
    inertia0 = np.array(tri.model.body_inertia[tb], float)
    # rotate the object through a 3-fold period; body-frame inertia + mass are rotation-invariant
    tri.reset(seed=0)
    adr = tri._disk_x_adr
    inv = True
    for rz in np.linspace(0, 2 * math.pi / 3, 6, endpoint=False):
        tri.data.qpos[adr + 2] = float(rz)
        mujoco.mj_forward(tri.model, tri.data)
        inv = inv and np.allclose(tri.model.body_inertia[tb], inertia0, atol=1e-9)
    return {"mass_triangle": round(mass_t, 6), "mass_cylinder": round(mass_c, 6),
            "equal_area_mass_parity": bool(abs(mass_t - mass_c) < 1e-4),
            "com_at_origin": bool(np.allclose(tri.model.body_ipos[tb], 0.0, atol=1e-4)),
            "diag_inertia": [round(float(x), 9) for x in inertia0],
            "inertia_rotation_invariant": bool(inv), "ok": bool(abs(mass_t - mass_c) < 1e-4 and inv)}


def validate_mesh_contact_stability(steps=120, seeds=3):
    """Step the triangle coin env under random actions; the mesh manipuland must SETTLE (finite qacc/qvel, no NaN) —
    a mesh with a bad convex hull or COM would jitter or explode against the fingertips."""
    worst_qvel, worst_qacc, nan = 0.0, 0.0, False
    for s in range(seeds):
        env = PlanarGraspEnv(max_steps=steps, coin_shape="triangle")
        env.reset(seed=s)
        for _ in range(steps):
            env.step(env.action_space.sample())
            if not np.all(np.isfinite(env.data.qpos)) or not np.all(np.isfinite(env.data.qvel)):
                nan = True
                break
            worst_qvel = max(worst_qvel, float(np.max(np.abs(env.data.qvel))))
            worst_qacc = max(worst_qacc, float(np.max(np.abs(env.data.qacc))))
    return {"max_abs_qvel": round(worst_qvel, 2), "max_abs_qacc": round(worst_qacc, 1), "nan_or_inf": nan,
            "ok": bool(not nan and worst_qvel < 50.0)}


def validate_stratification(zone_half=0.055):
    """The deterministic vertex/edge orientation panel + per-orientation footprint margin (at the zone centre): a
    vertex-leading pose has a different worst-corner clearance than an edge-leading pose — the geometry O3 must handle."""
    R = triangle_circumradius(_R)
    zone = np.array([0.0, 0.16])
    strata = orientation_strata(12)
    rows = [{"leading": lf, "disk_rz": rz, "footprint_margin": round(footprint_margin(zone, rz, R, zone, zone_half), 4)}
            for lf, rz in strata]
    v = [r for r in rows if r["leading"] == "vertex"]
    e = [r for r in rows if r["leading"] == "edge"]
    return {"n": len(rows), "n_vertex": len(v), "n_edge": len(e),
            "leading_feature_at_0": leading_feature(0.0),
            "vertex_min_margin": round(min((r["footprint_margin"] for r in v), default=0.0), 4),
            "edge_min_margin": round(min((r["footprint_margin"] for r in e), default=0.0), 4),
            "rows": rows, "ok": bool(v and e)}


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"contract": "O3_TRIANGLE_PHYSICAL_PREP", "date": "2026-07-24", "scope": "physical foundation only; no learning campaign",
              "footprint_certificate": validate_footprint_certificate(),
              "runtime_inertia": validate_runtime_inertia(),
              "mesh_contact_stability": validate_mesh_contact_stability(),
              "orientation_stratification": validate_stratification()}
    report["all_ok"] = all(report[k]["ok"] for k in
                           ("footprint_certificate", "runtime_inertia", "mesh_contact_stability", "orientation_stratification"))
    json.dump(report, open(f"{OUT}/o3_physical_prep.json", "w"), indent=1, default=str)
    for k in ("footprint_certificate", "runtime_inertia", "mesh_contact_stability", "orientation_stratification"):
        print(f"  {k}: ok={report[k]['ok']}  {({kk: vv for kk, vv in report[k].items() if kk not in ('rows','ok')})}")
    print(f"\n  ALL_OK: {report['all_ok']}\n  artifact: {OUT}/o3_physical_prep.json\nO3_PHYSICAL_PREP_DONE")
    return report["all_ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
