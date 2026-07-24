"""BALLTIP_INTERARM_FILTERED_V1 — robot-variant support: build the variant MuJoCo model from a HyMeKo robot spec and
report the collision policy (the §2 query + manifest).

The variant is selected entirely by choosing a robot `.hymeko` (spherical-tip radius + per-side collision-group masks live
in the spec, not in a script). The `inter_arm="enabled"` control (spherical tips WITH the original collision) is a documented
mask-only transform of the same spec — used only as the §5 regression control, never as a stale penalty.
"""
from __future__ import annotations

import re

import mujoco  # type: ignore[import-untyped]

from hymeko_rl.env.planar_grasp_env import compose_planar_scene, emit_galambos_v2_mjcf

CANONICAL_SPEC = "data/robotics/galambos_planar_v3.hymeko"
BALLTIP_SPEC = "data/robotics/galambos_planar_balltip_v1.hymeko"

# The three §5 regression variants (robot spec + inter-arm collision policy):
VARIANTS = {
    "canonical": (CANONICAL_SPEC, "original"),       # original geometry (tip 0.014) + original collision (1/3)
    "balltip_nofilter": (BALLTIP_SPEC, "enabled"),   # spherical tips (0.020) + original collision (control)
    "balltip_filtered": (BALLTIP_SPEC, "filtered"),  # spherical tips (0.020) + inter-arm filtering (the variant)
}


def read_collision_policy(spec_path: str = BALLTIP_SPEC) -> dict:
    """Parse the `galambos_collision {}` sidecar (numeric scalars) from a robot `.hymeko`. Returns the group masks and
    derives ``inter_arm`` = 'filtered' when left_contype != right_contype, else 'enabled'."""
    body = open(spec_path, encoding="utf-8").read()
    m = re.search(r"galambos_collision\s*\{([^}]*)\}", body)
    if not m:
        return {"inter_arm": "unspecified"}
    f = {k: int(v) for k, v in re.findall(r"(\w+)\s+(-?\d+)\s*;", m.group(1))}
    f["inter_arm"] = "filtered" if f.get("left_contype") != f.get("right_contype") else "enabled"
    return f


def build_arm_mjcf(robot_spec: str, inter_arm: str = "filtered") -> str:
    """Emit the arm MJCF from a robot spec; ``inter_arm='enabled'`` rewrites the per-side contype bits back to the
    canonical shared bit (1) — the control that keeps the spherical tips but restores the original collision policy."""
    arm = emit_galambos_v2_mjcf(robot_spec)
    if inter_arm == "enabled":
        arm = re.sub(r'contype="[48]"', 'contype="1"', arm)              # left(4)/right(8) -> shared canonical bit
    return arm


def build_variant_model(variant: str):
    """Build (and compile) the full scene MjModel for a named §5 variant. Returns (MjModel, arm_mjcf)."""
    spec, inter_arm = VARIANTS[variant]
    arm = build_arm_mjcf(spec, inter_arm)
    return mujoco.MjModel.from_xml_string(compose_planar_scene(arm)), arm


def _can_collide(m, a: int, b: int) -> bool:
    return bool((int(m.geom_contype[a]) & int(m.geom_conaffinity[b])) or (int(m.geom_contype[b]) & int(m.geom_conaffinity[a])))


def collision_manifest(m) -> dict:
    """The §2 collision manifest/query over a compiled MjModel: the exact per-side group masks, the derived
    ``inter_arm_collision`` field, and the concrete filtered / enabled body pairs (arm↔arm, arm↔coin, arm↔floor)."""
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
    body = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") for g in range(m.ngeom)}
    L = [g for g in range(m.ngeom) if body[g].endswith("_left")]
    R = [g for g in range(m.ngeom) if body[g].endswith("_right")]
    coin = [g for g in range(m.ngeom) if "disk" in names[g] or "coin" in names[g]]
    floor = [g for g in range(m.ngeom) if names[g] == "floor" or body[g] == "world"]
    lr = any(_can_collide(m, a, b) for a in L for b in R)
    filtered_pairs = sorted({f"{body[a]}|{body[b]}" for a in L for b in R})
    masks = {"left_contype": sorted({int(m.geom_contype[g]) for g in L}), "right_contype": sorted({int(m.geom_contype[g]) for g in R}),
             "coin_contype": sorted({int(m.geom_contype[g]) for g in coin}), "conaffinity": sorted({int(m.geom_conaffinity[g]) for g in L + R})}
    return {"inter_arm_collision": "filtered" if not lr else "enabled",
            "left_right_can_collide": lr,
            "arm_coin_enabled": all(any(_can_collide(m, a, b) for b in coin) for a in [L[0], R[0]] if coin),
            "arm_floor_enabled": all(any(_can_collide(m, a, b) for b in floor) for a in [L[0], R[0]] if floor),
            "group_masks": masks, "filtered_body_pairs": filtered_pairs,
            "fingertip_radii": sorted({round(float(m.geom_size[g][0]), 4) for g in range(m.ngeom) if int(m.geom_type[g]) == 2})}


def min_interarm_clearance(m, d) -> float:
    """Diagnostic (NOT part of the reward/certificate): the minimum surface-to-surface distance between any left-arm geom
    and any right-arm geom in the current configuration (negative = interpenetration). Used by the §6 exploit audit."""
    names_body = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") for g in range(m.ngeom)}
    L = [g for g in range(m.ngeom) if names_body[g].endswith("_left")]
    R = [g for g in range(m.ngeom) if names_body[g].endswith("_right")]
    best = float("inf")
    for a in L:
        for b in R:
            dist = mujoco.mj_geomDistance(m, d, a, b, 10.0, None)         # signed surface distance; <0 = penetration
            best = min(best, float(dist))
    return best
