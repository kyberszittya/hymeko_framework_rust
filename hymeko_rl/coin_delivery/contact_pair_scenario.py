"""RUBBER_TIP_LOW_DRAG_COIN scenario — decouple the two coin contact relationships so a physically-motivated material
model can be tested:

    fingertip ↔ coin   HIGH tangential friction  (a rubberised finger — grips, transfers useful shear)
    coin ↔ floor       LOW  tangential friction   (a hard coin sliding on a smooth table — easy to start moving)

Why NOT per-geom friction alone: MuJoCo combines two EQUAL-priority geoms' friction by the elementwise MAXIMUM, so
lowering the coin's geom friction cannot lower coin↔floor while the floor stays high — the earlier sweep was confounded by
exactly this. Why NOT explicit ``<pair>``: an explicit pair overrides the whole contact (solref/solimp), which changed the
CONTACT STIFFNESS and produced a 150 N normal-force explosion — it does not isolate friction.

The two relationships live in DIFFERENT model fields in this scene:
  * tip↔coin = a CONTACT (fingertip geom vs disk geom) → the friction is geom_friction, but the elementwise-max
    combination hides a per-geom change; the fix is geom PRIORITY: "when two geoms have different priority, the friction
    of the HIGHER-priority geom is used." Fingertip priority 2 ⇒ its friction wins the tip↔coin contact. Runtime scalar.
  * coin↔floor DRAG = NOT a contact here — the disk's planar slide DOFs carry VISCOUS DAMPING (``dof_damping`` ≈ 2.5;
    ``dof_frictionloss`` is 0), which is what resists sliding (a 0.5 m/s coin injected into the free scene decays in one
    step). So the "low-drag table" knob is the coin's slide ``dof_damping``, NOT the floor's contact friction. Runtime.

Both are runtime-mutable scalar fields — no recompile, no ``<pair>`` (which changed contact STIFFNESS → a 150 N normal-
force explosion), no contact-stiffness change. The frozen model files / SINGLE_TIP_LOW_FRICTION_COIN_V1 are untouched.
"""
from __future__ import annotations

import mujoco


def _gid(rl, name: str) -> int:
    g = mujoco.mj_name2id(rl.inner.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if g < 0:
        raise ValueError(f"geom {name!r} not found")
    return g


def setup_material_decoupling(rl):
    """Prepare the runtime handles for the RUBBER_TIP_LOW_DRAG material model. Sets fingertip priority so its contact
    friction wins the tip↔coin contact. Returns ``(tip_gids, disk_dof_adr, base_tip_mu, base_coin_damping)``. Idempotent."""
    m = rl.inner.model
    tip = [_gid(rl, "fingertip_left"), _gid(rl, "fingertip_right")]
    m.geom_priority[tip] = 2                                 # fingertip friction wins the tip↔coin contact (pri 2 > 0)
    m.geom_priority[_gid(rl, "disk")] = 0
    adr = int(rl.inner._disk_x_adr)                          # the disk's planar slide DOFs (x, y) carry the slide damping
    return tip, adr, round(float(m.geom_friction[tip[0], 0]), 4), round(float(m.dof_damping[adr]), 4)


def set_material(rl, tip_gids, disk_adr, tip_mu: float, coin_slide_damping: float,
                 coin_slide_frictionloss: float = 0.0) -> None:
    """Set the tip↔coin CONTACT friction (fingertip geom, priority-won) and the coin↔floor slide DRAG. The drag has two
    parts: a small VISCOUS ``dof_damping`` (numerical residual only) and a COULOMB ``dof_frictionloss`` (the physically-
    correct, speed-independent table friction — calibrated via the coast test). Both on the disk's two planar-slide DOFs.
    Runtime; no recompile."""
    m = rl.inner.model
    for g in tip_gids:
        m.geom_friction[g, 0:2] = tip_mu
    m.dof_damping[disk_adr:disk_adr + 2] = coin_slide_damping
    m.dof_frictionloss[disk_adr:disk_adr + 2] = coin_slide_frictionloss
    mujoco.mj_forward(m, rl.inner.data)


def effective_tip_coin_friction(rl, tip_gids, disk_geom):
    """Read the EFFECTIVE tangential friction MuJoCo assigned to an active fingertip↔disk contact (``d.contact[i].friction``
    [0], AFTER the priority combination). Returns the effective μ, or None if no such contact is active. This VERIFIES that
    geom_priority actually produces the intended tip↔coin friction — the user-requested separate check that the material
    model is what we think it is, independent of any delivery outcome."""
    d = rl.inner.data
    tip = set(int(g) for g in tip_gids)
    disk = int(disk_geom)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in tip and g2 == disk) or (g2 in tip and g1 == disk):
            return round(float(c.friction[0]), 4)
    return None
