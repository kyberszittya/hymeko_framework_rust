"""RUBBER_TIP_LOW_DRAG material-decoupling — the priority + friction + slide-damping writes take effect and leave the
model geometry (hence every cached id) unchanged. Integration test (needs the coin env)."""
import numpy as np

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup


def test_material_decoupling_sets_priority_friction_and_drag_without_reshaping():
    pi0, base, forbidden = _setup()
    rl, _gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=14300, tries=2)
    m = rl.inner.model
    ngeom, nq, nv = m.ngeom, m.nq, m.nv
    disk_geom = rl.inner._disk_geom
    tg, adr, base_tip, base_drag = setup_material_decoupling(rl)
    # fingertips get higher priority than the disk ⇒ their friction wins the tip↔coin contact
    assert all(m.geom_priority[g] > m.geom_priority[disk_geom] for g in tg)
    assert base_drag > 0 and base_tip > 0
    set_material(rl, tg, adr, tip_mu=2.5, coin_slide_damping=base_drag * 0.4)
    assert np.allclose(m.geom_friction[tg[0], 0:2], 2.5)                     # tip contact friction set
    assert np.allclose(m.dof_damping[adr:adr + 2], base_drag * 0.4)         # coin slide drag lowered
    # geometry unchanged ⇒ cached geom/qvel/body ids remain valid
    assert m.ngeom == ngeom and m.nq == nq and m.nv == nv and m.npair == 0
