"""COIN-DELIVERY-1 tests — delivery-directed primitive geometry (midpoint translates toward the zone), the
handoff-avoiding pulse, and the Stage-0 failure classifier. Mock inner env (no physics)."""
from __future__ import annotations

import types

import numpy as np

from hymeko_rl.experiments import coin_delivery1 as cd


def _inner(coin_xy, zone_xy=(0.044, 0.151), zone_half=0.04):
    from hymeko_rl.env.planar_grasp_env import coin_zone_direction
    m = types.SimpleNamespace(disk_pos=np.array([coin_xy[0], coin_xy[1], 0.0]))
    inner = types.SimpleNamespace(_planar_metrics=m, _zone_x=zone_xy[0], _zone_y=zone_xy[1], _zone_half=zone_half)
    # the coin geometry now lives on the env as `direction_to_zone`; bind it on the lightweight mock via the shared
    # pure function so this unit test exercises the real math without constructing a MuJoCo env.
    inner.direction_to_zone = lambda: coin_zone_direction(m.disk_pos, inner._zone_x, inner._zone_y)
    return inner


def test_dir_to_zone_points_at_zone():
    inner = _inner((-0.071, 0.189))
    d, n = cd._dir_to_zone(inner)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-6 and n > 0
    assert d[0] > 0 and d[1] < 0                                                   # coin left+above zone → move right+down


def test_primitives_valid_6d_and_carry_direction():
    inner = _inner((-0.071, 0.189)); d, _n = cd._dir_to_zone(inner)
    for name, prim in cd.PRIMS.items():
        a = prim(inner, 0)
        assert a.shape == (6,) and np.all(np.abs(a) <= 1.0 + 1e-6), name
        assert a[0] * d[0] >= 0 and a[1] * d[1] >= 0, name                         # midpoint translation toward the zone


def test_grasp_carry_squeezes_carry_pulse_releases():
    inner = _inner((-0.05, 0.18))
    assert cd.p_grasp_carry(inner, 0)[3] > 0                                       # grasp_carry always squeezes
    assert cd.p_carry_pulse(inner, 0)[3] > 0 and cd.p_carry_pulse(inner, 3)[3] < 0  # pulse releases on the 4th step


def test_stage0_classifier_branches():
    assert cd._class1(True, True, 0.0, 0.15) == "1_already_deliverable"
    assert cd._class1(False, True, 0.005, 0.15) == "2_grasped_no_transport"
    assert cd._class1(False, True, 0.05, 0.15) == "3_transport_starts_misses_zone"
    assert cd._class1(False, False, 0.0, 0.15) == "4_no_stable_acquisition"


def test_summ_aggregation():
    rs = [{"deliv": True, "handoff": True, "progress": 0.05, "final_dtz": 0.03},
          {"deliv": False, "handoff": True, "progress": 0.005, "final_dtz": 0.10}]
    s = cd._summ(rs, 2)
    assert s["delivery_success"] == 0.5 and s["grasp_no_delivery_rate"] == 0.5 and s["moved_coin_rate"] == 0.5
