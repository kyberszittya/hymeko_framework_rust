"""Tests for R11.4B dataset descriptor + sample serialization (fakes — no MuJoCo)."""
import types

import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_bc import dataset as D
from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    FEATURE_NAMES,
    N_FEATURES,
    BcSample,
    _f,
    descriptor,
    target_of,
    zone_of,
)


def _snap(zone: tuple[float, float] = (0.03, 0.15)) -> object:
    inner = types.SimpleNamespace(
        data=types.SimpleNamespace(qpos=np.arange(4.0), qvel=np.arange(4.0) + 10.0),
        _planar_metrics=types.SimpleNamespace(disk_vel=np.array([0.1, 0.2, 0.0])),
        _zone_x=zone[0], _zone_y=zone[1])
    branch = types.SimpleNamespace(inner=inner)
    return types.SimpleNamespace(branch=lambda: branch, prev_tau=np.array([0.5, 0.6, 0.7, 0.8]))


def _rc() -> object:
    oc = types.SimpleNamespace(bilateral_dwell=4, left_right_contact_delay=7, first_contact_relvel=0.03,
                               second_contact_relvel=None, coin_disp_capture_mm=15.0, terminal_coin_speed=0.17)
    params = types.SimpleNamespace(n=3.0, s=0.42, preload_start=0.05, bmax=0.98)
    return types.SimpleNamespace(result=types.SimpleNamespace(outcome=oc, params=params))


def test_feature_names_length() -> None:
    assert len(FEATURE_NAMES) == N_FEATURES == 30


def test_f_maps_none_and_nan_to_zero() -> None:
    assert _f(None) == 0.0 and _f(float("nan")) == 0.0 and _f(3.5) == 3.5


def test_bcsample_json_roundtrip() -> None:
    s = BcSample("bank_c0_0", "dev", 3, "recovered", [0.1] * 30, [0.04, 0.3, 0.0, 12.0, 13.0, 1.9], True, 9.9)
    assert BcSample.from_json(s.to_json()) == s


def test_target_of_prefers_relocated_else_zone() -> None:
    snap = _snap(zone=(0.03, 0.15))
    reloc = types.SimpleNamespace(target_xy=np.array([0.09, 0.16]))
    canon = types.SimpleNamespace(target_xy=None)
    assert np.allclose(target_of(reloc, snap), [0.09, 0.16])
    assert np.allclose(target_of(canon, snap), zone_of(snap))


def test_descriptor_shape_and_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "_coin_xy", lambda rl: np.array([0.1, 0.12]))
    scen = types.SimpleNamespace(target_xy=np.array([0.03, 0.15]))
    x = descriptor(scen, _rc(), _snap())
    assert x.shape == (N_FEATURES,)
    assert np.allclose(x[8:10], [0.1, 0.12])                 # coin pose
    assert np.allclose(x[12:14], [0.03, 0.15])               # target pose
    assert np.allclose(x[14:16], [0.03 - 0.1, 0.15 - 0.12])  # coin->target vector
    assert x[27] == 0.0                                      # second_contact_relvel None -> 0.0
