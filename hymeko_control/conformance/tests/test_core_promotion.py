"""Tests for the promoted generic certificate factories (core-v1).

These are scenario-independent: the extractor is supplied by the caller, so the
core names no scenario signal. Reward is never an input (reward-independence
preserved).
"""

from __future__ import annotations

from hymeko_control.cip.certificate import (
    stability_certificate,
    threshold_certificate,
)
from hymeko_control.cip.option import OptionEnd, ResponseTrace
from hymeko_control.cip.structured_state import ControlState
from hymeko_control.language.schema_v0 import CertificateKind

_STATE = ControlState(t=0, phase="X", signals={"d": 0.1})
_TRACE = ResponseTrace("o", ((0.0,),), ({"speed": 0.03, "upright": 0.8},),
                       OptionEnd.COMPLETED, {"option": "o", "speed": 0.03, "upright": 0.8})


def _speed(_s, tr):
    return float(tr.provenance["speed"])


def _upright(_s, tr):
    return float(tr.provenance["upright"])


def test_threshold_upper_bound() -> None:
    cert = threshold_certificate("speed_ok", CertificateKind.SAFETY, _speed, upper=0.06)
    assert cert.evaluate(_STATE, _TRACE)  # 0.03 <= 0.06
    fast = ResponseTrace("o", ((0.0,),), ({},), OptionEnd.COMPLETED,
                         {"option": "o", "speed": 0.5})
    assert not cert.evaluate(_STATE, fast)  # 0.5 > 0.06


def test_threshold_lower_bound_and_inverted_raises() -> None:
    cert = threshold_certificate("dist_reached", CertificateKind.SUCCESS,
                                 lambda s, t: s.signal("d"), upper=0.2)
    assert cert.evaluate(_STATE, _TRACE)
    try:
        threshold_certificate("bad", CertificateKind.SAFETY, _speed, lower=1.0, upper=0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("inverted bounds must raise")


def test_stability_certificate() -> None:
    cert = stability_certificate("no_fall", _upright, min_uprightness=0.5)
    assert cert.kind == CertificateKind.SAFETY
    assert cert.evaluate(_STATE, _TRACE)  # 0.8 >= 0.5
    fallen = ResponseTrace("o", ((0.0,),), ({},), OptionEnd.ABORTED,
                           {"option": "o", "upright": -0.3})
    assert not cert.evaluate(_STATE, fallen)  # -0.3 < 0.5


def test_reward_independence_of_promotion() -> None:
    import ast
    import inspect
    from pathlib import Path
    from hymeko_control.cip import certificate

    src = Path(inspect.getfile(certificate)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    args = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    assert "reward" not in names and "reward" not in args
