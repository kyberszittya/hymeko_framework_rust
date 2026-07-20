"""Coverage for the pure structural pieces added 2026-07-17 (§3 coverage rule): the canonical residual-template
library and the executable P-graph route logic. MuJoCo-driven orchestration (_prepare/_fit_eval, generators) is
exercised by the live runs; these test the deterministic, side-effect-free units."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.experiments.coin_toss_pgraph import CoinTossProcessGraph, build
from hymeko_rl.experiments.exp_coin_toss_cold_generators import _K, _SCALE, _TEMPLATES, _tmpl


@pytest.mark.parametrize("name", _TEMPLATES)
def test_template_shape_and_bound(name) -> None:
    seq = _tmpl(name, amp=0.12 * _SCALE, coord=0, coord2=2, rng=np.random.default_rng(0))
    assert len(seq) == _K
    for r in seq:
        assert r.shape == (4,) and r.dtype == np.float32
        assert np.all(np.abs(r) <= _SCALE * 0.15 + 1e-6)     # residual bound respected


def test_template_zero_amp_is_zero() -> None:
    for name in _TEMPLATES:
        seq = _tmpl(name, amp=0.0, coord=1, coord2=3, rng=np.random.default_rng(0))
        assert np.allclose(np.stack(seq), 0.0)               # no correction when amplitude is zero


def test_pulse_is_transient() -> None:
    seq = _tmpl("pulse", amp=0.1 * _SCALE, coord=0, coord2=1, rng=np.random.default_rng(0))
    assert seq[0][0] != 0.0 and seq[2][0] == 0.0             # pulse acts early, releases later


def test_push_counterpush_reverses() -> None:
    seq = _tmpl("push_counterpush", amp=0.1 * _SCALE, coord=2, coord2=0, rng=np.random.default_rng(0))
    assert np.sign(seq[0][2]) == -np.sign(seq[3][2])         # push then counter-push on the same coord


def test_pgraph_route_valid_and_invalid() -> None:
    g = CoinTossProcessGraph()
    r = g.route_cost_yield(["phase_extract", "candidate_gen", "paired_verify"])
    assert [s["unit"] for s in r["steps"]] == ["phase_extract", "candidate_gen", "paired_verify"]
    with pytest.raises(KeyError):
        g.route_cost_yield(["not_a_unit"])


def test_pgraph_renders_and_annotates_profile() -> None:
    g = build(".")                                            # annotate_from_results runs; profile is always available
    assert "simulate_continuation" in g.units and g.units["simulate_continuation"].cost != "unmeasured"
    assert g.to_mermaid().startswith("flowchart")
    assert "| unit |" in g.to_table()
