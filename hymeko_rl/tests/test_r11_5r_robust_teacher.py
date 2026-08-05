"""Tests for R11.5R robust teacher: perturbation bank, survival, lexicographic key + eligibility gate, funnel, CEM, status."""
import types

import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_teacher import robust_teacher as RT
from hymeko_rl.coin_delivery.delivery_teacher.robust_teacher import (
    PerturbationBank,
    RobustTeacherConfig,
    recert_status,
    robust_cem,
    robust_evaluate,
    survival_at,
)


def _m(*, k6: bool = True, safe: bool = True, dtz: float = 5.0) -> dict:
    return {"peak_qdot": 1.0 if safe else 5.0, "peak_coin_speed": 0.5 if safe else 2.0,
            "k6_delivered": k6, "dtz_end": dtz / 1000.0, "release_step": 30}


def _patch(monkeypatch: pytest.MonkeyPatch, nominal: dict, pert: "list[dict] | None" = None) -> None:
    seq = [nominal] + (pert or [])
    it = iter(seq)
    monkeypatch.setattr(RT, "rollout_primitive", lambda snap, theta, cfg: next(it, seq[-1]))
    monkeypatch.setattr(RT, "delivery_success", lambda m, cfg: bool(m["k6_delivered"]))


def test_perturbation_bank_shared_and_reproducible() -> None:
    b1, b2 = PerturbationBank((0.01,), 5, seed=0), PerturbationBank((0.01,), 5, seed=0)
    assert np.array_equal(b1.deltas(0.01, 3), b2.deltas(0.01, 3))          # reproducible
    assert np.array_equal(b1.deltas(0.01, 2), b1.deltas(0.01, 5)[:2])      # prefix-shared across k


def test_survival_and_cvar(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = iter([_m(dtz=6), _m(dtz=7), _m(k6=False, dtz=40)])                  # 3 perturbations: 2 K6, 1 miss@40mm
    monkeypatch.setattr(RT, "rollout_primitive", lambda snap, theta, cfg: next(seq))
    monkeypatch.setattr(RT, "delivery_success", lambda m, cfg: bool(m["k6_delivered"]))
    surv, cvar, n = survival_at(None, np.zeros(6), 0.01, 3, PerturbationBank((0.01,), 3, 0), 0.5)
    assert surv == round(2 / 3, 3) and cvar == 23.5 and n == 3               # survival 2/3; CVaR worst-2 = mean(40,7)


def test_eligibility_gate_nonk6_sorts_below_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = PerturbationBank((0.005, 0.01, 0.02), 6, 0)
    _patch(monkeypatch, _m(k6=False))                                        # failed nominal -> ineligible
    key_bad, rec_bad, npert = robust_evaluate(None, np.zeros(6), RobustTeacherConfig(), bank)
    assert key_bad[1] == 0 and npert == 0 and not rec_bad["nom_k6"]          # no perturbation rollouts for a failed nominal
    _patch(monkeypatch, _m(k6=True), [_m()] * 6)                             # K6 + all perturbations survive
    key_good, rec_good, _n = robust_evaluate(None, np.zeros(6), RobustTeacherConfig(), PerturbationBank((0.005, 0.01, 0.02), 6, 0))
    assert key_good > key_bad and rec_good["surv05"] == 1.0                  # eligible outranks ineligible


def test_higher_survival_wins_among_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = PerturbationBank((0.005, 0.01, 0.02), 6, 0)
    _patch(monkeypatch, _m(), [_m()] * 6)                                    # wide: all survive
    key_wide, _r, _n = robust_evaluate(None, np.zeros(6), RobustTeacherConfig(), bank)
    _patch(monkeypatch, _m(), [_m(k6=False, dtz=40)] * 6)                    # narrow: none survive
    key_narrow, _r2, _n2 = robust_evaluate(None, np.zeros(6), RobustTeacherConfig(), bank)
    assert key_wide > key_narrow                                            # wide basin sorts above narrow


def test_recert_status() -> None:
    cfg = RobustTeacherConfig()
    assert recert_status({"nom_k6": True, "safe": True, "surv1": 0.80}, cfg) == "WIDE_RECERTIFIED"
    assert recert_status({"nom_k6": True, "safe": True, "surv1": 0.40}, cfg) == "NARROW_ONLY"
    assert recert_status({"nom_k6": False}, cfg) == "NO_NOMINAL_K6"


def test_robust_cem_runs_and_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RT, "rollout_primitive", lambda snap, theta, cfg: _m())   # everything K6+survives
    monkeypatch.setattr(RT, "delivery_success", lambda m, cfg: True)
    spec = types.SimpleNamespace(search_idx=(0, 1, 2, 3, 4, 5), lo=(0.04,) * 6, hi=(0.2,) * 6,
                                 init_std=(0.05,) * 6, pop=4, iters=2, elite=2, assemble=lambda x: np.asarray(x, np.float64))
    res = robust_cem(None, spec, RobustTeacherConfig(), PerturbationBank((0.005, 0.01, 0.02), 6, 0), seed=0)
    assert res.compute["proposals"] == 8 and res.compute["sim_calls"] > 8 and res.certificate().nominal_k6
