"""COIN_FEEDBACK_BASELINE_RECONSTRUCTION_V1 harness unit tests (pure, deterministic; no env). Scripted traces exercise
the 11-metric derivation + phase-conditional contact legality; synthetic row-sets exercise the teacher qualification."""
import pytest

from hymeko_rl.coin_delivery.coin_baseline_reconstruction import ELEVEN_METRICS, RolloutTrace, qualify_teacher


class _Met:
    def __init__(self, lc, rc):
        self.left_contact = lc; self.right_contact = rc


class _Inner:
    def __init__(self):
        self._planar_metrics = _Met(False, False)


class _RL:
    """Minimal CoinRL4Dof stand-in driving RolloutTrace.record from a scripted (contact, dtz, speed, strict) step."""
    def __init__(self):
        self.inner = _Inner(); self._d = 0.0; self._s = 0.0; self._strict = 0; self._touched = False
    def _dtz(self):
        return self._d
    def _speed(self):
        return self._s
    def push(self, lc, rc, dtz, speed, strict):
        self.inner._planar_metrics = _Met(lc, rc); self._d = dtz; self._s = speed
        self._strict = strict; self._touched = self._touched or lc or rc


def _run(steps):
    """steps: list of (lc, rc, dtz, speed, strict[, reward]) — reward defaults to 0.0."""
    rl = _RL(); tr = RolloutTrace()
    for st in steps:
        lc, rc, dtz, speed, strict = st[:5]; r = st[5] if len(st) > 5 else 0.0
        rl.push(lc, rc, dtz, speed, strict); tr.record(rl, r)
    return tr.metrics()


def test_metrics_full_set_present_and_empty_safe():
    m = _run([(True, False, 0.5, 0.3, 0, 0.0)])
    for k in ELEVEN_METRICS:
        assert k in m
    empty = RolloutTrace().metrics()
    assert empty["n_steps"] == 0 and empty["strict_success"] is None


def test_push_delivery_contact_loss_after_stable_is_legal():
    # approach with contact, enter+settle (stable), THEN release (coast) and hold dwell to K=6 → legal, strict success
    steps = [(True, False, 0.20, 0.30, 0)]                      # transport, in contact
    steps += [(True, False, 0.015, 0.03, k) for k in range(1, 4)]  # entered+settled+contact, dwell rising
    steps += [(False, False, 0.015, 0.02, k) for k in range(4, 8)]  # released after stable placement (legal)
    m = _run(steps)
    assert m["first_contact"] and m["target_entry"]
    assert m["strict_success"] and m["max_dwell"] >= 6
    assert m["lost_required_contact"] is False                 # contact lost AFTER stable entry → legal
    assert m["required_contact_retention"] == 1.0              # contact held through the required (pre-stable) window


def test_contact_loss_before_entry_is_illegal():
    # acquire contact, then lose it while still far from the zone (before any stable entry) → illegal required loss
    steps = [(True, False, 0.30, 0.4, 0)] + [(False, False, 0.28, 0.4, 0) for _ in range(6)]
    m = _run(steps)
    assert m["first_contact"] and not m["target_entry"]
    assert m["lost_required_contact"] is True
    assert m["required_contact_retention"] < 0.5


def test_entry_velocity_and_braking():
    steps = [(True, False, 0.20, 0.50, 0), (True, False, 0.04, 0.40, 0), (True, False, 0.015, 0.02, 1)]
    m = _run(steps)
    assert m["entry_velocity"] == pytest.approx(0.40, abs=1e-6)  # coin speed at first dtz<=0.05
    assert m["braking"] == pytest.approx(0.40 - 0.02, abs=1e-6)  # max near-zone speed minus final speed


def _row(**kw):
    base = {"required_contact_retention": 1.0, "lost_required_contact": False, "target_exit": False,
            "max_dwell": 3, "strict_success": False, "total_return": 1.0, "braking": 0.1, "progress": 0.1}
    base.update(kw); return base


def test_qualify_teacher_qualified_when_planner_improves_safely():
    pi0 = [_row(max_dwell=2, strict_success=False) for _ in range(6)]
    plan = [_row(max_dwell=6, strict_success=True, total_return=2.0) for _ in range(6)]   # more dwell/strict, contact kept
    q = qualify_teacher(pi0, plan)
    assert q["qualified"] and q["verdict"] == "H30_TEACHER_QUALIFIED"


def test_qualify_teacher_unqualified_on_new_contact_loss():
    pi0 = [_row() for _ in range(6)]
    plan = [_row(required_contact_retention=0.2, lost_required_contact=True, max_dwell=6, strict_success=True) for _ in range(6)]
    q = qualify_teacher(pi0, plan)
    assert not q["qualified"] and "LOSES_REQUIRED_CONTACT" in q["verdict"]


def test_qualify_teacher_unqualified_on_no_improvement():
    pi0 = [_row(max_dwell=4, strict_success=True, total_return=2.0) for _ in range(6)]
    plan = [_row(max_dwell=4, strict_success=True, total_return=2.0) for _ in range(6)]    # identical → no gain
    q = qualify_teacher(pi0, plan)
    assert not q["qualified"] and "NO_IMPROVEMENT" in q["verdict"]
