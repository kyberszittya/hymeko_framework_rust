"""CONTACT_STABILIZED_PRIMITIVE_MPC_V1 contract tests (step 8), fast + deterministic via a fake env + injectable
snap/restore seam. Proves the 10 required properties before evaluation."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_primitive_mpc import (
    ACTION_SCALE,
    NEUTRAL_THETA,
    THETA_DIM,
    PrimitiveBounds,
    PrimitiveController,
    Theta,
    detect_mode,
    execution_guard,
    mode_offset,
    pack_theta,
    primitive_key,
    unpack_theta,
)


class _FakePi0:
    def action_mean(self, obs):                                  # pi_0 ≡ zeros ⇒ offset magnitude = |candidate action|
        return torch.zeros(obs.shape[0], 4)


class _Met:
    def __init__(self, c):
        self.left_contact = c; self.right_contact = False


class _Inner:
    class _Data:
        qvel = np.zeros(8, np.float32)
    def __init__(self, c):
        self._planar_metrics = _Met(c); self.data = _Inner._Data()


class _FakeRL:
    """Contact holds only when the executed action is gentle (|a| <= keep_tol) — so pi_0 (zeros) keeps contact and a
    large offset breaks it, letting the guard's blend behaviour be tested deterministically."""
    def __init__(self, *, dtz=0.2, speed=0.3, contact=True, strict=0, touched=True, keep_tol=0.6):
        self.dtz = dtz; self.speed = speed; self.contact = contact; self._strict = strict; self._touched = touched
        self._t = 0; self.keep_tol = keep_tol; self.inner = _Inner(contact); self.cf = None
    def obs(self):
        return np.zeros(48, np.float32)
    def _dtz(self):
        return self.dtz
    def _speed(self):
        return self.speed
    def step(self, a):
        self.contact = float(np.abs(a).max()) <= self.keep_tol; self.inner._planar_metrics = _Met(self.contact)
        self._t += 1; return self.obs(), 0.0, False, False, {}


def _snap(rl):
    return (rl.dtz, rl.speed, rl.contact, rl._strict, rl._touched, rl._t)


def _restore(rl, s):
    rl.dtz, rl.speed, rl.contact, rl._strict, rl._touched, rl._t = s; rl.inner._planar_metrics = _Met(rl.contact)


PI0 = _FakePi0()


# 1 — theta has exactly 10 dimensions
def test_theta_is_10d():
    assert THETA_DIM == 10 and len(NEUTRAL_THETA) == 10
    assert len(PrimitiveBounds().lo) == 10 and len(PrimitiveBounds().hi) == 10
    v = np.arange(10, dtype=np.float32); assert np.allclose(pack_theta(unpack_theta(v)), v)


# 2 — actions remain within [-4, 4]
def test_actions_clipped():
    rl = _FakeRL(strict=6, touched=True)                        # certified ⇒ guard admits candidate at α=1 (clip path)
    a, alpha, fb = execution_guard(rl, np.array([10, -10, 10, -10], np.float32), PI0, snap=_snap, restore=_restore)
    assert a.max() <= ACTION_SCALE and a.min() >= -ACTION_SCALE and alpha == 1.0


# 3 — SETTLE executes pi_0 bit-identically
def test_settle_is_pi0():
    assert np.all(mode_offset("SETTLE", unpack_theta(NEUTRAL_THETA)) == 0.0)
    rl = _FakeRL()
    ctrl = PrimitiveController(mode="SETTLE", snap=_snap, restore=_restore)
    # force SETTLE to persist: inside zone & slow
    rl.dtz = 0.01; rl.speed = 0.01
    a, diag = ctrl.step_action(rl, unpack_theta(NEUTRAL_THETA), PI0)
    assert diag["mode"] == "SETTLE" and np.allclose(a, np.zeros(4))   # == pi_0(obs) exactly


# 4 — mode transitions depend on current state, not elapsed time
def test_mode_transitions_state_based():
    th = unpack_theta(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.05, 0.06], np.float32))
    far = _FakeRL(dtz=0.3, speed=0.05); near = _FakeRL(dtz=0.03, speed=0.02)
    assert detect_mode(far, th, "PUSH") == "PUSH"                # far ⇒ keep pushing
    assert detect_mode(near, th, "PUSH") == "BRAKE"             # within brake_distance ⇒ brake
    assert detect_mode(near, th, "BRAKE") == "SETTLE"           # inside + slow ⇒ settle
    left = _FakeRL(dtz=0.2, speed=0.02); assert detect_mode(left, th, "SETTLE") == "BRAKE"   # coin left ⇒ re-brake


# 5 — guard alpha=0 executes pi_0 bit-identically (no admissible nonzero alpha)
def test_guard_fallback_is_pi0():
    rl = _FakeRL(keep_tol=0.0)                                   # only |a|=0 (pi_0) keeps contact ⇒ only α=0 admissible
    a, alpha, fb = execution_guard(rl, np.array([1.0, 1.0, 1.0, 1.0], np.float32), PI0, snap=_snap, restore=_restore)
    assert alpha == 0.0 and fb and np.allclose(a, np.zeros(4))


# 6 — unsafe candidate actions are blended toward pi_0
def test_guard_blends_toward_pi0():
    rl = _FakeRL(keep_tol=0.6)                                   # α=1 (|a|=1.0) breaks contact, α=0.5 (|a|=0.5) keeps it
    a, alpha, fb = execution_guard(rl, np.array([1.0, 1.0, 1.0, 1.0], np.float32), PI0, snap=_snap, restore=_restore)
    assert 0.0 < alpha < 1.0 and not fb and np.allclose(a, alpha * np.ones(4), atol=1e-6)


# 7 — release after certified placement is not penalized
def test_release_after_placement_legal():
    rl = _FakeRL(strict=6, touched=True, keep_tol=-1.0)         # would break contact, but certified ⇒ admitted at α=1
    a, alpha, fb = execution_guard(rl, np.array([0.5, 0, 0, 0], np.float32), PI0, snap=_snap, restore=_restore)
    assert alpha == 1.0 and not fb


# 8 — candidate simulation and real execution use the same controller (deterministic, identical output)
def test_same_controller_deterministic():
    th = unpack_theta(NEUTRAL_THETA.copy()); th = Theta(np.array([0.4] * 4, np.float32), np.zeros(4, np.float32), 0.05, 0.06)
    rl = _FakeRL(dtz=0.3, speed=0.05)
    c1 = PrimitiveController(mode="PUSH", snap=_snap, restore=_restore); a1, d1 = c1.step_action(rl, th, PI0)
    c2 = PrimitiveController(mode="PUSH", snap=_snap, restore=_restore); a2, d2 = c2.step_action(rl, th, PI0)
    assert np.allclose(a1, a2) and d1["mode"] == d2["mode"]      # same state+θ ⇒ same executed action


# 9 — state restoration is deterministic (guard never alters the real rollout)
def test_state_restoration_deterministic():
    rl = _FakeRL(dtz=0.2, speed=0.3, contact=True); before = _snap(rl)
    execution_guard(rl, np.array([1.0, 1.0, 1.0, 1.0], np.float32), PI0, snap=_snap, restore=_restore)
    assert _snap(rl) == before                                  # unchanged after the guard's branch-and-restore


# 10 — no raw open-loop action sequence is stored or executed
def test_no_open_loop_action_sequence():
    from hymeko_rl.coin_delivery.coin_primitive_mpc import PrimitiveMPCPolicy
    pol = PrimitiveMPCPolicy(PrimitiveBounds(), PI0)
    assert not any("action" in a and "seq" in a for a in dir(pol))   # no stored action sequence
    assert pol.theta is None and pol.replan_every == 4          # persists θ (10-D), recomputes actions live


def test_primitive_key_strict_outranks_progress():
    strict = {"feasible": True, "n_violations": 0, "any_strict": True, "max_dwell": 6, "min_dtz": 0.03,
              "excess_entry_speed": 0.0, "braking": 0.1, "intervention_rate": 0.2, "effort": 0.1}
    prog = {**strict, "any_strict": False, "min_dtz": 0.005}
    assert primitive_key(strict) > primitive_key(prog)
