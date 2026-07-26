"""R1 signed directional authority — pure logic + the Jacobian-equivariance test (the task's requested extension of the
equivariance check to B_coin). Under the target-frame mirror B_coin(Mx) = S_y B_coin(x) P_τ (S_y=diag(1,−1), P_τ = the
left-right joint permutation), the derived REACHABLE features must transform: forward push/brake invariant, lateral ±
swap, contact balance sign-flip."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.directional_authority import (
    contact_reaches, object_reaches, reachable)

PERM = [2, 3, 0, 1]                                     # left-right joint permutation (joints 0,1=left, 2,3=right)
S_Y = np.diag([1.0, -1.0])                              # target-frame mirror: along invariant, perp flips


def _perm_mat(perm):
    P = np.zeros((len(perm), len(perm)))
    for j, k in enumerate(perm):
        P[k, j] = 1.0                                  # (B @ P)[:, j] = B[:, perm[j]]
    return P


# ───────────────────────────── reachable closed form ─────────────────────────────
def test_reachable_symmetric_box_is_sum_abs_times_bound():
    c = np.array([1.0, -2.0, 0.5, -0.3])
    assert reachable(c, np.full(4, -0.3), np.full(4, 0.3)) == np.sum(np.abs(c)) * 0.3
    assert reachable(-c, np.full(4, -0.3), np.full(4, 0.3)) == reachable(c, np.full(4, -0.3), np.full(4, 0.3))  # symmetric ⇒ ±equal


def test_reachable_asymmetric_box_breaks_symmetry():
    c = np.array([1.0, 1.0, 0.0, 0.0])
    lb, ub = np.array([-0.1, -0.3, -0.3, -0.3]), np.array([0.3, 0.1, 0.3, 0.3])
    assert reachable(c, lb, ub) == 0.3 + 0.1            # +c picks ub
    assert reachable(-c, lb, ub) == 0.1 + 0.3          # −c picks lb (−·lb) → asymmetric headroom ⇒ +≠− in general
    # a genuinely asymmetric coefficient over this box gives different ±reach
    c2 = np.array([1.0, 0.0, 0.0, 0.0])
    assert reachable(c2, lb, ub) != reachable(-c2, lb, ub)     # 0.3 vs 0.1


# ───────────────────────────── Jacobian equivariance (object) ─────────────────────────────
def test_object_reaches_mirror_equivariance():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(2, 4))
    lb = -rng.uniform(0.05, 0.3, 4)                     # asymmetric box (so ± are genuinely distinguished)
    ub = rng.uniform(0.05, 0.3, 4)
    v_tf, sp = np.array([0.2, -0.1]), 0.2236
    a = object_reaches(B, lb, ub, v_tf, sp)
    # mirror: B → S_y B P_τ ; box permutes with the joints ; coin velocity perp component flips (S_y @ v)
    P = _perm_mat(PERM)
    Bm = S_Y @ B @ P
    lbm, ubm = lb[PERM], ub[PERM]
    vm = S_Y @ v_tf
    am = object_reaches(Bm, lbm, ubm, vm, sp)
    assert np.isclose(am["forward_push_reach"], a["forward_push_reach"]).all()      # forward INVARIANT
    assert np.isclose(am["forward_reverse_reach"], a["forward_reverse_reach"]).all()
    assert np.isclose(am["brake_opposed_reach"], a["brake_opposed_reach"]).all()    # brake (opposes v) INVARIANT
    assert np.allclose(am["lateral_reach_pair"], a["lateral_reach_pair"][::-1])     # lateral ± SWAP


# ───────────────────────────── Jacobian equivariance (contact balance) ─────────────────────────────
def test_contact_balance_sign_flips_under_mirror():
    rng = np.random.default_rng(1)
    Btau = rng.normal(size=(4, 4))                     # rows [Lvn,Lvt,Rvn,Rvt], cols = joints
    lb = -rng.uniform(0.05, 0.3, 4)
    ub = rng.uniform(0.05, 0.3, 4)
    c = contact_reaches(Btau, lb, ub)
    # contact mirror: swap L↔R normal rows (0↔2) and permute joint columns
    Btm = Btau.copy()
    Btm[[0, 2]] = Btau[[2, 0]]
    Btm = Btm[:, PERM]
    lbm, ubm = lb[PERM], ub[PERM]
    cm = contact_reaches(Btm, lbm, ubm)
    assert np.allclose(cm["normal_force_reach_pair"], c["normal_force_reach_pair"])   # total-normal INVARIANT (symmetric)
    assert np.isclose(cm["balance_reach_signed"], -c["balance_reach_signed"]).all()   # balance SIGN-FLIPS (antisymmetric)


def test_symmetric_state_has_zero_balance():
    # left == right rows ⇒ antisymmetric mode is zero ⇒ balance == 0 (a symmetric cradle picks either orientation)
    row = np.array([0.1, -0.2, 0.05, 0.0])
    Btau = np.array([row, [0.0] * 4, row, [0.0] * 4])   # Lvn == Rvn
    c = contact_reaches(Btau, np.full(4, -0.3), np.full(4, 0.3))
    assert abs(float(c["balance_reach_signed"][0])) < 1e-12
