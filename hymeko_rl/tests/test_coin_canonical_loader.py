"""Unit coverage for the canonical, experiment-free Coin loader/env factory (2026-07-21 canonicalization).

Covers the three production modules extracted out of the galambos/coin-toss experiment web:
    * :mod:`hymeko_rl.env.planar_arm_kinematics` — analytic planar 2-link IK + arm extraction.
    * :mod:`hymeko_rl.coin_delivery.env_factory` — canonical Coin env + C1 contact env + bank/contract/config.
    * :mod:`hymeko_rl.coin_delivery.e_approach` — the E-approach policy loader (fail-loud, drop-in).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from hymeko_rl.coin_delivery.e_approach import E_VALSELECT_CKPT, EApproachPolicy, load_e_approach_policy
from hymeko_rl.coin_delivery.env_factory import (VALID_EMBODIMENTS, coin_env_fingerprint, make_coin_contact_env,
                                                 make_coin_env)
from hymeko_rl.env.planar_arm_kinematics import ArmKin, extract_arms, ik_action, planar_2link_ik

_HAVE_CKPT = os.path.isfile(E_VALSELECT_CKPT)


# ── planar_arm_kinematics ────────────────────────────────────────────────────────────────────────────────────────────
def test_planar_2link_ik_roundtrip_reachable() -> None:
    """IK → forward kinematics recovers a reachable target (convention: tip at (0,0)=(−l1 sinj1−…, l1 cosj1+…))."""
    base, l1, l2 = (0.0, 0.0), 0.10, 0.08
    tgt = (0.05, 0.12)
    j1, j2 = planar_2link_ik(base, l1, l2, tgt)
    tip_x = -l1 * math.sin(j1) - l2 * math.sin(j1 + j2)
    tip_y = l1 * math.cos(j1) + l2 * math.cos(j1 + j2)
    assert math.hypot(tip_x - tgt[0], tip_y - tgt[1]) < 1e-6


def test_planar_2link_ik_clamps_out_of_reach() -> None:
    """An out-of-reach target is clamped to the reachable annulus (no NaN, finite joints)."""
    j1, j2 = planar_2link_ik((0.0, 0.0), 0.1, 0.1, (10.0, 10.0))
    assert math.isfinite(j1) and math.isfinite(j2)


def test_planar_2link_ik_rejects_nonpositive_links() -> None:
    with pytest.raises(ValueError):
        planar_2link_ik((0.0, 0.0), 0.0, 0.1, (0.1, 0.1))


def test_extract_arms_and_ik_action_on_real_model() -> None:
    env = make_coin_env(embodiment="POINT")
    arms = extract_arms(env.model)
    assert set(arms) == {"left", "right"} and all(isinstance(a, ArmKin) for a in arms.values())
    act = ik_action(((arms["left"], np.array([0.0, 0.1])), (arms["right"], np.array([0.0, 0.1]))), env)
    assert act.shape == (int(env.n_actions),) and np.all(np.isfinite(act))
    assert np.all(act >= env.action_space.low) and np.all(act <= env.action_space.high)


# ── env_factory ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_make_coin_env_default_is_deliver_v2b() -> None:
    env = make_coin_env(embodiment="POINT")
    fp = coin_env_fingerprint(env)
    assert fp["obs_space"] == [6, 8] and fp["reward_file"].endswith("galambos_task_deliver_v2b.hymeko")


def test_make_coin_env_rejects_unknown_embodiment() -> None:
    with pytest.raises(ValueError, match="embodiment"):
        make_coin_env(embodiment="NOPE")
    assert "POINT" in VALID_EMBODIMENTS


def test_make_coin_contact_env_wraps_the_coin_env() -> None:
    """The canonical C1 contact env (was pedc._env) builds over the Coin env with the pre-contact bank + POINT contract."""
    cf = make_coin_contact_env("POINT")
    fp = coin_env_fingerprint(cf._env)
    assert fp["obs_space"] == [6, 8]                            # inner env is the canonical Coin env
    assert getattr(cf, "_env", None) is not None


# ── e_approach ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_e_approach_loader_fail_loud_on_wrong_hash() -> None:
    """Never a silent substitution: wrong expected-hash raises (or FileNotFoundError in a checkpoint-less clone)."""
    with pytest.raises((ValueError, FileNotFoundError)):
        load_e_approach_policy(expected_checkpoint_hash="deadbeefdead")


def test_e_approach_loader_missing_checkpoint_is_fail_loud() -> None:
    with pytest.raises(FileNotFoundError):
        load_e_approach_policy(checkpoint_path="does/not/exist.pt", expected_checkpoint_hash=None)


@pytest.mark.skipif(not _HAVE_CKPT, reason="E_valselect_v2.pt is a gitignored external artifact (absent in a clone)")
def test_e_approach_policy_action_is_4dof_and_stable() -> None:
    pol = load_e_approach_policy()
    assert isinstance(pol, EApproachPolicy)
    env = make_coin_env(embodiment="POINT")
    env.reset(seed=1011)
    a = pol.action(env)
    assert a.shape == (4,) and a.dtype == np.float32 and np.all(np.isfinite(a))
