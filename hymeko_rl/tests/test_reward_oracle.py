"""Reward-Alignment Planner-Oracle — the oracle must reproduce the MEASURED galambos reward bug with ZERO RL,
and certify a fixed reward. Fast, deterministic (§3)."""
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.reward_oracle import Phase, _phase_reward, certify, tune_terminal_bonus

_GALAMBOS = "data/robotics/galambos_task.hymeko"


def _spec() -> RewardSpec:
    return RewardSpec.from_hymeko(_GALAMBOS)


def test_phase_reward_ordering_matches_the_annuity() -> None:
    # The per-step reward: delivering (in-zone) pays most, grasping-out next, far is a penalty. This ordering IS
    # the annuity that (with termination-on-success) creates the farming trap.
    s = _spec()
    r_far = _phase_reward(s, Phase.FAR, False, 4)
    r_out = _phase_reward(s, Phase.GRASPED_OUT, True, 4)
    r_in = _phase_reward(s, Phase.GRASPED_IN, True, 4)
    assert r_in > r_out > r_far                       # +15 > +5 > negative
    assert r_out > 0.0                                # grasping-out is a POSITIVE per-step annuity (the trap)


def test_current_reward_FAILS_certification_zero_RL() -> None:
    # THE POINT: the oracle reproduces the measured devolution (0.125 -> 0.083) with no training — the reward's
    # optimum is to FARM the grasp annuity, not deliver, because success terminates the reward stream.
    report = certify(_spec(), terminal_bonus=0.0)
    assert report.delivers is False
    # the optimum milks the +15/step in-zone annuity by oscillating in/out (hold, then `release` before success),
    # never sustaining delivery until forced at the horizon — the farmable-reward signature.
    assert "release" in report.trajectory


def test_terminal_bonus_fixes_it() -> None:
    # A large one-time delivery bonus makes delivering beat farming — the optimum flips to DELIVER.
    report = certify(_spec(), terminal_bonus=2000.0)
    assert report.delivers is True
    assert "DELIVER" in report.trajectory


def test_tune_finds_the_minimum_bonus() -> None:
    b = tune_terminal_bonus(_spec())
    assert b is not None and b > 0.0                  # a finite terminal bonus fixes it
    assert certify(_spec(), terminal_bonus=b).delivers          # and the returned value certifies
    assert not certify(_spec(), terminal_bonus=b * 0.5).delivers  # ...while half of it does not (it is ~minimal)


def test_certify_validates_inputs() -> None:
    import pytest
    with pytest.raises(ValueError):
        certify(_spec(), gamma=1.5)
