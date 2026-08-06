"""Regression tests for k-step returns in the canonical ReplayBuffer.sample_nstep (SAC target credit horizon). Uses
obs=[i,0] so the returned first-transition obs reveals the sampled start index, letting each drawn sample be checked
against the closed-form k-step return (powers of γ, termination stop, no episode-boundary crossing)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.train.replay import ReplayBuffer

_G = 0.9


def _build(rews, dones) -> ReplayBuffer:
    b = ReplayBuffer(1000, (2,), 1)
    for i, (r, dn) in enumerate(zip(rews, dones)):
        b.add(np.array([i, 0.0], np.float32), np.array([float(i)], np.float32), float(r),
              np.array([i + 1, 0.0], np.float32), bool(dn))
    return b


def _expected(rews, dones, start, n, gamma):
    R, steps, done, boot = 0.0, 0, 0.0, start
    for k in range(n):
        j = start + k
        if j >= len(rews):
            break
        R += gamma ** k * rews[j]
        steps += 1
        boot = j
        if dones[j]:
            done = 1.0
            break
    return R, gamma ** steps, done, boot + 1                       # boot+1 = next_obs value (obs=[i,0], next=[i+1,0])


def _check_all(b, rews, dones, n=3, draws=400):
    g = np.random.default_rng(0)
    bs = min(64, b.size)
    obs, act, R, s2, d, disc = b.sample_nstep(bs, n_step=n, gamma=_G, generator=g)
    seen = 0
    for _ in range(draws // 64 + 1):
        for i in range(len(R)):
            start = int(round(float(obs[i][0].item())))
            eR, eDisc, eDone, eNext = _expected(rews, dones, start, n, _G)
            assert abs(float(R[i]) - eR) < 1e-4, f"start {start}: R {float(R[i])} != {eR}"
            assert abs(float(disc[i]) - eDisc) < 1e-5
            assert float(d[i]) == eDone
            assert abs(float(s2[i][0]) - eNext) < 1e-4            # next_obs after the last accumulated transition
            assert abs(float(act[i][0]) - start) < 1e-4          # metadata aligned with the FIRST transition (§8)
            seen += 1
        obs, act, R, s2, d, disc = b.sample_nstep(bs, n_step=n, gamma=_G, generator=g)
    return seen


def test_1_three_rewards_correct_gamma_powers() -> None:
    rews = [1.0, 2.0, 3.0, 4.0, 5.0] * 16
    b = _build(rews, [False] * len(rews))
    assert _check_all(b, rews, [False] * len(rews)) > 200          # R = r_t + γ r_{t+1} + γ² r_{t+2}, verified per sample


def test_2_bootstrap_after_three_valid_transitions() -> None:
    rews = [1.0] * 80
    b = _build(rews, [False] * 80)
    obs, _a, _R, s2, d, disc = b.sample_nstep(64, n_step=3, gamma=_G, generator=np.random.default_rng(1))
    for i in range(len(d)):
        start = int(round(float(obs[i][0])))
        if start + 3 <= 80:
            assert abs(float(s2[i][0]) - (start + 3)) < 1e-4 and float(d[i]) == 0.0 and abs(float(disc[i]) - _G ** 3) < 1e-5


def test_3_accumulation_stops_at_termination() -> None:
    rews = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 12
    dones = [False, False, True, False, False, True] * 12         # episodes end every 3 transitions
    b = _build(rews, dones)
    _check_all(b, rews, dones)                                    # every sample stops at the first done in its window


def test_4_never_crosses_episode_boundary() -> None:
    rews = [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
    dones = [False, False, True, False, False, False]             # episode A ends at idx2; B is idx3..5 (reward 9)
    b = _build(rews, dones)
    obs, _a, R, _s2, _d, _disc = b.sample_nstep(6, n_step=3, gamma=_G, generator=np.random.default_rng(2))
    for i in range(len(R)):
        if int(round(float(obs[i][0]))) == 1:                     # start in episode A, one step before its terminal
            assert float(R[i]) < 3.0                              # must NOT pick up episode B's reward-9 transitions


def test_5_one_step_matches_sample() -> None:
    b = _build([1.0, 2.0, 3.0, 4.0], [False] * 4)
    r1 = b.sample_nstep(4, n_step=1, gamma=_G, generator=np.random.default_rng(5))
    r0 = b.sample(4, generator=np.random.default_rng(5))
    assert np.array_equal(r1[0].numpy(), r0[0].numpy()) and np.array_equal(r1[2].numpy(), r0[2].numpy())
    assert bool((r1[5] == _G).all())                             # disc is the constant γ


def test_6_demo_and_online_same_semantics() -> None:
    # tags (demo vs online) must not change n-step accumulation — same rewards/dones => same returns
    rews, dones = [1.0, 2.0, 3.0, 4.0, 5.0], [False] * 5
    plain = _build(rews, dones)
    tagged = ReplayBuffer(1000, (2,), 1)
    tagged.add_batch(np.array([[i, 0.0] for i in range(5)], np.float32), np.array([[float(i)] for i in range(5)], np.float32),
                     np.array(rews, np.float32), np.array([[i + 1, 0.0] for i in range(5)], np.float32),
                     np.array(dones, bool), tags=np.array([1, 2, 3, 0, 0], np.int16))
    a = plain.sample_nstep(5, n_step=3, gamma=_G, generator=np.random.default_rng(9))
    b = tagged.sample_nstep(5, n_step=3, gamma=_G, generator=np.random.default_rng(9))
    assert np.allclose(a[2].numpy(), b[2].numpy()) and np.allclose(a[5].numpy(), b[5].numpy())


def test_7_deterministic_for_fixed_rng() -> None:
    b = _build([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 8, [False] * 64)
    a = b.sample_nstep(32, n_step=3, gamma=_G, generator=np.random.default_rng(11))
    c = b.sample_nstep(32, n_step=3, gamma=_G, generator=np.random.default_rng(11))
    assert np.array_equal(a[2].numpy(), c[2].numpy()) and np.array_equal(a[0].numpy(), c[0].numpy())


def test_8_metadata_aligned_with_first_transition() -> None:
    rews = [10.0, 20.0, 30.0, 40.0, 50.0] * 14
    b = _build(rews, [False] * len(rews))
    obs, act, *_ = b.sample_nstep(64, n_step=3, gamma=_G, generator=np.random.default_rng(4))
    for i in range(len(act)):                                     # obs[0]==act[0]==the FIRST transition's index
        assert abs(float(obs[i][0]) - float(act[i][0])) < 1e-4
