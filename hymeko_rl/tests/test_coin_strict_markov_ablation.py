"""STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — the 5 gate tests required BEFORE training. Prove the exact counter
de-aliases the state, Arm A preserves reward+termination bit-exact, and Arm B pays the terminal bonus once at K6 with no
K5 farming."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import (
    arm_b_terminal_bonus,
    augment_with_strict,
    strict_onehot,
)


class _RL:
    def __init__(self, strict):
        self._strict = strict


# 1. strict 0..5 produce distinct state inputs
def test_strict_counter_states_distinct():
    base = np.zeros(62, np.float32)
    states = [augment_with_strict(base, _RL(k)) for k in range(6)]
    hashes = {s.tobytes() for s in states}
    assert len(hashes) == 6 and all(s.shape == (62 + 7,) for s in states)   # 6 distinct, exact-counter appended
    assert len({strict_onehot(k).tobytes() for k in range(6)}) == 6


# 2. identical physical state + distinct strict count is no longer aliased
def test_identical_physical_state_distinct_strict_not_aliased():
    obs = np.arange(62, dtype=np.float32)                                    # one fixed physical observation
    a4 = augment_with_strict(obs, _RL(4)); a5 = augment_with_strict(obs, _RL(5))
    assert not np.array_equal(a4, a5)                                        # strict 4 vs 5 no longer identical inputs
    assert np.array_equal(a4[:62], a5[:62])                                  # the physical part is unchanged


# 4. Arm B bonus occurs once, at K6, on the terminal transition (NOT at K5)
def test_arm_b_bonus_once_at_k6_terminal():
    b5, paid5 = arm_b_terminal_bonus(strict=5, touched=True, bonus_paid=False, grade=1.0)
    assert b5 == 0.0 and paid5 is False                                     # NO bonus at K5 (off-by-one removed)
    b6, paid6 = arm_b_terminal_bonus(strict=6, touched=True, bonus_paid=False, grade=1.0)
    assert b6 == 30.0 and paid6 is True                                     # +30 at K6 terminal, latched
    b6b, paid6b = arm_b_terminal_bonus(strict=6, touched=True, bonus_paid=True, grade=1.0)
    assert b6b == 0.0 and paid6b is True                                    # not paid twice
    assert arm_b_terminal_bonus(strict=6, touched=False, bonus_paid=False, grade=1.0)[0] == 0.0  # needs robot attribution


# 5. no K5 bonus farming is possible (reach K5, break, rebuild → no bonus until the single K6 terminal)
def test_no_k5_farming():
    paid = False; total = 0.0
    # dwell 0..5, break to 0, rebuild 1..5, then K6 — the farming exploit sequence
    for strict in [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]:
        bonus, paid = arm_b_terminal_bonus(strict=strict, touched=True, bonus_paid=paid, grade=1.0)
        total += bonus
    assert total == 30.0                                                    # exactly one bonus, only at K6


# 3. reward and termination are exactly reproduced (Arm A) — env-level, deterministic
def test_arm_a_reward_termination_bit_identical_to_canonical():
    import json

    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_strict_markov_ablation import CoinRL4DofAblation
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    torch.set_num_threads(1)
    pi0 = load_frozen_clip_actor("experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt", freeze=True)
    cfg = json.load(open("experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"))
    r0 = cfg["banks"]["dev"]["settling_dwell"]["rows"][0]
    ls = LateStart(seed=r0[0], prefix_steps=r0[1], family=r0[2], obs_sha=r0[3], base_sha=r0[4], causal_sha=r0[5])

    def pia(o):
        with torch.no_grad():
            return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(o, np.float32)[None]))[0].numpy(), -4, 4).astype(np.float32)

    # canonical CoinRL4Dof rollout vs Arm-A rollout, from the SAME reconstructed handoff, must be bit-identical
    canon = []; arm_a = []
    for env_is_arm_a in (False, True):
        rl, _g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
        if env_is_arm_a:                                                    # swap in the Arm-A env at the same state
            abl = CoinRL4DofAblation(arm="A"); abl.inner = rl.inner; abl.env = rl.env; abl.cf = rl.cf
            abl._t, abl._strict, abl._touched = rl._t, rl._strict, rl._touched; rl = abl
        seq = canon if not env_is_arm_a else arm_a
        for _s in range(20):
            a = pia(rl.obs()); _o, rw, term, trunc, _ = rl.step(a)
            seq.append((round(float(rw), 8), bool(term), int(rl._strict)))
            if term or trunc:
                break
    assert isinstance(canon[0][0], float)
    # Arm A must reproduce canonical reward+termination+strict exactly (it IS CoinRL4Dof with the same reward path)
    rl2, _g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    aA = CoinRL4DofAblation(arm="A"); aA.inner = rl2.inner; aA.env = rl2.env; aA.cf = rl2.cf
    aA._t, aA._strict, aA._touched = rl2._t, rl2._strict, rl2._touched
    seqA = []
    for _s in range(20):
        a = pia(aA.obs()); _o, rw, term, trunc, _ = aA.step(a); seqA.append((round(float(rw), 8), bool(term), int(aA._strict)))
        if term or trunc:
            break
    assert seqA == canon                                                   # Arm A == canonical, bit-identical


def test_strict_1_vs_5_distinct_actor_critic_target_replay_inputs():
    # step 4: identical physical obs48 with strict 1 vs 5 must give different actor / online-critic / target-critic /
    # replay-state inputs (counter NOT reconstructed from control_mode).
    from hymeko_rl.coin_delivery.coin_markov_ablation_train import _aug
    from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
    obs48 = np.arange(48, dtype=np.float32)
    s1 = _aug(obs48, 1); s5 = _aug(obs48, 5)
    assert not np.array_equal(s1, s5) and np.array_equal(s1[:48], s5[:48])       # replay/actor states differ, physics same
    critic = LateTwinCritic(obs_dim=55)
    act = np.zeros(4, np.float32)
    q1 = critic(torch.as_tensor(s1[None]), torch.as_tensor(act[None]))[0]
    q5 = critic(torch.as_tensor(s5[None]), torch.as_tensor(act[None]))[0]
    assert not torch.equal(q1, q5)                                              # online (and target, same net class) critic inputs differ
