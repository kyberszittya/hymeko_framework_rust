"""§2 RESIDUAL_CRITIC_STATE_V2 contract tests: streaming==batch, checkpoint resume, no future obs, dims, no
offline-phase-label leakage, versioned SHA."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_residual_critic_state import (
    RESIDUAL_CRITIC_STATE_DIM,
    ResidualCriticStateV2,
    build_critic_states_v2,
    residual_critic_state_v2_contract,
)


def _gate(t):
    return {"gate": float(t % 2), "mode": "LATE_CONTROL_ARMED" if t % 2 else "EARLY_CONTROL",
            "bilateral_counter": t % 3, "uni_counter": t % 5, "uni_side": "R" if t % 2 else None, "loss_counter": 0}


def _stream(n=20, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, 48)).astype(np.float32)
    act = rng.standard_normal((n, 4)).astype(np.float32)
    gates = [_gate(t) for t in range(n)]
    return obs, act, gates


def test_dim_163():
    assert RESIDUAL_CRITIC_STATE_DIM == 163
    st = ResidualCriticStateV2(); st.reset(np.zeros(48, np.float32))
    assert st.feature(_gate(0)).shape == (163,)


def test_streaming_equals_batch():
    obs, act, gates = _stream()
    n = len(obs)
    st = ResidualCriticStateV2(); st.reset(obs[0])
    stream = []
    for t in range(n):
        stream.append(st.feature(gates[t]))                       # feature at t (pre-decision)
        if t + 1 < n:
            st.push(obs[t + 1], act[t])                           # advance with executed action a_t -> obs_{t+1}
    stream = np.stack(stream)
    batch = build_critic_states_v2(obs, act, np.zeros(n, np.int32), gates)
    assert np.allclose(stream, batch, atol=1e-6)


def test_resume_reproduces_next_feature():
    obs, act, gates = _stream()
    st = ResidualCriticStateV2(); st.reset(obs[0])
    for t in range(6):
        st.feature(gates[t]); st.push(obs[t + 1], act[t])
    sd = st.state_dict()
    st2 = ResidualCriticStateV2(); st2.load_state_dict(sd)
    assert np.allclose(st.feature(gates[6]), st2.feature(gates[6]), atol=1e-7)
    st.push(obs[7], act[6]); st2.push(obs[7], act[6])
    assert np.allclose(st.feature(gates[7]), st2.feature(gates[7]), atol=1e-7)


def test_no_future_obs():
    # feature at t must not change if a FUTURE obs is later pushed (it only uses <= t)
    obs, act, gates = _stream()
    st = ResidualCriticStateV2(); st.reset(obs[0])
    st.push(obs[1], act[0]); st.push(obs[2], act[1])
    f2 = st.feature(gates[2]).copy()
    st.push(obs[3], act[2])                                       # future push
    st.load_state_dict({"obs": [obs[2], obs[1], obs[0]], "act": [act[1], act[0]]})
    assert np.allclose(st.feature(gates[2]), f2, atol=1e-7)


def test_gate_encoding_distinguishes_and_no_leakage():
    st = ResidualCriticStateV2(); st.reset(np.zeros(48, np.float32))
    a = st.feature({"gate": 1.0, "mode": "LATE_CONTROL_ARMED", "uni_side": "R"})
    b = st.feature({"gate": 0.0, "mode": "EARLY_CONTROL", "uni_side": None})
    assert not np.array_equal(a, b)
    leak = st.feature({"gate": 1.0, "mode": "LATE_CONTROL_ARMED", "uni_side": "R",
                       "disk_to_zone": 0.001, "success": True})       # extra keys ignored
    assert np.array_equal(a, leak)


def test_contract_sha_and_history_untouched():
    c = residual_critic_state_v2_contract()
    assert c["dim"] == 163 and len(c["sha256"]) == 64
    assert "pi_0 receives its original 48-dim" in c["base_actor_untouched"]
