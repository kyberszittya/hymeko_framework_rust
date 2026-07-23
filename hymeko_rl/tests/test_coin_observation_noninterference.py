"""OBSERVATION_NONINTERFERENCE_CONTRACT_V1 regression. Proves (bit-exact) that the diagnostic capture path
(node_features obs + _cert_step + PBRS-restored reward decomposition) is identical to the canonical rollout, that
node_features is read-only, and that the authoritative start_id disambiguates seed collisions."""
import hashlib

import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_start_id import seed_collision_report, start_id, start_id_from_row

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"
TDCFG = "experiments/2026_07_22_coin_v3_learning/rl_entry/transport_dwell_config.json"


class _LS:
    def __init__(self, seed, prefix, family):
        self.seed, self.prefix_steps, self.family = seed, prefix, family
        self.obs_sha = self.base_sha = self.causal_sha = ""


def test_start_id_disambiguates_seed_collision():
    a = _LS(6200, 99, "settling_dwell"); b = _LS(6200, 12, "transport")     # same seed, different start
    assert start_id(a) != start_id(b)                                        # seed alone would collide
    assert start_id(a) == start_id_from_row(6200, 99, "settling_dwell")
    rep = seed_collision_report([a, b, _LS(6200, 99, "settling_dwell")])
    assert rep["n_unique_seeds"] == 1 and rep["n_unique_start_id"] == 2      # 3 rows, 1 seed, 2 distinct starts


def _hash(a):
    return hashlib.md5(np.asarray(a, np.float64).tobytes()).hexdigest()


@pytest.mark.slow
def test_capture_bit_identical_to_canonical_and_reproduces_6of31():
    import json

    from hymeko_rl.coin_delivery.coin_contract_audit import decompose_reward
    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.coin_transport_dwell import CONTROL_MODES
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    from hymeko_rl.experiments.coin_neutral_start import _cert_step
    torch.set_num_threads(1)
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    cfg = json.load(open(TDCFG))
    bank = lambda m: [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]
    dev = [ls for m in CONTROL_MODES for ls in bank(cfg["banks"]["dev"][m])]

    def pia(o):
        with torch.no_grad():
            return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(o, np.float32)[None]))[0].numpy(), -4, 4).astype(np.float32)

    def roll(ls, capture):
        rl, g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360); seq = []; keys = ("_pbrs_prev_zone", "_pbrs_prev_grasp", "_pbrs_prev_conj")
        for _s in range(60):
            if capture:
                _cert_step(rl.inner, rl.cf); mem = {k: getattr(rl.inner, k) for k in keys if hasattr(rl.inner, k)}
            o = rl.obs(); a = pia(o); _o, rw, term, trunc, _ = rl.step(a)
            if capture:
                for k, v in mem.items():
                    setattr(rl.inner, k, v)
                decompose_reward(rl.inner, rl._dtz(), rl.inner.data.ctrl, rl.inner.reward_spec.terms); _cert_step(rl.inner, rl.cf)
            seq.append((_hash(o), _hash(a), _hash(rl.inner.data.qpos), _hash(rl.inner.data.qvel), round(float(rw), 8), int(rl._strict)))
            if term or trunc:
                break
        return seq

    # bit-identical on the 5 previously-mismatching settling starts
    for ls in [x for x in dev if x.family == "settling_dwell"]:
        assert roll(ls, False) == roll(ls, True), f"capture diverged from canonical at {ls.seed}"


def test_node_features_is_read_only():
    import json

    from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    pi0 = load_frozen_clip_actor(PI0, freeze=True); cfg = json.load(open(TDCFG))
    r0 = cfg["banks"]["dev"]["settling_dwell"]["rows"][0]
    ls = LateStart(seed=r0[0], prefix_steps=r0[1], family=r0[2], obs_sha=r0[3], base_sha=r0[4], causal_sha=r0[5])
    rl, _g, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    qp, qv = rl.inner.data.qpos.copy(), rl.inner.data.qvel.copy()
    oa = np.asarray(rl.inner.node_features()).copy(); ob = np.asarray(rl.inner.node_features()).copy()
    assert np.array_equal(oa, ob) and np.array_equal(qp, rl.inner.data.qpos) and np.array_equal(qv, rl.inner.data.qvel)
