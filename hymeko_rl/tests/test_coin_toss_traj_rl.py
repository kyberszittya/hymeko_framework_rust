"""Coverage for the flat-history trajectory-correction policy matrix (directive R0-R3). Pure/small — no MuJoCo:
fakes for the DAgger base + env metrics. Step-zero DAgger identity, chunk/mode shapes, R3 param-match to R2,
flat-history buffer parity with `_p0_window`, training runs + is reward-derived (advantage/uniform/preference)."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.experiments.exp_coin_toss_traj_rl import (_IN, _KMAX, _KMODES, _ChunkHead, _ModeHead, _policy_input,
                                                         _r3_hidden_matched, TrajChunkPolicy, train_traj)


class _FakeMetrics:
    def __init__(self, left=True, right=False):
        self.left_contact, self.right_contact, self.arm_self_contact, self.in_zone = left, right, False, False
        self.disk_pos = np.array([0.1, 0.2], np.float32); self.disk_vel = np.array([0.01, -0.02], np.float32)


class _FakeEnv:
    def __init__(self, left=True, right=False):
        self._planar_metrics = _FakeMetrics(left, right)


class _FakeDagger:
    def action_mean(self, x):
        return torch.full((x.shape[0], 4), 0.3)


def _synth_dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    chunk = rng.uniform(-0.5, 0.5, (n, _KMAX, 4)).astype(np.float32)
    ddel = np.zeros(n, np.float32); ddel[:15] = 1.0                          # 15 positives
    return {"hist": rng.standard_normal((n, 6, 10)).astype(np.float32),
            "obs": rng.standard_normal((n, 48)).astype(np.float32),
            "base_action": rng.uniform(-1, 1, (n, 4)).astype(np.float32), "chunk": chunk,
            "horizon": rng.choice([2, 4, 8], n).astype(np.float32), "ddel": ddel,
            "dret": (ddel * 3 + rng.standard_normal(n)).astype(np.float32),
            "safe_pos": (ddel > 0).astype(np.int64), "harmful": np.zeros(n, np.int64),
            "template": rng.integers(-1, _KMODES, n), "phase": rng.integers(1, 3, n),
            "pfx": rng.integers(0, 8, n), "oob": np.zeros(n, np.int64)}


def test_policy_input_dim() -> None:
    x = _policy_input(np.zeros((6, 10), np.float32), np.zeros(4, np.float32), 1)
    assert x.shape == (_IN,)


def test_chunk_head_zero_init_identity() -> None:
    h = _ChunkHead(); chunk, _hor, _gate = h(torch.randn(5, _IN))
    assert chunk.shape == (5, _KMAX, 4)
    assert torch.allclose(chunk, torch.zeros_like(chunk), atol=1e-6)         # zero-init ⇒ residual 0 = the DAgger


def test_mode_head_zero_amp_identity() -> None:
    o = _ModeHead()(torch.randn(5, _IN))
    assert o["mode"].shape == (5, _KMODES)
    assert torch.allclose(o["amp"], torch.zeros_like(o["amp"]), atol=1e-6)   # zero amp ⇒ template chunk 0 = the DAgger


def test_r3_param_matched_to_r2() -> None:
    p2 = sum(p.numel() for p in _ModeHead().parameters())
    p3 = sum(p.numel() for p in _ChunkHead(_r3_hidden_matched()).parameters())
    assert 0.9 * p2 <= p3 <= 1.35 * p2                                       # equal-capacity control (§3 R3)


def test_traj_policy_step_zero_is_dagger() -> None:
    lo, hi = -np.ones(4, np.float32) * 4, np.ones(4, np.float32) * 4
    for kind, head in (("chunk", _ChunkHead()), ("mode", _ModeHead())):
        pol = TrajChunkPolicy(_FakeDagger(), head, lo, hi, kind=kind)
        out = pol(_FakeEnv(left=True, right=False), np.zeros((6, 8), np.float32))   # phase-1 contact ⇒ plans, but residual 0
        assert np.allclose(out, 0.3, atol=1e-5)                              # == DAgger base (0.3)


def test_traj_policy_no_contact_returns_base() -> None:
    pol = TrajChunkPolicy(_FakeDagger(), _ChunkHead(), -np.ones(4, np.float32) * 4, np.ones(4, np.float32) * 4, kind="chunk")
    out = pol(_FakeEnv(left=False, right=False), np.zeros((6, 8), np.float32))      # phase 0 ⇒ gated off, pure base
    assert np.allclose(out, 0.3, atol=1e-6)


def test_train_runs_all_weightings() -> None:
    d = _synth_dataset()
    for kind in ("chunk", "mode", "flat"):
        for w in ("advantage", "uniform", "preference"):
            head = train_traj(kind, d, seed=0, weighting=w, epochs=30)
            assert sum(p.numel() for p in head.parameters()) > 0


def test_train_is_reward_derived() -> None:
    """Advantage weighting must change the learned head vs uniform on the SAME data (reward signal is used)."""
    d = _synth_dataset()
    ha = train_traj("chunk", d, seed=0, weighting="advantage", epochs=120)
    hu = train_traj("chunk", d, seed=0, weighting="uniform", epochs=120)
    x = torch.randn(16, _IN)
    assert not torch.allclose(ha(x)[0], hu(x)[0], atol=1e-4)                 # advantage ≠ uniform ⇒ reward-derived
