"""Coverage for the 2×2 input-output factorial (directive §2). Pure/small — reuses the traj-rl fakes."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.experiments.exp_coin_toss_factorial import FactorialHead, FactorialPolicy, _KF, train_factorial
from hymeko_rl.experiments.exp_coin_toss_traj_rl import _IN
from hymeko_rl.tests.test_coin_toss_traj_rl import _FakeDagger, _FakeEnv, _synth_dataset


def test_four_cells_forward_and_zero_init() -> None:
    for uh in (False, True):
        for fc in (False, True):
            h = FactorialHead(uh, fc); n = _IN if uh else 48
            r, gate = h(torch.randn(5, n))
            assert gate.shape == (5, 1)
            assert r.shape == ((5, _KF, 4) if fc else (5, 4))
            assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)         # zero-init ⇒ residual 0 = DAgger


def test_factorial_policy_step_zero_identity() -> None:
    lo, hi = -np.ones(4, np.float32) * 4, np.ones(4, np.float32) * 4
    for uh in (False, True):
        for fc in (False, True):
            pol = FactorialPolicy(_FakeDagger(), FactorialHead(uh, fc), lo, hi)
            out = pol(_FakeEnv(left=True, right=False), np.zeros((6, 8), np.float32))
            assert np.allclose(out, 0.3, atol=1e-5)                          # == DAgger base at init


def test_obs_cell_input_dim_48() -> None:
    h = FactorialHead(use_history=False, full_chunk=False)
    assert h.body[0].in_features == 48                                        # current-obs cell consumes obs(48)
    assert FactorialHead(use_history=True, full_chunk=False).body[0].in_features == _IN


def test_train_all_cells() -> None:
    d = _synth_dataset()
    for uh in (False, True):
        for fc in (False, True):
            head = train_factorial(uh, fc, d, seed=0, epochs=20)
            assert sum(p.numel() for p in head.parameters()) > 0
