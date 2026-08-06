"""Unit tests for the phase-gated hierarchical policy (handoff routing, not a residual)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.agents.phase_gated import PhaseGatedPolicy, phase_gated_action_fn


class _MLP(nn.Module):
    def action_mean(self, o: torch.Tensor) -> torch.Tensor:
        return torch.full((o.shape[0], 4), -9.0)      # marker: MLP action = -9

    forward = action_mean


class _PPC:
    def __init__(self) -> None:
        self.n = 0

    def reset(self) -> None:
        self.n = 0

    def action(self, _env) -> np.ndarray:
        self.n += 1
        return np.full(4, 7.0, np.float32)             # marker: PPC action = +7


class _Env:
    def __init__(self, left: float, right: float) -> None:
        self._z = np.array([left, right, 1, 0, 0], np.float32)

    def privileged_state(self) -> np.ndarray:
        return self._z

    def node_features(self) -> np.ndarray:
        return np.zeros((6, 8), np.float32)


def test_gate_routes_mlp_in_approach() -> None:
    pg = PhaseGatedPolicy(_MLP(), _PPC())
    a = pg.action(_Env(0.0, 0.0))                      # no contact → MLP
    assert np.allclose(a, -9.0) and pg.last_phase == "APPROACH"


def test_gate_routes_ppc_on_contact() -> None:
    pg = PhaseGatedPolicy(_MLP(), _PPC())
    assert np.allclose(pg.action(_Env(1.0, 0.0)), 7.0) and pg.last_phase == "PUSH"   # one fingertip → PPC
    assert np.allclose(pg.action(_Env(1.0, 1.0)), 7.0)                               # both → PPC


def test_ppc_fsm_advanced_every_step() -> None:
    ppc = _PPC()
    pg = PhaseGatedPolicy(_MLP(), ppc)
    pg.action(_Env(0.0, 0.0))                          # MLP executes, but PPC.action still called (FSM advances)
    pg.action(_Env(0.0, 0.0))
    assert ppc.n == 2                                  # PPC advanced on both steps despite MLP executing


def test_reset_resets_ppc_and_phase() -> None:
    ppc = _PPC()
    pg = PhaseGatedPolicy(_MLP(), ppc)
    pg.action(_Env(1.0, 1.0))
    pg.reset()
    assert ppc.n == 0 and pg.last_phase == "APPROACH"


def test_action_fn_adapter() -> None:
    pg = PhaseGatedPolicy(_MLP(), _PPC())
    fn = phase_gated_action_fn(pg)
    assert np.allclose(fn(_Env(0.0, 0.0), np.zeros((6, 8), np.float32)), -9.0)
