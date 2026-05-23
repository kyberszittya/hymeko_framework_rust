"""Controllers for the lateral-tracking benchmark.

All controllers share a uniform contract::

    controller.reset(track, initial_state)
    delta = controller.step(state, track, dt)

``delta`` is the front-wheel steering angle (rad).  Throttle/brake is
not part of this benchmark (constant longitudinal velocity).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn

from signedkan_wip.src.sequence.hsikan_seq import HSiKANSeqWindow
from signedkan_wip.src.sequence.clifford import CL2_DIM

from .bicycle import BicycleParams, BicycleState
from .tracks import Track


# ----------------------------------------------------------------- LQR
class LQRController:
    """Continuous-time LQR over a linearised error model.

    State error e = [lateral_err, heading_err]ᵀ.  At constant v, the
    linearised model is::

        ė_y   = v · e_ψ
        ė_ψ   = (v / L) · δ  − v · κ_ref

    Cost = ∫ e_yᵀ Q_y e_y + e_ψᵀ Q_ψ e_ψ + δᵀ R δ.  Solve via SciPy.
    """

    def __init__(self, params: BicycleParams,
                  q_y: float = 1.0, q_psi: float = 1.0, r: float = 0.5):
        self.p = params
        v = self.p.v_target
        A = np.array([[0.0, v], [0.0, 0.0]])
        B = np.array([[0.0], [v / self.p.L]])
        Q = np.diag([q_y, q_psi])
        R = np.array([[r]])
        # Riccati via scipy
        from scipy.linalg import solve_continuous_are
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)  # (1, 2)
        self.K = self.K.reshape(-1)            # → (2,)

    def reset(self, track: Track, state: BicycleState) -> None:
        pass

    def step(self, state: BicycleState, track: Track, dt: float) -> float:
        idx, e_y, psi_r, kappa_r = track.project(state.x, state.y)
        e_psi = ((state.psi - psi_r + np.pi) % (2.0 * np.pi)) - np.pi
        # LQR law (state-feedback) + feedforward for path curvature.
        delta_fb = -float(self.K[0] * e_y + self.K[1] * e_psi)
        delta_ff = float(np.arctan(self.p.L * kappa_r))  # curvature feed-forward
        return float(np.clip(delta_fb + delta_ff, -self.p.delta_max, self.p.delta_max))


# ----------------------------------------------------------- pure pursuit
class PurePursuitController:
    """Classical geometric pursuit (Coulter '92)."""

    def __init__(self, params: BicycleParams, lookahead: float = 6.0):
        self.p = params
        self.ld = float(lookahead)

    def reset(self, track: Track, state: BicycleState) -> None:
        pass

    def step(self, state: BicycleState, track: Track, dt: float) -> float:
        idx, _, _, _ = track.project(state.x, state.y)
        tx, ty = track.lookahead(idx, self.ld)
        dx = tx - state.x
        dy = ty - state.y
        # angle from vehicle heading to target
        alpha = np.arctan2(dy, dx) - state.psi
        alpha = ((alpha + np.pi) % (2.0 * np.pi)) - np.pi
        delta = np.arctan2(2.0 * self.p.L * np.sin(alpha), self.ld)
        return float(np.clip(delta, -self.p.delta_max, self.p.delta_max))


# ----------------------------------------------------------------- MPC
class MPCController:
    """Single-shooting nonlinear MPC over a finite horizon."""

    def __init__(self, params: BicycleParams,
                  horizon: int = 8, dt: float = 0.1,
                  q_y: float = 1.0, q_psi: float = 0.5, r: float = 0.05):
        self.p = params
        self.N = int(horizon)
        self.dt = float(dt)
        self.q_y, self.q_psi, self.r = float(q_y), float(q_psi), float(r)
        self._u_prev = np.zeros(self.N)

    def reset(self, track: Track, state: BicycleState) -> None:
        self._u_prev = np.zeros(self.N)

    def _rollout(self, u_seq: np.ndarray, state: BicycleState, track: Track
                  ) -> float:
        cost = 0.0
        s = state
        for k in range(self.N):
            delta = float(np.clip(u_seq[k], -self.p.delta_max, self.p.delta_max))
            # one Euler step (fast; MPC has many evaluations)
            s_new = BicycleState(
                x=s.x + self.dt * s.v * np.cos(s.psi),
                y=s.y + self.dt * s.v * np.sin(s.psi),
                psi=((s.psi + self.dt * s.v / self.p.L * np.tan(delta) + np.pi)
                      % (2.0 * np.pi)) - np.pi,
                v=s.v,
            )
            idx, e_y, psi_r, _ = track.project(s_new.x, s_new.y)
            e_psi = ((s_new.psi - psi_r + np.pi) % (2.0 * np.pi)) - np.pi
            cost += self.q_y * e_y ** 2 + self.q_psi * e_psi ** 2 + self.r * delta ** 2
            s = s_new
        return cost

    def step(self, state: BicycleState, track: Track, dt: float) -> float:
        # Warm-start: shift previous solution.
        u0 = np.concatenate([self._u_prev[1:], [self._u_prev[-1]]])
        from scipy.optimize import minimize
        bounds = [(-self.p.delta_max, self.p.delta_max)] * self.N
        res = minimize(self._rollout, u0, args=(state, track),
                        method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 20})
        self._u_prev = res.x
        return float(np.clip(res.x[0], -self.p.delta_max, self.p.delta_max))


# ----------------------------------------------------------------- HSIKAN
class HSIKANPolicy(nn.Module):
    """Windowed σ-cycle policy: state[K-history] → δ.

    Input features per step::

        [e_y, e_psi, kappa_ref]   (3 scalar channels)

    Lifted to a (C, 4) multivector per step (scalar slot of channel 0
    = the 3 features concatenated and projected).  σ_t = sign(e_y_t)
    — the natural signed-cycle quantity (which side of the path).
    The σ-cycle product over the window captures "consistently on one
    side" (oscillating left/right has product +1 or alternating;
    sustained drift has consistent sign).
    """

    def __init__(self, window: int = 8, n_channels: int = 4, delta_max: float = 0.6):
        super().__init__()
        self.window = int(window)
        self.n_channels = int(n_channels)
        self.delta_max = float(delta_max)
        # Lift 3 scalar features → n_channels Clifford scalars
        self.lift = nn.Linear(3, n_channels, bias=False)
        self.lift.weight.data.normal_(0.0, 0.1)
        self.block = HSiKANSeqWindow(K=self.window, n_channels=n_channels)
        self.head = nn.Linear(n_channels * CL2_DIM, 1)
        self.head.weight.data.normal_(0.0, 0.05)
        self.head.bias.data.zero_()

    def forward(self, feats: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # feats: (B, W, 3)  sigma: (B, W)
        per_channel = self.lift(feats)                # (B, W, C)
        mv = torch.zeros(per_channel.shape[0], per_channel.shape[1],
                          self.n_channels, CL2_DIM,
                          dtype=feats.dtype, device=feats.device)
        mv[..., 0] = per_channel
        out = self.block(mv, sigma)                   # (B, W, C, 4)
        last = out[:, -1].reshape(out.shape[0], -1)   # (B, C*4)
        delta = torch.tanh(self.head(last).squeeze(-1)) * self.delta_max
        return delta


class HSIKANController:
    """HSIKAN learned controller.

    Stores a fixed-length history of (e_y, e_psi, kappa_ref) and
    queries the policy at each step.  Training is delegated to the
    benchmark harness via :func:`imitation_train_hsikan`.
    """

    def __init__(self, params: BicycleParams, policy: Optional[HSIKANPolicy] = None,
                  window: int = 8):
        self.p = params
        self.window = int(window)
        self.policy = policy or HSIKANPolicy(window=window, delta_max=params.delta_max)
        self.history: Deque = deque(maxlen=self.window)
        self.sigma_history: Deque = deque(maxlen=self.window)

    def reset(self, track: Track, state: BicycleState) -> None:
        self.history.clear()
        self.sigma_history.clear()
        # prefill with zeros so the window is full at step 0
        for _ in range(self.window):
            self.history.append((0.0, 0.0, 0.0))
            self.sigma_history.append(0.0)

    def step(self, state: BicycleState, track: Track, dt: float) -> float:
        idx, e_y, psi_r, kappa_r = track.project(state.x, state.y)
        e_psi = ((state.psi - psi_r + np.pi) % (2.0 * np.pi)) - np.pi
        self.history.append((e_y, e_psi, kappa_r))
        self.sigma_history.append(float(np.sign(e_y)))
        feats = torch.tensor(list(self.history), dtype=torch.float32).unsqueeze(0)
        sigma = torch.tensor(list(self.sigma_history), dtype=torch.float32).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            delta = float(self.policy(feats, sigma).item())
        return float(np.clip(delta, -self.p.delta_max, self.p.delta_max))


__all__ = [
    "LQRController", "PurePursuitController", "MPCController",
    "HSIKANPolicy", "HSIKANController",
]
