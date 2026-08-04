r"""M2 — neural Lyapunov certificate of the centroidal capturability basin (torch, no §1: torch is pinned).

Where M1's quadratic certifies the pendulum's linear basin, the runner's basin is nonlinear, so ``V`` is an MLP:

    V_θ(x) = ‖φ_θ(x) − φ_θ(x*)‖²        (≥ 0 and V(x*) = 0 by construction; x* = (z0, 0, 0, 0))

fit with the same discrete-decrease idea as M1 (``relu(V(x⁺) − V(x))`` on non-falling rollouts), plus a term that
pushes ``V`` HIGH on falling states so the certified sublevel ``{V ≤ c}`` excludes them. Verification is by dense
sampling of the reduced state — EMPIRICAL, not a proof: it reports the fall-violation rate on ``{V ≤ c}`` and the
IoU vs the rollout-recoverable set. A formal (SMT / Lipschitz) guarantee is a later, separate step.

torch is used at the pinned ``==2.12.0`` (CORE.YAML); the version is NOT changed — using it is not a §1 edit.

# Preconditions: a stabilising ``CentroidalConfig``. # Postconditions: ``V(x*) = 0``, ``V ⪰ 0``; deterministic
#   for a fixed seed (torch + numpy seeded).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scenarios.humanoid.centroidal import (
    CentroidalConfig,
    _sample_grid,
    centroidal_rollout,
    centroidal_step,
)


def _lookahead(x0: np.ndarray, cfg: CentroidalConfig, steps: int) -> "tuple[np.ndarray, np.ndarray]":
    """Advance states ``steps`` closed-loop steps; return (x⁺, fell-during-lookahead mask). Dynamics only (no grad)."""
    state = x0.copy()
    fell = np.zeros(len(x0), dtype=bool)
    for i in range(steps):
        state = centroidal_step(state, i * cfg.dt, cfg)
        fell |= np.abs(state[:, 3]) > cfg.fall_pitch
    return state, fell


class NeuralLyapunovCertificate(nn.Module):
    """An MLP Lyapunov certificate ``V=‖φ(x)−φ(x*)‖²`` over the reduced centroidal state, with sampling verify."""

    def __init__(self, cfg: CentroidalConfig, hidden: int = 64, feat: int = 16, lookahead: int = 40,
                 seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.cfg, self.lookahead = cfg, lookahead
        # The fall dynamics are decoupled: (z, ż) is a stable SLIP gait, only (L, pitch) governs falling. So V is
        # over the fall-relevant error subspace (L, pitch) — constant on the z-bounce limit cycle (V decreases
        # genuinely, not spuriously oscillating), which the full-state version did not (it collapsed the level).
        self._errstar = torch.zeros((1, 2), dtype=torch.float32)
        self.phi = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, feat),
        )

    @staticmethod
    def _err(x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:4]                                     # (L, pitch): the converging error coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.phi(self._err(x)) - self.phi(self._errstar)   # V = ‖φ(L,pitch)−φ(0,0)‖²  ⇒  V⪰0, V(x*)=0
        return (d * d).sum(dim=1)

    def value(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.forward(torch.as_tensor(x, dtype=torch.float32)).numpy()

    def fit(self, x0: np.ndarray, iters: int = 800, lr: float = 3e-3, margin: float = 1e-3,
            sep: float = 2.0) -> "NeuralLyapunovCertificate":
        r"""Shape ``V`` to decrease on non-falling states and be high on falling ones (so ``{V≤c}`` excludes falls).

        The lookahead (dynamics, no grad) is precomputed once; only ``V`` carries gradients. # Pre: ``len(x0)>0``.
        """
        if len(x0) == 0:
            raise ValueError("fit requires at least one sample")
        xp_np, fell = _lookahead(x0, self.cfg, self.lookahead)
        x = torch.as_tensor(x0, dtype=torch.float32)
        xp = torch.as_tensor(xp_np, dtype=torch.float32)
        keep = torch.as_tensor(~fell)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            v, vp = self.forward(x), self.forward(xp)
            decrease = torch.relu(vp[keep] - v[keep] + margin).mean() if keep.any() else v.sum() * 0.0
            exclude = torch.relu(sep - v[~keep]).mean() if (~keep).any() else v.sum() * 0.0
            (decrease + exclude).backward()
            opt.step()
        return self

    def certified_level(self, x0: np.ndarray, tol: float = 1e-4) -> float:
        """Largest ``c`` such that ``{V ≤ c}`` neither increases ``V`` nor falls within the lookahead."""
        v = self.value(x0)
        xp, fell = _lookahead(x0, self.cfg, self.lookahead)
        bad = fell | (self.value(xp) - v > tol)
        return float(v[bad].min()) if bad.any() else float(v.max())

    def verify(self, cfg: CentroidalConfig | None = None) -> dict:
        """Empirical verification on a held-out sample: certified level, fall-violation rate, IoU vs recoverable."""
        cfg = cfg or self.cfg
        x0, recover = centroidal_rollout(cfg, _sample_grid(cfg, offset=0.5))   # held-out (shifted) ground truth
        v = self.value(x0)
        level = self.certified_level(x0)
        inside = v <= level
        certified = inside.astype(int)
        inter = int(np.sum((certified == 1) & (recover == 1)))
        union = int(np.sum((certified == 1) | (recover == 1)))
        return {"certified_level": level,
                "fall_violation_rate": float(np.mean(recover[inside] == 0)) if inside.any() else 0.0,
                "iou_vs_recoverable": float(inter / union) if union else 1.0,
                "n_certified": int(inside.sum()), "n": int(len(x0))}
