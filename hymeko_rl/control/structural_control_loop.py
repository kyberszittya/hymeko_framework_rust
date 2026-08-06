"""Closed-loop control with the StructuralActor — the first real loop (Hajdu, 2026-06-28).

A networked linear plant ``x_{k+1} = A_d x_k + B_d u_k`` (the Phase-2 ``make_plant``, discretised) is regulated
from a random perturbation. The controller is the **StructuralActor** (per-node walk-holonomy control, no
message-passing), trained through a *differentiable rollout* to minimise the finite-horizon LQR cost. We compare
its closed-loop suboptimality ρ = J(actor)/J* against an MLP controller and the LQR optimum (J*), over a chain
and a Steiner topology — i.e. control *in* the design geometry.

CPU-light (linear dynamics, small N); no MuJoCo. The MuJoCo ladder (velocity→2-link→SCARA→6-DOF→quadruped, as
``.hymeko`` descriptors) is the machine-bound follow-up.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are

from hymeko_rl.control.hypergraph_designs import k_uniform_random, steiner_triple_system, sunflower
from hymeko_rl.agents.structural_actor import StructuralActor
from hymeko_rl.control.structured_control import make_plant
from hymeko_rl.control.structured_mpc import discretize
from hymeko_rl.experiments.structural_probe import build_chain_graph


def _rollout_cost(controller, a_d: torch.Tensor, b_d: torch.Tensor, x0: torch.Tensor, *, steps: int,
                  q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Mean finite-horizon LQR cost ``Σ xᵀQx + uᵀRu`` of a (differentiable) closed loop from ``x0 (batch,N)``."""
    x = x0
    cost = x.new_zeros(())
    for _ in range(steps):
        u = controller(x)                                            # (batch, N) per-node control
        cost = cost + ((x @ q) * x).sum(-1).mean() + ((u @ r) * u).sum(-1).mean()
        x = x @ a_d.T + u @ b_d.T                                    # linear dynamics (differentiable)
    return cost


class _MLPController(nn.Module):
    def __init__(self, n: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class _ActorController(nn.Module):
    """Wraps a StructuralActor so its per-node ``control`` is the module ``forward`` (trainable end-to-end)."""

    def __init__(self, actor: StructuralActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor.control(x)


class _LQRController(nn.Module):
    """The optimal linear feedback ``u = −Kx`` (fixed; the reference, no trainable parameters)."""

    k: torch.Tensor

    def __init__(self, k: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("k", k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -(x @ self.k.T)


def _train(controller: nn.Module, a_d, b_d, *, n: int, steps: int, q, r, iters: int, batch: int,
           seed: int) -> float:
    gen = torch.Generator().manual_seed(seed)
    params = list(controller.parameters())
    if iters > 0 and params:
        opt = torch.optim.Adam(params, lr=0.01)
        for _ in range(iters):
            x0 = torch.randn(batch, n, generator=gen)
            opt.zero_grad()
            _rollout_cost(controller, a_d, b_d, x0, steps=steps, q=q, r=r).backward()
            opt.step()
    with torch.no_grad():
        x0 = torch.randn(256, n, generator=torch.Generator().manual_seed(seed + 7))
        return float(_rollout_cost(controller, a_d, b_d, x0, steps=steps, q=q, r=r))


def run_closed_loop(*, hg, hidden: int = 24, steps: int = 30, iters: int = 300, batch: int = 64, seed: int = 0,
                    dt: float = 0.1, eps: float = 0.9) -> dict[str, float]:
    """Train StructuralActor + MLP controllers on the plant over ``hg``; report ρ = J/J* vs the LQR optimum."""
    a_c, b_c = make_plant(hg, eps=eps)
    a_d_np, b_d_np = discretize(a_c, b_c, dt)
    n = a_d_np.shape[0]
    a_d, b_d = torch.as_tensor(a_d_np, dtype=torch.float32), torch.as_tensor(b_d_np, dtype=torch.float32)
    q, r = torch.eye(n), torch.eye(n)
    # LQR reference: optimal finite-horizon cost ≈ the discrete-LQR rollout under the optimal gain.
    p = solve_discrete_are(a_d_np, b_d_np, np.eye(n), np.eye(n))
    k_lqr = torch.as_tensor(np.linalg.solve(np.eye(n) + b_d_np.T @ p @ b_d_np, b_d_np.T @ p @ a_d_np),
                            dtype=torch.float32)
    j_lqr = _train(_LQRController(k_lqr), a_d, b_d, n=n, steps=steps, q=q, r=r, iters=0, batch=batch, seed=seed)
    torch.manual_seed(seed)
    j_actor = _train(_ActorController(StructuralActor(hg, 1, hidden, keep=8192)), a_d, b_d, n=n, steps=steps,
                     q=q, r=r, iters=iters, batch=batch, seed=seed)
    torch.manual_seed(seed)
    j_mlp = _train(_MLPController(n, hidden), a_d, b_d, n=n, steps=steps, q=q, r=r, iters=iters, batch=batch,
                   seed=seed)
    return dict(n=n, lqr_cost=round(j_lqr, 3), actor_rho=round(j_actor / j_lqr, 3),
                mlp_rho=round(j_mlp / j_lqr, 3))


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out-dir", default="reports/closed_loop")
    a = ap.parse_args(argv)
    graphs = {"chain9": build_chain_graph(9, seed=0),
              "kuniform3": k_uniform_random(9, 3, n_blocks=8, seed=0),
              "steiner9": steiner_triple_system(9),
              "sunflower": sunflower(core_size=2, petal_size=2, n_petals=3)}
    report = {}
    for name, hg in graphs.items():
        try:
            report[name] = run_closed_loop(hg=hg, iters=a.iters)
            print(f"  [{name}] {report[name]}", flush=True)
        except Exception as exc:   # noqa: BLE001 — one plant failing must not lose the rest of the overnight
            print(f"  [{name} FAILED: {type(exc).__name__}: {exc}]", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
