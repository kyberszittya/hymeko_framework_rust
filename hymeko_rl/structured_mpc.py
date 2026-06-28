"""Structured MPC — model-predictive control over the matched topology (Phase 2b of Kato's program).

MPC is *model*-predictive, so the "isomorphic controller from H" is the MPC whose internal prediction model has
coupling H. Under **input saturation** (where MPC ≠ LQR), a *matched* model predicts the true plant accurately
and controls well; a *mismatched* model's errors compound over the horizon and are amplified by the constraint.
The question: does topology-match matter *more* for constrained MPC than it did for unconstrained LQR (Phase 2,
where it was a marginal 2–5%)?

Reuses ``topology_zoo`` and ``structured_control.make_plant`` / ``unconstrained_lqr`` (the LQR baseline + the
oracle). The existing ``signedkan_wip/src/control`` MPCController is bicycle-nonlinear-specific (single input) —
not reusable for this networked linear MPC; built fresh, not duplicated. See
``docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/phase2b-structured-mpc.md``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import minimize

from hymeko_rl.structured_control import make_plant, mask_from_topology, unconstrained_lqr
from hymeko_rl.topology_zoo import TOPOLOGIES


def discretize(big_a: np.ndarray, big_b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation ``A_d = e^{A·dt}``, ``B_d = A⁻¹(A_d − I)B`` (A Hurwitz ⇒ invertible).
    # Postconditions ``A_d`` is the discrete transition; ``(A_d, B_d)`` shape ``(N, N)``."""
    a_d = expm(big_a * dt)
    b_d = np.linalg.solve(big_a, (a_d - np.eye(big_a.shape[0])) @ big_b)
    return a_d, b_d


class StructuredMPC:
    """Condensed finite-horizon MPC predicting with a (possibly mismatched) model ``(A_d, B_d)``; box input
    constraint ``|u| ≤ u_max``; terminal cost = the model's own discrete-LQR ``P_f`` (dual-mode MPC).

    # Preconditions ``(A_d, B_d)`` stabilisable; ``horizon ≥ 1``. # Invariants the returned control respects the
    box, so the closed loop never applies an infeasible input."""

    def __init__(self, a_d: np.ndarray, b_d: np.ndarray, q: np.ndarray, r: np.ndarray, *, horizon: int = 12,
                 u_max: float = math.inf) -> None:
        self.nx, self.nu = b_d.shape
        self.u_max, self.horizon = u_max, horizon
        p_f = solve_discrete_are(a_d, b_d, q, r)                       # terminal cost from the MODEL
        # Condensed prediction X = Φ x0 + Γ U over k=1..N.
        phi = np.zeros((horizon * self.nx, self.nx))
        gamma = np.zeros((horizon * self.nx, horizon * self.nu))
        a_pow = np.eye(self.nx)
        for k in range(horizon):
            a_pow = a_pow @ a_d                                        # = A_d^{k+1}
            phi[k * self.nx:(k + 1) * self.nx] = a_pow
            acc = b_d.copy()
            for j in range(k, -1, -1):                                # Γ[k, j] = A_d^{k-j} B_d
                gamma[k * self.nx:(k + 1) * self.nx, j * self.nu:(j + 1) * self.nu] = acc
                acc = a_d @ acc
        q_bar = np.kron(np.eye(horizon), q)
        q_bar[(horizon - 1) * self.nx:, (horizon - 1) * self.nx:] = p_f   # terminal block
        r_bar = np.kron(np.eye(horizon), r)
        self._hq = gamma.T @ q_bar @ gamma + r_bar
        self._lin = 2.0 * gamma.T @ q_bar @ phi                       # f = self._lin @ x0
        self._u = np.zeros(horizon * self.nu)                         # warm-start across steps

    def control(self, x: np.ndarray) -> np.ndarray:
        """Solve the box-QP ``min Uᵀ Hq U + (lin·x)ᵀ U s.t. |U|≤u_max`` and return the first input ``u₀``."""
        f = self._lin @ x
        if math.isinf(self.u_max):
            u = np.linalg.solve(2.0 * self._hq, -f)                   # unconstrained ⇒ MPC = LQR
        else:
            bounds = [(-self.u_max, self.u_max)] * self._u.size
            res = minimize(lambda u: float(u @ self._hq @ u + f @ u), self._u,
                           jac=lambda u: 2.0 * self._hq @ u + f, method="L-BFGS-B", bounds=bounds)
            u = res.x
            self._u = u                                               # warm-start next step
        return u[:self.nu]


def simulate_closed_loop(a_true: np.ndarray, b_true: np.ndarray, controller, x0: np.ndarray, *, steps: int,
                         q: np.ndarray, r: np.ndarray, u_max: float = math.inf) -> float:
    """Accumulated cost ``Σ (xᵀQx + uᵀRu)`` running ``controller`` on the TRUE plant; the applied input is
    saturated to the box (so a clip-LQR baseline is treated identically to the QP-feasible MPC)."""
    x = x0.astype(float).copy()
    cost = 0.0
    for _ in range(steps):
        u = controller.control(x)
        if not math.isinf(u_max):
            u = np.clip(u, -u_max, u_max)
        cost += float(x @ q @ x + u @ r @ u)
        x = a_true @ x + b_true @ u
    return cost


class SaturatedLQR:
    """The naive baseline MPC should beat: apply the (structured) LQR gain, then clip to the box."""

    def __init__(self, k: np.ndarray) -> None:
        self.k = k

    def control(self, x: np.ndarray) -> np.ndarray:
        return -self.k @ x


def run_mpc_topology_map(*, names: "list[str] | None" = None, n_nodes: int = 9, eps: float = 0.9, dt: float = 0.1,
                         horizon: int = 12, steps: int = 60, u_max: float = 0.6, x0_scale: float = 3.0,
                         graph_seed: int = 0) -> dict[str, object]:
    """Closed-loop cost of MPC(model=H) on each true plant, plus the saturated-LQR baseline and the unconstrained
    discrete-LQR optimum ``J_d``. # Postconditions ``ratio[plant][model] = cost / J_d``; ``best_model`` per plant."""
    names = names or list(TOPOLOGIES)
    hgs = {nm: TOPOLOGIES[nm](n_nodes, seed=graph_seed) for nm in names}
    q, r = np.eye(n_nodes), np.eye(n_nodes)
    x0 = x0_scale * np.ones(n_nodes)
    cont: dict[str, dict[str, float]] = {}
    jd: dict[str, float] = {}
    sat_lqr: dict[str, float] = {}
    for plant in names:
        a_true_c, b_c = make_plant(hgs[plant], eps=eps)
        a_true, b_d = discretize(a_true_c, b_c, dt)
        p_dare = solve_discrete_are(a_true, b_d, q, r)
        jd[plant] = round(float(x0 @ p_dare @ x0), 4)                 # discrete-LQR cost-to-go from x0
        # saturated structured-LQR over the MATCHED topology (clip the masked LQR gain)
        k_star, _ = unconstrained_lqr(a_true_c, b_c, q, r)
        k_matched = k_star * mask_from_topology(hgs[plant])
        sat_lqr[plant] = round(simulate_closed_loop(a_true, b_d, SaturatedLQR(k_matched), x0,
                                                    steps=steps, q=q, r=r, u_max=u_max) / jd[plant], 4)
        cont[plant] = {}
        for model in names:
            a_model_c, _ = make_plant(hgs[model], eps=eps)
            a_model, b_model = discretize(a_model_c, b_c, dt)
            mpc = StructuredMPC(a_model, b_model, q, r, horizon=horizon, u_max=u_max)
            c = simulate_closed_loop(a_true, b_d, mpc, x0, steps=steps, q=q, r=r, u_max=u_max)
            cont[plant][model] = round(c / jd[plant], 4)
    best_model = {plant: min(names, key=lambda m: cont[plant][m]) for plant in names}
    matched_is_best = {plant: cont[plant][plant] <= cont[plant][best_model[plant]] + 1e-3 for plant in names}
    worst_penalty = {plant: round(max(cont[plant].values()) - min(cont[plant].values()), 4) for plant in names}
    return dict(n_nodes=n_nodes, names=names, u_max=u_max, jd=jd, ratio=cont, saturated_lqr_matched=sat_lqr,
                best_model=best_model, matched_is_best=matched_is_best, worst_penalty=worst_penalty)


def plot_mpc_map(report: dict[str, object], out_path: str | Path) -> Path:
    """Heat-map of MPC closed-loop cost ratio (rows = true plant, cols = MPC model topology); ★ = best model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = report["names"]      # type: ignore[index]
    ratio = report["ratio"]      # type: ignore[index]
    best = report["best_model"]  # type: ignore[index]
    grid = np.array([[min(ratio[p][c], 10.0) for c in names] for p in names])
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(grid, cmap="cividis_r", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("MPC prediction-model topology")
    ax.set_ylabel("true plant topology")
    ax.set_title("Constrained structured-MPC cost ratio (1.0=discrete-LQR optimum; lower better)")
    for i, p in enumerate(names):
        for j, c in enumerate(names):
            star = "*" if best[p] == c else ""
            ax.text(j, i, f"{ratio[p][c]:.2f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 5.0 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cost / J_d (capped at 10)")
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="structured-MPC topology-model map (Phase 2b)")
    ap.add_argument("--n-nodes", type=int, default=9)
    ap.add_argument("--u-max", type=float, default=0.6)
    ap.add_argument("--out-dir", default="reports/structured_mpc")
    a = ap.parse_args(argv)
    report = run_mpc_topology_map(n_nodes=a.n_nodes, u_max=a.u_max)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mpc_map.json").write_text(json.dumps(report, indent=2, default=str))
    plot_mpc_map(report, out / "mpc_map")
    print(json.dumps({k: report[k] for k in ("best_model", "matched_is_best", "worst_penalty",
                                             "saturated_lqr_matched")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
