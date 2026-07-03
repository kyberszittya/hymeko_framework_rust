"""Structured control — the ``u = −Kx`` leg (Phase 2 of Kato's isomorphic-controllers program).

A controller topology H constrains a state-feedback gain K to a *sparsity pattern*: ``K_ij`` may be non-zero
only if ``i == j`` or ``(i, j) ∈ H`` — a distributed controller whose communication graph is H. For a networked
linear plant ``ẋ = A x + B u`` whose coupling is a plant graph, we synthesise the structured-LQR gain per
topology and ask which topology controls the plant most *efficiently*.

The honest metric is **suboptimality** ``ρ = J(K_H) / J*`` (``J*`` = the unconstrained LQR optimum), not raw
cost: a denser H is strictly less constrained, so cost is monotone in edge count and "lowest cost" is trivially
the complete topology. The result is that the *matched* topology reaches ``ρ ≈ 1`` at minimal sparsity while a
mismatched topology of equal density pays a penalty (or fails to stabilise). See
``docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/phase2-structured-control.md``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.control.topology_zoo import TOPOLOGIES


def make_plant(plant_hg: HypergraphState, *, a: float = 1.0, eps: float = 0.5,
               ) -> tuple[np.ndarray, np.ndarray]:
    """Networked plant ``ẋ = A x + B u`` with coupling from ``plant_hg``: ``A = −a·I + ε·Ŝ`` (Ŝ = degree-
    normalised signed adjacency), ``B = I``. # Preconditions ``a > ε`` ⇒ A Hurwitz (so K=0 stabilises any
    topology). # Postconditions ``A`` Hurwitz; ``A, B`` are ``(N, N)``."""
    a_pos, a_neg = plant_hg.dense_signed_adj()
    s = np.asarray(a_pos - a_neg, dtype=float)             # signed adjacency (symmetric)
    spectral = float(np.abs(np.linalg.eigvals(s)).max()) or 1.0   # ‖Ŝ‖=1 for every topology (consistent)
    n = plant_hg.n_vertices
    big_a = -a * np.eye(n) + eps * (s / spectral)
    if float(np.linalg.eigvals(big_a).real.max()) >= 0.0:  # invariant: open-loop stable
        raise ValueError("plant not Hurwitz; lower eps or raise a")
    return big_a, np.eye(n)


def mask_from_topology(hg: HypergraphState) -> np.ndarray:
    """The ``(N, N)`` 0/1 gain mask: diagonal + an entry wherever H couples two nodes (both arc directions are
    already present in ``hg.edges``)."""
    n = hg.n_vertices
    mask = np.eye(n)
    for i, j in hg.edges:
        mask[int(i), int(j)] = 1.0
    return (mask > 0).astype(float)


def _is_hurwitz(acl: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.linalg.eigvals(acl).real.max() < -tol)


def lqr_cost(big_a: np.ndarray, big_b: np.ndarray, k: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """LQR cost ``J(K) = tr(P)`` (``E[x₀x₀ᵀ]=I``) with ``(A−BK)ᵀP + P(A−BK) + (Q + KᵀRK) = 0``; ``inf`` if the
    closed loop is not Hurwitz. # Postconditions finite ⇒ ``A−BK`` Hurwitz and ``P`` PSD."""
    acl = big_a - big_b @ k
    if not _is_hurwitz(acl):
        return math.inf
    qk = q + k.T @ r @ k
    p = solve_continuous_lyapunov(acl.T, -qk)
    return float(np.trace(p))


def unconstrained_lqr(big_a: np.ndarray, big_b: np.ndarray, q: np.ndarray, r: np.ndarray,
                      ) -> tuple[np.ndarray, float]:
    """The unconstrained LQR optimum (= the complete-topology lower bound): ``K* = R⁻¹BᵀP``, ``J* = tr(P)`` with
    ``P`` from the algebraic Riccati equation."""
    p = solve_continuous_are(big_a, big_b, q, r)
    k = np.linalg.solve(r, big_b.T @ p)
    return k, float(np.trace(p))


def structured_lqr(big_a: np.ndarray, big_b: np.ndarray, mask: np.ndarray, q: np.ndarray, r: np.ndarray, *,
                   iters: int = 800, alpha0: float = 0.05, tol: float = 1e-7) -> tuple[np.ndarray, float, bool]:
    """Projected-gradient structured LQR: minimise ``J(K)`` over ``K ⊙ mask = K`` from ``K₀ = 0`` (valid since A
    is Hurwitz). Analytic gradient ``∇J = 2(RK − BᵀP)Σ`` with backtracking on the step (shrink if the proposal
    is unstable or worse). # Postconditions returns ``(K, J(K), converged)`` with ``K`` respecting the mask."""
    n = big_a.shape[0]
    # Warm-start from the masked unconstrained optimum if it stabilises (far better than K=0 near the stability
    # boundary, where descending from 0 underflows the line search). Fall back to K=0 otherwise.
    k = np.zeros((n, n))
    try:
        k_star, _ = unconstrained_lqr(big_a, big_b, q, r)
        k_masked = k_star * mask
        if _is_hurwitz(big_a - big_b @ k_masked):
            k = k_masked
    except (np.linalg.LinAlgError, ValueError):
        pass
    cost = lqr_cost(big_a, big_b, k, q, r)
    alpha, converged = alpha0, False
    for _ in range(iters):
        acl = big_a - big_b @ k
        p = solve_continuous_lyapunov(acl.T, -(q + k.T @ r @ k))
        sigma = solve_continuous_lyapunov(acl, -np.eye(n))
        grad = 2.0 * (r @ k - big_b.T @ p) @ sigma
        grad *= mask
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            converged = True
            break
        while alpha > 1e-8:                                # backtracking: stay stable and decrease cost
            cand = k - alpha * grad
            cand_cost = lqr_cost(big_a, big_b, cand, q, r)
            if cand_cost < cost - 1e-12:
                k, cost, alpha = cand, cand_cost, min(alpha * 1.3, alpha0)
                break
            alpha *= 0.5
        else:
            converged = True                              # step underflow ⇒ at a (local) stationary point
            break
    return k, cost, converged


def run_structured_map(*, names: "list[str] | None" = None, n_nodes: int = 9, a: float = 1.0, eps: float = 0.5,
                       graph_seed: int = 0, iters: int = 800) -> dict[str, object]:
    """The plant×controller **suboptimality** map ``ρ = J(K_H)/J*`` (≥ 1). # Postconditions
    ``rho[plant][controller]``; ``best_efficiency[plant]`` = the controller with the lowest ρ·(edges) trade
    among the *sparse* (non-complete) topologies."""
    names = names or list(TOPOLOGIES)
    hgs = {nm: TOPOLOGIES[nm](n_nodes, seed=graph_seed) for nm in names}
    masks = {nm: mask_from_topology(hgs[nm]) for nm in names}
    edges = {nm: int(hgs[nm].edges.shape[0]) // 2 for nm in names}
    q, r = np.eye(n_nodes), np.eye(n_nodes)
    rho: dict[str, dict[str, float]] = {}
    jstar: dict[str, float] = {}
    for plant in names:
        big_a, big_b = make_plant(hgs[plant], a=a, eps=eps)
        _, j_opt = unconstrained_lqr(big_a, big_b, q, r)
        jstar[plant] = round(j_opt, 5)
        rho[plant] = {}
        for ctrl in names:
            _, j_h, _ = structured_lqr(big_a, big_b, masks[ctrl], q, r, iters=iters)
            rho[plant][ctrl] = round(j_h / j_opt, 4) if math.isfinite(j_h) else math.inf
    sparse = [nm for nm in names if nm != "complete"]
    best_sparse = {plant: min(sparse, key=lambda c: rho[plant][c]) for plant in names}
    # matched is best among sparse, within a tie tolerance (coverage supersets also reach rho~1.0)
    matched_is_best = {plant: rho[plant][plant] <= rho[plant][best_sparse[plant]] + 5e-3
                       for plant in sparse}
    # topology sensitivity: how much worse the WORST sparse controller is (>0 ⇒ topology matters at all)
    worst_penalty = {plant: round(max(rho[plant][c] for c in sparse) - 1.0, 4) for plant in names}
    return dict(n_nodes=n_nodes, names=names, edges=edges, jstar=jstar, rho=rho,
                best_sparse=best_sparse, matched_is_best=matched_is_best, worst_penalty=worst_penalty)


def plot_structured_map(report: dict[str, object], out_path: str | Path) -> Path:
    """Suboptimality ρ heat-map (rows = plant, cols = controller; 1.0 = optimal) — Phase 2 figure (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = report["names"]    # type: ignore[index]
    rho = report["rho"]        # type: ignore[index]
    grid = np.array([[min(rho[p][c], 10.0) if math.isfinite(rho[p][c]) else 10.0 for c in names]
                     for p in names])
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.0 * len(names) + 1.5))
    im = ax.imshow(grid, cmap="magma_r", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("controller topology (gain sparsity)")
    ax.set_ylabel("plant topology (coupling)")
    ax.set_title("Structured-LQR suboptimality ρ = J(K_H)/J*  (1.0 = optimal; lower better)")
    for i, p in enumerate(names):
        for j, c in enumerate(names):
            val = rho[p][c]
            txt = "∞" if not math.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 5.0 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ (capped at 10)")
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="structured-control topology map (Phase 2)")
    ap.add_argument("--n-nodes", type=int, default=9)
    ap.add_argument("--eps", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--out-dir", default="reports/structured_control")
    a = ap.parse_args(argv)
    report = run_structured_map(n_nodes=a.n_nodes, eps=a.eps, iters=a.iters)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structured_map.json").write_text(json.dumps(report, indent=2, default=str))
    plot_structured_map(report, out / "structured_map")
    print(json.dumps({k: report[k] for k in ("best_sparse", "matched_is_best", "worst_penalty")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
