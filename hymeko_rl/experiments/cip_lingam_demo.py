"""CIP / DirectLiNGAM diagnostic demonstrator — run the diagnostic layer end-to-end and emit the §9 artifacts.

Modes (one file, one ``--mode`` flag — §6.5 #13):

* ``synthetic`` (default, runnable now) — sample a **known** linear non-Gaussian SEM shaped like the imitation
  failure chain (``approach_error → both_contact → dist_reduction → delivery``) plus an isolated
  ``reward_progress_disagreement`` (a reward that does not drive delivery = the reward-farming candidate). Run
  :class:`CausalDiagnosis`, then emit the three forms §9 requires: numerical (JSON + true-vs-recovered table),
  plotted (discovered DAG, true-vs-recovered adjacency heatmap). This validates the whole pipeline against
  ground truth — a bug in the measure or the orchestrator shows up as a mis-recovered edge.
* ``coin`` (Phase 2, gated) — build the frame from real coin-toss rollouts of a cached policy, then declare the
  discovered graph as ``.hymeko`` and cross-view verify. Gated behind the Phase-1 report per the plan; exits with
  a pointer until then (running real rollouts is a separate, user-gated step).

Doctrine: the discovered structure is a **proposal**; controlled ablations decide. The demo prints and stores
that disclaimer alongside the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from hymeko_rl.eval.causal import (  # noqa: E402
    CausalDiagnosis,
    DirectLiNGAM,
    RolloutFrame,
    sample_linear_sem,
)
from hymeko_rl.eval.evaluate import experiment_dir  # noqa: E402

# The synthetic ground-truth scenario. Indices: 0 approach_error, 1 both_contact, 2 dist_reduction, 3 delivery,
# 4 reward_progress_disagreement (isolated: a reward signal that does NOT cause delivery — the farming candidate).
_NAMES = ["approach_error", "both_contact", "dist_reduction", "delivery", "reward_progress_disagreement"]
_TRUE_EDGES: list[tuple[int, int, float]] = [(0, 1, -0.8), (1, 2, 0.7), (2, 3, 0.9)]


def synthetic_frame(seed: int, n: int) -> tuple[RolloutFrame, np.ndarray]:
    """Sample the ground-truth SEM and wrap it as a :class:`RolloutFrame` (with a method stratum)."""
    x, b_true = sample_linear_sem(_TRUE_EDGES, len(_NAMES), n, seed, noise="uniform")
    x[:, 4] = np.abs(x[:, 4])                       # disagreement magnitude is non-negative
    continuous = {name: x[:, i] for i, name in enumerate(_NAMES)}
    missing = {name: np.zeros(n, dtype=bool) for name in _NAMES}
    method = [["bc", "dagger", "td3_bc"][i % 3] for i in range(n)]
    frame = RolloutFrame(continuous=continuous, categorical={"method": method}, missing=missing, n=n)
    return frame, b_true


def _edge_set(adjacency: np.ndarray, names: list[str]) -> set[tuple[str, str, int]]:
    """Signed edge support ``{(cause, effect, sign)}`` of an adjacency matrix (``B[effect, cause]``)."""
    return {(names[j], names[i], int(np.sign(adjacency[i, j])))
            for i in range(len(names)) for j in range(len(names)) if adjacency[i, j] != 0.0}


def _true_adjacency() -> np.ndarray:
    b = np.zeros((len(_NAMES), len(_NAMES)))
    for cause, effect, w in _TRUE_EDGES:
        b[effect, cause] = w
    return b


def render_dag(order: list[int], adjacency: np.ndarray, names: list[str], out: Path, title: str) -> None:
    """Draw the discovered DAG left-to-right by causal order (graphviz absent on host → matplotlib layout)."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    pos = {node: (rank, 0.0 if rank % 2 == 0 else 0.9) for rank, node in enumerate(order)}
    for node, (px, py) in pos.items():
        ax.scatter([px], [py], s=2600, c="#e8eef7", edgecolors="#31507a", zorder=3, linewidths=1.6)
        ax.text(px, py, names[node].replace("_", "\n"), ha="center", va="center", fontsize=8.5, zorder=4)
    for i in range(len(names)):
        for j in range(len(names)):
            w = adjacency[i, j]
            if w == 0.0:
                continue
            color = "#c0392b" if w < 0 else "#1f6f43"
            arrow = FancyArrowPatch(pos[j], pos[i], arrowstyle="-|>", mutation_scale=16,
                                    connectionstyle="arc3,rad=0.18", lw=1.0 + 3.2 * abs(w),
                                    color=color, alpha=0.85, zorder=2)
            ax.add_patch(arrow)
            mx, my = (pos[j][0] + pos[i][0]) / 2, (pos[j][1] + pos[i][1]) / 2 + 0.16
            ax.text(mx, my, f"{w:+.2f}", ha="center", fontsize=7.5, color=color, zorder=5)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(-0.6, 1.6)
    ax.axis("off")
    ax.set_title(f"{title}\n(cause → effect; green +, red −; PROPOSED structure, not proof)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def render_adjacency_compare(true_b: np.ndarray, rec_b: np.ndarray, names: list[str], out: Path) -> None:
    """Side-by-side heatmaps of the true vs recovered adjacency (rows = effect, cols = cause)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    vmax = float(max(np.abs(true_b).max(), np.abs(rec_b).max(), 1e-6))
    for ax, mat, ttl in ((axes[0], true_b, "true B"), (axes[1], rec_b, "recovered B")):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("cause")
        ax.set_ylabel("effect")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("DirectLiNGAM adjacency — ground truth vs recovered", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run_synthetic(seed: int, n: int, out_base: str) -> Path:
    """Full synthetic demonstrator: diagnose, compare to ground truth, emit JSON + figures. Returns the run dir."""
    out_dir = experiment_dir(out_base, "cip_lingam_synthetic")
    print(f"[cip-lingam] synthetic demo: n={n}, seed={seed} -> {out_dir}", flush=True)

    frame, true_b = synthetic_frame(seed, n)
    result = DirectLiNGAM().fit(frame.continuous_matrix()[0], _NAMES)
    report = CausalDiagnosis().run(frame)

    true_edges = _edge_set(true_b, _NAMES)
    rec_edges = _edge_set(result.adjacency, _NAMES)
    recovered = sorted(true_edges & rec_edges)
    spurious = sorted(rec_edges - true_edges)
    missed = sorted(true_edges - rec_edges)
    recall = len(recovered) / max(1, len(true_edges))
    print(f"[cip-lingam] order = {report.causal_order}", flush=True)
    print(f"[cip-lingam] edge recall = {recall:.2f} ({len(recovered)}/{len(true_edges)}); "
          f"spurious = {len(spurious)}", flush=True)
    print(f"[cip-lingam] next intervention = {report.next_intervention}", flush=True)

    render_dag(result.order, result.adjacency, _NAMES, out_dir / "discovered_dag.png",
               "Discovered causal graph (synthetic ground-truth)")
    render_adjacency_compare(true_b, result.adjacency, _NAMES, out_dir / "adjacency_true_vs_recovered.png")

    summary = {
        "mode": "synthetic", "seed": seed, "n": n,
        "true_edges": [list(e) for e in sorted(true_edges)],
        "recovered_true_edges": [list(e) for e in recovered],
        "spurious_edges": [list(e) for e in spurious],
        "missed_edges": [list(e) for e in missed],
        "edge_recall": round(recall, 4),
        "diagnosis": report.as_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cip-lingam] wrote summary.json + 2 figures to {out_dir}", flush=True)
    return out_dir


def run_coin(_args: argparse.Namespace) -> None:
    """Phase-2 coin PoC — gated behind the Phase-1 report per the plan (running real rollouts is a separate step)."""
    print("[cip-lingam] mode 'coin' is Phase 2 (gated behind the Phase-1 report). It builds the frame from real "
          "coin-toss rollouts of a cached policy, then declares the discovered graph as .hymeko and cross-view "
          "verifies. Not run automatically; see docs/plans/2026-07-07-cip-directlingam-diagnostic/plan.pdf §Build "
          "order step 6.", flush=True)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CIP / DirectLiNGAM diagnostic demonstrator")
    parser.add_argument("--mode", choices=["synthetic", "coin"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=600, help="number of synthetic rollouts")
    parser.add_argument("--out", type=str, default="reports/figures", help="base dir for the timestamped run dir")
    args = parser.parse_args(argv)
    if args.mode == "synthetic":
        run_synthetic(args.seed, args.n, args.out)
    else:
        run_coin(args)


if __name__ == "__main__":
    main(sys.argv[1:])
