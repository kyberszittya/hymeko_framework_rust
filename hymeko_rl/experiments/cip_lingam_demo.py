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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from hymeko_rl.eval.causal import (  # noqa: E402
    CausalDiagnosis,
    CausalHypergraph,
    DirectLiNGAM,
    RolloutFrame,
    cross_view_verify,
    sample_linear_sem,
)
from hymeko_rl.eval.evaluate import experiment_dir  # noqa: E402

# --- Phase-2 coin PoC constants -------------------------------------------------------------------------------
_MAX_STEPS = 300
_DIFFICULTY = 0.3
# The two cached coin-toss policies (best-on-delivery checkpoints; NO new training — provenance in the report).
_COIN_POLICIES: tuple[tuple[str, str, str], ...] = (
    ("mlp", "mlp", "experiments/2026_07_05_18_34_coin_arch_ab_mlp/policies/coin_arch_ab_mlp_s0.pt"),
    ("hsikan", "sa_hsikan", "experiments/2026_07_05_18_42_coin_arch_ab_hsikan/policies/coin_arch_ab_hsikan_s0.pt"),
)
# Continuous rollout variables supplied alongside the CIP progress_score. monitor_score is DELIBERATELY excluded
# (it is the mean of the five gating sub-scores → perfectly collinear with them, which would make the linear
# model singular). total_reward is the RL training signal — the farming/disagreement probe.
_COIN_EXTRA_CONTINUOUS = ("approach_score", "contact_score", "delivery_score", "anti_exploit_score", "total_reward")

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


def _make_coin_env() -> Any:
    """The baseline planar-grasp (coin-toss) env — the true task, no shaped-training reward surgery."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    return PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFFICULTY)


def _load_coin_actor(env: Any, kind: str, checkpoint: str) -> Any:
    """Reconstruct the cached collaborative actor and load its weights (strict — a wrong arch RAISES, never
    silently loads a mismatched policy). Returns the actor in eval mode."""
    import torch

    from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
    actor, _critics = build_collaborative_offpolicy(env, kind=kind, hidden=64)
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def _rollout_policy(arch: str, kind: str, checkpoint: str, n: int, seed0: int,
                    ) -> "tuple[list[dict[str, Any]], dict[str, list[float]], float]":
    """Roll a cached policy ``n`` seeded episodes → (verdict maps, extra continuous columns, disagreement).

    Each episode is rolled twice with the SAME seed (deterministic greedy policy → identical trajectory): once
    through :func:`record_trajectory` for the monitor verdict, once through :func:`run_episode` for the summed
    RL reward. The per-arch reward↔monitor disagreement (``1 − concordance``) is sourced from the
    :class:`RewardConsistencyMonitor` (§6.1 — read, never re-derive) and attached to every verdict map so the
    CIP-export bridge presents it as a MEASURED value, not a default.
    """
    from hymeko_rl.eval.evaluate import greedy_action_fn, run_episode
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor
    from hymeko_rl.eval.task_monitor.context import record_trajectory
    from hymeko_rl.eval.task_monitor.root import TaskMonitor

    env = _make_coin_env()
    actor = _load_coin_actor(env, kind, checkpoint)
    action_fn = greedy_action_fn(actor)
    mon = TaskMonitor.from_env(env)

    verdicts = []
    rewards: list[float] = []
    for ep in range(n):
        seed = seed0 + ep
        verdict = mon.evaluate(record_trajectory(env, action_fn, seed))
        _outcome, total_reward = run_episode(env, action_fn, seed=seed)
        verdicts.append(verdict)
        rewards.append(float(total_reward))
        print(f"[cip-coin] {arch:6s} ep {ep + 1:2d}/{n} | pass={int(verdict.monitor_pass)} "
              f"score={verdict.monitor_score:+.3f} deliv={verdict.delivery_score:+.3f} "
              f"reward={total_reward:+.2f}", flush=True)
    if hasattr(env, "close"):
        env.close()

    rows = [{"policy": f"{arch}_ep{i}", "total_reward": rewards[i], "monitor_score": verdicts[i].monitor_score}
            for i in range(n)]
    disagreement = round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 6)

    verdict_maps: list[dict[str, Any]] = []
    for v in verdicts:
        d = v.as_dict()
        d["reward_progress_disagreement"] = disagreement       # measured, not defaulted (cip_export picks it up)
        verdict_maps.append(d)
    extra: dict[str, list[float]] = {
        "approach_score": [v.approach_score for v in verdicts],
        "contact_score": [v.contact_score for v in verdicts],
        "delivery_score": [v.delivery_score for v in verdicts],
        "anti_exploit_score": [v.anti_exploit_score for v in verdicts],
        "total_reward": rewards,
    }
    print(f"[cip-coin] {arch}: reward↔monitor disagreement (1−concordance) = {disagreement:.4f}", flush=True)
    return verdict_maps, extra, disagreement


def _build_coin_frame(n: int, seed0: int) -> "tuple[RolloutFrame, dict[str, float], list[str]]":
    """Roll every cached coin policy and stack into one architecture-stratified :class:`RolloutFrame`.

    A policy whose checkpoint does not match its declared architecture is SKIPPED (logged, never faked) so the
    rest of the diagnosis still runs. Returns ``(frame, per_arch_disagreement, included_arches)``.
    """
    all_maps: list[dict[str, Any]] = []
    extra: dict[str, list[float]] = {name: [] for name in _COIN_EXTRA_CONTINUOUS}
    arch_col: list[str] = []
    disagreement: dict[str, float] = {}
    included: list[str] = []
    for arch, kind, checkpoint in _COIN_POLICIES:
        if not Path(checkpoint).exists():
            print(f"[cip-coin] SKIP {arch}: checkpoint absent ({checkpoint})", flush=True)
            continue
        try:
            maps, arch_extra, dis = _rollout_policy(arch, kind, checkpoint, n, seed0)
        except (RuntimeError, ValueError) as err:                      # arch/weight mismatch — skip, do not fake
            print(f"[cip-coin] SKIP {arch}: could not load/roll ({type(err).__name__}: {err})", flush=True)
            continue
        all_maps.extend(maps)
        for name in _COIN_EXTRA_CONTINUOUS:
            extra[name].extend(arch_extra[name])
        arch_col.extend([arch] * len(maps))
        disagreement[arch] = dis
        included.append(arch)

    if not all_maps:
        raise RuntimeError("no coin policy could be rolled out; nothing to diagnose")
    frame = RolloutFrame.from_verdicts(all_maps, extra_continuous=extra, categoricals={"architecture": arch_col})
    return frame, disagreement, included


def run_coin(args: argparse.Namespace) -> Path:
    """Phase-2 coin PoC: diagnose real coin-toss rollouts, declare each discovered DAG as ``.hymeko``, cross-view
    verify it through the HyMeKo engine, and emit the §9 artifacts. Returns the run dir."""
    out_dir = experiment_dir(args.out, "cip_lingam_coin")
    n, seed0 = int(args.n), int(args.seed) + 9_000              # 9000 = the canonical eval-seed base (§galambos)
    print(f"[cip-coin] Phase-2 coin PoC: {n} episodes/arch, seed0={seed0} -> {out_dir}", flush=True)

    frame, disagreement, included = _build_coin_frame(n, seed0)
    reports = CausalDiagnosis().run_stratified(frame, stratify_by=["architecture"])

    summary: dict[str, Any] = {
        "mode": "coin", "n_episodes_per_arch": n, "seed0": seed0, "architectures": included,
        "reward_monitor_disagreement": disagreement,
        "checkpoints": {arch: ckpt for arch, _kind, ckpt in _COIN_POLICIES if arch in included},
        "strata": {},
        "_disclaimer": "DirectLiNGAM PROPOSES structure over real rollouts; controlled ablations DECIDE. "
                       "The .hymeko cross-view proves the declared DAG == the engine's tensor view, not causality.",
    }

    for arch in included:
        label = (arch,)
        report = reports[label]
        sub = frame.subset(frame.group_by(["architecture"])[label])
        matrix, kept, dropped = sub.continuous_matrix()
        entry: dict[str, Any] = {"diagnosis": report.as_dict(), "continuous_kept": kept, "continuous_dropped": dropped}

        if len(kept) >= 2 and matrix.shape[0] > len(kept):
            result = DirectLiNGAM().fit(matrix, kept)
            render_dag(result.order, result.adjacency, kept, out_dir / f"discovered_dag_{arch}.png",
                       f"Coin causal graph — {arch} (real rollouts, N={matrix.shape[0]})")
            cg = CausalHypergraph.from_lingam(result, f"CoinCausal{arch.capitalize()}")
            xview = cross_view_verify(cg, out_dir / f"causal_{arch}.hymeko")
            entry["cross_view"] = xview.as_dict()
            print(f"[cip-coin] {arch}: order={result.ordered_names()} | cross-view agree={xview.agree} "
                  f"backend={xview.backend} hash={xview.canonical_hash[:24]}", flush=True)
        else:
            entry["cross_view"] = None
            print(f"[cip-coin] {arch}: too few varying continuous columns to fit a DAG (kept={kept}); "
                  f"diagnosis-only for this stratum", flush=True)
        summary["strata"][arch] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cip-coin] wrote summary.json + {len(included)} DAG figure(s)/.hymeko to {out_dir}", flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CIP / DirectLiNGAM diagnostic demonstrator")
    parser.add_argument("--mode", choices=["synthetic", "coin"], default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=None,
                        help="synthetic: number of rollouts (default 600); coin: episodes per architecture "
                             "(default 20)")
    parser.add_argument("--out", type=str, default="reports/figures", help="base dir for the timestamped run dir")
    args = parser.parse_args(argv)
    if args.mode == "synthetic":
        run_synthetic(args.seed, args.n if args.n is not None else 600, args.out)
    else:
        args.n = args.n if args.n is not None else 20
        run_coin(args)


if __name__ == "__main__":
    main(sys.argv[1:])
