"""6D-0 SE(3) pose reach — measurement + figure + GIF.

The integration result for the frozen runtime on a real MuJoCo SE(3) task: measure the pose-certificate rate of
(a) the closed-loop 6-DoF DLS-IK expert (the ceiling), (b) the runtime with budget=0 (pure IK waypoint, no search),
(c) the runtime with search (budget 8 / 12) — the full StructuredState → LSTM → proposal → MultimodalBudgetSearch →
committed option → pose certificate pipeline. Emits numeric JSON + a bar figure; renders a GIF of the runtime driving
one reach (reusing render_rollout / _encode_gif — no new rendering code).
"""
import json
import os
import sys

import numpy as np

from hymeko_rl.env.se3_reach_env import SE3ReachEnv
from hymeko_rl.env.se3_reach_option import solve_reach

OUT = "reports/2026-07-24-se3-reach-6d0"


def _env():
    return SE3ReachEnv(control_mode="position", max_steps=200, reach_thresh=0.06, ang_thresh=0.35,
                       start_perturb=0.2, expert_gain=0.5)


def _run_expert(env, seed):
    env.reset(seed=seed)
    for _ in range(200):
        _o, _r, term, trunc, info = env.step(env.expert_action)
        if term and not info["death"]:
            return 1
        if term or trunc:
            return 0
    return 0


def _run_runtime(env, seed, budget):
    env.reset(seed=seed)
    prov = solve_reach(env, np.random.default_rng(500 + seed), budget=budget, horizon=160)
    for _ in range(200):
        _o, _r, term, trunc, info = env.step(prov.selected)
        if term and not info["death"]:
            return 1, prov
        if term or trunc:
            return 0, prov
    return 0, prov


def measure(n_seeds=30):
    env = _env()
    rows = {"expert": [], "runtime_b0": [], "runtime_b8": [], "runtime_b12": []}
    for s in range(n_seeds):
        rows["expert"].append(_run_expert(env, s))
        rows["runtime_b0"].append(_run_runtime(env, s, 0)[0])
        rows["runtime_b8"].append(_run_runtime(env, s, 8)[0])
        rows["runtime_b12"].append(_run_runtime(env, s, 12)[0])
    rate = {k: round(float(np.mean(v)), 3) for k, v in rows.items()}
    return rate, rows


def plot(rate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["expert\n(closed-loop IK)", "runtime b=0\n(IK waypoint)", "runtime b=8\n(+search)", "runtime b=12\n(+search)"]
    vals = [rate["expert"], rate["runtime_b0"], rate["runtime_b8"], rate["runtime_b12"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(labels, vals, color=["#444", "#37a", "#3a7", "#2a5"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("pose-certificate rate (pos ∧ ang)")
    ax.set_ylim(0, 1.05)
    ax.set_title("6D-0 SE(3) pose reach — frozen runtime vs the closed-loop IK ceiling\n"
                 "committed option = servo to an IK-proposed joint waypoint (K=1); search jitters it", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def render_gif(path, seed=13):
    """One runtime-driven reach as a GIF (reuse render_rollout + _encode_gif). Best-effort: needs an offscreen GL
    context; on failure, prints the reason and skips (the numeric result + plot stand on their own)."""
    try:
        from pathlib import Path

        from hymeko_rl.viz.render_reach import _encode_gif, render_rollout
        # A harder start (larger perturbation) so the arm's swing to the pose is visibly multi-frame; the committed
        # option is still one IK waypoint held under position control.
        env = SE3ReachEnv(control_mode="position", max_steps=160, reach_thresh=0.06, ang_thresh=0.35,
                          start_perturb=0.5, expert_gain=0.5)
        env.reset(seed=seed)
        prov = solve_reach(env, np.random.default_rng(500 + seed), budget=12, horizon=160)
        q_des = prov.selected

        def action_fn(_env, _obs):
            return q_des
        # 640×480: the default MuJoCo offscreen framebuffer is 640 wide (a larger one needs <global offwidth> in the
        # model XML). Watchable; below the §9 slide preference — noted in the report.
        frames = render_rollout(env, action_fn, seed=seed, height=480, width=640, max_frames=160)
        _encode_gif(frames, Path(path), fps=30)
        print(f"wrote {path} ({len(frames)} frames)")
        return True
    except Exception as e:  # noqa: BLE001 — GL context / renderer availability is environment-dependent
        print(f"GIF skipped ({type(e).__name__}: {e})")
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    rate, rows = measure(30)
    print("pose-certificate rate:", rate)
    manifest = {"contract": "SE3_POSE_REACH_6D0", "date": "2026-07-24",
                "runtime": "OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1 (frozen)",
                "task": "SE(3) pose reach (position + orientation), 4-DOF arm, position control",
                "certificate": "pos_err < 0.06 AND ang_err < 0.35 rad", "n_seeds": 30,
                "difficulty": {"start_perturb": 0.2, "note": "closable basic pose reach (reachable-by-FK targets)"},
                "pose_certificate_rate": rate, "per_seed": rows}
    json.dump(manifest, open(f"{OUT}/se3_reach_6d0.json", "w"), indent=1)
    plot(rate, f"{OUT}/se3_reach_6d0.png")
    render_gif(f"{OUT}/se3_reach_6d0.gif")
    print(f"artifact: {OUT}/se3_reach_6d0.json\nSE3_REACH_6D0_DONE")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
