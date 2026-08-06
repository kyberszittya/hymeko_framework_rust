"""BC-only delivery measurement (the 2026-07-05 handoff's decisive test).

Localizes the 0.30 (demonstrator) -> 0.10-0.20 (post-off-policy RL) delivery loss: is the BC clone already
below the teacher (cloning/arch gap), or does the off-policy refine degrade a good clone (trainer gap)?

Matches exp_galambos_coord_ab's config exactly: collab CTDE actor (sa_hsikan, hidden=64), 200 demo episodes
(success-only), 200 BC epochs, dwell-delivery on the BASELINE env, eval seed 9000, 3 seeds. Also measures the
demonstrator anchor under the SAME DwellMetric protocol (filter criterion == grading criterion, CLAUDE.md par.3).

One-off measurement driver: all logic is reused from hymeko_rl (par.6.1); only orchestration lives here.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

sys.path.insert(0, r"d:\hakiko_ai_ws\03_implementation\hymeko_framework_rust")

import numpy as np
import torch

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, experiment_dir
from hymeko_rl.experiments.exp_galambos_coord_ab import _both_contact_rate, make_env
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos, eval_delivery
from hymeko_rl.experiments.galambos_demo import GalambosDemonstrator
from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.train.campaign import tee_stdout

DIFFICULTY = 0.3
N_DEMOS = 200
BC_EPOCHS = 200
SEEDS = (0, 1, 2)
N_EVAL = 50
EVAL_SEED = 9_000


def demonstrator_delivery(n_episodes: int, seed0: int) -> float:
    """Dwell-delivery of the scripted demonstrator under the exact eval protocol used for the policies."""
    env = make_env(coord=False, difficulty=DIFFICULTY)
    demo = GalambosDemonstrator(env)

    def act(e: Any, _obs: np.ndarray) -> np.ndarray:
        if e._step == 0:                      # episode start (env.reset zeroes _step) -> reset the phase machine
            demo.reset()
        return demo.action(e)

    dwell = int(getattr(env, "success_steps", 1))
    res = eval_metric(env, act, DwellMetric("in_zone", dwell), n_episodes=n_episodes, seed0=seed0)
    return float(sum(res)) / max(1, n_episodes)


def main() -> int:
    exp = experiment_dir("experiments", "galambos_bc_only")
    (exp / "policies").mkdir(exist_ok=True)
    (exp / "gifs").mkdir(exist_ok=True)
    with tee_stdout(exp / "run.log"):
        t0 = time.perf_counter()
        print(f"[bc-only] measuring demonstrator anchor ({N_EVAL} eps, dwell rule)...", flush=True)
        demo_rate = demonstrator_delivery(N_EVAL, EVAL_SEED)
        print(f"[bc-only] demonstrator delivery = {demo_rate:.3f}", flush=True)

        env = make_env(coord=False, difficulty=DIFFICULTY)
        print(f"[bc-only] collecting {N_DEMOS} demo episodes (success-only)...", flush=True)
        obs, acts = collect_galambos_demos(env, N_DEMOS, 0)
        print(f"[bc-only] {len(obs)} demo transitions", flush=True)

        rows: list[dict[str, float]] = []
        for s in SEEDS:
            torch.manual_seed(s)
            np.random.seed(s)
            actor, _critics = build_collaborative_offpolicy(make_env(coord=False, difficulty=DIFFICULTY),
                                                            kind="sa_hsikan", hidden=64)
            losses = behaviour_clone(actor, obs, acts, n_epochs=BC_EPOCHS, seed=s)
            deliv = eval_delivery(make_env(coord=False, difficulty=DIFFICULTY), actor, N_EVAL, EVAL_SEED)
            both = _both_contact_rate(make_env(coord=False, difficulty=DIFFICULTY), actor, 12, EVAL_SEED)
            torch.save(actor.state_dict(), exp / "policies" / f"bc_only_s{s}.pt")
            rows.append({"seed": s, "bc_loss": round(losses[-1], 5), "delivery": round(deliv, 3),
                         "both_contact": round(both, 4)})
            print(f"[bc-only] seed {s}: delivery={deliv:.3f} both_contact={both:.4f} "
                  f"bc_loss={losses[-1]:.4f}", flush=True)

        dels = sorted(r["delivery"] for r in rows)
        summary = {"demonstrator_delivery": round(demo_rate, 3), "bc_delivery_median": dels[len(dels) // 2],
                   "bc_delivery_per_seed": dels, "seeds": rows,
                   "config": {"difficulty": DIFFICULTY, "n_demos": N_DEMOS, "bc_epochs": BC_EPOCHS,
                              "n_eval": N_EVAL, "eval_seed": EVAL_SEED, "demo_transitions": int(len(obs)),
                              "actor": "collab sa_hsikan hidden=64", "metric": "DwellMetric(in_zone, success_steps)"},
                   "wall_s": round(time.perf_counter() - t0, 1)}
        (exp / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        best = max(rows, key=lambda r: r["delivery"])
        try:
            from hymeko_rl.viz.campaign_viz import render_actor_gif
            actor, _ = build_collaborative_offpolicy(make_env(coord=False, difficulty=DIFFICULTY),
                                                     kind="sa_hsikan", hidden=64)
            actor.load_state_dict(torch.load(exp / "policies" / f"bc_only_s{best['seed']}.pt",
                                             map_location="cpu"))
            render_actor_gif(make_env(coord=False, difficulty=DIFFICULTY), actor,
                             str(exp / "gifs" / f"bc_only_s{best['seed']}"), seed=EVAL_SEED)
        except Exception as exc:  # noqa: BLE001 -- viz is best-effort, must not lose the measurement
            print(f"[gif skipped: {type(exc).__name__}: {exc}]", flush=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = ["demonstrator"] + [f"BC s{r['seed']}" for r in rows]
            vals = [demo_rate] + [r["delivery"] for r in rows]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(labels, vals, color=["#4477AA"] + ["#EE6677"] * len(rows))
            ax.axhline(demo_rate, ls="--", c="#4477AA", lw=1)
            ax.set_ylabel("dwell delivery rate (50 eps)")
            ax.set_title("Galambos: BC-only clone vs scripted demonstrator")
            fig.tight_layout()
            fig.savefig(exp / "bc_vs_demo.png", dpi=150)
        except Exception as exc:  # noqa: BLE001
            print(f"[plot skipped: {type(exc).__name__}: {exc}]", flush=True)

        print("\n=== BC-ONLY VERDICT ===")
        print(json.dumps(summary, indent=2))
        print(f"artifacts -> {exp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
