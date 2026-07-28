"""R10.2 Stage 2 (Boundary 4b) — conservative structured-option TD3 campaign (3 seeds) + the learning verdict.

Runs the frozen-coordinate TD3 (``torque_path_td3``) over 3 seeds, each HOME-start, <= 400 option-episodes, deterministic
eval every 25 (no exploration), and compares the post-TD3 actor to the frozen scaffold on the frozen dev panel (a paired
3-way: scaffold / zero-actor / post-TD3; the zero-actor equals the scaffold by construction and is the consistency check).

The learning claim is NOT nominal K6 (the scaffold already has it) — it is a state-dependent improvement under perturbation:
``STRUCTURED_OPTION_TD3_IMPROVES_OVER_SCAFFOLD`` requires, in >= 2/3 seeds, post-TD3 to deliver strictly more panel K6
than the scaffold while preserving nominal HOME K6, not regressing safety, and not adding boundary-route regressions.
Otherwise only ``STRUCTURED_OPTION_TD3_NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET`` is emitted (never an RL-impossibility claim).

Training perturbations (seed 12345) are DISJOINT from the frozen dev eval panel (seed 90210, member 0 = nominal is shared
by construction). Run: ``python -m hymeko_rl.experiments.coin_kinetic_structured_option_td3`` (``--smoke`` for a fast check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import torque_path_frozen as frz
from hymeko_rl.coin_delivery.theta_option import torque_path_td3 as td3
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

OUT = Path("reports/2026-07-28-r10-structured-option-torque-path-td3")
SEEDS = (0, 1, 2)
TRAIN_PERT_SEED = 12345
EVAL_PANEL_SEED = 90210
TRAIN_N = 32
EVAL_N = 16


def _improved(scaffold: dict, post: dict) -> bool:
    """A seed improves iff post-TD3 delivers strictly more panel K6 than the scaffold while preserving nominal HOME K6,
    not regressing safety, and not adding boundary-route regressions."""
    return bool(post["k6"] > scaffold["k6"] and post["nominal_k6"]
                and post["safe"] >= scaffold["safe"] and post["boundary"] <= scaffold["boundary"])


def run(out: Path = OUT, *, smoke: bool = False) -> dict:
    if smoke:
        seeds = (0,)
        cfg = td3.TD3Config(total_episodes=40, warmup_episodes=8, critic_warmup_updates=16, eval_every=20, batch=16,
                            rank_thetas=3)
        train_n, eval_n = 8, 6
    else:
        seeds, cfg, train_n, eval_n = SEEDS, td3.TD3Config(), TRAIN_N, EVAL_N
    rig = _rig()
    train = crl.perturbation_panel(n=train_n, seed=TRAIN_PERT_SEED)
    panel = crl.perturbation_panel(n=eval_n, seed=EVAL_PANEL_SEED)
    d_norm, sigma = frz.frozen_normalization(), frz.SIGMA

    results = [td3.train_seed(rig, train, panel, s, cfg, d_norm, sigma) for s in seeds]
    per_seed = [{"seed": r["seed"], "released": r["released"], "gate": r["gate"],
                 "scaffold": r["scaffold_eval"], "post": r["post_eval"],
                 "improved": _improved(r["scaffold_eval"], r["post_eval"])} for r in results]
    n_improved = sum(p["improved"] for p in per_seed)
    verdict = ("STRUCTURED_OPTION_TD3_IMPROVES_OVER_SCAFFOLD" if n_improved >= 2
               else "STRUCTURED_OPTION_TD3_NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET")
    summary = {
        "contract": "STRUCTURED_OPTION_TD3_V1", "parent_commit": "4697d263",
        "boundary": "4b (conservative structured-option TD3) — HOME-start, sigma=0.05 frozen, D frozen",
        "measurement_contract": {"seeds": list(seeds), "episodes_per_seed": cfg.total_episodes, "eval_every": cfg.eval_every,
                                 "train_perturbations": train_n, "eval_panel": eval_n, "train_seed": TRAIN_PERT_SEED,
                                 "eval_panel_seed": EVAL_PANEL_SEED, "no_exploration_at_eval": True, "smoke": smoke},
        "exploration_freeze": frz.REVIEW_DECISION,
        "per_seed": per_seed, "n_improved": n_improved, "verdict": verdict,
        "non_claims": ["nominal K6 is NOT the claim (the scaffold already has it)",
                       "eval on the DEV panel (not a final held-out transfer claim); s4/s7 sealed",
                       "no RL-impossibility claim; NO_IMPROVEMENT is bounded to this frozen budget"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "td3.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _print(r: dict) -> None:
    for p in r["per_seed"]:
        g = p["gate"]
        print(f"seed {p['seed']}: released={p['released']} improved={p['improved']} "
              f"scaffold_k6={p['scaffold']['k6']}/{p['scaffold']['n']} post_k6={p['post']['k6']}/{p['post']['n']} "
              f"nominal={p['post']['nominal_k6']} safe={p['post']['safe']} boundary={p['post']['boundary']} "
              f"| gate_spearman={g['spearman_q_vs_reward'] if g else None}")
    print(f"n_improved={r['n_improved']} -> {r['verdict']}")


if __name__ == "__main__":
    _print(run(smoke="--smoke" in sys.argv))
