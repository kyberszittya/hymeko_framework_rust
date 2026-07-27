"""Residual SAC over the certified PD-hold-q0 scaffold — extend the certified envelope.

The position-servo env makes ``a = 0`` the PD-hold-``q0`` controller, a *certified*
balance baseline (passes the unchanged Lyapunov certificate for pitch-rate ≤ ~0.3).
Beyond that the scaffold SURVIVES but overshoots (V_max > 0.055) → fails the
certificate. This trains a **bounded residual** (coin-R8 regime) on a HARDER
perturbation envelope and asks: does the learned residual raise the *certified*
pass-rate above the PD-hold scaffold on the same envelope?

Reward = alive − 2·V − control cost (V = COM Lyapunov). The reward-independent
``lyapunov_certificate`` is the eval-only gate. Baseline = a ≡ 0 (PD-hold), evaluated
on the identical seeds so the delta is the pure RL value-add.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.run_humanoid_sac [--steps N]
SIMULATION. Live [sac] progress every log_every steps (§3 never-run-blind).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .balance_env import BalanceConfig, HumanoidBalanceEnv
from .lyapunov import evaluate_lyapunov

_OUT = Path("reports/2026-07-27-humanoid-sac-residual")
_TRAIN = BalanceConfig(perturb_lo=0.4, perturb_hi=0.8)     # harder than PD-hold certified ~0.3


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return actor.action_mean(t).squeeze(0).cpu().numpy()


def _eval_policy(env, act_fn, seeds) -> tuple[float, float]:
    """Return (mean upright fraction, Lyapunov certificate pass rate) for a policy."""
    fracs, lyap_pass = [], 0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, up, steps, vs = False, 0, 0, []
        while not done:
            obs, _r, term, trunc, info = env.step(act_fn(obs))
            steps += 1
            vs.append(info["V"])
            if info["upright"]:
                up = steps
            done = term or trunc
        fracs.append(up / env.max_steps)
        lyap_pass += int(evaluate_lyapunov(vs)["passes"])
    return float(np.mean(fracs)), lyap_pass / len(seeds)


def _eval_balance(env, actor, seeds) -> tuple[float, float]:
    """SAC greedy-policy evaluation (upright fraction, certificate pass rate)."""
    return _eval_policy(env, lambda o: _greedy(actor, o), seeds)


def _eval_pd_hold(env, seeds) -> tuple[float, float]:
    """PD-hold scaffold baseline (a = 0) on the same seeds."""
    zero = np.zeros(env.model.nu)
    return _eval_policy(env, lambda _o: zero, seeds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    env = HumanoidBalanceEnv(cfg=_TRAIN, seed=0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    torch.manual_seed(0)
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                               action_dim=act_dim, action_scale=1.0, hidden=128)

    val_seeds = list(range(2000, 2008))                          # validation: checkpoint selection
    test_seeds = list(range(3000, 3012))                         # held-out: reported ONCE
    base_frac, base_rate = _eval_pd_hold(env, test_seeds)        # certified scaffold baseline (test)

    best_path = _OUT / "humanoid_sac_residual_best.pt"
    best = {"rate": -1.0}                                        # closure capture (not global state)

    def eval_fn(e, a) -> float:
        rate = _eval_balance(e, a, val_seeds)[1]                 # curve/selection = certified rate on VAL
        if rate > best["rate"]:                                  # keep the best-validation checkpoint
            best["rate"] = rate
            torch.save(a.state_dict(), best_path)
        return rate

    # ANNEAL alpha (init 0.1 -> 0.005): AUTO collapses entropy -> certified-rate instability (measured).
    cfg = SACConfig(total_steps=args.steps, start_steps=2_000, batch_size=256,
                    eval_every=10_000, log_every=5_000, seed=0,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    final_frac, final_rate = _eval_balance(env, actor, test_seeds)   # naive last checkpoint (test)
    best_actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                              action_dim=act_dim, action_scale=1.0, hidden=128)
    best_actor.load_state_dict(torch.load(best_path))
    best_frac, best_rate = _eval_balance(env, best_actor, test_seeds)  # best-val checkpoint (test)
    delta = round(best_rate - base_rate, 3)
    result = {
        "verdict": ("RESIDUAL_EXTENDS_CERTIFIED_ENVELOPE" if delta > 0.15
                    else "RESIDUAL_MATCHES_SCAFFOLD" if abs(delta) <= 0.15
                    else "RESIDUAL_REGRESSES_SCAFFOLD"),
        "envelope_perturb": [_TRAIN.perturb_lo, _TRAIN.perturb_hi],
        "pd_hold_certified_rate_test": round(base_rate, 3),
        "sac_best_val_certified_rate_val": round(best["rate"], 3),
        "sac_best_val_certified_rate_test": round(best_rate, 3),
        "sac_last_certified_rate_test": round(final_rate, 3),
        "certified_rate_delta_test": delta,
        "pd_hold_upright_fraction_test": round(base_frac, 3),
        "sac_best_upright_fraction_test": round(best_frac, 3),
        "eval_curve_certified_rate_val": [round(c, 3) for c in curve],
        "total_steps": args.steps,
        "note": "SIMULATION. Bounded residual over the CERTIFIED PD-hold-q0 scaffold (a=0). Lyapunov "
                "reward; reward-INDEPENDENT lyapunov_certificate is the gate. Baseline = PD-hold on the "
                "identical test seeds. VAL seeds select the checkpoint; TEST seeds reported once. "
                "sac_last vs sac_best exposes training (in)stability under the peak-V certificate.",
    }
    (_OUT / "sac_residual_gates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("verdict", "pd_hold_certified_rate_test", "sac_best_val_certified_rate_test",
                       "sac_last_certified_rate_test", "certified_rate_delta_test",
                       "eval_curve_certified_rate_val")}, indent=2))


if __name__ == "__main__":
    main()
