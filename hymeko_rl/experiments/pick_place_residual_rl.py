"""Residual RL (SAC vs PPO) on the frozen 0.83 hybrid-DAgger policy — the PROPER setup after the naive RL run.

Fixes the naive run's known flaws (coin-toss lessons): the RL action is a normalised BOUNDED residual on the frozen
base (``hymeko_rl.env.residual_pick_env``), so the action space is uniform ``[-1,1]^7`` (no SAC scale mismatch) and
the policy can only REFINE the 0.83 base, not drift off it. Compares SAC (off-policy) and PPO (on-policy) residuals
against the frozen base, on the stable-place reward.

Not a conclusion — a fair test. If a bounded residual lifts the base, RL earns its keep on this task; if both hold
~flat at the base, the honest read is that the base is near-optimal and residual RL has no headroom (still a real
answer, but only after the setup is fair). No CORE.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.env.residual_pick_env import ResidualPickEnv
from hymeko_rl.eval.evaluate import PlacementPrecisionMetric, eval_metric, greedy_action_fn
from hymeko_rl.experiments.gripper_pick_bc import build, eval_success, load_pick_actor
from hymeko_rl.train.ppo import PPOConfig, train_ppo
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
from hymeko_rl.viz.render_pick_place import fanuc_pick_env

_ENV = dict(expert_version=3, require_settle=True, max_steps=1000)
_BASE_CKPT = "experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt"
_OUT = Path("experiments/residual_rl")


@dataclass(frozen=True)
class ResidualCfg:
    """The residual-RL setup axes (kept in one config, not a Cartesian param list — §6.5 #1)."""
    delta: float = 0.25          # joint residual bound (rad)
    grip: float = 0.02           # grip residual bound (m)
    reward_mode: str = "base"    # "base" (env dense) | "settle" (anti-farm, metric-aligned)
    active_modes: "tuple[str, ...] | None" = None   # None=residual everywhere; else RL only in these automaton modes


def _load_base(kind: str) -> "torch.nn.Module":
    return load_pick_actor(_BASE_CKPT, fanuc_pick_env(**_ENV), kind=kind)   # canonical, compat-validated loader


def save_residual_policy(state_dict, kind: str, cfg: "ResidualCfg", algo: str, metric: float, out_path) -> Path:
    """Save a trained residual policy as a GUI-loadable `residual_clone` checkpoint (base ref + residual weights +
    the delta/grip/reward metadata `load_pick_policy` needs to rebuild base+bounded-residual)."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "residual_clone", "base_ckpt": _BASE_CKPT, "base_kind": kind, "residual_kind": kind,
                "hidden": 64, "delta_scale": cfg.delta, "grip_scale": cfg.grip, "reward_mode": cfg.reward_mode,
                "active_modes": list(cfg.active_modes) if cfg.active_modes else None, "observe_target": False,
                "algo": algo, "metric": float(metric), "residual_state_dict": state_dict}, p)
    return p


def _zero_anchor(policy: "torch.nn.Module") -> "torch.nn.Module":
    """Zero the mean-output head so ``action_mean(obs) ≡ 0`` at init → the residual STARTS at the base (0.875).

    A residual policy is only a residual if its learned START reproduces the base (the coin-toss zero-delta
    invariant). The env enforces zero-residual→base, but a freshly-built policy emits a *nonzero* mean
    (measured mean|residual|≈0.24 → base drops to ~0.62 before any training). Zeroing the last linear of the mean
    path fixes it: PPO ``actor_mean``; SAC ``mu`` (``Linear`` or ``PerNodeActionHead.head``), whose ``tanh(0)=0``."""
    head = getattr(policy, "actor_mean", None) or getattr(policy, "mu", None)
    if head is None:
        raise AttributeError(f"{type(policy).__name__} has no actor_mean/mu head to zero-anchor")
    head = getattr(head, "head", head)              # PerNodeActionHead wraps its final Linear as `.head`
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()
    return policy


class PickResidualExpertTeacher:
    """Reactive DAgger teacher for the residual env — the F-SAC-5 rollout-state anchor. At each visited state it
    returns the RESIDUAL that would make base+bounded-residual reproduce the scripted expert:
    ``r = clip((expert_action − base_action) / delta, −1, 1)``. `train_sac` queries it on the actor's OWN rollout
    states, so the SAC actor is anchored toward the expert where it actually goes (the covariate-shift fix)."""

    def reset(self) -> None:
        return None

    def action(self, env: ResidualPickEnv) -> np.ndarray:
        base_a = env._base_action(env._obs)
        expert_a = np.asarray(env.env.expert_action, dtype=np.float32)
        delta = np.where(env._delta != 0.0, env._delta, 1.0)
        return np.clip((expert_a - base_a) / delta, -1.0, 1.0).astype(np.float32)


def _renv(base, cfg: ResidualCfg) -> ResidualPickEnv:
    return ResidualPickEnv(base, delta_scale=cfg.delta, grip_scale=cfg.grip, reward_mode=cfg.reward_mode,
                           active_modes=cfg.active_modes, env_kwargs=dict(_ENV))


def _eval(base, residual, cfg: ResidualCfg, *, n_eval: int, seed: int = 20_000) -> float:
    # Eval THROUGH the wrapper so the same base+bounded(+mode-gated) correction runs at eval as in training —
    # LiftPlaceMetric reads info["reached"] regardless of reward_mode. (reward_mode is inert for eval.)
    return eval_success(_renv(base, cfg), residual, n_eval, seed=seed)[1]


def _eval_precision(base, residual, cfg: ResidualCfg, *, n_eval: int, seed: int = 20_000) -> float:
    """Median placement error (cm, object→target-centre at settle) over settled episodes — the precision headroom.
    Returns ``nan`` if nothing settled. Reuses the shared eval loop with :class:`PlacementPrecisionMetric`."""
    renv = _renv(base, cfg)
    res = eval_metric(renv, greedy_action_fn(residual), PlacementPrecisionMetric(),
                      n_episodes=n_eval, seed0=seed)
    errs = [err for placed, err in res if placed and np.isfinite(err)]
    return float(np.median(errs) * 100.0) if errs else float("nan")


def _ppo_residual(kind: str, seed: int, base, cfg: ResidualCfg, *, n_iters: int, n_eval: int) -> "dict":
    renv = _renv(base, cfg)
    torch.manual_seed(seed)
    res = _zero_anchor(build(kind, renv, 64))           # residual STARTS at the base (0.875), then refines
    pcfg = PPOConfig(n_iters=n_iters, n_steps=2048, ent_coef=0.003, bc_coef=0.0, seed=seed)
    print(f"  [ppo-res] seed {seed} → {n_iters} iters (δ={cfg.delta} reward={cfg.reward_mode})", flush=True)
    train_ppo(res, renv, pcfg)
    fin = _eval(base, res, cfg, n_eval=n_eval)
    err = _eval_precision(base, res, cfg, n_eval=n_eval)
    print(f"  [ppo-res] seed {seed} FINAL stable-place={fin:.3f} place-err={err:.2f}cm", flush=True)
    return {"algo": "ppo_residual", "seed": seed, "final": fin, "err": err,
            "state": copy.deepcopy(res).cpu().eval().state_dict()}


def _sac_residual(kind: str, seed: int, base, cfg: ResidualCfg, *, total_steps: int, n_eval: int,
                  sac_rehab: bool = False, rollout_anchor: bool = False, anchor_coef: float = 1.0) -> "dict":
    renv = _renv(base, cfg)
    feat = int(renv.observation_space.shape[-1])
    torch.manual_seed(seed)
    kw = {} if kind == "mlp" else {"hg_state": renv.hg}
    actor, critics = build_sac(kind, obs_dim=feat, flat_dim=renv.hg.n_vertices * feat,
                               action_dim=renv.n_actions, action_scale=1.0, hidden=64, **kw)   # uniform [-1,1]
    _zero_anchor(actor)                                 # residual STARTS at the base (0.875), then refines
    # sac_rehab (F-SAC-1/3/4): init_alpha 0.1, lr 3e-4, ANNEAL entropy — fixes the critic/entropy confound (isolated:
    # this alone still collapsed to 0.458, critic bounded). rollout_anchor (F-SAC-5): + a rollout-state DAgger anchor
    # to the expert on the actor's OWN visited states — the isolated remaining cause (off-policy covariate shift).
    rehab = sac_rehab or rollout_anchor
    # SACConfig.stable() = the coin-toss-validated correction (F-SAC-1), single source of truth — NOT re-typed here.
    # Scope (F-SAC-4/8/9/10 / F-PP-009): fixes the critic divergence, does NOT hold the off-policy collapse; the
    # working fix is rollout-state DAgger, not SAC. Kept here as the audit/discriminator path.
    scfg = (SACConfig.stable(total_steps=total_steps, seed=seed, start_steps=500,
                             rollout_anchor_coef=(anchor_coef if rollout_anchor else 0.0))
            if rehab else SACConfig(total_steps=total_steps, start_steps=500, seed=seed))
    teacher = PickResidualExpertTeacher() if rollout_anchor else None
    tag = f"-ANCHOR×{anchor_coef:g}" if rollout_anchor else ("-REHAB" if sac_rehab else "")
    print(f"  [sac-res{tag}] seed {seed} → {total_steps} steps (δ={cfg.delta} reward={cfg.reward_mode})", flush=True)
    train_sac(actor, critics, renv, scfg, dagger_teacher=teacher,
              eval_fn=lambda _e, a: _eval(base, a, cfg, n_eval=max(4, n_eval // 4)))
    fin = _eval(base, actor, cfg, n_eval=n_eval)
    err = _eval_precision(base, actor, cfg, n_eval=n_eval)
    print(f"  [sac-res] seed {seed} FINAL stable-place={fin:.3f} place-err={err:.2f}cm", flush=True)
    return {"algo": "sac_residual", "seed": seed, "final": fin, "err": err,
            "state": copy.deepcopy(actor).cpu().eval().state_dict()}


def run(kind: str = "hsikan", *, seeds=(0, 1, 2), cfg: "ResidualCfg | None" = None,
        algos=("ppo", "sac"), sac_rehab: bool = False, rollout_anchor: bool = False,
        anchor_coef: float = 1.0, save_policy_dir: "str | None" = None, smoke: bool = False) -> "dict":
    cfg = cfg or ResidualCfg()
    n_eval = 3 if smoke else 24
    ppo_iters = 2 if smoke else 60
    sac_steps = 1_500 if smoke else 40_000
    base = _load_base(kind)
    base_score = _eval(base, _ZeroResidual(), cfg, n_eval=n_eval)
    base_err = _eval_precision(base, _ZeroResidual(), cfg, n_eval=n_eval)
    print(f"\n===== residual RL on frozen base (stable-place {base_score:.3f}, place-err {base_err:.2f}cm) — {kind} "
          f"[δ={cfg.delta} grip={cfg.grip} reward={cfg.reward_mode} modes={cfg.active_modes or 'all'} "
          f"algos={list(algos)}] =====", flush=True)
    rows: "list[dict]" = []
    for seed in seeds:
        if "ppo" in algos:
            rows.append(_ppo_residual(kind, seed, base, cfg, n_iters=ppo_iters, n_eval=n_eval))
        if "sac" in algos:
            rows.append(_sac_residual(kind, seed, base, cfg, total_steps=sac_steps, n_eval=n_eval,
                                      sac_rehab=sac_rehab, rollout_anchor=rollout_anchor, anchor_coef=anchor_coef))
    out = {"kind": kind, "cfg": {"delta": cfg.delta, "grip": cfg.grip, "reward_mode": cfg.reward_mode,
                                 "active_modes": list(cfg.active_modes) if cfg.active_modes else None},
           "base_stable_place": base_score, "base_place_err_cm": base_err}
    for algo in (f"{a}_residual" for a in algos):
        rs = [r for r in rows if r["algo"] == algo]
        errs = [r["err"] for r in rs if np.isfinite(r["err"])]
        out[algo] = {"final_median": st.median([r["final"] for r in rs]),
                     "final_per_seed": [r["final"] for r in rs],
                     "delta_vs_base_median": st.median([r["final"] - base_score for r in rs]),
                     "err_cm_median": st.median(errs) if errs else float("nan"),
                     "err_cm_per_seed": [r["err"] for r in rs]}
        if save_policy_dir:                                        # save the best seed as a GUI-loadable residual ckpt
            best = max(rs, key=lambda r: r["final"])
            tag = f"{algo}_{kind}_d{cfg.delta}_{cfg.reward_mode}{'_rehab' if sac_rehab and 'sac' in algo else ''}.pt"
            path = save_residual_policy(best["state"], kind, cfg, algo, best["final"], Path(save_policy_dir) / tag)
            out[algo]["saved_policy"] = str(path)
            print(f"  saved best {algo} (stable-place {best['final']:.3f}) → {path}", flush=True)
    return out


class _ZeroResidual:
    """A residual that always outputs 0 → the adapter reproduces the base exactly (for the base score)."""

    def eval(self):
        return self

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 7)


def _save(out: "dict", cfg: ResidualCfg, kind: str) -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    tag = "all" if not cfg.active_modes else "-".join(cfg.active_modes)
    path = _OUT / f"residual_{kind}_d{cfg.delta}_{cfg.reward_mode}_{tag}.json"
    path.write_text(json.dumps({k: v for k, v in out.items()}, indent=2, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "sa_hsikan"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--delta", type=float, default=0.25, help="joint residual bound (rad)")
    ap.add_argument("--grip", type=float, default=0.02, help="grip residual bound (m)")
    ap.add_argument("--reward-mode", default="base", choices=("base", "settle", "precision"),
                    help="'base'=env dense; 'settle'=anti-farm metric-aligned; 'precision'=settle + centering bonus")
    ap.add_argument("--active-modes", nargs="+", default=None,
                    help="restrict RL residual to these hybrid-automaton modes (e.g. place release); "
                         "default = residual everywhere")
    ap.add_argument("--algo", nargs="+", default=["ppo", "sac"], choices=("ppo", "sac"),
                    help="which optimiser(s) to run; PPO refines a near-optimal base, SAC's entropy drifts it")
    ap.add_argument("--sac-rehab", action="store_true",
                    help="apply the F-SAC-1/3/4 fix to the SAC arm (init_alpha 0.1, lr 3e-4, ANNEAL entropy) — "
                         "tests whether the residual-SAC collapse was the entropy/critic confound (pitfall audit)")
    ap.add_argument("--anchor-coef", type=float, default=1.0,
                    help="strength of the rollout-state anchor (F-SAC-3: coef 1.0 is dwarfed by -Q; sweep to test if a "
                         "stronger anchor holds the base). Only used with --rollout-anchor.")
    ap.add_argument("--rollout-anchor", action="store_true",
                    help="F-SAC-5 fix: rehab config + a rollout-state DAgger anchor to the expert on the actor's own "
                         "visited states — the isolated remaining cause of the residual-SAC collapse")
    ap.add_argument("--save-policy", default=None, metavar="DIR",
                    help="save the best-seed residual policy per arm as a GUI-loadable residual_clone .pt in DIR "
                         "(e.g. experiments/pick_place_gui)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    cfg = ResidualCfg(delta=a.delta, grip=a.grip, reward_mode=a.reward_mode,
                      active_modes=tuple(a.active_modes) if a.active_modes else None)
    s = run(a.kind, seeds=tuple(a.seeds), cfg=cfg, algos=tuple(a.algo), sac_rehab=a.sac_rehab,
            rollout_anchor=a.rollout_anchor, anchor_coef=a.anchor_coef, save_policy_dir=a.save_policy, smoke=a.smoke)
    path = _save(s, cfg, a.kind)
    print(f"\n=== Residual RL vs frozen base (stable-place {s['base_stable_place']:.3f}, "
          f"place-err {s['base_place_err_cm']:.2f}cm) [δ={cfg.delta} reward={cfg.reward_mode} "
          f"modes={cfg.active_modes or 'all'}] ===")
    for algo in (f"{x}_residual" for x in a.algo):
        d = s[algo]
        print(f"  {algo:14s}: stable-place {d['final_median']:.3f} (Δ {d['delta_vs_base_median']:+.3f}) | "
              f"place-err {d['err_cm_median']:.2f}cm (base {s['base_place_err_cm']:.2f}) | "
              f"seeds {d['final_per_seed']}")
    print(f"saved → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
