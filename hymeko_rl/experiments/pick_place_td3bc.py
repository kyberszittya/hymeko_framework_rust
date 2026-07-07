"""FANUC pick-and-place, trained with **TD3+BC** on the reusable :class:`~hymeko_rl.train.campaign.Campaign`.

The registry recommends ``td3_bc`` for ``pick_place`` (``eval/tasks.py``), but the only ready pick script
(``pick_place_bc.py --algo td3``) is a BC-anchor-less TD3 *warm-start* that already collapsed on FANUC
(``project-fanuc-offpolicy-collapse``) and runs BC on CPU. This entrypoint closes that gap by driving the
**same** BC→TD3+BC loop the galambos A/B uses (``exp_galambos_coord_ab``) — GPU ``device="auto"``,
``td3_bc_config`` (bc_coef 2.5 + warm-start + critic_warmup 2000 + offline replay), best-checkpoint on
``place`` (so a collapse can never lose the BC floor), and self-contained ``experiments/<ts>/`` artifacts
(policies + gifs + results.json + run.log). Config + two closures, no algorithm re-implementation (§6.5 #3).

Oracle note: ``reward_oracle.certify`` models the galambos farming MDP only and cannot score the procedural
pick reward, so the pre-queue certify gate is **N/A** here (pick-place is not a dense-annuity farming trap).
Safety is the LiftPlaceMetric divergence guard (a blow-up is never a counted success) + the BC-floor
best-checkpoint. Pick-place is ~unsolved (real ≈0.125 lift); the honest target is "hold the BC floor, no
collapse", decided on multi-seed median/IQR.

    python -m hymeko_rl.experiments.pick_place_td3bc --smoke                 # 1 seed × 4k, path check
    python -m hymeko_rl.experiments.pick_place_td3bc --kind hsikan --seeds 0 # 1-seed production (1e5 steps)
    python -m hymeko_rl.experiments.pick_place_td3bc --kind hsikan           # 3-seed (0,1,2) × 1e5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium.spaces import Box

from hymeko_rl.experiments.gripper_pick_bc import collect_demos, eval_success
from hymeko_rl.train.campaign import Campaign, CampaignConfig
from hymeko_rl.train.ddpg import build_offpolicy
from hymeko_rl.viz.render_pick_place import fanuc_pick_env

_EVAL_SEED = 20_000
_LIFT_THRESH = 0.035   # matches fanuc_pick_env / gripper_pick_bc; the object-height success gate


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _build_factory(kind: str, hidden: int) -> "Any":
    """A Campaign ``build(env) -> (actor, [critic, critic])`` closed over the backbone: twin critics for TD3,
    structural kinds get the env hypergraph (``hg_state``). Mirrors ``pick_place_bc``'s off-policy construction.
    # Preconditions ``kind in {hsikan, mlp, signedkan, sa_hsikan, mixture}``; env exposes ``hg``/``n_actions``."""
    def _build(env: Any) -> "tuple[Any, list[Any]]":
        shape = env.observation_space.shape
        assert shape is not None
        nv, feat = int(shape[0]), int(shape[1])
        act_space = env.action_space
        assert isinstance(act_space, Box)
        scale = float(np.max(np.abs(act_space.high)))
        kw: "dict[str, Any]" = {} if kind == "mlp" else {"hg_state": env.hg}
        return build_offpolicy(kind, obs_dim=feat, flat_dim=nv * feat, action_dim=env.n_actions,
                               action_scale=scale, n_critics=2, hidden=hidden, **kw)
    return _build


def _measure_factory(n_eval: int) -> "Any":
    """A Campaign ``measure(make_env, actor) -> {lift, place}`` via the shared ``eval_success`` (LiftPlaceMetric
    divergence guard — a physics blow-up is a FAILURE, never a counted lift). ``select="place"`` is the true task
    success; ``lift`` is recorded as the achievable intermediate."""
    def _measure(make_env: Any, actor: Any) -> "dict[str, float]":
        env = make_env()
        try:
            lift, place = eval_success(env, actor, n_eval, seed=_EVAL_SEED, lift_thresh=_LIFT_THRESH)
        finally:
            _close_env(env)
        return {"lift": round(lift, 4), "place": round(place, 4)}
    return _measure


def _demos(env: Any, n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """The BC teacher: roll the scripted IK expert, keep only successful episodes (clean demos)."""
    return collect_demos(env, n, seed, only_success=True, lift_thresh=_LIFT_THRESH)


@dataclass(frozen=True)
class PickConfig:
    """One declaration site for the FANUC pick TD3+BC run (§6.5 #1: axes in config, not a growing signature).

    ``None`` budgets resolve to smoke (path check) or full defaults via :meth:`resolve`. ``smoke`` CAPS every
    budget so a fully specified config cannot turn ``--smoke`` into a full launch (the galambos lesson).
    """

    kind: str = "hsikan"
    hidden: int = 64
    smoke: bool = False
    seeds: "tuple[int, ...] | None" = None      # None → (0,) smoke / (0, 1, 2)
    total_steps: "int | None" = None            # None → 4_000 smoke / 100_000
    n_demos: "int | None" = None                # None → 6 smoke / 18
    bc_epochs: "int | None" = None              # None → 5 smoke / 80
    n_eval: "int | None" = None                 # None → 4 smoke / 8
    bc_batch: int = 128

    def resolve(self) -> "tuple[tuple[int, ...], int, int, int, int, int, int]":
        """Effective ``(seeds, total_steps, n_demos, bc_epochs, n_eval, eval_every, n_envs)``.
        # Postconditions under ``smoke``: 1 seed, ≤4k steps, ≤6 demos, ≤5 epochs, ≤4 eval, 2 envs."""
        seeds = self.seeds if self.seeds is not None else (0, 1, 2)
        total_steps = self.total_steps if self.total_steps is not None else 100_000
        n_demos = self.n_demos if self.n_demos is not None else 18
        bc_epochs = self.bc_epochs if self.bc_epochs is not None else 80
        n_eval = self.n_eval if self.n_eval is not None else 8
        if self.smoke:
            seeds, total_steps = seeds[:1], min(total_steps, 4_000)
            n_demos, bc_epochs, n_eval = min(n_demos, 6), min(bc_epochs, 5), min(n_eval, 4)
        eval_every = 1_000 if self.smoke else max(12_500, total_steps // 8)
        n_envs = 2 if self.smoke else 8
        return seeds, total_steps, n_demos, bc_epochs, n_eval, eval_every, n_envs


def run(cfg: PickConfig, base: "str | Any" = "experiments") -> "dict[str, Any]":
    """Build + run the FANUC pick TD3+BC Campaign; return the summary with launch provenance.
    ``base`` is the parent dir for the ``<ts>_fanuc_pick_td3bc_<kind>/`` output (tests point it at a tmp dir).
    # Postconditions writes one ``<base>/<ts>_fanuc_pick_td3bc_<kind>/`` with results.json + policies + gifs.
    """
    seeds, total_steps, n_demos, bc_epochs, n_eval, eval_every, n_envs = cfg.resolve()
    # compile pays off only past ~1e4 steps (fixed capture cost); skip it for the short smoke path check.
    offpolicy = {"compile": False} if cfg.smoke else None
    camp_cfg = CampaignConfig(
        name=f"fanuc_pick_td3bc_{cfg.kind}", select="place", seeds=seeds, total_steps=total_steps,
        eval_every=eval_every, n_demos=n_demos, bc_epochs=bc_epochs, n_eval=n_eval, n_envs=n_envs,
        device="auto", offpolicy=offpolicy, bc_batch=cfg.bc_batch)
    camp = Campaign(camp_cfg, make_env=fanuc_pick_env, build=_build_factory(cfg.kind, cfg.hidden),
                    measure=_measure_factory(n_eval), demos=_demos, gif=not cfg.smoke)
    print("[oracle] pick-place certify = N/A (reward_oracle models the galambos farming MDP only; "
          "pick reward is procedural lift/place, guarded by LiftPlaceMetric divergence + BC-floor checkpoint)",
          flush=True)
    result = camp.run()
    result["reward_certificate"] = ("N/A — reward_oracle.certify models the galambos grasp-and-deliver "
                                    "farming MDP only; pick-place reward is procedural (not a dense-annuity "
                                    "farming trap). Safety: LiftPlaceMetric divergence guard + BC-floor checkpoint.")
    result["kind"] = cfg.kind
    result["budget"] = {"seeds": list(seeds), "total_steps": total_steps, "n_demos": n_demos,
                        "bc_epochs": bc_epochs, "n_eval": n_eval, "n_envs": n_envs}
    return result


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "sa_hsikan"))
    ap.add_argument("--smoke", action="store_true", help="1 seed × 4k steps path check on the GPU")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit seeds (e.g. --seeds 0)")
    ap.add_argument("--steps", type=int, default=None, help="off-policy env-step budget per seed")
    args = ap.parse_args(argv)
    cfg = PickConfig(kind=args.kind, smoke=args.smoke,
                     seeds=tuple(args.seeds) if args.seeds is not None else None, total_steps=args.steps)
    summary = run(cfg)
    print("\n=== FANUC pick-place TD3+BC ===")
    print(f"  {summary['kind']}: place median = {summary.get('place_median', '?')} "
          f"per-seed={summary.get('place_per_seed', '?')}")
    print(json.dumps({k: v for k, v in summary.items() if k != "seeds"}, indent=2, default=str)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
