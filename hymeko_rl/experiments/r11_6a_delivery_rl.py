"""R11.6A — reward-driven coin delivery RL from a certified grasp. Build the certified-handoff bank, BC-warm-start a TD3
actor to the teacher theta, train N seeds on the shaped physical reward, select by dev K6, gate.

Reuses: option_rl (TD3 semi-MDP + selection), delivery_theta_env (env + reward), delivery_bc (handoff reconstruction +
teacher theta from the R11.4B dataset), forward_displacement (frozen rollout + K6). The policy learns to deliver WITHOUT
per-instance CEM and WITHOUT imitating the teacher; the 56 teacher theta are warm-start + positive reference only.

Genuinely new vs the prior walls: full structured theta (R8/R9 residual wall), multi-scenario target-conditioned shaped
reward (R10.2 sigma-ball wall), 56 demos across 51 scenarios (R7 2-cradle coverage wall). TD3, not SAC.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.delivery_bc.dataset import BcSample, bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.evaluate import rollout_theta
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import (
    CoinDeliveryThetaOptionEnv,
    DeliveryReward,
    ScenarioHandoff,
    fit_obs_standardizer,
)
from hymeko_rl.option_rl import SemiMDPConfig, make_actor, train_semi_mdp

R11_4B_DATASET = Path("reports/2026-08-03-r11-4b-bc/dataset")
DEFAULT_OUT = Path("reports/2026-08-04-r11-6a-delivery-rl")
BC_HELD_OUT_REF = 0.25            # R11.4B best held-out (1-NN); R11.6A must beat this without per-instance search


def _load_teacher(dataset_dir: Path) -> "list[BcSample]":
    out: list[BcSample] = []
    for f in sorted(glob.glob(str(dataset_dir / "extract_*.jsonl"))):
        for line in Path(f).open():
            if line.strip():
                d = json.loads(line)
                if not d.get("omitted"):
                    out.append(BcSample.from_json(d))
    return out


def _build_handoff(ctx: tuple, smp: BcSample) -> "ScenarioHandoff | None":
    cfg, conf, obj = ctx
    rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(smp.scenario_id), smp.seed)
    if rc is None:
        return None
    snap = rc.result.outcome.snapshot
    theta = np.array(smp.theta, np.float64)
    if not rollout_theta(snap, theta)["k6"]:                                # teacher theta must reproduce K6 (fidelity)
        return None
    return ScenarioHandoff(smp.scenario_id, smp.split, smp.seed, snap, np.array(smp.x, np.float64), theta)


def build_bank(ctx: tuple, smps: "list[BcSample]", splits: tuple, limit: int) -> "list[ScenarioHandoff]":
    """Reconstruct + cache the certified handoffs (train first so a --limit smoke keeps train coverage)."""
    chosen = [s for s in smps if s.split in splits]
    chosen.sort(key=lambda s: (s.split != "train", s.scenario_id))
    if limit:
        chosen = chosen[:limit]
    out = [_build_handoff(ctx, s) for s in chosen]
    return [h for h in out if h is not None]


def bc_init_actor(actor: Any, env: CoinDeliveryThetaOptionEnv, epochs: int, lr: float, seed: int) -> Any:
    """Warm-start the DetActor toward the teacher theta (reference, not a fixed target); trains on standardized train obs."""
    torch.manual_seed(seed)
    x_np, y_np = env.warmstart_data()
    x, y = torch.tensor(x_np), torch.tensor(y_np)
    opt = torch.optim.Adam(actor.parameters(), lr)
    for _ in range(epochs):
        opt.zero_grad()
        (((actor(x) - y) ** 2).mean()).backward()
        opt.step()
    return actor


def eval_actor(actor: Any, env: CoinDeliveryThetaOptionEnv, idx: "list[int]") -> dict[str, float]:
    """Mean-action (no exploration, no search) closed-loop K6 + safety over scenarios ``idx``."""
    k6 = safe = 0.0
    for i in idx:
        o = env.reset(i)
        with torch.no_grad():
            z = actor.mean_action(torch.as_tensor(o[None]).float())[0].numpy()
        _o2, _r, _done, info = env.step(z)
        k6 += info["k6"]
        safe += float(info["safe"])
    n = max(1, len(idx))
    return {"k6": round(k6 / n, 3), "safe": round(safe / n, 3)}


def _fixed_action_k6(env: CoinDeliveryThetaOptionEnv, idx: "list[int]", action: np.ndarray) -> dict[str, float]:
    k6 = safe = 0.0
    for i in idx:
        env.reset(i)
        _o2, _r, _done, info = env.step(action)
        k6 += info["k6"]
        safe += float(info["safe"])
    n = max(1, len(idx))
    return {"k6": round(k6 / n, 3), "safe": round(safe / n, 3)}


def _make_dev_eval(env: CoinDeliveryThetaOptionEnv, dev_idx: "list[int]") -> Any:
    def fn(actor: Any) -> "tuple[float, dict[str, float]]":
        r = eval_actor(actor, env, dev_idx)
        return r["k6"], {"safe": r["safe"]}
    return fn


def train_seed(env: CoinDeliveryThetaOptionEnv, base_actor: Any, dev_idx: "list[int]", seed: int,
               cfg: SemiMDPConfig) -> dict[str, Any]:
    actor = copy.deepcopy(base_actor)
    ckpts, history = train_semi_mdp("td3", env, actor, _make_dev_eval(env, dev_idx), cfg,
                                    obs_dim=env.obs_dim, act_dim=env.act_dim, seed=seed)
    best = make_actor("td3", env.obs_dim, env.act_dim)
    best.load_state_dict(ckpts["best_val"])
    return {"seed": seed, "train": eval_actor(best, env, env.indices("train")),
            "dev": eval_actor(best, env, dev_idx), "final_dev": history[-1]["dev_score"] if history else 0.0}


def _r11_6a_verdict(passed: bool, tr: float, safe: bool, warmstart_train: float) -> str:
    if passed:
        return "R11_6A_REWARD_DRIVEN_DELIVERY_LEARNS"
    if tr < 0.60 and tr <= warmstart_train + 1e-9:
        return "R11_6A_ACTION_COORDINATE_INSUFFICIENT"         # can't beat the teacher warm-start
    if not safe:
        return "R11_6A_REWARD_MISSPECIFIED"                    # found unsafe/degenerate optima
    return "R11_6A_OPTIMIZATION_STALLED"


def gate(seeds: "list[dict]", box_center: dict, warmstart: dict) -> dict[str, Any]:
    """R11.6A PASS: mean train K6 >= .60, mean dev K6 >= .50, 0 safety regression, and beats box-center + the R11.4B BC
    held-out (0.25) — i.e. RL amortizes delivery (no per-instance search) where BC could not even fit train (0.386)."""
    tr = round(float(np.mean([s["train"]["k6"] for s in seeds])), 3)
    dv = round(float(np.mean([s["dev"]["k6"] for s in seeds])), 3)
    safe = all(s["train"]["safe"] >= 0.999 and s["dev"]["safe"] >= 0.999 for s in seeds)
    beats = dv > box_center["k6"] and dv > BC_HELD_OUT_REF
    passed = tr >= 0.60 and dv >= 0.50 and safe and beats
    verdict = _r11_6a_verdict(passed, tr, safe, warmstart["train"]["k6"])
    return {"mean_train_k6": tr, "mean_dev_k6": dv, "safety_ok": safe, "box_center_k6": box_center["k6"],
            "warmstart_train_k6": warmstart["train"]["k6"], "warmstart_dev_k6": warmstart["dev"]["k6"],
            "bc_heldout_ref": BC_HELD_OUT_REF, "beats_baselines": beats, "per_seed": seeds, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=R11_4B_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="cap scenarios (smoke); 0 = all train+dev")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total-options", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--bc-epochs", type=int, default=300)
    ap.add_argument("--expl-noise", type=float, default=0.2, help="TD3 exploration std (lower = gentler around warm-start)")
    args = ap.parse_args()

    ctx = bc_context()
    smps = _load_teacher(args.dataset_dir)
    bank = build_bank(ctx, smps, ("train", "dev"), args.limit)
    print(f"bank: {len(bank)} handoffs ({sum(h.split=='train' for h in bank)} train / {sum(h.split=='dev' for h in bank)} dev)", flush=True)
    std = fit_obs_standardizer(bank)
    env = CoinDeliveryThetaOptionEnv(bank, std, DeliveryReward(), seed=0)
    dev_idx = env.indices("dev") or env.indices("train")[:2]

    base = make_actor("td3", env.obs_dim, env.act_dim)
    bc_init_actor(base, env, args.bc_epochs, 1e-3, seed=0)
    warmstart = {"train": eval_actor(base, env, env.indices("train")), "dev": eval_actor(base, env, dev_idx)}
    box_center = _fixed_action_k6(env, env.indices("train") + dev_idx, np.zeros(env.act_dim, np.float32))
    print(f"warm-start: train {warmstart['train']} dev {warmstart['dev']} | box-center {box_center}", flush=True)

    cfg = SemiMDPConfig(total_options=args.total_options, warmup_options=args.warmup, expl_noise=args.expl_noise)
    seeds = []
    for s in range(args.seeds):
        res = train_seed(env, base, dev_idx, s, cfg)
        print(f"[seed {s}] train {res['train']} dev {res['dev']}", flush=True)
        seeds.append(res)

    g = gate(seeds, box_center, warmstart)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gate.json").write_text(json.dumps(g, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in g.items() if k != "per_seed"}, indent=2), flush=True)
    print("R11_6A_DELIVERY_RL_DONE", flush=True)


if __name__ == "__main__":
    main()
