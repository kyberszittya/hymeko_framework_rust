"""R11.6B — robustness-seeking generalization RL. A0 (BC) / A1 (nominal reward, = v2.1) / A2 (robust reward) on the SAME
panel; only the reward semantics differ between A1 and A2, so any dev gain is attributable to the robustness objective.

Wide-basin teachers (R11.4B basin audit) are the immutable positive anchor; narrow ones are ordinary replay / stress.
Selection is combined + robustness-aware (train nominal floor preserved AND dev nominal + dev robust). After the gate, the
frozen policy runs the UNTOUCHED test panel ONCE. Reuses the v2.1 bank / warm-start / TD3 trainer; new = the robust env.
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

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import (
    CoinDeliveryThetaOptionEnv,
    DeliveryReward,
    box_to_theta,
    fit_obs_standardizer,
)
from hymeko_rl.coin_delivery.theta_option.robust_delivery import RobustCoinDeliveryEnv, RobustRewardConfig
from hymeko_rl.experiments.r11_6a_delivery_rl import (
    _load_teacher,
    _make_combined_eval,
    bc_init_actor,
    build_bank,
    eval_actor,
    teacher_positives,
    train_td3_anchored,
)
from hymeko_rl.option_rl import SemiMDPConfig, make_actor
from hymeko_rl.option_rl.core import OptionTransition

R11_4B_DATASET = Path("reports/2026-08-03-r11-4b-bc/dataset")
BASIN_DIR = Path("reports/2026-08-03-r11-4b-bc/basin")
DEFAULT_OUT = Path("reports/2026-08-05-r11-6b-robust-rl")
SELECT_SEED = 909
TEST_SEED = 707


def wide_basin_ids(basin_dir: Path, scale: str = "0.01", thresh: float = 0.5) -> "set[str]":
    """Scenario ids whose stored teacher theta survive >= ``thresh`` at the given perturbation scale (the R11.4B audit)."""
    out: set[str] = set()
    for f in sorted(glob.glob(str(basin_dir / "basin_*.jsonl"))):
        for line in Path(f).open():
            if line.strip():
                r = json.loads(line)
                if "error" not in r and float(r["survival"].get(scale, 0.0)) >= thresh:
                    out.add(r["scenario_id"])
    return out


def teacher_positives_filtered(env: CoinDeliveryThetaOptionEnv, reward_scale: float, keep_ids: "set[str] | None"
                               ) -> "list[OptionTransition]":
    """Immutable teacher positives, optionally restricted to ``keep_ids`` (wide-basin only for A2)."""
    pos = teacher_positives(env, reward_scale)
    if keep_ids is None:
        return pos
    train = env.indices("train")
    return [p for p, i in zip(pos, train) if env.handoff_id(i) in keep_ids]


def _actor_theta(actor: Any, env: Any, idx: int) -> np.ndarray:
    o = env.reset(idx)
    with torch.no_grad():
        z = actor.mean_action(torch.as_tensor(o[None]).float())[0].numpy()
    return box_to_theta(z)


def robust_dev(env: RobustCoinDeliveryEnv, actor: Any, idx: "list[int]", min_survival: float, seed: int) -> dict[str, float]:
    """Fixed-seed robust evaluation: nominal K6 rate + robust-success rate (nominal K6 AND survival >= min) over ``idx``."""
    nom = rob = 0.0
    for i in idx:
        cert = env.certify(_actor_theta(actor, env, i), i, seed=seed)
        nom += float(cert.nominal_k6)
        rob += float(cert.is_robust(min_survival))
    n = max(1, len(idx))
    return {"nominal": round(nom / n, 3), "robust": round(rob / n, 3)}


def _make_robust_select(nominal_env: CoinDeliveryThetaOptionEnv, robust_env: RobustCoinDeliveryEnv, dev_idx: "list[int]",
                        train_sub: "list[int]", ws_train_sub: float, margin: float, min_survival: float) -> Any:
    """Combined + robustness-aware selection: score = dev nominal + dev robust IF the train subset is preserved, else -1."""
    def fn(actor: Any) -> "tuple[float, dict[str, float]]":
        trn = eval_actor(actor, nominal_env, train_sub)["k6"]
        if trn < ws_train_sub - margin:
            return -1.0, {"train_sub": trn}
        dev_nom = eval_actor(actor, nominal_env, dev_idx)["k6"]
        dev_rob = robust_dev(robust_env, actor, dev_idx, min_survival, SELECT_SEED)["robust"]
        return dev_nom + dev_rob, {"train_sub": trn, "dev_nom": dev_nom, "dev_rob": dev_rob}
    return fn


def _train_arm(train_env: Any, nominal_env: CoinDeliveryThetaOptionEnv, robust_env: RobustCoinDeliveryEnv, base: Any,
               dev_idx: "list[int]", cfg: SemiMDPConfig, positives: list, eval_fn: Any, seed: int,
               min_survival: float) -> dict[str, Any]:
    """Train one seed on ``train_env`` (nominal for A1, robust for A2); evaluate nominally + robustly on the shared envs."""
    actor = copy.deepcopy(base)
    ckpts, _hist = train_td3_anchored(train_env, actor, eval_fn, cfg, positives, 0.25, seed)
    best = make_actor("td3", nominal_env.obs_dim, nominal_env.act_dim)
    best.load_state_dict(ckpts["best_val"])
    return {"seed": seed, "train": eval_actor(best, nominal_env, nominal_env.indices("train")),
            "dev": eval_actor(best, nominal_env, dev_idx),
            "dev_robust": robust_dev(robust_env, best, dev_idx, min_survival, SELECT_SEED), "state": ckpts["best_val"]}


def _run_arm(name: str, train_env: Any, c: dict, positives: list, eval_fn: Any, n_seeds: int) -> "list[dict]":
    out = []
    for s in range(n_seeds):
        res = _train_arm(train_env, c["nom"], c["rob"], c["base"], c["dev_idx"], c["cfg"], positives, eval_fn, s,
                         c["rcfg"].robust_min_survival)
        print(f"[{name} seed {s}] train {res['train']} dev {res['dev']} dev_robust {res['dev_robust']}", flush=True)
        out.append(res)
    return out


_STABLE = "R11_6B_ROBUST_OBJECTIVE_STABLE_BUT_NO_GENERALIZATION_GAIN"


def _6b_verdict(safe: bool, no_collapse: bool, tr: float, dev_gain: int, rob: float, n_seed_gain: int, n_seeds: int) -> str:
    if not no_collapse:
        return "R11_6B_LOCAL_ROBUSTNESS_REWARD_INSUFFICIENT"           # drifted to narrow (train collapsed in a seed)
    if not (safe and tr >= 0.75):
        return _STABLE
    if dev_gain >= 2 and rob >= 0.50 and n_seed_gain >= (n_seeds + 1) // 2:
        return "R11_6B_ROBUST_REWARD_GENERALIZATION_PASS"
    return _STABLE                                                     # objective stable but no unseen gain (coverage limit)


def gate_6b(a2: "list[dict]", a0_dev: dict, ws_train: float, min_survival: float, n_dev: int) -> dict[str, Any]:
    """PASS: train nominal >=0.75 no-collapse, dev nominal +2 scenarios over A0/BC, dev robust >=50%, seeds>=2/3 dev-gain,
    0 safety. Else STABLE_BUT_NO_GENERALIZATION_GAIN (train held, no dev gain) or LOCAL_ROBUSTNESS_REWARD_INSUFFICIENT."""
    tr = round(float(np.mean([s["train"]["k6"] for s in a2])), 3)
    dv = round(float(np.mean([s["dev"]["k6"] for s in a2])), 3)
    rob = round(float(np.mean([s["dev_robust"]["robust"] for s in a2])), 3)
    dev_gain_scen = round((dv - a0_dev["k6"]) * n_dev)                    # dev K6 gain expressed in # scenarios
    n_seed_gain = sum(1 for s in a2 if s["dev"]["k6"] > a0_dev["k6"] + 1e-9)
    safe = all(s["train"]["safe"] >= 0.999 and s["dev"]["safe"] >= 0.999 for s in a2)
    no_collapse = all(s["train"]["k6"] >= ws_train - 0.15 for s in a2)
    verdict = _6b_verdict(safe, no_collapse, tr, dev_gain_scen, rob, n_seed_gain, len(a2))
    return {"a2_mean_train_k6": tr, "a2_mean_dev_k6": dv, "a2_mean_dev_robust": rob, "a0_bc_dev_k6": a0_dev["k6"],
            "dev_gain_scenarios": dev_gain_scen, "seeds_with_dev_gain": f"{n_seed_gain}/{len(a2)}",
            "train_no_collapse": no_collapse, "safety_ok": safe, "verdict": verdict}


def _test_certificates(env: RobustCoinDeliveryEnv, actor: Any, min_survival: float) -> dict[str, Any]:
    """Freeze the policy and run the UNTOUCHED test panel ONCE (never in reward/selection)."""
    test_idx = env.indices("test")
    certs = []
    for i in test_idx:
        cert = env.certify(_actor_theta(actor, env, i), i, seed=TEST_SEED)
        certs.append({"scenario_id": env.handoff_id(i), **cert.to_dict(), "robust": cert.is_robust(min_survival)})
    n = max(1, len(test_idx))
    return {"test_nominal_k6": round(sum(c["nominal_k6"] for c in certs) / n, 3),
            "test_robust": round(sum(c["robust"] for c in certs) / n, 3), "certificates": certs}


def _build_context(args: argparse.Namespace) -> dict[str, Any]:
    ctx = bc_context()
    bank = build_bank(ctx, _load_teacher(R11_4B_DATASET), ("train", "dev", "test"), args.limit)
    std = fit_obs_standardizer(bank)
    rcfg = RobustRewardConfig(k=args.k, sigma=args.sigma)
    nom = CoinDeliveryThetaOptionEnv(bank, std, DeliveryReward(), seed=0)
    rob = RobustCoinDeliveryEnv(bank, std, DeliveryReward(), rcfg, seed=0)
    dev_idx = nom.indices("dev") or nom.indices("train")[:2]
    wide = wide_basin_ids(BASIN_DIR)
    base = make_actor("td3", nom.obs_dim, nom.act_dim)
    bc_init_actor(base, nom, 300, 1e-3, seed=0)
    a0 = eval_actor(base, nom, dev_idx)                                                    # A0: BC warm-start, no RL
    train_sub = nom.indices("train")[:14]
    ws = eval_actor(base, nom, train_sub)["k6"]
    cfg = SemiMDPConfig(total_options=args.total_options, warmup_options=args.warmup, expl_noise=0.1)
    n_wide = sum(1 for i in nom.indices("train") if nom.handoff_id(i) in wide)
    print(f"bank {len(bank)} ({len(nom.indices('train'))}tr/{len(dev_idx)}dev/{len(nom.indices('test'))}test); "
          f"wide-basin train anchors {n_wide}; A0(BC) dev {a0} | ws train_sub {ws}", flush=True)
    return {"nom": nom, "rob": rob, "base": base, "dev_idx": dev_idx, "a0": a0, "ws": ws, "cfg": cfg, "rcfg": rcfg,
            "a1_pos": teacher_positives(nom, cfg.reward_scale),                    # A1: all positives, nominal reward
            "a1_eval": _make_combined_eval(nom, dev_idx, train_sub, ws, 0.15),
            "a2_pos": teacher_positives_filtered(rob, cfg.reward_scale, wide),     # A2: wide-basin positives, robust reward
            "a2_eval": _make_robust_select(nom, rob, dev_idx, train_sub, ws, 0.15, rcfg.robust_min_survival)}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--total-options", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.01)
    return ap.parse_args()


def _maybe_test_panel(g: dict, a2: "list[dict]", c: dict, ms: float) -> None:
    """On PASS only, freeze the best A2 policy and run the UNTOUCHED test panel once."""
    if g["verdict"] != "R11_6B_ROBUST_REWARD_GENERALIZATION_PASS":
        return
    actor = make_actor("td3", c["nom"].obs_dim, c["nom"].act_dim)
    actor.load_state_dict(max(a2, key=lambda s: s["dev"]["k6"] + s["dev_robust"]["robust"])["state"])
    g["test_panel"] = _test_certificates(c["rob"], actor, ms)


def _mean_arm(arm: "list[dict]") -> dict[str, float]:
    return {"mean_dev_k6": round(float(np.mean([s["dev"]["k6"] for s in arm])), 3),
            "mean_dev_robust": round(float(np.mean([s["dev_robust"]["robust"] for s in arm])), 3)}


def _write_result(out: Path, g: dict, a1: "list[dict]", a2: "list[dict]") -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {**g, "a1": [{k: v for k, v in s.items() if k != "state"} for s in a1],
               "a2": [{k: v for k, v in s.items() if k != "state"} for s in a2]}
    (out / "gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in g.items() if k != "test_panel"}, indent=2), flush=True)
    print("R11_6B_ROBUST_RL_DONE", flush=True)


def main() -> None:
    args = _parse_args()
    c = _build_context(args)
    ms = c["rcfg"].robust_min_survival
    a1 = _run_arm("A1", c["nom"], c, c["a1_pos"], c["a1_eval"], args.seeds)     # nominal reward (= v2.1)
    a2 = _run_arm("A2", c["rob"], c, c["a2_pos"], c["a2_eval"], args.seeds)     # robust reward (R11.6B)
    g = gate_6b(a2, c["a0"], c["ws"], ms, len(c["dev_idx"]))
    g["a1_nominal_reference"] = _mean_arm(a1)
    _maybe_test_panel(g, a2, c, ms)
    _write_result(args.out, g, a1, a2)


if __name__ == "__main__":
    main()
