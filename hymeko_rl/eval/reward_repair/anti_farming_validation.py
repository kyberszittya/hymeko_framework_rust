"""Offline anti-farming validation for the monitor-aligned reward — DECOUPLED counterfactual trajectories.

The R2 scripted rollouts are collinear (contact≈delivery), so they cannot test farming suppression. Here we hand-
build eight **diagnostic / counterfactual** trajectory classes (NOT environment performance rollouts) that
deliberately decouple contact/proximity from delivery, and score the three reward variants (``original``,
``mw_in_place_off``, ``monitor_aligned``) on each. No env, no PPO — pure trajectory scoring.

Each step carries both the MetaWorld reward components (``in_place_reward``, ``grasp_reward``, ``near_object``,
``obj_to_target``) so ``original``/``off`` (declared HyMeKo weights) are computable, and the geometry/monitor
signals (hand-object distance, object height, obj→target, grasp/near/success) so ``monitor_aligned`` is computable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.eval.cip.monitor_aligned_reward import monitor_aligned_components, monitor_aligned_step

_N = 20                      # steps per trajectory


def _step(ip: float, gr: float, near: float, grasp: float, ott: float, oz: float, ho: float,
          succ: float = 0.0) -> "dict[str, float]":
    """One trajectory step: reward components (ip/gr/near/ott) + geometry/monitor (oz/ho/grasp/near/success)."""
    return {"in_place": ip, "grasp_reward": gr, "near": near, "grasp": grasp, "ott": ott, "oz": oz, "ho": ho,
            "success": succ}


def _lin(a: float, b: float, i: int, n: int = _N) -> float:
    return a + (b - a) * i / (n - 1)


def _traj_far() -> "list[dict[str, float]]":
    return [_step(0.05, 0.0, 0, 0, 0.30, 0.02, 0.50) for _ in range(_N)]


def _traj_approach() -> "list[dict[str, float]]":
    out = []
    for i in range(_N):
        ho = _lin(0.50, 0.05, i)
        out.append(_step(0.05, min(0.3, 0.50 - ho), 1.0 if ho < 0.06 else 0.0, 0, 0.30, 0.02, ho))
    return out


def _traj_hover_farm() -> "list[dict[str, float]]":
    return [_step(0.15, 0.4, 1, 0, 0.30, 0.02, 0.03) for _ in range(_N)]


def _traj_bare_contact_farm() -> "list[dict[str, float]]":
    return [_step(0.20, 0.7, 1, 1, 0.30, 0.02, 0.02) for _ in range(_N)]


def _traj_grasp_lift_no_delivery() -> "list[dict[str, float]]":
    return [_step(0.30, 0.7, 1, 1, 0.30, 0.02 + min(0.13, i * 0.02), 0.02) for i in range(_N)]


def _traj_delivery() -> "list[dict[str, float]]":
    return [_step(_lin(0.30, 0.80, i), 0.7, 1, 1, _lin(0.30, 0.08, i), 0.15, 0.02) for i in range(_N)]


def _traj_success() -> "list[dict[str, float]]":
    return [_step(_lin(0.30, 1.0, i), 0.7, 1, 1, _lin(0.30, 0.02, i), 0.15, 0.02, 1.0 if i >= _N - 2 else 0.0)
            for i in range(_N)]


def _traj_proxy_exploit() -> "list[dict[str, float]]":
    return [_step(0.70, 0.7, 1, 1, 0.30, 0.02, 0.02) for _ in range(_N)]   # high in_place PROXY, zero delivery


TRAJECTORIES = {"far": _traj_far, "approach": _traj_approach, "hover_farm": _traj_hover_farm,
                "bare_contact_farm": _traj_bare_contact_farm, "grasp_lift_no_delivery": _traj_grasp_lift_no_delivery,
                "delivery": _traj_delivery, "success": _traj_success, "proxy_exploit": _traj_proxy_exploit}
_FARMING = ("hover_farm", "bare_contact_farm", "proxy_exploit")


def _declared_weights(spec: str) -> "tuple[float, float, float, float]":
    from hymeko_rl.env.reward import read_reward_terms
    w = dict(read_reward_terms(spec))
    return w["mw_in_place"], w["mw_grasp"], w["mw_near"], w["mw_dist"]


def _variant_rewards(traj: "list[dict[str, float]]", wts: "tuple[float, float, float, float]",
                     ) -> "tuple[dict[str, np.ndarray], list[dict[str, float]]]":
    """Per-step reward arrays for the three variants + the monitor_aligned per-step component breakdown."""
    w_ip, w_gr, w_near, w_dist = wts
    orig, off, ma = [], [], []
    comps: list[dict[str, float]] = []
    d_prev, oz0, ott_prev, oz_prev = traj[0]["ho"], traj[0]["oz"], traj[0]["ott"], traj[0]["oz"]
    for s in traj:
        orig.append(w_ip * s["in_place"] + w_gr * s["grasp_reward"] + w_near * s["near"] - w_dist * s["ott"])
        off.append(w_gr * s["grasp_reward"] + w_near * s["near"] - w_dist * s["ott"])
        sig = {"d": s["ho"], "d_prev": d_prev, "obj_z": s["oz"], "obj_z0": oz0, "obj_z_prev": oz_prev, "ott": s["ott"],
               "ott_prev": ott_prev, "grasp": s["grasp"], "near": s["near"], "success": s["success"]}
        ma.append(monitor_aligned_step(sig))
        comps.append(monitor_aligned_components(sig))
        d_prev, ott_prev, oz_prev = s["ho"], s["ott"], s["oz"]
    return ({"original": np.asarray(orig), "mw_in_place_off": np.asarray(off), "monitor_aligned": np.asarray(ma)},
            comps)


def _first_success(traj: "list[dict[str, float]]") -> int:
    for i, s in enumerate(traj):
        if s["success"] > 0.5:
            return i
    return len(traj)


def _component_entropy(comps: "list[dict[str, float]] | None") -> "float | None":
    """Entropy of the monitor_aligned component |contribution| shares over the steps — higher = more balanced shaping."""
    if not comps:
        return None
    keys = ("approach", "contact", "lift", "delivery", "success", "stagnation")
    shares = np.asarray([sum(abs(c[k]) for c in comps) for k in keys])
    tot = float(shares.sum())
    if tot < 1e-9:
        return 0.0
    p = shares / tot
    p = p[p > 0]
    return round(float(-(p * np.log(p)).sum()), 3)


def _class_metrics(traj: "list[dict[str, float]]", reward: np.ndarray, comps: "list[dict[str, float]] | None",
                   fs: int) -> "dict[str, Any]":
    """Reward + density diagnostics for one (trajectory class, variant)."""
    pre = reward[:fs] if fs < len(reward) else reward
    succ_bonus = float(np.sum([c["success"] for c in comps])) if comps else 0.0
    return {"total": round(float(reward.sum()), 3), "mean": round(float(reward.mean()), 4),
            "max": round(float(reward.max()), 3), "nonzero_fraction": round(float(np.mean(np.abs(reward) > 1e-6)), 3),
            "pre_success_total": round(float(pre.sum()), 3), "success_bonus": round(succ_bonus, 3),
            "monitor_success": int(any(s["success"] > 0.5 for s in traj)),
            "delivery_score": round(traj[0]["ott"] - traj[-1]["ott"], 3),
            "progress_score": round(float(np.mean([s["in_place"] for s in traj])), 3),
            "dense_pre_success": round(float(np.mean(np.abs(pre) > 1e-6)), 3),
            "pre_success_reward_std": round(float(np.std(pre)), 4),
            "pre_success_reward_mean_abs": round(float(np.mean(np.abs(pre))), 4),
            "component_entropy": _component_entropy(comps[:fs] if (comps and fs < len(comps)) else comps)}


def _global_metrics(totals: "dict[str, dict[str, float]]", quality: "dict[str, float]") -> "dict[str, Any]":
    """Per variant: reward↔monitor disagreement over the 8 classes + farming-suppression ratios."""
    from hymeko_rl.eval.task_monitor.consistency import RewardConsistencyMonitor
    out: dict[str, Any] = {}
    for variant, t in totals.items():
        rows = [{"policy": c, "total_reward": t[c], "monitor_score": quality[c]} for c in t]
        dis = round(1.0 - float(RewardConsistencyMonitor().check_reward_alignment(rows).score), 4)
        farm_avg = float(np.mean([t[c] for c in _FARMING]))
        out[variant] = {"disagreement": dis,
                        "proxy_over_success_ratio": round(t["proxy_exploit"] / t["success"], 3) if t["success"] else None,
                        "farming_over_delivery_ratio": round(farm_avg / t["delivery"], 3) if t["delivery"] else None}
    return out


def _verdict(per_class: "dict[str, Any]", glob: "dict[str, Any]") -> "dict[str, Any]":
    """The five health checks for monitor_aligned (see the report)."""
    def ma(c: str, k: str) -> float:
        return float(per_class[c]["monitor_aligned"][k])
    totals = {c: ma(c, "total") for c in TRAJECTORIES}
    dense = all(per_class[c]["monitor_aligned"]["dense_pre_success"] > 0.5
                and per_class[c]["monitor_aligned"]["pre_success_reward_std"] > 0.0 for c in ("approach", "delivery"))
    delivery_beats_farm = all(totals["delivery"] > totals[f] for f in _FARMING)
    success_strongest = totals["success"] == max(totals.values())
    farm_low = all(totals[f] < 0.5 * totals["delivery"] for f in _FARMING)
    lower_disagree = glob["monitor_aligned"]["disagreement"] <= glob["original"]["disagreement"]
    checks = {"1_dense_pre_success_shaping": bool(dense), "2_delivery_beats_farming": bool(delivery_beats_farm),
              "3_success_strongest": bool(success_strongest), "4_farming_reward_capped_low": bool(farm_low),
              "5_lower_disagreement_than_original": bool(lower_disagree)}
    return {"checks": checks, "healthy": all(checks.values())}


def run_anti_farming_validation(spec: str = "data/robotics/metaworld_reward.hymeko", out_dir: Any = None,
                                ) -> "dict[str, Any]":
    """Score original / mw_in_place_off / monitor_aligned on the eight decoupled counterfactual trajectory classes."""
    out = Path(out_dir) if out_dir is not None else Path("reports/figures/2026_07_10_anti_farming_validation")
    out.mkdir(parents=True, exist_ok=True)
    wts = _declared_weights(spec)
    variants = ("original", "mw_in_place_off", "monitor_aligned")
    per_class: dict[str, Any] = {}
    quality: dict[str, float] = {}
    for cname, builder in TRAJECTORIES.items():
        traj = builder()
        fs = _first_success(traj)
        rewards, comps = _variant_rewards(traj, wts)
        per_class[cname] = {v: _class_metrics(traj, rewards[v], comps if v == "monitor_aligned" else None, fs)
                            for v in variants}
        quality[cname] = round((traj[0]["ott"] - traj[-1]["ott"]) + 2.0 * float(any(s["success"] > 0.5 for s in traj)), 3)
    totals = {v: {c: per_class[c][v]["total"] for c in TRAJECTORIES} for v in variants}
    glob = _global_metrics(totals, quality)
    verdict = _verdict(per_class, glob)
    summary = {"classes": list(TRAJECTORIES), "declared_weights": {"in_place": wts[0], "grasp": wts[1],
               "near": wts[2], "dist": wts[3]}, "per_class": per_class, "task_quality": quality,
               "global": glob, "verdict": verdict}
    (out / "anti_farming_validation.json").write_text(json.dumps(summary, indent=2, default=float))
    _plot(per_class, glob, out)
    for c in TRAJECTORIES:
        print(f"[anti-farm] {c:22s} total  orig {per_class[c]['original']['total']:8.2f} | "
              f"off {per_class[c]['mw_in_place_off']['total']:8.2f} | ma {per_class[c]['monitor_aligned']['total']:8.2f}",
              flush=True)
    print(f"[anti-farm] disagreement  orig {glob['original']['disagreement']} off "
          f"{glob['mw_in_place_off']['disagreement']} ma {glob['monitor_aligned']['disagreement']} | "
          f"proxy/success ma {glob['monitor_aligned']['proxy_over_success_ratio']} vs orig "
          f"{glob['original']['proxy_over_success_ratio']} | HEALTHY={verdict['healthy']}", flush=True)
    return summary


def _plot(per_class: "dict[str, Any]", glob: "dict[str, Any]", out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    classes = list(TRAJECTORIES)
    variants = ("original", "mw_in_place_off", "monitor_aligned")
    cols = {"original": "#2c3e50", "mw_in_place_off": "#c0392b", "monitor_aligned": "#2c8"}
    x = np.arange(len(classes))
    w = 0.27
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for j, v in enumerate(variants):
        ax[0].bar(x + (j - 1) * w, [per_class[c][v]["total"] for c in classes], w, label=v, color=cols[v])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(classes, rotation=25, ha="right", fontsize=8)
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_ylabel("total trajectory reward")
    ax[0].set_title("Total reward per counterfactual trajectory class (farming should score LOW)")
    ax[0].legend(fontsize=8)
    ratios = ["proxy_over_success_ratio", "farming_over_delivery_ratio"]
    xr = np.arange(len(ratios))
    for j, v in enumerate(variants):
        ax[1].bar(xr + (j - 1) * w, [glob[v][r] if glob[v][r] is not None else 0 for r in ratios], w, color=cols[v], label=v)
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_xticks(xr)
    ax[1].set_xticklabels(["proxy / success", "farming / delivery"])
    ax[1].set_ylabel("reward ratio (lower = better suppression)")
    ax[1].set_title("Farming suppression — monitor_aligned should be lowest")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "anti_farming_validation.png", dpi=130)
    plt.close(fig)
