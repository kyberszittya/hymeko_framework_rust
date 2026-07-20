"""COIN-DELIVERY-RL-1 — post-handoff transport & centering (PPO residual around grasp_carry).

The first delivery-CORRECTED RL experiment. Under the delivery semantics established by DELIVERY-ENV-0
(F-COIN-DELIVERY-SEMANTICS): handoff is a phase event, center-reach (disk_to_zone <= 0.02) is task success, horizon 120.
A scripted grasp_carry acquisition prefix runs to handoff; a PPO **residual** policy then controls the transport suffix.

    a_exec = clip(a_grasp_carry(inner) + delta*tanh(policy(obs)), lo, hi)   (zero residual == scripted, tested Stage 0)

PPO is chosen first (on-policy) to avoid depending on an unvetted off-policy critic (the SAC OOD-overestimation risk).
Staged: Stage 0 hard sanity gates → Stage 1 single-seed PPO smoke (50k-100k post-handoff steps) → Stage 2 3-seed
confirmation (only if promising). NO env/reward/dynamics/CORE change; the training harness is non-invasive. No handoff
success claim; no delivery-solved claim from one seed; no kato15 until a local 3-seed positive.

Reuses ``train.ppo.train_ppo`` (unchanged), ``agents.policy.build_policy``, ``train.coin_delivery_rl`` (the harness).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.coin_delivery_rl import (
    DeliveryRLConfig,
    delivery_reward,
    eval_delivery,
    greedy_action_fn,
    make_delivery_rl_env,
    residual_action,
    scripted_action_fn,
    zero_residual_head,
)
from hymeko_rl.train.ppo import PPOConfig, train_ppo

_OUT = Path("experiments/2026_07_20_coin_delivery_rl1")
_HELD = range(64_000, 64_090)
_OBS_DIM = 41           # len(ACTOR_FIELDS)
_ACT_DIM = 6


def _log(m: str) -> None:
    print(m, flush=True)


# ── Stage 0 — hard sanity gates ────────────────────────────────────────────────────────────────────────────────────
def _reward_gates(cfg: DeliveryRLConfig) -> dict[str, Any]:
    """Gates 4-7 on the pure reward: progress>0, holding<=0, away<0, center is the strongest event."""
    prog = delivery_reward(0.10, 0.088, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    hold = delivery_reward(0.10, 0.100, entered_zone_now=False, center_now=False, stalled=True, dropped=False, cfg=cfg)
    away = delivery_reward(0.10, 0.112, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    zone = delivery_reward(0.05, 0.039, entered_zone_now=True, center_now=False, stalled=False, dropped=False, cfg=cfg)
    cent = delivery_reward(0.03, 0.019, entered_zone_now=False, center_now=True, stalled=False, dropped=False, cfg=cfg)
    return {"progress": round(prog, 4), "holding": round(hold, 4), "away": round(away, 4),
            "zone_event": round(zone, 4), "center_event": round(cent, 4),
            "g4_progress_pos": prog > 0, "g5_holding_nonpos": hold <= 0, "g6_away_neg": away < 0,
            "g7_center_strongest": cent > zone > prog > 0}


def _handoff_not_terminal(cfg: DeliveryRLConfig, env: Any, seed: int = 64_000) -> dict[str, Any]:
    """Gate 2: the episode CONTINUES past the handoff phase event (suffix steps run after handoff)."""
    obs, info = env.reset(seed=seed)
    handoff_at_reset = bool(info["handoff_event"])
    steps = 0
    af = scripted_action_fn()
    for _ in range(cfg.horizon):
        obs, _r, term, trunc, _i = env.step(af(obs))
        steps += 1
        if term or trunc:
            break
    return {"handoff_reached_in_prefix": handoff_at_reset, "suffix_steps_ran": steps,
            "g2_handoff_not_terminal": handoff_at_reset and steps > 1}


def _zero_residual_matches(cfg: DeliveryRLConfig, held) -> dict[str, Any]:
    """Gate 1: zero residual == scripted grasp_carry, both as a pure action identity and end-to-end metric identity."""
    base = np.array([0.6, -0.4, 0.0, 0.8, 0.0, 0.0], np.float32)
    identity = bool(np.allclose(residual_action(base, np.zeros(_ACT_DIM, np.float32), cfg.delta, cfg.lo, cfg.hi),
                                np.clip(base, cfg.lo, cfg.hi)))
    ac = _build_residual_policy(cfg, seed=0)                 # zero-init head → greedy action_mean == 0 everywhere
    seeds = list(held)[:20]
    scripted = eval_delivery(scripted_action_fn(), seeds, cfg)
    zero_init = eval_delivery(greedy_action_fn(ac), seeds, cfg)
    same = (abs(scripted["center_reach"] - zero_init["center_reach"]) < 1e-9
            and abs(scripted["zone_entry"] - zero_init["zone_entry"]) < 1e-9
            and zero_init["residual_norm_med"] in (0.0, None))
    return {"action_identity": identity, "scripted_center": scripted["center_reach"],
            "zero_init_center": zero_init["center_reach"], "zero_init_residual_norm": zero_init["residual_norm_med"],
            "g1_zero_residual_is_scripted": bool(identity and same)}


def stage0_gates(cfg: DeliveryRLConfig, held=_HELD) -> dict[str, Any]:
    _log("=== Stage 0 — hard sanity gates ===")
    env = make_delivery_rl_env(cfg)
    scripted = eval_delivery(scripted_action_fn(), held, cfg, env=env)
    g1 = _zero_residual_matches(cfg, held)
    g2 = _handoff_not_terminal(cfg, env)
    rg = _reward_gates(cfg)
    # Gate 3: scripted reaches ~zone 0.73 / center 0.52-0.56 (DELIVERY-ENV-0 reference; tolerant band)
    g3 = bool(scripted["zone_entry"] >= 0.55 and 0.45 <= scripted["center_reach"] <= 0.75)
    gates = {"g1": g1["g1_zero_residual_is_scripted"], "g2": g2["g2_handoff_not_terminal"], "g3_scripted_band": g3,
             "g4": rg["g4_progress_pos"], "g5": rg["g5_holding_nonpos"], "g6": rg["g6_away_neg"],
             "g7": rg["g7_center_strongest"]}
    all_pass = all(gates.values())
    out = {"scripted_full_held": {k: scripted[k] for k in ("n", "zone_entry", "center_reach", "handoff_event",
                                                            "contact_lost", "grasp_no_delivery", "final_dtz_med",
                                                            "time_to_center_med", "return_med")},
           "gate1_zero_residual": g1, "gate2_handoff": g2, "reward_gates": rg, "gates": gates, "all_pass": all_pass}
    for k, v in gates.items():
        _log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    _log(f"  scripted (n={scripted['n']}): zone={scripted['zone_entry']} center={scripted['center_reach']} "
         f"final_dtz_med={scripted['final_dtz_med']} → Stage-0 {'PASS' if all_pass else 'FAIL'}")
    return out


# ── policy + PPO ────────────────────────────────────────────────────────────────────────────────────────────────────
def _build_residual_policy(cfg: DeliveryRLConfig, *, seed: int) -> Any:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ac = build_policy("mlp", obs_dim=_OBS_DIM, action_dim=_ACT_DIM, log_std_init=-1.6)
    zero_residual_head(ac)                                   # start AT the scripted controller (zero residual)
    return ac


def _ppo_config(steps: int, seed: int) -> PPOConfig:
    n_steps = 1024
    return PPOConfig(n_iters=max(1, steps // n_steps), n_steps=n_steps, lr=3e-4, clip=0.2,
                     update_epochs=8, ent_coef=0.0, seed=seed)


def _train_residual(cfg: DeliveryRLConfig, steps: int, seed: int) -> tuple[Any, list[dict]]:
    ac = _build_residual_policy(cfg, seed=seed)
    env = make_delivery_rl_env(cfg)
    pcfg = _ppo_config(steps, seed)
    metrics: list[dict] = []
    t0 = time.perf_counter()

    def _on_iter(i: int, n: int) -> None:                   # live observability (§3): never run blind
        if metrics and (i % 5 == 0 or i == n - 1):
            m = metrics[-1]
            el = time.perf_counter() - t0
            eta = el / max(1, i) * (n - i)
            _log(f"  [ppo {i}/{n}] return={m['return']:+.3f} ploss={m['policy_loss']:+.3f} "
                 f"vloss={m['value_loss']:.3f} clip={m['clip_frac']:.3f} std={m['action_std']:.3f} "
                 f"| {el:.0f}s ETA {eta:.0f}s")

    _log(f"=== training PPO residual ({pcfg.n_iters}×{pcfg.n_steps}={pcfg.n_iters * pcfg.n_steps} steps, seed {seed}) ===")
    train_ppo(ac, env, pcfg, on_iteration=_on_iter, metrics_out=metrics)
    _log(f"  trained in {time.perf_counter() - t0:.0f}s ({len(metrics)} iters)")
    return ac, metrics


# ── Stage 1 — single-seed PPO smoke ──────────────────────────────────────────────────────────────────────────────────
def _promising(scripted: dict, trained: dict) -> tuple[bool, dict]:
    """Primary gate: center-reach up, OR equal center-reach with materially better precision/time — zone-entry kept."""
    center_up = trained["center_reach"] > scripted["center_reach"] + 1e-9
    zone_kept = trained["zone_entry"] >= scripted["zone_entry"] - 0.05
    center_equal = abs(trained["center_reach"] - scripted["center_reach"]) <= 0.02
    better_prec = (trained["final_dtz_med"] is not None and scripted["final_dtz_med"] is not None
                   and trained["final_dtz_med"] < scripted["final_dtz_med"] - 0.002)
    faster = (trained["time_to_center_med"] is not None and scripted["time_to_center_med"] is not None
              and trained["time_to_center_med"] < scripted["time_to_center_med"] - 3)
    checks = {"center_up": center_up, "zone_kept": zone_kept,
              "equal_center_better_precision_or_time": center_equal and (better_prec or faster)}
    return bool(zone_kept and (center_up or checks["equal_center_better_precision_or_time"])), checks


_EVAL_KEYS = ("n", "zone_entry", "center_reach", "handoff_event", "contact_lost", "grasp_no_delivery",
              "final_dtz_med", "min_dtz_med", "time_to_zone_med", "time_to_center_med", "return_med",
              "residual_norm_med", "saturation_med")


def _slim(ev: dict) -> dict:
    return {k: ev[k] for k in _EVAL_KEYS}


def stage1_smoke(cfg: DeliveryRLConfig, steps: int, seed: int, held=_HELD) -> dict[str, Any]:
    _log(f"\n=== Stage 1 — single-seed PPO smoke (seed {seed}, {steps} steps) ===")
    env = make_delivery_rl_env(cfg)
    scripted = eval_delivery(scripted_action_fn(), held, cfg, env=env)
    ac0 = _build_residual_policy(cfg, seed=seed)
    zero_init = eval_delivery(greedy_action_fn(ac0), held, cfg, env=env)
    ac, metrics = _train_residual(cfg, steps, seed)
    trained = eval_delivery(greedy_action_fn(ac), held, cfg, env=env)
    ok, checks = _promising(scripted, trained)
    _log(f"  scripted    zone={scripted['zone_entry']} center={scripted['center_reach']} "
         f"final_dtz={scripted['final_dtz_med']} t2center={scripted['time_to_center_med']}")
    _log(f"  zero-init   zone={zero_init['zone_entry']} center={zero_init['center_reach']} (== scripted expected)")
    _log(f"  trained     zone={trained['zone_entry']} center={trained['center_reach']} "
         f"final_dtz={trained['final_dtz_med']} t2center={trained['time_to_center_med']} "
         f"resid={trained['residual_norm_med']} sat={trained['saturation_med']}")
    _log(f"  → promising={ok} {checks}")
    return {"seed": seed, "steps": steps, "scripted": _slim(scripted), "zero_init": _slim(zero_init),
            "trained": _slim(trained), "promising": ok, "gate_checks": checks,
            "train_curve": [round(m["return"], 4) for m in metrics], "ac": ac}


# ── Stage 2 — 3-seed confirmation (conditional) ──────────────────────────────────────────────────────────────────────
def stage2_confirm(cfg: DeliveryRLConfig, steps: int, seeds: tuple[int, ...], held=_HELD) -> dict[str, Any]:
    _log(f"\n=== Stage 2 — {len(seeds)}-seed confirmation ===")
    env = make_delivery_rl_env(cfg)
    scripted = eval_delivery(scripted_action_fn(), held, cfg, env=env)
    trained_evals = []
    for sd in seeds:
        ac, _m = _train_residual(cfg, steps, sd)
        ev = eval_delivery(greedy_action_fn(ac), held, cfg, env=env)
        trained_evals.append(ev)
        _log(f"  seed {sd}: zone={ev['zone_entry']} center={ev['center_reach']} final_dtz={ev['final_dtz_med']}")

    def _mi(key: str) -> dict:
        vals = [e[key] for e in trained_evals if e[key] is not None]
        return {"median": round(float(np.median(vals)), 4), "iqr": round(float(np.subtract(*np.percentile(vals, [75, 25]))), 4)} if vals else {}

    med = {k: _mi(k) for k in ("zone_entry", "center_reach", "final_dtz_med", "time_to_center_med", "return_med")}
    center_med = med["center_reach"].get("median", 0.0)
    zone_med = med["zone_entry"].get("median", 0.0)
    passed = bool(zone_med >= 0.70 and center_med > scripted["center_reach"] + 1e-9)
    _log(f"  scripted center={scripted['center_reach']} | trained median zone={zone_med} center={center_med} "
         f"→ Stage-2 {'PASS' if passed else 'no-improvement'}")
    return {"seeds": list(seeds), "steps": steps, "scripted": _slim(scripted),
            "trained_per_seed": [_slim(e) for e in trained_evals], "median_iqr": med,
            "gate_median_zone_ge_070_and_center_gt_scripted": passed}


# ── graphical output (§9) ────────────────────────────────────────────────────────────────────────────────────────────
def _plot(s1: dict, out: Path) -> "str | None":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:  # noqa: BLE001
        _log(f"  [plot] matplotlib unavailable: {ex}")
        return None
    arms = ["scripted", "zero_init", "trained"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, (key, ttl) in enumerate((("zone_entry", "zone-entry rate"), ("center_reach", "center-reach rate"))):
        vals = [s1[a][key] for a in arms]
        ax[j].bar(arms, vals, color=["#8d99ae", "#adb5bd", "#2a9d8f"], edgecolor="black", zorder=3)
        for i, v in enumerate(vals):
            ax[j].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax[j].set_title(ttl)
        ax[j].set_ylim(0, 1.0)
        ax[j].grid(axis="y", alpha=0.3)
    curve = s1.get("train_curve", [])
    ax[2].plot(range(len(curve)), curve, color="#2a9d8f", lw=1.6)
    ax[2].set_title("PPO training return (per iter)")
    ax[2].set_xlabel("iteration")
    ax[2].grid(alpha=0.3)
    fig.suptitle("COIN-DELIVERY-RL-1 — PPO residual vs scripted grasp_carry (delivery semantics, center_tol 0.02)",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "coin_delivery_rl1.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def _gif(cfg: DeliveryRLConfig, ac: Any, out: Path, seeds: tuple[int, ...] = (64_005, 64_009)) -> "str | None":
    """Side-by-side scripted vs trained-residual rollout under delivery semantics (§9 animated output)."""
    try:
        import imageio.v2 as imageio
        import mujoco
    except Exception as ex:  # noqa: BLE001
        _log(f"  [gif] deps unavailable: {ex}")
        return None
    from hymeko_rl.train.coin_delivery_rl import greedy_action_fn as _gaf, scripted_action_fn as _saf
    env = make_delivery_rl_env(cfg)
    inner = env.inner
    rend = mujoco.Renderer(inner.model, height=300, width=380)

    def roll_frames(action_fn: Any, seed: int) -> list:
        obs, _i = env.reset(seed=seed)
        frames = []
        for _ in range(cfg.horizon):
            rend.update_scene(inner.data)
            frames.append(np.asarray(rend.render(), np.uint8))
            obs, _r, term, trunc, _info = env.step(action_fn(obs))
            if term or trunc:
                break
        return frames

    all_frames: list = []
    for sd in seeds:
        fs = roll_frames(_saf(), sd)
        ft = roll_frames(_gaf(ac), sd)
        n = max(len(fs), len(ft))
        fs += [fs[-1]] * (n - len(fs))
        ft += [ft[-1]] * (n - len(ft))
        strip = np.full((n, 300, 6, 3), 20, np.uint8)
        for a, b, s in zip(fs, ft, strip):
            all_frames.append(np.concatenate([a, s, b], axis=1))
        all_frames += [all_frames[-1]] * 6
    rend.close()
    p = out / "coin_delivery_rl1_scripted_vs_trained.gif"
    imageio.mimsave(p, all_frames, fps=16, loop=0)
    _log(f"  [gif] wrote {p} (left=scripted grasp_carry, right=trained residual)")
    return str(p)


# ── driver ────────────────────────────────────────────────────────────────────────────────────────────────────────
def run(*, steps: int = 51_200, seed: int = 0, stage2_seeds: tuple[int, ...] = (0, 1, 2),
        gif: bool = True, held=_HELD) -> dict[str, Any]:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    cfg = DeliveryRLConfig()

    s0 = stage0_gates(cfg, held)
    if not s0["all_pass"]:
        out = {"stage0": s0, "verdict": "STAGE0_FAIL_gates_not_met", "wall_s": round(time.perf_counter() - t0, 1)}
        (_OUT / "manifests" / "coin_delivery_rl1.json").write_text(json.dumps(out, indent=2, default=float))
        _log("[COIN-DELIVERY-RL-1] Stage 0 FAILED — not training. See manifest.")
        return out

    s1 = stage1_smoke(cfg, steps, seed, held)
    plot = _plot(s1, _OUT)
    gif_path = _gif(cfg, s1["ac"], _OUT) if gif else None
    s2: dict[str, Any] = {}
    if s1["promising"]:
        s2 = stage2_confirm(cfg, steps, stage2_seeds, held)

    trained, scripted = s1["trained"], s1["scripted"]
    kato15_justified = bool(s2.get("gate_median_zone_ge_070_and_center_gt_scripted", False))
    verdict = _verdict(s1, s2)
    out = {"config": {"delta": cfg.delta, "horizon": cfg.horizon, "center_tol": cfg.center_tol,
                      "prefix_cap": cfg.prefix_cap, "reward_weights": {"w_progress": cfg.w_progress, "w_zone": cfg.w_zone,
                      "w_center": cfg.w_center, "w_stall": cfg.w_stall, "w_drop": cfg.w_drop}},
           "stage0": s0, "stage1": {k: v for k, v in s1.items() if k != "ac"}, "stage2": s2,
           "scripted_center": scripted["center_reach"], "trained_center": trained["center_reach"],
           "scripted_zone": scripted["zone_entry"], "trained_zone": trained["zone_entry"],
           "reward_driven_improvement": bool(trained["center_reach"] > scripted["center_reach"] + 1e-9),
           "kato15_justified": kato15_justified, "verdict": verdict, "plot": plot, "gif": gif_path,
           "single_seed_point_estimate": not bool(s2), "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "coin_delivery_rl1.json").write_text(json.dumps(out, indent=2, default=float))
    _log(f"\n[COIN-DELIVERY-RL-1] verdict={verdict} | scripted center {scripted['center_reach']} → trained "
         f"{trained['center_reach']} | kato15_justified={kato15_justified} | {out['wall_s']}s")
    return out


def _verdict(s1: dict, s2: dict) -> str:
    """Case A (center up) / B (precision-or-time up, center kept) / C (degrades) / D (matches, no improvement)."""
    sc, tr = s1["scripted"], s1["trained"]
    if tr["center_reach"] > sc["center_reach"] + 1e-9:
        return "CASE_A_rl_improves_center_reach" + ("_confirmed_3seed" if s2.get(
            "gate_median_zone_ge_070_and_center_gt_scripted") else "_single_seed")
    if s1["gate_checks"].get("equal_center_better_precision_or_time"):
        return "CASE_B_rl_refines_precision_or_time"
    if tr["center_reach"] < sc["center_reach"] - 0.02:
        return "CASE_C_rl_degrades_scripted_needs_audit"
    return "CASE_D_matches_scripted_near_local_ceiling"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="COIN-DELIVERY-RL-1 — PPO residual transport & centering")
    ap.add_argument("--steps", type=int, default=51_200, help="post-handoff PPO env steps (Stage 1 smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--stage0-only", action="store_true", help="run only the Stage-0 gates (no training)")
    a = ap.parse_args(argv)
    if a.stage0_only:
        s0 = stage0_gates(DeliveryRLConfig())
        return 0 if s0["all_pass"] else 1
    run(steps=a.steps, seed=a.seed, gif=not a.no_gif)
    return 0


if __name__ == "__main__":
    sys.exit(main())
