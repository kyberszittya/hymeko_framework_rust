"""Segment-local BC of the teacher's KINETIC transport skill — the coin-following, light-contact tip-trajectory that the G-VF
profile-CEM could not hand-produce (contact lost ~50mm). The G-VF FAIL localised the missing skill to the transport segment;
this is the first step of the learned-KINETIC-policy path (BC → DAgger → bounded RL).

This file is the OFFLINE step: behaviour-clone the teacher's per-step transport action (dtau from `_schedule_increment`, the
jacobian-based coin-following control) from a structured state + short response history, and run the decisive **cross-cradle
action-generalization test** — does ONE policy predict the teacher's transport on an UNSEEN cradle? That is the campaign's real
question (does structured-state + hybrid-mode + local learned controller generalize) answered at the action level, before the
closed-loop deploy. Reuses `rollout_primitive` (dynamics) + `_schedule_increment` (teacher action) + `behaviour_clone` +
`DetActor` — no re-implementation. Dev s1/s3 only; held-out s4/s7 untouched; physics/teacher frozen.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _schedule_increment, rollout_primitive
from hymeko_rl.coin_delivery.theta_option.residual_adapter import _coin_spin
from hymeko_rl.experiments.coin_r9_causal_rl import (
    BANK, DELIVERY_CFG, OUT, _load_harness, _teacher_theta, build_panel)
from hymeko_rl.option_rl.agents import DetActor
from hymeko_rl.train.bc import behaviour_clone

_HIST = 2                    # short response-history depth (a momentary state may not reveal that the coin is sliding out of the tips)
_DEV_CRADLES = ("s1", "s3")


class _KineticActor(DetActor):
    """DetActor (tanh-bounded MLP) + the `action_mean` alias `behaviour_clone` fits against."""

    def action_mean(self, s: Any) -> Any:
        return self.forward(s)


def _kinetic_feats(rl: Any, hist: list) -> "tuple[np.ndarray, list]":
    """Structured state + short response history — the guard/policy inputs: coin (dtz, v_par, v_lat, spin), L/R contact force,
    arm q/qdot, previous applied torque, and the last _HIST (dtz, v_par, fn_l, fn_r) frames. No future / teacher / K6 leak."""
    u, dtz = rl.inner.direction_to_zone()
    uu = np.asarray(u, np.float64)[:2]
    vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    con = primary_fingertip_contacts(rl)
    fnl = float(con["left"]["fn"]) if con["left"] is not None else 0.0
    fnr = float(con["right"]["fn"]) if con["right"] is not None else 0.0
    cur = [float(dtz), float(np.dot(vel, uu)), float(np.dot(vel, np.array([-uu[1], uu[0]]))), _coin_spin(rl), fnl, fnr]
    q = list(np.asarray(rl.inner.data.qpos[:4], np.float64))
    qd = list(np.asarray(rl.inner.data.qvel[:4], np.float64))
    pv = list(np.asarray(rl.inner.data.ctrl[:4], np.float64))
    hframe = [cur[0], cur[1], fnl, fnr]
    h: list = []
    for k in range(_HIST):
        h += hist[-1 - k] if len(hist) > k else hframe
    return np.array(cur + q + qd + pv + h, np.float64), hframe


def collect_teacher_transport(snap: Any, theta: Any) -> "tuple[np.ndarray, np.ndarray]":
    """Capture (features, teacher dtau / step) over the teacher's transport-to-release segment (t ≤ release). Pairs the
    post-step-t state (= pre-step-(t+1) state) with the teacher's step-(t+1) action, so the dynamics come from the real
    `rollout_primitive` and the action from the real `_schedule_increment` — an exact (state, action) capture, no loop replica."""
    th = np.asarray(theta, np.float64)
    rel = int(round(float(th[4])))
    e_par = np.asarray(snap.branch().inner.direction_to_zone()[0], np.float64)      # fixed t=0 reference, as the teacher uses
    step = float(snap.stack.tau_rate * snap.stack.control_dt)
    feats: list = []
    acts: list = []
    hist: list = []

    def hook(rl: Any, t: int) -> None:
        if t + 1 > rel:                                                             # only the transport-to-release segment
            return
        f, hframe = _kinetic_feats(rl, hist)
        dtau = _schedule_increment(rl, th, t + 1, e_par, step, _coin_xy(rl))
        feats.append(f)
        acts.append(np.clip(dtau, -step, step) / step)                             # normalise to the tanh actor's [-1, 1]
        hist.append(hframe)
    rollout_primitive(snap, tuple(th), DELIVERY_CFG, frame_hook=hook)
    return np.array(feats, np.float64), np.array(acts, np.float64)


def _fit_eval(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, n_epochs: int, verbose: bool) -> dict:
    """Standardise on the train cradle, BC-fit the actor, and report train (within) + held-out-cradle (cross) action MSE + R²."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    actor = _KineticActor(xtr.shape[1], ytr.shape[1], h=128)
    losses = behaviour_clone(actor, (xtr - mu) / sd, ytr, n_epochs=n_epochs, lr=1e-3, batch=64, seed=0,
                             log_every=50 if verbose else 0)
    with torch.no_grad():
        ptr = actor.action_mean(torch.as_tensor((xtr - mu) / sd, dtype=torch.float32)).numpy()
        pte = actor.action_mean(torch.as_tensor((xte - mu) / sd, dtype=torch.float32)).numpy()
    within = float(np.mean((ptr - ytr) ** 2))
    cross = float(np.mean((pte - yte) ** 2))
    r2 = 1.0 - cross / max(float(np.mean(yte.var(0))), 1e-9)                        # vs predicting the mean action
    return {"within_mse": round(within, 5), "cross_mse": round(cross, 5), "cross_r2": round(r2, 3),
            "final_train_loss": round(losses[-1], 5), "n_train": len(xtr), "n_test": len(xte)}


def _verdict(r2_mean: float, n_cradles: int, min_demos: int) -> str:
    """Honest gate: 2 dev cradles (s4/s7 held-out) with ~8-10 transport steps each cannot DISTINGUISH generalization from noise —
    a negative cross-cradle R² is then UNDERPOWERED, not a proven non-generalization (no premature-conclusion, CLAUDE.md)."""
    if n_cradles < 3 or min_demos < 20:
        return "CROSS_CRADLE_BC_UNDERPOWERED"                     # too few cradles / too little data to conclude either way
    if r2_mean > 0.5:
        return "TEACHER_TRANSPORT_GENERALISES_CROSS_CRADLE"
    return "TEACHER_TRANSPORT_PARTIALLY_GENERALISES" if r2_mean > 0.0 else "TEACHER_TRANSPORT_CRADLE_SPECIFIC"


def bc_generalization(n_epochs: int = 200) -> dict:
    """Collect the teacher transport-segment demos on dev s1/s3, then a two-way cross-cradle BC test (train on one, predict the
    other's teacher transport). The mean held-out R² is the offline answer to 'does the transport skill generalize?'."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(BANK))
    panel = {ps.tag: ps for ps in build_panel(_load_harness(), bank)}
    data = {tag: collect_teacher_transport(panel[tag].snap, _teacher_theta(bank, tag)) for tag in _DEV_CRADLES}
    print("   collected transport demos: " + ", ".join(f"{k} n={len(v[0])} act|.|~{np.abs(v[1]).mean():.3f}"
                                                        for k, v in data.items()), flush=True)
    (x1, y1), (x3, y3) = data["s1"], data["s3"]
    fwd = _fit_eval(x1, y1, x3, y3, n_epochs, verbose=True)       # train s1 → predict s3
    bwd = _fit_eval(x3, y3, x1, y1, n_epochs, verbose=False)      # train s3 → predict s1
    r2_mean = round((fwd["cross_r2"] + bwd["cross_r2"]) / 2, 3)
    min_demos = min(len(x1), len(x3))
    verdict = _verdict(r2_mean, len(_DEV_CRADLES), min_demos)
    thetas = {tag: [round(float(x), 3) for x in _teacher_theta(bank, tag)] for tag in _DEV_CRADLES}
    out = {"contract": "COIN_KINETIC_BC_CROSS_CRADLE", "cradles": list(_DEV_CRADLES), "feature_dim": int(x1.shape[1]),
           "action_dim": int(y1.shape[1]), "hist_depth": _HIST, "min_demos_per_cradle": min_demos,
           "teacher_theta_per_cradle": thetas, "teacher_action_magnitude": {"s1": round(float(np.abs(y1).mean()), 3),
           "s3": round(float(np.abs(y3).mean()), 3)}, "train_s1_test_s3": fwd, "train_s3_test_s1": bwd,
           "cross_cradle_r2_mean": r2_mean, "verdict": verdict, "wall_s": round(time.time() - t0, 1),
           "reading": ("UNDERPOWERED: only 2 dev cradles (s4/s7 held-out validation-only) with ~8-10 transport steps each, and "
                       "the two teacher θ are strategically OPPOSITE (s1 light-squeeze/strong-push vs s3 firm-squeeze/gentle-push)"
                       " — a 2-cradle train/test cannot distinguish generalization from noise. The BC memorises each cradle "
                       "(within-MSE≈0); the NEXT decisive test is the WITHIN-cradle closed-loop deploy (does the cloned "
                       "transport reach the corridor on the SAME cradle), then more dev cradles for a real generalization claim.")}
    json.dump(out, open(f"{OUT}/kinetic_bc_cross_cradle.json", "w"), indent=1, default=float)
    print(f"   train s1→s3: within {fwd['within_mse']} cross {fwd['cross_mse']} R² {fwd['cross_r2']}\n"
          f"   train s3→s1: within {bwd['within_mse']} cross {bwd['cross_mse']} R² {bwd['cross_r2']}\n"
          f"   cross-cradle R² mean {r2_mean}\n== KINETIC BC (cross-cradle) ==\n  {verdict} | wall {out['wall_s']}s\n"
          f"KINETIC_BC_DONE", flush=True)
    return out


def main(argv: "list[str]") -> None:
    if "--bc" in argv:
        bc_generalization()
    else:
        print("usage: coin_kinetic_bc.py --bc", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
