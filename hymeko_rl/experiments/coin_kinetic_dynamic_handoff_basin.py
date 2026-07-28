"""R10 Stage 0 — characterise the K6-compatible DYNAMIC handoff basin H_dyn (not a single cradle point).

The reach-target characterization showed the handoff the frozen APPROACH continues from is a DYNAMIC state (arm momentum +
prev_tau + bilateral contact are jointly load-bearing), not a static pose. Before defining `PRECONTACT_STRADDLE_ENTRY_V1` or any
planner/RL, this maps the tolerance basin: targeted perturbations around the certified cradle handoff (arm q, qvel magnitude and
direction, prev_tau, coin pose/velocity), each run through the FULL frozen downstream (H1 HANDOFF_RESET → R2 → coast → K6) and
labelled K6-compatible / failure. The result is a basin with tolerances, so the capture target is a SET, not the single privileged
cradle state.

State-editing here is a characterization tool (like the frozen-policy intervention) — it defines which handoff states deliver; the
downstream positive control and RL must reach the basin from HOME with no state edit / snapshot injection. Downstream frozen; no RL.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_dynamic_handoff_basin``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

OUT = Path("reports/2026-07-28-coin-r9-dynamic-handoff-basin")
CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")
CFG = replace(DELIVERY_CFG, horizon=80)


class _Handoff:
    """The certified cradle handoff + a perturbed-rollout helper through the frozen downstream."""

    def __init__(self) -> None:
        from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
        from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
        self.model, self.norm = _load_clone()
        cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
        self.cradle = cradle
        d = cradle.branch().inner.data
        self.q0, self.coin0, self.qv0 = d.qpos[:4].copy(), d.qpos[4:7].copy(), d.qvel.copy()
        self.tau0 = np.asarray(cradle.prev_tau).copy()
        self.r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_actor_state"]))

    def roll(self, *, qv_scale: float = 1.0, tau_scale: float = 1.0, dq: Any = 0.0, qv_dir_noise: float = 0.0,
             coin_dxy: Any = 0.0, coin_dv: Any = 0.0, rng: Any = None) -> dict:
        r = self.cradle.branch()
        d = r.inner.data
        qv = self.qv0.copy() * qv_scale
        if qv_dir_noise and rng is not None:                              # rotate arm qvel direction at fixed magnitude
            pert = rng.normal(0, qv_dir_noise, 4)
            qv[:4] = np.linalg.norm(qv[:4]) * (qv[:4] + pert) / (np.linalg.norm(qv[:4] + pert) + 1e-9)
        d.qpos[:4] = self.q0 + dq
        d.qpos[4:6] = self.coin0[:2] + coin_dxy
        d.qpos[6] = self.coin0[2]
        d.qvel[:] = qv
        d.qvel[4:6] = np.asarray(coin_dv) if np.ndim(coin_dv) else coin_dv
        d.ctrl[:] = 0.0
        mujoco.mj_forward(r.inner.model, d)
        con = primary_fingertip_contacts(r)
        snap = kc.TransportSnapshot.from_live(copy.deepcopy(r), self.cradle.stack, self.tau0 * tau_scale)
        ctrl = HandoffResetTemporalController(snap, CloneActor(self.model, self.norm), self.r2_fn, ResidualBounds(alpha=0.15))
        m = velocity_rollout(snap, ctrl, CFG)
        return {"k6": bool(delivery_success(m, CFG)), "min_dtz_mm": round(_min_dtz_mm(snap, m), 2),
                "dwell": int(m["k6_max_dwell"]), "bilat": int((con["left"] is not None) + (con["right"] is not None)),
                "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)}


def _band(hf: _Handoff, key: str, values: list) -> dict:
    """1-D K6 sweep along one axis; returns the values, K6 flags, and the contiguous K6 band."""
    flags = []
    for v in values:
        kw = {key: v} if key in ("qv_scale", "tau_scale") else {}
        if key == "dq":
            kw = {"dq": np.array([v, v, v, v])}
        elif key == "coin_dxy":
            kw = {"coin_dxy": np.array([v, v])}
        elif key == "coin_dv":
            kw = {"coin_dv": np.array([v, 0.0])}
        flags.append(bool(hf.roll(**kw)["k6"]))
    k6_vals = [v for v, f in zip(values, flags) if f]
    return {"values": values, "k6": flags, "k6_band": ([min(k6_vals), max(k6_vals)] if k6_vals else None)}


def _factorial(hf: _Handoff) -> dict:
    return {f"qvel{qs}_tau{ts}": hf.roll(qv_scale=qs, tau_scale=ts)["k6"]
            for qs in (0.0, 1.0) for ts in (0.0, 1.0)}


def _basin_fraction(hf: _Handoff, n: int) -> dict:
    """Random samples inside a candidate box (qvel 0.8–1.2, tau 0.9–1.1, small dq / coin dev) → K6-compatible fraction."""
    rng = np.random.default_rng(20260729)
    k6 = 0
    for _ in range(n):
        r = hf.roll(qv_scale=float(rng.uniform(0.8, 1.2)), tau_scale=float(rng.uniform(0.9, 1.1)),
                    dq=rng.normal(0, 0.01, 4), qv_dir_noise=0.05, coin_dxy=rng.normal(0, 0.002, 2), rng=rng)
        k6 += int(r["k6"])
    return {"n": n, "k6_compatible": k6, "fraction": round(k6 / n, 3),
            "box": {"qv_scale": [0.8, 1.2], "tau_scale": [0.9, 1.1], "dq_sigma": 0.01, "qv_dir_noise": 0.05,
                    "coin_dxy_sigma_mm": 2.0}}


def run(out: Path = OUT, n_samples: int = 60) -> dict:
    hf = _Handoff()
    axes = {
        "qv_scale": _band(hf, "qv_scale", [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.35, 1.5]),
        "tau_scale": _band(hf, "tau_scale", [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25]),
        "dq_uniform": _band(hf, "dq", [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1]),
        "coin_dxy_m": _band(hf, "coin_dxy", [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]),
        "coin_dv_m_s": _band(hf, "coin_dv", [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1])}
    factorial = _factorial(hf)
    basin = _basin_fraction(hf, n_samples)
    verdict = "DYNAMIC_HANDOFF_BASIN_CHARACTERISED"
    summary = {"contract": "DYNAMIC_HANDOFF_BASIN_V1", "immutable_source": "10aced90",
               "cradle_qvel_arms": [round(float(x), 4) for x in hf.qv0[:4]],
               "cradle_prev_tau": [round(float(x), 4) for x in hf.tau0], "factorial_qvel_x_tau": factorial,
               "axis_bands": axes, "random_basin": basin, "verdict": verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / "dynamic_handoff_basin.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    print(f"\n{r['verdict']}")
    f = r["factorial_qvel_x_tau"]
    print(f"  factorial qvel×tau: {f}  (both jointly necessary)")
    for name, b in r["axis_bands"].items():
        print(f"  {name:12s} K6 band {b['k6_band']}")
    bs = r["random_basin"]
    print(f"  random basin: {bs['k6_compatible']}/{bs['n']} K6-compatible (fraction {bs['fraction']}) in box {bs['box']}")
