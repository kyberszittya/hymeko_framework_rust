"""R10.2 Stage 2 (Boundary 3) — structured-option coordinate conditioning + episodic-exploration admissibility.

Runs the pre-registered Boundary-3 contract (no training, no reward, no actor update):

  * **3A non-zero terminal-offset audit** — with ONLY the terminal block active (``ds=dp=db=k1=k2=0``, ``dtau_T != 0``),
    per-joint +/- and a few mixed-sign combinations, logging requested / executed terminal offset, sign-correctness,
    relative tracking error, and the slew / action-clip / torque-clamp saturation masks. Isolating ``dtau_T`` keeps a
    changed phase-time or bmax from being confounded with the terminal tracking error.
  * **3A local sensitivity** — a full 15-D central-difference audit (``+/- eps`` in z-space) of the executed action,
    terminal torque, terminal ``q``/``qvel`` and downstream ``min_dtz``, with a standardised terminal-state Jacobian SVD
    (effective/stable rank, dead dimensions, collinear pairs). Ill-conditioning is handled BEFORE TD3, not excavated after.
  * **3B normalized episodic exploration** — one frozen per-dimension ``D`` (``theta = sigma*D*z``), then the three
    pre-registered scales ``sigma in {0.05, 0.10, 0.20}`` under the fixed contract: 3 seeds x 32 option-episodes = 96 /
    sigma, the frozen dev panel, a single theta draw per episode, NO per-step noise, the frozen downstream + K6 monitor.

Emits ``reports/2026-07-28-r10-structured-option-torque-path-td3/admissibility.json`` with:
``TERMINAL_OFFSET_TRACKING`` (PASS/FAIL), ``LOCAL_THETA_SENSITIVITY`` (PASS/REDUCE_COORDINATE), the sigma result table,
and the proposed smallest admissible sigma. The final sigma freeze is approved separately; TD3 does not start here.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_structured_option_admissibility`` (add ``--smoke`` for a fast check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import torque_path_conditioning as cond
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

OUT = Path("reports/2026-07-28-r10-structured-option-torque-path-td3")
SIGMAS = (0.05, 0.10, 0.20)
SEEDS = (0, 1, 2)
N_EPISODES = 32
PANEL_N = 16
K6_MM = 10.0                       # strict-K6 min_dtz gate (mm) used elsewhere in the arc
NEAR_BOUNDARY_MM = 25.0            # a safe non-K6 within this min_dtz is an "informative" near-boundary case
DEGENERATE_DEV = 1e-4             # terminal deviation below this = perturbation vanished (degenerate/duplicate)
CLIP_DOMINATES = 0.5             # mean action-clip fraction above this = clipping dominates the episode
TRACK_FLOOR = 0.30               # executed terminal offset must reach >= 30% of requested, unless mask-explained
TERMINAL_COMBOS = ((1, -1, 1, -1), (-1, -1, 1, 1), (1, 1, -1, -1))     # pre-registered mixed-sign terminal probes


# --------------------------------------------------------------------------------------------------------------------
# Panel rollers + tubes (built once; reused across every episode)
# --------------------------------------------------------------------------------------------------------------------
def _panel_rollers(rig: dict, panel: "list[crl.Perturbation]") -> "list[tuple[Any, dict]]":
    out = []
    for pert in panel:
        snap = crl.perturb_ready(rig["ready"], rig["stack"], pert)
        roller = tpo.TorquePathCaptureRoll(snap, rig["ref"], rig["stack"], rig["pi0"], rig["coin"])
        out.append((roller, tpo.record_phase_tube(roller)))
    return out


# --------------------------------------------------------------------------------------------------------------------
# 3A — isolated non-zero terminal-offset audit
# --------------------------------------------------------------------------------------------------------------------
def _terminal_probe(roller: Any, tube: dict, z_dt: np.ndarray) -> dict:
    z = np.zeros(tpo.THETA_DIM, dtype=np.float32)
    z[11:15] = z_dt
    rep = tpo.terminal_offset_report(roller, z, tube)
    req, exe = rep["requested"], rep["executed"]
    active = np.array(rep["terminal_slew_limited"]) | np.array(rep["terminal_action_clipped"]) \
        | np.array(rep["terminal_torque_clamped"])
    sign_ok = [bool(req[j] == 0.0 or np.sign(exe[j]) == np.sign(req[j])) for j in range(4)]
    tracked = [bool(req[j] == 0.0 or abs(exe[j]) >= TRACK_FLOOR * abs(req[j]) or active[j]) for j in range(4)]
    return {"requested": [round(float(x), 5) for x in req], "executed": [round(float(x), 5) for x in exe],
            "rel_err": round(float(np.linalg.norm(exe - req) / (np.linalg.norm(req) + 1e-9)), 4),
            "sign_correct": sign_ok, "tracked_or_saturated": tracked,
            "slew_limited": rep["terminal_slew_limited"], "action_clipped": rep["terminal_action_clipped"],
            "torque_clamped": rep["terminal_torque_clamped"]}


def _terminal_offset_audit(nominal: Any, tube: dict) -> dict:
    slew = nominal.slew
    mag = cond._D_CLAMP[1] * tpo.ThetaScales().terminal_frac    # request near the top of the decoder's terminal band
    probes = {}
    for j in range(4):
        for sign, tag in ((1.0, "pos"), (-1.0, "neg")):
            z = np.zeros(4)
            z[j] = sign
            probes[f"joint{j}_{tag}"] = _terminal_probe(nominal, tube, z)
    for c, combo in enumerate(TERMINAL_COMBOS):
        probes[f"combo{c}"] = _terminal_probe(nominal, tube, np.array(combo, float))
    single = [v for k, v in probes.items() if k.startswith("joint")]
    all_sign = all(all(p["sign_correct"]) for p in single)
    all_tracked = all(all(p["tracked_or_saturated"]) for p in single)
    return {"slew": round(float(slew), 5), "requested_magnitude_note": round(float(mag), 5),
            "probes": probes, "all_single_sign_correct": all_sign, "all_single_tracked_or_saturated": all_tracked}


# --------------------------------------------------------------------------------------------------------------------
# 3B — normalized episodic-exploration admissibility
# --------------------------------------------------------------------------------------------------------------------
def _band(md: float) -> str:
    return "lt10" if md < K6_MM else ("b10_25" if md <= NEAR_BOUNDARY_MM else ("b25_50" if md <= 50.0 else "ge50"))


def _episode(roller: Any, tube: dict, down: Any, theta: np.ndarray) -> dict:
    res = roller.rollout(theta)
    k6, md, safe, kinds = down.deliver_with_trace(res["snapshot"])
    clip_frac = float(np.mean([np.mean(m.action_clipped) for m in res["masks"]]))
    dev = float(np.linalg.norm(np.asarray(res["prev"]) - np.asarray(tube["tau0_terminal"])))
    reset = kinds.count("HANDOFF_RESET")
    return {"k6": bool(k6), "min_dtz": float(md), "safe": bool(safe), "reset": reset, "clip_frac": clip_frac,
            "terminal_dev": dev, "band": _band(md),
            "safety_violation": not safe, "boundary_violation": reset != 1,
            "informative": bool(safe and reset == 1 and not k6),        # any safe, boundary-ok non-delivery is a negative
            "near_boundary": bool(safe and reset == 1 and not k6 and md <= NEAR_BOUNDARY_MM),
            "degenerate": dev < DEGENERATE_DEV}


_COUNT_KEYS = ("k6", "safety_violation", "boundary_violation", "informative", "near_boundary", "degenerate")


def _seed_scale(rollers: list, down: Any, d_norm: np.ndarray, sigma: float, seed: int, n_ep: int) -> dict:
    rng = np.random.default_rng(seed)
    eps = [_episode(*rollers[e % len(rollers)], down, cond.sample_theta(sigma, d_norm, rng.standard_normal(tpo.THETA_DIM)))
           for e in range(n_ep)]
    agg = {k: int(sum(e[k] for e in eps)) for k in _COUNT_KEYS}
    agg["bands"] = {b: int(sum(e["band"] == b for e in eps)) for b in ("lt10", "b10_25", "b25_50", "ge50")}
    agg["mean_clip_frac"] = round(float(np.mean([e["clip_frac"] for e in eps])), 4)
    agg["dev_std"] = round(float(np.std([e["terminal_dev"] for e in eps])), 5)
    agg["min_dtz_median"] = round(float(np.median([e["min_dtz"] for e in eps])), 2)
    agg["n"] = len(eps)
    return agg


def _scale_admissibility(rollers: list, down: Any, d_norm: np.ndarray, sigma: float, seeds: tuple, n_ep: int) -> dict:
    per = {s: _seed_scale(rollers, down, d_norm, sigma, s, n_ep) for s in seeds}
    tot = len(seeds) * n_ep
    safety = sum(v["safety_violation"] for v in per.values())
    boundary = sum(v["boundary_violation"] for v in per.values())
    degen = sum(v["degenerate"] for v in per.values())
    k6_seeds = sum(v["k6"] >= 1 for v in per.values())
    info_seeds = sum(v["informative"] >= 1 for v in per.values())
    clip_dom = bool(np.mean([v["mean_clip_frac"] for v in per.values()]) > CLIP_DOMINATES)
    admissible = all([safety == 0, boundary == 0, k6_seeds >= 2, info_seeds >= 2, degen <= 0.2 * tot, not clip_dom])
    return {"sigma": sigma, "per_seed": per, "safety_violations": safety, "boundary_violations": boundary,
            "degenerate_total": degen, "k6_seeds": k6_seeds, "informative_seeds": info_seeds,
            "clip_dominates": clip_dom, "admissible": bool(admissible)}


# --------------------------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------------------------
TASK_DEAD_MM = 5.0                 # |Δmin_dtz per unit z| below this = the dim barely moves the task (task-dead)
TASK_REDUNDANT_RATIO = 2.0        # a collinear pair is truly redundant only if task sensitivities agree within this ratio


def _task_conditioning(sens: dict) -> dict:
    """Task-aware reading of the local sensitivity: a dim is 'dead' only if it barely moves ``min_dtz`` (not merely the
    tightly-constrained terminal state); a collinear pair is 'redundant' only if the two dims ALSO have similar task
    (``min_dtz``) sensitivity — terminal-state-collinear-but-task-distinct dims are advisory, not a defect."""
    mds = np.array([r.d_min_dtz for r in sens["responses"]])
    task_dead = [i for i in range(tpo.THETA_DIM) if abs(mds[i]) < TASK_DEAD_MM]
    redundant = []
    for i, k, cos in sens["collinear_pairs"]:
        a, b = abs(mds[i]), abs(mds[k])
        ratio = max(a, b) / max(min(a, b), 1e-9)
        if ratio <= TASK_REDUNDANT_RATIO:
            redundant.append([i, k, round(cos, 4), round(float(ratio), 2)])
    return {"task_dead_dims": task_dead, "task_redundant_pairs": redundant}


def _sensitivity_summary(sens: dict, task: dict) -> dict:
    return {"singular_values": [round(float(x), 4) for x in sens["singular_values"]],
            "effective_rank": sens["effective_rank"], "stable_rank": sens["stable_rank"],
            "terminal_state_dead_dims": sens["dead_dims"], "terminal_state_collinear_pairs": sens["collinear_pairs"],
            "task_dead_dims": task["task_dead_dims"], "task_redundant_pairs": task["task_redundant_pairs"],
            "per_dim_effect": [round(float(x), 5) for x in sens["effects"]],
            "per_dim_min_dtz_sens": [round(float(r.d_min_dtz), 4) for r in sens["responses"]],
            "per_dim_action_rms": [round(float(r.d_action_rms), 5) for r in sens["responses"]]}


def _recommend(table: dict, sigmas: tuple) -> dict:
    """The smallest sigma meeting every admissibility criterion EXCEPT the strict zero-boundary-regression rule (0 safety
    violations, K6 in >=2/3 seeds, informative in >=2/3 seeds, clipping not dominating). The residual blocker, if any, is a
    small count of SAFE boundary-route regressions (reset != 1) — a review decision (relax the tolerance vs probe a
    smaller sigma), not a safety failure."""
    for s in sorted(sigmas):
        v = table[f"sigma_{s}"]
        if all([v["safety_violations"] == 0, v["k6_seeds"] >= 2, v["informative_seeds"] >= 2, not v["clip_dominates"]]):
            return {"sigma": s, "residual_boundary_regressions": v["boundary_violations"],
                    "note": "smallest sigma clean on safety + K6 + informative + clipping; residual = SAFE boundary-route "
                            "variation only (reset != 1), a review call on tolerance vs a smaller sigma"}
    return {"sigma": None, "residual_boundary_regressions": None, "note": "no sigma met the relaxed criteria"}


def run(out: Path = OUT, *, smoke: bool = False) -> dict:
    seeds, n_ep, sigmas = ((0,), 4, (0.10,)) if smoke else (SEEDS, N_EPISODES, SIGMAS)
    rig = _rig()
    panel = crl.perturbation_panel(n=PANEL_N, seed=90210)
    rollers = _panel_rollers(rig, panel)
    nominal, nominal_tube = rollers[0]                       # panel member 0 is the un-perturbed nominal READY

    nk6, nmd, nsafe, nkinds = rig["down"].deliver_with_trace(
        nominal.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))["snapshot"])
    nominal_k6 = {"k6": bool(nk6), "min_dtz_mm": round(nmd, 3), "safe": bool(nsafe),
                  "handoff_resets": nkinds.count("HANDOFF_RESET")}
    terminal = _terminal_offset_audit(nominal, nominal_tube)
    sens = cond.axis_sensitivity(nominal, rig["down"])
    task = _task_conditioning(sens)
    d_norm = cond.freeze_normalization(sens["responses"])
    table = {f"sigma_{s}": _scale_admissibility(rollers, rig["down"], d_norm, s, seeds, n_ep) for s in sigmas}

    admissible = [s for s in sigmas if table[f"sigma_{s}"]["admissible"]]
    recommendation = _recommend(table, sigmas)
    sensitivity_ok = all([not task["task_dead_dims"], not task["task_redundant_pairs"], sens["effective_rank"] >= 10])
    verdicts = {
        "TERMINAL_OFFSET_TRACKING": ("PASS" if terminal["all_single_sign_correct"]
                                     and terminal["all_single_tracked_or_saturated"] else "FAIL"),
        "LOCAL_THETA_SENSITIVITY": ("PASS" if sensitivity_ok else "REDUCE_COORDINATE"),
    }
    summary = {
        "contract": "STRUCTURED_OPTION_ADMISSIBILITY_V1", "parent_commit": "c8e90e11",
        "boundary": "3 (coordinate conditioning + episodic admissibility) — NO training/reward/actor-update",
        "measurement_contract": {"sigmas": list(sigmas), "seeds": list(seeds), "episodes_per_seed_scale": n_ep,
                                 "panel_n": PANEL_N, "single_theta_per_episode": True, "per_step_noise": False,
                                 "smoke": smoke},
        "nominal_zero_theta_k6": nominal_k6,
        "frozen_normalization_D": [round(float(x), 4) for x in d_norm],
        "terminal_offset_audit": terminal,
        "local_sensitivity": _sensitivity_summary(sens, task),
        "sigma_table": table,
        "proposed_smallest_admissible_sigma": (min(admissible) if admissible else None),
        "recommendation": recommendation,
        "verdicts": verdicts,
        "non_claims": ["NO reward-driven policy trained", "NO actor update", "sigma freeze is approved SEPARATELY",
                       "TD3/SAC/PPO do not start until the sigma is frozen"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "admissibility.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _print(r: dict) -> None:
    print(f"nominal zero-theta K6: {r['nominal_zero_theta_k6']}")
    t = r["terminal_offset_audit"]
    print(f"terminal-offset: sign_correct={t['all_single_sign_correct']} tracked/sat={t['all_single_tracked_or_saturated']}")
    s = r["local_sensitivity"]
    print(f"sensitivity: eff_rank={s['effective_rank']} stable_rank={s['stable_rank']} "
          f"task_dead={s['task_dead_dims']} task_redundant={s['task_redundant_pairs']} "
          f"(term-state collinear advisory={s['terminal_state_collinear_pairs']})")
    print(f"D = {r['frozen_normalization_D']}")
    for k, v in r["sigma_table"].items():
        bands = {b: sum(s["bands"][b] for s in v["per_seed"].values()) for b in ("lt10", "b10_25", "b25_50", "ge50")}
        print(f"  {k}: admissible={v['admissible']} k6_seeds={v['k6_seeds']} info_seeds={v['informative_seeds']} "
              f"safety={v['safety_violations']} boundary={v['boundary_violations']} degen={v['degenerate_total']} "
              f"clip_dom={v['clip_dominates']} bands={bands}")
    print(f"strict admissible sigma: {r['proposed_smallest_admissible_sigma']} | recommendation: {r['recommendation']}")
    for k, v in r["verdicts"].items():
        print(f"  {v} {k}")


if __name__ == "__main__":
    _print(run(smoke="--smoke" in sys.argv))
