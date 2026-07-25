"""H2 — B_v IDENTIFICATION & VALIDATION benchmark (reproducible; frozen V2 material / V4 stack; O3 paused; no RL).

Identify the governed-servo control-to-contact-velocity response Jacobian B_v = ∂v_rel,t+1/∂Δq_target at the EXACT
acquisition HANDOFF of CERTIFIED straddle cradles (both-contact ∧ internal-force certificate feasible ∧ straddle), with
the TRUE exported controller state (prev_tau + q_target — no re-synthesis, no settle). v_rel is measured at the COMMON
MuJoCo contact point (both bodies transported there). Then VALIDATE the one-step predictiveness against simulation
(held-out perturbations, two ε scales, central vs one-sided, repeat-eval determinism), CONTRAST against the geometric
fingertip Jacobian (guard: B_v ≠ J_tip), and run a PINNED positive-control (coin held → the FD/velocity pipeline check,
NOT active-stabilizability evidence). Emits the machine-readable artifact + plots. Does NOT build the H2 QP.

Run:  python -m hymeko_rl.experiments.bv_identification_benchmark [--smoke]
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_pair_scenario import set_material, setup_material_decoupling  # noqa: E402
from hymeko_rl.coin_delivery.contact_velocity import (  # noqa: E402
    BvConfig, CradleSnapshot, InvalidCradleSnapshot, identify_Bv, one_step_vrel, onesided_Bv, replay_consistency, validate_Bv)
from hymeko_rl.coin_delivery.cooperative_launch import CooperativeConfig, straddle_directed_acquire, tip_midpoint  # noqa: E402
from hymeko_rl.env.governed_arm import V3Stack  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
REPORT_DIR = "reports/2026-07-26-h2-bv-identification"
# The CERTIFIED straddling-cradle subset (reachable both-contact ∧ internal-force certificate feasible) from the E3
# embodiment audit (cradle_embodiment_audit.json): states s1, s3, s4, s7.
CERTIFIED_SEEDS = [14250, 14750, 15000, 15750]


def _load_frozen():
    v4 = json.load(open(f"{OUT}/dynamics_contract_v4.json"))["frozen_contract"]
    v2 = json.load(open(f"{OUT}/rubber_tip_v2_material.json"))["frozen_material"]
    mu = float(np.mean(list(v2["coin_floor_mu_eff_by_v0"].values())))
    stack = V3Stack(v4["qdot_soft"], v4["qdot_hard"], v4["armature"], v4["damping"], v4["friction"],
                    v4["kp"], v4["kv"], v4["tau_rate"], over_hard_brake=v4["over_hard_brake"])
    return stack, CooperativeConfig(coast_mu=mu), v2, mu


def _make_env(pi0, base, forbidden, v2, seed, tries):
    rl, _gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=seed, tries=tries)
    tg, adr, _bt, _bd = setup_material_decoupling(rl)
    set_material(rl, tg, adr, v2["tip_coin_friction"], v2["coin_slide_viscous_damping"], v2["coin_slide_coulomb_frictionloss"])
    return rl


def acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg, seed, tries):
    """Acquire a CERTIFIED both-contact straddle cradle (keep the pin), exporting the exact handoff controller state.
    Source gate (fix per review): both_contact ∧ internal-force certificate feasible ∧ straddle (n_dot<0). Sweeps the two
    audit axes and takes the most-negative-ndot certified hit. Returns (rl, pin_saved, handoff, meta) or (None,…)."""
    rl0 = _make_env(pi0, base, forbidden, v2, seed, tries)
    mid = tip_midpoint(rl0)
    u, _dtz = rl0.inner.direction_to_zone()
    e = np.asarray(u, np.float64)
    axes = {"zone_cross": np.array([-e[1], e[0]]), "zone_par": e}
    best = None
    attempts = {}
    for name, ax in axes.items():
        rl = _make_env(pi0, base, forbidden, v2, seed, tries)
        out = straddle_directed_acquire(rl, stack, mid.astype(np.float32), ax, cfg=cfg, keep_pin=True)
        cert_ok = bool(out["both_contact"] and out["cradle_certificate"]["feasible"] and out["straddle"]["straddles"])
        attempts[name] = {"both_contact": bool(out["both_contact"]), "n_dot": out["straddle"]["n_dot"],
                          "cert_feasible": bool(out["cradle_certificate"]["feasible"]), "certified": cert_ok}
        if cert_ok and (best is None or out["straddle"]["n_dot"] < best[3]):
            handoff = {"prev_tau": out["final_tau"], "q_target": out["final_q_target"]}
            best = (rl, out["pin_saved"], handoff, out["straddle"]["n_dot"], name)
    if best is None:
        return None, None, None, {"seed": seed, "certified": False, "attempts": attempts}
    rl, saved, handoff, ndot, name = best
    return rl, saved, handoff, {"seed": seed, "certified": True, "axis": name, "n_dot": ndot, "attempts": attempts}


# --------------------------------------------------------------------------------------------------------------------
# Geometric-fingertip-Jacobian contrast (the guard: B_v ≠ J_tip)
# --------------------------------------------------------------------------------------------------------------------
def kinematic_vrel_jacobian(snap: CradleSnapshot, dt: float) -> np.ndarray:
    """The PURELY-GEOMETRIC predictor a naive fingertip Jacobian gives: perturb each joint POSITION by ε (kinematics
    only — mj_forward, NO governed dynamics, NO contact solve, coin FIXED), read the tip displacement at the frozen
    contact point, and form v_rel = (Δp_tip/dt) projected on the frozen frame. If the identified governed B_v equalled
    this, B_v would BE the geometric Jacobian."""
    import copy

    import mujoco
    rl = snap.branch()
    m, d = rl.inner.model, rl.inner.data
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    q0 = d.qpos[:4].copy()
    p_l0, p_r0 = d.geom_xpos[gl][:2].copy(), d.geom_xpos[gr][:2].copy()
    frl, frr = snap.frames["left"], snap.frames["right"]
    eps = 1e-5
    Bk = np.zeros((4, 4))
    for j in range(4):
        dd = copy.deepcopy(rl)
        dd.inner.data.qpos[:4] = q0
        dd.inner.data.qpos[j] = q0[j] + eps
        mujoco.mj_forward(dd.inner.model, dd.inner.data)
        vpl = (dd.inner.data.geom_xpos[gl][:2] - p_l0) / eps / dt   # d(tip)/dq · (1/dt) → tip velocity per unit Δq
        vpr = (dd.inner.data.geom_xpos[gr][:2] - p_r0) / eps / dt
        Bk[0, j], Bk[1, j] = frl["n"] @ vpl, frl["t"] @ vpl        # coin FIXED ⇒ v_rel = v_tip
        Bk[2, j], Bk[3, j] = frr["n"] @ vpr, frr["t"] @ vpr
    return Bk


# --------------------------------------------------------------------------------------------------------------------
# Per-state analysis
# --------------------------------------------------------------------------------------------------------------------
def _determinism_check(snap: CradleSnapshot, cfg: BvConfig) -> float:
    g_a, _va, _da = one_step_vrel(snap, np.zeros(4), cfg)
    g_b, _vb, _db = one_step_vrel(snap, np.zeros(4), cfg)
    return float(np.max(np.abs(g_a - g_b)))


def _pick_identification(snap: CradleSnapshot, cfg: BvConfig):
    ids = [identify_Bv(snap, e, cfg) for e in cfg.eps_scales]
    ids.sort(key=lambda r: (-sum(r["col_valid"]), -r["eps"]))
    return ids[0], ids


def _col_gap(Ba, Bb, va, vb):
    gaps = []
    for j in range(4):
        if va[j] and vb[j]:
            denom = np.linalg.norm(Ba[:, j]) + np.linalg.norm(Bb[:, j])
            if denom > 1e-9:
                gaps.append(float(np.linalg.norm(Ba[:, j] - Bb[:, j]) / denom))
    return round(max(gaps), 4) if gaps else None


def _summarise_levels(records: list[dict]) -> dict:
    out = {}
    for level in ("small", "medium", "near_boundary"):
        rs = [r for r in records if r["level"] == level]
        if not rs:
            continue
        rel = [r["rel_err"] for r in rs if r["rel_err"] is not None]
        cos = [r["cosine"] for r in rs if r["cosine"] is not None]
        out[level] = {"n": len(rs), "median_rel_err": (round(float(np.median(rel)), 4) if rel else None),
                      "median_cosine": (round(float(np.median(cos)), 4) if cos else None),
                      "contact_retained_frac": round(float(np.mean([r["contact_retained"] for r in rs])), 3),
                      "contract_ok_frac": round(float(np.mean([r["contract_ok"] for r in rs])), 3)}
    return out


def _round_mat(M):
    return [[(round(float(x), 5) if np.isfinite(x) else None) for x in row] for row in M]


def analyse_snapshot(snap: CradleSnapshot, cfg: BvConfig, dt: float) -> dict:
    """B_v identification + validation + geometric contrast on one snapshot (free or pinned)."""
    det = _determinism_check(snap, cfg)
    best, ids = _pick_identification(snap, cfg)
    Bv, g0, col_valid = best["Bv"], best["g0"], best["col_valid"]
    one_sided = onesided_Bv(snap, best["eps"], cfg)
    Bk = kinematic_vrel_jacobian(snap, dt)
    finite = np.isfinite(Bv)
    bv_norm = float(np.linalg.norm(np.where(finite, Bv, 0.0)))
    diff_norm = float(np.linalg.norm(np.where(finite, Bv - Bk, 0.0)))
    records = validate_Bv(snap, Bv, g0, col_valid, cfg)
    col_diag = {j: dict(best["cols"][j]) for j in range(4)}
    col_classes = [best["cols"][j]["class"] for j in range(4)]
    n_active = int(sum(1 for c in col_classes if c == "ACTIVE"))
    mechanism = _rate_limiter_mechanism(snap, best)
    return {"operating_point": snap.operating_point, "released_coin": snap.released,
            "pre_release_hash": snap.pre_release_hash, "post_release_hash": snap.post_release_hash,
            "post_release_recert": snap.recert, "determinism_max_abs": round(det, 9), "eps_used": best["eps"],
            "col_valid": col_valid, "n_valid_cols": int(sum(col_valid)), "col_classes": col_classes,
            "n_active_cols": n_active, "col_diag": col_diag,
            "handoff_prev_tau": [round(float(x), 4) for x in snap.prev_tau], "arm_saturated": snap.arm_saturated,
            "fn0": [round(x, 4) for x in snap.fn0], "straddle0": round(snap.straddle0, 4),
            "eps_agreement_rel_gap": _col_gap(ids[0]["Bv"], ids[1]["Bv"], ids[0]["col_valid"], ids[1]["col_valid"]) if len(ids) > 1 else None,
            "central_vs_onesided_rel_gap": _col_gap(Bv, one_sided["Bv"], col_valid, one_sided["col_valid"]),
            "Bv": _round_mat(Bv), "g0": [round(float(x), 6) for x in g0],
            "geometric_jacobian_contrast": {"bv_norm": round(bv_norm, 5), "bv_minus_jtip_norm": round(diff_norm, 5),
                                            "relative_difference": round(diff_norm / (bv_norm + 1e-12), 4),
                                            "Bv_kinematic": _round_mat(Bk)},
            "rate_limiter_mechanism": mechanism, "validation": records, "level_summary": _summarise_levels(records)}


def _rate_limiter_mechanism(snap: CradleSnapshot, best: dict) -> dict:
    """Why the columns are dead: the nominal raw PD torque is already |raw − prev_tau| beyond the per-step torque-rate
    slew (τ̇·dt), so the perturbation kp·ε is fully absorbed by the rate clip (identical applied torque). Reports the
    per-joint gap vs the slew step and vs the slew step + kp·ε."""
    d0 = best["diag0"]
    rate_step = float(snap.stack.tau_rate * snap.stack.control_dt)
    kp_eps = float(snap.stack.kp * best["eps"])
    gap = np.abs(np.asarray(d0["raw_pd_tau"], np.float64) - snap.prev_tau)
    return {"rate_step_Nm": round(rate_step, 4), "kp_times_eps_Nm": round(kp_eps, 4),
            "raw_pd_tau": d0["raw_pd_tau"], "prev_tau": [round(float(x), 4) for x in snap.prev_tau],
            "abs_gap_raw_minus_prev": [round(float(x), 4) for x in gap],
            "absorbed_per_joint": [bool(g >= rate_step + kp_eps) for g in gap],
            "all_absorbed": bool(np.all(gap >= rate_step + kp_eps))}


# --------------------------------------------------------------------------------------------------------------------
# Verdict (plan §5) — DIAGNOSTIC aggregation; a final verdict is only stamped once the operating point is settled
# --------------------------------------------------------------------------------------------------------------------
def _overnight_gate(states: list[dict]) -> dict:
    """The measurement-rigor gate that MUST pass before the full 4-seed validation runs: ≥1 admissible post-release
    certified snapshot, ≥2 ACTIVE B_v columns in some state (a usable one-step Jacobian exists), bit-reproducible
    branches, replay-consistent handoff transfer, every zero column physically classified. If it fails, STOP with the
    negative/characterised report — do NOT auto-run overnight."""
    free = [s for s in states if s.get("free")]
    if not free:
        return {"passed": False, "reason": "no certified snapshot analysed", "checks": {}}
    checks = {
        "post_release_admissible": any(s["free"]["post_release_recert"]["admissible"] for s in free),
        "some_state_ge2_active_cols": any(s["free"]["n_active_cols"] >= 2 for s in free),
        "bit_reproducible": all(s["free"]["determinism_max_abs"] <= 1e-8 for s in free),
        "replay_consistent": all(s.get("replay", {}).get("consistent", False) for s in free),
        "all_zero_cols_classified": all(all(c in ("ACTIVE", "DEAD_ZONE_IDENTICAL_POST_GOVERNOR_TORQUE",
                                                  "INVALID_MODE_SWITCH", "ZERO_BUT_TORQUE_DIFFERED")
                                           for c in s["free"]["col_classes"]) for s in free)}
    passed = bool(all(checks.values()))
    reason = "all gate checks passed" if passed else "; ".join(f"{k}=FAIL" for k, v in checks.items() if not v)
    return {"passed": passed, "reason": reason, "checks": checks}


def _level_stats(free: list[dict], level: str):
    vals = [s["level_summary"].get(level) for s in free if s["level_summary"].get(level)]
    if not vals:
        return None
    return {"rel": float(np.median([v["median_rel_err"] for v in vals if v["median_rel_err"] is not None] or [np.inf])),
            "cos": float(np.median([v["median_cosine"] for v in vals if v["median_cosine"] is not None] or [-1.0])),
            "retain": float(np.median([v["contact_retained_frac"] for v in vals]))}


def _level_good(x, cfg: BvConfig) -> bool:
    return bool(x and x["rel"] <= cfg.rel_ok and x["cos"] >= cfg.cos_ok and x["retain"] >= 0.8)


def _is_dead_zone(free: list[dict]) -> bool:
    """Every certified handoff yields 0 ACTIVE columns, all classified as identical-post-governor-torque dead zones."""
    ok_cls = ("DEAD_ZONE_IDENTICAL_POST_GOVERNOR_TORQUE", "INVALID_MODE_SWITCH")
    return bool(all(s["n_active_cols"] == 0 for s in free)
                and all(all(c in ok_cls for c in s["col_classes"]) for s in free))


def _positive_verdict(free: list[dict], cfg: BvConfig) -> dict:
    """The trust-region verdict once a usable (≥2 ACTIVE column) Jacobian exists on a reproducible branch."""
    levels = {lv: _level_stats(free, lv) for lv in ("small", "medium", "near_boundary")}
    nb = levels["near_boundary"]
    if nb and nb["retain"] < 0.5:
        return {"verdict": "BV_IDENTIFICATION_UNRELIABLE_UNDER_CONTACT_MODE_SWITCHING",
                "reason": f"near-boundary contact retention {nb['retain']:.2f} < 0.5", "levels": levels}
    good = {lv: _level_good(levels[lv], cfg) for lv in levels}
    if good["small"] and good["medium"] and good["near_boundary"]:
        v = "CONTROL_TO_CONTACT_VELOCITY_JACOBIAN_VALIDATED"
    elif good["small"]:
        v = "LOCAL_BV_VALID_ONLY_ON_SMALL_TRUST_REGION"
    else:
        v = "CONTROL_RESPONSE_PARTIALLY_OBSERVED"
    return {"verdict": v, "levels": levels}


def decide_verdict(states: list[dict], cfg: BvConfig) -> dict:
    free = [s["free"] for s in states if s.get("free")]
    if not free:
        return {"verdict": "CONTROL_RESPONSE_PARTIALLY_OBSERVED", "reason": "no certified straddle cradle acquired"}
    if _is_dead_zone(free):     # one-step Δq_target authority null — rate limiter absorbs ±ε (a precise negative)
        return {"verdict": "ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT",
                "reason": "every certified post-release handoff: 0 ACTIVE B_v columns; ±ε absorbed by the rate limiter "
                          "(identical post-governor torque) — one-step position-target authority is null"}
    nondet = [s for s in free if s["determinism_max_abs"] > 1e-8]
    max_cols = max(s["n_active_cols"] for s in free)
    if nondet or max_cols < 2:
        return {"verdict": "CONTROL_RESPONSE_PARTIALLY_OBSERVED",
                "reason": f"nondeterministic={len(nondet)} max_active_cols={max_cols}"}
    return _positive_verdict(free, cfg)


def _plot(states: list[dict], path: str) -> None:
    """The mechanism figure: WHY B_v ≡ 0. Left — per-joint one-step slew debt |raw_pd − prev_tau| against the torque-rate
    step and the step+kp·ε authority band (every bar above the band ⇒ the ±ε command is absorbed). Right — the resulting
    B_v column norms (all ≈ 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    free = [s["free"] for s in states if s.get("free")]
    labels = [f"s{s['state']}·seed{s['seed']}" for s in states if s.get("free")]
    joints = np.arange(4)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / max(1, len(free))
    rate_step = free[0]["rate_limiter_mechanism"]["rate_step_Nm"] if free else 0.3
    band = rate_step + max((f["rate_limiter_mechanism"]["kp_times_eps_Nm"] for f in free), default=0.36)
    for i, (f, lab) in enumerate(zip(free, labels)):
        ax[0].bar(joints + i * width, f["rate_limiter_mechanism"]["abs_gap_raw_minus_prev"], width, label=lab)
        ax[1].bar(joints + i * width, [f["col_diag"][j]["col_norm"] for j in range(4)], width, label=lab)
    ax[0].axhline(rate_step, ls="--", c="k", lw=1, label=f"slew step τ̇·dt = {rate_step}")
    ax[0].axhline(band, ls=":", c="r", lw=1.2, label=f"step + kp·ε ≈ {band:.2f} (absorption)")
    ax[0].set_xticks(joints + 0.4 - width / 2)
    ax[0].set_xticklabels([f"j{j}" for j in joints])
    ax[0].set_ylabel("|raw_pd − prev_tau|  (N·m)")
    ax[0].set_title("one-step slew debt vs command authority")
    ax[0].legend(fontsize=7)
    ax[1].set_xticks(joints + 0.4 - width / 2)
    ax[1].set_xticklabels([f"j{j}" for j in joints])
    ax[1].set_ylabel("‖B_v[:,j]‖")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_title("resulting B_v column norms (≈ 0)")
    ax[1].legend(fontsize=7)
    fig.suptitle("H2 B_v ≡ 0 at the certified handoff — torque-rate slew absorbs the one-step Δq_target command")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(REPORT_DIR, exist_ok=True)
    t_start = time.time()
    stack, cfg_coop, v2, mu = _load_frozen()
    bcfg = BvConfig()
    dt = stack.control_dt
    pi0, base, forbidden = _setup()
    seeds = CERTIFIED_SEEDS[:2] if smoke else CERTIFIED_SEEDS
    tries = 3
    print(f"B_v IDENTIFICATION — certified-handoff control-to-contact-velocity | μ={mu:.3f} dt={dt} "
          f"eps={bcfg.eps_scales} levels={bcfg.dq_levels} | seeds={seeds}", flush=True)
    states, acquire_log = [], []
    for si, seed in enumerate(seeds):
        t0 = time.time()
        rl, saved, handoff, meta = acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg_coop, seed, tries)
        acquire_log.append(meta)
        if rl is None:
            print(f"  s{si} seed{seed}: no CERTIFIED straddle — skip ({time.time()-t0:.1f}s)", flush=True)
            continue
        entry = {"state": si, "seed": seed, "acquire": meta}
        try:
            free = CradleSnapshot(rl, stack, saved, handoff["prev_tau"], handoff["q_target"], release_coin=True,
                                  coast_mu=mu, cfg=bcfg)
            entry["free"] = analyse_snapshot(free, bcfg, dt)
            entry["replay"] = replay_consistency(rl, stack, handoff["prev_tau"], handoff["q_target"], free)
        except InvalidCradleSnapshot as e:
            entry["free_invalid"] = str(e)
            print(f"  s{si} seed{seed}: INVALID free snapshot — {e} ({time.time()-t0:.1f}s)", flush=True)
            states.append(entry)
            continue
        try:                                                     # PINNED positive control (pipeline check, not evidence)
            pinned = CradleSnapshot(rl, stack, saved, handoff["prev_tau"], handoff["q_target"], release_coin=False,
                                    coast_mu=mu, cfg=bcfg)
            entry["pinned_control"] = analyse_snapshot(pinned, bcfg, dt)
        except InvalidCradleSnapshot as e:
            entry["pinned_invalid"] = str(e)
        states.append(entry)
        fs = entry["free"]
        rc = fs["post_release_recert"]
        print(f"  s{si} seed{seed}: prev_tau={fs['handoff_prev_tau']} sat={fs['arm_saturated']} fn0={fs['fn0']} | "
              f"recert[dual{int(rc['dual_contact'])} strd{int(rc['straddle'])} cert{int(rc['cert_feasible'])} qdot{rc['qdot_max']}] "
              f"replay_ok={entry['replay']['consistent']} det={fs['determinism_max_abs']:.1e} | "
              f"cols valid={fs['n_valid_cols']}/4 ACTIVE={fs['n_active_cols']}/4 classes={fs['col_classes']} | "
              f"pinned ACTIVE={entry.get('pinned_control',{}).get('n_active_cols','-')} ({time.time()-t0:.1f}s)", flush=True)

    gate = _overnight_gate(states)
    verdict = decide_verdict(states, bcfg)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    manifest = {
        "contract": "H2_CONTROL_TO_CONTACT_VELOCITY_IDENTIFICATION", "date": "2026-07-26",
        "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2", "coast_mu": round(mu, 3), "certified_seeds": CERTIFIED_SEEDS,
        "frozen_conventions": {
            "decision_variable": "dq_target in R^4 (joint position-servo target perturbation; nominal dq=0 = acquisition handoff q_target)",
            "prediction_horizon": "exactly one control step", "control_dt": dt, "substeps": stack.substeps,
            "output_ordering": "[v_n_L, v_t_L, v_n_R, v_t_R]",
            "relative_velocity": "v_rel = v_tip(x_c) - v_coin(x_c) at the COMMON MuJoCo contact point x_c; both bodies rigid-body-transported to x_c; v_n>0 = closing/compression",
            "projection_frame": "ACTUAL MuJoCo contact normal at snapshot t (oriented tip->coin), FROZEN",
            "coin": "FREE (pin released) for identification; PINNED variant = positive control (pipeline check only)",
            "operating_point": "EXACT acquisition handoff (exported prev_tau + q_target; NO settle, NO re-synthesised seed)",
            "source_gate": "certified straddle: both_contact AND internal_force_feasibility.feasible AND straddle(n_dot<0)",
            "fd": "central, two eps scales; one-sided compared near constraints", "eps_scales": list(bcfg.eps_scales),
            "fn_floor_N": bcfg.fn_floor, "dq_levels": list(bcfg.dq_levels), "val_dirs_per_level": bcfg.n_val_dirs,
            "val_seed": bcfg.val_seed,
            "validity": "SAME contact mode: same contact identities AND both Fn>=floor AND straddle retained AND motion contract"},
        "n_states_requested": len(seeds), "n_states_analysed": len([s for s in states if s.get("free")]),
        "acquire_log": acquire_log, "states": states, "overnight_gate": gate, "verdict": verdict,
        "peak_rss_gb": round(peak_rss_gb, 3), "wall_s": round(time.time() - t_start, 1), "h2_qp_built": False}
    art = f"{REPORT_DIR}/bv_identification.json"
    json.dump(manifest, open(art, "w"), indent=1, default=float)
    if [s for s in states if s.get("free")]:
        _plot(states, f"{REPORT_DIR}/bv_rate_limiter_mechanism.png")
    print(f"\n== B_v IDENTIFICATION ==\n  certified states analysed: {manifest['n_states_analysed']}/{len(seeds)} | "
          f"peak RSS {peak_rss_gb:.2f} GB | wall {manifest['wall_s']}s\n"
          f"  overnight-gate: {'PASS' if gate['passed'] else 'FAIL'} ({gate['reason']})\n"
          f"  → {verdict['verdict']}\n  reason: {verdict.get('reason', verdict.get('levels'))}\n"
          f"  artifact: {art}\n  H2 QP built: NO"
          f"\n  {'gate PASSED — full 4-seed validation may run' if gate['passed'] else 'gate FAILED — STOP, negative/characterised report; do NOT run overnight'}\nBV_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
