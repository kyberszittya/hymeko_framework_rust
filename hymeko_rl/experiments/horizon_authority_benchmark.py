"""H2 SESSION 2 — LIFTED-HORIZON AUTHORITY vs CRADLE COLLAPSE benchmark (reproducible; frozen V2/V4; O3 paused; no RL).

For every CERTIFIED straddle cradle (s1,s3 development · s4,s7 held-out), from the EXACT acquisition handoff (free coin,
exported prev_tau + q_target), run HELD_OFFSET rollouts to H_MAX=40 and extract the horizon hierarchy H_tau ≤ H_state ≤
H_vrel versus H_collapse, assemble B_v(H) at the frozen horizons, and SELECT the control-interface route (A/B/C) by the
frozen decision tree. A ONE_STEP_PULSE control contrasts sustained vs impulse authority. Emits the machine-readable
artifact + the race/rank/authority figures. Does NOT build the H2 QP; does NOT tune acquisition; does NOT start RL.

Trace-equivalence guard: at H=1 the HELD rollout must reproduce the Session-1 dead zone (rank 0, no authority) — printed.

Run:  python -m hymeko_rl.experiments.horizon_authority_benchmark [--smoke]
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.contact_velocity import BvConfig, CradleSnapshot, InvalidCradleSnapshot  # noqa: E402
from hymeko_rl.coin_delivery.horizon_authority import (  # noqa: E402
    HORIZONS, HorizonConfig, analyse_state_horizon, decide_campaign_route)
from hymeko_rl.coin_delivery.mobile_conditioning import ConditionConfig, condition_mobile_handoff  # noqa: E402
from hymeko_rl.experiments.bv_identification_benchmark import (  # noqa: E402
    CERTIFIED_SEEDS, _load_frozen, acquire_certified_straddle)
from hymeko_rl.experiments.video_coin_variants import _setup  # noqa: E402

REPORT_DIR = "reports/2026-07-27-coin-solution-to-rl"
DEV_IDS, HELDOUT_IDS = [0, 1], [2, 3]           # s1,s3 development · s4,s7 held-out (CERTIFIED_SEEDS order)
STATE_TAG = {0: "s1", 1: "s3", 2: "s4", 3: "s7"}


def _analyse_seed(idx, seed, harness, cfg: HorizonConfig, condition: bool = False) -> dict:
    """Acquire a certified straddle at ``seed``, optionally run the Route-A mobile low-debt conditioning (free-coin
    preload settle), snapshot the free-coin handoff, run the lifted-horizon analysis."""
    pi0, base, forbidden, v2, stack, cfg_coop, mu = harness
    rl, saved, handoff, meta = acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg_coop, seed, tries=3)
    entry = {"state": idx, "tag": STATE_TAG.get(idx, f"s{idx}"), "seed": seed,
             "split": ("development" if idx in DEV_IDS else "held_out"), "acquire": meta, "conditioned": bool(condition)}
    if rl is None:
        entry["skipped"] = "no CERTIFIED straddle cradle acquired"
        return entry
    prev_tau, q_target = handoff["prev_tau"], handoff["q_target"]
    if condition:                                    # Route A — mobile low-debt handoff (free coin, no pin), then measure
        cond = condition_mobile_handoff(rl, stack, saved, handoff["prev_tau"], handoff["q_target"],
                                        coop=cfg_coop, ccfg=ConditionConfig())
        entry["conditioning"] = {
            "settled": cond["settled"], "steps_used": cond["steps_used"], "fn_final": list(cond["fn_final"]),
            "balance_final": cond["balance_final"], "qdot_final": cond["qdot_final"],
            "prev_tau": ([round(float(x), 4) for x in cond["prev_tau"]] if cond["prev_tau"] is not None else None),
            "q_target": [round(float(x), 4) for x in cond["q_target"]], "fn_hist": cond["fn_hist"]}
        if cond["prev_tau"] is None:
            entry["invalid"] = "conditioning produced no controller state (no contact)"
            return entry
        prev_tau, q_target = cond["prev_tau"], cond["q_target"]
    try:
        snap = CradleSnapshot(rl, stack, saved, prev_tau, q_target, release_coin=True, coast_mu=mu, cfg=cfg.bv)
        entry["analysis"] = analyse_state_horizon(snap, cfg)
    except InvalidCradleSnapshot as e:
        entry["invalid"] = str(e)
    return entry


def _print_state(entry: dict, dt_wall: float) -> None:
    if entry.get("conditioning"):
        c = entry["conditioning"]
        fh = c["fn_hist"]
        trace = " ".join(f"({fl:.2f},{fr:.2f})" for fl, fr in fh[::max(1, len(fh) // 8)]) if fh else "-"
        print(f"  {entry['tag']} cond: settled={c['settled']} steps={c['steps_used']} fn_final={c['fn_final']} "
              f"balance={c['balance_final']} qdot={c['qdot_final']} | fn_hist~ {trace}", flush=True)
    if "analysis" not in entry:
        print(f"  {entry['tag']} seed{entry['seed']}: {entry.get('skipped', entry.get('invalid','?'))} "
              f"({dt_wall:.1f}s)", flush=True)
        return
    a = entry["analysis"]
    print(f"  {entry['tag']} seed{entry['seed']} [{entry['split']}]: prev_tau={a['handoff_prev_tau']} "
          f"fn0={a['fn0']} strd0={a['straddle0']} | H_tau={a['H_tau']} H_state={a['H_state']} H_vrel={a['H_vrel']} "
          f"H_collapse={a['H_collapse_baseline']} | rank2_onset={a['rank2_onset']} "
          f"repro={a['rank2_onset_reproducible']} sustain={a['rank2_sustains_margin']} usable={a['usable_authority']} "
          f"| H=1 rank={a['rank_series'][list(a['rank_series'])[0]][0]} → {a['route']} ({dt_wall:.1f}s)", flush=True)


def _campaign_states(states: list[dict]) -> dict:
    return {e["state"]: e["analysis"] for e in states if "analysis" in e}


# --------------------------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------------------------
def _plot(states: list[dict], path: str, cfg: HorizonConfig) -> None:
    """Three-panel presentation figure: (1) the horizon RACE (H_tau/H_state/H_vrel vs H_collapse per state), (2) the
    B_v rank versus control step per state with the collapse line, (3) the contact-velocity authority ‖Δv_rel‖ growth
    at the frozen horizons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ana = [(e["tag"], e["analysis"]) for e in states if "analysis" in e]
    if not ana:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (1) race
    ev_c = {"H_tau": "tab:blue", "H_state": "tab:cyan", "H_vrel": "tab:orange", "H_collapse_baseline": "tab:red"}
    for row, (tag, a) in enumerate(ana):
        for key, c in ev_c.items():
            v = a[key]
            if v is not None:
                ax[0].scatter([v], [row], color=c, s=70, zorder=3,
                              label=key.replace("_baseline", "") if row == 0 else None)
        if a["rank2_onset"] and a["usable_authority"]:
            ax[0].axvspan(a["rank2_onset"], a["H_collapse_baseline"], ymin=(row + 0.1) / len(ana),
                          ymax=(row + 0.9) / len(ana), color="tab:green", alpha=0.15)
    ax[0].set_yticks(range(len(ana)))
    ax[0].set_yticklabels([t for t, _ in ana])
    ax[0].set_xlabel("control step H")
    ax[0].set_title("horizon race: authority vs collapse")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].grid(True, axis="x", ls=":", alpha=0.5)
    # (2) rank vs step
    steps = np.arange(1, cfg.h_max + 1)
    for tag, a in ana:
        e0 = list(a["rank_series"])[0]
        ax[1].plot(steps, a["rank_series"][e0], label=f"{tag} held", lw=1.6)
        ax[1].plot(steps, a["pulse_rank_series"], ls="--", lw=1.0, alpha=0.7, label=f"{tag} pulse")
        ax[1].axvline(a["H_collapse_baseline"], color="tab:red", ls=":", lw=0.8)
    ax[1].axhline(2, color="k", ls="--", lw=0.8, label="rank=2 (usable)")
    ax[1].set_xlabel("control step H")
    ax[1].set_ylabel("rank B_v(H)")
    ax[1].set_title("controllable rank vs horizon (held vs pulse)")
    ax[1].legend(fontsize=6, ncol=2)
    # (3) authority growth
    for tag, a in ana:
        e0 = list(a["columns"])[0]
        peak_cols = a["columns"][e0]
        agg = np.max([c["dvrel_norm_at_horizons"] for c in peak_cols], axis=0)
        ax[2].plot(list(HORIZONS), agg, marker="o", label=tag)
    ax[2].axhline(cfg.vrel_tol, color="k", ls="--", lw=0.8, label=f"‖Δv_rel‖ floor {cfg.vrel_tol}")
    ax[2].set_xlabel("horizon H")
    ax[2].set_ylabel("max col ‖Δv_rel‖ (m/s)")
    ax[2].set_title("contact-velocity authority growth")
    ax[2].set_yscale("symlog", linthresh=1e-4)
    ax[2].legend(fontsize=7)
    fig.suptitle("H2 Session 2 — lifted-horizon authority versus cradle collapse")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(smoke=False, condition=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(REPORT_DIR, exist_ok=True)
    t_start = time.time()
    stack, cfg_coop, v2, mu = _load_frozen()
    cfg = HorizonConfig(bv=BvConfig())
    pi0, base, forbidden = _setup()
    harness = (pi0, base, forbidden, v2, stack, cfg_coop, mu)
    seeds = CERTIFIED_SEEDS[:2] if smoke else CERTIFIED_SEEDS
    tag = "conditioned (ROUTE_A)" if condition else "baseline"
    print(f"HORIZON AUTHORITY [{tag}] — lifted-horizon vs cradle collapse | μ={mu:.3f} H={HORIZONS} "
          f"eps={cfg.bv.eps_scales} margin={cfg.margin} | seeds={seeds} ({'smoke:dev' if smoke else 'full'})", flush=True)
    states = []
    for idx, seed in enumerate(seeds):
        t0 = time.time()
        entry = _analyse_seed(idx, seed, harness, cfg, condition=condition)
        _print_state(entry, time.time() - t0)
        states.append(entry)
    cstates = _campaign_states(states)
    campaign = decide_campaign_route(cstates, DEV_IDS, HELDOUT_IDS)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)
    manifest = {
        "contract": "H2_LIFTED_HORIZON_AUTHORITY_VS_CRADLE_COLLAPSE", "session": "H2_SESSION_2",
        "variant": ("ROUTE_A_MOBILE_LOW_DEBT_CONDITIONED" if condition else "BASELINE_RAW_HANDOFF"),
        "date": "2026-07-27", "wall_clock_note": "executed 2026-07-26/27 JST overnight campaign",
        "builds_on": "ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT (5f2f0133, frozen)",
        "physics": "RUBBER_TIP_LOW_DRAG_COIN_V2", "coast_mu": round(mu, 3),
        "frozen_split": {"development": ["s1", "s3"], "held_out": ["s4", "s7"]},
        "frozen_conventions": {
            "primary_semantics": "HELD_OFFSET (Δq held on servo target for all H steps)",
            "control_semantics": "ONE_STEP_PULSE (Δq on step 1 only)",
            "horizons": list(HORIZONS), "h_max": cfg.h_max, "eps_scales": list(cfg.bv.eps_scales),
            "H_tau": "first step ±ε post-governor torque sequences diverge",
            "H_state": "first step q/qdot diverge", "H_vrel": "first step measurable ‖Δv_rel‖>floor, same contact mode",
            "H_collapse": "first step the cradle-alive predicate fails (NOT the one-step FD drift gate)",
            "tau_tol_Nm": cfg.tau_tol, "state_tol": cfg.state_tol, "vrel_tol_mps": cfg.vrel_tol,
            "rank_abs_tol": cfg.rank_abs_tol, "rank_rel_tol": cfg.rank_rel_tol, "usability_margin_steps": cfg.margin,
            "repro_rel_gap": cfg.repro_rel_gap,
            "usable_authority": "rank≥2 both ε AND reproducible AND sustains margin AND onset<H_collapse",
            "route_tree": {"ROUTE_C": "usable (H_vrel+margin<H_collapse, rank≥2)",
                           "ROUTE_A": "rank-2 authority exists but unusable (timing) → lower-debt handoff",
                           "ROUTE_B": "no rank-2 authority before collapse → slew-admissible Δτ"}},
        "n_states": len(seeds), "states": states, "campaign_route": campaign,
        "peak_rss_gb": round(peak_rss_gb, 3), "wall_s": round(time.time() - t_start, 1), "h2_qp_built": False}
    suffix = "_conditioned" if condition else ""
    art = f"{REPORT_DIR}/authority_recovery{suffix}.json"
    json.dump(manifest, open(art, "w"), indent=1, default=float)
    _plot(states, f"{REPORT_DIR}/horizon_authority_race{suffix}.png", cfg)
    print(f"\n== HORIZON AUTHORITY ==\n  states analysed: {len(cstates)}/{len(seeds)} | peak RSS {peak_rss_gb:.2f} GB "
          f"| wall {manifest['wall_s']}s\n  dev routes: {campaign['dev_routes']} | held-out: {campaign['heldout_routes']}"
          f"\n  → CAMPAIGN ROUTE: {campaign['route']}\n  reason: {campaign['reason']}\n  artifact: {art}\nHORIZON_DONE",
          flush=True)
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, condition="--condition" in sys.argv)
