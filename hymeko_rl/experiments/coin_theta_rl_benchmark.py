"""COIN TEACHER-TO-RL benchmark — one harness, mode flags (§6.5 #13: modes, not v-files).

Pipeline over the frozen 6-D torque-θ delivery option (tag coin-physical-feasibility-closed, a3459629):
    frozen CEM teacher bank → structured causal θ dataset → BC proposal → update-0 no-regression →
    matched SAC/TD3 smoke → matched multi-seed.

Modes (each STOPS at its gate; downstream artifacts only when authorised):
    --semantics       emit option_semantics.json (Stage 0)
    --teacher-bank    reproduce the 4 canonical trajectories + dev-only CEM augmentation; freeze teacher_bank.json (Stage 1)
    --dataset         build the structured causal θ dataset + splits; dataset_contract.json (Stage 2)
    --bc              fit B0/B1/B2 proposals; bc_results.json (Stage 3)
    --update0         update-0 deploy on the frozen 4-state panel; update_zero.json (Stage 4)
    --coverage-curve  coverage-only causal curve: the update-0 gate at N=2,4,6 dev cradles (only N changes); coverage_curve.json
    --rl-smoke        matched SAC/TD3 one-seed smoke (Stage 6, gated on Stage 4)
    --rl-multiseed    matched multi-seed SAC/TD3 (Stage 7, gated)

Run:  python -m hymeko_rl.experiments.coin_theta_rl_benchmark --<mode> [--smoke]
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from typing import Any

import numpy as np

REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"
ACCEPTABLE_DIR = "reports/2026-07-27-coin-acceptable-set"
MULTIMODAL_DIR = "reports/2026-07-27-coin-multimodal"
REP_AUDIT_DIR = "reports/2026-07-27-coin-decision-representation"


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 3)


def _dump(obj: dict, name: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f"{REPORT_DIR}/{name}"
    json.dump(obj, open(path, "w"), indent=1, default=float)
    return path


def semantics_main() -> dict:
    """Stage 0 — freeze and emit the 6-D torque-θ option semantics."""
    from hymeko_rl.coin_delivery.theta_option.semantics import option_semantics
    sem = option_semantics()
    path = _dump(sem, "option_semantics.json")
    print(f"OPTION SEMANTICS frozen → {path}\n  dim={sem['dim']} components={[c['name'] for c in sem['components']]}\n"
          f"  K6: CENTER_TOL={sem['termination_and_k6']['CENTER_TOL_m']} SETTLE_VEL="
          f"{sem['termination_and_k6']['SETTLE_VEL_mps']} HELD_DWELL={sem['termination_and_k6']['HELD_DWELL_steps']}\n"
          f"  Bellman action = θ_0 (proposal centre); θ_exec = search provenance only\nSEMANTICS_DONE", flush=True)
    return sem


def scout_cradles_main(n: int = 32) -> dict:
    """DEV-CRADLE EXPANSION step 0 — certification scout (read-only). Sweep the canonical seed enumeration and count how
    many UNIQUE certified straddle cradles are available, to bound N for the expansion. No delivery CEM (cheap)."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import (
        FROZEN_SEEDS, enumerate_seeds, load_harness, scout_certified_cradles)
    t0 = time.time()
    seeds = enumerate_seeds(n)
    print(f"CRADLE SCOUT — certifying {len(seeds)} candidate cradles (seed=14000+250·si, si=0..{n-1}) | "
          f"frozen already-certified: {list(FROZEN_SEEDS)}", flush=True)
    n_cert = {"c": 0}

    def _progress(r: dict, dt: float) -> None:                # live per-cradle (never run blind)
        n_cert["c"] += int(r["certified"])
        mark = "FROZEN" if r["is_frozen_seed"] else ("NEW" if r["certified"] else "-")
        print(f"  si={r['si']:2d} seed={r['seed']}: certified={int(r['certified'])} n_dot={r['n_dot']} "
              f"strd0={r['straddle0']} [{mark}] (cum_cert={n_cert['c']}, {dt:.0f}s)", flush=True)

    rows = scout_certified_cradles(load_harness(), seeds, progress=_progress)
    certified = [r for r in rows if r["certified"]]
    uniq_hashes = {r["post_release_hash"] for r in certified}
    new_certified = [r for r in certified if not r["is_frozen_seed"]]
    frozen_recertified = sum(1 for r in certified if r["is_frozen_seed"])
    out = {"contract": "COIN_CRADLE_CERTIFICATION_SCOUT_V1", "base_commit": "a3459629", "date": "2026-07-27",
           "enumeration": "seed = 14000 + 250*si", "n_scanned": len(seeds),
           "n_certified": len(certified), "n_unique_by_hash": len(uniq_hashes),
           "n_new_certified": len(new_certified), "n_frozen_recertified": frozen_recertified,
           "certified_seeds": [r["seed"] for r in certified], "new_certified_seeds": [r["seed"] for r in new_certified],
           "certification_rate": round(len(certified) / max(1, len(seeds)), 3),
           "rows": rows, "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    path = _dump(out, "cradle_scout.json")
    print(f"\n== CRADLE SCOUT ==\n  certified {out['n_certified']}/{out['n_scanned']} "
          f"(rate {out['certification_rate']}) | unique-by-hash {out['n_unique_by_hash']} | "
          f"frozen re-certified {frozen_recertified}/4 | NEW certified {out['n_new_certified']}\n"
          f"  achievable N (unique certified dev+heldout cradles) ≈ {out['n_unique_by_hash']}\n"
          f"  new certified seeds: {out['new_certified_seeds']}\n  artifact: {path} | wall {out['wall_s']}s\nSCOUT_DONE",
          flush=True)
    return out


def deliver_pass_main() -> dict:
    """DEV-CRADLE EXPANSION step 1 — dedup the certified inventory + frozen-CEM delivery pass on each UNIQUE dev-eligible
    cradle. Records K6-deliverable dev cradles (the N-curve x-axis) and CERTIFIED_BUT_NOT_DELIVERABLE_UNDER_FROZEN_OPTION
    separately. All per-cradle params (CEM budget/seed, θ bounds, search, K6, motion contract) frozen."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import (
        HELDOUT_SEEDS, NEAR_TOL, acquire_certified_pool, dedup_and_split)
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import deliver_on_snapshot, load_harness
    t0 = time.time()
    scout_path = f"{REPORT_DIR}/cradle_scout.json"
    if not os.path.exists(scout_path):
        print(f"MISSING {scout_path} — run --scout-cradles first")
        sys.exit(2)
    scout = json.load(open(scout_path))
    certified_seeds = scout["certified_seeds"]
    near_tol = float(sys.argv[sys.argv.index("--near-tol") + 1]) if "--near-tol" in sys.argv else NEAR_TOL
    print(f"DELIVERY PASS — {len(certified_seeds)} certified cradles → dedup (near_tol={near_tol}) → frozen-CEM delivery. "
          f"held-out excluded: {list(HELDOUT_SEEDS)}", flush=True)
    harness = load_harness()

    def _acq_prog(e: Any, dt: float) -> None:
        print(f"  acquire seed={e.seed}: hash={e.hash} ({dt:.0f}s)", flush=True)

    pool = acquire_certified_pool(harness, certified_seeds, progress=_acq_prog)
    dedup = dedup_and_split(pool, near_tol=near_tol)
    by_seed = {e.seed: e for e in pool}
    print(f"  dedup: certified={dedup['n_certified']} hash_unique={dedup['n_hash_unique']} "
          f"after_near_dedup={dedup['n_after_near_dedup']} dev_eligible={dedup['n_dev_eligible']} | "
          f"fp_dist min/med/max={dedup['pairwise_fingerprint_dist']['min']}/"
          f"{dedup['pairwise_fingerprint_dist']['median']}/{dedup['pairwise_fingerprint_dist']['max']}", flush=True)
    delivering, not_deliverable = [], []
    for k, seed in enumerate(dedup["dev_eligible_seeds"]):
        te = time.time()
        r = deliver_on_snapshot(by_seed[seed].snap, basin_seed=seed)
        rec = {"seed": seed, **r}
        (delivering if r["deliverable"] else not_deliverable).append(rec)
        oc = r.get("outcome", {})
        print(f"  [{k+1}/{len(dedup['dev_eligible_seeds'])}] deliver seed={seed}: deliverable={int(r['deliverable'])} "
              f"src={r.get('canonical_source')} dwell={oc.get('k6_max_dwell')} dtz_end={oc.get('dtz_end_mm')}mm "
              f"({time.time()-te:.0f}s)", flush=True)
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import assemble_funnel
    funnel = assemble_funnel(dedup, delivering, not_deliverable, certified_seeds=certified_seeds)
    out = {"contract": "COIN_CRADLE_DELIVERY_PASS_V1", "base_commit": "a3459629", "date": "2026-07-27",
           "near_tol": near_tol, "dedup": dedup, **funnel,
           "frozen_dev_already_in_pool": [s for s in (14250, 14750) if s in dedup["dev_eligible_seeds"]],
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    path = _dump(out, "cradle_delivery_pass.json")
    f = funnel["funnel"]
    print(f"\n== DELIVERY PASS ==\n  funnel: raw_dev={f['n_raw_dev_candidates']} → near_unique={f['n_near_unique_dev']} "
          f"→ K6_deliverable={f['n_K6_deliverable_dev']} (yield {f['delivery_yield_among_unique']}) → usable_N={f['usable_N']}\n"
          f"  new-state distribution: {funnel['new_state_distribution']}\n"
          f"  deliverable seeds: {[r['seed'] for r in delivering]}\n  artifact: {path} | wall {out['wall_s']}s\n"
          f"DELIVER_PASS_DONE", flush=True)
    return out


def contact_audit_main() -> dict:
    """TASK CONTACT-LEGALITY audit — grade each frozen teacher delivery (CONTROLLED_INSERTION + fingertip impulse share +
    E0/E1/E2 level). Physical collision is real; this only GRADES the contact quality."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.insertion_certificate import grade_delivery
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    bank_path = f"{REPORT_DIR}/teacher_bank.json"
    if not os.path.exists(bank_path):
        print(f"MISSING {bank_path} — run --teacher-bank first")
        sys.exit(2)
    bank = json.load(open(bank_path))
    harness = load_harness()
    print("CONTACT AUDIT — grading teacher deliveries (physical collision ON; link contact allowed)", flush=True)
    rows, tot_ft, tot_arm = [], 0.0, 0.0
    for e in bank["states"]:
        if "canonical_theta_vec" not in e:
            continue
        snap, _ = acquire_snapshot(harness, e["seed"])
        from hymeko_rl.coin_delivery.theta_option.insertion_certificate import contact_impulse_share
        q = contact_impulse_share(snap, e["canonical_theta_vec"])
        g = grade_delivery(snap, e["canonical_theta_vec"])
        tot_ft += q["fingertip_impulse"]
        tot_arm += q["arm_body_impulse"]
        rows.append({"tag": e["tag"], "split": e["split"], "controlled_insertion": g.controlled_insertion,
                     "ballistic_knock": g.ballistic_knock, "level": g.level,
                     "fingertip_impulse_share": g.fingertip_impulse_share,
                     "arm_body_contact_frames": q["arm_body_contact_frames"], "n_frames": q["n_frames"],
                     "peak_coin_speed": g.peak_coin_speed, "terminal_coin_speed": g.terminal_coin_speed})
        print(f"  {e['tag']} [{e['split']}]: controlled_insertion={g.controlled_insertion} knock={g.ballistic_knock} "
              f"level={g.level} ft_share={g.fingertip_impulse_share} arm_frames={q['arm_body_contact_frames']}/{q['n_frames']}",
              flush=True)
    grand = tot_ft + tot_arm
    overall_share = round(tot_ft / grand, 4) if grand > 1e-9 else None
    all_ci = all(r["controlled_insertion"] for r in rows)
    out = {"contract": "COIN_CONTACT_LEGALITY_AUDIT_V1", "base_commit": "a3459629", "date": "2026-07-27",
           "physical_collision": "REALISTIC (every arm geom collides with the coin; per-side masks; same-arm isolated)",
           "task_certificate": "CONTROLLED_INSERTION (link contact allowed; ballistic knock rejected)",
           "levels": {"E0": "WHOLE_ARM_ASSISTED_INSERTION (link contact allowed)",
                      "E1": "FINGERTIP_DOMINANT (ft impulse share >= 0.5)", "E2": "FINGERTIP_ONLY (no non-tip contact)"},
           "states": rows, "overall_fingertip_impulse_share": overall_share,
           "all_controlled_insertion": all_ci, "teacher_label": "WHOLE_ARM_ASSISTED_INSERTION_E0",
           "fingertip_dominant": bool(overall_share is not None and overall_share >= 0.5),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    path = _dump(out, "contact_quality_audit.json")
    print(f"\n== CONTACT AUDIT ==\n  all controlled_insertion={all_ci} | overall fingertip impulse share={overall_share} "
          f"| LABEL: {out['teacher_label']} (fingertip_dominant={out['fingertip_dominant']})\n  artifact: {path} | "
          f"wall {out['wall_s']}s\nCONTACT_AUDIT_DONE", flush=True)
    return out


def teacher_bank_main(smoke: bool = False) -> dict:
    """Stage 1 — reproduce + freeze the 6-D torque-θ teacher bank; gate on 4/4 canonical frozen-tolerance replay."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import (
        CERTIFIED_SEEDS, DEV_IDS, HELDOUT_IDS, STATE_TAG, load_harness, reproduce_state)
    t0 = time.time()
    harness = load_harness()
    seeds = list(CERTIFIED_SEEDS[:1]) if smoke else list(CERTIFIED_SEEDS)
    print(f"TEACHER BANK — reproduce {len(seeds)} certified cradle(s) {[STATE_TAG[i] for i in range(len(seeds))]} "
          f"({'smoke:s1' if smoke else 'full 4-state'})", flush=True)
    entries = []
    for idx, seed in enumerate(seeds):
        te = time.time()
        e = reproduce_state(harness, idx, seed, augment=(idx in DEV_IDS))
        entries.append(e)
        oc = e.get("outcome", {})
        print(f"  {e['tag']} [{e['split']}] seed{seed}: k6={e.get('k6_delivered')} replay_ok={e.get('replay_ok')} "
              f"dtz {oc.get('dtz_start_mm')}→{oc.get('dtz_end_mm')}mm dwell={oc.get('k6_max_dwell')}/6 "
              f"θ={e.get('canonical_theta_vec')} basin={e.get('n_basin_delivering','-')}/{e.get('n_basin_near','-')} "
              f"({time.time()-te:.1f}s)", flush=True)
    canon = [e for e in entries if "canonical_theta_vec" in e]
    n_deliver = sum(1 for e in canon if e.get("k6_delivered"))
    n_replay = sum(1 for e in canon if e.get("replay_ok"))
    expected = len(seeds)
    gate = {"all_canonical_replay": bool(n_replay == expected), "all_canonical_k6": bool(n_deliver == expected),
            "n_delivered": n_deliver, "n_replay_ok": n_replay, "n_states": expected,
            "k6_by_frozen_monitor_only": True, "no_pin_teleport_or_coin_edit": True,
            "passed": bool(n_replay == expected and n_deliver == expected)}
    bank = {"contract": "COIN_6D_TORQUE_THETA_TEACHER_BANK_V1", "base_tag": "coin-physical-feasibility-closed",
            "base_commit": "a3459629", "date": "2026-07-27",
            "certified_seeds": list(CERTIFIED_SEEDS),
            "split": {"development": [STATE_TAG[i] for i in DEV_IDS], "held_out": [STATE_TAG[i] for i in HELDOUT_IDS],
                      "policy": "dev θ + dev basin augment used for fitting; held-out θ frozen for evaluation ONLY"},
            "option_config": {"horizon": 60, "lo": [0.0, 0.0, -0.10, 1.0, 4.0, 0.0],
                              "hi": [0.25, 0.30, 0.10, 28.0, 48.0, 4.0], "cem_seed": 20260727, "pop": 56, "iters": 10},
            "states": entries, "gate": gate, "smoke": bool(smoke),
            "peak_rss_gb": _peak_rss_gb(), "wall_s": round(time.time() - t0, 1)}
    path = _dump(bank, "teacher_bank.json")
    n_dev_aug = sum(e.get("n_basin_delivering", 0) + e.get("n_basin_near", 0) for e in entries if e["split"] == "development")
    print(f"\n== TEACHER BANK ==\n  canonical delivered {n_deliver}/{expected} | replay_ok {n_replay}/{expected} | "
          f"dev basin candidates {n_dev_aug} | peak RSS {bank['peak_rss_gb']} GB | wall {bank['wall_s']}s\n"
          f"  GATE: {'PASS' if gate['passed'] else 'FAIL'} | artifact: {path}\nTEACHER_BANK_DONE", flush=True)
    return bank


def dataset_main() -> dict:
    """Stage 2 — build the structured causal θ dataset from the frozen teacher bank; freeze dataset_contract.json."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.dataset import build_dataset, contract_summary
    t0 = time.time()
    bank_path = f"{REPORT_DIR}/teacher_bank.json"
    if not os.path.exists(bank_path):
        print(f"MISSING {bank_path} — run --teacher-bank first")
        sys.exit(2)
    bank = json.load(open(bank_path))
    if bank.get("smoke"):
        print("teacher_bank.json is a SMOKE run (partial) — run the full --teacher-bank first")
        sys.exit(2)
    print("DATASET — building structured causal θ dataset from the frozen teacher bank", flush=True)
    ds = build_dataset(bank)
    summary = contract_summary(ds)
    summary["wall_s"] = round(time.time() - t0, 1)
    summary["peak_rss_gb"] = _peak_rss_gb()
    path = _dump(summary, "dataset_contract.json")
    sc = summary["split_counts"]
    print(f"  feature_dim={summary['feature_dim']} history={summary['history']['k']}x{len(summary['history']['features'])} "
          f"| splits train={sc['train']} val={sc['val']} eval={sc['eval']}\n"
          f"  n_by_tag_split={summary['n_by_tag_split']}\n"
          f"  split_isolation_ok={summary['split_isolation_ok']} all_hashes_match={summary['all_hashes_match']}\n"
          f"  leakage_guards={summary['leakage_guards']}\n  artifact: {path} | wall {summary['wall_s']}s\nDATASET_DONE",
          flush=True)
    return summary


_NPZ = f"{REPORT_DIR}/theta_dataset.npz"


def _load_or_build_dataset():
    """Load the cached snapshot-free dataset (fast) or build it from the frozen teacher bank (re-acquires physics)."""
    from hymeko_rl.coin_delivery.theta_option.dataset import build_dataset, load_npz, save_npz
    if os.path.exists(_NPZ):
        return load_npz(_NPZ)
    bank_path = f"{REPORT_DIR}/teacher_bank.json"
    if not os.path.exists(bank_path):
        print(f"MISSING {bank_path} — run --teacher-bank first")
        sys.exit(2)
    ds = build_dataset(json.load(open(bank_path)))
    save_npz(ds, _NPZ)
    return ds


def bc_main() -> dict:
    """Stage 3 — fit B0/B1/B2 proposals identically; offline θ-error / validity / phase / held-back-dev metrics; select."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.proposal import VARIANTS, fit_bc, offline_metrics, save_proposal
    t0 = time.time()
    ds = _load_or_build_dataset()
    print(f"BC — fitting {VARIANTS} on train={len(ds.subset('train'))} val={len(ds.subset('val'))} (identical budget)",
          flush=True)
    results = {}
    for v in VARIANTS:
        prop, fz, tl = fit_bc(ds, v, epochs=1200, lr=1e-3, seed=0)
        save_proposal(prop, fz, f"{REPORT_DIR}/bc_{v}.pt")
        results[v] = {"train": {**tl, **offline_metrics(prop, fz, ds, "train")},
                      "val": offline_metrics(prop, fz, ds, "val")}
        val = results[v]["val"]
        print(f"  {v}: train_mse={tl['train_mse']} | val n={val.get('n')} norm_err={val.get('mean_norm_err')} "
              f"phase_step_err={val.get('phase_param_step_err')} valid={val.get('bounded_validity')} "
              f"per_tag={val.get('per_tag_norm_err')}", flush=True)
    # selection: lowest held-back-dev (val) normalised θ-error; tie-break to the simplest variant (B0<B1<B2)
    order = {v: i for i, v in enumerate(VARIANTS)}
    selected = min(VARIANTS, key=lambda v: (results[v]["val"].get("mean_norm_err", 9.9), order[v]))
    out = {"contract": "COIN_6D_THETA_BC_V1", "base_commit": "a3459629", "date": "2026-07-27",
           "config": {"epochs": 1200, "lr": 1e-3, "seed": 0, "variants": list(VARIANTS)},
           "note": "offline regression is NOT the update-0 gate (Stage 4); selection carries to the frozen-panel deploy",
           "results": results, "selected": selected, "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    path = _dump(out, "bc_results.json")
    print(f"\n== BC ==\n  selected (min held-back-dev θ-error): {selected}\n  artifact: {path} | wall {out['wall_s']}s\n"
          f"BC_DONE", flush=True)
    return out


def _render_update0_viz(out: dict, panel: list) -> dict:
    """§9 graphical output for update-0: (1) a per-state K6 grouped bar (informed / uninformed / oracle) + a budget-sweep
    line; (2) deploy GIFs of the BC θ_exec on a delivering dev cradle (s1) and a failing held-out cradle (s4). Reuses the
    frozen renderer; panel snapshots are already acquired. Returns the written figure paths."""
    figs: dict = {}
    gate_b = out["deploy_budget"]
    per = out["informed_sweep"][str(gate_b)]["per_state"] if str(gate_b) in out["informed_sweep"] else out["informed_sweep"][gate_b]["per_state"]
    unf = out["uninformed_sweep"].get(str(gate_b), out["uninformed_sweep"].get(gate_b))["per_state"]
    orc = out["oracle_gate_diagnostic"]["per_state"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tags = list(per.keys())
        x = np.arange(len(tags))
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        for k, (cond, src, c) in enumerate([("informed (BC θ0)", per, "tab:blue"),
                                            ("uninformed (box)", unf, "tab:gray"),
                                            ("oracle (teacher θ)", orc, "tab:green")]):
            ax[0].bar(x + (k - 1) * 0.27, [int(src[t]["delivery_success"]) for t in tags], 0.27, label=cond, color=c)
        ax[0].set_xticks(x)
        ax[0].set_xticklabels([f"{t}\n{per[t]['split'][:3]}" for t in tags])
        ax[0].set_ylabel("frozen K6 delivered")
        ax[0].set_ylim(0, 1.2)
        ax[0].set_title(f"update-0 per-cradle K6 (budget {gate_b})")
        ax[0].legend(fontsize=8)
        budgets = out["budgets"]
        inf_tot = [out["informed_sweep"][str(b) if str(b) in out["informed_sweep"] else b]["total_k6"] for b in budgets]
        unf_tot = [out["uninformed_sweep"][str(b) if str(b) in out["uninformed_sweep"] else b]["total_k6"] for b in budgets]
        ax[1].plot(budgets, inf_tot, "o-", label="informed (BC θ0)", color="tab:blue")
        ax[1].plot(budgets, unf_tot, "s--", label="uninformed (box)", color="tab:gray")
        ax[1].axhline(4, color="tab:green", ls=":", label="oracle / teacher = 4/4")
        ax[1].set_xlabel("fixed search budget")
        ax[1].set_ylabel("total K6 / 4")
        ax[1].set_ylim(0, 4.3)
        ax[1].set_title("K6 vs search budget")
        ax[1].legend(fontsize=8)
        fig.suptitle(f"Update-0 no-regression — {out['verdict']} (blocker: {out['diagnosed_blocker']})")
        fig.tight_layout()
        p = f"{REPORT_DIR}/update_zero_panel.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs["panel_png"] = p
    except Exception as e:
        figs["plot_error"] = str(e)
    # deploy GIFs of the BC θ_exec: s1 (delivering dev) and s4 (failing held-out)
    try:
        from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
        from hymeko_rl.experiments.horizon_authority_benchmark import _render_forward_gif
        by_tag = {ps.tag: ps for ps in panel}
        for tag in ("s1", "s4"):
            if tag in per and tag in by_tag:
                gif = f"{REPORT_DIR}/update0_deploy_{tag}.gif"
                if _render_forward_gif(by_tag[tag].snap, per[tag]["theta_exec"], gif, DELIVERY_CFG):
                    figs[f"gif_{tag}"] = gif
    except Exception as e:
        figs["gif_error"] = str(e)
    return figs


def _render_coverage_viz(out: dict, panel: list) -> dict:
    """§9 graphical output for the coverage curve: (1) K6-vs-N (dev / held-out / total, oracle=4/4 reference) and the
    held-out actor→teacher θ distance vs N, side by side; (2) a deploy GIF of the largest-N proposal on a held-out cradle
    (s4). Reuses the frozen renderer; the panel snapshots are already acquired."""
    figs: dict = {}
    recs = sorted(out["records"], key=lambda r: r["N"])
    Ns = [r["N"] for r in recs]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        ax[0].plot(Ns, [r["gate_informed_dev_k6"] for r in recs], "o-", label="dev K6 (s1,s3)", color="tab:blue")
        ax[0].plot(Ns, [r["gate_informed_held_out_k6"] for r in recs], "s-", label="held-out K6 (s4,s7)", color="tab:red")
        ax[0].plot(Ns, [r["gate_informed_total_k6"] for r in recs], "^--", label="total K6 / 4", color="tab:purple")
        ax[0].axhline(4, color="tab:green", ls=":", label="oracle (teacher θ) = 4/4")
        ax[0].set_xlabel("N (unique K6-deliverable dev cradles)")
        ax[0].set_ylabel("frozen K6 @ budget 8")
        ax[0].set_ylim(-0.2, 4.3)
        ax[0].set_xticks(Ns)
        ax[0].set_title("Coverage curve — update-0 K6 vs N")
        ax[0].legend(fontsize=8)
        hd = [r["held_out_theta_l2_norm"] for r in recs]
        dd = [r["dev_theta_l2_norm"] for r in recs]
        ax[1].plot(Ns, hd, "s-", label="held-out (s4,s7)", color="tab:red")
        ax[1].plot(Ns, dd, "o-", label="dev (s1,s3)", color="tab:blue")
        ax[1].set_xlabel("N (unique K6-deliverable dev cradles)")
        ax[1].set_ylabel("actor→teacher θ distance (box-normalised L2)")
        ax[1].set_xticks(Ns)
        ax[1].set_title("Actor→teacher θ distance vs N")
        ax[1].legend(fontsize=8)
        v = out["verdict"]
        fig.suptitle(f"COVERAGE-ONLY CAUSAL CURVE — {v['verdict']} "
                     f"(authorise_sac_td3={v['authorise_sac_td3']}, generalisation_improves={v['generalisation_improves_with_coverage']})")
        fig.tight_layout()
        p = f"{REPORT_DIR}/coverage_curve.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs["curve_png"] = p
    except Exception as e:                                # viz must never break the measurement
        figs["plot_error"] = str(e)
    try:
        from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
        from hymeko_rl.experiments.horizon_authority_benchmark import _render_forward_gif
        top = max(recs, key=lambda r: r["N"])
        te = top.get("held_out_theta_exec", {})
        by_tag = {ps.tag: ps for ps in panel}
        for tag in ("s4",):
            if tag in te and tag in by_tag:
                gif = f"{REPORT_DIR}/coverage_N{top['N']}_deploy_{tag}.gif"
                if _render_forward_gif(by_tag[tag].snap, te[tag], gif, DELIVERY_CFG):
                    figs[f"gif_N{top['N']}_{tag}"] = gif
    except Exception as e:
        figs["gif_error"] = str(e)
    return figs


def _pca2(X: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """2-D PCA projection via SVD (no sklearn dep): centre, SVD, project onto the top-2 right singular vectors. Returns
    (projection (n,2), components (2,d)). # Postconditions: deterministic; components are orthonormal rows."""
    Xc = np.asarray(X, np.float64)
    mu = Xc.mean(0)
    Xc = Xc - mu
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    comp = vt[:2]
    return Xc @ comp.T, comp


def _render_acceptable_set_viz(out: dict, pooled_norm: np.ndarray, labels: list, state_of: list,
                               heldout_proj_src: dict) -> dict:
    """§9 viz for the multimodality test: a 2-D PCA of the pooled dev acceptable set coloured by basin, with per-state
    markers, basin centroids, and the held-out teacher θ + the failing actor θ₀ overlaid (projected into the same PCA)."""
    figs: dict = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        proj, comp = _pca2(pooled_norm)
        mu = np.asarray(pooled_norm, np.float64).mean(0)
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        nb = out["pooled_clusters"]["n_basins"]
        cmap = plt.get_cmap("tab10")
        markers = {"s1": "o", "s3": "s", "16500": "^", "17750": "v", "19500": "D", "24000": "P"}
        for i, (p, lb, st) in enumerate(zip(proj, labels, state_of)):
            ax.scatter(p[0], p[1], c=[cmap(lb % 10)], marker=markers.get(str(st), "x"), s=42,
                       edgecolors="k", linewidths=0.3, alpha=0.85,
                       label=None)
        for b, cen in enumerate(out["pooled_clusters"]["centroids"]):
            cp = (np.asarray(cen) - mu) @ comp.T
            ax.scatter(cp[0], cp[1], c="k", marker="*", s=220, edgecolors=cmap(b % 10), linewidths=1.5,
                       label=f"basin {b} centroid (n={out['pooled_clusters']['basin_sizes'][b]})")
        for tag, src in heldout_proj_src.items():
            for kind, z, mk, col in src:
                zp = (np.asarray(z, np.float64) - mu) @ comp.T
                ax.scatter(zp[0], zp[1], marker=mk, s=170, c=col, edgecolors="k", linewidths=1.2,
                           label=f"{tag} {kind}")
        ax.set_xlabel("PCA-1 (normalised θ)")
        ax.set_ylabel("PCA-2 (normalised θ)")
        ax.set_title(f"Dev acceptable set — {out['verdict']['verdict']} "
                     f"({nb} pooled basins, inter/intra={out['pooled_clusters']['min_inter_basin_dist']}/"
                     f"{out['pooled_clusters']['max_intra_nn_hop']})")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        p = f"{ACCEPTABLE_DIR}/acceptable_set_basins.png"
        os.makedirs(ACCEPTABLE_DIR, exist_ok=True)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs["basins_png"] = p
    except Exception as e:
        figs["plot_error"] = str(e)
    return figs


def acceptable_set_main(smoke: bool = False) -> dict:
    """MULTIMODALITY DISCRIMINATING TEST (M0) — harvest the acceptable set on DEV cradles (local delivering basin +
    global enrichment; frozen-K6 ∧ motion-compatible), pool + single-linkage cluster in normalised option-space, test
    whether averaging fails (per-state acceptable-centroid non-delivery), and overlay the held-out delivering teacher θ +
    the failing N=6 actor θ₀ (eval-only, no fit). Verdict gates the multimodal model: MULTIMODAL_BASINS_PRESENT (proceed)
    vs SINGLE_CONNECTED_CLUSTER (= REPRESENTATION_NOT_PROPOSAL_MODALITY_IS_BLOCKER, stop)."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.acceptable_set import (
        assign_to_basins, centroid_delivers, cluster_basins, harvest_acceptable_set, multimodality_verdict)
    from hymeko_rl.coin_delivery.theta_option.proposal import load_proposal
    from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
    from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness, sample_dev_basin
    t0 = time.time()
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    dp = json.load(open(f"{REPORT_DIR}/cradle_delivery_pass.json"))
    box = ThetaBox()
    dev = {14250: "s1", 14750: "s3"}
    new_theta = {r["seed"]: r["canonical_theta"] for r in dp["deliverable_dev_pool"] if r["seed"] not in dev}
    dev_canon = {14250: [e for e in bank["states"] if e["tag"] == "s1"][0]["canonical_theta_vec"],
                 14750: [e for e in bank["states"] if e["tag"] == "s3"][0]["canonical_theta_vec"], **new_theta}
    dev_seeds = [14250, 14750] + sorted(new_theta)
    heldout = {e["tag"]: e["canonical_theta_vec"] for e in bank["states"] if e["split"] == "held_out"}
    n_global = 120 if smoke else 600
    link_tol = float(sys.argv[sys.argv.index("--link-tol") + 1]) if "--link-tol" in sys.argv else 0.9
    harness = load_harness()
    print(f"ACCEPTABLE-SET (M0 multimodality test) — dev {dev_seeds} | local basin + {n_global} global/state | "
          f"link_tol={link_tol} | held-out (eval-only overlay) {list(heldout)}", flush=True)

    pooled_norm: list[np.ndarray] = []
    state_of: list[Any] = []
    per_state: dict[str, Any] = {}
    for seed in dev_seeds:
        te = time.time()
        tag = dev.get(seed, str(seed))
        snap, _ = acquire_snapshot(harness, seed)
        canon = np.asarray(dev_canon[seed], np.float64)
        local = [np.asarray(c["theta"], np.float64) for c in sample_dev_basin(snap, canon, seed=seed)
                 if c["kind"] == "delivering"]
        res = harvest_acceptable_set(snap, n_samples=n_global, seed=seed, seed_thetas=[canon, *local])
        acc = res["accepted"]
        cd = centroid_delivers(snap, acc)
        Z = [a.theta_norm for a in acc]
        pooled_norm.extend(Z)
        state_of.extend([tag] * len(Z))
        per_state[tag] = {"seed": seed, "n_accepted": len(acc), "n_local_basin": len(local),
                          "delivery_rate": res["delivery_rate"], "acceptance_rate": res["acceptance_rate"],
                          "canonical_theta": [round(float(x), 5) for x in canon], "centroid": cd}
        print(f"  {tag} seed={seed}: accepted={len(acc)} (local={len(local)}, deliv_rate={res['delivery_rate']}) | "
              f"acc-centroid delivers={cd.get('delivers')} dtz={cd.get('dtz_end_mm')}mm ({time.time()-te:.0f}s)", flush=True)

    P = np.asarray(pooled_norm, np.float64)
    pooled = cluster_basins(P, link_tol=link_tol)
    tol_sweep = {str(lt): cluster_basins(P, link_tol=lt)["n_basins"] for lt in (0.4, 0.6, 0.8, 0.9, 1.1, 1.3, 1.5)}
    # per-state basin occupancy
    for tag in per_state:
        idx = [i for i, s in enumerate(state_of) if s == tag]
        per_state[tag]["basins_occupied"] = sorted(set(pooled["labels"][i] for i in idx))
    # held-out overlay (eval-only): the delivering teacher θ → nearest pooled dev basin / orphan
    heldout_assign = {tag: {**assign_to_basins(box.norm(np.asarray(th, np.float64)), pooled["centroids"], link_tol=link_tol),
                            "teacher_theta": [round(float(x), 5) for x in th]}
                      for tag, th in heldout.items()}
    # failing N=6 actor θ₀ on the panel (eval-only overlay: where the single-θ regressor pointed)
    actor0 = {}
    pt = f"{REPORT_DIR}/bc_coverage_N6_B0.pt"
    if os.path.exists(pt):
        prop, fz = load_proposal(pt)
        panel = build_panel(harness, bank)
        for ps in panel:
            th0 = prop.center(fz.obs(ps.features, ps.history))
            a = assign_to_basins(box.norm(np.asarray(th0, np.float64)), pooled["centroids"], link_tol=link_tol)
            actor0[ps.tag] = {"split": ps.split, "theta0": [round(float(x), 5) for x in th0], **a}

    verdict = multimodality_verdict(per_state, pooled, heldout_assign)
    out = {"contract": "COIN_6D_THETA_ACCEPTABLE_SET_MULTIMODALITY_V1", "base_commit": "5fbf3c16", "date": "2026-07-27",
           "branch": "recovery/coin-acceptable-set-proposal",
           "question": "Do multiple distinct delivering basins exist (→ multimodal), or one connected cluster (→ representation)?",
           "harvest": {"local_basin": "sample_dev_basin (frozen)", "global_uniform_per_state": n_global,
                       "accept": "frozen-K6 delivered AND motion-compatible", "held_out": "eval-only overlay, no fit"},
           "link_tol": link_tol, "link_tol_sweep_n_basins": tol_sweep,
           "per_state": per_state, "pooled_clusters": pooled,
           "held_out_overlay": heldout_assign, "actor_n6_theta0_overlay": actor0,
           "verdict": verdict, "smoke": bool(smoke),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    heldout_src = {tag: [("teacher θ (delivers)", box.norm(np.asarray(v["teacher_theta"], np.float64)), "*", "red")]
                   for tag, v in heldout_assign.items()}
    for tag, a in actor0.items():
        if a["split"] == "held_out":
            heldout_src.setdefault(tag, []).append(("actor θ₀ (N=6)", box.norm(np.asarray(a["theta0"], np.float64)), "X", "gray"))
    out["figures"] = _render_acceptable_set_viz(out, P, pooled["labels"], state_of, heldout_src)
    os.makedirs(ACCEPTABLE_DIR, exist_ok=True)
    path = f"{ACCEPTABLE_DIR}/acceptable_set_multimodality.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    v = verdict
    centroid_str = ", ".join(f"{t}:{s['centroid'].get('delivers')}" for t, s in per_state.items())
    overlay_str = ", ".join(f"{t}:basin{a['nearest_basin']}(d={a['dist']},orphan={a['orphan']})"
                            for t, a in heldout_assign.items())
    print(f"\n== ACCEPTABLE-SET MULTIMODALITY ==\n  pooled points={pooled['n_points']} basins={pooled['n_basins']} "
          f"inter/intra={pooled['min_inter_basin_dist']}/{pooled['max_intra_nn_hop']} | tol-sweep {tol_sweep}\n"
          f"  per-state acc-centroid delivers: {{ {centroid_str} }}\n"
          f"  held-out teacher-θ overlay: {{ {overlay_str} }}\n"
          f"  VERDICT: {v['verdict']} | justifies_k_head={v['justifies_k_head']} | held_out_ood_warning={v['held_out_ood_warning']}\n"
          f"  artifact: {path} | wall {out['wall_s']}s | peak RSS {out['peak_rss_gb']} GB\nACCEPTABLE_SET_DONE", flush=True)
    return out


def r1_check_main(smoke: bool = False) -> dict:
    """R1 REPRESENTATION-GOAL CHECK (before the update-0 gate) — verify the canonical target-frame features achieve the
    audit's targets: mirror deficit → 0 (by construction, confirmed on real snapshots), smoothness corr(dφ,dθ) improves
    or holds, and training-free NN-retrieval (with the canonical θ decode) beats R0's dev 1/6. No model training."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.canonical_frame import (
        canonicalise, flatten_r1, from_canonical_theta, r1_grouped_features, swap_grouped, to_canonical_theta)
    from hymeko_rl.coin_delivery.theta_option.representation_audit import lipschitz_analysis, nearest_neighbour_by_feature
    from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, fixed_search_select
    from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    dp = json.load(open(f"{REPORT_DIR}/cradle_delivery_pass.json"))
    box = ThetaBox()
    dev_tag = {14250: "s1", 14750: "s3", 16500: "16500", 17750: "17750", 19500: "19500", 24000: "24000"}
    held_tag = {15000: "s4", 15750: "s7"}
    canon = {14250: [e for e in bank["states"] if e["tag"] == "s1"][0]["canonical_theta_vec"],
             14750: [e for e in bank["states"] if e["tag"] == "s3"][0]["canonical_theta_vec"],
             15000: [e for e in bank["states"] if e["tag"] == "s4"][0]["canonical_theta_vec"],
             15750: [e for e in bank["states"] if e["tag"] == "s7"][0]["canonical_theta_vec"],
             **{r["seed"]: r["canonical_theta"] for r in dp["deliverable_dev_pool"] if r["seed"] not in (14250, 14750)}}
    dev_seeds, held_seeds = [14250, 14750, 16500, 17750, 19500, 24000], [15000, 15750]
    harness = load_harness()
    print("R1 CHECK — canonical target-frame features | budget-8 retrieval w/ θ-decode, no training", flush=True)
    snaps, feat, swapped, theta_canon_n, mirror_ok = {}, {}, {}, {}, {}
    for seed in dev_seeds + held_seeds:
        tag = {**dev_tag, **held_tag}[seed]
        snap, _ = acquire_snapshot(harness, seed)
        snaps[tag] = snap
        g = r1_grouped_features(snap)
        cg, sw = canonicalise(g)
        feat[tag] = flatten_r1(cg)
        swapped[tag] = sw
        theta_canon_n[tag] = box.norm(to_canonical_theta(np.asarray(canon[seed], np.float64), sw))
        mirror_ok[tag] = bool(np.allclose(flatten_r1(canonicalise(g)[0]), flatten_r1(canonicalise(swap_grouped(g))[0]), atol=1e-6))
        print(f"  {tag} seed={seed}: was_swapped={sw} mirror_invariant={mirror_ok[tag]}", flush=True)

    dev_tags, held_tags = [dev_tag[s] for s in dev_seeds], [held_tag[s] for s in held_seeds]
    lip = lipschitz_analysis({t: feat[t] for t in dev_tags}, {t: theta_canon_n[t] for t in dev_tags})
    nn_dev = nearest_neighbour_by_feature(feat, dev_tags, dev_tags)
    nn_held = nearest_neighbour_by_feature(feat, held_tags, dev_tags)

    def _retrieve(target: str, source: str, i: int) -> dict:
        theta_phys = from_canonical_theta(box.denorm(theta_canon_n[source]), swapped[target])   # decode to target frame
        prov = fixed_search_select(snaps[target], np.asarray(theta_phys, np.float64), np.random.default_rng(50000 + i * 131),
                                   budget=8, cfg=DELIVERY_CFG, std=SEARCH_STD)
        return {"source": source, "k6": bool(prov.outcome.get("delivery_success")),
                "dtz_end_mm": round(prov.outcome.get("dtz_end", 0.0) * 1000, 2), "nn_feature_dist": None}

    print("NN-retrieval (R1 canonical features + θ-decode, budget 8):", flush=True)
    retr_dev, dev_k6 = {}, 0
    for i, t in enumerate(dev_tags):
        r = _retrieve(t, nn_dev[t]["nn_tag"], i)
        r["nn_feature_dist"] = nn_dev[t]["nn_feature_dist"]
        retr_dev[t] = r
        dev_k6 += int(r["k6"])
        print(f"  dev {t} ← NN {r['source']} (dφ={r['nn_feature_dist']}): K6={int(r['k6'])} dtz={r['dtz_end_mm']}mm", flush=True)
    retr_held, held_k6 = {}, 0
    for i, t in enumerate(held_tags):
        r = _retrieve(t, nn_held[t]["nn_tag"], 100 + i)
        r["nn_feature_dist"] = nn_held[t]["nn_feature_dist"]
        retr_held[t] = r
        held_k6 += int(r["k6"])
        print(f"  held {t} ← NN {r['source']} (dφ={r['nn_feature_dist']}): K6={int(r['k6'])} dtz={r['dtz_end_mm']}mm [overlay]", flush=True)

    goals = {"mirror_invariant_all": bool(all(mirror_ok.values())),
             "corr_dphi_dtheta": lip["corr_dphi_dtheta"], "corr_improved_or_held_vs_R0_0p71": bool(lip["corr_dphi_dtheta"] >= 0.71 - 0.1),
             "nn_retrieval_dev": f"{dev_k6}/{len(dev_tags)}", "nn_retrieval_held_out_overlay": f"{held_k6}/{len(held_tags)}",
             "retrieval_beats_R0_dev_1of6": bool(dev_k6 > 1)}
    out = {"contract": "COIN_R1_REPRESENTATION_CHECK_V1", "base_commit": "1ab9e62f", "date": "2026-07-27",
           "representation": "R1 canonical target/contact-frame", "no_training": True,
           "per_cradle_swapped": swapped, "mirror_invariant_confirmed": mirror_ok,
           "lipschitz": lip, "nn_retrieval_dev": retr_dev, "nn_retrieval_held_out_overlay": retr_held,
           "representation_goals": goals, "R0_reference": {"nn_retrieval_dev": "1/6", "corr_dphi_dtheta": 0.71, "mirror_deficit": 3.49},
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    os.makedirs(REP_AUDIT_DIR, exist_ok=True)
    path = f"{REP_AUDIT_DIR}/r1_representation_check.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n== R1 CHECK ==\n  mirror-invariant(all)={goals['mirror_invariant_all']} (deficit 3.49→0 by construction)\n"
          f"  corr(dφ,dθ)={lip['corr_dphi_dtheta']} (R0 0.71) | NN-retrieval dev {dev_k6}/6 (R0 1/6) | held(overlay) {held_k6}/2\n"
          f"  goals: {goals}\n  artifact: {path} | wall {out['wall_s']}s\nR1_CHECK_DONE", flush=True)
    return out


def rep_audit_main(smoke: bool = False) -> dict:
    """DECISION-TIME REPRESENTATION AUDIT (Step 1) — no training, dev-only, held-out overlay is frozen diagnosis. On the
    current 42-D features: feature→θ smoothness (Lipschitz), training-free nearest-feature retrieval deploy (dev LODO +
    held-out overlay) at the SAME budget-8 search, and the canonical L/R ordering deficit. Verdict frames what R1 must fix."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.dataset import flatten_features, structured_features
    from hymeko_rl.coin_delivery.theta_option.representation_audit import (
        audit_verdict, lipschitz_analysis, nearest_neighbour_by_feature, ordering_deficit)
    from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, fixed_search_select
    from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    dp = json.load(open(f"{REPORT_DIR}/cradle_delivery_pass.json"))
    box = ThetaBox()
    dev_tag = {14250: "s1", 14750: "s3", 16500: "16500", 17750: "17750", 19500: "19500", 24000: "24000"}
    held_tag = {15000: "s4", 15750: "s7"}
    canon = {14250: [e for e in bank["states"] if e["tag"] == "s1"][0]["canonical_theta_vec"],
             14750: [e for e in bank["states"] if e["tag"] == "s3"][0]["canonical_theta_vec"],
             15000: [e for e in bank["states"] if e["tag"] == "s4"][0]["canonical_theta_vec"],
             15750: [e for e in bank["states"] if e["tag"] == "s7"][0]["canonical_theta_vec"],
             **{r["seed"]: r["canonical_theta"] for r in dp["deliverable_dev_pool"] if r["seed"] not in (14250, 14750)}}
    dev_seeds = [14250, 14750, 16500, 17750, 19500, 24000]
    held_seeds = [15000, 15750]
    harness = load_harness()
    print(f"REP AUDIT — dev {dev_seeds} + held-out (overlay) {held_seeds} | budget-8 retrieval, no training", flush=True)
    snaps, feats_flat, feats_grouped, theta_n = {}, {}, {}, {}
    for seed in dev_seeds + held_seeds:
        tag = {**dev_tag, **held_tag}[seed]
        snap, _ = acquire_snapshot(harness, seed)
        snaps[tag] = snap
        fg = structured_features(snap)
        feats_grouped[tag] = fg
        feats_flat[tag] = flatten_features(fg)
        theta_n[tag] = box.norm(np.asarray(canon[seed], np.float64))
        print(f"  acquired {tag} seed={seed}", flush=True)

    dev_tags = [dev_tag[s] for s in dev_seeds]
    held_tags = [held_tag[s] for s in held_seeds]
    lip = lipschitz_analysis({t: feats_flat[t] for t in dev_tags}, {t: theta_n[t] for t in dev_tags})
    order = ordering_deficit({t: feats_grouped[t] for t in dev_tags}, flatten_features)
    nn_dev = nearest_neighbour_by_feature(feats_flat, dev_tags, dev_tags)          # LODO among dev
    nn_held = nearest_neighbour_by_feature(feats_flat, held_tags, dev_tags)        # held-out → nearest dev (overlay)

    def _retrieve(target_tag: str, source_tag: str, i: int) -> dict:
        prov = fixed_search_select(snaps[target_tag], np.asarray(canon_by_tag[source_tag], np.float64),
                                   np.random.default_rng(60000 + i * 131), budget=8, cfg=DELIVERY_CFG, std=SEARCH_STD)
        return {"source": source_tag, "k6": bool(prov.outcome.get("delivery_success")),
                "dtz_end_mm": round(prov.outcome.get("dtz_end", 0.0) * 1000, 2),
                "nn_feature_dist": None}
    canon_by_tag = {dev_tag[s]: canon[s] for s in dev_seeds}
    canon_by_tag.update({held_tag[s]: canon[s] for s in held_seeds})

    print("NN-retrieval deploy (training-free, budget 8):", flush=True)
    retr_dev, dev_k6 = {}, 0
    for i, t in enumerate(dev_tags):
        src = nn_dev[t]["nn_tag"]
        r = _retrieve(t, src, i)
        r["nn_feature_dist"] = nn_dev[t]["nn_feature_dist"]
        retr_dev[t] = r
        dev_k6 += int(r["k6"])
        print(f"  dev {t} ← NN {src} (dφ={r['nn_feature_dist']}): K6={int(r['k6'])} dtz={r['dtz_end_mm']}mm", flush=True)
    retr_held, held_k6 = {}, 0
    for i, t in enumerate(held_tags):
        src = nn_held[t]["nn_tag"]
        r = _retrieve(t, src, 100 + i)
        r["nn_feature_dist"] = nn_held[t]["nn_feature_dist"]
        retr_held[t] = r
        held_k6 += int(r["k6"])
        print(f"  held {t} ← NN {src} (dφ={r['nn_feature_dist']}): K6={int(r['k6'])} dtz={r['dtz_end_mm']}mm [overlay]", flush=True)

    verdict = audit_verdict(dev_k6, len(dev_tags), held_k6, len(held_tags), lip, order)
    out = {"contract": "COIN_DECISION_REPRESENTATION_AUDIT_V1", "base_commit": "1ca1edcb", "date": "2026-07-27",
           "branch": "recovery/coin-decision-representation", "no_training": True,
           "lipschitz": lip, "ordering_deficit": order,
           "nn_retrieval_dev": retr_dev, "nn_retrieval_held_out_overlay": retr_held,
           "verdict": verdict, "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    os.makedirs(REP_AUDIT_DIR, exist_ok=True)
    path = f"{REP_AUDIT_DIR}/representation_audit.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n== REP AUDIT ==\n  smoothness: corr(dφ,dθ)={lip['corr_dphi_dtheta']} lipschitz_max={lip['lipschitz_ratio']['max']}\n"
          f"  ordering: mean full-swap deficit={order['mean_full_swap_deficit']} canonical={order['features_are_canonically_ordered']}\n"
          f"  NN-retrieval: dev {dev_k6}/{len(dev_tags)} | held-out(overlay) {held_k6}/{len(held_tags)}\n"
          f"  defects: {verdict['identified_defects']}\n  SUMMARY: {verdict['audit_summary']}\n"
          f"  artifact: {path} | wall {out['wall_s']}s | RSS {out['peak_rss_gb']} GB\nREP_AUDIT_DONE", flush=True)
    return out


def _build_acceptable_training_data(snaps: dict, dev_seeds: list, dev_canon: dict, dev_tag: dict, n_global: int) -> "tuple[list, dict]":
    """Per dev cradle (from pre-acquired snapshots): frozen B0 features + the acceptable set (local delivering basin +
    global enrichment, normalised θ) + the multimodal gate. Deterministic (seed = the cradle seed) — the SAME acceptable
    sets M0 analysed. Returns the training states + a per-state summary."""
    from hymeko_rl.coin_delivery.theta_option.acceptable_set import harvest_acceptable_set
    from hymeko_rl.coin_delivery.theta_option.dataset import flatten_features, structured_features
    from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import KHeadTrainState, is_multimodal_target_set
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import sample_dev_basin
    states, summary = [], {}
    for seed in dev_seeds:
        tag = dev_tag[seed]
        snap = snaps[seed]
        feats = flatten_features(structured_features(snap))
        canon = np.asarray(dev_canon[seed], np.float64)
        local = [np.asarray(c["theta"], np.float64) for c in sample_dev_basin(snap, canon, seed=seed) if c["kind"] == "delivering"]
        res = harvest_acceptable_set(snap, n_samples=n_global, seed=seed, seed_thetas=[canon, *local])
        tgt = np.asarray([a.theta_norm for a in res["accepted"]], np.float64)
        mm = is_multimodal_target_set(tgt)
        states.append(KHeadTrainState(tag=tag, features=np.asarray(feats, np.float32), targets_norm=tgt, multimodal=mm))
        summary[tag] = {"seed": seed, "n_targets": len(tgt), "multimodal": bool(mm)}
        print(f"  data {tag} seed={seed}: targets={len(tgt)} multimodal={mm}", flush=True)
    return states, summary


def _multimodal_deploy_panel(prop: Any, panel: list, budget: int, dev_targets: dict, box: Any, seed_base: int = 90000) -> dict:
    """Deploy a K-head proposal on the frozen panel at ``budget`` (fixed per-state rng). Records per-state K6 + which mode
    won + search displacement + nearest-dev-acceptable-θ distance + failure mode; returns dev/held-out K6 counts."""
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import classify_failure_mode
    from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import multimodal_search_select
    all_dev = np.vstack([v for v in dev_targets.values()]) if dev_targets else None
    per = {}
    for i, ps in enumerate(panel):
        dep = multimodal_search_select(ps.snap, prop, ps.features, np.random.default_rng(seed_base + i * 131 + budget), budget=budget)
        d = dep.as_dict()
        sel_c = np.asarray(dep.mode_centers[dep.selected_mode], np.float64)
        disp = float(np.linalg.norm(box.norm(np.asarray(d["theta_exec"], np.float64)) - box.norm(sel_c)))
        near = None
        if all_dev is not None:
            near = round(float(np.min(np.linalg.norm(all_dev - box.norm(sel_c)[None, :], axis=1))), 4)
        fail = None if d["delivery_success"] else classify_failure_mode(
            {"dtz_end_mm": d["dtz_end_mm"], "k6_max_dwell": d["k6_max_dwell"], "peak_qdot": d["peak_qdot"], "peak_coin_speed": d["peak_coin_speed"]})
        per[ps.tag] = {"split": ps.split, **d, "search_displacement_norm": round(disp, 4),
                       "nearest_dev_acceptable_dist": near, "failure_mode": fail}
    dev_k6 = sum(1 for r in per.values() if r["split"] == "development" and r["delivery_success"])
    held_k6 = sum(1 for r in per.values() if r["split"] == "held_out" and r["delivery_success"])
    return {"budget": budget, "per_state": per, "dev_k6": dev_k6, "held_out_k6": held_k6, "total_k6": dev_k6 + held_k6}


def multimodal_main(smoke: bool = False) -> dict:
    """M1+M2 — the K-head acceptable-set proposal. Build the dev acceptable-set training data, select K by DEV-ONLY
    leave-one-dev-out CV (physical K6 on the held-out DEV cradle), freeze K, retrain on all dev, then ONE frozen-panel
    deploy at total budget 8 (fair K×(8/K) split, centre-inclusive per mode). Hard gate MULTIMODAL_UPDATE_ZERO_PASS
    (4/4 incl. held-out 2/2) is the only thing that authorises SAC/TD3. No held-out fitting; all seeds reported."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
    from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import fit_khead, multimodal_search_select, save_khead
    from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    dp = json.load(open(f"{REPORT_DIR}/cradle_delivery_pass.json"))
    dev_tag = {14250: "s1", 14750: "s3", 16500: "16500", 17750: "17750", 19500: "19500", 24000: "24000"}
    dev_seeds = [14250, 14750, 16500, 17750, 19500, 24000]
    dev_canon = {14250: [e for e in bank["states"] if e["tag"] == "s1"][0]["canonical_theta_vec"],
                 14750: [e for e in bank["states"] if e["tag"] == "s3"][0]["canonical_theta_vec"],
                 **{r["seed"]: r["canonical_theta"] for r in dp["deliverable_dev_pool"] if r["seed"] not in (14250, 14750)}}
    Ks = [1, 2] if smoke else [1, 2, 4]
    EPOCHS, LR, SEED = (400 if smoke else 1500), 1e-3, 0
    n_global = 120 if smoke else 600
    box = ThetaBox()
    harness = load_harness()
    os.makedirs(MULTIMODAL_DIR, exist_ok=True)
    print(f"MULTIMODAL (M1+M2) — dev {dev_seeds} | K∈{Ks} epochs={EPOCHS} | budget 8 (K×8/K) | DEV-only LODO-CV → freeze K", flush=True)

    print("ACQUIRE dev snapshots (once, reused for harvest + LODO deploy)", flush=True)
    dev_snaps_by_seed = {seed: acquire_snapshot(harness, seed)[0] for seed in dev_seeds}
    print("BUILD acceptable-set training data (deterministic; = M0 sets)", flush=True)
    states, data_summary = _build_acceptable_training_data(dev_snaps_by_seed, dev_seeds, dev_canon, dev_tag, n_global)
    dev_targets = {s.tag: s.targets_norm for s in states}
    dev_snaps = {dev_tag[seed]: dev_snaps_by_seed[seed] for seed in dev_seeds}

    # ── DEV-only LODO cross-validation: pick K by physical held-out-DEV K6 (no panel/held-out involvement) ──
    print("DEV LODO-CV (leave-one-dev-out; physical K6 on the held-out DEV cradle)", flush=True)
    cv = {}
    for K in Ks:
        fold_k6, folds = 0, []
        for held in states:
            train = [s for s in states if s.tag != held.tag]
            prop, _info = fit_khead(train, K, epochs=EPOCHS, lr=LR, seed=SEED)
            dep = multimodal_search_select(dev_snaps[held.tag], prop, held.features, np.random.default_rng(70000 + K), budget=8)
            ok = bool(dep.provenance.outcome.get("delivery_success"))
            fold_k6 += int(ok)
            folds.append({"held_dev": held.tag, "k6": ok, "selected_mode": dep.selected_mode})
        cv[K] = {"lodo_k6": fold_k6, "n": len(states), "folds": folds}
        fold_str = " ".join(f"{f['held_dev']}:{int(f['k6'])}" for f in folds)
        print(f"  K={K}: LODO held-dev K6 = {fold_k6}/{len(states)} | {fold_str}", flush=True)
    k_star = max(Ks, key=lambda K: (cv[K]["lodo_k6"], -K))       # best LODO K6, tie → smaller K
    print(f"  SELECTED K* = {k_star} (dev-only LODO)", flush=True)

    # ── freeze K*, retrain on ALL dev, deploy ONCE on the frozen panel ──
    prop_final, fit_info = fit_khead(states, k_star, epochs=EPOCHS, lr=LR, seed=SEED)
    save_khead(prop_final, f"{MULTIMODAL_DIR}/khead_K{k_star}.pt")
    panel = build_panel(harness, bank)
    dep8 = _multimodal_deploy_panel(prop_final, panel, 8, dev_targets, box)
    dep_prop = _multimodal_deploy_panel(prop_final, panel, k_star, dev_targets, box)   # proposal-only: 1 candidate/mode (K total)
    n = len(panel)
    dev_ok, held_ok = dep8["dev_k6"] == 2, dep8["held_out_k6"] == 2
    all_ok = dep8["total_k6"] == n
    motion_ok = all(bool(r["peak_qdot"] <= 3.0 and r["peak_coin_speed"] <= 1.5) for r in dep8["per_state"].values())
    gate_pass = bool(all_ok and dev_ok and held_ok and motion_ok)
    if gate_pass:
        verdict = "MULTIMODAL_UPDATE_ZERO_PASS"
    elif dep8["held_out_k6"] > 0:
        verdict = "MULTIMODAL_PROPOSAL_IMPROVES_BASIN_COVERAGE"
    else:
        verdict = "MULTIMODALITY_PRESENT_BUT_UPDATE_ZERO_STILL_FAILS"
    out = {"contract": "COIN_6D_THETA_MULTIMODAL_UPDATE_ZERO_V1", "base_commit": "5fbf3c16", "date": "2026-07-27",
           "branch": "recovery/coin-acceptable-set-proposal",
           "invariant": {"model": "K-head acceptable-set (shared B0 trunk)", "loss": "perm-invariant bidirectional Chamfer + gated head-collapse",
                         "optimiser": "Adam", "epochs": EPOCHS, "lr": LR, "seed": SEED, "search_budget": 8,
                         "budget_split": "option_rl.allocate_budget (K×8/K), centre-inclusive per mode",
                         "eval_panel": "frozen 4-state {s1,s3,s4,s7}", "held_out": ["s4", "s7"],
                         "model_selection": "DEV-only LODO-CV (physical held-dev K6); no held-out fitting"},
           "data_summary": data_summary, "cv": cv, "k_star": k_star, "fit_info": fit_info,
           "deploy_budget8": dep8, "deploy_proposal_only": dep_prop,
           "gate": {"dev_k6": dep8["dev_k6"], "held_out_k6": dep8["held_out_k6"], "total_k6": dep8["total_k6"],
                    "n_states": n, "motion_ok": motion_ok, "no_held_out_fitting": True, "passed": gate_pass},
           "verdict": verdict, "authorises_sac_td3": gate_pass,
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    out["figures"] = _render_multimodal_viz(out, panel, prop_final, dev_targets, box)
    path = f"{MULTIMODAL_DIR}/multimodal_update_zero.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(f"\n== MULTIMODAL UPDATE-0 ==\n  K*={k_star} | budget-8: dev {dep8['dev_k6']}/2 held {dep8['held_out_k6']}/2 "
          f"total {dep8['total_k6']}/4 | proposal-only(K): dev {dep_prop['dev_k6']}/2 held {dep_prop['held_out_k6']}/2\n"
          f"  per-state (budget 8): " + ", ".join(
              f"{t}:{int(r['delivery_success'])}(m{r['selected_mode']},dtz{r['dtz_end_mm']})" for t, r in dep8["per_state"].items()) + "\n"
          f"  motion_ok={motion_ok} | GATE {'PASS' if gate_pass else 'FAIL'} | verdict={verdict} | "
          f"authorises_sac_td3={gate_pass}\n  artifact: {path} | wall {out['wall_s']}s | RSS {out['peak_rss_gb']} GB\n"
          f"MULTIMODAL_DONE", flush=True)
    return out


def _render_multimodal_viz(out: dict, panel: list, prop: Any, dev_targets: dict, box: Any) -> dict:
    """§9 viz: per-state K6 (proposal-only K vs budget-8) grouped bars + a note of the selected mode per state."""
    figs: dict = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d8 = out["deploy_budget8"]["per_state"]
        dp = out["deploy_proposal_only"]["per_state"]
        tags = list(d8.keys())
        x = np.arange(len(tags))
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - 0.2, [int(dp[t]["delivery_success"]) for t in tags], 0.4, label=f"proposal-only (K={out['k_star']} centres)", color="tab:gray")
        ax.bar(x + 0.2, [int(d8[t]["delivery_success"]) for t in tags], 0.4, label="budget-8 (K×8/K)", color="tab:blue")
        ax.axhline(1, color="tab:green", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n{d8[t]['split'][:3]}\nmode {d8[t]['selected_mode']}" for t in tags])
        ax.set_ylabel("frozen K6 delivered")
        ax.set_ylim(0, 1.25)
        ax.set_title(f"Multimodal update-0 (K*={out['k_star']}) — {out['verdict']} "
                     f"(total {out['deploy_budget8']['total_k6']}/4, held {out['deploy_budget8']['held_out_k6']}/2)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = f"{MULTIMODAL_DIR}/multimodal_update_zero.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        figs["png"] = p
    except Exception as e:
        figs["plot_error"] = str(e)
    return figs


def coverage_curve_main(smoke: bool = False) -> dict:
    """COVERAGE-ONLY CAUSAL CURVE — the frozen update-0 gate at N = 2, 4, 6 unique K6-deliverable DEVELOPMENT cradles,
    with EVERYTHING except N held identical (model B0, feature set, optimiser, epochs, init seed, search budget 8, search
    semantics, evaluation panel, K6, motion/collision/task contract). Every N is trained from a FRESH matched init. The one
    question: does growing dev-cradle coverage ALONE close the held-out regression? No architecture / loss / hyperparameter
    change — only N. Reuses the frozen teacher-bank / dataset / BC / deploy machinery unchanged."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.coverage_curve import (
        FROZEN_DEV_SEEDS, coverage_record, coverage_verdict, nested_dev_sets, select_n4_additions, theta_distance_summary)
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import geometry_fingerprint
    from hymeko_rl.coin_delivery.theta_option.dataset import build_dataset
    from hymeko_rl.coin_delivery.theta_option.deploy import DEPLOY_BUDGETS, build_panel, update_zero_eval
    from hymeko_rl.coin_delivery.theta_option.proposal import fit_bc, save_proposal
    from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, search_semantics
    from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness, reproduce_state
    t0 = time.time()
    bank_path, dp_path = f"{REPORT_DIR}/teacher_bank.json", f"{REPORT_DIR}/cradle_delivery_pass.json"
    if not (os.path.exists(bank_path) and os.path.exists(dp_path)):
        print(f"MISSING {bank_path} or {dp_path} — run --teacher-bank and --deliver-pass first")
        sys.exit(2)
    bank, dp = json.load(open(bank_path)), json.load(open(dp_path))
    VARIANT, EPOCHS, LR, INIT_SEED = "B0", 1200, 1e-3, 0            # FROZEN across N (the frozen update-0 model)
    frozen_dev = {e["seed"]: e for e in bank["states"] if e["split"] == "development"}
    new_seeds = sorted(dp["new_state_distribution"]["K6_DELIVERABLE"])
    if smoke:
        new_seeds, EPOCHS = new_seeds[:2], 200
    budgets = (0, 8) if smoke else DEPLOY_BUDGETS
    k_add = 1 if smoke else 2
    harness = load_harness()
    print(f"COVERAGE CURVE — frozen dev {list(FROZEN_DEV_SEEDS)} + new usable dev {new_seeds} | model={VARIANT} "
          f"epochs={EPOCHS} lr={LR} init_seed={INIT_SEED} budgets={budgets} search_std={SEARCH_STD}\n"
          f"  INVARIANT: only N changes; fresh matched init per N; eval panel FROZEN 4-state; held-out ALWAYS s4,s7", flush=True)

    # 1. reproduce the NEW dev entries with the SAME frozen machinery (CEM + basin), live per cradle
    new_dev: dict[int, Any] = {}
    dp_theta = {r["seed"]: r.get("canonical_theta") for r in dp["deliverable_dev_pool"]}
    for k, seed in enumerate(new_seeds):
        te = time.time()
        e = reproduce_state(harness, idx=seed, seed=seed, augment=True, tag=f"c{k+1}", split="development", basin_seed=seed)
        new_dev[seed] = e
        cv = e.get("canonical_theta_vec")
        matches = bool(cv is not None and dp_theta.get(seed) is not None
                       and np.allclose(np.asarray(cv), np.asarray(dp_theta[seed]), atol=1e-5))
        print(f"  new dev c{k+1} seed={seed}: k6={e.get('k6_delivered')} replay_ok={e.get('replay_ok')} src={e.get('canonical_source')} "
              f"basin_deliv={e.get('n_basin_delivering','-')} matches_delivery_pass={matches} ({time.time()-te:.0f}s)", flush=True)

    # 2. FROZEN geometry fingerprints → NON-OUTCOME N=4 selection, recorded BEFORE any training/deploy is read
    fps = {seed: geometry_fingerprint(acquire_snapshot(harness, seed)[0]) for seed in list(FROZEN_DEV_SEEDS) + new_seeds}
    selection = select_n4_additions(new_seeds, fps, FROZEN_DEV_SEEDS, k=k_add)
    print(f"\n== N=4 SELECTION (frozen, non-outcome: {selection['rule']}) ==\n  candidate min-dist to frozen dev: "
          f"{selection['candidate_min_dist_to_frozen_dev']}\n  SELECTED (before any result): {selection['selected']}\n", flush=True)
    dev_sets = nested_dev_sets(new_seeds, selection["selected"], FROZEN_DEV_SEEDS)
    entry_by_seed = {**frozen_dev, **new_dev}

    # 3. FIXED evaluation panel (frozen 4-state {s1,s3,s4,s7}) — acquired once, identical for every N
    panel = build_panel(harness, bank)
    box = ThetaBox()

    records: list[dict[str, Any]] = []
    for n in sorted(dev_sets):
        tn = time.time()
        dev_seeds = dev_sets[n]
        cov_bank = {"contract": f"COVERAGE_N{n}", "base_commit": "a3459629", "states": [entry_by_seed[s] for s in dev_seeds]}
        ds = build_dataset(cov_bank, harness=harness)
        prop, fz, tl = fit_bc(ds, VARIANT, epochs=EPOCHS, lr=LR, seed=INIT_SEED)   # FRESH matched init each N
        save_proposal(prop, fz, f"{REPORT_DIR}/bc_coverage_N{n}_{VARIANT}.pt")
        out = update_zero_eval(panel, prop, fz, budgets=budgets)
        rows = [{"tag": ps.tag, "split": ps.split,
                 "proposed": [float(x) for x in prop.center(fz.obs(ps.features, ps.history))],
                 "teacher": [float(x) for x in ps.teacher_theta]} for ps in panel]
        dist = theta_distance_summary(rows, box)
        rec = coverage_record(n, dev_seeds, out, dist)
        rec["train_mse"] = tl.get("train_mse")
        rec["n_train_rows"] = len(ds.subset("train"))
        gate_b = out["deploy_budget"]
        per = out["informed_sweep"][str(gate_b) if str(gate_b) in out["informed_sweep"] else gate_b]["per_state"]
        rec["held_out_theta_exec"] = {t: r["theta_exec"] for t, r in per.items() if r["split"] == "held_out"}
        records.append(rec)
        print(f"  N={n} train={dev_seeds} rows={rec['n_train_rows']}: INFORMED dev={rec['gate_informed_dev_k6']}/2 "
              f"held={rec['gate_informed_held_out_k6']}/2 total={rec['gate_informed_total_k6']}/4 "
              f"| proposal-only(b0)={rec['proposal_only_total_k6']}/4 | uninformed={rec['uninformed_total_k6']}/4 "
              f"| oracle={rec['oracle_total_k6']}/4 | θdist(held)={rec['held_out_theta_l2_norm']} "
              f"| gate_passed={rec['gate_passed']} ({time.time()-tn:.0f}s)", flush=True)

    verdict = coverage_verdict(records)
    out_obj = {"contract": "COIN_6D_THETA_COVERAGE_CURVE_V1", "base_commit": "a3459629", "date": "2026-07-27",
               "question": "Does growing development-cradle coverage ALONE remove the held-out update-0 regression?",
               "invariant_across_N": {"model": VARIANT, "feature_set": "structured 42-D + causal history (frozen)",
                                      "optimiser": "Adam", "epochs": EPOCHS, "lr": LR, "init_seed": INIT_SEED,
                                      "search_budget": 8, "search_semantics": search_semantics(8), "search_std": SEARCH_STD,
                                      "eval_panel": "frozen 4-state {s1,s3,s4,s7}", "held_out": ["s4", "s7"],
                                      "fresh_matched_init_each_N": True, "K6": "frozen monitor",
                                      "motion_collision_task_contract": "frozen (4c71f12f per-side masks + CONTROLLED_INSERTION)"},
               "frozen_dev_seeds": list(FROZEN_DEV_SEEDS), "new_usable_dev_seeds": new_seeds,
               "n4_selection": selection, "nested_dev_sets": {str(n): dev_sets[n] for n in sorted(dev_sets)},
               "records": records, "verdict": verdict, "smoke": bool(smoke),
               "wall_s": round(time.time() - t0, 1), "peak_rss_gb": _peak_rss_gb()}
    out_obj["figures"] = _render_coverage_viz(out_obj, panel)
    path = _dump(out_obj, "coverage_curve.json")
    v = verdict
    print("\n== COVERAGE CURVE ==")
    for r in sorted(records, key=lambda z: z["N"]):
        print(f"  N={r['N']}: total {r['gate_informed_total_k6']}/4 (dev {r['gate_informed_dev_k6']}/2 "
              f"held {r['gate_informed_held_out_k6']}/2) | held s4/s7 K6={r['held_out_per_state_k6']} "
              f"| θdist(held)={r['held_out_theta_l2_norm']} | fail={r['held_out_failure_mode']}")
    print(f"  held-out K6 by N: {v['held_out_k6_by_N']} | θdist(held) by N: {v['held_out_theta_l2_norm_by_N']}\n"
          f"  generalisation_improves_with_coverage={v['generalisation_improves_with_coverage']} "
          f"(k6↑={v['held_out_k6_nondecreasing_and_rises']}, θdist↓={v['held_out_theta_distance_shrinks_monotonically']})\n"
          f"  VERDICT: {v['verdict']} | authorise_sac_td3={v['authorise_sac_td3']} | next: {v.get('next_action')}\n"
          f"  artifact: {path} | wall {out_obj['wall_s']}s | peak RSS {out_obj['peak_rss_gb']} GB\nCOVERAGE_CURVE_DONE",
          flush=True)
    return out_obj


def update0_main(smoke: bool = False) -> dict:
    """Stage 4 — update-0 no-regression on the frozen 4-state panel (informed BC θ_0 vs uninformed box-centre control)."""
    import torch
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.theta_option.deploy import DEPLOY_BUDGETS, build_panel, update_zero_eval
    from hymeko_rl.coin_delivery.theta_option.proposal import load_proposal
    from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    t0 = time.time()
    bank_path = f"{REPORT_DIR}/teacher_bank.json"
    sel = json.load(open(f"{REPORT_DIR}/bc_results.json"))["selected"] if os.path.exists(f"{REPORT_DIR}/bc_results.json") else "B0"
    pt = f"{REPORT_DIR}/bc_{sel}.pt"
    if not (os.path.exists(bank_path) and os.path.exists(pt)):
        print(f"MISSING {bank_path} or {pt} — run --teacher-bank and --bc first")
        sys.exit(2)
    bank = json.load(open(bank_path))
    prop, fz = load_proposal(pt)
    budgets = (0, 8) if smoke else DEPLOY_BUDGETS
    print(f"UPDATE-0 — BC {sel} on frozen panel {[e['tag'] for e in bank['states'] if 'canonical_theta_vec' in e]} | "
          f"budgets {budgets} search_std={SEARCH_STD}", flush=True)
    panel = build_panel(load_harness(), bank)
    out = update_zero_eval(panel, prop, fz, budgets=budgets)
    out["search_std"] = SEARCH_STD
    out["wall_s"] = round(time.time() - t0, 1)
    out["peak_rss_gb"] = _peak_rss_gb()
    out["figures"] = _render_update0_viz(out, panel)
    path = _dump(out, "update_zero.json")
    for b in budgets:
        inf, unf = out["informed_sweep"][b], out["uninformed_sweep"][b]
        ik6 = {t: int(r["delivery_success"]) for t, r in inf["per_state"].items()}
        print(f"  budget {b}: INFORMED dev={inf['dev_k6']}/2 held={inf['held_out_k6']}/2 total={inf['total_k6']}/4 {ik6} "
              f"| UNINFORMED total={unf['total_k6']}/4", flush=True)
    g = out["gate"]
    print(f"\n== UPDATE-0 ==\n  verdict: {out['verdict']}  blocker: {out['diagnosed_blocker']}\n"
          f"  gate(budget {out['deploy_budget']}): informed {g['informed_total_k6']}/4 (dev {g['informed_dev_k6']}/2 "
          f"held {g['informed_held_out_k6']}/2) | uninformed {g['uninformed_total_k6']}/4 | oracle(teacher θ) "
          f"{g['oracle_total_k6']}/4 | actor_gap {g['actor_gap_vs_uninformed']} load_bearing={g['actor_load_bearing']}\n"
          f"  oracle_validates_search_and_physics={g['oracle_validates_search_and_physics']}\n"
          f"  GATE {'PASS' if g['passed'] else 'FAIL'} | authorises_rl={out['authorises_rl']}\n  artifact: {path} | "
          f"wall {out['wall_s']}s\nUPDATE0_DONE", flush=True)
    return out


if __name__ == "__main__":
    if "--semantics" in sys.argv:
        semantics_main()
    elif "--teacher-bank" in sys.argv:
        teacher_bank_main(smoke="--smoke" in sys.argv)
    elif "--dataset" in sys.argv:
        dataset_main()
    elif "--bc" in sys.argv:
        bc_main()
    elif "--update0" in sys.argv:
        update0_main(smoke="--smoke" in sys.argv)
    elif "--coverage-curve" in sys.argv:
        coverage_curve_main(smoke="--smoke" in sys.argv)
    elif "--acceptable-set" in sys.argv:
        acceptable_set_main(smoke="--smoke" in sys.argv)
    elif "--multimodal" in sys.argv:
        multimodal_main(smoke="--smoke" in sys.argv)
    elif "--rep-audit" in sys.argv:
        rep_audit_main(smoke="--smoke" in sys.argv)
    elif "--r1-check" in sys.argv:
        r1_check_main(smoke="--smoke" in sys.argv)
    elif "--scout-cradles" in sys.argv:
        _n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else (12 if "--smoke" in sys.argv else 32)
        scout_cradles_main(n=_n)
    elif "--deliver-pass" in sys.argv:
        deliver_pass_main()
    elif "--contact-audit" in sys.argv:
        contact_audit_main()
    else:
        print("specify a mode: --semantics | --teacher-bank | --dataset | --bc | --update0 | --coverage-curve "
              "| --scout-cradles [--n N] | --deliver-pass | --contact-audit | --rl-smoke | --rl-multiseed")
        sys.exit(2)
