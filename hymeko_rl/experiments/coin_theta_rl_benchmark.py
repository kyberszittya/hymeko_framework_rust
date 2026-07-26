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
    elif "--scout-cradles" in sys.argv:
        _n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else (12 if "--smoke" in sys.argv else 32)
        scout_cradles_main(n=_n)
    elif "--deliver-pass" in sys.argv:
        deliver_pass_main()
    else:
        print("specify a mode: --semantics | --teacher-bank | --dataset | --bc | --update0 | --scout-cradles [--n N] "
              "| --rl-smoke | --rl-multiseed")
        sys.exit(2)
