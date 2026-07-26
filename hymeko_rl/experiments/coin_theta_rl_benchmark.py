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


if __name__ == "__main__":
    if "--semantics" in sys.argv:
        semantics_main()
    elif "--teacher-bank" in sys.argv:
        teacher_bank_main(smoke="--smoke" in sys.argv)
    elif "--dataset" in sys.argv:
        dataset_main()
    elif "--bc" in sys.argv:
        bc_main()
    else:
        print("specify a mode: --semantics | --teacher-bank | --dataset | --bc | --update0 | --rl-smoke | --rl-multiseed")
        sys.exit(2)
