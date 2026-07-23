"""RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1 entry (label-only; no critic, no actor, no SAC).

Modes:
  freeze <pi0> <config.json>            capture the panel, write the FROZEN preregistration (state manifest, candidate
                                        manifest, K values, metric suite, thresholds, bootstrap) — commit BEFORE running.
  run    <pi0> <config.json> <out.json> [scale]  re-capture (verify manifest SHA), run the sweep, write results+verdict.
"""
import hashlib
import json
import sys
import time


sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_counterfactual_labels import capture_state_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_hold_sweep import (  # noqa: E402
    BENEFIT_EPS,
    FROZEN_ISO_SEED,
    K_VALUES,
    NONNEG_THRESHOLDS,
    hold_candidates,
    metrics_by_K_family,
    paired_bootstrap_vs_k1,
    sweep_group,
    sweep_verdict,
)
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

DEV_BANK = (6100, 6200)          # SAME bank family as the (disjoint-from-policy) critic std_dev; recaptured fresh
N_ISO = 3
METRIC_SUITE = ["median_abs_dG", "iqr_abs_dG", "frac_nonneg_1", "frac_nonneg_5", "frac_nonneg_10",
                "median_best_worst_gap", "beneficial_frac", "neutral_frac", "harmful_frac", "best_action_stable",
                "prob_contact_break", "prob_target_exit", "median_dwell_change", "strict_success_gain",
                "median_eff_clip_loss"]
BOOTSTRAP = {"n_boot": 4000, "by": "state_group_id", "statistic": "median_abs_dG_per_group", "seed": 0}
LEVERAGE_GATE = {"rule": "paired ci95(K - K1) of per-group median|ΔG| excludes 0 above; monotone up to eligible K",
                 "eligible_K_requires": "ci95_low > 0"}


def _state_manifest(groups):
    rows = [[g.group_id, g.seed, g.family, g.t] for g in groups]
    return {"rows": rows, "n": len(rows), "sha16": hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]}


def _candidate_manifest():
    cands = hold_candidates(N_ISO)
    rows = [[n, [round(float(x), 6) for x in d], m["magnitude"], m["dir"]] for n, d, m in cands]
    return {"n": len(rows), "rows": rows, "iso_seed": FROZEN_ISO_SEED,
            "sha16": hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]}


def _panel(pi0, per_family):
    return capture_state_panel(pi0, range(*DEV_BANK), per_family=per_family, label=False)


def freeze(pi0_path, config_path, per_family=10):
    pi0 = load_frozen_clip_actor(pi0_path, freeze=True)
    groups = _panel(pi0, per_family)
    cfg = {
        "sweep": "RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1", "date": "2026-07-23",
        "pi0_sha": hashlib.sha256(open(pi0_path, "rb").read()).hexdigest()[:8],
        "dev_bank": list(DEV_BANK), "per_family": per_family,
        "K_values": list(K_VALUES), "magnitudes": [0.0, 0.01, 0.025, 0.05, 0.10, 0.25],
        "directions": "signed actuator bases (8) + frozen deterministic isotropic (%d, seed %d)" % (N_ISO, FROZEN_ISO_SEED),
        "residual_bound": 0.25, "gamma": 0.99, "horizon": 360,
        "nonneg_thresholds": list(NONNEG_THRESHOLDS), "benefit_eps": BENEFIT_EPS,
        "metric_suite": METRIC_SUITE, "bootstrap": BOOTSTRAP, "leverage_gate": LEVERAGE_GATE,
        "state_manifest": _state_manifest(groups), "candidate_manifest": _candidate_manifest(),
        "contract": {"residual_execution": "for k in range(K): action=clip(pi0(o)+gate*delta,-4,4); gate-off ⇒ pi0 exact; "
                                           "after K attempted steps continue pi0 only",
                     "matched_states_across_K": True, "identical_candidates_across_K": True,
                     "each_branch_run_twice": True, "no_critic_no_actor_no_sac": True},
        "possible_conclusions": ["RESIDUAL_SIGNAL_INCREASES_WITH_HOLD_HORIZON", "RESIDUAL_SIGNAL_HAS_FINITE_TEMPORAL_WINDOW",
                                 "RESIDUAL_SIGNAL_FLAT_ACROSS_HOLD_HORIZON", "RESIDUAL_HOLD_SWEEP_UNDERPOWERED"],
    }
    json.dump(cfg, open(config_path, "w"), indent=1)
    print(f"FROZEN {config_path}: {cfg['state_manifest']['n']} states (per_family={per_family}), "
          f"{cfg['candidate_manifest']['n']} candidates, K={cfg['K_values']}")
    print(f"  state_manifest sha {cfg['state_manifest']['sha16']}  candidate_manifest sha {cfg['candidate_manifest']['sha16']}")


def run(pi0_path, config_path, out_path, scale="full"):
    cfg = json.load(open(config_path))
    per_family = cfg["per_family"] if scale != "smoke" else 3
    log = lambda *a: print(*a, flush=True)
    pi0 = load_frozen_clip_actor(pi0_path, freeze=True)
    log(f"[{time.strftime('%H:%M:%S')}] recapturing frozen panel (per_family={per_family})...")
    groups = _panel(pi0, per_family)
    sm = _state_manifest(groups)
    if scale != "smoke" and sm["sha16"] != cfg["state_manifest"]["sha16"]:
        raise AssertionError(f"state manifest drift: {sm['sha16']} != frozen {cfg['state_manifest']['sha16']}")
    cands = hold_candidates(N_ISO)
    log(f"  {len(groups)} states, {len(cands)} candidates, K={cfg['K_values']}; running sweep (each branch x2)...")
    rl = CoinRL4Dof(); results = {}
    t0 = time.time()
    for i, g in enumerate(groups):
        results[g.group_id] = sweep_group(rl, pi0, g, cands, tuple(cfg["K_values"]))
        if i % 5 == 0 or i == len(groups) - 1:
            log(f"    group {i+1}/{len(groups)} ({g.family}) {time.time()-t0:.0f}s")
    mets = metrics_by_K_family(groups, results, tuple(cfg["K_values"]))
    paired = paired_bootstrap_vs_k1(groups, results, tuple(cfg["K_values"]), n_boot=cfg["bootstrap"]["n_boot"])
    verdict = sweep_verdict(paired)
    # determinism certificate
    det = all(all(all(results[gid][K]["det_ok"]) for K in cfg["K_values"]) for gid in results)
    out = {"sweep": cfg["sweep"], "pi0_sha": cfg["pi0_sha"], "scale": scale,
           "state_manifest_sha": sm["sha16"], "candidate_manifest_sha": _candidate_manifest()["sha16"],
           "deterministic_x2": det, "metrics_by_K_family": {str(k): v for k, v in mets.items()},
           "paired_bootstrap_vs_K1": {str(k): v for k, v in paired.items()}, "verdict": verdict}
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    log("\n== leverage (per-group median|ΔG|), paired vs K=1 ==")
    for K in cfg["K_values"]:
        p = paired[str(K)] if str(K) in paired else paired[K]
        log(f"   K={K:2d}  median_leverage {p['median_leverage']:.3f}  Δvs K1 {p['mean_paired_diff']:+.3f}  ci95 {p['ci95']}")
    log(f"\ndeterministic_x2={det}  →  {verdict}\nwrote {out_path}\nHOLD_SWEEP_DONE")


def main():
    mode = sys.argv[1]
    if mode == "freeze":
        freeze(sys.argv[2], sys.argv[3])
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "full")
    else:
        raise SystemExit("mode must be freeze|run")


if __name__ == "__main__":
    main()
