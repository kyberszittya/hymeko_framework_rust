"""Run the COIN discounted-alignment gate (Option B §9) and write the result + raw evidence.

Load-bearing gate (directive §6): discounted(strict K=6 delivery) > discounted(every non-success class) under every
configured γ, and no repeatable non-success cycle's infinite upper bound dominates strict delivery (§7). The internal
ordering among failures is diagnostic, not a gate.

Usage: ``python -m hymeko_rl.coin_delivery.run_discounted_alignment``  →  writes
``experiments/2026_07_23_coin_hymeko_recovery/logs/discounted_alignment_v3.json`` and prints the verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

from hymeko_rl.coin_delivery.discounted_alignment import (
    FAILURE_CONTROLLERS,
    FARMING_LOOPS,
    bundle_hashes,
    cycle_upper_bound,
    discounted_return,
    resolve_gammas,
    strict_delivery_reference,
    _rollout,
)
from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig, make_delivery_rl_env

_OUT = Path("experiments/2026_07_23_coin_hymeko_recovery/logs/discounted_alignment_v3.json")
_SEEDS = (0, 1, 2)


def run() -> dict:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    gammas = resolve_gammas()
    manifest = bundle_hashes(cfg)

    # strict-delivery references: (A) deterministic demonstration, (B) frozen chain ecological cross-check.
    strict_rows = [strict_delivery_reference(env, seed=s) for s in _SEEDS]
    failures = {name: [_rollout(env, ctrl, seed=s, horizon=cfg.horizon) for s in _SEEDS]
                for name, ctrl in FAILURE_CONTROLLERS.items()}

    def _median_return(rows, gamma):
        vals = sorted(discounted_return(r["rewards"], gamma) for r in rows)
        return vals[len(vals) // 2]

    results: dict = {"gate": "COIN_DISCOUNTED_REWARD_ALIGNMENT", "gammas": gammas, "manifest": manifest,
                     "seeds": list(_SEEDS), "per_gamma": {}}
    verdict = "COIN_DISCOUNTED_REWARD_ALIGNMENT_PASS"
    for label, gamma in gammas.items():
        strict_g = _median_return(strict_rows, gamma)
        fails_g = {name: _median_return(rows, gamma) for name, rows in failures.items()}
        dominated = [n for n, g in fails_g.items() if g >= strict_g]
        # no-farming: infinite repeated-cycle upper bound for each repeatable loop must not reach strict.
        farm = {}
        for name, L in FARMING_LOOPS.items():
            rows = failures[name]
            longest = max(rows, key=lambda r: len(r["rewards"]))["rewards"]
            cyc = longest[:L] if len(longest) >= L else longest
            ub = cycle_upper_bound(cyc, gamma)
            farm[name] = {"cycle_len": L, "cycle_upper_bound": round(ub, 3), "dominates_strict": bool(ub >= strict_g)}
        farmers = [n for n, f in farm.items() if f["dominates_strict"]]
        ok = (not dominated) and (not farmers)
        if not ok:
            verdict = "COIN_DISCOUNTED_ALIGNMENT_BLOCKED"
        results["per_gamma"][label] = {
            "gamma": gamma, "strict_delivery_return": round(strict_g, 3),
            "failure_returns": {n: round(v, 3) for n, v in fails_g.items()},
            "failures_dominating_strict": dominated, "no_farming": farm, "farmers_dominating_strict": farmers,
            "strict_dominates_all": ok,
        }
    results["verdict"] = verdict
    results["strict_reference_detail"] = [
        {"seed": s, "steps": r["steps"], "terminated": r["terminated"], "final_dwell": r["final_dwell"]}
        for s, r in zip(_SEEDS, strict_rows)]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps({"verdict": res["verdict"], "per_gamma": {
        k: {"strict": v["strict_delivery_return"], "worst_failure": max(v["failure_returns"].values()),
            "strict_dominates_all": v["strict_dominates_all"]} for k, v in res["per_gamma"].items()}}, indent=1))
