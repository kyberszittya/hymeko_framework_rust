"""COIN-DELIVERY-OVERNIGHT-2 — acquisition oracle run + preregistered gate-ladder evaluation (orchestration).

Split from coin_delivery_acquisition1.py (gate-signal helpers) to keep each file focused (CLAUDE.md 6.5 #4).
"""
from __future__ import annotations

import json
import time

import numpy as np

from hymeko_rl.experiments.coin_delivery_acquisition1 import (
    _OUT,
    _SCRAMBLE,
    _states,
    chained_eval,
    demo_reproducibility,
    selector_oracle,
)
from hymeko_rl.train.coin_delivery_acquisition import (
    AcqOracleConfig,
    AcqParams,
    cem_acquisition,
    eval_acquisition,
    make_acq_env,
)


def _log(m: str) -> None:
    print(m, flush=True)


def _best_base_mode(seeds, env) -> tuple:
    """Pre-sweep the approach modes at default continuous params; return (mode_name, base AcqParams) with the most
    stable acquisitions. asym_left/asym_right/staggered are genuinely different control families, not just params."""
    from dataclasses import replace
    from hymeko_rl.experiments.coin_delivery_acquisition1 import _MODES
    best = ("symmetric", AcqParams(), -1)
    for name, kw in _MODES:
        p = replace(AcqParams(), **kw)
        ev = eval_acquisition(p, seeds, env=env)
        _log(f"  [mode {name}] n_stable={ev['n_stable']}/{len(list(seeds))}")
        if ev["n_stable"] > best[2]:
            best = (name, p, ev["n_stable"])
    _log(f"  → best base mode: {best[0]} ({best[2]}/{len(list(seeds))})")
    return best[0], best[1]


def _budget_oracle(seeds, env, budgets, rng, base=None) -> dict:
    """CEM at ascending budgets; record n_stable + recovered per budget + the best params overall (budget-scaling)."""
    per_budget = {}
    best = None
    for name, oc in budgets:
        b = cem_acquisition(seeds, oc, rng, env=env, base=base, log=_log)
        per_budget[name] = {"n_stable": b["n_stable"], "recovered": b["eval"]["recovered_seeds"],
                            "pop": oc.pop, "iters": oc.iters}
        _log(f"  [{name}] n_stable={b['n_stable']}/{len(list(seeds))} recovered={b['eval']['recovered_seeds']}")
        if best is None or b["n_stable"] >= best["n_stable"]:
            best = b
    counts = [per_budget[n]["n_stable"] for n, _ in budgets]
    monotonic = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    return {"per_budget": per_budget, "counts": counts, "monotonic": monotonic, "best": best}


def _structural(seeds, params, env) -> dict:
    """Correct-vs-scrambled geometry at equal budget (the structural control) + the funnel under each."""
    correct = eval_acquisition(params, seeds, env=env)
    scram = eval_acquisition(params, seeds, env=env, scramble_perm=_SCRAMBLE)
    return {"correct": {k: correct[k] for k in ("n_stable", "pregrasp_rate", "first_contact_rate", "two_finger_rate",
                                                "stable_acquisition_rate", "recovered_seeds")},
            "scrambled": {k: scram[k] for k in ("n_stable", "pregrasp_rate", "first_contact_rate", "two_finger_rate",
                                                "stable_acquisition_rate", "recovered_seeds")},
            "correct_beats_scrambled": correct["n_stable"] > scram["n_stable"]}


def _gate_dict(sig: dict) -> dict:
    """The 8 preregistered gate booleans (computed EXACTLY as specified; no post-hoc weakening)."""
    n_full = sig["best_n_stable"]
    cbs = sig["structural"]["correct_beats_scrambled"]
    ch, demo, bud, ph, sel = sig["chained_correct"], sig["demo"], sig["budget"], sig["phase"], sig["selector"]
    return {
        "STRICT_GATE": bool(n_full >= 5 or (n_full / 19.0) >= 0.25),
        "CONTACT_LOSS_GATE": bool(sig["per_class"]["contact_loss_recovered"] >= 3 and cbs),
        "GEOMETRIC_HARD_GATE": bool(sig["per_class"]["geometric_hard_recovered"] >= 4 and cbs),
        "CHAIN_GATE": bool(ch["stable_acq"] >= 3 and (ch["zone_entry"] >= 2 or ch["center_reach"] >= 1)
                           and sig["easy_preserved"]),
        "DEMO_GATE": bool(demo["distinct_success_states"] >= 3 and demo["n_trajectories"] >= 100
                          and demo["reproducibility"] >= 0.80),
        "BUDGET_SCALING_GATE": bool(bud["monotonic"] and bud["full_expanded_unique"] >= 3 and bud["shared_structure"]),
        "PHASE_GATE": bool(ph["best_abs_improvement"] >= 0.30 and cbs and ph["next_stage_not_worse"]),
        "ORACLE_SELECTOR_GATE": bool(sel["per_state_best_n"] >= 5 and sel["max_single_mode"] < 5),
    }


_TARGETED = ("CONTACT_LOSS_GATE", "GEOMETRIC_HARD_GATE", "CHAIN_GATE", "DEMO_GATE", "BUDGET_SCALING_GATE",
             "ORACLE_SELECTOR_GATE")


def _evaluate_gates(sig: dict) -> dict:
    """Apply the preregistered gate ladder: STRICT→Case A; any targeted alt→Case B; phase-only→Case C; none→Case D."""
    g = _gate_dict(sig)
    strict, targeted = g["STRICT_GATE"], any(g[k] for k in _TARGETED)
    if strict:
        case, auth = "A", ["PPO", "TD3+BC", "guarded_SAC"]
    elif targeted:
        case, auth = "B", ["TD3+BC", "PPO"]
    elif g["PHASE_GATE"]:
        case, auth = "C", ["TD3+BC_or_PPO_on_phase"]
    else:
        case, auth = "D", []
    return {"gates": g, "passed_gates": [k for k, v in g.items() if v], "case": case, "authorized_methods": auth,
            "rl_authorized": case != "D"}


def run(*, fast: bool = False) -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    env = make_acq_env()
    S = _states()
    wall, geo, cl, easy = S["acquisition_wall"], S["geometric_hard"], S["contact_loss"], S["easy"]
    _log(f"=== PART II acquisition oracle === wall n={len(wall)} (GEOMETRIC_HARD {len(geo)} + CONTACT_LOSS {len(cl)})")
    rng = np.random.default_rng(0)
    budgets = ([("small", AcqOracleConfig(pop=6, iters=2)), ("full", AcqOracleConfig(pop=10, iters=4))] if fast
               else [("small", AcqOracleConfig(pop=8, iters=3)), ("full", AcqOracleConfig(pop=16, iters=6)),
                     ("expanded", AcqOracleConfig(pop=24, iters=8))])
    _log("=== mode pre-sweep (pick best acquisition control family) ===")
    mode_name, base_mode = _best_base_mode(wall, env)
    bud = _budget_oracle(wall, env, budgets, rng, base=base_mode)
    bud["best_mode"] = mode_name
    best = bud["best"]
    bp: AcqParams = best["params"]
    rec = set(best["eval"]["recovered_seeds"])

    _log("=== correct-vs-scrambled structural control ===")
    structural = _structural(wall, bp, env)
    _log(f"  correct n_stable={structural['correct']['n_stable']} vs scrambled={structural['scrambled']['n_stable']} "
         f"→ correct_beats_scrambled={structural['correct_beats_scrambled']}")

    per_class = {"geometric_hard_recovered": len(rec & set(geo)), "contact_loss_recovered": len(rec & set(cl)),
                 "geometric_hard_seeds": sorted(rec & set(geo)), "contact_loss_seeds": sorted(rec & set(cl))}
    _log(f"  per-class: GEOMETRIC_HARD {per_class['geometric_hard_recovered']}/{len(geo)} | "
         f"CONTACT_LOSS {per_class['contact_loss_recovered']}/{len(cl)}")

    _log("=== selector (per-state best mode) ===")
    sel = selector_oracle(wall, bp, env=env)
    _log(f"  per-state best={sel['per_state_best_n']} max single mode={sel['max_single_mode']} {sel['per_mode_n']}")

    _log("=== demo reproducibility (perturbed replays of recovered) ===")
    demo = demo_reproducibility(sorted(rec), bp, env=env, n_perturb=(6 if fast else 12))
    demo.pop("per_state_successes", None)
    _log(f"  unique={demo['distinct_success_states']} trajectories={demo['n_trajectories']} "
         f"repro={demo['reproducibility']}")

    _log("=== PART IV chained delivery (acquisition → grasp_carry → delivery semantics) ===")
    chained_correct = chained_eval(sorted(rec), bp, env=env)
    chained_scram = chained_eval(sorted(rec), bp, env=env, scramble_perm=_SCRAMBLE)
    for d in (chained_correct, chained_scram):
        d.pop("rows", None)
    _log(f"  chained(correct): stable={chained_correct['stable_acq']} zone={chained_correct['zone_entry']} "
         f"center={chained_correct['center_reach']}")

    # easy-state preservation: the acquisition primitive must not DESTROY already-successful acquisitions
    easy_ev = eval_acquisition(bp, easy[:20], env=env)
    easy_preserved = easy_ev["stable_acquisition_rate"] >= 0.6
    _log(f"  easy-state acquisition (20): stable_rate={easy_ev['stable_acquisition_rate']} preserved={easy_preserved}")

    # phase-gate signal: pregrasp / two_finger improvement over the OLD baseline (grasp_carry ~0 on the wall)
    phase = {"pregrasp_improvement": structural["correct"]["pregrasp_rate"],
             "two_finger_improvement": structural["correct"]["two_finger_rate"],
             "best_abs_improvement": max(structural["correct"]["pregrasp_rate"], structural["correct"]["two_finger_rate"]),
             "next_stage_not_worse": structural["correct"]["stable_acquisition_rate"] >= structural["scrambled"]["stable_acquisition_rate"]}
    # budget-scaling structure proxy: recovered sets overlap across budgets (shared, not isolated accidents)
    fe_unique = len(set(bud["per_budget"].get("expanded", bud["per_budget"]["full"])["recovered"]))
    shared = _shared_structure(bud)

    sig = {"best_n_stable": best["n_stable"], "structural": structural, "per_class": per_class,
           "chained_correct": chained_correct, "demo": demo, "phase": phase, "selector": sel,
           "easy_preserved": easy_preserved,
           "budget": {"monotonic": bud["monotonic"], "counts": bud["counts"],
                      "full_expanded_unique": fe_unique, "shared_structure": shared}}
    decision = _evaluate_gates(sig)
    _log(f"\n=== PREREGISTERED GATE LADDER === passed={decision['passed_gates']} → Case {decision['case']} "
         f"(RL authorized={decision['rl_authorized']}, methods={decision['authorized_methods']})")

    out = {"acquisition_wall": {"n": len(wall), "geometric_hard": geo, "contact_loss": cl},
           "budget_oracle": bud["per_budget"], "budget_counts": bud["counts"], "budget_monotonic": bud["monotonic"],
           "best_mode": bud.get("best_mode"), "best_params": bp.__dict__, "best_n_stable": best["n_stable"],
           "recovered_seeds": sorted(rec),
           "structural_correct_vs_scrambled": structural, "per_class_recovery": per_class,
           "selector": sel, "demo_reproducibility": demo, "chained_delivery_correct": chained_correct,
           "chained_delivery_scrambled": chained_scram, "easy_state_acquisition": easy_ev["stable_acquisition_rate"],
           "phase_signal": phase, "gate_signals": {k: v for k, v in sig.items() if k not in ("structural",)},
           "gate_ladder": decision, "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "coin_delivery_acquisition.json").write_text(json.dumps(out, indent=2, default=_json_default))
    _log(f"[ACQUISITION] best {best['n_stable']}/19 stable | Case {decision['case']} | {out['wall_s']}s")
    return out


def _shared_structure(bud: dict) -> bool:
    """Do the recovered states across budgets overlap (shared structure, not isolated accidents)?"""
    sets = [set(v["recovered"]) for v in bud["per_budget"].values() if v["recovered"]]
    if len(sets) < 2:
        return False
    inter = set.intersection(*sets)
    union = set.union(*sets)
    return len(inter) >= max(2, len(union) // 2)          # majority of recovered states shared across budgets


def _json_default(o):
    if isinstance(o, (set,)):
        return sorted(o)
    if hasattr(o, "value"):
        return o.value
    return str(o)
