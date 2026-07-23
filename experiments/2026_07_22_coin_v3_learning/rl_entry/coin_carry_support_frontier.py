"""CARRY_SUPPORT_FRONTIER_V1 — factorized support-frontier for the carry coverage limit (diagnostic/dev panel).

CARRY_HANDOFF_COVERAGE showed PARTIAL coverage (~1/3) at one support setting. This separates WHICH axis limits it — a
single combined 0.4/50/160 run would only show "something helped". On the SAME 30 held-out strict-0 carry states (manifest-
verified), same per-state CEM seeds, BUDGET-MATCHED random control:

  C0 mag .20 len 30 H 120 (ref)  C1 .20/30/160 (horizon)  C2 .20/50/160 (temporal)
  C3 .40/30/160 (amplitude)      C4 .40/50/160 (upper)    CB .20/30/120 2× iters (search budget)

Corrected per review: (1) material gate = NET new (n_new − n_lost) ≥ threshold; (2) horizon-fair safety = any_exit
(fraction of states with ≥1 full-containment exit), not raw counts; (3) CANDIDATE_CLASS_PLATEAU is only asserted if C0 AND
the widest config are optimizer-STABLE across 2–3 independent CEM seeds — otherwise NO_TESTED_SUPPORT_AXIS_MATERIALLY_HELPS
(not a proof the class is wrong); (4) panel manifest verified per state (strict==0 ∧ gate_mult==1.0 ∧ family ∈ carry);
(5) conclusion is PRIMARILY about contact_retention — transport (n≈5) is exploratory, braking rare. No critic training.
"""
import json
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_handoff import carry_cem, carry_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import (  # noqa: E402
    LateStart,
    build_boundary_panel,
    late_start_bank_manifest,
    reconstruct_handoff,
    verify_reconstruction,
)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
INIT_STD, ELITE_FRAC = 0.10, 0.25
CONFIGS = [("C0", 0.20, 30, 120, 24, 4), ("C1", 0.20, 30, 160, 24, 4), ("C2", 0.20, 50, 160, 24, 4),
           ("C3", 0.40, 30, 160, 24, 4), ("C4", 0.40, 50, 160, 24, 4), ("CB", 0.20, 30, 120, 24, 8)]
MATERIAL = 5                                             # pre-registered NET new K6-successes on the 30-panel (≥+0.167)
STABLE_JACCARD = 0.6                                     # optimizer-stability bar for a plateau claim


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _cem(rl0, gate0, pi0, base, adim, seed, mag, length, H, shots, iters):
    return carry_cem(rl0, gate0, pi0, base, adim, np.random.default_rng(seed), shots=shots, length=length, iters=iters,
                     init_std=INIT_STD, mag_max=mag, elite_frac=ELITE_FRAC, horizon=H)


def _jaccard(sets):
    js = [len(a & b) / max(len(a | b), 1) for a, b in combinations(sets, 2)]
    return round(float(np.mean(js)), 3) if js else 1.0


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    forbidden = set(ls.seed for ls in tb) | set(ls.seed for ls in db)
    configs = CONFIGS[:3] if smoke else CONFIGS
    want = 8 if smoke else 30

    log("[panel] rebuilding + VERIFYING the held-out strict-0 carry panel (manifest)...")
    panel, _c, _s = build_boundary_panel(pi0, range(6300, 8200), forbidden, want=want,
                                         families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templates, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)                                # actual identity verification, not just the return
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"], f"panel identity mismatch {ls.seed}: {v}"
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0 and rec.family in FAMS_CARRY, f"bad carry state {ls.seed}"
        templates.append((rl, gate)); fam.append(ls.family)
    n = len(templates)
    manifest = late_start_bank_manifest(panel)
    log(f"[panel] {n} carry states (manifest sha16 {manifest['sha16']}) | families { {k: v for k, v in sorted(Counter(fam).items())} }")

    # per config: per-state CEM (default seed) + budget-matched random
    cemo = {c[0]: [] for c in configs}; rnd = {c[0]: [] for c in configs}; Hof = {c[0]: c[3] for c in configs}
    for name, mag, length, H, shots, iters in configs:
        for i in range(n):
            rl0, gate0 = templates[i]
            cemo[name].append(_cem(rl0, gate0, pi0, base, adim, 200 + i, mag, length, H, shots, iters))
            rnd[name].append(carry_random(rl0, gate0, pi0, base, adim, np.random.default_rng(300 + i),
                                          shots=shots * iters, length=length, mag_max=mag, horizon=H))
        log(f"  [{name}] mag {mag} len {length} H {H} budget {shots}x{iters}: CEM K6 {round(float(np.mean([o['k6'] for o in cemo[name]])), 3)} "
            f"rand {round(float(np.mean([o['k6'] for o in rnd[name]])), 3)} handoff {round(float(np.mean([o['reached_handoff'] for o in cemo[name]])), 3)}")

    # multi-seed optimizer stability for C0 and the widest tested config (to separate plateau from CEM-instability)
    widest = configs[-2][0] if len(configs) >= 5 else configs[-1][0]     # C4 in the full set
    stab = {}
    for name in dict.fromkeys(["C0", widest]):
        mag, length, H, shots, iters = next((c[1], c[2], c[3], c[4], c[5]) for c in configs if c[0] == name)
        seed_sets = [{i for i in range(n) if cemo[name][i]["k6"] == 1}]   # default-seed solved set (already computed)
        for extra in (500, 700):
            seed_sets.append({i for i in range(n) if _cem(templates[i][0], templates[i][1], pi0, base, adim, extra + i, mag, length, H, shots, iters)["k6"] == 1})
        stab[name] = {"jaccard": _jaccard(seed_sets), "seed_solved_sizes": [len(s) for s in seed_sets]}
        log(f"  [stability {name}] cross-seed Jaccard {stab[name]['jaccard']} sizes {stab[name]['seed_solved_sizes']}")

    solved = {name: {i for i in range(n) if cemo[name][i]["k6"] == 1} for name in cemo}
    base_solved = solved[configs[0][0]]
    res = {}
    for name in cemo:
        s = solved[name]; exits = [o["contain_exit_ct"] for o in cemo[name]]
        res[name] = {"cem_K6": round(len(s) / n, 3), "rand_K6": round(float(np.mean([o["k6"] for o in rnd[name]])), 3),
                     "handoff": round(float(np.mean([o["reached_handoff"] for o in cemo[name]])), 3),
                     "any_exit": round(float(np.mean([e > 0 for e in exits])), 3),
                     "exit_per_100": round(float(np.mean([e / Hof[name] * 100 for e in exits])), 3),
                     "n_new": len(s - base_solved), "n_lost": len(base_solved - s), "net_new": len(s - base_solved) - len(base_solved - s),
                     "new_vs_C0": sorted(s - base_solved), "lost_vs_C0": sorted(base_solved - s), "solved_states": sorted(s)}
    core = set.intersection(*solved.values()) if solved else set(); union = set.union(*solved.values()) if solved else set()
    per_fam = {f: {"n": sum(x == f for x in fam), **{name: round(float(np.mean([cemo[name][i]["k6"] for i in range(n) if fam[i] == f])), 3) for name in cemo}}
               for f in sorted(set(fam))}

    def cov(name):
        return res[name]["cem_K6"]
    best = max((c[0] for c in configs), key=cov)
    material = (res[best]["net_new"] >= MATERIAL and res[best]["n_lost"] <= 1 and cov(best) > res[best]["rand_K6"]
                and res[best]["any_exit"] <= res["C0"]["any_exit"] + 0.1)
    names = [c[0] for c in configs]
    axis = []
    if "C1" in names and cov("C1") > cov("C0") + 0.1: axis.append("horizon")
    if "C2" in names and cov("C2") > cov("C1") + 0.1: axis.append("temporal_length")
    if "C3" in names and cov("C3") > cov("C1") + 0.1: axis.append("amplitude")
    if "C4" in names and cov("C4") > max(cov("C2"), cov("C3")) + 0.1: axis.append("temporal_amplitude_interaction")
    if "CB" in names and cov("CB") > cov("C0") + 0.1: axis.append("search_budget")
    optimizer_stable = all(stab[k]["jaccard"] >= STABLE_JACCARD for k in stab)

    if material:
        verdict = f"SUPPORT_LIMITED_{'_'.join(axis).upper() or 'BEST_' + best}"
        nxt = (f"widening ({'+'.join(axis) or best}) materially raises coverage to {cov(best)} (net +{res[best]['net_new']}, "
               f"CEM>{res[best]['rand_K6']} random, any_exit {res[best]['any_exit']}) → support-limited on {axis or [best]}; "
               "distil Phase 4 from the smallest widening's CEM first actions (BC→DAgger→RL), then RE-CONFIRM on a FRESH "
               "carry panel. Conclusion primarily for contact_retention; transport underpowered")
    elif not axis and optimizer_stable:
        verdict = "CANDIDATE_CLASS_PLATEAU_PI0_OFFSET_LIKELY_WRONG_FOR_CARRY_CONTACT_RETENTION"
        nxt = ("no tested support axis materially helps AND the CEM optimizer is stable across seeds (Jaccard "
               f"{[stab[k]['jaccard'] for k in stab]}) → for contact_retention the pi_0+offset candidate class is likely the "
               "limit; go carry-SPECIFIC (push/brake/release parametrization) for the DAgger expert. transport underpowered")
    else:
        verdict = "NO_TESTED_SUPPORT_AXIS_MATERIALLY_HELPS"
        nxt = ("no axis clears the NET-material bar; " + ("optimizer UNSTABLE across seeds (Jaccard "
               f"{[stab[k]['jaccard'] for k in stab]}) so the plateau is not yet a candidate-class proof — increase CEM budget"
               if not optimizer_stable else "solved set is not a clean plateau — inspect per-state / widen further") +
               ". Conclusion primarily for contact_retention; transport underpowered")

    out = {"contract": "CARRY_SUPPORT_FRONTIER_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "note": "DEV panel after config selection — re-confirm the winner on a FRESH held-out carry panel; conclusion primarily contact_retention",
           "panel_manifest_sha16": manifest["sha16"],
           "configs": [{"name": c[0], "mag_max": c[1], "length": c[2], "horizon": c[3], "shots": c[4], "iters": c[5]} for c in configs],
           "panel": {"n": n, "families": {k: v for k, v in sorted(Counter(fam).items())}, "all_held_out": True, "strict": 0},
           "results": res, "solved_core": sorted(core), "solved_union": sorted(union),
           "optimizer_stability": stab, "optimizer_stable": optimizer_stable, "per_family": per_fam,
           "material_gate": {"threshold_net_new": MATERIAL, "best": best, "met": material, "best_net_new": res[best]["net_new"]},
           "axis": axis, "verdict": verdict, "next_lever": nxt}
    json.dump(out, open(f"{D}/carry_support_frontier_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_SUPPORT_FRONTIER_V1 (factorized; net-material; horizon-fair exit; multi-seed stability) ==")
    for name in names:
        r = res[name]
        log(f"  {name}: CEM K6 {r['cem_K6']} (rand {r['rand_K6']}) handoff {r['handoff']} any_exit {r['any_exit']} | vs C0 net {r['net_new']:+d} (+{r['n_new']}/-{r['n_lost']})")
    log(f"  core {sorted(core)} | union {sorted(union)} | optimizer_stable {optimizer_stable} {[ (k, stab[k]['jaccard']) for k in stab]}")
    log("  per family: " + " ".join(f"{f}[n{per_fam[f]['n']}](" + "/".join(f"{per_fam[f][c[0]]}" for c in configs) + ")" for f in per_fam))
    log(f"→ {verdict}  |  next: {nxt}")
    log(f"wrote {D}/carry_support_frontier_v1.json\nCARRY_SUPPORT_FRONTIER_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
