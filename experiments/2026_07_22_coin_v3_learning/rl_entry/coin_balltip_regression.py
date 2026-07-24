"""BALLTIP_INTERARM_FILTERED_V1 §5+§6 — 4-way matched-panel regression + inter-arm-clearance exploit audit.

Compares the deployed carry-option controller (`θ_center from the proposal → fixed b=8 search_select → committed
push/brake/release → frozen settling pi_0`) across FOUR robot variants on a MATCHED fixed panel:

  1. canonical_clamp  — the REAL frozen E0 eval robot (CONCAVE_CLAMP fingertip)
  2. point_sphere     — galambos_planar_v3 sphere r0.014 (clamp→sphere control)
  3. balltip_nofilter — ball tip r0.020 + ORIGINAL collision (radius control)
  4. balltip_filtered — ball tip r0.020 + INTER-ARM collision filtering (the variant under test)

Identical across variants (the only difference is the robot): initial state (canonical E0 handoff, transplanted by
shared qpos layout), env seed, search seed, proposal θ_center (from the canonical obs), search budget b=8, settling
pi_0, certificate, eval horizon. Records per rollout: K6, max_dwell, handoff, containment-exit, effort, completion,
and the §6 min inter-arm clearance (+ overlap-steps: clearance<0 ⇒ arm-through-arm, impossible with the filtered
contacts absent). Emits JSON + a grouped-bar figure (§9 graphical). Does NOT modify the frozen baseline.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_carry_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import (  # noqa: E402
    LateStart, build_boundary_panel, reconstruct_handoff)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import (  # noqa: E402
    PANEL_VARIANTS, build_variant_rl, min_interarm_clearance, transplant_handoff)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
OUT = "reports/2026-07-24-balltip-regression"
FAMS = ("contact_retention", "transport", "braking")
PROP_CKPT = f"{D}/carry_proposal_refined.pt"        # the deployed proposal (frozen update-0 controller)
SEARCH_SEED = 9000
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}
VARIANTS = list(PANEL_VARIANTS)


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _fresh(variant, rl_c, ls):
    """A fresh env for one variant at the canonical handoff. canonical_clamp = the reconstruct itself (deep-copied);
    every other variant is built on its own robot and the canonical physical+certificate state is transplanted in."""
    if variant == "canonical_clamp":
        return copy.deepcopy(rl_c)
    return transplant_handoff(build_variant_rl(variant, seed=int(ls.seed)), rl_c)


def _clearance_trace(rl_v, gate, pi0, base, theta, horizon):
    """Re-run the COMMITTED option with a non-behavioral clearance hook: min inter-arm clearance over the rollout and
    the number of overlap steps (clearance < 0 ⇒ interpenetration only reachable when inter-arm contact is filtered)."""
    m, d = rl_v.inner.model, rl_v.inner.data
    acc = {"min_clr": float("inf"), "overlap_steps": 0, "n": 0}

    def hook(_phase, _strict):
        c = min_interarm_clearance(m, d)
        acc["min_clr"] = min(acc["min_clr"], c)
        acc["overlap_steps"] += int(c < 0.0)
        acc["n"] += 1

    o = structured_carry_rollout(rl_v, gate, pi0, base, np.asarray(theta, np.float32), horizon=horizon, frame_hook=hook)
    return o, acc


def _exploit_replay(rl_c, ls, gate, pi0, base, theta, horizon, o_filt, clr_filt):
    """§6 discriminating exploit test: replay the FILTERED variant's committed option θ on the NO-FILTER robot (collision
    RE-ENABLED, same geometry+θ). ``exploit`` iff the outcome CHANGES (the collision blocks what filtering allowed) — i.e.
    the filtered result relied on arm pass-through. If the outcome is identical, the negative clearance was a harmless
    query artefact (coin-hugging fingertips), not a load-bearing exploit."""
    o_re, clr_re = _clearance_trace(transplant_handoff(build_variant_rl("balltip_nofilter", seed=int(ls.seed)), rl_c),
                                    gate, pi0, base, theta, horizon)
    changed = (o_re["k6"] != o_filt["k6"]) or (o_re["reached_handoff"] != o_filt["reached_handoff"])
    return {"exploit": bool(changed), "nofilter_replay_k6": int(o_re["k6"]),
            "nofilter_replay_handoff": int(o_re["reached_handoff"]),
            "clr_gap": round(float(clr_re["min_clr"] - clr_filt["min_clr"]), 5)}  # >0 ⇒ filtered went deeper than collision allows


def run_panel(pi0, prop, base, panel, horizon=160, log=print):
    """The 4-way matched-panel sweep. Returns per-variant per-state records (each with the outcome + clearance audit +,
    for the filtered variant, the §6 collision-re-enabled exploit replay)."""
    recs = {v: [] for v in VARIANTS}
    for i, (rl_c, gate_c, ls) in enumerate(panel):
        c_center = prop.theta(rl_c.obs())                                # SAME proposal θ_center for all variants
        for v in VARIANTS:
            theta, o = search_select(_fresh(v, rl_c, ls), gate_c, c_center, pi0, base,
                                     np.random.default_rng(SEARCH_SEED + i), b=8, horizon=horizon)
            o2, clr = _clearance_trace(_fresh(v, rl_c, ls), gate_c, pi0, base, theta, horizon)
            assert o2["k6"] == o["k6"] and o2["reached_handoff"] == o["reached_handoff"], f"non-deterministic {v} @ {i}"
            rec = {"i": i, "seed": int(ls.seed), "family": ls.family, "k6": int(o["k6"]),
                   "max_dwell": int(o["max_dwell"]), "reached_handoff": int(o["reached_handoff"]),
                   "contain_exit_ct": int(o["contain_exit_ct"]), "touched": int(o["touched"]),
                   "effort": float(o["effort"]), "completion": int(o["completion"]),
                   "min_clearance": round(float(clr["min_clr"]), 5), "overlap_steps": int(clr["overlap_steps"])}
            if v == "balltip_filtered":                                  # §6 exploit test only meaningful for the filtered variant
                rec.update(_exploit_replay(rl_c, ls, gate_c, pi0, base, theta, horizon, o, clr))
            recs[v].append(rec)
        if (i + 1) % 4 == 0 or i == 0:
            k6s = {v: sum(r["k6"] for r in recs[v]) for v in VARIANTS}
            log(f"  [{i + 1}/{len(panel)}] K6 so far " + " ".join(f"{v.split('_')[0][:4]}:{k6s[v]}" for v in VARIANTS), flush=True)
    return recs


def summarize(recs, n):
    """Per-variant aggregate: K6/handoff/exit rates, mean dwell/effort/completion, clearance stats, overlap-exploit
    count, and the solved (K6) state set (for cross-variant solved-set overlap)."""
    out = {}
    for v, rs in recs.items():
        clrs = [r["min_clearance"] for r in rs]
        out[v] = {"n": n, "k6": sum(r["k6"] for r in rs), "k6_rate": round(sum(r["k6"] for r in rs) / n, 4),
                  "handoff": sum(r["reached_handoff"] for r in rs), "exit": sum(r["contain_exit_ct"] for r in rs),
                  "mean_dwell": round(float(np.mean([r["max_dwell"] for r in rs])), 3),
                  "mean_effort": round(float(np.mean([r["effort"] for r in rs])), 3),
                  "mean_completion": round(float(np.mean([r["completion"] for r in rs])), 2),
                  "min_clearance": round(float(np.min(clrs)), 5), "mean_min_clearance": round(float(np.mean(clrs)), 5),
                  "overlap_query_states": sum(1 for r in rs if r["overlap_steps"] > 0),   # raw mj_geomDistance<0 (artefact-prone)
                  "total_overlap_steps": sum(r["overlap_steps"] for r in rs),
                  "exploit_states": sum(1 for r in rs if r.get("exploit")),               # §6: filtered outcome relied on pass-through
                  "solved_states": sorted(r["i"] for r in rs if r["k6"])}
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vs = VARIANTS
    lbl = [v.replace("_", "\n") for v in vs]
    colors = ["#555", "#3a7", "#38c", "#c63"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(vs))
    ax[0].bar(x, [summary[v]["k6_rate"] for v in vs], color=colors)
    ax[0].set_title("K6 delivery rate")
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("rate")
    ax[1].bar(x, [summary[v]["mean_min_clearance"] for v in vs], color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("mean min inter-arm clearance (m)\n(query artefact — see §6 exploit test)")
    ax[2].bar(x, [summary[v]["exploit_states"] for v in vs], color=colors)
    ax[2].set_title("§6 exploit states\n(filtered pass-through blocked by collision)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(lbl, fontsize=8)
    fig.suptitle("BALLTIP_INTERARM_FILTERED_V1 — 4-way matched-panel regression (frozen controller, b=8)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    prop = load_proposal(PROP_CKPT)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    want = 4 if smoke else 24
    raw, _c, _s = build_boundary_panel(pi0, range(14000, 15200), forbidden, want=want, families=FAMS,
                                       strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    panel = [(*reconstruct_handoff(pi0, ls, horizon=360)[:2], ls) for ls in raw]
    print(f"[balltip §5] {len(panel)} held-out matched states | variants {VARIANTS} | search seed {SEARCH_SEED}", flush=True)
    recs = run_panel(pi0, prop, base, panel)
    summary = summarize(recs, len(panel))
    manifest = {"contract": "BALLTIP_INTERARM_FILTERED_V1", "section": "§5+§6", "date": "2026-07-24", "smoke": smoke,
                "baseline": BASELINE, "n_states": len(panel), "search_seed": SEARCH_SEED,
                "proposal_ckpt": PROP_CKPT.split("/")[-1], "variants": PANEL_VARIANTS, "summary": summary, "records": recs}
    json.dump(manifest, open(f"{OUT}/regression.json", "w"), indent=1, default=float)
    plot(summary, f"{OUT}/regression.png")
    print("\n== BALLTIP §5 4-way matched-panel regression ==", flush=True)
    hdr = f"{'variant':18} {'K6':>6} {'handoff':>8} {'exit':>5} {'dwell':>6} {'min_clr':>8} {'exploit':>8}"
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        s = summary[v]
        print(f"{v:18} {s['k6']:>3}/{s['n']:<2} {s['handoff']:>8} {s['exit']:>5} {s['mean_dwell']:>6} "
              f"{s['min_clearance']:>8} {s['exploit_states']:>8}", flush=True)
    print(f"\n  §6: raw mj_geomDistance<0 is artefact-prone (coin-hugging tips); 'exploit' = filtered outcome that "
          f"a collision-on replay BLOCKS.\n  artifacts: {OUT}/regression.json + regression.png\nBALLTIP_REGRESSION_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
