"""O3 — triangular-prism bounded PHYSICAL experiment: does the vertex/edge-contact structure create separate strategy
basins where the multimodal policy search helps (as it did on 6D-1's contact-FREE geometrically-separated basins), or
does contact candidate-localization stay the wall (the O2 boxes finding)?

The physical foundation is validated (o3_triangle_physical_prep, all_ok). This is the gated experiment. Arms (the
user's list), all on the FRESH-reconstruct triangle distribution, ball-tip embodiment (BALLTIP_COIN_BASELINE_V1),
orientation-stratified by leading feature (vertex-leading vs edge-leading toward the zone):
  A pi0-only                — the clamp-trained base, no carry option
  B zero-shot proposal      — the frozen CYLINDER proposal + b8 (cross-shape transfer)
  C single-head refit       — an O3 triangle proposal (fit on fresh triangle labels) + b8
  D K-mode proposal (b0)    — the O3 templates as K modes, direct (no within-mode search)
  E K-mode + equal-budget   — MultimodalBudgetSearch over K modes at the SAME total budget as C
  F explicit controller     — geometry-aware push θ
  G structured expert       — 192-shot search (ceiling)
Plus the equal-budget K sub-ablation K1/K2/K4/K6 at a fixed total budget (the 6D-1 question in contact).

Grade: strict K6 AND the FULL-FOOTPRINT certificate (every triangle vertex in the zone — the stricter O3 predicate),
plus contact retention, disk-rz rotation range, target-entry, and the deploy-expert gap, split by vertex/edge leading.
"""
import copy
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
import torch  # noqa: E402

from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, load_proposal, save_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import structured_random_best_with_support  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel, reconstruct_handoff  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.coin_delivery.triangle_footprint import full_footprint_certified, leading_feature, triangle_circumradius  # noqa: E402
from hymeko_rl.option_rl import MultimodalBudgetSearch  # noqa: E402

from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402
from coin_kmode_budget_ablation import CoinCarryScorer, CoinJitterGenerator, TemplateKModeProposal  # noqa: E402
from coin_object_o2 import (  # noqa: E402
    EVAL_H, EXPERT_SHOTS, _ball_tf, _geometry_theta, _rz_adr, _set_orientation, _stage_metrics)

OUT = "reports/2026-07-24-o3-triangle-experiment"
OLD_PROP = f"{D}/carry_proposal_balltip_fresh_v1.pt"
O3_PROP = f"{D}/carry_proposal_o3_triangle_v1.pt"
FAMS = ("contact_retention", "transport", "braking")
_R = 0.020
_CIRC = triangle_circumradius(_R)
ZONE_XY, ZONE_HALF = np.array([0.0, 0.16]), 0.055        # EnvSpec target_zone (zone_x, zone_y, zone_half)
BUDGET, K_VALUES = 12, (1, 2, 4, 6)
RK = {"geom": "POINT", "arm_mjcf_transform": _ball_tf, "coin_shape": "triangle", "disk_radius_override": _R}


def _disk_xy(rl):
    return np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2]


def _full_footprint(rl):
    return int(full_footprint_certified(_disk_xy(rl), float(rl.inner.data.qpos[_rz_adr(rl)]), _CIRC, ZONE_XY, ZONE_HALF))


def fresh_o3_bank(pi0, cand_ls, want, log):
    """Fresh triangle reconstructions, orientation-stratified by LEADING FEATURE (vertex vs edge toward the +y zone).
    Cycle deterministic orientations spanning one 3-fold period; keep strict==0 carry starts."""
    orients = [(leading_feature(rz), float(rz)) for rz in np.linspace(0.0, 2 * math.pi / 3, 6, endpoint=False)]
    panel, oi = [], 0
    for ls in cand_ls:
        try:
            rl, gate, _h, _r = reconstruct_handoff(pi0, ls, **RK)
        except ValueError:
            continue
        if int(rl._strict) != 0:
            continue
        lead, theta = orients[oi % len(orients)]
        oi += 1
        _set_orientation(rl, theta)
        if int(rl._strict) != 0:
            continue
        panel.append({"rl": rl, "gate": gate, "leading": lead, "orient_rad": round(theta, 4), "seed": int(ls.seed)})
        if len(panel) >= want:
            break
    log(f"    triangle fresh handoffs {len(panel)} (circumradius {_CIRC:.4f})")
    return panel


def _kmode_arm(rl, gate, prop, pi0, base, k, budget, rng):
    """K-mode proposal over the O3 templates + MultimodalBudgetSearch at ``budget`` (k=... modes). Returns outcome dict."""
    scorer = CoinCarryScorer(rl, gate, pi0, base)
    kprop = TemplateKModeProposal(prop, rl.obs(), k)
    prov = MultimodalBudgetSearch(CoinJitterGenerator(), scorer, budget=budget).select(kprop, rl.obs(), rng)
    return prov.outcome, int(prov.selected_mode)


def _metrics_with_footprint(rl, gate, pi0, base, theta):
    o = _stage_metrics(rl, gate, pi0, base, theta)          # runs the θ, tracks contact/dtz_min/rz range
    o["full_footprint"] = _full_footprint(rl)               # final disk pose after the committed rollout
    return o


def eval_state(item, old_prop, o3_prop, pi0, base, adim, i):
    rl, gate = item["rl"], item["gate"]
    obs = rl.obs()
    out = {}
    o = sequence_then_pi0(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H)
    out["pi0"] = {"k6": int(o["k6"]), "reached_handoff": int(o["reached_handoff"]), "full_footprint": _full_footprint(rl)}
    for name, prop in (("zeroshot", old_prop), ("single_head", o3_prop)):
        th, _o = search_select(copy.deepcopy(rl), copy.deepcopy(gate), prop.theta(obs), pi0, base,
                               np.random.default_rng(9000 + i), b=8, horizon=EVAL_H)
        out[name] = _summ2(_metrics_with_footprint(copy.deepcopy(rl), gate, pi0, base, th))
    # D K-mode direct (b0) and E K-mode + equal-budget search (budget = the single-head's b8)
    okd, _md = _kmode_arm(copy.deepcopy(rl), gate, o3_prop, pi0, base, K_VALUES[-1], 0, np.random.default_rng(9100 + i))
    out["kmode_b0"] = {"k6": int(okd.get("k6", 0)), "full_footprint": _full_footprint(rl)}
    oke, _me = _kmode_arm(copy.deepcopy(rl), gate, o3_prop, pi0, base, K_VALUES[-1], 8, np.random.default_rng(9100 + i))
    out["kmode_search"] = {"k6": int(oke.get("k6", 0)), "full_footprint": _full_footprint(rl)}
    out["explicit"] = _summ2(_metrics_with_footprint(copy.deepcopy(rl), gate, pi0, base, _geometry_theta(copy.deepcopy(rl))))
    th_e, _oe, _s = structured_random_best_with_support(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base,
                                                        np.random.default_rng(9000 + i), shots=EXPERT_SHOTS, horizon=EVAL_H)
    out["expert"] = _summ2(_metrics_with_footprint(copy.deepcopy(rl), gate, pi0, base, th_e))
    # equal-budget K sub-ablation K1/K2/K4/K6 at BUDGET
    out["kbudget"] = {}
    for k in K_VALUES:
        ok, _mk = _kmode_arm(copy.deepcopy(rl), gate, o3_prop, pi0, base, k, BUDGET, np.random.default_rng(9200 + i))
        out["kbudget"][f"K{k}"] = int(ok.get("k6", 0))
    return out


def _summ2(o):
    keys = ("k6", "reached_handoff", "touched", "max_dwell", "contact_frac", "dtz_min", "rot_range", "full_footprint", "target_entry")
    return {k: (int(o[k]) if k in ("k6", "reached_handoff", "touched", "full_footprint", "target_entry") else o.get(k)) for k in keys if k in o}


def main(smoke=False):
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    old_prop = load_proposal(OLD_PROP)
    want = 6 if smoke else 24

    # ---- fit the O3 triangle single-head proposal on fresh triangle teacher labels ----
    log("[O3] fitting the fresh triangle proposal (frozen-reward teacher on fresh triangle handoffs)...")
    tr_ls, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(8 if smoke else 80),
                                         families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    obs_b, th_b, _p = generate_bank(pi0, base, tr_ls, shots=(24 if smoke else 128), reconstruct_kwargs=RK, log=log)
    kk = min(6, max(2, len(obs_b) // 4))
    o3_prop, fit = fit_proposal(np.asarray(obs_b, np.float32), np.asarray(th_b, np.float32), kk,
                                clf_epochs=80 if smoke else 300, res_epochs=80 if smoke else 300)
    save_proposal(o3_prop, O3_PROP)
    log(f"[O3] triangle proposal K={kk} res_mse {fit['res_mse']:.3f} → {O3_PROP}")

    # ---- fresh eval panel (orientation-stratified), disjoint seeds ----
    ev_ls, _c, _s = build_boundary_panel(pi0, range(14000, 15600), forbidden, want=(12 if smoke else 60),
                                         families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    bank = fresh_o3_bank(pi0, ev_ls, want, log)
    recs = [{"leading": it["leading"], "orient_rad": it["orient_rad"], **eval_state(it, old_prop, o3_prop, pi0, base, adim, i)}
            for i, it in enumerate(bank)]
    agg = _aggregate(recs)
    verdict = _verdict(agg)
    manifest = {"contract": "O3_TRIANGLE_EXPERIMENT", "date": "2026-07-24", "smoke": smoke, "distribution": "fresh_reconstruct_triangle",
                "embodiment": "collision-on ball-tip (frozen)", "grade": "strict K6 AND full-footprint certificate",
                "circumradius": round(_CIRC, 5), "budget": BUDGET, "K_values": list(K_VALUES), "n": len(recs),
                "proposal_fit": fit, "aggregate": agg, "records": recs, "verdict": verdict}
    json.dump(manifest, open(f"{OUT}/o3_triangle.json", "w"), indent=1, default=float)
    log(f"\n== O3 triangle experiment ==  → {verdict}\n  artifact: {OUT}/o3_triangle.json\nO3_TRIANGLE_DONE")
    return manifest


def _rate(recs, arm, key="k6"):
    vals = [r[arm].get(key) for r in recs if arm in r and r[arm].get(key) is not None]
    return round(float(np.mean(vals)), 3) if vals else 0.0


def _aggregate(recs):
    arms = ("pi0", "zeroshot", "single_head", "kmode_b0", "kmode_search", "explicit", "expert")
    agg = {"n": len(recs)}
    for a in arms:
        agg[a] = {"k6": _rate(recs, a, "k6"), "full_footprint": _rate(recs, a, "full_footprint")}
    for extra in ("contact_frac", "rot_range", "dtz_min"):
        agg[extra] = {a: _rate(recs, a, extra) for a in ("single_head", "explicit", "expert")}
    # equal-budget K sub-ablation
    agg["kbudget"] = {f"K{k}": round(float(np.mean([r["kbudget"][f"K{k}"] for r in recs if "kbudget" in r])), 3)
                      for k in K_VALUES} if recs else {}
    # split by leading feature (vertex vs edge)
    for lead in ("vertex", "edge"):
        sub = [r for r in recs if r.get("leading") == lead]
        agg[f"{lead}_k6"] = {a: _rate(sub, a, "k6") for a in ("single_head", "kmode_search", "expert")}
    return agg


def _verdict(agg):
    kb = agg.get("kbudget", {})
    if not kb:
        return "NO_DATA"
    k1, kmax = kb.get("K1", 0), kb.get(f"K{K_VALUES[-1]}", 0)
    deploy, expert = agg["single_head"]["k6"], agg["expert"]["k6"]
    if kmax > k1 + 0.1:
        return f"CONTACT_MULTIMODAL_HELPS (Kmax {kmax} > K1 {k1}); vertex/edge basins separable"
    if expert > deploy + 0.15:
        return f"CONTACT_CANDIDATE_LOCALIZATION_WALL (expert {expert} >> deploy {deploy}, but K-mode flat: K1 {k1} Kmax {kmax})"
    return f"NO_MULTIMODAL_ADVANTAGE_IN_CONTACT (K1 {k1} Kmax {kmax}, deploy {deploy} expert {expert})"


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
