"""CARRY_OPTION_ACTOR_V1 — corrective diagnostic: WHY (if) deterministic MSE-BC fails to distil the option bank.

The θ-multimodality inspection showed several carry states admit multiple valid θ*, so a single-mode MSE regressor
mode-averages into an invalid θ. This isolates the two competing explanations on the SAME disjoint held-out panel, all
executed through the SAME committed-macro controller so only the θ-selection differs:

  NEAREST_RETRIEVAL   nearest bank obs → its stored θ (non-parametric; keeps each state's own mode)
  TEMPLATE_CLASSIFY   k-means the bank θ into K canonical templates; MLP classifies obs→template; execute the medoid θ
                      (parametric, multimodality-respecting: classification, not regression)
  RANDOM_BANK_THETA   a fixed random bank θ per state (control: is any confident θ as good as the selected one?)
  GLOBAL_ROBUST_THETA the single most jitter-robust bank θ applied everywhere (does ONE macro cover many states?)

If NEAREST beats pi_0 but BC does not → multimodality-of-regression (fix: classify/MDN). If NEAREST also fails → the
option θ are state-specific knife-edge and do not transfer open-loop (the option must stay closed-loop / re-planned).
"""
import json
import sys
from collections import Counter

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_handoff import sequence_then_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_option import option_controller_rollout  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, T_MAX, T_MIN, structured_random  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import LateStart, build_boundary_panel, reconstruct_handoff, verify_reconstruction  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
FAMS_CARRY = ("contact_retention", "transport", "braking")
SHOTS, EVAL_H = 64, 160


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _panel(pi0, seeds, forbidden, want):
    panel, _c, _s = build_boundary_panel(pi0, seeds, forbidden, want=want, families=FAMS_CARRY, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    templ, fam = [], []
    for ls in panel:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["gate_ok"]
        rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        assert int(rl._strict) == 0 and rec.gate_mult == 1.0
        templ.append((rl, gate)); fam.append(ls.family)
    return templ, fam


def _norm(t):
    t = np.asarray(t, np.float32)
    return np.concatenate([t[:, :12] / A_BOUND, (t[:, 12:] - T_MIN) / (T_MAX - T_MIN)], -1)


def _kmeans(X, K, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1); lab = d.argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
    medoid = [int(np.argmin(((X - C[k]) ** 2).sum(-1) + (lab != k) * 1e9)) for k in range(K)]
    return lab, medoid


class _Clf(nn.Module):
    def __init__(self, K, obs=48, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, o):
        return self.net(o)


def _train_clf(obs, lab, K, *, epochs, seed):
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(obs, np.float32)); y = torch.as_tensor(np.asarray(lab, np.int64))
    clf = _Clf(K); opt = torch.optim.Adam(clf.parameters(), 3e-3); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lf(clf(x), y); loss.backward(); opt.step()
    return clf


def _roll_fixed(templ, theta_of, pi0, base):
    """Execute a per-state fixed θ (theta_of(i)) through the committed-macro controller; K6/handoff/exit rates."""
    import copy
    return [option_controller_rollout(copy.deepcopy(templ[i][0]), copy.deepcopy(templ[i][1]), theta_of(i), pi0, base, horizon=EVAL_H) for i in range(len(templ))]


def main(smoke=False):
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)
    import copy

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    adim = pi0.action_dim; base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = set(ls.seed for ls in _bank(cfg["banks"]["late_train"])) | set(ls.seed for ls in _bank(cfg["banks"]["late_dev"]))
    shots = 16 if smoke else SHOTS
    K = 4 if smoke else 8

    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    bank_obs, bank_th = z["obs"].astype(np.float32), z["theta"].astype(np.float32)
    prov = json.load(open(f"{D}/carry_option_teacher_bank_v1.json"))["provenance"]
    robusts = np.array([p["robust_k6"] if p.get("robust_k6") is not None else 0.0 for p in prov if p["confident"]], np.float32)
    log(f"[bank] {len(bank_obs)} confident θ | robust_k6 mean {round(float(robusts.mean()),3)} max {round(float(robusts.max()),3)}")

    ev_templ, ev_fam = _panel(pi0, range(11000, 13000), forbidden, 10 if smoke else 30)
    log(f"[panel] disjoint eval {len(ev_templ)} ({dict(Counter(ev_fam))})")

    # NEAREST_RETRIEVAL: nearest bank obs (L2) → its θ
    def nearest_theta(i):
        o = ev_templ[i][0].obs()
        return bank_th[int(((bank_obs - o) ** 2).sum(1).argmin())]
    # TEMPLATE_CLASSIFY: k-means θ (normalized) → medoid templates; classify obs → template
    lab, medoid = _kmeans(_norm(bank_th), K, seed=0)
    templates = bank_th[medoid]
    clf = _train_clf(bank_obs, lab, K, epochs=(60 if smoke else 300), seed=0)
    with torch.no_grad():
        ev_ids = clf(torch.as_tensor(np.stack([ev_templ[i][0].obs() for i in range(len(ev_templ))]))).argmax(1).numpy()
    def template_theta(i):
        return templates[ev_ids[i]]
    # RANDOM_BANK_THETA control + GLOBAL_ROBUST_THETA
    rng = np.random.default_rng(7)
    rand_ids = rng.integers(0, len(bank_th), len(ev_templ))
    def random_theta(i):
        return bank_th[rand_ids[i]]
    global_robust = bank_th[int(robusts.argmax())]
    def global_theta(_i):
        return global_robust

    methods = {"NEAREST_RETRIEVAL": nearest_theta, "TEMPLATE_CLASSIFY": template_theta,
               "RANDOM_BANK_THETA": random_theta, "GLOBAL_ROBUST_THETA": global_theta}
    outs = {name: _roll_fixed(ev_templ, fn, pi0, base) for name, fn in methods.items()}
    pi0_out = [sequence_then_pi0(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.zeros((0, adim), np.float32), horizon=EVAL_H) for i in range(len(ev_templ))]
    exp_out = [structured_random(copy.deepcopy(ev_templ[i][0]), copy.deepcopy(ev_templ[i][1]), pi0, base, np.random.default_rng(900 + i), shots=shots, horizon=EVAL_H) for i in range(len(ev_templ))]

    def rate(o, k="k6"):
        return round(float(np.mean([x[k] for x in o])), 3)
    def anyexit(o):
        return round(float(np.mean([x["contain_exit_ct"] > 0 for x in o])), 3)
    agg = {"pi_0": {"K6": rate(pi0_out), "handoff": rate(pi0_out, "reached_handoff"), "any_exit": anyexit(pi0_out)},
           "structured_expert": {"K6": rate(exp_out), "handoff": rate(exp_out, "reached_handoff"), "any_exit": anyexit(exp_out)}}
    for name, o in outs.items():
        agg[name] = {"K6": rate(o), "handoff": rate(o, "reached_handoff"), "any_exit": anyexit(o)}

    pk6, ek6 = agg["pi_0"]["K6"], agg["structured_expert"]["K6"]
    nk6 = agg["NEAREST_RETRIEVAL"]["K6"]; tk6 = agg["TEMPLATE_CLASSIFY"]["K6"]; rk6 = agg["RANDOM_BANK_THETA"]["K6"]
    best_distil = max(nk6, tk6); frac = best_distil / ek6 if ek6 > 0 else 0.0
    classify_helps = tk6 > max(nk6, rk6) + 1e-9                          # multimodality-respecting selection beats nearest/random
    if ek6 <= pk6 + 0.05:
        mech = "OPTION_ADVANTAGE_NOT_REPRODUCED_ON_THIS_EVAL"
        finding = f"expert {ek6} ≈ pi_0 {pk6} on this eval panel — the carry-option advantage did not reproduce here; do not conclude from this panel alone"
    elif best_distil <= pk6 + 0.05:
        mech = "OPTION_THETA_DO_NOT_TRANSFER_OPEN_LOOP"
        finding = f"live expert {ek6} > pi_0 {pk6} but no fixed-θ distillation transfers (nearest {nk6}, template {tk6}) → option θ are state-specific/knife-edge (robust_k6 {round(float(robusts.mean()),3)}); the option must stay closed-loop / re-searched per state"
    elif frac >= 0.6:
        mech = "OPTION_DISTILLATION_TRANSFERS"
        finding = f"best distillation {best_distil} reaches {round(frac,2)}× the per-state expert {ek6} and beats pi_0 {pk6} → deterministic MSE-BC failed by mode-averaging, distillation itself is viable"
    else:
        mech = "OPTION_DISTILLATION_COMPOUND_BLOCK"
        finding = (f"compound: multimodality-respecting classification {'helps' if classify_helps else 'does not help'} over BC/nearest "
                   f"(template {tk6} vs nearest {nk6} vs random {rk6}, BC 0.0), BUT the best open-loop distillation {best_distil} reaches only "
                   f"{round(frac,2)}× the per-state expert {ek6} → knife-edge state-specific θ basins (robust_k6 {round(float(robusts.mean()),3)}) dominate; "
                   "open-loop distillation is blocked, the option advantage needs per-state search or its cheap amortization")

    out = {"contract": "CARRY_OPTION_DIAGNOSTIC_V1", "date": "2026-07-24", "no_new_campaign": True, "smoke": smoke,
           "eval_n": len(ev_templ), "eval_families": dict(Counter(ev_fam)), "K_templates": K,
           "template_cluster_sizes": dict(Counter(int(x) for x in lab)), "bank_robust_k6_mean": round(float(robusts.mean()), 3),
           "aggregate": agg, "mechanism": mech, "finding": finding}
    json.dump(out, open(f"{D}/carry_option_diagnostic_v1.json", "w"), indent=1, default=float)

    log("\n== CARRY_OPTION_DIAGNOSTIC_V1 (same eval panel, same executor, only θ-selection differs) ==")
    for name in ("pi_0", "structured_expert", "NEAREST_RETRIEVAL", "TEMPLATE_CLASSIFY", "RANDOM_BANK_THETA", "GLOBAL_ROBUST_THETA"):
        a = agg[name]; log(f"  {name:20}: K6 {a['K6']} | handoff {a['handoff']} | any_exit {a['any_exit']}")
    log(f"→ {mech}\n  {finding}")
    log(f"wrote {D}/carry_option_diagnostic_v1.json\nCARRY_OPTION_DIAGNOSTIC_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
