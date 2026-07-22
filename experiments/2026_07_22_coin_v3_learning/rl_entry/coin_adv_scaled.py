"""Scaled advantage-critic development (§13). The 32-state panel overfit (+0.996 in / +0.111 held-out); scale the
counterfactual training data ~6x + weight decay and re-audit generalization on a disjoint dev panel with the full
actor-relevant metric suite (centered corr, margin-aware ranking, +gradA vs -gradA empirical). Fresh disjoint banks.
"""
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "dev", "/private/tmp/claude-501/-Users-kyberszittya-hakiko-ai-ws-03-implementation-hymeko-framework-rust/63ad1b54-314a-48f8-b561-ba4a163f847c/scratchpad/coin_residual_critic_dev.py")
sys.argv = ["x", sys.argv[1]]
dev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dev)
from hymeko_rl.coin_delivery.coin_residual_critic import encode_controller_states  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/adv_scaled.json"
BOUND = 0.25
ADV_TRAIN_BIG = list(range(6100, 6300))   # 200 seeds
ADV_DEV = list(range(7100, 7140))
ADV_SEALED = list(range(7140, 7156))


def train_adv(fam, steps=15000, seed=0, wd=1e-4):
    torch.manual_seed(seed)
    O, E, DE, DG = [], [], [], []
    for f, states in fam.items():
        for st in states:
            for a, dg in zip(st["cand_actions"], np.array(st["G"]) - st["G0"]):
                O.append(st["obs"]); E.append(st["cstate"]); DE.append((np.asarray(a) - st["base"]).astype(np.float32)); DG.append(float(dg))
    O = torch.tensor(np.stack(O)); Enc = encode_controller_states(E); DEt = torch.tensor(np.stack(DE)); DGt = torch.tensor(np.array(DG, np.float32))
    adv = dev.ResidualAdvantageCritic(); opt = torch.optim.Adam(adv.parameters(), lr=1e-3, weight_decay=wd)
    n = len(O); rng = np.random.default_rng(seed)
    for _i in range(steps):
        idx = rng.integers(0, n, min(512, n)); i2 = rng.integers(0, n, min(512, n))
        pred = adv(O[idx], Enc[idx], DEt[idx])
        mse = ((pred - DGt[idx]) ** 2).mean()
        rank = torch.relu(-torch.sign(DGt[idx] - DGt[i2]) * (pred - adv(O[i2], Enc[i2], DEt[i2]))).mean()
        opt.zero_grad(); (mse + 0.5 * rank).backward(); opt.step()
    return adv, n


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); rl = CoinRL4Dof()
    print("building LARGE adv train panel (200 seeds)...", flush=True)
    tr = dev.build_panel(pi0, ADV_TRAIN_BIG, per_family=60)
    nstates = {f: len(tr[f]) for f in tr}
    print(f"  train states {nstates} total {sum(nstates.values())}", flush=True)
    print("building adv dev panel...", flush=True)
    dv = dev.build_panel(pi0, ADV_DEV, per_family=10)
    adv, npairs = train_adv(tr)
    print(f"  trained on {npairs} candidate pairs", flush=True)

    def adv_q(st, a, h, A=adv):
        O = torch.tensor(st["obs"][None]); E = encode_controller_states([st["cstate"]])
        de = torch.tensor((np.asarray(a) - st["base"]).astype(np.float32)[None])
        with torch.no_grad():
            return float(A(O, E, de))

    def adv_grad(st, A=adv):
        O = torch.tensor(st["obs"][None]); E = encode_controller_states([st["cstate"]])
        de = torch.zeros(1, 4, requires_grad=True); v = A(O, E, de); v.backward()
        return de.grad[0].numpy()

    # in-sample vs held-out generalization corr
    def corr(fam):
        pr, dg = [], []
        for f, states in fam.items():
            for st in states:
                for a, g in zip(st["cand_actions"], np.array(st["G"]) - st["G0"]):
                    pr.append(adv_q(st, a, "Q1")); dg.append(float(g))
        pr = np.array(pr); dg = np.array(dg)
        return float(np.corrcoef(pr, dg)[0, 1]) if np.std(pr) > 1e-9 else 0.0
    c_in, c_dev = corr(tr), corr(dv)
    print(f"  generalization: in-sample corr {c_in:+.3f} | held-out corr {c_dev:+.3f}", flush=True)
    res = dev.audit_panel(dv, adv_q, adv_grad, rl, pi0, "adv-scaled")
    dp = dev.__dict__

    def fam_pass(r, f):
        x = r.get(f, {})
        return x.get("n", 0) > 0 and x.get("centered_corr_Q1_vs_dG", -1) > 0.2 and x.get("gradQ1_wins", 0) > 0.55
    passed = (c_dev > 0.3 and fam_pass(res, "transport") and fam_pass(res, "contact_retention")
              and res.get("transport", {}).get("harmful_rej", 0) > 0.6)
    out = {"pi0_sha": hashlib.sha256(open(PI0, "rb").read()).hexdigest()[:8], "train_states": nstates,
           "train_pairs": npairs, "in_sample_corr": round(c_in, 3), "held_out_corr": round(c_dev, 3),
           "dev_metrics": res, "advantage_development_pass": bool(passed),
           "verdict": "RESIDUAL_ADVANTAGE_CRITIC_DEVELOPMENT_PASS" if passed else "ADVANTAGE_DEV_INSUFFICIENT"}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"\ngeneralization in {c_in:+.2f} dev {c_dev:+.2f} | dev pass {passed}\n{out['verdict']}", flush=True)
    print("ADV_SCALED_DONE", flush=True)


if __name__ == "__main__":
    main()
