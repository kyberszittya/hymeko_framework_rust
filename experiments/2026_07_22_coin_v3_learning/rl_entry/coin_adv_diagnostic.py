"""Discriminating diagnostic for the advantage critic: does it fit the return-marginal dG IN-SAMPLE? If in-sample
corr is high but held-out is low/negative -> overfitting (small data). If in-sample is ALSO low -> dG is not a
function of the 48-dim obs+state+delta (Markov insufficiency: node_features lacks coin velocity). This decides whether
RESIDUAL_CRITIC_ROUTE_BLOCKED is honest or premature.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "dev", "/private/tmp/claude-501/-Users-kyberszittya-hakiko-ai-ws-03-implementation-hymeko-framework-rust/63ad1b54-314a-48f8-b561-ba4a163f847c/scratchpad/coin_residual_critic_dev.py")
sys.argv = ["x", sys.argv[1]]
dev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dev)
from hymeko_rl.coin_delivery.coin_residual_critic import encode_controller_states  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

PI0 = sys.argv[1]


def corr_pred_dg(adv, fam):
    pr, dg = [], []
    for f, states in fam.items():
        for st in states:
            base = st["base"]; G = np.array(st["G"]); G0 = st["G0"]
            for a, g in zip(st["cand_actions"], G - G0):
                O = torch.tensor(st["obs"][None]); E = encode_controller_states([st["cstate"]])
                de = torch.tensor((np.asarray(a) - base).astype(np.float32)[None])
                with torch.no_grad():
                    pr.append(float(adv(O, E, de)))
                dg.append(float(g))
    pr = np.array(pr); dg = np.array(dg)
    c = float(np.corrcoef(pr, dg)[0, 1]) if np.std(pr) > 1e-9 and np.std(dg) > 1e-9 else 0.0
    return c, len(pr), float(np.std(dg))


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    print("building ADV_TRAIN panel...", flush=True)
    tr = dev.build_panel(pi0, dev.ADV_TRAIN)
    print("building ADV_DEV panel...", flush=True)
    dv = dev.build_panel(pi0, dev.ADV_DEV)
    n_train_pairs = sum(len(st["G"]) for f in tr for st in tr[f])
    print(f"train candidate pairs {n_train_pairs}", flush=True)
    # train the advantage critic (same as dev harness) and also a HEAVILY-trained version to test fit ceiling
    for steps in (4000, 20000):
        adv = dev.train_adv_critic(tr, steps=steps)
        c_in, n_in, sd_in = corr_pred_dg(adv, tr)
        c_dev, n_dev, sd_dev = corr_pred_dg(adv, dv)
        print(f"  steps={steps}: IN-SAMPLE corr(pred,dG)={c_in:+.3f} (n={n_in}) | HELD-OUT corr={c_dev:+.3f} (n={n_dev})",
              flush=True)
    diag = {"in_sample_corr_4k": None}
    adv = dev.train_adv_critic(tr, steps=20000)
    c_in, _, _ = corr_pred_dg(adv, tr); c_dev, _, _ = corr_pred_dg(adv, dv)
    verdict = ("OVERFIT_MARKOV_OK" if c_in > 0.6 and c_dev < 0.3 else
               "TARGET_UNLEARNABLE_FROM_OBS" if c_in < 0.4 else "LEARNABLE")
    diag = {"in_sample_corr": round(c_in, 3), "held_out_corr": round(c_dev, 3), "dg_std_train": round(sd_in, 3),
            "interpretation": verdict}
    json.dump(diag, open("/tmp/adv_diag.json", "w"), indent=1)
    print(f"\nINTERPRETATION: {verdict} (in-sample {c_in:+.3f}, held-out {c_dev:+.3f})", flush=True)
    print("ADV_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
