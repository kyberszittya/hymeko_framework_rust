"""O2 K-mode vs single-head fit control — the direct multimodal test (fit-level, on the O2 box labels).

The O2 geometry-probe showed obs+geom does NOT beat obs (0.967×) yet near-obs winning θ are distant within a shape (2.88):
`MULTIMODAL_OPTION_STRUCTURE_DOMINANT`. The decisive control the synthesis calls for: on the SAME box teacher labels, does a
K-MODE (best-of-K) head collapse the residual a single deterministic MSE head leaves? If best-of-K error ≪ single-head error
where obs+geom did not help, the `MultimodalProposal` requirement is directly confirmed at the fit level (physical validation
is O3-triangle). Ladder: single-det → K-mode(obs) → K-mode(obs+geom). No GRU; no LSTM here (labels are non-sequential).
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_object_o2 import SHAPES  # noqa: E402
from coin_object_o2_geometry_probe import per_shape_bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import norm_theta  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-o2-square-rectangle-fresh-reconstruct"
FAMS = ("contact_retention", "transport", "braking")


def kmode_val_error(X, Y, Kmode, *, seed=0, epochs=600, h=128):
    """Best-of-K head: MLP X → K·θ; loss = min_k ||Y − θ_k||² (winner-take-all). Kmode=1 is the single deterministic head.
    Returns validation best-of-K MSE (80/20 split) — the error the BEST of the K modes leaves."""
    import torch
    torch.manual_seed(seed)
    n, od = len(X), Y.shape[1]
    idx = np.random.default_rng(seed).permutation(n)
    ntr = int(0.8 * n)
    xt, yt = torch.as_tensor(X[idx[:ntr]]), torch.as_tensor(Y[idx[:ntr]])
    xv, yv = torch.as_tensor(X[idx[ntr:]]), torch.as_tensor(Y[idx[ntr:]])
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, Kmode * od))
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    def best_of_k(x, y):
        pred = net(x).reshape(len(x), Kmode, od)
        d = ((pred - y[:, None, :]) ** 2).mean(-1)         # per-mode MSE
        return d.min(1).values.mean()                       # best mode wins

    for _ in range(epochs):
        opt.zero_grad()
        best_of_k(xt, yt).backward()
        opt.step()
    with torch.no_grad():
        return round(float(best_of_k(xv, yv)), 4)


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    def log(*a):
        print(*a, flush=True)
    import os
    os.makedirs(OUT, exist_ok=True)

    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    want, shots = (8, 24) if smoke else (35, 96)

    Xs, Ys, Gs = [], [], []
    for sh in (["square_1_1"] if smoke else list(SHAPES)):
        cand, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(12 if smoke else 140),
                                            families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
        X, Y, G = per_shape_bank(pi0, base, sh, cand, shots, want, log)
        Xs.append(X)
        Ys.append(Y)
        Gs.append(G)
        log(f"  [{sh}] labels {len(X)}")
    X = np.concatenate(Xs)
    Y = norm_theta(np.concatenate(Ys))
    XG = np.concatenate([X, np.concatenate(Gs)], 1)
    log(f"[kmode] combined box labels {len(X)}")

    Ks = [1, 2, 4, 6]
    obs_err = {k: kmode_val_error(X, Y, k, epochs=(150 if smoke else 600)) for k in Ks}
    geom_err = {k: kmode_val_error(XG, Y, k, epochs=(150 if smoke else 600)) for k in Ks}
    ratio_single_over_bestK = round(obs_err[1] / max(1e-9, obs_err[max(Ks)]), 3)
    ratio_obs_over_geom_bestK = round(obs_err[max(Ks)] / max(1e-9, geom_err[max(Ks)]), 3)

    if ratio_single_over_bestK >= 1.5:
        verdict = "MULTIMODAL_PROPOSAL_RECOVERS_FIT" + ("_GEOMETRY_HELPS_MODES" if ratio_obs_over_geom_bestK >= 1.3 else "_GEOMETRY_MINOR")
    else:
        verdict = "MULTIMODAL_FIT_NOT_DECISIVE"
    out = {"contract": "O2_KMODE_CONTROL", "date": "2026-07-24", "smoke": smoke, "n_labels": len(X), "Ks": Ks,
           "val_bestK_err_obs": obs_err, "val_bestK_err_obs_plus_geom": geom_err,
           "single_over_bestK_ratio": ratio_single_over_bestK, "obs_over_geom_bestK_ratio": ratio_obs_over_geom_bestK,
           "verdict": verdict}
    json.dump(out, open(f"{OUT}/o2_kmode_control.json", "w"), indent=1, default=float)

    log("\n== O2 K-mode vs single-head fit control ==")
    log(f"  best-of-K val MSE (obs):      {obs_err}")
    log(f"  best-of-K val MSE (obs+geom): {geom_err}")
    log(f"  single(K=1)/best-of-K({max(Ks)}) ratio: {ratio_single_over_bestK}×  |  obs/obs+geom @ best-K: {ratio_obs_over_geom_bestK}×")
    log(f"→ {verdict}\n  artifact: {OUT}/o2_kmode_control.json\nO2_KMODE_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
