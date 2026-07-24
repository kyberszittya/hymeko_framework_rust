"""O2 geometry-aliasing diagnostics (DIAGNOSTIC CONTROLS — no assimilation implementation).

The combined square+2:1+3:1 proposal fit at res_mse 0.946 (vs 0.13 square-only) suggests GEOMETRY ALIASING: the shape-blind
obs maps near-identical states to DIFFERENT winning θ depending on the object. Three controls test that hypothesis
directly, before any RepresentationAdapter is built:

  1. PER-SHAPE fit — fit a proposal per shape; report square / 2:1 / 3:1 res_mse separately (is the aggregate 0.946 driven
     by cross-shape averaging, or is each shape individually hard?).
  2. SHAPE-CONDITIONED PROBE — a plain MLP regressor obs→θ vs (obs + [hx, hy, hx/hy, sin rz, cos rz])→θ on the SAME labels.
     If the geometry descriptor collapses the validation error, the missing geometric representation is DIRECTLY shown.
  3. TEACHER MULTIMODALITY — for near-obs neighbours, θ-distance within a shape vs across shapes. If near-obs pairs from
     different shapes have DISTANT winning θ, the shape-blind map is information-theoretically underdetermined.

This does NOT implement the RepresentationAdapter; it is a diagnostic that says whether one is warranted.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_balltip_proposal import D, _bank  # noqa: E402
from coin_object_o2 import SHAPES, _ball_tf, _hxy  # noqa: E402
from coin_carry_option_teacher_bank import generate_bank  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_proposal import fit_proposal, norm_theta  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_late_start import build_boundary_panel  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

OUT = "reports/2026-07-24-o2-square-rectangle-fresh-reconstruct"
FAMS = ("contact_retention", "transport", "braking")


def geom_desc(shape, rz):
    hx, hy = _hxy(SHAPES[shape])
    return np.array([hx, hy, hx / hy, math.sin(rz), math.cos(rz)], np.float32)


def per_shape_bank(pi0, base, shape, cand_ls, shots, want, log):
    """Fresh box teacher labels for one shape, recording the object geometry descriptor per label (hx,hy,rz)."""
    hx, hy = _hxy(SHAPES[shape])
    rk = {"geom": "POINT", "arm_mjcf_transform": _ball_tf, "coin_shape": "box", "disk_radius_override": hx, "disk_radius_y_override": hy}
    obs, theta, geom = [], [], []
    o, t, _p = generate_bank(pi0, base, cand_ls[:want * 3], shots=shots, reconstruct_kwargs=rk, log=log)
    for ob, th in zip(o, t):
        obs.append(ob)
        theta.append(th)
        geom.append(geom_desc(shape, 0.0))                              # teacher states are axis-start (rz≈0); descriptor carries hx,hy
        if len(obs) >= want:
            break
    return np.asarray(obs, np.float32), np.asarray(theta, np.float32), np.asarray(geom, np.float32)


def _mlp_val_mse(X, Y, *, seed=0, epochs=400, h=128):
    """Train a plain 2-layer MLP X→Y (θ in norm space), 80/20 split, return validation MSE."""
    import torch
    torch.manual_seed(seed)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    ntr = int(0.8 * n)
    tr, va = idx[:ntr], idx[ntr:]
    xt = torch.as_tensor(X[tr])
    yt = torch.as_tensor(Y[tr])
    xv = torch.as_tensor(X[va])
    yv = torch.as_tensor(Y[va])
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], h), torch.nn.ReLU(), torch.nn.Linear(h, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return round(float(((net(xv) - yv) ** 2).mean()), 4)


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
    want, shots = (8, 24) if smoke else (45, 128)

    banks = {}
    for sh in (["square_1_1"] if smoke else list(SHAPES)):
        cand, _c, _s = build_boundary_panel(pi0, range(9000, 10800), forbidden, want=(12 if smoke else 140),
                                            families=FAMS, strict_primary=(0,), strict_fill=(), per_seed_cap=3)
        X, Y, G = per_shape_bank(pi0, base, sh, cand, shots, want, log)
        banks[sh] = (X, Y, G)
        log(f"  [{sh}] labels {len(X)}")

    # 1. per-shape fit MSE (each shape alone) + combined
    per_shape = {}
    for sh, (X, Y, _G) in banks.items():
        if len(X) < 4:
            per_shape[sh] = None
            continue
        kk = min(6, max(2, len(X) // 4))
        _p, fit = fit_proposal(X, Y, kk, clf_epochs=(60 if smoke else 200), res_epochs=(60 if smoke else 200))
        per_shape[sh] = {"n": len(X), "K": kk, "res_mse": round(fit["res_mse"], 4)}
    Xc = np.concatenate([banks[s][0] for s in banks])
    Yc = np.concatenate([banks[s][1] for s in banks])
    Gc = np.concatenate([banks[s][2] for s in banks])
    kkc = min(6, max(2, len(Xc) // 4))
    _pc, fitc = fit_proposal(Xc, Yc, kkc, clf_epochs=(60 if smoke else 200), res_epochs=(60 if smoke else 200))
    combined_res_mse = round(fitc["res_mse"], 4)

    # 2. shape-conditioned probe: obs vs obs+geom (val MSE of a plain MLP; θ in norm space)
    Yn = norm_theta(Yc)
    mse_obs = _mlp_val_mse(Xc, Yn, epochs=(120 if smoke else 400))
    mse_obs_geom = _mlp_val_mse(np.concatenate([Xc, Gc], 1), Yn, epochs=(120 if smoke else 400))
    probe_ratio = round(mse_obs / max(1e-9, mse_obs_geom), 3)

    # 3. teacher multimodality: for near-obs pairs, θ-distance within-shape vs across-shape
    shape_of = np.concatenate([[i] * len(banks[s][0]) for i, s in enumerate(banks)])
    Dobs = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
    Dth = np.linalg.norm(norm_theta(Yc)[:, None, :] - norm_theta(Yc)[None, :, :], axis=2)
    np.fill_diagonal(Dobs, np.inf)
    within, across = [], []
    for i in range(len(Xc)):
        j = int(np.argmin(Dobs[i]))                                    # nearest-obs neighbour
        (within if shape_of[i] == shape_of[j] else across).append(float(Dth[i, j]))
    multimodality = {"nearest_obs_theta_dist_within_shape": round(float(np.mean(within)), 3) if within else None,
                     "nearest_obs_theta_dist_across_shape": round(float(np.mean(across)), 3) if across else None,
                     "n_within": len(within), "n_across": len(across)}

    out = {"contract": "O2_GEOMETRY_PROBE", "date": "2026-07-24", "smoke": smoke,
           "per_shape_fit": per_shape, "combined_res_mse": combined_res_mse,
           "shape_conditioned_probe": {"val_mse_obs": mse_obs, "val_mse_obs_plus_geom": mse_obs_geom,
                                       "improvement_ratio_obs_over_geom": probe_ratio,
                                       "geom_descriptor": "[hx, hy, hx/hy, sin(rz), cos(rz)]"},
           "teacher_multimodality": multimodality}
    json.dump(out, open(f"{OUT}/o2_geometry_probe.json", "w"), indent=1, default=float)

    log("\n== O2 geometry-aliasing diagnostics ==")
    log(f"  per-shape res_mse: {[(s, per_shape[s]['res_mse'] if per_shape[s] else None) for s in per_shape]} | combined {combined_res_mse}")
    log(f"  shape-conditioned probe val MSE: obs {mse_obs} vs obs+geom {mse_obs_geom} (obs/geom ratio {probe_ratio}×)")
    log(f"  teacher multimodality (nearest-obs θ-dist): within-shape {multimodality['nearest_obs_theta_dist_within_shape']} "
        f"vs across-shape {multimodality['nearest_obs_theta_dist_across_shape']}")
    verdict = ("GEOMETRY_CONDITIONING_WARRANTED" if probe_ratio >= 1.5 else "GEOMETRY_CONDITIONING_NOT_DECISIVE")
    log(f"→ {verdict}\n  artifact: {OUT}/o2_geometry_probe.json\nO2_GEOM_PROBE_DONE")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
