"""§5 conditional action-conflict, density-CONTROLLED, in FULL_ACTION_OBS_HISTORY_V1 space.

A HANDOFF_FEEDBACK_ONLY / B OPEN_LOOP_MIXED_LEGACY / C H30_FEEDBACK_ONLY / D HANDOFF_PLUS_H30.
Two matched estimators to defeat the dataset-size/density confound (the directive's "matched distance thresholds"):
  (1) SIZE-MATCHED k-NN: subsample every dataset's per-phase samples to the common minimum, then k-NN — same density.
  (2) FIXED-RADIUS: standardize the space on the pooled union, count only neighbours within a matched radius r.
Report per phase (TRANSPORT/SETTLING/STRICT_DWELL) + radius/k sensitivity. §7 gate: C transport & settling conflict
materially < B under matched density.
"""
import glob
import json
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.full_action_bc import load_trajectory_dataset
from hymeko_rl.coin_delivery.full_action_obs_history import build_history_features

PHN = ["APPROACH", "CONTACT_ACQ", "BILATERAL", "TRANSPORT", "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]
BASE, FB = sys.argv[1], sys.argv[2]
RNG = np.random.default_rng(0)


def base_oh(dirs, patterns, teacher=None, index=None):
    d = load_trajectory_dataset(dirs, patterns=patterns)
    obs, act, phase, traj = d["obs"], d["act"], d["phase"], d["traj"]
    if teacher is not None:
        keep = {r["seed"] for r in index if r.get("delivered") and r.get("teacher") == teacher}
        seeds = [int(f.split("traj_")[1].split(".npz")[0]) for f in d["files"]]
        ktids = {i for i, s in enumerate(seeds) if s in keep}
        m = np.isin(traj, list(ktids))
        obs, act, phase, traj = obs[m], act[m], phase[m], traj[m]
    return build_history_features(obs, act, traj), act, phase


def fb_oh(fb_dir):
    o, a, p = [], [], []
    for f in sorted(glob.glob(fb_dir + "/fb_*.npz")):
        z = np.load(f)
        o.append(z["obs_hist"])
        a.append(z["act"])
        p.append(z["phase"])
    return np.concatenate(o), np.concatenate(a), np.concatenate(p)


def _disp(nb):
    """neighbourhood action dispersion (mean L2 of neighbour actions to their local mean)."""
    c = nb - nb.mean(axis=1, keepdims=True)
    return np.linalg.norm(c, axis=-1).mean(axis=1)          # (N,)


def knn_matched(oh, act, n_sub, k=8):
    """Subsample to n_sub (matched density) then k-NN conditional conflict. Magnitude-gated cosine disagreement."""
    if len(oh) < k + 2 or n_sub < k + 2:
        return None
    sel = RNG.choice(len(oh), size=min(n_sub, len(oh)), replace=False)
    oh, act = oh[sel], act[sel]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(oh)
    _, idx = nn.kneighbors(oh)
    nb = act[idx[:, 1:]]
    disp = _disp(nb)
    scale = np.linalg.norm(act, axis=1) + 0.2
    mag = np.linalg.norm(act, axis=1)
    hi = mag > 0.5
    cos_dis = None
    if hi.sum() > k + 1:
        nbh = act[idx[hi][:, 1:]]
        md = nbh.mean(axis=1, keepdims=True)
        num = (nbh * md).sum(-1)
        den = np.linalg.norm(nbh, axis=-1) * np.linalg.norm(md, axis=-1) + 1e-9
        cos_dis = round(float(1.0 - (num / den).mean()), 4)
    return {"n": int(len(oh)), "cond_disp": round(float(disp.mean()), 4),
            "conflict_rate": round(float((disp / scale > 0.6).mean()), 4), "cos_dis": cos_dis}


def radius_conflict(oh_std, act, r):
    """Fixed-radius (standardized space): among samples with >=3 neighbours within r, mean action dispersion + rate."""
    nn = NearestNeighbors(radius=r).fit(oh_std)
    dist, idx = nn.radius_neighbors(oh_std)
    disps, confl, ncov = [], [], 0
    for i in range(len(oh_std)):
        nbrs = [j for j in idx[i] if j != i]
        if len(nbrs) < 3:
            continue
        ncov += 1
        nb = act[nbrs]
        d = float(np.linalg.norm(nb - nb.mean(0), axis=1).mean())
        disps.append(d)
        confl.append(d / (np.linalg.norm(act[i]) + 0.2) > 0.6)
    if not disps:
        return None
    return {"coverage": round(ncov / len(oh_std), 3), "cond_disp": round(float(np.mean(disps)), 4),
            "conflict_rate": round(float(np.mean(confl)), 4)}


def main():
    index = [json.loads(x) for x in open(BASE + "/index.jsonl")]
    ds = {"A_HANDOFF_ONLY": base_oh([BASE], ("traj_*.npz",), "handoff", index),
          "B_OPEN_LOOP_MIXED": base_oh([BASE], ("traj_*.npz",)),
          "C_H30_FEEDBACK": fb_oh(FB)}
    a, c = ds["A_HANDOFF_ONLY"], ds["C_H30_FEEDBACK"]
    ds["D_HANDOFF_PLUS_H30"] = (np.concatenate([a[0], c[0]]), np.concatenate([a[1], c[1]]), np.concatenate([a[2], c[2]]))
    for n, (oh, act, ph) in ds.items():
        print(f"{n}: {len(oh)} samples", flush=True)
    # common standardizer on the pooled union (same metric space for all)
    scaler = StandardScaler().fit(np.concatenate([ds[k][0] for k in ds]))
    out = {"size_matched_knn": {}, "fixed_radius": {}}
    for ph in (3, 5, 6):
        name = PHN[ph]
        # common per-phase sample floor (size match)
        counts = {k: int((ds[k][2] == ph).sum()) for k in ds}
        n_sub = min(counts.values())
        out["size_matched_knn"][name] = {"n_matched": n_sub, "raw_counts": counts, "datasets": {}}
        for k in ds:
            oh, act, phase = ds[k]
            m = phase == ph
            out["size_matched_knn"][name]["datasets"][k] = knn_matched(oh[m], act[m], n_sub, k=8)
        # fixed radius (standardized), radius chosen from B's median NN dist for fairness
        out["fixed_radius"][name] = {}
        for r in (6.0, 9.0):
            out["fixed_radius"][name][f"r={r}"] = {}
            for k in ds:
                oh, act, phase = ds[k]
                m = phase == ph
                ohs = scaler.transform(oh[m]) if m.sum() > 5 else None
                out["fixed_radius"][name][f"r={r}"][k] = radius_conflict(ohs, act[m], r) if ohs is not None else None
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("\n=== SIZE-MATCHED k-NN conflict_rate (matched density) B vs C ===")
    for ph in ("TRANSPORT", "SETTLING", "STRICT_DWELL"):
        d = out["size_matched_knn"][ph]
        b, c = d["datasets"]["B_OPEN_LOOP_MIXED"], d["datasets"]["C_H30_FEEDBACK"]
        if b and c:
            print(f"  {ph:<13} n_matched={d['n_matched']:<5} B conflict={b['conflict_rate']:.3f} cos_dis={b['cos_dis']}  "
                  f"|| C conflict={c['conflict_rate']:.3f} cos_dis={c['cos_dis']}")
    print("\n=== FIXED-RADIUS r=9 (standardized) conflict_rate B vs C ===")
    for ph in ("TRANSPORT", "SETTLING", "STRICT_DWELL"):
        rr = out["fixed_radius"][ph]["r=9.0"]
        b, c = rr["B_OPEN_LOOP_MIXED"], rr["C_H30_FEEDBACK"]
        if b and c:
            print(f"  {ph:<13} B conflict={b['conflict_rate']:.3f} cov={b['coverage']}  || C conflict={c['conflict_rate']:.3f} cov={c['coverage']}")


if __name__ == "__main__":
    main()
