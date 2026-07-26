"""DECISION-TIME REPRESENTATION AUDIT (Step 1) — is the current 42-D feature space one in which the cradle→working-θ map
is learnable AT ALL? No training; development-only; the held-out overlay is frozen DIAGNOSIS, never model selection.

Coverage (COVERAGE_ALONE_INSUFFICIENT) and proposal modality (MULTIMODALITY_PRESENT_BUT_UPDATE_ZERO_STILL_FAILS) are both
excluded. Before engineering a new representation (R1/R2), audit WHY the current one fails to generalise:

  1. SMOOTHNESS / coordinate breaks — do feature-close cradles have θ-close solutions? Pairwise ‖Δθ‖ vs ‖Δφ‖, their
     correlation, and the Lipschitz ratio ‖Δθ‖/‖Δφ‖ (a large max ⇒ a coordinate-dependent break: a small feature change
     maps to a large θ change, which no smooth regressor can track).
  2. NEAREST-NEIGHBOUR RETRIEVAL (training-free) — the simplest possible feature→θ generalisation: propose the
     nearest-feature OTHER cradle's canonical θ and run the SAME budget-8 search. If even retrieval delivers, the feature
     space clusters delivering strategies (the learned model was the weak link); if it fails, feature-proximity does not
     predict θ-transferability — a representation defect. Run dev leave-one-out + a frozen held-out overlay.
  3. CANONICAL LEFT-RIGHT ORDERING — the current features concatenate left-then-right in FIXED order (no canonicalisation),
     so a cradle and its mirror get different vectors. Quantify the order-sensitivity (swap the per-side entries and
     measure the feature displacement) — a structural defect R1's canonical ordering is meant to remove.

The verdict frames what R1 must fix; it does not itself select a model.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def pairwise(vecs: dict[str, np.ndarray]) -> "tuple[list[str], np.ndarray]":
    """Tags + the symmetric pairwise L2 distance matrix over a {tag: vector} dict (deterministic tag order)."""
    tags = sorted(vecs)
    X = np.asarray([np.asarray(vecs[t], np.float64) for t in tags])
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return tags, D


def lipschitz_analysis(feats: dict[str, np.ndarray], thetas_norm: dict[str, np.ndarray]) -> dict[str, Any]:
    """Pairwise feature distance vs canonical-θ distance across cradles: the Pearson correlation (does feature-proximity
    predict θ-proximity?) and the Lipschitz ratio ‖Δθ‖/‖Δφ‖ (min/median/max). Low correlation OR a large max ratio ⇒ the
    feature→θ map is not smoothly learnable in this coordinate system. # Postconditions: correlation ∈ [-1,1]."""
    tags, Dphi = pairwise(feats)
    _tt, Dtheta = pairwise({t: thetas_norm[t] for t in tags})
    iu = np.triu_indices(len(tags), k=1)
    dphi, dth = Dphi[iu], Dtheta[iu]
    ratios = dth / np.maximum(dphi, 1e-9)
    corr = float(np.corrcoef(dphi, dth)[0, 1]) if len(dphi) > 1 and dphi.std() > 0 and dth.std() > 0 else 0.0
    return {"tags": tags, "n_pairs": int(len(dphi)),
            "feature_dist": {"min": round(float(dphi.min()), 4), "median": round(float(np.median(dphi)), 4), "max": round(float(dphi.max()), 4)},
            "theta_dist": {"min": round(float(dth.min()), 4), "median": round(float(np.median(dth)), 4), "max": round(float(dth.max()), 4)},
            "corr_dphi_dtheta": round(corr, 4),
            "lipschitz_ratio": {"min": round(float(ratios.min()), 4), "median": round(float(np.median(ratios)), 4), "max": round(float(ratios.max()), 4)},
            "nearest_feature_pair_is_nearest_theta_pair": bool(np.argmin(dphi) == np.argmin(dth))}


def nearest_neighbour_by_feature(feats_all: dict[str, np.ndarray], query_tags: "list[str]",
                                 candidate_tags: "list[str]") -> dict[str, Any]:
    """For each tag in ``query_tags``, its nearest cradle among ``candidate_tags`` by feature distance (self excluded;
    deterministic tie-break to the lower tag). ``feats_all`` must contain every query AND candidate tag. Used for
    training-free retrieval (propose the neighbour's canonical θ) — dev LODO uses candidates=dev; the held-out overlay uses
    candidates=dev. # Postconditions: never maps a tag to itself."""
    out = {}
    for t in query_tags:
        cands = [(float(np.linalg.norm(np.asarray(feats_all[t], np.float64) - np.asarray(feats_all[c], np.float64))), c)
                 for c in candidate_tags if c != t]
        cands.sort(key=lambda z: (z[0], z[1]))
        out[t] = {"nn_tag": cands[0][1], "nn_feature_dist": round(cands[0][0], 4)} if cands else {"nn_tag": None}
    return out


# ── canonical L/R ordering deficit ──
# feature groups (dataset.FEATURE_ORDER) that encode per-side quantities in FIXED left-then-right order:
_CONTACT_PAIR_GROUPS = ("fn", "normal", "xc_rel")   # fn=[L,R], normal=[n_l(2),n_r(2)], xc_rel=[xc_l(2),xc_r(2)]
_JOINT_GROUPS = ("q", "qdot", "prev_tau", "saturated")   # 4 joints; ASSUME [0:2]=left arm, [2:4]=right arm


def swap_lr(feat_grouped: dict[str, np.ndarray], *, include_joints: bool = False) -> dict[str, np.ndarray]:
    """Return the feature dict with per-SIDE entries left↔right swapped. Contact pairs always; the 4-joint groups
    (assuming joints 0,1=left / 2,3=right) only when ``include_joints``. Shared quantities (dtz, target axis, coin state,
    straddle) are unchanged. This is the permutation a mirror-symmetric cradle would induce (a true mirror ALSO negates the
    perpendicular vector component — this swap is a lower bound on the mismatch)."""
    g = {k: np.asarray(v, np.float64).copy() for k, v in feat_grouped.items()}
    for grp in _CONTACT_PAIR_GROUPS:
        if grp in g:
            v = g[grp]
            h = v.shape[0] // 2
            g[grp] = np.concatenate([v[h:], v[:h]])
    if include_joints:
        for grp in _JOINT_GROUPS:
            if grp in g and g[grp].shape[0] == 4:
                g[grp] = g[grp][[2, 3, 0, 1]]
        if "slew_head" in g and g["slew_head"].shape[0] == 8:   # [up(4), down(4)] per joint
            s = g["slew_head"]
            g["slew_head"] = np.concatenate([s[[2, 3, 0, 1]], s[4:][[2, 3, 0, 1]]])
    return g


def ordering_deficit(feats_grouped: dict[str, dict[str, np.ndarray]], flatten_fn: Any) -> dict[str, Any]:
    """Per-cradle order-sensitivity: ‖φ − φ_swapLR‖ (contact-only and contact+joints), normalised by ‖φ‖. A nonzero
    deficit proves the features are NOT canonically ordered — a mirror-symmetric cradle maps to a different vector, so the
    model cannot share the mirrored relation. # Postconditions: deficits ≥ 0."""
    per = {}
    for tag, fg in feats_grouped.items():
        base = flatten_fn(fg)
        d_contact = float(np.linalg.norm(base - flatten_fn(swap_lr(fg, include_joints=False))))
        d_full = float(np.linalg.norm(base - flatten_fn(swap_lr(fg, include_joints=True))))
        nrm = float(np.linalg.norm(base)) + 1e-9
        per[tag] = {"contact_swap_deficit": round(d_contact, 4), "full_swap_deficit": round(d_full, 4),
                    "contact_swap_deficit_rel": round(d_contact / nrm, 4), "full_swap_deficit_rel": round(d_full / nrm, 4)}
    mean_c = round(float(np.mean([v["contact_swap_deficit"] for v in per.values()])), 4)
    mean_f = round(float(np.mean([v["full_swap_deficit"] for v in per.values()])), 4)
    return {"per_cradle": per, "mean_contact_swap_deficit": mean_c, "mean_full_swap_deficit": mean_f,
            "features_are_canonically_ordered": bool(mean_f < 1e-6)}


def audit_verdict(nn_dev_k6: int, n_dev: int, nn_held_k6: int, n_held: int, lipschitz: dict[str, Any],
                  ordering: dict[str, Any]) -> dict[str, Any]:
    """Summarise what the audit says about the current 42-D coordinate system. RETRIEVABLE_ON_DEV ⇔ nearest-feature
    retrieval already delivers on dev LODO (feature-proximity predicts θ-transfer there); a large Lipschitz max or a low
    dφ–dθ correlation flags coordinate breaks; a nonzero ordering deficit flags the missing canonical L/R frame. These
    frame R1 — they do not select a model. # Postconditions: booleans reflect the measured diagnostics."""
    retrievable_dev = bool(nn_dev_k6 >= max(1, n_dev - 1))     # retrieval works on (almost) every dev LODO fold
    retrievable_held = bool(nn_held_k6 >= 1)
    smooth = bool(lipschitz["corr_dphi_dtheta"] > 0.3 and lipschitz["lipschitz_ratio"]["max"] < 5.0)
    canonical = bool(ordering["features_are_canonically_ordered"])
    defects = []
    if not retrievable_dev:
        defects.append("FEATURE_PROXIMITY_DOES_NOT_PREDICT_THETA_TRANSFER_ON_DEV")
    if not smooth:
        defects.append("NON_SMOOTH_OR_COORDINATE_DEPENDENT_MAP")
    if not canonical:
        defects.append("NO_CANONICAL_LEFT_RIGHT_ORDERING")
    return {"nn_retrieval_dev_k6": f"{nn_dev_k6}/{n_dev}", "nn_retrieval_held_out_k6": f"{nn_held_k6}/{n_held}",
            "retrieval_works_on_dev": retrievable_dev, "retrieval_works_on_held_out": retrievable_held,
            "map_is_smooth": smooth, "features_canonically_ordered": canonical,
            "identified_defects": defects,
            "audit_summary": ("CURRENT_42D_SUPPORTS_RETRIEVAL" if retrievable_dev
                              else "CURRENT_42D_DOES_NOT_ADMIT_A_LEARNABLE_MAP_AS_IS"),
            "note": "held-out overlay is frozen diagnosis, not model selection"}
