"""Acceptable-set harvest + MULTIMODALITY DISCRIMINATING TEST — the gate before any K-head multimodal proposal.

The coverage curve closed the coverage axis: a single-θ features-only regressor deploys 2/4 (held-out 0/2 at every N),
while the oracle (teacher θ + the same budget-8 search) is 4/4 — so a delivering θ exists but the actor's single θ₀ misses
it. Two hypotheses remain, and they demand different fixes:

    (A) the actor AVERAGES multiple valid modes into a physically-bad mean   → a multimodal (K-head) proposal is the fix;
    (B) the feature representation cannot read out the right mode at all      → REPRESENTATION is the blocker, not modality.

This module runs the discriminating test BEFORE committing to (A). On DEVELOPMENT cradles only, it harvests the
acceptable set — every θ that is frozen-K6 successful AND motion-contract compatible — by GLOBAL sampling of the option
box (local jitter can only see one basin), clusters the successful θ in normalised option-space (single-linkage connected
components), and tests two direct multimodality signatures:

    • ≥2 well-separated delivering basins exist (pooled and/or per state);
    • the acceptable-set CENTROID (what an MSE regressor targets) is itself NON-delivering — direct evidence that
      averaging modes yields a bad θ.

Held-out s4/s7 are eval-only: their ALREADY-recorded delivering teacher θ is overlaid for ANALYSIS (does it fall inside a
dev basin → multimodal-recoverable, or outside every basin → OOD/representational) — no held-out θ feeds any fit.

Verdict → `MULTIMODAL_BASINS_PRESENT` (justifies the K-head; proceed) or `SINGLE_CONNECTED_CLUSTER`
(= REPRESENTATION_NOT_PROPOSAL_MODALITY_IS_BLOCKER; stop — a multimodal head cannot help).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import SETTLE_VEL
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
from hymeko_rl.coin_delivery.theta_option.semantics import DIM, DELIVERY_CFG, ThetaBox
from hymeko_rl.env.motion_contract import MotionLimits


def is_motion_compatible(m: dict[str, Any], limits: MotionLimits = MotionLimits()) -> bool:
    """The frozen motion contract on a rollout: bounded joint speed, bounded coin speed, and braked to rest (terminal coin
    speed below the settle threshold). K6 already requires the terminal settle; this makes the acceptance explicit and
    independent of the K6 predicate. # Postconditions: True ⇒ within all three bounds."""
    return bool(m["peak_qdot"] <= limits.joint_vel_hard
                and m["peak_coin_speed"] <= limits.ee_speed_hard
                and m["terminal_coin_speed"] < SETTLE_VEL)


@dataclass(frozen=True)
class AcceptableTheta:
    """One accepted solution: frozen-K6 delivered AND motion-compatible. ``theta`` legal, ``theta_norm`` in [-1,1]^6."""

    theta: np.ndarray
    theta_norm: np.ndarray
    dtz_end_mm: float
    terminal_coin_speed: float
    peak_qdot: float
    peak_coin_speed: float


def harvest_acceptable_set(snap: CradleSnapshot, cfg: Any = DELIVERY_CFG, *, n_samples: int, seed: int,
                           limits: MotionLimits = MotionLimits(), seed_thetas: "list[np.ndarray] | None" = None,
                           progress: "Any | None" = None) -> dict[str, Any]:
    """Harvest the acceptable set at ``snap`` by GLOBAL uniform sampling of the normalised option box (plus any
    ``seed_thetas`` — e.g. the known canonical + its local basin — to guarantee the primary basin is represented). Keep
    each θ that delivers frozen-K6 AND is motion-compatible. Global sampling (not local jitter) is what can reveal a
    SECOND basin. # Preconditions: n_samples ≥ 0. # Postconditions: every kept θ satisfies delivery_success ∧
    is_motion_compatible at stored precision; deterministic given ``seed``."""
    box = ThetaBox()
    rng = np.random.default_rng(seed)
    zs = list(rng.uniform(-1.0, 1.0, size=(int(n_samples), DIM)))
    if seed_thetas:
        zs = [box.norm(np.asarray(t, np.float64)) for t in seed_thetas] + zs
    accepted: list[AcceptableTheta] = []
    n_deliver = 0
    for i, z in enumerate(zs):
        theta = box.clip(box.denorm(np.asarray(z, np.float64)))
        m = rollout_primitive(snap, tuple(np.asarray(theta, np.float64)), cfg)
        deliv = bool(delivery_success(m, cfg))
        n_deliver += int(deliv)
        if deliv and is_motion_compatible(m, limits):
            accepted.append(AcceptableTheta(
                theta=np.asarray(theta, np.float64), theta_norm=box.norm(theta),
                dtz_end_mm=round(m["dtz_end"] * 1000, 2), terminal_coin_speed=round(m["terminal_coin_speed"], 4),
                peak_qdot=round(m["peak_qdot"], 4), peak_coin_speed=round(m["peak_coin_speed"], 4)))
        if progress is not None and (i + 1) % max(1, len(zs) // 10) == 0:
            progress(i + 1, len(zs), len(accepted))
    n_total = len(zs)
    return {"n_sampled": n_total, "n_delivering": n_deliver, "n_accepted": len(accepted),
            "acceptance_rate": round(len(accepted) / max(1, n_total), 4),
            "delivery_rate": round(n_deliver / max(1, n_total), 4), "accepted": accepted}


def _union_find(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def cluster_basins(thetas_norm: np.ndarray, *, link_tol: float) -> dict[str, Any]:
    """Single-linkage connected-components clustering in normalised θ-space: two accepted θ are in the same basin if a
    chain of accepted θ links them with every hop < ``link_tol``. Returns basin labels, count, per-basin size/centroid,
    the minimum INTER-basin centroid distance, and the maximum INTRA-basin nearest-neighbour hop (the separation the
    verdict reads). # Postconditions: labels ∈ [0, n_basins); n_basins ≥ 1 when n ≥ 1."""
    X = np.asarray(thetas_norm, np.float64)
    n = len(X)
    if n == 0:
        return {"n_points": 0, "n_basins": 0, "labels": [], "basin_sizes": [], "centroids": [],
                "min_inter_basin_dist": None, "max_intra_nn_hop": None, "link_tol": link_tol}
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] < link_tol]
    roots = _union_find(n, edges)
    uniq = sorted(set(roots))
    remap = {r: k for k, r in enumerate(uniq)}
    labels = [remap[r] for r in roots]
    nb = len(uniq)
    centroids = [X[[i for i in range(n) if labels[i] == b]].mean(0) for b in range(nb)]
    sizes = [int(sum(1 for lb in labels if lb == b)) for b in range(nb)]
    # min inter-basin centroid distance
    inter = None
    if nb >= 2:
        cds = [float(np.linalg.norm(centroids[a] - centroids[b])) for a in range(nb) for b in range(a + 1, nb)]
        inter = round(min(cds), 4)
    # max intra-basin nearest-neighbour hop (how tight each basin is)
    intra_hops = []
    for b in range(nb):
        idx = [i for i in range(n) if labels[i] == b]
        if len(idx) >= 2:
            sub = D[np.ix_(idx, idx)]
            np.fill_diagonal(sub, np.inf)
            intra_hops.append(float(sub.min(1).max()))
    max_intra = round(max(intra_hops), 4) if intra_hops else None
    return {"n_points": n, "n_basins": nb, "labels": labels, "basin_sizes": sizes,
            "centroids": [c.tolist() for c in centroids], "min_inter_basin_dist": inter,
            "max_intra_nn_hop": max_intra, "link_tol": link_tol}


def centroid_delivers(snap: CradleSnapshot, accepted: list[AcceptableTheta], cfg: Any = DELIVERY_CFG,
                      limits: MotionLimits = MotionLimits()) -> dict[str, Any]:
    """Roll the acceptable-set CENTROID (the normalised-space mean — what an MSE regressor over these targets converges to)
    through the frozen option. If the centroid is NON-delivering while its members deliver, that is direct evidence the
    single-θ regressor's failure is AVERAGING (mechanism A). # Postconditions: 'delivers' is the frozen monitor verdict on
    the centroid θ. Returns {} if the set is empty."""
    if not accepted:
        return {}
    box = ThetaBox()
    z_mean = np.mean([a.theta_norm for a in accepted], axis=0)
    theta = box.clip(box.denorm(z_mean))
    m = rollout_primitive(snap, tuple(np.asarray(theta, np.float64)), cfg)
    return {"centroid_theta": [round(float(x), 5) for x in theta],
            "delivers": bool(delivery_success(m, cfg)), "motion_ok": is_motion_compatible(m, limits),
            "dtz_end_mm": round(m["dtz_end"] * 1000, 2), "k6_max_dwell": int(m["k6_max_dwell"]),
            "terminal_coin_speed": round(m["terminal_coin_speed"], 4)}


def assign_to_basins(query_norm: np.ndarray, centroids: list, *, link_tol: float) -> dict[str, Any]:
    """Assign a query θ (e.g. a held-out teacher θ, or the failing actor θ₀) to the nearest pooled basin, or mark it an
    ORPHAN if it is farther than ``link_tol`` from every basin centroid. An orphan held-out delivering θ means the
    delivering region is OUT-OF-DISTRIBUTION for the dev basins (a K-head trained on dev could not propose it →
    representational, not modality). # Postconditions: nearest_basin ∈ [0,K) or None (orphan)."""
    if not centroids:
        return {"nearest_basin": None, "dist": None, "orphan": True}
    C = np.asarray(centroids, np.float64)
    d = np.linalg.norm(C - np.asarray(query_norm, np.float64)[None, :], axis=1)
    j = int(np.argmin(d))
    return {"nearest_basin": j, "dist": round(float(d[j]), 4), "orphan": bool(d[j] >= link_tol)}


def multimodality_verdict(per_state: dict[str, Any], pooled_clusters: dict[str, Any],
                          heldout_assignment: dict[str, Any]) -> dict[str, Any]:
    """The discriminating verdict. MULTIMODAL_BASINS_PRESENT iff there is genuine mode structure a K-head could exploit:
    ≥2 well-separated pooled basins (min inter-basin dist > max intra-basin hop), OR ≥1 dev state whose acceptable-set
    centroid is NON-delivering (averaging is demonstrably harmful). Otherwise SINGLE_CONNECTED_CLUSTER — the acceptable set
    is one blob the MSE centre already lands in, so the blocker is REPRESENTATION, not modality. The held-out overlay is
    reported as corroboration: a held-out delivering θ that is an ORPHAN (outside every dev basin) warns that even a
    perfect K-head, trained on dev, could not propose it (OOD). # Postconditions: exactly one verdict string."""
    nb = pooled_clusters.get("n_basins", 0)
    inter = pooled_clusters.get("min_inter_basin_dist")
    intra = pooled_clusters.get("max_intra_nn_hop")
    well_separated = bool(nb >= 2 and inter is not None and (intra is None or inter > intra))
    centroid_nondeliver = [s for s, v in per_state.items() if v.get("centroid", {}).get("delivers") is False]
    any_centroid_fails = len(centroid_nondeliver) > 0
    multimodal = bool(well_separated or any_centroid_fails)
    heldout_orphans = [t for t, a in heldout_assignment.items() if a.get("orphan")]
    return {"verdict": "MULTIMODAL_BASINS_PRESENT" if multimodal else "SINGLE_CONNECTED_CLUSTER",
            "modality_is_blocker": multimodal,
            "blocker_if_not": None if multimodal else "REPRESENTATION_NOT_PROPOSAL_MODALITY_IS_BLOCKER",
            "pooled_n_basins": nb, "pooled_well_separated": well_separated,
            "min_inter_basin_dist": inter, "max_intra_nn_hop": intra,
            "states_with_nondelivering_centroid": centroid_nondeliver,
            "held_out_orphans_outside_dev_basins": heldout_orphans,
            "held_out_ood_warning": bool(heldout_orphans),
            "justifies_k_head": multimodal, "authorises_multimodal_model": multimodal}
