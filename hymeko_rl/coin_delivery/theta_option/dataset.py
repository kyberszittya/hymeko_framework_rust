"""STAGE 2 — the structured CAUSAL θ dataset (input = decision-time state; target = proposal θ_0).

Every feature is read at the option-initiation frozen handoff (t=0, BEFORE any option action) or from a θ-independent
pre-decision probe — so nothing depends on the option outcome (no future leakage, no K6 as input, no θ_exec as a label).
The target is the proposal centre θ_0 (the Bellman action), stored normalised in the frozen θ box.

Splits (frozen): train = development teacher data (s1,s3 delivering θ); validation = held-back development delivering θ;
evaluation = the frozen 4-state panel canonical θ (s1,s3 + held-out s4,s7). Held-out labels are recorded in the eval
split ONLY and never used for fitting or normalisation. Normalisation uses fixed physical scales (a 2-point dev std is
degenerate) recorded in the machine-readable contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, _coin_xy
from hymeko_rl.env.governed_arm import pd_governed_torque
from hymeko_rl.env.motion_contract import govern_torque
from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox

# ── fixed physical normalisation scales per feature GROUP (a 2-point dev std is degenerate; these are honest, in-contract) ──
NORM_SCALES: dict[str, float] = {
    "dtz": 0.20, "e_par": 1.0, "coin_xy": 0.30, "coin_vel": 0.50, "straddle": 1.0, "fn": 5.0,
    "normal": 1.0, "xc_rel": 0.05, "q": 3.0, "qdot": 3.0, "prev_tau": 2.0, "slew_head": 0.30, "saturated": 1.0,
}
HISTORY_K = 8                                        # steps of the θ-independent passive-hold causal probe
HIST_FEATURES = ("dtz", "coin_speed", "fn_left", "fn_right", "straddle", "qdot_max")


def _tau_slew_step(snap: CradleSnapshot) -> float:
    return float(snap.stack.tau_rate * snap.stack.control_dt)


def structured_features(snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """The causal structured state at the frozen handoff (t=0), grouped by entity. Read-only (a fresh branch is inspected,
    never stepped). # Postconditions: every value is finite; nothing depends on the option outcome (fully causal)."""
    rl = snap.branch()
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    e_par, dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(e_par, np.float64)[:2]
    coin_xy = _coin_xy(rl)
    coin_vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    con = primary_fingertip_contacts(rl)
    c = coin_xy
    lft, rgt = con["left"], con["right"]
    n_l = lft["n"] if lft is not None else np.zeros(2)
    n_r = rgt["n"] if rgt is not None else np.zeros(2)
    xc_l = (lft["x_c"] - c) if lft is not None else np.zeros(2)
    xc_r = (rgt["x_c"] - c) if rgt is not None else np.zeros(2)
    d = rl.inner.data
    slew = _tau_slew_step(snap)
    return {
        "dtz": np.array([float(dtz)]), "e_par": e_par, "coin_xy": coin_xy, "coin_vel": coin_vel,
        "straddle": np.array([float(snap.straddle0)]), "fn": np.array(snap.fn0, np.float64),
        "normal": np.concatenate([n_l, n_r]), "xc_rel": np.concatenate([xc_l, xc_r]),
        "q": np.asarray(d.qpos[:4], np.float64), "qdot": np.asarray(d.qvel[:4], np.float64),
        "prev_tau": np.asarray(snap.prev_tau, np.float64),
        "slew_head": np.concatenate([np.minimum(snap.hi - snap.prev_tau, slew) / slew,
                                     np.minimum(snap.prev_tau - snap.lo, slew) / slew]),
        "saturated": np.asarray(snap.arm_saturated, np.float64),
    }


FEATURE_ORDER = ("dtz", "e_par", "coin_xy", "coin_vel", "straddle", "fn", "normal", "xc_rel",
                 "q", "qdot", "prev_tau", "slew_head", "saturated")


def _scale_of(group: str) -> float:
    return NORM_SCALES["slew_head"] if group == "slew_head" else NORM_SCALES.get(group, 1.0)


def flatten_features(feat: dict[str, np.ndarray], *, normalise: bool = True) -> np.ndarray:
    """Concatenate the grouped features in the FROZEN order, optionally applying the fixed per-group physical scale.
    slew_head is already scale-normalised in `structured_features`. # Postconditions: 1-D float32; deterministic."""
    parts = []
    for g in FEATURE_ORDER:
        v = np.asarray(feat[g], np.float64)
        parts.append(v / _scale_of(g) if (normalise and g != "slew_head") else v)
    return np.concatenate(parts).astype(np.float32)


def feature_names() -> list[str]:
    sizes = {"dtz": 1, "e_par": 2, "coin_xy": 2, "coin_vel": 2, "straddle": 1, "fn": 2, "normal": 4, "xc_rel": 4,
             "q": 4, "qdot": 4, "prev_tau": 4, "slew_head": 8, "saturated": 4}
    return [f"{g}[{i}]" for g in FEATURE_ORDER for i in range(sizes[g])]


def causal_history(snap: CradleSnapshot, k: int = HISTORY_K) -> np.ndarray:
    """A θ-INDEPENDENT causal probe: roll ``k`` steps of the passive POSITION hold (pd servo at q_hold — the null 'do
    nothing' action) from the handoff and record per-step (dtz, coin_speed, fn_L, fn_R, straddle, qdot_max). Pre-decision
    and deterministic (no option outcome, no θ) — a legitimate decision-time sensor stream for the temporal encoders.
    # Postconditions: shape (k, 6) float32."""
    rl = snap.branch()
    d = rl.inner.data
    prev_tau = snap.prev_tau.copy()

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    rows = []
    try:
        for _ in range(int(k)):
            a = pd_governed_torque(d.qpos[:4].copy(), d.qvel[:4].copy(), snap.q_hold, snap.stack, prev_tau, snap.lo, snap.hi)
            prev_tau = a
            step_ablation(rl, np.asarray(a, np.float32), "A")
            _u, dtz = rl.inner.direction_to_zone()
            con = primary_fingertip_contacts(rl)
            fn_l = float(con["left"]["fn"]) if con["left"] is not None else 0.0
            fn_r = float(con["right"]["fn"]) if con["right"] is not None else 0.0
            straddle = float(con["left"]["n"] @ con["right"]["n"]) if (con["left"] is not None and con["right"] is not None) else 0.0
            rows.append([float(dtz), _coin_speed(rl), fn_l, fn_r, straddle, float(np.max(np.abs(d.qvel[:4])))])
    finally:
        mujoco.set_mjcb_control(None)
    return np.asarray(rows, np.float32)


@dataclass
class DatasetRow:
    """One (state → θ_0) example. ``features`` normalised; ``history`` the causal probe; ``theta`` legal, ``theta_norm``
    in [-1,1]; ``split`` ∈ {train, val, eval}; ``kind`` ∈ {canonical, basin_delivering}; ``eval_only`` marks held-out."""

    tag: str
    split: str
    kind: str
    features: np.ndarray
    history: np.ndarray
    theta: np.ndarray
    theta_norm: np.ndarray
    eval_only: bool
    k6_delivered: bool


@dataclass
class ThetaDataset:
    rows: list[DatasetRow] = field(default_factory=list)
    contract: dict[str, Any] = field(default_factory=dict)

    def subset(self, split: str) -> list[DatasetRow]:
        return [r for r in self.rows if r.split == split]

    def xy(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        rs = self.subset(split)
        return (np.asarray([r.features for r in rs], np.float32),
                np.asarray([r.theta_norm for r in rs], np.float32))

    def history_of(self, split: str) -> np.ndarray:
        return np.asarray([r.history for r in self.subset(split)], np.float32)


VAL_EVERY = 4                                        # every VAL_EVERY-th delivering-basin candidate is held back for validation


def _feature_bundle(snap: CradleSnapshot) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    feat = structured_features(snap)
    return flatten_features(feat, normalise=True), causal_history(snap), feat


def build_dataset(bank: dict[str, Any], harness: Any = None) -> ThetaDataset:
    """Assemble the structured causal θ dataset from the frozen teacher bank. Re-acquires each cradle's snapshot
    (verifying its post-release hash matches the bank — cross-stage replay consistency), extracts the causal features +
    history ONCE per snapshot, and attaches θ labels: dev canonical + delivering-basin θ (train/val), held-out canonical
    (eval-only). # Preconditions: `bank` from a non-smoke teacher_bank run. # Postconditions: no held-out row has
    split in {train,val}; every θ label is a legal, delivering θ from the bank."""
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    harness = harness or load_harness()
    box = ThetaBox()
    rows: list[DatasetRow] = []
    hash_checks = []
    for e in bank["states"]:
        if "canonical_theta_vec" not in e:
            continue
        tag, split_grp = e["tag"], e["split"]
        eval_only = bool(split_grp == "held_out")
        snap, _meta = acquire_snapshot(harness, e["seed"])
        if snap is None:
            raise RuntimeError(f"dataset build: could not re-acquire snapshot for {tag}")
        hash_checks.append({"tag": tag, "bank_hash": e["snapshot"]["post_release_hash"],
                            "reacquired_hash": snap.post_release_hash,
                            "match": bool(snap.post_release_hash == e["snapshot"]["post_release_hash"])})
        fvec, hist, _feat = _feature_bundle(snap)
        canon = np.asarray(e["canonical_theta_vec"], np.float64)

        def _row(theta: np.ndarray, split: str, kind: str, k6: bool) -> DatasetRow:
            return DatasetRow(tag=tag, split=split, kind=kind, features=fvec.copy(), history=hist.copy(),
                              theta=box.clip(theta), theta_norm=box.norm(theta), eval_only=eval_only, k6_delivered=k6)

        # eval-panel label (all four states): the canonical delivering θ
        rows.append(_row(canon, "eval", "canonical", True))
        if eval_only:
            continue                                 # held-out: eval ONLY, never train/val (no augmentation)
        rows.append(_row(canon, "train", "canonical", True))     # dev canonical → train
        deliver = [c for c in e.get("basin_candidates", []) if c["kind"] == "delivering"]
        for i, c in enumerate(deliver):              # dev delivering basin → train/val (deterministic holdout)
            split = "val" if (i % VAL_EVERY == 0) else "train"
            rows.append(_row(np.asarray(c["theta"], np.float64), split, "basin_delivering", True))

    n_by_tag_split: dict[str, int] = {}
    for r in rows:
        key = f"{r.tag}:{r.split}"
        n_by_tag_split[key] = n_by_tag_split.get(key, 0) + 1
    contract: dict[str, Any] = {
        "contract": "COIN_6D_THETA_DATASET_V1", "base_commit": "a3459629", "teacher_bank": bank.get("contract"),
        "target": "proposal centre θ_0 (Bellman action), normalised in the frozen θ box",
        "feature_order": list(FEATURE_ORDER), "feature_dim": int(rows[0].features.shape[0]) if rows else 0,
        "feature_names": feature_names(), "normalisation": "fixed physical per-group scales (2-point dev std is degenerate)",
        "norm_scales": NORM_SCALES, "history": {"k": HISTORY_K, "features": list(HIST_FEATURES),
                                                "probe": "θ-independent passive position-hold from the handoff (causal, pre-decision)"},
        "splits": {"train": "development canonical + delivering-basin θ",
                   "val": f"held-back development delivering-basin θ (every {VAL_EVERY}th)",
                   "eval": "frozen 4-state panel canonical θ (s1,s3 + held-out s4,s7)"},
        "leakage_guards": ["no K6 result in inputs", "no θ_exec as a label", "no post-option measurement",
                           "held-out labels are eval-only", "features read at t=0 or from a θ-independent pre-decision probe"],
        "split_counts": {s: sum(1 for r in rows if r.split == s) for s in ("train", "val", "eval")},
        "n_by_tag_split": n_by_tag_split,
        "snapshot_hash_checks": hash_checks,
        "all_hashes_match": bool(all(h["match"] for h in hash_checks)),
    }
    return ThetaDataset(rows=rows, contract=contract)


def save_npz(ds: ThetaDataset, path: str) -> None:
    """Cache the (snapshot-free) dataset rows + contract to ``path`` so the offline BC fit reloads in <1s instead of
    re-acquiring physics. Physics stages (update-0, RL) still re-acquire snapshots (mujoco state is not serialisable)."""
    import json as _json
    n = len(ds.rows)
    np.savez(path, features=np.asarray([r.features for r in ds.rows], np.float32),
             history=np.asarray([r.history for r in ds.rows], np.float32),
             theta=np.asarray([r.theta for r in ds.rows], np.float32),
             theta_norm=np.asarray([r.theta_norm for r in ds.rows], np.float32),
             tag=np.asarray([r.tag for r in ds.rows]), split=np.asarray([r.split for r in ds.rows]),
             kind=np.asarray([r.kind for r in ds.rows]),
             eval_only=np.asarray([r.eval_only for r in ds.rows]),
             k6=np.asarray([r.k6_delivered for r in ds.rows]),
             contract=np.asarray([_json.dumps(ds.contract)]), n=np.asarray([n]))


def load_npz(path: str) -> ThetaDataset:
    """Reconstruct a snapshot-free `ThetaDataset` from `save_npz` (features/history/θ/splits) for the offline BC fit."""
    import json as _json
    z = np.load(path, allow_pickle=True)
    rows = [DatasetRow(tag=str(z["tag"][i]), split=str(z["split"][i]), kind=str(z["kind"][i]),
                       features=z["features"][i], history=z["history"][i], theta=z["theta"][i],
                       theta_norm=z["theta_norm"][i], eval_only=bool(z["eval_only"][i]), k6_delivered=bool(z["k6"][i]))
            for i in range(int(z["n"][0]))]
    return ThetaDataset(rows=rows, contract=_json.loads(str(z["contract"][0])))


def contract_summary(ds: ThetaDataset) -> dict[str, Any]:
    """The machine-readable dataset contract (dataset_contract.json) + a per-split preview of the (tag, kind, θ) rows —
    small enough to store in full because the dataset is tiny."""
    preview = [{"tag": r.tag, "split": r.split, "kind": r.kind, "eval_only": r.eval_only,
                "theta": [round(float(x), 5) for x in r.theta]} for r in ds.rows]
    return {**ds.contract, "rows": preview,
            "split_isolation_ok": bool(all(not (r.eval_only and r.split in ("train", "val")) for r in ds.rows))}
