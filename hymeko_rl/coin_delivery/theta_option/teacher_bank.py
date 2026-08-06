"""STAGE 1 — reproduce and freeze the 6-D torque-θ teacher bank.

Re-acquire each CERTIFIED straddle cradle (s1,s3 development · s4,s7 held-out) at the frozen seeds, reproduce the
canonical delivering θ with the frozen deterministic CEM (`forward_displacement.cem_search` under `DELIVERY_CFG`), and
record full per-trajectory provenance (snapshot hashes, θ, K6 dwell trace, motion-contract metrics, a non-behavioural
per-step physical trace). Dev states additionally get delivering/near-delivering **basin** candidates (label
augmentation from s1,s3 ONLY); held-out states get exactly the canonical θ, retained for frozen evaluation only.

Nothing here alters physics, the K6 monitor, the split, or the successful trajectories — it reproduces and records them.
The per-step torque recorded is the post-governor `data.ctrl` (the true torque MuJoCo integrated); the frozen K6 verdict
comes only from the authoritative `rollout_primitive`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, InvalidCradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import (
    _coin_speed, _coin_xy, cem_search, delivery_success, rollout_primitive, score as fd_score)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, DIM, THETA_NAMES, ThetaBox
from hymeko_rl.experiments.bv_identification_benchmark import (
    CERTIFIED_SEEDS, _load_frozen, acquire_certified_straddle)  # noqa: F401  (CERTIFIED_SEEDS re-exported for the harness)
from hymeko_rl.experiments.video_coin_variants import _setup

STATE_TAG = {0: "s1", 1: "s3", 2: "s4", 3: "s7"}
DEV_IDS, HELDOUT_IDS = (0, 1), (2, 3)
REPLAY_TOL = 1e-6                                     # frozen-tolerance replay: |trace K6 dwell − authoritative| within this


def load_harness() -> tuple[Any, ...]:
    """Load the frozen V2/V4 stack + acquisition harness (single source: the bv/horizon benchmarks)."""
    stack, cfg_coop, v2, mu = _load_frozen()
    pi0, base, forbidden = _setup()
    return pi0, base, forbidden, v2, stack, cfg_coop, mu


def acquire_snapshot(harness: tuple[Any, ...], seed: int, *, tries: int = 3, object_spec: Any = None) -> tuple[Any, ...]:
    """Acquire the certified straddle at ``seed`` and build the free-coin `CradleSnapshot` (the exact frozen handoff).
    Returns (snap|None, acquire_meta). Deterministic given the frozen harness + seed. ``object_spec`` (R11.7A U6)
    selects the manipuland; None ⇒ the HyMeKo scene's reference coin (the frozen O0 behaviour)."""
    pi0, base, forbidden, v2, stack, cfg_coop, mu = harness
    rl, saved, handoff, meta = acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg_coop, seed,
                                                          tries=tries, object_spec=object_spec)
    if rl is None:
        return None, meta
    from hymeko_rl.coin_delivery.contact_velocity import BvConfig
    try:
        snap = CradleSnapshot(rl, stack, saved, handoff["prev_tau"], handoff["q_target"],
                              release_coin=True, coast_mu=mu, cfg=BvConfig())
    except InvalidCradleSnapshot as e:
        return None, {**meta, "invalid_snapshot": str(e)}
    return snap, meta


FROZEN_SEEDS = (14250, 14750, 15000, 15750)          # the 4 already-certified cradles (s1,s3 dev · s4,s7 held-out)


def enumerate_seeds(n: int) -> list[int]:
    """The canonical cradle-embodiment enumeration ``seed = 14000 + 250·si`` (from bimanual_cradle_embodiment_audit),
    extended to ``si ∈ [0,n)``. Each seed_lo drives a distinct scene reconstruction (a candidate straddle cradle)."""
    return [14000 + 250 * si for si in range(int(n))]


def scout_certified_cradles(harness: tuple[Any, ...], seeds: list[int],
                            progress: "Any | None" = None) -> list[dict[str, Any]]:
    """CERTIFICATION SCOUT (read-only): for each seed, attempt to acquire a certified straddle cradle (both-contact ∧
    internal-force-feasible ∧ straddle) and build its free-coin snapshot. Records certification, the straddle n_dot, and
    the post-release state hash (a uniqueness signature) — NO delivery CEM (cheap). ``progress(row, dt)`` (optional) is
    called live per cradle so the sweep is never blind. Bounds the achievable N of UNIQUE development cradles for the
    expansion. # Postconditions: one row per seed; a certified row has a snapshot hash."""
    import time as _time
    rows: list[dict[str, Any]] = []
    for si, seed in enumerate(seeds):
        t0 = _time.time()
        snap, meta = acquire_snapshot(harness, seed)
        certified = snap is not None
        row = {"si": si, "seed": seed, "certified": bool(certified),
               "acquire_certified": bool(meta.get("certified", False)),
               "invalid_snapshot": meta.get("invalid_snapshot"),
               "n_dot": (round(float(meta["n_dot"]), 4) if meta.get("n_dot") is not None else None),
               "axis": meta.get("axis"),
               "post_release_hash": (snap.post_release_hash if snap else None),
               "straddle0": (round(float(snap.straddle0), 4) if snap else None),
               "fn0": ([round(float(x), 4) for x in snap.fn0] if snap else None),
               "is_frozen_seed": bool(seed in FROZEN_SEEDS)}
        rows.append(row)
        if progress is not None:
            progress(row, _time.time() - t0)
    return rows


def _phase_of(t: int, theta: Any) -> str:
    ramp, rel = float(theta[3]), float(theta[4])
    return "PUSH" if t <= ramp else ("BRAKE" if t <= rel else "RELEASE")


def trace_teacher(snap: CradleSnapshot, theta: Any, cfg: Any = DELIVERY_CFG) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Roll the frozen physical option for ``theta`` via the AUTHORITATIVE `rollout_primitive`, capturing a NON-behavioural
    per-step physical trace through a frame_hook (q, qdot, coin pose/velocity, post-governor ctrl, contact Fn/straddle,
    dtz, K6 dwell, phase). The returned metrics ARE `rollout_primitive`'s (frozen K6 verdict); the hook only observes.
    # Postconditions: `len(trace) == cfg.horizon` on a completed rollout; no pin/teleport/coin edit (all inside the frozen
    rollout)."""
    trace: list[dict[str, Any]] = []
    dwell = {"k6": 0}

    def hook(rl: Any, t: int) -> None:
        d = rl.inner.data
        con = primary_fingertip_contacts(rl)
        fn_l = float(con["left"]["fn"]) if con["left"] is not None else 0.0
        fn_r = float(con["right"]["fn"]) if con["right"] is not None else 0.0
        both = con["left"] is not None and con["right"] is not None
        straddle = (float(con["left"]["n"] @ con["right"]["n"]) if both else None)
        _u, dtz = rl.inner.direction_to_zone()
        sp = _coin_speed(rl)
        from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, SETTLE_VEL
        dwell["k6"] = dwell["k6"] + 1 if (dtz <= CENTER_TOL and sp < SETTLE_VEL) else 0
        trace.append({"t": int(t), "phase": _phase_of(t, theta),
                      "q": [round(float(x), 5) for x in d.qpos[:4]],
                      "qdot_max": round(float(np.max(np.abs(d.qvel[:4]))), 5),
                      "coin_xy": [round(float(x), 5) for x in _coin_xy(rl)],
                      "coin_speed": round(sp, 5), "dtz": round(float(dtz), 5),
                      "ctrl_post_gov": [round(float(x), 5) for x in d.ctrl[:4]],
                      "fn_left": round(fn_l, 4), "fn_right": round(fn_r, 4),
                      "both_contact": bool(both), "straddle": (round(straddle, 4) if straddle is not None else None),
                      "k6_dwell": int(dwell["k6"])})

    m = rollout_primitive(snap, tuple(np.asarray(theta, np.float64)), cfg, frame_hook=hook)
    return m, trace


def _outcome_summary(m: dict[str, Any], cfg: Any = DELIVERY_CFG) -> dict[str, Any]:
    return {"k6_delivered": bool(m["k6_delivered"]), "k6_max_dwell": int(m["k6_max_dwell"]),
            "delivery_success": bool(delivery_success(m, cfg)), "touched": bool(m["touched"]),
            "dtz_start_mm": round(m["dtz_start"] * 1000, 2), "dtz_end_mm": round(m["dtz_end"] * 1000, 2),
            "gap_closed": round(m["gap_closed"], 3), "forward_mm": round(m["forward"] * 1000, 2),
            "terminal_coin_speed": round(m["terminal_coin_speed"], 4),
            "peak_qdot": round(m["peak_qdot"], 4), "peak_coin_speed": round(m["peak_coin_speed"], 4),
            "contact_lost_steps": int(m["contact_lost_steps"]), "score": round(float(fd_score(m, cfg)), 5)}


def snapshot_provenance(snap: CradleSnapshot) -> dict[str, Any]:
    """The frozen-handoff provenance shared by every trajectory from this cradle (hashes, controller state, contact)."""
    return {"pre_release_hash": snap.pre_release_hash, "post_release_hash": snap.post_release_hash,
            "operating_point": snap.operating_point, "prev_tau": [round(float(x), 5) for x in snap.prev_tau],
            "q_hold": [round(float(x), 5) for x in snap.q_hold], "straddle0": round(float(snap.straddle0), 4),
            "fn0": [round(float(x), 4) for x in snap.fn0], "arm_saturated": list(snap.arm_saturated),
            "actuator_lo": [round(float(x), 4) for x in snap.lo], "actuator_hi": [round(float(x), 4) for x in snap.hi],
            "recert_admissible": bool(snap.recert["admissible"])}


_STORE_DECIMALS = 6                                   # canonical/basin θ storage precision (robust to s3-class fragility)


def _round_theta(theta: Any) -> np.ndarray:
    return np.round(np.asarray(theta, np.float64), _STORE_DECIMALS)


def _verify_deliver(snap: CradleSnapshot, theta: Any, cfg: Any = DELIVERY_CFG) -> tuple[bool, dict[str, Any]]:
    """Roll ``theta`` at ITS GIVEN (stored) precision and return (delivers-K6-with-motion-contract, metrics). The
    delivery verdict is the frozen monitor's — this is the only reproduction test that matters for a stored θ."""
    m = rollout_primitive(snap, tuple(np.asarray(theta, np.float64)), cfg)
    return bool(delivery_success(m, cfg)), m


def sample_dev_basin(snap: CradleSnapshot, canonical_theta: Any, cfg: Any = DELIVERY_CFG, *,
                     n_jitter: int = 48, jitter_std: float = 0.18, seed: int = 0) -> list[dict[str, Any]]:
    """Dev-ONLY label augmentation: sample θ around the canonical solution (jitter in normalised z-space), ROUND to the
    storage precision, roll THAT rounded θ through the frozen option, and KEEP the ones that deliver (K6) or nearly
    deliver (dtz_end ≤ 40 mm, touched) AT STORED PRECISION — a cheap deterministic, reproducible map of the delivering
    BASIN at a development cradle. Never applied to held-out. Verifying the rounded θ (not the full-precision draw) makes
    every kept candidate reproduce K6 exactly."""
    box = ThetaBox()
    rng = np.random.default_rng(seed)
    z0 = box.norm(np.asarray(canonical_theta, np.float64))
    kept: list[dict[str, Any]] = []
    for _ in range(int(n_jitter)):
        theta = _round_theta(box.denorm(z0 + jitter_std * rng.standard_normal(DIM)))
        deliv, m = _verify_deliver(snap, theta, cfg)
        near = bool(m["dtz_end"] <= 0.040 and m["touched"])
        if deliv or near:
            kept.append({"theta": [round(float(x), _STORE_DECIMALS) for x in theta],
                         "kind": "delivering" if deliv else "near", "outcome": _outcome_summary(m, cfg)})
    return kept


def robust_canonical(snap: CradleSnapshot, cem_best_theta: Any, basin: list[dict[str, Any]], cfg: Any = DELIVERY_CFG) -> dict[str, Any]:
    """Select the canonical delivering θ that REPRODUCES frozen K6 at stored precision. The candidate pool is the rounded
    CEM optimum ∪ (dev-only) the delivering basin candidates; the canonical is the highest frozen-score θ that delivers at
    stored precision. For a robust cradle (s1,s4,s7) the CEM optimum wins; for a fragile one (s3, whose knife-edge optimum
    does not survive quantisation) a robust delivering basin candidate is selected — a dev-side provenance/replay repair
    using ONLY that cradle's own frozen snapshot + K6 monitor (no cross-state / no held-out label). Returns the chosen θ,
    its metrics, the source, and the fragility diagnostic."""
    cem_round = _round_theta(cem_best_theta)
    cem_deliv, cem_m = _verify_deliver(snap, cem_round, cfg)
    cem_vec = [round(float(x), _STORE_DECIMALS) for x in cem_round]
    if cem_deliv:                                     # faithful: the frozen CEM optimum reproduces K6 at stored precision
        return {"theta": cem_round, "metrics": cem_m, "source": "cem_best", "cem_best_reproduces": True,
                "cem_best_theta": cem_vec}
    # Fragile CEM optimum (does not survive quantisation) — repair from the delivering basin (dev only; empty for held-out)
    scored = []
    for c in basin:
        if c["kind"] != "delivering":
            continue
        th = _round_theta(c["theta"])
        deliv, m = _verify_deliver(snap, th, cfg)
        if deliv:
            scored.append((float(fd_score(m, cfg)), th, m))
    if scored:
        scored.sort(key=lambda z: z[0], reverse=True)
        _s, th, m = scored[0]
        return {"theta": th, "metrics": m, "source": "basin_repair", "cem_best_reproduces": False,
                "cem_best_theta": cem_vec}
    return {"theta": cem_round, "metrics": cem_m, "source": "cem_best_NONDELIVERING", "cem_best_reproduces": False,
            "cem_best_theta": cem_vec}


def deliver_on_snapshot(snap: CradleSnapshot, cfg: Any = DELIVERY_CFG, *, basin_seed: int = 0) -> dict[str, Any]:
    """Frozen-CEM delivery test on an already-acquired snapshot (no re-acquisition). Runs the SAME frozen search
    (cem_search) + fragility repair (basin → robust_canonical) as the teacher bank, and reports whether the cradle is
    K6-DELIVERABLE under the frozen 6-D option — with the canonical θ if so. All per-cradle params (CEM budget, seed, θ
    bounds, K6, motion contract) are frozen. # Postconditions: `deliverable` is the frozen monitor's verdict on the
    stored (rounded) canonical θ."""
    res = cem_search(snap, cfg)
    cem_best = res["best_theta"]
    if cem_best is None:
        return {"deliverable": False, "reason": "cem_no_feasible_theta", "canonical_theta": None}
    basin = sample_dev_basin(snap, cem_best, cfg, seed=basin_seed)
    rc = robust_canonical(snap, cem_best, basin, cfg)
    m = rc["metrics"]
    deliverable = bool(delivery_success(m, cfg))
    return {"deliverable": deliverable, "canonical_theta": [round(float(x), 6) for x in rc["theta"]],
            "canonical_source": rc["source"], "cem_optimum_reproduces": bool(rc["cem_best_reproduces"]),
            "outcome": _outcome_summary(m, cfg), "n_basin_delivering": int(sum(1 for c in basin if c["kind"] == "delivering"))}


def reproduce_state(harness: tuple[Any, ...], idx: int, seed: int, cfg: Any = DELIVERY_CFG, *, augment: bool = False,
                    tag: "str | None" = None, split: "str | None" = None, basin_seed: "int | None" = None) -> dict[str, Any]:
    """Reproduce one cradle's canonical delivering θ (frozen CEM), verify frozen-tolerance replay, record provenance +
    per-step trace, and (dev only, ``augment``) harvest delivering-basin candidates. Returns the bank entry.

    For the frozen 4-state panel, ``tag``/``split`` are derived from ``idx`` (DEV_IDS/HELDOUT_IDS). For dev-cradle
    EXPANSION (the coverage curve) pass an explicit ``tag`` + ``split="development"`` to reproduce a NEW development cradle
    with the SAME frozen machinery (identical CEM search + basin augmentation); ``basin_seed`` defaults to a deterministic
    function of ``idx``. # Preconditions: split ∈ {development, held_out}. # Postconditions: a dev entry carries
    ``canonical_theta_vec`` + (``augment``) ``basin_candidates``; defaults reproduce the frozen-panel behaviour exactly."""
    tag = tag if tag is not None else STATE_TAG.get(idx, f"s{idx}")
    split = split if split is not None else ("development" if idx in DEV_IDS else "held_out")
    is_dev = split == "development"
    snap, meta = acquire_snapshot(harness, seed)
    entry: dict[str, Any] = {"state": idx, "tag": tag, "seed": seed, "split": split, "acquire": meta}
    if snap is None:
        entry["skipped"] = "no certified straddle / invalid snapshot"
        return entry
    res = cem_search(snap, cfg)                       # authoritative frozen search (deterministic CEM seed)
    cem_best = res["best_theta"]
    if cem_best is None:
        entry["failed"] = "CEM found no feasible θ"
        entry["snapshot"] = snapshot_provenance(snap)
        return entry
    # Dev cradles harvest the delivering basin FIRST (used both for canonical robustness repair AND as label augmentation);
    # held-out cradles are NEVER refined — their canonical is the frozen CEM optimum (recorded for evaluation only).
    basin: list[dict[str, Any]] = []
    if augment and is_dev:
        basin = sample_dev_basin(snap, cem_best, cfg, seed=(basin_seed if basin_seed is not None else idx * 1000 + 7))
    rc = robust_canonical(snap, cem_best, basin, cfg)
    canon_theta = rc["theta"]
    metrics, trace = trace_teacher(snap, canon_theta, cfg)    # rich per-step trace of the chosen canonical θ
    # Frozen-tolerance replay of the STORED canonical θ: deterministic on a second roll AND delivers frozen K6.
    m2 = rollout_primitive(snap, tuple(np.asarray(canon_theta, np.float64)), cfg)
    deterministic = bool(m2["k6_delivered"] == metrics["k6_delivered"] and m2["k6_max_dwell"] == metrics["k6_max_dwell"]
                         and abs(m2["dtz_end"] - metrics["dtz_end"]) <= REPLAY_TOL)
    replay_ok = bool(deterministic and metrics["k6_delivered"] and delivery_success(metrics, cfg))
    entry.update({
        "canonical_theta": {n: round(float(v), 6) for n, v in zip(THETA_NAMES, canon_theta)},
        "canonical_theta_vec": [round(float(x), 6) for x in canon_theta],
        "canonical_source": rc["source"],               # cem_best | basin | cem_best_NONDELIVERING
        "cem_optimum_theta": rc["cem_best_theta"],       # the raw frozen CEM optimum (mobile_teacher θ)
        "cem_optimum_reproduces_at_stored_precision": bool(rc["cem_best_reproduces"]),
        "search": {"cem_seed": cfg.cem_seed, "pop": cfg.pop, "iters": cfg.iters, "elite": cfg.elite,
                   "passive_drift_mm": round(res["passive_drift"] * 1000, 3), "history": res["history"]},
        "snapshot": snapshot_provenance(snap),
        "outcome": _outcome_summary(metrics, cfg),
        "replay_ok": replay_ok, "deterministic": deterministic, "k6_delivered": bool(metrics["k6_delivered"]),
        "trace": trace,
    })
    if basin:
        entry["basin_candidates"] = basin
        entry["n_basin_delivering"] = int(sum(1 for c in basin if c["kind"] == "delivering"))
        entry["n_basin_near"] = int(sum(1 for c in basin if c["kind"] == "near"))
    return entry
