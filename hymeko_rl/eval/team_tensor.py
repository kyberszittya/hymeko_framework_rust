"""Shared cooperative team tensor z_team for the coin-toss / Galambos two-fingertip task (v8 audit).

A CANONICAL shared representation both fingertip agents + the (would-be) centralized critic should consume — coin +
target + left/right fingertip pose & velocity, the relative vectors, aperture / squeeze, the contact flags, fingertip-
vs body-attributed progress, and the temporal/cooperative signals (contact-lost-after-handoff, sustained-PUSH,
previous-step contact) that a scalar delivery reward omits. This module DEFINES the schema and audits its COVERAGE
against the existing V2 semantic tensor (a strong z_team candidate) — it does not change the reward or train anything.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hymeko_rl.eval.semantic_tensor import REL_FIELDS

# ---- the canonical shared team-tensor schema (the contract; order is load-bearing → hashed)
TEAM_FIELDS = (
    "coin_x", "coin_y", "coin_vx", "coin_vy", "target_x", "target_y",
    "lft_x", "lft_y", "lft_vx", "lft_vy", "rft_x", "rft_y", "rft_vx", "rft_vy",
    "l_to_coin_x", "l_to_coin_y", "r_to_coin_x", "r_to_coin_y",
    "coin_to_target_x", "coin_to_target_y", "mid_to_coin_x", "mid_to_coin_y",
    "aperture", "squeeze_dir_x", "squeeze_dir_y", "squeeze_mag",
    "left_contact", "right_contact", "both_contact", "arm_body_contact",
    "ft_progress", "body_progress", "contact_lost_after_handoff", "sustained_push_duration",
    "phase_APPROACH", "phase_CONTACT", "phase_PUSH", "phase_DELIVERY",
    "high_coin_y", "prev_left_contact", "prev_right_contact",
)

# ---- joint action fields (the cooperative fix: the centralized critic sees BOTH agents' actions).
# Action layout is pinned: left arm = motors [0,1], right arm = motors [2,3] (n_actions = 4).
JOINT_ACTION_FIELDS = ("aL_1", "aL_2", "aR_1", "aR_2")

# ---- canonical consumer subsets (every consumer reads ONE of these, in this exact order).
#   ACTOR : decentralized policy input — team STATE, no joint action (avoids the critic-only privileged signal).
#   CRITIC: centralized CTDE critic — team state + the JOINT action (a_L, a_R). This is the missing cooperation term.
#   VIEW  : full logging / audit view — everything.
ACTOR_FIELDS = TEAM_FIELDS
CRITIC_FIELDS = TEAM_FIELDS + JOINT_ACTION_FIELDS
VIEW_FIELDS = TEAM_FIELDS + JOINT_ACTION_FIELDS

# name -> index into the flat ACTOR/TEAM observation, so consumers read a NAMED field instead of a magic position.
_TEAM_FIELD_INDEX = {name: i for i, name in enumerate(TEAM_FIELDS)}


def field_index(name: str) -> int:
    """Index of a named field in ``TEAM_FIELDS``/``ACTOR_FIELDS`` (the flat contact-formation observation). Lets
    coin code read ``obs[field_index("mid_to_coin_x")]`` rather than a hard-coded ``obs[20]``.

    # Preconditions ``name`` is a TEAM_FIELDS field. # Raises ``KeyError`` (loud, not a silent mis-index) otherwise."""
    try:
        return _TEAM_FIELD_INDEX[name]
    except KeyError as exc:
        raise KeyError(f"{name!r} is not a TEAM_FIELDS observation field; valid names: {TEAM_FIELDS}") from exc


def field_indices(*names: str) -> list[int]:
    """The indices of several named observation fields, in order (e.g. ``field_indices('mid_to_coin_x','mid_to_coin_y')``)."""
    return [field_index(n) for n in names]

# left/right role-symmetric field pairs (for the L/R role-consistency invariant check).
_LR_PAIRS = (
    ("lft_x", "rft_x"), ("lft_y", "rft_y"), ("lft_vx", "rft_vx"), ("lft_vy", "rft_vy"),
    ("l_to_coin_x", "r_to_coin_x"), ("l_to_coin_y", "r_to_coin_y"),
    ("left_contact", "right_contact"), ("prev_left_contact", "prev_right_contact"),
    ("aL_1", "aR_1"), ("aL_2", "aR_2"),
)

# which team fields the V2 semantic tensor (per-vertex node matrix + relational block) already carries
_SEMANTIC_COVERS = {
    "coin_x", "coin_y", "target_x", "target_y", "lft_x", "lft_y", "rft_x", "rft_y",
    "l_to_coin_x", "l_to_coin_y", "r_to_coin_x", "r_to_coin_y", "coin_to_target_x", "coin_to_target_y",
    "aperture", "squeeze_dir_x", "squeeze_dir_y", "left_contact", "right_contact", "both_contact",
    "arm_body_contact", "ft_progress", "body_progress",
    "phase_APPROACH", "phase_CONTACT", "phase_PUSH", "phase_DELIVERY",
}


def team_schema_hash() -> str:
    return hashlib.md5("|".join(("TEAM", *TEAM_FIELDS)).encode()).hexdigest()


def view_schema_hash() -> str:
    """Stable field-order hash of the FULL runtime view (state + joint action). Order is load-bearing."""
    return hashlib.md5("|".join(("VIEW", *VIEW_FIELDS)).encode()).hexdigest()


def canonical_subsets() -> dict[str, tuple[str, ...]]:
    """The three canonical consumer views. A consumer MUST read exactly one of these (in order)."""
    return {"actor": ACTOR_FIELDS, "critic": CRITIC_FIELDS, "view": VIEW_FIELDS}


def _is_ordered_subsequence(sub: tuple[str, ...], full: tuple[str, ...]) -> bool:
    """True iff ``sub`` appears in ``full`` in the same relative order (no reordering, no invented fields)."""
    it = iter(full)
    return all(f in it for f in sub)


def subsets_are_canonical() -> dict[str, Any]:
    """Guard: every consumer subset is an in-order subsequence of VIEW_FIELDS — no reordering, no stray field.
    This is what "consumers read canonical subsets only" means operationally."""
    subs = canonical_subsets()
    checks = {name: _is_ordered_subsequence(fields, VIEW_FIELDS) for name, fields in subs.items()}
    # critic MUST carry the joint action (the cooperative fix); actor MUST NOT (decentralized execution).
    critic_has_action = all(f in CRITIC_FIELDS for f in JOINT_ACTION_FIELDS)
    actor_has_action = any(f in ACTOR_FIELDS for f in JOINT_ACTION_FIELDS)
    ok = all(checks.values()) and critic_has_action and not actor_has_action
    return {"per_subset": checks, "critic_sees_joint_action": critic_has_action,
            "actor_excludes_joint_action": not actor_has_action, "all_canonical": ok}


def left_right_field_pairs() -> tuple[tuple[str, str], ...]:
    """Symmetric (left, right) field pairs — swapping arm roles must swap exactly these (role consistency)."""
    return _LR_PAIRS


def coverage_vs_semantic() -> dict:
    """Which z_team fields the semantic tensor already carries vs the GAPS a cooperative critic would miss."""
    present = [f for f in TEAM_FIELDS if f in _SEMANTIC_COVERS]
    missing = [f for f in TEAM_FIELDS if f not in _SEMANTIC_COVERS]
    return {"n_fields": len(TEAM_FIELDS), "n_present_in_semantic": len(present),
            "coverage_frac": round(len(present) / len(TEAM_FIELDS), 3),
            "gaps": missing, "semantic_rel_fields": list(REL_FIELDS)}


def consumers_audit() -> dict:
    """Which tensor each learner/controller actually consumes — the shared-tensor question."""
    return {
        "entry_policy_E": "node_features (6-vertex per-vertex geometry, 48-d)",
        "push_option_theta1": "PushObs (coin, INDEXED tips[k,2], zone ray u) — a compact push view",
        "v7_sac_critic": "a 6-d hand-picked context (coin_y-emphasized); NOT the joint (a_L,a_R)",
        "semantic_tensor_z_team": "an OFFLINE artifact — NOT consumed by any controller or critic as the substrate",
        "shared_team_tensor_used": False,
        "critic_sees_joint_action": False,
        "left_right_ordering_consistent": True,   # semantic tensor pins is_left_tip / is_right_tip on their vertices
    }


@dataclass(frozen=True)
class _Geom:
    """The purely-geometric slice of z_team read from the live env (no history, no velocities, no action)."""
    coin: np.ndarray
    coin_v: np.ndarray
    zone: np.ndarray
    lft: np.ndarray
    rft: np.ndarray
    aperture: float
    sdir: np.ndarray
    smag: float
    left_contact: bool
    right_contact: bool
    arm_body_contact: bool


def _read_geometry(env: Any) -> _Geom:
    """Read the coin/target/fingertip geometry + contact flags from the env. Left/right roles pinned to the two
    fingertip sites (``arms['left']``/``arms['right']``). Shared by the snapshot and the stateful tracker (§6.1)."""
    m = env._planar_metrics
    # np.array (COPY), NOT np.asarray: site_xpos/disk_pos are live MuJoCo buffers updated IN PLACE by mj_step. A view
    # stored as `TeamTensorTracker._prev_lft` would alias the current position next step, making the finite-difference
    # fingertip velocity read exactly 0 forever (v14e audit caught this: lft_vx dead across the whole push window).
    coin = np.array(m.disk_pos[:2], np.float64)
    coin_v = np.array(m.disk_vel[:2], np.float64)
    zone = np.array([float(env._zone_x), float(env._zone_y)])
    from hymeko_rl.experiments.galambos_demo import _extract_arms
    arms = _extract_arms(env.model)
    lft = np.array(env.data.site_xpos[arms["left"].tip_site][:2], np.float64)
    rft = np.array(env.data.site_xpos[arms["right"].tip_site][:2], np.float64)
    mid = 0.5 * (lft + rft)
    squeeze = coin - mid
    smag = float(np.linalg.norm(squeeze))
    lg = getattr(m, "legality", None)
    return _Geom(coin=coin, coin_v=coin_v, zone=zone, lft=lft, rft=rft,
                 aperture=float(np.linalg.norm(lft - rft)), sdir=squeeze / (smag + 1e-9), smag=smag,
                 left_contact=bool(m.left_contact), right_contact=bool(m.right_contact),
                 arm_body_contact=bool(getattr(lg, "arm_body_contact", False) or False))


def _geom_record(g: _Geom, *, lft_v: np.ndarray, rft_v: np.ndarray, ft_progress: float, body_progress: float,
                 contact_lost: float, sustained: float, phase: str, prev_left: bool, prev_right: bool,
                 action: np.ndarray, high_coin_y_thresh: float) -> dict[str, float]:
    """Assemble the full VIEW record (all TEAM_FIELDS + JOINT_ACTION_FIELDS) from geometry + history + action."""
    mid = 0.5 * (g.lft + g.rft)
    to_t = g.zone - g.coin
    a = np.asarray(action, np.float64).ravel()
    a = a if a.size >= 4 else np.pad(a, (0, 4 - a.size))
    ph = {p: 1.0 if phase == p else 0.0 for p in ("APPROACH", "CONTACT", "PUSH", "DELIVERY")}
    both = g.left_contact and g.right_contact
    vals = [
        g.coin[0], g.coin[1], g.coin_v[0], g.coin_v[1], g.zone[0], g.zone[1],
        g.lft[0], g.lft[1], lft_v[0], lft_v[1], g.rft[0], g.rft[1], rft_v[0], rft_v[1],
        g.coin[0] - g.lft[0], g.coin[1] - g.lft[1], g.coin[0] - g.rft[0], g.coin[1] - g.rft[1],
        to_t[0], to_t[1], g.coin[0] - mid[0], g.coin[1] - mid[1],
        g.aperture, g.sdir[0], g.sdir[1], g.smag,
        float(g.left_contact), float(g.right_contact), float(both), float(g.arm_body_contact),
        ft_progress, body_progress, contact_lost, sustained,
        ph["APPROACH"], ph["CONTACT"], ph["PUSH"], ph["DELIVERY"],
        1.0 if g.coin[1] > high_coin_y_thresh else 0.0, float(prev_left), float(prev_right),
        a[0], a[1], a[2], a[3],
    ]
    return {f: float(v) for f, v in zip(VIEW_FIELDS, vals)}


def extract_team_features(env: Any, *, prev_left: bool, prev_right: bool, sustained_steps: int,
                          contact_lost: bool, phase: str = "PUSH", high_coin_y_thresh: float = 0.15) -> np.ndarray:
    """Snapshot z_team vector (TEAM_FIELDS order, velocities = 0 — no history). Kept for offline parity checks; the
    stateful :class:`TeamTensorTracker` is the runtime substrate that fills velocities and temporal signals."""
    g = _read_geometry(env)
    rec = _geom_record(g, lft_v=np.zeros(2), rft_v=np.zeros(2), ft_progress=0.0, body_progress=0.0,
                       contact_lost=float(contact_lost), sustained=float(sustained_steps), phase=phase,
                       prev_left=prev_left, prev_right=prev_right, action=np.zeros(4),
                       high_coin_y_thresh=high_coin_y_thresh)
    return np.array([rec[f] for f in TEAM_FIELDS], dtype=np.float32)


@dataclass
class TeamTensorTracker:
    """Stateful runtime extractor for the shared cooperative tensor z_team — the first-class substrate a CTDE critic /
    role-shared actors would consume. Unlike the snapshot form it fills EVERY field with a live value: fingertip
    velocities (finite-difference), per-step fingertip- vs body-attributed progress, previous-step contact, sustained
    two-fingertip duration, contact-lost-after-handoff, and the joint action.

    Preconditions: ``step`` receives the env AFTER its ``step`` (metrics current) and the action just applied.
    Postconditions: returns a dict over VIEW_FIELDS with all fields finite (100% coverage).
    Invariants: left/right roles pinned to ``arms['left']``/``arms['right']`` across the episode.
    """

    high_coin_y_thresh: float = 0.15
    progress_eps: float = 5e-4
    _dt: float = field(default=0.0, init=False)
    _prev_lft: np.ndarray | None = field(default=None, init=False)
    _prev_rft: np.ndarray | None = field(default=None, init=False)
    _prev_dist: float | None = field(default=None, init=False)
    _prev_left: bool = field(default=False, init=False)
    _prev_right: bool = field(default=False, init=False)
    _handoff_seen: bool = field(default=False, init=False)
    _sustained: int = field(default=0, init=False)

    def reset(self, env: Any) -> None:
        """Reset the per-episode history. Reads the control dt from the env (``timestep * frame_skip``)."""
        self._dt = float(env.model.opt.timestep) * int(getattr(env, "frame_skip", 1)) or 1.0
        self._prev_lft = self._prev_rft = None
        self._prev_dist = None
        self._prev_left = self._prev_right = False
        self._handoff_seen = False
        self._sustained = 0

    def step(self, env: Any, action: np.ndarray, phase: str) -> dict[str, float]:
        """Emit the full z_team record for the current env state + the ``action`` just applied under ``phase``."""
        g = _read_geometry(env)
        lft_v = self._finite_diff(self._prev_lft, g.lft)
        rft_v = self._finite_diff(self._prev_rft, g.rft)
        dist = float(env._planar_metrics.disk_to_zone)
        ft_p, body_p = self._progress(dist, g)
        both = g.left_contact and g.right_contact
        self._handoff_seen = self._handoff_seen or (both and phase in ("CONTACT", "PUSH"))
        contact_lost = 1.0 if (self._handoff_seen and not both) else 0.0
        self._sustained = self._sustained + 1 if both else 0
        rec = _geom_record(g, lft_v=lft_v, rft_v=rft_v, ft_progress=ft_p, body_progress=body_p,
                           contact_lost=contact_lost, sustained=float(self._sustained), phase=phase,
                           prev_left=self._prev_left, prev_right=self._prev_right, action=action,
                           high_coin_y_thresh=self.high_coin_y_thresh)
        self._prev_lft, self._prev_rft, self._prev_dist = g.lft, g.rft, dist
        self._prev_left, self._prev_right = g.left_contact, g.right_contact
        return rec

    def _finite_diff(self, prev: np.ndarray | None, cur: np.ndarray) -> np.ndarray:
        return np.zeros(2) if prev is None else (cur - prev) / self._dt

    def _progress(self, dist: float, g: _Geom) -> tuple[float, float]:
        """Per-step toward-zone displacement attributed to fingertip contact vs body-only contact (MonitorContext
        convention). Zero on the first step (no previous distance)."""
        if self._prev_dist is None:
            return 0.0, 0.0
        toward = max(0.0, self._prev_dist - dist)
        if toward < self.progress_eps:
            return 0.0, 0.0
        if g.left_contact or g.right_contact:
            return toward, 0.0
        if g.arm_body_contact:
            return 0.0, toward
        return 0.0, 0.0


def project(record: dict[str, float], fields: tuple[str, ...]) -> np.ndarray:
    """Project a z_team record onto a canonical subset (``ACTOR_FIELDS`` / ``CRITIC_FIELDS`` / ``VIEW_FIELDS``)."""
    return np.array([record[f] for f in fields], dtype=np.float32)


def coverage(record: dict[str, float]) -> dict[str, Any]:
    """100%-coverage guard: every VIEW field present and finite (no placeholder / NaN / inf)."""
    missing = [f for f in VIEW_FIELDS if f not in record]
    nonfinite = [f for f in VIEW_FIELDS if f in record and not np.isfinite(record[f])]
    covered = len(VIEW_FIELDS) - len(missing) - len(nonfinite)
    return {"n_fields": len(VIEW_FIELDS), "n_covered": covered,
            "coverage_frac": round(covered / len(VIEW_FIELDS), 3),
            "missing": missing, "nonfinite": nonfinite,
            "full_coverage": not missing and not nonfinite}
