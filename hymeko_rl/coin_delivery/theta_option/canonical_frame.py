"""R1 — canonical target/contact-frame representation (equivariant to the left-right mirror).

The Step-1 audit found the 42-D features have NO canonical left-right frame: swapping L↔R moves a cradle's feature vector
by 3.49 (> the 2.58 distance between two *different* cradles), so a cradle and its own mirror look more different than two
distinct cradles. R1 removes that defect by (a) expressing everything in the TARGET frame (along e_par = coin→zone, perp =
⊥) so physically-similar cradles are close, and (b) CANONICALISING the left-right ordering.

Crucially the canonicalisation is carried through BOTH the state AND the θ label (an equivariance, not just input
invariance):

    mirror S:   φ(Sx) = φ(x)            (canonical features are mirror-invariant)
                θ_canonical = T_θ(θ)     (the 6-D label transforms: balance = the L/R steer, index 2, NEGATES)

    deploy:     physical state → canonicalise (record `was_swapped`) → predict canonical θ_0 →
                decode T_θ back if swapped → execute on the physical left/right arms.

Without the label transform the input becomes invariant but the SAME input still carries two different labels — the
aliasing the audit exposed. The mirror in the target frame is exactly: swap the two sides AND negate the perpendicular
components (a reflection across the target axis). S is an involution; the canonical representative is the S-orbit member
whose aggregate perpendicular key is ≥ 0 (a stable, deterministic tie-break — no frame-to-frame ordering jitter).
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot, primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy

BALANCE_IDX = 2                                      # θ = (squeeze, forward, BALANCE=L/R steer, ramp, release, brake)

# R1 target-frame feature layout: group -> kind.
#   SHARED_ALONG : mirror-invariant scalar/along quantity (unchanged by S)
#   SHARED_PERP  : shared perpendicular quantity (negated by S)
#   SIDE_ALONG   : per-side [left,right] along/magnitude (sides swapped by S, no sign change)
#   SIDE_PERP    : per-side [left,right] perpendicular (sides swapped AND negated by S)
SHARED_ALONG, SHARED_PERP, SIDE_ALONG, SIDE_PERP = "SHARED_ALONG", "SHARED_PERP", "SIDE_ALONG", "SIDE_PERP"
R1_LAYOUT: tuple[tuple[str, str], ...] = (
    # ── v1 target/contact-frame geometry ──
    ("dtz", SHARED_ALONG), ("coin_vel_along", SHARED_ALONG), ("coin_vel_perp", SHARED_PERP), ("straddle", SHARED_ALONG),
    ("tip_coin_along", SIDE_ALONG), ("tip_coin_perp", SIDE_PERP),
    ("normal_along", SIDE_ALONG), ("normal_perp", SIDE_PERP),
    ("fn", SIDE_ALONG), ("slew_head_up", SIDE_ALONG), ("slew_head_dn", SIDE_ALONG),
    # ── v2 configuration-dependent CONTROL AUTHORITY (B_τ = ∂vrel4/∂Δτ) ──
    # B_τ singular values / condition / active-rank are invariant under the mirror (orthogonal row/col transforms);
    # per-side response magnitude swaps sides. Kept as magnitudes ⇒ no sign-flip ambiguity in τ space.
    ("btau_svals", SHARED_ALONG), ("btau_summary", SHARED_ALONG), ("btau_side_auth", SIDE_ALONG),
    # ── v2 contact kinematics / forces (contact frame → target-consistent) ──
    ("vrel_normal", SIDE_ALONG), ("vrel_tangent", SIDE_PERP), ("friction_util", SIDE_ALONG),
    # ── v2 real control state (achievable next slew-admissible step) ──
    ("prev_tau_arm", SIDE_ALONG),
    # ── v3 SIGNED directional reachable authority (object frame; equivariant) ──
    # forward push/reverse & brake are along ±e_par ⇒ mirror-INVARIANT; lateral ± swaps as a pair; the B_coin governor-
    # attenuation is a scalar provenance flag.
    ("forward_push_reach", SHARED_ALONG), ("forward_reverse_reach", SHARED_ALONG),
    ("lateral_reach_pair", SIDE_ALONG), ("brake_opposed_reach", SHARED_ALONG),
    ("bcoin_min_attenuation", SHARED_ALONG),
    # ── v3 SIGNED contact/internal authority (B_τ null-space): total-normal ± invariant; L/R balance SIGN-FLIPS ──
    ("normal_force_reach_pair", SHARED_ALONG), ("balance_reach_signed", SHARED_PERP),
)
R1_GROUP_ORDER = tuple(name for name, _ in R1_LAYOUT)
R1_KIND = dict(R1_LAYOUT)
# groups whose flattened length differs from the kind default (SHARED_* = 1, SIDE_* = 2)
R1_GROUP_LEN = {"btau_svals": 4, "btau_summary": 2, "normal_force_reach_pair": 2}
# FIXED per-group physical scales (causal — no cross-cradle leak) so no component dominates the distance/model. The same
# scale applies to both sides of a SIDE group (⇒ commutes with the swap ⇒ mirror-safe). Chosen from the physical ranges.
R1_NORM_SCALES = {
    "dtz": 0.20, "coin_vel_along": 0.50, "coin_vel_perp": 0.50, "straddle": 1.0,
    "tip_coin_along": 0.05, "tip_coin_perp": 0.05, "normal_along": 1.0, "normal_perp": 1.0,
    "fn": 5.0, "slew_head_up": 1.0, "slew_head_dn": 1.0,
    "btau_svals": 0.40, "btau_summary": 0.50, "btau_side_auth": 0.40,
    "vrel_normal": 0.20, "vrel_tangent": 0.20, "friction_util": 1.0, "prev_tau_arm": 2.0,
    "forward_push_reach": 0.10, "forward_reverse_reach": 0.10, "lateral_reach_pair": 0.10,
    "brake_opposed_reach": 0.10, "bcoin_min_attenuation": 1.0, "normal_force_reach_pair": 0.25,
    "balance_reach_signed": 0.10,
}


def group_len(name: str) -> int:
    """Flattened length of a feature group (explicit override, else 1 for SHARED_* and 2 for SIDE_*)."""
    return R1_GROUP_LEN.get(name, 1 if R1_KIND[name] in (SHARED_ALONG, SHARED_PERP) else 2)


def _perp(u: np.ndarray) -> np.ndarray:
    """The +90° rotation of the unit target direction (the target-frame perpendicular axis)."""
    return np.array([-u[1], u[0]], np.float64)


def r1_grouped_features(snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """Extract the R1 target/contact-frame features at the frozen handoff (read-only; a fresh branch is inspected). Every
    2-D physical quantity is projected onto (e_par = coin→zone, e_perp = ⊥); per-side entries are [left, right].
    # Postconditions: keys == R1_GROUP_ORDER; SIDE groups are length-2; all finite."""
    rl = snap.branch()
    mujoco.mj_forward(rl.inner.model, rl.inner.data)
    u, dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float64)[:2]
    n = float(np.linalg.norm(e_par))
    e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
    e_perp = _perp(e_par)
    coin = _coin_xy(rl)
    v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    con = primary_fingertip_contacts(rl)

    def _side(side: str, key: str, default: np.ndarray) -> np.ndarray:
        c = con[side]
        return np.asarray(c[key], np.float64) if c is not None else np.asarray(default, np.float64)

    xcl, xcr = _side("left", "x_c", coin), _side("right", "x_c", coin)
    nl, nr = _side("left", "n", np.zeros(2)), _side("right", "n", np.zeros(2))
    fn_l = float(con["left"]["fn"]) if con["left"] is not None else 0.0
    fn_r = float(con["right"]["fn"]) if con["right"] is not None else 0.0
    slew = float(snap.stack.tau_rate * snap.stack.control_dt)
    up = np.minimum(snap.hi - snap.prev_tau, slew) / slew          # per-joint up-headroom (4)
    dn = np.minimum(snap.prev_tau - snap.lo, slew) / slew          # per-joint down-headroom (4)
    ba = _btau_authority(snap)                                     # B_τ singular / condition / per-side authority
    ck = _contact_kinematics(rl, snap)                            # per-side v_n, v_t, friction utilisation
    from hymeko_rl.coin_delivery.theta_option.directional_authority import (
        contact_internal_authority, object_authority)
    oa = object_authority(snap)                                   # v3 signed object reachable authority (B_coin)
    ci = contact_internal_authority(snap)                        # v3 signed contact/internal authority (B_τ null-space)
    pt = np.asarray(snap.prev_tau, np.float64)
    # per-ARM slew headroom = mean over that arm's joints (joints 0,1 = left, 2,3 = right); a magnitude ⇒ no sign flip
    return {
        "dtz": np.array([float(dtz)]),
        "coin_vel_along": np.array([float(v @ e_par)]),
        "coin_vel_perp": np.array([float(v @ e_perp)]),
        "straddle": np.array([float(snap.straddle0)]),
        "tip_coin_along": np.array([float((xcl - coin) @ e_par), float((xcr - coin) @ e_par)]),
        "tip_coin_perp": np.array([float((xcl - coin) @ e_perp), float((xcr - coin) @ e_perp)]),
        "normal_along": np.array([float(nl @ e_par), float(nr @ e_par)]),
        "normal_perp": np.array([float(nl @ e_perp), float(nr @ e_perp)]),
        "fn": np.array([fn_l, fn_r]),
        "slew_head_up": np.array([float(up[:2].mean()), float(up[2:].mean())]),
        "slew_head_dn": np.array([float(dn[:2].mean()), float(dn[2:].mean())]),
        "btau_svals": ba["svals"], "btau_summary": ba["summary"], "btau_side_auth": ba["side_auth"],
        "vrel_normal": ck["vrel_normal"], "vrel_tangent": ck["vrel_tangent"], "friction_util": ck["friction_util"],
        "prev_tau_arm": np.array([float(np.linalg.norm(pt[:2])), float(np.linalg.norm(pt[2:]))]),
        "forward_push_reach": oa["forward_push_reach"], "forward_reverse_reach": oa["forward_reverse_reach"],
        "lateral_reach_pair": oa["lateral_reach_pair"], "brake_opposed_reach": oa["brake_opposed_reach"],
        "bcoin_min_attenuation": oa["bcoin_min_attenuation"],
        "normal_force_reach_pair": ci["normal_force_reach_pair"], "balance_reach_signed": ci["balance_reach_signed"],
    }


def _btau_authority(snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """Configuration-dependent one-step control authority from B_τ = ∂vrel4/∂Δτ (contact-frame rows [Lvn,Lvt,Rvn,Rvt]).
    Returns mirror-safe, WELL-SCALED scalars: the 4 singular values + [active-rank fraction, Frobenius norm] (invariant
    under the mirror's orthogonal row/col action) and the per-side response magnitude [‖left rows‖, ‖right rows‖] (swaps
    sides). The raw condition number is deliberately dropped — it is O(100+) (it would dominate the unnormalised distance)
    AND redundant with the singular values. Reuses the frozen Route-B `identify_Btau`. # Postconditions: all finite,
    O(1); magnitudes ⇒ no τ-sign ambiguity."""
    from hymeko_rl.coin_delivery.torque_authority import TorqueAuthorityConfig, identify_Btau
    info = identify_Btau(snap, 0.05, TorqueAuthorityConfig())
    B = np.nan_to_num(np.asarray(info["Btau"], np.float64), nan=0.0)
    sv = np.sort(np.linalg.svd(B, compute_uv=False))[::-1]
    return {"svals": sv.astype(np.float64),
            "summary": np.array([float(info["n_active"]) / 4.0, float(np.linalg.norm(B))]),
            "side_auth": np.array([float(np.linalg.norm(B[:2])), float(np.linalg.norm(B[2:]))])}


def _contact_kinematics(rl: Any, snap: CradleSnapshot) -> dict[str, np.ndarray]:
    """Per-side contact-relative kinematics at the handoff: normal-approach velocity v_n (SIDE_ALONG — symmetric),
    tangential velocity v_t (SIDE_PERP — flips under reflection), and a friction-utilisation slip proxy |v_t|/(|v_n|+ε)
    (SIDE_ALONG magnitude). Reuses `measure_contact_velocities` on the frozen frames. Missing contact ⇒ zeros."""
    from hymeko_rl.coin_delivery.contact_velocity import measure_contact_velocities
    try:
        per = measure_contact_velocities(rl, snap.frames)["per"]
    except Exception:
        per = {"left": None, "right": None}

    def _vn(side: str) -> float:
        c = per.get(side)
        return float(c["v_n"]) if c is not None else 0.0

    def _vt(side: str) -> float:
        c = per.get(side)
        return float(c["v_t"]) if c is not None else 0.0

    lvn, rvn, lvt, rvt = _vn("left"), _vn("right"), _vt("left"), _vt("right")
    # BOUNDED slip fraction ∈ [0,1] (NOT the unbounded |v_t|/|v_n| ratio, which explodes as v_n→0 and dominated the
    # distance at ~96% — a scale-dominance bug the per-group audit caught, same class as the B_τ condition number).
    def _slip(vt: float, vn: float) -> float:
        return abs(vt) / (abs(vt) + abs(vn) + 1e-3)
    return {"vrel_normal": np.array([lvn, rvn]), "vrel_tangent": np.array([lvt, rvt]),
            "friction_util": np.array([_slip(lvt, lvn), _slip(rvt, rvn)])}


def swap_grouped(g: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The mirror S on grouped R1 features: SHARED_ALONG unchanged; SHARED_PERP negated; SIDE_ALONG sides swapped;
    SIDE_PERP sides swapped AND negated. S is an involution (S∘S = identity). # Postconditions: same keys/shapes."""
    out = {}
    for name in g:
        v = np.asarray(g[name], np.float64).copy()
        kind = R1_KIND[name]
        if kind == SHARED_ALONG:
            out[name] = v
        elif kind == SHARED_PERP:
            out[name] = -v
        elif kind == SIDE_ALONG:
            out[name] = v[::-1]
        else:                                                       # SIDE_PERP: swap sides and negate
            out[name] = -v[::-1]
    return out


def perp_key(g: dict[str, np.ndarray]) -> float:
    """The canonical-ordering key: the sum of every perpendicular-tagged entry. Antisymmetric under S
    (``perp_key(swap(g)) == -perp_key(g)``), so ``perp_key ≥ 0`` selects a unique S-orbit representative. Aggregating all
    perp signals makes an exact zero (which would make the tie-break arbitrary) physically negligible."""
    k = 0.0
    for name in g:
        if R1_KIND[name] in (SHARED_PERP, SIDE_PERP):
            k += float(np.sum(g[name]))
    return k


def canonicalise(g: dict[str, np.ndarray]) -> "tuple[dict[str, np.ndarray], bool]":
    """Map g to its canonical S-orbit representative (perp_key ≥ 0). Returns (canonical grouped features, was_swapped).
    # Postconditions: canonicalise(g) == canonicalise(swap(g)) (mirror-invariant); deterministic."""
    if perp_key(g) < 0.0:
        return swap_grouped(g), True
    return {k: np.asarray(v, np.float64).copy() for k, v in g.items()}, False


def flatten_r1(g: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the grouped features in the frozen R1 group order, each divided by its FIXED per-group physical scale
    (so no component dominates the distance), → a 1-D float32 vector. The per-group scale is identical for both sides ⇒ it
    commutes with the mirror swap ⇒ the canonicalisation invariance is preserved. # Postconditions: length == r1_feature_dim();
    deterministic; every component O(1)."""
    return np.concatenate([np.asarray(g[name], np.float64).ravel() / R1_NORM_SCALES.get(name, 1.0)
                           for name in R1_GROUP_ORDER]).astype(np.float32)


def r1_canonical_features(snap: CradleSnapshot) -> "tuple[np.ndarray, bool]":
    """The R1 deploy entry: extract target-frame features, canonicalise, flatten. Returns (canonical feature vector,
    was_swapped) — ``was_swapped`` is needed to decode a predicted canonical θ back onto the physical arms."""
    g, swapped = canonicalise(r1_grouped_features(snap))
    return flatten_r1(g), swapped


def swap_theta(theta: np.ndarray) -> np.ndarray:
    """The label transform T_θ: negate the L/R steer (balance, index 2); all other components unchanged. Involution
    (T_θ∘T_θ = identity). Balance bounds are symmetric (±0.10) so the result stays legal. This is the θ-side of the mirror
    equivariance — a canonical θ predicted for the canonical (possibly swapped) state must be un-transformed before it is
    executed on the physical arms."""
    t = np.asarray(theta, np.float64).copy()
    t[BALANCE_IDX] = -t[BALANCE_IDX]
    return t


def to_canonical_theta(theta: np.ndarray, was_swapped: bool) -> np.ndarray:
    """Transform a PHYSICAL θ into the canonical frame (apply T_θ iff the state was swapped) — used to build canonical
    training labels from physical teacher θ."""
    return swap_theta(theta) if was_swapped else np.asarray(theta, np.float64).copy()


def from_canonical_theta(theta_canonical: np.ndarray, was_swapped: bool) -> np.ndarray:
    """Decode a predicted CANONICAL θ back onto the physical left/right arms (undo T_θ iff the state was swapped). Since
    T_θ is an involution, the decode is the same negation. # Postconditions: from_canonical_theta(to_canonical_theta(θ,s),s)
    == θ."""
    return swap_theta(theta_canonical) if was_swapped else np.asarray(theta_canonical, np.float64).copy()


def r1_feature_dim() -> int:
    """Length of the flattened R1 vector (sum of per-group lengths across the frozen layout)."""
    return sum(group_len(name) for name in R1_GROUP_ORDER)
