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
    ("dtz", SHARED_ALONG), ("coin_vel_along", SHARED_ALONG), ("coin_vel_perp", SHARED_PERP), ("straddle", SHARED_ALONG),
    ("tip_coin_along", SIDE_ALONG), ("tip_coin_perp", SIDE_PERP),
    ("normal_along", SIDE_ALONG), ("normal_perp", SIDE_PERP),
    ("fn", SIDE_ALONG), ("slew_head_up", SIDE_ALONG), ("slew_head_dn", SIDE_ALONG),
)
R1_GROUP_ORDER = tuple(name for name, _ in R1_LAYOUT)
R1_KIND = dict(R1_LAYOUT)


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
    }


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
    """Concatenate the grouped features in the frozen R1 group order → a 1-D float32 vector. # Postconditions: length is
    fixed (4 shared + 7×2 − … per the layout); deterministic."""
    return np.concatenate([np.asarray(g[name], np.float64).ravel() for name in R1_GROUP_ORDER]).astype(np.float32)


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
    """Length of the flattened R1 vector (4 shared-1 groups + 7 per-side-2 groups = 4 + 14 = 18)."""
    return sum(1 if R1_KIND[name] in (SHARED_ALONG, SHARED_PERP) else 2 for name in R1_GROUP_ORDER)
