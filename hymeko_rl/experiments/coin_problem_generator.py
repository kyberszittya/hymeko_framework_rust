"""HyMeKo structured problem generator for Coin Delivery (minimal adapter on the canonical PlanarSnapshot path, §2).

Emits ONLY initial/target configurations (PlanarSnapshot + zone) with structural labels + parent identity + provenance;
it never controls the actor. Three families (§4): CERTIFIED_NEIGHBORHOOD (variants around known-certified parents),
ATTRIBUTION_BOUNDARY (contrastive variants around the bulldoze/plow boundary), LEFT_RIGHT_SYMMETRY (canonical mirrors).
Each variant changes exactly ONE named relation. Reuses env.planar_snapshot.{PlanarSnapshot,snapshot_planar,
restore_planar} + coin_delivery.provenance.snapshot_hash; validity is PHYSICAL only (never policy-success, §5).

qpos layout: [j1_left, j2_left, j1_right, j2_right, coin_x, coin_y, coin_theta]; zone = (zx, zy) per config.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import PlanarSnapshot, restore_planar, snapshot_planar

CERTIFIED_NEIGHBORHOOD = "CERTIFIED_NEIGHBORHOOD"
ATTRIBUTION_BOUNDARY = "ATTRIBUTION_BOUNDARY"
LEFT_RIGHT_SYMMETRY = "LEFT_RIGHT_SYMMETRY"
_LX, _LY, _CX, _CY, _CT = 0, 1, 2, 3, 4          # arm joint idx (left shoulder/elbow, right shoulder/elbow) ... coin
_JL1, _JL2, _JR1, _JR2, _COINX, _COINY, _COINT = 0, 1, 2, 3, 4, 5, 6


@dataclass(frozen=True)
class GeneratedConfig:
    """One generated Coin-Delivery problem: a restorable config + structural provenance (never a policy outcome)."""
    config_id: str
    parent_seed: int
    family: str
    changed_relation: str
    params: dict
    state_hash: str
    left_reachable: bool
    right_reachable: bool
    target_dir: tuple
    not_initially_successful: bool
    snapshot: PlanarSnapshot

    def provenance(self) -> dict:
        d = {k: getattr(self, k) for k in ("config_id", "parent_seed", "family", "changed_relation", "params",
                                           "state_hash", "left_reachable", "right_reachable", "target_dir",
                                           "not_initially_successful")}
        return d


# ── named single-relation perturbations (each returns a NEW snapshot; changes exactly one relation) ────────────────
def _perturb(snap: PlanarSnapshot, relation: str, p: float) -> PlanarSnapshot:
    q = snap.qpos.copy()
    zx, zy = snap.zone
    if relation == "coin_lateral_offset":
        q[_COINX] += p
    elif relation == "coin_to_target_angle":                     # rotate coin about the zone by angle p (rad)
        dx, dy = q[_COINX] - zx, q[_COINY] - zy
        c, s = np.cos(p), np.sin(p)
        q[_COINX], q[_COINY] = zx + c * dx - s * dy, zy + s * dx + c * dy
    elif relation == "lr_contact_gap":                           # elbows apart/together (left +, right -)
        q[_JL2] += p
        q[_JR2] -= p
    elif relation == "lr_reach_asymmetry":                       # left shoulder shifts only (asymmetric reach)
        q[_JL1] += p
    elif relation == "bilateral_convergence":                    # both elbows converge toward the coin
        q[_JL2] += p
        q[_JR2] += p
    else:
        raise ValueError(f"unknown relation {relation!r}")
    return dataclasses.replace(snap, qpos=q, qvel=np.zeros_like(snap.qvel), qacc_warmstart=np.zeros_like(snap.qacc_warmstart))


def _mirror(snap: PlanarSnapshot) -> PlanarSnapshot:
    """Canonical left-right mirror: swap the L/R arm joints (by named actuator identity, not raw index math) with a
    sign flip, reflect the coin x about the zone x, negate coin theta. Zone x is reflected about itself (unchanged)."""
    q = snap.qpos.copy()
    zx, zy = snap.zone
    q[_JL1], q[_JR1] = -snap.qpos[_JR1], -snap.qpos[_JL1]        # left shoulder <-> right shoulder (mirror)
    q[_JL2], q[_JR2] = -snap.qpos[_JR2], -snap.qpos[_JL2]        # left elbow <-> right elbow
    q[_COINX] = 2.0 * zx - snap.qpos[_COINX]                     # reflect coin x about the zone
    q[_COINT] = -snap.qpos[_COINT]
    return dataclasses.replace(snap, qpos=q, qvel=np.zeros_like(snap.qvel), qacc_warmstart=np.zeros_like(snap.qacc_warmstart))


# ── validity (PHYSICAL only; never policy-success) ────────────────────────────────────────────────────────────────
def validate(env, snap: PlanarSnapshot) -> tuple[bool, str, dict]:
    """Restore + physically validate a generated config. # Postconditions returns (ok, reason, facts). Rejects: starts
    in-zone, non-deterministic restore, invalid geometry/penetration, arm unreachable, dtz<=0."""
    try:
        restore_planar(env.inner, snap)
    except Exception as exc:                                     # noqa: BLE001 (any restore failure = invalid)
        return False, f"restore_error:{type(exc).__name__}", {}
    q1 = env.inner.data.qpos.copy()
    m = env.inner.planar_metrics
    restore_planar(env.inner, snap)                              # deterministic-restore check
    if not np.allclose(q1, env.inner.data.qpos):
        return False, "nondeterministic_restore", {}
    if bool(m.in_zone):
        return False, "starts_in_zone", {}
    dtz = float(m.disk_to_zone)
    if not np.isfinite(dtz) or dtz <= 1e-4:
        return False, "invalid_dtz", {}
    ltip = float(np.linalg.norm(np.asarray(m.left_tip_dist))) if np.ndim(m.left_tip_dist) else float(m.left_tip_dist)
    rtip = float(np.linalg.norm(np.asarray(m.right_tip_dist))) if np.ndim(m.right_tip_dist) else float(m.right_tip_dist)
    left_ok, right_ok = ltip < 0.30, rtip < 0.30                # coin within each fingertip's reach band
    if not (left_ok and right_ok):
        return False, "arm_unreachable", {}
    coin = np.asarray(m.disk_pos[:2], np.float64)
    if not (-0.35 < coin[0] < 0.35 and 0.02 < coin[1] < 0.34):   # coin on the table (no ejected/penetrating geometry)
        return False, "invalid_geometry", {}
    d = np.array([snap.zone[0], snap.zone[1]]) - coin
    facts = dict(dtz=dtz, left_reachable=left_ok, right_reachable=right_ok,
                 target_dir=(float(d[0]), float(d[1])), in_zone=False, ltip=ltip, rtip=rtip)
    return True, "ok", facts


def _parent_snapshot(env, seed: int) -> PlanarSnapshot:
    env.reset(seed=int(seed))
    return snapshot_planar(env.inner)


_FAMILY_A_RELATIONS = ("coin_lateral_offset", "coin_to_target_angle", "lr_contact_gap",
                       "lr_reach_asymmetry", "bilateral_convergence")
_FAMILY_B_RELATIONS = ("lr_contact_gap", "lr_reach_asymmetry", "coin_lateral_offset")   # bulldoze<->plow boundary knobs


def generate(env, *, n_per_family: int, rng: np.random.Generator) -> tuple[list[GeneratedConfig], dict]:
    """Generate ~``n_per_family`` valid configs for EACH of the 3 families (A/B from parents by single-relation
    perturbation; C by canonical mirror of A+B). Returns (configs, rejection_counts). Validity is physical only."""
    rej: dict = {}
    out: list[GeneratedConfig] = []

    def _try(parent_seed, family, relation, p, snap):
        ok, reason, facts = validate(env, snap)
        if not ok:
            rej[reason] = rej.get(reason, 0) + 1
            return None
        h = snapshot_hash(snap)
        cid = f"{family[:4]}_{parent_seed}_{relation}_{p:+.3f}_{h[:8]}"
        return GeneratedConfig(cid, int(parent_seed), family, relation, {"p": float(p)}, h,
                               facts["left_reachable"], facts["right_reachable"], facts["target_dir"], True, snap)

    def _fill(family, parents, relations, target):
        got = []
        amp = 0.02
        while len(got) < target and amp < 0.5:
            for ps in parents:
                base = _parent_snapshot(env, ps)
                for rel in relations:
                    p = float(rng.uniform(-amp, amp))
                    if abs(p) < amp * 0.3:
                        p += np.sign(p or 1.0) * amp * 0.3
                    cfg = _try(ps, family, rel, p, _perturb(base, rel, p))
                    if cfg is not None and cfg.state_hash not in {c.state_hash for c in got}:
                        got.append(cfg)
                    if len(got) >= target:
                        break
                if len(got) >= target:
                    break
            amp += 0.02
        return got[:target]

    a = _fill(CERTIFIED_NEIGHBORHOOD, (64102, 64201), _FAMILY_A_RELATIONS, n_per_family)
    b = _fill(ATTRIBUTION_BOUNDARY, (64111,), _FAMILY_B_RELATIONS, n_per_family)
    # family C: canonical mirror of a balanced subset of A+B
    src = (a + b)
    rng.shuffle(src)
    c = []
    for base in src:
        snap = _mirror(base.snapshot)
        ok, reason, facts = validate(env, snap)
        if not ok:
            rej[f"mirror_{reason}"] = rej.get(f"mirror_{reason}", 0) + 1
            continue
        h = snapshot_hash(snap)
        c.append(GeneratedConfig(f"MIRR_{base.config_id}_{h[:8]}", base.parent_seed, LEFT_RIGHT_SYMMETRY,
                                 f"mirror({base.changed_relation})", base.params, h, facts["left_reachable"],
                                 facts["right_reachable"], facts["target_dir"], True, snap))
        if len(c) >= n_per_family:
            break
    out = a + b + c
    return out, rej


def split_freeze(configs: list[GeneratedConfig], *, n_train_per: int, n_held_per: int, out_dir: Path,
                 rng: np.random.Generator) -> dict:
    """Split each family into disjoint TRAIN/HELD-OUT sets (frozen, hashed). Returns the manifest (ids + hashes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_fam: dict = {}
    for c in configs:
        by_fam.setdefault(c.family, []).append(c)
    train, held = [], []
    for fam, cs in by_fam.items():
        idx = rng.permutation(len(cs))
        train += [cs[i] for i in idx[:n_train_per]]
        held += [cs[i] for i in idx[n_train_per:n_train_per + n_held_per]]
    for name, cs in (("train", train), ("held", held)):
        with open(out_dir / f"{name}_configs.pkl", "wb") as f:
            pickle.dump(cs, f)
    manifest = dict(
        n_train=len(train), n_held=len(held),
        train_by_family={f: sum(c.family == f for c in train) for f in by_fam},
        held_by_family={f: sum(c.family == f for c in held) for f in by_fam},
        train_ids=[c.config_id for c in train], held_ids=[c.config_id for c in held],
        train_hashes=[c.state_hash for c in train], held_hashes=[c.state_hash for c in held],
        corpus_sha=hashlib.sha256("".join(sorted(c.state_hash for c in train + held)).encode()).hexdigest())
    (out_dir / "generator_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    return manifest


def load_configs(path: Path) -> list[GeneratedConfig]:
    with open(path, "rb") as f:
        return pickle.load(f)
