"""HyMeKo semantic manifold tensor for the coin-toss / Galambos task (coin_toss_v2, Part A).

This is NOT a flat observation. It is a fixed-schema *structured* representation of the task manifold: the
10-vertex Galambos task hypergraph (arm links + fingertips + coin + zone + grasp/goal hubs) with a signed
incidence, per-vertex features that place each **relation on its entity** (contact + tip-distance on the fingertip
vertices, coin→zone on the coin/goal-hub), a global relational block, and a monitor-component block used as the
supervised LABEL. HSiKAN consumes (node-feature matrix + signed incidence); an MLP consumes the flattened node
matrix (structure-blind) — the fair Part-B contrast. The schema carries a field-order hash (a tensor contract).

Entities/relations are reused (PlanarGraspMetrics, node_features, MonitorContext, handoff_quality_features); the
only derived quantities are push-direction alignment and strict stable-delivery (both flagged in the v2 report).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---- vertex roles (7) for the 10 task-graph vertices
ROLES = ("arm_base", "arm_link", "fingertip", "coin", "zone", "grasp_hub", "goal_hub")
_VERTEX_ROLE = {
    "base_left": "arm_base", "base_right": "arm_base",
    "link1_left": "arm_link", "link1_right": "arm_link",
    "link2_left": "fingertip", "link2_right": "fingertip",
    "coin": "coin", "zone": "zone", "grasp_hub": "grasp_hub", "goal_hub": "goal_hub",
}
_LEFT_TIP_LABEL, _RIGHT_TIP_LABEL = "link2_left", "link2_right"

# ---- schema field names (the tensor contract; order is load-bearing → hashed)
_GEOM = ("qpos", "qvel", "x", "y", "dcoin_x", "dcoin_y", "dzone_x", "dzone_y")   # from env.node_features()
NODE_FIELDS = (*(f"role_{r}" for r in ROLES), *_GEOM, "is_left_tip", "is_right_tip", "contact", "tip_dist")
REL_FIELDS = ("left_tip_dist", "right_tip_dist", "two_finger_symmetry", "both_contact", "arm_body_contact",
              "body_only_progress", "ft_progress_ratio", "coin_to_target_x", "coin_to_target_y",
              "coin_to_target_dist", "push_alignment", "coin_speed",
              "phase_APPROACH", "phase_CONTACT", "phase_PUSH", "phase_DELIVERY", "handoff_quality_score")
MON_FIELDS = ("approach_score", "contact_score", "progress_score", "delivery_score", "anti_exploit_score",
              "stable_delivery", "monitor_score", "monitor_pass")
N_VERTICES = 10


def schema_field_order_hash() -> str:
    """md5 over the ordered schema field names (roles + node + relational + monitor). A change in ANY field name
    or order changes the hash — the tensor contract (mirrors PipelineSchemaLedger's field-order hash)."""
    payload = "|".join(("ROLES", *ROLES, "NODE", *NODE_FIELDS, "REL", *REL_FIELDS, "MON", *MON_FIELDS,
                        f"N_VERTICES={N_VERTICES}"))
    return hashlib.md5(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SemanticSchema:
    roles: tuple[str, ...] = ROLES
    node_fields: tuple[str, ...] = NODE_FIELDS
    rel_fields: tuple[str, ...] = REL_FIELDS
    mon_fields: tuple[str, ...] = MON_FIELDS
    n_vertices: int = N_VERTICES

    @property
    def node_dim(self) -> int:
        return len(self.node_fields)

    @property
    def field_order_hash(self) -> str:
        return schema_field_order_hash()

    def as_dict(self) -> dict:
        return {"roles": list(self.roles), "node_fields": list(self.node_fields), "rel_fields": list(self.rel_fields),
                "mon_fields": list(self.mon_fields), "n_vertices": self.n_vertices,
                "node_dim": self.node_dim, "rel_dim": len(self.rel_fields), "mon_dim": len(self.mon_fields),
                "field_order_hash": self.field_order_hash}


@dataclass
class SemanticTensor:
    node_matrix: np.ndarray          # (N, node_dim)  — HSiKAN input (with incidence); MLP input flattened
    relational: np.ndarray           # (rel_dim,)     — global relational block
    monitor: np.ndarray              # (mon_dim,)     — monitor components (LABELS)
    schema_hash: str

    def flat(self) -> np.ndarray:
        """Structure-blind flattening: node matrix + relational block (for the MLP baselines)."""
        return np.concatenate([self.node_matrix.reshape(-1), self.relational]).astype(np.float32)

    def validate(self, schema: SemanticSchema) -> None:
        # Preconditions: shapes match the schema and the field-order hash is current.
        assert self.node_matrix.shape == (schema.n_vertices, schema.node_dim), self.node_matrix.shape
        assert self.relational.shape == (len(schema.rel_fields),), self.relational.shape
        assert self.monitor.shape == (len(schema.mon_fields),), self.monitor.shape
        assert self.schema_hash == schema.field_order_hash, "field-order hash mismatch (tensor contract violated)"


def role_onehot(label: str) -> np.ndarray:
    v = np.zeros(len(ROLES), dtype=np.float32)
    v[ROLES.index(_VERTEX_ROLE.get(label, "arm_link"))] = 1.0
    return v


def extract_node_matrix(env: Any, node_features: np.ndarray) -> np.ndarray:
    """Per-vertex features = role one-hot + geometry (node_features) + per-vertex contact/tip-distance placed on
    the fingertip vertices (relations on their entities). ``node_features`` is env.node_features() (N, 8)."""
    labels = env.hg.vertex_labels
    m = env._planar_metrics
    tip = {_LEFT_TIP_LABEL: (bool(m.left_contact), float(m.left_tip_dist)),
           _RIGHT_TIP_LABEL: (bool(m.right_contact), float(m.right_tip_dist))}
    rows = []
    for i, lab in enumerate(labels):
        contact, tip_dist = tip.get(lab, (False, 0.0))
        rows.append(np.concatenate([role_onehot(lab), node_features[i].astype(np.float32),
                                    np.array([1.0 if lab == _LEFT_TIP_LABEL else 0.0,
                                              1.0 if lab == _RIGHT_TIP_LABEL else 0.0,
                                              1.0 if contact else 0.0, tip_dist], dtype=np.float32)]))
    return np.asarray(rows, dtype=np.float32)


def push_alignment(coin_vel: np.ndarray, coin: np.ndarray, zone: np.ndarray) -> float:
    """Derived: cos-alignment of the coin's velocity with the coin→zone direction (∈[-1,1]; 0 if either is ~0)."""
    to_zone = np.asarray(zone) - np.asarray(coin)
    nv, nz = float(np.linalg.norm(coin_vel)), float(np.linalg.norm(to_zone))
    return float(np.dot(coin_vel, to_zone) / (nv * nz)) if nv > 1e-9 and nz > 1e-9 else 0.0


def stable_delivery(ctx: Any, success_steps: int) -> float:
    """Derived strict stable-delivery: 1.0 if held for k consec AND still held at episode end; 0.5 if held k
    somewhere but lost; 0.0 if never held k. (DeliveryMonitor only exposes the k-consec-anywhere form.)"""
    held = bool(ctx.max_dwell >= success_steps)
    terminal = bool(len(ctx.dwell) and ctx.dwell[-1] >= success_steps)
    return 1.0 if (held and terminal) else (0.5 if held else 0.0)


def extract_relational(env: Any, ctx: Any, phase: str, handoff_score: float, contract: Any) -> np.ndarray:
    """Global relational block from a MonitorContext (episode) + a phase label + handoff-quality score."""
    m = env._planar_metrics
    coin = np.asarray(m.disk_pos, dtype=np.float64)
    zone = np.array([float(env._zone_x), float(env._zone_y)], dtype=np.float64)
    sym = abs(float(m.left_tip_dist) - float(m.right_tip_dist))
    ft_ratio = float(ctx.ft_prog / (ctx.total_prog + 1e-9)) if ctx.total_prog > 1e-9 else 0.0
    to_zone = zone - coin
    ph = {p: 1.0 if phase == p else 0.0 for p in ("APPROACH", "CONTACT", "PUSH", "DELIVERY")}
    return np.array([float(m.left_tip_dist), float(m.right_tip_dist), sym,
                     float(ctx.both_contact.mean()), float(ctx.body_contact.any()),
                     float(ctx.body_prog), ft_ratio, float(to_zone[0]), float(to_zone[1]),
                     float(np.linalg.norm(to_zone)), push_alignment(np.asarray(m.disk_vel), coin, zone),
                     float(m.disk_speed), ph["APPROACH"], ph["CONTACT"], ph["PUSH"], ph["DELIVERY"],
                     float(handoff_score)], dtype=np.float32)


def extract_monitor(verdict: Any, ctx: Any, contract: Any) -> np.ndarray:
    """Monitor-component LABEL block from a TaskVerdict + MonitorContext."""
    return np.array([float(verdict.approach_score), float(verdict.contact_score), float(verdict.progress_score),
                     float(verdict.delivery_score), float(verdict.anti_exploit_score),
                     stable_delivery(ctx, contract.success_steps), float(verdict.monitor_score),
                     1.0 if verdict.monitor_pass else 0.0], dtype=np.float32)
