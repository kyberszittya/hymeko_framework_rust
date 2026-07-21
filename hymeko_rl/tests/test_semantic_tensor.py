"""Tensor-contract tests for the HyMeKo semantic manifold tensor (coin_toss_v2 Part A)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.eval.semantic_tensor import (
    MON_FIELDS, NODE_FIELDS, REL_FIELDS, ROLES, SemanticSchema, SemanticTensor,
    push_alignment, role_onehot, schema_field_order_hash, stable_delivery)


def test_schema_hash_stable_and_sensitive() -> None:
    h1 = schema_field_order_hash()
    assert h1 == schema_field_order_hash()                       # deterministic
    assert len(h1) == 32 and SemanticSchema().field_order_hash == h1


def test_schema_dims() -> None:
    s = SemanticSchema()
    assert s.n_vertices == 10
    assert s.node_dim == len(NODE_FIELDS) == len(ROLES) + 8 + 4  # role onehot + geom(8) + [is_left,is_right,contact,tip_dist]
    assert len(REL_FIELDS) == 17 and len(MON_FIELDS) == 8


def test_role_onehot() -> None:
    assert role_onehot("coin")[ROLES.index("coin")] == 1.0 and role_onehot("coin").sum() == 1.0
    assert role_onehot("link2_left")[ROLES.index("fingertip")] == 1.0
    assert role_onehot("base_right")[ROLES.index("arm_base")] == 1.0
    assert role_onehot("grasp_hub")[ROLES.index("grasp_hub")] == 1.0


def test_push_alignment_sign() -> None:
    coin, zone = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert push_alignment(np.array([0.1, 0.0]), coin, zone) > 0.99      # moving toward zone
    assert push_alignment(np.array([-0.1, 0.0]), coin, zone) < -0.99    # away
    assert push_alignment(np.array([0.0, 0.0]), coin, zone) == 0.0      # static → 0


class _Ctx:
    def __init__(self, max_dwell, dwell_last):
        self.max_dwell = max_dwell
        self.dwell = np.array([0, dwell_last])


def test_stable_delivery_tiers() -> None:
    assert stable_delivery(_Ctx(6, 6), 5) == 1.0     # held k and still held at end
    assert stable_delivery(_Ctx(6, 0), 5) == 0.5     # held k somewhere but lost
    assert stable_delivery(_Ctx(2, 0), 5) == 0.0     # never held k


def test_tensor_validate_and_flat() -> None:
    s = SemanticSchema()
    t = SemanticTensor(node_matrix=np.zeros((10, s.node_dim), np.float32),
                       relational=np.zeros(len(REL_FIELDS), np.float32),
                       monitor=np.zeros(len(MON_FIELDS), np.float32), schema_hash=s.field_order_hash)
    t.validate(s)                                                 # correct shapes + hash → no raise
    assert t.flat().shape == (10 * s.node_dim + len(REL_FIELDS),)


def test_validate_rejects_bad_hash() -> None:
    s = SemanticSchema()
    t = SemanticTensor(np.zeros((10, s.node_dim), np.float32), np.zeros(len(REL_FIELDS), np.float32),
                       np.zeros(len(MON_FIELDS), np.float32), schema_hash="deadbeef")
    try:
        t.validate(s)
    except AssertionError:
        return
    raise AssertionError("validate must reject a stale field-order hash")
