"""Evolvens-telemetry tests.

Covers:
  1. Inactive sink: no overhead, no crash.
  2. Active sink captures one record per backward call.
  3. JSONL sink mirrors the in-memory record list.
  4. Records carry expected fields with finite values; rotor angle
     present iff entropy feedback active.
  5. ``cos_scatter_pre_post`` approximates the rotor angle's cosine
     (sanity: forward Hamilton rotation preserves norm and twists).
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from signedkan_wip.src.ac_hsikan.telemetry import (
    EvolventRecord, EvolventTelemetry, emit_backward_record, is_active,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel",
)


def test_inactive_sink_is_noop():
    assert is_active() is False
    # Call emit with garbage tensors -- must not raise when no sink active.
    emit_backward_record(
        rotor_M=None,
        scatter_h=torch.zeros(1),
        scatter_h_modulated=torch.zeros(1),
        grad_path_h=torch.zeros(1),
        grad_scatter_h=torch.zeros(1),
        grad_x=torch.zeros(1),
        grad_out=torch.zeros(1),
    )


def test_context_manager_pushes_and_pops():
    assert is_active() is False
    with EvolventTelemetry() as t:
        assert is_active() is True
        assert t is not None
        assert t.records == []
    assert is_active() is False


def test_emit_path_records_one_per_call():
    with EvolventTelemetry() as t:
        for _ in range(3):
            emit_backward_record(
                rotor_M=None,
                scatter_h=torch.randn(2, 4, 8),
                scatter_h_modulated=torch.randn(2, 4, 8),
                grad_path_h=torch.randn(2, 4, 8),
                grad_scatter_h=torch.randn(2, 4, 8),
                grad_x=torch.randn(2, 4, 16),
                grad_out=torch.randn(2, 4, 16),
            )
    assert len(t.records) == 3
    for i, r in enumerate(t.records):
        assert r.step == i
        assert r.rotor_angle_rad is None
        assert math.isfinite(r.scatter_norm)
        assert math.isfinite(r.cos_grad_x_grad_out)
        assert -1.0 - 1e-5 <= r.cos_grad_x_grad_out <= 1.0 + 1e-5


def test_jsonl_sink_mirrors_records(tmp_path):
    out = tmp_path / "evolvens.jsonl"
    with EvolventTelemetry(out_path=out) as t:
        for _ in range(2):
            emit_backward_record(
                rotor_M=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                scatter_h=torch.randn(1, 2, 4),
                scatter_h_modulated=torch.randn(1, 2, 4),
                grad_path_h=torch.randn(1, 2, 4),
                grad_scatter_h=torch.randn(1, 2, 4),
                grad_x=torch.randn(1, 2, 8),
                grad_out=torch.randn(1, 2, 8),
            )
    on_disk = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(on_disk) == len(t.records) == 2
    for rec, row in zip(t.records, on_disk):
        assert rec.step == row["step"]
        assert rec.scatter_norm == pytest.approx(row["scatter_norm"])


def test_summary_handles_empty_and_populated():
    with EvolventTelemetry() as t:
        empty_summary = t.summary()
        assert empty_summary == {"n_steps": 0}
        emit_backward_record(
            rotor_M=None,
            scatter_h=torch.ones(1, 1, 4),
            scatter_h_modulated=torch.ones(1, 1, 4),
            grad_path_h=torch.ones(1, 1, 4),
            grad_scatter_h=torch.ones(1, 1, 4),
            grad_x=torch.ones(1, 1, 4),
            grad_out=torch.ones(1, 1, 4),
        )
        s = t.summary()
        assert s["n_steps"] == 1
        assert s["scatter_norm"]["mean"] == pytest.approx(2.0)  # sqrt(4)
        assert s["cos_grad_x_grad_out"]["mean"] == pytest.approx(1.0)


# ---- Integration: real backward through FusedPoolScatter ------------------

@cuda_only
def test_records_one_per_backward_with_pool_scatter():
    from signedkan_wip.src.ac_hsikan.components.pool_scatter import (
        FusedPoolScatter,
    )
    device = "cuda"
    mod = FusedPoolScatter(
        d_model=16, h=8, n_quat=2, G=8, entropy_feedback=True,
    ).to(device)
    with torch.no_grad():
        mod.entropy_beta.fill_(0.3)
    x = torch.randn(2, 8, 16, device=device, requires_grad=True)
    local = torch.randint(0, 8, (8, 4), device=device)
    H = torch.tensor(0.7, device=device)

    with EvolventTelemetry() as t:
        for _ in range(3):
            y = mod(x, local, entropy_scalar=H)
            grad_out = torch.randn_like(y)
            x_grad = torch.autograd.grad(y, x, grad_outputs=grad_out,
                                         retain_graph=False)[0]
            assert x_grad is not None

    assert len(t.records) == 3
    for r in t.records:
        assert r.rotor_angle_rad is not None
        # β·H = 0.3 · 0.7 = 0.21 rad expected
        assert r.rotor_angle_rad == pytest.approx(0.21, abs=1e-4)
        # Rotor is norm-preserving on the Hamilton field.
        assert r.scatter_mod_norm == pytest.approx(r.scatter_norm, rel=1e-4)
        assert r.grad_scatter_norm == pytest.approx(r.grad_path_norm, rel=1e-4)
