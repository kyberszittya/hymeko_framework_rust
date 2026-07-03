"""Gradient-field / error-space evolvens telemetry.

The "evolvens" framing (involute, in differential geometry: the curve
traced by a point on a taut string as it is unwound from another curve)
captures the way the **gradient field** and the **error field** twist
around each other through the pool-scatter primitive when the entropy
Hamilton rotor is active:

    scatter_h            ─┐
        │     (forward)   │
        ▼  M ⊗            │
    scatter_h_modulated   ▼
        │  W_back         W_back
        ▼                 │
       out ───── grad_out ┘
                  │
                  ▼  Wb.T
            grad_path_h
                  │  M* ⊗   (reverse rotor)
                  ▼
            grad_scatter_h
                  │  (closed-form bwd → dQ, dK, dV)
                  ▼
                grad_x

The relevant scalars at every backward call are:

    rotor_angle_rad         arccos(M[:, 0].mean())   -- the twist
    scatter_norm            ||scatter_h||_F
    scatter_mod_norm        ||M ⊗ scatter_h||_F        ≈ scatter_norm
    grad_path_norm          ||grad_path_h||_F          -- error post-W_back
    grad_scatter_norm       ||M* ⊗ grad_path_h||_F     ≈ grad_path_norm
    grad_x_norm             ||grad_x||_F               -- field at input
    grad_out_norm           ||grad_out||_F             -- field at output
    cos_grad_x_grad_out     cos angle (layer's twist on grad)
    cos_scatter_pre_post    cos angle scatter_h vs scatter_h_modulated
                            (the forward-rotor's effective angle in feat-space)

Usage::

    with EvolventTelemetry() as t:
        for step in training_loop():
            loss.backward()
    for r in t.records:
        print(r.step, r.rotor_angle_rad, r.cos_grad_x_grad_out)

Stream to disk::

    with EvolventTelemetry(out_path="evolvens.jsonl"):
        ...

Outside a ``with`` block the emit path is a no-op (single contextvar read).
"""
from __future__ import annotations

import json
import math
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

import torch


# ---- Record schema --------------------------------------------------------

@dataclass
class EvolventRecord:
    """One backward-pass observation of the gradient/error fusion."""
    step: int
    rotor_angle_rad: float | None
    scatter_norm: float
    scatter_mod_norm: float
    grad_path_norm: float
    grad_scatter_norm: float
    grad_x_norm: float
    grad_out_norm: float
    cos_grad_x_grad_out: float
    cos_scatter_pre_post: float


# ---- Context-managed sink -------------------------------------------------

_ACTIVE: ContextVar["EvolventTelemetry | None"] = ContextVar(
    "_evolvent_active", default=None,
)


class EvolventTelemetry:
    """Context manager + records list (+ optional JSONL stream).

    The sink is published via a ``contextvars.ContextVar``: nested ``with``
    blocks push/pop transparently, and the dispatch is thread- and
    coroutine-safe. Emit is a single contextvar read when no sink is active.
    """

    def __init__(self, out_path: str | Path | None = None) -> None:
        self.records: list[EvolventRecord] = []
        self._out_path: Path | None = Path(out_path) if out_path else None
        self._fh: IO[str] | None = None
        self._token = None
        self._next_step = 0

    # ── lifecycle ────────────────────────────────────────────────
    def __enter__(self) -> "EvolventTelemetry":
        if self._out_path is not None:
            self._fh = open(self._out_path, "w")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ── emit ─────────────────────────────────────────────────────
    def emit(self, record: EvolventRecord) -> None:
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(asdict(record)) + "\n")
            self._fh.flush()

    def next_step(self) -> int:
        s = self._next_step
        self._next_step += 1
        return s

    # ── post-hoc analysis helpers ────────────────────────────────
    def summary(self) -> dict:
        """Per-field {mean, std, min, max} across recorded steps."""
        if not self.records:
            return {"n_steps": 0}
        keys = [k for k in asdict(self.records[0]).keys()
                if k != "step"]
        out: dict = {"n_steps": len(self.records)}
        for k in keys:
            vals = [getattr(r, k) for r in self.records
                    if getattr(r, k) is not None]
            if not vals:
                out[k] = None
                continue
            t = torch.tensor(vals, dtype=torch.float64)
            out[k] = {
                "mean": float(t.mean().item()),
                "std":  float(t.std().item()) if len(t) > 1 else 0.0,
                "min":  float(t.min().item()),
                "max":  float(t.max().item()),
            }
        return out


def is_active() -> bool:
    """True iff an :class:`EvolventTelemetry` is currently in scope."""
    return _ACTIVE.get() is not None


# ---- Emission entry point (called from pool_scatter backward) -------------

def _fnorm(t: torch.Tensor) -> float:
    return float(t.detach().float().norm().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    denom = af.norm() * bf.norm()
    if denom.item() < 1e-30:
        return 0.0
    return float(((af * bf).sum() / denom).item())


def active_sink() -> "EvolventTelemetry | None":
    """Return the currently-active sink, or ``None``.

    Use this at forward-time (in the main thread, where the contextvar is set)
    to snapshot the sink onto an autograd ``ctx``. PyTorch's backward runs in
    a worker thread that does NOT inherit the contextvar.
    """
    return _ACTIVE.get()


def emit_backward_record(
    rotor_M: torch.Tensor | None,
    scatter_h: torch.Tensor,
    scatter_h_modulated: torch.Tensor,
    grad_path_h: torch.Tensor,
    grad_scatter_h: torch.Tensor,
    grad_x: torch.Tensor,
    grad_out: torch.Tensor,
    sink: "EvolventTelemetry | None" = None,
) -> None:
    """Compute the per-step evolvens metrics and push to the given sink.

    If ``sink`` is None, falls back to the contextvar-active sink (only
    visible from the thread that set it). Pass the sink explicitly when
    calling from an autograd backward worker thread.
    """
    if sink is None:
        sink = _ACTIVE.get()
    if sink is None:
        return

    rotor_angle: float | None = None
    if rotor_M is not None and rotor_M.numel() > 0:
        # For pure-imaginary axis n with |n|=1 we built M = (cos θ, sin θ·n);
        # per quaternion block θ_q = arccos(M[q, 0]). Report the mean across
        # blocks -- one scalar per step is enough at this granularity.
        m0_mean = rotor_M[:, 0].detach().float().mean().clamp(-1.0, 1.0)
        rotor_angle = math.acos(float(m0_mean.item()))

    rec = EvolventRecord(
        step=sink.next_step(),
        rotor_angle_rad=rotor_angle,
        scatter_norm=_fnorm(scatter_h),
        scatter_mod_norm=_fnorm(scatter_h_modulated),
        grad_path_norm=_fnorm(grad_path_h),
        grad_scatter_norm=_fnorm(grad_scatter_h),
        grad_x_norm=_fnorm(grad_x),
        grad_out_norm=_fnorm(grad_out),
        cos_grad_x_grad_out=_cos(grad_x, grad_out),
        cos_scatter_pre_post=_cos(scatter_h, scatter_h_modulated),
    )
    sink.emit(rec)


__all__ = [
    "EvolventRecord",
    "EvolventTelemetry",
    "emit_backward_record",
    "is_active",
]
