r"""M2⁺ — HSTL runtime monitor of the certified basin: the online companion to the offline certificate.

The M2 certificate proves an OFFLINE set ``{V ≤ c}``; this watches an EXECUTING trajectory against
certificate-tied HSTL (HyMeKo/Hypergraph Signal Temporal Logic) safety specs and reports quantitative
robustness — a signed safety margin, live, with early warning before a fall.

Two specs, in robust-STL min/max semantics:
    G(cert_margin ≥ 0)   — always inside the certified sublevel, cert_margin = c − V(x)
    G(fall_margin > 0)    — always upright,                       fall_margin = fall_pitch − |pitch|
The running robustness is the min margin over the trace (the safety envelope); ``lead_steps`` is how many steps
the certificate margin goes negative BEFORE the actual fall — the monitor's early warning.

Reuse, not reinvention (§6.1): the pure-Python HTL evaluator ``hymeko_neuro.eval.htl`` is the reference backend.
The Rust ``hymeko_monitor`` crate is the fast version, but its PyO3 binding is NOT built in this venv, so
``make_monitor(..., "rust")`` is a documented slot behind the SAME ``MonitorBackend`` interface (mirroring
``coin_carry_monitor.py``'s §2A/§2B pattern) — no claim it runs here until the binding is built and parity-tested.

# Preconditions: a fitted certificate exposing ``value(x)``; a ``CentroidalConfig``. # Postconditions:
#   deterministic; the monitor's robustness is the robust-STL margin of the executed trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from hymeko_neuro.eval.htl import HtlMonitor, HypergraphEvent

from scenarios.humanoid.centroidal import CentroidalConfig, centroidal_step

SPEC_STAY_CERTIFIED = "G(cert_margin >= 0)"      # always inside {V ≤ c}
SPEC_NEVER_FALL = "G(fall_margin > 0)"           # always upright


@runtime_checkable
class MonitorBackend(Protocol):
    """The narrow runtime-monitor interface (Python impl below; the Rust STL crate is the same-interface slot)."""

    def observe(self, t: float, signals: Mapping[str, float]) -> float: ...   # append event, return robustness
    def robustness(self) -> float: ...
    def satisfied(self) -> bool: ...


class PythonHtlBackend:
    """Reference backend over the pure-Python HTL evaluator (robust-STL min/max semantics)."""

    def __init__(self, spec: str, horizon: int = 4096) -> None:
        self._monitor = HtlMonitor(spec, horizon=horizon)

    def observe(self, t: float, signals: Mapping[str, float]) -> float:
        return float(self._monitor.observe(HypergraphEvent(t=float(t), scalar_signals=dict(signals))))

    def robustness(self) -> float:
        return float(self._monitor.robustness())

    def satisfied(self) -> bool:
        return bool(self._monitor.satisfied())


def make_monitor(spec: str, backend: str = "python", horizon: int = 4096) -> MonitorBackend:
    """Backend factory (mirrors ``coin_carry_monitor.make_monitor``): ``python`` ships; ``rust`` is a built-slot."""
    if backend == "python":
        return PythonHtlBackend(spec, horizon)
    if backend == "rust":
        raise NotImplementedError(
            "the Rust hymeko_monitor STL backend is not built in this venv (import hymeko_monitor is empty). "
            "Build the PyO3 binding, then wire a RustHtlBackend behind this same MonitorBackend interface and "
            "parity-test it against PythonHtlBackend before use.")
    raise ValueError(f"unknown monitor backend {backend!r} (use 'python'; 'rust' once its binding is built)")


def certificate_signals(state: np.ndarray, certificate, level: float, cfg: CentroidalConfig) -> "dict[str, float]":
    """Map one reduced state ``(z, ż, L, pitch)`` to the HSTL scalar signals: the certificate + fall margins."""
    v = float(certificate.value(state.reshape(1, 4))[0])
    return {"V": v, "cert_margin": level - v, "fall_margin": cfg.fall_pitch - abs(float(state[3]))}


@dataclass(frozen=True)
class MonitorResult:
    """Verdict of one monitored run: satisfaction, the robust-STL margins, and the early-warning lead."""

    satisfied_certified: bool
    satisfied_upright: bool
    min_cert_margin: float           # robust-STL robustness of G(cert_margin ≥ 0) = min over the trace
    min_fall_margin: float
    fell: bool
    fall_step: int                   # step of the actual fall (−1 if none)
    warn_step: int                   # first step the certificate margin goes negative (−1 if none)
    lead_steps: int                  # fall_step − warn_step: steps of early warning (−1 if no fall)
    steps: int


def monitor_trajectory(x0: np.ndarray, certificate, level: float, cfg: CentroidalConfig,
                       backend: str = "python") -> MonitorResult:
    r"""Run one centroidal trajectory from ``x0`` through the HSTL runtime monitor. # Preconditions: ``x0`` shape (4,).

    Watches ``G(cert_margin ≥ 0)`` and ``G(fall_margin > 0)`` online; returns the robust-STL margins, whether the
    run fell, and the early-warning lead (how far ahead the certificate margin flagged the fall).
    """
    if x0.shape != (4,):
        raise ValueError(f"x0 must be a single reduced state of shape (4,), got {x0.shape}")
    steps = int(round(cfg.horizon / cfg.dt))
    cert_mon = make_monitor(SPEC_STAY_CERTIFIED, backend, steps + 1)
    fall_mon = make_monitor(SPEC_NEVER_FALL, backend, steps + 1)
    state = x0.astype(float).copy()
    fell, fall_step, warn_step = False, -1, -1
    for i in range(steps):
        sig = certificate_signals(state, certificate, level, cfg)
        cert_mon.observe(i, sig)
        fall_mon.observe(i, sig)
        if warn_step < 0 and sig["cert_margin"] < 0.0:
            warn_step = i
        state = centroidal_step(state.reshape(1, 4), i * cfg.dt, cfg)[0]
        if not fell and abs(float(state[3])) > cfg.fall_pitch:
            fell, fall_step = True, i
    return MonitorResult(
        satisfied_certified=cert_mon.satisfied(), satisfied_upright=fall_mon.satisfied(),
        min_cert_margin=cert_mon.robustness(), min_fall_margin=fall_mon.robustness(),
        fell=fell, fall_step=fall_step, warn_step=warn_step,
        lead_steps=(fall_step - warn_step) if (fell and warn_step >= 0) else -1, steps=steps)
