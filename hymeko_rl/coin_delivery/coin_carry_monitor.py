"""§2A — the delivery verdict as an online trace-MONITOR whose semantics come from the `.hymeko`, not Python constants.

The executor emits only a per-step `TraceSample` of physical signals; a `MonitorBackend` computes the single verdict
(strict progression, containment enter/exit, handoff, K6/delivered, terminal, failure reason). The tolerances and the K6
dwell are read from `coin_carry_option_v1.hymeko`'s `@certificate` node — so the same hypergraph description that drives the
phase automaton (§1) also defines the certificate (§2A). The narrow `MonitorBackend` interface lets §2B swap a
`RustStlMonitorBackend` (PyO3 over the real `hymeko_monitor` STL crate) behind the SAME spec without touching the executor.

Verdict contract (frozen, matches `coin_rl_env`): strict_ok = dtz≤center_tol ∧ speed<settle_vel; strict = strict+1 if
strict_ok else 0; handoff = strict≥1; containment = dtz≤center_tol; K6 = max_strict≥held_dwell ∧ touched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hymeko_rl.coin_delivery.coin_carry_fsm import load_carry_automaton


@dataclass(frozen=True)
class MonitorSpec:
    """Certificate tolerances + dwell, sourced from the `.hymeko` `@certificate` node (NOT hard-coded)."""

    center_tol: float
    settle_vel: float
    held_dwell: int
    entry_tol: float


def load_carry_monitor_spec(path: str | None = None) -> MonitorSpec:
    """Read the certificate semantics from the carry-option `.hymeko` (reuses the §1 automaton parser).

    # Errors ``KeyError`` if the `@certificate` fields are absent from the profile."""
    spec = load_carry_automaton() if path is None else load_carry_automaton(path)
    p = spec.params
    return MonitorSpec(center_tol=p["center_tol"], settle_vel=p["settle_vel"],
                       held_dwell=int(p["held_dwell"]), entry_tol=p["entry_tol"])


@dataclass
class TraceSample:
    """One step of physical signals the executor emits. The monitor derives ALL certificate state from these."""

    dtz: float
    speed: float
    touched: bool
    contact: bool
    terminated: bool


@runtime_checkable
class MonitorBackend(Protocol):
    """The narrow monitor interface (§2A = Python impl below; §2B = Rust STL impl behind the SAME methods)."""

    def reset(self, spec: MonitorSpec, initial_dtz: float, initial_touched: bool = False) -> None: ...

    def observe(self, sample: TraceSample) -> dict: ...   # returns the per-step delta {strict, exited, handoff_now}

    def verdict(self) -> dict: ...

    def snapshot(self) -> dict: ...


class PythonTaskMonitorBackend:
    """Reference Python backend — reproduces the frozen strict-dwell certificate from the trace, emitting the full temporal
    event sequence (containment enter/exit, handoff, delivered) so parity is checked on the whole trace, not just final K6."""

    def __init__(self):
        self.spec: MonitorSpec | None = None
        self.strict = 0; self.max_strict = 0; self.touched = False
        self.was_contained = False; self.contain_exit = 0; self.handoff = False; self.delivered = False
        self.terminal = False; self.failure_reason: str | None = None
        self.events: list[tuple[int, str]] = []; self._t = 0

    def reset(self, spec: MonitorSpec, initial_dtz: float, initial_touched: bool = False) -> None:
        self.spec = spec
        self.strict = 0; self.max_strict = 0; self.touched = bool(initial_touched)
        self.was_contained = initial_dtz <= spec.center_tol
        self.contain_exit = 0; self.handoff = False; self.delivered = False
        self.terminal = False; self.failure_reason = None; self.events = []; self._t = 0

    def observe(self, s: TraceSample) -> dict:
        sp = self.spec
        self._t += 1
        strict_ok = (s.dtz <= sp.center_tol) and (s.speed < sp.settle_vel)
        self.strict = self.strict + 1 if strict_ok else 0
        self.max_strict = max(self.max_strict, self.strict)
        self.touched = self.touched or bool(s.touched)
        contained = s.dtz <= sp.center_tol
        exited = self.was_contained and not contained
        if exited:
            self.contain_exit += 1; self.events.append((self._t, "containment_exit"))
        elif contained and not self.was_contained:
            self.events.append((self._t, "containment_enter"))
        self.was_contained = contained
        handoff_now = False
        if self.strict >= 1 and not self.handoff:
            self.handoff = True; handoff_now = True; self.events.append((self._t, "handoff"))
        if self.max_strict >= sp.held_dwell and self.touched and not self.delivered:
            self.delivered = True; self.events.append((self._t, "delivered"))
        if s.terminated:
            self.terminal = True
        return {"strict": self.strict, "exited": int(exited), "handoff_now": int(handoff_now)}

    def verdict(self) -> dict:
        k6 = int(self.max_strict >= self.spec.held_dwell and self.touched)
        return {"k6": k6, "reached_handoff": int(self.max_strict >= 1), "contain_exit_ct": self.contain_exit,
                "max_strict": self.max_strict, "terminal": int(self.terminal),
                "failure_reason": self.failure_reason, "events": tuple(self.events)}

    def snapshot(self) -> dict:
        return {"strict": self.strict, "max_strict": self.max_strict, "contain_exit": self.contain_exit,
                "handoff": int(self.handoff), "delivered": int(self.delivered)}


def make_monitor(backend: str = "python") -> MonitorBackend:
    """Backend factory. §2A ships 'python'; §2B will add 'rust' (PyO3 over hymeko_monitor) behind the same interface."""
    if backend == "python":
        return PythonTaskMonitorBackend()
    raise ValueError(f"unknown monitor backend {backend!r} (only 'python' in §2A; 'rust' arrives with §2B)")
