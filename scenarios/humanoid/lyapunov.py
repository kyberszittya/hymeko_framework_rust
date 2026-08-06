"""COM-Lyapunov for the humanoid + the generic Lyapunov certificate.

Verifies the humanoid balance closed loop against the SAME reward-independent
Lyapunov conditions used on AIBO. The whole-body COM is the underactuated
coordinate; the Lyapunov function is a positive-definite energy in the balance
error (COM height loss + COM offset from support + COM velocity + torso tilt):

    V(s) = 1/2 [ w_h·(com_z − h_ref)² + w_xy·‖com_xy − support_xy‖²
                 + w_v·‖com_vel‖² + w_up·(1 − uprightness)² ]

V → 0 iff the COM holds its standing height over the support, at rest, upright.
A FALL (tip or collapse) makes V diverge, so the Lyapunov certificate rejects it.

NOTE: ``evaluate_lyapunov`` / ``lyapunov_certificate`` are generic and duplicate the
AIBO implementation across scenario branches -- a deliberate CORE-PROMOTION candidate
(generalizes ``stability_certificate``); unify at the core-promotion review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from hymeko_control.cip.certificate import Certificate
from hymeko_control.language.schema_v0 import CertificateKind

VFn = Callable[[dict], float]


@dataclass(frozen=True)
class HumanoidCOMLyapunov:
    """Whole-body COM Lyapunov energy over the balance error."""

    h_ref: float = 0.645        # MEASURED standing COM height (was mis-set to 0.818, a pelvis-top height)
    w_h: float = 4.0
    w_xy: float = 2.0
    w_v: float = 0.3
    w_up: float = 1.0

    def __call__(self, sig: dict) -> float:
        return 0.5 * (
            self.w_h * (float(sig.get("com_z", self.h_ref)) - self.h_ref) ** 2
            + self.w_xy * float(sig.get("com_xy_off", 0.0)) ** 2
            + self.w_v * float(sig.get("com_speed", 0.0)) ** 2
            + self.w_up * (1.0 - float(sig.get("uprightness", 1.0))) ** 2
        )


def evaluate_lyapunov(v_series: Sequence[float], *, descent_tol: float = 5e-3,
                      converge_eps: float = 0.05, min_descent_frac: float = 0.9) -> dict:
    """V >= 0, near-monotone descent (dV <= tol on >= min_descent_frac of steps),
    convergence (Vfinal <= eps) and net decrease. See the AIBO report for rationale."""
    vs = [float(v) for v in v_series]
    if len(vs) < 2:
        return {"passes": False, "reason": "too short"}
    steps = list(zip(vs, vs[1:]))
    frac = sum(1 for a, b in steps if b <= a + descent_tol) / len(steps)
    # Lyapunov stability = V >= 0, near-monotone non-increasing, BOUNDED (no growth
    # beyond max(V0, eps)), and converged. Bounded+converged (not strict net-decrease)
    # so a start-at-equilibrium trajectory (V ~ 0 throughout) also certifies stable.
    bounded = max(vs) <= max(vs[0], converge_eps) + descent_tol
    return {
        "V0": round(vs[0], 4), "Vfinal": round(vs[-1], 4), "Vmax": round(max(vs), 4),
        "descent_fraction": round(frac, 3), "nonnegative": min(vs) >= -1e-9,
        "converged": vs[-1] <= converge_eps, "bounded": bounded,
        "passes": bool(min(vs) >= -1e-9 and frac >= min_descent_frac
                       and vs[-1] <= converge_eps and bounded),
    }


def lyapunov_certificate(name: str, v_fn: VFn, **kw) -> Certificate:
    """Generic reward-independent CIP-0 SAFETY certificate over a trace (V by v_fn)."""

    def _fn(_state: Any, trace: Any) -> bool:
        return evaluate_lyapunov([v_fn(s) for s in trace.signals], **kw)["passes"]

    return Certificate(name, CertificateKind.SAFETY, _fn)
