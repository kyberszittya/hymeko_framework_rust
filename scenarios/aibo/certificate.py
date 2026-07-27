"""CIP-AIBO-01 external certificate suite (reward-independent, over state+trace).

SIMULATION certificates. Safety dominates success. ``no_fall`` and
``speed_bounded_at_stop`` are genuine physics predicates on the simulated body.
"""

from __future__ import annotations

from hymeko_control.cip.certificate import Certificate, CertificateSuite
from hymeko_control.language.schema_v0 import CertificateKind


def _approach_align_stop(_state, trace) -> bool:
    """SUCCESS: waypoint reached, body halted, held for dwell (at the HOLD option)."""
    p = trace.provenance
    return bool(p.get("held") and p.get("reached") and p.get("halted"))


def _no_fall(_state, trace) -> bool:
    """SAFETY: torso stayed upright, no divergence, this option."""
    return not trace.provenance.get("fell")


def _speed_bounded_at_stop(_state, trace) -> bool:
    """SAFETY: during STOP/HOLD the planar body speed is below threshold (vacuous elsewhere)."""
    if trace.provenance.get("mode") in ("STOP", "HOLD"):
        return bool(trace.provenance.get("halted"))
    return True


def aibo_certificate_suite() -> CertificateSuite:
    return CertificateSuite(certificates=(
        Certificate("approach_align_stop", CertificateKind.SUCCESS, _approach_align_stop),
        Certificate("no_fall", CertificateKind.SAFETY, _no_fall),
        Certificate("speed_bounded_at_stop", CertificateKind.SAFETY, _speed_bounded_at_stop),
    ))
