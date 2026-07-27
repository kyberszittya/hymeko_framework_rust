"""CIP-PNP-01 external certificate suite.

Reward-independent predicates over (state, trace) only. They read the response
trace's summary provenance (computed by the adapter from the physics signals),
never a reward. Safety dominates success (enforced by ``CertificateSuite``).
"""

from __future__ import annotations

from hymeko_control.cip.certificate import Certificate, CertificateSuite
from hymeko_control.language.schema_v0 import CertificateKind


def _placed_and_settled(_state, trace) -> bool:
    """SUCCESS: object placed stably and settle dwell held."""
    return bool(trace.provenance.get("placed_stable"))


def _object_not_dropped(_state, trace) -> bool:
    """SAFETY: no post-lift ground contact and no divergence/death this option."""
    return not trace.provenance.get("any_death") and not trace.provenance.get("dropped")


def _contact_retained_in_carry(_state, trace) -> bool:
    """SAFETY: bilateral contact held throughout any CARRY option (vacuous otherwise)."""
    return bool(trace.provenance.get("carry_contact_ok", True))


def pnp_certificate_suite() -> CertificateSuite:
    """The frozen external certificate for CIP-PNP-01."""
    return CertificateSuite(certificates=(
        Certificate("placed_and_settled", CertificateKind.SUCCESS, _placed_and_settled),
        Certificate("object_not_dropped", CertificateKind.SAFETY, _object_not_dropped),
        Certificate("contact_retained_in_carry", CertificateKind.SAFETY,
                    _contact_retained_in_carry),
    ))
