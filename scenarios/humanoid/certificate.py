"""CIP-HUM-01 external certificate suite (reward-independent, over state+trace).

Safety dominates success. NOTE: ``support_margin_maintained`` is VACUOUS on the
fixed-base HyMeKo humanoid (the pelvis is welded; falling is impossible). It is
kept in the contract for the eventual floating-base version but must be read as
"not genuinely exercised" here (see the report). The genuine safety guarantee is
``joint_limits_bounded``.
"""

from __future__ import annotations

from hymeko_control.cip.certificate import Certificate, CertificateSuite
from hymeko_control.language.schema_v0 import CertificateKind


def _stand_reach_recover(_state, trace) -> bool:
    """SUCCESS: reached RECOVER with the effector returned home, no divergence."""
    return bool(trace.provenance.get("recover_home_ok"))


def _joint_limits_bounded(_state, trace) -> bool:
    """SAFETY (genuine): joints in range, velocity bounded, no divergence."""
    return (
        bool(trace.provenance.get("limits_ok"))
        and not trace.provenance.get("diverged")
        and float(trace.provenance.get("max_qvel", 0.0)) < 40.0
    )


def _support_margin_maintained(_state, trace) -> bool:
    """SAFETY (VACUOUS on fixed base): torso upright. Not a genuine balance test."""
    return bool(trace.provenance.get("torso_upright", True))


def hum_certificate_suite() -> CertificateSuite:
    return CertificateSuite(certificates=(
        Certificate("stand_reach_recover", CertificateKind.SUCCESS, _stand_reach_recover),
        Certificate("joint_limits_bounded", CertificateKind.SAFETY, _joint_limits_bounded),
        Certificate("support_margin_maintained", CertificateKind.SAFETY,
                    _support_margin_maintained),
    ))
