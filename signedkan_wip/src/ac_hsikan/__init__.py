"""AC-HSiKAN: Attention-Cycle HSiKAN.

Signed-cycle pool as a transformer self-attention substitute.

See:
  docs/plans/2026-06-04-ac-hsikan/plan.{md,tex,pdf,tikz,mmd}

Public API:
    from signedkan_wip.src.ac_hsikan import (
        AcHsikanConfig, SignHead, AcHsikanLayer, AcHsikanClassifier,
    )
"""
from .config import AcHsikanConfig
from .layer import AcHsikanLayer
from .model import AcHsikanClassifier
from .sign_head import QuaternionSignHead, SignHead, build_sign_head
from .telemetry import EvolventRecord, EvolventTelemetry

__all__ = ["AcHsikanConfig", "AcHsikanClassifier", "AcHsikanLayer",
           "QuaternionSignHead", "SignHead", "build_sign_head",
           "EvolventRecord", "EvolventTelemetry"]
