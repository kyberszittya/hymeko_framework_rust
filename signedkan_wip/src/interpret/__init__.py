"""HSIKAN interpretability tooling — 2026-05-20.

Fuzzy signature view: makes HSIKAN's per-cycle reasoning chain
observable. For a query edge, ``extract_signature`` returns the
full set of cycles that touched it, each tagged with its sign-
product vote, fuzzy membership weight, and per-cycle embedding.

See `docs/plans/2026-05-20-fuzzy-signature-view/plan.pdf` for
the design.
"""
from .fuzzy_signature import (
    CycleContribution,
    FuzzySignature,
    extract_signature,
    plot_signature,
)
from .gomb_signature import (
    GombCycleContribution,
    GombFuzzySignature,
    extract_gomb_signature,
    plot_gomb_signature,
)

__all__ = [
    "CycleContribution",
    "FuzzySignature",
    "extract_signature",
    "plot_signature",
    "GombCycleContribution",
    "GombFuzzySignature",
    "extract_gomb_signature",
    "plot_gomb_signature",
]
