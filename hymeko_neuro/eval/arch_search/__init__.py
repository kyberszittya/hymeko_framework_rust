"""Architecture-search algorithms (MSG / ABB / SSG) — 2026-05-21.

P-graph-derived discrete search machinery, applied to the
model-architecture axis-product:

- **MSG (Maximum Structure Generation):** Cartesian product over
  the architectural axes. Each output is one
  :class:`ArchCandidate`.
- **ABB (Algorithmic Bounding Branch):** prune candidates whose
  predicted GPU memory / wall time / parameter count exceeds the
  configured caps — BEFORE they get launched. Avoids wasting
  cycle-enum / GPU-startup on cells we already know will OOM.
- **SSG (Subset Structure Generation):** filter to a Pareto-
  dominant subset along (wall, expected-lift). Optional; useful
  for report-quality runs where structurally-dominated candidates
  add noise.

See `docs/plans/2026-05-21-msg-abb-arch-search/plan.pdf` for the
design and the bound-prediction calibration.
"""
from .abb import (
    ArchCandidate, msg_enumerate, abb_prune, ssg_pareto,
)

__all__ = [
    "ArchCandidate",
    "msg_enumerate",
    "abb_prune",
    "ssg_pareto",
]
