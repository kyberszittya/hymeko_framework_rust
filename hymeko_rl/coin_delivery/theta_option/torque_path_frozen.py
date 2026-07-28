"""R10.2 Stage 2 (Boundary 4) — the FROZEN structured-option exploration coordinate for TD3.

The 15-D torque-path coordinate (`torque_path_option`), its per-dimension normalization ``D``, and the episodic
exploration scale ``sigma`` are frozen here from the Boundary-3 admissibility review (commit ``be4cf935``). They must not
be retuned during training; a drift-guard test recomputes ``D`` and compares it to :data:`FROZEN_D`.

Honest provenance of the decision — the strict pre-registered admissibility gate did **NOT** formally PASS; this is a
*research-review-accepted* training distribution (``STRUCTURED_THETA_EXPLORATION_REVIEW_ACCEPTED``): at ``sigma=0.05`` every
seed produced K6 positives and physically-distinct safe negatives with **zero physical safety violations (0/96)**; the sole
exception is a rare (1/96) SAFE boundary-route variation (``reset != 1``). Do not rewrite this as a formal strict PASS.
"""
from __future__ import annotations

import numpy as np

# Frozen per-dimension normalization D (equalises the physical effect per unit z), from `freeze_normalization` on the
# Boundary-3 sensitivity audit at the frozen medoid scaffold (commit be4cf935). Full precision; drift-guarded by a test.
FROZEN_D: "tuple[float, ...]" = (
    1.0, 0.5832159355552171, 0.5, 2.0, 2.0, 1.2288103986489867, 1.7171496112568918, 1.542056507947858,
    2.0, 0.5, 0.5, 1.0950502640421496, 0.9746322914071653, 0.7705626678950782, 0.8876543728311997)

SIGMA: float = 0.05        # the frozen episodic exploration scale (theta = sigma * D * z); approved for training only

REVIEW_DECISION = {
    "verdict": "STRUCTURED_THETA_EXPLORATION_REVIEW_ACCEPTED",
    "sigma": SIGMA,
    "strict_preregistered_admissibility": False,     # the strict zero-boundary-regression gate did NOT pass
    "reviewed_training_admissibility": True,          # research-review accepted as a training distribution
    "exception": "1/96 safe boundary-route variation (reset != 1)",
    "physical_safety_violations": "0/96",
    "source_commit": "be4cf935",
    "reset_ne_1_contract": ["NOT unsafe", "NOT in positive replay", "terminal boundary-penalty",
                            "own boundary_route_variation category", "counts as FAILURE in evaluation"],
}


def frozen_normalization() -> np.ndarray:
    """The frozen per-dimension normalization ``D`` as a length-15 array. # Postconditions: matches :data:`FROZEN_D`."""
    return np.array(FROZEN_D, dtype=float)
