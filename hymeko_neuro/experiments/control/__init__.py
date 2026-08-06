"""HSIKAN vehicle-control benchmark.

A bicycle-kinematic lateral-tracking task with four controllers:

- ``LQRController``        — optimal-linear (Riccati solve at construct).
- ``PurePursuitController`` — classical geometric path tracker.
- ``MPCController``        — nonlinear model-predictive (scipy.optimize).
- ``HSIKANController``     — learned (windowed σ-cycle) policy.

See ``docs/plans/2026-05-21-hsikan-timeseries-control/`` for the
design.  Signed-cycle reading of vehicle control: σ_t = sign(lateral
error at step t).  Π σ_t over a window = "consistently on one side
of the path" → HSIKAN's natural representation.
"""

from .bicycle import (
    BicycleParams,
    BicycleState,
    BicycleVehicle,
)
from .controllers import (
    HSIKANController,
    LQRController,
    MPCController,
    PurePursuitController,
)
from .tracks import (
    Track,
    sinusoid_track,
    s_curve_track,
    straight_track,
)

__all__ = [
    "BicycleParams", "BicycleState", "BicycleVehicle",
    "Track", "sinusoid_track", "s_curve_track", "straight_track",
    "LQRController", "PurePursuitController", "MPCController",
    "HSIKANController",
]
