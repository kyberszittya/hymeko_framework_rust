"""R11.4B conditioned delivery BC — assimilating the certified scenario-specific teacher theta-schedules into one
coin/target/handoff-conditioned policy pi: descriptor -> theta in R^6, delivering strict K6 WITHOUT per-scenario
delivery-CEM and WITHOUT the deliverability oracle.

Three concerns, one per module:
  * ``dataset``  — deterministic teacher-replay theta-extraction + the input descriptor + the scenario-level split.
  * ``models``   — the ``ThetaPolicy`` Protocol and its impls (mean / nearest-schedule / ridge / small MLP).
  * ``evaluate`` — the closed-loop physical strict-K6 harness (no CEM/oracle/lookup; 0 safety regression).

The transport primitive, solver, ranking, and demonstration bank are READ-ONLY here (this package only consumes them).
"""
from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    FEATURE_NAMES,
    BcSample,
    bc_context,
    best_theta_full,
    descriptor,
    extract_sample,
    fresh_rig,
    reconstruct_capture,
)
from hymeko_rl.coin_delivery.delivery_bc.evaluate import (
    CLOSED_LOOP_CFG,
    r2_delivery,
    rollout_theta,
    theta_basin_survival,
)
from hymeko_rl.coin_delivery.delivery_bc.models import (
    THETA_HI,
    THETA_LO,
    MeanThetaPolicy,
    MlpBcPolicy,
    NearestSchedulePolicy,
    RidgePolicy,
    Standardizer,
    ThetaPolicy,
    clip_theta,
)

__all__ = [
    "FEATURE_NAMES", "BcSample", "bc_context", "best_theta_full", "descriptor", "extract_sample", "fresh_rig",
    "reconstruct_capture",
    "CLOSED_LOOP_CFG", "r2_delivery", "rollout_theta", "theta_basin_survival",
    "THETA_HI", "THETA_LO", "MeanThetaPolicy", "MlpBcPolicy", "NearestSchedulePolicy", "RidgePolicy",
    "Standardizer", "ThetaPolicy", "clip_theta",
]
