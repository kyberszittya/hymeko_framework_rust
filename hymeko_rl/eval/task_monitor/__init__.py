"""Hierarchical formal task monitor for the galambos coin-delivery task.

An UNLEARNED, position-based external verifier (conceptually a top-down camera): NOT the reward RL optimises,
NOT the metrics we report — a deterministic verifier over trajectories. The contract (object roles, geometric /
contact-quality / temporal predicates, success formula) is generated from the HyMeKo task contract carried by
the env, so roles / zone / contacts / success stay single-source-of-truth.

Hierarchy (:class:`TaskMonitor` composes the submonitors):

1. :class:`GeometryMonitor` — coin-target + fingertip-coin + arm-body-coin distances, zone membership.
2. :class:`ApproachMonitor` — fingertips approach the coin before it is displaced, not pushed away.
3. :class:`ContactMonitor` — left/right fingertip contact, both-fingertip engagement, duration + timing.
4. :class:`ProgressMonitor` — distance decreases, fingertip- vs body-attributed progress.
5. :class:`DeliveryMonitor` — coin enters zone and holds for k steps (stable delivery).
6. :class:`AntiExploitMonitor` — body-driven / body-assisted / no-engagement delivery.
7. :class:`RewardConsistencyMonitor` — reward-vs-monitor and critic-vs-monitor misalignment (cross-cutting).
8. :class:`TensorContractMonitor` — obs / privileged-z / reward-feature / contact-quality schema + field order.

:class:`PipelineSchemaLedger` (with :class:`TransitionSchema`) is the *live* sibling of monitor 8: it guards the
rollout → replay-serialize → replay-load → critic-train → eval path at run time, aborting on the first schema /
field-order drift (wired into ``train_offpolicy``). :class:`PolicyProvenanceLedger` is its identity counterpart:
the schema guard proves the *tensors* are right, the provenance guard proves the *policy* (checkpoint md5, param
hash, action checksum, anchor identity) is the one the run claims — a scripted controller can never masquerade as
a learned actor.

Use the monitor as an external evaluator first (diagnostics, acceptance tests, reward-/critic-vs-monitor
misalignment), NOT inside the reward.
"""
from __future__ import annotations

from typing import Any

from .cip_export import CIP_DEFAULTS, CIP_VARIABLES, CipExport, export_cip_variables
from .consistency import RewardConsistencyMonitor, TensorContractMonitor
from .context import MonitorContext, SupportsDistanceSeries, contiguous_runs, record_trajectory
from .contract import MonitorContract, TensorContract
from .metaworld import (
    CoffeePushContext,
    CoffeePushMonitor,
    CoffeePushProgressMonitor,
    CoffeePushSubmonitor,
    CoffeePushSuccessMonitor,
    CoffeePushVerdict,
    DialTurnContext,
    DialTurnMonitor,
    DialTurnOvershootMonitor,
    DialTurnProgressMonitor,
    DialTurnSubmonitor,
    DialTurnSuccessMonitor,
    DialTurnVerdict,
    MetaWorldSubmonitor,
)
from .pipeline import (
    STAGES,
    PipelineSchemaError,
    PipelineSchemaLedger,
    TransitionField,
    TransitionSchema,
    flat_dim,
)
from .provenance import (
    LEARNED_ROLES,
    PolicyProvenance,
    PolicyProvenanceError,
    PolicyProvenanceLedger,
    PolicyRole,
    action_checksum,
    canonical_obs_batch,
    file_md5,
    param_hash,
)
from .root import TaskMonitor, TaskVerdict
from .submonitors import (
    AntiExploitMonitor,
    ApproachMonitor,
    ContactMonitor,
    DeliveryMonitor,
    GeometryMonitor,
    ProgressMonitor,
    StagnationMonitor,
    StagnationVerdict,
    SubVerdict,
    TrajectoryMonitor,
    default_submonitors,
)

__all__ = [
    "MonitorContract", "TensorContract", "MonitorContext", "record_trajectory", "contiguous_runs",
    "SubVerdict", "TrajectoryMonitor", "GeometryMonitor", "ApproachMonitor", "ContactMonitor",
    "ProgressMonitor", "DeliveryMonitor", "AntiExploitMonitor", "StagnationMonitor", "StagnationVerdict",
    "default_submonitors",
    "RewardConsistencyMonitor", "TensorContractMonitor", "TaskMonitor", "TaskVerdict",
    "PipelineSchemaLedger", "PipelineSchemaError", "TransitionSchema", "TransitionField", "flat_dim", "STAGES",
    "PolicyProvenanceLedger", "PolicyProvenanceError", "PolicyProvenance", "PolicyRole", "LEARNED_ROLES",
    "file_md5", "param_hash", "action_checksum", "canonical_obs_batch",
    "export_cip_variables", "CipExport", "CIP_VARIABLES", "CIP_DEFAULTS",
    "SupportsDistanceSeries", "MetaWorldSubmonitor",
    "CoffeePushMonitor", "CoffeePushVerdict", "CoffeePushContext", "CoffeePushSubmonitor",
    "CoffeePushProgressMonitor", "CoffeePushSuccessMonitor",
    "DialTurnMonitor", "DialTurnVerdict", "DialTurnContext", "DialTurnSubmonitor",
    "DialTurnProgressMonitor", "DialTurnSuccessMonitor", "DialTurnOvershootMonitor",
    "monitor_policy",
]


def monitor_policy(make_env: Any, make_action_fn: Any, n: int, seed0: int = 9000) -> dict[str, Any]:
    """Back-compat convenience: aggregate the root :class:`TaskMonitor` over ``n`` episodes of a policy."""
    env = make_env()
    mon = TaskMonitor.from_env(env)
    env.close()
    return mon.evaluate_policy(make_env, make_action_fn, n, seed0)
