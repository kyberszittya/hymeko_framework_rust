"""CIP-export bridge — convert reward-independent monitor verdicts into named CIP scalar variables.

CIP (Causal Information Prioritization) consumes a factored set of scalar variables. This module is the thin,
generic adapter that lifts a :class:`~hymeko_rl.eval.task_monitor.root.TaskVerdict` (or its serialized
``as_dict()`` mapping) into the named CIP variables, **without** touching the reward and **without** fabricating
measurements that do not exist yet.

Design rules (see ``docs/plans/2026-07-04-hymeko-code-agent/hymeko_to_cip.md``):

* Every CIP variable always appears in :attr:`CipExport.variables` — a value that could not be sourced gets a
  **stable, deterministic default** and its name is listed in :attr:`CipExport.missing`. A consumer filters on
  ``missing`` before feeding LiNGAM/CIP; the default is never presented as a measurement.
* ``stagnation_duration`` / ``stagnated`` are populated iff a ``StagnationVerdict`` is present under
  ``sub_verdicts["stagnation"]`` (compose ``default_submonitors() + [StagnationMonitor()]``).
* ``reward_progress_disagreement`` is populated from a ``RewardConsistencyMonitor`` output when the caller
  attaches it (top-level ``reward_progress_disagreement`` scalar, or a ``reward_consistency`` sub-verdict) — a
  single-trajectory verdict carries no cross-policy disagreement, so it is legitimately missing there.
* ``forbidden_contact_count`` / ``clearance_min`` / ``phase_transition_failure`` have no monitor producing them
  yet (audit gap C.2); they are read opportunistically (top-level or any sub-verdict field of that name) so the
  bridge picks them up unchanged once the violation submonitors land, and default-and-missing until then.

The bridge is read-only: it never mutates the input verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from .root import TaskVerdict

# Canonical CIP variable order (the initial factored set; extend as violation submonitors land).
CIP_VARIABLES: tuple[str, ...] = (
    "success_monitor_pass",
    "progress_score",
    "stagnation_duration",
    "stagnated",
    "forbidden_contact_count",
    "clearance_min",
    "phase_transition_failure",
    "reward_progress_disagreement",
)

# Deterministic defaults for a variable that could not be sourced (never presented as a measurement).
CIP_DEFAULTS: dict[str, float | int | bool] = {
    "success_monitor_pass": False,
    "progress_score": 0.0,
    "stagnation_duration": 0,
    "stagnated": False,
    "forbidden_contact_count": 0,
    "clearance_min": 0.0,
    "phase_transition_failure": False,
    "reward_progress_disagreement": 0.0,
}

_MISSING = object()   # sentinel distinct from any real value (a genuine ``0`` / ``False`` is not "missing")


@dataclass(frozen=True)
class CipExport:
    """The CIP-ready variables plus provenance: which were defaulted (``missing``) and where the rest came from."""

    variables: dict[str, float | int | bool]
    missing: tuple[str, ...]
    source_keys: tuple[str, ...]


def _get(obj: Any, key: str, default: Any = _MISSING) -> Any:
    """Read ``key`` from a mapping (``.get``) or an object (``getattr``); ``default`` if absent. Read-only."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _find_scalar(verdict: Any, sub: Any, key: str) -> Any:
    """Look for ``key`` at the top level, then in any sub-verdict. ``_MISSING`` if nowhere — never fabricated."""
    top = _get(verdict, key)
    if top is not _MISSING:
        return top
    if isinstance(sub, Mapping):
        for sv in sub.values():
            val = _get(sv, key)
            if val is not _MISSING:
                return val
    return _MISSING


def _reward_disagreement(verdict: Any) -> tuple[Optional[float], Optional[str]]:
    """Derive ``reward_progress_disagreement`` from a RewardConsistencyMonitor output, if the caller attached one.

    Accepts (in priority order): a pre-computed top-level ``reward_progress_disagreement`` scalar; a
    ``reward_consistency`` sub-verdict carrying a concordance ``score`` (disagreement = ``1 - score``, 0 == aligned)
    or, failing that, a ``passed`` flag (``0.0`` aligned / ``1.0`` inverted). Returns ``(None, None)`` if absent.
    """
    pre = _get(verdict, "reward_progress_disagreement")
    if pre is not _MISSING:
        return float(pre), "reward_progress_disagreement"
    rc = _get(verdict, "reward_consistency")
    if rc is not _MISSING:
        score = _get(rc, "score")
        if score is not _MISSING:
            return round(1.0 - float(score), 6), "reward_consistency.score"
        passed = _get(rc, "passed")
        if passed is not _MISSING:
            return (0.0 if passed else 1.0), "reward_consistency.passed"
    return None, None


def export_cip_variables(verdict: "TaskVerdict | Mapping[str, Any]") -> CipExport:
    """Convert a monitor ``TaskVerdict`` (or its ``as_dict()`` mapping) into named CIP scalar variables.

    # Preconditions
    * ``verdict`` is a :class:`TaskVerdict`, its ``as_dict()`` mapping, or a partial mapping of the same shape.

    # Postconditions
    * Every name in :data:`CIP_VARIABLES` is present in the returned ``variables`` (a defaulted value if unsourced).
    * ``missing`` lists exactly the variables that were defaulted; ``source_keys`` names where the rest came from.
    * The input ``verdict`` is not mutated (read-only).
    """
    if verdict is None:
        raise TypeError("export_cip_variables: verdict must be a TaskVerdict or mapping, got None")
    sub = _get(verdict, "sub_verdicts", {})
    stag = sub.get("stagnation") if isinstance(sub, Mapping) else None

    variables: dict[str, float | int | bool] = {}
    missing: list[str] = []
    source: list[str] = []

    def put(name: str, raw: Any, cast: Any, src: str) -> None:
        if raw is _MISSING:
            variables[name] = CIP_DEFAULTS[name]
            missing.append(name)
        else:
            variables[name] = cast(raw)
            source.append(src)

    # Task-level aspects (present on any real verdict).
    put("success_monitor_pass", _get(verdict, "monitor_pass"), bool, "monitor_pass")
    put("progress_score", _get(verdict, "progress_score"), float, "progress_score")

    # Stagnation (present iff a StagnationVerdict was composed in).
    put("stagnation_duration", _get(stag, "stagnation_duration"), int, "sub_verdicts.stagnation.stagnation_duration")
    put("stagnated", _get(stag, "stagnated"), bool, "sub_verdicts.stagnation.stagnated")

    # Not-yet-implemented violation variables (audit gap C.2): read opportunistically, else default + missing.
    put("forbidden_contact_count", _find_scalar(verdict, sub, "forbidden_contact_count"), int,
        "forbidden_contact_count")
    put("clearance_min", _find_scalar(verdict, sub, "clearance_min"), float, "clearance_min")
    put("phase_transition_failure", _find_scalar(verdict, sub, "phase_transition_failure"), bool,
        "phase_transition_failure")

    # Reward/monitor disagreement (cross-cutting RewardConsistencyMonitor output, attached by the caller).
    rpd, rpd_src = _reward_disagreement(verdict)
    if rpd is None:
        variables["reward_progress_disagreement"] = CIP_DEFAULTS["reward_progress_disagreement"]
        missing.append("reward_progress_disagreement")
    else:
        variables["reward_progress_disagreement"] = rpd
        source.append(rpd_src or "reward_progress_disagreement")

    return CipExport(variables=variables, missing=tuple(missing), source_keys=tuple(source))
