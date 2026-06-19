"""The reward as a declarative term spec — the in-memory form of the
``meta_reward.hymeko`` vocabulary (the reward half of the agent description).

A :class:`RewardSpec` is an ordered list of ``(term_kind, weight)`` pairs; the scalar
reward is ``Σ weight · term(state)``. Each term kind maps to an extractor (Strategy) over
the live env state. :meth:`RewardSpec.from_hymeko` reads the terms + weights straight from a
``.hymeko`` task profile's ``reward_spec``, so a new reward needs only a new ``.hymeko`` —
the env's ``step`` no longer hard-codes ``-dist``.

Term kinds mirror ``data/robotics/meta_reward.hymeko``. Only the reaching task's terms are
implemented; the rest are a registry entry away.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from hymeko_rl.env._profile import read_bundle
from hymeko_rl.env.safety import CLEAN_SAFETY

if TYPE_CHECKING:
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

# The env is duck-typed: a term reads only the attributes it needs (reach_thresh, _last_safety,
# _planar_metrics, …), so terms serve both ArmReachEnv and PlanarGraspEnv. Hence `Any` here.
RewardTerm = Callable[[Any, float, np.ndarray], float]


# ── reward-term extractors (Strategy) ────────────────────────────────────────
# Each returns the *unweighted* term value given the env, the EE-to-target distance, and
# the applied action; RewardSpec.evaluate scales by the declared weight.
def _term_reach_distance(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -dist


def _term_success_bonus(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return 1.0 if dist < env.reach_thresh else 0.0


def _term_action_cost(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    a = np.asarray(action, dtype=np.float64)
    return -float(a @ a)   # -‖action‖²


# ── safety / configuration terms (read the env's last SafetyState) ───────────
# The weighted magnitude (the bounded terminal penalty) lives in the .hymeko `weight`; here the
# unweighted term is just the indicator/shape. Fall back to a clean state so the terms are inert
# (0) on an env that has not computed a safety state.
def _term_ground_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).ground_contact else 0.0


def _term_self_collision_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).self_collision else 0.0


def _term_joint_limit_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    # -(1 - margin)²: 0 mid-range, rising smoothly to -1 at a joint limit.
    margin = getattr(env, "_last_safety", CLEAN_SAFETY).joint_margin
    return -((1.0 - margin) ** 2)


def _term_below_ground_penalty(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    return -1.0 if getattr(env, "_last_safety", CLEAN_SAFETY).below_ground else 0.0


# ── planar grasping terms (read the env's PlanarGraspMetrics; 0 on a non-grasp env) ──────────
# The dense pull (-‖disk - zone‖) reuses `reach_distance` by passing disk_to_zone as the distance.
def _term_both_contact(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    m = getattr(env, "_planar_metrics", None)
    return 1.0 if (m is not None and m.left_contact and m.right_contact) else 0.0


def _term_in_zone(env: "ArmReachEnv", dist: float, action: np.ndarray) -> float:
    m = getattr(env, "_planar_metrics", None)
    return 1.0 if (m is not None and m.in_zone) else 0.0


# kind -> extractor. Defaults match meta_reward.hymeko.
_REWARD_TERMS: dict[str, RewardTerm] = {
    "reach_distance": _term_reach_distance,
    "success_bonus": _term_success_bonus,
    "action_cost": _term_action_cost,
    "ground_penalty": _term_ground_penalty,
    "self_collision_penalty": _term_self_collision_penalty,
    "joint_limit_penalty": _term_joint_limit_penalty,
    "below_ground_penalty": _term_below_ground_penalty,
    "both_contact": _term_both_contact,
    "in_zone": _term_in_zone,
}


@dataclass(frozen=True)
class RewardSpec:
    """An ordered tuple of ``(term_kind, weight)`` → the scalar reward ``Σ weight·term``.

    # Preconditions Non-empty; every kind is in :data:`_REWARD_TERMS`.
    # Postconditions ``evaluate`` returns a finite float; an all-zero-weight spec yields 0.
    """

    terms: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("RewardSpec must declare at least one term")
        unknown = [k for k, _ in self.terms if k not in _REWARD_TERMS]
        if unknown:
            raise ValueError(
                f"unknown reward term(s) {unknown}; known: {sorted(_REWARD_TERMS)}")

    def evaluate(self, env: Any, dist: float, action: np.ndarray) -> float:
        """Scalar reward from the live state: ``Σ weight · term(env, dist, action)``. ``env`` is
        duck-typed (ArmReachEnv or PlanarGraspEnv) — each term reads only what it needs."""
        return float(sum(w * _REWARD_TERMS[k](env, dist, action) for k, w in self.terms))

    @classmethod
    def from_hymeko(cls, profile_path: str | Path) -> "RewardSpec":
        """Build the spec from a ``.hymeko`` task profile's ``reward_spec`` bundle."""
        return cls(terms=read_reward_terms(profile_path))


def read_reward_terms(profile_path: str | Path) -> tuple[tuple[str, float], ...]:
    """Read a profile's ``reward_spec`` → ordered ``(term_kind, weight)`` pairs.

    The weight is each term instance's ``weight`` field (default ``1.0`` if absent). See
    :func:`hymeko_rl.env._profile.read_bundle` for the (narrow, B-003-bridge) parse.

    # Errors ``FileNotFoundError``; ``ValueError`` (no/!1 reward_spec, undeclared member).
    """
    out: list[tuple[str, float]] = []
    for _name, kind, body in read_bundle(profile_path, "reward_spec"):
        match = re.search(r"weight\s+(-?[\d.]+)", body)
        out.append((kind, float(match.group(1)) if match else 1.0))
    return tuple(out)


# The reaching task's default reward: dense negative distance to the goal (weight 1.0) —
# identical to the env's former procedural `-dist`.
REACH_REWARD = RewardSpec((("reach_distance", 1.0),))
