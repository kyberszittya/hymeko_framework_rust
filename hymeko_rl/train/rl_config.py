"""Canonical, configurable RL selection for the Coin-Delivery factorial: policy × strategy × semantic-critic mode, with
LOUD compatibility validation. Only the identifiers this task needs are executable (SAC_SINGLE_ACTOR / DIRECT /
TASK_ONLY / TASK_AND_MECHANISM); the future actor-bank/strategy identifiers exist as *validated unsupported* choices that
fail loudly rather than silently degrading. The mechanism target derives ONLY from canonical named observation fields.
"""
from __future__ import annotations

from enum import Enum

import torch

from hymeko_rl.eval.team_tensor import field_index


class PolicyKind(str, Enum):
    SCRIPTED_A1 = "SCRIPTED_A1"
    SCRIPTED_A4 = "SCRIPTED_A4"
    BC = "BC"
    SAC_SINGLE_ACTOR = "SAC_SINGLE_ACTOR"
    SAC_CONTACT_ACTOR_BANK = "SAC_CONTACT_ACTOR_BANK"


class Strategy(str, Enum):
    DIRECT = "DIRECT"
    HYMEKO_CONTACT_MODE = "HYMeko_CONTACT_MODE"
    CRITIC_SELECTED = "CRITIC_SELECTED"


class CriticMode(str, Enum):
    TASK_ONLY = "TASK_ONLY"
    TASK_AND_MECHANISM = "TASK_AND_MECHANISM"


# implementations executable in THIS task (others are validated-but-unsupported → fail loud, never silent fallback)
SUPPORTED_POLICIES = {PolicyKind.SAC_SINGLE_ACTOR, PolicyKind.SCRIPTED_A1, PolicyKind.SCRIPTED_A4, PolicyKind.BC}
_ACTOR_BANK_POLICIES = {PolicyKind.SAC_CONTACT_ACTOR_BANK}
SUPPORTED_STRATEGIES = {Strategy.DIRECT}
SUPPORTED_CRITIC_MODES = {CriticMode.TASK_ONLY, CriticMode.TASK_AND_MECHANISM}

# pre-registered mechanism weight (§5: fixed, NOT tuned per run) — the actor maximises Q_task + MECH_COEF·Q_mechanism.
MECH_COEF = 0.5
_MECH_BOTH = field_index("both_contact")            # clean BILATERAL fingertip contact (28)
_MECH_BODY = field_index("arm_body_contact")        # arm-body↔coin shove (29) — a body-shove, not fingertip-attributed


class UnsupportedRLConfig(ValueError):
    """A requested policy/strategy/critic combination is not implemented — raised loudly, never a silent fallback."""


def validate_rl_config(policy: PolicyKind, strategy: Strategy, critic_mode: CriticMode, *,
                       obs_dim: int | None = None, checkpoint_obs_dim: int | None = None) -> None:
    """§3 loud compatibility gate. # Raises :class:`UnsupportedRLConfig` on any invalid or not-yet-implemented combo."""
    if not isinstance(policy, PolicyKind):
        raise UnsupportedRLConfig(f"unknown policy {policy!r}; valid: {[p.value for p in PolicyKind]}")
    if not isinstance(strategy, Strategy):
        raise UnsupportedRLConfig(f"unknown strategy {strategy!r}; valid: {[s.value for s in Strategy]}")
    if not isinstance(critic_mode, CriticMode):
        raise UnsupportedRLConfig(f"unknown critic mode {critic_mode!r}; valid: {[c.value for c in CriticMode]}")
    if strategy is Strategy.CRITIC_SELECTED and critic_mode is not CriticMode.TASK_AND_MECHANISM:
        raise UnsupportedRLConfig("CRITIC_SELECTED requires TASK_AND_MECHANISM (nothing to select actions with otherwise)")
    if strategy is Strategy.HYMEKO_CONTACT_MODE and policy not in _ACTOR_BANK_POLICIES:
        raise UnsupportedRLConfig("HYMeko_CONTACT_MODE requires an actor bank (SAC_CONTACT_ACTOR_BANK)")
    if policy in _ACTOR_BANK_POLICIES:
        raise UnsupportedRLConfig("SAC_CONTACT_ACTOR_BANK is not implemented yet (validated-unsupported; use "
                                  "SAC_SINGLE_ACTOR) — it must never silently fall back to the single actor")
    if policy not in SUPPORTED_POLICIES:
        raise UnsupportedRLConfig(f"policy {policy.value} not executable in this task")
    if strategy not in SUPPORTED_STRATEGIES:
        raise UnsupportedRLConfig(f"strategy {strategy.value} not executable in this task (only DIRECT)")
    if critic_mode not in SUPPORTED_CRITIC_MODES:
        raise UnsupportedRLConfig(f"critic mode {critic_mode.value} not executable in this task")
    if checkpoint_obs_dim is not None and obs_dim is not None and checkpoint_obs_dim != obs_dim:
        raise UnsupportedRLConfig(f"checkpoint obs_dim {checkpoint_obs_dim} != policy obs_dim {obs_dim} "
                                  "(architecture-incompatible checkpoint)")


def mechanism_reward(obs: torch.Tensor) -> torch.Tensor:
    """The bounded [0,1] per-step mechanism-VALIDITY signal (§4): clean BILATERAL fingertip contact WITHOUT an arm-body
    shove, from canonical NAMED observation fields only (both_contact ∧ ¬arm_body_contact). NOT an environment reward —
    a separate semantic target for Q_mechanism. # Preconditions ``obs`` is the flat ACTOR_FIELDS observation."""
    both = obs[..., _MECH_BOTH].clamp(0.0, 1.0)
    body = obs[..., _MECH_BODY].clamp(0.0, 1.0)
    return both * (1.0 - body)
