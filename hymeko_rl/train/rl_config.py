"""Canonical, configurable RL selection for the Coin-Delivery factorial: policy × strategy × semantic-critic mode, with
LOUD compatibility validation. Only the identifiers this task needs are executable (SAC_SINGLE_ACTOR / DIRECT /
TASK_ONLY / TASK_AND_MECHANISM); the future actor-bank/strategy identifiers exist as *validated unsupported* choices that
fail loudly rather than silently degrading. The mechanism target derives ONLY from canonical named observation fields.
"""
from __future__ import annotations

from enum import Enum, IntEnum

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


# implementations executable NOW (others are validated-but-unsupported → fail loud, never silent fallback)
SUPPORTED_POLICIES = {PolicyKind.SAC_SINGLE_ACTOR, PolicyKind.SAC_CONTACT_ACTOR_BANK,
                      PolicyKind.SCRIPTED_A1, PolicyKind.SCRIPTED_A4, PolicyKind.BC}
_ACTOR_BANK_POLICIES = {PolicyKind.SAC_CONTACT_ACTOR_BANK}
SUPPORTED_STRATEGIES = {Strategy.DIRECT, Strategy.HYMEKO_CONTACT_MODE}
SUPPORTED_CRITIC_MODES = {CriticMode.TASK_ONLY, CriticMode.TASK_AND_MECHANISM}

# pre-registered mechanism weight (§5: fixed, NOT tuned per run) — the actor maximises Q_task + MECH_COEF·Q_mechanism.
MECH_COEF = 0.5
_MECH_BOTH = field_index("both_contact")            # clean BILATERAL fingertip contact (28)
_MECH_BODY = field_index("arm_body_contact")        # arm-body↔coin shove (29) — a body-shove, not fingertip-attributed

# HYMeko_CONTACT_MODE strategy — inspectable named-field gates (NOT a learned opaque gate). All by field name.
# NOTE (measured 2026-07-21): the phase one-hot is stuck at CONTACT in the coin direct_env, so it carries no
# hysteresis signal; the ``prev_*_contact`` fields DO vary, so a 1-frame prev-bilateral hold is the stability seat.
# ``contact_lost_after_handoff`` is a LATCH — it must NOT gate transport (it would bar re-entry after any loss); it
# only labels the reposition reason.
_F_LEFT = field_index("left_contact")               # 26
_F_RIGHT = field_index("right_contact")             # 27
_F_LOST = field_index("contact_lost_after_handoff")  # 32 — latched; reason-only, never a transport gate
_F_PREV_LEFT = field_index("prev_left_contact")     # 39 — previous-frame contact (real per-frame hysteresis signal)
_F_PREV_RIGHT = field_index("prev_right_contact")   # 40


class ContactModeReason(IntEnum):
    """Why the HYMeko_CONTACT_MODE strategy selected a mode — logged per step (inspectable, §4)."""
    TRANSPORT_VALID_BILATERAL = 0                   # both fingertips, no shove → the transport corridor
    TRANSPORT_HYSTERESIS_HOLD = 1                   # was bilateral last frame, ≥1 fingertip now, no shove → 1-frame hold
    REPOSITION_NO_CONTACT = 2                       # neither fingertip on the coin → approach
    REPOSITION_ONE_SIDED = 3                        # exactly one fingertip → repair to bilateral
    REPOSITION_LOST = 4                             # bilateral lost after handoff → re-establish
    REPOSITION_BODY_SHOVE = 5                       # arm-body↔coin contact → back off to a clean fingertip grasp


def select_contact_mode(obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """The explicit HYMeko contact-mode gate: TRANSPORT when the coin is cleanly bracketed for targetward motion,
    REPOSITION otherwise. Reads canonical NAMED fields only; ``prev_*_contact`` supplies a 1-frame hysteresis hold so a
    single-frame bilateral flicker does not thrash the mode.

    # Preconditions ``obs`` is the flat ACTOR_FIELDS observation (…, 41). # Postconditions returns
    ``(transport_mask (…,) bool, reason (…,) long)`` — True ⇒ ACTOR_TRANSPORT; ``reason`` is a :class:`ContactModeReason`.
    """
    left = obs[..., _F_LEFT] > 0.5
    right = obs[..., _F_RIGHT] > 0.5
    both = obs[..., _MECH_BOTH] > 0.5
    body = obs[..., _MECH_BODY] > 0.5
    lost = obs[..., _F_LOST] > 0.5
    prev_both = (obs[..., _F_PREV_LEFT] > 0.5) & (obs[..., _F_PREV_RIGHT] > 0.5)
    clean = both & ~body                                            # valid bilateral transport configuration (CURRENT)
    # 1-frame hysteresis: was bilateral last frame, still ≥1 fingertip and no shove → hold TRANSPORT through a flicker
    stable = prev_both & (left | right) & ~body & ~clean
    transport = clean | stable
    reason = torch.full(obs.shape[:-1], int(ContactModeReason.REPOSITION_NO_CONTACT), dtype=torch.long,
                        device=obs.device)
    reason = torch.where(left ^ right, torch.full_like(reason, int(ContactModeReason.REPOSITION_ONE_SIDED)), reason)
    reason = torch.where(lost & ~(left | right), torch.full_like(reason, int(ContactModeReason.REPOSITION_LOST)), reason)
    reason = torch.where(body, torch.full_like(reason, int(ContactModeReason.REPOSITION_BODY_SHOVE)), reason)
    reason = torch.where(stable, torch.full_like(reason, int(ContactModeReason.TRANSPORT_HYSTERESIS_HOLD)), reason)
    reason = torch.where(clean, torch.full_like(reason, int(ContactModeReason.TRANSPORT_VALID_BILATERAL)), reason)
    return transport, reason


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
    if policy in _ACTOR_BANK_POLICIES and strategy is not Strategy.HYMEKO_CONTACT_MODE:
        raise UnsupportedRLConfig("SAC_CONTACT_ACTOR_BANK requires the HYMeko_CONTACT_MODE strategy "
                                  "(the bank has no meaning without an explicit mode selector)")
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
