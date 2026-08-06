"""v16 — cooperative_push_objective_v0: the training signal for the standalone push primitive (transport + grip). Kept
separate from v2b (certified env reward, comparability) and the TaskMonitor (external verifier). Rewards delivery and
progress-toward-target ONLY when two-fingertip contact is active and balanced; penalises contact loss before delivery,
one-finger pushing, body-only progress, arm/body contact, excessive squeeze, and uncontrolled coin displacement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CooperativePushConfig:
    w_progress: float = 20.0           # progress toward the zone (any contact)
    w_progress_both: float = 30.0      # progress toward the zone WHILE both fingertips are on the coin (the goal)
    w_both: float = 0.4                # small per-step both-contact bonus
    w_balance: float = 1.0             # balanced two-fingertip contact (symmetric tip-to-coin distances)
    w_contact_loss: float = 4.0        # both → not-both transition before delivery
    w_one_finger: float = 2.0          # exactly one fingertip in contact
    w_body: float = 6.0                # arm/body contact (terminal safety penalty applied in the MDP)
    w_squeeze_excess: float = 0.5      # penalise crushing squeeze (both tips far inside the coin)
    w_lateral_disp: float = 0.5        # uncontrolled sideways coin displacement (off the coin→zone line)
    w_delivery: float = 12.0           # terminal delivery bonus (applied in the MDP)
    balance_tol: float = 0.02
    near_coin: float = 0.06


def coop_push_step_reward(m: Any, d_dist: float, prev_both: bool, prev_lateral: float, cfg: CooperativePushConfig
                          ) -> float:
    """Per-step reward. `d_dist` = previous coin-to-zone distance minus current (positive = progress). `prev_lateral`
    is the previous perpendicular offset of the coin from the coin→zone line (for the uncontrolled-displacement term)."""
    both = bool(m.left_contact and m.right_contact)
    one_finger = bool(m.left_contact) ^ bool(m.right_contact)
    progress = max(0.0, float(d_dist))
    r = cfg.w_progress * progress
    if both:
        balanced = abs(float(m.left_tip_dist) - float(m.right_tip_dist)) < cfg.balance_tol
        r += cfg.w_progress_both * progress + cfg.w_both + (cfg.w_balance if balanced else 0.0)
    if prev_both and not both:
        r -= cfg.w_contact_loss                                    # harmful contact loss (before delivery)
    if one_finger:
        r -= cfg.w_one_finger
    # excessive squeeze: both tips pressed well inside the coin surface
    if float(m.left_tip_dist) < 0.0 and float(m.right_tip_dist) < 0.0:
        r -= cfg.w_squeeze_excess
    return float(r)


@dataclass(frozen=True)
class CooperativePushConfigV1:
    """v16b stabilised variant. The v0 audit found a HOLD-without-push local optimum: unconditional per-step both-contact
    + balance bonuses accumulated to out-score delivery (hold return +18.9 vs transport-that-delivers −12.8), so SAC
    correctly learned to hold. v1 GATES the contact/balance bonus on PROGRESS (reward two-fingertip contact only WHILE
    the coin is advancing), and drops the unconditional both-contact bonus. Semantics preserved (reward preservation
    DURING delivery); only the degenerate hold optimum is removed. Delivery + progress-while-both now dominate."""

    w_progress: float = 25.0           # progress toward the zone (any contact)
    w_progress_both: float = 60.0      # progress WHILE both fingertips on the coin (the goal — dominant term)
    w_balance_progress: float = 10.0   # balanced two-fingertip contact, ONLY while progressing (no idle hold reward)
    w_contact_loss: float = 3.0        # ONE-TIME penalty on the both→not transition before delivery (not per-step)
    w_body: float = 6.0
    w_delivery: float = 30.0           # LARGE terminal bonus so delivering dominates the safe-but-useless hold
    balance_tol: float = 0.02
    progress_eps: float = 5e-4         # minimum toward-zone displacement to count as "progressing"


def coop_push_step_reward_v1(m: Any, d_dist: float, prev_both: bool, cfg: CooperativePushConfigV1) -> float:
    """v1 step reward: the both-contact / balance bonus is GATED on progress (holding-without-progress earns nothing),
    and contact loss is a ONE-TIME event penalty — NOT a per-step one-finger accumulation, which in v0 reached ~−80 and
    made safe holding out-score imperfect transport. Delivery (large terminal) + progress-while-both dominate."""
    both = bool(m.left_contact and m.right_contact)
    progress = max(0.0, float(d_dist))
    progressing = progress > cfg.progress_eps
    r = cfg.w_progress * progress
    if both and progressing:
        balanced = abs(float(m.left_tip_dist) - float(m.right_tip_dist)) < cfg.balance_tol
        r += cfg.w_progress_both * progress + (cfg.w_balance_progress * progress if balanced else 0.0)
    if prev_both and not both:
        r -= cfg.w_contact_loss                                    # harmful contact-loss EVENT (once, before delivery)
    return float(r)


def lateral_offset(m: Any, zone: np.ndarray) -> float:
    """Perpendicular distance of the coin from the coin→zone line at episode start — a proxy for sideways drift."""
    coin = np.asarray(m.disk_pos[:2], np.float64)
    to_zone = zone - coin
    n = np.linalg.norm(to_zone)
    if n < 1e-9:
        return 0.0
    u = to_zone / n
    perp = np.array([-u[1], u[0]])
    return float(abs(np.dot(coin, perp)))
