"""Contact-formation cooperative objective (v12) — the short-horizon micro-MDP reward, SEPARATE from the frozen
TaskMonitor verifier and from v2b. Rewards forming BALANCED two-fingertip contact and holding it toward a
handoff-ready state; penalizes one-fingertip-only contact, contact loss, arm/body contact, and excessive coin
displacement before stable contact. Computed from monitor-physical env metrics only (no policy internals)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any



@dataclass
class ContactFormationConfig:
    w_approach: float = 6.0        # + DENSE shaping: both tips getting closer to the coin (the gradient to contact)
    w_contact: float = 1.5        # + per-fingertip in contact (MONOTONIC: one-finger is progress toward both, not a failure)
    w_both: float = 2.0           # + both-fingertip bonus on top of the per-finger term
    w_balance: float = 1.0        # + symmetric left/right tip distance while in contact
    w_one_finger: float = 0.3     # − MILD nudge off a STUCK one-finger grip (not a barrier on the formation path)
    w_body: float = 5.0           # − arm/body contact
    w_displacement: float = 1.0   # − coin moved a lot before stable contact
    w_handoff: float = 5.0        # + terminal handoff-ready bonus
    near_coin: float = 0.06       # tip "near the coin" scale for the balance term
    approach_scale: float = 0.08  # tip-distance normaliser for the approach-shaping term
    hold_k: int = 4               # both-contact steps required for handoff-ready
    # v12b hold-shaping (default 0 → v12 formation-only behaviour is unchanged)
    w_streak: float = 0.0         # + consecutive both-contact streak (the K=2→3→4 hold gradient)
    w_flicker: float = 0.0        # − a both→not-both transition (grip flicker)
    w_linger: float = 0.0         # − LINGERING one-finger (sustained ≥ linger_thresh steps; NOT the transient)
    w_lost_after_both: float = 0.0  # − losing contact after both-contact was established
    linger_thresh: int = 3        # one-finger run length beyond which it is "lingering", not a transient
    streak_cap: int = 6           # cap the streak bonus so it does not dominate


def hold_shaping(streak: int, flicker: bool, one_finger_run: int, lost_after_both: bool,
                 cfg: ContactFormationConfig) -> float:
    """v12b hold terms (history-dependent; the env supplies the running counters). Rewards building a consecutive
    both-contact streak; penalizes flicker, a LINGERING (not transient) one-finger grip, and losing contact after
    both was formed. Zero when the v12b weights are all 0."""
    r = cfg.w_streak * float(min(streak, cfg.streak_cap))
    r -= cfg.w_flicker if flicker else 0.0
    r -= cfg.w_linger if one_finger_run >= cfg.linger_thresh else 0.0
    r -= cfg.w_lost_after_both if lost_after_both else 0.0
    return float(r)


def contact_step_reward(m: Any, coin_disp: float, cfg: ContactFormationConfig) -> float:
    """Dense per-step contact-formation reward. The DOMINANT term is dense approach shaping (tips → coin) so a
    pre-contact policy has a gradient to follow; the per-fingertip contact term is MONOTONIC (one-finger = progress
    toward both, only mildly nudged, NOT penalized as a barrier on the path). ``coin_disp`` = |Δcoin| this step."""
    left, right = bool(m.left_contact), bool(m.right_contact)
    both = left and right
    body = bool(getattr(getattr(m, "legality", None), "arm_body_contact", False) or False)
    lt, rt = float(m.left_tip_dist), float(m.right_tip_dist)
    balance = max(0.0, 1.0 - abs(lt - rt) / cfg.near_coin)          # 1 when symmetric, 0 when very asymmetric
    closeness = 2.0 - min(lt, cfg.approach_scale) / cfg.approach_scale - min(rt, cfg.approach_scale) / cfg.approach_scale
    r = cfg.w_approach * 0.5 * closeness                            # dense gradient toward two-fingertip contact
    r += cfg.w_contact * (float(left) + float(right))              # monotonic: 0 → 1 → 2 fingers
    r += cfg.w_both if both else 0.0
    r += cfg.w_balance * balance if (left or right) else 0.0
    r -= cfg.w_one_finger if (left != right) else 0.0              # mild only (progress, not failure, during formation)
    r -= cfg.w_body if body else 0.0
    r -= cfg.w_displacement * float(coin_disp)
    return float(r)


def is_handoff_ready(both_history: list[bool], cfg: ContactFormationConfig) -> bool:
    """Handoff-ready = both fingertips in contact for the last ``hold_k`` steps (a preserved, balanced grip)."""
    return len(both_history) >= cfg.hold_k and all(both_history[-cfg.hold_k:])


def is_safety_violation(m: Any) -> bool:
    """Hard shield: any arm/body contact aborts the contact-formation episode (exploit/body guard)."""
    return bool(getattr(getattr(m, "legality", None), "arm_body_contact", False) or False)


def contact_metrics(both_frac: float, one_finger_frac: float, balance: float, preserved: bool, handoff_ready: bool,
                    body_frac: float, coin_drift: float, high_coin_y: bool) -> dict:
    """The Gate-1 micro-MDP metric bundle for one episode."""
    return {"both_contact_frac": round(float(both_frac), 4), "one_finger_frac": round(float(one_finger_frac), 4),
            "balance": round(float(balance), 4), "contact_preserved": 1.0 if preserved else 0.0,
            "handoff_ready": 1.0 if handoff_ready else 0.0, "body_frac": round(float(body_frac), 4),
            "coin_drift": round(float(coin_drift), 5), "high_coin_y_contact": 1.0 if (handoff_ready and high_coin_y) else 0.0}
