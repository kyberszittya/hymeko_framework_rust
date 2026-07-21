"""Neutral-start Coin Delivery — connect APPROACH → GRASP_ACQUIRE → (proven) TRANSPORT → BRAKE_HOLD_CERTIFY.

The initial-pose audit showed the RL policy was never trained on the approach: `ContactFormationEnv.reset` restores a
contact-prepared bank snapshot and a scripted `grasp_carry` prefix grips the coin, so the policy only ever saw the
transport suffix (5/9), and delivers 0/9 from a true neutral pose. This module makes the hidden prefix EXPLICIT:

- :class:`NeutralCoinDeliveryEnv` — a delivery env whose reset restores the **canonical neutral pose** (no bank
  snapshot, pads open, no contact) and then runs a **controllable** number of scripted `grasp_carry` steps. That count
  is the reverse-curriculum knob: 0 = true neutral (N5), rising to full-handoff = contact-prepared (N0). It fails loud
  if a NEUTRAL_START run begins with contact.
- The scripted `grasp_carry` is now visible curriculum/demonstration data, not a hidden reset side-effect.

Reuses the existing env chain, `p_grasp_carry`, the transport checkpoint, and `DeliveryCertifier`. No contact-geometry /
wrist / critic / replay / n-step / force-closure change (per the task constraints).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.delivery_certificate import CertStep
from hymeko_rl.train.coin_delivery_rl import CoinDeliveryTrainEnv, DeliveryRLConfig, p_grasp_carry


class NeutralCoinDeliveryEnv(CoinDeliveryTrainEnv):
    """CoinDeliveryTrainEnv whose reset starts from the CANONICAL NEUTRAL pose (no bank snapshot) and runs exactly
    ``prefix_steps`` scripted ``grasp_carry`` steps (the reverse-curriculum knob). ``prefix_steps=0`` = true neutral.

    # Preconditions the wrapped ContactFormationEnv's ``_restore`` is neutralised (no-op) so the planar reset stands.
    # Invariants NEUTRAL_START (prefix_steps==0) begins with no fingertip contact, else it raises.
    """

    def __init__(self, cf, cfg=None, *, prefix_steps: int = 0, direct: bool = True) -> None:
        super().__init__(cf, cfg or DeliveryRLConfig())
        self.prefix_steps = int(prefix_steps)
        if direct:
            self._base_override = lambda inner, t: np.zeros(self.action_space.shape[0], np.float32)
            self._delta_override = 1.0

    def set_stage(self, prefix_steps: int) -> None:
        self.prefix_steps = int(prefix_steps)

    def _neutral_contact_reset(self, seed):
        """Reset the ContactFormationEnv to the CANONICAL NEUTRAL planar pose (PlanarGraspEnv.reset — arm zeros, coin
        placed for the seed, pads open, no contact) WITHOUT restoring a contact-prepared bank snapshot; mirror the
        ContactFormationEnv bookkeeping the bank path would have set."""
        cf = self.env
        cf._env.reset(seed=seed)                                     # planar neutral reset (no bank snapshot)
        cf._tracker.reset(cf._env)
        cf._t = 0
        cf._both_hist = []
        cf._prev_coin = np.asarray(cf._env._planar_metrics.disk_pos[:2], np.float64)
        cf._high = cf._streak = cf._of_run = 0
        cf._had_both = False
        return cf._obs(np.zeros(4, np.float32))

    def reset(self, *, seed=None):
        obs = self._neutral_contact_reset(seed)                      # true canonical neutral (no bank, no hidden pre-roll)
        self.env._horizon = self.cfg.prefix_cap + self.cfg.horizon + 8
        self._reset_state()
        self._last_obs = np.asarray(obs, np.float32)
        self._start_obs = self._last_obs.copy()
        met0 = self.inner._planar_metrics
        if self.prefix_steps == 0 and (met0.left_contact or met0.right_contact):
            raise RuntimeError("NEUTRAL_START invariant violated: initial fingertip contact at prefix_steps=0 "
                               "(hidden pre-roll or non-neutral snapshot leaked into the reset)")
        for _ in range(self.prefix_steps):                          # explicit, curriculum-controlled scripted approach
            acquire = np.clip(p_grasp_carry(self.inner, 0), self.cfg.lo, self.cfg.hi).astype(np.float32)
            obs, _r, _t, _tr, sinfo = self.env.step(acquire)
            self._last_obs = np.asarray(obs, np.float32)
            self._had_both = self._had_both or self._both()
            if sinfo.get("handoff_ready"):
                self._handoff = True
                break
        self._prev_dtz = self._start_dtz = self._dtz()
        self._prev_both = self._both()
        return self._last_obs, {"handoff_event": self._handoff}


def neutral_env(*, prefix_steps: int = 0):
    """Build an E0 NeutralCoinDeliveryEnv (direct-action) whose reset is a true canonical-neutral start (no bank
    snapshot restore); returns (env, contact_env)."""
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _e0_env
    _base, cf = _e0_env()
    cf._restore = lambda item: None                                 # neutralise the bank-snapshot restore ⇒ neutral pose
    env = NeutralCoinDeliveryEnv(cf, DeliveryRLConfig(), prefix_steps=prefix_steps)
    return env, cf


def _clearance(inner) -> float:
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    return float(inner.planar_metrics.disk_to_zone) - (disk_r + float(inner._zone_half))


def _cert_step(inner, cf) -> CertStep:
    from hymeko_rl.experiments.coin_wristed_delivery import _both_pad_contact
    met = inner._planar_metrics
    lg = getattr(met, "legality", None)
    lf = bool(lg.left_fingertip_contact) if lg is not None else bool(met.left_contact)
    rf = bool(lg.right_fingertip_contact) if lg is not None else bool(met.right_contact)
    body = bool(lg.arm_body_contact) if lg is not None else False
    imp = float(lg.arm_body_contact_impulse) if lg is not None else 0.0
    v = inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]
    _both, fl, fr = _both_pad_contact(cf)
    return CertStep(disk_to_zone=float(met.disk_to_zone), disk_speed=float(np.linalg.norm(v)),
                    left_fingertip=lf, right_fingertip=rf, arm_body_contact=body, arm_body_impulse=imp,
                    force_left=fl, force_right=fr)
