"""R6 — the grip-held brake-to-stop controller with certified release.

R5 proved the released-coast dissipation is unobservable before the release decision. R6 removes that prediction: keep the
coin in the OBSERVABLE HELD mode and brake it to rest inside the zone under held bilateral contact, then RELEASE only from a
certified rest state (`release_certificate`). The strategy is structurally different from the R4/R5 coast-in (which released
early to coast), so it is a separate controller class (§6.5 #8), reusing the R4/R5 substrate: the SnapContext authority, the
R3 decoder magnitudes, the monotone PhaseMachine, and the continuous `closed_loop_rollout` driver interface.

Phase logic — the transport→brake trigger is the R4 stopping guard `d_stop = v∥²/(2·a_brake)` with the MEASURED braking
authority `a_brake` (grip held throughout, so a fast cradle cannot escape):

    HELD-TRANSPORT (PUSH, grip held)  while  d_stop < dtz − zone            # can still push and stop in time
    GRIP-HELD BRAKE (velocity feedback, grip held)  when  d_stop ≥ dtz − zone
    ↳ decay squeeze as the coin slows so the release certificate can pass
    CERTIFIED RELEASE  when the N-frame certificate arms → frozen K6 dwell

θ magnitudes (squeeze / forward / balance / brake_gain) come from the decoded intent; the timing roles are neutralised and
the phase is state-driven (monotone PUSH→BRAKE→RELEASE). NEVER a teacher θ; one continuous trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.closed_loop_intent import BRAKE, PUSH, RELEASE, PhaseMachine, SnapContext
from hymeko_rl.coin_delivery.theta_option.physical_intent import PhysicalIntent
from hymeko_rl.coin_delivery.theta_option.release_certificate import ReleaseCertMonitor, ReleaseCertParams
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox

_EPS = 1e-9


@dataclass
class BrakeToStopParams:
    """FROZEN (dev-tuned) brake-to-stop parameters; the release certificate carries its own tolerances."""

    a_brake_gain: float = 1.0               # scales the measured brake authority in d_stop = v∥²/(2·gain·a_brake)
    a_brake_floor: float = 0.5              # m/s² floor so d_stop cannot blow up at ~zero authority
    brake_margin: float = 0.005             # m — enter BRAKE when d_stop ≥ dtz − release_zone + brake_margin
    release_zone: float = 0.02              # m — stop target is the zone edge
    min_push_steps: int = 2                 # guarantee a short held-transport before braking
    decay_speed: float = 0.08               # m/s — below this the coin is settling ⇒ decay the squeeze
    squeeze_decay: float = 0.85             # per-step multiplicative squeeze decay during the settle (→ low fn for release)
    squeeze_floor: float = 0.15             # keep enough squeeze (× θ[0]) to retain contact while decaying
    cert: ReleaseCertParams = field(default_factory=ReleaseCertParams)


class BrakeToStopController:
    """Drives one continuous held brake-to-stop trajectory. `theta_for_step` is what `closed_loop_rollout` calls each step;
    `before_release`/`release_boundary_step` give the frozen contact-retention bookkeeping window. # Postconditions: NEVER a
    teacher θ; box-legal, phase-monotone; grip held until a certified release; cert/phase trace kept for provenance."""

    def __init__(self, snap: CradleSnapshot, base_intent: PhysicalIntent, theta0: np.ndarray,
                 params: BrakeToStopParams = BrakeToStopParams(), cfg: Any = DELIVERY_CFG,
                 snap_ctx: "SnapContext | None" = None) -> None:
        self.snap, self.base_intent, self.p, self.cfg = snap, base_intent, params, cfg
        self.box = ThetaBox()
        sc = snap_ctx or SnapContext.build(snap, cfg)
        self.theta0 = self.box.clip(np.asarray(theta0, np.float64))
        self.horizon = sc.horizon
        dt = float(sc.ctx.control_dt)
        self.a_brake = max(params.a_brake_gain * float(sc.authority["brake_opposed_reach"]) / max(dt, _EPS),
                           params.a_brake_floor)
        self.authority = sc.authority
        self.reset()

    def reset(self) -> None:
        self.pm = PhaseMachine()
        self.cert = ReleaseCertMonitor(self.p.cert)
        self._sqz_scale = 1.0
        self.cert_trace: list[dict[str, Any]] = []
        self.phase_trace: list[dict[str, Any]] = []

    @property
    def phase(self) -> int:
        return self.pm.phase

    def before_release(self, t: int) -> bool:
        return self.pm.phase < RELEASE

    @property
    def release_boundary_step(self) -> "int | None":
        return self.pm.release_step

    def _live(self, rl: Any) -> "tuple[float, float]":
        u, dtz = rl.inner.direction_to_zone()
        e_par = np.asarray(u, np.float64)[:2]
        n = float(np.linalg.norm(e_par))
        e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
        v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        return float(dtz), float(v @ e_par)

    def _want_phase(self, rl: Any, t: int, dtz: float, v_par: float) -> int:
        """State-driven monotone phase: held-transport until d_stop reaches the zone, then grip-held brake, then RELEASE
        only when the N-frame certificate arms."""
        if self.pm.phase >= RELEASE:
            return RELEASE
        armed, diag = self.cert.update(rl, t)                 # evaluate the release certificate every step
        self.cert_trace.append(diag)
        if armed:
            return RELEASE
        if self.pm.phase < RELEASE and t <= self.p.min_push_steps:
            return max(self.pm.phase, PUSH)                    # a short guaranteed held-transport
        d_stop = (v_par * v_par) / (2.0 * self.a_brake) if v_par > 0 else 0.0
        if d_stop >= max(float(dtz) - self.p.release_zone, 0.0) + self.p.brake_margin:
            return max(self.pm.phase, BRAKE)                   # brake NOW to stop inside the zone (grip held)
        return max(self.pm.phase, PUSH)                        # keep transporting under held grip

    def theta_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        dtz, v_par = self._live(rl)
        want = self._want_phase(rl, t, dtz, v_par)
        theta_mag = self.theta0.copy()
        theta_mag[3] = theta_mag[4] = self.horizon             # timing neutralised; phase is state-driven
        # squeeze decay during the settle so grip pressure (fn) drops enough for the release certificate, floored to retain
        # contact. Applied while braking/settling and the coin is slow.
        if want >= BRAKE and abs(v_par) < self.p.decay_speed:
            self._sqz_scale = max(self.p.squeeze_floor, self._sqz_scale * self.p.squeeze_decay)
        theta_mag[0] = theta_mag[0] * self._sqz_scale
        eff, phase = self.pm.step(theta_mag, t, force_phase=want)
        self.phase_trace.append({"t": int(t), "phase": phase, "dtz_mm": round(dtz * 1000, 2),
                                 "v_par": round(v_par, 5), "sqz_scale": round(self._sqz_scale, 3)})
        return self.box.clip(eff)
