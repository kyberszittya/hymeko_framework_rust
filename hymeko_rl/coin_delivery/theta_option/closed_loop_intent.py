"""R4 Stage 2 — the deterministic basin-aware intent-correction law + the monotone phase machine + the closed-loop
controller.

The measured physics (per-step traces of the four cradles) show the delivering strategy is **PUSH-then-COAST**: build
coin momentum with the grip, then RELEASE (relax the grip, no active brake) and let the coin coast into the zone under
friction and settle. The open-loop R3 predictor fails the held-out cradles precisely by breaking this: s7 runs a long
velocity-feedback BRAKE that kills the momentum 105 mm short; s4 pushes too weakly and over-squeezes, so the coin coasts
only to 64 mm. Neither over-shoots — both stop short from over-control.

R4's NEW element is a causal feedback law over the physical INTENT that (a) replaces the open-loop timing with a
**coast-in trigger** — RELEASE the moment the measured momentum can coast the coin into the zone (`v² / 2·a_friction ≥
dtz − zone`), never braking unless a large overshoot is predicted — and (b) corrects the magnitude roles (cut over-squeeze,
ensure push momentum, trim over-braking) from the measured contact/progress response. The one physical constant, the coast
deceleration `a_friction`, is cradle-agnostic and dev-tuned; the trigger reproduces the teacher release timing (so it is a
no-op on an already-delivering plan, the C0 requirement) and fixes the held-out over-control. The correction updates the R3
intent roles; the FROZEN R3 decoder turns the corrected intent into the cradle-specific θ magnitudes. Everything is in the
canonical frame → mirror-equivariant (forward/braking/squeeze read invariant signals; lateral reads the canonicalised
perpendicular). The phase machine keeps PUSH→BRAKE→RELEASE monotone (no chatter / reverse).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.authority_decoder import DecoderResult, decode_from_canonical
from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, r1_grouped_features
from hymeko_rl.coin_delivery.theta_option.closed_loop_state import (
    CoastEstimator, MeasureContext, ProbeReference, ResponseState, measure_response)
from hymeko_rl.coin_delivery.theta_option.directional_authority import contact_internal_authority, object_authority
from hymeko_rl.coin_delivery.theta_option.physical_intent import INTENT_HI, INTENT_LO, PhysicalIntent, slew_of
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, ThetaBox

PUSH, BRAKE, RELEASE = 0, 1, 2
_PHASE_NAME = {PUSH: "PUSH", BRAKE: "BRAKE", RELEASE: "RELEASE"}
_EPS = 1e-9


@dataclass(frozen=True)
class CorrectionParams:
    """The FROZEN (dev-tuned in C1) coast-in + magnitude-correction parameters. `a_friction` is the coast deceleration
    (a physical constant of the coin/surface, cradle-agnostic); the rest are conservative deadbanded gains. Every
    correction is clipped to the intent box."""

    # regime-aware passthrough: a high-response / brake-hold cradle (very large object forward_push_reach) whose grip cannot
    # coast-settle a fast coin defers to its decoded open-loop θ (which already delivers — "do no harm"); the coast-in
    # applies only to coast-type cradles. ∞ by default ⇒ uniform coast-in (passthrough OFF).
    passthrough_reach: float = float("inf")  # object forward_push_reach above which the controller passes the decoded θ through
    # R5 adaptive coast identification (OFF by default ⇒ exact R4 behaviour): estimate â_eff online + a grip-held
    # brake-probe to seed the estimate before the first release decision.
    adaptive: bool = False                  # use the online CoastEstimator's â_eff instead of the fixed a_friction
    probe_steps: int = 3                    # grip-hold coast-probe length after the push (seeds â_eff; holds a fast coin)
    probe_brake_gain: float = 0.0           # brake gain DURING the probe: 0 ⇒ pure grip-hold coast ⇒ â_eff ≈ friction (NOT
                                            #   the decoded brake gain, which measures braking, not coasting — the A0 bug)
    est_a_lo: float = 0.20                  # â_eff clamp band
    est_a_hi: float = 1.20
    est_k: int = 8                          # estimator sliding window
    # coast-in phase trigger
    a_friction: float = 0.55                # m/s² — coast deceleration (R4 fixed model; R5 prior when adaptive)
    release_zone: float = 0.02              # m — coast target is dtz − release_zone (coast into the zone)
    coast_safety: float = 1.0               # release when coast_reach ≥ coast_safety·(dtz − release_zone)
    overshoot_margin: float = 0.06          # m — BRAKE-trim only if coast_reach exceeds the target by this much
    min_push_steps: int = 4                 # never RELEASE before this many steps (ensure some momentum is built)
    # magnitude correction (applied from the probe onward, at the cadence)
    w_probe: int = 6                        # probe window before the first magnitude correction
    c_corr: int = 4                         # magnitude re-correction cadence after the probe
    initial_source: str = "r3_predictor"    # {"r3_predictor", "conservative_prior"}
    push_squeeze_cap: float = 0.10          # cap squeeze during the build (reduce over-grip that stalls the coin)
    k_squeeze_cut: float = 0.6              # fraction of the excess squeeze removed per correction
    # coast-deficit forward boost (ChatGPT-advised; OFF by default k_forward_deficit=0): boost only when the PREDICTED
    # coast falls short of the corridor (d_remaining − coast_reach > deadband) AND contact is stable — so a fast cradle
    # (s3, deficit ≤ 0) is never over-driven, only a momentum-starved one (s7).
    k_forward_deficit: float = 0.0          # Δforward per m of coast deficit
    deficit_deadband: float = 0.02          # m — ignore a small coast deficit
    fn_floor: float = 0.05                  # N — below this min fₙ (or contact lost) ⇒ squeeze up (retention)
    k_squeeze_up: float = 0.08              # ↑ squeeze on contact-retention risk
    # cross-track / spin (canonical perpendicular)
    k_lateral: float = 0.25                 # Δlateral per (m/s) of canonical v_perp
    k_spin: float = 0.02                    # Δlateral per (rad/s) of canonical spin
    lateral_deadband: float = 0.05          # m/s — ignore small perpendicular drift (C0 no-op)

    def as_dict(self) -> "dict[str, Any]":
        return {k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def coast_reach(v_parallel: float, a_friction: float) -> float:
    """The along-track distance a coin at closing speed ``v_parallel`` will travel before friction brings it to rest:
    v²/(2·a_friction). Only the forward (toward-zone) component counts; a receding coin reaches 0. # Postconditions: ≥ 0;
    monotone increasing in v_parallel; monotone decreasing in a_friction."""
    v = max(float(v_parallel), 0.0)
    return v * v / (2.0 * max(float(a_friction), _EPS))


def coastin_phase(dtz: float, v_parallel: float, p: CorrectionParams, elapsed: int, phase_floor: int,
                  min_push: "int | None" = None, a_eff: "float | None" = None) -> int:
    """The coast-in brake-vs-coast DECISION applied AFTER the decoded push window. PUSH (build/extend) while the momentum
    cannot yet coast to the zone; RELEASE once it can (coast in, NO brake); BRAKE only if a large overshoot is predicted
    (a fast / brake-hold cradle like s3). ``min_push`` (default the decoded θ ramp, else `p.min_push_steps`) is the build
    window that RESPECTS each cradle's tuned push duration — the generic coast-in must not release earlier than the θ's own
    ramp, or it destabilises the precisely-timed teacher plans (the C0 lesson). # Postconditions: ≥ phase_floor; never
    returns PUSH once RELEASE/BRAKE was reached."""
    mp = int(p.min_push_steps if min_push is None else min_push)
    target = max(float(dtz) - p.release_zone, 0.0)
    reach = coast_reach(v_parallel, p.a_friction if a_eff is None else a_eff)   # R5: online â_eff replaces the fixed coast
    # Overshoot detection comes FIRST — a fast / brake-hold cradle (s3) must arrest the moment it has too much momentum,
    # even inside the build window, or it accelerates past the grip and flies through the zone.
    if reach >= target + p.overshoot_margin:
        return max(phase_floor, BRAKE)                        # too much momentum ⇒ trim / arrest
    if phase_floor < RELEASE and elapsed <= mp:
        return max(phase_floor, PUSH)                         # respect the decoded push duration (do not release early)
    if reach >= p.coast_safety * target:
        return max(phase_floor, RELEASE)                      # enough to coast into the zone ⇒ release & settle
    return max(phase_floor, PUSH)                             # still short ⇒ extend the build


def _canon_perp(resp: ResponseState, was_swapped: bool) -> "tuple[float, float]":
    """Canonicalise the reflection-antisymmetric response signals (v_perp, spin): flip iff was_swapped, so a cradle and its
    mirror hand the law the SAME canonical perpendicular error. # Postconditions: invariant under (mirror, swap-flip)."""
    s = -1.0 if was_swapped else 1.0
    return s * resp.v_perpendicular, s * resp.coin_spin


class IntentCorrector:
    """The deterministic canonical-frame MAGNITUDE correction over the R3 roles (the coast-in TIMING is applied by the
    controller's phase trigger, not here). Pure given (intent, response, was_swapped); no physics, no future, no K6, no
    teacher. # Postconditions: every returned intent is clipped to the frozen intent box; forward/braking/squeeze deltas
    depend only on mirror-invariant signals; lateral on the canonicalised perpendicular."""

    def __init__(self, params: CorrectionParams = CorrectionParams()) -> None:
        self.p = params

    def correct(self, intent: PhysicalIntent, resp: ResponseState, *, was_swapped: bool) -> "tuple[PhysicalIntent, dict[str, Any]]":
        p = self.p
        v = intent.to_vector().copy()
        # roles: 0 forward_drive, 1 peak_velocity, 2 lateral, 3 squeeze, 4 brake_entry, 5 braking_demand, 6 release
        fired: list[str] = []
        contact_risk = (not resp.both_contact) or (min(resp.fn) < p.fn_floor)
        v_perp_c, spin_c = _canon_perp(resp, was_swapped)
        # ── anti-over-control (the held-out failure mode): cut the over-squeeze that stalls the coast. NEVER touch braking
        #    or forward — the coast-in TIMING (release when the coin can coast in; BRAKE-trim only when it overshoots) owns
        #    the momentum, and boosting forward over-drives a fast cradle (s3) straight through the zone. ──
        if v[3] > p.push_squeeze_cap and not contact_risk:
            v[3] = v[3] - p.k_squeeze_cut * (v[3] - p.push_squeeze_cap)      # ↓ over-squeeze (frees the coast)
            fired.append("cut_squeeze")
        # ── coast-deficit forward boost (only when the PREDICTED coast lands short ∧ contact stable ⇒ s7, not s3) ──
        deficit = max(resp.dtz - p.release_zone, 0.0) - coast_reach(resp.v_parallel, p.a_friction)
        if p.k_forward_deficit > 0.0 and deficit > p.deficit_deadband and not contact_risk:
            v[0] = v[0] + p.k_forward_deficit * (deficit - p.deficit_deadband)
            fired.append("boost_forward")
        # ── contact-retention risk ⇒ squeeze up (overrides the cap) ──
        if contact_risk:
            v[3] = v[3] + p.k_squeeze_up
            fired.append("contact_squeeze")
        # ── cross-track / spin (canonical perpendicular; mirror-equivariant sign; deadbanded for C0) ──
        if abs(v_perp_c) > p.lateral_deadband:
            v[2] = v[2] - (p.k_lateral * v_perp_c + p.k_spin * spin_c)
            fired.append("cross_track")
        corrected = PhysicalIntent.from_vector(np.clip(v, INTENT_LO, INTENT_HI))
        log = {"fired": fired, "contact_risk": bool(contact_risk), "coast_deficit": round(float(deficit), 5),
               "coin_speed": round(resp.coin_speed, 5), "canon_v_perp": round(v_perp_c, 5),
               "delta_intent": [round(float(a), 5) for a in (corrected.to_vector() - intent.to_vector())]}
        return corrected, log


class PhaseMachine:
    """Monotone PUSH(0)→BRAKE(1)→RELEASE(2). Given the current θ (ramp=θ[3], rel=θ[4]), step t, and an optional STATE
    ``force_phase``, advance the phase floor (never regress), then TIMING-CLAMP a copy of θ so the frozen
    `_schedule_increment` yields exactly the (monotone) phase. # Postconditions: phase non-decreasing; the returned θ's
    ramp/rel realise the returned phase under _schedule_increment's `t≤ramp / t≤rel` test; neutral on a constant θ with
    force_phase=PUSH."""

    def __init__(self) -> None:
        self.phase = PUSH
        self.release_step: int | None = None

    def step(self, theta: np.ndarray, t: int, *, force_phase: int = PUSH) -> "tuple[np.ndarray, int]":
        th = np.asarray(theta, np.float64).copy()
        ramp, rel = float(th[3]), float(th[4])
        desired = PUSH if t <= ramp else (BRAKE if t <= rel else RELEASE)
        # ``force_phase`` lets a STATE trigger bring the phase forward (e.g. release early to coast in); combined
        # monotonically with the θ-implied phase and the running floor, it can only ADVANCE, never revert.
        self.phase = max(self.phase, desired, int(force_phase))
        if self.phase == PUSH:
            th[3] = max(ramp, float(t))                        # ensure t ≤ ramp'  (stay in PUSH)
            th[4] = max(rel, th[3])
        elif self.phase == BRAKE:
            th[3] = min(ramp, float(t - 1))                    # t > ramp'  ⇒ not PUSH
            th[4] = max(rel, float(t))                         # t ≤ rel'   ⇒ not yet RELEASE
        else:                                                  # RELEASE
            th[4] = min(rel, float(t - 1))                     # t > rel'   ⇒ RELEASE
            th[3] = min(ramp, th[4])
            if self.release_step is None:
                self.release_step = int(t)
        return th, self.phase


@dataclass
class SnapContext:
    """The snapshot-dependent (θ-independent) decode + measurement context, computed ONCE per cradle and shared across all
    budget candidates (the authority FD is the expensive part). Reuses the FROZEN R3 canonical geometry + handoff
    authority (`identify_Bcoin`/`object_authority`/`contact_internal_authority`) — identical to what the R3 decoder reads."""

    g_canon: "dict[str, np.ndarray]"
    was_swapped: bool
    slew: float
    horizon: float
    authority: "dict[str, float]"
    ctx: MeasureContext

    @staticmethod
    def build(snap: CradleSnapshot, cfg: Any = DELIVERY_CFG) -> "SnapContext":
        g_canon, was_swapped = canonicalise(r1_grouped_features(snap))
        oa, ci = object_authority(snap), contact_internal_authority(snap)
        authority = {"forward_push_reach": float(oa["forward_push_reach"][0]),
                     "brake_opposed_reach": float(oa["brake_opposed_reach"][0]),
                     "lateral_reach": float(np.mean(oa["lateral_reach_pair"])),
                     "normal_force_reach": float(np.mean(ci["normal_force_reach_pair"])),
                     "balance_reach_signed": float(ci["balance_reach_signed"][0])}
        ctx = MeasureContext(authority=authority, control_dt=float(snap.stack.control_dt), m_coin=_coin_mass(snap),
                             slew=slew_of(snap), lo=np.asarray(snap.lo, np.float64), hi=np.asarray(snap.hi, np.float64),
                             prev_tau=np.asarray(snap.prev_tau, np.float64))
        return SnapContext(g_canon=g_canon, was_swapped=bool(was_swapped), slew=slew_of(snap),
                           horizon=float(cfg.horizon), authority=authority, ctx=ctx)


class ClosedLoopController:
    """Drives one continuous closed-loop trajectory. Every step it reads the live coin state and applies the coast-in
    phase trigger (state-driven RELEASE, never braking unless overshoot); at the probe/cadence it measures the full causal
    response and re-decodes a corrected MAGNITUDE intent (deterministic, NO extra search budget). Reuses the FROZEN R3 pure
    decoder core `decode_from_canonical` with the snapshot's pre-computed canonical geometry + handoff authority, so a
    re-decode is physics-free. `theta_for_step` is what `closed_loop_rollout` calls each step.

    # Preconditions: `probe_theta0` is a legal θ (the search candidate for the initial segment); `base_intent` is the
    initial physical intent (R3 predictor or conservative prior). # Postconditions: NEVER a teacher θ; every returned θ is
    box-legal and phase-monotone; corrections/responses logged for provenance."""

    def __init__(self, snap: CradleSnapshot, base_intent: PhysicalIntent, probe_theta0: np.ndarray,
                 corrector: IntentCorrector, params: CorrectionParams = CorrectionParams(),
                 cfg: Any = DELIVERY_CFG, snap_ctx: "SnapContext | None" = None) -> None:
        self.snap, self.base_intent, self.corrector, self.p, self.cfg = snap, base_intent, corrector, params, cfg
        self.box = ThetaBox()
        self.probe_theta0 = self.box.clip(np.asarray(probe_theta0, np.float64))
        sc = snap_ctx or SnapContext.build(snap, cfg)
        self.g_canon, self.was_swapped, self.slew, self.horizon = sc.g_canon, sc.was_swapped, sc.slew, sc.horizon
        self.authority, self.ctx = sc.authority, sc.ctx
        self.reset()

    def reset(self) -> None:
        self.pm = PhaseMachine()
        self.cur_intent = self.base_intent
        self.cur_theta = np.asarray(self.probe_theta0, np.float64).copy()
        self.ref: ProbeReference | None = None
        self.corrections: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.est = CoastEstimator(a_prior=self.p.a_friction, a_lo=self.p.est_a_lo, a_hi=self.p.est_a_hi, k_est=self.p.est_k)
        self._prev_v: float | None = None
        self._prev_push = False
        self.a_eff_trace: list[dict[str, Any]] = []

    @property
    def phase(self) -> int:
        return self.pm.phase

    def before_release(self, t: int) -> bool:
        """Contact-retention bookkeeping window = PUSH+BRAKE (phase < RELEASE). Called AFTER `theta_for_step(t)`."""
        return self.pm.phase < RELEASE

    @property
    def release_boundary_step(self) -> "int | None":
        return self.pm.release_step

    def _live_state(self, rl: Any) -> "tuple[float, float]":
        """Cheap per-step (dtz, v_parallel) read for the coast-in trigger (no full ResponseState)."""
        u, dtz = rl.inner.direction_to_zone()
        e_par = np.asarray(u, np.float64)[:2]
        n = float(np.linalg.norm(e_par))
        e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
        v = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
        return float(dtz), float(v @ e_par)

    def _redecode(self, intent: PhysicalIntent) -> DecoderResult:
        return decode_from_canonical(intent, self.g_canon, self.was_swapped, self.slew, self.horizon)

    def _should_correct(self, t: int) -> bool:
        if t < self.p.w_probe + 1:
            return False
        return t == self.p.w_probe + 1 or (t - (self.p.w_probe + 1)) % max(1, self.p.c_corr) == 0

    def theta_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        if self.ref is None:
            self.ref = ProbeReference.capture(rl, t)
        if self.authority["forward_push_reach"] > self.p.passthrough_reach:
            # regime-aware: a high-response / brake-hold cradle defers to its decoded open-loop θ (own timing delivers).
            eff, _phase = self.pm.step(self.cur_theta, t)
            return self.box.clip(eff)
        dtz, v_par = self._live_state(rl)
        if self._should_correct(t):                            # magnitude re-decode from the measured response
            self.ctx = MeasureContext(**{**self.ctx.__dict__, "prev_tau": np.asarray(prev_tau, np.float64)})
            resp = measure_response(rl, self.ref, self.ctx, phase=_PHASE_NAME[self.pm.phase], t=t, horizon=int(self.horizon))
            self.cur_intent, log = self.corrector.correct(self.cur_intent, resp, was_swapped=self.was_swapped)
            self.cur_theta = np.asarray(self._redecode(self.cur_intent).physical_theta, np.float64)
            self.responses.append({"t": int(t), **resp.as_dict()})
            self.corrections.append({"t": int(t), "intent": [round(float(x), 5) for x in self.cur_intent.to_vector()], **log})
        # RESPECT the decoded push duration (θ ramp) as the build window, then let the coast-in decide brake-vs-coast.
        # KEEP the decoded ramp in θ[3] — it drives the PUSH ramp fraction min(1, t/ramp); the phase machine's PUSH clamp
        # (max(ramp,t)) then gives a correct ramp-up then a full push, so the coin is NOT under-pushed. Only θ[4] (release)
        # is neutralised so the θ never self-releases; the coast-in ``force_phase`` decides RELEASE (coast) vs BRAKE
        # (overshoot). Only the MAGNITUDE roles (squeeze/forward/brake_gain/balance) come from the decode.
        # R5: feed the online coast estimator with this step's along-track velocity change (valid only on a no-active-push
        # decelerating step), then read â_eff; a grip-held BRAKE-probe after the push seeds the estimate + holds a fast coin.
        if self._prev_v is not None:
            self.est.update(self._prev_v, v_par, self.ctx.control_dt, active_push=self._prev_push)
        theta_mag = self.cur_theta.copy()
        ramp = int(round(float(theta_mag[3])))
        theta_mag[4] = self.horizon
        if self.p.adaptive:
            a_eff = self.est.estimate()
            probe_end = ramp + self.p.probe_steps
            if ramp < t <= probe_end and self.pm.phase < RELEASE:
                want = max(self.pm.phase, BRAKE)               # grip-hold probe: measure â_eff without releasing the coin
                theta_mag[5] = self.p.probe_brake_gain         # ≈0 brake ⇒ the coin coasts under FRICTION, grip held
            else:
                want = coastin_phase(dtz, v_par, self.p, elapsed=t, phase_floor=self.pm.phase, min_push=probe_end, a_eff=a_eff)
            self.a_eff_trace.append({"t": int(t), "a_eff": round(float(a_eff), 5), "n": self.est.n_samples,
                                     "v_par": round(float(v_par), 5), "phase": _PHASE_NAME[want]})
        else:
            want = coastin_phase(dtz, v_par, self.p, elapsed=t, phase_floor=self.pm.phase, min_push=ramp)
        eff, phase = self.pm.step(theta_mag, t, force_phase=want)
        self._prev_v, self._prev_push = v_par, (phase == PUSH)
        return self.box.clip(eff)


def _coin_mass(snap: CradleSnapshot) -> float:
    """The coin (disk) body mass from the frozen model (a constant of the cradle). # Postconditions: > 0."""
    import mujoco
    rl = snap.branch()
    m = rl.inner.model
    bid = int(rl.inner._disk_body) if getattr(rl.inner, "_disk_body", None) is not None else mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, "disk")
    return float(m.body_mass[bid]) if bid is not None and bid >= 0 else 0.05


def initial_intent_prior() -> PhysicalIntent:
    """A conservative dev-safe initial intent (a moderate push-and-coast plan): decent forward, low squeeze, no active
    brake — matching the measured PUSH-then-COAST delivering strategy, so the coast-in correction can only refine it.
    Frozen; carries no held-out information."""
    return PhysicalIntent(forward_drive=0.10, peak_velocity=0.30, lateral=0.0, squeeze=0.08,
                          brake_entry=0.20, braking_demand=0.20, release=0.20).clip()
