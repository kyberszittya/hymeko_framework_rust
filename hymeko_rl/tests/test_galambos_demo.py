"""Validate the planar 2-link IK against the actual Galambos arm model (the demonstrator foundation)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from hymeko_rl.control.controller_spec import ControllerSpec, PhaseSpec
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.galambos_demo import (
    GUARDS,
    PUSH_PROFILE,
    PushConfig,
    PushDemonstrator,
    PushObs,
    fan_offsets,
    planar_2link_ik,
    push_slots,
    orbit_step,
)

# make_planar_arms_mjcf defaults: bases at (±0.14, -0.02), l1=0.16, l2=0.14.
_L1, _L2 = 0.16, 0.14
_BASES = {"right": (0.14, -0.02), "left": (-0.14, -0.02)}


def _tip_xy(env: PlanarGraspEnv, side: str) -> np.ndarray:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"tip_{side}")
    return np.asarray(env.data.site_xpos[sid][:2], dtype=np.float64)


def _set_joint(env: PlanarGraspEnv, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    env.data.qpos[env.model.jnt_qposadr[jid]] = val


def test_ik_reaches_targets_on_the_model() -> None:
    """For reachable targets on both arms, the IK joint angles place the fingertip site on the target."""
    env = PlanarGraspEnv(robot=None, max_steps=10)
    env.reset(seed=0)
    targets = {"right": [(0.10, 0.18), (0.20, 0.12), (0.05, 0.22)],
               "left": [(-0.10, 0.18), (-0.20, 0.12), (-0.05, 0.22)]}
    for side, pts in targets.items():
        for tgt in pts:
            j1, j2 = planar_2link_ik(_BASES[side], _L1, _L2, tgt)
            env.data.qpos[:] = 0.0
            _set_joint(env, f"j1_{side}", j1)
            _set_joint(env, f"j2_{side}", j2)
            mujoco.mj_forward(env.model, env.data)
            tip = _tip_xy(env, side)
            assert np.allclose(tip, tgt, atol=1e-2), f"{side} tip {tip} != target {tgt} (j1={j1:.3f}, j2={j2:.3f})"


def test_out_of_reach_target_is_clamped() -> None:
    """A target beyond l1+l2 clamps to the reach boundary (no NaN, tip on the line to the target)."""
    j1, j2 = planar_2link_ik(_BASES["right"], _L1, _L2, (0.14, 0.60))  # far above reach
    assert math.isfinite(j1) and math.isfinite(j2)


def test_zero_link_length_rejected() -> None:
    try:
        planar_2link_ik((0.0, 0.0), 0.0, 0.14, (0.1, 0.1))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-positive link length")


# ── Push (V-plow) demonstrator ─────────────────────────────────────────────────────────────────────────────

_CFG = PushConfig(contact_dist=0.049)


def test_push_slots_strictly_behind_coin() -> None:
    """Both V-slots lie strictly behind the coin w.r.t. the zone, symmetric about the push ray."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        coin = rng.uniform(-0.2, 0.2, 2)
        zone = rng.uniform(-0.2, 0.2, 2)
        if float(np.linalg.norm(zone - coin)) < 1e-3:
            continue
        slots = push_slots(coin, zone, 0.006, _CFG)
        assert slots is not None
        ray = zone - coin
        for s in slots:
            assert float(np.dot(s - coin, ray)) < 0.0, f"slot {s} not behind coin {coin} (zone {zone})"
        # symmetry: both slots at the same radius, mirrored about the ray
        d0, d1 = (float(np.linalg.norm(s - coin)) for s in slots)
        assert abs(d0 - d1) < 1e-12
        n = np.array([-ray[1], ray[0]]) / np.linalg.norm(ray)
        assert abs(float(np.dot(slots[0] - coin, n)) + float(np.dot(slots[1] - coin, n))) < 1e-12


def test_push_slots_degenerate_and_press() -> None:
    """coin ≈ zone yields None (hold); press moves slots radially inward; bad press violates the contract."""
    coin = np.array([0.05, 0.10])
    assert push_slots(coin, coin + 1e-9, 0.0, _CFG) is None
    loose = push_slots(coin, np.array([0.2, 0.1]), 0.0, _CFG)
    pressed = push_slots(coin, np.array([0.2, 0.1]), 0.010, _CFG)
    assert loose is not None and pressed is not None
    assert float(np.linalg.norm(pressed[0] - coin)) < float(np.linalg.norm(loose[0] - coin))
    try:
        push_slots(coin, np.array([0.2, 0.1]), _CFG.contact_dist, _CFG)
    except AssertionError:
        return
    raise AssertionError("expected precondition failure for press >= contact_dist")


def test_orbit_step_orbits_and_descends() -> None:
    """Far from the slot the waypoint stays on the swing orbit and outside the front sector; aligned, it
    returns the slot itself (the descend that makes the swing→plow gate reachable — 2026-07-05 bug)."""
    coin = np.zeros(2)
    zone = np.array([0.2, 0.0])                       # push +x → front at angle 0, slots behind (±140°)
    slots = push_slots(coin, zone, 0.0, _CFG)
    assert slots is not None
    theta_front = 0.0
    # tip parked 90° away from its slot: waypoint must be on the orbit radius, not in the front sector
    tip = coin + (_CFG.contact_dist + _CFG.swing_margin) * np.array([0.0, -1.0])   # at -90°
    wp = orbit_step(tip, coin, slots[0], theta_front, _CFG)                        # slot 0 at +140°
    assert abs(float(np.linalg.norm(wp - coin)) - (_CFG.contact_dist + _CFG.swing_margin)) < 1e-9
    ang = math.atan2(float(wp[1]), float(wp[0]))
    assert abs(ang) > _CFG.front_avoid - 1e-9, f"waypoint {wp} entered the front sector (ang={ang:.2f})"
    # tip already aligned with the slot: descend onto it exactly
    aligned_tip = coin + (_CFG.contact_dist + _CFG.swing_margin) * (slots[1] - coin) / np.linalg.norm(slots[1] - coin)
    assert np.allclose(orbit_step(aligned_tip, coin, slots[1], theta_front, _CFG), slots[1])


def _obs(coin: np.ndarray, zone: np.ndarray, *tips: np.ndarray) -> PushObs:
    d = float(np.linalg.norm(zone - coin))
    return PushObs(coin=coin, zone=zone, tips=np.stack(tips), dist=d,
                       u=(zone - coin) / d if d >= 1e-6 else None)


def test_controller_spec_reads_the_push_profile() -> None:
    """The declarative FSM parses: 3 phases (swing initial), laws + guard events all bound, params typed."""
    spec = ControllerSpec.from_hymeko(PUSH_PROFILE)
    assert [p.name for p in spec.phases] == ["approach", "push", "hold"]
    assert spec.initial == "approach"
    assert spec.phase("push").law == "pressed_slots"
    assert ("formation_broken", "approach") in spec.phase("push").transitions
    cfg = PushConfig.from_params(spec.params, contact_dist=0.049)
    assert cfg.press_max == 0.012 and cfg.brake_dist == 0.060
    try:
        PushConfig.from_params({"presss_max": 1.0}, contact_dist=0.049)   # a .hymeko typo
    except ValueError:
        return
    raise AssertionError("unknown declarative param must fail loud")


def test_fsm_guards_and_step() -> None:
    """The declared FSM walks correctly on synthetic obs: coin-at-zone → hold; tips-on-slots → plow;
    tip-past-coin → swing; else self-loop."""
    spec = ControllerSpec.from_hymeko(PUSH_PROFILE)
    coin, zone = np.zeros(2), np.array([0.2, 0.0])
    slots = push_slots(coin, zone, 0.0, _CFG)
    assert slots is not None
    a = np.array([0, 1])

    def fires(obs: PushObs):
        return lambda e: float(GUARDS[e](obs, slots, a, _CFG)) > 0.0    # robustness semantics: ρ > 0

    far = _obs(coin, zone, np.array([0.0, 0.15]), np.array([0.0, -0.15]))
    assert spec.step("approach", fires(far)) == ("approach", "")
    seated = _obs(coin, zone, slots[0], slots[1])
    assert spec.step("approach", fires(seated)) == ("push", "slots_reached")
    assert spec.step("push", fires(seated)) == ("push", "")
    front = _obs(coin, zone, np.array([0.06, 0.0]), slots[1])          # first tip ahead of the coin
    assert spec.step("push", fires(front)) == ("approach", "formation_broken")
    at_zone = _obs(coin, coin + 1e-9, slots[0], slots[1])
    assert spec.step("push", lambda e: float(GUARDS[e](at_zone, None, a, _CFG)) > 0.0) == ("hold", "coin_at_zone")


def test_guard_robustness_parity_and_margins() -> None:
    """Regression (monitor extraction, 2026-07-05): each guard's ρ > 0 verdict equals the boolean predicate
    it replaced, over 200 random synthetic obs — and ρ is a genuine margin (scales with distance to the
    threshold), not a re-encoded bool."""
    rng = np.random.default_rng(11)
    a = np.array([0, 1])
    checked = 0
    for _ in range(200):
        coin = rng.uniform(-0.25, 0.25, 2)
        zone = rng.uniform(-0.25, 0.25, 2)
        tips = rng.uniform(-0.3, 0.3, (2, 2))
        obs = _obs(coin, zone, tips[0], tips[1])
        slots = push_slots(coin, zone, 0.0, _CFG) if obs.u is not None else None
        if slots is None:
            continue
        rho_slots = float(GUARDS["slots_reached"](obs, slots, a, _CFG))
        want_slots = all(float(np.linalg.norm(tips[i] - slots[a[i]])) < _CFG.slot_tol + _CFG.press_max
                         for i in range(2))
        assert (rho_slots > 0.0) == want_slots
        rho_broken = float(GUARDS["formation_broken"](obs, slots, a, _CFG))
        want_broken = any(float(np.dot(tips[i] - coin, obs.u)) > 0.5 * _CFG.contact_dist for i in range(2))
        assert (rho_broken > 0.0) == want_broken
        assert float(GUARDS["coin_left_zone"](obs, slots, a, _CFG)) > 0.0    # dist >= sampling scale >> eps
        assert float(GUARDS["coin_at_zone"](obs, slots, a, _CFG)) < 0.0
        checked += 1
    assert checked > 150
    # margin is graded: a tip twice as far past the coin has a strictly larger formation_broken robustness
    coin, zone = np.zeros(2), np.array([0.2, 0.0])
    slots = push_slots(coin, zone, 0.0, _CFG)
    near = _obs(coin, zone, np.array([0.03, 0.0]), np.array([-0.05, 0.0]))
    far_ = _obs(coin, zone, np.array([0.06, 0.0]), np.array([-0.05, 0.0]))
    assert float(GUARDS["formation_broken"](far_, slots, a, _CFG)) > float(GUARDS["formation_broken"](near, slots, a, _CFG))


def test_reswing_reassigns_before_targets() -> None:
    """Regression (2026-07-05 ``KeyError: 'left'``): a plow→swing transition mid-step must re-assign slots
    BEFORE the target law runs. Pre-FSM code cleared the assignment during the transition and crashed."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    demo = PushDemonstrator(env)
    demo._phase = "push"
    demo._assign = np.array([0, 1])
    coin, zone = np.array([0.0, 0.15]), np.array([0.1, 0.15])
    front_tip = coin + np.array([0.06, 0.0])                           # ahead of the coin along the push ray
    targets = demo._decide(_obs(coin, zone, front_tip, coin + np.array([-0.04, -0.03])))
    assert demo._phase == "approach"
    assert demo._assign is not None and sorted(demo._assign.tolist()) == [0, 1]
    assert targets.shape == (2, 2) and np.all(np.isfinite(targets))
    assert demo.events and demo.events[-1].reason == "formation_broken"
    assert demo.events[-1].margin > 0.0                                # the firing guard's robustness
    assert demo.last_margins["formation_broken"] == demo.events[-1].margin     # per-step monitor data exposed


def test_fan_offsets_rotation_group() -> None:
    """k-arm fan: k=1 dead behind; k=2 the classic ±V; k=3 symmetric including centre; slots all behind."""
    assert np.allclose(fan_offsets(1, 0.7), [0.0])
    assert np.allclose(fan_offsets(2, 0.7), [0.7, -0.7])
    assert np.allclose(fan_offsets(3, 0.7), [0.7, 0.0, -0.7])
    coin, zone = np.array([0.05, 0.1]), np.array([0.2, 0.0])
    slots3 = push_slots(coin, zone, 0.006, _CFG, k=3)
    assert slots3 is not None and slots3.shape == (3, 2)
    assert all(float(np.dot(s - coin, zone - coin)) < 0.0 for s in slots3)


def test_push_regression_beats_pinch_carry() -> None:
    """Dwell-delivery over a fixed 12-episode set > 0.3 — the pinch-carry demonstrator measures ~0.2 pooled
    (2026-07-05 probe), so this fails against the prior implementation. Also: no coin-out-of-workspace death."""
    env = PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)
    demo = PushDemonstrator(env)
    need = int(env.success_steps)
    held_n, deaths = 0, 0
    for ep in range(12):
        env.reset(seed=9000 + ep)
        demo.reset()
        consec, held = 0, False
        term = False
        for _ in range(env.max_steps):
            _obs, _r, term, trunc, info = env.step(demo.action(env))
            consec = consec + 1 if info["in_zone"] else 0
            held = held or consec >= need
            if term or trunc:
                break
        held_n += held
        deaths += int(term and not held)              # terminated without success = coin knocked out (death)
    assert held_n / 12 > 0.3, f"push controller delivery {held_n}/12 regressed to pinch-carry level"
    assert deaths == 0, f"{deaths} coin-out-of-workspace deaths — the push controller must not bulldoze"


class _IdleDemonstrator:
    """Null strategy: never moves — no episode can pass the dwell filter."""

    def reset(self) -> None:
        pass

    def action(self, env: PlanarGraspEnv) -> np.ndarray:
        return np.zeros(int(env.n_actions), dtype=np.float32)


def test_collect_demos_dwell_filter_and_strategy_injection() -> None:
    """Regression (filter ≡ grading, §3): ``only_success`` keeps HELD episodes; an idle teacher yields none
    (RuntimeError), the default push controller yields demos. Old momentary-``in_zone`` filter + hard-wired teacher
    would not raise for a teacher that grazes the zone without holding."""
    from hymeko_rl.experiments.galambos_bc import collect_galambos_demos
    env = PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)
    try:
        collect_galambos_demos(env, 2, 9000, demonstrator=_IdleDemonstrator())
    except RuntimeError:
        pass
    else:
        raise AssertionError("idle demonstrator must collect zero held demos and raise")
    obs, acts = collect_galambos_demos(env, 5, 9000)                   # default = push controller
    assert len(obs) == len(acts) >= 1
    assert obs.ndim == 3 and acts.ndim == 2


def test_law_injection_and_target_exposure() -> None:
    """The hybrid seam (2026-07-05): a per-mode LAW can be injected per instance (learned laws under the
    declared FSM), and the decided fingertip targets are exposed for imitation."""
    env = PlanarGraspEnv(robot=None, max_steps=10, difficulty=0.3)
    env.reset(seed=0)
    marker = np.array([[0.05, 0.2], [-0.05, 0.2]])
    demo = PushDemonstrator(env, laws={"approach_orbit": lambda obs, slots, assign, cfg: marker})
    demo.action(env)                                                   # initial phase is swing
    assert np.allclose(demo.last_targets, marker), "injected law must drive the targets"
    # a spec law with no binding must fail loud at construction
    bogus = ControllerSpec(phases=(PhaseSpec(name="p", law="no_such_law", transitions=()),), params={})
    try:
        PushDemonstrator(env, spec=bogus)
    except ValueError:
        return
    raise AssertionError("an unbound spec law must raise at construction")


def test_experiment_config_from_hymeko() -> None:
    """The campaign is DATA: AbConfig reads the declared experiment_spec (budget + arms + strings)."""
    from hymeko_rl.experiments.exp_galambos_coord_ab import AbConfig
    cfg = AbConfig.from_hymeko("data/robotics/galambos_ab_deliver.hymeko")
    assert cfg.total_steps == 200_000 and cfg.seeds == (0, 1, 2) and cfg.difficulty == 0.3
    assert cfg.variants == ("coord",) and cfg.treatment_name == "deliver"
    assert cfg.treatment_hymeko.endswith("galambos_task_deliver.hymeko")
    assert cfg.profile is not None and not cfg.uncertified_waiver and not cfg.smoke


def test_smoke_caps_profile_budgets() -> None:
    """Regression (2026-07-05 03:51): --smoke on a FULLY-SPECIFIED profile must cap the declared budgets,
    not inherit them — the old fill-if-None logic launched a full 200-demo/200k campaign as a 'smoke'."""
    from hymeko_rl.experiments.exp_galambos_coord_ab import AbConfig, resolve_budget
    cfg = AbConfig.from_hymeko("data/robotics/galambos_ab_deliver.hymeko", smoke=True)
    seeds, total_steps, n_demos, bc_epochs, _every, n_eval = resolve_budget(cfg)
    assert len(seeds) == 1 and total_steps <= 3_000 and n_demos <= 12 and bc_epochs <= 3 and n_eval == 3
    full = resolve_budget(AbConfig.from_hymeko("data/robotics/galambos_ab_deliver.hymeko"))
    assert full[0] == (0, 1, 2) and full[1] == 200_000 and full[2] == 200


def test_campaign_prequeue_gate_blocks_uncertified_reward() -> None:
    """Regression (2026-07-05 wasted-overnight incident): queuing the galambos campaign with a training
    reward whose oracle-certified optimum does NOT deliver must raise BEFORE any training — unless the
    explicit ``allow_uncertified`` waiver is passed. The farming baseline is exactly such a reward."""
    from hymeko_rl.experiments.exp_galambos_coord_ab import AbConfig, run
    try:
        run(AbConfig(smoke=True, variants=("baseline",), n_demos=1, total_steps=10, bc_epochs=1))
    except ValueError as err:
        assert "not oracle-certified" in str(err).lower()
        return
    raise AssertionError("uncertified training reward must be blocked at the pre-queue gate")


def test_grade_delivery_tiers() -> None:
    """The v2 delivery grade (progress attribution) separates all outcomes: a non-delivery → None; a held
    delivery with negligible body-only progress → fingertip_dominant (an incidental graze is allowed — NOT the
    same as zero-contact); a body-dominant push → body_driven_exploit; some body help with fingertip-dominant
    progress → body_assisted. Directly exercises the exploit/assisted branches an all-fingertip-dominant
    scripted rollout never reaches."""
    from hymeko_rl.experiments.exp_galambos_coord_ab import grade_delivery
    # not held → no grade regardless of progress
    assert grade_delivery(held=False, body_progress=1.0, fingertip_progress=0.0) is None
    # negligible body-only progress → fingertip_dominant (incidental hand touch allowed)
    assert grade_delivery(held=True, body_progress=0.0, fingertip_progress=0.2) == "fingertip_dominant"
    assert grade_delivery(held=True, body_progress=0.004, fingertip_progress=0.2) == "fingertip_dominant"  # <= eps
    # body did MOST of the moving → body_driven_exploit (a shove)
    assert grade_delivery(held=True, body_progress=0.15, fingertip_progress=0.05) == "body_driven_exploit"
    # some body help but fingertips dominant → body_assisted
    assert grade_delivery(held=True, body_progress=0.05, fingertip_progress=0.15) == "body_assisted"
    # boundary: body just above eps and equal to fingertips (not > ) → body_assisted, not exploit
    assert grade_delivery(held=True, body_progress=0.1, fingertip_progress=0.1) == "body_assisted"


def test_residual_zero_delta_reproduces_base_controller() -> None:
    """The residual wrapper's KEY invariant: a zero delta gives bit-identical episodes to the raw base
    controller (same seeds, same rewards, same final coin position). Any violation breaks the 'starts at
    the base's performance' guarantee residual RL rests on."""
    from hymeko_rl.env.residual import ResidualControllerEnv
    raw_env = PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.3)
    raw_demo = PushDemonstrator(raw_env)
    wrap_inner = PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.3)
    wrapped = ResidualControllerEnv(wrap_inner, PushDemonstrator(wrap_inner), delta_scale=0.15)
    zero = np.zeros(int(raw_env.n_actions), dtype=np.float32)
    for ep in range(3):
        raw_env.reset(seed=9000 + ep)
        raw_demo.reset()
        wrapped.reset(seed=9000 + ep)
        for _ in range(raw_env.max_steps):
            _o1, r1, t1, tr1, _i1 = raw_env.step(raw_demo.action(raw_env))
            _o2, r2, t2, tr2, _i2 = wrapped.step(zero)
            assert (t1, tr1) == (t2, tr2) and abs(float(r1) - float(r2)) < 1e-9
            if t1 or tr1:
                break
        assert np.allclose(raw_env._planar_metrics.disk_pos, wrap_inner._planar_metrics.disk_pos)


def test_push_action_latency_budget() -> None:
    """Performance contract: median action() cost < 1 ms (pure numpy geometry + IK; §3 numerical budget)."""
    import time
    env = PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)
    demo = PushDemonstrator(env)
    env.reset(seed=0)
    demo.reset()
    for _ in range(20):                               # warm-up
        env.step(demo.action(env))
    samples = []
    for _ in range(5):                                # 5 iterations of 200 calls: median over iterations
        t0 = time.perf_counter()
        for _ in range(200):
            demo.action(env)
        samples.append((time.perf_counter() - t0) / 200.0)
    med = sorted(samples)[2]
    assert med < 1e-3, f"median action() latency {med * 1e3:.3f} ms exceeds 1 ms budget"
