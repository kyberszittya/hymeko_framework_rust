"""Spring-legged AIBO — series-elastic knees store & return launch energy; loading is motor-bounded.

Certifies: (1) the spring model builds and stands; (2) a loaded spring released launches the body
airborne where a rigid knee cannot (elastic energy return); (3) the fast release is a passive spring
(exceeds the motor speed cap — legitimate, not a fake motor); (4) the realistic knee motor can only
statically load a tiny deflection (a launch spring needs dynamic/body-weight loading).
"""

from __future__ import annotations

import mujoco
import pytest

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.spring_leg import (
    KNEE_MOTOR_TAU_REALISTIC,
    ElasticLaunch,
    SpringLegSpec,
    add_knee_springs,
    build_spring_legged,
    stands,
)

_MOTOR_SPEED_CAP = 8.0  # rad/s — the realistic joint-speed motion contract (a geared motor)


@pytest.fixture(scope="module")
def mjcf() -> str:
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                           max_steps=200)
    return env._mjcf


@pytest.fixture(scope="module")
def spring_model(mjcf: str) -> mujoco.MjModel:
    return build_spring_legged(mjcf, SpringLegSpec(stiffness=150.0, springref=0.0, damping=0.05))


# --- spec contract ---------------------------------------------------------------------------

def test_spec_rejects_non_positive_stiffness() -> None:
    with pytest.raises(ValueError):
        SpringLegSpec(stiffness=0.0)
    with pytest.raises(ValueError):
        SpringLegSpec(stiffness=-1.0)


def test_spec_rejects_negative_damping() -> None:
    with pytest.raises(ValueError):
        SpringLegSpec(damping=-0.1)


def test_stored_energy_is_half_k_deflection_squared() -> None:
    spec = SpringLegSpec(stiffness=100.0, springref=0.0)
    assert spec.stored_energy(-1.0, n_legs=4) == pytest.approx(4 * 0.5 * 100.0 * 1.0**2)
    # monotone in |deflection|
    assert spec.stored_energy(-1.0) > spec.stored_energy(-0.5) > 0.0


# --- MJCF injection --------------------------------------------------------------------------

def test_add_knee_springs_sets_all_four_knees(mjcf: str) -> None:
    out = add_knee_springs(mjcf, SpringLegSpec(stiffness=120.0, springref=0.0, damping=0.04))
    assert out.count('stiffness="120.0"') == 4      # all four knees, no more
    assert out.count('springref="0.0"') == 4
    m = mujoco.MjModel.from_xml_string(out)
    for leg in ("fl", "fr", "bl", "br"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"knee_{leg}")
        assert float(m.jnt_stiffness[jid]) == pytest.approx(120.0)


def test_add_knee_springs_missing_knee_raises() -> None:
    with pytest.raises(ValueError):
        add_knee_springs("<mujoco/>", SpringLegSpec())


# --- the model stands ------------------------------------------------------------------------

def test_spring_model_builds_and_stands(spring_model: mujoco.MjModel) -> None:
    settled, peak = stands(spring_model)
    assert 0.10 < settled < 0.30                 # a plausible standing height
    assert peak - settled < 0.10                 # steady — it does NOT spontaneously launch


# --- elastic energy return: the launch -------------------------------------------------------

def test_spring_launches_airborne_where_rigid_does_not(mjcf: str, spring_model: mujoco.MjModel) -> None:
    rigid = mujoco.MjModel.from_xml_string(mjcf)
    rigid_r = ElasticLaunch(rigid, load_angle=-1.0).launch()
    spring_r = ElasticLaunch(spring_model, load_angle=-1.0).launch()
    # the rigid geared knee stores no energy — no launch
    assert not rigid_r.airborne
    assert rigid_r.rise < 0.05
    # the series-elastic knee returns stored PE as an airborne launch
    assert spring_r.airborne
    assert spring_r.flight_steps > 50
    assert spring_r.rise > 0.15
    assert spring_r.stored_energy > 0.0


def test_release_speed_exceeds_motor_cap_proving_passive_spring(spring_model: mujoco.MjModel) -> None:
    # the launch velocity comes from the PASSIVE spring, which is not bound by the geared motor's
    # speed cap — this is legitimate series elasticity, distinct from a fabricated fast motor.
    r = ElasticLaunch(spring_model, load_angle=-1.0).launch()
    assert r.peak_release_speed > _MOTOR_SPEED_CAP


def test_launch_is_deterministic(spring_model: mujoco.MjModel) -> None:
    a = ElasticLaunch(spring_model, load_angle=-1.0).launch()
    b = ElasticLaunch(spring_model, load_angle=-1.0).launch()
    assert a.rise == b.rise and a.flight_steps == b.flight_steps


# --- honest loading constraint (the realistic motor) -----------------------------------------

def test_static_load_limit_shrinks_with_stiffness() -> None:
    soft_theta, _ = SpringLegSpec(stiffness=30.0).static_load_limit()
    stiff_theta, _ = SpringLegSpec(stiffness=150.0).static_load_limit()
    assert soft_theta > stiff_theta > 0.0        # a stiffer spring is harder to statically load


def test_realistic_motor_cannot_statically_load_a_launch_spring() -> None:
    # with the realistic ~5 N·m knee motor, the statically-loadable energy is tiny (< a few J),
    # far below the ~300 J the launch demo stores — hence a launch-capable spring must be loaded
    # DYNAMICALLY (body weight / landing momentum), the honest SLIP constraint.
    _theta, e_knee = SpringLegSpec(stiffness=150.0).static_load_limit(tau_max=KNEE_MOTOR_TAU_REALISTIC)
    assert 4 * e_knee < 2.0
    launch_stored = SpringLegSpec(stiffness=150.0).stored_energy(-1.0)
    assert launch_stored > 50 * (4 * e_knee)     # the launch stores orders of magnitude more
