"""Spring-legged (series-elastic knee) AIBO variant — energy-storing legs that can launch.

Motivation. The rigid-leg full-body hop execution (`running_mpc` report) rose only +2 cm and never
left the ground: the geared knee, capped at the realistic motor speed, cannot extend fast enough to
launch the 5.7 kg body. Real jumping robots and animals solve this with **series elasticity** — a
spring in the leg is loaded *slowly* (within the motor's limits) and released *fast* (a passive
catapult), decoupling the launch velocity from the motor velocity.

This module builds a spring-legged variant by injecting a MuJoCo joint spring (stiffness toward a
rest angle, low damping = energy return) into each knee of the emitted quadruped, and demonstrates
the elastic launch: a loaded (compressed) spring, when released, carries the body airborne — where
the rigid knee cannot.

Honesty note. The high knee speed at release is a **passive spring**, not a commanded motor: it is
physically legitimate series elasticity, *distinct* from the retracted capture-step exploit (which
was a fake 27 rad/s *motor*). The loading, however, is bounded by the realistic knee motor
(~5 N·m for a 5.7 kg robot, not the ±50 N·m MJCF placeholder — see :data:`KNEE_MOTOR_TAU_REALISTIC`
and `static_load_limit`): a small motor can only statically load a soft spring to a tiny deflection,
so a launch-capable spring must be loaded dynamically (body weight / landing momentum — the SLIP
regime). That the loading is body-weight-driven, not motor-driven, is the whole point of elasticity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

LEGS: tuple[str, ...] = ("fl", "fr", "bl", "br")

# The emitted quadruped knee actuator declares ``ctrlrange="-50 50"`` (gear 1), but 50 N·m is a
# placeholder, far too large for a 5.7 kg robot: standing needs only ~1.5–3 N·m per knee
# (body-weight/4 × shin moment arm), so a realistic AIBO-class knee servo is ~3–8 N·m. We use a
# realistic value for the loading analysis — this is the honest constraint (a small motor cannot
# statically load a launch-capable spring; that energy must come from body weight / landing).
KNEE_MOTOR_TAU_REALISTIC: float = 5.0  # N·m — realistic AIBO-class knee torque (not the ±50 placeholder)


@dataclass(frozen=True)
class SpringLegSpec:
    """A series-elastic knee: restoring torque ``stiffness·(springref − q)`` with low damping.

    # Preconditions
    ``stiffness > 0`` (a real spring); ``damping ≥ 0``. ``springref`` is the spring's rest angle —
    set near the straight standing knee (≈ 0) so the spring is neutral at stance and supports
    standing, and a crouch (``q < springref``) loads it.
    """

    stiffness: float = 150.0
    springref: float = 0.0
    damping: float = 0.05

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError(f"spring stiffness must be > 0, got {self.stiffness}")
        if self.damping < 0.0:
            raise ValueError(f"damping must be >= 0, got {self.damping}")

    def stored_energy(self, load_angle: float, n_legs: int = 4) -> float:
        """Elastic PE (J) stored across ``n_legs`` knees when each is held at ``load_angle``."""
        deflection = self.springref - load_angle
        return n_legs * 0.5 * self.stiffness * deflection**2

    def static_load_limit(self, tau_max: float = KNEE_MOTOR_TAU_REALISTIC) -> tuple[float, float]:
        """Deflection (rad) and per-knee PE (J) the motor can *statically* load against this spring.

        The motor crouches until its torque equals the spring's: ``tau_max = stiffness·θ`` ⇒
        ``θ* = tau_max/stiffness``, storing ``½·stiffness·θ*²`` per knee. Postcondition: returns
        ``(θ*, E*)`` with ``θ* ≥ 0``; a stiffer spring reaches a smaller motor-loadable deflection.
        """
        theta = tau_max / self.stiffness
        return theta, 0.5 * self.stiffness * theta**2


def add_knee_springs(mjcf: str, spec: SpringLegSpec) -> str:
    """Inject a series-elastic spring into each knee joint of ``mjcf``.

    # Preconditions
    ``mjcf`` contains one self-closing ``<joint name="knee_{leg}" .../>`` per leg in :data:`LEGS`.
    # Postconditions
    Each knee carries ``stiffness>0`` toward ``springref`` with the spec's (low) damping, replacing
    any prior damping — an energy-storing compliant knee. Other joints are untouched.
    """
    out = mjcf
    for leg in LEGS:
        match = re.search(rf'<joint name="knee_{leg}"[^>]*?/>', out)
        if match is None:
            raise ValueError(f"knee_{leg} joint not found in mjcf")
        stripped = re.sub(r'\s*damping="[^"]*"', "", match.group(0))
        injected = (stripped[:-2]
                    + f' stiffness="{spec.stiffness}" springref="{spec.springref}"'
                    + f' damping="{spec.damping}"/>')
        out = out.replace(match.group(0), injected)
    return out


def build_spring_legged(mjcf: str, spec: SpringLegSpec) -> mujoco.MjModel:
    """Build a spring-legged MuJoCo model from an emitted quadruped ``mjcf`` string.

    Precondition: ``mjcf`` is a valid quadruped MJCF (e.g. ``QuadrupedGoalEnv._mjcf``). Postcondition:
    returns a model whose four knees are series-elastic per ``spec``; the returned model stands.
    """
    return mujoco.MjModel.from_xml_string(add_knee_springs(mjcf, spec))


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of an elastic launch. ``airborne`` iff all paws cleared the ground."""

    rise: float
    flight_steps: int
    peak_release_speed: float
    stored_energy: float
    return_efficiency: float

    @property
    def airborne(self) -> bool:
        return self.flight_steps > 0


@dataclass
class ElasticLaunch:
    """Load the knee springs (compress), release the actuator, measure the passive launch.

    The spring returns stored PE as an airborne launch that a rigid knee cannot produce. Loading is
    modelled as a compressed initial pose (in a real hop the motor crouches to ``load_angle`` within
    its limits, or body weight loads it — see :meth:`SpringLegSpec.static_load_limit`); release is
    actuator-off, so the launch is **pure spring**.

    # Preconditions
    ``model`` has ``knee_{leg}``, ``paw_{leg}`` and a ``base`` free joint; ``load_angle`` is a
    compressed (crouched) knee angle. # Invariants: no actuator torque is applied during release.
    """

    model: mujoco.MjModel
    load_angle: float = -1.0
    settle_steps: int = 120
    release_steps: int = 220
    clearance: float = 0.03

    def _addr(self):
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        knee_q = {n: int(m.jnt_qposadr[jid(f"knee_{n}")]) for n in LEGS}
        knee_v = {n: int(m.jnt_dofadr[jid(f"knee_{n}")]) for n in LEGS}
        paw = {n: bid(f"paw_{n}") for n in LEGS}
        return bid("torso"), knee_q, knee_v, paw, int(m.jnt_qposadr[jid("base")])

    def _knee_stiffness(self) -> float:
        return float(self.model.jnt_stiffness[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "knee_fl")])

    def launch(self) -> LaunchResult:
        """Set the loaded (compressed) pose, release the actuator, integrate; measure the launch."""
        m = self.model
        d = mujoco.MjData(m)
        torso, knee_q, knee_v, paw, base = self._addr()
        mujoco.mj_forward(m, d)
        d.qpos[base + 2] = 0.2
        for n in LEGS:
            d.qpos[knee_q[n]] = self.load_angle  # compress the springs (load)
        mujoco.mj_forward(m, d)
        z0 = float(d.xpos[torso, 2])
        paw0 = min(float(d.xpos[paw[n], 2]) for n in LEGS)
        max_z, flight, peak_speed = z0, 0, 0.0
        for _ in range(self.release_steps):  # actuator OFF — pure passive spring
            mujoco.mj_step(m, d)
            max_z = max(max_z, float(d.xpos[torso, 2]))
            peak_speed = max(peak_speed, max(abs(float(d.qvel[knee_v[n]])) for n in LEGS))
            if min(float(d.xpos[paw[n], 2]) for n in LEGS) - paw0 > self.clearance:
                flight += 1
        rise = max_z - z0
        k = self._knee_stiffness()
        stored = 4 * 0.5 * k * self.load_angle**2 if k > 0 else 0.0
        total_mass = float(np.sum(m.body_mass))
        eff = (total_mass * 9.81 * rise) / stored if stored > 0 else 0.0
        return LaunchResult(rise=rise, flight_steps=flight, peak_release_speed=peak_speed,
                            stored_energy=stored, return_efficiency=eff)


def stands(model: mujoco.MjModel, steps: int = 300) -> tuple[float, float]:
    """Settle ``model`` from its default pose; return ``(mean settled torso z, max torso z)``.

    A spring-legged model with ``springref`` near the standing knee should settle steadily (it does
    not spontaneously launch): ``max_z`` stays close to the settled height.
    """
    d = mujoco.MjData(model)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    mujoco.mj_forward(model, d)
    zs = [float(d.xpos[torso, 2])]
    for _ in range(steps):
        mujoco.mj_step(model, d)
        zs.append(float(d.xpos[torso, 2]))
    return float(np.mean(zs[-60:])), float(np.max(zs))
