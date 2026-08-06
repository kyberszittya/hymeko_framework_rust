"""Declarative RL scenarios: build a :class:`PickPlaceEnv` from one ``.hymeko`` scenario model.

A scenario (``meta_scenario.hymeko`` vocabulary) bundles the scene geometry with ``@"…"`` references to the
robot (a kinematics composite) and the reward profile. :class:`ScenarioSpec` reads that model and constructs the
environment — the generalization of :meth:`RewardSpec.from_hymeko` from *reward* to the *whole scenario*, so the
RL problem is described in HyMeKo rather than hand-parameterized in Python. A parity test guards that the
scenario-built env is behaviorally identical to the former Python config (``fanuc_pick_env``).

The parse is the same documented bridge as :mod:`hymeko_rl.env._profile` (regex over the profile shape) until
the engine's structured IR is exposed to Python (B-003).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hymeko_rl.env._profile import read_imports, read_scene_fields

if TYPE_CHECKING:
    from hymeko_rl.env.pick_place_env import PickPlaceEnv

# scene fields that are scalars vs vectors, and their PickPlaceEnv kwarg arity. Drives both the parse→kwargs
# mapping and the precondition check, so a missing/mis-shaped field fails loudly at load, not deep in the env.
_SCALARS = ("mount_height", "table_top", "box_mass", "lift_thresh", "place_radius")
_VECTORS = {"obj_radius": 2, "target_xy": 2, "arm_home": 6}


@dataclass(frozen=True)
class ScenarioSpec:
    """A whole RL scenario read from a ``.hymeko`` model: scene geometry + robot + reward references.

    # Invariants ``obj_radius``/``target_xy`` are 2-tuples, ``arm_home`` a 6-tuple; ``robot`` is a real file.
    """

    robot: str
    reward_profile: str | None
    mount_height: float
    table_top: float
    box_mass: float
    lift_thresh: float
    place_radius: float
    max_steps: int
    obj_radius: tuple[float, ...]
    target_xy: tuple[float, ...]
    arm_home: tuple[float, ...]

    @classmethod
    def from_hymeko(cls, path: str | Path) -> "ScenarioSpec":
        """Build the spec from a scenario ``.hymeko`` (scene fields + robot/reward imports).

        # Preconditions ``path`` declares one ``scene`` with the scalar fields ``_SCALARS`` + ``max_steps`` and
          the vector fields ``_VECTORS`` (correct arity), and imports a robot (kinematics) profile.
        # Postconditions Returns a fully-populated spec; ``reward_profile`` is ``None`` only if no reward import.
        # Errors ``ValueError`` on a missing/mis-shaped field or a missing robot import; ``FileNotFoundError``.
        """
        fields = read_scene_fields(path)
        missing = [k for k in (*_SCALARS, "max_steps", *_VECTORS) if k not in fields]
        if missing:
            raise ValueError(f"{path}: scenario scene missing fields {missing}")

        def scalar(name: str) -> float:
            value = fields[name]
            if isinstance(value, tuple):
                raise ValueError(f"{path}: scene field {name!r} must be a scalar, got {value!r}")
            return value

        def vector(name: str, arity: int) -> tuple[float, ...]:
            value = fields[name]
            if not isinstance(value, tuple) or len(value) != arity:
                raise ValueError(f"{path}: scene field {name!r} must be a {arity}-vector, got {value!r}")
            return value

        robot, reward = cls._resolve_imports(path)
        if robot is None:
            raise ValueError(f"{path}: scenario imports no robot (kinematics) profile")
        return cls(
            robot=robot, reward_profile=reward,
            mount_height=scalar("mount_height"), table_top=scalar("table_top"), box_mass=scalar("box_mass"),
            lift_thresh=scalar("lift_thresh"), place_radius=scalar("place_radius"),
            max_steps=int(scalar("max_steps")),
            obj_radius=vector("obj_radius", 2), target_xy=vector("target_xy", 2),
            arm_home=vector("arm_home", 6),
        )

    @staticmethod
    def _resolve_imports(path: str | Path) -> tuple[str | None, str | None]:
        """Classify the scenario's ``@"…"`` imports into (robot, reward) by content — a reward profile declares
        a ``reward_spec``; a robot composite declares ``kinematics``. Mirrors the compiler's resolution rather
        than coupling to filenames; the meta vocabulary import matches neither and is skipped."""
        robot = reward = None
        for imported in read_imports(path):
            if not imported.is_file():
                continue
            text = imported.read_text(encoding="utf-8")
            if "reward_spec" in text:
                reward = str(imported)
            elif "kinematics" in text:
                robot = str(imported)
        return robot, reward

    def build(self, **overrides: Any) -> "PickPlaceEnv":
        """Construct the env from this scenario. ``overrides`` patch any field (e.g. ``max_steps`` for a test).

        # Postconditions Returns a ready-to-``reset`` :class:`PickPlaceEnv` equivalent to the former
          ``fanuc_pick_env`` Python config when built from ``pick_place_scenario.hymeko``.
        """
        from hymeko_rl.env.pick_place_env import PickPlaceEnv

        cfg: dict[str, Any] = dict(
            robot=self.robot, reward_profile=self.reward_profile,
            mount_height=self.mount_height, table_top=self.table_top, box_mass=self.box_mass,
            lift_thresh=self.lift_thresh, place_radius=self.place_radius, max_steps=self.max_steps,
            obj_radius=self.obj_radius, target_xy=self.target_xy, arm_home=self.arm_home,
        )
        cfg.update(overrides)
        return PickPlaceEnv(**cfg)
