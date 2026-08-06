"""The RL agent description — the MDP as data (the fourth role of the unification).

The agent's observation/action/reward structure is itself a HyMeKo artifact: an
``.hymeko`` *agent profile* that compiles through the same IR the geometry and the
networks use. ``AgentSpec`` is the typed in-memory form; the ``.hymeko`` loader
(``from_hymeko``) is the one wiring step left.

The observation/state-space half is now **wired**: ``meta_observation.hymeko`` (vocabulary)
+ ``arm_reach_observation.hymeko`` (example) declare per-vertex + global channels whose
star-expansion is the obs tensor, and :class:`hymeko_rl.env.observation.ObservationSpec`
reads them — ``ObservationSpec.from_hymeko(profile)`` returns the ordered channel kinds and
``.obs_dim`` their summed width, and ``ArmReachEnv.node_features`` now *assembles* the obs
from that spec instead of hard-coding the column layout (see
``reports/2026-06-19-hymeko-rl-declarative-observation.md``). ``AgentSpec.from_hymeko``
(this module) is the remaining MDP-level wrapper: combine that per-vertex ``obs_dim`` with
the kinematic model's vertex count and the action/reward profile. The reader routes through
a narrow profile parse until the engine's import-resolving snapshot reaches Python (B-003).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSpec:
    """A continuous-control MDP description — the MDP composed *as data* from the same ``.hymeko``
    profiles the geometry and the networks use (the fourth role of the unification).

    # Preconditions
    ``obs_dim, action_dim, n_vertices >= 1``; ``action_high > action_low``; ``max_steps >= 1``.
    # Invariants
    Immutable (frozen) — the env and policy read one consistent spec; the action
    bounds are symmetric per-dim unless a richer profile overrides. ``obs_dim`` is the **per-vertex**
    observation width; the full observation is ``(n_vertices, obs_dim)``.
    """

    obs_dim: int
    action_dim: int
    action_low: float
    action_high: float
    reward: str          # named reward strategy / task-profile stem, e.g. "arm_reach_safe_task"
    max_steps: int = 200
    n_vertices: int = 1

    def __post_init__(self) -> None:
        if self.obs_dim < 1 or self.action_dim < 1 or self.n_vertices < 1:
            raise ValueError(
                f"obs_dim/action_dim/n_vertices must be >= 1; got "
                f"{self.obs_dim}/{self.action_dim}/{self.n_vertices}")
        if self.action_high <= self.action_low:
            raise ValueError(
                f"action_high must exceed action_low; got "
                f"[{self.action_low}, {self.action_high}]")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1; got {self.max_steps}")

    @classmethod
    def from_hymeko(cls, robot: str | Path, *, obs_profile: str | Path,
                    task_profile: str | Path | None = None, control_mode: str = "position",
                    max_steps: int = 200) -> "AgentSpec":
        """Compose the MDP spec from the ``.hymeko`` sources — the "MDP as data" capstone.

        ``obs_dim`` and the kinematic ``n_vertices`` come from the observation profile + the emitted
        robot's hypergraph; ``action_dim`` and the action bounds from the emitted model's actuators;
        ``reward`` names the task profile (its terms are read by :class:`RewardSpec` at env build).

        # Preconditions ``robot`` / ``obs_profile`` exist; the CLI is built.
        # Postconditions an :class:`AgentSpec` whose dims/bounds match an :class:`ArmReachEnv` built
        from the same sources (a test asserts the parity).
        """
        import mujoco

        from hymeko_rl.env.arm_world import emit_arm_mjcf
        from hymeko_rl.env.observation import ObservationSpec
        from hymeko_rl.agents.hypergraph_state import HypergraphState

        mjcf = emit_arm_mjcf(robot, control_mode=control_mode)
        model = mujoco.MjModel.from_xml_string(mjcf)
        hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        obs = ObservationSpec.from_hymeko(obs_profile)
        return cls(
            obs_dim=obs.obs_dim,
            action_dim=int(model.nu),
            action_low=float(model.actuator_ctrlrange[:, 0].min()),
            action_high=float(model.actuator_ctrlrange[:, 1].max()),
            reward=Path(task_profile).stem if task_profile else "none",
            max_steps=max_steps,
            n_vertices=hg.n_vertices,
        )
