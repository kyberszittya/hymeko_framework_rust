"""The observation as a declarative channel spec — the in-memory form of the
``meta_observation.hymeko`` vocabulary (the agent-description role of the unification).

An :class:`ObservationSpec` is an *ordered list of channel kinds*; its star-expansion is
the per-vertex observation tensor ``(N_vertices, Σ dim)``. Each channel kind maps to a
**dim** and an **extractor** (Strategy) that reads the live MuJoCo state into its columns.
:meth:`ObservationSpec.from_hymeko` reads the ordered channels straight from a
``.hymeko`` observation profile, so a new observation needs only a new ``.hymeko`` — the
env's ``node_features`` is no longer hand-coded.

Channel kinds mirror ``data/robotics/meta_observation.hymeko``. Only the kinds the reaching
task uses are implemented here; the rest are a registry entry away.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from hymeko_rl.env._profile import read_bundle

if TYPE_CHECKING:
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

Extractor = Callable[["ArmReachEnv"], np.ndarray]


# ── channel extractors (Strategy) ────────────────────────────────────────────
# Each returns a ``(N_vertices, dim)`` float32 block. Per-vertex channels scatter a
# per-joint scalar onto its child vertex; global channels broadcast across all vertices.
def _scatter_joint_scalar(env: "ArmReachEnv", source: np.ndarray) -> np.ndarray:
    out = np.zeros((env.hg.n_vertices, 1), dtype=np.float32)
    for j in range(env.model.njnt):
        v = int(env._jnt_vtx[j])
        if 0 <= v < env.hg.n_vertices:
            out[v, 0] = source[j]
    return out


def _extract_joint_position(env: "ArmReachEnv") -> np.ndarray:
    return _scatter_joint_scalar(env, env.data.qpos)


def _extract_joint_velocity(env: "ArmReachEnv") -> np.ndarray:
    return _scatter_joint_scalar(env, env.data.qvel)


def _broadcast(env: "ArmReachEnv", vec: np.ndarray) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(vec, dtype=np.float32), (env.hg.n_vertices, len(vec))).copy()


def _extract_target_position(env: "ArmReachEnv") -> np.ndarray:
    return _broadcast(env, env._target)


def _extract_ee_error(env: "ArmReachEnv") -> np.ndarray:
    return _broadcast(env, env._target - env._ee_pos())


@dataclass(frozen=True)
class _ChannelImpl:
    """A channel kind's width and its live-state extractor — the dim and the Strategy
    bundled so the two cannot drift."""
    dim: int
    extract: Extractor


# kind -> (dim, extractor). Dims match meta_observation.hymeko.
_CHANNELS: dict[str, _ChannelImpl] = {
    "joint_position": _ChannelImpl(1, _extract_joint_position),
    "joint_velocity": _ChannelImpl(1, _extract_joint_velocity),
    "target_position": _ChannelImpl(3, _extract_target_position),
    "ee_error": _ChannelImpl(3, _extract_ee_error),
}


@dataclass(frozen=True)
class ObservationSpec:
    """An ordered tuple of channel kinds → the per-vertex observation layout.

    # Preconditions Non-empty; every kind is in :data:`_CHANNELS`.
    # Postconditions ``assemble`` returns ``(N_vertices, obs_dim)`` float32; the column
    order matches ``channels`` (the same layout the policy was sized against).
    """

    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("ObservationSpec must declare at least one channel")
        unknown = [k for k in self.channels if k not in _CHANNELS]
        if unknown:
            raise ValueError(
                f"unknown observation channel(s) {unknown}; "
                f"known: {sorted(_CHANNELS)}")

    @property
    def obs_dim(self) -> int:
        """Per-vertex feature width Σ dim."""
        return sum(_CHANNELS[k].dim for k in self.channels)

    def assemble(self, env: "ArmReachEnv") -> np.ndarray:
        """Build the live ``(N_vertices, obs_dim)`` observation from the env state."""
        blocks = [_CHANNELS[k].extract(env) for k in self.channels]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    @classmethod
    def from_hymeko(cls, profile_path: str | Path) -> "ObservationSpec":
        """Build the spec from a ``.hymeko`` observation profile."""
        return cls(channels=read_channels(profile_path))


def read_channels(profile_path: str | Path) -> tuple[str, ...]:
    """Read the ordered channel *kinds* of a profile's ``observation_space``.

    Pairs each ``observation_space`` member with its declared channel kind, in arc order.
    See :func:`hymeko_rl.env._profile.read_bundle` for the (narrow, B-003-bridge) parse.

    # Postconditions Member kinds in arc order (unknown kinds are returned as-is and
    rejected by :class:`ObservationSpec`).
    # Errors ``FileNotFoundError``; ``ValueError`` (no/!1 observation_space, undeclared member).
    """
    return tuple(kind for _name, kind, _body in read_bundle(profile_path, "observation_space"))


# The reaching task's default observation: per-joint [qpos, qvel] + broadcast
# [target(3), ee_error(3)] = 8. Identical layout to the former hand-coded node_features.
REACH_OBSERVATION = ObservationSpec(
    ("joint_position", "joint_velocity", "target_position", "ee_error"))
