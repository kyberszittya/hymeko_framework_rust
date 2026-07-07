"""Monitor contracts, drawn from the HyMeKo task contract carried by the env (single-source-of-truth).

Two contracts:

* :class:`MonitorContract` — the geometric / temporal thresholds the trajectory submonitors verify against
  (zone geometry, dwell success rule, fingertip-near threshold, body-progress tolerance). Read from
  :class:`~hymeko_rl.env.env_spec.EnvSpec` / the contact-legality roles on the env.
* :class:`TensorContract` — the field-ordering + dimension schema of the tensors that cross the RL pipeline
  (observation, privileged CTDE ``z``, reward feature, contact-quality). Read from the env so the
  :class:`~hymeko_rl.eval.task_monitor.consistency.TensorContractMonitor` can detect a schema drift between
  rollout, replay serialisation, replay loading, critic training, and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MonitorContract:
    """Declarative geometric / temporal thresholds for the coin-delivery task (single-source from the env)."""

    zone_half: float                 # target radius (EnvSpec.zone_half)
    success_steps: int               # k consecutive in-zone steps = stable delivery (EnvSpec.success)
    near_coin: float = 0.06          # fingertip "near the coin" distance threshold
    body_eps: float = 0.005          # max body-only toward-zone progress for a fingertip-dominant pass
    progress_eps: float = 0.002      # per-step toward-zone displacement counted as real progress (noise floor)
    ft_dominant_frac: float = 0.15   # body_prog must be <= max(body_eps, frac * total_prog)
    min_two_finger_steps: int = 1    # engagement duration floor

    @classmethod
    def from_env(cls, env: Any) -> "MonitorContract":
        """Read the thresholds from the env's HyMeKo contract (zone geometry + dwell success rule)."""
        return cls(zone_half=float(env._zone_half), success_steps=int(getattr(env, "success_steps", 1)))


@dataclass(frozen=True)
class TensorContract:
    """Field-ordering + dimension schema of the pipeline tensors, read from the env (single-source)."""

    obs_dim: int
    privileged_dim: int
    node_feat_dim: int
    privileged_fields: tuple[str, ...] = ("left_contact", "right_contact", "phase0", "phase1", "phase2")

    @classmethod
    def from_env(cls, env: Any) -> "TensorContract":
        obs_dim = int(env.observation_space.shape[0]) if getattr(env, "observation_space", None) is not None \
            else int(getattr(env, "_obs_dim", 0))
        priv = env.privileged_state()
        return cls(obs_dim=obs_dim, privileged_dim=int(priv.shape[0]),
                   node_feat_dim=int(getattr(env, "_FEAT", 0)))
