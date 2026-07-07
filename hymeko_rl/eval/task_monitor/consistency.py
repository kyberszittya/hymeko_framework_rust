"""Cross-cutting monitors — they verify the *pipeline* rather than a single trajectory.

* :class:`RewardConsistencyMonitor` — compares the RL reward (and the critic Q) against the external monitor
  verdict across a set of policies / actions, and flags the two misalignment modes the RL debugging depends on:
  reward increasing while the monitor score worsens, and Q ranking a worse-monitored action above a better one.
* :class:`TensorContractMonitor` — verifies the observation / privileged-``z`` / reward-feature / contact-quality
  tensor schema against :class:`~hymeko_rl.eval.task_monitor.contract.TensorContract`, and hashes field ordering
  so a drift between rollout, replay serialisation, replay loading, critic training and evaluation is detectable.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .submonitors import SubVerdict


def _rank_inversions(order_a: list[float], order_b: list[float]) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` where ``a`` ranks i above j but ``b`` ranks j above i (discordant pairs)."""
    inv = []
    n = len(order_a)
    for i in range(n):
        for j in range(i + 1, n):
            if (order_a[i] - order_a[j]) * (order_b[i] - order_b[j]) < 0:
                inv.append((i, j))
    return inv


class RewardConsistencyMonitor:
    """Detect reward-vs-monitor and critic-vs-monitor misalignment (external ground-truth = the monitor score)."""

    name = "reward_consistency"

    def check_reward_alignment(self, rows: list[dict[str, Any]]) -> SubVerdict:
        """``rows``: ``[{policy, total_reward, monitor_score}]``. Flag any pair where reward is higher but the
        monitor score is lower (reward rewards a worse-monitored policy)."""
        if len(rows) < 2:
            return SubVerdict(self.name + ":reward", True, 1.0, [], [], [])
        reward = [float(r["total_reward"]) for r in rows]
        mon = [float(r["monitor_score"]) for r in rows]
        name = [str(r["policy"]) for r in rows]
        inv = _rank_inversions(reward, mon)
        aligned = not inv
        # Spearman-style concordance as the robustness score
        conc = 1.0 - 2.0 * len(inv) / max(1, len(rows) * (len(rows) - 1) // 2)
        viol = []
        for i, j in inv:
            hi, lo = (i, j) if reward[i] > reward[j] else (j, i)   # reward ranks hi above lo
            viol.append(f"reward prefers {name[hi]} but monitor prefers {name[lo]}")
        return SubVerdict(self.name + ":reward", aligned, float(conc), viol, [i for i, _ in inv], [])

    def check_critic_alignment(self, q_by_action: dict[str, float],
                               score_by_action: dict[str, float]) -> SubVerdict:
        """``q_by_action`` / ``score_by_action`` keyed by the same action names. Flag any pair where Q ranks an
        action above another whose external monitor/outcome score is better (critic ranks bad above good)."""
        keys = [k for k in q_by_action if k in score_by_action]
        if len(keys) < 2:
            return SubVerdict(self.name + ":critic", True, 1.0, [], [], [])
        q = [float(q_by_action[k]) for k in keys]
        s = [float(score_by_action[k]) for k in keys]
        inv = _rank_inversions(q, s)
        aligned = not inv
        conc = 1.0 - 2.0 * len(inv) / max(1, len(keys) * (len(keys) - 1) // 2)
        viol = []
        for i, j in inv:
            hi, lo = (i, j) if q[i] > q[j] else (j, i)   # Q ranks hi above lo; monitor disagrees
            viol.append(f"Q ranks {keys[hi]} above {keys[lo]} but monitor prefers {keys[lo]}")
        return SubVerdict(self.name + ":critic", aligned, float(conc), viol, [], [])


class TensorContractMonitor:
    """Verify pipeline tensor schema + field-order integrity across the RL stages."""

    name = "tensor_contract"

    def __init__(self, contract: Any) -> None:
        self.contract = contract

    @staticmethod
    def field_hash(names: tuple[str, ...]) -> str:
        """Order-sensitive hash of a tensor's field layout (the same fields in a different order → different hash)."""
        return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]

    @staticmethod
    def tensor_signature(arr: np.ndarray) -> str:
        """Shape + dtype signature (a dimension drift between stages → different signature)."""
        a = np.asarray(arr)
        return f"{a.shape[-1] if a.ndim else 0}:{a.dtype}"

    def check_env(self, env: Any) -> SubVerdict:
        """Verify the live env's obs / privileged / node-feature dims match the declared schema."""
        c = self.contract
        obs_dim = int(env.observation_space.shape[0]) if getattr(env, "observation_space", None) is not None \
            else int(getattr(env, "_obs_dim", 0))
        priv = int(env.privileged_state().shape[0])
        feat = int(getattr(env, "_FEAT", 0))
        viol = []
        if obs_dim != c.obs_dim:
            viol.append(f"obs_dim {obs_dim}!={c.obs_dim}")
        if priv != c.privileged_dim:
            viol.append(f"privileged_dim {priv}!={c.privileged_dim}")
        if feat != c.node_feat_dim:
            viol.append(f"node_feat_dim {feat}!={c.node_feat_dim}")
        if len(c.privileged_fields) != c.privileged_dim:
            viol.append(f"privileged_fields len {len(c.privileged_fields)}!={c.privileged_dim}")
        return SubVerdict(self.name + ":env", not viol, 1.0 if not viol else 0.0, viol, [], [])

    def verify_stages(self, stage_fields: dict[str, tuple[str, ...]]) -> SubVerdict:
        """``stage_fields``: field-name tuples recorded at rollout / replay-serialize / replay-load / critic /
        eval. Pass iff every stage hashes to the same field ordering."""
        hashes = {stage: self.field_hash(fields) for stage, fields in stage_fields.items()}
        uniq = set(hashes.values())
        aligned = len(uniq) <= 1
        viol = [] if aligned else [f"field-order drift across stages: {hashes}"]
        return SubVerdict(self.name + ":stages", aligned, 1.0 if aligned else 0.0, viol, [], [])
