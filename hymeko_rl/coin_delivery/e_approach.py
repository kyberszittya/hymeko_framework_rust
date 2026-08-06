"""Canonical E-approach policy loader — production, experiment-free.

Extracted from ``experiments.exp_v3_handoff_gate._load_e`` (+ ``exp_option_retest._fresh_actor``) so the recovered
E_valselect approach loads without dragging in the galambos/coin-toss experiment web. The actor is built by the
production :func:`hymeko_rl.agents.multichannel_ctde.build_collaborative_offpolicy` over the canonical Coin env
(:func:`hymeko_rl.coin_delivery.env_factory.make_coin_env`); the state_dict is loaded verbatim. Behaviour is identical
to the old loader (proven bit-for-bit against frozen fixtures). No experiment module is imported here.

The E-approach inference contract is ``node_features → action_mean`` (a 4-DoF arm action), wrapped in
:class:`EApproachPolicy`, which also exposes ``.action_mean(tensor)`` and ``.state_dict()`` as a drop-in for the old
raw actor.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

# the frozen deploy checkpoint (read-only); md5 b822a660 == this sha256 prefix
E_VALSELECT_CKPT = "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt"
E_VALSELECT_SHA256 = "7dbbf1a7782f"          # sha256[:12] of the E_valselect_v2.pt bytes (frozen 2026-07-21)


class EApproachPolicy:
    """The learned neutral→pre-contact approach policy. ``action(env)`` runs the ``node_features → action_mean``
    inference; ``action_mean``/``state_dict`` delegate to the wrapped actor for drop-in compatibility with the old
    ``_load_e()`` return value. # Invariants the wrapped actor is in eval mode and never mutated by inference."""

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    def action(self, env: Any) -> np.ndarray:
        """node_features(env) → action_mean, as a float32 numpy action (the E-approach inference contract)."""
        nf = np.asarray(env.node_features(), dtype=np.float32)
        with torch.no_grad():
            return self._actor.action_mean(torch.as_tensor(nf[None], dtype=torch.float32))[0].numpy()

    def action_mean(self, x: Any) -> Any:                       # drop-in for the old raw-actor callers
        return self._actor.action_mean(x)

    def state_dict(self) -> Any:
        return self._actor.state_dict()

    @property
    def actor(self) -> Any:
        return self._actor


def _sha256_prefix(path: str, n: int) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:n]


def load_e_approach_policy(checkpoint_path: str = E_VALSELECT_CKPT, *, device: str = "cpu",
                           expected_checkpoint_hash: str | None = E_VALSELECT_SHA256,
                           embodiment: str = "POINT") -> EApproachPolicy:
    """Load the E-approach policy from ``checkpoint_path`` over the canonical Coin env, fail-loud on any mismatch.

    # Preconditions the checkpoint exists and its sha256 prefix equals ``expected_checkpoint_hash`` (when given);
      ``device`` is a valid torch device string; ``embodiment`` is a valid Coin embodiment.
    # Postconditions returns an eval-mode :class:`EApproachPolicy` whose params match the checkpoint exactly.
    # Errors ``FileNotFoundError`` / ``ValueError`` (hash, device, state_dict key/shape mismatch) — never a silent
      substitution.
    """
    import os

    from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
    from hymeko_rl.coin_delivery.env_factory import make_coin_env
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"E-approach checkpoint missing: {checkpoint_path} (no fabricated policy)")
    if expected_checkpoint_hash is not None:
        got = _sha256_prefix(checkpoint_path, len(expected_checkpoint_hash))
        if got != expected_checkpoint_hash:
            raise ValueError(f"E-approach checkpoint hash mismatch: {checkpoint_path} is {got}, "
                             f"expected {expected_checkpoint_hash} (refusing to load a different checkpoint)")
    try:
        torch.device(device)
    except Exception as exc:
        raise ValueError(f"unsupported device {device!r}: {exc}") from exc
    env = make_coin_env(embodiment=embodiment)                  # for actor dims (node_features / action layout)
    actor = build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]
    state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = actor.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"E-approach state_dict mismatch: missing={list(missing)} unexpected={list(unexpected)} "
                         f"(checkpoint/architecture incompatible)")
    actor.eval()
    return EApproachPolicy(actor)
