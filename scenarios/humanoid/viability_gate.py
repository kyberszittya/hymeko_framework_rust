r"""Learn the humanoid's viability boundary from walking rollouts — a learned Lyapunov/ROA gate.

A state is **viable** if the policy does not fall within `horizon` steps from it. We roll out a walker,
label every visited reduced-state (uprightness, forward tilt, pitch-rate, forward velocity, CoM-forward
offset), and fit a logistic boundary ``P(viable | state)``. `balance_env` then **gates the forward
reward** by that probability — the policy is paid for speed only while provably recoverable, so it walks
*within* the certified region (stability-constrained RL). This extends the M0–M2 viability ladder
(`viability.py` / `neural_certificate.py`) to the walking humanoid, and *learns* the boundary from the
policy's own falls (co-adapting cert ↔ policy).

# Preconditions: the env exposes ``viability_state()`` and a ``max_steps`` horizon. # Postconditions:
#   ``learn_from_policy`` writes a `.npz` (mean/std/w/b) `balance_env` loads via ``viability_boundary``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LearnedViabilityBoundary:
    r"""A logistic viability classifier ``P(viable) = σ((x−μ)/σ · w + b)`` over the reduced state.

    The same logistic form as `viability.LearnedBoundary`, here over the 5-D humanoid walking state and
    fit by batch gradient descent (no sklearn). Standardisation (μ, σ) makes the fit well-conditioned.
    """

    def __init__(self) -> None:
        self.mean = self.std = self.w = None
        self.b = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, *, iters: int = 1200, lr: float = 0.2
            ) -> "LearnedViabilityBoundary":
        """Fit on states ``x`` (n×d) and viability labels ``y`` (n,). # Preconditions: x, y aligned; y∈{0,1}."""
        assert x.ndim == 2 and x.shape[0] == y.shape[0], "x (n×d) and y (n,) must align"
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mean) / self.std
        w, b = np.zeros(z.shape[1]), 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(z @ w + b)))
            grad = p - y
            w -= lr * (z.T @ grad) / len(y)
            b -= lr * float(grad.mean())
        self.w, self.b = w, float(b)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """P(viable) for each row of ``x``."""
        z = (np.atleast_2d(x) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(z @ self.w + self.b)))

    def save(self, path: "str | Path") -> None:
        np.savez(path, mean=self.mean, std=self.std, w=self.w, b=self.b)


def collect_labelled(env, policy_fn, seeds, horizon: int) -> "tuple[np.ndarray, np.ndarray]":
    r"""Roll out ``policy_fn`` and label each visited state: viable iff it is > ``horizon`` steps before
    the episode's fall (states within ``horizon`` of a fall are non-viable; a non-falling episode is all
    viable). Returns ``(states n×5, labels n)``."""
    xs: "list[np.ndarray]" = []
    ys: "list[int]" = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        ep: "list[np.ndarray]" = []
        fell = False
        for _ in range(env.max_steps):
            ep.append(env.viability_state())
            obs, _r, term, trunc, _i = env.step(policy_fn(obs))
            if term:
                fell = True
                break
            if trunc:
                break
        n = len(ep)
        for k, state in enumerate(ep):
            viable = 1 if (not fell or (n - k) > horizon) else 0
            xs.append(state)
            ys.append(viable)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def learn_from_policy(env, policy_fn, seeds, *, horizon: int = 200,
                      out_path: "str | Path") -> "tuple[LearnedViabilityBoundary, float]":
    r"""Collect labelled states from ``policy_fn`` on ``env``, fit the boundary, save it. Returns
    ``(boundary, train_accuracy)``."""
    x, y = collect_labelled(env, policy_fn, seeds, horizon)
    boundary = LearnedViabilityBoundary().fit(x, y)
    boundary.save(out_path)
    acc = float(((boundary.predict(x) > 0.5).astype(float) == y).mean())
    return boundary, acc
