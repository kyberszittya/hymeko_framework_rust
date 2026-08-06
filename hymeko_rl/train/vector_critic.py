"""Vector critics + constraint-projected gradient — the multidimensional replacement for a single scalar critic.

Trains one critic per :data:`~hymeko_rl.train.search_objective.COMPONENTS` signal (approach / contact / progress /
delivery / antiexploit / body_progress), each predicting that component's short-return under the frozen DAgger
policy. Then a PCGrad-style projection builds an update direction that raises the objectives (delivery + progress)
while respecting the constraints (contact must not drop, anti-exploit must not drop, body-progress must not rise).
No actor is trained here — this is the critic + gradient-geometry diagnostic that gates whether a vector-projected
actor smoke is worth running.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.train.ddpg import _polyak
from hymeko_rl.train.search_objective import COMPONENTS


def build_vector_critics(env: Any) -> dict[str, Any]:
    """One centralized QCritic per component signal (same architecture as the CTDE critic; reuses the builder)."""
    return {name: build_collaborative_offpolicy(env, kind="mlp", hidden=64, n_critics=1, privileged=True)[1][0]
            for name in COMPONENTS}


@dataclass
class VectorCriticConfig:
    steps: int = 3000
    lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1000


def train_vector_critics(critics: dict[str, Any], frozen_actor: Any, data: dict[str, np.ndarray],
                         cfg: VectorCriticConfig) -> dict[str, Any]:
    """Fit each component critic to ``r_comp + gamma*(1-d)*Q_comp_target(s', a_pi(s'), z')`` on the DAgger dataset
    (``data`` holds obs/action/next_obs/done/z/z_next + one reward column per component). Actor stays frozen."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    frozen_actor.to(dev).eval()
    for p in frozen_actor.parameters():
        p.requires_grad_(False)
    tgt = {n: copy.deepcopy(c).to(dev) for n, c in critics.items()}
    opt = {n: torch.optim.Adam(c.to(dev).parameters(), lr=cfg.lr) for n, c in critics.items()}
    rng = np.random.default_rng(cfg.seed)
    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    n = int(data["obs"].shape[0])
    obs, act, nobs, done = t(data["obs"]), t(data["action"]), t(data["next_obs"]), t(data["done"])
    z, z2 = t(data["z"]), t(data["z_next"])
    comp = {c: t(data[f"r_{c}"]) for c in COMPONENTS}
    for step in range(1, cfg.steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, size=cfg.batch_size), device=dev)
        s, a, s2, d, zz, zz2 = obs[idx], act[idx], nobs[idx], done[idx], z[idx], z2[idx]
        with torch.no_grad():
            a2 = frozen_actor(s2)
        for name, c in critics.items():
            with torch.no_grad():
                y = comp[name][idx] + cfg.gamma * (1 - d) * tgt[name](s2, a2, zz2)
            loss = F.mse_loss(c(s, a, zz), y)
            opt[name].zero_grad()
            loss.backward()
            opt[name].step()
            _polyak(tgt[name], c, cfg.tau)
        if cfg.log_every and step % cfg.log_every == 0:
            with torch.no_grad():
                means = {k: float(critics[k](s, a, zz).mean()) for k in COMPONENTS}
            print(f"  [vector-critic] step {step}/{cfg.steps} "
                  + " ".join(f"{k}:{means[k]:+.2f}" for k in COMPONENTS), flush=True)
    return critics


def train_vector_critics_mc(critics: dict[str, Any], data: dict[str, np.ndarray],
                            cfg: VectorCriticConfig) -> dict[str, list[float]]:
    """Fit each component critic by **supervised regression to the measured Monte-Carlo return** ``mc_r_c`` — the
    fair-retest target (exact, action-conditioned, frozen continuation) that replaces the bootstrap of
    :func:`train_vector_critics`. No target network, no actor: the OOD random-action rows already carry their
    (low) measured returns, so plain MSE fits the off-manifold shape.

    # Preconditions: ``data`` has ``obs``/``action``/``z`` and ``mc_r_{component}`` for every ``COMPONENTS`` name.
    # Postconditions: returns per-component loss history; ``critics`` are updated in place (trained)."""
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    t = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    obs, act, z = t(data["obs"]), t(data["action"]), t(data["z"])
    tgt = {c: t(data[f"mc_r_{c}"]) for c in COMPONENTS}
    opt = {n: torch.optim.Adam(c.to(dev).parameters(), lr=cfg.lr) for n, c in critics.items()}
    rng = np.random.default_rng(cfg.seed)
    n = int(obs.shape[0])
    hist: dict[str, list[float]] = {c: [] for c in COMPONENTS}
    for step in range(1, cfg.steps + 1):
        idx = torch.as_tensor(rng.integers(0, n, size=min(cfg.batch_size, n)), device=dev)
        s, a, zz = obs[idx], act[idx], z[idx]
        losses = {}
        for name, c in critics.items():
            loss = F.mse_loss(c(s, a, zz), tgt[name][idx])
            opt[name].zero_grad()
            loss.backward()
            opt[name].step()
            losses[name] = float(loss.detach())
        if not all(np.isfinite(v) for v in losses.values()):
            raise FloatingPointError(f"non-finite critic loss at step {step}: {losses}")
        for name in COMPONENTS:
            hist[name].append(losses[name])
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"  [vector-critic/mc] step {step}/{cfg.steps} "
                  + " ".join(f"{k}:{losses[k]:.3f}" for k in COMPONENTS), flush=True)
    return hist


def action_gradient(critic: Any, obs: np.ndarray, action: np.ndarray, z: np.ndarray) -> np.ndarray:
    """∇_a Q(s, a, z) at a single (s, a, z), as a numpy vector (grad of the scalar critic w.r.t. the action)."""
    a = torch.tensor(action[None], dtype=torch.float32, requires_grad=True)
    q = critic(torch.as_tensor(obs[None]), a, torch.as_tensor(z[None])).sum()
    q.backward()
    return a.grad[0].detach().numpy()


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else 0.0


def _project_out(g: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Remove the component of ``g`` along ``ref`` (PCGrad conflict removal)."""
    nn = float(np.dot(ref, ref))
    return g - (float(np.dot(g, ref)) / nn) * ref if nn > 1e-12 else g


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def projected_gradient(grads: dict[str, np.ndarray], *, normalize: bool = False,
                       extra_up: tuple[str, ...] = (), extra_down: tuple[str, ...] = ()) -> tuple[np.ndarray, dict]:
    """Constraint-projected update: start from ∇delivery+∇progress, project away directions that would REDUCE
    contact / anti-exploit (``up`` constraints), or INCREASE body-progress (``down`` constraint). Returns the
    (unit) direction + which projections fired.

    ``normalize=True`` unit-normalizes each component gradient before combining/projecting, so a large-magnitude
    critic (e.g. the discounted-contact-indicator return, which is O(10) vs O(1) delivery) cannot dominate the
    geometry by scale alone — the fair-test variant. ``extra_up`` / ``extra_down`` add further named constraints.
    Defaults reproduce the original behavior exactly."""
    gof = (lambda k: _unit(grads[k])) if normalize else (lambda k: grads[k])
    g = gof("delivery") + gof("progress")
    fired: dict[str, bool] = {}
    for k in ("contact", "antiexploit", *extra_up):        # must not decrease → remove conflicting part (dot<0)
        if float(np.dot(g, gof(k))) < 0.0:
            g = _project_out(g, gof(k))
            fired[k] = True
    for k in ("body_progress", *extra_down):               # must not increase → remove aligned part (dot>0)
        if float(np.dot(g, gof(k))) > 0.0:
            g = _project_out(g, gof(k))
            fired[k] = True
    norm = float(np.linalg.norm(g))
    unit = g / norm if norm > 1e-9 else g
    return unit, {"projections_fired": fired, "norm_before_unit": round(norm, 5), "normalized": normalize}
