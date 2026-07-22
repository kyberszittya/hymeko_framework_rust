"""CANONICAL FULL-ACTION BC (2026-07-22, v3 learning §5) — a standalone policy ``u = bc(node_features)`` cloned from
the executed expert actions. NO scripted acquisition, NO scripted carry, NO residual-over-base, NO online expert
switching, NO state injection: the deployed BC drives the whole task from true neutral through ``inner.step``.

Architecture: flat MLP (48 → 256 → 256 → 4), MSE regression to the executed actuator command. Deployment rolls the BC
from a neutral reset and grades with the SAME strict K=6 delivery certificate as the expert.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FullActionBC(nn.Module):
    """``node_features`` (flat 48) → 4-DoF actuator command."""

    def __init__(self, obs_dim: int = 48, action_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act(self, node_features_flat: np.ndarray) -> np.ndarray:
        return self.net(torch.as_tensor(node_features_flat[None], dtype=torch.float32))[0].numpy()


def train_bc(obs: np.ndarray, act: np.ndarray, *, epochs: int = 200, lr: float = 1e-3, batch: int = 256,
             seed: int = 0, val: "tuple[np.ndarray, np.ndarray] | None" = None) -> tuple[FullActionBC, dict]:
    """Behaviour-clone ``obs → act`` (MSE). Returns the policy + train/val loss history. Seeded (reproducible A/B)."""
    torch.manual_seed(seed)
    bc = FullActionBC(obs.shape[1], act.shape[1])
    opt = torch.optim.Adam(bc.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    ob = torch.as_tensor(obs, dtype=torch.float32)
    ac = torch.as_tensor(act, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    hist = {"train": [], "val": []}
    n = len(obs)
    for _ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            loss = lossfn(bc(ob[b]), ac[b])
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(b)
        hist["train"].append(ep_loss / n)
        if val is not None:
            with torch.no_grad():
                hist["val"].append(float(lossfn(bc(torch.as_tensor(val[0], dtype=torch.float32)),
                                                 torch.as_tensor(val[1], dtype=torch.float32))))
    bc.eval()
    return bc, hist


def dagger_transport_labels(bc, seeds, *, grasp_hold: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """On-policy DAgger for the failing TRANSPORT phase: roll the BC (``inner.step``) to its OWN grasp state, then hand
    off to the expert handoff transport via ``env.step`` (which maintains the transport obs) and record
    ``(node_features flat 48, executed ctrl 4)`` — the expert's correct action on the states the BC actually reaches.
    The approach phase is already competent (9/9 contact), so DAgger targets transport (the observed failure)."""
    from hymeko_rl.coin_delivery.full_action_dataset import _handoff_transport
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    tfn = _greedy_action_fn(_handoff_transport())
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    obs: list[np.ndarray] = []
    act: list[np.ndarray] = []
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        bi = 0
        for _k in range(160):                                    # BC drives the approach on-policy
            m = inner._planar_metrics
            bi = bi + 1 if (m.left_contact and m.right_contact) else 0
            if bi >= grasp_hold:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            inner.step(np.asarray(bc.act(nf), np.float32))
        cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)   # transition to transport
        cf._t = 0
        cf._both_hist = []
        env._suffix_t = 0
        env._prev_dtz = env._dtz()
        env._prev_both = env._both()
        o = cf._obs(np.zeros(4, np.float32))
        for _t in range(200):                                    # expert handoff labels on the BC-reached grasp state
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
            obs.append(nf)
            act.append(np.asarray(inner.data.ctrl[:4], np.float32).copy())
    return np.asarray(obs, np.float32), np.asarray(act, np.float32)


def load_trajectory_dataset(dataset_dir: str) -> dict:
    """Load all certified full-trajectory ``traj_*.npz`` under ``dataset_dir`` into flat arrays + per-sample phase +
    trajectory id (splits are at the trajectory level; never split one trajectory's steps across train/val)."""
    import glob
    import os
    files = sorted(glob.glob(os.path.join(dataset_dir, "traj_*.npz")))
    obs, act, phase, traj = [], [], [], []
    for i, f in enumerate(files):
        d = np.load(f)
        obs.append(d["obs"])
        act.append(d["act"])
        phase.append(d["phase"])
        traj.append(np.full(len(d["act"]), i, np.int32))
    return {"obs": np.concatenate(obs), "act": np.concatenate(act), "phase": np.concatenate(phase),
            "traj": np.concatenate(traj), "n_traj": len(files), "files": files}


def phase_balanced_weights(phase: np.ndarray, n_phases: int = 7) -> np.ndarray:
    """Per-sample weight inversely proportional to phase frequency, so each RUNTIME PHASE contributes equal total mass
    (the short load-bearing contact/settle/dwell phases are not drowned by the long approach). Returns normalised."""
    counts = np.bincount(phase, minlength=n_phases).astype(np.float64)
    w = 1.0 / np.maximum(counts[phase], 1.0)
    return (w / w.sum()).astype(np.float64)


def train_bc_phase_balanced(obs: np.ndarray, act: np.ndarray, phase: np.ndarray, *, epochs: int = 300,
                            lr: float = 1e-3, batch: int = 256, seed: int = 0, steps_per_epoch: int = 200,
                            val: "tuple[np.ndarray, np.ndarray, np.ndarray] | None" = None) -> tuple[FullActionBC, dict]:
    """Phase-balanced behaviour cloning: each mini-batch is sampled by :func:`phase_balanced_weights` so the rare
    contact/settle/dwell transitions are seen as often as the abundant approach ones. Deterministic per ``seed``."""
    torch.manual_seed(seed)
    bc = FullActionBC(obs.shape[1], act.shape[1])
    opt = torch.optim.Adam(bc.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    ob = torch.as_tensor(obs, dtype=torch.float32)
    ac = torch.as_tensor(act, dtype=torch.float32)
    w = phase_balanced_weights(phase)
    rng = np.random.default_rng(seed)
    hist: dict = {"train": [], "val": [], "val_by_phase": []}
    for _ep in range(epochs):
        ep = 0.0
        for _b in range(steps_per_epoch):
            idx = rng.choice(len(obs), size=batch, p=w)
            opt.zero_grad()
            loss = lossfn(bc(ob[idx]), ac[idx])
            loss.backward()
            opt.step()
            ep += float(loss.detach())
        hist["train"].append(ep / steps_per_epoch)
        if val is not None:
            vo, va, vp = val
            with torch.no_grad():
                pred = bc(torch.as_tensor(vo, dtype=torch.float32))
                hist["val"].append(float(lossfn(pred, torch.as_tensor(va, dtype=torch.float32))))
                per = {int(p): float(((pred.numpy()[vp == p] - va[vp == p]) ** 2).mean())
                       for p in np.unique(vp)}
                hist["val_by_phase"].append(per)
    bc.eval()
    return bc, hist


def eval_bc_delivery(policy, seeds, *, horizon: int = 360) -> dict:
    """Deploy ``policy`` (``.act(flat48)->4`` or None for zero-action) from NEUTRAL and grade strict K=6 delivery with
    the certificate — no scripted base, no expert online, all motion through ``inner.step``."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    fc = grasp = deliv = 0
    per = []
    for s in seeds:
        env.set_stage(0)
        env.reset(seed=int(s))
        cert = DeliveryCertifier(initial_clearance=_clearance(inner))
        touched = False
        for _k in range(horizon):
            cert.update(_cert_step(inner, cf))
            m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            if cert.delivery_certified:
                break
            nf = np.asarray(inner.node_features(), np.float32).flatten()
            a = np.zeros(4, np.float32) if policy is None else np.asarray(policy.act(nf), np.float32)
            inner.step(a)
        d = bool(cert.delivery_certified)
        fc += int(touched)
        grasp += int(getattr(cert, "ever_grasped", False) or touched)
        deliv += int(d)
        per.append((int(s), d))
    n = max(1, len(list(seeds)))
    return {"n": n, "first_contact": fc, "grasp": grasp, "deliver": deliv,
            "deliver_rate": round(deliv / n, 4), "delivered_seeds": [s for s, d in per if d]}
