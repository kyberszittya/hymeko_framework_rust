"""K2 — the KINETIC feedback clone: a small recurrent actor imitating the K1-A first-action labels, deployed teacher-free.

K2-A trains a small GRU + MLP head on the 32 K1-A feedback labels (41-D causal observation → 4-D bounded, slew-normalised Δτ),
with the input normalisation FROZEN from the bank and delivering / near-slip states slightly up-weighted. The terminal-failure
states never enter the action loss. K2-B deploys the clone in the FROZEN chain — frozen APPROACH → learned KINETIC clone →
frozen G0/release → frozen coast → frozen K6 — with NO teacher/CEM in the loop: `KineticCloneController` owns only the KINETIC
transport Δτ; the frozen `_kinetic_targets` release / contact-risk / horizon guards still fire, and every torque/slew/joint
limit is the frozen scaffold's.

Mandatory controls (tested): the recurrent forward is batch == streaming (a sequence processed at once equals step-by-step with
the carried hidden state); the hidden-state reset is deterministic; the emitted Δτ is bounded by ± slew; no teacher is imported
into the deploy path; the K0 entry and frozen downstream are unchanged (at zero KINETIC steps the controller reproduces the
frozen APPROACH bit-for-bit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC, ApproachParams, HybridApproachController
from hymeko_rl.coin_delivery.theta_option.kinetic_contract import FEATURE_NAMES, kinetic_observe
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams

OBS_DIM = len(FEATURE_NAMES)                       # 41
ACT_DIM = 4                                        # per-joint slew-normalised Δτ (the KINETIC action basis)


@dataclass
class NormStats:
    """Input normalisation FROZEN from the K1-A bank (never recomputed at deploy). # Invariants: mean/std are length-OBS_DIM."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mean) / self.std

    def as_dict(self) -> dict:
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std]}

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "NormStats":
        obs = np.asarray(obs, np.float64)
        return cls(mean=obs.mean(0), std=obs.std(0) + 1e-6)


class KineticClone(nn.Module):
    """A small recurrent clone: GRU encoder over the causal observation stream + a tanh-bounded MLP head. # Preconditions:
    input (B, T, OBS_DIM). # Postconditions: output (B, T, ACT_DIM) ∈ (−1, 1); `forward` carries an optional hidden state so a
    streamed step equals the batched sequence."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, act_dim))

    def forward(self, seq: torch.Tensor, h0: "torch.Tensor | None" = None) -> "tuple[torch.Tensor, torch.Tensor]":
        out, h_n = self.gru(seq, h0)
        return torch.tanh(self.head(out)), h_n


class CloneActor:
    """Streaming deploy wrapper: frozen normalisation + a carried GRU hidden state. `act(obs)` returns the 4-D bounded Δτ (slew
    units) for one step; `reset()` zeroes the hidden state deterministically. # Invariants: eval mode + no_grad ⇒ deterministic;
    no teacher/future/K6 is read (only the 41-D observation)."""

    def __init__(self, model: KineticClone, norm: NormStats) -> None:
        self.model = model.eval()
        self.norm = norm
        self.h: "torch.Tensor | None" = None

    def reset(self) -> None:
        self.h = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = self.norm.apply(obs).astype(np.float32)
        with torch.no_grad():
            a, self.h = self.model(torch.from_numpy(x).view(1, 1, -1), self.h)
        return a.view(-1).numpy().astype(np.float64)


@dataclass
class CloneTrainConfig:
    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: int = 64
    w_deliver: float = 1.5                          # slight up-weight for delivering states
    w_near_slip: float = 1.3                        # slight up-weight for near-slip states
    near_slip_thresh: float = 0.25


def _sample_weights(bank: list[dict], cfg: CloneTrainConfig) -> np.ndarray:
    """Per-label weight: delivering and near-slip states are slightly up-weighted (mean-normalised). Terminal-failure states
    are not in `bank` at all, so their (near-zero) actions never enter the loss."""
    w = np.ones(len(bank), np.float64)
    for i, r in enumerate(bank):
        if r.get("delivers_k6"):
            w[i] *= cfg.w_deliver
        if float(r["descriptor"]["slip"]) > cfg.near_slip_thresh:
            w[i] *= cfg.w_near_slip
    return w / w.mean()


def train_clone(bank: list[dict], cfg: CloneTrainConfig = CloneTrainConfig(),
                *, seed: int = 0) -> "tuple[KineticClone, NormStats, list[float]]":
    """Behaviour-clone the K1-A feedback labels (single-step, length-1 sequences) under a weighted Huber loss. Deterministic
    given `seed`. # Preconditions: `bank` the feedback_labels records (obs + first_action_norm). # Postconditions: returns
    (trained model, frozen NormStats, per-epoch loss); at deploy the recurrence carries across the KINETIC transport."""
    torch.manual_seed(seed)
    obs = np.array([r["obs"] for r in bank], np.float64)
    act = np.array([r["first_action_norm"] for r in bank], np.float64)
    norm = NormStats.from_obs(obs)
    x = torch.from_numpy(norm.apply(obs).astype(np.float32)).unsqueeze(1)      # (N, 1, OBS_DIM)
    y = torch.from_numpy(act.astype(np.float32)).unsqueeze(1)                  # (N, 1, ACT_DIM)
    w = torch.from_numpy(_sample_weights(bank, cfg).astype(np.float32)).view(-1, 1, 1)
    model = KineticClone(hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    huber = nn.HuberLoss(reduction="none")
    history: list[float] = []
    model.train()
    for _epoch in range(cfg.epochs):
        opt.zero_grad()
        pred, _h = model(x)
        loss = (huber(pred, y) * w).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return model, norm, history


class KineticCloneController(HybridApproachController):
    """Frozen APPROACH + frozen KINETIC transitions, with the learned clone owning ONLY the KINETIC transport Δτ. In the KINETIC
    transport state the clone's bounded Δτ (slew units) replaces the hand-tuned servo; every phase transition (release /
    contact-risk brake / horizon) remains the frozen `_kinetic_targets` logic, and the release/coast/K6 downstream is unchanged.
    # Postconditions: |Δτ| ≤ slew; no teacher in the loop; at zero KINETIC steps identical to the frozen APPROACH scaffold."""

    def __init__(self, snap: Any, actor: CloneActor, params: TipTransportParams = TipTransportParams(),
                 approach: "ApproachParams | None" = None, cfg: Any = DELIVERY_CFG) -> None:
        self.actor = actor
        ap = approach if approach is not None else ApproachParams(kinetic_transport=True)
        super().__init__(snap, params, ap, cfg, enabled=True)

    def reset(self) -> None:
        super().reset()
        self.actor.reset()
        self._obs_hist: list = []
        self.clone_trace: list[dict[str, Any]] = []

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        bt = self._base_targets(rl, t)                                        # runs the frozen phase machine (may transition)
        if self.phase == KINETIC and not bt["in_release"]:                    # the KINETIC transport state ⇒ the clone acts
            obs, hframe = kinetic_observe(rl, self._obs_hist)
            self._obs_hist.append(hframe)
            a = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
            dtau = np.clip(a * self.slew, -self.slew, self.slew)
            self._log_clone(rl, t, "KINETIC_CLONE", bt, a)
            return dtau
        dtau = self._servo(rl, bt, bt["base_qref"], bt["base_sqz"], self.p.k_v)  # frozen APPROACH / brake / release
        self._log_clone(rl, t, "FROZEN", bt, None)
        return dtau

    def _log_clone(self, rl: Any, t: int, kind: str, bt: dict, a: "np.ndarray | None") -> None:
        con = primary_fingertip_contacts(rl)
        self.clone_trace.append({
            "t": int(t), "kind": kind, "phase": int(self.phase), "dtz_mm": round(bt["dtz"] * 1000, 2),
            "v_par": round(bt["v_par"], 5), "fn_l": round(float(con["left"]["fn"]) if con["left"] else 0.0, 4),
            "fn_r": round(float(con["right"]["fn"]) if con["right"] else 0.0, 4),
            "action": [round(float(x), 5) for x in a] if a is not None else None})
