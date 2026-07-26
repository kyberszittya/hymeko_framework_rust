"""STAGE 3 — the BC / proposal actor: structured state → proposal centre θ_0.

Three variants share one dataset, optimiser budget, output bounds, and evaluation seeds; they differ ONLY in the flat
observation the RL-ready `DetActor` consumes (so the winner plugs straight into the semi-MDP engine and the fixed search):

    B0  features(42)
    B1  features(42) ⊕ causal-history flattened (8×6)
    B2  features(42) ⊕ LSTM temporal embedding of the causal history (option_rl.LSTMTemporalEncoder)

The target is the normalised proposal centre θ_0 (Tanh-bounded, always a legal option after `ThetaBox.denorm`). Offline
regression alone is NOT the update-0 gate (Stage 4) — a low θ-error can still fail delivery, and the fixed search is what
must reproduce K6. `Featurizer` turns (features, history) into the variant's flat obs; the frozen LSTM (B2) is part of it,
so a trained B2 actor is a plain flat-obs `DetActor` at deploy — identical RL wiring to B0/B1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.theta_option.dataset import HISTORY_K, ThetaDataset
from hymeko_rl.coin_delivery.theta_option.semantics import DIM, THETA_NAMES, ThetaBox
from hymeko_rl.option_rl.agents import DetActor
from hymeko_rl.option_rl.temporal import LSTMTemporalEncoder

FEATURE_DIM = 42
HIST_DIM = 6
LSTM_EMB = 8
VARIANTS = ("B0", "B1", "B2")


def obs_dim(variant: str) -> int:
    return {"B0": FEATURE_DIM, "B1": FEATURE_DIM + HISTORY_K * HIST_DIM, "B2": FEATURE_DIM + LSTM_EMB}[variant]


class Featurizer:
    """Maps (features(42), history(K,6)) → the variant's flat obs. B2 owns a (trained, then frozen) LSTM encoder; B0/B1
    are pure concatenations. Deterministic. # Postconditions: returns a 1-D float32 of length ``obs_dim(variant)``."""

    def __init__(self, variant: str, lstm: "LSTMTemporalEncoder | None" = None):
        self.variant = variant
        self.lstm = lstm

    def _embed(self, history: np.ndarray) -> np.ndarray:
        assert self.lstm is not None, "Featurizer._embed is only reachable for B2 (a frozen LSTM must be set)"
        with torch.no_grad():
            X = torch.as_tensor(np.asarray(history, np.float32))[None]     # (1,K,6)
            emb, _ = self.lstm(X)                                          # (1,K,emb)
            return emb[0, -1].numpy()                                      # last-step embedding

    def obs(self, features: np.ndarray, history: np.ndarray) -> np.ndarray:
        f = np.asarray(features, np.float32).reshape(-1)
        if self.variant == "B0":
            return f
        if self.variant == "B1":
            return np.concatenate([f, np.asarray(history, np.float32).reshape(-1)]).astype(np.float32)
        return np.concatenate([f, self._embed(history)]).astype(np.float32)


class LSTMProposalNet(nn.Module):
    """B2 training net: encode the causal history with the shared LSTM, concat its last embedding with the features, map
    to θ_norm (Tanh). After training, the LSTM is frozen and reused by the `Featurizer`; the head becomes a `DetActor`
    over [features ⊕ emb] so deploy/RL see a flat obs identical to B0/B1."""

    def __init__(self, h: int = 128):
        super().__init__()
        self.lstm = LSTMTemporalEncoder(in_dim=HIST_DIM, hidden=32, out_dim=LSTM_EMB)
        self.head = nn.Sequential(nn.Linear(FEATURE_DIM + LSTM_EMB, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                  nn.Linear(h, DIM), nn.Tanh())

    def forward(self, feats: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        emb, _ = self.lstm(hist)                                          # (B,K,emb)
        return self.head(torch.cat([feats, emb[:, -1]], -1))              # (B,6)


@dataclass
class BCProposal:
    """A trained proposal conforming to `option_rl.ProposalPolicy` (via `center`) AND the coin `.theta(obs)` idiom. Holds
    the flat-obs `DetActor` and the θ box; `center(obs)`/`theta(obs)` return a LEGAL θ_0. The obs must already be the
    variant's featurised vector (build it with the paired `Featurizer`)."""

    variant: str
    actor: DetActor
    box: ThetaBox

    def _theta(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.actor.mean_action(torch.as_tensor(np.asarray(obs, np.float32))[None])[0].numpy()
        return self.box.denorm(z)

    def center(self, obs: np.ndarray) -> np.ndarray:
        return self._theta(obs)

    def theta(self, obs: np.ndarray) -> np.ndarray:
        return self._theta(obs)


def _train_detactor(X: np.ndarray, Y: np.ndarray, *, epochs: int, lr: float, seed: int) -> tuple[DetActor, float]:
    torch.manual_seed(seed)
    actor = DetActor(X.shape[1], DIM)
    opt = torch.optim.Adam(actor.parameters(), lr)
    x, y = torch.as_tensor(X), torch.as_tensor(Y)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((actor.mean_action(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return actor, float(loss.item())


def fit_bc(ds: ThetaDataset, variant: str, *, epochs: int = 1200, lr: float = 1e-3, seed: int = 0) -> tuple[BCProposal, Featurizer, dict[str, Any]]:
    """Fit the BC proposal for ``variant`` on the TRAIN split (development delivering θ). B0/B1 train a flat-obs
    `DetActor`; B2 trains the LSTM+head jointly then freezes the LSTM and distils the head into a flat-obs `DetActor` over
    [features ⊕ frozen-emb]. Deterministic given ``seed``. Returns (proposal, featurizer, train_loss)."""
    box = ThetaBox()
    train = ds.subset("train")
    feats = np.asarray([r.features for r in train], np.float32)
    hists = np.asarray([r.history for r in train], np.float32)
    Y = np.asarray([r.theta_norm for r in train], np.float32)
    if variant in ("B0", "B1"):
        fz = Featurizer(variant)
        X = np.asarray([fz.obs(f, h) for f, h in zip(feats, hists)], np.float32)
        actor, tl = _train_detactor(X, Y, epochs=epochs, lr=lr, seed=seed)
        return BCProposal(variant, actor, box), fz, {"train_mse": round(tl, 6)}
    # B2: joint LSTM+head, then freeze LSTM and distil into a flat-obs DetActor over [features ⊕ frozen-emb]
    torch.manual_seed(seed)
    net = LSTMProposalNet()
    opt = torch.optim.Adam(net.parameters(), lr)
    tf, th, ty = torch.as_tensor(feats), torch.as_tensor(hists), torch.as_tensor(Y)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(tf, th) - ty) ** 2).mean()
        loss.backward()
        opt.step()
    net.lstm.eval()
    for p in net.lstm.parameters():
        p.requires_grad_(False)
    fz = Featurizer("B2", lstm=net.lstm)
    X = np.asarray([fz.obs(f, h) for f, h in zip(feats, hists)], np.float32)
    actor, tl = _train_detactor(X, Y, epochs=epochs, lr=lr, seed=seed)   # distil the frozen-emb head into a DetActor
    return BCProposal("B2", actor, box), fz, {"train_mse": round(tl, 6), "joint_lstm_mse": round(float(loss.item()), 6)}


def save_proposal(prop: BCProposal, fz: Featurizer, path: str) -> None:
    """Persist a trained proposal (variant + `DetActor` weights + obs_dim + the frozen B2 LSTM if present) for update-0
    and the RL init handoff."""
    blob: dict[str, Any] = {"variant": prop.variant, "obs_dim": obs_dim(prop.variant),
                            "actor": prop.actor.state_dict()}
    if prop.variant == "B2" and fz.lstm is not None:
        blob["lstm"] = fz.lstm.state_dict()
    torch.save(blob, path)


def load_proposal(path: str) -> tuple[BCProposal, Featurizer]:
    """Load a proposal saved by `save_proposal`. Reconstructs the flat-obs `DetActor` and (B2) the frozen LSTM featurizer."""
    blob = torch.load(path, weights_only=False)
    variant = str(blob["variant"])
    actor = DetActor(int(blob["obs_dim"]), DIM)
    actor.load_state_dict(blob["actor"])
    actor.eval()
    lstm = None
    if variant == "B2":
        lstm = LSTMTemporalEncoder(in_dim=HIST_DIM, hidden=32, out_dim=LSTM_EMB)
        lstm.load_state_dict(blob["lstm"])
        lstm.eval()
    return BCProposal(variant, actor, ThetaBox()), Featurizer(variant, lstm=lstm)


# ── offline metrics (NOT the update-0 gate) ──
def offline_metrics(prop: BCProposal, fz: Featurizer, ds: ThetaDataset, split: str) -> dict[str, Any]:
    """Per-component θ error (legal units), normalised error, bounded-action validity, and phase-sensitive error on a
    split. `ramp_steps`/`release_step` are the phase params (in control steps); the other four are continuous torque
    params. # Postconditions: bounded_validity == 1.0 (Tanh output is always in-box)."""
    rows = ds.subset(split)
    if not rows:
        return {"n": 0}
    box = prop.box
    preds = np.asarray([prop.theta(fz.obs(r.features, r.history)) for r in rows], np.float64)
    tgts = np.asarray([r.theta for r in rows], np.float64)
    abs_err = np.abs(preds - tgts)
    norm_err = np.abs(box.norm(preds) - box.norm(tgts))
    per_comp = {THETA_NAMES[i]: round(float(abs_err[:, i].mean()), 5) for i in range(DIM)}
    phase_idx, cont_idx = [3, 4], [0, 1, 2, 5]
    return {"n": len(rows), "mean_abs_err_per_component": per_comp,
            "mean_norm_err": round(float(norm_err.mean()), 5),
            "phase_param_step_err": round(float(abs_err[:, phase_idx].mean()), 4),
            "continuous_param_err": round(float(abs_err[:, cont_idx].mean()), 5),
            "bounded_validity": round(float(np.mean(np.all((box.norm(preds) >= -1.0001) & (box.norm(preds) <= 1.0001), axis=1))), 4),
            "per_tag_norm_err": {tag: round(float(np.mean([ne.mean() for ne, r in zip(norm_err, rows) if r.tag == tag])), 5)
                                 for tag in sorted({r.tag for r in rows})}}
