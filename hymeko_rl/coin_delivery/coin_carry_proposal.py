"""Multimodality-preserving OPTION proposal + the fixed local-search wrapper (shared by Stage 4c refinement and Stage 5 RL).

The diagnostic proved a deterministic single-vector MSE proposal MODE-AVERAGES the multimodal θ target. The fix (kept the
same across refinement and RL): keep the multimodality in a DISCRETE template choice a classifier handles, and regress only
the *unimodal within-mode residual*:

    obs → template classifier → template id → template medoid θ  +  template-conditioned residual r(obs, id) → θ_center.

The fixed deployed/RL wrapper then samples ``b`` structured candidates AROUND θ_center, selects the best by the frozen
canonical local-search score, and returns the selected committed option — the search stays load-bearing; the proposal only
decides *where* to search. Normalisation: amplitudes /A_BOUND, durations → [0,1]; residual regressed in that space.
"""
import copy

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_carry_option import teacher_theta
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, DIM, T_MAX, T_MIN, structured_carry_rollout, structured_random_around


def norm_theta(theta):
    """θ → normalised (amps in [-1,1], durs in [0,1]). Preconditions: theta[...,:12]∈±A_BOUND, theta[...,12:]∈[T_MIN,T_MAX]."""
    t = np.asarray(theta, np.float32)
    return np.concatenate([t[..., :12] / A_BOUND, (t[..., 12:] - T_MIN) / (T_MAX - T_MIN)], -1)


def denorm_theta(z):
    """Inverse of norm_theta, with a hard clip back to legal bounds (postcondition: result is legal θ)."""
    z = np.asarray(z, np.float32)
    amp = np.clip(z[..., :12] * A_BOUND, -A_BOUND, A_BOUND)
    dur = np.clip(z[..., 12:] * (T_MAX - T_MIN) + T_MIN, T_MIN, T_MAX)
    return np.concatenate([amp, dur], -1)


def kmeans(X, K, iters=40, seed=0):
    """Lloyd k-means in normalised θ space; returns (labels, medoid indices). Preconditions: len(X) ≥ K."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
    medoid = [int(np.argmin(((X - C[k]) ** 2).sum(-1) + (lab != k) * 1e9)) for k in range(K)]
    return lab, medoid


class _Clf(nn.Module):
    def __init__(self, K, obs=48, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, o):
        return self.net(o)


class _Residual(nn.Module):
    """(obs48 ⊕ template one-hot) → normalised residual θ (bounded ±RES_CAP by tanh so the proposal stays near its mode)."""

    RES_CAP = 1.0

    def __init__(self, K, obs=48, h=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(obs + K, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, DIM))

    def forward(self, o, onehot):
        return torch.tanh(self.net(torch.cat([o, onehot], -1))) * self.RES_CAP


class TemplateResidualProposal:
    """Deterministic obs→θ_center map that preserves multimodality via a discrete template + a unimodal residual.

    # Invariants: `templates` are legal θ; `theta(obs)` returns legal θ (denorm clips). The map is single-valued per obs
    (so it can itself be distilled by plain MSE for RL init, unlike the multimodal teacher)."""

    def __init__(self, templates, clf, residual):
        self.templates = np.asarray(templates, np.float32)
        self.templates_norm = norm_theta(self.templates)
        self.clf, self.residual = clf, residual
        self.K = len(templates)

    def _ids(self, obs):
        with torch.no_grad():
            return self.clf(torch.as_tensor(np.asarray(obs, np.float32))).argmax(-1).numpy()

    def theta(self, obs48):
        o = np.asarray(obs48, np.float32)
        single = o.ndim == 1
        o2 = o[None] if single else o
        ids = self._ids(o2)
        onehot = np.eye(self.K, dtype=np.float32)[ids]
        with torch.no_grad():
            res = self.residual(torch.as_tensor(o2), torch.as_tensor(onehot)).numpy()
        z = self.templates_norm[ids] + res
        out = denorm_theta(z)
        return out[0] if single else out


def fit_proposal(obs, theta, K, *, clf_epochs=300, res_epochs=300, seed=0):
    """Fit template classifier + template-conditioned residual on labelled (obs, θ). Residual target is unimodal within a
    template (all θ assigned to a template cluster near its medoid) ⇒ MSE is appropriate (unlike a global θ MSE)."""
    torch.manual_seed(seed)
    obs = np.asarray(obs, np.float32); theta = np.asarray(theta, np.float32)
    zt = norm_theta(theta)
    lab, medoid = kmeans(zt, K, seed=seed)
    templates = theta[medoid]; templates_norm = norm_theta(templates)
    clf = _Clf(K); optc = torch.optim.Adam(clf.parameters(), 3e-3); lf = nn.CrossEntropyLoss()
    x = torch.as_tensor(obs); y = torch.as_tensor(lab.astype(np.int64))
    for _ in range(clf_epochs):
        optc.zero_grad(); loss = lf(clf(x), y); loss.backward(); optc.step()
    res = _Residual(K)
    onehot = torch.as_tensor(np.eye(K, dtype=np.float32)[lab])
    tgt = torch.as_tensor((zt - templates_norm[lab]).astype(np.float32))
    optr = torch.optim.Adam(res.parameters(), 3e-3)
    for _ in range(res_epochs):
        optr.zero_grad(); pred = res(x, onehot); rl = ((pred - tgt) ** 2).sum(-1).mean(); rl.backward(); optr.step()
    return TemplateResidualProposal(templates, clf, res), {"clf_ce": float(loss.item()), "res_mse": float(rl.item()), "cluster_sizes": np.bincount(lab, minlength=K).tolist()}


def save_proposal(prop, path):
    """Persist a TemplateResidualProposal (templates + classifier + residual) for the RL init handoff."""
    torch.save({"templates": prop.templates, "K": prop.K, "clf": prop.clf.state_dict(), "residual": prop.residual.state_dict()}, path)


def load_proposal(path):
    d = torch.load(path, weights_only=False)
    clf = _Clf(d["K"]); clf.load_state_dict(d["clf"])
    res = _Residual(d["K"]); res.load_state_dict(d["residual"])
    return TemplateResidualProposal(d["templates"], clf, res)


def search_select(rl, gate, center, pi0, base, rng, *, b, std_amp=0.6, std_dur=2.0, horizon=160):
    """The FIXED local-search wrapper: sample ``b`` structured candidates AROUND ``center``, select the best by the frozen
    canonical structured_score, return (θ_selected, outcome). ``b==0`` executes the center directly (no rescue). This is the
    single source of truth for both deployment eval and the RL environment — the budget/dist/executor/score are frozen."""
    if b <= 0:
        theta = np.asarray(center, np.float32)
        return theta, structured_carry_rollout(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, theta, horizon=horizon)
    return structured_random_around(copy.deepcopy(rl), copy.deepcopy(gate), pi0, base, rng, shots=b, center=np.asarray(center, np.float32), std_amp=std_amp, std_dur=std_dur, horizon=horizon)


def canonical_label(rl, gate, pi0, base, rng, *, shots, horizon=160):
    """Strong per-state search → a refinement/RL proposal-improvement target θ*. ABSTAIN (None) unless the option delivers
    K6 or at least a valid handoff — never turn a merely-least-bad candidate into a confident label."""
    theta, admissible, out = teacher_theta(rl, gate, pi0, base, rng, shots=shots, horizon=horizon)
    if int(out["k6"]) == 1 or int(out["reached_handoff"]) == 1:
        return theta.astype(np.float32), out
    return None, out
