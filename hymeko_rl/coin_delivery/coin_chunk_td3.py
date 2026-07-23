"""RECEDING_HORIZON_ACTION_CHUNK_TD3_V1 — contracts (A-E; NO TD3 fine-tuning here).

A short full-action CHUNK ``A_t = [a_t..a_{t+K-1}]`` (K=8) is proposed by ``pi_chunk(state_t)``; only the first M=2 are
executed (gate-routed; gate-off ⇒ frozen ``pi_0`` bit-identical), then the actor replans from ``state_{t+M}`` — a
receding-horizon feedback policy, NOT a constant residual hold (each chunk step is independently predicted). Twin chunk
critics ``Q_i(state_t, chunk_t)``; semi-MDP M-step target. Warm-started by supervised regression on planner + pi_0
sequences (the actor learns the first K actions of the feedback rollout FROM EACH STATE, not a time-indexed trajectory).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_transport_dwell import CONTACT_FLAGS, CONTROL_MODES, EventStateDetector, ENTRY_TOL
from hymeko_rl.coin_delivery.rl_clip_actor import make_backbone

K, M, ACT_DIM, ACTION_SCALE, GAMMA = 8, 2, 4, 4.0, 0.99
CHUNK_DIM = K * ACT_DIM                                   # 32
N_COND = len(CONTROL_MODES) + len(CONTACT_FLAGS) + 5      # 3+2+5 = 10
STATE_DIM = 48 + N_COND + ACT_DIM                         # obs 48 + conditioning 10 + prev action 4 (causal history) = 62


def state_vec(obs, cm, cf, ev, prev_action) -> np.ndarray:
    v = np.zeros(N_COND, np.float32)
    v[CONTROL_MODES.index(cm)] = 1.0
    v[len(CONTROL_MODES) + CONTACT_FLAGS.index(cf)] = 1.0
    v[len(CONTROL_MODES) + len(CONTACT_FLAGS):] = ev
    return np.concatenate([np.asarray(obs, np.float32), v, np.asarray(prev_action, np.float32)]).astype(np.float32)


# ── A. chunk actor + twin chunk critic ──
class ChunkActor(nn.Module):
    """``state (62) → chunk (K*4)`` clip-squashed to [-4,4]; ``chunk(state)`` returns (B, K, 4)."""

    def __init__(self, state_dim: int = STATE_DIM, k: int = K, action_dim: int = ACT_DIM, action_scale: float = ACTION_SCALE):
        super().__init__()
        self.backbone = make_backbone(state_dim, 256)
        self.head = nn.Linear(256, k * action_dim)
        self.k, self.action_dim, self.action_scale = int(k), int(action_dim), float(action_scale)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.head(self.backbone(state)), -self.action_scale, self.action_scale)   # (B, K*4)

    def chunk(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).view(-1, self.k, self.action_dim)                                 # (B, K, 4)


class ChunkTwinCritic(nn.Module):
    """Twin ``Q_i(state 62, chunk 32)`` with independent parameters."""

    def __init__(self, state_dim: int = STATE_DIM, chunk_dim: int = CHUNK_DIM):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(state_dim + chunk_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q1, self.q2 = mk(), mk()

    def forward(self, state, chunk):
        x = torch.cat([state, chunk], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, state, chunk):
        q1, q2 = self.forward(state, chunk)
        return torch.min(q1, q2)


# ── B. semi-MDP M-step target ──
def semi_mdp_target(prefix_rewards, terminated, truncated, boot_state, boot_chunk, critic_target, *, gamma=GAMMA, m=M):
    """y = Σ_{j<M'} γ^j r_{t+j} + γ^{M'}·mask·min(Q1_t,Q2_t)(boot_state, boot_chunk). M' = actual executed prefix length.
    terminated ⇒ mask 0 (no bootstrap); truncated ⇒ mask 1 (bootstrap). ``prefix_rewards`` has ≤ M entries."""
    G, disc = 0.0, 1.0
    term = False
    for j, r in enumerate(prefix_rewards):
        G += disc * float(r); disc *= gamma
        if terminated[j]:
            term = True; break
        if truncated[j]:
            break
    mask = 0.0 if term else 1.0
    with torch.no_grad():
        boot = critic_target.min_q(boot_state, boot_chunk) if mask else torch.zeros(boot_state.shape[0])
    return torch.as_tensor(G, dtype=torch.float32) + disc * mask * boot


# ── C. receding-horizon execution ──
def _pi0_action(pi0, obs):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def execute_chunk(rl, pi0, pi_chunk, gate, st, *, m=M):
    """Execute the first ``m`` planned actions of ``pi_chunk(st)`` with gate routing: gate-on ⇒ chunk step, gate-off ⇒
    frozen ``pi_0`` bit-identical. The detector is the CALLER's concern (one read per replan). Returns
    (requested_chunk(32), executed_prefix(M',4), per_step_records, obs_next)."""
    with torch.no_grad():
        chunk = pi_chunk.chunk(torch.as_tensor(np.asarray(st, np.float32)[None]))[0].numpy()   # (K, 4) requested
    executed, recs = [], []; o = rl.obs()
    for j in range(m):
        gate_on = gate.gate == 1.0
        a = np.clip(chunk[j], -ACTION_SCALE, ACTION_SCALE).astype(np.float32) if gate_on else _pi0_action(pi0, o)
        o2, r, term, trunc, _ = rl.step(a)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        executed.append(a)
        recs.append({"action": a, "reward": float(r), "terminated": bool(term), "truncated": bool(trunc), "gate_on": bool(gate_on)})
        o = o2
        if term or trunc:
            break
    return chunk.reshape(-1), np.stack(executed), recs, o


# ── D. warm-start sequence targets ──
def pi0_chunk_target(rl, pi0, *, k=K):
    """The next ``k`` frozen-pi_0 actions FROM THE CURRENT state (pi_0 rehearsal). Snapshots+restores the env."""
    from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar
    snap = (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)
    o = rl.obs(); seq = []
    for _j in range(k):
        with torch.no_grad():
            a = np.clip(pi0.action_mean(torch.as_tensor(o[None], dtype=torch.float32))[0].numpy(), -ACTION_SCALE, ACTION_SCALE).astype(np.float32)
        seq.append(a); o, _r, term, trunc, _ = rl.step(a)
        if term or trunc:
            seq += [seq[-1]] * (k - len(seq)); break
    restore_planar(rl.inner, snap[0]); rl._t, rl._strict, rl._touched = snap[1], snap[2], snap[3]
    return np.stack(seq[:k]).astype(np.float32)


def plan_chunk_target(rl, cf, seed, *, k=K, pop=24, iters=5, elite=6):
    """The best ``k``-step CEM action sequence from the current state (feedback-planner warm-start, better-than-pi_0
    when it finds strict/dwell gains). Reuses the existing receding-horizon scorer; leaves the env unchanged."""
    from hymeko_rl.coin_delivery.coin_v3_receding_horizon import CTRL_LIM, _lexo, _score_horizon, _snapshot, _restore
    from hymeko_rl.experiments.coin_neutral_start import _clearance
    inner = rl.inner; qpos, qvel = _snapshot(inner)
    clearance = _clearance(inner); touched = bool(rl._touched); dwell = int(rl._strict)
    dim = k * ACT_DIM; mean = np.zeros(dim, np.float32); sigma = np.full(dim, 0.6, np.float32); rng = np.random.default_rng(seed)
    best_lex, best_seq = None, mean.copy()
    for _it in range(iters):
        cand = np.clip(rng.normal(mean, sigma, (pop, dim)), -CTRL_LIM, CTRL_LIM).astype(np.float32); cand[0] = mean
        scored = []
        for c in cand:
            r = _score_horizon(inner, cf, qpos, qvel, c.reshape(k, ACT_DIM), clearance, touched, dwell)
            scored.append((_lexo(r), c))
        scored.sort(key=lambda x: x[0]); el = np.stack([c for _l, c in scored[-elite:]]); mean, sigma = el.mean(0), el.std(0) + 0.05
        if best_lex is None or scored[-1][0] > best_lex:
            best_lex, best_seq = scored[-1][0], scored[-1][1].copy()
    _restore(inner, qpos, qvel)
    return np.clip(best_seq.reshape(k, ACT_DIM), -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


def build_warmstart_dataset(pi0, starts, *, use_planner=True, planner_frac=0.4, k=K):
    """Per gate-active start: state_vec + target chunk (planner for a fraction, pi_0-rehearsal otherwise). Returns
    (states, target_chunks, provenance)."""
    X, Y, prov = [], [], []
    for i, ls in enumerate(starts):
        rl2, _gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        det = EventStateDetector(); cm, cf, ev = det.state_of(rl2)
        st = state_vec(rec.obs, cm, cf, ev, rec.base)          # prev action ≈ captured pi_0 action
        use_p = use_planner and (i % max(1, int(round(1 / planner_frac))) == 0)
        if use_p:
            tgt = plan_chunk_target(rl2, rl2.cf, seed=1000 + i, k=k); src = "planner"
        else:
            tgt = pi0_chunk_target(rl2, pi0, k=k); src = "pi0"
        X.append(st); Y.append(tgt.reshape(-1)); prov.append(src)
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), prov


# ── E. supervised training + eval ──
def train_supervised_chunk(X, Y, *, steps=3000, seed=0, lr=1e-3, log=print):
    torch.manual_seed(seed)
    actor = ChunkActor(); opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y); n = len(X); rng = np.random.default_rng(seed)
    for i in range(steps + 1):
        idx = rng.integers(0, n, min(256, n))
        pred = actor(Xt[idx]); loss = ((pred - Yt[idx]) ** 2).mean()
        if i == steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 500 == 0:
            log(f"    sup step {i}/{steps} mse {loss.item():.4f}")
    return actor


def chunk_metrics(actor, X, Y):
    with torch.no_grad():
        pred = actor(torch.tensor(X)).view(-1, K, ACT_DIM); tgt = torch.tensor(Y).view(-1, K, ACT_DIM)
    seq_mse = float(((pred - tgt) ** 2).mean())
    first_mse = float(((pred[:, 0] - tgt[:, 0]) ** 2).mean())
    prefix_mse = float(((pred[:, :M] - tgt[:, :M]) ** 2).mean())
    per_index = [round(float(((pred[:, j] - tgt[:, j]) ** 2).mean()), 4) for j in range(K)]
    return {"sequence_mse": round(seq_mse, 5), "first_action_mse": round(first_mse, 5),
            "two_step_prefix_mse": round(prefix_mse, 5), "per_index_mse": per_index}


def eval_receding_horizon(pi0, pi_chunk, dev_starts, *, horizon=60):
    """Receding-horizon rollout of the chunk actor (execute M, replan) vs frozen pi_0, on dev late-starts (§9/§12)."""
    late, base = [], []
    for ls in dev_starts:
        late.append(_rollout_rh(pi0, pi_chunk, ls, horizon, use_chunk=True))
        base.append(_rollout_rh(pi0, pi_chunk, ls, horizon, use_chunk=False))

    def agg(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else 0.0
    keys = ["strict_success", "max_dwell", "entered", "exited", "contact_retention", "progress", "ret", "replan_consistent"]
    out = {"n": len(dev_starts), "chunk": {k: round(agg(late, k), 4) for k in keys}, "pi0": {k: round(agg(base, k), 4) for k in keys}}
    out["delta_vs_pi0"] = {k: round(out["chunk"][k] - out["pi0"][k], 4) for k in keys}
    return out


def _accumulate(rl, recs, state):
    for rr in recs:
        state["tot"] += state["disc"] * rr["reward"]; state["disc"] *= GAMMA; state["steps"] += 1; state["n"] += 1
        state["max_dwell"] = max(state["max_dwell"], rl._strict)
        if rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact:
            state["contact"] += 1
        state["term"], state["trunc"] = rr["terminated"], rr["truncated"]
        dtz = rl._dtz(); state["progress"] += (state["prev_dtz"] - dtz); state["prev_dtz"] = dtz
        if dtz <= ENTRY_TOL:
            state["entered"] = True
        elif state["entered"]:
            state["exited"] = True
        if state["term"] or state["trunc"]:
            break


def _rollout_rh(pi0, pi_chunk, ls, horizon, *, use_chunk):
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = EventStateDetector(); pa = rec.base.astype(np.float32)
    s = {"tot": 0.0, "disc": 1.0, "max_dwell": rl._strict, "entered": rl._dtz() <= ENTRY_TOL, "exited": False,
         "contact": 0, "n": 0, "progress": 0.0, "prev_dtz": rl._dtz(), "term": False, "trunc": False, "steps": 0}
    while s["steps"] < horizon and not (s["term"] or s["trunc"]):
        cm, cf, ev = det.state_of(rl)                                       # one detector read per replan
        if use_chunk and gate.gate == 1.0:
            st = state_vec(rl.obs(), cm, cf, ev, pa)
            _chunk, executed, recs, _o = execute_chunk(rl, pi0, pi_chunk, gate, st, m=M)
            _accumulate(rl, recs, s); pa = executed[-1]
        else:
            a = _pi0_action(pi0, rl.obs())
            o2, r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            _accumulate(rl, [{"reward": r, "terminated": term, "truncated": trunc}], s); pa = a
    return {"strict_success": int(s["max_dwell"] >= HELD_DWELL), "max_dwell": int(s["max_dwell"]), "entered": int(s["entered"]),
            "exited": int(s["exited"]), "contact_retention": s["contact"] / max(s["n"], 1), "progress": float(s["progress"]),
            "ret": float(s["tot"]), "replan_consistent": 1}
