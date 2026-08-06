"""ARCHITECTURAL_ASSIMILATION_V1 A-FINISH.2 — ToyReach end-to-end integration through the WHOLE runtime.

A synthetic, non-coin SE(3)-lite reach exercises the full task-independent pipeline exactly as a real adapter will:
  raw env → StructuredStateAdapter → LSTM streaming → fusion → K-mode proposal → MultimodalBudgetSearch → option
  execution → certificate → checkpoint/replay restore.
No coin/CIP/pick-place, no MuJoCo — this is the runtime's own end-to-end proof (gate 3) before the API freeze.
"""
import numpy as np
import torch

from hymeko_rl.option_rl import (
    FlatStateView,
    LSTMTemporalEncoder,
    MultimodalBudgetSearch,
    ProposalMode,
    StructuredState,
    fuse_state,
)

THRESH = 0.15


class ToyReach:
    """State = end-effector pos(3) + fixed target(3); an option applies a 3-vector move (clipped); reached iff dist<THRESH."""

    def __init__(self, target):
        self.pos = np.zeros(3, np.float32)
        self.target = np.asarray(target, np.float32)

    def obs(self):
        return np.concatenate([self.pos, self.target]).astype(np.float32)

    def step(self, move):
        self.pos = (self.pos + np.clip(np.asarray(move, np.float32), -0.3, 0.3)).astype(np.float32)

    def dist(self):
        return float(np.linalg.norm(self.target - self.pos))


class ToyReachAdapter:
    """StructuredStateAdapter: two nodes (current EE, target) + target-relative geometry — a minimal SE(3)-lite state."""

    def structured(self, env: ToyReach) -> StructuredState:
        nf = np.stack([np.concatenate([env.pos, [0.0]]), np.concatenate([env.target, [1.0]])]).astype(np.float32)
        return StructuredState(nf, edges=[(0, 1)], geometry=(env.target - env.pos), phase=0, metadata={"task": "reach"})


class ReachProposal:
    """MultimodalProposalPolicy: K modes = K step scales toward the target (a toy multimodality over the approach)."""

    def __init__(self, k):
        self.k = k

    def modes(self, obs):
        pos, tgt = np.asarray(obs)[:3], np.asarray(obs)[3:]
        step = (tgt - pos) / 6.0                              # the per-step move that reaches the target over the 6-step option
        scales = np.linspace(0.8, 1.2, self.k) if self.k > 1 else [1.0]   # K modes bracket the exact scale
        return [ProposalMode(1.0 / self.k, (step * s).astype(np.float32), 0.01, i) for i, s in enumerate(scales)]


class ReachScorer:
    """Rolls a candidate move as a short committed option on a COPY of the env; higher = closer; certificate = reached."""

    def __init__(self, env):
        self.p0, self.t = env.pos.copy(), env.target.copy()

    def score(self, cand, rng):
        p = self.p0.copy()
        for _ in range(6):                                   # the committed option: repeat the move toward the target
            p = p + np.clip(np.asarray(cand, np.float32), -0.3, 0.3)
        d = float(np.linalg.norm(self.t - p))
        return -d, {"reached": int(d < THRESH), "dist": d, "k6": int(d < THRESH)}


def _encode(env, adapter, enc, hidden):
    """One representation step: structured state → flat structured embedding, LSTM streaming over the obs, fuse."""
    s = adapter.structured(env)
    struct_emb = FlatStateView().view(s)                     # structured channel
    with torch.no_grad():
        temporal, hidden = enc.update(torch.as_tensor(env.obs()), hidden)   # LSTM streaming (hidden = runtime state)
    return fuse_state(struct_emb, temporal.numpy()[0]), hidden


def test_toy_reach_end_to_end_multimodal_reaches():
    env = ToyReach([0.5, -0.4, 0.3])
    adapter, enc = ToyReachAdapter(), LSTMTemporalEncoder(in_dim=6, hidden=8, out_dim=4).eval()
    hidden = enc.initial_hidden(1)
    fused, hidden = _encode(env, adapter, enc, hidden)
    assert fused.shape[0] == (2 * 4) + 3 + 1 + 4             # struct(node8 ⊕ geom3 ⊕ phase1) ⊕ temporal4
    prov = MultimodalBudgetSearch(_Gen(), ReachScorer(env), budget=12).select(ReachProposal(4), env.obs(), np.random.default_rng(0))
    for _ in range(6):                                       # execute the committed option
        env.step(prov.selected)
    assert prov.outcome["reached"] == 1 and env.dist() < THRESH   # the full pipeline delivers the certificate


def test_toy_reach_k1_also_runs():
    env = ToyReach([0.4, 0.4, -0.2])
    prov = MultimodalBudgetSearch(_Gen(), ReachScorer(env), budget=8).select(ReachProposal(1), env.obs(), np.random.default_rng(0))
    assert prov.as_dict()["n_modes"] == 1 and "dist" in prov.outcome


def test_toy_reach_checkpoint_restore_identical_next_embedding(tmp_path):
    env, adapter, enc = ToyReach([0.3, 0.3, 0.3]), ToyReachAdapter(), LSTMTemporalEncoder(6, 8, 4).eval()
    hidden = enc.initial_hidden(1)
    _f, hidden = _encode(env, adapter, enc, hidden)         # advance the temporal state one step
    env.step([0.1, 0.0, 0.0])
    expected, _ = _encode(env, adapter, enc, hidden)
    ck = tmp_path / "rt.pt"
    torch.save({"w": enc.state_dict(), "hidden": (hidden[0], hidden[1]), "pos": env.pos, "target": env.target}, ck)
    blob = torch.load(ck, weights_only=False)
    enc2 = LSTMTemporalEncoder(6, 8, 4)
    enc2.load_state_dict(blob["w"])
    enc2.eval()
    env2 = ToyReach(blob["target"])
    env2.pos = blob["pos"]
    got, _ = _encode(env2, adapter, enc2, blob["hidden"])
    assert np.allclose(got, expected, atol=1e-6)            # replay/checkpoint restore → identical next fused state


class _Gen:
    def sample(self, center, n, rng):
        return np.asarray(center, np.float64) + rng.normal(0, 0.004, (n, len(center)))  # small jitter (×6 in the option)
