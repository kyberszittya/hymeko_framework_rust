"""HSiKAN behavioral cloning on CartPole.

Minimal CartPole simulator (no gym dependency) + an analytical
LQR-style expert that keeps the pole upright.  Collect (state,
action) pairs from expert rollouts, build a k-NN signed graph over
states with action-agreement signs (P1, action ∈ {0, 1}), train
HSiKAN as a binary-action classifier, then evaluate by rolling out
the trained policy and measuring mean episode length.

Acceptance: HSiKAN-policy mean episode length close to the expert's
(typically 200 — the maximum in the standard CartPole-v1 setup).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig, MultiLayerSignedKANConfig,
)
from .run_tabular_smoke import build_M_vt, build_per_arity
from signedkan_wip.src.tabular_signed_graph import build_signed_graph_from_tabular


# CartPole-v1 physics constants (from OpenAI Gym).
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSPOLE + MASSCART
LENGTH = 0.5  # half-pole length
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates
THETA_THRESHOLD = 12 * 2 * math.pi / 360  # 12 degrees
X_THRESHOLD = 2.4
MAX_STEPS = 200


def step(state: np.ndarray, action: int) -> tuple[np.ndarray, bool]:
    """One step of CartPole physics. action: 0 = left, 1 = right."""
    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG
    costheta, sintheta = math.cos(theta), math.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc
    new_state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
    done = (
        abs(x) > X_THRESHOLD
        or abs(theta) > THETA_THRESHOLD
    )
    return new_state, done


def expert_action(state: np.ndarray) -> int:
    """Hand-tuned LQR-flavoured policy for CartPole.

    Rule: action = 1 (push right) iff K · state > 0, where K is a
    classic CartPole gain matrix.  Stabilises the pole near upright
    indefinitely (caps at MAX_STEPS = 200).
    """
    K = np.array([1.0, 1.0, 30.0, 5.0])
    return int((K @ state) > 0)


def collect_trajectories(n_episodes: int, seed: int = 0) -> tuple[np.ndarray,
                                                                    np.ndarray]:
    """Run `n_episodes` of expert rollouts, return concatenated
    (states, actions)."""
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for ep in range(n_episodes):
        state = rng.uniform(-0.05, 0.05, size=4)
        for _ in range(MAX_STEPS):
            a = expert_action(state)
            states.append(state.copy())
            actions.append(a)
            state, done = step(state, a)
            if done:
                break
    return np.array(states), np.array(actions, dtype=np.int64)


def evaluate_policy(policy_fn, n_episodes: int = 20,
                    seed: int = 0) -> float:
    """Roll out `policy_fn(state) -> action` for `n_episodes`, return
    mean episode length (max MAX_STEPS = 200)."""
    rng = np.random.default_rng(seed)
    lengths = []
    for ep in range(n_episodes):
        state = rng.uniform(-0.05, 0.05, size=4)
        ep_len = 0
        for _ in range(MAX_STEPS):
            a = int(policy_fn(state))
            state, done = step(state, a)
            ep_len += 1
            if done:
                break
        lengths.append(ep_len)
    return float(np.mean(lengths))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-episodes", type=int, default=50)
    ap.add_argument("--k-nn", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--n-eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Collect expert demonstrations.
    print(f"[expert] collecting {args.n_train_episodes} episodes...")
    X, y = collect_trajectories(args.n_train_episodes, seed=args.seed)
    print(f"[expert] states={X.shape}, actions={y.shape}, "
          f"action_balance={(y == 1).mean():.3f}")

    # Reference: expert mean episode length.
    expert_len = evaluate_policy(
        expert_action, n_episodes=args.n_eval_episodes,
        seed=args.seed + 100,
    )
    print(f"[expert] mean episode length: {expert_len:.1f} / {MAX_STEPS}")

    # Reference: random policy.
    rng_pol = np.random.default_rng(args.seed)
    random_len = evaluate_policy(
        lambda s: rng_pol.integers(0, 2),
        n_episodes=args.n_eval_episodes, seed=args.seed + 100,
    )
    print(f"[random] mean episode length: {random_len:.1f} / {MAX_STEPS}")

    # 2. Build k-NN signed graph (P1 — action agreement).
    g = build_signed_graph_from_tabular(
        X, y=y, k=args.k_nn, protocol="p1",
    )
    print(f"[graph] n_nodes={g.n_nodes}, n_edges={g.edges.shape[0]}, "
          f"pos_frac={(g.signs == 1).mean():.3f}")

    arities = (3, 4)
    per_arity_tuples, arities_used = build_per_arity(
        g, arities, max_k=3000, seed=args.seed,
    )

    per_arity_inputs: list[tuple[torch.Tensor, ...]] = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, g.n_nodes, device)
        rows = np.zeros(triad_v.shape[0], dtype=np.int64)
        cols = np.arange(triad_v.shape[0], dtype=np.int64)
        vals = np.ones(triad_v.shape[0], dtype=np.float32) / max(
            1, triad_v.shape[0]
        )
        idx = torch.tensor(np.stack([rows, cols]),
                            dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        M_e_dummy = torch.sparse_coo_tensor(
            idx, v, (1, triad_v.shape[0]),
        ).coalesce()
        per_arity_inputs.append(
            (triad_v_t, triad_sigma_t, M_vt, M_e_dummy)
        )

    # Standardise states for vertex-feature input.
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=args.hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none",
            use_residual=True),
        arities=tuple(arities_used),
        init_arity_logits=tuple([0.0] * len(arities_used)),
        vertex_feat_dim=4,
    )
    model = MixedAritySignedKAN(cfg).to(device)
    head = nn.Linear(args.hidden, 2).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=5e-3,
    )

    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train(); head.train()
        _ = model.encode_edges(per_arity_inputs, vertex_features=Xs_t)
        h_v = model._final_h_v
        logits = head(h_v)
        loss = F.cross_entropy(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y_t).float().mean().item()
            print(f"  epoch {ep+1:3d}  loss={loss.item():.4f}  "
                  f"train_acc={acc:.3f}")
    train_s = time.time() - t0

    # 3. Roll out the trained HSiKAN policy.
    # Build a 1-NN lookup: for a query state, find the closest training
    # state's vertex embedding and predict action from its head logits.
    # (This is the simplest way to query an inductive policy from a
    # transductive node-classifier; matches how graph-based BC works in
    # practice.)
    model.eval(); head.eval()
    with torch.no_grad():
        _ = model.encode_edges(per_arity_inputs, vertex_features=Xs_t)
        h_v_train = model._final_h_v.cpu().numpy()  # (n_train, h)
    X_mean, X_std = X.mean(0), X.std(0) + 1e-12

    def hsikan_policy(state: np.ndarray) -> int:
        # Project state to standardised feature space.
        s = (state - X_mean) / X_std
        # 1-NN over training states.
        dists = np.linalg.norm(Xs - s[None, :], axis=1)
        nn_idx = int(dists.argmin())
        h = torch.tensor(h_v_train[nn_idx], dtype=torch.float32,
                          device=device)
        with torch.no_grad():
            return int(head(h.unsqueeze(0)).argmax(-1).item())

    hsikan_len = evaluate_policy(
        hsikan_policy, n_episodes=args.n_eval_episodes,
        seed=args.seed + 100,
    )

    alpha_vec = [float(a) for a in
                  model.alpha().detach().cpu().tolist()]
    out = dict(
        task="cartpole_bc",
        n_train_episodes=args.n_train_episodes,
        n_train_states=int(X.shape[0]),
        n_edges=int(g.edges.shape[0]),
        hidden=args.hidden, arities=list(arities_used), alpha=alpha_vec,
        expert_mean_len=expert_len,
        random_mean_len=random_len,
        hsikan_mean_len=hsikan_len,
        hsikan_recover_pct=float(
            100 * (hsikan_len - random_len) /
                  max(1.0, expert_len - random_len)
        ),
        n_params=sum(p.numel() for p in
                      list(model.parameters()) + list(head.parameters())),
        train_s=train_s, n_epochs=args.n_epochs, seed=args.seed,
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
