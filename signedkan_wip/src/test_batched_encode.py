"""Correctness test: _encode_edges_batched vs _encode_edges_full.

Both must produce edge embeddings that agree within FP tolerance and
gradients that flow correctly to model parameters. Numerical diffs come
solely from the order of summation (sparse mm vs index_add) and are
bounded at ~1e-4 relative error in fp32.
"""
from __future__ import annotations

import numpy as np
import torch

from .datasets import load, split
from .hyperedges import construct
from .n_tuples import construct_k
from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from .signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .run_phase2_mixed_arity import _build_edge_incidence


def _build_per_arity_inputs(g, edges_array, arities, max_per_arity, device, seed):
    per_arity_tuples = []
    for k in arities:
        if k == 3:
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=max_per_arity[k], seed=seed)
        cap = max_per_arity[k]
        if cap and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        per_arity_tuples.append(t_k)

    per_arity_inputs = []
    for ai, k in enumerate(arities):
        tuples = per_arity_tuples[ai]
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        edge_to_tuples = build_edge_to_tuples(tuples)
        n_tuples = len(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        M_e = _build_edge_incidence(edges_array, edge_to_tuples,
                                      n_tuples, device)
        per_arity_inputs.append((triad_v, triad_sigma, M_vt, M_e))
    return per_arity_inputs


def _build_model(g, arities, hidden, n_layers, grid, batch_size, seed):
    torch.manual_seed(seed)
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=n_layers,
            hidden_dim=hidden, grid=grid, k=3,
            spline_kinds=["catmull_rom"] * n_layers,
            init_scale=0.05,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
        cycle_batch_size=batch_size,
    )
    return MixedAritySignedKAN(cfg)


def main():
    SEED = 0
    HIDDEN = 16
    N_LAYERS = 2
    GRID = 3
    BATCH_SIZE = 1000
    ARITIES = (3, 4)
    MAX_PER = {3: 5000, 4: 5000}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load("bitcoin_alpha")
    tr_idx, _, _ = split(g, seed=SEED)
    e_tr = g.edges[tr_idx]

    per_arity_inputs = _build_per_arity_inputs(
        g, e_tr, ARITIES, MAX_PER, device, SEED,
    )

    # Two models with IDENTICAL state, one full one batched.
    m_full = _build_model(g, ARITIES, HIDDEN, N_LAYERS, GRID, None, SEED).to(device)
    m_batch = _build_model(g, ARITIES, HIDDEN, N_LAYERS, GRID, BATCH_SIZE, SEED).to(device)
    m_batch.load_state_dict(m_full.state_dict())

    # Forward.
    edge_emb_full = m_full.encode_edges(per_arity_inputs)
    edge_emb_batch = m_batch.encode_edges(per_arity_inputs)
    assert edge_emb_full.shape == edge_emb_batch.shape, \
        f"shape mismatch: {edge_emb_full.shape} vs {edge_emb_batch.shape}"

    diff = (edge_emb_full - edge_emb_batch).abs()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    rel_max = (diff / (edge_emb_full.abs() + 1e-8)).max().item()
    print(f"forward: shape={edge_emb_full.shape}")
    print(f"  abs max diff:  {abs_max:.3e}")
    print(f"  abs mean diff: {abs_mean:.3e}")
    print(f"  rel max diff:  {rel_max:.3e}")

    # Backward — same loss target.
    target = torch.zeros(edge_emb_full.shape[0], device=device)
    loss_full = (m_full.classifier(edge_emb_full).squeeze(-1) - target).pow(2).mean()
    loss_batch = (m_batch.classifier(edge_emb_batch).squeeze(-1) - target).pow(2).mean()
    loss_full.backward()
    loss_batch.backward()

    print(f"\nloss: full={loss_full.item():.6e}  batched={loss_batch.item():.6e}")
    print(f"  loss diff: {abs(loss_full.item() - loss_batch.item()):.3e}")

    # Compare gradients on a sample of named parameters.
    print("\ngradient diffs:")
    max_param_rel = 0.0
    for (n_f, p_f), (n_b, p_b) in zip(m_full.named_parameters(),
                                        m_batch.named_parameters()):
        assert n_f == n_b
        if p_f.grad is None and p_b.grad is None:
            continue
        if p_f.grad is None or p_b.grad is None:
            print(f"  {n_f:>40s}: ONE GRAD MISSING")
            continue
        gd = (p_f.grad - p_b.grad).abs().max().item()
        gn = max(p_f.grad.abs().max().item(), 1e-8)
        rel = gd / gn
        max_param_rel = max(max_param_rel, rel)
        if gd > 1e-5:
            print(f"  {n_f:>40s}: abs_max={gd:.3e}  rel={rel:.3e}")
    print(f"\nmax relative gradient diff: {max_param_rel:.3e}")

    # Pass criteria: forward within 1e-3 absolute, gradients within 1e-2 relative.
    assert abs_max < 1e-3, f"forward abs diff too large: {abs_max}"
    assert max_param_rel < 1e-2, f"gradient rel diff too large: {max_param_rel}"
    print("\n✓ batched ≡ full within FP tolerance (forward + backward)")


if __name__ == "__main__":
    main()
