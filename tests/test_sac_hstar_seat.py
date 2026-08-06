"""SAC structural-entropy (H★) exploration seat — the HSiKAN-only exploration signal.

Certifies the seat's contract: the config field defaults to 0 (backward-compatible, no change to
non-HSiKAN runs); H★ is computable from a signedkan per-node actor's per-vertex activations; and it
is bounded in [0, 1] normalized. This is the HyMeKo-native exploration signal an MLP cannot provide.
"""

from __future__ import annotations

import dataclasses

import torch

from hymeko_rl.agents.star_entropy import star_expansion_entropy
from hymeko_rl.train.sac import SACConfig, build_sac
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def test_struct_entropy_coef_defaults_to_zero() -> None:
    fields = {f.name: f.default for f in dataclasses.fields(SACConfig)}
    assert "struct_entropy_coef" in fields
    assert fields["struct_entropy_coef"] == 0.0            # off by default → no change to existing runs


def test_hstar_computable_from_signedkan_actor() -> None:
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph"), seed=0)
    actor, _ = build_sac("signedkan", obs_dim=4, flat_dim=20, action_dim=4, action_scale=1.0,
                         hidden=32, actor_head="per_node", act_vertices=env._abd_vtx, hg_state=env.hg)
    assert hasattr(actor, "_node_acts")                    # HSiKAN per-node actor exposes per-vertex acts
    obs = torch.randn(8, 5, 4)
    hstar = star_expansion_entropy(actor._node_acts(obs))
    assert hstar.shape == (8,)
    assert torch.all(hstar >= -1e-6) and torch.all(hstar <= 1.0 + 1e-6)   # normalized entropy in [0, 1]


def test_mlp_actor_has_no_node_acts() -> None:
    # the H★ seat is HSiKAN-only: an MLP (pooled) actor has no per-vertex activations, so the seat is
    # inert for it (the guard `hasattr(actor, "_node_acts")` skips it) — MLP runs are unchanged.
    mlp, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=32)
    assert not hasattr(mlp, "_node_acts")
