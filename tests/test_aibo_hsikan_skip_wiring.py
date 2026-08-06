"""HSiKAN highway/skip wiring for the omni runner — the 'H' in HSiKAN, wired through build_sac.

The AIBO omni runs so far used ``skip="none"`` (plain signed-conv, the default). The runner now
exposes ``--skip {none,residual,highway}``; ``highway`` is the Schmidhuber gate — the namesake 'H'.
These lock the invariant the runner relies on: ``build_sac("signedkan", skip="highway")`` genuinely
adds per-layer highway gate parameters (so the flag is not a silent no-op), ``skip="none"`` does not,
and ``sa_hsikan`` builds regardless (it ignores skip — the holonomy operator is a structural constant).
"""

from __future__ import annotations

import torch

from hymeko_rl.train.sac import build_sac
from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def _omni_env() -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph"), seed=0)


def _signedkan(env: ResidualTrotEnv, skip: str):
    return build_sac("signedkan", obs_dim=4, flat_dim=env._n_vtx * 4, action_dim=4, action_scale=1.0,
                     hidden=32, actor_head="per_node", act_vertices=env._abd_vtx, hg_state=env.hg, skip=skip)


def _has_highway_gate(module: torch.nn.Module) -> bool:
    return any("skip.gate" in nm for nm, _ in module.named_parameters())


def test_highway_skip_adds_gate_parameters() -> None:
    env = _omni_env()
    actor, _ = _signedkan(env, "highway")
    assert _has_highway_gate(actor)                        # the 'H' is genuinely wired, not a no-op


def test_none_skip_has_no_gate() -> None:
    env = _omni_env()
    actor, _ = _signedkan(env, "none")
    assert not _has_highway_gate(actor)                    # the prior default = plain signed-conv


def test_highway_actor_more_params_than_plain() -> None:
    # the gate is a real per-layer Linear, so highway strictly grows the parameter count
    env = _omni_env()
    plain, _ = _signedkan(env, "none")
    highway, _ = _signedkan(env, "highway")
    assert sum(p.numel() for p in highway.parameters()) > sum(p.numel() for p in plain.parameters())


def test_highway_forward_matches_action_shape() -> None:
    env = _omni_env()
    actor, critics = _signedkan(env, "highway")
    obs = torch.randn(3, env._n_vtx, 4)
    with torch.no_grad():
        a = actor.action_mean(obs)
        q = critics[0](obs, a)
    assert a.shape == (3, 4) and q.shape == (3,)


def test_sa_hsikan_builds_and_ignores_skip() -> None:
    # sa_hsikan is a structural constant (Bᴸ collapse) — skip is not applicable; it must still build.
    env = _omni_env()
    actor, _ = build_sac("sa_hsikan", obs_dim=4, flat_dim=env._n_vtx * 4, action_dim=4, action_scale=1.0,
                         hidden=32, actor_head="per_node", act_vertices=env._abd_vtx, hg_state=env.hg, skip="none")
    obs = torch.randn(3, env._n_vtx, 4)
    with torch.no_grad():
        a = actor.action_mean(obs)
    assert a.shape == (3, 4)
    assert not _has_highway_gate(actor)                    # no highway inside the holonomy readout
