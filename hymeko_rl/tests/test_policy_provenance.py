"""Unit tests for the PolicyProvenanceLedger — the identity guard (checkpoint md5 / param hash / action checksum /
anchor identity). Tiny torch modules; the real-checkpoint validation is ``scratchpad/provenance_validate.py``."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from hymeko_rl.eval.task_monitor import (
    PolicyProvenanceError,
    PolicyProvenanceLedger,
    PolicyRole,
    action_checksum,
    canonical_obs_batch,
    file_md5,
    param_hash,
)

OBS = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)


class TinyActor(nn.Module):
    """A minimal deterministic actor exposing ``action_mean`` (the convention greedy_action_fn uses)."""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(8, 4)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(4, 8, generator=g))
            self.lin.bias.copy_(torch.randn(4, generator=g))

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.lin(obs.reshape(obs.shape[0], -1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_mean(obs)


def _load(path) -> TinyActor:
    a = TinyActor(999)
    a.load_state_dict(torch.load(path))
    return a


def test_param_hash_and_checksum_determinism():
    assert param_hash(TinyActor(1)) == param_hash(TinyActor(1)) != param_hash(TinyActor(2))
    x = TinyActor(1).action_mean(torch.as_tensor(OBS)).detach().numpy()
    assert action_checksum(x) == action_checksum(x.copy())
    assert action_checksum(x) != action_checksum(x + 1.0)


def test_canonical_obs_batch_deterministic():
    class Env:
        def reset(self, seed=None):
            return np.random.default_rng(seed).standard_normal((6, 8)).astype(np.float32), {}

        def close(self):
            pass

    b1 = canonical_obs_batch(lambda: Env(), n=4, seed0=100)
    b2 = canonical_obs_batch(lambda: Env(), n=4, seed0=100)
    assert b1.shape == (4, 6, 8) and np.array_equal(b1, b2)


def test_rl_init_passes_when_actor_anchor_are_the_selected(tmp_path):
    p = tmp_path / "sel.pt"
    torch.save(TinyActor(1).state_dict(), p)
    led = PolicyProvenanceLedger(OBS)
    led.register_checkpoint("selected", PolicyRole.DAGGER_VAL_SELECTED, str(p), _load(p),
                            arch="mlp", seed=1, dagger_stage="d3")
    led.register_checkpoint("rl_actor", PolicyRole.RL_ACTOR, str(p), _load(p), arch="mlp", seed=1)
    led.register_checkpoint("anchor", PolicyRole.DAGGER_VAL_SELECTED, str(p), _load(p), arch="mlp", seed=1)
    led.assert_checkpoint_matches("selected", file_md5(str(p)))
    led.assert_rl_init("rl_actor", "anchor", "selected")
    assert led.action_mse("rl_actor", "anchor") == pytest.approx(0.0)
    # queued expectations (what train_offpolicy runs at RL init)
    led.expect_checkpoint_matches("selected", file_md5(str(p)))
    led.expect_rl_init("rl_actor", "anchor", "selected")
    assert led.verify_or_abort().passed


def test_rl_init_fails_on_wrong_actor(tmp_path):
    p = tmp_path / "sel.pt"
    torch.save(TinyActor(1).state_dict(), p)
    led = PolicyProvenanceLedger(OBS)
    led.register_checkpoint("selected", PolicyRole.DAGGER_VAL_SELECTED, str(p), _load(p), arch="mlp")
    led.register_checkpoint("rl_actor", PolicyRole.RL_ACTOR, None, TinyActor(2), arch="mlp")   # different params
    led.register_checkpoint("anchor", PolicyRole.DAGGER_VAL_SELECTED, str(p), _load(p), arch="mlp")
    with pytest.raises(PolicyProvenanceError, match="param hash"):
        led.assert_rl_init("rl_actor", "anchor", "selected")
    led.expect_rl_init("rl_actor", "anchor", "selected")
    assert not led.verify().passed


def test_checkpoint_md5_mismatch_fails(tmp_path):
    p = tmp_path / "sel.pt"
    torch.save(TinyActor(1).state_dict(), p)
    led = PolicyProvenanceLedger(OBS)
    led.register_checkpoint("selected", PolicyRole.DAGGER_VAL_SELECTED, str(p), _load(p), arch="mlp")
    with pytest.raises(PolicyProvenanceError, match="md5"):
        led.assert_checkpoint_matches("selected", "deadbeefdeadbeef")


def test_scripted_cannot_be_a_learned_actor():
    led = PolicyProvenanceLedger(OBS)
    led.register_scripted("scripted_teacher")
    r = led.records["scripted_teacher"]
    assert r.is_scripted and r.role is PolicyRole.SCRIPTED_TEACHER and r.param_hash is None
    with pytest.raises(PolicyProvenanceError, match="not a valid learned actor"):
        led.assert_learned_not_scripted("scripted_teacher")
    # a scripted CALLABLE has no state_dict → cannot be registered as a learned checkpoint (assertion 4)
    with pytest.raises(PolicyProvenanceError, match="no state_dict"):
        led.register_checkpoint("fake", PolicyRole.RL_ACTOR, None, lambda env, obs: obs, arch="scripted")


def test_learned_role_required_for_checkpoint():
    led = PolicyProvenanceLedger(OBS)
    with pytest.raises(PolicyProvenanceError, match="not a learned role"):
        led.register_checkpoint("x", PolicyRole.SCRIPTED_TEACHER, None, TinyActor(1), arch="mlp")


def test_report_fields(tmp_path):
    p = tmp_path / "sel.pt"
    torch.save(TinyActor(1).state_dict(), p)
    led = PolicyProvenanceLedger(OBS)
    led.register_checkpoint("rl_actor", PolicyRole.RL_ACTOR, str(p), _load(p),
                            arch="mlp", seed=1, dagger_stage="d3")
    f = led.report_fields(actor_name="rl_actor", reward_file="galambos_task_deliver_v2b.hymeko",
                          env_file="galambos_env_v2.hymeko")
    assert f["actor_checkpoint_hash"] == file_md5(str(p))
    assert f["dagger_stage"] == "d3" and f["reward_file"].endswith("v2b.hymeko")
    assert f["policy_provenance"] == "PASS"
