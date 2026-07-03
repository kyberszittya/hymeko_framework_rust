"""The RL campaign framework (Strategy + Template Method): self-contained artifacts, run.log, best-checkpoint.

Uses the cart-pole fixture + the MLP off-policy builder as a trivial architecture; the metric is a controlled
sequence so best-checkpoint selection is testable without depending on RL variance."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hymeko_rl.campaign import Campaign, CampaignConfig, compare, tee_stdout
from hymeko_rl.ddpg import build_offpolicy
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf

_MJCF = emit_cartpole_mjcf()


def _make_env() -> InvertedPendulumEnv:
    return InvertedPendulumEnv(mjcf=_MJCF)


def _build(env: InvertedPendulumEnv) -> tuple:
    ss = env.observation_space.shape
    assert ss is not None
    feat, flat = int(ss[1]), int(ss[0]) * int(ss[1])
    return build_offpolicy("mlp", obs_dim=feat, flat_dim=flat, action_dim=1, action_scale=env.force_mag,
                           n_critics=2)


def _seq_measure(scores: list[float]):
    """A measure Strategy that yields a controlled score sequence (padded with its last value)."""
    box = {"i": 0}

    def m(_make_env, _actor) -> dict[str, float]:
        v = scores[min(box["i"], len(scores) - 1)]
        box["i"] += 1
        return {"score": v}

    return m


def _cfg(name: str, **kw) -> CampaignConfig:
    base = dict(name=name, select="score", seeds=(0,), total_steps=60, eval_every=30, n_envs=1)
    base.update(kw)
    return CampaignConfig(**base)   # type: ignore[arg-type]


def test_tee_stdout_writes_every_line_and_restores() -> None:
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "log.txt"
        before = sys.stdout
        with tee_stdout(p):
            print("hello-tee")
        assert sys.stdout is before                        # stdout restored
        assert "hello-tee" in p.read_text(encoding="utf-8")   # and the line was captured


def test_campaign_config_is_frozen() -> None:
    c = CampaignConfig(name="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "y"                                       # type: ignore[misc]


def test_run_is_self_contained_with_log(tmp_path: Path) -> None:
    out = Campaign(_cfg("toy"), _make_env, _build, _seq_measure([0.4]), demos=None, gif=False).run(base=tmp_path)
    d = Path(out["dir"])
    assert (d / "results.json").exists()
    assert (d / "run.log").exists()                        # THE fix: the log lives with the artifacts
    assert list((d / "policies").glob("toy_s0.pt"))        # best-checkpoint policy saved
    assert "CURVE toy s0" in (d / "run.log").read_text(encoding="utf-8")
    assert out["score_median"] == 0.4


def test_best_checkpoint_keeps_the_peak_not_the_endpoint(tmp_path: Path) -> None:
    # score rises then falls; best-checkpoint must keep the PEAK (0.9), never the lower endpoint.
    out = Campaign(_cfg("peak", total_steps=120, eval_every=30), _make_env, _build,
                   _seq_measure([0.1, 0.9, 0.3, 0.3, 0.3]), gif=False).run(base=tmp_path)
    seed = out["seeds"][0]
    curve = [p["score"] for p in seed["curve"]]
    assert seed["peak"]["score"] == max(curve)
    assert seed["peak"]["score"] == 0.9                    # the middle peak, not the endpoint


def test_compare_runs_each_architecture(tmp_path: Path) -> None:
    res = compare(_cfg("ab"), _make_env, {"a": _build, "b": _build}, _seq_measure([0.5, 0.5, 0.5, 0.5]),
                  base=tmp_path)
    assert set(res["verdict"]) == {"a", "b"}               # one Campaign per architecture, keyed by arch
    assert set(res["runs"]) == {"a", "b"}
    assert res["runs"]["a"]["name"] == "ab_a"              # each arch's dir/name is namespaced
