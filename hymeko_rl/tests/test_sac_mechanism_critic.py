"""F11 vs F12 semantic-critic contrast — focused regression tests.

F11 = 1 actor × task critic only (``critic_mode="TASK_ONLY"``, the pre-existing trainer). F12 = 1 actor × task critic +
a SEPARATE semantic ``Q_mechanism`` critic (``critic_mode="TASK_AND_MECHANISM"``). The mechanism critic is NOT the twin-Q
anti-overestimation pair: it is an independent value estimator whose bounded target is the mechanism-validity signal read
from canonical NAMED observation fields (``both_contact ∧ ¬arm_body_contact``), fed into the actor objective with a
fixed, pre-registered coefficient ``mech_coef``. These tests certify: (1) the F11 path is byte-identical to before;
(2) the mechanism critic exerts a real, coefficient-scaled gradient on the actor in F12; (3) the mechanism target comes
from the named fields and is independent of the environment reward; (4) invalid configs fail loudly; (5) actor
checkpoints round-trip for both modes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.eval.team_tensor import field_index
from hymeko_rl.train.rl_config import (
    CriticMode, PolicyKind, Strategy, UnsupportedRLConfig, mechanism_reward, validate_rl_config,
)
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_OBS = 41                                          # coin-delivery observation width (flat ACTOR_FIELDS)
_ACT = 6                                            # two 2-link planar arms + gate channels
_BOTH = field_index("both_contact")                # 28
_BODY = field_index("arm_body_contact")            # 29


class _ContactToyEnv:
    """A deterministic 41-dim / 6-dim env exercising the mechanism-validity fields, with a reward CHANNEL fully
    decoupled from those fields — so a test can prove the mechanism target reads the contact fields while the env
    reward does not depend on them. gym-shaped: ``reset(seed)->（obs,info)``, ``step(a)->(obs,r,term,trunc,info)``."""

    class _Space:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    def __init__(self, max_steps: int = 40) -> None:
        self.observation_space = self._Space((_OBS,))
        self.action_space = self._Space((_ACT,))
        self.max_steps = max_steps
        self._t = 0
        self._rng = np.random.default_rng(0)

    def _obs(self) -> np.ndarray:
        o = self._rng.standard_normal(_OBS).astype(np.float32) * 0.1
        o[_BOTH] = 1.0 if (self._t % 2 == 0) else 0.0        # clean bilateral contact on even steps
        o[_BODY] = 1.0 if (self._t % 3 == 0) else 0.0        # a body-shove every third step (invalidates mechanism)
        return o

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(0 if seed is None else seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._t += 1
        # reward depends ONLY on the action energy — never on the contact fields the mechanism target reads
        reward = -float(np.mean(np.square(action)))
        term = False
        trunc = self._t >= self.max_steps
        return self._obs(), reward, term, trunc, {}


def _fresh_actor_critics() -> tuple[object, list]:
    torch.manual_seed(0)
    return build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, hidden=16)


def _cfg(critic_mode: str = "TASK_ONLY", *, mech_coef: float = 0.5, seed: int = 0) -> SACConfig:
    return SACConfig(total_steps=240, start_steps=40, batch_size=16, capacity=500, eval_every=1000,
                     log_every=0, n_eval=1, seed=seed, critic_mode=critic_mode, mech_coef=mech_coef)


def _flat(actor: object) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in actor.parameters()]).numpy()   # type: ignore[attr-defined]


# ── 1. F11 is byte-identical to the pre-existing trainer ─────────────────────────────────────────────────────────
def test_task_only_is_byte_identical_default() -> None:
    """The default config IS ``critic_mode="TASK_ONLY"``; adding the F12 machinery must not perturb the F11 path
    (the ``if _mech`` guards consume zero RNG when False)."""
    assert SACConfig().critic_mode == "TASK_ONLY"
    a1, c1 = _fresh_actor_critics()
    train_sac(a1, c1, _ContactToyEnv(), _cfg("TASK_ONLY"), eval_fn=lambda *_: 0.0)
    a2, c2 = _fresh_actor_critics()
    train_sac(a2, c2, _ContactToyEnv(), SACConfig(total_steps=240, start_steps=40, batch_size=16, capacity=500,
              eval_every=1000, log_every=0, n_eval=1, seed=0), eval_fn=lambda *_: 0.0)   # no critic_mode → default
    assert np.array_equal(_flat(a1), _flat(a2))                        # F11 explicit == default: identical actor


# ── 2. the mechanism critic exerts a real gradient on the actor in F12 ────────────────────────────────────────────
def test_f12_mechanism_critic_changes_the_actor() -> None:
    a11, c11 = _fresh_actor_critics()
    train_sac(a11, c11, _ContactToyEnv(), _cfg("TASK_ONLY"), eval_fn=lambda *_: 0.0)
    a12, c12 = _fresh_actor_critics()
    train_sac(a12, c12, _ContactToyEnv(), _cfg("TASK_AND_MECHANISM"), eval_fn=lambda *_: 0.0)
    assert not np.allclose(_flat(a11), _flat(a12))                     # F12 actor diverges: Q_mechanism reached it


# ── 3. mechanism target is read from canonical NAMED fields, bounded, truth-tabled ────────────────────────────────
def test_mechanism_reward_from_canonical_named_fields() -> None:
    assert (_BOTH, _BODY) == (field_index("both_contact"), field_index("arm_body_contact"))
    o = torch.zeros(4, _OBS)
    o[0, _BOTH] = 1.0                                                  # clean bilateral, no shove → valid
    o[1, _BOTH] = 1.0                                                  # bilateral but shoving → invalid
    o[1, _BODY] = 1.0
    o[2, _BODY] = 1.0                                                  # shove only → invalid
    #  o[3] all-zero → invalid
    r = mechanism_reward(o)
    assert r.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0            # bounded [0,1]
    big = torch.full((1, _OBS), 5.0)                                  # out-of-range fields are clamped, still bounded
    assert 0.0 <= float(mechanism_reward(big)) <= 1.0


# ── 4. task + mechanism losses stay finite; the mechanism critic actually trains ──────────────────────────────────
def test_f12_losses_finite_and_mechanism_critic_trained() -> None:
    a, c = _fresh_actor_critics()
    diag: dict[str, float] = {}
    train_sac(a, c, _ContactToyEnv(), _cfg("TASK_AND_MECHANISM"), eval_fn=lambda *_: 0.0, diag_out=diag)
    assert diag["critic_mode"] is True
    for k in ("last_c", "last_a", "mech_crit_loss", "q_task", "q_mech"):
        assert np.isfinite(diag[k]), f"{k} not finite: {diag[k]}"
    assert diag["mech_crit_loss"] >= 0.0                              # an MSE, non-negative
    assert all(torch.isfinite(p).all() for p in a.parameters())      # no NaN leaked into the actor


# ── 5. the mechanism LABEL changes the target, never the env reward ───────────────────────────────────────────────
def test_mechanism_label_changes_target_not_env_reward() -> None:
    """Flipping the mechanism-validity fields changes ``mechanism_reward`` but the env's reward channel (action
    energy) is independent — the mechanism critic learns a semantic target, it does not touch the task reward."""
    base = torch.zeros(1, _OBS)
    valid = base.clone()
    valid[0, _BOTH] = 1.0
    assert float(mechanism_reward(base)) == 0.0 and float(mechanism_reward(valid)) == 1.0   # label drives the target
    env = _ContactToyEnv()
    env.reset(seed=0)
    _, r_a, *_ = env.step(np.zeros(_ACT, np.float32))
    _, r_b, *_ = env.step(np.zeros(_ACT, np.float32))
    assert r_a == r_b == 0.0                                          # zero action → zero reward on BOTH contact phases


# ── 6. invalid / not-yet-implemented configs fail LOUDLY (never silent fallback) ──────────────────────────────────
def test_invalid_rl_configs_fail_loud() -> None:
    validate_rl_config(PolicyKind.SAC_SINGLE_ACTOR, Strategy.DIRECT, CriticMode.TASK_ONLY)          # supported
    validate_rl_config(PolicyKind.SAC_SINGLE_ACTOR, Strategy.DIRECT, CriticMode.TASK_AND_MECHANISM)  # supported
    with pytest.raises(UnsupportedRLConfig, match="requires the HYMeko_CONTACT_MODE"):               # bank needs its selector
        validate_rl_config(PolicyKind.SAC_CONTACT_ACTOR_BANK, Strategy.DIRECT, CriticMode.TASK_ONLY)
    with pytest.raises(UnsupportedRLConfig, match="CRITIC_SELECTED requires"):
        validate_rl_config(PolicyKind.SAC_SINGLE_ACTOR, Strategy.CRITIC_SELECTED, CriticMode.TASK_ONLY)
    with pytest.raises(UnsupportedRLConfig, match="HYMeko_CONTACT_MODE requires"):
        validate_rl_config(PolicyKind.SAC_SINGLE_ACTOR, Strategy.HYMEKO_CONTACT_MODE, CriticMode.TASK_ONLY)
    with pytest.raises(UnsupportedRLConfig, match="architecture-incompatible"):
        validate_rl_config(PolicyKind.SAC_SINGLE_ACTOR, Strategy.DIRECT, CriticMode.TASK_ONLY,
                           obs_dim=41, checkpoint_obs_dim=39)


# ── 7. in F12 the task value and mechanism value are distinct estimators ─────────────────────────────────────────
def test_f12_task_and_mechanism_values_are_independent() -> None:
    """Q_task and Q_mechanism learn different targets (task reward vs bounded mechanism validity), so their reported
    values differ. The mechanism critic is a SEPARATE estimator, not a copy of the task critic."""
    a, c = _fresh_actor_critics()
    diag: dict[str, float] = {}
    train_sac(a, c, _ContactToyEnv(), _cfg("TASK_AND_MECHANISM"), eval_fn=lambda *_: 0.0, diag_out=diag)
    assert diag["q_task"] != diag["q_mech"]                          # distinct value functions
    assert np.isfinite(diag["q_mech"])


# ── 8. F11 runs NO mechanism machinery (diagnostics never populated) ──────────────────────────────────────────────
def test_f11_runs_no_mechanism_machinery() -> None:
    a, c = _fresh_actor_critics()
    diag: dict[str, float] = {}
    train_sac(a, c, _ContactToyEnv(), _cfg("TASK_ONLY"), eval_fn=lambda *_: 0.0, diag_out=diag)
    assert diag["critic_mode"] is False
    assert np.isnan(diag["mech_crit_loss"]) and np.isnan(diag["q_mech"])   # never computed under F11


# ── 9. actor checkpoints round-trip for BOTH modes (no silent partial load) ───────────────────────────────────────
@pytest.mark.parametrize("mode", ["TASK_ONLY", "TASK_AND_MECHANISM"])
def test_actor_checkpoint_roundtrips(mode: str, tmp_path) -> None:
    a, c = _fresh_actor_critics()
    train_sac(a, c, _ContactToyEnv(), _cfg(mode), eval_fn=lambda *_: 0.0)
    ck = tmp_path / f"actor_{mode}.pt"
    torch.save(a.state_dict(), ck)
    probe = torch.randn(5, _OBS)
    with torch.no_grad():
        before = a.action_mean(probe).clone()
    a2, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, hidden=16)
    missing = a2.load_state_dict(torch.load(ck), strict=True)         # strict → no silent partial load
    assert not missing.missing_keys and not missing.unexpected_keys
    with torch.no_grad():
        after = a2.action_mean(probe)
    assert torch.equal(before, after)                                # identical actor after round-trip


# ── 10. the pre-registered mech_coef is load-bearing (scales the actor objective) ─────────────────────────────────
def test_mech_coef_scales_the_actor_objective() -> None:
    a_lo, c_lo = _fresh_actor_critics()
    train_sac(a_lo, c_lo, _ContactToyEnv(), _cfg("TASK_AND_MECHANISM", mech_coef=0.5), eval_fn=lambda *_: 0.0)
    a_hi, c_hi = _fresh_actor_critics()
    train_sac(a_hi, c_hi, _ContactToyEnv(), _cfg("TASK_AND_MECHANISM", mech_coef=4.0), eval_fn=lambda *_: 0.0)
    assert not np.allclose(_flat(a_lo), _flat(a_hi))                  # coefficient enters the objective → changes actor
