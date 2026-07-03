"""VectorizedEnv (lock-step N-env rollout) + ReplayBuffer.add_batch — the vectorised-rollout primitives."""
import numpy as np

from hymeko_rl.env.vector_env import VectorizedEnv
from hymeko_rl.train.replay import ReplayBuffer


class _StubEnv:
    """A tiny Gymnasium-shaped env: random obs, terminates after ``horizon`` steps."""

    def __init__(self, dim: int = 2, horizon: int = 3) -> None:
        self.dim, self.horizon, self.t = dim, horizon, 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int = 0):
        self.t = 0
        self._rng = np.random.default_rng(seed)
        return self._rng.standard_normal(self.dim).astype(np.float32), {}

    def step(self, action):
        self.t += 1
        obs = self._rng.standard_normal(self.dim).astype(np.float32)
        return obs, 1.0, self.t >= self.horizon, False, {}


def test_vectorized_env_shapes_and_autoreset() -> None:
    ve = VectorizedEnv(lambda: _StubEnv(dim=2, horizon=3), n_envs=4, seed=0)
    assert ve.obs.shape == (4, 2)
    acts = np.zeros((4, 1), dtype=np.float32)
    term = np.zeros(4)
    for _ in range(3):                                   # horizon 3 -> step 3 terminates all
        next_obs, rew, term, trunc = ve.step(acts)
        assert next_obs.shape == (4, 2) and rew.shape == (4,) and term.shape == (4,) and trunc.shape == (4,)
    assert term.all()                                    # all done at the horizon
    assert ve.obs.shape == (4, 2)                        # self.obs advanced to the (live) reset obs


def test_vectorized_env_seed_determinism() -> None:
    a = VectorizedEnv(lambda: _StubEnv(), 3, seed=7).obs
    b = VectorizedEnv(lambda: _StubEnv(), 3, seed=7).obs
    assert np.allclose(a, b)
    c = VectorizedEnv(lambda: _StubEnv(), 3, seed=8).obs
    assert not np.allclose(a, c)                          # different seed -> different rollout


def test_replay_add_batch_parity() -> None:
    """N add_batch == N sequential add (same ring state)."""
    cap, n = 100, 8
    rb1, rb2 = ReplayBuffer(cap, (3,), 2), ReplayBuffer(cap, (3,), 2)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, 3)).astype(np.float32)
    act = rng.standard_normal((n, 2)).astype(np.float32)
    rew = rng.standard_normal(n).astype(np.float32)
    nob = rng.standard_normal((n, 3)).astype(np.float32)
    don = (rng.random(n) < 0.3).astype(np.float32)
    for i in range(n):
        rb1.add(obs[i], act[i], float(rew[i]), nob[i], bool(don[i]))
    rb2.add_batch(obs, act, rew, nob, don)
    assert rb1.size == rb2.size == n and rb1._ptr == rb2._ptr
    assert np.allclose(rb1._obs, rb2._obs) and np.allclose(rb1._act, rb2._act)
    assert np.allclose(rb1._rew, rb2._rew) and np.allclose(rb1._done, rb2._done)


def test_replay_add_batch_wraparound() -> None:
    """A batch crossing the ring boundary matches sequential adds."""
    rb1, rb2 = ReplayBuffer(10, (2,), 1), ReplayBuffer(10, (2,), 1)
    z2, z1 = np.zeros(2, np.float32), np.zeros(1, np.float32)
    for buf in (rb1, rb2):
        for _ in range(8):
            buf.add(z2, z1, 0.0, z2, False)              # fill to ptr=8
    rng = np.random.default_rng(1)
    n = 5
    obs = rng.standard_normal((n, 2)).astype(np.float32)
    act = rng.standard_normal((n, 1)).astype(np.float32)
    rew = rng.standard_normal(n).astype(np.float32)
    for i in range(n):
        rb1.add(obs[i], act[i], float(rew[i]), obs[i], False)
    rb2.add_batch(obs, act, rew, obs, np.zeros(n, np.float32))
    assert rb1._ptr == rb2._ptr == 3 and rb1.size == rb2.size == 10   # wrapped past capacity
    assert np.allclose(rb1._obs, rb2._obs) and np.allclose(rb1._rew, rb2._rew)
