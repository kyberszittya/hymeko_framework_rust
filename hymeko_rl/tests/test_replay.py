"""ReplayBuffer — the SoA off-policy ring, incl. the asymmetric-CTDE (MADDPG) privileged (z, z') stream.

The privileged stream is additive: a ``priv_dim=0`` buffer is byte-for-byte the plain ``(s,a,r,s',d)`` ring,
and adding a priv stream must NOT perturb the base-transition draws — the bit-identity guard that keeps
cart-pole / SAC / joint runs unchanged.
"""
import numpy as np
import pytest
import torch

from hymeko_rl.train.replay import ReplayBuffer


def _fill(buf: ReplayBuffer, n: int, *, priv: bool = False) -> None:
    for i in range(n):
        o = np.full((3, 2), i, dtype=np.float32)
        a = np.full(4, -i, dtype=np.float32)
        if priv:
            buf.add(o, a, float(i), o + 1, bool(i % 2),
                    priv=np.full(5, i, np.float32), priv_next=np.full(5, i + 0.5, np.float32))
        else:
            buf.add(o, a, float(i), o + 1, bool(i % 2))


def test_nonpriv_sample_arity_and_shapes() -> None:
    buf = ReplayBuffer(50, (3, 2), 4)
    _fill(buf, 40)
    s, a, r, s2, d = buf.sample(8, generator=np.random.default_rng(0))    # exactly 5 tensors (unchanged contract)
    assert tuple(s.shape) == (8, 3, 2) and tuple(a.shape) == (8, 4)
    assert tuple(r.shape) == (8,) and tuple(d.shape) == (8,)


def test_priv_stream_does_not_perturb_base_draws() -> None:
    # Bit-identity guard: a priv_dim>0 buffer, sampled with the SAME generator seed, returns the IDENTICAL base
    # (s,a,r,s',d) as a plain buffer holding the same data — the extra arrays never touch the base RNG path.
    plain = ReplayBuffer(50, (3, 2), 4)
    priv = ReplayBuffer(50, (3, 2), 4, priv_dim=5)
    _fill(plain, 40)
    _fill(priv, 40, priv=True)
    base = plain.sample(8, generator=np.random.default_rng(123))
    withz = priv.sample_with_priv(8, generator=np.random.default_rng(123))
    for x, y in zip(base, withz[:5]):
        assert torch.equal(x, y)
    assert tuple(withz[5].shape) == (8, 5) and tuple(withz[6].shape) == (8, 5)


def test_priv_roundtrip_values() -> None:
    buf = ReplayBuffer(10, (1,), 1, priv_dim=5)
    for i in range(5):
        buf.add(np.zeros(1, np.float32), np.zeros(1, np.float32), 0.0, np.zeros(1, np.float32), False,
                priv=np.full(5, i, np.float32), priv_next=np.full(5, -i, np.float32))
    _s, _a, _r, _s2, _d, z, z2 = buf.sample_with_priv(5, generator=np.random.default_rng(1))
    for row_z, row_z2 in zip(z.numpy(), z2.numpy()):      # each stored z is full-of-k, its z' full-of-(-k)
        k = row_z[0]
        assert np.allclose(row_z, k) and np.allclose(row_z2, -k)


def test_priv_add_requires_priv_args() -> None:
    buf = ReplayBuffer(10, (1,), 1, priv_dim=5)
    with pytest.raises(ValueError):                       # priv_dim>0 buffer must be fed z on add()
        buf.add(np.zeros(1, np.float32), np.zeros(1, np.float32), 0.0, np.zeros(1, np.float32), False)


def test_sample_with_priv_requires_priv_buffer() -> None:
    buf = ReplayBuffer(10, (1,), 1)                       # priv_dim=0
    buf.add(np.zeros(1, np.float32), np.zeros(1, np.float32), 0.0, np.zeros(1, np.float32), False)
    with pytest.raises(ValueError):
        buf.sample_with_priv(1, generator=np.random.default_rng(0))


def test_add_batch_priv() -> None:
    buf = ReplayBuffer(20, (2,), 3, priv_dim=5)
    n = 6
    obs = np.zeros((n, 2), np.float32)
    act = np.zeros((n, 3), np.float32)
    rew = np.arange(n, dtype=np.float32)
    done = np.zeros(n, np.float32)
    z = np.tile(np.arange(5, dtype=np.float32), (n, 1))
    buf.add_batch(obs, act, rew, obs, done, priv=z, priv_next=z + 1)
    assert buf.size == n
    out = buf.sample_with_priv(n, generator=np.random.default_rng(0))
    assert tuple(out[5].shape) == (n, 5) and tuple(out[6].shape) == (n, 5)
