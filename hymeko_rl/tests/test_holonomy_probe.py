"""The holonomy-discriminator (T1 parity): the label is a clean Z2 cycle holonomy, read by the multiplicative
transport (rotor) but NOT by additive message-passing (B^N / HSiKAN), the MLP, or a linear probe."""
import pytest
import torch

from hymeko_rl.experiments.holonomy_probe import make_parity_data, run_holonomy_probe


def test_parity_data_is_the_cycle_holonomy() -> None:
    s, y = make_parity_data(8, 100, seed=0)
    assert s.shape == (100, 8) and y.shape == (100,)
    assert set(s.unique().tolist()) <= {-1.0, 1.0}
    assert torch.allclose(y, (s.prod(dim=1) > 0).float())       # label == product-of-signs > 0


def test_ring_size_floor_raises() -> None:
    with pytest.raises(ValueError):
        make_parity_data(2, 10, seed=0)


def test_transport_reads_holonomy_additive_does_not() -> None:
    # the decisive verification (small/fast): transport solves; additive (B^N/HSiKAN) + linear are at chance.
    arms = run_holonomy_probe(ring_size=12, seeds=2, epochs=300, n_train=400, n_test=400)["arms"]
    assert arms["transport"]["acc"] > 0.9                       # the rotor transport reads the holonomy
    assert arms["additive"]["acc"] < 0.7                        # additive B^N (HSiKAN mechanism) cannot
    assert arms["linear"]["acc"] < 0.65                         # confound guard: not linearly separable
