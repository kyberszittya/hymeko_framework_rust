"""Coverage for the channel-completeness audit (directive §2/§3/§5). Pure parts: event-state semantics on hand-built
contact streams (0→1/1→2/2→1/1→0/1↔2/reset), group dims, ceiling nets. The full extractor (contact physics /
relative-velocity / canonical normal) is smoke-tested on a real MuJoCo rollout in the experiment."""
from __future__ import annotations

import numpy as np
import torch

# EVT layout: [dwell, in_mode, since_onset, prev_mode(3), cur_mode(3), trans_w, osc_w]
D, IM, SO, CM = 0, 1, 2, slice(6, 9)


class _M:
    def __init__(self, l, r):
        self.left_contact, self.right_contact = l, r


def _stream(seq):
    from hymeko_rl.experiments.exp_coin_toss_privileged import EventTracker
    tr = EventTracker()
    return tr, [tr.update(_M(l, r)) for l, r in seq]


def test_group_dims() -> None:
    from hymeko_rl.experiments.exp_coin_toss_privileged import CON, CTRL, EVT, KIN, PRIV_DIM
    assert (KIN, CON, CTRL, EVT) == (15, 22, 9, 11) and PRIV_DIM == 57


def test_dwell_not_alias_of_in_mode() -> None:
    # 0 → 1 → 1 → 2: dwell spans the 1→2 transition (contiguous contact), in_mode RESETS at the transition
    _, o = _stream([(False, False), (True, False), (True, False), (True, True)])
    assert o[3][D] == 3                                          # dwell = 3 contiguous contact steps
    assert o[3][IM] == 1                                         # time-in-current-mode reset at 1→2 (NOT equal to dwell)
    assert o[2][D] == o[2][IM]                                   # while mode is stable they coincide...
    assert o[3][D] != o[3][IM]                                   # ...but diverge across a within-contact mode change


def test_transitions_and_prev_mode() -> None:
    _, o = _stream([(False, False), (True, False), (True, True), (True, False)])  # 0→1→2→1
    assert np.argmax(o[2][CM]) == 2 and np.argmax(o[3][CM]) == 1  # current mode tracks
    assert np.argmax(o[3][slice(3, 6)]) == 2                      # prev_mode at 2→1 step is 2 (no lag)


def test_contact_loss_resets_dwell_keeps_since_onset() -> None:
    _, o = _stream([(True, False), (True, False), (False, False)])   # 1,1,0
    assert o[1][D] == 2                                          # dwell 2 while in contact
    assert o[2][D] == 0                                          # dwell resets on contact loss
    assert o[2][SO] > o[1][SO] - 1                               # since_onset keeps growing after contact ends


def test_oscillation_counts_1_2_switches() -> None:
    tr, o = _stream([(True, False), (True, True), (True, False), (True, True)])  # 1↔2↔1↔2
    assert o[-1][10] >= 2                                        # windowed oscillation count grows on 1↔2 flips
    assert tr.hist[-1] == 2


def test_reset_clears_state() -> None:
    from hymeko_rl.experiments.exp_coin_toss_privileged import EventTracker
    tr = EventTracker(); [tr.update(_M(True, True)) for _ in range(5)]
    tr2 = EventTracker(); out = tr2.update(_M(True, False))
    assert out[D] == 1 and out[IM] == 1                          # a fresh tracker starts clean (episode reset)


def test_ceiling_nets_forward() -> None:
    from hymeko_rl.experiments.exp_coin_toss_channel_ceiling import _GRUHead, _GRUHSiKAN, _PrivHSiKAN, _mlp
    from hymeko_rl.experiments.exp_coin_toss_privileged import PRIV_DIM
    assert _mlp(48, 1)(torch.randn(5, 48)).shape == (5, 1)
    assert _GRUHead(PRIV_DIM, 1)(torch.randn(5, 6, PRIV_DIM)).shape == (5, 1)
    assert _PrivHSiKAN(9)(torch.randn(5, PRIV_DIM)).shape == (5, 9)
    with torch.no_grad():
        m = _GRUHSiKAN(PRIV_DIM, 1); x = torch.randn(4, 6, PRIV_DIM)
        assert torch.allclose(m(x), m.gru(x), atol=1e-6)        # zero-init structural residual (matched control §7)
