"""FEEDBACK_CHUNK_WARMSTART_V2 contract tests (no TD3): frozen prefix weights + prefix-weighted loss emphasis; dense
feedback dataset provenance (planner vs pi_0 fallback)."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_chunk_td3 import ACT_DIM, CHUNK_DIM, K, M
from hymeko_rl.coin_delivery.coin_feedback_chunk_v2 import PREFIX_WEIGHTS, prefix_weighted_loss

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


def test_prefix_weights_frozen_and_emphasize_executed_prefix():
    assert np.allclose(PREFIX_WEIGHTS, [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1])
    assert len(PREFIX_WEIGHTS) == K and PREFIX_WEIGHTS[0] == PREFIX_WEIGHTS[M - 1] == 1.0
    assert PREFIX_WEIGHTS[:M].sum() > PREFIX_WEIGHTS[M:].sum()      # executed prefix is load-bearing


def test_prefix_weighted_loss_penalizes_first_actions_more():
    tgt = torch.zeros(1, CHUNK_DIM)
    err_first = torch.zeros(1, K, ACT_DIM); err_first[0, 0] = 1.0    # error only on action 0 (weight 1.0)
    err_last = torch.zeros(1, K, ACT_DIM); err_last[0, K - 1] = 1.0  # error only on action K-1 (weight 0.1)
    lf = prefix_weighted_loss(err_first.view(1, -1), tgt); ll = prefix_weighted_loss(err_last.view(1, -1), tgt)
    assert float(lf) > float(ll)                                    # same magnitude error weighted higher on the prefix


@pytest.mark.slow
def test_feedback_dataset_dense_and_labeled():
    from collections import Counter

    from hymeko_rl.coin_delivery.coin_feedback_chunk_v2 import build_feedback_dataset
    from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank
    from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    starts = build_late_start_bank(pi0, range(6000, 6030), per_family=1)
    X, Y, prov, stats = build_feedback_dataset(pi0, starts, horizon=30)
    assert X.shape[1] == 62 and Y.shape[1] == CHUNK_DIM and len(X) == stats["n_examples"] > len(starts)  # dense: >1/traj
    assert set(Counter(prov)) <= {"planner", "pi0_fallback"}
    assert stats["n_planner_improving"] + stats["n_pi0_fallback"] == stats["n_examples"]
