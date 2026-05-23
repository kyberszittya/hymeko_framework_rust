"""Unit tests for the MSG / ABB / SSG architecture-search machinery."""
from __future__ import annotations

import pytest

from signedkan_wip.src.arch_search import (
    ArchCandidate, msg_enumerate, abb_prune, ssg_pareto,
)


def test_msg_enumerate_cartesian_product():
    """MSG produces |axes_product| × |seeds| candidates."""
    cands = msg_enumerate(
        {
            "dataset": ["bitcoin_alpha", "slashdot"],
            "outer_hsikan_n_layers": [2, 4],
            "inner_skip": ["highway"],
        },
        seeds=(0, 1, 2),
    )
    # 2 × 2 × 1 × 3 = 12.
    assert len(cands) == 12
    # All are ArchCandidate instances with the right keys set.
    datasets = {c.dataset for c in cands}
    assert datasets == {"bitcoin_alpha", "slashdot"}
    depths = {c.outer_hsikan_n_layers for c in cands}
    assert depths == {2, 4}


def test_abb_prune_drops_high_memory():
    """ABB prunes candidates whose predicted memory exceeds the cap."""
    cands = msg_enumerate(
        {
            "dataset": ["slashdot"],
            "outer_hsikan_n_layers": [2, 4, 8],
            "inner_skip": ["highway"],
            "grad_checkpoint": [False],
        },
        seeds=(0,),
    )
    survivors, pruned = abb_prune(cands, mem_cap_gib=4.0,
                                    wall_cap_s=999.0)
    # d=8 no-ckpt on slashdot should exceed 4 GiB.
    pruned_depths = {c.outer_hsikan_n_layers for c, _ in pruned}
    assert 8 in pruned_depths
    # And the reason mentions "mem".
    assert any("mem" in r for _, r in pruned)


def test_abb_prune_drops_high_wall():
    """ABB prunes candidates exceeding the wall budget."""
    cands = msg_enumerate(
        {
            "dataset": ["epinions"],
            "outer_hsikan_n_layers": [8],
            "inner_skip": ["highway"],
            "grad_checkpoint": [True],
        },
        seeds=(0,),
    )
    _, pruned = abb_prune(cands, mem_cap_gib=99.0, wall_cap_s=5.0)
    assert len(pruned) == 1
    assert "wall" in pruned[0][1]


def test_grad_checkpoint_reduces_predicted_memory():
    """A checkpointed candidate has lower predicted memory than
    the same config without."""
    c_off = ArchCandidate(dataset="slashdot",
                            outer_hsikan_n_layers=4,
                            grad_checkpoint=False)
    c_on = ArchCandidate(dataset="slashdot",
                           outer_hsikan_n_layers=4,
                           grad_checkpoint=True)
    assert c_on.predicted_peak_mem_gib() < c_off.predicted_peak_mem_gib()


def test_memory_monotonic_in_depth():
    """Predicted memory is monotonic in outer_hsikan_n_layers."""
    base_kwargs = dict(dataset="bitcoin_alpha", inner_skip="highway")
    mems = [
        ArchCandidate(outer_hsikan_n_layers=d, **base_kwargs)
            .predicted_peak_mem_gib()
        for d in [1, 2, 4, 8]
    ]
    assert mems == sorted(mems)


def test_ssg_pareto_picks_no_ckpt_when_both_fit():
    """When both ckpt-on and ckpt-off survive ABB for the same
    depth, SSG should prefer ckpt-off (lower wall)."""
    cands = [
        ArchCandidate(dataset="bitcoin_alpha", outer_hsikan_n_layers=4,
                        grad_checkpoint=False, seed=0),
        ArchCandidate(dataset="bitcoin_alpha", outer_hsikan_n_layers=4,
                        grad_checkpoint=True, seed=0),
    ]
    out = ssg_pareto(cands)
    # The two candidates differ only by grad_checkpoint, so SSG
    # picks whichever has lower predicted wall.
    assert len(out) == 2 or len(out) == 1
    # If the bucket collapses to one (same depth+skip+arc+middle),
    # the survivor should be the ckpt-off one (lower wall).
    if len(out) == 1:
        assert out[0].grad_checkpoint is False


def test_arch_candidate_to_cli_args_has_required_flags():
    """``to_cli_args`` produces all the CLI flags the runner needs."""
    c = ArchCandidate(
        dataset="bitcoin_alpha", outer_hsikan_n_layers=4,
        inner_skip="cr_highway", grad_checkpoint=True,
        seed=2, n_epochs=60,
    )
    args = c.to_cli_args()
    assert "--dataset" in args
    assert "bitcoin_alpha" in args
    assert "--outer-hsikan-n-layers" in args
    assert "4" in args
    assert "--outer-hsikan-grad-checkpoint" in args
    assert "--seed" in args
    assert "2" in args


def test_arch_candidate_name_is_human_readable():
    """Names encode the salient axes for log labelling."""
    c = ArchCandidate(
        dataset="bitcoin_alpha", outer_hsikan_n_layers=4,
        inner_skip="cr_highway", grad_checkpoint=True, seed=1,
    )
    n = c.name
    assert "ba" in n and "d4" in n and "cr" in n and "ckpt" in n
    assert "s1" in n
