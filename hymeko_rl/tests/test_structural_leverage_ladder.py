"""Stage 2 tests — DeepSets neutral control + the richness ladder (machinery; verdicts need full epochs)."""
from __future__ import annotations

import torch

from hymeko_rl.experiments.exp_structural_leverage_ladder import (
    DeepSetsBackbone,
    LadderConfig,
    _build,
    _match_width,
    build_deepsets,
    plot_ladder,
    plot_neutral_control,
    run_ladder,
    run_neutral_control,
)
from hymeko_rl.experiments.structural_probe import build_toy_graph

_SMOKE = dict(seeds=2, epochs=12, n_train=64, n_test=128)


def test_deepsets_forward_shape_and_finite() -> None:
    bb = DeepSetsBackbone(1, 16)
    x = torch.randn(4, 7, 1)
    out = bb(x)
    assert out.shape == (4, 16)
    assert bool(torch.isfinite(out).all())


def test_deepsets_is_permutation_invariant() -> None:
    # the defining property: no message passing → node order cannot matter (so it CANNOT compute B²x).
    bb = DeepSetsBackbone(1, 16)
    x = torch.randn(4, 7, 1)
    perm = torch.randperm(7)
    assert torch.allclose(bb(x), bb(x[:, perm, :]), atol=1e-6)


def test_deepsets_ignores_graph_structure() -> None:
    # _build("deepsets", ...) must not depend on the graph (structure-blind per-node) — same params on any graph.
    hg = build_toy_graph()
    p1 = _build("deepsets", hg, 32, 2).n_params()
    p2 = build_deepsets(32, n_layers=2).n_params()
    assert p1 == p2


def test_width_match_is_close_to_target() -> None:
    hg = build_toy_graph()
    target = _build("hsikan", hg, 32, 2).n_params()
    for kind in ("mlp", "deepsets"):
        w = _match_width(kind, hg, target, 2)
        got = _build(kind, hg, w, 2).n_params()
        assert abs(got - target) / target < 0.10          # within 10% of HSiKAN params


def test_neutral_control_smoke() -> None:
    r = run_neutral_control(LadderConfig(**_SMOKE))
    assert r["verdict"]["label"] in ("SUPPORTED", "WEAKENED", "FALSIFIED")
    for target in ("structural", "bag"):
        for key in ("hsikan_true", "hsikan_scrambled", "deepsets", "mlp"):
            assert f"{target}/{key}" in r["cells"]
    # params matched across the three architectures (within the search grid).
    assert r["params"]["hsikan"] > 0


def test_ladder_smoke() -> None:
    r = run_ladder(LadderConfig(**_SMOKE, lengths=(4, 8)))
    assert [row["n_nodes"] for row in r["rows"]] == [4, 8]
    assert r["verdict"]["label"] in ("SUPPORTED", "WEAKENED", "FALSIFIED")
    assert len(r["verdict"]["structure_benefit_by_length"]) == 2
    assert len(r["verdict"]["msgpass_benefit_by_length"]) == 2


def test_plots_write_png(tmp_path) -> None:
    nc = run_neutral_control(LadderConfig(**_SMOKE))
    ld = run_ladder(LadderConfig(**_SMOKE, lengths=(4, 8)))
    assert plot_neutral_control(nc, tmp_path / "nc").exists()
    assert plot_ladder(ld, tmp_path / "ld").exists()
