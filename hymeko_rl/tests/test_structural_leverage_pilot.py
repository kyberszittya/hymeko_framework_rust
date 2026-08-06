"""Stage 1 tests — the structural-leverage pilot: verdict logic (pure) + a cheap end-to-end smoke."""
from __future__ import annotations

from hymeko_rl.experiments.exp_structural_leverage_pilot import (
    PilotConfig,
    _mlp_gap_framing,
    _scramble_framing,
    build_toy_graph,
    plot_pilot,
    run_pilot,
)
from hymeko_rl.experiments.incidence_scramble import scramble_signed_incidence
from hymeko_rl.experiments.structural_probe import build_model


def _pt(*, struct_ratio_o: float, struct_ratio_s: float, struct_collapse: float | None,
        bag_ratio_o: float, bag_ratio_s: float, struct_benefit: float = 3.0,
        bag_benefit: float = 1.0) -> dict:
    return {
        "structural": {"ratio_mlp_over_hsikan_original": struct_ratio_o,
                       "ratio_mlp_over_hsikan_scrambled": struct_ratio_s, "collapse_frac": struct_collapse,
                       "structure_benefit": struct_benefit},
        "bag": {"ratio_mlp_over_hsikan_original": bag_ratio_o,
                "ratio_mlp_over_hsikan_scrambled": bag_ratio_s, "collapse_frac": None,
                "structure_benefit": bag_benefit},
    }


def _rob(*, all_degrade: bool = True) -> dict:
    return {"all_scrambles_degrade": all_degrade}


# ── MLP-gap framing (pre-registered, confounded) ────────────────────────────────────────────────────────
def test_mlp_gap_framing_supported() -> None:
    v = _mlp_gap_framing(_pt(struct_ratio_o=2.5, struct_ratio_s=1.2, struct_collapse=0.7,
                             bag_ratio_o=1.0, bag_ratio_s=1.0))
    assert v["label"] == "SUPPORTED"


def test_mlp_gap_framing_falsified_on_fake_flat_advantage() -> None:
    # HSiKAN "wins" the flat control too (2.0×) → the naive framing correctly flags a non-structural win.
    v = _mlp_gap_framing(_pt(struct_ratio_o=2.5, struct_ratio_s=1.2, struct_collapse=0.7,
                             bag_ratio_o=2.0, bag_ratio_s=2.0))
    assert v["checks"]["flat_control_tie"] is False
    assert v["label"] in ("WEAKENED", "FALSIFIED")


# ── scramble framing (architecture-controlled, primary) ─────────────────────────────────────────────────
def test_scramble_framing_supported() -> None:
    # structure helps 3× on structural, is neutral (1×) on bag, and every scramble degrades → H2 supported.
    v = _scramble_framing(_pt(struct_ratio_o=1.4, struct_ratio_s=0.5, struct_collapse=4.0,
                              bag_ratio_o=18.0, bag_ratio_s=17.0, struct_benefit=3.0, bag_benefit=1.05),
                          _rob(all_degrade=True))
    assert v["label"] == "SUPPORTED"
    assert all(v["checks"].values())


def test_scramble_framing_falsified_when_structure_irrelevant() -> None:
    # structure barely helps (1.1×) even on the structural target → H2 not supported.
    v = _scramble_framing(_pt(struct_ratio_o=1.4, struct_ratio_s=1.3, struct_collapse=0.1,
                              bag_ratio_o=1.0, bag_ratio_s=1.0, struct_benefit=1.1, bag_benefit=1.0),
                          _rob(all_degrade=True))
    assert v["checks"]["structure_helps_on_structural"] is False
    assert v["label"] == "FALSIFIED"


def test_scramble_framing_weakened_when_not_robust() -> None:
    # structure helps on structural but not every scramble degrades → weakened, not fully supported.
    v = _scramble_framing(_pt(struct_ratio_o=1.4, struct_ratio_s=0.5, struct_collapse=4.0,
                              bag_ratio_o=18.0, bag_ratio_s=17.0, struct_benefit=3.0, bag_benefit=1.05),
                          _rob(all_degrade=False))
    assert v["label"] == "WEAKENED"


def test_mlp_baseline_is_structure_blind() -> None:
    # the MLP flattens the per-node obs, so its param count is identical on the true vs scrambled graph
    # (it never uses the adjacency) — the shared-baseline assumption the pilot relies on.
    hg = build_toy_graph()
    hg_scr = scramble_signed_incidence(hg, seed=0)
    p_true = build_model("mlp", hg, 40).n_params()
    p_scr = build_model("mlp", hg_scr, 40).n_params()
    assert p_true == p_scr


def test_pilot_smoke_runs() -> None:
    # cheap end-to-end: 2 seeds, 15 epochs — checks wiring, the params-match assert, and report shape.
    report = run_pilot(PilotConfig(seeds=2, epochs=15, n_train=64, n_test=128))
    assert set(report["per_target"]) == {"structural", "bag"}
    assert report["verdict"]["primary_label"] in ("SUPPORTED", "WEAKENED", "FALSIFIED")
    assert report["verdict"]["mlp_gap_framing"]["label"] in ("SUPPORTED", "WEAKENED", "FALSIFIED")
    # HSiKAN params identical across conditions (the assert inside run_pilot would have fired otherwise).
    assert report["cells"]["structural/hsikan_original"]["n_params"] == \
        report["cells"]["structural/hsikan_scrambled"]["n_params"]
    # scramble actually changed the graph (else the H2 arm is a no-op).
    assert report["scramble_stats"]["n_edges_changed"] >= 1
    assert report["scramble_stats"]["signed_degree_preserved"] is True
    # robustness sweep present and structurally sound.
    assert report["scramble_robustness"]["n_scramble_seeds"] == 5
    assert len(report["scramble_robustness"]["hsikan_scrambled_per_seed"]) == 5


def test_pilot_plot_writes_png(tmp_path) -> None:
    report = run_pilot(PilotConfig(seeds=2, epochs=15, n_train=64, n_test=128))
    out = plot_pilot(report, tmp_path / "pilot")
    assert out.exists() and out.suffix == ".png"
