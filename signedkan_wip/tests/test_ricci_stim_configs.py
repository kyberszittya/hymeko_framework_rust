"""Tests pinning the Ricci-Stim ablation-config table.

Specifically encodes the 2026-05-17 operational decision:
Config F (Bochner full, no SDRF) is the canonical recommended
config; Config E (with SDRF) is preserved for measurement
reproduction but is *not* the operational baseline.

Reference: reports/2026-05-16-gomb-soma-hodge-vectorize.md §11.
"""
from __future__ import annotations

from signedkan_wip.experiments.run_ricci_stim_cluttered_mnist import (
    CONFIGS,
)


def test_config_f_exists_and_is_no_sdrf():
    """Config F = the canonical operational config = Bochner full, no SDRF.
    Drift on this is a regression — file an issue, do not silently flip."""
    assert "F" in CONFIGS, (
        "Config F must exist as the canonical no-SDRF recommendation. "
        "See reports/2026-05-16-gomb-soma-hodge-vectorize.md §11 for "
        "the measurement that motivated dropping SDRF (Config D 0.174 "
        "vs Config E 0.141 mAP50_proxy)."
    )
    f = CONFIGS["F"]
    assert f["use_sdrf"] is False
    assert f["bochner_alpha"] == 0.1
    assert f["bochner_beta"] == 0.1


def test_config_e_preserves_sdrf_for_reproducibility():
    """Config E preserves the legacy SDRF-on definition. We keep it so
    older orchestrators / measurements stay byte-identical reproducible."""
    assert "E" in CONFIGS
    assert CONFIGS["E"]["use_sdrf"] is True


def test_config_f_and_config_d_have_identical_hyperparams():
    """Config F is materially the same as Config D — the difference is
    *semantic role* (F = canonical operational, D = ablation cell). If
    Config D is ever changed in the ablation grid, F must follow."""
    assert CONFIGS["F"] == CONFIGS["D"]


def test_config_a_is_pure_baseline():
    """Config A is the no-Hodge no-Ricci no-SDRF control. Must stay
    structurally bare so the orthogonal-ablation table interprets clean.
    """
    a = CONFIGS["A"]
    assert a["bochner_alpha"] == 0.0
    assert a["bochner_beta"] == 0.0
    assert a["use_sdrf"] is False


def test_ablation_grid_covers_three_axes_cleanly():
    """The ablation grid covers Hodge (α), Ricci (β), SDRF as orthogonal
    on/off axes:
        A: 0,0,off   — baseline
        B: 0.1,0,off — Hodge only
        C: 0,0.1,off — Ricci only
        D: 0.1,0.1,off — Bochner full (Hodge × Ricci)
        E: 0.1,0.1,on  — Bochner + SDRF (the 2026-05-15 planned headline)
        F: 0.1,0.1,off — Bochner only, 2026-05-17 canonical
    """
    grid = {(c["bochner_alpha"], c["bochner_beta"], c["use_sdrf"]): name
            for name, c in CONFIGS.items()}
    # Each (α, β, sdrf) cell maps to at least one config (some duplicate
    # under intent — D and F are intentional copies).
    assert (0.0, 0.0, False) in grid
    assert (0.1, 0.0, False) in grid
    assert (0.0, 0.1, False) in grid
    assert (0.1, 0.1, False) in grid
    assert (0.1, 0.1, True) in grid
