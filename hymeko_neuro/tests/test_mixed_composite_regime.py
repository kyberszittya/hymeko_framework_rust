"""Tests for the HSiKAN-mixed × protocol-axis Composite-regime pipeline.

Covers (CLAUDE.md §3 layers):
  * Unit   — unit→knob mapping, knob→env translation, budget gate.
  * Unit   — regime solve: Composite prunes the quaternion by-product
             that Canonical keeps (regression for the no-excess
             composition, and for the pre-fix empty-structure bug).
  * Integ. — end-to-end solve→map→env (dry-run, no GPU): the selected
             structure spans every mandatory protocol axis and yields a
             well-formed HSIKAN_* env patch.

The torch/GPU production-scale smoke is the standalone driver
(`run_hsikan_mixed_composite_smoke.py`), run under
`systemd-run --user -p MemoryMax=16G`; it is the performance gate and
asserts the RSS/wall budget itself (see the report).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hymeko_neuro.experiments.runs.run_hsikan_mixed_composite_smoke import (
    _parse_time_v_rss_gib,
    _repo_root,
    check_budget,
    main,
    solve_regime,
    structure_to_env,
)
from hymeko_neuro.experiments.hsikan_pgraph_mapping import merge_structure_knobs

SWEEP = Path("data/hsikan/sweep_msg_mixed_protocols.hymeko")

# The mandatory axes of the protocol sweep; a feasible ABB solution
# must select exactly one unit from each.
_AXIS_PREFIXES = ("struct_", "attn_", "gate_", "dm_", "model_h", "train_")


@pytest.fixture(scope="module")
def repo() -> Path:
    return _repo_root()


@pytest.fixture(scope="module")
def dump_built(repo: Path) -> None:
    """Ensure the pgraph dump binary is buildable/present; skip if not."""
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    if rel.exists() or dbg.exists():
        return
    try:
        subprocess.run(
            ["cargo", "build", "-p", "hymeko_pgraph", "--bin",
             "hymeko_pgraph_dump", "--release"],
            cwd=str(repo), check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:  # pragma: no cover
        pytest.skip(f"cannot build hymeko_pgraph_dump: {err}")


# ----------------------------------------------------------------------
# Unit — unit→knob mapping
# ----------------------------------------------------------------------
def test_merge_baseline_abb_selection():
    """The cost-minimal ABB unit set maps to the baseline protocol combo."""
    knobs = merge_structure_knobs(
        ["attn_none", "dm_off", "gate_scalar", "model_h8", "struct_mixed",
         "train_short"]
    )
    assert knobs == {
        "attention_kind": "none",
        "direct_messaging": False,
        "per_edge_gate": False,
        "hidden": 8,
        "mixed_tuples": "c3,c4,w2,w3",
        "n_epochs": 10,
    }


def test_merge_rich_selection():
    """A non-baseline selection maps to the non-trivial protocol knobs."""
    knobs = merge_structure_knobs(
        ["struct_mixed", "attn_dot", "gate_edge_cr", "dm_on", "model_h32",
         "train_long"]
    )
    assert knobs["attention_kind"] == "dot"
    assert knobs["per_edge_gate"] is True
    assert knobs["direct_messaging"] is True
    assert knobs["hidden"] == 32
    assert knobs["n_epochs"] == 60


def test_merge_collision_precedence():
    """Later unit on the same axis wins (merge contract)."""
    assert merge_structure_knobs(["attn_none", "attn_dot"])["attention_kind"] == "dot"


def test_merge_unknown_unit_raises():
    with pytest.raises(KeyError):
        merge_structure_knobs(["attn_none", "not_a_real_unit"])


# ----------------------------------------------------------------------
# Unit — knob→env translation
# ----------------------------------------------------------------------
def test_structure_to_env_booleans_and_strings():
    env, cli = structure_to_env(
        {
            "mixed_tuples": "c3,c4,w2,w3",
            "attention_kind": "quaternion",
            "per_edge_gate": True,
            "direct_messaging": False,
            "hidden": 16,
            "n_epochs": 60,
        }
    )
    assert env["HSIKAN_MIXED_TUPLES"] == "c3,c4,w2,w3"
    assert env["HSIKAN_ATTENTION_M_E"] == "quaternion"
    assert env["HSIKAN_PER_EDGE_GATE"] == "1"
    assert env["HSIKAN_DIRECT_MESSAGING"] == "0"
    # hidden / n_epochs are CLI args, not env.
    assert cli == {"hidden": 16, "n_epochs": 60}
    assert "HSIKAN_HIDDEN" not in env


def test_structure_to_env_partial():
    """Missing keys are simply absent — no spurious env vars."""
    env, cli = structure_to_env({"attention_kind": "none"})
    assert env == {"HSIKAN_ATTENTION_M_E": "none"}
    assert cli == {}


def test_structure_to_env_attention_inherits_topk_cap():
    """Enabling attention auto-sets the enumeration cap (contract
    preservation: attention disables cycle-batching and OOMs uncapped —
    measured 2026-05-27 on a 7.6 GiB GPU)."""
    for kind in ("dot", "quaternion"):
        env, _ = structure_to_env({"attention_kind": kind})
        assert env["HSIKAN_ATTENTION_M_E"] == kind
        assert env["HSIKAN_TOPK_MODE"] == "per_vertex"
        assert int(env["HSIKAN_TOPK_K"]) > 0


def test_structure_to_env_none_attention_no_cap():
    """attention=none must NOT impose a top-K cap (baseline path keeps
    the proven uncapped BA-mixed recipe)."""
    env, _ = structure_to_env({"attention_kind": "none"})
    assert "HSIKAN_TOPK_MODE" not in env
    assert "HSIKAN_TOPK_K" not in env


# ----------------------------------------------------------------------
# Unit — budget gate (pure function)
# ----------------------------------------------------------------------
def test_parse_time_v_rss():
    stderr = (
        "\tCommand being timed: \"python\"\n"
        "\tMaximum resident set size (kbytes): 1677721\n"
        "\tExit status: 0\n"
    )
    gib = _parse_time_v_rss_gib(stderr)
    assert abs(gib - 1677721 / (1024 * 1024)) < 1e-9


def test_parse_time_v_rss_absent_is_nan():
    import math
    assert math.isnan(_parse_time_v_rss_gib("no rss line here"))


def test_check_budget_pass():
    assert check_budget(auc=0.93, wall_s=120.0, peak_rss_gib=5.0,
                        mem_cap_gib=7.0, wall_cap_s=180.0) == []


@pytest.mark.parametrize(
    "auc,wall,rss,n_expected",
    [
        (0.93, 999.0, 5.0, 1),   # wall over
        (0.93, 10.0, 99.0, 1),   # rss over
        (1.5, 10.0, 5.0, 1),     # auc out of range
        (None, 10.0, 5.0, 1),    # auc non-numeric
        (1.5, 999.0, 99.0, 3),   # all three
    ],
)
def test_check_budget_violations(auc, wall, rss, n_expected):
    v = check_budget(auc=auc, wall_s=wall, peak_rss_gib=rss,
                    mem_cap_gib=7.0, wall_cap_s=180.0)
    assert len(v) == n_expected


# ----------------------------------------------------------------------
# Unit/regression — regime semantics (Composite prunes the by-product)
# ----------------------------------------------------------------------
def _solve(repo: Path, regime: str) -> dict:
    return solve_regime(repo, repo / SWEEP, regime)


def test_canonical_keeps_quaternion(repo, dump_built):
    """Canonical maximal structure admits the wasteful quaternion unit."""
    data = _solve(repo, "canonical")
    assert "attn_quaternion" in data["msg_units"]


def test_composite_prunes_quaternion(repo, dump_built):
    """Composite (canonical+no-excess) drops the by-product producer."""
    data = _solve(repo, "canonical+no-excess")
    assert "attn_quaternion" not in data["msg_units"]
    # No-excess alone must agree (Composite is their intersection).
    ne = _solve(repo, "no-excess")
    assert "attn_quaternion" not in ne["msg_units"]


def test_composite_abb_nonempty_and_spans_all_axes(repo, dump_built):
    """The empty-structure regression guard + one-unit-per-axis check."""
    data = _solve(repo, "canonical+no-excess")
    units = data["abb"]["units"]
    assert units, "Composite ABB returned an empty structure"
    for prefix in _AXIS_PREFIXES:
        selected = [u for u in units if u.startswith(prefix)]
        assert len(selected) == 1, (
            f"axis {prefix!r}: expected exactly one selected unit, got {selected}"
        )


def test_solve_raises_on_unknown_regime(repo, dump_built):
    with pytest.raises(RuntimeError):
        _solve(repo, "not-a-regime")


# ----------------------------------------------------------------------
# Integration — end-to-end solve→map→env (dry-run, no GPU)
# ----------------------------------------------------------------------
def test_dry_run_end_to_end(repo, dump_built, capsys):
    rc = main(["--dry-run", "--regime", "canonical+no-excess"])
    assert rc == 0
    prov = json.loads(capsys.readouterr().out)
    assert prov["regime"] == "canonical+no-excess"
    assert prov["abb_units"], "no ABB units selected"
    # The fixed mixed family is always forwarded.
    assert prov["env_patch"]["HSIKAN_MIXED_TUPLES"] == "c3,c4,w2,w3"
    # Every mandatory axis resolved into the merged structure.
    s = prov["structure"]
    assert s["mixed_tuples"] == "c3,c4,w2,w3"
    assert "attention_kind" in s and "per_edge_gate" in s and "direct_messaging" in s
    assert prov["hidden"] >= 1 and prov["n_epochs"] >= 1
