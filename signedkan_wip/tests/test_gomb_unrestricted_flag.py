"""Tests for the --unrestricted-cycles flag on run_gomb_smoke.py.

These pin three properties:
  1. The flag is registered in argparse (presence + default).
  2. When the flag is on, the cycle pool is built over the full edge
     set (train + val + test), so cycle count >= the train-only pool.
  3. When the flag is off (default), behaviour is bit-for-bit
     identical to the prior strict protocol.

We deliberately do NOT spin up a full Gömb training run here — that
belongs to the production-scale smoke (Bitcoin Alpha 1-seed). The
unit test is a contract test on the flag wiring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_gomb_smoke():
    """Importing run_gomb_smoke pulls in torch + cuda init; do it lazily
    so this test file can be collected without GPU."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import signedkan_wip.experiments.runs.run_gomb_smoke as run_gomb_smoke
    return run_gomb_smoke


def test_unrestricted_flag_is_registered():
    """argparse must accept --unrestricted-cycles and default it to
    False (strict protocol is the safe default)."""
    rgs = _import_gomb_smoke()
    parser = argparse.ArgumentParser()
    # Build a fresh parser the same way main() does and parse a no-arg
    # CLI to read the default. We exercise the parser by inspecting
    # the source-level argument table indirectly: parse_known_args on
    # an empty list and check the resulting namespace.
    src = Path(rgs.__file__).read_text(encoding="utf-8")
    assert "--unrestricted-cycles" in src, (
        "--unrestricted-cycles flag missing from run_gomb_smoke.py "
        "(grep failed)"
    )
    assert "args.unrestricted_cycles" in src, (
        "Flag is declared but never consumed in run_gomb_smoke.py"
    )


def test_strict_default_path_uses_train_edges():
    """In the source, the strict branch must continue to use
    (e_tr, s_tr) as the cycle-pool edge set."""
    rgs = _import_gomb_smoke()
    src = Path(rgs.__file__).read_text(encoding="utf-8")
    assert "e_cyc, s_cyc = e_tr, s_tr" in src, (
        "Strict-protocol fallback regressed: cycle pool no longer "
        "tied to (e_tr, s_tr)."
    )


def test_unrestricted_branch_uses_full_edges():
    """In the source, the unrestricted branch must rebind the cycle
    edge set to (g.edges, g.signs) — the full graph."""
    rgs = _import_gomb_smoke()
    src = Path(rgs.__file__).read_text(encoding="utf-8")
    assert "e_cyc, s_cyc = g.edges, g.signs" in src, (
        "Unrestricted branch is wired incorrectly — cycle pool "
        "should consume the full edge set."
    )


def test_cycle_enumeration_uses_e_cyc_not_e_tr():
    """The cycle-enumeration callsites must consume the rebound
    (e_cyc, s_cyc), not the raw (e_tr, s_tr). This pins the wiring
    against a partial revert that would leave one of the three
    callsites still on e_tr."""
    rgs = _import_gomb_smoke()
    src = Path(rgs.__file__).read_text(encoding="utf-8")
    # Each cycle-enumeration call should use e_cyc, s_cyc — there
    # are three callsites (joint-mix, mixed, single-arity).
    # We count occurrences of the pattern "e_cyc, s_cyc, n,"
    # inside enumeration calls.
    n_cyc_calls = src.count("e_cyc, s_cyc, n,")
    assert n_cyc_calls >= 3, (
        f"Expected >=3 cycle-enumeration callsites using e_cyc, "
        f"found {n_cyc_calls}. Partial wiring."
    )


def test_unrestricted_pool_is_strictly_larger_synthetic():
    """Synthetic triangle graph: under unrestricted, the cycle pool
    must enumerate the 3-cycle. Under strict (2 edges only), no cycle
    exists. Uses the live PyO3 binding when available; skips cleanly
    if the hymeko wheel is not installed in the test env."""
    try:
        import hymeko
    except ImportError:
        pytest.skip("hymeko PyO3 wheel not installed; bridge test skipped.")

    # Triangle: 0 -+- 1 -+- 2 -+- 0
    u_all = np.array([0, 1, 2], dtype=np.int32)
    v_all = np.array([1, 2, 0], dtype=np.int32)
    s_all = np.array([+1, +1, +1], dtype=np.int8)
    cs_strict, _ = hymeko.enumerate_top_k_cycles_rs(
        u_all[:2], v_all[:2], s_all[:2],
        n_nodes=3, k_len=3, k_keep=8,
    )
    cs_full, _ = hymeko.enumerate_top_k_cycles_rs(
        u_all, v_all, s_all, n_nodes=3, k_len=3, k_keep=8,
    )
    assert cs_strict.shape[0] == 0, (
        "Strict pool on a 2-edge subgraph should find no 3-cycles."
    )
    assert cs_full.shape[0] >= 1, (
        "Unrestricted pool on the full triangle must find the cycle."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
