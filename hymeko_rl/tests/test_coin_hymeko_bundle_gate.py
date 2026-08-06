"""HYMEKO_COIN_SPEC_BUNDLE_RUNTIME gate (Option B §9) — the whole executable Coin spec bundle is load-bearing and
self-consistent through the canonical outer runtime, with one combined bundle hash. This is the umbrella gate over the
component sentinels (control / scene / graph / checkpoint / alignment); it fails loudly if any spec becomes non-load-
bearing or a Python fallback creeps back in.
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.bundle_gate import run


def test_bundle_gate_passes():
    r = run()
    failed = [k for k, v in r["checks"].items() if not v]
    assert not failed, f"bundle gate checks failed: {failed}"
    assert r["verdict"] == "HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS"


def test_bundle_exposes_a_single_combined_hash():
    r = run()
    assert r["combined_bundle_hash"] and len(r["combined_bundle_hash"]) == 16
    # training and evaluation bind to the SAME hash: alignment manifest hash == bundle-hashes combined hash.
    assert r["checks"]["single_combined_hash"]


def test_canonical_mode_has_no_python_fallback():
    r = run()
    assert r["checks"]["no_python_fallback"], "canonical mode must hard-fail on a missing spec value (no fallback)"


def test_checkpoint_graph_fingerprint_matches_the_bundle():
    r = run()
    assert r["checks"]["checkpoint_graph_compatible"]
    assert r["checks"]["all_checkpoints_compatible"]
