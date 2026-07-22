"""COIN §10 reproduction audit test: every PRESENT frozen artifact reproduces on both the historical (legacy) and the
canonical v2 runtime (step-zero action delta 0.0), and the absent artifacts are reported ARTIFACT_NOT_PRESENT rather
than invented.
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.reproduction_audit import run


def test_all_present_artifacts_reproduce_on_both_runtimes():
    r = run()
    assert r["all_present_reproduce"], [row for row in r["rows"] if row["status"] == "COMPAT_DELTA"]


def test_absent_artifacts_are_reported_not_invented():
    r = run()
    # the quarantined full-action + missing relay-bridge artifacts must be flagged, never fabricated.
    assert set(r["absent"]) == {"1_relay_bridge", "6_corrected_bridge", "7_full_action_BC"}
    for row in r["rows"]:
        if row["status"] == "ARTIFACT_NOT_PRESENT":
            assert "ledger_ref" in row and "expected_path" in row


def test_graph_state_reproductions_share_the_semantic_fingerprint():
    r = run()
    for row in r["rows"]:
        if row["status"] == "REPRODUCED_BOTH_RUNTIMES" and "graph_fp_match" in row:
            assert row["graph_fp_match"], f"{row['item']}: v2 graph fingerprint != legacy"
