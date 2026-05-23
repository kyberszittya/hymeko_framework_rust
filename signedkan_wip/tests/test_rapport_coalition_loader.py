"""Round-trip tests for the rapport coalition .hymeko loader.

Verifies that triad_hri.hymeko parses to a Coalition struct with
the expected agents, relations, cycle, policies, and that all
cross-reference integrity checks (relation src/dst → agent;
cycle members → relation) hold.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAD_PATH = REPO_ROOT / "data" / "coalitions" / "triad_hri.hymeko"


@pytest.fixture(scope="module")
def triad():
    from signedkan_wip.src.rapport.coalition import load_coalition
    return load_coalition(TRIAD_PATH)


def test_triad_has_three_agents(triad):
    assert sorted(triad.agent_names()) == ["alice", "bob", "r1"]
    assert triad.agents["alice"].kind == "human"
    assert triad.agents["bob"].kind == "human"
    assert triad.agents["r1"].kind == "robot"


def test_triad_has_three_relations(triad):
    assert sorted(triad.relation_names()) == ["r_ab", "r_ar", "r_br"]
    r = triad.relations["r_ab"]
    assert r.src == "alice"
    assert r.dst == "bob"
    assert r.sign == 1
    assert r.magnitude == 1.0
    assert r.kind == "interpersonal"
    # hri_relation kind for human↔robot edges
    assert triad.relations["r_ar"].kind == "hri_relation"
    assert triad.relations["r_br"].kind == "hri_relation"


def test_triad_has_one_cycle_with_three_members(triad):
    assert triad.cycle_names() == ["triad"]
    c = triad.cycles["triad"]
    assert sorted(c.members) == ["r_ab", "r_ar", "r_br"]


def test_triad_has_three_policies_with_actions(triad):
    assert sorted(triad.policy_names()) == ["mediate", "repair", "withdraw"]
    assert triad.policies["repair"].action == "signal_alignment"
    assert triad.policies["mediate"].action == "mediation_offer"
    assert triad.policies["withdraw"].action == "withdraw"
    # Conditions must be non-empty strings referencing sigma(triad).
    for p in triad.policies.values():
        assert "sigma" in p.condition


def test_relation_src_dst_must_be_agents(tmp_path):
    """If a relation references a non-existent agent, the loader
    must raise ValueError with a clear message."""
    from signedkan_wip.src.rapport.coalition import load_coalition
    bad = tmp_path / "bad_triad.hymeko"
    bad.write_text("""
bad_description {
    @"meta_hri.hymeko";
    using hri_meta as hri;
}

bad: hri {
    alice: hri.human {}
    r_orphan: hri.interpersonal {
        from alice;
        to   ghost;
        sign 1;
        magnitude 1.0;
    }
}
""")
    # Copy meta_hri.hymeko alongside the bad file so the parser can
    # resolve the @import. The parser doesn't actually resolve the
    # import semantically; it just needs the syntax to parse.
    meta = REPO_ROOT / "data" / "coalitions" / "meta_hri.hymeko"
    (tmp_path / "meta_hri.hymeko").write_text(meta.read_text())
    with pytest.raises(ValueError, match=r"dst=.ghost. not an agent"):
        load_coalition(bad)


def test_cycle_members_must_be_relations(tmp_path):
    """If a cycle references a non-existent relation, ValueError."""
    from signedkan_wip.src.rapport.coalition import load_coalition
    bad = tmp_path / "bad_cycle.hymeko"
    bad.write_text("""
bad_description {
    @"meta_hri.hymeko";
    using hri_meta as hri;
}

bad: hri {
    alice: hri.human {}
    bob:   hri.human {}
    r_ab: hri.interpersonal {
        from alice;
        to   bob;
        sign 1;
        magnitude 1.0;
    }
    bad_cycle: hri.sigma_cycle {
        members [r_ab, r_ghost];
    }
}
""")
    meta = REPO_ROOT / "data" / "coalitions" / "meta_hri.hymeko"
    (tmp_path / "meta_hri.hymeko").write_text(meta.read_text())
    with pytest.raises(ValueError, match=r"member .r_ghost. not a relation"):
        load_coalition(bad)
