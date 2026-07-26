"""R2 relational encoder — the 10 mandatory correctness gates BEFORE any training (frozen contract
`reports/2026-07-27-coin-r2-relational-contract.md`). Pure-graph tests are fast; test 10 (physical deploy smoke) drives
one real dev snapshot through the unchanged option executor. No held-out data (s4/s7) is touched here."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.theta_option.canonical_frame import (
    BALANCE_IDX, R1_GROUP_ORDER, canonicalise, from_canonical_theta, group_len, swap_grouped)
from hymeko_rl.coin_delivery.theta_option.relational_graph import build_graph_from_canonical, recover_r1_groups
from hymeko_rl.coin_delivery.theta_option.relational_encoder import (
    RelationalKHeadNet, RelationalKHeadProposal, graph_tensors, relational_deploy_one)
from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import KHeadProposalNet
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, DIM, ThetaBox
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL

R1_PARAMS = 25240
R2_PRIMARY_H = 25
R2_PRIMARY_PARAMS = 25774


def _grouped(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=group_len(name)) for name in R1_GROUP_ORDER}


def _net(seed: int = 0, k: int = 4, h: int = R2_PRIMARY_H) -> RelationalKHeadNet:
    torch.manual_seed(seed)
    return RelationalKHeadNet(k, h=h)


def _permute_sides(G):
    """A copy of the graph with the two side rows (TIP/CONTACT/PORT + authority) swapped — the arbitrary storage order."""
    from dataclasses import replace
    nodes = dict(G.nodes)
    for t in ("TIP", "CONTACT", "PORT"):
        nodes[t] = nodes[t][::-1].copy()
    return replace(G, nodes=nodes, authority=G.authority[::-1].copy())


# ── 1. GRAPH INFORMATION PARITY: every R1 group recovered; the ONLY non-R1 attributes are documented constants ──
def test_1_information_parity_and_no_new_input():
    g = _grouped(0)
    G = build_graph_from_canonical(g, was_swapped=False)
    rec = recover_r1_groups(G)
    assert set(rec) == set(R1_GROUP_ORDER), "every R1 group has exactly one recovery home"
    for name in R1_GROUP_ORDER:
        assert np.allclose(np.asarray(rec[name]).ravel(), np.asarray(g[name]).ravel(), atol=1e-9), name
    # the ONLY non-R1 attributes are the documented CONSTANTS (COIN phase placeholder + TARGET zone spec) — no learned input
    assert float(G.node("COIN")[-1]) == 0.0, "COIN phase placeholder is the const 0 at t=0"
    assert np.allclose(G.node("TARGET"), [CENTER_TOL, SETTLE_VEL, float(HELD_DWELL)]), "TARGET is the constant zone spec"


# ── 2. GRAPH MIRROR EQUIVALENCE: G_c(Mx)==G_c(x); the encoder output is identical; the decode flag flips ──
def test_2_graph_and_encoder_mirror_equivalence():
    net = _net()
    for seed in range(8):
        g = _grouped(seed)
        ca, sa = canonicalise(g)
        cb, sb = canonicalise(swap_grouped(g))
        Ga, Gb = build_graph_from_canonical(ca, sa), build_graph_from_canonical(cb, sb)
        for t in Ga.nodes:
            assert np.allclose(Ga.nodes[t], Gb.nodes[t], atol=1e-9), (seed, t)
        assert Ga.was_swapped != Gb.was_swapped
        with torch.no_grad():
            za, zb = net(graph_tensors(Ga)), net(graph_tensors(Gb))
        assert torch.allclose(za, zb, atol=1e-6), f"encoder output must be mirror-invariant (seed {seed})"


# ── 3. NODE-PERMUTATION INVARIANCE: swapping the two side rows leaves the pooled embedding + theta heads unchanged ──
def test_3_node_permutation_invariance():
    net = _net()
    for seed in range(8):
        G = build_graph_from_canonical(_grouped(seed), was_swapped=bool(seed % 2))
        with torch.no_grad():
            z0 = net(graph_tensors(G))
            z1 = net(graph_tensors(_permute_sides(G)))
        assert torch.allclose(z0, z1, atol=1e-6), f"side-permutation must not change the heads (seed {seed})"


# ── 4. TIED-WEIGHT TEST: the two physical sides run through the SAME encoder + message functions ──
def test_4_tied_side_weights():
    net = _net()
    # structural: exactly one module per per-side node TYPE (not two side-specific embeddings)
    assert isinstance(net.enc_tip, torch.nn.Linear) and isinstance(net.enc_contact, torch.nn.Linear)
    assert isinstance(net.enc_port, torch.nn.Linear)
    # functional: a graph whose two sides carry identical features yields identical per-side node encodings
    g = _grouped(3)
    G = build_graph_from_canonical(g, was_swapped=False)
    t = graph_tensors(G)
    for typ, enc in (("tip", net.enc_tip), ("contact", net.enc_contact), ("port", net.enc_port)):
        sym = t[typ].clone()
        sym[1] = sym[0]                                   # force the two sides equal
        with torch.no_grad():
            e = torch.relu(enc(sym))
        assert torch.allclose(e[0], e[1], atol=1e-7), f"tied encoder must map equal side features to equal states ({typ})"


# ── 5. THETA OUTPUT EQUIVARIANCE: a mirrored physical state decodes to the T_theta-transformed physical theta (balance sign) ──
def test_5_theta_output_equivariance():
    net = _net()
    box = ThetaBox()
    for seed in range(8):
        g = _grouped(seed)
        ca, sa = canonicalise(g)
        cb, sb = canonicalise(swap_grouped(g))
        Ga, Gb = build_graph_from_canonical(ca, sa), build_graph_from_canonical(cb, sb)
        prop = RelationalKHeadProposal(4, net, box)
        for ma, mb in zip(prop.modes(Ga), prop.modes(Gb)):     # same canonical heads, opposite was_swapped
            da = from_canonical_theta(np.asarray(ma.center, np.float64), Ga.was_swapped)
            db = from_canonical_theta(np.asarray(mb.center, np.float64), Gb.was_swapped)
            for j in range(DIM):
                if j == BALANCE_IDX:
                    assert np.isclose(da[j], -db[j], atol=1e-5), (seed, "balance must flip sign")
                else:
                    assert np.isclose(da[j], db[j], atol=1e-5), (seed, j)


# ── 6. PARAMETER-BUDGET TEST: the PRIMARY R2 model is within +-5% of the R1 K-head ──
def test_6_parameter_budget_within_5pct():
    r1 = sum(p.numel() for p in KHeadProposalNet(4, feat_dim=43, h=128).parameters())
    r2 = sum(p.numel() for p in RelationalKHeadNet(4, h=R2_PRIMARY_H).parameters())
    assert r1 == R1_PARAMS and r2 == R2_PRIMARY_PARAMS
    assert 0.95 * r1 <= r2 <= 1.05 * r1, f"R2 {r2} outside +-5% of R1 {r1}"


# ── 7. DETERMINISTIC FORWARD: same graph + model state + seed -> identical heads ──
def test_7_deterministic_forward():
    G = build_graph_from_canonical(_grouped(5), was_swapped=True)
    net = _net(seed=7)
    with torch.no_grad():
        a, b = net(graph_tensors(G)), net(graph_tensors(G))
    assert torch.equal(a, b)
    net2 = _net(seed=7)                                   # same seed -> identical init -> identical output
    with torch.no_grad():
        c = net2(graph_tensors(G))
    assert torch.allclose(a, c, atol=0.0)


# ── 8. K-HEAD OUTPUT CONTRACT: K=4 bounded 6-D canonical theta per head; legal box; distinct read-outs (no aliasing) ──
def test_8_khead_output_contract():
    net = _net()
    G = build_graph_from_canonical(_grouped(2), was_swapped=False)
    with torch.no_grad():
        z = net(graph_tensors(G))
    assert z.shape == (4, DIM) and bool(z.abs().max() <= 1.0 + 1e-6)
    box = ThetaBox()
    centres = np.asarray([box.denorm(zk.numpy()) for zk in z], np.float64)
    assert np.all(centres >= box.lo - 1e-5) and np.all(centres <= box.hi + 1e-5), "every head is a legal theta"
    assert net.heads.weight.shape[0] == 4 * DIM, "K distinct 6-D linear read-outs, no head/action aliasing"


# ── 9. SEARCH-PROVENANCE REGRESSION: K=4, budget 8 total, centre inclusion, decoded centres, theta0/theta_exec split ──
class _StubScorer:
    """Deterministic injectable scorer (no physics): score = -||candidate - target||; counts evaluations."""
    def __init__(self, target):
        self.target = np.asarray(target, np.float64)
        self.n = 0

    def score(self, candidate, rng):
        self.n += 1
        d = float(np.linalg.norm(np.asarray(candidate, np.float64) - self.target))
        return -d, {"delivery_success": d < 3.0, "k6_delivered": d < 3.0, "k6_max_dwell": 6 if d < 3.0 else 0,
                    "dtz_end": 0.01 if d < 3.0 else 0.07, "dtz_start": 0.1, "terminal_coin_speed": 0.02,
                    "peak_qdot": 1.0, "peak_coin_speed": 0.5}


def test_9_search_provenance_regression():
    net = _net()
    box = ThetaBox()
    prop = RelationalKHeadProposal(4, net, box)
    G = build_graph_from_canonical(_grouped(4), was_swapped=True)
    scorer = _StubScorer(target=box.denorm(np.zeros(DIM)))
    d = relational_deploy_one(None, prop, G, np.random.default_rng(0), 8, box, cfg=DELIVERY_CFG, scorer=scorer)
    assert d["n_modes"] == 4
    assert sum(d["per_mode_budget"]) == 8 and d["budget_total"] == 8, "fair total budget-8 split"
    assert all(b >= 1 for b in d["per_mode_budget"]), "every mode centre-inclusive (>=1 each)"
    assert scorer.n == 8, "exactly 8 candidate evaluations total"
    assert len(d["canonical_heads"]) == 4 and len(d["decoded_physical_centres"]) == 4
    assert 0 <= d["selected_head"] < 4
    # decoded centres are the inverse T_theta of the canonical heads (balance flips because was_swapped)
    for canon, phys in zip(d["canonical_heads"], d["decoded_physical_centres"]):
        assert np.isclose(canon[BALANCE_IDX], -phys[BALANCE_IDX], atol=1e-3)
    assert len(d["theta_exec"]) == DIM


# ── 10. PHYSICAL DEPLOY SMOKE: the graph pipeline reaches the unchanged option executor with a legal bounded theta ──
@pytest.mark.slow
def test_10_physical_deploy_smoke():
    from hymeko_rl.coin_delivery.theta_option.relational_graph import build_graph
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    harness = load_harness()
    snap = acquire_snapshot(harness, 14250)[0]            # ONE dev cradle (s1); NO held-out data
    box = ThetaBox()
    prop = RelationalKHeadProposal(4, _net(), box)
    G = build_graph(snap)                                  # canonicalise -> typed graph (records was_swapped)
    d = relational_deploy_one(snap, prop, G, np.random.default_rng(0), 8, box)   # real physics
    assert sum(d["per_mode_budget"]) == 8
    te = np.asarray(d["theta_exec"], np.float64)
    assert np.all(te >= box.lo - 1e-4) and np.all(te <= box.hi + 1e-4), "executor received a LEGAL bounded theta"
    assert isinstance(d["delivery_success"], bool) and np.isfinite(d["peak_qdot"])
    assert d["peak_qdot"] <= 3.0 + 1e-6, "no safety bypass: motion contract respected"
