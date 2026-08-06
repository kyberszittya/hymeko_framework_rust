"""Tests for R11.6D Phase 4 transportability retrieval: score, signatures, top-1 selection, LOSO, AUROC."""
from hymeko_rl.coin_delivery.transport_retrieval import (
    TransportSignature,
    TransportWeights,
    build_signatures,
    cell_index,
    evaluate_handoff,
    rank_theta,
    score,
    signature_from_cells,
)
from hymeko_rl.experiments.r11_6d_transport_retrieval import _auroc


def _sig(transport: float, k6: float = 1.0, contact: float = 1.0, lo: float = -1.0, hi: float = 1.0) -> TransportSignature:
    return TransportSignature(transport, 0.0, lo, hi, 0.0, 0.0, k6, contact)


def test_score_penalises_undershoot_and_overshoot_separately() -> None:
    w = TransportWeights(alpha=2.0, beta=0.5)                     # undershoot hurts 4x more than overshoot
    qf = {"d_required_mm": 100.0, "bearing": 0.0}
    under = score(qf, _sig(70.0), w)                             # transports 70, needs 100 -> undershoot 30
    over = score(qf, _sig(130.0), w)                            # transports 130 -> overshoot 30
    exact = score(qf, _sig(100.0), w)
    assert exact == 0.0 and under == -2.0 * 30 and over == -0.5 * 30 and under < over < exact


def test_score_angle_gap_and_reward_terms() -> None:
    qf = {"d_required_mm": 100.0, "bearing": 2.0}                 # bearing outside [-1, 1]
    w = TransportWeights(alpha=1.0, beta=1.0, gamma=10.0, eta=50.0, rho=20.0)
    s = score(qf, _sig(100.0, k6=0.8, contact=0.9, lo=-1.0, hi=1.0), w)
    assert s == -10.0 * 1.0 + 50.0 * 0.8 + 20.0 * 0.9            # angle gap = bearing 2 - hi 1 = 1


def test_rank_prefers_transport_match() -> None:
    qf = {"d_required_mm": 100.0, "bearing": 0.0}
    sigs = {"near": _sig(70.0), "match": _sig(100.0), "over": _sig(140.0)}
    assert rank_theta(qf, sigs, TransportWeights(alpha=1.0, beta=1.0))[0] == "match"


def _cell(h, t, split, tr, k6, dtz, bearing=0.0, safe=True, contact=1.0, under=0.0, over=0.0) -> dict:
    return {"handoff": h, "theta": t, "split": split, "projected_transport_mm": tr, "k6": k6, "safe": safe,
            "dtz_mm": dtz, "bearing": bearing, "contact_retention": contact, "undershoot_mm": under, "overshoot_mm": over}


def test_signature_typical_transport_and_k6_rate() -> None:
    cells = [_cell("h1", "tA", "train", 70.0, True, 5.0), _cell("h2", "tA", "train", 90.0, False, 40.0),
             _cell("h3", "tA", "train", 80.0, True, 8.0)]
    sig = signature_from_cells(cells)
    assert sig.typical_transport_mm == 80.0 and sig.k6_rate == round(2 / 3, 3)


def test_build_signatures_filters_train_and_excludes() -> None:
    cells = [_cell("h1", "tA", "train", 70.0, True, 5.0), _cell("hd", "tA", "dev", 200.0, False, 99.0),
             _cell("h2", "tA", "train", 90.0, True, 6.0)]
    sig = build_signatures(cells, exclude=frozenset({"h2"}))["tA"]
    assert sig.typical_transport_mm == 70.0                       # dev cell excluded; h2 excluded -> only h1


def test_evaluate_handoff_top1_regret_and_top3() -> None:
    # from handoff H: tMatch delivers (dtz 5), tOther delivers (dtz 12), tMiss fails.
    cells = [_cell("H", "tMatch", "train", 100.0, True, 5.0), _cell("H", "tOther", "train", 100.0, True, 12.0),
             _cell("H", "tMiss", "train", 60.0, False, 40.0)]
    idx = cell_index(cells)
    sigs = {"tMatch": _sig(100.0), "tOther": _sig(100.0), "tMiss": _sig(60.0)}
    r = evaluate_handoff(idx, "H", {"d_required_mm": 100.0, "bearing": 0.0}, sigs,
                         ["tMatch", "tOther", "tMiss"], TransportWeights(alpha=1.0, beta=1.0))
    assert r["k6"] and r["top3_deliverable"] and r["oracle_dtz"] == 5.0 and r["regret"] == round(r["sel_dtz"] - 5.0, 2)


def test_auroc_separates_deliverable() -> None:
    assert _auroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert _auroc([True, False], [0.1, 0.9]) == 0.0
    assert _auroc([True, True], [0.5, 0.6]) is None              # undefined when one class empty
