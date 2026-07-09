"""Wire the HyMeKo reward source-of-truth into the LiNGAM-SH mechanism pipeline (integration step).

Takes the HyMeKo-declared MetaWorld reward (`data/robotics/metaworld_reward.hymeko`), turns its terms into a
``{reward terms} → {total_reward}`` mechanism proposal, and evaluates it against the observed pairwise ``B`` (from
the coffee-push multi-seed DirectLiNGAM evidence) alongside three baselines — **no mechanism**, **raw pairwise
edges**, and **common-child-only**. Uses the existing Step 3A/3B factorization tools; selects; cross-view-verifies
the declared reward mechanism. No training; no discovery from scratch (the reward mechanism comes from the HyMeKo
declaration, the common-child from the evidence).

Doctrine: this compares *reconstruction of the pairwise ``B``* — it does **not** claim causal truth from grouping.
The HyMeKo reward mechanism is grounded in the reward *fidelity* (R², `metaworld_reward.py`), not in ``B``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .metaworld_reward import _DEFAULT_SPEC, reward_mechanism_proposal


def _pairwise_proposals(edges: "Sequence[tuple[str, str, float]]") -> list[Any]:
    """One degenerate ``{cause} → {effect}`` mechanism proposal per pairwise edge (the raw-edges baseline)."""
    from hymeko_rl.eval.causal import MechanismProposal
    out = []
    for i, (cause, effect, w) in enumerate(edges):
        out.append(MechanismProposal(name=f"pw{i}_{cause}_{effect}", tail=(cause,), head=(effect,),
                                     strength=abs(float(w)), sign=int(np.sign(w)) or 1, confidence=1.0,
                                     source="raw_pairwise", evidence={}))
    return out


def _score(variables: "Sequence[str]", edges: "Sequence[tuple[str, str, float]]", proposals: list[Any],
           ) -> "dict[str, Any]":
    """Least-squares fit of a candidate set → its reconstruction metrics of ``B``."""
    from hymeko_rl.eval.causal import fit_sigma_least_squares
    fac = fit_sigma_least_squares(variables, edges, proposals)
    m = fac.metrics
    return {"n_mechanisms": len(proposals), "n_parameters": int(m["n_parameters"]),
            "fro_error": m["fro_error"], "explained_energy": m["explained_energy"],
            "relative_error": m["relative_error"]}


def compare_reward_mechanisms(variables: "Sequence[str]", edges: "Sequence[tuple[str, str, float]]", *,
                              spec_path: str = _DEFAULT_SPEC, out_path: "str | Path | None" = None,
                              ) -> dict[str, Any]:
    """Compare the HyMeKo reward mechanism vs none / raw-pairwise / common-child on reconstructing ``B``.

    # Postconditions returns per-candidate metrics + the HyMeKo reward mechanism's cross-view result; writes JSON
      to ``out_path`` if given.
    """
    from hymeko_rl.eval.causal import (
        cross_view_verify,
        propose_common_child,
        proposals_to_causal_hypergraph,
    )

    reward_prop = reward_mechanism_proposal(spec_path, available=variables)
    common = propose_common_child(edges)
    candidates: dict[str, list[Any]] = {
        "none": [], "raw_pairwise": _pairwise_proposals(edges),
        "common_child": common, "hymeko_reward": [reward_prop]}
    scores = {name: _score(variables, edges, props) for name, props in candidates.items()}

    # cross-view the HyMeKo reward mechanism (representation + engine verification)
    cg = proposals_to_causal_hypergraph(variables, [reward_prop], name="HymekoRewardMechanism")
    acyclic = cg.check_acyclicity().acyclic
    xview = cross_view_verify(cg, Path(out_path).parent / "hymeko_reward_mechanism.hymeko") if out_path \
        else cross_view_verify(cg, Path("/tmp/_hymeko_reward_mechanism.hymeko"))

    result: dict[str, Any] = {
        "variables": list(variables), "edges": [[c, e, round(w, 6)] for c, e, w in edges], "spec": spec_path,
        "reward_mechanism": {"tail": list(reward_prop.tail), "head": list(reward_prop.head),
                             "strength": reward_prop.strength, "sign": reward_prop.sign},
        "scores": scores, "common_child_proposals": [{"tail": list(p.tail), "head": list(p.head)} for p in common],
        "cross_view": {"acyclic": acyclic, **xview.as_dict()},
        "_disclaimer": "Compares reconstruction of the pairwise B; does NOT claim causal truth from grouping. The "
                       "HyMeKo reward mechanism is grounded in reward fidelity (R2), not in B.",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def _coffee_push_multiseed_b(path: str) -> "tuple[list[str], list[tuple[str, str, float]]]":
    """Representative (variables, directed edges) from the coffee-push multi-seed aggregate (stable edges)."""
    agg = json.loads(Path(path).read_text())["aggregate"]["edges"]
    edges: list[tuple[str, str, float]] = []
    variables: list[str] = []
    for key, v in agg.items():
        if v.get("presence", 0.0) < 0.5 or "dominant_direction" not in v:
            continue
        cause, effect = v["dominant_direction"].split("->")
        edges.append((cause, effect, float(v["median_weight"])))
        variables += [cause, effect]
    return list(dict.fromkeys(variables)), edges


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    from hymeko_rl.eval.evaluate import experiment_dir
    parser = argparse.ArgumentParser(description="HyMeKo reward SoT → LiNGAM-SH mechanism integration")
    parser.add_argument("--multiseed", type=str,
                        default="reports/figures/2026_07_08_16_39_cip_metaworld_multiseed/"
                                "coffee_push_multiseed_summary.json")
    parser.add_argument("--out", type=str, default="reports/figures")
    args = parser.parse_args(argv)
    variables, edges = _coffee_push_multiseed_b(args.multiseed)
    out_dir = experiment_dir(args.out, "cip_reward_mechanism_integration")
    res = compare_reward_mechanisms(variables, edges, out_path=out_dir / "reward_mechanism_comparison.json")
    print(f"[reward-mech] variables={variables}")
    print(f"[reward-mech] reward mechanism tail={res['reward_mechanism']['tail']} -> total_reward")
    for name, s in res["scores"].items():
        print(f"  {name:14s} explained_energy={s['explained_energy']:+.4f} params={s['n_parameters']} "
              f"fro_error={s['fro_error']:.4f}")
    print(f"[reward-mech] cross_view agree={res['cross_view']['agree']} acyclic={res['cross_view']['acyclic']} "
          f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
