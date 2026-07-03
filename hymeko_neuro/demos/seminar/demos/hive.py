"""Demo 2 — HIVE compilation, live (how HyMeKo represents a hypergraph).

Pure transform, no model. One source through surface ``.hymeko`` → canonical IR
→ tensor encodings (star COO / clique matrix), side by side, plus the
**canonicalisation** punchline: the engine's Blake3 fingerprint is invariant to
*how you write* the graph (declaration order) and changes on a structural edit.

Honest framing (verified 2026-06-10, see
``docs/plans/2026-06-10-canonical-hash-iso-invariance/``): the hash is invariant
to node/edge **declaration order**, NOT to relabeling or within-edge member
order — node identity is semantic in HyMeKo. So we claim "same model written two
ways → same fingerprint", not abstract graph-isomorphism invariance.

Acceptance (SEMINAR_DEMOS §2): declaration-order-permuted inputs hash equal; a
one-edge change hashes differently. No model loaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..base import REPO_ROOT, DemoContext, DemoResult

# A controlled signed chain used for the canonicalisation proof. parse_dsl on
# these is reliable (unlike the robot fixtures, whose @"..." includes are
# resolved relative to cwd and currently fail to load).
_CHAIN_BASE = """G{}
g{
  n0 {} n1 {} n2 {} n3 {}
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
  @e2{ (~n2, ~n3); }
}
"""
_CHAIN_EDGE_REORDER = """G{}
g{
  n0 {} n1 {} n2 {} n3 {}
  @e2{ (~n2, ~n3); }
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
}
"""
_CHAIN_NODE_REORDER = """G{}
g{
  n3 {} n2 {} n1 {} n0 {}
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
  @e2{ (~n2, ~n3); }
}
"""
_CHAIN_STRUCTURAL_CHANGE = """G{}
g{
  n0 {} n1 {} n2 {} n3 {}
  @e0{ (~n0, ~n1); }
  @e1{ (~n1, ~n2); }
  @e2{ (~n0, ~n3); }
}
"""


class HiveDemo:
    name = "hive"
    help = "HIVE compilation: surface .hymeko -> IR -> COO/CSR + canonical-hash invariance."

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--src", default="data/typical_graphs/fano_graph.hymeko",
            help="self-contained .hymeko source to compile (no @-includes).",
        )
        parser.add_argument(
            "--no-write", action="store_true", help="skip the JSON report.",
        )

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult:
        try:
            import hymeko
        except ImportError as err:
            raise RuntimeError(
                "Demo 2 needs the built `hymeko` module. Build it once with "
                "`.venv/Scripts/maturin.exe develop --manifest-path "
                "hymeko_py/Cargo.toml`."
            ) from err

        src = Path(args.src)
        if not src.is_absolute():
            src = REPO_ROOT / src
        if not src.is_file():
            raise FileNotFoundError(f"source not found: {src}")

        compiled = self._compile(hymeko, src)
        proof = self._canonicalisation_proof(hymeko)

        result = DemoResult(name=self.name)
        src_rel = os.path.relpath(src, REPO_ROOT)
        result.metrics.update({
            "source": src_rel,
            "n_nodes": compiled["n_nodes"],
            "n_edges": compiled["n_edges"],
            "star_coo_shape": compiled["star_coo_shape"],
            "star_coo_nnz": compiled["star_coo_nnz"],
            "clique_shape": compiled["clique_shape"],
            "clique_nnz": compiled["clique_nnz"],
            "canonical_hash": compiled["canonical_hash"],
        })
        for k in ("n_nodes", "n_edges", "star_coo_nnz", "clique_nnz", "canonical_hash"):
            result.provenance[k] = f"{src_rel} via hymeko engine"

        ok = proof["order_invariant"] and proof["structural_sensitive"]
        result.metrics["canonicalisation"] = "PASS" if ok else "FAIL"
        result.notes.append(
            f"canonicalisation: decl-order permutations hash equal "
            f"({proof['order_invariant']}); one-edge change differs "
            f"({proof['structural_sensitive']}) -> {'PASS' if ok else 'FAIL'}"
        )
        result.notes.append(
            "the fingerprint is invariant to declaration order, not to "
            "relabeling / within-edge order (node identity is semantic)."
        )
        result.notes.append(
            "star incidence grows O(|E|·d), clique O(|E|·d²): per arity-d edge, "
            "d incidences vs C(d,2) clique pairs — see the 3D viewer for the gap."
        )

        if not args.no_write:
            result.artifacts.append(
                self._write_report(ctx.out_dir, src, compiled, proof)
            )
        return result

    @staticmethod
    def _compile(hymeko: Any, src: Path) -> dict[str, Any]:
        eng = hymeko.PyHypergraphEngine()
        ir = eng.load_file(str(src))
        star = eng.compile_star_expansion(ir)
        clique = eng.compile_clique_expansion(ir)
        return {
            "n_nodes": ir.node_count,
            "n_edges": ir.edge_count,
            "star_coo_shape": list(star.shape),
            "star_coo_nnz": star.nnz,
            "clique_shape": list(clique.shape),
            "clique_nnz": clique.nnz,
            "canonical_hash": ir.canonical_hash,
            "snapshot": json.loads(ir.snapshot_json()),
            "dot": ir.to_dot("compiled"),
        }

    @staticmethod
    def _canonicalisation_proof(hymeko: Any) -> dict[str, Any]:
        def h(src: str) -> str:
            return str(hymeko.PyHypergraphEngine().parse_dsl(src).canonical_hash)

        base = h(_CHAIN_BASE)
        edge_reorder = h(_CHAIN_EDGE_REORDER)
        node_reorder = h(_CHAIN_NODE_REORDER)
        changed = h(_CHAIN_STRUCTURAL_CHANGE)
        return {
            "hash_base": base,
            "hash_edge_reorder": edge_reorder,
            "hash_node_reorder": node_reorder,
            "hash_structural_change": changed,
            "order_invariant": base == edge_reorder == node_reorder,
            "structural_sensitive": base != changed,
        }

    @staticmethod
    def _write_report(
        out_dir: Path, src: Path, compiled: dict[str, Any], proof: dict[str, Any],
    ) -> Path:
        report = {
            "source": str(src),
            "surface_text": src.read_text(encoding="utf-8"),
            "ir_snapshot": compiled["snapshot"],
            "ir_dot": compiled["dot"],
            "star_coo": {
                "shape": compiled["star_coo_shape"], "nnz": compiled["star_coo_nnz"],
            },
            "clique": {
                "shape": compiled["clique_shape"], "nnz": compiled["clique_nnz"],
            },
            "canonical_hash": compiled["canonical_hash"],
            "canonicalisation_proof": proof,
        }
        path = out_dir / "hive_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path


__all__ = ["HiveDemo"]
