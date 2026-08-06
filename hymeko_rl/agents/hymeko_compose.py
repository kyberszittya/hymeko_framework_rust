"""Collect the files that COMPOSE a HyMeKo description — the transitive ``@"…"`` import closure.

A HyMeKo model is built from modular files: a scenario imports a robot composite (which imports its kinematics
vocabulary + link definitions) and a reward profile (which imports the reward vocabulary), and so on. This tool
walks that import graph and reports the whole composition — a "bill of materials" for a model, and a clean
demonstration of HyMeKo's modular, reusable structure.

    python -m hymeko_rl.agents.hymeko_compose data/robotics/pick_place_scenario.hymeko
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from hymeko_rl.env._profile import read_decls, read_imports


@dataclass
class CompositionNode:
    """One file in a model's composition: its path, the decls it contributes, and its imported children.

    ``dup`` marks a file already visited earlier in the walk (shown once with its decls, then as a back-reference)
    so import diamonds/cycles terminate. ``missing`` marks an import whose file is absent on disk.
    """

    path: Path
    decls: list[tuple[str, str]] = field(default_factory=list)
    imports: list["CompositionNode"] = field(default_factory=list)
    dup: bool = False
    missing: bool = False


def collect_composition(path: str | Path, _seen: set[Path] | None = None) -> CompositionNode:
    """The composition tree rooted at ``path``: every ``@"…"``-imported file, transitively.

    # Postconditions a tree whose every node is a real file (``missing`` nodes flag dangling imports); a file
      imported on more than one path appears in full at its first visit and as ``dup`` thereafter (no infinite
      recursion on diamonds/cycles).
    """
    path = Path(path).resolve()
    seen = _seen if _seen is not None else set()
    if not path.is_file():
        return CompositionNode(path, missing=True)
    if path in seen:
        return CompositionNode(path, decls=read_decls(path), dup=True)
    seen.add(path)
    node = CompositionNode(path, decls=read_decls(path))
    node.imports = [collect_composition(imp, seen) for imp in read_imports(path)]
    return node


def unique_files(node: CompositionNode) -> list[Path]:
    """The distinct existing files that compose the model, in first-visit order (the bill of materials)."""
    out: list[Path] = []
    seen: set[Path] = set()

    def walk(n: CompositionNode) -> None:
        if n.missing or n.path in seen:
            return
        seen.add(n.path)
        out.append(n.path)
        for child in n.imports:
            walk(child)

    walk(node)
    return out


def to_composition_dot(root: CompositionNode) -> str:
    """Render the model composition as a HYPERGRAPH in Graphviz DOT: each ``.hymeko`` model is a vertex, and each
    model's set of imports is a **composition hyperedge** (a hub node joining the composite to its parts — the
    star-expansion of "X is composed of {A, B, …}"). Since each model is itself a hypergraph, this is a
    hypergraph-of-hypergraphs (the bundle-of-bundles view). Render with: ``dot -Tsvg out.dot -o out.svg``.

    # Postconditions DOT text; shared imports appear once (one vertex, multiple incident hyperedges)."""
    lines = ["digraph hymeko_composition {", "  rankdir=LR;",
             '  node [shape=box, style=rounded, fontsize=10, fontname="Helvetica"];',
             '  edge [color="#2a8a55"];']
    seen: set[Path] = set()
    hub = [0]

    def emit(node: CompositionNode) -> None:
        if node.missing:
            return
        name = node.path.name
        if node.path not in seen:
            seen.add(node.path)
            lines.append(f'  "{name}" [label="{name}\\n{len(node.decls)} decl(s)"];')
        if node.dup:
            return                                              # its hyperedge was emitted at first visit
        kids = [k for k in node.imports if not k.missing]
        if kids:
            h = f"compose{hub[0]}"
            hub[0] += 1
            lines.append(f'  {h} [shape=point, width=0.07, color="#2a8a55"];   // composition hyperedge')
            lines.append(f'  "{name}" -> {h} [arrowhead=none];')
            for k in kids:
                lines.append(f'  {h} -> "{k.path.name}";')
            for k in node.imports:
                emit(k)

    emit(root)
    lines.append("}")
    return "\n".join(lines)


def format_tree(node: CompositionNode, *, root: Path | None = None, prefix: str = "", last: bool = True) -> str:
    """A readable import tree: each file with the decls it contributes; ``↩`` = already shown, ``✗`` = missing."""
    root = root or node.path.parent
    try:
        name = node.path.relative_to(root).as_posix()
    except ValueError:
        name = node.path.name
    tag = " ✗ (missing)" if node.missing else " ↩ (already composed)" if node.dup else ""
    decls = "" if (node.dup or node.missing or not node.decls) else \
        "   {" + ", ".join(f"{n}:{k}" for n, k in node.decls[:6]) + ("…" if len(node.decls) > 6 else "") + "}"
    connector = "" if prefix == "" else ("└─ " if last else "├─ ")
    line = f"{prefix}{connector}{name}{tag}{decls}"
    child_prefix = prefix + ("" if prefix == "" else ("   " if last else "│  "))
    lines = [line]
    kids = node.imports
    for i, child in enumerate(kids):
        lines.append(format_tree(child, root=root, prefix=child_prefix, last=(i == len(kids) - 1)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="path to a .hymeko file")
    ap.add_argument("--list", action="store_true", help="just print the flat bill of materials")
    ap.add_argument("--dot", help="write the composition hypergraph as Graphviz DOT to this path")
    a = ap.parse_args(argv)
    node = collect_composition(a.model)
    files = unique_files(node)
    if a.dot:
        Path(a.dot).write_text(to_composition_dot(node), encoding="utf-8")
        print(f"wrote composition hypergraph (DOT) to {a.dot}  (render: dot -Tsvg {a.dot} -o out.svg)")
    if a.list:
        for f in files:
            print(f.as_posix())
    elif not a.dot:
        root = Path(a.model).resolve().parent
        print(format_tree(node, root=root))
        print(f"\n{len(files)} file(s) compose {Path(a.model).name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
