"""Export the kinematic-graph fingerprints the browser demo consumes.

The browser page (``index.html``) is static; it needs one JSON blob
describing each canonical mechanism: its links, its *signed* joints
(sign = rotational vs translational), and — the centrepiece — its
**cycle-arity histogram**. That histogram is the structural fingerprint
HSiKAN's learned αₖ regime compass converges to: parallel mechanisms
spike at k=4 (four-bar) or k=6 (Stewart / delta), serial chains and
trees stay flat (no closed loops).

Two code paths, picked at runtime:

* **real pipeline** — if ``hymeko_neuro.experiments.kinematic`` imports
  (i.e. the full torch/numpy stack is installed), the cycle histogram
  is taken from the canonical ``kinematic_loop_summary`` so the numbers
  match every other report in the repo verbatim.
* **stdlib fallback** — otherwise a dependency-free URDF reader +
  simple-cycle enumerator reproduces the same ``cycles_per_arity``
  semantics (undirected simple cycles of length k, deduped by
  rotation+reflection). Used on machines without the research stack.

Run:
    python demo_web/export_kinematic_data.py
Output:
    demo_web/kinematic_data.json
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = Path(__file__).resolve().parent / "kinematic_data.json"
MAX_K = 6

# Sign convention mirrors hymeko_neuro/experiments/kinematic/graph.py::JOINT_SIGN.
JOINT_SIGN: dict[str, int] = {
    "revolute": +1,
    "continuous": +1,
    "planar": +1,
    "floating": +1,
    "prismatic": -1,
}


# ── Curated catalogue ───────────────────────────────────────────────
# Canonical fixtures with a clean, teachable cycle signature. ``family``
# is ground truth for these hand-built mechanisms (not a prediction).

@dataclass(frozen=True)
class MechSpec:
    id: str
    label: str
    rel_path: str
    family: str
    blurb: str


CATALOGUE: tuple[MechSpec, ...] = (
    MechSpec(
        "four_bar", "4-bar planar linkage",
        "data/robotics/synthetic_fixtures/four_bar.urdf", "parallel (planar)",
        "One closed loop of 4 links. The textbook planar parallel "
        "mechanism — the regime compass spikes at k=4.",
    ),
    MechSpec(
        "stewart", "Stewart platform (hexapod)",
        "data/robotics/synthetic_fixtures/stewart.urdf", "parallel (spatial)",
        "6-DOF spatial parallel manipulator. A dense forest of k=6 "
        "loops — the canonical 'k=6 spike'.",
    ),
    MechSpec(
        "delta_3rrr", "Delta 3-RRR",
        "data/robotics/synthetic_fixtures/delta_3rrr.urdf", "parallel (spatial)",
        "Three parallel legs, sparser than Stewart but the SAME k=6 "
        "arity signature — αₖ is regime, not magnitude.",
    ),
    MechSpec(
        "serial_4", "Serial arm — 4-DOF",
        "data/robotics/synthetic_fixtures/serial_4.urdf", "serial (open chain)",
        "Open chain. No closed loops → the cycle pool is empty → the "
        "regime compass is flat. The 'null' reference.",
    ),
    MechSpec(
        "serial_7", "Serial arm — 7-DOF (redundant)",
        "data/robotics/synthetic_fixtures/serial_7.urdf", "serial (open chain)",
        "A 7-DOF redundant manipulator (WAM-shaped). Still an open "
        "chain — longer, but no cycles.",
    ),
)


# ── URDF reader (stdlib only) ───────────────────────────────────────

@dataclass
class Joint:
    name: str
    parent: str
    child: str
    jtype: str
    sign: int
    axis: tuple[float, float, float]


@dataclass
class Mechanism:
    links: list[str] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)


def read_urdf(path: Path) -> Mechanism:
    """Parse links + movable joints from a URDF (fixed joints skipped).

    Preconditions: ``path`` is a readable URDF XML file.
    Postconditions: every returned joint references links present in
    ``links``; sign ∈ {+1, −1} per :data:`JOINT_SIGN`.
    """
    root = ET.parse(str(path)).getroot()
    mech = Mechanism()
    mech.links = [el.get("name", f"link{i}")
                  for i, el in enumerate(root.findall("link"))]
    known = set(mech.links)
    for j_el in root.findall("joint"):
        jtype = (j_el.get("type") or "fixed").strip().lower()
        if jtype == "fixed":
            continue
        p_el, c_el = j_el.find("parent"), j_el.find("child")
        if p_el is None or c_el is None:
            continue
        parent, child = p_el.get("link"), c_el.get("link")
        if parent not in known or child not in known:
            continue
        ax_el = j_el.find("axis")
        axis = (0.0, 0.0, 1.0)
        if ax_el is not None and ax_el.get("xyz"):
            parts = [float(v) for v in ax_el.get("xyz").split()]
            if len(parts) == 3:
                axis = (parts[0], parts[1], parts[2])
        mech.joints.append(Joint(
            name=j_el.get("name", "?"), parent=parent, child=child,
            jtype=jtype, sign=JOINT_SIGN.get(jtype, +1), axis=axis,
        ))
    return mech


# ── Cycle-arity histogram (stdlib fallback) ─────────────────────────

def _adjacency(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def count_simple_cycles_by_arity(
    n_nodes: int, edges: list[tuple[int, int]], max_k: int = MAX_K,
) -> dict[int, int]:
    """Count undirected simple cycles of each length 3..max_k.

    Each cycle is counted once (deduped by its frozenset of vertices is
    insufficient — two distinct cycles can share a vertex set — so we
    canonicalise the ordered vertex ring under rotation + reflection).

    Matches the semantics of
    ``hymeko_neuro/experiments/kinematic/graph.py::kinematic_loop_summary``'s
    ``cycles_per_arity`` for the small mechanisms in the catalogue.
    """
    adj = _adjacency(n_nodes, edges)
    seen: set[tuple[int, ...]] = set()
    counts: dict[int, int] = {k: 0 for k in range(3, max_k + 1)}

    def canon(ring: list[int]) -> tuple[int, ...]:
        # Smallest rotation of the ring or its reverse, anchored at min.
        best: tuple[int, ...] | None = None
        for seq in (ring, ring[::-1]):
            m = seq.index(min(seq))
            rot = tuple(seq[m:] + seq[:m])
            if best is None or rot < best:
                best = rot
        return best  # type: ignore[return-value]

    # DFS from each start; only extend to neighbours > start to halve work,
    # close the cycle when we return to start with length in [3, max_k].
    def dfs(start: int, current: int, path: list[int], visited: set[int]):
        if len(path) > max_k:
            return
        for nxt in adj[current]:
            if nxt == start and len(path) >= 3:
                key = canon(path)
                if key not in seen:
                    seen.add(key)
                    counts[len(path)] += 1
            elif nxt > start and nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, nxt, path, visited)
                path.pop()
                visited.discard(nxt)

    for s in range(n_nodes):
        dfs(s, s, [s], {s})
    return counts


def cycles_via_real_pipeline(path: Path) -> dict[int, int] | None:
    """Use the canonical loop summary if the research stack imports."""
    try:
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from hymeko_neuro.experiments.kinematic import (  # type: ignore
            kinematic_loop_summary, urdf_to_signed_graph,
        )
    except Exception:
        return None
    g, _names, joints = urdf_to_signed_graph(path)
    summary = kinematic_loop_summary(g, joints, max_k=MAX_K)
    return {int(k): int(v) for k, v in summary["cycles_per_arity"].items()}


# ── Assembly ────────────────────────────────────────────────────────

def topology_signature(cycle_counts: dict[int, int], n_links: int,
                        edges: list[tuple[int, int]]) -> str:
    n_cycles = sum(max(0, v) for v in cycle_counts.values())
    if n_cycles == 0:
        deg = [0] * n_links
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return "open chain" if (max(deg) if deg else 0) <= 2 else "tree"
    if cycle_counts.get(4, 0) > 0 and cycle_counts.get(6, 0) == 0:
        return "4-bar / planar parallel"
    if cycle_counts.get(6, 0) > 0:
        return "Stewart / delta / spatial parallel"
    return "mixed serial-parallel"


def build_entry(spec: MechSpec, used_real: list[bool]) -> dict | None:
    path = REPO_ROOT / spec.rel_path
    if not path.is_file():
        print(f"  [skip] {spec.id}: URDF missing ({path})")
        return None
    mech = read_urdf(path)
    idx = {name: i for i, name in enumerate(mech.links)}
    edges = [(idx[j.parent], idx[j.child]) for j in mech.joints]

    real = cycles_via_real_pipeline(path)
    if real is not None:
        cycle_counts = real
        used_real.append(True)
    else:
        cycle_counts = count_simple_cycles_by_arity(len(mech.links), edges)
        used_real.append(False)

    n_rev = sum(1 for j in mech.joints if j.sign == +1)
    n_pris = sum(1 for j in mech.joints if j.sign == -1)
    return {
        "id": spec.id,
        "label": spec.label,
        "family": spec.family,
        "blurb": spec.blurb,
        "n_links": len(mech.links),
        "n_joints": len(mech.joints),
        "n_revolute": n_rev,
        "n_prismatic": n_pris,
        "topology": topology_signature(cycle_counts, len(mech.links), edges),
        "links": mech.links,
        "joints": [
            {
                "name": j.name, "parent": j.parent, "child": j.child,
                "type": j.jtype, "sign": j.sign, "axis": list(j.axis),
            }
            for j in mech.joints
        ],
        "edges": [[u, v] for u, v in edges],
        "cycles_per_arity": {str(k): int(v) for k, v in cycle_counts.items()},
    }


def main() -> None:
    used_real: list[bool] = []
    entries = [e for s in CATALOGUE if (e := build_entry(s, used_real))]
    source = ("real kinematic_loop_summary" if used_real and all(used_real)
              else "stdlib cycle enumerator (fallback)")
    payload = {
        "schema": 1,
        "cycle_source": source,
        "max_k": MAX_K,
        "mechanisms": entries,
    }
    blob = json.dumps(payload, indent=2)
    OUT_PATH.write_text(blob, encoding="utf-8")
    # Also emit a <script src>-loadable wrapper so index.html works on a
    # bare double-click (file://), where fetch() is CORS-blocked.
    js_path = OUT_PATH.with_suffix(".js")
    js_path.write_text(f"window.KINEMATIC_DATA = {blob};\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"wrote {js_path}  ({len(entries)} mechanisms, "
          f"cycles via {source})")
    for e in entries:
        spikes = {k: v for k, v in e["cycles_per_arity"].items() if v}
        print(f"  {e['id']:<12s} {e['topology']:<28s} "
              f"cycles={spikes or 'none'}")


if __name__ == "__main__":
    main()
