"""Kinematic-graph analysis for the demo.

Wraps ``kinematic_graph.urdf_to_signed_graph`` and the cycle-summary
helper into a single ``KinematicBundle`` the GUI can render. v0 is
purely descriptive (no learned model — that comes once the structural
view is locked in).

A robot's kinematic structure is a signed graph (links = vertices,
movable joints = edges, sign = rotational vs translational). Closed
kinematic loops show up as cycles; parallel manipulators have
many k=4..6 cycles, serial arms have none. HSiKAN's αₖ vector — which
the demo already shows for signed-link prediction — is the natural
"kinematic family signature" once we train a classifier.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..datasets import SignedGraph
from ..kinematic_graph import (
    KinematicJoint, kinematic_loop_summary, urdf_to_signed_graph,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF_REGISTRY = Path(__file__).resolve().parent / "kinematic_models.yaml"


@dataclass
class URDFEntry:
    """Catalogue entry for one URDF in the demo."""

    id: str
    label: str
    path: Path
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.path.is_file()


@dataclass
class KinematicBundle:
    """Structural snapshot of a robot's kinematic graph.

    All fields are pure data — no torch tensors, no GPU state. Safe to
    pickle, hash, and ship to a Gradio session state.
    """

    name: str                 # human-readable id (e.g. "moveo")
    urdf_path: Path           # resolved absolute path
    graph: SignedGraph
    link_names: list[str]
    joints: list[KinematicJoint]
    cycle_counts: dict[int, int]   # arity → number of cycles

    @property
    def n_links(self) -> int:
        return self.graph.n_nodes

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def n_revolute(self) -> int:
        return sum(1 for j in self.joints
                    if j.joint_type in ("revolute", "continuous"))

    @property
    def n_prismatic(self) -> int:
        return sum(1 for j in self.joints if j.joint_type == "prismatic")

    @property
    def is_open_chain(self) -> bool:
        return sum(self.cycle_counts.values()) == 0

    def balance_summary(self) -> dict[str, float | int]:
        """Davis-balance metrics on the joint signs.

        For kinematic graphs, the *signs* encode joint kind (rev/pris),
        not trust. Balance has no kinematic interpretation, but the
        statistics are still a useful structural fingerprint that
        HSiKAN consumes downstream.
        """
        signs = self.graph.signs
        n_pos = int((signs == 1).sum())
        n_neg = int((signs == -1).sum())
        n_total = signs.shape[0]
        return {
            "n_edges": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_fraction": (n_pos / n_total) if n_total else 0.0,
        }


def _resolve_urdf(raw: str | Path) -> Path:
    """Absolute paths kept as-is; otherwise resolved against REPO_ROOT."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_urdf_bundle(
    urdf_path: str | Path,
    name: str | None = None,
    max_k_cycles: int = 6,
) -> KinematicBundle:
    """Parse a URDF and enumerate cycles up to ``max_k_cycles``.

    Cycle enumeration uses a hard cap (10 000 per arity) so a 5 000-link
    tree won't OOM. For interactive use on real robots (drchubo at 52
    links, moveo at 8) the cap is never hit.
    """
    path = _resolve_urdf(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    g, link_names, joints = urdf_to_signed_graph(path)
    summary = kinematic_loop_summary(g, joints, max_k=max_k_cycles)
    cycle_counts = {int(k): int(v) for k, v in summary["cycles_per_arity"].items()}
    return KinematicBundle(
        name=name or path.stem,
        urdf_path=path,
        graph=g,
        link_names=link_names,
        joints=joints,
        cycle_counts=cycle_counts,
    )


def topology_signature(bundle: KinematicBundle) -> str:
    """One-line label classifying the mechanism's topology.

    Coarse heuristic for v0; replace with a trained classifier in v0.5.
    """
    n_cycles = sum(bundle.cycle_counts.values())
    if n_cycles == 0:
        # Serial chain or tree.
        deg = np.zeros(bundle.n_links, dtype=np.int64)
        for u, v in bundle.graph.edges:
            deg[u] += 1; deg[v] += 1
        max_deg = int(deg.max()) if bundle.n_links else 0
        if max_deg <= 2:
            return "open chain"
        return "tree"
    # Some closed loops present.
    if bundle.cycle_counts.get(4, 0) > 0 and bundle.cycle_counts.get(6, 0) == 0:
        return "4-bar / planar parallel"
    if bundle.cycle_counts.get(6, 0) > 0:
        return "Stewart / delta / spatial parallel"
    return "mixed serial-parallel"


def load_urdf_registry(
    path: str | Path | None = None,
) -> list[URDFEntry]:
    """Load the URDF catalogue.

    Precedence: explicit ``path`` arg > ``HYMEKO_URDF_REGISTRY`` env
    var > the packaged default. Missing-file errors degrade to an empty
    list with a printed warning — the GUI falls back to upload-only.
    """
    if path is None:
        env = os.environ.get("HYMEKO_URDF_REGISTRY")
        path = Path(env) if env else DEFAULT_URDF_REGISTRY
    path = Path(path).expanduser()
    if not path.is_file():
        print(f"[demo.kinematic] WARNING: no URDF registry at {path}")
        return []
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    out: list[URDFEntry] = []
    for raw in doc.get("urdfs", []):
        try:
            out.append(URDFEntry(
                id=str(raw["id"]),
                label=str(raw.get("label", raw["id"])),
                path=_resolve_urdf(raw["path"]),
                notes=str(raw.get("notes", "")).strip(),
            ))
        except KeyError as e:
            print(f"[demo.kinematic] skipping malformed URDF entry "
                  f"(missing {e.args[0]!r}): {raw!r}")
    return out


def urdf_dropdown_choices(entries: list[URDFEntry]) -> list[tuple[str, str]]:
    """``(label, id)`` pairs for a Gradio Dropdown.

    Available URDFs first; missing files tagged ``[MISSING]``.
    """
    avail = sorted([e for e in entries if e.available], key=lambda e: e.label)
    miss = sorted([e for e in entries if not e.available], key=lambda e: e.label)
    return ([(e.label, e.id) for e in avail]
            + [(f"[MISSING] {e.label}", e.id) for e in miss])


def find_urdf_by_id(entries: list[URDFEntry], entry_id: str) -> URDFEntry | None:
    for e in entries:
        if e.id == entry_id:
            return e
    return None


__all__ = [
    "REPO_ROOT",
    "DEFAULT_URDF_REGISTRY",
    "KinematicBundle",
    "URDFEntry",
    "find_urdf_by_id",
    "load_urdf_bundle",
    "load_urdf_registry",
    "topology_signature",
    "urdf_dropdown_choices",
]
