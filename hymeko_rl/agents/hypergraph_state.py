"""Robot geometry -> signed kinematic hypergraph -> torch tensor (the obs structure).

Csaba's pipeline to Kato: describe the arm in HyMeKo, **hymeko -> mjcf**, *then* the
observation as a hypergraph with an equivalent tensor representation. This module is that
last hop: from the compiled MJCF, extract the kinematic hypergraph —

* **vertices** = links (MJCF bodies, excluding the world);
* **hyperedges** = joints, each connecting parent -> child with a *signed* direction
  (down-chain +1, up-chain -1) so signed message passing distinguishes propagating
  toward the end-effector from propagating toward the base.

``from_hymeko`` runs ``hymeko emit -f mjcf`` (the CLI resolves the ``@"...";`` imports the
standalone PyO3 ``load_file`` cannot — a tracked engine bug) so the SAME ``.hymeko`` source
yields both the scene and the obs tensor. ``from_mjcf`` is the dependency-free core.

Refinement (tracked): the canonical ``.hymeko`` star-expansion (engine
``compile_star_expansion``) is the ideal direct source once the import resolver is fixed;
the MJCF-derived hypergraph is structurally faithful (same compiled robot) meanwhile.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HypergraphState:
    """A robot's signed kinematic hypergraph as plain arrays + a topology hash.

    # Invariants
    ``edges`` is ``(E, 2)`` int (parent_vertex, child_vertex) into ``vertex_labels``;
    ``signs`` is ``(E,)`` in ``{+1, -1}``; ``topo_hash`` changes iff the topology changes
    (the re-encode gate). One joint contributes two directed arcs (down +1, up -1).
    """

    vertex_labels: tuple[str, ...]
    edges: np.ndarray
    signs: np.ndarray
    topo_hash: str

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_labels)

    def semantic_fingerprint(self) -> str:
        """A hash of the SEMANTIC graph — vertex labels+order, signed edges — independent of the raw MJCF text
        (unlike ``topo_hash``). Two models with the same semantic policy graph but different physical MJCF (e.g. the
        legacy arm vs the v2 arm projected to 6 vertices) share this fingerprint. Used by the graph-contract
        equivalence gate and the canonical bundle manifest."""
        payload = repr((tuple(self.vertex_labels),
                        [tuple(map(int, e)) for e in np.asarray(self.edges).tolist()],
                        [int(s) for s in np.asarray(self.signs).tolist()]))
        return "sem:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def from_mjcf(cls, mjcf: str | Path, *, is_path: bool = True) -> HypergraphState:
        """Extract the signed kinematic hypergraph from an MJCF (path or XML string).

        # Preconditions the MJCF compiles; it has >= 1 joint.
        # Postconditions ``n_vertices == (nbody - 1) - #non-semantic-helper-bodies`` (world dropped, plus any body
        # explicitly flagged non-semantic via the ``hymeko_non_semantic_bodies`` metadata marker); each inter-link
        # joint contributes a down +1 and an up -1 arc.
        """
        text = Path(mjcf).read_text(encoding="utf-8") if is_path else str(mjcf)
        model = (mujoco.MjModel.from_xml_path(str(mjcf)) if is_path
                 else mujoco.MjModel.from_xml_string(text))
        # SEMANTIC GRAPH PROJECTION (spec-driven, explicit): bodies named in the emitted `hymeko_non_semantic_bodies`
        # metadata marker are geometry/emission helpers (e.g. v2 fingertip child bodies) — they stay physical/contact-
        # active in the MjModel but are NOT promoted to policy-graph vertices. No marker → every non-world body is a
        # vertex (legacy behaviour, byte-identical for every other robot). Helpers must be LEAF bodies (no semantic
        # child depends on a dropped vertex); a flagged non-leaf would orphan its subtree and is rejected below.
        mk = re.search(r'name\s*=\s*"hymeko_non_semantic_bodies"\s+data\s*=\s*"([^"]*)"', text)
        non_semantic = set(mk.group(1).split()) if mk else set()
        drop_ids = {b for b in range(1, model.nbody)
                    if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) in non_semantic}
        semantic = [b for b in range(1, model.nbody) if b not in drop_ids]
        # vertices: the semantic bodies, remapped to contiguous 0-based indices (world + helpers dropped).
        labels = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in semantic]
        body2vtx = {b: i for i, b in enumerate(semantic)}
        edges: list[tuple[int, int]] = []
        signs: list[int] = []
        for j in range(model.njnt):
            child = int(model.jnt_bodyid[j])
            parent = int(model.body_parentid[child])
            if child in drop_ids:
                continue  # a joint whose moving body is a dropped helper (helpers are welded leaves → no joint here)
            if parent in drop_ids:
                # a semantic child jointed to a DROPPED body would be orphaned → the helper is not a leaf. Reject
                # loudly rather than silently disconnect the chain.
                raise ValueError(f"non-semantic body {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent)!r} "
                                 f"has a jointed semantic child; graph_node-0 helpers must be leaf bodies")
            if parent not in body2vtx or child not in body2vtx:
                continue  # a joint to the world frame (free/base) has no parent vertex
            p, c = body2vtx[parent], body2vtx[child]
            edges.append((p, c))   # down-chain (parent -> child)
            signs.append(+1)
            edges.append((c, p))   # up-chain   (child  -> parent)
            signs.append(-1)
        if not edges:
            raise ValueError("MJCF has no inter-link joints; not a kinematic chain")
        return cls(
            vertex_labels=tuple(labels),
            edges=np.asarray(edges, dtype=np.int64),
            signs=np.asarray(signs, dtype=np.int64),
            topo_hash="blake-na:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32],
        )

    @classmethod
    def from_hymeko(cls, source: str | Path, *, model_name: str = "robot",
                    binary: str | Path | None = None) -> HypergraphState:
        """Emit MJCF from a ``.hymeko`` via the CLI (imports resolved), then extract.

        # Errors ``FileNotFoundError`` if the ``hymeko`` CLI is not built;
        ``RuntimeError`` if the emit fails (exit code propagated with stderr).
        """
        hb = Path(binary) if binary else _find_cli()
        if hb is None:
            raise FileNotFoundError(
                "hymeko CLI not found; build it with `cargo build -p hymeko_cli`")
        proc = subprocess.run(
            [str(hb), "emit", "-f", "mjcf", str(source), "-n", model_name],
            capture_output=True, text=True, cwd=_REPO,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hymeko emit -f mjcf failed (exit {proc.returncode}): {proc.stderr[:300]}")
        return cls.from_mjcf(proc.stdout, is_path=False)

    def star_expansion(self) -> "HypergraphState":
        """The bipartite **star-expansion**: each joint becomes a HUB node mediating its parent and child, so
        the graph carries the original vertices PLUS one hyperedge-hub per joint, and message passing flows
        ``vertex → hub → vertex``. The result is a drop-in :class:`HypergraphState` for the HSiKAN backbone — a
        richer structural prior in which the joints (the actuated DOF) are first-class nodes, not just edges.

        Each kinematic joint here is binary, so every hub has degree 2; the *form* generalises unchanged to
        higher-arity hubs once the canonical ``.hymeko`` star-expansion (true ``>2`` hyperedges) is available via
        the engine's transitive-import resolver (memory ``project-engine-transitive-imports``).

        # Preconditions ``self`` has ``>= 1`` down-chain (``+1``) arc (i.e. at least one joint).
        # Postconditions ``n_vertices == N + J`` (``J`` = #joints); each hub ``N+k`` connects exactly its
          parent and child, the down(``+1``)/up(``-1``) signs routed through it; ``topo_hash`` is suffixed
          ``":star"`` so a re-encode gate distinguishes it.
        """
        n = self.n_vertices
        joints = [(int(p), int(c)) for (p, c), s in zip(self.edges, self.signs) if s > 0]
        if not joints:
            raise ValueError("star_expansion requires >= 1 down-chain (+1) arc (a joint)")
        labels = list(self.vertex_labels) + [
            f"hub:{self.vertex_labels[p]}~{self.vertex_labels[c]}" for p, c in joints]
        edges: list[tuple[int, int]] = []
        signs: list[int] = []
        for k, (p, c) in enumerate(joints):
            hub = n + k
            edges += [(p, hub), (hub, c)]   # down-chain  parent → hub → child   (+1)
            signs += [1, 1]
            edges += [(c, hub), (hub, p)]   # up-chain    child  → hub → parent  (-1)
            signs += [-1, -1]
        return HypergraphState(
            vertex_labels=tuple(labels),
            edges=np.asarray(edges, dtype=np.int64),
            signs=np.asarray(signs, dtype=np.int64),
            topo_hash=self.topo_hash + ":star",
        )

    def with_task_hyperedges(self, *, new_entities: Sequence[str],
                             hyperedges: Sequence[tuple[str, Sequence[int]]]) -> "HypergraphState":
        """Augment with a **task layer**: add ``new_entities`` as vertices (e.g. the manipulated object and the
        goal), then add each hyperedge as a HUB node joined to its members — so non-robot entities and the
        relations that decide the task (a GRASP hyperedge ``{fingers, object}``, a GOAL hyperedge
        ``{object, target}``) become first-class graph structure the HSiKAN message-passes over, instead of flat
        broadcast features. This is the fix for the Galambos HSiKAN==MLP tie: the coin and zone were absent from
        the graph, so the structural prior had nothing task-relevant to reason over (memory
        ``project-galambos-hsikan-tie-rootcause``).

        Members are indices into the **augmented** vertex set: base vertices ``0..N-1``, then ``new_entities`` in
        order, then earlier hubs (so a hyperedge may reference an earlier hub — edge-on-edge incidence). Each
        member↔hub incidence mirrors :meth:`star_expansion`'s signing (member→hub ``+1``, hub→member ``-1``).

        # Preconditions every member index is ``< `` the number of vertices defined before its hub
          (base + entities + earlier hubs); each hyperedge has ``>= 1`` member.
        # Postconditions ``n_vertices == N + len(new_entities) + len(hyperedges)``; ``topo_hash`` suffixed
          ``":task"`` (re-encode gate).
        # Errors ``ValueError`` on an empty or out-of-range hyperedge.
        """
        labels = list(self.vertex_labels) + list(new_entities)
        edges = [list(map(int, e)) for e in self.edges.tolist()]
        signs = [int(s) for s in self.signs.tolist()]
        next_idx = self.n_vertices + len(new_entities)
        for hub_label, members in hyperedges:
            if not members:
                raise ValueError(f"task hyperedge {hub_label!r} has no members")
            if any(not 0 <= int(m) < next_idx for m in members):
                raise ValueError(
                    f"task hyperedge {hub_label!r} references an out-of-range member "
                    f"(only {next_idx} vertices defined before it)")
            hub = next_idx
            next_idx += 1
            labels.append(hub_label)
            for m in members:
                edges.append([int(m), hub])
                signs.append(1)         # member → hub  (+1, into the relation)
                edges.append([hub, int(m)])
                signs.append(-1)        # hub → member  (-1, out of the relation)
        return HypergraphState(
            vertex_labels=tuple(labels),
            edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            signs=np.asarray(signs, dtype=np.int64),
            topo_hash=self.topo_hash + ":task",
        )

    def dense_signed_adj(self, device: torch.device | str = "cpu",
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-normalised dense ``(A_pos, A_neg)`` of shape ``(N, N)``.

        Dense is correct here: kinematic graphs are tiny (N <= ~20), and a dense
        ``A`` batches cleanly as ``einsum('nm,bmd->bnd', A, x)`` over rollout features
        (a sparse ``mm`` would not). ``A[i, j]`` weights neighbour ``j``'s message to ``i``.
        """
        n = self.n_vertices
        a_pos = torch.zeros(n, n)
        a_neg = torch.zeros(n, n)
        for (src, dst), s in zip(self.edges, self.signs):
            # message flows src -> dst; receiver dst aggregates from src.
            (a_pos if s > 0 else a_neg)[int(dst), int(src)] += 1.0
        a_pos = _row_normalise(a_pos)
        a_neg = _row_normalise(a_neg)
        return a_pos.to(device), a_neg.to(device)


def _row_normalise(a: torch.Tensor) -> torch.Tensor:
    deg = a.sum(dim=1, keepdim=True).clamp_min(1.0)
    return a / deg


def _find_cli() -> Path | None:
    for rel in ("target/debug/hymeko.exe", "target/debug/hymeko",
                "target/release/hymeko.exe", "target/release/hymeko"):
        p = _REPO / rel
        if p.exists():
            return p
    found = shutil.which("hymeko")
    return Path(found) if found else None
