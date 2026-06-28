"""Store a trained policy *as a HyMeKo hypergraph*, and read it back — the storage thesis as an artifact.

The identity this rests on: a weight matrix :math:`W \\in \\mathbb{R}^{m\\times n}` is the weighted
biadjacency of a bipartite graph, which is the **star expansion** of a hypergraph (``m`` vertices, ``n``
hyperedges, incidence ``B[i,j] = W[i,j]``). So every learned tensor of an actor-critic round-trips:

    state_dict  --policy_to_hymeko-->  .hymeko (valid HyMeKo, each tensor a weighted incidence)
                <--hymeko_to_policy--   (re-read, reconstruct, bit-exact)

For the cart-pole HSiKAN policy the structural ``a_pos/a_neg`` *are* the kinematic hypergraph's signed
incidence (the star expansion of the robot), and the learned ``w_*`` matrices are stored as incidence
tensors too. Weights are written full-precision and inline (the policy is small); for a large net the
same scheme stores the structure inline and the bulk in a content-addressed blob (see the architecture
plan, P4). :func:`weight_to_hypergraph` exposes the matrix↔hypergraph identity for visualisation.
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Per-tensor storage tiers (binary-tensor-storage plan): T0 inline-decimal (auditable, small/structural),
# T1 inline-binary base64-float32 (~2× smaller, self-contained), T2 content-addressed external `.npz` blob
# (structure inline + bulk by reference, sha256-verified) — for nets too large to inline.
TIERS = ("t0", "t1", "t2", "auto")
_AUTO_T0_MAX = 64           # auto: <= this many elements stay human-readable decimals (T0)
_AUTO_T1_MAX = 1_000_000    # auto: <= this many elements inline as binary (T1); above → external blob (T2)


def _sha256(flat: np.ndarray) -> str:
    return hashlib.sha256(flat.astype("<f4").tobytes()).hexdigest()


@dataclass(frozen=True)
class StarHypergraph:
    """A weight matrix as the star expansion of a weighted bipartite hypergraph.

    Rows are vertices, columns are hyperedges; ``incidence[i, j]`` is the weight on the star edge from
    vertex ``i`` to hyperedge ``j``. ``star_edges`` is the explicit ``(vertex, hyperedge, weight)`` list
    (the bipartite/star view).

    # Invariants ``incidence.shape == (n_vertices, n_hyperedges)``; ``hypergraph_to_weight`` inverts.
    """

    n_vertices: int
    n_hyperedges: int
    incidence: np.ndarray

    @property
    def star_edges(self) -> list[tuple[int, int, float]]:
        return [(i, j, float(self.incidence[i, j]))
                for i in range(self.n_vertices) for j in range(self.n_hyperedges)]


def weight_to_hypergraph(w: np.ndarray) -> StarHypergraph:
    """A 2-D weight matrix → its weighted star-expansion hypergraph (rows=vertices, cols=hyperedges).

    # Preconditions ``w`` is 2-D. # Postconditions :func:`hypergraph_to_weight` recovers ``w`` exactly.
    """
    a = np.asarray(w)
    if a.ndim != 2:
        raise ValueError(f"weight_to_hypergraph expects a 2-D matrix; got shape {a.shape}")
    return StarHypergraph(a.shape[0], a.shape[1], a.copy())


def hypergraph_to_weight(h: StarHypergraph) -> np.ndarray:
    """Inverse of :func:`weight_to_hypergraph`: the star-expansion incidence → the weight matrix."""
    return h.incidence.copy()


# ── HyMeKo (de)serialisation ──────────────────────────────────────────────────
def _fnum(x: float) -> str:
    """Shortest **positional** decimal that round-trips this value as float32 — no scientific notation
    (HyMeKo's number lexer rejects the ``e`` in ``1.5e-05``), so a stored policy is *valid HyMeKo*."""
    s = np.format_float_positional(np.float32(x), unique=True, trim="0")
    if s.endswith("."):
        s += "0"
    return "0.0" if s in ("-0.0", "-0", "-0.") else s


def _fmt_list(flat: np.ndarray) -> str:
    return "[" + ", ".join(_fnum(x) for x in flat) + "]"


def _encode_b64(flat: np.ndarray) -> str:
    """Little-endian float32 bytes → base64 (T1). A *string* in the .hymeko, so no number-lexer issue."""
    return base64.b64encode(flat.astype("<f4").tobytes()).decode("ascii")


def _decode_b64(s: str) -> np.ndarray:
    """Base64 → little-endian float32 array (inverse of :func:`_encode_b64`)."""
    return np.frombuffer(base64.b64decode(s.encode("ascii")), dtype="<f4")


def _provenance_block(meta: "dict[str, object]") -> list[str]:
    """A ``provenance { … }`` decl recording who/how trained the policy (algo, backbone, strategy, …).
    String values are quoted; numbers are bare — both valid HyMeKo."""
    lines = ["provenance {"]
    for k, v in meta.items():
        lines.append(f'    {k} "{v}";' if isinstance(v, str) else f"    {k} {v};")
    lines += ["}", ""]
    return lines


_PROV_BLOCK = re.compile(r"provenance\s*\{(.*?)\n\}", re.DOTALL)
_PROV_STR = re.compile(r'(\w+)\s+"([^"]*)";')
_PROV_NUM = re.compile(r"(\w+)\s+(-?[\d.]+);")


def read_provenance(path: str | Path) -> "dict[str, str]":
    """Read the ``provenance`` block from a stored ``.hymeko`` (algo/backbone/strategy/…). ``{}`` if absent."""
    m = _PROV_BLOCK.search(Path(path).read_text(encoding="utf-8"))
    if m is None:
        return {}
    body = m.group(1)
    out: dict[str, str] = {k: v for k, v in _PROV_STR.findall(body)}
    for k, v in _PROV_NUM.findall(body):
        out.setdefault(k, v)
    return out


def policy_to_hymeko(state_dict: "dict[str, torch.Tensor]", path: str | Path, *,
                     tier: str = "auto", blob_path: str | Path | None = None,
                     meta: "dict[str, object] | None" = None) -> Path:
    """Write a trained ``state_dict`` as a valid ``.hymeko`` — each tensor a weighted-incidence decl.

    ``tier`` selects the per-tensor encoding: ``"t0"`` inline-decimal (auditable), ``"t1"`` inline-binary
    base64-float32 (~2× smaller, self-contained), ``"t2"`` content-addressed external ``.npz`` blob
    (structure inline + bulk by reference, ``sha256``-verified), ``"auto"`` keeps small/structural tensors
    (``<= _AUTO_T0_MAX`` elements) as readable decimals, the bulk as binary (T1), and anything larger than
    ``_AUTO_T1_MAX`` as a blob (T2). Every tier round-trips bit-exact; the tier is a storage choice, never
    semantics. For any T2 tensor a sibling ``.npz`` is written (``blob_path``, default ``<stem>.weights.npz``).

    # Preconditions every value is a tensor; ``tier in TIERS``. # Postconditions the file at ``path`` is
    valid HyMeKo (``hymeko validate`` passes) and :func:`hymeko_to_policy` reconstructs it bit-exact.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be in {TIERS}; got {tier!r}")
    path = Path(path)
    blob = Path(blob_path) if blob_path is not None else path.with_suffix(".weights.npz")
    blob_data: dict[str, np.ndarray] = {}   # npz_key -> float32 array, written once at the end
    # Self-contained (inline tensor vocab) so the stored policy validates anywhere — a portable artifact,
    # not coupled to data/nn/meta_nn.hymeko's location. Mirrors meta_nn's tensor typing.
    lines = [
        "// GENERATED by hymeko_rl.policy_store — a trained actor-critic stored AS a HyMeKo hypergraph.",
        "// Each tensor is the weighted incidence of a bipartite hypergraph (its star expansion); the",
        "// cart-pole's a_pos/a_neg are the kinematic graph's signed incidence. Round-trips bit-exact.",
        "",
        "stored_policy_description {}",
        "",
        *(_provenance_block(meta) if meta else []),
        "nn {",
        "    tensors {",
        "        meta_tensor {}",
        "        t_activation: + <isa> meta_tensor {}",
        "    }",
        "}",
        "",
        "stored_policy: nn.tensors {",
    ]
    for idx, (key, tensor) in enumerate(state_dict.items()):
        arr = tensor.detach().cpu().numpy()
        shape = list(arr.shape) if arr.ndim else [1]
        flat = arr.reshape(-1).astype("<f4")
        if tier == "t2" or (tier == "auto" and flat.size > _AUTO_T1_MAX):
            chosen = "t2"
        elif tier == "t1" or (tier == "auto" and flat.size > _AUTO_T0_MAX):
            chosen = "t1"
        else:
            chosen = "t0"
        lines.append(f"    w_{idx}: nn.tensors.t_activation {{")
        lines.append(f'        key "{key}";')
        lines.append(f"        shape [{', '.join(str(s) for s in shape)}];")
        if chosen == "t2":
            npz_key = f"w_{idx}"
            blob_data[npz_key] = flat
            lines.append('        dtype "f32";')
            lines.append(f'        blob "{blob.name}";')
            lines.append(f'        npz_key "{npz_key}";')
            lines.append(f'        sha256 "{_sha256(flat)}";')
        elif chosen == "t1":
            lines.append('        dtype "f32";')
            lines.append(f'        data_b64 "{_encode_b64(flat)}";')
        else:
            lines.append(f"        data {_fmt_list(flat.astype(np.float64))};")
        lines.append("    }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if blob_data:
        # numpy's savez stub mis-types **kwds (as `allow_pickle: bool`); the call is correct.
        np.savez(blob, **blob_data)  # type: ignore[arg-type]
    return path


# One stored tensor decl `w_N: … { … }` — body captured for per-field extraction (handles all tiers).
_BLOCK_RE = re.compile(r"w_\d+:\s*[^{]*\{(.*?)\n\s*\}", re.DOTALL)
_FIELD = {k: re.compile(rf'{k}\s+"([^"]*)"') for k in ("key", "data_b64", "blob", "npz_key", "sha256")}
_SHAPE_RE = re.compile(r"shape\s+\[([^\]]*)\]")
_DATA_RE = re.compile(r"data\s+\[([^\]]*)\]")


def hymeko_to_policy(path: str | Path) -> "dict[str, torch.Tensor]":
    """Read a ``.hymeko`` written by :func:`policy_to_hymeko` back into a ``state_dict``.

    Dispatches per tensor on the present field — ``data`` (T0 decimal), ``data_b64`` (T1 binary), or
    ``blob``/``npz_key``/``sha256`` (T2 external, sha256-verified). The blob is resolved relative to the
    ``.hymeko``'s directory.

    # Errors ``ValueError`` if a T2 tensor's blob is missing/stale (``sha256`` mismatch).
    # Postconditions returns float32 tensors keyed by the original ``state_dict`` keys, shapes restored.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    blobs: dict[str, Any] = {}   # rel filename -> loaded npz (cached)

    def _blob(rel: str) -> Any:
        if rel not in blobs:
            blobs[rel] = np.load(path.parent / rel)
        return blobs[rel]

    out: dict[str, torch.Tensor] = {}
    for body in _BLOCK_RE.findall(text):
        key = _FIELD["key"].search(body).group(1)                       # type: ignore[union-attr]
        shape = [int(s) for s in _SHAPE_RE.search(body).group(1).split(",") if s.strip()]  # type: ignore[union-attr]
        if (mb := _FIELD["blob"].search(body)) is not None:             # T2 external blob
            npz_key = _FIELD["npz_key"].search(body).group(1)           # type: ignore[union-attr]
            flat = np.asarray(_blob(mb.group(1))[npz_key], dtype="<f4")
            want = _FIELD["sha256"].search(body).group(1)               # type: ignore[union-attr]
            if _sha256(flat) != want:
                raise ValueError(f"sha256 mismatch for {key!r}: blob '{mb.group(1)}' is missing or stale")
        elif (m64 := _FIELD["data_b64"].search(body)) is not None:      # T1 inline-binary
            flat = _decode_b64(m64.group(1))
        else:                                                          # T0 inline-decimal
            data_s = _DATA_RE.search(body).group(1)                     # type: ignore[union-attr]
            flat = np.array([float(x) for x in data_s.split(",") if x.strip()], dtype=np.float32)
        out[key] = torch.tensor(np.asarray(flat, dtype=np.float32).reshape(shape), dtype=torch.float32)
    if not out:
        raise ValueError(f"no stored tensors parsed from {path}")
    return out
