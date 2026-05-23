"""Dataset utilities for HSiKAN / Gömb / HyMeYOLO experiments.

Moved here from ``signedkan_wip/src/datasets*.py`` on 2026-05-19
as part of Slice E of the directory-reorganisation plan
(``docs/plans/2026-05-19-signedkan-wip-organize/``). The original
flat naming (``datasets.py``, ``datasets_continuous.py``,
``datasets_meshes.py``, ``datasets_small.py``, ``datasets_synth.py``)
is replaced by a single package with topical submodules:

* :mod:`.legacy` — the original ``datasets.py`` (Konect / Bitcoin
  / Slashdot / Epinions loaders, the ``SignedGraph`` dataclass,
  ``load`` / ``split`` / ``deduplicate_pairs`` entry points).
* :mod:`.continuous` — continuous-valued signed graphs
  (``WeightedSignedGraph``).
* :mod:`.meshes` — polyhedral meshes (mixed-polytope datasets).
* :mod:`.small` — small synthetic test graphs (karate, SBM,
  hierarchical).
* :mod:`.synth` — sklearn-style synthetic datasets adapted for
  signed graphs (make_moons, make_circles, make_regression).

External callers continue to use the flat name they always used,
because this ``__init__`` re-exports the public surface verbatim:

.. code-block:: python

    from signedkan_wip.src.datasets import load, split, SignedGraph

is identical to what pre-Slice-E code wrote.

Object-oriented note: ``SignedGraph`` is the canonical
public dataclass; ``WeightedSignedGraph`` is its continuous-valued
sibling. A future refactor may unify them under a common
``SignedGraphBase`` ABC; not done in Slice E to keep the move
file-system-only.
"""

# ─── Re-export the public API of the (formerly-flat) modules ────────

from .legacy import (
    DATA_DIR,
    URLS,
    FORMATS,
    SignedGraph,
    download,
    load,
    deduplicate_pairs,
    split,
)
from .continuous import (
    WeightedSignedGraph,
    load_continuous,
)
from .meshes import (
    build_polyhedron,
    build_polyhedral_mesh,
    build_mixed_polytope_dataset,
)
from .small import (
    karate_faction_signed,
    sbm_signed,
    hierarchical_signed,
)
from .synth import (
    make_moon_signed_graph,
    make_circles_signed_graph,
    make_regression_signed_graph,
)

__all__ = [
    # legacy.py — the headline API
    "DATA_DIR",
    "URLS",
    "FORMATS",
    "SignedGraph",
    "download",
    "load",
    "deduplicate_pairs",
    "split",
    # continuous.py
    "WeightedSignedGraph",
    "load_continuous",
    # meshes.py
    "build_polyhedron",
    "build_polyhedral_mesh",
    "build_mixed_polytope_dataset",
    # small.py
    "karate_faction_signed",
    "sbm_signed",
    "hierarchical_signed",
    # synth.py
    "make_moon_signed_graph",
    "make_circles_signed_graph",
    "make_regression_signed_graph",
]
