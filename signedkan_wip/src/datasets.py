"""SignedKAN — Phase 1.1–1.4: Bitcoin Alpha + OTC loaders.

Public source: SNAP signed-network datasets.
- Bitcoin Alpha: https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- Bitcoin OTC:   https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

CSV format: source, target, rating ∈ [-10, 10], unix-timestamp.
We binarise rating → sign (rating > 0 ⇒ +1; rating < 0 ⇒ −1; 0 dropped).

Run: python3 -m src.datasets --download bitcoin_alpha bitcoin_otc
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "bitcoin_alpha": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz",
    "bitcoin_otc":   "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
    "slashdot":      "https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz",
    "epinions":      "https://snap.stanford.edu/data/soc-sign-epinions.txt.gz",
    # WikiSigned k=2 (Maniu et al.): ~12K nodes, ~140K signed votes,
    # ~80% positive.  SNAP no longer hosts the bare .txt.gz; the KONECT
    # mirror packages it inside a tar.bz2 with a different layout
    # (handled by FORMATS["wikisigned"] = "konect_signed").
    "wikisigned":    "http://konect.cc/files/download.tsv.wikisigned-k2.tar.bz2",
    # KONECT: Wikipedia admin elections (Leskovec et al.). 7118 nodes,
    # ~100K signed votes (support/oppose). Same konect 4-col format.
    "wiki_elec":     "http://konect.cc/files/download.tsv.elec.tar.bz2",
    # KONECT: Wikipedia edit-war / conflict network. 118K nodes,
    # 2.9M multi-signed edges (continuous weights → binarised on load).
    "wiki_conflict": "http://konect.cc/files/download.tsv.wikiconflict.tar.bz2",
}

# Two on-disk file formats:
#   "bitcoin"    — comma-separated, 4 cols: src, dst, rating, timestamp.
#                  rating ∈ [-10, 10]; binarised rating > 0 → +1.
#   "snap_signed" — tab-separated, 3 cols: src, dst, sign ∈ {+1, -1}.
FORMATS = {
    "bitcoin_alpha": "bitcoin",
    "bitcoin_otc":   "bitcoin",
    "slashdot":      "snap_signed",
    "epinions":      "snap_signed",
    "wikisigned":    "konect_signed",
    "wiki_elec":     "konect_signed",
    "wiki_conflict": "konect_signed",
}


@dataclass
class SignedGraph:
    edges: np.ndarray   # (E, 2) src, dst
    signs: np.ndarray   # (E,) +1 / -1
    n_nodes: int

    def stats(self) -> dict:
        n_pos = int((self.signs == +1).sum())
        n_neg = int((self.signs == -1).sum())
        return {
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_frac": n_pos / max(1, n_pos + n_neg),
        }


def download(name: str) -> Path:
    fmt = FORMATS[name]
    ext = "csv" if fmt == "bitcoin" else "txt"
    out = DATA_DIR / f"{name}.{ext}"
    if out.exists():
        return out
    print(f"  downloading {name} from {URLS[name]} ...")
    req = urllib.request.Request(
        URLS[name], headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    # konect_signed datasets ship as `.tar.bz2` containing an
    # `out.<name>` file with `% header\nsrc tgt weight\n...`.  Extract
    # the inner file and rewrite it to the canonical `<name>.txt`
    # location for downstream parsing.
    if fmt == "konect_signed":
        import io
        import tarfile
        # Slug-to-inner-file mapping: konect packs `out.<slug>` inside
        # `<slug>/`.  Our dataset names use Python-safe underscores;
        # konect URLs use dashes/underscores per the original slug.
        konect_slug = {
            "wikisigned":    "wikisigned-k2",
            "wiki_elec":     "elec",
            "wiki_conflict": "wikiconflict",
        }.get(name, name)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:bz2") as tf:
            inner = next(
                (m for m in tf.getmembers()
                 if m.name.endswith(f"out.{konect_slug}")),
                None,
            )
            if inner is None:
                # Some KONECT archives use a different inner name
                # (e.g. moreno_sampson → out.moreno_sampson_sampson).
                # Fall back to the first `out.` member.
                inner = next(
                    (m for m in tf.getmembers()
                     if "/out." in m.name),
                    None,
                )
            if inner is None:
                raise RuntimeError(
                    f"konect_signed archive missing inner out.* for {name}"
                )
            fh = tf.extractfile(inner)
            assert fh is not None
            text = fh.read().decode("utf-8")
    else:
        text = gzip.decompress(raw).decode("utf-8")
    out.write_text(text)
    print(f"  wrote {out}")
    return out


def load(name: str) -> SignedGraph:
    # Programmatic datasets (Phase 6 small/synthetic stitch).
    if name == "karate":
        from .datasets_small import karate_faction_signed
        return karate_faction_signed()
    if name.startswith("sbm_"):
        # sbm_n200_k4_s0 → sbm_signed(n_nodes=200, n_communities=4, seed=0)
        # sbm_n200 / sbm_n400 → same with k=4, seed=0 (matches run_final_cell
        # ``sbm_n{N}`` shorthand used in SOTA tables).
        from .datasets_small import sbm_signed
        parts = name.split("_")
        n = int(parts[1][1:])
        if len(parts) >= 3 and parts[2].startswith("k"):
            k = int(parts[2][1:])
        else:
            k = 4
        if len(parts) >= 4 and parts[3].startswith("s"):
            seed_part = int(parts[3][1:])
        else:
            seed_part = 0
        g, _ = sbm_signed(n_nodes=n, n_communities=k, seed=seed_part)
        return g
    if name.startswith("hier_"):
        # hier_n240_s0 → hierarchical_signed(n_nodes=240, seed=0)
        from .datasets_small import hierarchical_signed
        parts = name.split("_")
        n = int(parts[1][1:])
        seed_part = int(parts[2][1:]) if len(parts) > 2 else 0
        g, _ = hierarchical_signed(n_nodes=n, seed=seed_part)
        return g
    if name.startswith("sbmsweep_"):
        # sbmsweep_pos85_s0 → SBM with pos_in=0.85, seed=0
        from .datasets_small import sbm_signed
        parts = name.split("_")
        pos_in = int(parts[1][3:]) / 100.0
        seed_part = int(parts[2][1:]) if len(parts) > 2 else 0
        g, _ = sbm_signed(n_nodes=200, n_communities=4,
                            p_in=0.20, p_out=0.05,
                            pos_in=pos_in, pos_out=0.15,
                            noise=0.05, seed=seed_part)
        return g
    raw_path = download(name)
    fmt = FORMATS[name]
    edges = []
    signs = []
    nodes = set()
    with raw_path.open() as f:
        if fmt == "bitcoin":
            reader = csv.reader(f)
        elif fmt == "konect_signed":
            # `% header\nsrc tgt weight\n...` whitespace-separated.
            # Use a manual line parser since csv.reader chokes on
            # mixed tabs+spaces.
            reader = (line.split() for line in f if line.strip())
        else:  # SNAP signed (tab-separated)
            reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#") or row[0].startswith("%"):
                continue
            if len(row) < 3:
                continue
            try:
                s = int(row[0])
                t = int(row[1])
                # konect multisigned datasets (wiki_conflict) use
                # continuous float weights; binarise on sign.
                rf = float(row[2])
            except ValueError:
                continue
            if rf == 0.0:
                continue
            edges.append((s, t))
            signs.append(1 if rf > 0 else -1)
            nodes.add(s); nodes.add(t)
    # Re-index to 0..N-1.
    node_list = sorted(nodes)
    remap = {n: i for i, n in enumerate(node_list)}
    edges_arr = np.array([(remap[s], remap[t]) for s, t in edges], dtype=np.int64)
    signs_arr = np.array(signs, dtype=np.int8)
    g = SignedGraph(
        edges=edges_arr, signs=signs_arr, n_nodes=len(node_list)
    )
    # konect multisigned (wiki_conflict) packs many edges per pair;
    # collapse via majority-vote so downstream cycle enumeration sees
    # a clean simple signed graph.
    if fmt == "konect_signed" and name in ("wiki_conflict",):
        g = deduplicate_pairs(g, merge="majority")
    return g


def deduplicate_pairs(g: SignedGraph,
                       merge: str = "majority") -> SignedGraph:
    """Collapse duplicate (u, v) edges (undirected) into a single edge per
    pair, removing the leak source where (u, v) and (v, u) — or repeated
    ratings of the same pair across time — straddle a train/val/test
    split and let the held-out edge's pair survive in g_features.

    ``merge`` controls how multiple sign entries for the same pair are
    combined:
      - "majority": majority vote (ties → +1).
      - "first":    keep the first entry's sign (insertion order).
      - "last":     keep the last entry's sign.

    The returned SignedGraph has at most one edge per undirected pair.
    Vertex IDs are preserved.
    """
    pair_signs: dict[tuple[int, int], list[int]] = {}
    for (u, v), s in zip(g.edges, g.signs):
        key = (int(min(u, v)), int(max(u, v)))
        pair_signs.setdefault(key, []).append(int(s))

    out_edges = np.empty((len(pair_signs), 2), dtype=np.int64)
    out_signs = np.empty(len(pair_signs), dtype=np.int8)
    for i, (key, signs) in enumerate(pair_signs.items()):
        u, v = key
        out_edges[i, 0] = u
        out_edges[i, 1] = v
        if merge == "majority":
            n_pos = sum(1 for s in signs if s == 1)
            n_neg = len(signs) - n_pos
            out_signs[i] = 1 if n_pos >= n_neg else -1
        elif merge == "first":
            out_signs[i] = signs[0]
        elif merge == "last":
            out_signs[i] = signs[-1]
        else:
            raise ValueError(f"unknown merge strategy: {merge!r}")
    return SignedGraph(
        edges=out_edges, signs=out_signs, n_nodes=g.n_nodes,
    )


def split(g: SignedGraph, train: float = 0.8, val: float = 0.1,
          test: float = 0.1, seed: int = 42) -> tuple:
    """Random edge split; returns (train_idx, val_idx, test_idx)."""
    assert abs(train + val + test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    perm = rng.permutation(g.edges.shape[0])
    n = len(perm)
    i_train = int(n * train)
    i_val = i_train + int(n * val)
    return perm[:i_train], perm[i_train:i_val], perm[i_val:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", nargs="+", default=list(URLS),
                    choices=list(URLS))
    args = ap.parse_args()
    for name in args.download:
        g = load(name)
        s = g.stats()
        print(f"\n{name}: nodes={s['n_nodes']}  edges={s['n_edges']}  "
              f"pos={s['n_pos']}  neg={s['n_neg']}  "
              f"pos_frac={s['pos_frac']:.3f}")
        tr, va, te = split(g)
        print(f"  split: train={len(tr)}  val={len(va)}  test={len(te)}")


if __name__ == "__main__":
    main()
