"""IMDB Large Movie Review dataset — manual downloader + Hungarian-format loader.

Standalone implementation (no torchtext / HuggingFace dependency).
Mirrors the Reddit Hyperlinks pattern from 2026-05-12: when the
upstream library doesn't ship in CORE, fall back to curl + tarball.

Source: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
(Maas et al., ACL 2011; the canonical IMDB binary-sentiment corpus.)

After extraction:
    aclImdb/train/pos/*.txt   12,500 positive reviews
    aclImdb/train/neg/*.txt   12,500 negative reviews
    aclImdb/test/pos/*.txt    12,500 positive
    aclImdb/test/neg/*.txt    12,500 negative
    aclImdb/imdb.vocab        88,584-line word list (not used; we build our own)

Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


# Canonical Stanford URL — has been stable since 2011.
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_TARBALL = "aclImdb_v1.tar.gz"
IMDB_ROOT = "aclImdb"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def download_imdb(root: Path | str = "data/imdb",
                  url: str = IMDB_URL,
                  timeout: int = 600) -> Path:
    """Download + extract aclImdb if not already present.

    Idempotent: if ``<root>/aclImdb/`` exists with both train/ and
    test/ subtrees, returns immediately. Otherwise curl-fetches the
    84 MB tarball and untars it.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    aclimdb_dir = root / IMDB_ROOT
    if (aclimdb_dir / "train" / "pos").is_dir() and \
       (aclimdb_dir / "test" / "pos").is_dir():
        return aclimdb_dir
    tarball = root / IMDB_TARBALL
    if not tarball.exists():
        print(f"[imdb] downloading {url} → {tarball}")
        subprocess.run(
            ["curl", "-L", "-o", str(tarball), url],
            check=True, timeout=timeout,
        )
    print(f"[imdb] extracting {tarball} → {root}")
    subprocess.run(
        ["tar", "-xzf", str(tarball), "-C", str(root)],
        check=True, timeout=timeout,
    )
    if not aclimdb_dir.is_dir():
        raise RuntimeError(
            f"aclImdb extraction failed: {aclimdb_dir} not present"
        )
    return aclimdb_dir


# Whitespace + HTML tag stripping. IMDB reviews use <br /> as line break.
_HTML_RE = re.compile(r"<[^>]+>")
_SPLIT_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """Cleanup + whitespace tokenisation.

    1. Strip HTML tags (mostly ``<br />``).
    2. Lowercase.
    3. Split on whitespace.

    Keeps punctuation attached to tokens (e.g. ``"good."``); this
    is the standard ``torchtext.basic_english``-light protocol.
    """
    text = _HTML_RE.sub(" ", text)
    return [t for t in _SPLIT_RE.split(text.lower().strip()) if t]


def _iter_split(aclimdb_dir: Path, split: str
                ) -> Iterable[tuple[int, list[str]]]:
    """Yield (label, tokens) for every file under <aclimdb>/<split>/{pos,neg}.

    Label is 1 for pos, 0 for neg.
    Iteration order is deterministic (sorted filenames).
    """
    for label, sub in [(1, "pos"), (0, "neg")]:
        d = aclimdb_dir / split / sub
        if not d.is_dir():
            raise FileNotFoundError(f"IMDB split missing: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix != ".txt":
                continue
            yield label, tokenize(p.read_text(encoding="utf-8", errors="replace"))


def build_imdb_vocab(
    aclimdb_dir: Path,
    vocab_size: int = 20_000,
    min_freq: int = 2,
) -> dict[str, int]:
    """Build a frequency-based vocabulary from the train split.

    - Token 0 is reserved for ``<pad>``.
    - Token 1 is reserved for ``<unk>``.
    - The remaining (vocab_size - 2) slots are filled with the most
      frequent tokens (with frequency >= min_freq).
    - Ties on frequency are broken alphabetically (deterministic).
    """
    counter: Counter[str] = Counter()
    for _label, tokens in _iter_split(aclimdb_dir, "train"):
        counter.update(tokens)
    # Sort by (-count, token) so ties break alphabetically.
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for tok, cnt in ordered:
        if len(vocab) >= vocab_size:
            break
        if cnt < min_freq:
            break
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int],
    L_max: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Map tokens → ids of length L_max, truncating or right-padding.

    Returns (ids, mask) — both shape (L_max,), dtype int64 / bool.
    The mask is 1 where the position is a real token, 0 where padding.
    """
    ids = np.full((L_max,), PAD_ID, dtype=np.int64)
    mask = np.zeros((L_max,), dtype=bool)
    truncated = tokens[:L_max]
    for i, t in enumerate(truncated):
        ids[i] = vocab.get(t, UNK_ID)
        mask[i] = True
    return ids, mask


@dataclass
class IMDBSplit:
    """In-memory IMDB split with tokens already vocab-encoded."""
    ids: np.ndarray   # (N, L_max) int64
    mask: np.ndarray  # (N, L_max) bool
    labels: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def shuffle(self, seed: int) -> "IMDBSplit":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        return IMDBSplit(
            ids=self.ids[perm], mask=self.mask[perm], labels=self.labels[perm],
        )


def materialise_split(
    aclimdb_dir: Path,
    split: str,
    vocab: dict[str, int],
    L_max: int = 200,
) -> IMDBSplit:
    """Read all files in <split>, encode, pack into an IMDBSplit."""
    ids_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    labels: list[int] = []
    for label, tokens in _iter_split(aclimdb_dir, split):
        ids, mask = encode_tokens(tokens, vocab, L_max=L_max)
        ids_rows.append(ids)
        mask_rows.append(mask)
        labels.append(label)
    return IMDBSplit(
        ids=np.stack(ids_rows, axis=0),
        mask=np.stack(mask_rows, axis=0),
        labels=np.array(labels, dtype=np.int64),
    )


def load_imdb(
    root: Path | str = "data/imdb",
    vocab_size: int = 20_000,
    L_max: int = 200,
    min_freq: int = 2,
    download: bool = True,
) -> tuple[IMDBSplit, IMDBSplit, dict[str, int]]:
    """End-to-end loader: returns (train, test, vocab).

    Caches the materialised splits + vocab under ``<root>/cache/`` keyed
    by ``(vocab_size, L_max, min_freq)``. Second invocation is ~5 s
    (read the cached .npz) instead of ~60 s (re-iterate 50 k files).
    """
    root = Path(root)
    aclimdb_dir = download_imdb(root) if download else (root / IMDB_ROOT)
    if not aclimdb_dir.is_dir():
        raise FileNotFoundError(
            f"{aclimdb_dir} not found; pass download=True or extract manually"
        )
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"vs{vocab_size}_L{L_max}_mf{min_freq}"
    train_cache = cache_dir / f"train_{key}.npz"
    test_cache = cache_dir / f"test_{key}.npz"
    vocab_cache = cache_dir / f"vocab_{key}.json"
    if train_cache.exists() and test_cache.exists() and vocab_cache.exists():
        with vocab_cache.open() as f:
            vocab = json.load(f)
        tr = np.load(train_cache)
        te = np.load(test_cache)
        return (
            IMDBSplit(ids=tr["ids"], mask=tr["mask"], labels=tr["labels"]),
            IMDBSplit(ids=te["ids"], mask=te["mask"], labels=te["labels"]),
            vocab,
        )
    vocab = build_imdb_vocab(aclimdb_dir, vocab_size=vocab_size, min_freq=min_freq)
    train = materialise_split(aclimdb_dir, "train", vocab, L_max=L_max)
    test = materialise_split(aclimdb_dir, "test", vocab, L_max=L_max)
    np.savez_compressed(train_cache,
                         ids=train.ids, mask=train.mask, labels=train.labels)
    np.savez_compressed(test_cache,
                         ids=test.ids, mask=test.mask, labels=test.labels)
    with vocab_cache.open("w") as f:
        json.dump(vocab, f)
    return train, test, vocab


def split_to_tensors(s: IMDBSplit, device: str | torch.device = "cpu"
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an IMDBSplit to (ids, mask, labels) GPU tensors."""
    return (
        torch.from_numpy(s.ids).to(device),
        torch.from_numpy(s.mask).to(device),
        torch.from_numpy(s.labels).to(device),
    )
