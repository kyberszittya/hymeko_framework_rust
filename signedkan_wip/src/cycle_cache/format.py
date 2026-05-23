"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..runtime_config import get_runtime
from .pack import _pack_and_drop, _unpack_to_ntuples

def _cache_format() -> str:
    """Disk format for new cache writes.  Reads auto-detect from the
    file's magic bytes / suffix, so existing `.npz` caches continue
    to work regardless of this setting.

    Set ``HYMEKO_CACHE_FORMAT=cbor`` to write the cross-language
    CBOR format (RFC 8949).  Default is ``npz`` (legacy)."""
    return get_runtime().cycle_cache.cache_format.lower()


# CBOR top-level map keys (stable wire format, version 1).
# The on-disk layout is:
#   {
#     "format_version": uint8 = 1,
#     "v_shape":  [n_cycles, k]   (array of int),
#     "v_dtype":  "int64",
#     "v_buf":    bytes (n_cycles * k * 8 bytes, little-endian int64),
#     "sigma_shape": [n_cycles, k],
#     "sigma_dtype": "int8",
#     "sigma_buf":   bytes (n_cycles * k bytes),
#     "edge_signs_shape": [n_cycles, k] | null,
#     "edge_signs_dtype": "int8" | null,
#     "edge_signs_buf":   bytes | null,
#   }
# Rust readers (ciborium / serde_cbor) get the same map; the byte
# buffers are np.frombuffer-equivalent on the Rust side.
_CBOR_FORMAT_VERSION = 1


def _save_packed_cbor(path: pathlib.Path, v: np.ndarray, sigma: np.ndarray,
                       edge_signs: np.ndarray | None) -> None:
    import cbor2
    v = np.ascontiguousarray(v, dtype=np.int64)
    sigma = np.ascontiguousarray(sigma, dtype=np.int8)
    payload: dict = {
        "format_version": _CBOR_FORMAT_VERSION,
        "v_shape":  list(v.shape),
        "v_dtype":  str(v.dtype),
        "v_buf":    v.tobytes(),
        "sigma_shape": list(sigma.shape),
        "sigma_dtype": str(sigma.dtype),
        "sigma_buf":   sigma.tobytes(),
        "edge_signs_shape": None,
        "edge_signs_dtype": None,
        "edge_signs_buf":   None,
    }
    if edge_signs is not None:
        es = np.ascontiguousarray(edge_signs, dtype=np.int8)
        payload["edge_signs_shape"] = list(es.shape)
        payload["edge_signs_dtype"] = str(es.dtype)
        payload["edge_signs_buf"]   = es.tobytes()
    with path.open("wb") as fh:
        cbor2.dump(payload, fh)


def _load_packed_cbor(
    path: pathlib.Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import cbor2
    with path.open("rb") as fh:
        payload = cbor2.load(fh)
    fmt_v = int(payload.get("format_version", 1))
    if fmt_v != _CBOR_FORMAT_VERSION:
        raise ValueError(
            f"unsupported cbor cache format version {fmt_v}; "
            f"expected {_CBOR_FORMAT_VERSION}"
        )
    v = np.frombuffer(
        payload["v_buf"], dtype=payload["v_dtype"],
    ).reshape(tuple(payload["v_shape"]))
    sigma = np.frombuffer(
        payload["sigma_buf"], dtype=payload["sigma_dtype"],
    ).reshape(tuple(payload["sigma_shape"]))
    if payload.get("edge_signs_buf") is None:
        edge_signs = None
    else:
        edge_signs = np.frombuffer(
            payload["edge_signs_buf"], dtype=payload["edge_signs_dtype"],
        ).reshape(tuple(payload["edge_signs_shape"]))
    return v, sigma, edge_signs


def _detect_format(path: pathlib.Path) -> str:
    """Auto-detect cache format from extension; fall back to .npz
    detection by magic bytes for the (legacy, non-suffixed) case."""
    if path.suffix == ".cbor":
        return "cbor"
    if path.suffix == ".npz":
        return "npz"
    # Fall back: peek at magic bytes.  CBOR has no fixed magic; .npz
    # starts with the PK\x03\x04 ZIP signature.
    try:
        with path.open("rb") as fh:
            head = fh.read(4)
        if head[:2] == b"PK":
            return "npz"
        return "cbor"
    except FileNotFoundError:
        return "npz"


def _save_packed(path: pathlib.Path, v: np.ndarray, sigma: np.ndarray,
                  edge_signs: np.ndarray) -> None:
    fmt = _cache_format()
    if fmt == "cbor":
        # Write to `<key>.cbor` regardless of the path's `.npz` suffix
        # produced by the historical caller — that's the only place
        # the suffix differs.
        out_path = path.with_suffix(".cbor")
        _save_packed_cbor(out_path, v, sigma, edge_signs)
    else:
        np.savez(path, v=v, sigma=sigma, edge_signs=edge_signs)


def _load_packed(path: pathlib.Path
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # Auto-detect format from the on-disk file.  Try .cbor first if
    # the caller passed a .npz path that no longer exists (i.e., the
    # cache was last written under HYMEKO_CACHE_FORMAT=cbor).
    if not path.exists():
        cbor_path = path.with_suffix(".cbor")
        if cbor_path.exists():
            return _load_packed_cbor(cbor_path)
    fmt = _detect_format(path)
    if fmt == "cbor":
        return _load_packed_cbor(path)
    arr = np.load(path)
    v = arr["v"]
    sigma = arr["sigma"]
    edge_signs = arr["edge_signs"] if "edge_signs" in arr.files else None
    return v, sigma, edge_signs


