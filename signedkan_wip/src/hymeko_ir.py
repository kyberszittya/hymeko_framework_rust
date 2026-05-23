"""Shared HyMeKo-IR helpers used by hymeko_driver.py and
hymeko_train_walker.py.  Single source of truth for parsing,
walking, and querying the parsed-dict shape produced by
``hymeko.parse_hymeko_rs`` (the PyO3 wheel)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hymeko  # PyO3 wheel; provides parse_hymeko_rs


def read_hymeko(path: str) -> dict:
    """Parse a .hymeko file via the Rust bridge → nested dict."""
    src = Path(path).read_text()
    return hymeko.parse_hymeko_rs(src)


def all_items(tree: dict) -> list[dict]:
    """Flatten the tree one level into the context body (the canonical
    convention used across all our .hymeko files: a description-name
    wrapper containing one ``context`` block)."""
    out = []
    for it in tree.get("items", []):
        if it["kind"] == "node" and it.get("body"):
            out.extend(it["body"])
        else:
            out.append(it)
    return out


def has_tag(item: dict, tag: str) -> bool:
    return tag in (item.get("tags") or [])


def has_base(item: dict, base_name: str) -> bool:
    """True if the item inherits from ``base_name`` via ``:`` syntax."""
    for b in item.get("bases") or []:
        if b.get("path") and b["path"][-1] == base_name:
            return True
    return False


def child_value(item: dict, child_name: str, default: Any = None) -> Any:
    """Find a child node by name and return its scalar value, with
    surrounding double-quotes stripped if the value is a string."""
    body = item.get("body") or []
    for c in body:
        if c.get("kind") == "node" and c.get("name") == child_name:
            v = c.get("value", default)
            if isinstance(v, str):
                v = v.strip('"')
            return v
    return default


def parse_arch(arch_path: str) -> dict:
    """Walk an architecture .hymeko (e.g. data/hsikan/arch_mixed_k34.hymeko)
    and return the normalized config dict expected by both the driver
    and the train-walker:

        {"hidden": int, "grid": int, "arities": tuple[int, ...],
         "spline_kind": str, "n_layers": int, "name": str}
    """
    tree = read_hymeko(arch_path)
    items = all_items(tree)
    layers = [it for it in items if has_base(it, "signedkan_layer")
              or has_base(it, "walk_layer")]
    if not layers:
        raise ValueError(f"no signedkan_layer / walk_layer in {arch_path}")
    arities = sorted({int(child_value(l, "arity", 3)) for l in layers
                      if has_base(l, "signedkan_layer")})
    hidden = int(child_value(layers[0], "hidden", 16))
    grid = int(child_value(layers[0], "grid", 5))
    spline_kind = child_value(layers[0], "spline_kind", "catmull_rom") \
        or "catmull_rom"
    n_layers = max(
        1,
        len([l for l in layers if has_base(l, "signedkan_layer")
             and int(child_value(l, "arity", 3)) == arities[0]])
    )
    return {
        "hidden": hidden,
        "grid": grid,
        "arities": tuple(arities or [3]),
        "spline_kind": spline_kind,
        "n_layers": n_layers,
        "name": tree.get("name", "HSiKAN"),
    }
