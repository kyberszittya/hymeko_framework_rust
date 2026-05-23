"""Convert a P-graph Studio `.pgip` (SQLite) project to a HyMeKo
P-graph source file.

The `.pgip` schema is the one used by P-graph Studio (Friedler group):

    materials      (id, name, typeId, unitPrice, minFlow, maxFlow)
    materialTypes  (id, name)            # 0=Intermediate, 1=Raw, 2=Product
    units          (id, name, weight, fixCapitalCost, propCapitalCost,
                    fixOperatingCost, propOperatingCost, minSize, maxSize)
    inputOutput    (id, unitId, materialId, isInput, flowRate)

The conversion:
- Material with typeId=1 (Raw)        -> <material, raw>
- Material with typeId=2 (Product)    -> <material, product>
- Material with typeId=0 (Intermediate) -> <material>
- Unit's scalar cost = `weight` (Friedler 1992 form)
- Unit's multi-cost = (fixCapitalCost, propCapitalCost,
                       fixOperatingCost, propOperatingCost) when any are
                       non-zero (Stage P-mo form, alphabetised to
                       [fixCapital, fixOperating, propCapital, propOperating])
- Per-unit body has one signed hyperarc:
    -<material>   for inputs  (isInput=1)
    +<material>   for outputs (isInput=0)

Usage:
    python scripts/pgip_to_hymeko.py \\
        data/pgraph/Chapter3/example3_2.pgip \\
        data/pgraph/Chapter3/example3_2.hymeko
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path


def _sanitize(name: str) -> str:
    """HyMeKo identifiers must match `[A-Za-z_][A-Za-z0-9_]*`. Replace
    every non-conforming character with `_`."""
    if not name:
        return "_"
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not re.match(r"[A-Za-z_]", out[0]):
        out = "_" + out
    return out


def convert(pgip_path: Path, out_path: Path | None = None) -> str:
    conn = sqlite3.connect(str(pgip_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Materials and their type names.
    cur.execute("SELECT id, name, typeId FROM materials ORDER BY id;")
    materials = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT id, name FROM materialTypes;")
    type_name = {r["id"]: r["name"] for r in cur.fetchall()}

    # Units (id -> row).
    cur.execute(
        "SELECT id, name, weight, fixCapitalCost, propCapitalCost, "
        "fixOperatingCost, propOperatingCost FROM units ORDER BY id;"
    )
    units = [dict(r) for r in cur.fetchall()]

    # inputOutput grouped by unit.
    cur.execute(
        "SELECT unitId, materialId, isInput, flowRate FROM inputOutput "
        "ORDER BY unitId, isInput DESC;"
    )
    io_by_unit: dict[int, list[dict]] = {}
    for r in cur.fetchall():
        io_by_unit.setdefault(r["unitId"], []).append(dict(r))

    mat_name = {m["id"]: _sanitize(m["name"]) for m in materials}

    title = _sanitize(pgip_path.stem)
    lines: list[str] = []
    lines.append(f"// Converted from {pgip_path.name} by scripts/pgip_to_hymeko.py")
    lines.append(f"// Source: P-graph Studio .pgip (SQLite) format,")
    lines.append(f"// materials={len(materials)}  units={len(units)}")
    lines.append("//")
    lines.append("// Type-tag mapping:")
    for tid, tname in sorted(type_name.items()):
        lines.append(f"//   typeId={tid} → {tname}")
    lines.append("")
    lines.append(f"{title} {{}}")
    lines.append("")
    lines.append("context")
    lines.append("{")

    # Materials, grouped by typeId for readability.
    for type_id, header in [(1, "raw"), (2, "product"), (0, "intermediate")]:
        items = [m for m in materials if m["typeId"] == type_id]
        if not items:
            continue
        lines.append(f"    // ── {type_name.get(type_id, 'Type'+str(type_id))} materials ──")
        for m in items:
            tags = ["material"]
            if type_id == 1:
                tags.append("raw")
            elif type_id == 2:
                tags.append("product")
            tag_str = ", ".join(tags)
            lines.append(f"    {mat_name[m['id']]} <{tag_str}>;")
        lines.append("")

    # Units.
    lines.append("    // ── Operating units ──")
    for u in units:
        uname = _sanitize(u["name"])
        weight = float(u["weight"])
        # Multi-cost annotation (Stage P-mo) — only emit `cost <dim> N;`
        # children if at least one of the four cost columns is non-zero.
        multi = (
            float(u["fixCapitalCost"])
            + float(u["propCapitalCost"])
            + float(u["fixOperatingCost"])
            + float(u["propOperatingCost"])
        )
        body_lines: list[str] = []
        if multi > 0:
            for (col, dim) in [
                ("fixCapitalCost",      "fixed_capex"),
                ("propCapitalCost",     "prop_capex"),
                ("fixOperatingCost",    "fixed_opex"),
                ("propOperatingCost",   "prop_opex"),
            ]:
                v = float(u[col])
                if v != 0.0:
                    body_lines.append(f"        cost <{dim}> {v};")
        # Hyperarc.
        ios = io_by_unit.get(u["id"], [])
        arc_refs: list[str] = []
        for io in ios:
            sign = "-" if io["isInput"] else "+"
            arc_refs.append(f"{sign}{mat_name[io['materialId']]}")
        body_lines.append(f"        ({', '.join(arc_refs)});")

        lines.append(f"    @{uname} <unit> {weight} {{")
        lines.extend(body_lines)
        lines.append("    }")

    lines.append("}")
    lines.append("")
    content = "\n".join(lines)

    if out_path is not None:
        out_path.write_text(content)
    return content


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pgip", help="Input .pgip file")
    p.add_argument(
        "out",
        nargs="?",
        default=None,
        help="Output .hymeko file (default: same path with .hymeko ext)",
    )
    args = p.parse_args()

    src = Path(args.pgip)
    if not src.is_file():
        print(f"error: {src} not found", file=sys.stderr)
        sys.exit(1)
    dst = Path(args.out) if args.out else src.with_suffix(".hymeko")
    txt = convert(src, dst)
    print(f"wrote {dst}  ({len(txt)} bytes, {txt.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
