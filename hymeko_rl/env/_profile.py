"""Narrow readers for ``.hymeko`` profiles — shared by the observation and reward specs.

Both specs read a *bundle* (an ``observation_space`` / ``reward_spec`` / ``termination_spec``) and its
ordered member channels/terms. This is a documented **bridge** until the engine's import-resolving
snapshot is exposed to Python (B-003): the CLI has no typed structured IR dump (``--json`` only on
``entropy``/``rewrite``; ``inspect`` prints ``kind=Node/Edge`` without the semantic kind), so a parse
of the regular profile form (``@name: ns.kind { body }`` + a bundle's ``(+ a, + b …)`` arc) is the
robust option. It does **not** parse arbitrary HyMeKo — only this profile shape.

**Import-aware (2026-06-19, APPROVED-CORE-EDIT: xprofile-instance-refs).** Since the engine now
resolves cross-profile instance references, this reader follows ``@"…"`` imports and merges their
``@``-decls, so a bundle member declared in an imported profile (referenced as ``alias.member`` via a
``using <desc>.<content> as alias``) resolves — mirroring the compiler, not diverging from it. The
last dotted segment of a member is the decl name (``arr.dist`` → ``dist``); local decls shadow imports.
"""
from __future__ import annotations

import re
from pathlib import Path

# @name: <ns-path>.kind { body }  — the last dotted segment is the channel/term kind.
_DECL = re.compile(r"@(\w+)\s*:\s*[\w.]*\.(\w+)\s*\{(.*?)\}", re.DOTALL)
# @"some/file.hymeko"  — an import directive.
_IMPORT = re.compile(r'@"([^"]+)"')


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())


def _local_decls(body: str) -> dict[str, tuple[str, str]]:
    return {m.group(1): (m.group(2), m.group(3)) for m in _DECL.finditer(body)}


def _gather_decls(path: Path, _seen: set[Path] | None = None) -> dict[str, tuple[str, str]]:
    """All ``@``-decls reachable from ``path`` — its own plus those of its (transitively) imported
    files. Imported decls are merged first so a file's **local** decls shadow imported same-name
    decls. Import cycles terminate via the visited set; a missing import file is skipped silently
    (the compiler reports it — the shim need not duplicate that error)."""
    _seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in _seen or not path.is_file():
        return {}
    _seen.add(path)
    body = _strip_comments(path.read_text(encoding="utf-8"))  # .hymeko is UTF-8, not the OS locale
    decls: dict[str, tuple[str, str]] = {}
    for imp in _IMPORT.findall(body):
        decls.update(_gather_decls(path.parent / imp, _seen))
    decls.update(_local_decls(body))  # local overrides imported
    return decls


def read_scene_fields(profile: str | Path, kind: str = "scene") -> dict[str, float | tuple[float, ...]]:
    """Read a scenario profile's single ``@name: ns.<kind> { … }`` body → ``{field: scalar | tuple}``.

    Field syntax is the HyMeKo value form already used by scene/robot profiles: ``name 0.12;`` (scalar) and
    ``name [a, b, c];`` (vector). A bracketed value becomes a ``tuple[float, …]``; a bare number a ``float``.
    This is the same documented bridge as :func:`read_bundle` (regex over the profile shape, not a full parse).

    # Preconditions ``profile`` exists and declares exactly one ``<kind>`` instance.
    # Postconditions Returns one entry per ``name value;`` field, in file order.
    # Errors ``FileNotFoundError``; ``ValueError`` (no ``<kind>`` declaration).
    """
    text = _strip_comments(Path(profile).read_text(encoding="utf-8"))
    decl = re.search(rf"@\w+\s*:\s*[\w.]*\.{kind}\s*\{{(.*?)\}}", text, re.DOTALL)
    if decl is None:
        raise ValueError(f"{profile}: no '{kind}' declaration found")
    out: dict[str, float | tuple[float, ...]] = {}
    for fm in re.finditer(r"(\w+)\s+(\[[^\]]*\]|-?[\d.]+)\s*;", decl.group(1)):
        name, val = fm.group(1), fm.group(2)
        out[name] = (tuple(float(x) for x in val[1:-1].split(",")) if val.startswith("[")
                     else float(val))
    return out


def read_imports(profile: str | Path) -> list[Path]:
    """The ``@"…"`` import paths declared in ``profile``, each resolved relative to it (existence not checked)."""
    text = _strip_comments(Path(profile).read_text(encoding="utf-8"))
    return [Path(profile).parent / imp for imp in _IMPORT.findall(text)]


def read_decls(profile: str | Path) -> list[tuple[str, str]]:
    """The ``@name: ns.kind { … }`` declarations made locally in ``profile`` → ``(name, kind)`` pairs, in file
    order (the last dotted segment of the type path is the kind). What this file *contributes* to a model."""
    text = _strip_comments(Path(profile).read_text(encoding="utf-8"))
    return [(m.group(1), m.group(2)) for m in _DECL.finditer(text)]


def read_bundle(profile: str | Path, spec_kind: str) -> list[tuple[str, str, str, float | None]]:
    """Read a profile's single ``spec_kind`` bundle → its members as ordered
    ``(name, kind, body, arc_weight)`` quadruples, in the bundle's arc order. Members may be declared
    in imported profiles (resolved by last dotted segment, e.g. ``arr.dist`` → ``dist``).

    ``arc_weight`` is the optional numeric annotation carried by the bundle's arc itself
    (``(+ approach 4.0, …)``) — the hyperedge defines the weight of each relationship, rather than the
    member node carrying a ``weight`` attribute. ``None`` when the arc has no weight (e.g.
    ``observation_space`` channels, or a reward bundle still using the body-``weight`` form).

    # Preconditions ``profile`` exists and declares exactly one ``spec_kind``; every arc member is a
    declared ``@``-edge (here or in an imported profile).
    # Postconditions Order follows the bundle's ``(+ … )`` arc, not declaration order.
    # Errors ``FileNotFoundError``; ``ValueError`` (no/multiple ``spec_kind``, empty bundle,
    undeclared member).
    """
    profile = Path(profile)
    own_body = _strip_comments(profile.read_text(encoding="utf-8"))
    own_decls = _local_decls(own_body)            # the bundle itself must be declared locally
    decls = _gather_decls(profile)                # members may be local or imported

    specs = [n for n, (k, _) in own_decls.items() if k == spec_kind]
    if len(specs) != 1:
        raise ValueError(
            f"{profile}: expected exactly one {spec_kind}, found {specs or 'none'}")
    # arc members: ``+ ref [weight]`` — the optional trailing number is the arc's weight (the hyperedge
    # carries the weight, not the member node). Keep the full dotted ref; the last segment is the decl name.
    parsed = re.findall(r"\+\s*([\w.]+)(?:\s+(-?\d+(?:\.\d+)?))?", own_decls[specs[0]][1])
    if not parsed:
        raise ValueError(f"{profile}: {spec_kind} {specs[0]!r} has no member channels")
    members = [(ref.split(".")[-1], float(w) if w else None) for ref, w in parsed]
    missing = [m for m, _ in members if m not in decls]
    if missing:
        raise ValueError(f"{profile}: {spec_kind} members {missing} are not declared (here or imported)")
    return [(m, decls[m][0], decls[m][1], w) for m, w in members]


# Signed arc with an optional numeric weight: ``+ member 4.0`` / ``- member`` / ``~ member 0.5``.
_SIGNED_ARC = re.compile(r"([+\-~])\s*([\w.]+)(?:\s+(-?\d+(?:\.\d+)?))?")


def read_arc_weights(profile: str | Path, edge_name: str) -> list[tuple[str, str, float | None]]:
    """Read the signed arcs **and their weights** of a named hyperedge — the general HyMeKo arc-weight
    capability (reward/observation/strategy bundles are specific uses via :func:`read_bundle`; this reads any
    declared hyperedge by name).

    ``@edge_name: ns.kind { (+ a 4.0, - b, ~ c 0.5); }`` → ``[("+", "a", 4.0), ("-", "b", None), ("~", "c", 0.5)]``.
    The weight is a HyMeKo arc annotation (``RefAtom.anno.value`` = ``Value::Num``); a missing weight is ``None``.
    The member is the last dotted segment of the reference (``arr.dist`` → ``dist``). Non-numeric arc payloads
    (e.g. a joint's ``[[pos],[rpy]]`` transform) are read as weightless (``None``).

    # Preconditions ``profile`` exists and declares ``@edge_name``.
    # Postconditions one ``(sign, member, weight)`` triple per arc, in arc order.
    # Errors ``FileNotFoundError``; ``ValueError`` (no such hyperedge, or it has no arcs).
    """
    text = _strip_comments(Path(profile).read_text(encoding="utf-8"))
    decl = re.search(rf"@{re.escape(edge_name)}\s*:\s*[^{{]*?\{{(.*?)\}}", text, re.DOTALL)
    if decl is None:
        raise ValueError(f"{profile}: no hyperedge named {edge_name!r}")
    arcs = _SIGNED_ARC.findall(decl.group(1))
    if not arcs:
        raise ValueError(f"{profile}: hyperedge {edge_name!r} has no arcs")
    return [(sign, ref.split(".")[-1], float(w) if w else None) for sign, ref, w in arcs]
