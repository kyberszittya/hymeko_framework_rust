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


def read_bundle(profile: str | Path, spec_kind: str) -> list[tuple[str, str, str]]:
    """Read a profile's single ``spec_kind`` bundle → its members as ordered ``(name, kind, body)``
    triples, in the bundle's arc order. Members may be declared in imported profiles (resolved by
    last dotted segment, e.g. ``arr.dist`` → ``dist``).

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
    # arc members; keep the full dotted ref, then take the last segment as the decl name.
    refs = re.findall(r"\+\s*([\w.]+)", own_decls[specs[0]][1])
    if not refs:
        raise ValueError(f"{profile}: {spec_kind} {specs[0]!r} has no member channels")
    members = [r.split(".")[-1] for r in refs]
    missing = [m for m in members if m not in decls]
    if missing:
        raise ValueError(f"{profile}: {spec_kind} members {missing} are not declared (here or imported)")
    return [(m, decls[m][0], decls[m][1]) for m in members]
