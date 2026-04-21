#!/usr/bin/env python3
"""Quality audit driver. Produces a JSON summary that the caller post-processes into Markdown.

Coverage:
  1. Fan-out      — name-based grep of callees referenced inside each Rust fn / Python def
  2. Code length  — file SLOC (non-blank non-pure-comment) + per-function NLOC from lizard
  3. CCN          — lizard
  4. Id length    — regex scan per-language
  5. Nesting      — lizard token-based depth proxy (max indent for Python, brace-depth for Rust)
  6. Fog          — textstat on doc-comment blocks + README + docs/*.md
  7. DIT          — Python MRO walk + Rust trait-bound chain
  8. Method fan-in/fan-out — aggregated from (1) filtered to methods
  9. WMC          — sum of CCN over methods in each class/impl
 10. Overloaded methods — same name across multiple impl/class contexts
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
SKIP_DIR_PARTS = {"target", "__pycache__", "generated", ".git", "node_modules",
                  "archive", "steps", "input", "hymeko_core/target"}
SKIP_PATH_SUBSTR = ["/target/", "/__pycache__/", "/generated/",
                    "archive.zip", "/hymeko_query/target/", "data/nn/",
                    "/steps/", "/input/"]
IDENT_LOOP_ALLOW = {"i", "j", "k", "x", "y", "z", "t", "n", "m", "a", "b", "c",
                    "e", "f", "r", "p", "q", "v", "u"}

# Hard short-circuits for cost/noise
MAX_FILES = None  # None = all


def should_skip(p: Path) -> bool:
    s = str(p)
    if any(sub in s for sub in SKIP_PATH_SUBSTR):
        return True
    if any(part in SKIP_DIR_PARTS for part in p.parts):
        return True
    return False


def walk_sources():
    rust, python = [], []
    for base, dirs, files in os.walk(ROOT):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_PARTS]
        for f in files:
            p = Path(base) / f
            if should_skip(p):
                continue
            if f.endswith(".rs"):
                rust.append(p)
            elif f.endswith(".py"):
                python.append(p)
    return rust, python


# ─── Lizard ──────────────────────────────────────────────────────────

def run_lizard(files: list[Path]) -> list[dict[str, Any]]:
    """Return list of per-function dicts."""
    if not files:
        return []
    # Chunk to avoid ARG_MAX
    chunk = 200
    rows: list[dict[str, Any]] = []
    for i in range(0, len(files), chunk):
        batch = [str(p) for p in files[i:i + chunk]]
        try:
            out = subprocess.run(
                ["lizard", "--csv", "-l", "rust", "-l", "python"] + batch,
                capture_output=True, text=True, timeout=120,
            )
        except Exception as e:
            print(f"lizard error: {e}", file=sys.stderr)
            continue
        for line in out.stdout.splitlines():
            # Columns: NLOC,CCN,tokens,params,length,location,file,name,long_name,start,end
            # location and file are quoted; name/long_name quoted.
            parts = parse_csv_row(line)
            if len(parts) < 11:
                continue
            try:
                rows.append({
                    "nloc": int(parts[0]),
                    "ccn": int(parts[1]),
                    "tokens": int(parts[2]),
                    "params": int(parts[3]),
                    "length": int(parts[4]),
                    "location": parts[5],
                    "file": parts[6],
                    "name": parts[7],
                    "long_name": parts[8],
                    "start": int(parts[9]),
                    "end": int(parts[10]),
                })
            except ValueError:
                continue
    return rows


def parse_csv_row(line: str) -> list[str]:
    """Split a lizard CSV row with quoted fields."""
    out = []
    cur = []
    in_quote = False
    for c in line:
        if c == '"':
            in_quote = not in_quote
        elif c == ',' and not in_quote:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


# ─── File-level SLOC ─────────────────────────────────────────────────

RUST_LINE_COMMENT = re.compile(r"^\s*(//|/\*|\*/|\*).*$")
PY_LINE_COMMENT = re.compile(r"^\s*#.*$")


def file_sloc(path: Path, lang: str) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    sloc = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if lang == "rust" and RUST_LINE_COMMENT.match(line):
            continue
        if lang == "python" and PY_LINE_COMMENT.match(line):
            continue
        sloc += 1
    return sloc


# ─── Identifier length ───────────────────────────────────────────────

# Rust declaration keywords + capturing the identifier
RUST_IDENT = re.compile(
    r"\b(?:fn|let(?:\s+mut)?|struct|const|static|type|trait|enum|mod)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def rust_idents(path: Path) -> list[tuple[str, int]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    out = []
    for m in RUST_IDENT.finditer(text):
        name = m.group(1)
        # Compute line
        line = text[:m.start()].count("\n") + 1
        out.append((name, line))
    return out


def python_idents(path: Path) -> list[tuple[str, int]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, node.lineno))
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.append((tgt.id, node.lineno))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append((node.target.id, node.lineno))
    return out


def classify_ident(name: str) -> str | None:
    # `_` and `_foo` are intentional placeholders, not identifiers.
    if name == "_" or name.startswith("_"):
        return None
    if len(name) < 3:
        if name.lower() in IDENT_LOOP_ALLOW:
            return None
        return "short"
    if len(name) > 30:
        return "long"
    return None


# ─── Nesting depth ───────────────────────────────────────────────────

def rust_max_nesting(text: str, start: int, end: int) -> int:
    # Count maximum brace depth between start and end lines.
    lines = text.splitlines()
    depth = 0
    max_depth = 0
    for i in range(start - 1, min(end, len(lines))):
        for c in lines[i]:
            if c == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == '}':
                depth = max(0, depth - 1)
    # Rust function body is inside its own brace; subtract 1 to get internal nesting.
    return max(0, max_depth - 1)


def py_max_nesting(source: str, start: int, end: int) -> int:
    # Use indentation relative to the def line.
    lines = source.splitlines()
    if start - 1 >= len(lines):
        return 0
    def_line = lines[start - 1]
    base_indent = len(def_line) - len(def_line.lstrip(" "))
    max_depth = 0
    for i in range(start, min(end, len(lines))):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(lines[i]) - len(lines[i].lstrip(" "))
        # 4-space convention
        depth = max(0, (indent - base_indent) // 4 - 1)
        max_depth = max(max_depth, depth)
    return max_depth


# ─── Fan-in / fan-out (name-based, cheap) ───────────────────────────

IDENT_TOKEN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")  # ≥3 chars to reduce noise


def extract_body(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[start - 1:end])


def function_callees(body: str, known_fn_names: set[str]) -> set[str]:
    # Tokens inside the body that match any known function name.
    seen = set()
    for m in IDENT_TOKEN.finditer(body):
        name = m.group(1)
        if name in known_fn_names:
            seen.add(name)
    return seen


# ─── OO family ───────────────────────────────────────────────────────

def python_dit_per_class(paths: list[Path]) -> list[tuple[Path, str, int]]:
    results = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (OSError, SyntaxError):
            continue
        class_bases: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)
                class_bases[node.name] = bases
        # Compute DIT locally (bases declared in-file only — imports not resolved)
        def dit(name: str, visited: set[str] | None = None) -> int:
            visited = visited or set()
            if name in visited:
                return 0
            visited = visited | {name}
            bases = class_bases.get(name, [])
            # Skip common base classes that are noise
            bases = [b for b in bases if b not in {"object", "Exception", "TypedDict",
                                                    "Enum", "IntEnum", "Protocol"}]
            if not bases:
                return 0
            return 1 + max(dit(b, visited) for b in bases)
        for name in class_bases:
            d = dit(name)
            if d > 0:
                results.append((p, name, d))
    return results


TRAIT_DEF_RE = re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*([^{]+))?\s*\{",
                           re.MULTILINE)


def rust_trait_graph(paths: list[Path]) -> dict[str, list[str]]:
    """Map trait name → list of super-traits."""
    graph: dict[str, list[str]] = {}
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in TRAIT_DEF_RE.finditer(text):
            name = m.group(1)
            supers_str = m.group(2) or ""
            # Strip lifetimes, where-clauses, type params
            supers = []
            for s in re.split(r"[+,]", supers_str):
                s = s.strip().split("<")[0].strip()
                s = re.sub(r"'[a-z_]+", "", s).strip()
                if s and s[0].isupper():
                    # Keep only last path segment
                    supers.append(s.split("::")[-1])
            graph[name] = supers
    return graph


def rust_dit(graph: dict[str, list[str]]) -> list[tuple[str, int]]:
    memo: dict[str, int] = {}
    def depth(name: str, seen: frozenset[str] = frozenset()) -> int:
        if name in seen:
            return 0
        if name in memo:
            return memo[name]
        bases = graph.get(name, [])
        if not bases:
            memo[name] = 0
            return 0
        d = 1 + max((depth(b, seen | {name}) for b in bases), default=0)
        memo[name] = d
        return d
    return [(name, depth(name)) for name in graph]


# ─── Overloaded methods ──────────────────────────────────────────────

IMPL_RE = re.compile(r"^\s*impl(?:<[^>]+>)?\s+(?:[^{]+?\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)",
                     re.MULTILINE)
FN_IN_IMPL_RE = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
                           re.MULTILINE)


def rust_overloaded(paths: list[Path]) -> list[tuple[str, str, int]]:
    """(type_name, method_name, count) where method is defined in ≥ 3 impl blocks."""
    # Type → method → number of distinct impl blocks defining it.
    per_type: dict[str, Counter] = defaultdict(Counter)
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        # Split on impl ... { blocks — crude but workable.
        i = 0
        while i < len(text):
            m = IMPL_RE.search(text, i)
            if not m:
                break
            type_name = m.group(1)
            # Find opening brace
            brace = text.find("{", m.end())
            if brace == -1:
                break
            # Walk to matching close brace
            depth = 1
            j = brace + 1
            while j < len(text) and depth:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            body = text[brace + 1:j - 1]
            for fm in FN_IN_IMPL_RE.finditer(body):
                per_type[type_name][fm.group(1)] += 1
            i = j
    out = []
    for ty, methods in per_type.items():
        for name, n in methods.items():
            if n >= 3:
                out.append((ty, name, n))
    out.sort(key=lambda t: -t[2])
    return out


# ─── Fog index ───────────────────────────────────────────────────────

def extract_doc_paragraphs(paths: list[Path], kind: str) -> list[tuple[Path, str]]:
    """Return list of (path, paragraph_text) pairs."""
    out = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        paragraphs: list[str] = []
        if kind == "md":
            # Split by blank lines, skip headings, code blocks
            in_code = False
            current: list[str] = []
            for line in text.splitlines():
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if line.strip().startswith("#"):
                    continue
                if not line.strip():
                    if current:
                        paragraphs.append(" ".join(current).strip())
                        current = []
                    continue
                current.append(line.strip())
            if current:
                paragraphs.append(" ".join(current).strip())
        elif kind == "rust":
            # /// doc comments grouped by contiguous run
            current: list[str] = []
            for line in text.splitlines():
                m = re.match(r"^\s*///\s?(.*)$", line)
                if m:
                    current.append(m.group(1))
                else:
                    if current:
                        paragraphs.append(" ".join(current).strip())
                        current = []
            if current:
                paragraphs.append(" ".join(current).strip())
        elif kind == "python":
            # Docstrings
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Module)):
                    ds = ast.get_docstring(node)
                    if ds:
                        paragraphs.append(" ".join(ds.split()))
        for para in paragraphs:
            if len(para.split()) >= 60:
                out.append((p, para))
    return out


def fog(paragraph: str) -> float:
    try:
        import textstat
        return float(textstat.gunning_fog(paragraph))
    except Exception:
        return -1.0


# ─── Main ────────────────────────────────────────────────────────────

def main() -> int:
    rust_files, py_files = walk_sources()
    print(f"[info] {len(rust_files)} Rust + {len(py_files)} Python files", file=sys.stderr)

    funcs = run_lizard(rust_files + py_files)
    print(f"[info] lizard parsed {len(funcs)} functions", file=sys.stderr)

    # Build the known-function-name index for fan-out
    rust_fns = {f["name"] for f in funcs if f["file"].endswith(".rs")}
    py_fns = {f["name"] for f in funcs if f["file"].endswith(".py")}

    # Read file contents once — cached for body extraction.
    file_cache: dict[str, str] = {}
    def read(path_str: str) -> str:
        if path_str not in file_cache:
            try:
                file_cache[path_str] = Path(path_str).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                file_cache[path_str] = ""
        return file_cache[path_str]

    # Fan-out + nesting per function (lizard doesn't give nesting directly, we compute)
    per_fn: list[dict[str, Any]] = []
    fan_in: Counter = Counter()
    for f in funcs:
        text = read(f["file"])
        body = extract_body(text, f["start"], f["end"])
        known = rust_fns if f["file"].endswith(".rs") else py_fns
        callees = function_callees(body, known) - {f["name"]}
        if f["file"].endswith(".rs"):
            nesting = rust_max_nesting(text, f["start"], f["end"])
        else:
            nesting = py_max_nesting(text, f["start"], f["end"])
        for c in callees:
            fan_in[c] += 1
        per_fn.append({
            "file": f["file"].lstrip("./"),
            "name": f["name"],
            "start": f["start"],
            "end": f["end"],
            "nloc": f["nloc"],
            "ccn": f["ccn"],
            "nesting": nesting,
            "fanout": len(callees),
        })

    # Attach fan-in
    for row in per_fn:
        row["fanin"] = fan_in.get(row["name"], 0)

    # File-level SLOC
    file_slocs: list[tuple[str, int, str]] = []
    for p in rust_files:
        s = file_sloc(p, "rust")
        file_slocs.append((str(p.relative_to(ROOT)), s, "rust"))
    for p in py_files:
        s = file_sloc(p, "python")
        file_slocs.append((str(p.relative_to(ROOT)), s, "python"))

    # Identifier length
    idents = {"short": [], "long": []}
    for p in rust_files:
        for name, line in rust_idents(p):
            c = classify_ident(name)
            if c:
                idents[c].append({"file": str(p.relative_to(ROOT)), "line": line,
                                   "name": name, "len": len(name)})
    for p in py_files:
        for name, line in python_idents(p):
            c = classify_ident(name)
            if c:
                idents[c].append({"file": str(p.relative_to(ROOT)), "line": line,
                                   "name": name, "len": len(name)})

    # DIT
    py_dits = python_dit_per_class(py_files)
    rust_traits = rust_trait_graph(rust_files)
    rust_dits = rust_dit(rust_traits)

    # WMC — group lizard rows by containing type
    # For Rust: lizard function name is often just the method name; we need to tie
    # back to the surrounding `impl Type` block. Use IMPL_RE offsets.
    wmc_rust: dict[tuple[str, str], int] = defaultdict(int)
    for p in rust_files:
        text = read(str(p))
        # Find impl blocks and their ranges
        impls: list[tuple[str, int, int]] = []  # (type, start_line, end_line)
        i = 0
        while i < len(text):
            m = IMPL_RE.search(text, i)
            if not m:
                break
            type_name = m.group(1)
            brace = text.find("{", m.end())
            if brace == -1:
                break
            depth = 1
            j = brace + 1
            while j < len(text) and depth:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            start_line = text[:brace].count("\n") + 1
            end_line = text[:j].count("\n") + 1
            impls.append((type_name, start_line, end_line))
            i = j
        for f in funcs:
            if not f["file"].endswith(".rs"):
                continue
            if Path(f["file"]).resolve() != p.resolve():
                continue
            for ty, s, e in impls:
                if s < f["start"] and f["end"] <= e:
                    wmc_rust[(str(p.relative_to(ROOT)), ty)] += f["ccn"]
                    break

    # Python WMC
    wmc_python: dict[tuple[str, str], int] = defaultdict(int)
    for p in py_files:
        try:
            tree = ast.parse(read(str(p)))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Find matching lizard row by file + line range
                        for f in funcs:
                            if f["file"].endswith(".py") and Path(f["file"]).resolve() == p.resolve():
                                if f["start"] >= child.lineno and f["end"] <= (child.end_lineno or child.lineno):
                                    wmc_python[(str(p.relative_to(ROOT)), node.name)] += f["ccn"]
                                    break

    # Overloaded methods (Rust only; Python overloading rare)
    rust_overload = rust_overloaded(rust_files)

    # Fog index
    md_paths = [p for p in Path("docs").rglob("*.md") if not should_skip(p)]
    md_paths += [Path("README.md")] if Path("README.md").exists() else []
    # Exclude auto-generated changelog bodies (historical noise)
    md_paths = [p for p in md_paths if "changelog" not in str(p).lower()]
    fog_samples = []
    for p, para in extract_doc_paragraphs(md_paths, "md"):
        f = fog(para)
        if f >= 0:
            fog_samples.append({"path": str(p), "fog": f, "words": len(para.split())})
    rust_fog = []
    for p, para in extract_doc_paragraphs(rust_files, "rust"):
        f = fog(para)
        if f >= 0:
            rust_fog.append({"path": str(p), "fog": f, "words": len(para.split())})

    summary = {
        "per_fn": per_fn,
        "file_slocs": file_slocs,
        "idents": idents,
        "py_dits": [(str(p), n, d) for p, n, d in py_dits],
        "rust_dits": rust_dits,
        "wmc_rust": [(f, t, v) for (f, t), v in wmc_rust.items()],
        "wmc_python": [(f, t, v) for (f, t), v in wmc_python.items()],
        "rust_overload": rust_overload,
        "fog_md": fog_samples,
        "fog_rust": rust_fog,
        "n_rust_files": len(rust_files),
        "n_py_files": len(py_files),
    }
    json.dump(summary, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
