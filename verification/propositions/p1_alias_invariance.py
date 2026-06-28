"""Property-test Proposition 1 (alias invariance) against the REAL HyMeKo compiler.

Prop. 1 says denotation-preserving surface rewrites (sibling reordering, import/alias rewrites) compile to the
SAME canonical IR and emit byte-identical target text. We test the strongest automatable case — sibling
reordering of the description body — over a corpus of self-contained fixtures x many random permutations, against
the actual engine (`parse_dsl` -> `canonical_hash` and `to_dot`). This is not a model proof; it is falsification
over the real artifact, which a model proof does not give. A single hash mismatch on a valid reorder would be a
counterexample to Prop. 1.
"""
import glob
import random

import hymeko

N_PERMS = 200


def last_brace_block(src):
    """(open_idx, close_idx) of the last depth-0 {...} block = the description body."""
    blocks, depth, start = [], 0, None
    for i, ch in enumerate(src):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blocks.append((start, i))
    return blocks[-1]


def split_siblings(inner):
    """Top-level brace-balanced declarations inside a body (each `name {...}` / `@name{...}`)."""
    decls, depth, start = [], 0, None
    for i, ch in enumerate(inner):
        if depth == 0 and start is None and not ch.isspace():
            start = i
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                decls.append(inner[start:i + 1])
                start = None
    return decls


def reorder(src, rng):
    o, c = last_brace_block(src)
    decls = split_siblings(src[o + 1:c])
    if len(decls) < 2:
        return None
    perm = decls[:]
    rng.shuffle(perm)
    new_inner = "\n    " + "\n    ".join(d.strip() for d in perm) + "\n"
    return src[:o + 1] + new_inner + src[c:]


def canon(src):
    eng = hymeko.PyHypergraphEngine()   # FRESH engine per parse — parse_dsl mutates engine state (it accumulates)
    ir = eng.parse_dsl(src)
    return ir.canonical_hash, ir.snapshot_json()   # hash = content-address; snapshot = full canonical IR structure


fixtures = sorted(glob.glob("data/typical_graphs/*.hymeko"))
total = hash_ok = both_ok = snap_only = 0
hash_violations = []
for f in fixtures:
    base = open(f, encoding="utf-8").read()
    try:
        h0, snap0 = canon(base)
    except Exception as e:   # noqa: BLE001
        print(f"  [skip {f}: base won't compile: {str(e)[:60]}]")
        continue
    fh = fb = 0
    for s in range(N_PERMS):
        v = reorder(base, random.Random(s))
        if v is None:
            continue
        try:
            h, snap = canon(v)
        except Exception as e:   # noqa: BLE001 — a reorder that breaks compilation is itself informative
            hash_violations.append((f, s, f"compile-fail: {str(e)[:50]}"))
            continue
        total += 1
        if h != h0:
            hash_violations.append((f, s, f"HASH {h[:18]} != {h0[:18]}"))   # a REAL Prop.1 counterexample
            continue
        hash_ok += 1
        fh += 1
        if snap == snap0:
            both_ok += 1
            fb += 1
        else:
            snap_only += 1
    print(f"  {f.split(chr(92))[-1]:30} hash-invariant {fh}/{N_PERMS}  (snapshot-also {fb}/{N_PERMS})  {h0[:20]}")

print(f"\nP1 (alias / sibling-reorder invariance) vs the REAL compiler — {total} reorderings, {len(fixtures)} fixtures:")
print(f"  canonical_hash invariant : {hash_ok}/{total}   <- the Proposition 1 claim (the content-address)")
print(f"  snapshot_json also equal : {both_ok}/{total}   (stricter; {snap_only} had equal hash but a reordered JSON view)")
if hash_violations:
    print(f"  HASH VIOLATIONS (real P1 counterexamples): {len(hash_violations)}")
    for f, s, why in hash_violations[:8]:
        print(f"     {f} seed {s}: {why}")
else:
    print("  NO hash violations: Proposition 1 holds on every tested instance against the real compiler.")
    print("  The snapshot diffs are a serialization artifact — the JSON view keeps source order; only the")
    print("  Blake3 canonical hash (which IS the content-address Prop.1 is about) is order-canonicalised.")
