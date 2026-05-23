#!/usr/bin/env python3
"""
generate_fixtures.py — synthetic HyMeKo fixture generator for the
scaling study.

Adapted from the original draft artefact to emit the *real* HyMeKo
surface syntax used in this workspace (see data/robotics/mini_arm.hymeko).
Each fixture imports `meta_kinematics.hymeko`; the generator copies that
file (and any siblings it pulls in via `@"..."` imports) into the
fixtures root so the `ModuleStore` path resolver can find them.

Three families:

  chain(n)        : serial kinematic chain, n links + n revolute joints.
                    Every hyperedge arity-2, d̄ = 2.

  tree(n, k)      : rooted tree of n links, branching factor k,
                    n revolute joints. Realistic robot morphology proxy.

  highArity(m, d) : m hyperedges each of arity d over a shared vertex
                    pool of ⌈md/2⌉, stress fixture for Proposition 4's
                    ρ→1 asymptote. Uses `conti_joint` as the carrier
                    hyperedge type so the root decl still compiles; the
                    bench harness skips the robotics emitters for this
                    family because the output isn't a well-typed robot.

Each fixture is written to <out>/<family>/<name>/<name>.hymeko with a
sibling copy of meta_kinematics.hymeko (self-contained unit of
compilation). An index.json manifest captures (|V|, |E|, mean_arity,
bytes) per fixture.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Surface-syntax templates                                                    #
# --------------------------------------------------------------------------- #

HEADER_TMPL_LONG = (
    '{root}_description {{\n'
    '    @"meta_kinematics.hymeko";\n'
    '}}\n'
    '\n'
    '{root}: meta_kinematics.kinematics.elements,\n'
    '        meta_kinematics.kinematics.geometry,\n'
    '        meta_kinematics.kinematics.axes\n'
    '{{\n'
)

# Aliased (idiomatic) form: declare short aliases for the repeated
# meta_kinematics.kinematics.* paths in the description header, then
# use them throughout. This mirrors the hand-written fixture pattern
# in data/robotics/anthropomorphic_arm_using.hymeko.
HEADER_TMPL_ALIASED = (
    '{root}_description {{\n'
    '    @"meta_kinematics.hymeko";\n'
    '    using kinematics.elements as el;\n'
    '    using kinematics.geometry as geo;\n'
    '    using kinematics.axes as ax;\n'
    '    using kinematics.conti_joint as cj;\n'
    '}}\n'
    '\n'
    '{root}: el, geo, ax\n'
    '{{\n'
)

# Current default: retain the long (naive) form. Aliased output is
# opt-in via --aliased so the original measurement baseline is not
# silently altered. Toggled via `set_aliased(True)` before generation.
_ALIASED = False

def set_aliased(value: bool) -> None:
    global _ALIASED
    _ALIASED = value

def _header() -> str:
    return HEADER_TMPL_ALIASED if _ALIASED else HEADER_TMPL_LONG

# Legacy alias for call sites that still reference HEADER_TMPL.format(...)
# — they route through _header() so aliased mode is picked up.
class _HeaderProxy:
    def format(self, **kw): return _header().format(**kw)
HEADER_TMPL = _HeaderProxy()

FOOTER = '}\n'


def link_block(name: str, mass: float, ox: float, oy: float, oz: float,
               shape: str = "box", dims: tuple[float, ...] = (0.1, 0.1, 0.1),
               aliased: bool | None = None) -> str:
    if aliased is None:
        aliased = _ALIASED
    """Emit a link declaration with nested geometry + visual/collision refs."""
    dim_list = ", ".join(f"{d:.3f}" for d in dims)
    link_type = "el.link" if aliased else "meta_kinematics.kinematics.elements.link"
    geom_type = (f"geo.{shape}" if aliased
                 else f"meta_kinematics.kinematics.geometry.{shape}")
    return (
        f"    {name}: {link_type} {{\n"
        f"        mass {mass:.3f};\n"
        f"        link_geometry: {geom_type} {{\n"
        f"            dimension [{dim_list}];\n"
        f"        }}\n"
        f"        visual    -> link_geometry;\n"
        f"        collision -> link_geometry;\n"
        f"        origin [{ox:.3f}, {oy:.3f}, {oz:.3f}];\n"
        f"    }}\n"
    )


def revolute_joint_block(name: str, parent: str, child: str,
                         axis: str = "AXIS_Z",
                         origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.1),
                         origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
                         aliased: bool | None = None) -> str:
    if aliased is None:
        aliased = _ALIASED
    ox, oy, oz = origin_xyz
    rx, ry, rz = origin_rpy
    joint_type = "cj" if aliased else "meta_kinematics.kinematics.conti_joint"
    axis_ref = f"ax.{axis}" if aliased else f"meta_kinematics.kinematics.axes.{axis}"
    return (
        f"    @{name}: {joint_type} {{\n"
        f"        (+ {parent} [[{ox:.3f}, {oy:.3f}, {oz:.3f}], [{rx:.3f}, {ry:.3f}, {rz:.3f}]],\n"
        f"         - {child},\n"
        f"         - {axis_ref});\n"
        f"    }}\n"
    )


def high_arity_hyperedge_block(name: str, participants: list[tuple[str, str]],
                               aliased: bool | None = None) -> str:
    if aliased is None:
        aliased = _ALIASED
    """
    Emit a hyperedge of arbitrary arity as a `conti_joint`, with each
    participant signed `+` or `-` per the provided list. Used only for
    the highArity stress fixtures — the emitters don't run on this
    family.
    """
    joint_type = "cj" if aliased else "meta_kinematics.kinematics.conti_joint"
    axis_ref = "ax.AXIS_Z" if aliased else "meta_kinematics.kinematics.axes.AXIS_Z"
    lines = [f"    @{name}: {joint_type} {{"]
    parts = [f"{sign} {vname}" for (sign, vname) in participants]
    parts.append(f"- {axis_ref}")
    inner = ",\n         ".join(parts)
    lines.append(f"        ({inner});")
    lines.append("    }")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Fixture statistics                                                          #
# --------------------------------------------------------------------------- #

@dataclass
class FixtureStats:
    family: str
    name: str
    n_vertices: int
    n_hyperedges: int
    mean_arity: float
    source_bytes: int
    path: str
    robot_name: str


# --------------------------------------------------------------------------- #
# Generators                                                                  #
# --------------------------------------------------------------------------- #

def gen_chain(n: int, seed: int = 0) -> tuple[str, FixtureStats]:
    """Serial chain: l0 → l1 → ... → l_{n-1} via revolute joints."""
    rng = random.Random(seed + n)
    root = f"chain_{n}"
    lines = [HEADER_TMPL.format(root=root)]

    for i in range(n):
        lines.append(link_block(
            f"l{i}",
            mass=rng.uniform(0.1, 5.0),
            ox=rng.uniform(-0.1, 0.1),
            oy=0.0,
            oz=i * 0.1,
            shape="cylinder",
            dims=(0.05, 0.1),
        ))

    # n-1 internal joints connecting consecutive links
    for i in range(n - 1):
        lines.append(revolute_joint_block(
            f"j{i}", parent=f"l{i}", child=f"l{i+1}",
        ))

    lines.append(FOOTER)
    src = "".join(lines)

    n_v = n              # n links (no world anchor — root decl itself is the anchor)
    n_e = max(n - 1, 0)
    return src, FixtureStats(
        family="chain", name=f"chain_{n}", n_vertices=n_v, n_hyperedges=n_e,
        mean_arity=2.0 if n_e > 0 else 0.0,
        source_bytes=len(src.encode()), path="", robot_name=root,
    )


def gen_tree(n: int, branching: int = 3, seed: int = 0) -> tuple[str, FixtureStats]:
    """Rooted tree: n links, branching factor k, n-1 joints."""
    rng = random.Random(seed + n)
    root = f"tree_{n}_k{branching}"
    lines = [HEADER_TMPL.format(root=root)]

    for i in range(n):
        lines.append(link_block(
            f"l{i}",
            mass=rng.uniform(0.1, 5.0),
            ox=rng.uniform(-0.2, 0.2),
            oy=rng.uniform(-0.2, 0.2),
            oz=rng.uniform(0.0, 0.5),
            shape="box",
            dims=(0.1, 0.1, 0.1),
        ))

    children_count = [0] * n
    for i in range(1, n):
        candidates = [p for p in range(i) if children_count[p] < branching]
        if not candidates:
            candidates = list(range(i))
        parent = rng.choice(candidates)
        children_count[parent] += 1
        lines.append(revolute_joint_block(
            f"j{i-1}", parent=f"l{parent}", child=f"l{i}",
        ))

    lines.append(FOOTER)
    src = "".join(lines)
    n_v = n
    n_e = max(n - 1, 0)
    return src, FixtureStats(
        family="tree", name=f"tree_{n}_k{branching}",
        n_vertices=n_v, n_hyperedges=n_e,
        mean_arity=2.0 if n_e > 0 else 0.0,
        source_bytes=len(src.encode()), path="", robot_name=root,
    )


def _build_topology_humanoid(n_fingers_per_hand: int = 0,
                             ) -> list[tuple[str, str, str]]:
    """
    Return ordered list of (parent, child, joint_name) tuples defining the
    kinematic tree (excludes the root link itself). Used by both the
    .hymeko and .urdf humanoid emitters to guarantee identical topology.

    Humanoid morphology (Atlas-class, 28 links baseline):
      pelvis (root)
        torso / neck / head
        L arm: shoulder/upper_arm/elbow/forearm/wrist/hand
        R arm: (mirror)
        L leg: hip/upper_leg/knee/lower_leg/ankle/foot
        R leg: (mirror)
      Optional: n_fingers_per_hand × (knuckle/proximal/middle/distal).
    """
    edges: list[tuple[str, str, str]] = []

    # Spine
    edges.append(("pelvis", "torso", "j_pelvis_torso"))
    edges.append(("torso",  "neck",  "j_torso_neck"))
    edges.append(("neck",   "head",  "j_neck_head"))

    def arm_chain(side: str) -> None:
        parts = ["shoulder", "upper_arm", "elbow",
                 "forearm", "wrist", "hand"]
        parent = "torso"
        for p in parts:
            name = f"{side}_{p}"
            edges.append((parent, name, f"j_{name}"))
            parent = name
        # Fingers anchored at the hand
        for i in range(n_fingers_per_hand):
            f_parts = ["knuckle", "proximal", "middle", "distal"]
            parent = f"{side}_hand"
            for fp in f_parts:
                name = f"{side}_finger{i}_{fp}"
                edges.append((parent, name, f"j_{name}"))
                parent = name

    def leg_chain(side: str) -> None:
        parts = ["hip", "upper_leg", "knee",
                 "lower_leg", "ankle", "foot"]
        parent = "pelvis"
        for p in parts:
            name = f"{side}_{p}"
            edges.append((parent, name, f"j_{name}"))
            parent = name

    arm_chain("L")
    arm_chain("R")
    leg_chain("L")
    leg_chain("R")
    return edges


def _build_topology_quadruped(leg_dof: int = 5, tail_segments: int = 0,
                              ) -> list[tuple[str, str, str]]:
    """
    Quadruped morphology (Spot/ANYmal-class):
      body (root)
        head
        4 × leg: N-DOF chain from body → foot
        Optional: tail (tail_segments-link chain)
    """
    edges: list[tuple[str, str, str]] = []
    edges.append(("body", "head", "j_body_head"))

    leg_link_names = ["hip", "upper", "knee", "lower", "ankle",
                      "tarsus", "foot"]
    if leg_dof > len(leg_link_names):
        leg_dof = len(leg_link_names)

    for corner in ("FL", "FR", "RL", "RR"):
        parent = "body"
        for k in range(leg_dof):
            name = f"{corner}_{leg_link_names[k]}"
            edges.append((parent, name, f"j_{name}"))
            parent = name

    for t in range(tail_segments):
        parent = "body" if t == 0 else f"tail{t-1}"
        edges.append((parent, f"tail{t}", f"j_tail{t}"))
    return edges


def gen_humanoid(n_fingers_per_hand: int = 0, seed: int = 0,
                 ) -> tuple[str, FixtureStats]:
    rng = random.Random(seed + 7919 + n_fingers_per_hand)
    edges = _build_topology_humanoid(n_fingers_per_hand)
    root = f"humanoid_f{n_fingers_per_hand}"

    # Collect unique link names: every edge parent + every edge child + root
    link_names = {"pelvis"}
    for (p, c, _j) in edges:
        link_names.add(p); link_names.add(c)

    lines = [HEADER_TMPL.format(root=root)]
    # Emit one link block per name, deterministic order
    for name in sorted(link_names):
        lines.append(link_block(
            name, mass=rng.uniform(0.2, 4.0),
            ox=rng.uniform(-0.1, 0.1), oy=rng.uniform(-0.1, 0.1),
            oz=rng.uniform(0.0, 0.3),
            shape="box", dims=(0.08, 0.08, 0.1),
        ))
    # Joints in topology order
    for (parent, child, jname) in edges:
        lines.append(revolute_joint_block(jname, parent=parent, child=child))
    lines.append(FOOTER)
    src = "".join(lines)
    n_v = len(link_names)
    n_e = len(edges)
    return src, FixtureStats(
        family="humanoid", name=root, n_vertices=n_v,
        n_hyperedges=n_e, mean_arity=2.0,
        source_bytes=len(src.encode()), path="", robot_name=root,
    )


def gen_quadruped(leg_dof: int = 5, tail_segments: int = 0, seed: int = 0,
                  ) -> tuple[str, FixtureStats]:
    rng = random.Random(seed + 31337 + leg_dof * 13 + tail_segments)
    edges = _build_topology_quadruped(leg_dof, tail_segments)
    name = f"quadruped_d{leg_dof}_t{tail_segments}"

    link_names = {"body"}
    for (p, c, _j) in edges:
        link_names.add(p); link_names.add(c)

    lines = [HEADER_TMPL.format(root=name)]
    for lname in sorted(link_names):
        lines.append(link_block(
            lname, mass=rng.uniform(0.3, 5.0),
            ox=rng.uniform(-0.15, 0.15), oy=rng.uniform(-0.15, 0.15),
            oz=rng.uniform(0.0, 0.2),
            shape="box", dims=(0.1, 0.1, 0.1),
        ))
    for (parent, child, jname) in edges:
        lines.append(revolute_joint_block(jname, parent=parent, child=child))
    lines.append(FOOTER)
    src = "".join(lines)
    n_v = len(link_names)
    n_e = len(edges)
    return src, FixtureStats(
        family="quadruped", name=name, n_vertices=n_v,
        n_hyperedges=n_e, mean_arity=2.0,
        source_bytes=len(src.encode()), path="", robot_name=name,
    )


def gen_high_arity(m: int, d: int, seed: int = 0) -> tuple[str, FixtureStats]:
    """
    Stress fixture for Prop 4: m hyperedges each of arity d over a pool
    of ⌈m*d / 2⌉ shared vertices.
    """
    rng = random.Random(seed + m + d * 1000)
    n_v = max(d + 1, (m * d) // 2)
    root = f"ha_m{m}_d{d}"
    lines = [HEADER_TMPL.format(root=root)]

    for i in range(n_v):
        lines.append(link_block(
            f"v{i}", mass=1.0, ox=0.0, oy=0.0, oz=0.0,
            shape="box", dims=(0.05, 0.05, 0.05),
        ))

    for e in range(m):
        participants = rng.sample(range(n_v), d)
        half = d // 2
        signed = []
        for idx, v in enumerate(participants):
            sign = "+" if idx < half else "-"
            signed.append((sign, f"v{v}"))
        lines.append(high_arity_hyperedge_block(f"e{e}", signed))

    lines.append(FOOTER)
    src = "".join(lines)
    return src, FixtureStats(
        family="highArity", name=f"ha_m{m}_d{d}",
        n_vertices=n_v, n_hyperedges=m, mean_arity=float(d),
        source_bytes=len(src.encode()), path="", robot_name=root,
    )


def gen_fixed_pool_high_arity(n_pool: int, m: int, d: int,
                              seed: int = 0) -> tuple[str, "FixtureStats"]:
    """
    Asymptote-targeting fixture for Proposition 4 (storage overhead).

    Holds the vertex pool size n constant at `n_pool`, the hyperedge
    count constant at `m`, and sweeps the arity `d`. Under this design
    the bound

        (n + m) / (m * d̄)  →  0  as  d̄ → ∞,

    witnessing the asymptote ρ → 1 claimed by the proposition. The
    sibling `gen_high_arity` family grows n linearly with d (`n_v =
    max(d+1, m*d/2)`), which keeps the bound at ~0.55 across the swept
    range and therefore cannot witness the asymptote — see
    docs/storage_overhead_asymptote.md for the full analysis.
    """
    if d > n_pool:
        raise ValueError(
            f"gen_fixed_pool_high_arity: arity d={d} exceeds pool size "
            f"n_pool={n_pool}; each hyperedge must have d distinct "
            f"participants drawn from the pool"
        )
    rng = random.Random(seed + n_pool * 10_000 + m * 100 + d)
    name = f"hap_n{n_pool}_m{m}_d{d}"
    lines = [HEADER_TMPL.format(root=name)]

    for i in range(n_pool):
        lines.append(link_block(
            f"v{i}", mass=1.0, ox=0.0, oy=0.0, oz=0.0,
            shape="box", dims=(0.05, 0.05, 0.05),
        ))

    for e in range(m):
        participants = rng.sample(range(n_pool), d)
        half = d // 2
        signed: list[tuple[str, str]] = []
        for idx, v in enumerate(participants):
            sign = "+" if idx < half else "-"
            signed.append((sign, f"v{v}"))
        lines.append(high_arity_hyperedge_block(f"e{e}", signed))

    lines.append(FOOTER)
    src = "".join(lines)
    return src, FixtureStats(
        family="highArityFixedPool", name=name,
        n_vertices=n_pool, n_hyperedges=m, mean_arity=float(d),
        source_bytes=len(src.encode()), path="", robot_name=name,
    )


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

DEFAULT_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_HA_ARITIES = [2, 3, 5, 10, 20, 50]
DEFAULT_HA_M = 200
# Asymptote-targeting fixture for Prop 4: pool n held fixed, arity d
# swept up to n. With m fixed at DEFAULT_HAP_M and n_pool at
# DEFAULT_HAP_N, the bound (n+m)/(m·d̄) shrinks as 1/d̄.
DEFAULT_HAP_N = 200
DEFAULT_HAP_M = 200
DEFAULT_HAP_ARITIES = [2, 5, 10, 20, 50, 100, 200]
DEFAULT_HUMANOID_FINGERS = [0, 2, 5]    # base / two fingers / five fingers
DEFAULT_QUADRUPED_DOFS   = [3, 5, 7]    # typical, extended, high-DOF
DEFAULT_QUADRUPED_TAILS  = [0, 3]       # without / with tail


def write_fixture(
    out_dir: Path, family: str, stats: FixtureStats, src: str,
    meta_src: Path,
) -> FixtureStats:
    fam_dir = out_dir / family / stats.name
    fam_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fam_dir / f"{stats.name}.hymeko"
    fixture_path.write_text(src, encoding="utf-8")
    # Copy meta_kinematics alongside so the ModuleStore resolver finds it
    shutil.copy2(meta_src, fam_dir / "meta_kinematics.hymeko")
    stats.path = str(fixture_path.relative_to(out_dir))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for fixtures and manifest")
    ap.add_argument("--meta", type=Path,
                    default=Path(__file__).resolve().parents[2]
                                 / "data" / "robotics" / "meta_kinematics.hymeko",
                    help="Path to meta_kinematics.hymeko (default: workspace data/robotics/)")
    ap.add_argument("--sizes", type=str, default=None,
                    help="Comma-separated chain/tree sizes (default: log-spaced 1..5000)")
    ap.add_argument("--arities", type=str, default=None,
                    help="Comma-separated arities for highArity family")
    ap.add_argument("--ha-m", type=int, default=DEFAULT_HA_M)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-high-arity", action="store_true")
    ap.add_argument("--aliased", action="store_true",
                    help="Emit the idiomatic aliased form of each fixture: "
                         "declare `using kinematics.elements as el` + sibling "
                         "aliases in the description header and use the short "
                         "forms throughout. Default: emit the naive "
                         "fully-qualified form to preserve the measurement "
                         "baseline.")
    args = ap.parse_args()

    set_aliased(args.aliased)

    if not args.meta.exists():
        ap.error(f"meta_kinematics not found at {args.meta}")

    sizes = ([int(x) for x in args.sizes.split(",")]
             if args.sizes else DEFAULT_SIZES)
    arities = ([int(x) for x in args.arities.split(",")]
               if args.arities else DEFAULT_HA_ARITIES)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[FixtureStats] = []

    for n in sizes:
        src, stats = gen_chain(n, seed=args.seed)
        manifest.append(write_fixture(args.out, "chain", stats, src, args.meta))

    for n in sizes:
        src, stats = gen_tree(n, branching=3, seed=args.seed)
        manifest.append(write_fixture(args.out, "tree", stats, src, args.meta))

    if not args.skip_high_arity:
        for d in arities:
            src, stats = gen_high_arity(args.ha_m, d, seed=args.seed)
            manifest.append(write_fixture(args.out, "highArity", stats, src, args.meta))

        # Asymptote-targeting variant: pool n held fixed, arity d swept
        # — witnesses (n+m)/(m·d̄) → 0 for fixed n, m as d → ∞.
        # See docs/storage_overhead_asymptote.md for the math.
        for d in DEFAULT_HAP_ARITIES:
            if d > DEFAULT_HAP_N:
                continue  # arity cannot exceed the available pool
            src, stats = gen_fixed_pool_high_arity(
                DEFAULT_HAP_N, DEFAULT_HAP_M, d, seed=args.seed,
            )
            manifest.append(write_fixture(
                args.out, "highArityFixedPool", stats, src, args.meta,
            ))

    # Realistic-morphology check fixtures (Atlas / Spot-class)
    for nf in DEFAULT_HUMANOID_FINGERS:
        src, stats = gen_humanoid(n_fingers_per_hand=nf, seed=args.seed)
        manifest.append(write_fixture(args.out, "humanoid", stats, src, args.meta))
    for dof in DEFAULT_QUADRUPED_DOFS:
        for ts in DEFAULT_QUADRUPED_TAILS:
            src, stats = gen_quadruped(leg_dof=dof, tail_segments=ts, seed=args.seed)
            manifest.append(write_fixture(args.out, "quadruped", stats, src, args.meta))

    (args.out / "index.json").write_text(
        json.dumps([asdict(s) for s in manifest], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} fixtures to {args.out}/")
    print(f"Manifest: {args.out}/index.json")


if __name__ == "__main__":
    main()
