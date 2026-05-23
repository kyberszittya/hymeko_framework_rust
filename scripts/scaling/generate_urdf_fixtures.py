#!/usr/bin/env python3
"""
generate_urdf_fixtures.py — URDF fixtures mirroring the .hymeko fixtures
for the head-to-head scaling comparison.

Each output is a plain URDF file (valid xacro input — xacro is a
superset of URDF). Same sizes and topology as the .hymeko generator,
deterministic from the same seed. We write `.urdf` files so xacro,
gz sdf, and mujoco all consume the identical input — any timing
difference is toolchain cost, not input-size drift.

Usage:
    python generate_urdf_fixtures.py --out ./urdf_fixtures
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<robot name="{name}">\n'
FOOTER = '</robot>\n'


def link_xml(name: str, mass: float, shape: str, dims: tuple[float, ...]) -> str:
    if shape == "box":
        geom = f'<box size="{dims[0]:.3f} {dims[1]:.3f} {dims[2]:.3f}"/>'
    elif shape == "cylinder":
        geom = f'<cylinder radius="{dims[0]:.3f}" length="{dims[1]:.3f}"/>'
    else:
        geom = f'<sphere radius="{dims[0]:.3f}"/>'
    return (
        f'  <link name="{name}">\n'
        f'    <inertial><mass value="{mass:.3f}"/>'
        f'<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>\n'
        f'    <visual><geometry>{geom}</geometry></visual>\n'
        f'    <collision><geometry>{geom}</geometry></collision>\n'
        f'  </link>\n'
    )


def joint_xml(name: str, parent: str, child: str,
              xyz: tuple[float, float, float] = (0.0, 0.0, 0.1),
              rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
              axis: tuple[int, int, int] = (0, 0, 1)) -> str:
    return (
        f'  <joint name="{name}" type="revolute">\n'
        f'    <parent link="{parent}"/>\n'
        f'    <child link="{child}"/>\n'
        f'    <origin xyz="{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}" '
        f'rpy="{rpy[0]:.3f} {rpy[1]:.3f} {rpy[2]:.3f}"/>\n'
        f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n'
        f'    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>\n'
        f'  </joint>\n'
    )


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


def gen_chain(n: int, seed: int = 0) -> tuple[str, FixtureStats]:
    rng = random.Random(seed + n)
    name = f"chain_{n}"
    lines = [HEADER.format(name=name)]
    for i in range(n):
        lines.append(link_xml(
            f"l{i}", mass=rng.uniform(0.1, 5.0),
            shape="cylinder", dims=(0.05, 0.1),
        ))
    for i in range(n - 1):
        lines.append(joint_xml(f"j{i}", f"l{i}", f"l{i+1}"))
    lines.append(FOOTER)
    src = "".join(lines)
    return src, FixtureStats(
        family="chain", name=name, n_vertices=n, n_hyperedges=max(n - 1, 0),
        mean_arity=2.0 if n > 1 else 0.0,
        source_bytes=len(src.encode()), path="", robot_name=name,
    )


def gen_tree(n: int, branching: int = 3, seed: int = 0) -> tuple[str, FixtureStats]:
    rng = random.Random(seed + n)
    name = f"tree_{n}_k{branching}"
    lines = [HEADER.format(name=name)]
    for i in range(n):
        lines.append(link_xml(
            f"l{i}", mass=rng.uniform(0.1, 5.0),
            shape="box", dims=(0.1, 0.1, 0.1),
        ))
    children_count = [0] * n
    for i in range(1, n):
        cands = [p for p in range(i) if children_count[p] < branching]
        if not cands:
            cands = list(range(i))
        parent = rng.choice(cands)
        children_count[parent] += 1
        lines.append(joint_xml(f"j{i-1}", f"l{parent}", f"l{i}"))
    lines.append(FOOTER)
    src = "".join(lines)
    return src, FixtureStats(
        family="tree", name=name, n_vertices=n, n_hyperedges=max(n - 1, 0),
        mean_arity=2.0 if n > 1 else 0.0,
        source_bytes=len(src.encode()), path="", robot_name=name,
    )


def _topo_humanoid(n_fingers_per_hand: int) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    edges.append(("pelvis", "torso", "j_pelvis_torso"))
    edges.append(("torso",  "neck",  "j_torso_neck"))
    edges.append(("neck",   "head",  "j_neck_head"))
    for side in ("L", "R"):
        parts = ["shoulder", "upper_arm", "elbow",
                 "forearm", "wrist", "hand"]
        parent = "torso"
        for p in parts:
            n = f"{side}_{p}"
            edges.append((parent, n, f"j_{n}"))
            parent = n
        for i in range(n_fingers_per_hand):
            parent = f"{side}_hand"
            for fp in ["knuckle", "proximal", "middle", "distal"]:
                n = f"{side}_finger{i}_{fp}"
                edges.append((parent, n, f"j_{n}"))
                parent = n
    for side in ("L", "R"):
        parts = ["hip", "upper_leg", "knee",
                 "lower_leg", "ankle", "foot"]
        parent = "pelvis"
        for p in parts:
            n = f"{side}_{p}"
            edges.append((parent, n, f"j_{n}"))
            parent = n
    return edges


def _topo_quadruped(leg_dof: int, tail_segments: int) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    edges.append(("body", "head", "j_body_head"))
    leg_link_names = ["hip", "upper", "knee", "lower", "ankle", "tarsus", "foot"]
    leg_dof = min(leg_dof, len(leg_link_names))
    for corner in ("FL", "FR", "RL", "RR"):
        parent = "body"
        for k in range(leg_dof):
            n = f"{corner}_{leg_link_names[k]}"
            edges.append((parent, n, f"j_{n}"))
            parent = n
    for t in range(tail_segments):
        parent = "body" if t == 0 else f"tail{t-1}"
        edges.append((parent, f"tail{t}", f"j_tail{t}"))
    return edges


def _topo_fixture(family: str, name: str, edges: list[tuple[str, str, str]],
                  root_link: str, seed: int) -> tuple[str, FixtureStats]:
    rng = random.Random(seed + sum(ord(c) for c in name))
    links = {root_link}
    for (p, c, _j) in edges:
        links.add(p); links.add(c)
    lines = [HEADER.format(name=name)]
    for l in sorted(links):
        lines.append(link_xml(
            l, mass=rng.uniform(0.3, 4.0),
            shape="box", dims=(0.08, 0.08, 0.1),
        ))
    for (parent, child, jname) in edges:
        lines.append(joint_xml(jname, parent, child))
    lines.append(FOOTER)
    src = "".join(lines)
    return src, FixtureStats(
        family=family, name=name,
        n_vertices=len(links), n_hyperedges=len(edges),
        mean_arity=2.0, source_bytes=len(src.encode()),
        path="", robot_name=name,
    )


def gen_humanoid(n_fingers_per_hand: int, seed: int = 0) -> tuple[str, FixtureStats]:
    edges = _topo_humanoid(n_fingers_per_hand)
    return _topo_fixture("humanoid", f"humanoid_f{n_fingers_per_hand}",
                         edges, "pelvis", seed)


def gen_quadruped(leg_dof: int, tail_segments: int,
                  seed: int = 0) -> tuple[str, FixtureStats]:
    edges = _topo_quadruped(leg_dof, tail_segments)
    return _topo_fixture("quadruped",
                         f"quadruped_d{leg_dof}_t{tail_segments}",
                         edges, "body", seed)


DEFAULT_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_HUMANOID_FINGERS = [0, 2, 5]
DEFAULT_QUADRUPED_DOFS   = [3, 5, 7]
DEFAULT_QUADRUPED_TAILS  = [0, 3]


def write_fixture(out_dir: Path, family: str, stats: FixtureStats, src: str) -> FixtureStats:
    fam = out_dir / family / stats.name
    fam.mkdir(parents=True, exist_ok=True)
    p = fam / f"{stats.name}.urdf"
    p.write_text(src, encoding="utf-8")
    stats.path = str(p.relative_to(out_dir))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = ([int(x) for x in args.sizes.split(",")]
             if args.sizes else DEFAULT_SIZES)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[FixtureStats] = []
    for n in sizes:
        if n < 2:
            continue  # URDF needs at least 1 link + 0 joints; xacro OK, but skip n=1
        src, s = gen_chain(n, seed=args.seed)
        manifest.append(write_fixture(args.out, "chain", s, src))
        src, s = gen_tree(n, branching=3, seed=args.seed)
        manifest.append(write_fixture(args.out, "tree", s, src))

    for nf in DEFAULT_HUMANOID_FINGERS:
        src, s = gen_humanoid(nf, seed=args.seed)
        manifest.append(write_fixture(args.out, "humanoid", s, src))
    for dof in DEFAULT_QUADRUPED_DOFS:
        for ts in DEFAULT_QUADRUPED_TAILS:
            src, s = gen_quadruped(dof, ts, seed=args.seed)
            manifest.append(write_fixture(args.out, "quadruped", s, src))

    (args.out / "index.json").write_text(
        json.dumps([asdict(s) for s in manifest], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} URDF fixtures to {args.out}/")


if __name__ == "__main__":
    main()
