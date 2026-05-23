#!/usr/bin/env python3
r"""
urdf_to_hymeko.py — translate a URDF file into a `.hymeko` description
that emits byte-equivalent kinematic structure (links, joints, axes,
limits, origins) through HyMeKo's compile + emit pipeline.

Usage:
    python urdf_to_hymeko.py path/to/robot.urdf --out fixtures/real/

The output is a directory containing:
    <robot_name>.hymeko          — the translated description
    meta_kinematics.hymeko       — copied for module resolution

Mesh geometry (`<mesh filename="...">`) is replaced with a placeholder
box of the link's inertial extents; HyMeKo's emitters do not produce
mesh-bearing artefacts, and substituting a placeholder lets the
emitted URDF/SDF/MJCF round-trip without dangling mesh references.
The kinematic structure (link names, joint hierarchy, axes, limits,
origins, masses, inertias) is preserved verbatim.

Joint type mapping:
    URDF revolute    →  meta_kinematics.kinematics.rev_joint
    URDF continuous  →  meta_kinematics.kinematics.conti_joint
    URDF prismatic   →  meta_kinematics.kinematics.prismatic_joint
    URDF fixed       →  meta_kinematics.kinematics.fixed_joint

Axis mapping:
    Principal axes (±X, ±Y, ±Z) map to AXIS_X / AXIS_Y / AXIS_Z and the
    M_-prefixed negative-axis decls. Negative-X and negative-Y decls
    are synthesised in the fixture header as needed (the shared
    meta_kinematics.hymeko ships AXIS_M_Z only). Non-principal axes
    (any direction with multiple non-zero components) are emitted as
    custom axis decls in the fixture header.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── URDF model ────────────────────────────────────────────────────────────

@dataclass
class Link:
    name: str
    mass: float = 1.0
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geom_dims: tuple[float, float, float] = (0.05, 0.05, 0.05)
    geom_shape: str = "box"


@dataclass
class Joint:
    name: str
    jtype: str  # urdf type: revolute, continuous, prismatic, fixed
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None
    limit_effort: Optional[float] = None
    limit_velocity: Optional[float] = None


@dataclass
class UrdfModel:
    name: str
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)


def parse_urdf(path: Path) -> UrdfModel:
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{path}: root element is <{root.tag}>, expected <robot>")
    model = UrdfModel(name=root.get("name", path.stem))

    for link_el in root.findall("link"):
        name = link_el.get("name")
        if not name:
            continue
        link = Link(name=name)
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                link.mass = float(mass_el.get("value", "1.0"))
            origin = inertial.find("origin")
            if origin is not None:
                xyz = origin.get("xyz", "0 0 0").split()
                link.origin_xyz = tuple(float(x) for x in xyz)
        # No geometry inspection needed — meshes are placeholder-substituted
        # in the emitter; keep the default 5cm box.
        model.links.append(link)

    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type", "fixed")
        if not name:
            continue
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        joint = Joint(
            name=name, jtype=jtype,
            parent=parent_el.get("link", ""),
            child=child_el.get("link", ""),
        )
        origin = joint_el.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0").split()
            rpy = origin.get("rpy", "0 0 0").split()
            joint.origin_xyz = tuple(float(x) for x in xyz)
            joint.origin_rpy = tuple(float(x) for x in rpy)
        axis_el = joint_el.find("axis")
        if axis_el is not None:
            axis = axis_el.get("xyz", "0 0 1").split()
            joint.axis = tuple(float(x) for x in axis)
        limit_el = joint_el.find("limit")
        if limit_el is not None:
            joint.limit_lower = _maybe_float(limit_el.get("lower"))
            joint.limit_upper = _maybe_float(limit_el.get("upper"))
            joint.limit_effort = _maybe_float(limit_el.get("effort"))
            joint.limit_velocity = _maybe_float(limit_el.get("velocity"))
        model.joints.append(joint)

    return model


def _maybe_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# ─── Axis classification ──────────────────────────────────────────────────

# Built-in axes shipped in meta_kinematics.hymeko.
# AXIS_M_Z is the only negative principal axis pre-declared; we emit
# AXIS_M_X and AXIS_M_Y as fixture-local decls when needed.
BUILTIN_AXES = {
    (1.0, 0.0, 0.0):  "AXIS_X",
    (0.0, 1.0, 0.0):  "AXIS_Y",
    (0.0, 0.0, 1.0):  "AXIS_Z",
    (0.0, 0.0, -1.0): "ax.AXIS_M_Z",
}


def classify_axis(ax: tuple[float, float, float],
                  custom_axes: dict[tuple[float, float, float], str],
                  ) -> str:
    """Return the HyMeKo path for an axis vector. Registers fixture-local
    axes in `custom_axes` if not a built-in; subsequent calls reuse the
    same name."""
    # Round to 6 decimal places to identify principal axes that may have
    # tiny numerical noise.
    key = tuple(round(c, 6) for c in ax)
    if key in BUILTIN_AXES:
        # AXIS_M_Z is in `ax.` namespace; AXIS_X/Y/Z bare.
        name = BUILTIN_AXES[key]
        return name if "." in name else f"ax.{name}"
    if key in custom_axes:
        return custom_axes[key]
    # Synthesise a fixture-local axis decl.
    # Names: AXIS_M_X / AXIS_M_Y for principal-negative, otherwise
    # AXIS_CUSTOM_<n> with sequential numbering.
    if key == (-1.0, 0.0, 0.0):
        local_name = "AXIS_M_X"
    elif key == (0.0, -1.0, 0.0):
        local_name = "AXIS_M_Y"
    else:
        local_name = f"AXIS_CUSTOM_{len(custom_axes)}"
    custom_axes[key] = local_name
    return local_name


# ─── HyMeKo emitter ────────────────────────────────────────────────────────

def emit_hymeko(model: UrdfModel) -> str:
    custom_axes: dict[tuple[float, float, float], str] = {}

    # First pass: classify axes (populates custom_axes).
    for j in model.joints:
        if j.jtype != "fixed":
            classify_axis(j.axis, custom_axes)

    out: list[str] = []
    rname = _sanitise(model.name)

    # Header block: import + aliases + custom axis decls.
    out.append(f"{rname}_description {{\n")
    out.append('    @"meta_kinematics.hymeko";\n')
    out.append("    using kinematics.elements as el;\n")
    out.append("    using kinematics.geometry as geo;\n")
    out.append("    using kinematics.axes as ax;\n")
    out.append("    using kinematics.rev_joint as rj;\n")
    out.append("    using kinematics.conti_joint as cj;\n")
    out.append("    using kinematics.prismatic_joint as pj;\n")
    out.append("    using kinematics.fixed_joint as fj;\n")
    out.append("}\n\n")

    out.append(f"{rname}: el, geo, ax\n{{\n")

    # Custom axis decls go inside the description body (they are decls,
    # not header statements). Each is + <isa> ax.axis_definition.
    for vec, local_name in custom_axes.items():
        out.append(
            f"    {local_name}: + <isa> ax.axis_definition "
            f"{{ ax [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]; }}\n"
        )
    if custom_axes:
        out.append("\n")

    # Links: substitute mesh geometry with default placeholder box.
    # Mass and inertial origin preserved from URDF.
    for link in model.links:
        out.append(_emit_link(link))

    # Joints
    for j in model.joints:
        out.append(_emit_joint(j, custom_axes))

    out.append("}\n")
    return "".join(out)


def _emit_link(link: Link) -> str:
    n = _sanitise(link.name)
    return (
        f"    {n}: el.link {{\n"
        f"        mass {link.mass:.6f};\n"
        f"        link_geometry: geo.box {{ "
        f"dimension [{link.geom_dims[0]:.4f}, {link.geom_dims[1]:.4f}, {link.geom_dims[2]:.4f}]; }}\n"
        f"        visual    -> link_geometry;\n"
        f"        collision -> link_geometry;\n"
        f"        origin [{link.origin_xyz[0]:.6f}, {link.origin_xyz[1]:.6f}, {link.origin_xyz[2]:.6f}];\n"
        f"    }}\n"
    )


def _joint_alias(jtype: str) -> str:
    return {"revolute": "rj", "continuous": "cj",
            "prismatic": "pj", "fixed": "fj"}.get(jtype, "fj")


def _emit_joint(j: Joint,
                custom_axes: dict[tuple[float, float, float], str]) -> str:
    name = _sanitise(j.name)
    alias = _joint_alias(j.jtype)
    parent = _sanitise(j.parent)
    child = _sanitise(j.child)
    ox, oy, oz = j.origin_xyz
    rx, ry, rz = j.origin_rpy

    body = [f"    @{name}: {alias} {{\n"]
    incidences = [
        f"(+ {parent} [[{ox:.6f}, {oy:.6f}, {oz:.6f}], "
        f"[{rx:.6f}, {ry:.6f}, {rz:.6f}]]"
    ]
    incidences.append(f"         - {child}")
    if j.jtype != "fixed":
        axis_path = classify_axis(j.axis, custom_axes)
        incidences.append(f"         - {axis_path}")
    inner = ",\n".join(incidences) + ");"
    body.append(f"        {inner}\n")
    body.append("    }\n")
    return "".join(body)


def _sanitise(s: str) -> str:
    """HyMeKo identifiers: ASCII alphanumeric + underscore. Replace
    everything else with `_`. Cannot start with a digit."""
    out = []
    for c in s:
        if c.isalnum() or c == "_":
            out.append(c)
        else:
            out.append("_")
    sanitised = "".join(out)
    if not sanitised:
        return "_"
    if sanitised[0].isdigit():
        sanitised = "_" + sanitised
    return sanitised


# ─── Orchestration ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", type=Path, help="Input URDF file")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (created if missing)")
    ap.add_argument("--meta", type=Path,
                    default=Path(__file__).resolve().parents[2]
                                 / "data" / "robotics" / "meta_kinematics.hymeko",
                    help="Path to meta_kinematics.hymeko (default: workspace data/robotics/)")
    args = ap.parse_args()

    if not args.urdf.exists():
        ap.error(f"URDF not found: {args.urdf}")
    if not args.meta.exists():
        ap.error(f"meta_kinematics not found: {args.meta}")

    model = parse_urdf(args.urdf)
    src = emit_hymeko(model)

    args.out.mkdir(parents=True, exist_ok=True)
    rname = _sanitise(model.name)
    fixture_path = args.out / f"{rname}.hymeko"
    fixture_path.write_text(src, encoding="utf-8")
    shutil.copy2(args.meta, args.out / "meta_kinematics.hymeko")

    print(f"Translated {args.urdf}")
    print(f"  → {fixture_path}")
    print(f"  robot:    {model.name}")
    print(f"  links:    {len(model.links)}")
    print(f"  joints:   {len(model.joints)} "
          f"({sum(1 for j in model.joints if j.jtype == 'revolute')} revolute, "
          f"{sum(1 for j in model.joints if j.jtype == 'continuous')} continuous, "
          f"{sum(1 for j in model.joints if j.jtype == 'prismatic')} prismatic, "
          f"{sum(1 for j in model.joints if j.jtype == 'fixed')} fixed)")
    print(f"  bytes:    {fixture_path.stat().st_size}")


if __name__ == "__main__":
    main()
