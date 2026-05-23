#!/usr/bin/env python3
"""Emit GZ-ready SDF models for the triad rapport demo from HyMeKo.

For each agent model:
  1. Run ``hymeko emit -f sdf <robotics_file>.hymeko`` to produce the
     kinematic-structure SDF (links + joints + geometries + colours).
  2. Inject the gz-specific runtime extensions that the SDF emitter
     doesn't (yet) generate: PosePublisher, DiffDrive, camera sensor.
  3. Write the final SDF to ``data/models/<name>/model.sdf``.

The runtime extensions are intentionally additive: they hook into the
named links (``body`` / ``head`` / ``chassis`` / ``left_wheel`` /
``right_wheel``) that the HyMeKo file defines. If the .hymeko file
adds a new link, the SDF emit still works; only the plugin attachment
needs to know the link names.

Run:
    python scripts/emit_triad_sdf.py

Plan: docs/plans/2026-05-18-gz-rapport-demo/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTICS_DIR = REPO_ROOT / "data" / "robotics"
MODELS_DIR = REPO_ROOT / "data" / "models"


# ─── Plugin / sensor XML snippets (parsed and grafted in) ────────────


HUMAN_POSE_PUBLISHER = dedent("""
    <plugin
      filename="gz-sim-pose-publisher-system"
      name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose>
      <publish_model_pose>true</publish_model_pose>
      <update_frequency>30</update_frequency>
    </plugin>
""").strip()


R1_DIFF_DRIVE = dedent("""
    <plugin
      filename="gz-sim-diff-drive-system"
      name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.40</wheel_separation>
      <wheel_radius>0.05</wheel_radius>
      <odom_publish_frequency>10</odom_publish_frequency>
      <topic>cmd_vel</topic>
      <max_linear_acceleration>2.0</max_linear_acceleration>
      <max_angular_acceleration>2.0</max_angular_acceleration>
    </plugin>
""").strip()


R1_POSE_PUBLISHER = HUMAN_POSE_PUBLISHER


R1_CAMERA_LINK = dedent("""
    <link name="camera_link">
      <pose>0.13 0 0 0 0 0</pose>
      <inertial>
        <mass>0.05</mass>
        <inertia>
          <ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name="camera_visual">
        <geometry><box><size>0.03 0.05 0.025</size></box></geometry>
        <material>
          <ambient>0.05 0.05 0.05 1</ambient>
          <diffuse>0.05 0.05 0.05 1</diffuse>
        </material>
      </visual>
      <sensor name="camera" type="camera">
        <pose>0 0 0 0 0 0</pose>
        <camera>
          <horizontal_fov>1.92</horizontal_fov>
          <image>
            <width>320</width>
            <height>240</height>
            <format>R8G8B8</format>
          </image>
          <clip><near>0.05</near><far>10.0</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <topic>r1/camera/image</topic>
      </sensor>
    </link>
""").strip()


R1_CAMERA_JOINT = dedent("""
    <joint name="camera_joint" type="fixed">
      <parent>head</parent>
      <child>camera_link</child>
    </joint>
""").strip()


# ─── Heading-arrow visual (a small forward-facing cylinder visual) ───


HUMAN_HEADING_ARROW = dedent("""
    <visual name="heading_arrow">
      <pose relative_to="head">0.18 0 0 0 1.5707963 0</pose>
      <geometry>
        <cylinder><radius>0.015</radius><length>0.20</length></cylinder>
      </geometry>
      <material>
        <ambient>0.85 0.30 0.10 1</ambient>
        <diffuse>0.85 0.30 0.10 1</diffuse>
      </material>
    </visual>
""").strip()


R1_HEADING_ARROW = dedent("""
    <visual name="heading_arrow">
      <pose relative_to="head">0.16 0 0 0 1.5707963 0</pose>
      <geometry>
        <cylinder><radius>0.012</radius><length>0.18</length></cylinder>
      </geometry>
      <material>
        <ambient>0.85 0.30 0.10 1</ambient>
        <diffuse>0.85 0.30 0.10 1</diffuse>
      </material>
    </visual>
""").strip()


# ─── emit + patch ────────────────────────────────────────────────────


def emit_hymeko_sdf(hymeko_path: Path, model_name: str) -> str:
    """Run `hymeko emit -f sdf` and return the SDF text."""
    out = subprocess.check_output(
        ["cargo", "run", "--quiet", "--bin", "hymeko",
         "--", "emit", "-f", "sdf",
         "--name", model_name,
         str(hymeko_path)],
        cwd=str(REPO_ROOT),
    )
    return out.decode("utf-8")


def graft_xml_into_link(model_root: ET.Element,
                        link_name: str,
                        xml_snippet: str) -> None:
    """Append parsed XML elements to a named <link> in the model."""
    for link in model_root.findall("link"):
        if link.get("name") == link_name:
            for child in ET.fromstring(f"<root>{xml_snippet}</root>"):
                link.append(child)
            return
    raise ValueError(f"link {link_name!r} not found in model")


def graft_xml_into_model(model_root: ET.Element, xml_snippet: str) -> None:
    """Append parsed XML elements directly to <model>."""
    for child in ET.fromstring(f"<root>{xml_snippet}</root>"):
        model_root.append(child)


def inject_material(model: ET.Element, link_name: str,
                    rgba: tuple[float, float, float, float]) -> None:
    """Add an `<material><ambient>...<diffuse>...</material>` block to
    each `<visual>` of the named link.

    The HyMeKo SDF emitter does not yet propagate `color` directives
    from the source .hymeko file. We inject them here from the same
    constants the .hymeko file declares (body_color, skin_color,
    chassis_color, head_color, wheel_color), keeping the colour
    constants synchronised — when the emitter learns to do this
    natively, this helper goes away.
    """
    r, g, b, a = rgba
    rgba_str = f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}"
    for link in model.findall("link"):
        if link.get("name") != link_name:
            continue
        for visual in link.findall("visual"):
            # Skip if a material is already present.
            if visual.find("material") is not None:
                continue
            mat = ET.SubElement(visual, "material")
            amb = ET.SubElement(mat, "ambient"); amb.text = rgba_str
            dif = ET.SubElement(mat, "diffuse"); dif.text = rgba_str
        return
    raise ValueError(f"link {link_name!r} not found")


def inject_link_pose(model: ET.Element, link_name: str, pose_xyz_rpy: str) -> None:
    """Insert a <pose> element as the first child of <link> with the
    given name. The HyMeKo SDF emitter doesn't yet derive link-level
    poses from joint origins; this wrapper supplies them based on the
    joint topology declared in the source .hymeko file."""
    for link in model.findall("link"):
        if link.get("name") == link_name:
            pose_el = ET.Element("pose")
            pose_el.text = pose_xyz_rpy
            link.insert(0, pose_el)
            return
    raise ValueError(f"link {link_name!r} not found")


def patch_human_sdf(sdf_text: str) -> str:
    """Add pose-publisher + heading-arrow visual to the human SDF.
    The head link also gets a model-frame pose so it sits atop the
    body (the HyMeKo neck joint declared this offset, but the SDF
    emitter doesn't yet propagate joint origins to link poses)."""
    root = ET.fromstring(sdf_text)
    model = root.find("model")
    if model is None:
        raise RuntimeError("emitted SDF has no <model> element")
    # The HyMeKo neck joint says head sits 0.80 above body. The body
    # visual pose put the cylinder at z=0.75 (centre-of-mass). The head
    # link pose in the model frame is therefore z=1.55 (body z=0.75 +
    # head offset within body=0.80). The visual layer already places
    # it correctly via relative_to="head"; we just need the link's
    # own pose so collision works at the right height.
    inject_link_pose(model, "head", "0 0 1.55 0 0 0")
    # Colours from triad_human.hymeko's body_color / skin_color.
    inject_material(model, "body", (0.42, 0.62, 0.83, 1.0))
    inject_material(model, "head", (0.95, 0.83, 0.70, 1.0))
    graft_xml_into_link(model, "head", HUMAN_HEADING_ARROW)
    graft_xml_into_model(model, HUMAN_POSE_PUBLISHER)
    return _prettify(root)


def patch_r1_sdf(sdf_text: str) -> str:
    """Add diff-drive + pose publisher + camera link/joint + heading
    arrow to the r1 SDF, and inject link poses from the HyMeKo joint
    declarations (the SDF emitter doesn't yet propagate joint origins
    to link poses, so the wheels and head end up at the origin without
    this patch)."""
    root = ET.fromstring(sdf_text)
    model = root.find("model")
    if model is None:
        raise RuntimeError("emitted SDF has no <model> element")
    # Link poses derived from triad_r1.hymeko joint origins:
    #   neck:              chassis → head at (0, 0, 0.35)
    #   left_wheel_joint:  chassis → left_wheel at (0, 0.20, -0.05), rpy (-π/2, 0, 0)
    #   right_wheel_joint: chassis → right_wheel at (0, -0.20, -0.05), rpy (-π/2, 0, 0)
    # The chassis itself sits at z=0.10 (per its visual pose).
    inject_link_pose(model, "head", "0 0 0.45 0 0 0")
    inject_link_pose(model, "left_wheel",  "0 0.20 0.05 1.5707963 0 0")
    inject_link_pose(model, "right_wheel", "0 -0.20 0.05 1.5707963 0 0")
    # Colours from triad_r1.hymeko's chassis/head/wheel_color directives.
    inject_material(model, "chassis",      (0.30, 0.30, 0.35, 1.0))
    inject_material(model, "head",         (0.95, 0.55, 0.20, 1.0))
    inject_material(model, "left_wheel",   (0.05, 0.05, 0.05, 1.0))
    inject_material(model, "right_wheel",  (0.05, 0.05, 0.05, 1.0))
    # Heading arrow + camera + plugins.
    graft_xml_into_link(model, "head", R1_HEADING_ARROW)
    graft_xml_into_model(model, R1_CAMERA_LINK)
    graft_xml_into_model(model, R1_CAMERA_JOINT)
    graft_xml_into_model(model, R1_DIFF_DRIVE)
    graft_xml_into_model(model, R1_POSE_PUBLISHER)
    return _prettify(root)


def _prettify(root: ET.Element) -> str:
    """Round-trip through minidom for human-readable indent."""
    import xml.dom.minidom as minidom
    rough = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(MODELS_DIR))
    args = ap.parse_args()
    out_root = Path(args.out_dir)

    targets = [
        ("triad_human",
         ROBOTICS_DIR / "triad_human.hymeko",
         patch_human_sdf,
         out_root / "triad_human" / "model.sdf"),
        ("triad_r1",
         ROBOTICS_DIR / "triad_r1.hymeko",
         patch_r1_sdf,
         out_root / "triad_r1" / "model.sdf"),
    ]
    for name, hymeko_path, patch_fn, out_path in targets:
        print(f"--- {name} ---", file=sys.stderr)
        if not hymeko_path.exists():
            raise FileNotFoundError(f"missing input: {hymeko_path}")
        print(f"  emitting from {hymeko_path}", file=sys.stderr)
        raw = emit_hymeko_sdf(hymeko_path, name)
        print(f"  patching with gz extensions", file=sys.stderr)
        final = patch_fn(raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final)
        print(f"  wrote {out_path}  ({len(final.splitlines())} lines)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
