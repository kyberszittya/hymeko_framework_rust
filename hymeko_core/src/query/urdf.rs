//! URDF XML generation from kinematic model.

use crate::ir::ir::Ir;
use crate::query::engine::{NameResolver, QueryEngine};
use crate::query::kinematic::*;
use crate::query::predicate::*;

/// Predefined queries for URDF generation.
pub fn urdf_queries() -> Vec<NamedQuery> {
    vec![
        NamedQuery { label: "links".into(),
            predicate: Predicate::node().and(Predicate::inherits("link")) },
        NamedQuery { label: "fixed_joints".into(),
            predicate: Predicate::edge().and(Predicate::inherits("fixed_joint")) },
        NamedQuery { label: "continuous_joints".into(),
            predicate: Predicate::edge().and(Predicate::inherits("conti_joint")) },
        NamedQuery { label: "revolute_joints".into(),
            predicate: Predicate::edge().and(Predicate::inherits("rev_joint")) },
        NamedQuery { label: "prismatic_joints".into(),
            predicate: Predicate::edge().and(Predicate::inherits("prismatic_joint")) },
        NamedQuery { label: "axes".into(),
            predicate: Predicate::node().and(Predicate::inherits("axis_definition")) },
    ]
}

/// Generate URDF XML string from a compiled IR.
pub fn generate_urdf<R: NameResolver>(ir: &Ir, resolver: &R, robot_name: &str) -> String {
    let engine = QueryEngine::new(ir, resolver);
    let model = extract_kinematic_model(&engine, robot_name);

    let mut out = String::with_capacity(4096);
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!("<robot name=\"{}\">\n", xml_escape(robot_name)));

    // Links
    for link in &model.links {
        out.push_str(&format!("  <link name=\"{}\">\n", xml_escape(&link.name)));

        if let Some(mass) = link.mass {
            out.push_str("    <inertial>\n");
            out.push_str(&format!("      <mass value=\"{}\"/>\n", mass));
            out.push_str("    </inertial>\n");
        }

        if let Some(ref geom) = link.geometry {
            for tag in &["visual", "collision"] {
                out.push_str(&format!("    <{}>\n", tag));
                if let Some(ref origin) = link.origin {
                    emit_origin_list(&mut out, origin, 6);
                }
                out.push_str("      <geometry>\n");
                emit_geometry(&mut out, geom, 8);
                out.push_str("      </geometry>\n");
                if *tag == "visual" {
                    if let Some(ref color) = link.color {
                        out.push_str("      <material name=\"color\">\n");
                        out.push_str(&format!(
                            "        <color rgba=\"{} {} {} {}\"/>\n",
                            color.get(0).unwrap_or(&0.5),
                            color.get(1).unwrap_or(&0.5),
                            color.get(2).unwrap_or(&0.5),
                            color.get(3).unwrap_or(&1.0),
                        ));
                        out.push_str("      </material>\n");
                    }
                }
                out.push_str(&format!("    </{}>\n", tag));
            }
        }

        out.push_str("  </link>\n\n");
    }

    // Joints
    for joint in &model.joints {
        out.push_str(&format!(
            "  <joint name=\"{}\" type=\"{}\">\n",
            xml_escape(&joint.name), joint.joint_type.urdf_str()
        ));
        out.push_str(&format!(
            "    <parent link=\"{}\"/>\n", xml_escape(&joint.parent_link)
        ));
        out.push_str(&format!(
            "    <child link=\"{}\"/>\n", xml_escape(&joint.child_link)
        ));

        if let Some(xyz) = joint.origin_xyz {
            let rpy = joint.origin_rpy_rad().unwrap_or([0.0; 3]);
            out.push_str(&format!(
                "    <origin xyz=\"{} {} {}\" rpy=\"{:.4} {:.4} {:.4}\"/>\n",
                xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]
            ));
        }

        if joint.joint_type != JointType::Fixed {
            if let Some(ax) = joint.axis {
                out.push_str(&format!(
                    "    <axis xyz=\"{} {} {}\"/>\n",
                    ax[0] as i32, ax[1] as i32, ax[2] as i32
                ));
            }
        }

        if let Some(ref lim) = joint.limits {
            out.push_str(&format!(
                "    <limit lower=\"{}\" upper=\"{}\" effort=\"{}\" velocity=\"{}\"/>\n",
                lim.lower, lim.upper, lim.effort, lim.velocity
            ));
        }

        out.push_str("  </joint>\n\n");
    }

    out.push_str("</robot>\n");
    out
}

/// Schema validation: check that every joint references known links.
pub fn validate_robot_schema<R: NameResolver>(ir: &Ir, resolver: &R) -> Vec<String> {
    let engine = QueryEngine::new(ir, resolver);
    let model = extract_kinematic_model(&engine, "validation");
    let mut errors = Vec::new();

    let link_names: std::collections::HashSet<&str> =
        model.links.iter().map(|l| l.name.as_str()).collect();

    for joint in &model.joints {
        if !link_names.contains(joint.parent_link.as_str()) {
            errors.push(format!(
                "Joint '{}' references unknown parent link '{}'",
                joint.name, joint.parent_link
            ));
        }
        if !link_names.contains(joint.child_link.as_str()) {
            errors.push(format!(
                "Joint '{}' references unknown child link '{}'",
                joint.name, joint.child_link
            ));
        }
    }

    errors
}

// ---- XML helpers ----

fn emit_origin_list(out: &mut String, vals: &[f64], indent: usize) {
    let pad: String = " ".repeat(indent);
    if vals.len() >= 3 {
        out.push_str(&format!(
            "{pad}<origin xyz=\"{} {} {}\"/>\n",
            vals[0], vals[1], vals[2]
        ));
    }
}

fn emit_geometry(out: &mut String, geom: &GeometryInfo, indent: usize) {
    let pad: String = " ".repeat(indent);
    match geom.shape {
        GeometryShape::Box => {
            let d = &geom.dimensions;
            if d.len() >= 3 {
                out.push_str(&format!("{pad}<box size=\"{} {} {}\"/>\n", d[0], d[1], d[2]));
            }
        }
        GeometryShape::Cylinder => {
            let d = &geom.dimensions;
            if d.len() >= 2 {
                out.push_str(&format!(
                    "{pad}<cylinder radius=\"{}\" length=\"{}\"/>\n", d[0], d[1]
                ));
            }
        }
        GeometryShape::Sphere => {
            let d = &geom.dimensions;
            if !d.is_empty() {
                out.push_str(&format!("{pad}<sphere radius=\"{}\"/>\n", d[0]));
            }
        }
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
