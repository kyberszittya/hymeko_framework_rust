//! Text generation from query results.
//! Dispatches to format-specific generators (URDF, SDF, MuJoCo MJCF).

use hymeko::ir::ir::Ir;
use crate::{NameResolver, QueryEngine};
use crate::formats::sdf::generate_sdf;
use crate::formats::urdf::generate_urdf;
use crate::kinematics::joints::JointType;
use crate::kinematics::kinematic::{extract_kinematic_model, GeometryShape, KinematicModel};

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Urdf,
    Sdf17,
    Mjcf,
    DotGraph,
}

/// Unified codegen entry point.
///
/// 1. Runs predefined kinematic queries
/// 2. Extracts KinematicModel
/// 3. Generates target format string
pub fn generate_description<R: NameResolver>(
    ir: &Ir,
    resolver: &R,
    robot_name: &str,
    format: OutputFormat,
) -> Result<String, CodegenError> {
    match format {
        // URDF and SDF have their own extraction pipelines
        OutputFormat::Urdf => Ok(generate_urdf(ir, resolver, robot_name)),
        OutputFormat::Sdf17 => Ok(generate_sdf(ir, resolver, robot_name)),
        // MJCF and DOT use the shared KinematicModel
        OutputFormat::Mjcf | OutputFormat::DotGraph => {
            let engine = QueryEngine::new(ir, resolver);
            let model = extract_kinematic_model(&engine, robot_name);
            match format {
                OutputFormat::Mjcf => Ok(generate_mjcf(&model, robot_name)),
                OutputFormat::DotGraph => Ok(generate_dot(&model, robot_name)),
                _ => unreachable!(),
            }
        }
    }
}

#[derive(Debug)]
pub enum CodegenError {
    QueryFailed(String),
    MissingField(String),
    InvalidTopology(String),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueryFailed(s) => write!(f, "Query failed: {s}"),
            Self::MissingField(s) => write!(f, "Missing required field: {s}"),
            Self::InvalidTopology(s) => write!(f, "Invalid topology: {s}"),
        }
    }
}

/// MuJoCo MJCF generation (new domain transform)
fn generate_mjcf(model: &KinematicModel, robot_name: &str) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str(&format!("<mujoco model=\"{robot_name}\">\n"));
    out.push_str("  <compiler angle=\"radian\" meshdir=\"meshes/\"/>\n");
    out.push_str("  <worldbody>\n");

    // Build kinematic tree by finding the root link (no parent joint)
    let child_links: std::collections::HashSet<&str> = model.joints.iter()
        .map(|j| j.child_link.as_str())
        .collect();

    let root_links: Vec<_> = model.links.iter()
        .filter(|l| !child_links.contains(l.name.as_str()))
        .collect();

    for root in &root_links {
        emit_mjcf_body(&mut out, &model, &root.name, 2);
    }

    out.push_str("  </worldbody>\n");

    // Actuators
    out.push_str("  <actuator>\n");
    for j in &model.joints {
        if j.joint_type != JointType::Fixed {
            out.push_str(&format!(
                "    <motor name=\"{}_motor\" joint=\"{}\" gear=\"1\"/>\n",
                j.name, j.name
            ));
        }
    }
    out.push_str("  </actuator>\n");
    out.push_str("</mujoco>\n");
    out
}

fn emit_mjcf_body(
    out: &mut String,
    model: &KinematicModel,
    link_name: &str,
    indent: usize,
) {
    let pad = " ".repeat(indent);
    let link = model.links.iter().find(|l| l.name == link_name);

    out.push_str(&format!("{pad}<body name=\"{link_name}\">\n"));

    if let Some(link) = link {
        // Inertial
        if let Some(mass) = link.mass {
            out.push_str(&format!(
                "{pad}  <inertial mass=\"{mass}\" pos=\"0 0 0\" diaginertia=\"{i} {i} {i}\"/>\n",
                i = mass * 0.01
            ));
        }

        // Geometry
        if let Some(ref geom) = link.geometry {
            let (gtype, size) = match geom.shape {
                GeometryShape::Box => {
                    // MuJoCo box size is half-extents
                    let dims = &geom.dimensions;
                    let s = if dims.len() >= 3 {
                        format!("{} {} {}", dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0)
                    } else {
                        "0.1 0.1 0.1".to_string()
                    };
                    ("box", s)
                }
                GeometryShape::Cylinder => {
                    let dims = &geom.dimensions;
                    let s = if dims.len() >= 2 {
                        format!("{} {}", dims[0], dims[1] / 2.0) // radius, half-length
                    } else {
                        "0.1 0.05".to_string()
                    };
                    ("cylinder", s)
                }
                GeometryShape::Sphere => {
                    let dims = &geom.dimensions;
                    let s = if !dims.is_empty() {
                        format!("{}", dims[0])
                    } else {
                        "0.1".to_string()
                    };
                    ("sphere", s)
                }
            };
            out.push_str(&format!("{pad}  <geom type=\"{gtype}\" size=\"{size}\"/>\n"));
        }
    }

    // Find child joints from this link
    let child_joints: Vec<_> = model.joints.iter()
        .filter(|j| j.parent_link == link_name)
        .collect();

    for joint in child_joints {
        let axis = joint.axis.unwrap_or([0.0, 0.0, 1.0]);
        out.push_str(&format!(
            "{pad}  <joint name=\"{}\" type=\"{}\" axis=\"{} {} {}\"",
            joint.name,
            mjcf_joint_type(joint.joint_type),
            axis[0], axis[1], axis[2],
        ));
        if let Some(ref lim) = joint.limits {
            out.push_str(&format!(" range=\"{} {}\"", lim.lower, lim.upper));
        }
        out.push_str("/>\n");

        emit_mjcf_body(out, model, &joint.child_link, indent + 2);
    }

    out.push_str(&format!("{pad}</body>\n"));
}

fn mjcf_joint_type(jtype: JointType) -> &'static str {
    match jtype {
        JointType::Revolute | JointType::Continuous => "hinge",
        JointType::Prismatic => "slide",
        JointType::Fixed => "fixed",
    }
}

/// DOT graph generation for visualization
fn generate_dot(model: &KinematicModel, robot_name: &str) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str(&format!("digraph {robot_name} {{\n"));
    out.push_str("  rankdir=TB;\n");
    out.push_str("  node [shape=box, style=filled, fillcolor=lightblue];\n");

    for link in &model.links {
        out.push_str(&format!("  \"{}\" [label=\"{}\"];\n", link.name, link.name));
    }

    out.push_str("  edge [color=red, fontcolor=red];\n");
    for joint in &model.joints {
        out.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{} ({})\"];\n",
            joint.parent_link, joint.child_link,
            joint.name, joint.joint_type.urdf_str()
        ));
    }

    out.push_str("}\n");
    out
}