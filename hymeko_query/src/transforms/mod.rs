//! Domain transform ecosystem for HyMeKo.
//!
//! Module structure:
//!   mod.rs            — DomainTransform trait, Registry, Config, Diagnostic
//!   model_view.rs     — ModelView enum, ModelKind, extract() generic functions
//!   transform_engine.rs — TransformEngine (CLI orchestrator, generic on R)
//!
//! The separation ensures:
//!   - DomainTransform is dyn-compatible (no generics)
//!   - Extraction is generic on R: NameResolver (concrete, sized)
//!   - TransformEngine bridges the two layers

pub mod model_view;
pub mod transform_engine;

// Re-exports
pub use model_view::{ModelView, ModelKind, extract, extract_kinematic};
pub use transform_engine::TransformEngine;

use crate::kinematics::kinematic::{GeometryShape, KinematicModel};
use crate::kinematics::joints::JointType;
use std::path::Path;

// ─── Core trait (dyn-compatible: no generics) ─────────────────────────────

/// A domain transform generates output from an extracted model.
///
/// **Dyn-safe**: stored as `Box<dyn DomainTransform>` in the registry.
/// Never touches `Ir` or `NameResolver` — those are handled by
/// `model_view::extract()` before this trait is called.
pub trait DomainTransform {
    /// Unique identifier (e.g., "urdf", "sdf", "mjcf", "dot").
    fn name(&self) -> &'static str;

    /// Output file extension.
    fn extension(&self) -> &'static str;

    /// What model kind this transform consumes.
    fn accepts(&self) -> ModelKind;

    /// Generate output. Returns `None` if model kind doesn't match.
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String>;

    /// Validate before generation. Default checks joint topology.
    fn validate(&self, model: &ModelView) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        if let Some(km) = model.as_kinematic() {
            let link_names: std::collections::HashSet<&str> =
                km.links.iter().map(|l| l.name.as_str()).collect();
            for joint in &km.joints {
                if !link_names.contains(joint.parent_link.as_str()) {
                    diags.push(Diagnostic::error(format!(
                        "Joint '{}': unknown parent '{}'", joint.name, joint.parent_link
                    )));
                }
                if !link_names.contains(joint.child_link.as_str()) {
                    diags.push(Diagnostic::error(format!(
                        "Joint '{}': unknown child '{}'", joint.name, joint.child_link
                    )));
                }
            }
        }
        diags
    }
}

// ─── Configuration ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TransformConfig {
    pub robot_name: String,
    pub emit_comments: bool,
    pub indent: String,
    pub options: std::collections::HashMap<String, String>,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            robot_name: "robot".into(),
            emit_comments: true,
            indent: "  ".into(),
            options: Default::default(),
        }
    }
}

impl TransformConfig {
    pub fn with_name(mut self, name: &str) -> Self {
        self.robot_name = name.into(); self
    }
    pub fn with_option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.into(), value.into()); self
    }
}

// ─── Diagnostics ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagLevel { Info, Warning, Error }

impl Diagnostic {
    pub fn info(msg: impl Into<String>) -> Self { Self { level: DiagLevel::Info, message: msg.into() } }
    pub fn warning(msg: impl Into<String>) -> Self { Self { level: DiagLevel::Warning, message: msg.into() } }
    pub fn error(msg: impl Into<String>) -> Self { Self { level: DiagLevel::Error, message: msg.into() } }
    pub fn is_error(&self) -> bool { self.level == DiagLevel::Error }
}

// ─── Registry ─────────────────────────────────────────────────────────────

pub struct TransformRegistry {
    pub transforms: Vec<Box<dyn DomainTransform>>,
}

impl Default for TransformRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(UrdfTransform));
        reg.register(Box::new(SdfTransform));
        reg.register(Box::new(MjcfTransform));
        reg.register(Box::new(DotTransform));
        reg
    }
}

impl TransformRegistry {
    pub fn new() -> Self { Self { transforms: Vec::new() } }

    pub fn register(&mut self, t: Box<dyn DomainTransform>) {
        self.transforms.push(t);
    }

    pub fn get(&self, name: &str) -> Option<&dyn DomainTransform> {
        self.transforms.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub fn by_extension(&self, ext: &str) -> Option<&dyn DomainTransform> {
        self.transforms.iter().find(|t| t.extension() == ext).map(|t| t.as_ref())
    }

    pub fn available(&self) -> Vec<&str> {
        self.transforms.iter().map(|t| t.name()).collect()
    }

    /// Emit all formats from a pre-extracted model.
    pub fn emit_all(&self, model: &ModelView, config: &TransformConfig) -> Vec<(String, String)> {
        self.transforms.iter().filter_map(|t| {
            let output = t.emit(model, config)?;
            Some((format!("{}.{}", config.robot_name, t.extension()), output))
        }).collect()
    }

    /// Write all formats to disk.
    pub fn write_all(
        &self, model: &ModelView, config: &TransformConfig, dir: &Path,
    ) -> std::io::Result<Vec<std::path::PathBuf>> {
        let mut paths = Vec::new();
        for (filename, content) in self.emit_all(model, config) {
            let path = dir.join(&filename);
            std::fs::write(&path, &content)?;
            paths.push(path);
        }
        Ok(paths)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Transform implementations
// ═══════════════════════════════════════════════════════════════════════════

// ─── URDF ─────────────────────────────────────────────────────────────────

pub struct UrdfTransform;

impl DomainTransform for UrdfTransform {
    fn name(&self) -> &'static str { "urdf" }
    fn extension(&self) -> &'static str { "urdf" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        // TODO: delegate to crate::formats::urdf::generate_urdf_from_model(km)
        Some(emit_urdf_stub(km, &config.robot_name))
    }
}

// ─── SDF ──────────────────────────────────────────────────────────────────

pub struct SdfTransform;

impl DomainTransform for SdfTransform {
    fn name(&self) -> &'static str { "sdf" }
    fn extension(&self) -> &'static str { "sdf" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        // TODO: delegate to crate::formats::sdf::generate_sdf_from_model(km)
        Some(emit_sdf_stub(km, &config.robot_name))
    }
}

// ─── MJCF ─────────────────────────────────────────────────────────────────

pub struct MjcfTransform;

impl DomainTransform for MjcfTransform {
    fn name(&self) -> &'static str { "mjcf" }
    fn extension(&self) -> &'static str { "xml" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }

    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_mjcf(km, config))
    }

    fn validate(&self, model: &ModelView) -> Vec<Diagnostic> {
        let mut diags = Vec::new();
        if let Some(km) = model.as_kinematic() {
            let mut child_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for j in &km.joints {
                *child_counts.entry(j.child_link.as_str()).or_insert(0) += 1;
            }
            for (name, count) in &child_counts {
                if *count > 1 {
                    diags.push(Diagnostic::error(format!(
                        "'{}' is child in {} joints — MJCF requires tree", name, count
                    )));
                }
            }
        }
        diags
    }
}

// ─── DOT ──────────────────────────────────────────────────────────────────

pub struct DotTransform;

impl DomainTransform for DotTransform {
    fn name(&self) -> &'static str { "dot" }
    fn extension(&self) -> &'static str { "dot" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_dot(km, config))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Emit functions
// ═══════════════════════════════════════════════════════════════════════════

fn emit_urdf_stub(model: &KinematicModel, name: &str) -> String {
    // Stub — replace with full generation from formats::urdf
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str(&format!("<robot name=\"{}\">\n", xml_escape(name)));
    out.push_str("  <!-- TODO: delegate to formats::urdf::generate_urdf_from_model -->\n");
    out.push_str("</robot>\n");
    out
}

fn emit_sdf_stub(model: &KinematicModel, name: &str) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<sdf version=\"1.7\">\n");
    out.push_str(&format!("  <model name=\"{}\">\n", name));
    out.push_str("    <!-- TODO: delegate to formats::sdf::generate_sdf_from_model -->\n");
    out.push_str("  </model>\n");
    out.push_str("</sdf>\n");
    out
}

fn emit_mjcf(model: &KinematicModel, config: &TransformConfig) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str(&format!("<mujoco model=\"{}\">\n", config.robot_name));
    out.push_str("  <compiler angle=\"radian\" meshdir=\".\"/>\n");
    out.push_str("  <option gravity=\"0 0 -9.81\" timestep=\"0.001\"/>\n\n");

    // Assets
    out.push_str("  <asset>\n");
    for link in &model.links {
        if let Some(ref c) = link.color {
            out.push_str(&format!(
                "    <material name=\"mat_{}\" rgba=\"{} {} {} {}\"/>\n",
                link.name, c.get(0).unwrap_or(&0.5), c.get(1).unwrap_or(&0.5),
                c.get(2).unwrap_or(&0.5), c.get(3).unwrap_or(&1.0),
            ));
        }
    }
    out.push_str("  </asset>\n\n");

    out.push_str("  <worldbody>\n");
    for root in find_roots(model) {
        emit_mjcf_body(&mut out, model, &root, 4);
    }
    out.push_str("  </worldbody>\n\n");

    out.push_str("  <actuator>\n");
    for j in &model.joints {
        if j.joint_type != JointType::Fixed {
            out.push_str(&format!("    <motor name=\"act_{}\" joint=\"{}\" gear=\"1\"/>\n", j.name, j.name));
        }
    }
    out.push_str("  </actuator>\n");
    out.push_str("</mujoco>\n");
    out
}

fn emit_mjcf_body(out: &mut String, model: &KinematicModel, link_name: &str, indent: usize) {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let incoming = model.joints.iter().find(|j| j.child_link == link_name);

    out.push_str(&format!("{}<body name=\"{}\"", pad, link_name));
    if let Some(j) = incoming {
        if let Some(xyz) = j.origin_xyz {
            out.push_str(&format!(" pos=\"{} {} {}\"", xyz[0], xyz[1], xyz[2]));
        }
        if let Some(rpy) = j.origin_rpy_rad() {
            if rpy.iter().any(|&v| v.abs() > 1e-9) {
                out.push_str(&format!(" euler=\"{:.4} {:.4} {:.4}\"", rpy[0], rpy[1], rpy[2]));
            }
        }
    }
    out.push_str(">\n");

    if let Some(link) = model.links.iter().find(|l| l.name == link_name) {
        if let Some(mass) = link.mass {
            let i = mass * 0.01;
            out.push_str(&format!("{}<inertial mass=\"{}\" diaginertia=\"{} {} {}\"/>\n", pad2, mass, i, i, i));
        }
        if let Some(ref geom) = link.geometry {
            let gs = match geom.shape {
                GeometryShape::Box if geom.dimensions.len() >= 3 => {
                    let d = &geom.dimensions;
                    format!("type=\"box\" size=\"{} {} {}\"", d[0]/2.0, d[1]/2.0, d[2]/2.0)
                }
                GeometryShape::Cylinder if geom.dimensions.len() >= 2 => {
                    let d = &geom.dimensions;
                    format!("type=\"cylinder\" size=\"{} {}\"", d[0], d[1]/2.0)
                }
                GeometryShape::Sphere if !geom.dimensions.is_empty() =>
                    format!("type=\"sphere\" size=\"{}\"", geom.dimensions[0]),
                _ => String::new(),
            };
            if !gs.is_empty() {
                let mat = link.color.as_ref().map(|_| format!(" material=\"mat_{}\"", link.name)).unwrap_or_default();
                out.push_str(&format!("{}<geom {}{}/>  \n", pad2, gs, mat));
            }
        }
    }

    if let Some(j) = incoming {
        if j.joint_type != JointType::Fixed {
            let jtype = match j.joint_type {
                JointType::Revolute | JointType::Continuous => "hinge",
                JointType::Prismatic => "slide",
                JointType::Fixed => unreachable!(),
            };
            let ax = j.axis.unwrap_or([0.0, 0.0, 1.0]);
            let range = j.limits.as_ref().map(|lim| {
                let d2r = std::f64::consts::PI / 180.0;
                format!(" range=\"{:.4} {:.4}\"", lim.lower * d2r, lim.upper * d2r)
            }).unwrap_or_default();
            out.push_str(&format!("{}<joint name=\"{}\" type=\"{}\" axis=\"{} {} {}\"{}/>\n",
                                  pad2, j.name, jtype, ax[0], ax[1], ax[2], range));
        }
    }

    for child in model.joints.iter().filter(|j| j.parent_link == link_name).map(|j| j.child_link.as_str()) {
        emit_mjcf_body(out, model, child, indent + 2);
    }

    out.push_str(&format!("{}</body>\n", pad));
}

fn emit_dot(model: &KinematicModel, config: &TransformConfig) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str(&format!("digraph \"{}\" {{\n", config.robot_name));
    out.push_str("  rankdir=TB;\n");
    out.push_str("  node [shape=box, style=rounded, fontname=\"sans-serif\", fontsize=10];\n");
    out.push_str("  edge [fontname=\"sans-serif\", fontsize=9];\n\n");

    for link in &model.links {
        let label = link.mass.map_or(link.name.clone(), |m| format!("{}\\n{:.1} kg", link.name, m));
        out.push_str(&format!("  \"{}\" [label=\"{}\"];\n", link.name, label));
    }
    out.push('\n');

    for j in &model.joints {
        let style = match j.joint_type {
            JointType::Fixed => "dashed",
            JointType::Revolute | JointType::Continuous => "bold",
            JointType::Prismatic => "dotted",
        };
        let axis_label = j.axis.map(|ax| {
            let letter = if ax[0].abs() > 0.5 { "X" } else if ax[1].abs() > 0.5 { "Y" } else { "Z" };
            format!("{}\\n({})", j.name, letter)
        }).unwrap_or_else(|| j.name.clone());
        out.push_str(&format!("  \"{}\" -> \"{}\" [label=\"{}\", style={}];\n",
                              j.parent_link, j.child_link, axis_label, style));
    }

    out.push_str("}\n");
    out
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn find_roots(model: &KinematicModel) -> Vec<String> {
    // Find parent_link names that never appear as child_link.
    // This handles frames like "world" that aren't in model.links.
    let children: std::collections::HashSet<&str> =
        model.joints.iter().map(|j| j.child_link.as_str()).collect();
    let parents: std::collections::HashSet<&str> =
        model.joints.iter().map(|j| j.parent_link.as_str()).collect();
    let mut roots: Vec<String> = parents.difference(&children)
        .map(|s| s.to_string())
        .collect();
    // Fallback: if no joints exist, use links not appearing as children
    if roots.is_empty() {
        roots = model.links.iter()
            .filter(|l| !children.contains(l.name.as_str()))
            .map(|l| l.name.clone())
            .collect();
    }
    roots
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;")
        .replace('>', "&gt;").replace('"', "&quot;")
}