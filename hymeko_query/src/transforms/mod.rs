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

use std::fmt::Write;

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

    /// Subdirectory name under `<workspace>/transforms/` that holds the
    /// transform's `queries.hymeko` + `template.<ext>` pair. Returns
    /// `None` for transforms that don't (yet) have a template; the
    /// registry's `render_from_templates` then falls back to whatever
    /// the caller wires up (typically the legacy hard-coded emitters
    /// in `formats/*.rs`). This is the hook that makes the generation
    /// pipeline data-driven — templates are *files*, not `push_str`
    /// calls.
    fn template_dir(&self) -> Option<&'static str> {
        None
    }

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
        reg.register(Box::new(GazeboWorldTransform));
        reg.register(Box::new(MermaidTransform));
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

    /// Render a registered transform through the **template engine**
    /// (`hymeko_query::rewrite::template::execute_transform`) — the
    /// canonical data-driven path. The transform's output is produced
    /// by rendering the `template.<ext>` file against query results
    /// from its `queries.hymeko`, *not* by Rust-side `push_str`
    /// calls. Every format with a registered template (urdf, sdf,
    /// mjcf, dot, gazebo, mermaid) can go through this entry point.
    ///
    /// `transforms_root` is the workspace-level `transforms/` directory
    /// (i.e. the one containing `urdf/`, `sdf/`, `mjcf/`, …). In tests
    /// this is resolved via `env!("CARGO_MANIFEST_DIR")/../transforms`;
    /// in production the CLI passes the path from its config.
    ///
    /// Returns `None` when the transform has no registered template
    /// directory. Returns `Some(Err)` on I/O / parse failure so callers
    /// can surface the error.
    pub fn render_from_templates<R: crate::traits::NameResolver>(
        &self,
        name: &str,
        ir: &hymeko::ir::ir::Ir,
        resolver: &R,
        config: &TransformConfig,
        transforms_root: &Path,
    ) -> Option<Result<String, String>> {
        let t = self.get(name)?;
        let subdir = t.template_dir()?;
        let dir = transforms_root.join(subdir);
        Some(render_via_template(ir, resolver, name, &dir, config))
    }
}

/// Load `queries.hymeko` + `template.<ext>` from `dir`, build a
/// [`crate::rewrite::template::TransformSpec`], and hand off to
/// [`crate::rewrite::template::execute_transform`].
///
/// Extension matching is done by scanning the directory for a single
/// file whose name starts with `"template."` — this keeps the helper
/// ignorant of per-format quirks like `template.urdf.xml` vs
/// `template.world.sdf` vs `template.mmd`.
fn render_via_template<R: crate::traits::NameResolver>(
    ir: &hymeko::ir::ir::Ir,
    resolver: &R,
    name: &str,
    dir: &Path,
    config: &TransformConfig,
) -> Result<String, String> {
    use crate::rewrite::template::{execute_transform, TransformSpec};

    let query_path = dir.join("queries.hymeko");
    let query_source = std::fs::read_to_string(&query_path)
        .map_err(|e| format!("reading {}: {e}", query_path.display()))?;

    let template_source = find_template_file(dir)
        .map_err(|e| format!("locating template in {}: {e}", dir.display()))?;

    let spec = TransformSpec {
        name: name.to_string(),
        query_source,
        template_source,
    };

    let mut cfg_map: std::collections::HashMap<String, String> =
        config.options.clone();
    cfg_map.entry("robot_name".to_string()).or_insert(config.robot_name.clone());
    // Gazebo world uses {{config:world_name}} — default to `empty`
    // if the caller didn't specify it.
    cfg_map.entry("world_name".to_string()).or_insert("empty".to_string());

    execute_transform(ir, resolver, &spec, &cfg_map)
}

/// Find the single `template.*` file in a transform directory and
/// return its contents.
fn find_template_file(dir: &Path) -> Result<String, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("template."))
        {
            return std::fs::read_to_string(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()));
        }
    }
    Err("no `template.*` file found".to_string())
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
        Some(emit_urdf_stub(km, &config.robot_name))
    }
    fn template_dir(&self) -> Option<&'static str> { Some("urdf") }
}

// ─── SDF ──────────────────────────────────────────────────────────────────

pub struct SdfTransform;

impl DomainTransform for SdfTransform {
    fn name(&self) -> &'static str { "sdf" }
    fn extension(&self) -> &'static str { "sdf" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_sdf_stub(km, &config.robot_name))
    }
    fn template_dir(&self) -> Option<&'static str> { Some("sdf") }
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
    fn template_dir(&self) -> Option<&'static str> { Some("mjcf") }

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

// ─── Mermaid flowchart ───────────────────────────────────────────────────

/// Emits a Mermaid `flowchart TD` of the kinematic chain — renders inline
/// on GitHub, in the VS Code Mermaid preview, Obsidian, and most docs
/// sites without any Graphviz toolchain. Lossier than DOT for N-ary
/// hyperedges (Mermaid has no native hyperedge), but gives browser-native
/// rendering with zero external dependency.
pub struct MermaidTransform;

impl DomainTransform for MermaidTransform {
    fn name(&self) -> &'static str { "mermaid" }
    fn extension(&self) -> &'static str { "mmd" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_mermaid(km, config))
    }
    fn template_dir(&self) -> Option<&'static str> { Some("mermaid") }
}

// ─── Gazebo world (gz sim) ────────────────────────────────────────────────

/// `DomainTransform` front-end for the Gazebo world emitter.
///
/// The full emitter ([`crate::formats::gazebo::generate_gazebo_world`])
/// needs access to the raw `Ir` + `NameResolver` to walk `sim_plugin` /
/// `control_plugin` hyperedges — information that the `ModelView`
/// abstraction doesn't carry. The registry entry therefore emits a
/// plugins-stripped world skeleton (physics + ground plane + inline
/// robot model) so round-trip + count assertions still work; callers
/// that need the full plugin-populated output use the free function
/// directly, the same way URDF/SDF consumers already do.
pub struct GazeboWorldTransform;

impl DomainTransform for GazeboWorldTransform {
    fn name(&self) -> &'static str { "gazebo" }
    fn extension(&self) -> &'static str { "world.sdf" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_gazebo_world_stub(km, &config.robot_name, "default"))
    }
    fn template_dir(&self) -> Option<&'static str> { Some("gazebo") }
    fn validate(&self, model: &ModelView) -> Vec<Diagnostic> {
        // Same tree-topology check as MJCF — gz sim requires a DAG.
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
                        "gz sim: link `{}` is child of {} joints — tree required",
                        name, count
                    )));
                }
            }
        }
        diags
    }
}

impl DomainTransform for DotTransform {
    fn name(&self) -> &'static str { "dot" }
    fn extension(&self) -> &'static str { "dot" }
    fn accepts(&self) -> ModelKind { ModelKind::Kinematic }
    fn emit(&self, model: &ModelView, config: &TransformConfig) -> Option<String> {
        let km = model.as_kinematic()?;
        Some(emit_dot(km, config))
    }
    fn template_dir(&self) -> Option<&'static str> { Some("dot") }
}

// ═══════════════════════════════════════════════════════════════════════════
// Emit functions
// ═══════════════════════════════════════════════════════════════════════════

fn emit_urdf_stub(model: &KinematicModel, name: &str) -> String {
    // Delegate to the model-view rich emitter. Set the model name so the
    // `<robot name=...>` header matches the requested `config.robot_name`.
    let mut m = model.clone();
    m.name = name.to_string();
    crate::formats::urdf::generate_urdf_from_model(&m)
}

fn emit_gazebo_world_stub(_model: &KinematicModel, robot_name: &str, world_name: &str) -> String {
    // This deliberately mirrors the shape of
    // `generate_gazebo_world` but drops the inline robot model and the
    // extracted `sim_plugin` / `control_plugin` tags — the `ModelView`
    // abstraction doesn't expose the raw `Ir`, which is required to
    // walk those hyperedges. The stub is still useful as a launchable
    // floor-only world and as a round-trip anchor for tests that
    // exercise the registry surface.
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<sdf version=\"1.8\">\n");
    out.push_str(&format!("  <world name=\"{}\">\n", xml_escape(world_name)));
    out.push_str("    <!-- TODO: delegate to formats::gazebo::generate_gazebo_world -->\n");
    out.push_str("    <!-- Robot model (\"");
    out.push_str(&xml_escape(robot_name));
    out.push_str("\") + sim_plugin / control_plugin tags populated by the full emitter -->\n");
    out.push_str("    <plugin filename=\"gz-sim-physics-system\" name=\"gz::sim::systems::Physics\"/>\n");
    out.push_str("    <plugin filename=\"gz-sim-user-commands-system\" name=\"gz::sim::systems::UserCommands\"/>\n");
    out.push_str("    <plugin filename=\"gz-sim-scene-broadcaster-system\" name=\"gz::sim::systems::SceneBroadcaster\"/>\n");
    out.push_str("  </world>\n");
    out.push_str("</sdf>\n");
    out
}

fn emit_sdf_stub(model: &KinematicModel, name: &str) -> String {
    let mut m = model.clone();
    m.name = name.to_string();
    crate::formats::sdf::generate_sdf_from_model(&m)
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

fn emit_mermaid(model: &KinematicModel, config: &TransformConfig) -> String {
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "%% Generated by hymeko_query::transforms::MermaidTransform");
    let _ = writeln!(out, "%% Robot: {}", config.robot_name);
    let _ = writeln!(out, "flowchart TD");

    // Shared class definitions — links are rectangles, roots (world-style
    // frames that don't appear in model.links) are pill-shaped.
    let _ = writeln!(out, "    classDef link fill:#FFE4B5,stroke:#8B4513,stroke-width:2px,color:#000;");
    let _ = writeln!(out, "    classDef root fill:#DDD,stroke:#555,stroke-width:2px,color:#000;");
    let _ = writeln!(out);

    let link_ids: std::collections::HashSet<&str> =
        model.links.iter().map(|l| l.name.as_str()).collect();
    let roots = find_roots(model);

    // Emit explicit link nodes.
    for link in &model.links {
        let mass_suffix = link
            .mass
            .map(|m| format!("<br/>{:.2} kg", m))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "    {}[\"<b>{}</b>{mass_suffix}\"]:::link",
            mermaid_id(&link.name),
            escape_label(&link.name)
        );
    }

    // Emit root frames (world-style anchors that aren't `link`-typed in
    // the fixture — they appear only as joint parents).
    for r in &roots {
        if link_ids.contains(r.as_str()) {
            continue;
        }
        let _ = writeln!(
            out,
            "    {}([\"{}\"]):::root",
            mermaid_id(r),
            escape_label(r)
        );
    }
    let _ = writeln!(out);

    // Emit joint edges. Revolute / continuous arrows use the default
    // solid style; fixed joints are dashed (`-.->`); prismatic are
    // dotted (`-- dotted -->` via label trick).
    for j in &model.joints {
        let axis_letter = j.axis.map(|ax| {
            if ax[0].abs() > 0.5 { 'X' } else if ax[1].abs() > 0.5 { 'Y' } else { 'Z' }
        });
        let label = match (j.joint_type, axis_letter) {
            (JointType::Fixed, _) => format!("{} (fixed)", j.name),
            (JointType::Revolute, Some(a)) => format!("{} (rev, {a})", j.name),
            (JointType::Continuous, Some(a)) => format!("{} (cont, {a})", j.name),
            (JointType::Prismatic, Some(a)) => format!("{} (prismatic, {a})", j.name),
            (JointType::Revolute, None) => format!("{} (rev)", j.name),
            (JointType::Continuous, None) => format!("{} (cont)", j.name),
            (JointType::Prismatic, None) => format!("{} (prismatic)", j.name),
        };
        let arrow = match j.joint_type {
            JointType::Fixed => "-.->|\"",
            _ => "-->|\"",
        };
        let _ = writeln!(
            out,
            "    {} {arrow}{}\"| {}",
            mermaid_id(&j.parent_link),
            escape_label(&label),
            mermaid_id(&j.child_link)
        );
    }

    out
}

/// Sanitise a HyMeKo identifier so it's a legal Mermaid node id.
/// Mermaid ids can have ASCII alnum + underscore; we replace anything
/// else with `_`.
fn mermaid_id(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Escape characters that would break a Mermaid `["label"]` literal.
fn escape_label(s: &str) -> String {
    s.replace('"', "&quot;").replace('|', "&vert;")
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