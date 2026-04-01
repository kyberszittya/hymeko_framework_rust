//! Kinematic model extraction from query results.
//!
//! This is the shared intermediate that URDF, SDF, Gazebo, and
//! Isaac Sim transforms all consume. It navigates the IR using
//! DeclIds from query results, extracting links, joints, topology,
//! geometry, axes, and origin transforms.

use crate::common::ids::DeclId;
use crate::ir::ir::{Ir, ValueR, SignedRefR};
use crate::query::engine::{NameResolver, QueryEngine};
use crate::query::predicate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointType {
    Fixed,
    Continuous,
    Revolute,
    Prismatic,
}

impl JointType {
    pub fn urdf_str(&self) -> &'static str {
        match self {
            Self::Fixed      => "fixed",
            Self::Continuous => "continuous",
            Self::Revolute   => "revolute",
            Self::Prismatic  => "prismatic",
        }
    }

    pub fn sdf_str(&self) -> &'static str {
        match self {
            Self::Fixed      => "fixed",
            Self::Continuous => "revolute", // SDF 1.7 has no continuous
            Self::Revolute   => "revolute",
            Self::Prismatic  => "prismatic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryShape {
    Box,
    Cylinder,
    Sphere,
}

#[derive(Debug, Clone)]
pub struct GeometryInfo {
    pub shape: GeometryShape,
    pub dimensions: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LinkInfo {
    pub did: DeclId,
    pub name: String,
    pub mass: Option<f64>,
    pub geometry: Option<GeometryInfo>,
    pub origin: Option<Vec<f64>>,
    pub color: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct JointLimits {
    pub lower: f64,
    pub upper: f64,
    pub effort: f64,
    pub velocity: f64,
}

#[derive(Debug, Clone)]
pub struct JointInfo {
    pub did: DeclId,
    pub name: String,
    pub joint_type: JointType,
    pub parent_link: String,
    pub child_link: String,
    pub axis: Option<[f64; 3]>,
    pub origin_xyz: Option<[f64; 3]>,
    pub origin_rpy_deg: Option<[f64; 3]>,
    pub limits: Option<JointLimits>,
}

impl JointInfo {
    /// Convert origin RPY from degrees to radians.
    pub fn origin_rpy_rad(&self) -> Option<[f64; 3]> {
        self.origin_rpy_deg.map(|rpy| {
            let d2r = std::f64::consts::PI / 180.0;
            [rpy[0] * d2r, rpy[1] * d2r, rpy[2] * d2r]
        })
    }
}

#[derive(Debug, Clone)]
pub struct KinematicModel {
    pub name: String,
    pub links: Vec<LinkInfo>,
    pub joints: Vec<JointInfo>,
}

/// Extract a complete kinematic model from the IR.
pub fn extract_kinematic_model<R: NameResolver>(
    engine: &QueryEngine<'_, R>,
    robot_name: &str,
) -> KinematicModel {
    let ir = engine.ir();
    let res = engine.resolver();

    // Query links
    let link_results = engine.query(
        &Predicate::node().and(Predicate::inherits("link"))
    );

    let links: Vec<LinkInfo> = link_results.matches.iter().map(|(did, name)| {
        let mass = find_child_num(ir, res, *did, "mass");
        let geometry = extract_geometry(ir, res, *did);
        let origin = find_child_list(ir, res, *did, "origin");

        // Color via reference: `color -> diff_robot.body_color;`
        let color = find_child_ref_value_list(ir, res, *did, "color");

        LinkInfo {
            did: *did,
            name: name.clone(),
            mass,
            geometry,
            origin,
            color,
        }
    }).collect();

    // Query joints by type
    let mut joints = Vec::new();
    let joint_types = [
        ("fixed_joint", JointType::Fixed),
        ("conti_joint", JointType::Continuous),
        ("rev_joint", JointType::Revolute),
        ("prismatic_joint", JointType::Prismatic),
    ];

    for (base_name, jtype) in &joint_types {
        let results = engine.query(
            &Predicate::edge().and(Predicate::inherits(base_name))
        );

        for (did, name) in &results.matches {
            let topo = extract_joint_topology(ir, res, *did);
            joints.push(JointInfo {
                did: *did,
                name: name.clone(),
                joint_type: *jtype,
                parent_link: topo.parent_link,
                child_link: topo.child_link,
                axis: topo.axis,
                origin_xyz: topo.origin_xyz,
                origin_rpy_deg: topo.origin_rpy,
                limits: None,
            });
        }
    }

    KinematicModel {
        name: robot_name.to_string(),
        links,
        joints,
    }
}

// ============================================================
// IR navigation helpers
// ============================================================

/// Find a direct child node by name and return its numeric value.
fn find_child_num<R: NameResolver>(ir: &Ir, res: &R, did: DeclId, name: &str) -> Option<f64> {
    ir.decl_children(did).find_map(|cid| {
        let child = &ir.decl_nodes[cid.0];
        if res.resolve(child.name) == name {
            match &child.anno.value {
                Some(ValueR::Num(v)) => Some(*v),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// Find a direct child node by name and return its list value as Vec<f64>.
fn find_child_list<R: NameResolver>(ir: &Ir, res: &R, did: DeclId, name: &str) -> Option<Vec<f64>> {
    ir.decl_children(did).find_map(|cid| {
        let child = &ir.decl_nodes[cid.0];
        if res.resolve(child.name) == name {
            extract_num_list(&child.anno.value)
        } else {
            None
        }
    })
}

/// Find a child that has a Ref value (like `color -> body_color;`),
/// follow the ref to the target, and extract its value as Vec<f64>.
fn find_child_ref_value_list<R: NameResolver>(
    ir: &Ir, res: &R, did: DeclId, child_name: &str,
) -> Option<Vec<f64>> {
    ir.decl_children(did).find_map(|cid| {
        let child = &ir.decl_nodes[cid.0];
        if res.resolve(child.name) != child_name { return None; }
        match &child.anno.value {
            Some(ValueR::Ref(target_did)) => {
                if target_did.is_none() { return None; }
                let target = &ir.decl_nodes[target_did.0];
                extract_num_list(&target.anno.value)
            }
            _ => None,
        }
    })
}

fn extract_num_list(val: &Option<ValueR>) -> Option<Vec<f64>> {
    match val {
        Some(ValueR::List(items)) => {
            let nums: Vec<f64> = items.iter().filter_map(|v| {
                match v {
                    ValueR::Num(n) => Some(*n),
                    ValueR::List(inner) => {
                        // Nested list: flatten one level
                        None // handled separately in weight extraction
                    }
                    _ => None,
                }
            }).collect();
            if nums.is_empty() { None } else { Some(nums) }
        }
        _ => None,
    }
}

/// Extract geometry from a child named `link_geometry` that inherits
/// from box/cylinder/sphere.
fn extract_geometry<R: NameResolver>(ir: &Ir, res: &R, link_did: DeclId) -> Option<GeometryInfo> {
    ir.decl_children(link_did).find_map(|cid| {
        let child = &ir.decl_nodes[cid.0];
        let cname = res.resolve(child.name);
        if cname != "link_geometry" { return None; }

        if let Some(nid) = ir.as_node(cid) {
            for base in &ir.nodes[nid.0].bases {
                let target = base.target();
                if target.is_none() { continue; }
                let base_name = res.resolve(ir.decl_nodes[target.0].name);
                let shape = match base_name {
                    "box"      => Some(GeometryShape::Box),
                    "cylinder" => Some(GeometryShape::Cylinder),
                    "sphere"   => Some(GeometryShape::Sphere),
                    _ => None,
                };
                if let Some(shape) = shape {
                    let dims = find_child_list(ir, res, cid, "dimension")
                        .unwrap_or_default();
                    return Some(GeometryInfo { shape, dimensions: dims });
                }
            }
        }
        None
    })
}

// ============================================================
// Joint topology extraction
// ============================================================

struct JointTopology {
    parent_link: String,
    child_link: String,
    axis: Option<[f64; 3]>,
    origin_xyz: Option<[f64; 3]>,
    origin_rpy: Option<[f64; 3]>,
}

/// Extract parent/child link, axis, and origin from joint arc refs.
///
/// The convention in HyMeKo:
///   `+ base_link, [[x,y,z],[r,p,y]] - wheel_fr, - AXIS_Z`
///
/// - Plus ref → parent link (origin weights on this ref)
/// - Minus ref to something inheriting `link` → child link
/// - Minus ref to something inheriting `axis_definition` → joint axis
fn extract_joint_topology<R: NameResolver>(
    ir: &Ir, res: &R, edge_did: DeclId,
) -> JointTopology {
    let mut topo = JointTopology {
        parent_link: "unknown".to_string(),
        child_link: "unknown".to_string(),
        axis: None,
        origin_xyz: None,
        origin_rpy: None,
    };

    let Some(eid) = ir.as_edge(edge_did) else { return topo; };
    let edge_rec = &ir.edges[eid.0];

    for &arc_id in &edge_rec.arcs {
        let arc = &ir.arcs[arc_id.0];
        for sref in &arc.refs {
            let sign = sref.sign();
            let target = sref.target();
            if target.is_none() { continue; }

            let target_name = res.resolve(ir.decl_nodes[target.0].name);
            let is_axis = check_inherits_simple(ir, res, target, "axis_definition", 4);
            let is_link = check_inherits_simple(ir, res, target, "link", 4);

            if sign == 1 && is_link {
                topo.parent_link = target_name.to_string();
                // Extract origin from the weight annotation on this ref
                let atom = sref.atom();
                if let Some(ref weights) = atom.weights {
                    extract_origin_from_weights(weights, &mut topo);
                }
            } else if sign == -1 && is_link {
                topo.child_link = target_name.to_string();
            } else if sign == -1 && is_axis {
                let ax_vals = find_child_list(ir, res, target, "ax");
                if let Some(v) = ax_vals {
                    if v.len() >= 3 {
                        topo.axis = Some([v[0], v[1], v[2]]);
                    }
                }
            }
        }
    }

    topo
}

/// Extract xyz and rpy from weight annotations like `[[x,y,z],[r,p,y]]`.
fn extract_origin_from_weights(weights: &[ValueR], topo: &mut JointTopology) {
    // The weights Vec typically has structure:
    // [List([Num(x), Num(y), Num(z)]), List([Num(r), Num(p), Num(y)])]
    if let Some(xyz) = extract_3vec_from_value(weights.first()) {
        topo.origin_xyz = Some(xyz);
    }
    if let Some(rpy) = extract_3vec_from_value(weights.get(1)) {
        topo.origin_rpy = Some(rpy);
    }
}

fn extract_3vec_from_value(val: Option<&ValueR>) -> Option<[f64; 3]> {
    match val? {
        ValueR::List(items) if items.len() >= 3 => {
            let mut arr = [0.0f64; 3];
            for (i, item) in items.iter().take(3).enumerate() {
                if let ValueR::Num(n) = item {
                    arr[i] = *n;
                }
            }
            Some(arr)
        }
        _ => None,
    }
}

/// Simple transitive inheritance check (standalone, not requiring QueryEngine).
fn check_inherits_simple<R: NameResolver>(
    ir: &Ir, res: &R, did: DeclId, base_name: &str, depth: usize,
) -> bool {
    if depth == 0 || did.is_none() { return false; }

    if let Some(nid) = ir.as_node(did) {
        for base_ref in &ir.nodes[nid.0].bases {
            let target = base_ref.target();
            if target.is_none() { continue; }
            let tname = res.resolve(ir.decl_nodes[target.0].name);
            if tname == base_name { return true; }
            if check_inherits_simple(ir, res, target, base_name, depth - 1) { return true; }
        }
    }
    if let Some(eid) = ir.as_edge(did) {
        for base_ref in &ir.edges[eid.0].bases {
            let target = base_ref.target();
            if target.is_none() { continue; }
            let tname = res.resolve(ir.decl_nodes[target.0].name);
            if tname == base_name { return true; }
            if check_inherits_simple(ir, res, target, base_name, depth - 1) { return true; }
        }
    }

    false
}
