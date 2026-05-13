//! Lower a parsed HyMeKo [`Description`] into a [`PGraphSchema`] +
//! P-graph side-tables (raws, products, costs).
//!
//! ## Encoding contract — idiomatic HyMeKo
//!
//! P-graphs map onto the canonical signed-incidence hypergraph IR
//! by reading:
//!
//! - **HyMeKo nodes** (no `@` prefix) tagged `<material>`  →  M-set.
//! - **HyMeKo hyperedges** (`@`-prefix) tagged `<unit>`    →  O-set.
//!
//! This is the natural mapping: a P-graph operating unit *is* a
//! hyperedge connecting input materials to output materials, so it
//! lives on the edge side of the IR. A material is a passive
//! resource type, so it lives on the node side.
//!
//! ### Sub-tags on materials
//!
//! - `<raw>`     — material is in the raw-material set $R$.
//! - `<product>` — material is in the required-product set $P$.
//!
//! ### Unit body
//!
//! An edge body holds **one** signed [`HyperArc`] whose signs encode
//! the unit's I/O signature:
//!
//! - `-Material` ⇒ Material is consumed (directed schema edge
//!   `Material → unit`).
//! - `+Material` ⇒ Material is produced (directed schema edge
//!   `unit → Material`).
//!
//! ### Cost
//!
//! The unit's scalar cost is the edge's numeric value, e.g.
//! `@Mixer <unit> 100 { (-A, -B, +C); }`. Default is $1.0$.
//!
//! For backwards compatibility a `cost N;` child node is also
//! recognised inside an edge body.

use std::collections::{BTreeMap, BTreeSet};

use hymeko::common::ids::{DeclId, EdgeId};
use parser::ast::{Anno, Description, EdgeDecl, HyperItem, NodeDecl, SignedRef, Value};
use thiserror::Error;

use crate::schema::{PGraphError, PGraphSchema, PNodeKind};

/// Errors raised while lowering a parsed [`Description`] into a P-graph.
#[derive(Debug, Error)]
pub enum LowerError {
    /// A declaration carried a tag set that doesn't classify it as
    /// either a Material or an Operating Unit.
    #[error("decl `{name}` has ambiguous P-graph role: {detail}")]
    AmbiguousRole {
        /// Offending name.
        name: String,
        /// Free-form detail.
        detail: String,
    },
    /// A unit body referenced a material name that is not declared.
    #[error("unit `{unit}` references unknown material `{target}`")]
    UnknownMaterial {
        /// Unit that contained the bad reference.
        unit: String,
        /// Referenced name.
        target: String,
    },
    /// A unit's hyperarc used a `~` (neutral) sign — P-graph edges must
    /// be signed (consume `-` or produce `+`).
    #[error("unit `{unit}` has a neutral (~) reference; P-graph edges must be + or -")]
    NeutralRef {
        /// Offending unit name.
        unit: String,
    },
    /// The lowered schema failed the bipartite invariant.
    #[error(transparent)]
    SchemaError(#[from] PGraphError),
}

/// Lowered P-graph: schema plus the side-tables MSG/SSG/ABB need.
#[derive(Debug, Clone)]
pub struct LoweredPGraph {
    /// Bipartite Material/Operating-Unit overlay.
    pub schema: PGraphSchema,
    /// Mapping from source name to assigned [`DeclId`].
    pub name_to_decl: BTreeMap<String, DeclId>,
    /// Reverse mapping for diagnostics.
    pub decl_to_name: BTreeMap<DeclId, String>,
    /// Materials in the raw-material set $R$.
    pub raws: BTreeSet<DeclId>,
    /// Materials in the required-product set $P$.
    pub products: BTreeSet<DeclId>,
    /// All M-nodes (mirrors `schema.m_nodes()` for convenience).
    pub materials: BTreeSet<DeclId>,
    /// All O-nodes (mirrors `schema.o_nodes()`).
    pub units: BTreeSet<DeclId>,
    /// Per-unit cost (default 1.0).
    pub costs: BTreeMap<DeclId, f64>,
    /// Per-unit *consumed* materials (set of M-nodes appearing with
    /// `-` sign in the unit's hyperarc).
    pub unit_inputs: BTreeMap<DeclId, BTreeSet<DeclId>>,
    /// Per-unit *produced* materials (set of M-nodes appearing with
    /// `+` sign in the unit's hyperarc).
    pub unit_outputs: BTreeMap<DeclId, BTreeSet<DeclId>>,
}

/// Lower a parsed [`Description`] into a P-graph.
///
/// Materials are collected from `HyperItem::Node` decls tagged
/// `<material>`; units are collected from `HyperItem::Edge` decls
/// tagged `<unit>`. A wrapper such as `context { ... }` is followed
/// transparently — the lowering recurses one level into untagged
/// node bodies.
pub fn lower<'a>(d: &Description<'a, &'a str>) -> Result<LoweredPGraph, LowerError> {
    // ── 1. Walk top-level items, separating materials (nodes) and
    //       units (edges).
    let mut mats: Vec<&NodeDecl<'a, &'a str>> = Vec::new();
    let mut units: Vec<&EdgeDecl<'a, &'a str>> = Vec::new();
    for item in &d.items {
        collect(item, &mut mats, &mut units);
    }

    // ── 2. Assign DeclIds, classify, build the kind map and side-sets.
    let mut name_to_decl: BTreeMap<String, DeclId> = BTreeMap::new();
    let mut decl_to_name: BTreeMap<DeclId, String> = BTreeMap::new();
    let mut kinds: BTreeMap<DeclId, PNodeKind> = BTreeMap::new();
    let mut raws: BTreeSet<DeclId> = BTreeSet::new();
    let mut products: BTreeSet<DeclId> = BTreeSet::new();

    for n in &mats {
        if !is_material(&n.anno) {
            return Err(LowerError::AmbiguousRole {
                name: n.inner.name.to_string(),
                detail: "node lacks <material> tag".into(),
            });
        }
        let id = intern(&mut name_to_decl, &mut decl_to_name, n.inner.name);
        kinds.insert(id, PNodeKind::Material);
        if n.anno.tags.iter().any(|t| *t == "raw") {
            raws.insert(id);
        }
        if n.anno.tags.iter().any(|t| *t == "product") {
            products.insert(id);
        }
    }
    for e in &units {
        if !is_unit(&e.anno) {
            return Err(LowerError::AmbiguousRole {
                name: e.inner.name.to_string(),
                detail: "@-edge lacks <unit> tag".into(),
            });
        }
        let id = intern(&mut name_to_decl, &mut decl_to_name, e.inner.name);
        kinds.insert(id, PNodeKind::OperatingUnit);
    }

    // ── 3. Walk each unit's body for the I/O hyperarc + cost.
    let mut edges: BTreeMap<EdgeId, (DeclId, DeclId)> = BTreeMap::new();
    let mut costs: BTreeMap<DeclId, f64> = BTreeMap::new();
    let mut unit_inputs: BTreeMap<DeclId, BTreeSet<DeclId>> = BTreeMap::new();
    let mut unit_outputs: BTreeMap<DeclId, BTreeSet<DeclId>> = BTreeMap::new();
    let mut next_edge: usize = 0;

    for e in &units {
        let unit_id = name_to_decl[e.inner.name];
        let unit_name = e.inner.name.to_string();
        unit_inputs.entry(unit_id).or_default();
        unit_outputs.entry(unit_id).or_default();

        // Cost source 1: the edge's value (idiomatic).
        let cost = match &e.anno.value {
            Some(Value::Num(v)) => *v,
            _ => 1.0,
        };
        costs.insert(unit_id, cost);

        for it in &e.inner.body {
            match it {
                HyperItem::Arc(a) => {
                    for sref in &a.inner.refs {
                        let (sign, target_name) = match sref {
                            SignedRef::Plus(r) => (1, leaf(&r.target.path)),
                            SignedRef::Minus(r) => (-1, leaf(&r.target.path)),
                            SignedRef::Neutral(_) => {
                                return Err(LowerError::NeutralRef {
                                    unit: unit_name.clone(),
                                });
                            }
                        };
                        let mat_id = *name_to_decl.get(target_name).ok_or_else(|| {
                            LowerError::UnknownMaterial {
                                unit: unit_name.clone(),
                                target: target_name.to_string(),
                            }
                        })?;
                        if sign < 0 {
                            edges.insert(EdgeId::new(next_edge), (mat_id, unit_id));
                            unit_inputs.get_mut(&unit_id).unwrap().insert(mat_id);
                        } else {
                            edges.insert(EdgeId::new(next_edge), (unit_id, mat_id));
                            unit_outputs.get_mut(&unit_id).unwrap().insert(mat_id);
                        }
                        next_edge += 1;
                    }
                }
                // Cost source 2: a `cost N;` child node — back-compat.
                HyperItem::Node(child) => {
                    if child.inner.name == "cost" {
                        if let Some(Value::Num(v)) = &child.anno.value {
                            costs.insert(unit_id, *v);
                        }
                    }
                }
                HyperItem::Edge(_) => { /* nested edges ignored */ }
            }
        }
    }

    // ── 4. Construct the schema (bipartite invariant check).
    let schema = PGraphSchema::try_new(kinds.clone(), edges)?;

    let materials: BTreeSet<DeclId> = kinds
        .iter()
        .filter_map(|(d, k)| (*k == PNodeKind::Material).then_some(*d))
        .collect();
    let units_set: BTreeSet<DeclId> = kinds
        .iter()
        .filter_map(|(d, k)| (*k == PNodeKind::OperatingUnit).then_some(*d))
        .collect();

    Ok(LoweredPGraph {
        schema,
        name_to_decl,
        decl_to_name,
        raws,
        products,
        materials,
        units: units_set,
        costs,
        unit_inputs,
        unit_outputs,
    })
}

// ─── helpers ────────────────────────────────────────────────────────

fn intern(
    name_to_decl: &mut BTreeMap<String, DeclId>,
    decl_to_name: &mut BTreeMap<DeclId, String>,
    name: &str,
) -> DeclId {
    if let Some(id) = name_to_decl.get(name) {
        return *id;
    }
    let id = DeclId::new(name_to_decl.len());
    name_to_decl.insert(name.to_string(), id);
    decl_to_name.insert(id, name.to_string());
    id
}

fn is_material(a: &Anno<'_, &str>) -> bool {
    a.tags.iter().any(|t| *t == "material")
}

fn is_unit(a: &Anno<'_, &str>) -> bool {
    a.tags.iter().any(|t| *t == "unit")
}

/// Walk an item tree, collecting tagged materials and tagged units.
/// Untagged nodes with bodies are descended into (this lets a
/// `context { ... }` wrapper hold the actual P-graph items without
/// itself being part of the P-graph).
fn collect<'a, 'b>(
    item: &'b HyperItem<'a, &'a str>,
    mats: &mut Vec<&'b NodeDecl<'a, &'a str>>,
    units: &mut Vec<&'b EdgeDecl<'a, &'a str>>,
) {
    match item {
        HyperItem::Node(n) => {
            if is_material(&n.anno) {
                mats.push(n);
            } else if let Some(body) = &n.inner.body {
                // Untagged node — likely a `context { ... }` wrapper.
                for child in body {
                    collect(child, mats, units);
                }
            }
        }
        HyperItem::Edge(e) => {
            if is_unit(&e.anno) {
                units.push(e);
            }
            // Untagged edges are ignored (they belong to other passes).
        }
        HyperItem::Arc(_) => { /* top-level arcs are ignored */ }
    }
}

fn leaf<'a>(path: &'a [&'a str]) -> &'a str {
    path.last().copied().unwrap_or("")
}
