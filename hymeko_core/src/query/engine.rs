use crate::common::ids::{DeclId, SymId};
use crate::ir::ir::{Ir, SignedRefR, ValueR};
use crate::query::predicate::*;

pub trait NameResolver {
    fn resolve(&self, id: SymId) -> &str;
}

impl NameResolver for crate::resolution::interner::Interner {
    fn resolve(&self, id: SymId) -> &str {
        self.resolve(id)
    }
}

impl NameResolver for crate::resolution::string_table::StringTable {
    fn resolve(&self, id: SymId) -> &str {
        self.resolve(id)
    }
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub matches: Vec<(DeclId, String)>,
}

impl QueryResult {
    pub fn len(&self) -> usize { self.matches.len() }
    pub fn is_empty(&self) -> bool { self.matches.is_empty() }

    pub fn decl_ids(&self) -> Vec<DeclId> {
        self.matches.iter().map(|(id, _)| *id).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.matches.iter().map(|(_, n)| n.as_str()).collect()
    }
}

// ============================================================
// Query configuration
// ============================================================

/// Configuration for the query engine.
pub struct QueryConfig {
    /// Maximum transitive inheritance depth (prevents infinite loops).
    pub max_inherit_depth: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self { max_inherit_depth: 8 }
    }
}

// ============================================================
// Query engine
// ============================================================

/// Domain-agnostic pattern-matching engine over a compiled `Ir`.
///
/// Evaluates [`Predicate`] trees against every declaration
/// in the IR and returns the set of matching `DeclId`s. It is
/// intentionally stateless — all domain knowledge lives in the
/// predicates (see [`crate::query::urdf`] for an example).
pub struct QueryEngine<'a, R: NameResolver> {
    ir: &'a Ir,
    resolver: &'a R,
    config: QueryConfig,
}

impl<'a, R: NameResolver> QueryEngine<'a, R> {
    pub fn new(ir: &'a Ir, resolver: &'a R) -> Self {
        Self { ir, resolver, config: QueryConfig::default() }
    }

    pub fn with_config(ir: &'a Ir, resolver: &'a R, config: QueryConfig) -> Self {
        Self { ir, resolver, config }
    }

    /// Expose IR for domain transforms that need direct access.
    pub fn ir(&self) -> &Ir { self.ir }

    /// Expose resolver for domain transforms.
    pub fn resolver(&self) -> &R { self.resolver }

    /// Run a single predicate against every declaration in the IR.
    pub fn query(&self, predicate: &Predicate) -> QueryResult {
        let mut matches = Vec::new();
        for (idx, decl) in self.ir.decl_nodes.iter().enumerate() {
            let did = DeclId(idx);
            if did.is_none() { continue; }
            if self.matches(did, predicate) {
                let name = self.resolver.resolve(decl.name).to_string();
                matches.push((did, name));
            }
        }
        QueryResult { matches }
    }

    /// Run multiple named queries in a single pass.
    pub fn query_all(&self, queries: &[NamedQuery]) -> Vec<(String, QueryResult)> {
        queries.iter()
            .map(|nq| (nq.label.clone(), self.query(&nq.predicate)))
            .collect()
    }

    /// Evaluate a predicate against a single declaration.
    ///
    /// Public so that domain modules (e.g. `urdf.rs`) can
    /// test individual elements without running a full scan.
    pub fn matches(&self, did: DeclId, pred: &Predicate) -> bool {
        if did.is_none() { return false; }
        let decl = &self.ir.decl_nodes[did.0];

        match pred {
            Predicate::Any => true,

            Predicate::And(subs) => subs.iter().all(|p| self.matches(did, p)),
            Predicate::Or(subs)  => subs.iter().any(|p| self.matches(did, p)),
            Predicate::Not(inner) => !self.matches(did, inner),

            Predicate::Kind(kind) => decl.kind == *kind,

            Predicate::Named(name) =>
                self.resolver.resolve(decl.name) == name.as_str(),

            Predicate::NamePrefix(prefix) =>
                self.resolver.resolve(decl.name).starts_with(prefix.as_str()),

            Predicate::InheritsFrom(base_name) =>
                self.check_inherits(did, base_name, self.config.max_inherit_depth),

            Predicate::HasTag(tag) =>
                decl.anno.tags.iter().any(|&t| self.resolver.resolve(t) == tag.as_str()),

            Predicate::HasChild(child_pred) =>
                self.ir.decl_children(did).any(|cid| self.matches(cid, child_pred)),

            Predicate::HasParent(parent_pred) => {
                let pid = self.ir.parent(did);
                pid.is_some() && self.matches(pid, parent_pred)
            }

            Predicate::HasValue(vp) => {
                match &decl.anno.value {
                    Some(ValueR::Num(v)) => vp.matches_num(*v),
                    Some(ValueR::Str(sid)) => vp.matches_str(self.resolver.resolve(*sid)),
                    _ => matches!(vp, ValuePredicate::Any),
                }
            }

            Predicate::ChildValue(child_name, vp) => {
                self.ir.decl_children(did).any(|cid| {
                    let child = &self.ir.decl_nodes[cid.0];
                    let name_ok = self.resolver.resolve(child.name) == child_name.as_str();
                    let val_ok = match &child.anno.value {
                        Some(ValueR::Num(v)) => vp.matches_num(*v),
                        Some(ValueR::Str(sid)) => vp.matches_str(self.resolver.resolve(*sid)),
                        _ => matches!(vp, ValuePredicate::Any),
                    };
                    name_ok && val_ok
                })
            }

            Predicate::HasPlusRef(tp)    => self.check_arc_ref(did, Some(1), tp),
            Predicate::HasMinusRef(tp)   => self.check_arc_ref(did, Some(-1), tp),
            Predicate::HasNeutralRef(tp) => self.check_arc_ref(did, Some(0), tp),
            Predicate::HasRef(tp)        => self.check_arc_ref(did, None, tp),
        }
    }

    // ---- Internal helpers ----

    /// Walk the `bases` vector transitively up to `depth` levels,
    /// returning `true` if any ancestor's resolved name equals
    /// `base_name`.
    ///
    /// Both `+ <isa> base` and `: base` syntax produce entries in the
    /// `bases` vector (as different `SignedRefR` variants), so this
    /// function handles all inheritance styles uniformly.
    fn check_inherits(&self, did: DeclId, base_name: &str, depth: usize) -> bool {
        if depth == 0 { return false; }

        let bases = self.get_bases(did);
        for base_ref in bases {
            let target = base_ref.target();
            if target.is_none() { continue; }

            let target_name = self.resolver.resolve(self.ir.decl_nodes[target.0].name);
            if target_name == base_name { return true; }
            if self.check_inherits(target, base_name, depth - 1) { return true; }
        }
        false
    }

    /// Retrieve the inheritance list (`bases`) for a declaration.
    /// Works for both nodes and edges. Returns empty slice for arcs.
    fn get_bases(&self, did: DeclId) -> &[SignedRefR] {
        if let Some(nid) = self.ir.as_node(did) {
            return &self.ir.nodes[nid.0].bases;
        }
        if let Some(eid) = self.ir.as_edge(did) {
            return &self.ir.edges[eid.0].bases;
        }
        &[]
    }

    /// Check whether an edge declaration has at least one arc ref
    /// with the given sign whose target matches `target_pred`.
    /// `sign = None` means any sign is accepted.
    fn check_arc_ref(
        &self,
        did: DeclId,
        sign: Option<i8>,
        target_pred: &Predicate,
    ) -> bool {
        let Some(eid) = self.ir.as_edge(did) else { return false; };
        let edge_rec = &self.ir.edges[eid.0];

        for &arc_id in &edge_rec.arcs {
            let arc = &self.ir.arcs[arc_id.0];
            for sref in &arc.refs {
                if let Some(expected) = sign {
                    if sref.sign() != expected { continue; }
                }
                let target = sref.target();
                if self.matches(target, target_pred) { return true; }
            }
        }
        false
    }
}