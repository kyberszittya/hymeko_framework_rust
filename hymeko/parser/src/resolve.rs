use std::collections::HashMap;

use crate::ast::*;
use crate::common::ids::{DeclId, SymId};
use crate::common::pathkey::PathKey;
use crate::interner::Interner;
use crate::ir::ir::SignedRefR;

#[derive(Debug)]
pub struct Index {
    pub by_path: HashMap<PathKey, DeclId>, // FQ path -> unique DeclId




}

impl Index {
    /// Returns an iterator over all (PathKey, DeclId) pairs in the index
    pub fn iter(&self) -> impl Iterator<Item = (&PathKey, &DeclId)> {
        self.by_path.iter()
    }
}

#[derive(Debug)]
pub enum ResolveError {
    DuplicateDecl { path: String },
    UnresolvedRef { from_scope: String, target: String },
    AmbiguousRef { from_scope: String, target: String, candidates: Vec<String> },
}

pub fn build_index_sym(d: &AstSym, it: &Interner) -> Result<Index, ResolveError> {
    let mut idx = Index { by_path: HashMap::new() };
    let mut next: u32 = 0;

    // header nodes
    for n in &d.header {
        index_node(&mut idx, &mut next, &[], n, it)?;
    }

    // top-level items
    index_items(&mut idx, &mut next, &[], &d.items, it)?;

    Ok(idx)
}

fn index_items(
    idx: &mut Index,
    next: &mut u32,
    scope: &[SymId],
    items: &[HyperItem<SymId>],
    it: &Interner,
) -> Result<(), ResolveError> {
    for item in items {
        match item {
            HyperItem::Node(n) => index_node(idx, next, scope, n, it)?,
            HyperItem::Edge(e) => index_edge(idx, next, scope, e, it)?,
            HyperItem::Arc(_) => {}
        }
    }
    Ok(())
}

fn add_decl(
    idx: &mut Index,
    next: &mut u32,
    key: PathKey,
    it: &Interner,
) -> Result<DeclId, ResolveError> {
    if idx.by_path.contains_key(&key) {
        return Err(ResolveError::DuplicateDecl { path: fmt_path(&key, it) });
    }
    let id = DeclId(*next);
    *next += 1;
    idx.by_path.insert(key, id);
    Ok(id)
}

fn index_node(
    idx: &mut Index,
    next: &mut u32,
    scope: &[SymId],
    n: &NodeDecl<SymId>,
    it: &Interner,
) -> Result<(), ResolveError> {
    let mut fq = scope.to_vec();
    fq.push(n.inner.name);
    let _id = add_decl(idx, next, PathKey(fq), it)?;

    if let Some(body) = &n.inner.body {
        let mut child = scope.to_vec();
        child.push(n.inner.name);
        index_items(idx, next, &child, body, it)?;
    }
    Ok(())
}

fn index_edge(
    idx: &mut Index,
    next: &mut u32,
    scope: &[SymId],
    e: &EdgeDecl<SymId>,
    it: &Interner,
) -> Result<(), ResolveError> {
    let mut fq = scope.to_vec();
    fq.push(e.inner.name);
    let _id = add_decl(idx, next, PathKey(fq), it)?;

    let mut child = scope.to_vec();
    child.push(e.inner.name);
    index_items(idx, next, &child, &e.inner.body, it)?;
    Ok(())
}

/// ***EZ a resolve: RefPath + Scope -> DeclId***
/// 0 találat: error, 2+ találat: error (nincs "50 cél")
pub fn resolve_ref_to_declid(
    idx: &Index,
    scope: &[SymId],
    target: &Ref<SymId>,
    it: &Interner,
) -> Result<DeclId, ResolveError> {
    let target_key = PathKey(target.path.clone());

    let mut hit: Option<(PathKey, DeclId)> = None;

    // scope->root: [scope..k] + target
    for k in (0..=scope.len()).rev() {
        let mut fq = scope[..k].to_vec();
        fq.extend_from_slice(&target_key.0);
        let fk = PathKey(fq);

        if let Some(&did) = idx.by_path.get(&fk) {
            if hit.is_some() {
                // második találat -> ambiguous azonnal
                let first = hit.take().unwrap().0;
                return Err(ResolveError::AmbiguousRef {
                    from_scope: fmt_scope(scope, it),
                    target: fmt_path(&target_key, it),
                    candidates: vec![fmt_path(&first, it), fmt_path(&fk, it)],
                });
            }
            hit = Some((fk, did));
        }
    }

    match hit {
        Some((_k, did)) => Ok(did),
        None => Err(ResolveError::UnresolvedRef {
            from_scope: fmt_scope(scope, it),
            target: fmt_path(&target_key, it),
        }),
    }
}


/// Arc refs “lefordítása” DeclId-ra (innentől egyértelmű)
pub fn resolve_arc_refs(
    idx: &Index,
    scope: &[SymId],
    arc: &HyperArc<SymId>,
    it: &Interner,
) -> Result<Vec<SignedRefR>, ResolveError> {
    let mut out = Vec::with_capacity(arc.inner.refs.len());

    for sref in &arc.inner.refs {
        let atom = match sref {
            SignedRef::Plus(x) | SignedRef::Minus(x) | SignedRef::Neutral(x) => x,
        };
        let did = resolve_ref_to_declid(idx, scope, &atom.target, it)?;

        out.push(match sref {
            SignedRef::Plus(_) => SignedRefR::Plus(did),
            SignedRef::Minus(_) => SignedRefR::Minus(did),
            SignedRef::Neutral(_) => SignedRefR::Neutral(did),
        });
    }

    Ok(out)
}

/// Opcionális: teljes fában minden ref validálása (és arc refek leképzése)
pub fn validate_all_refs_sym(d: &AstSym, idx: &Index, it: &Interner) -> Result<(), ResolveError> {
    validate_items(&[], &d.items, idx, it)
}

fn validate_items(
    scope: &[SymId],
    items: &[HyperItem<SymId>],
    idx: &Index,
    it: &Interner,
) -> Result<(), ResolveError> {
    for item in items {
        match item {
            HyperItem::Node(n) => {
                if let Some(body) = &n.inner.body {
                    let mut child = scope.to_vec();
                    child.push(n.inner.name);
                    validate_items(&child, body, idx, it)?;
                }
            }
            HyperItem::Edge(e) => {
                let mut child = scope.to_vec();
                child.push(e.inner.name);
                validate_items(&child, &e.inner.body, idx, it)?;
            }
            HyperItem::Arc(a) => {
                // arc refs
                let _ = resolve_arc_refs(idx, scope, a, it)?;
                // value-ben lévő Ref-ek is:
                if let Some(v) = &a.anno.value {
                    validate_value(scope, v, idx, it)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_value(
    scope: &[SymId],
    v: &Value<SymId>,
    idx: &Index,
    it: &Interner,
) -> Result<(), ResolveError> {
    match v {
        Value::Ref(r) => { let _ = resolve_ref_to_declid(idx, scope, r, it)?; }
        Value::List(xs) => for x in xs { validate_value(scope, x, idx, it)?; }
        Value::Str(_) | Value::Num(_) => {}
    }
    Ok(())
}

fn fmt_scope(scope: &[SymId], it: &Interner) -> String {
    if scope.is_empty() { "<root>".into() }
    else { scope.iter().map(|&s| it.resolve(s)).collect::<Vec<_>>().join(".") }
}

fn fmt_path(p: &PathKey, it: &Interner) -> String {
    p.0.iter().map(|&s| it.resolve(s)).collect::<Vec<_>>().join(".")
}