// src/resolve.rs
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use parser::ast::{Anno, EdgeDecl, HyperArc, HyperItem, NodeDecl, Ref, SignedRef, Value};
use crate::common::ids::{DeclId, SymId};
use crate::common::pathkey::PathKey;
use crate::ir::ir::{AnnoR, RefAtomR, SignedRefR, ValueR};
use crate::resolution::interner::Interner;
use crate::sym_ast::AstSym;

#[derive(Debug, 
    Serialize, Deserialize)]
pub struct Index {
    pub by_path: HashMap<PathKey, DeclId>,
}

impl Index {
    pub fn iter(&self) -> impl Iterator<Item = (&PathKey, &DeclId)> {
        self.by_path.iter()
    }
}

#[derive(Debug)]
pub enum ResolveError {
    DuplicateDecl { path: String },
    UnresolvedRef { from_scope: String, target: String },
    AmbiguousRef { from_scope: String, target: String, candidates: Vec<String> },
    UnexpectedTopLevelArc {detail: String}
}

pub fn build_index_sym<'a>(d: &AstSym<'a>, it: &Interner) -> Result<Index, ResolveError> {
    let mut idx = Index { by_path: HashMap::new() };
    let mut next: usize = 0;

    for n in &d.header {
        index_node(&mut idx, &mut next, &[], n, it)?;
    }

    index_items(&mut idx, &mut next, &[], &d.items, it)?;

    Ok(idx)
}

pub fn build_index_sym_with_prefix<'a>(
    d: &AstSym<'a>,
    prefix: &[SymId],
    it: &Interner,
    idx: &mut Index,
    next: &mut usize,
) -> Result<(), ResolveError> {
    // header node decl-ek
    for n in &d.header {
        index_node(idx, next, prefix, n, it)?;
    }

    // items
    index_items(idx, next, prefix, &d.items, it)?;
    Ok(())
}

fn index_items<'a>(
    idx: &mut Index,
    next: &mut usize,
    scope: &[SymId],
    items: &[HyperItem<'a, SymId>],
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

fn resolve_value<'a>(
    idx: &Index,
    scope: &[SymId],
    v: &Value<'a, SymId>,
    it: &mut Interner,
) -> Result<ValueR, ResolveError> {
    Ok(match v {
        Value::Str(s) => ValueR::Str(it.intern(s)), // Bridging to owned IR
        Value::Num(x) => ValueR::Num(*x),
        Value::List(xs) => ValueR::List(xs.iter().map(|x| resolve_value(idx, scope, x, it)).collect::<Result<_,_>>()?),
        Value::Ref(r) => {
            let did = resolve_ref_to_declid(idx, scope, r, it)?;
            ValueR::Ref(did)
        }
    })
}

pub fn resolve_anno<'a>(
    idx: &Index,
    scope: &[SymId],
    a: &Anno<'a, SymId>,
    it: &mut Interner,
) -> Result<AnnoR, ResolveError> {
    Ok(AnnoR {
        tags: a.tags.clone(), // They are already SymIds! No interning needed.
        value: match &a.value {
            Some(v) => Some(resolve_value(idx, scope, v, it)?),
            None => None,
        },
    })
}

fn add_decl(
    idx: &mut Index,
    next: &mut usize,
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

fn index_node<'a>(
    idx: &mut Index,
    next: &mut usize,
    scope: &[SymId],
    n: &NodeDecl<'a, SymId>,
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

pub fn resolve_arc_anno<'a>(
    idx: &Index,
    scope: &[SymId],
    arc: &HyperArc<'a, SymId>,
    it: &mut Interner,
) -> Result<AnnoR, ResolveError> {
    resolve_anno(idx, scope, &arc.anno, it)
}

fn index_edge<'a>(
    idx: &mut Index,
    next: &mut usize,
    scope: &[SymId],
    e: &EdgeDecl<'a, SymId>,
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

fn fmt_path_from_slice(p: &[SymId], it: &Interner) -> String {
    p.iter()
        .map(|&s| it.resolve(s))
        .collect::<Vec<_>>()
        .join(".")
}

pub fn resolve_ref_to_declid(
    idx: &Index,
    scope: &[SymId],
    target: &Ref<SymId>,
    it: &Interner,
) -> Result<DeclId, ResolveError> {
    let mut hit: Option<(PathKey, DeclId)> = None;

    // 1. One single scratch buffer for the entire search [cite: 2026-02-08]
    let mut fq_buffer = Vec::with_capacity(scope.len() + target.path.len());

    for k in (0..=scope.len()).rev() {
        fq_buffer.clear();
        fq_buffer.extend_from_slice(&scope[..k]);
        fq_buffer.extend_from_slice(&target.path);

        // 2. We only 'own' it if we find a hit and need to store it [cite: 2026-02-08]
        // Note: Unless you use a custom lookup wrapper, .get() usually requires
        // the same type as the key (PathKey). We'll assume PathKey(Vec<SymId>) here.
        let fk = PathKey(fq_buffer.clone());

        if let Some(&did) = idx.by_path.get(&fk) {
            if let Some((ref first_key, _)) = hit {
                return Err(ResolveError::AmbiguousRef {
                    from_scope: fmt_scope(scope, it),
                    target: fmt_path_from_slice(&target.path, it), // Use a slice formatter [cite: 2026-02-08]
                    candidates: vec![fmt_path(first_key, it), fmt_path(&fk, it)],
                });
            }
            hit = Some((fk, did));
        }
    }

    match hit {
        Some((_, did)) => Ok(did),
        None => Err(ResolveError::UnresolvedRef {
            from_scope: fmt_scope(scope, it),
            target: fmt_path_from_slice(&target.path, it), // No clone needed [cite: 2026-02-08]
        }),
    }
}

pub fn resolve_node_bases<'a>(
    idx: &Index,
    scope: &[SymId],
    bases: &[SignedRef<'a, SymId>],
    it: &mut Interner,
) -> Result<Vec<SignedRefR>, ResolveError> {
    let mut out = Vec::with_capacity(bases.len());

    for sref in bases {
        let atom = match sref {
            SignedRef::Plus(x) | SignedRef::Minus(x) | SignedRef::Neutral(x) => x,
        };

        let did = resolve_ref_to_declid(idx, scope, &atom.target, it)?;

        // Tag-ek + value ugyanúgy, mint arcoknál
        let anno_r = resolve_anno(idx, scope, &atom.anno, it)?;

        let weights_r = match &atom.anno.value {
            Some(Value::List(ws)) => {
                let resolved = ws
                    .iter()
                    .map(|w| resolve_value(idx, scope, w, it))
                    .collect::<Result<Vec<ValueR>, _>>()?;
                Some(resolved)
            }
            Some(v) => {
                let resolved = resolve_value(idx, scope, v, it)?;
                Some(vec![resolved])
            }
            None => None,
        };

        let atom_r = RefAtomR {
            target: did,
            weights: weights_r,
            anno: anno_r,
        };

        let sref_r = match sref {
            SignedRef::Plus(_) => SignedRefR::Plus(atom_r),
            SignedRef::Minus(_) => SignedRefR::Minus(atom_r),
            SignedRef::Neutral(_) => SignedRefR::Neutral(atom_r),
        };

        out.push(sref_r);
    }

    Ok(out)
}

pub fn resolve_arc_refs<'a>(
    idx: &Index,
    scope: &[SymId],
    arc: &HyperArc<'a, SymId>,
    it: &mut Interner,
) -> Result<Vec<SignedRefR>, ResolveError> {
    let mut out = Vec::with_capacity(arc.inner.refs.len());

    for sref in &arc.inner.refs {
        let atom = match sref {
            SignedRef::Plus(x) | SignedRef::Minus(x) | SignedRef::Neutral(x) => x,
        };
        let did = resolve_ref_to_declid(idx, scope, &atom.target, it)?;

        // Delegate directly to the unified annotation resolver
        let anno_r = resolve_anno(idx, scope, &atom.anno, it)?;

        // Map the unified value back to IR weights if necessary
        let weights_r = match &atom.anno.value {
            Some(Value::List(ws)) => {
                let resolved = ws
                    .iter()
                    .map(|w| resolve_value(idx, scope, w, it))
                    .collect::<Result<Vec<ValueR>, _>>()?;
                Some(resolved)
            }
            Some(v) => {
                let resolved = resolve_value(idx, scope, v, it)?;
                Some(vec![resolved])
            }
            None => None,
        };

        let atom_r = RefAtomR {
            target: did,
            weights: weights_r,
            anno: anno_r
        };

        let sref_r = match sref {
            SignedRef::Plus(_) => SignedRefR::Plus(atom_r),
            SignedRef::Minus(_) => SignedRefR::Minus(atom_r),
            SignedRef::Neutral(_) => SignedRefR::Neutral(atom_r),
        };

        out.push(sref_r);
    }

    Ok(out)
}

pub fn validate_all_refs_sym<'a>(d: &AstSym<'a>, idx: &Index, it: &mut Interner) -> Result<(), ResolveError> {
    validate_items(&[], &d.items, idx, it)
}

pub fn validate_all_refs_sym_with_prefix<'a>(
    d: &AstSym<'a>,
    prefix: &[SymId],
    idx: &Index,
    it: &mut Interner,
) -> Result<(), ResolveError> {
    validate_items(prefix, &d.items, idx, it)
}

fn validate_items<'a>(
    scope: &[SymId],
    items: &[HyperItem<'a, SymId>],
    idx: &Index,
    it: &mut Interner,
) -> Result<(), ResolveError> {
    for item in items {
        match item {
            HyperItem::Node(n) => {
                let _ = resolve_node_bases(idx, scope, &n.inner.bases, it)?;
                if let Some(body) = &n.inner.body {
                    let mut child = scope.to_vec();
                    child.push(n.inner.name);
                    validate_items(&child, body, idx, it)?;
                }
            }
            HyperItem::Edge(e) => {
                let _ = resolve_node_bases(idx, scope, &e.inner.bases, it)?;
                let mut child = scope.to_vec();
                child.push(e.inner.name);
                validate_items(&child, &e.inner.body, idx, it)?;
            }
            HyperItem::Arc(a) => {
                let _ = resolve_arc_refs(idx, scope, a, it)?;
                if let Some(v) = &a.anno.value {
                    validate_value(scope, v, idx, it)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_value<'a>(
    scope: &[SymId],
    v: &Value<'a, SymId>,
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