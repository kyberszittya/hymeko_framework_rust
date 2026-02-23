use std::collections::HashMap;

use crate::common::ids::SymId;
use crate::resolution::interner::Interner;
use crate::resolution::resolve::{build_index_sym_with_prefix, validate_all_refs_sym_with_prefix, Index, ResolveError};
use crate::sym_ast::AstSym;

pub struct ModuleView<'a> {
    pub alias: SymId,
    pub ast: AstSym<'a>,
}

pub fn build_index_modules<'a>(
    root: &AstSym<'a>,
    imported: &[ModuleView<'a>],
    it: &Interner,
) -> Result<Index, ResolveError> {
    let mut idx = Index { by_path: HashMap::new() };
    let mut next: u32 = 0;

    // root a globális névtérben (prefix = [])
    build_index_sym_with_prefix(root, &[], it, &mut idx, &mut next)?;

    // importok alias alatt
    for m in imported {
        build_index_sym_with_prefix(&m.ast, &[m.alias], it, &mut idx, &mut next)?;
    }

    Ok(idx)
}

pub fn validate_all_refs_modules<'a>(
    root: &AstSym<'a>,
    imported: &[ModuleView<'a>],
    idx: &Index,
    it: &mut Interner,
) -> Result<(), ResolveError> {
    validate_all_refs_sym_with_prefix(root, &[], idx, it)?;

    for m in imported {
        validate_all_refs_sym_with_prefix(&m.ast, &[m.alias], idx, it)?;
    }

    Ok(())
}