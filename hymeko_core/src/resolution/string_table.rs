use std::sync::Arc;
use crate::common::ids::SymId;
use crate::resolution::interner::Interner;

// ================================================
// StringTable: lightweight wrapper around the Interner for Python access (frozen interner snapshot)
// ================================================


#[derive(Clone, Debug)]
pub struct StringTable(Arc<Vec<String>>);

impl StringTable {
    pub fn from_interner(interner: &Interner) -> Self {
        let strings = interner.iter().map(|(_, s)| s.to_string()).collect();
        Self(Arc::new(strings))
    }

    pub fn from_vec(strings: Vec<String>) -> Self {
        Self(Arc::new(strings))
    }

    #[inline]
    pub fn resolve(&self, id: SymId) -> &str {
        &self.0[id.0]
    }
}
