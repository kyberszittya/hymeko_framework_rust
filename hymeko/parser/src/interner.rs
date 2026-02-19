use std::collections::HashMap;
use crate::common::ids::SymId;

#[derive(Default)]
pub struct Interner {
    map: HashMap<String, SymId>,
    vec: Vec<String>,
}

impl Interner {
    pub fn new() -> Self { Self::default() }

    pub fn intern(&mut self, s: &str) -> SymId {
        if let Some(&id) = self.map.get(s) { return id; }
        let id = SymId(self.vec.len() as u32);
        self.vec.push(s.to_owned());
        self.map.insert(s.to_owned(), id);
        id
    }

    pub fn resolve(&self, id: SymId) -> &str {
        &self.vec[id.0 as usize]
    }

    /// Iterator over all interned strings and their SymIds.
    pub fn iter(&self) -> impl Iterator<Item = (SymId, &str)> {
        self.vec.iter().enumerate().map(|(i, s)| (SymId(i as u32), s.as_str()))
    }
}

impl Interner {
    pub fn get_id(&self, s: &str) -> Option<SymId> {
        self.map.get(s).copied()
    }
}