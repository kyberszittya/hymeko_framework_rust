use std::collections::HashMap;
use crate::common::SymId;

#[derive(Default)]
pub struct Interner {
    map: HashMap<String, SymId>,
    vec: Vec<String>,
}

impl Interner{
    pub fn new() -> Self { Self::default() }
    pub fn intern(&mut self, s: &str) -> SymId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = SymId(self.vec.len() as u32);
        self.vec.push(s.to_owned());
        self.map.insert(self.vec[id.0 as usize].clone(), id);
        id
    }

    pub fn resolve(&self, id: SymId) -> &str {
        &self.vec[id.0 as usize]
    }
}