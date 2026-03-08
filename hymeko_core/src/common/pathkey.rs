use std::borrow::Borrow;
use serde::{Deserialize, Serialize};
use crate::common::ids::SymId;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd,
    Serialize, Deserialize)]
pub struct PathKey(pub Vec<SymId>);


impl PathKey {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = SymId> + '_ {
        self.0.iter().copied()
    }
}


impl Borrow<[SymId]> for PathKey {
    fn borrow(&self) -> &[SymId] { &self.0 }
}