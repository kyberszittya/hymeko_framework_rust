use crate::common::ids::SymId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathKey(pub Vec<SymId>);

impl PathKey {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = SymId> + '_ {
        self.0.iter().copied()
    }
}
