use crate::common::ids::SymId;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathKey(pub Vec<SymId>);