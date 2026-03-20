use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
    Serialize, Deserialize)]
pub struct DeclId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash,
    Serialize, Deserialize)]
pub struct NodeId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash,
    Serialize, Deserialize)]
pub struct EdgeId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash,
    Serialize, Deserialize)]
pub struct HyperArcId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord,
    Serialize, Deserialize)]
pub struct SymId(pub usize);

