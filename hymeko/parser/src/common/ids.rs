#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeclId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HyperArcId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(pub u32);

impl DeclId {
    pub const NONE: Self = Self(u32::MAX);

    #[inline(always)]
    pub fn is_none(self) -> bool { self.0 == u32::MAX }

    #[inline(always)]
    pub fn is_some(self) -> bool { self.0 != u32::MAX }
}