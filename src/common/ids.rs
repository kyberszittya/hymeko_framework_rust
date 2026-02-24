#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeclId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HyperArcId(pub usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(pub u32);

impl DeclId {
    pub const NONE: Self = Self(usize::MAX);

    #[inline(always)]
    pub fn is_none(self) -> bool { self.0 == usize::MAX }

    #[inline(always)]
    pub fn is_some(self) -> bool { self.0 != usize::MAX }
}