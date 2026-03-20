use crate::common::ids::{DeclId, HyperArcId, NodeId};

impl DeclId {
    pub const NONE: Self = Self(usize::MAX);

    #[inline(always)]
    pub fn is_none(self) -> bool { self.0 == usize::MAX }

    #[inline(always)]
    pub fn is_some(self) -> bool { self.0 != usize::MAX }
}

impl NodeId {
    pub const NONE: Self = Self(usize::MAX);
    #[inline(always)]
    pub fn is_none(self) -> bool { self.0 == usize::MAX }
    #[inline(always)]
    pub fn is_some(self) -> bool { self.0 != usize::MAX }
}

impl HyperArcId {
    pub const NONE: Self = Self(usize::MAX);
    #[inline(always)]
    pub fn is_none(self) -> bool { self.0 == usize::MAX }
    #[inline(always)]
    pub fn is_some(self) -> bool { self.0 != usize::MAX }
}