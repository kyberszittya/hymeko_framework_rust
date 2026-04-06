use hymeko::common::ids::SymId;
use hymeko::resolution::interner::Interner;

pub trait NameResolver {
    fn resolve(&self, id: SymId) -> &str;
}

impl NameResolver for Interner {
    #[inline]
    fn resolve(&self, id: SymId) -> &str {
        Interner::resolve(self, id)
    }
}