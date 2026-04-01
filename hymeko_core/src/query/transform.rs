//! Domain transform trait — pluggable output format interface.
//!
//! A domain transform declares query patterns and generates
//! output from results. This makes the query engine domain-agnostic.

use crate::ir::ir::Ir;
use crate::query::engine::{NameResolver, QueryResult};
use crate::query::predicate::NamedQuery;

/// A domain transform declares what queries it needs and produces output.
pub trait DomainTransform {
    type Config;
    type Output;

    /// Human-readable name of this transform.
    fn name(&self) -> &str;

    /// The query patterns this transform needs.
    fn queries(&self) -> Vec<NamedQuery>;

    /// Generate output from query results and IR.
    fn generate<R: NameResolver>(
        &self,
        config: &Self::Config,
        results: &[(String, QueryResult)],
        ir: &Ir,
        resolver: &R,
    ) -> Self::Output;
}
