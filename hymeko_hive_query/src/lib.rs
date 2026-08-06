//! HIVE association-query parser.
//!
//! This crate owns the small LALRPOP language that LLMs and humans can use to
//! describe HIVE queries. It lowers into typed [`hymeko_hive::AssociationQuery`]
//! values; execution remains in `hymeko_hive`.

use thiserror::Error;

use hymeko_hive::AssociationQuery;

#[allow(clippy::all, dead_code, unused_imports)]
mod grammar {
    include!(concat!(env!("OUT_DIR"), "/query.rs"));
}

/// Syntax error for the association-query subset.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct QuerySyntaxError(pub String);

/// Parse a HIVE association query.
///
/// Examples:
/// - `assoc node kind type(trait)`
/// - `assoc relation endpoint(-) node.kind(behavior)`
/// - `KIND(trait)`
/// - `HASARCREF(-, KIND(behavior))`
pub fn parse_query(input: &str) -> Result<AssociationQuery, QuerySyntaxError> {
    grammar::QueryParser::new()
        .parse(input)
        .map_err(|err| QuerySyntaxError(err.to_string()))
}

#[cfg(test)]
mod tests {
    use hymeko_hive::{AssociationKind, AssociationObject, AssociationQuery, EntitySelector, Sign};

    use super::parse_query;

    #[test]
    fn parses_association_kind_query() {
        let query = parse_query("assoc node kind type(trait)").unwrap();
        assert_eq!(query, AssociationQuery::node_kind("trait"));
    }

    #[test]
    fn parses_relation_endpoint_node_kind_query() {
        let query = parse_query("assoc relation endpoint(-) node.kind(behavior)").unwrap();
        assert_eq!(
            query,
            AssociationQuery::relation_endpoint_node_type(Sign::Minus, "behavior")
        );
    }

    #[test]
    fn parses_specific_subject_and_object() {
        let query = parse_query("assoc relation(rule_finish) endpoint(plus) node(codex)").unwrap();
        assert_eq!(
            query,
            AssociationQuery {
                subject: EntitySelector::Relation("rule_finish".into()),
                association: AssociationKind::Endpoint {
                    sign: Some(Sign::Plus)
                },
                object: AssociationObject::Node("codex".into()),
            }
        );
    }

    #[test]
    fn parses_legacy_kind_alias() {
        assert_eq!(
            parse_query("KIND(trait)").unwrap(),
            AssociationQuery::node_kind("trait")
        );
    }

    #[test]
    fn parses_legacy_hasarcref_alias() {
        assert_eq!(
            parse_query("HASARCREF(-, KIND(behavior))").unwrap(),
            AssociationQuery::relation_endpoint_node_type(Sign::Minus, "behavior")
        );
    }

    #[test]
    fn rejects_unknown_syntax() {
        assert!(parse_query("select * from hive").is_err());
    }

    #[test]
    fn parses_readme_robot_and_fano_examples() {
        for query in [
            "assoc node kind type(link)",
            "assoc relation kind type(rev_joint)",
            "assoc node inherits type(meta_controller)",
            "assoc relation endpoint(-) node.kind(link)",
            "assoc relation endpoint(-) node(tool)",
            "assoc relation endpoint(-) node(AXIS_Z)",
            "assoc relation kind type(control_plugin)",
            "assoc relation kind type(sim_plugin)",
            "assoc node kind type(point)",
            "assoc relation kind type(line)",
            "assoc relation endpoint(~) node(n0)",
            "assoc relation endpoint(~) node.kind(point)",
        ] {
            parse_query(query).unwrap_or_else(|err| panic!("failed to parse `{query}`: {err}"));
        }
    }
}
