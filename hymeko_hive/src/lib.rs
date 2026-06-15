//! HIVE: a database-like canonical store for signed typed hypergraphs.
//!
//! This crate is intentionally below AKOIRE/HOTARU. Planners may propose
//! [`HiveTransaction`] values, but only [`HiveStore::commit`] mutates canonical
//! state. The first implementation is in-memory and small on purpose: it pins
//! transaction semantics before wiring parser lowering, transport, or storage.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod generators;

/// Content hash of a canonical HIVE state.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HiveHash(String);

impl HiveHash {
    /// Borrow the lowercase hex encoding.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HiveHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Stable node identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl From<&str> for NodeId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for NodeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Stable relation/hyperedge identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationId(pub String);

impl From<&str> for RelationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for RelationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Transaction identifier, supplied by AKOIRE/HOTARU or a distributed actor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub String);

impl From<&str> for TransactionId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for TransactionId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Signed incidence endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Sign {
    Plus,
    Minus,
    Neutral,
}

impl Sign {
    fn parse_minimal(input: &str) -> Result<Self, QueryParseError> {
        match input {
            "+" | "plus" => Ok(Self::Plus),
            "-" | "minus" => Ok(Self::Minus),
            "~" | "neutral" => Ok(Self::Neutral),
            other => Err(QueryParseError::UnknownSign(other.to_string())),
        }
    }
}

/// Literal attribute value attached to a node or relation.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum AttributeValue {
    Bool(bool),
    Float(f64),
    Int(i64),
    Text(String),
}

/// Node record in canonical HIVE state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiveNode {
    pub id: NodeId,
    pub ty: String,
    pub attributes: BTreeMap<String, AttributeValue>,
}

impl HiveNode {
    #[must_use]
    pub fn new(id: impl Into<NodeId>, ty: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ty: ty.into(),
            attributes: BTreeMap::new(),
        }
    }
}

/// Relation/hyperedge record in canonical HIVE state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiveRelation {
    pub id: RelationId,
    pub ty: String,
    pub endpoints: Vec<(Sign, NodeId)>,
    pub attributes: BTreeMap<String, AttributeValue>,
}

impl HiveRelation {
    #[must_use]
    pub fn new(
        id: impl Into<RelationId>,
        ty: impl Into<String>,
        endpoints: Vec<(Sign, NodeId)>,
    ) -> Self {
        Self {
            id: id.into(),
            ty: ty.into(),
            endpoints,
            attributes: BTreeMap::new(),
        }
    }
}

/// One proposed mutation over HIVE state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HiveDelta {
    AddNode(HiveNode),
    RemoveNode {
        id: NodeId,
    },
    AddRelation(HiveRelation),
    RemoveRelation {
        id: RelationId,
    },
    AttachNodeAttribute {
        id: NodeId,
        key: String,
        value: AttributeValue,
    },
    AttachRelationAttribute {
        id: RelationId,
        key: String,
        value: AttributeValue,
    },
}

/// HIVE's first query subset.
///
/// This mirrors the useful core of the existing HyMeKo predicate language:
/// `KIND(type)` becomes [`HiveQuery::NodesByType`] or
/// [`HiveQuery::RelationsByType`], and `HASARCREF(sign, KIND(type))` becomes
/// [`HiveQuery::RelationsByEndpointNodeType`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HiveQuery {
    NodesByType(String),
    RelationsByType(String),
    NodesWithAttribute(String),
    RelationsWithAttribute(String),
    RelationsByEndpointNode { sign: Sign, node: NodeId },
    RelationsByEndpointNodeType { sign: Sign, node_type: String },
}

impl HiveQuery {
    /// Parse the minimal LLM-to-HIVE query description format.
    ///
    /// Supported forms:
    /// - `node.type:<type>`
    /// - `relation.type:<type>`
    /// - `node.attr:<key>`
    /// - `relation.attr:<key>`
    /// - `relation.endpoint:<sign>:<node_id>`
    /// - `relation.endpoint_type:<sign>:<node_type>`
    ///
    /// `sign` accepts `+`, `-`, `~`, `plus`, `minus`, or `neutral`.
    pub fn parse_minimal(input: &str) -> Result<Self, QueryParseError> {
        let mut parts = input.trim().split(':');
        let head = parts
            .next()
            .filter(|part| !part.trim().is_empty())
            .ok_or(QueryParseError::Empty)?;

        match head.trim() {
            "node.type" => {
                let value = parse_one_value(parts, input)?;
                Ok(Self::NodesByType(value))
            }
            "relation.type" => {
                let value = parse_one_value(parts, input)?;
                Ok(Self::RelationsByType(value))
            }
            "node.attr" => {
                let value = parse_one_value(parts, input)?;
                Ok(Self::NodesWithAttribute(value))
            }
            "relation.attr" => {
                let value = parse_one_value(parts, input)?;
                Ok(Self::RelationsWithAttribute(value))
            }
            "relation.endpoint" => {
                let (sign, value) = parse_sign_and_value(parts, input)?;
                Ok(Self::RelationsByEndpointNode {
                    sign,
                    node: value.into(),
                })
            }
            "relation.endpoint_type" => {
                let (sign, value) = parse_sign_and_value(parts, input)?;
                Ok(Self::RelationsByEndpointNodeType {
                    sign,
                    node_type: value,
                })
            }
            other => Err(QueryParseError::UnknownHead(other.to_string())),
        }
    }
}

fn parse_one_value<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    original: &str,
) -> Result<String, QueryParseError> {
    let value = parts
        .next()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .ok_or_else(|| QueryParseError::Malformed(original.to_string()))?;
    if parts.next().is_some() {
        return Err(QueryParseError::Malformed(original.to_string()));
    }
    Ok(value.to_string())
}

fn parse_sign_and_value<'a>(
    mut parts: impl Iterator<Item = &'a str>,
    original: &str,
) -> Result<(Sign, String), QueryParseError> {
    let sign = parts
        .next()
        .map(str::trim)
        .ok_or_else(|| QueryParseError::Malformed(original.to_string()))
        .and_then(Sign::parse_minimal)?;
    let value = parts
        .next()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .ok_or_else(|| QueryParseError::Malformed(original.to_string()))?;
    if parts.next().is_some() {
        return Err(QueryParseError::Malformed(original.to_string()));
    }
    Ok((sign, value.to_string()))
}

/// Minimal query parser error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum QueryParseError {
    #[error("empty query")]
    Empty,
    #[error("unknown query head: {0}")]
    UnknownHead(String),
    #[error("unknown sign: {0}")]
    UnknownSign(String),
    #[error("malformed query: {0}")]
    Malformed(String),
}

/// Owned query result so callers can use it across transaction boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct QueryResult {
    pub nodes: Vec<NodeId>,
    pub relations: Vec<RelationId>,
}

/// Entity selector for the generalized association query model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntitySelector {
    AnyNode,
    AnyRelation,
    Node(NodeId),
    Relation(RelationId),
}

/// Object selector for the generalized association query model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssociationObject {
    Any,
    Type(String),
    Attribute(String),
    Node(NodeId),
    NodeType(String),
}

/// Association label. Legacy predicates are projections of these:
/// `KIND(x)` = `Kind -> Type(x)`, and `HASARCREF(s, p)` =
/// `Endpoint(s) -> p`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssociationKind {
    Kind,
    Inherits,
    Attribute,
    Endpoint { sign: Option<Sign> },
}

/// General association pattern: subject --association--> object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssociationQuery {
    pub subject: EntitySelector,
    pub association: AssociationKind,
    pub object: AssociationObject,
}

impl AssociationQuery {
    #[must_use]
    pub fn node_kind(ty: impl Into<String>) -> Self {
        Self {
            subject: EntitySelector::AnyNode,
            association: AssociationKind::Kind,
            object: AssociationObject::Type(ty.into()),
        }
    }

    #[must_use]
    pub fn relation_kind(ty: impl Into<String>) -> Self {
        Self {
            subject: EntitySelector::AnyRelation,
            association: AssociationKind::Kind,
            object: AssociationObject::Type(ty.into()),
        }
    }

    #[must_use]
    pub fn relation_endpoint_node_type(sign: Sign, node_type: impl Into<String>) -> Self {
        Self {
            subject: EntitySelector::AnyRelation,
            association: AssociationKind::Endpoint { sign: Some(sign) },
            object: AssociationObject::NodeType(node_type.into()),
        }
    }
}

/// One realized association match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssociationMatch {
    pub subject: EntitySelector,
    pub association: AssociationKind,
    pub object: AssociationObject,
}

/// Atomic mutation request. All deltas apply, or none do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiveTransaction {
    pub id: TransactionId,
    pub parent_hash: HiveHash,
    pub actor: String,
    pub seq: u64,
    pub deltas: Vec<HiveDelta>,
}

impl HiveTransaction {
    #[must_use]
    pub fn new(
        id: impl Into<TransactionId>,
        parent_hash: HiveHash,
        actor: impl Into<String>,
        seq: u64,
        deltas: Vec<HiveDelta>,
    ) -> Self {
        Self {
            id: id.into(),
            parent_hash,
            actor: actor.into(),
            seq,
            deltas,
        }
    }
}

/// Successful commit or replay receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitReceipt {
    pub tx_id: TransactionId,
    pub old_hash: HiveHash,
    pub new_hash: HiveHash,
    pub generation: u64,
    pub status: CommitStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitStatus {
    Applied,
    Replayed,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum CommitError {
    #[error("parent hash mismatch: expected {expected}, got {actual}")]
    ParentHashMismatch {
        expected: HiveHash,
        actual: HiveHash,
    },
    #[error("sequence mismatch for actor {actor}: expected {expected}, got {actual}")]
    SequenceMismatch {
        actor: String,
        expected: u64,
        actual: u64,
    },
    #[error("delta {index} failed: {reason}")]
    DeltaFailed { index: usize, reason: DeltaError },
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DeltaError {
    #[error("node already exists: {0:?}")]
    NodeExists(NodeId),
    #[error("node not found: {0:?}")]
    NodeMissing(NodeId),
    #[error("node has incident relations: {0:?}")]
    NodeHasIncidentRelations(NodeId),
    #[error("relation already exists: {0:?}")]
    RelationExists(RelationId),
    #[error("relation not found: {0:?}")]
    RelationMissing(RelationId),
    #[error("relation must have at least one endpoint")]
    EmptyRelation,
}

/// Immutable snapshot of canonical HIVE state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HiveState {
    nodes: BTreeMap<NodeId, HiveNode>,
    relations: BTreeMap<RelationId, HiveRelation>,
}

impl HiveState {
    #[must_use]
    pub fn nodes(&self) -> &BTreeMap<NodeId, HiveNode> {
        &self.nodes
    }

    #[must_use]
    pub fn relations(&self) -> &BTreeMap<RelationId, HiveRelation> {
        &self.relations
    }

    #[must_use]
    pub fn hash(&self) -> HiveHash {
        let mut canonical = String::new();
        for (id, node) in &self.nodes {
            canonical.push_str("node\0");
            canonical.push_str(&id.0);
            canonical.push('\0');
            canonical.push_str(&node.ty);
            canonical.push('\0');
            write_attributes(&mut canonical, &node.attributes);
        }
        for (id, relation) in &self.relations {
            canonical.push_str("relation\0");
            canonical.push_str(&id.0);
            canonical.push('\0');
            canonical.push_str(&relation.ty);
            canonical.push('\0');
            for (sign, node_id) in &relation.endpoints {
                canonical.push_str(match sign {
                    Sign::Plus => "+",
                    Sign::Minus => "-",
                    Sign::Neutral => "~",
                });
                canonical.push_str(&node_id.0);
                canonical.push('\0');
            }
            write_attributes(&mut canonical, &relation.attributes);
        }
        HiveHash(blake3::hash(canonical.as_bytes()).to_hex().to_string())
    }

    fn apply_delta(&mut self, delta: &HiveDelta) -> Result<(), DeltaError> {
        match delta {
            HiveDelta::AddNode(node) => {
                if self.nodes.contains_key(&node.id) {
                    return Err(DeltaError::NodeExists(node.id.clone()));
                }
                self.nodes.insert(node.id.clone(), node.clone());
                Ok(())
            }
            HiveDelta::RemoveNode { id } => {
                if !self.nodes.contains_key(id) {
                    return Err(DeltaError::NodeMissing(id.clone()));
                }
                if self
                    .relations
                    .values()
                    .any(|rel| rel.endpoints.iter().any(|(_, node_id)| node_id == id))
                {
                    return Err(DeltaError::NodeHasIncidentRelations(id.clone()));
                }
                self.nodes.remove(id);
                Ok(())
            }
            HiveDelta::AddRelation(relation) => {
                if self.relations.contains_key(&relation.id) {
                    return Err(DeltaError::RelationExists(relation.id.clone()));
                }
                if relation.endpoints.is_empty() {
                    return Err(DeltaError::EmptyRelation);
                }
                for (_, node_id) in &relation.endpoints {
                    if !self.nodes.contains_key(node_id) {
                        return Err(DeltaError::NodeMissing(node_id.clone()));
                    }
                }
                self.relations.insert(relation.id.clone(), relation.clone());
                Ok(())
            }
            HiveDelta::RemoveRelation { id } => {
                if self.relations.remove(id).is_none() {
                    return Err(DeltaError::RelationMissing(id.clone()));
                }
                Ok(())
            }
            HiveDelta::AttachNodeAttribute { id, key, value } => {
                let node = self
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| DeltaError::NodeMissing(id.clone()))?;
                node.attributes.insert(key.clone(), value.clone());
                Ok(())
            }
            HiveDelta::AttachRelationAttribute { id, key, value } => {
                let relation = self
                    .relations
                    .get_mut(id)
                    .ok_or_else(|| DeltaError::RelationMissing(id.clone()))?;
                relation.attributes.insert(key.clone(), value.clone());
                Ok(())
            }
        }
    }
}

fn write_attributes(out: &mut String, attributes: &BTreeMap<String, AttributeValue>) {
    for (key, value) in attributes {
        out.push_str("attr\0");
        out.push_str(key);
        out.push('\0');
        match value {
            AttributeValue::Bool(v) => out.push_str(if *v { "b:true" } else { "b:false" }),
            AttributeValue::Float(v) => out.push_str(&format!("f:{:016x}", v.to_bits())),
            AttributeValue::Int(v) => out.push_str(&format!("i:{v}")),
            AttributeValue::Text(v) => {
                out.push_str("t:");
                out.push_str(v);
            }
        }
        out.push('\0');
    }
}

/// In-memory HIVE store with atomic transaction commits.
#[derive(Debug, Clone)]
pub struct HiveStore {
    state: HiveState,
    index: HiveIndex,
    generation: u64,
    applied: BTreeMap<TransactionId, CommitReceipt>,
    next_seq_by_actor: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Default)]
struct HiveIndex {
    nodes_by_type: BTreeMap<String, BTreeSet<NodeId>>,
    relations_by_type: BTreeMap<String, BTreeSet<RelationId>>,
    nodes_by_attribute: BTreeMap<String, BTreeSet<NodeId>>,
    relations_by_attribute: BTreeMap<String, BTreeSet<RelationId>>,
    relations_by_endpoint: BTreeMap<(Sign, NodeId), BTreeSet<RelationId>>,
}

impl HiveIndex {
    fn build(state: &HiveState) -> Self {
        let mut index = Self::default();
        for (id, node) in &state.nodes {
            index
                .nodes_by_type
                .entry(node.ty.clone())
                .or_default()
                .insert(id.clone());
            for key in node.attributes.keys() {
                index
                    .nodes_by_attribute
                    .entry(key.clone())
                    .or_default()
                    .insert(id.clone());
            }
        }
        for (id, relation) in &state.relations {
            index
                .relations_by_type
                .entry(relation.ty.clone())
                .or_default()
                .insert(id.clone());
            for key in relation.attributes.keys() {
                index
                    .relations_by_attribute
                    .entry(key.clone())
                    .or_default()
                    .insert(id.clone());
            }
            for (sign, node_id) in &relation.endpoints {
                index
                    .relations_by_endpoint
                    .entry((*sign, node_id.clone()))
                    .or_default()
                    .insert(id.clone());
            }
        }
        index
    }
}

impl Default for HiveStore {
    fn default() -> Self {
        Self::new()
    }
}

impl HiveStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: HiveState::default(),
            index: HiveIndex::default(),
            generation: 0,
            applied: BTreeMap::new(),
            next_seq_by_actor: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn state(&self) -> &HiveState {
        &self.state
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn state_hash(&self) -> HiveHash {
        self.state.hash()
    }

    /// Commit a transaction atomically. Replaying an already-applied
    /// transaction id returns the original receipt and does not mutate state.
    pub fn commit(&mut self, tx: HiveTransaction) -> Result<CommitReceipt, CommitError> {
        if let Some(receipt) = self.applied.get(&tx.id) {
            let mut replayed = receipt.clone();
            replayed.status = CommitStatus::Replayed;
            return Ok(replayed);
        }

        let old_hash = self.state_hash();
        if tx.parent_hash != old_hash {
            return Err(CommitError::ParentHashMismatch {
                expected: old_hash,
                actual: tx.parent_hash,
            });
        }

        let expected_seq = *self.next_seq_by_actor.get(&tx.actor).unwrap_or(&0);
        if tx.seq != expected_seq {
            return Err(CommitError::SequenceMismatch {
                actor: tx.actor,
                expected: expected_seq,
                actual: tx.seq,
            });
        }

        let mut staged = self.state.clone();
        for (index, delta) in tx.deltas.iter().enumerate() {
            staged
                .apply_delta(delta)
                .map_err(|reason| CommitError::DeltaFailed { index, reason })?;
        }

        self.state = staged;
        self.index = HiveIndex::build(&self.state);
        self.generation += 1;
        self.next_seq_by_actor.insert(tx.actor, tx.seq + 1);

        let receipt = CommitReceipt {
            tx_id: tx.id.clone(),
            old_hash,
            new_hash: self.state_hash(),
            generation: self.generation,
            status: CommitStatus::Applied,
        };
        self.applied.insert(tx.id, receipt.clone());
        Ok(receipt)
    }

    /// Applied transaction ids, useful for distributed log reconciliation.
    #[must_use]
    pub fn applied_transactions(&self) -> BTreeSet<&TransactionId> {
        self.applied.keys().collect()
    }

    /// Run a typed HIVE query over the current committed state.
    #[must_use]
    pub fn query(&self, query: &HiveQuery) -> QueryResult {
        match query {
            HiveQuery::NodesByType(ty) => QueryResult {
                nodes: ids_from_set(self.index.nodes_by_type.get(ty)),
                relations: Vec::new(),
            },
            HiveQuery::RelationsByType(ty) => QueryResult {
                nodes: Vec::new(),
                relations: ids_from_set(self.index.relations_by_type.get(ty)),
            },
            HiveQuery::NodesWithAttribute(key) => QueryResult {
                nodes: ids_from_set(self.index.nodes_by_attribute.get(key)),
                relations: Vec::new(),
            },
            HiveQuery::RelationsWithAttribute(key) => QueryResult {
                nodes: Vec::new(),
                relations: ids_from_set(self.index.relations_by_attribute.get(key)),
            },
            HiveQuery::RelationsByEndpointNode { sign, node } => QueryResult {
                nodes: Vec::new(),
                relations: ids_from_set(
                    self.index.relations_by_endpoint.get(&(*sign, node.clone())),
                ),
            },
            HiveQuery::RelationsByEndpointNodeType { sign, node_type } => {
                let mut relations = BTreeSet::new();
                if let Some(nodes) = self.index.nodes_by_type.get(node_type) {
                    for node in nodes {
                        if let Some(found) =
                            self.index.relations_by_endpoint.get(&(*sign, node.clone()))
                        {
                            relations.extend(found.iter().cloned());
                        }
                    }
                }
                QueryResult {
                    nodes: Vec::new(),
                    relations: relations.into_iter().collect(),
                }
            }
        }
    }

    /// Run a generalized association query and return realized association
    /// triples. This is the common substrate under `KIND`, attribute lookup,
    /// and `HASARCREF`-style endpoint queries.
    #[must_use]
    pub fn query_associations(&self, query: &AssociationQuery) -> Vec<AssociationMatch> {
        match &query.association {
            AssociationKind::Kind => self.query_kind_associations(query),
            AssociationKind::Inherits => Vec::new(),
            AssociationKind::Attribute => self.query_attribute_associations(query),
            AssociationKind::Endpoint { sign } => self.query_endpoint_associations(query, *sign),
        }
    }

    fn query_kind_associations(&self, query: &AssociationQuery) -> Vec<AssociationMatch> {
        match (&query.subject, &query.object) {
            (EntitySelector::AnyNode, AssociationObject::Type(ty)) => self
                .query(&HiveQuery::NodesByType(ty.clone()))
                .nodes
                .into_iter()
                .map(|id| AssociationMatch {
                    subject: EntitySelector::Node(id),
                    association: AssociationKind::Kind,
                    object: AssociationObject::Type(ty.clone()),
                })
                .collect(),
            (EntitySelector::AnyRelation, AssociationObject::Type(ty)) => self
                .query(&HiveQuery::RelationsByType(ty.clone()))
                .relations
                .into_iter()
                .map(|id| AssociationMatch {
                    subject: EntitySelector::Relation(id),
                    association: AssociationKind::Kind,
                    object: AssociationObject::Type(ty.clone()),
                })
                .collect(),
            (EntitySelector::Node(id), AssociationObject::Type(ty)) => self
                .state
                .nodes
                .get(id)
                .filter(|node| &node.ty == ty)
                .map(|_| {
                    vec![AssociationMatch {
                        subject: EntitySelector::Node(id.clone()),
                        association: AssociationKind::Kind,
                        object: AssociationObject::Type(ty.clone()),
                    }]
                })
                .unwrap_or_default(),
            (EntitySelector::Relation(id), AssociationObject::Type(ty)) => self
                .state
                .relations
                .get(id)
                .filter(|relation| &relation.ty == ty)
                .map(|_| {
                    vec![AssociationMatch {
                        subject: EntitySelector::Relation(id.clone()),
                        association: AssociationKind::Kind,
                        object: AssociationObject::Type(ty.clone()),
                    }]
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn query_attribute_associations(&self, query: &AssociationQuery) -> Vec<AssociationMatch> {
        match (&query.subject, &query.object) {
            (EntitySelector::AnyNode, AssociationObject::Attribute(key)) => self
                .query(&HiveQuery::NodesWithAttribute(key.clone()))
                .nodes
                .into_iter()
                .map(|id| AssociationMatch {
                    subject: EntitySelector::Node(id),
                    association: AssociationKind::Attribute,
                    object: AssociationObject::Attribute(key.clone()),
                })
                .collect(),
            (EntitySelector::AnyRelation, AssociationObject::Attribute(key)) => self
                .query(&HiveQuery::RelationsWithAttribute(key.clone()))
                .relations
                .into_iter()
                .map(|id| AssociationMatch {
                    subject: EntitySelector::Relation(id),
                    association: AssociationKind::Attribute,
                    object: AssociationObject::Attribute(key.clone()),
                })
                .collect(),
            (EntitySelector::Node(id), AssociationObject::Attribute(key)) => self
                .state
                .nodes
                .get(id)
                .filter(|node| node.attributes.contains_key(key))
                .map(|_| {
                    vec![AssociationMatch {
                        subject: EntitySelector::Node(id.clone()),
                        association: AssociationKind::Attribute,
                        object: AssociationObject::Attribute(key.clone()),
                    }]
                })
                .unwrap_or_default(),
            (EntitySelector::Relation(id), AssociationObject::Attribute(key)) => self
                .state
                .relations
                .get(id)
                .filter(|relation| relation.attributes.contains_key(key))
                .map(|_| {
                    vec![AssociationMatch {
                        subject: EntitySelector::Relation(id.clone()),
                        association: AssociationKind::Attribute,
                        object: AssociationObject::Attribute(key.clone()),
                    }]
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    fn query_endpoint_associations(
        &self,
        query: &AssociationQuery,
        sign: Option<Sign>,
    ) -> Vec<AssociationMatch> {
        let candidate_relations: Vec<RelationId> = match &query.subject {
            EntitySelector::AnyRelation => self.state.relations.keys().cloned().collect(),
            EntitySelector::Relation(id) if self.state.relations.contains_key(id) => {
                vec![id.clone()]
            }
            _ => Vec::new(),
        };

        let mut out = Vec::new();
        for relation_id in candidate_relations {
            let relation = &self.state.relations[&relation_id];
            for (endpoint_sign, node_id) in &relation.endpoints {
                if sign.is_some_and(|wanted| wanted != *endpoint_sign) {
                    continue;
                }
                let object_matches = match &query.object {
                    AssociationObject::Any => true,
                    AssociationObject::Node(wanted) => wanted == node_id,
                    AssociationObject::NodeType(ty) => self
                        .state
                        .nodes
                        .get(node_id)
                        .is_some_and(|node| &node.ty == ty),
                    _ => false,
                };
                if object_matches {
                    out.push(AssociationMatch {
                        subject: EntitySelector::Relation(relation_id.clone()),
                        association: AssociationKind::Endpoint {
                            sign: Some(*endpoint_sign),
                        },
                        object: AssociationObject::Node(node_id.clone()),
                    });
                }
            }
        }
        out
    }
}

fn ids_from_set<T: Clone + Ord>(set: Option<&BTreeSet<T>>) -> Vec<T> {
    set.map_or_else(Vec::new, |ids| ids.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(
        store: &HiveStore,
        id: &str,
        actor: &str,
        seq: u64,
        deltas: Vec<HiveDelta>,
    ) -> HiveTransaction {
        HiveTransaction::new(id, store.state_hash(), actor, seq, deltas)
    }

    fn node(id: &str, ty: &str) -> HiveDelta {
        HiveDelta::AddNode(HiveNode::new(id, ty))
    }

    #[test]
    fn empty_store_has_stable_hash() {
        let a = HiveStore::new();
        let b = HiveStore::new();
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.generation(), 0);
    }

    #[test]
    fn commit_adds_nodes_and_relation() {
        let mut store = HiveStore::new();
        let receipt = store
            .commit(tx(
                &store,
                "tx-1",
                "akoire",
                0,
                vec![
                    node("codex", "agent"),
                    node("epistemic_care", "trait"),
                    HiveDelta::AddRelation(HiveRelation::new(
                        "has_trait",
                        "has_attribute",
                        vec![
                            (Sign::Plus, "codex".into()),
                            (Sign::Plus, "epistemic_care".into()),
                        ],
                    )),
                ],
            ))
            .unwrap();

        assert_eq!(receipt.status, CommitStatus::Applied);
        assert_eq!(store.generation(), 1);
        assert_eq!(store.state().nodes().len(), 2);
        assert_eq!(store.state().relations().len(), 1);
        assert_ne!(receipt.old_hash, receipt.new_hash);
    }

    #[test]
    fn batch_is_atomic_when_later_delta_fails() {
        let mut store = HiveStore::new();
        let before_hash = store.state_hash();
        let err = store
            .commit(tx(
                &store,
                "tx-bad",
                "akoire",
                0,
                vec![
                    node("codex", "agent"),
                    HiveDelta::AddRelation(HiveRelation::new(
                        "bad",
                        "has_attribute",
                        vec![(Sign::Plus, "missing".into())],
                    )),
                ],
            ))
            .unwrap_err();

        assert!(matches!(
            err,
            CommitError::DeltaFailed {
                index: 1,
                reason: DeltaError::NodeMissing(_)
            }
        ));
        assert_eq!(store.generation(), 0);
        assert_eq!(store.state_hash(), before_hash);
        assert!(store.state().nodes().is_empty());
    }

    #[test]
    fn stale_parent_hash_is_rejected() {
        let mut store = HiveStore::new();
        let stale = store.state_hash();
        store
            .commit(tx(&store, "tx-1", "akoire", 0, vec![node("a", "agent")]))
            .unwrap();

        let err = store
            .commit(HiveTransaction::new(
                "tx-2",
                stale,
                "akoire",
                1,
                vec![node("b", "agent")],
            ))
            .unwrap_err();

        assert!(matches!(err, CommitError::ParentHashMismatch { .. }));
        assert_eq!(store.state().nodes().len(), 1);
    }

    #[test]
    fn replay_by_transaction_id_is_idempotent() {
        let mut store = HiveStore::new();
        let transaction = tx(&store, "tx-1", "akoire", 0, vec![node("a", "agent")]);
        let first = store.commit(transaction.clone()).unwrap();
        let second = store.commit(transaction).unwrap();

        assert_eq!(first.status, CommitStatus::Applied);
        assert_eq!(second.status, CommitStatus::Replayed);
        assert_eq!(first.new_hash, second.new_hash);
        assert_eq!(store.generation(), 1);
        assert_eq!(store.state().nodes().len(), 1);
    }

    #[test]
    fn actor_sequence_must_be_contiguous() {
        let mut store = HiveStore::new();
        let err = store
            .commit(tx(&store, "tx-2", "hotaru", 1, vec![node("a", "agent")]))
            .unwrap_err();

        assert!(matches!(
            err,
            CommitError::SequenceMismatch {
                expected: 0,
                actual: 1,
                ..
            }
        ));
        assert_eq!(store.generation(), 0);
    }

    #[test]
    fn query_subset_matches_nodes_relations_and_signed_endpoint_types() {
        let mut store = HiveStore::new();
        store
            .commit(tx(
                &store,
                "tx-1",
                "akoire",
                0,
                vec![
                    node("codex", "agent"),
                    node("epistemic_care", "trait"),
                    node("implement_and_verify", "behavior"),
                    HiveDelta::AttachNodeAttribute {
                        id: "epistemic_care".into(),
                        key: "weight".to_string(),
                        value: AttributeValue::Float(0.96),
                    },
                    HiveDelta::AddRelation(HiveRelation::new(
                        "rule_finish_with_evidence",
                        "commits_to",
                        vec![
                            (Sign::Plus, "epistemic_care".into()),
                            (Sign::Minus, "implement_and_verify".into()),
                        ],
                    )),
                ],
            ))
            .unwrap();

        let traits = store.query(&HiveQuery::NodesByType("trait".to_string()));
        assert_eq!(traits.nodes, vec![NodeId::from("epistemic_care")]);

        let weighted = store.query(&HiveQuery::NodesWithAttribute("weight".to_string()));
        assert_eq!(weighted.nodes, vec![NodeId::from("epistemic_care")]);

        let rules = store.query(&HiveQuery::RelationsByType("commits_to".to_string()));
        assert_eq!(
            rules.relations,
            vec![RelationId::from("rule_finish_with_evidence")]
        );

        let minus_behavior = store.query(&HiveQuery::RelationsByEndpointNodeType {
            sign: Sign::Minus,
            node_type: "behavior".to_string(),
        });
        assert_eq!(
            minus_behavior.relations,
            vec![RelationId::from("rule_finish_with_evidence")]
        );
    }

    #[test]
    fn association_queries_generalize_kind_attribute_and_endpoint_predicates() {
        let mut store = HiveStore::new();
        store
            .commit(tx(
                &store,
                "tx-1",
                "akoire",
                0,
                vec![
                    node("codex", "agent"),
                    node("epistemic_care", "trait"),
                    node("implement_and_verify", "behavior"),
                    HiveDelta::AttachNodeAttribute {
                        id: "epistemic_care".into(),
                        key: "weight".to_string(),
                        value: AttributeValue::Float(0.96),
                    },
                    HiveDelta::AddRelation(HiveRelation::new(
                        "rule_finish_with_evidence",
                        "commits_to",
                        vec![
                            (Sign::Plus, "epistemic_care".into()),
                            (Sign::Minus, "implement_and_verify".into()),
                        ],
                    )),
                ],
            ))
            .unwrap();

        let kind_matches = store.query_associations(&AssociationQuery::node_kind("trait"));
        assert_eq!(
            kind_matches,
            vec![AssociationMatch {
                subject: EntitySelector::Node("epistemic_care".into()),
                association: AssociationKind::Kind,
                object: AssociationObject::Type("trait".to_string()),
            }]
        );

        let attr_matches = store.query_associations(&AssociationQuery {
            subject: EntitySelector::AnyNode,
            association: AssociationKind::Attribute,
            object: AssociationObject::Attribute("weight".to_string()),
        });
        assert_eq!(
            attr_matches,
            vec![AssociationMatch {
                subject: EntitySelector::Node("epistemic_care".into()),
                association: AssociationKind::Attribute,
                object: AssociationObject::Attribute("weight".to_string()),
            }]
        );

        let endpoint_matches = store.query_associations(
            &AssociationQuery::relation_endpoint_node_type(Sign::Minus, "behavior"),
        );
        assert_eq!(
            endpoint_matches,
            vec![AssociationMatch {
                subject: EntitySelector::Relation("rule_finish_with_evidence".into()),
                association: AssociationKind::Endpoint {
                    sign: Some(Sign::Minus)
                },
                object: AssociationObject::Node("implement_and_verify".into()),
            }]
        );
    }

    #[test]
    fn minimal_llm_query_descriptions_parse_to_typed_hive_queries() {
        assert_eq!(
            HiveQuery::parse_minimal("node.type:trait").unwrap(),
            HiveQuery::NodesByType("trait".to_string())
        );
        assert_eq!(
            HiveQuery::parse_minimal("relation.type:commits_to").unwrap(),
            HiveQuery::RelationsByType("commits_to".to_string())
        );
        assert_eq!(
            HiveQuery::parse_minimal("node.attr:weight").unwrap(),
            HiveQuery::NodesWithAttribute("weight".to_string())
        );
        assert_eq!(
            HiveQuery::parse_minimal("relation.endpoint_type:-:behavior").unwrap(),
            HiveQuery::RelationsByEndpointNodeType {
                sign: Sign::Minus,
                node_type: "behavior".to_string()
            }
        );
        assert_eq!(
            HiveQuery::parse_minimal("relation.endpoint:plus:codex").unwrap(),
            HiveQuery::RelationsByEndpointNode {
                sign: Sign::Plus,
                node: "codex".into()
            }
        );
    }

    #[test]
    fn minimal_llm_query_descriptions_reject_ambiguous_input() {
        assert!(matches!(
            HiveQuery::parse_minimal(""),
            Err(QueryParseError::Empty)
        ));
        assert!(matches!(
            HiveQuery::parse_minimal("node.type"),
            Err(QueryParseError::Malformed(_))
        ));
        assert!(matches!(
            HiveQuery::parse_minimal("relation.endpoint:*:codex"),
            Err(QueryParseError::UnknownSign(_))
        ));
        assert!(matches!(
            HiveQuery::parse_minimal("select * from nodes"),
            Err(QueryParseError::UnknownHead(_))
        ));
    }
}
