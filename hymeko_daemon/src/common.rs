use hymeko::ir::ir::Ir;
use std::sync::Arc;

#[derive(Debug)]
#[allow(dead_code)]
// Reserved for alternate ingress wire formats; current paths use `ExecutableQuery` only.
pub enum IngressPayload {
    RawUtf8(Vec<u8>),
    CborEncoded(Vec<u8>),
    CompiledIr(Vec<u8>),
}

#[derive(Clone, Copy, PartialEq)]
pub enum IngressFormat {
    RawUtf8,
    CborEncoded,
    CompiledIr,
}

#[derive(Debug)]
pub struct ExecutableQuery {
    pub ir: Arc<Ir>,
}
