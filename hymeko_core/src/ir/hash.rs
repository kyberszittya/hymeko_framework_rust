use serde::{Deserialize, Serialize};
use crate::common::ids::SymId;
use crate::common::pathkey::PathKey;
use crate::ir::ir::DeclKind;
use crate::resolution::interner::Interner;
use crate::resolution::resolve::Index;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug,
    Serialize, Deserialize)]
pub struct HashId(pub [u8; 32]);

fn path_bytes(path: &[SymId], it: &Interner) -> Vec<u8> {
    // pl: "fano.n0" UTF-8
    let s = path.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
    s.into_bytes()
}

pub fn hash_doc(idx: &Index, it: &Interner) -> HashId {
    let mut h = blake3::Hasher::new();
    h.update(b"hymeko-doc-v1");
    let mut scratch = Vec::with_capacity(256);
    for k in idx.by_path.keys() {
        scratch.clear(); // Resets length to 0, but KEEPS the allocated capacity

        for (i, &sym) in k.0.iter().enumerate() {
            if i > 0 {
                scratch.push(b'.');
            }
            scratch.extend_from_slice(it.resolve(sym).as_bytes());
        }
        scratch.push(b'\n');

        // One single function call to Blake3 per path, passing a contiguous block of memory.
        h.update(&scratch);
    }
    HashId(*h.finalize().as_bytes())
}

pub fn hash_decl(doc: HashId, kind: DeclKind, path: &PathKey, it: &Interner) -> HashId {
    let mut h = blake3::Hasher::new();
    h.update(b"hymeko-decl-v1");
    h.update(&doc.0);
    h.update(match kind {
        DeclKind::Node => b"N",
        DeclKind::Edge => b"E",
        DeclKind::HyperArc => b"A",
    });
    let pb = path.0.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
    h.update(pb.as_bytes());
    HashId(*h.finalize().as_bytes())
}