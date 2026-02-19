use crate::common::ids::SymId;
use crate::common::pathkey::PathKey;
use crate::interner::Interner;
use crate::ir::ir::DeclKind;
use crate::resolve::{Index};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct HashId(pub [u8; 32]);

fn path_bytes(path: &[SymId], it: &Interner) -> Vec<u8> {
    // pl: "fano.n0" UTF-8
    let s = path.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
    s.into_bytes()
}

pub fn hash_doc(idx: &Index, it: &Interner) -> HashId {
    let mut keys: Vec<&PathKey> = idx.by_path.keys().collect();
    keys.sort_by(|a, b| {
        let sa = a.0.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
        let sb = b.0.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
        sa.cmp(&sb)
    });

    let mut h = blake3::Hasher::new();
    h.update(b"hymeko-doc-v1");
    for k in keys {
        let pb = k.0.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
        h.update(pb.as_bytes());
        h.update(b"\n");
    }
    HashId(*h.finalize().as_bytes())
}

pub fn hash_decl(doc: HashId, kind: DeclKind, path: &PathKey, it: &Interner) -> HashId {
    let mut h = blake3::Hasher::new();
    h.update(b"hymeko-decl-v1");
    h.update(&doc.0);
    h.update(match kind { DeclKind::Node => b"N", DeclKind::Edge => b"E" });
    let pb = path.0.iter().map(|&x| it.resolve(x)).collect::<Vec<_>>().join(".");
    h.update(pb.as_bytes());
    HashId(*h.finalize().as_bytes())
}