use blake3::Hasher;
use crate::common::ids::DeclId;
use crate::ir::ir::{AnnoR, Ir, SignedRefR, ValueR};

// Assuming HashId wraps a [u8; 32]
use crate::ir::hash::HashId;
use crate::resolution::interner::Interner;

pub fn compute_merkle_hashes(ir: &mut Ir, interner: &Interner) {
    let num_decls = ir.decl_kind.len();

    if ir.decl_hash.len() < num_decls {
        ir.decl_hash.resize(num_decls, None);
    }

    // Process backwards to ensure children are hashed before parents [cite: 2026-02-08].
    for i in (0..num_decls).rev() {
        let did = DeclId(i as u32);
        let mut hasher = Hasher::new();

        // 1. Hash Kind and Name
        hasher.update(&[ir.decl_kind[i] as u8]);
        let name_str = interner.resolve(ir.decl_name[i]);
        hasher.update(name_str.as_bytes());

        // 2. Hash Annotations
        hash_anno(&ir.decl_anno[i], &mut hasher, ir, interner);

        // 3. Hash Children (Merkle linkage) [cite: 2026-02-08]
        for child_did in ir.decl_children(did) {
            if let Some(h) = ir.decl_hash[child_did.0 as usize] {
                hasher.update(&h.0);
            }
        }

        // 4. Special handling for Arcs (they contain references/weights)
        if let Some(aid) = ir.as_arc(did) {
            let arc = &ir.arcs[aid.0 as usize];
            for sref in &arc.refs {
                hash_signed_ref(sref, &mut hasher, ir, interner);
            }
        }

        ir.decl_hash[i] = Some(HashId(hasher.finalize().into()));
    }
}

fn hash_anno(anno: &AnnoR, hasher: &mut Hasher, ir: &Ir, it: &Interner) {
    // 1. Hash the number of tags to prevent "concatenation collisions" [cite: 2026-02-08]
    let num_tags = anno.tags.len() as u64;
    hasher.update(&num_tags.to_le_bytes());

    // 2. Resolve and hash each tag string
    for &tag_sid in &anno.tags {
        // Ensure you are passing the SymId, not a reference to it [cite: 2026-02-08]
        let tag_str = it.resolve(tag_sid);
        hasher.update(tag_str.as_bytes());
    }

    // 3. Hash the presence and content of the value
    match &anno.value {
        Some(v) => {
            hasher.update(&[1]); // Presence flag (discriminant)
            hash_value(v, hasher, ir, it);
        }
        None => {
            hasher.update(&[0]); // Absence flag
        }
    }
}

fn hash_value(v: &ValueR, hasher: &mut Hasher, ir: &Ir, it: &Interner) {
    match v {
        ValueR::Str(sid) => {
            hasher.update(&[0]); // Type tag
            hasher.update(it.resolve(*sid).as_bytes());
        }
        ValueR::Num(n) => {
            hasher.update(&[1]);
            hasher.update(&n.to_le_bytes());
        }
        ValueR::List(xs) => {
            hasher.update(&[2]);
            hasher.update(&(xs.len() as u64).to_le_bytes());
            for x in xs {
                hash_value(x, hasher, ir, it);
            }
        }
        ValueR::Ref(did) => {
            hasher.update(&[3]);
            // Use the HashId of the target to stay location-independent [cite: 2026-02-08].
            if let Some(h) = ir.decl_hash[did.0 as usize] {
                hasher.update(&h.0);
            }
        }
    }
}

fn hash_signed_ref(sref: &SignedRefR, hasher: &mut Hasher, ir: &Ir, it: &Interner) {
    // Hash the variant/sign
    let (tag, atom) = match sref {
        SignedRefR::Plus(a) => (1u8, a),
        SignedRefR::Minus(a) => (2u8, a),
        SignedRefR::Neutral(a) => (3u8, a),
    };
    hasher.update(&[tag]);

    // Hash the Target's HashId [cite: 2026-02-08]
    if let Some(h) = ir.decl_hash[atom.target.0 as usize] {
        hasher.update(&h.0);
    }

    // Hash Atom-specific annotations and weights
    hash_anno(&atom.anno, hasher, ir, it);
    if let Some(weights) = &atom.weights {
        hasher.update(&[1]);
        for w in weights {
            hash_value(w, hasher, ir, it);
        }
    } else {
        hasher.update(&[0]);
    }
}