use blake3::Hasher;
use crate::common::ids::DeclId;
use crate::interner::Interner;
use crate::ir::ir::{Ir, ValueR};

// Assuming HashId wraps a [u8; 32]
use crate::ir::hash::HashId;


fn hash_value(v: &ValueR, hasher: &mut Hasher, ir: &Ir, it: &Interner) {
    match v {
        ValueR::Str(sid) => {
            hasher.update(&[0]); // Variant tag
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
            // We hash the HashId of the target, NOT the DeclId index.
            // This ensures the hash is independent of memory layout [cite: 2026-02-08].
            if let Some(target_hash) = ir.decl_hash[did.0 as usize] {
                hasher.update(&target_hash.0);
            }
        }
    }
}

pub fn compute_merkle_hashes(ir: &mut Ir, interner: &Interner) {
    let num_decls = ir.decl_kind.len();

    // Resize the hash array if it isn't fully allocated
    if ir.decl_hash.len() < num_decls {
        ir.decl_hash.resize(num_decls, None);
    }

    // A Japanese Teacher's DOD trick: Iterate strictly backwards.
    // Because children are parsed after parents, Child ID > Parent ID.
    // By going backwards, children are guaranteed to be hashed before their parents [cite: 2025-10-31, 2026-02-08].
    for i in (0..num_decls).rev() {
        let did = DeclId(i as u32);

        let mut hasher = Hasher::new();

        // 1. Hash the structural kind (Node, Edge, Arc)
        let kind = ir.decl_kind(did);
        hasher.update(&[kind as u8]);

        // 2. Hash the exact string name directly from the Interner
        let name_sym = ir.decl_name[did.0 as usize];
        let name_str = interner.resolve(name_sym);
        hasher.update(name_str.as_bytes());

        // 3. Hash the children linearly
        // Because we are going backwards, the child hashes are already computed [cite: 2026-02-08].
        for child_did in ir.decl_children(did) {
            let child_hash = ir.decl_hash[child_did.0 as usize]
                .expect("Child hash must exist due to reverse topological order");
            hasher.update(&child_hash.0); // Assuming HashId(pub [u8; 32])
        }

        // 4. (Optional) Hash annotations and Arc references here if you want them
        // to strictly define the element's identity.

        // Store the finalized 32-byte cryptographic hash
        let anno = &ir.decl_anno[did.0 as usize];
        hasher.update(&(anno.tags.len() as u64).to_le_bytes());
        for &tag_sid in &anno.tags {
            hasher.update(interner.resolve(tag_sid).as_bytes());
        }

        if let Some(val) = &anno.value {
            hasher.update(&[1]); // Presence tag
            hash_value(val, &mut hasher, ir, interner);
        } else {
            hasher.update(&[0]);
        }

        // 5. Finalize the Content-Addressable ID
        let final_hash = hasher.finalize();
        ir.decl_hash[did.0 as usize] = Some(HashId(final_hash.into()));
    }
}