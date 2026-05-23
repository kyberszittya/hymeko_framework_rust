//! Round-trip + error-path tests for the wire format.

use hymeko_emitter::editor_ir::{IRDelta, Vertex};
use hymeko_wire::{FLAG_ZSTD, HEADER_SIZE, MAGIC, VERSION, WireError, decode_delta, encode_delta};

fn sample_delta() -> IRDelta {
    IRDelta::AddVertex {
        data: Vertex {
            name: "base_link".into(),
            level: 0,
            attributes: Vec::new(),
            position: None,
        },
    }
}

#[test]
fn roundtrip_compressed() {
    let delta = sample_delta();
    let packet = encode_delta(&delta, 42, 7, true).unwrap();
    assert!(packet.len() > HEADER_SIZE);
    let (header, out) = decode_delta(&packet).unwrap();
    // Copy packed fields into locals before asserting — the
    // `#[repr(C, packed)]` header has unaligned fields and taking
    // references to them is undefined behaviour.
    let magic = header.magic;
    let version = header.version;
    let patch_id = header.patch_id;
    let seq = header.delta_seq;
    let flags = header.flags;
    assert_eq!(magic, MAGIC);
    assert_eq!(version, VERSION);
    assert_eq!(patch_id, 42);
    assert_eq!(seq, 7);
    assert!(flags & FLAG_ZSTD != 0);
    match out {
        IRDelta::AddVertex { data } => assert_eq!(data.name, "base_link"),
        _ => panic!("unexpected variant"),
    }
}

#[test]
fn roundtrip_uncompressed() {
    let delta = sample_delta();
    let packet = encode_delta(&delta, 1, 1, false).unwrap();
    let (header, _) = decode_delta(&packet).unwrap();
    let flags = header.flags;
    assert_eq!(flags & FLAG_ZSTD, 0);
}

#[test]
fn bad_magic_is_rejected() {
    let delta = sample_delta();
    let mut packet = encode_delta(&delta, 0, 0, true).unwrap().to_vec();
    // Corrupt the magic bytes.
    packet[0] = 0xFF;
    packet[1] = 0xFF;
    match decode_delta(&packet).unwrap_err() {
        WireError::BadMagic => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn checksum_mismatch_is_caught() {
    let delta = sample_delta();
    let mut packet = encode_delta(&delta, 0, 0, true).unwrap().to_vec();
    // Corrupt a byte in the payload (past the header) so the checksum
    // still points at the original payload hash.
    let payload_start = HEADER_SIZE;
    packet[payload_start] ^= 0xFF;
    match decode_delta(&packet).unwrap_err() {
        WireError::ChecksumMismatch { .. } => {}
        other => panic!("expected ChecksumMismatch, got {other:?}"),
    }
}

#[test]
fn packet_shorter_than_header_errors() {
    match decode_delta(&[0u8; 10]).unwrap_err() {
        WireError::TooShort => {}
        WireError::BadMagic => {}
        other => panic!("expected TooShort/BadMagic, got {other:?}"),
    }
}

#[test]
fn encoded_packet_has_magic_prefix() {
    let packet = encode_delta(&sample_delta(), 0, 0, true).unwrap();
    let prefix = u32::from_le_bytes(packet[0..4].try_into().unwrap());
    assert_eq!(prefix, MAGIC);
}

#[test]
fn large_batch_delta_roundtrips() {
    // Exercises the zstd path on a meaningfully-sized payload.
    let mut deltas = Vec::new();
    for i in 0..500 {
        deltas.push(IRDelta::AddVertex {
            data: Vertex {
                name: format!("node_{i:04}"),
                level: (i as i8) % 5,
                attributes: Vec::new(),
                position: None,
            },
        });
    }
    let batch = IRDelta::Batch { deltas };
    let packet = encode_delta(&batch, 10, 10, true).unwrap();
    let (_, decoded) = decode_delta(&packet).unwrap();
    match decoded {
        IRDelta::Batch { deltas } => assert_eq!(deltas.len(), 500),
        _ => panic!("expected Batch"),
    }
}
