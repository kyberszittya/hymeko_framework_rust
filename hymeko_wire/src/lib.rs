//! `hymeko_wire` — CBOR + zstd + xxhash3 packet format for
//! `IRDelta` gossip over future P2P transports (`hymeko_p2p`).
//!
//! Packet layout (on the wire):
//!
//! ```text
//! ┌─────────────── PacketHeader (packed, little-endian) ────────────────┐
//! │ u32 magic (0x484D4B4F "HMKO")                                       │
//! │ u16 version                                                         │
//! │ u8  ir_level                                                        │
//! │ u8  flags          bit 0x01 = zstd compressed                       │
//! │ u64 patch_id                                                        │
//! │ u64 delta_seq                                                       │
//! │ u32 checksum       xxh3_32 of the payload (post-compression)        │
//! ├─────────────────────────── Payload ─────────────────────────────────┤
//! │ CBOR-encoded IRDelta, optionally zstd-compressed                    │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use bytes::Bytes;
use hymeko_emitter::editor_ir::IRDelta;
use thiserror::Error;
use xxhash_rust::xxh3::xxh3_64;

/// Magic prefix: ASCII "HMKO" in little-endian order.
pub const MAGIC: u32 = 0x484D_4B4F;

/// Current wire version. Increment on backwards-incompatible changes to
/// the payload shape.
pub const VERSION: u16 = 1;

/// Flag bit — payload is zstd-compressed.
pub const FLAG_ZSTD: u8 = 0x01;

/// On-wire header. `#[repr(C, packed)]` keeps a stable 32-byte layout
/// across compiler versions. All integer fields are little-endian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, packed)]
pub struct PacketHeader {
    pub magic: u32,
    pub version: u16,
    pub ir_level: u8,
    pub flags: u8,
    pub patch_id: u64,
    pub delta_seq: u64,
    /// xxh3_32 of the (post-compression) payload. Stored as u32 so the
    /// header stays 32 bytes even though xxh3 produces 64 bits natively.
    pub checksum: u32,
}

pub const HEADER_SIZE: usize = core::mem::size_of::<PacketHeader>();

impl PacketHeader {
    pub fn new(patch_id: u64, delta_seq: u64, checksum: u32, flags: u8) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            ir_level: 0,
            flags,
            patch_id,
            delta_seq,
            checksum,
        }
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6] = self.ir_level;
        buf[7] = self.flags;
        buf[8..16].copy_from_slice(&self.patch_id.to_le_bytes());
        buf[16..24].copy_from_slice(&self.delta_seq.to_le_bytes());
        buf[24..28].copy_from_slice(&self.checksum.to_le_bytes());
        // 4 trailing bytes of padding to hit the 32-byte size.
        buf
    }

    fn from_bytes(buf: &[u8]) -> Result<Self, WireError> {
        if buf.len() < HEADER_SIZE {
            return Err(WireError::TooShort);
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(WireError::BadMagic);
        }
        Ok(Self {
            magic,
            version: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            ir_level: buf[6],
            flags: buf[7],
            patch_id: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            delta_seq: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            checksum: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
        })
    }
}

#[derive(Debug, Error)]
pub enum WireError {
    #[error("packet too short")]
    TooShort,
    #[error("bad magic number")]
    BadMagic,
    #[error("unsupported wire version {0}")]
    UnsupportedVersion(u16),
    #[error("checksum mismatch (expected {expected:#x}, got {got:#x})")]
    ChecksumMismatch { expected: u32, got: u32 },
    #[error("cbor encode error: {0}")]
    CborEncode(String),
    #[error("cbor decode error: {0}")]
    CborDecode(String),
    #[error("zstd compress error: {0}")]
    Compress(String),
    #[error("zstd decompress error: {0}")]
    Decompress(String),
}

/// Encode an `IRDelta` for the wire. Compression is applied unless the
/// caller explicitly passes `compress = false` (useful for tiny deltas
/// where the zstd framing overhead dwarfs the payload savings).
pub fn encode_delta(
    delta: &IRDelta,
    patch_id: u64,
    seq: u64,
    compress: bool,
) -> Result<Bytes, WireError> {
    // 1. CBOR encode.
    let mut cbor = Vec::new();
    ciborium::into_writer(delta, &mut cbor).map_err(|e| WireError::CborEncode(e.to_string()))?;

    // 2. Optional zstd.
    let (payload, flags) = if compress {
        let compressed =
            zstd::encode_all(cbor.as_slice(), 3).map_err(|e| WireError::Compress(e.to_string()))?;
        (compressed, FLAG_ZSTD)
    } else {
        (cbor, 0)
    };

    // 3. Header + checksum.
    let checksum = (xxh3_64(&payload) & 0xFFFF_FFFF) as u32;
    let header = PacketHeader::new(patch_id, seq, checksum, flags);

    // 4. Assemble.
    let mut packet = Vec::with_capacity(HEADER_SIZE + payload.len());
    packet.extend_from_slice(&header.to_bytes());
    packet.extend_from_slice(&payload);
    Ok(Bytes::from(packet))
}

/// Decode a wire packet back into an `IRDelta`, validating magic, version
/// and checksum along the way.
pub fn decode_delta(packet: &[u8]) -> Result<(PacketHeader, IRDelta), WireError> {
    let header = PacketHeader::from_bytes(packet)?;
    if header.version != VERSION {
        return Err(WireError::UnsupportedVersion(header.version));
    }
    let payload = &packet[HEADER_SIZE..];

    let got = (xxh3_64(payload) & 0xFFFF_FFFF) as u32;
    if got != header.checksum {
        return Err(WireError::ChecksumMismatch {
            expected: header.checksum,
            got,
        });
    }

    let cbor: Vec<u8> = if (header.flags & FLAG_ZSTD) != 0 {
        zstd::decode_all(payload).map_err(|e| WireError::Decompress(e.to_string()))?
    } else {
        payload.to_vec()
    };

    let delta: IRDelta =
        ciborium::from_reader(cbor.as_slice()).map_err(|e| WireError::CborDecode(e.to_string()))?;
    Ok((header, delta))
}
