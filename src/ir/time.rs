use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ns() -> i128 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime before UNIX_EPOCH");
    (d.as_secs() as i128) * 1_000_000_000 + (d.subsec_nanos() as i128)
}

pub fn now_ns_u128() -> u128 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime before UNIX_EPOCH");
    (d.as_secs() as u128) * 1_000_000_000 + (d.subsec_nanos() as u128)
}

pub fn now_ms() -> i64 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("SystemTime before UNIX_EPOCH");
    (d.as_secs() as i64) * 1000 + (d.subsec_millis() as i64)
}