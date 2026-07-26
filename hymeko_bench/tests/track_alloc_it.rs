//! End-to-end coverage for the tracking allocator: this integration binary
//! installs `TrackingAlloc` as the process global allocator (which a
//! `--lib` unit-test harness cannot), so `measure` sees real allocations.
//!
//! The assertions live in a SINGLE test function on purpose: `measure`
//! requires single-threaded execution (the global counters are process-wide),
//! and cargo runs distinct `#[test]` fns in the same binary on parallel
//! threads. One function ⇒ one thread ⇒ reliable deltas.

use std::hint::black_box;

use hymeko_bench::{measure, peak_bytes};

#[global_allocator]
static GLOBAL: hymeko_bench::TrackingAlloc = hymeko_bench::TrackingAlloc;

#[test]
fn tracking_allocator_measures_retained_and_peak() {
    // (1) a retained 4 KiB allocation shows up as retained bytes >= 4096.
    let (v, retained, peak) = measure(|| {
        let buf = vec![0u8; 4096];
        black_box(&buf);
        buf
    });
    assert_eq!(v.len(), 4096);
    assert!(retained >= 4096, "retained {retained} < 4096");
    assert!(peak >= retained, "peak {peak} < retained {retained}");
    drop(v);

    // (2) a large transient freed before return: peak captures it, and exceeds
    //     what is retained afterwards.
    let (kept, retained2, peak2) = measure(|| {
        let transient = vec![0u8; 1 << 20]; // 1 MiB, freed before return
        black_box(&transient);
        drop(transient);
        vec![0u8; 4096]
    });
    assert_eq!(kept.len(), 4096);
    assert!(peak2 >= (1 << 20), "peak {peak2} missed the 1 MiB transient");
    assert!(peak2 > retained2, "peak {peak2} !> retained {retained2}");
    drop(kept);

    // (3) peak_bytes is a running high-water mark: a fresh large alloc can only
    //     raise it.
    let before = peak_bytes();
    let big = vec![0u8; 2 << 20];
    black_box(&big);
    let after = peak_bytes();
    assert!(after >= before, "peak went down: {after} < {before}");
    drop(big);
}
