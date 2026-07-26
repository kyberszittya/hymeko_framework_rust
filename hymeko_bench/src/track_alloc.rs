//! Process-wide heap-accounting global allocator for the camera-ready
//! memory benchmarks (SMC 2026 Task 1 / Task 2).
//!
//! Wraps the system allocator and maintains two atomic counters: the current
//! live byte count and the high-water mark. This is the dependency-free
//! equivalent of `stats_alloc` (net resident) plus `dhat` (peak) that the
//! benchmark prompt names — implemented in-tree so no new Cargo dependency is
//! added to a `CORE.YAML`-pinned tree.
//!
//! # Why a global
//! A `#[global_allocator]` is, by language design, the single process-wide
//! allocation hook; there is no non-global way to observe every allocation.
//! This is the narrowly-justified exception to the project's
//! "no module-level mutable state" rule (CLAUDE.md §6.5 #11): the atomics are
//! confined to this benchmark crate, are never read inside a hot loop for
//! control flow, and exist solely for measurement.
//!
//! # Preconditions / Invariants
//! - Delta measurements (`current_bytes` before/after a region) are only exact
//!   when the region runs **single-threaded** with all scratch allocations
//!   freed before the closing snapshot. The Task 1/2 harnesses satisfy this.
//! - Counters are monotone-consistent: every tracked `alloc`/`realloc` grow is
//!   matched by the corresponding `dealloc`/`realloc` shrink.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Heap-accounting allocator delegating to the system allocator.
pub struct TrackingAlloc;

#[inline]
fn record_alloc(size: usize) {
    // `fetch_add` returns the previous value; the post-increment live total is
    // `prev + size`. Bump the high-water mark to at least that.
    let live = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[inline]
fn record_free(size: usize) {
    CURRENT.fetch_sub(size, Ordering::Relaxed);
}

// SAFETY: every branch delegates to `System` with the caller's exact layout and
// only wraps the returned pointer with counter bookkeeping; no aliasing or
// layout mutation is introduced.
unsafe impl GlobalAlloc for TrackingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        record_free(layout.size());
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            record_alloc(layout.size());
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            // Net the delta so the live total reflects the resize.
            record_free(layout.size());
            record_alloc(new_size);
        }
        p
    }
}

/// Current live (allocated − freed) byte count.
#[must_use]
pub fn current_bytes() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

/// High-water mark of the live byte count since process start (or last
/// [`reset_peak`]).
#[must_use]
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Reset the high-water mark to the current live total, so a subsequent
/// [`peak_bytes`] measures only the peak reached *after* this call.
pub fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Run `f`, returning `(value, retained_bytes, peak_delta_bytes)` where
/// `retained_bytes` is the net live-heap growth still held when `f` returns and
/// `peak_delta_bytes` is the transient high-water mark reached during `f`, both
/// relative to entry. `retained_bytes` saturates at 0 (a region that frees more
/// than it allocates reports 0 retained).
///
/// # Preconditions
/// Must be called single-threaded; concurrent allocations on another thread
/// corrupt the delta.
pub fn measure<T, F: FnOnce() -> T>(f: F) -> (T, usize, usize) {
    let base = current_bytes();
    reset_peak();
    let value = f();
    let after = current_bytes();
    let peak = peak_bytes();
    let retained = after.saturating_sub(base);
    let peak_delta = peak.saturating_sub(base);
    (value, retained, peak_delta)
}

// NOTE: `measure`/`peak_bytes` can only be exercised when this crate's
// `TrackingAlloc` is installed as the process `#[global_allocator]`. A
// `cargo test --lib` harness does not install it, so the end-to-end coverage
// lives in `tests/track_alloc_it.rs`, which declares the global allocator.
