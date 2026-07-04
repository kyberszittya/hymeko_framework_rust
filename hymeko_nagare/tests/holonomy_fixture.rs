//! Frozen seed-53 dataset fixture for the holonomy benchmark line.
//!
//! The 2026-07-01/02 Nagare holonomy results (entropy-pool local learner,
//! gate ablations, fitted projection gate) are all anchored on
//! `make_dataset` with seed 53 and the task-index seed offsets used by
//! `examples/entropy_pool_learning_compare.rs`. This fixture freezes those
//! six datasets (3 tasks x train/test) as FNV-1a-64 content hashes over
//! the exact f32 bit patterns, plus an 8-value hex preview for debugging.
//!
//! If `rand`'s `StdRng` stream or the generators ever drift, this test
//! turns the silent benchmark invalidation into a loud failure.
//!
//! Regenerate deliberately with:
//! `cargo test -p hymeko_nagare --test holonomy_fixture -- --ignored`

use std::fmt::Write as _;
use std::path::PathBuf;

use hymeko_nagare::holonomy::{Dataset, Task, make_dataset};

const FIXTURE_SEED: u64 = 53;
const N_TRAIN: usize = 192;
const N_TEST: usize = 96;
const N_POINTS: usize = 32;
const PREVIEW_LEN: usize = 8;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("moons_spiral_xor_seed53.txt")
}

/// FNV-1a 64-bit over a byte stream.
fn fnv1a64(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_x(data: &Dataset) -> u64 {
    fnv1a64(data.x.iter().flat_map(|v| v.to_bits().to_le_bytes()))
}

fn hash_y(data: &Dataset) -> u64 {
    fnv1a64(data.y.iter().flat_map(|&v| (v as u64).to_le_bytes()))
}

fn preview(data: &Dataset) -> String {
    data.x
        .iter()
        .take(PREVIEW_LEN)
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

/// The six benchmark datasets with the exact seed offsets of the compare
/// example: train `seed + idx * 100`, test `seed + idx * 100 + 1`.
fn fixture_cases() -> Vec<(Task, &'static str, usize, u64)> {
    [Task::Moons, Task::Spiral, Task::Xor]
        .iter()
        .enumerate()
        .flat_map(|(idx, &task)| {
            let base = FIXTURE_SEED + idx as u64 * 100;
            [
                (task, "train", N_TRAIN, base),
                (task, "test", N_TEST, base + 1),
            ]
        })
        .collect()
}

fn render_fixture() -> String {
    let mut out = String::new();
    out.push_str("# Frozen seed-53 moons/spiral/xor datasets (Nagare holonomy benchmark line)\n");
    out.push_str("# task split samples points seed x_fnv1a64 y_fnv1a64 x_preview_hex\n");
    for (task, split, samples, seed) in fixture_cases() {
        let data = make_dataset(task, samples, N_POINTS, seed);
        writeln!(
            out,
            "{} {} {} {} {} {:016x} {:016x} {}",
            task.as_str(),
            split,
            samples,
            N_POINTS,
            seed,
            hash_x(&data),
            hash_y(&data),
            preview(&data)
        )
        .expect("string write cannot fail");
    }
    out
}

#[test]
fn seed53_datasets_match_frozen_fixture() {
    let path = fixture_path();
    let frozen = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
    let current = render_fixture();
    // Compare line-by-line for a readable failure message.
    let frozen_lines: Vec<&str> = frozen.lines().filter(|l| !l.starts_with('#')).collect();
    let current_lines: Vec<&str> = current.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(
        frozen_lines.len(),
        current_lines.len(),
        "fixture row count changed"
    );
    for (frozen_line, current_line) in frozen_lines.iter().zip(current_lines.iter()) {
        assert_eq!(
            frozen_line, current_line,
            "seed-53 dataset drifted from the frozen fixture; if the change is \
             deliberate, regenerate with `-- --ignored` and re-anchor the benchmarks"
        );
    }
}

#[test]
#[ignore = "writes the fixture; run deliberately after an intentional generator change"]
fn regenerate_seed53_fixture() {
    let path = fixture_path();
    std::fs::create_dir_all(path.parent().expect("fixture path has a parent"))
        .expect("create fixtures dir");
    std::fs::write(&path, render_fixture()).expect("write fixture");
}
