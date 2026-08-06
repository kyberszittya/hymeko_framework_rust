//! Small performance checks for native CR and Chebyshev-CR kernels.

use std::{
    env,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use hymeko_nagare::{
    catmull_rom_forward, chebyshev_cr_forward, chebyshev_deploy_forward, chebyshev_knot_basis,
};

#[derive(Clone, Debug)]
struct Case {
    name: &'static str,
    n: usize,
    channels: usize,
    grid: usize,
    k: usize,
}

#[derive(Clone, Debug)]
struct Timing {
    mean_us_per_sample: f64,
    median_us_per_sample: f64,
    max_us_per_sample: f64,
}

#[derive(Clone, Debug)]
struct Row {
    case: Case,
    cr: Timing,
    cheb_cr_train: Timing,
    cheb_deploy: Timing,
    cr_checksum: f32,
    cheb_cr_checksum: f32,
    cheb_deploy_checksum: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (out, repeats) = parse_args()?;
    let cases = [
        Case {
            name: "tiny_descriptor",
            n: 1,
            channels: 16,
            grid: 8,
            k: 5,
        },
        Case {
            name: "small_descriptions",
            n: 8,
            channels: 32,
            grid: 12,
            k: 5,
        },
        Case {
            name: "toy_point_batch",
            n: 96 * 48,
            channels: 32,
            grid: 12,
            k: 5,
        },
    ];

    let rows: Vec<_> = cases.iter().map(|case| bench_case(case, repeats)).collect();
    for row in &rows {
        println!(
            "{} cr={:.4}us cheb_cr={:.4}us deploy={:.4}us",
            row.case.name,
            row.cr.median_us_per_sample,
            row.cheb_cr_train.median_us_per_sample,
            row.cheb_deploy.median_us_per_sample
        );
    }
    if let Some(path) = out {
        write_json(&path, &rows)?;
    }
    Ok(())
}

fn bench_case(case: &Case, repeats: usize) -> Row {
    let x = make_x(case.n, case.channels);
    let cr_coef = make_coef(case.channels, case.grid);
    let cheb_coef = make_coef(case.channels, case.k);
    let basis = chebyshev_knot_basis(case.grid, case.k);

    let cr_checksum = catmull_rom_forward(&cr_coef, &x, case.n, case.channels, case.grid)
        .0
        .iter()
        .sum();
    let cheb_cr_checksum =
        chebyshev_cr_forward(&cheb_coef, &x, case.n, case.channels, case.grid, case.k)
            .0
            .iter()
            .sum();
    let cheb_deploy_checksum =
        chebyshev_deploy_forward(&cheb_coef, &x, case.n, case.channels, case.k)
            .iter()
            .sum();

    Row {
        case: case.clone(),
        cr: time(repeats, case.n, || {
            catmull_rom_forward(&cr_coef, &x, case.n, case.channels, case.grid).0
        }),
        cheb_cr_train: time(repeats, case.n, || {
            let control = control_points_fast(&cheb_coef, &basis, case.channels, case.grid, case.k);
            catmull_rom_forward(&control, &x, case.n, case.channels, case.grid).0
        }),
        cheb_deploy: time(repeats, case.n, || {
            chebyshev_deploy_forward(&cheb_coef, &x, case.n, case.channels, case.k)
        }),
        cr_checksum,
        cheb_cr_checksum,
        cheb_deploy_checksum,
    }
}

fn time<F>(repeats: usize, n: usize, mut f: F) -> Timing
where
    F: FnMut() -> Vec<f32>,
{
    for _ in 0..20 {
        let out = f();
        assert!(out.iter().all(|v| v.is_finite()));
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let out = f();
        assert!(out.iter().all(|v| v.is_finite()));
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / n as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Timing {
        mean_us_per_sample: values.iter().sum::<f64>() / values.len() as f64,
        median_us_per_sample: values[values.len() / 2],
        max_us_per_sample: values[values.len() - 1],
    }
}

fn make_x(n: usize, channels: usize) -> Vec<f32> {
    (0..n * channels)
        .map(|i| {
            let v = ((i * 37 + 11) % 1009) as f32 / 1008.0;
            2.4 * v - 1.2
        })
        .collect()
}

fn make_coef(channels: usize, width: usize) -> Vec<f32> {
    (0..channels * width)
        .map(|i| {
            let v = ((i * 97 + 23) % 997) as f32 / 996.0;
            0.2 * v - 0.1
        })
        .collect()
}

fn control_points_fast(
    coef: &[f32],
    basis: &[f32],
    channels: usize,
    grid: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; channels * grid];
    for ch in 0..channels {
        for g in 0..grid {
            let mut acc = 0.0;
            for term in 0..k {
                acc += coef[ch * k + term] * basis[g * k + term];
            }
            out[ch * grid + g] = acc;
        }
    }
    out
}

fn parse_args() -> Result<(Option<PathBuf>, usize), io::Error> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut out = None;
    let mut repeats = 500usize;
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| io::Error::other(format!("missing value for {key}")))?;
        match key.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--repeats" => {
                repeats = value
                    .parse()
                    .map_err(|_| io::Error::other("bad --repeats"))?
            }
            _ => return Err(io::Error::other(format!("unknown flag {key}"))),
        }
        i += 2;
    }
    Ok((out, repeats))
}

fn write_json(path: &PathBuf, rows: &[Row]) -> io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(f, "  \"cases\": [")?;
    for (idx, row) in rows.iter().enumerate() {
        let comma = if idx + 1 == rows.len() { "" } else { "," };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"case\": \"{}\",", row.case.name)?;
        writeln!(f, "      \"n\": {},", row.case.n)?;
        writeln!(f, "      \"channels\": {},", row.case.channels)?;
        writeln!(f, "      \"grid\": {},", row.case.grid)?;
        writeln!(f, "      \"k\": {},", row.case.k)?;
        write_timing(&mut f, "cr", &row.cr, true)?;
        write_timing(&mut f, "cheb_cr_train", &row.cheb_cr_train, true)?;
        write_timing(&mut f, "cheb_deploy", &row.cheb_deploy, true)?;
        writeln!(f, "      \"cr_checksum\": {:.9},", row.cr_checksum)?;
        writeln!(
            f,
            "      \"cheb_cr_checksum\": {:.9},",
            row.cheb_cr_checksum
        )?;
        writeln!(
            f,
            "      \"cheb_deploy_checksum\": {:.9}",
            row.cheb_deploy_checksum
        )?;
        writeln!(f, "    }}{comma}")?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

fn write_timing(f: &mut File, name: &str, timing: &Timing, comma: bool) -> io::Result<()> {
    writeln!(f, "      \"{name}\": {{")?;
    writeln!(
        f,
        "        \"mean_us_per_sample\": {:.9},",
        timing.mean_us_per_sample
    )?;
    writeln!(
        f,
        "        \"median_us_per_sample\": {:.9},",
        timing.median_us_per_sample
    )?;
    writeln!(
        f,
        "        \"max_us_per_sample\": {:.9}",
        timing.max_us_per_sample
    )?;
    writeln!(f, "      }}{}", if comma { "," } else { "" })?;
    Ok(())
}
