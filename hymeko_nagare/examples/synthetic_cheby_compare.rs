//! Compare Nagare/Rust against a PyTorch synthetic Chebyshev classifier fixture.

use std::{
    collections::HashMap,
    env, fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use hymeko_nagare::{LinearLayer, chebyshev_deploy_forward, linear_forward};

const CHEBYSHEV_DOMAIN_SCALE: f32 = 0.5;

#[derive(Clone, Debug)]
struct TensorBlock {
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Case {
    task: String,
    samples: usize,
    hidden: usize,
    k: usize,
    tensors: HashMap<String, TensorBlock>,
}

impl Case {
    fn tensor(&self, name: &str) -> &[f32] {
        &self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .values
    }

    fn layer(&self, prefix: &str, in_dim: usize, out_dim: usize) -> LinearLayer {
        LinearLayer {
            w: self.tensor(&format!("{prefix}_w")).to_vec(),
            b: self.tensor(&format!("{prefix}_b")).to_vec(),
            in_dim,
            out_dim,
        }
    }
}

#[derive(Clone, Debug)]
struct Report {
    task: String,
    max_abs_logits: f32,
    acc_random_weights: f32,
    median_us_per_sample: f64,
    mean_us_per_sample: f64,
    max_us_per_sample: f64,
    n_params: usize,
    param_bytes: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fixture, out, repeats) = parse_args()?;
    let cases = parse_fixture(&fs::read_to_string(fixture)?)?;
    let reports: Vec<_> = cases.iter().map(|case| report(case, repeats)).collect();
    for row in &reports {
        println!(
            "{} max_abs={:.3e} acc={:.3} median_us={:.4}",
            row.task, row.max_abs_logits, row.acc_random_weights, row.median_us_per_sample
        );
    }
    if let Some(path) = out {
        write_json(&path, &reports)?;
    }
    Ok(())
}

fn forward(case: &Case) -> Vec<f32> {
    let input = case.layer("input", 2, case.hidden);
    let head = case.layer("head", case.hidden, 2);
    let z = linear_forward(&input, case.tensor("x"));
    let scaled: Vec<f32> = z.iter().map(|v| v * CHEBYSHEV_DOMAIN_SCALE).collect();
    let h = chebyshev_deploy_forward(
        case.tensor("cheb_coef"),
        &scaled,
        case.samples,
        case.hidden,
        case.k,
    );
    linear_forward(&head, &h)
}

fn report(case: &Case, repeats: usize) -> Report {
    let logits = forward(case);
    let max_abs_logits = max_abs(&logits, case.tensor("logits"));
    let acc_random_weights = accuracy(&logits, case.tensor("y"));
    let timing = time_case(case, repeats);
    let n_params = case.tensor("input_w").len()
        + case.tensor("input_b").len()
        + case.tensor("cheb_coef").len()
        + case.tensor("head_w").len()
        + case.tensor("head_b").len();
    Report {
        task: case.task.clone(),
        max_abs_logits,
        acc_random_weights,
        median_us_per_sample: timing.0,
        mean_us_per_sample: timing.1,
        max_us_per_sample: timing.2,
        n_params,
        param_bytes: n_params * std::mem::size_of::<f32>(),
    }
}

fn time_case(case: &Case, repeats: usize) -> (f64, f64, f64) {
    for _ in 0..20 {
        let out = forward(case);
        assert!(out.iter().all(|v| v.is_finite()));
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let out = forward(case);
        assert!(out.iter().all(|v| v.is_finite()));
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / case.samples as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    (
        values[values.len() / 2],
        values.iter().sum::<f64>() / values.len() as f64,
        values[values.len() - 1],
    )
}

fn accuracy(logits: &[f32], y: &[f32]) -> f32 {
    let mut correct = 0usize;
    for i in 0..y.len() {
        let pred = usize::from(logits[2 * i + 1] > logits[2 * i]);
        correct += usize::from(pred == y[i] as usize);
    }
    correct as f32 / y.len() as f32
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn parse_fixture(text: &str) -> Result<Vec<Case>, String> {
    let mut lines = text.lines().enumerate();
    let Some((_, header)) = lines.next() else {
        return Err("empty fixture".to_string());
    };
    if header.trim() != "HYMEKO_SYNTHETIC_CHEBY_COMPARE_V1" {
        return Err(format!("bad header {header:?}"));
    }
    let Some((_, cases_line)) = lines.next() else {
        return Err("missing cases line".to_string());
    };
    let expected = parse_usize_line(cases_line, "cases")?;
    let mut cases = Vec::with_capacity(expected);

    while let Some((line_no, line)) = lines.next() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(task) = line.strip_prefix("case ") else {
            return Err(format!("line {}: expected case", line_no + 1));
        };
        let samples = parse_required_usize(&mut lines, "samples")?;
        let hidden = parse_required_usize(&mut lines, "hidden")?;
        let k = parse_required_usize(&mut lines, "k")?;
        let mut tensors = HashMap::new();
        loop {
            let Some((inner_no, inner)) = lines.next() else {
                return Err("unterminated case".to_string());
            };
            let inner = inner.trim();
            if inner == "endcase" {
                break;
            }
            let Some(rest) = inner.strip_prefix("tensor ") else {
                return Err(format!("line {}: expected tensor", inner_no + 1));
            };
            let mut parts = rest.split_whitespace();
            let name = parts.next().ok_or("missing tensor name")?.to_string();
            let len: usize = parts
                .next()
                .ok_or("missing tensor len")?
                .parse()
                .map_err(|_| "bad tensor len".to_string())?;
            let mut values = Vec::with_capacity(len);
            loop {
                let Some((value_no, value_line)) = lines.next() else {
                    return Err(format!("unterminated tensor {name}"));
                };
                let value_line = value_line.trim();
                if value_line == "endtensor" {
                    break;
                }
                for token in value_line.split_whitespace() {
                    values.push(
                        token
                            .parse::<f32>()
                            .map_err(|_| format!("line {}: bad float", value_no + 1))?,
                    );
                }
            }
            if values.len() != len {
                return Err(format!("tensor {name} length mismatch"));
            }
            tensors.insert(name, TensorBlock { values });
        }
        cases.push(Case {
            task: task.to_string(),
            samples,
            hidden,
            k,
            tensors,
        });
    }
    if cases.len() != expected {
        return Err(format!("expected {expected} cases, parsed {}", cases.len()));
    }
    Ok(cases)
}

fn parse_required_usize<'a, I>(lines: &mut I, key: &str) -> Result<usize, String>
where
    I: Iterator<Item = (usize, &'a str)>,
{
    let Some((line_no, line)) = lines.next() else {
        return Err(format!("missing {key}"));
    };
    parse_usize_line(line.trim(), key).map_err(|e| format!("line {}: {e}", line_no + 1))
}

fn parse_usize_line(line: &str, key: &str) -> Result<usize, String> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some(key) {
        return Err(format!("expected {key}, got {line:?}"));
    }
    parts
        .next()
        .ok_or_else(|| format!("missing value for {key}"))?
        .parse()
        .map_err(|_| format!("bad usize for {key}"))
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>, usize), io::Error> {
    let mut fixture = None;
    let mut out = None;
    let mut repeats = 300usize;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| io::Error::other(format!("missing value for {key}")))?;
        match key.as_str() {
            "--fixture" => fixture = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--repeats" => repeats = value.parse().map_err(|_| io::Error::other("bad repeats"))?,
            _ => return Err(io::Error::other(format!("unknown flag {key}"))),
        }
        i += 2;
    }
    Ok((
        fixture.ok_or_else(|| io::Error::other("missing --fixture"))?,
        out,
        repeats,
    ))
}

fn write_json(path: &PathBuf, rows: &[Report]) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(f, "  \"cases\": [")?;
    for (idx, row) in rows.iter().enumerate() {
        let comma = if idx + 1 == rows.len() { "" } else { "," };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"task\": \"{}\",", row.task)?;
        writeln!(f, "      \"max_abs_logits\": {:.9e},", row.max_abs_logits)?;
        writeln!(
            f,
            "      \"acc_random_weights\": {:.6},",
            row.acc_random_weights
        )?;
        writeln!(
            f,
            "      \"median_us_per_sample\": {:.9},",
            row.median_us_per_sample
        )?;
        writeln!(
            f,
            "      \"mean_us_per_sample\": {:.9},",
            row.mean_us_per_sample
        )?;
        writeln!(
            f,
            "      \"max_us_per_sample\": {:.9},",
            row.max_us_per_sample
        )?;
        writeln!(f, "      \"n_params\": {},", row.n_params)?;
        writeln!(f, "      \"param_bytes\": {}", row.param_bytes)?;
        writeln!(f, "    }}{comma}")?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}
