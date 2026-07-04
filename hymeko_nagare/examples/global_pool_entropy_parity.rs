//! PyTorch/Nagare forward parity for global-pool entropy feedback.
//!
//! The fixture format is deliberately plain text so this harness does not add
//! JSON/parser dependencies to Nagare.

use std::{
    collections::HashMap,
    env, fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use hymeko_nagare::{linear_forward, LinearLayer};

#[derive(Clone, Debug)]
pub struct TensorBlock {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct FixtureCase {
    pub task: String,
    pub samples: usize,
    pub points: usize,
    pub hidden: usize,
    pub tensors: HashMap<String, TensorBlock>,
}

impl FixtureCase {
    fn tensor(&self, name: &str) -> &[f32] {
        &self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
            .values
    }

    fn layer(&self, prefix: &str, in_dim: usize, out_dim: usize) -> LinearLayer {
        let w = self.tensor(&format!("{prefix}_w")).to_vec();
        let b = self.tensor(&format!("{prefix}_b")).to_vec();
        assert_eq!(w.len(), in_dim * out_dim, "{prefix}_w shape mismatch");
        assert_eq!(b.len(), out_dim, "{prefix}_b shape mismatch");
        LinearLayer {
            w,
            b,
            in_dim,
            out_dim,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub cases: Vec<FixtureCase>,
}

#[derive(Clone, Debug)]
pub struct ForwardOutput {
    pub logits: Vec<f32>,
    pub logits_first: Vec<f32>,
    pub entropy: Vec<f32>,
    pub allocation_bytes: usize,
    pub allocation_count: usize,
}

#[derive(Clone, Debug)]
struct AllocCounter {
    bytes: usize,
    count: usize,
}

impl AllocCounter {
    fn add_f32(&mut self, len: usize) {
        self.bytes += len * std::mem::size_of::<f32>();
        self.count += 1;
    }
}

#[derive(Clone, Debug)]
pub struct CaseReport {
    pub task: String,
    pub max_abs_logits: f32,
    pub max_abs_logits_first: f32,
    pub max_abs_entropy: f32,
    pub timing: Timing,
    pub allocation_bytes_per_forward: usize,
    pub allocation_count_per_forward: usize,
    pub n_params: usize,
    pub param_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub mean_us_per_sample: f64,
    pub median_us_per_sample: f64,
    pub min_us_per_sample: f64,
    pub max_us_per_sample: f64,
}

pub fn parse_fixture_text(text: &str) -> Result<Fixture, String> {
    let mut lines = text.lines().enumerate().peekable();
    let Some((_, header)) = lines.next() else {
        return Err("empty fixture".to_string());
    };
    if header.trim() != "HYMEKO_GLOBAL_POOL_ENTROPY_PARITY_V1" {
        return Err(format!("bad fixture header {header:?}"));
    }
    let Some((_, cases_line)) = lines.next() else {
        return Err("missing cases line".to_string());
    };
    let expected_cases = parse_usize_line(cases_line, "cases")?;
    let mut cases = Vec::with_capacity(expected_cases);

    while let Some((line_no, line)) = lines.next() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(task) = line.strip_prefix("case ") else {
            return Err(format!("line {}: expected case, got {line:?}", line_no + 1));
        };
        let samples = parse_required_usize(&mut lines, "samples")?;
        let points = parse_required_usize(&mut lines, "points")?;
        let hidden = parse_required_usize(&mut lines, "hidden")?;
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
                return Err(format!(
                    "line {}: expected tensor or endcase, got {inner:?}",
                    inner_no + 1
                ));
            };
            let mut parts = rest.split_whitespace();
            let name = parts
                .next()
                .ok_or_else(|| format!("line {}: missing tensor name", inner_no + 1))?
                .to_string();
            let len: usize = parts
                .next()
                .ok_or_else(|| format!("line {}: missing tensor len", inner_no + 1))?
                .parse()
                .map_err(|_| format!("line {}: bad tensor len", inner_no + 1))?;
            let shape = parts
                .next()
                .unwrap_or("")
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| format!("bad shape item {s:?}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
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
                            .map_err(|_| format!("line {}: bad float {token:?}", value_no + 1))?,
                    );
                }
            }
            if values.len() != len {
                return Err(format!(
                    "tensor {name} length mismatch: header {len}, values {}",
                    values.len()
                ));
            }
            tensors.insert(name, TensorBlock { values, shape });
        }
        cases.push(FixtureCase {
            task: task.to_string(),
            samples,
            points,
            hidden,
            tensors,
        });
    }

    if cases.len() != expected_cases {
        return Err(format!(
            "case count mismatch: header {expected_cases}, parsed {}",
            cases.len()
        ));
    }
    Ok(Fixture { cases })
}

fn parse_required_usize<'a, I>(lines: &mut I, key: &str) -> Result<usize, String>
where
    I: Iterator<Item = (usize, &'a str)>,
{
    let Some((line_no, line)) = lines.next() else {
        return Err(format!("missing {key} line"));
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

pub fn forward_case(case: &FixtureCase) -> ForwardOutput {
    let mut alloc = AllocCounter { bytes: 0, count: 0 };
    let hidden = case.hidden;
    let batch = case.samples;
    let points = case.points;
    let embed = case.layer("embed", 2, hidden);
    let first = case.layer("first", 3 * hidden, 2);
    let update = case.layer("update", 4 * hidden + 1, hidden);
    let head = case.layer("head", 3 * hidden, 2);

    let embed_pre = linear_forward(&embed, case.tensor("x"));
    alloc.add_f32(embed_pre.len());
    let h = relu(&embed_pre);
    alloc.add_f32(h.len());
    let pooled = global_pool(&h, batch, points, hidden);
    alloc.add_f32(pooled.len());
    let logits_first = linear_forward(&first, &pooled);
    alloc.add_f32(logits_first.len());
    let entropy = entropy_feature(&logits_first);
    alloc.add_f32(entropy.len());
    let update_x = update_input(&h, &pooled, &entropy, batch, points, hidden);
    alloc.add_f32(update_x.len());
    let update_pre = linear_forward(&update, &update_x);
    alloc.add_f32(update_pre.len());
    let h2 = relu(&update_pre);
    alloc.add_f32(h2.len());
    let pooled2 = global_pool(&h2, batch, points, hidden);
    alloc.add_f32(pooled2.len());
    let logits = linear_forward(&head, &pooled2);
    alloc.add_f32(logits.len());

    ForwardOutput {
        logits,
        logits_first,
        entropy,
        allocation_bytes: alloc.bytes,
        allocation_count: alloc.count,
    }
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

pub fn global_pool(input: &[f32], batch: usize, points: usize, hidden: usize) -> Vec<f32> {
    assert_eq!(input.len(), batch * points * hidden);
    let mut out = vec![0.0; batch * hidden * 3];
    let inv_points = 1.0 / points as f32;
    for b in 0..batch {
        for h in 0..hidden {
            let mut sum = 0.0;
            let mut max_val = f32::NEG_INFINITY;
            for p in 0..points {
                let v = input[(b * points + p) * hidden + h];
                sum += v;
                max_val = max_val.max(v);
            }
            let mu = sum * inv_points;
            let mut var = 0.0;
            for p in 0..points {
                let d = input[(b * points + p) * hidden + h] - mu;
                var += d * d;
            }
            out[b * hidden * 3 + h] = mu;
            out[b * hidden * 3 + hidden + h] = (var * inv_points).sqrt();
            out[b * hidden * 3 + 2 * hidden + h] = max_val;
        }
    }
    out
}

fn entropy_feature(logits: &[f32]) -> Vec<f32> {
    let batch = logits.len() / 2;
    let mut out = vec![0.0; batch];
    for b in 0..batch {
        let a = logits[2 * b];
        let c = logits[2 * b + 1];
        let m = a.max(c);
        let ea = (a - m).exp();
        let ec = (c - m).exp();
        let p0 = ea / (ea + ec);
        let p1 = ec / (ea + ec);
        out[b] = -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
    }
    out
}

fn update_input(
    h: &[f32],
    pooled: &[f32],
    entropy: &[f32],
    batch: usize,
    points: usize,
    hidden: usize,
) -> Vec<f32> {
    let in_dim = 4 * hidden + 1;
    let mut out = vec![0.0; batch * points * in_dim];
    for b in 0..batch {
        for p in 0..points {
            let dst = (b * points + p) * in_dim;
            let src_h = (b * points + p) * hidden;
            out[dst..dst + hidden].copy_from_slice(&h[src_h..src_h + hidden]);
            out[dst + hidden..dst + 4 * hidden]
                .copy_from_slice(&pooled[b * 3 * hidden..(b + 1) * 3 * hidden]);
            out[dst + 4 * hidden] = entropy[b];
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn param_count(case: &FixtureCase) -> usize {
    [
        "embed_w", "embed_b", "first_w", "first_b", "update_w", "update_b", "head_w", "head_b",
    ]
    .iter()
    .map(|name| case.tensor(name).len())
    .sum()
}

fn time_case(case: &FixtureCase, repeats: usize) -> Timing {
    for _ in 0..20 {
        let _ = forward_case(case);
    }
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let _ = forward_case(case);
        times.push(start.elapsed().as_secs_f64() * 1.0e6 / case.samples as f64);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    Timing {
        mean_us_per_sample: times.iter().sum::<f64>() / times.len() as f64,
        median_us_per_sample: times[times.len() / 2],
        min_us_per_sample: times[0],
        max_us_per_sample: times[times.len() - 1],
    }
}

pub fn report_case(case: &FixtureCase, repeats: usize) -> CaseReport {
    let out = forward_case(case);
    let n_params = param_count(case);
    CaseReport {
        task: case.task.clone(),
        max_abs_logits: max_abs(&out.logits, case.tensor("logits")),
        max_abs_logits_first: max_abs(&out.logits_first, case.tensor("logits_first")),
        max_abs_entropy: max_abs(&out.entropy, case.tensor("entropy")),
        timing: time_case(case, repeats),
        allocation_bytes_per_forward: out.allocation_bytes,
        allocation_count_per_forward: out.allocation_count,
        n_params,
        param_bytes: n_params * std::mem::size_of::<f32>(),
    }
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>, usize), String> {
    let mut fixture = None;
    let mut out = None;
    let mut repeats = 300usize;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--fixture" => fixture = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--repeats" => repeats = value.parse().map_err(|_| "bad --repeats".to_string())?,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 2;
    }
    Ok((
        fixture.ok_or_else(|| "missing --fixture".to_string())?,
        out,
        repeats,
    ))
}

fn write_json(path: &PathBuf, reports: &[CaseReport]) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(f, "  \"cases\": [")?;
    for (idx, report) in reports.iter().enumerate() {
        let comma = if idx + 1 == reports.len() { "" } else { "," };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"task\": \"{}\",", report.task)?;
        writeln!(
            f,
            "      \"max_abs_logits\": {:.9e},",
            report.max_abs_logits
        )?;
        writeln!(
            f,
            "      \"max_abs_logits_first\": {:.9e},",
            report.max_abs_logits_first
        )?;
        writeln!(
            f,
            "      \"max_abs_entropy\": {:.9e},",
            report.max_abs_entropy
        )?;
        writeln!(
            f,
            "      \"median_us_per_sample\": {:.6},",
            report.timing.median_us_per_sample
        )?;
        writeln!(
            f,
            "      \"mean_us_per_sample\": {:.6},",
            report.timing.mean_us_per_sample
        )?;
        writeln!(
            f,
            "      \"max_us_per_sample\": {:.6},",
            report.timing.max_us_per_sample
        )?;
        writeln!(
            f,
            "      \"allocation_bytes_per_forward\": {},",
            report.allocation_bytes_per_forward
        )?;
        writeln!(
            f,
            "      \"allocation_count_per_forward\": {},",
            report.allocation_count_per_forward
        )?;
        writeln!(f, "      \"n_params\": {},", report.n_params)?;
        writeln!(f, "      \"param_bytes\": {}", report.param_bytes)?;
        writeln!(f, "    }}{comma}")?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (fixture_path, out, repeats) = parse_args().map_err(io::Error::other)?;
    let text = fs::read_to_string(fixture_path)?;
    let fixture = parse_fixture_text(&text).map_err(io::Error::other)?;
    let reports: Vec<_> = fixture
        .cases
        .iter()
        .map(|case| report_case(case, repeats))
        .collect();
    for report in &reports {
        println!(
            "{} max_abs={:.3e} median_us={:.2} alloc_bytes={}",
            report.task,
            report.max_abs_logits,
            report.timing.median_us_per_sample,
            report.allocation_bytes_per_forward
        );
    }
    if let Some(path) = out {
        write_json(&path, &reports)?;
    }
    Ok(())
}
