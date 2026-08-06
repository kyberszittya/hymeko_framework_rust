//! Compare entropy-pool local learning with the backprop-like holonomy toy.
//!
//! Orchestration shell only: the learner, projection, pooling, features,
//! datasets, and metrics live in `hymeko_nagare::holonomy`; the
//! backprop-like baseline lives in the `holonomy_entropy_toys` example.

use std::{
    env,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
};

#[allow(dead_code)]
#[path = "holonomy_entropy_toys.rs"]
mod holonomy_entropy_toys;

use holonomy_entropy_toys::{Config as BackpropConfig, run_suite as run_backprop_suite};
use hymeko_nagare::holonomy::{
    Dataset, EntropyPoolLocalLearner, GateMode, Metrics, Task, Timing, corrupt_dataset,
    evaluate_local, make_dataset,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub tasks: Vec<Task>,
    pub n_train: usize,
    pub n_test: usize,
    pub n_points: usize,
    pub hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub seed: u64,
    pub out: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tasks: vec![Task::Moons, Task::Spiral, Task::Xor],
            n_train: 192,
            n_test: 96,
            n_points: 32,
            hidden: 24,
            epochs: 50,
            batch_size: 32,
            lr: 0.05,
            seed: 53,
            out: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompareRow {
    pub task: Task,
    pub local: Metrics,
    pub local_timing: Timing,
    pub backprop_acc: f32,
    pub backprop_loss: f32,
    pub backprop_timing_median_us: f64,
    pub local_params: usize,
    pub backprop_params: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StressKind {
    Clean,
    Noisy,
    Missing,
    FewShotNoisyMissing,
}

impl StressKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Noisy => "noisy",
            Self::Missing => "missing",
            Self::FewShotNoisyMissing => "fewshot_noisy_missing",
        }
    }

    fn train_samples(self, cfg: &Config) -> usize {
        match self {
            Self::FewShotNoisyMissing => cfg.n_train.min(32),
            _ => cfg.n_train,
        }
    }

    fn noise_std(self) -> f32 {
        match self {
            Self::Noisy => 0.18,
            Self::FewShotNoisyMissing => 0.22,
            _ => 0.0,
        }
    }

    fn missing_rate(self) -> f32 {
        match self {
            Self::Missing => 0.35,
            Self::FewShotNoisyMissing => 0.45,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StressRow {
    pub task: Task,
    pub stress: StressKind,
    pub entropy_metrics: Metrics,
    pub constant_metrics: Metrics,
    pub projection_metrics: Metrics,
    pub entropy_timing: Timing,
    pub constant_timing: Timing,
    pub projection_timing: Timing,
    pub entropy_params: usize,
    pub constant_params: usize,
    pub projection_params: usize,
}

pub fn run_suite(cfg: &Config) -> Vec<CompareRow> {
    let bp_cfg = BackpropConfig {
        tasks: cfg.tasks.clone(),
        n_train: cfg.n_train,
        n_test: cfg.n_test,
        n_points: cfg.n_points,
        hidden: cfg.hidden,
        epochs: cfg.epochs,
        batch_size: cfg.batch_size,
        lr: 0.002,
        seed: cfg.seed,
        out: None,
    };
    let backprop = run_backprop_suite(&bp_cfg);
    cfg.tasks
        .iter()
        .enumerate()
        .map(|(idx, &task)| {
            let train = make_dataset(task, cfg.n_train, cfg.n_points, cfg.seed + idx as u64 * 100);
            let test = make_dataset(
                task,
                cfg.n_test,
                cfg.n_points,
                cfg.seed + idx as u64 * 100 + 1,
            );
            let mut local = EntropyPoolLocalLearner::new(cfg.seed + idx as u64 * 17);
            local.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + idx as u64,
            );
            let local_metrics = evaluate_local(&local, &test);
            let local_timing = forward_timing(&local, &test, 120);
            let bp = &backprop[idx];
            CompareRow {
                task,
                local: local_metrics,
                local_timing,
                backprop_acc: bp.test.acc,
                backprop_loss: bp.test.loss,
                backprop_timing_median_us: bp.timing.median_us_per_sample,
                local_params: local.n_params(),
                backprop_params: bp.n_params,
            }
        })
        .collect()
}

pub fn run_stress_ablation(cfg: &Config) -> Vec<StressRow> {
    let stresses = [
        StressKind::Clean,
        StressKind::Noisy,
        StressKind::Missing,
        StressKind::FewShotNoisyMissing,
    ];
    let mut rows = Vec::new();
    for (task_idx, &task) in cfg.tasks.iter().enumerate() {
        for (stress_idx, &stress) in stresses.iter().enumerate() {
            let train = make_dataset(
                task,
                stress.train_samples(cfg),
                cfg.n_points,
                cfg.seed + task_idx as u64 * 100 + stress_idx as u64 * 1_000,
            );
            let test = make_dataset(
                task,
                cfg.n_test,
                cfg.n_points,
                cfg.seed + task_idx as u64 * 100 + stress_idx as u64 * 1_000 + 1,
            );
            let train = corrupt_dataset(
                &train,
                stress.noise_std(),
                stress.missing_rate(),
                cfg.seed + 20_000 + stress_idx as u64,
            );
            let test = corrupt_dataset(
                &test,
                stress.noise_std(),
                stress.missing_rate(),
                cfg.seed + 30_000 + stress_idx as u64,
            );
            let mut entropy_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Entropy,
            );
            let mut constant_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Constant,
            );
            let mut projection_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Projection,
            );
            entropy_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            constant_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            projection_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            rows.push(StressRow {
                task,
                stress,
                entropy_metrics: evaluate_local(&entropy_model, &test),
                constant_metrics: evaluate_local(&constant_model, &test),
                projection_metrics: evaluate_local(&projection_model, &test),
                entropy_timing: forward_timing(&entropy_model, &test, 80),
                constant_timing: forward_timing(&constant_model, &test, 80),
                projection_timing: forward_timing(&projection_model, &test, 80),
                entropy_params: entropy_model.n_params(),
                constant_params: constant_model.n_params(),
                projection_params: projection_model.n_params(),
            });
        }
    }
    rows
}

fn forward_timing(model: &EntropyPoolLocalLearner, data: &Dataset, repeats: usize) -> Timing {
    // Kept local (not the shared closure protocol) so the timed body stays
    // byte-identical with the 2026-07-02 measurements this example anchors.
    for _ in 0..20 {
        let _ = model.predict_dataset(data);
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let _ = model.predict_dataset(data);
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / data.samples as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Timing {
        median_us_per_sample: values[values.len() / 2],
        mean_us_per_sample: values.iter().sum::<f64>() / values.len() as f64,
        max_us_per_sample: values[values.len() - 1],
    }
}

fn write_json(
    path: &PathBuf,
    cfg: &Config,
    rows: &[CompareRow],
    stress_rows: &[StressRow],
) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(
        file,
        "  \"comparison\": \"entropy_pool_local_vs_backprop_like_holonomy\","
    )?;
    writeln!(file, "  \"n_train\": {},", cfg.n_train)?;
    writeln!(file, "  \"n_test\": {},", cfg.n_test)?;
    writeln!(file, "  \"n_points\": {},", cfg.n_points)?;
    writeln!(file, "  \"hidden\": {},", cfg.hidden)?;
    writeln!(file, "  \"epochs\": {},", cfg.epochs)?;
    writeln!(file, "  \"rows\": [")?;
    for (idx, row) in rows.iter().enumerate() {
        let comma = if idx + 1 == rows.len() { "" } else { "," };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", row.task.as_str())?;
        writeln!(file, "      \"local_acc\": {:.6},", row.local.acc)?;
        writeln!(file, "      \"local_loss\": {:.6},", row.local.loss)?;
        writeln!(file, "      \"local_entropy\": {:.6},", row.local.entropy)?;
        writeln!(
            file,
            "      \"local_clifford_error\": {:.6},",
            row.local.clifford_error
        )?;
        writeln!(
            file,
            "      \"local_median_us_per_sample\": {:.6},",
            row.local_timing.median_us_per_sample
        )?;
        writeln!(
            file,
            "      \"local_mean_us_per_sample\": {:.6},",
            row.local_timing.mean_us_per_sample
        )?;
        writeln!(
            file,
            "      \"local_max_us_per_sample\": {:.6},",
            row.local_timing.max_us_per_sample
        )?;
        writeln!(file, "      \"backprop_acc\": {:.6},", row.backprop_acc)?;
        writeln!(file, "      \"backprop_loss\": {:.6},", row.backprop_loss)?;
        writeln!(
            file,
            "      \"backprop_median_us_per_sample\": {:.6},",
            row.backprop_timing_median_us
        )?;
        writeln!(file, "      \"local_params\": {},", row.local_params)?;
        writeln!(file, "      \"backprop_params\": {}", row.backprop_params)?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ],")?;
    writeln!(file, "  \"stress_rows\": [")?;
    for (idx, row) in stress_rows.iter().enumerate() {
        let comma = if idx + 1 == stress_rows.len() {
            ""
        } else {
            ","
        };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", row.task.as_str())?;
        writeln!(file, "      \"stress\": \"{}\",", row.stress.as_str())?;
        writeln!(
            file,
            "      \"entropy_gate\": \"{}\",",
            GateMode::Entropy.as_str()
        )?;
        writeln!(
            file,
            "      \"entropy_acc\": {:.6},",
            row.entropy_metrics.acc
        )?;
        writeln!(
            file,
            "      \"entropy_loss\": {:.6},",
            row.entropy_metrics.loss
        )?;
        writeln!(
            file,
            "      \"entropy_clifford_error\": {:.6},",
            row.entropy_metrics.clifford_error
        )?;
        writeln!(
            file,
            "      \"entropy_median_us_per_sample\": {:.6},",
            row.entropy_timing.median_us_per_sample
        )?;
        writeln!(
            file,
            "      \"constant_gate\": \"{}\",",
            GateMode::Constant.as_str()
        )?;
        writeln!(
            file,
            "      \"constant_acc\": {:.6},",
            row.constant_metrics.acc
        )?;
        writeln!(
            file,
            "      \"constant_loss\": {:.6},",
            row.constant_metrics.loss
        )?;
        writeln!(
            file,
            "      \"constant_clifford_error\": {:.6},",
            row.constant_metrics.clifford_error
        )?;
        writeln!(
            file,
            "      \"constant_median_us_per_sample\": {:.6},",
            row.constant_timing.median_us_per_sample
        )?;
        writeln!(
            file,
            "      \"projection_gate\": \"{}\",",
            GateMode::Projection.as_str()
        )?;
        writeln!(
            file,
            "      \"projection_acc\": {:.6},",
            row.projection_metrics.acc
        )?;
        writeln!(
            file,
            "      \"projection_loss\": {:.6},",
            row.projection_metrics.loss
        )?;
        writeln!(
            file,
            "      \"projection_clifford_error\": {:.6},",
            row.projection_metrics.clifford_error
        )?;
        writeln!(
            file,
            "      \"projection_median_us_per_sample\": {:.6},",
            row.projection_timing.median_us_per_sample
        )?;
        writeln!(file, "      \"entropy_params\": {},", row.entropy_params)?;
        writeln!(file, "      \"constant_params\": {},", row.constant_params)?;
        writeln!(
            file,
            "      \"projection_params\": {}",
            row.projection_params
        )?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--tasks" => {
                cfg.tasks = value
                    .split(',')
                    .map(Task::from_str)
                    .collect::<Result<Vec<_>, _>>()?
            }
            "--n-train" => cfg.n_train = value.parse().map_err(|_| "bad --n-train".to_string())?,
            "--n-test" => cfg.n_test = value.parse().map_err(|_| "bad --n-test".to_string())?,
            "--n-points" => {
                cfg.n_points = value.parse().map_err(|_| "bad --n-points".to_string())?
            }
            "--hidden" => cfg.hidden = value.parse().map_err(|_| "bad --hidden".to_string())?,
            "--epochs" => cfg.epochs = value.parse().map_err(|_| "bad --epochs".to_string())?,
            "--batch-size" => {
                cfg.batch_size = value.parse().map_err(|_| "bad --batch-size".to_string())?
            }
            "--lr" => cfg.lr = value.parse().map_err(|_| "bad --lr".to_string())?,
            "--seed" => cfg.seed = value.parse().map_err(|_| "bad --seed".to_string())?,
            "--out" => cfg.out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {key}")),
        }
        i += 2;
    }
    Ok(cfg)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args().map_err(io::Error::other)?;
    let rows = run_suite(&cfg);
    let stress_rows = run_stress_ablation(&cfg);
    for row in &rows {
        println!(
            "{} local acc={:.3} median_us={:.2}; backprop acc={:.3} median_us={:.2}",
            row.task.as_str(),
            row.local.acc,
            row.local_timing.median_us_per_sample,
            row.backprop_acc,
            row.backprop_timing_median_us
        );
    }
    for row in &stress_rows {
        println!(
            "{} {} entropy acc={:.3}; constant acc={:.3}; projection acc={:.3}",
            row.task.as_str(),
            row.stress.as_str(),
            row.entropy_metrics.acc,
            row.constant_metrics.acc,
            row.projection_metrics.acc
        );
    }
    if let Some(path) = &cfg.out {
        write_json(path, &cfg, &rows, &stress_rows)?;
    }
    Ok(())
}
