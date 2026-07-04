//! Compare entropy-pool local learning with the backprop-like holonomy toy.

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

use holonomy_entropy_toys::{
    clifford_probability_error, quaternion_periodic_features, run_suite as run_backprop_suite,
    Config as BackpropConfig, Dataset,
};
pub use holonomy_entropy_toys::{make_dataset, Task};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

const VERTEX_FEATURES: usize = 7;
const STRUCTURAL_FEATURES: usize = 4 * VERTEX_FEATURES;
const LOCAL_FEATURES: usize = STRUCTURAL_FEATURES + 1;
const PROJECTION_RANK: usize = 6;
const PROJECTION_ALPHA: f32 = 0.72;

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
pub struct Metrics {
    pub acc: f32,
    pub loss: f32,
    pub entropy: f32,
    pub clifford_error: f32,
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub median_us_per_sample: f64,
    pub mean_us_per_sample: f64,
    pub max_us_per_sample: f64,
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
pub enum GateMode {
    Entropy,
    Constant,
    Projection,
}

impl GateMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Entropy => "entropy",
            Self::Constant => "constant",
            Self::Projection => "projection",
        }
    }
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

#[derive(Clone, Debug)]
pub struct EntropyPoolLocalLearner {
    w: Vec<f32>,
    b: [f32; 2],
    gate_mode: GateMode,
    projection_basis: [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK],
}

impl EntropyPoolLocalLearner {
    pub fn new(seed: u64) -> Self {
        Self::new_with_gate(seed, GateMode::Entropy)
    }

    pub fn new_with_gate(seed: u64, gate_mode: GateMode) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = vec![0.0; LOCAL_FEATURES * 2];
        for value in &mut w {
            *value = (rng.random::<f32>() * 2.0 - 1.0) * 0.01;
        }
        Self {
            w,
            b: [0.0, 0.0],
            gate_mode,
            projection_basis: default_holonomy_projection_basis(),
        }
    }

    pub fn logits_one(&self, phi: &[f32]) -> [f32; 2] {
        assert_eq!(phi.len(), LOCAL_FEATURES);
        let mut out = self.b;
        for (i, &v) in phi.iter().enumerate() {
            out[0] += v * self.w[2 * i];
            out[1] += v * self.w[2 * i + 1];
        }
        out
    }

    pub fn predict_dataset(&self, data: &Dataset) -> Vec<f32> {
        let structural = structural_pool_features(data);
        let mut logits = vec![0.0; data.samples * 2];
        for sample in 0..data.samples {
            let phi = entropy_augmented_phi(
                self,
                &structural[sample * STRUCTURAL_FEATURES..(sample + 1) * STRUCTURAL_FEATURES],
            );
            let row = self.logits_one(&phi);
            logits[2 * sample] = row[0];
            logits[2 * sample + 1] = row[1];
        }
        logits
    }

    pub fn train(&mut self, data: &Dataset, epochs: usize, batch_size: usize, lr: f32, seed: u64) {
        let structural = structural_pool_features(data);
        if self.gate_mode == GateMode::Projection {
            self.projection_basis = learn_holonomy_projection_basis(&structural, &data.y);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..data.samples).collect();
        for _ in 0..epochs {
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(batch_size) {
                for &sample in chunk {
                    let base = sample * STRUCTURAL_FEATURES;
                    let phi =
                        entropy_augmented_phi(self, &structural[base..base + STRUCTURAL_FEATURES]);
                    let logits = self.logits_one(&phi);
                    let (p0, p1) = softmax2(logits[0], logits[1]);
                    let y0 = f32::from(data.y[sample] == 0);
                    let y1 = f32::from(data.y[sample] == 1);
                    let entropy = entropy2(p0, p1);
                    let gate = match self.gate_mode {
                        GateMode::Entropy => 0.25 + entropy,
                        GateMode::Constant | GateMode::Projection => 1.0,
                    };
                    let delta = [y0 - p0, y1 - p1];
                    for (i, &value) in phi.iter().enumerate() {
                        self.w[2 * i] += lr * gate * value * delta[0];
                        self.w[2 * i + 1] += lr * gate * value * delta[1];
                    }
                    self.b[0] += lr * gate * delta[0];
                    self.b[1] += lr * gate * delta[1];
                }
            }
        }
    }

    pub fn n_params(&self) -> usize {
        let projection_params = if self.gate_mode == GateMode::Projection {
            STRUCTURAL_FEATURES * PROJECTION_RANK
        } else {
            0
        };
        self.w.len() + self.b.len() + projection_params
    }
}

pub fn structural_pool_features(data: &Dataset) -> Vec<f32> {
    let lifted = quaternion_periodic_features(&data.x, data.samples, data.points);
    let mut out = vec![0.0; data.samples * STRUCTURAL_FEATURES];
    let inv_points = 1.0 / data.points as f32;
    for sample in 0..data.samples {
        for channel in 0..VERTEX_FEATURES {
            let mut sum = 0.0;
            let mut max_value = f32::NEG_INFINITY;
            let mut positive = 0usize;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                sum += value;
                max_value = max_value.max(value);
                positive += usize::from(value >= 0.0);
            }
            let mean = sum * inv_points;
            let mut var = 0.0;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                let d = value - mean;
                var += d * d;
            }
            let pos = positive as f32 * inv_points;
            let sign_entropy = binary_entropy(pos);
            let base = sample * STRUCTURAL_FEATURES;
            out[base + channel] = mean;
            out[base + VERTEX_FEATURES + channel] = (var * inv_points + 1.0e-6).sqrt();
            out[base + 2 * VERTEX_FEATURES + channel] = max_value;
            out[base + 3 * VERTEX_FEATURES + channel] = sign_entropy;
        }
    }
    out
}

pub fn evaluate_local(model: &EntropyPoolLocalLearner, data: &Dataset) -> Metrics {
    let logits = model.predict_dataset(data);
    let ce = cross_entropy(&logits, &data.y);
    Metrics {
        acc: ce.acc,
        loss: ce.loss,
        entropy: ce.entropy,
        clifford_error: clifford_probability_error(&logits, &data.y),
    }
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

fn entropy_augmented_phi(model: &EntropyPoolLocalLearner, structural: &[f32]) -> Vec<f32> {
    let mut warm = vec![0.0; LOCAL_FEATURES];
    warm[..STRUCTURAL_FEATURES].copy_from_slice(structural);
    warm[STRUCTURAL_FEATURES] = 1.0;
    let logits = model.logits_one(&warm);
    let (p0, p1) = softmax2(logits[0], logits[1]);
    warm[STRUCTURAL_FEATURES] = match model.gate_mode {
        GateMode::Entropy => entropy2(p0, p1),
        GateMode::Constant | GateMode::Projection => 1.0,
    };
    if model.gate_mode == GateMode::Projection {
        project_onto_holonomy_subspace(&mut warm, &model.projection_basis);
    }
    warm
}

pub fn project_onto_holonomy_axis(phi: &mut [f32]) {
    let basis = default_holonomy_projection_basis();
    project_onto_holonomy_subspace(phi, &basis);
}

pub fn project_onto_holonomy_subspace(
    phi: &mut [f32],
    basis: &[[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK],
) {
    assert_eq!(phi.len(), LOCAL_FEATURES);
    let mut projected = [0.0f32; STRUCTURAL_FEATURES];
    for axis in basis {
        let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
        if norm2 <= 1.0e-12 {
            continue;
        }
        let dot = phi[..STRUCTURAL_FEATURES]
            .iter()
            .zip(axis.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>();
        let scale = dot / norm2;
        for (dst, &axis_value) in projected.iter_mut().zip(axis.iter()) {
            *dst += scale * axis_value;
        }
    }
    for i in 0..STRUCTURAL_FEATURES {
        phi[i] = PROJECTION_ALPHA * projected[i] + (1.0 - PROJECTION_ALPHA) * phi[i];
    }
}

pub fn learn_holonomy_projection_basis(
    structural: &[f32],
    labels: &[usize],
) -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
    assert_eq!(structural.len(), labels.len() * STRUCTURAL_FEATURES);
    let mut class_sum = [[0.0f32; STRUCTURAL_FEATURES]; 2];
    let mut class_count = [0usize; 2];
    for (sample, &label) in labels.iter().enumerate() {
        let label = label.min(1);
        class_count[label] += 1;
        let base = sample * STRUCTURAL_FEATURES;
        for i in 0..STRUCTURAL_FEATURES {
            class_sum[label][i] += structural[base + i];
        }
    }
    for label in 0..2 {
        let scale = 1.0 / class_count[label].max(1) as f32;
        for value in &mut class_sum[label] {
            *value *= scale;
        }
    }
    let mut candidates = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK + 2];
    for i in 0..STRUCTURAL_FEATURES {
        candidates[0][i] = class_sum[1][i] - class_sum[0][i];
        candidates[1][i] = 0.5 * (class_sum[0][i] + class_sum[1][i]);
    }
    let class_delta = candidates[0];
    copy_channel_group(&mut candidates[2], &class_delta, &[0, 1, 2]);
    copy_channel_group(&mut candidates[3], &class_delta, &[3, 4]);
    copy_channel_group(&mut candidates[4], &class_delta, &[5, 6]);
    for channel in 0..VERTEX_FEATURES {
        candidates[5][VERTEX_FEATURES + channel] =
            class_sum[0][VERTEX_FEATURES + channel] + class_sum[1][VERTEX_FEATURES + channel];
        candidates[5][3 * VERTEX_FEATURES + channel] = class_sum[0][3 * VERTEX_FEATURES + channel]
            + class_sum[1][3 * VERTEX_FEATURES + channel];
    }
    let defaults = default_holonomy_projection_basis();
    candidates[6] = defaults[2];
    candidates[7] = defaults[4];
    orthonormalize_candidates(&candidates)
}

fn copy_channel_group(dst: &mut [f32; STRUCTURAL_FEATURES], src: &[f32], channels: &[usize]) {
    for &channel in channels {
        dst[channel] = src[channel];
        dst[VERTEX_FEATURES + channel] = src[VERTEX_FEATURES + channel];
        dst[2 * VERTEX_FEATURES + channel] = src[2 * VERTEX_FEATURES + channel];
        dst[3 * VERTEX_FEATURES + channel] = src[3 * VERTEX_FEATURES + channel];
    }
}

fn default_holonomy_projection_basis() -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
    let mut candidates = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK];
    copy_channel_group(&mut candidates[0], &[1.0; STRUCTURAL_FEATURES], &[0, 1, 2]);
    copy_channel_group(&mut candidates[1], &[1.0; STRUCTURAL_FEATURES], &[3, 4]);
    copy_channel_group(&mut candidates[2], &[1.0; STRUCTURAL_FEATURES], &[5, 6]);
    for channel in 0..VERTEX_FEATURES {
        candidates[3][VERTEX_FEATURES + channel] = 1.0;
        candidates[4][2 * VERTEX_FEATURES + channel] = 1.0;
        candidates[5][3 * VERTEX_FEATURES + channel] = 1.0;
    }
    orthonormalize_candidates(&candidates)
}

fn orthonormalize_candidates<const N: usize>(
    candidates: &[[f32; STRUCTURAL_FEATURES]; N],
) -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
    let mut basis = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK];
    let mut rank = 0usize;
    for candidate in candidates {
        if rank == PROJECTION_RANK {
            break;
        }
        let mut vector = *candidate;
        for _ in 0..2 {
            for axis in basis.iter().take(rank) {
                remove_axis_component(&mut vector, axis);
            }
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= 1.0e-7 {
            continue;
        }
        for (dst, value) in basis[rank].iter_mut().zip(vector.iter()) {
            *dst = *value / norm;
        }
        rank += 1;
    }
    basis
}

fn remove_axis_component(vector: &mut [f32; STRUCTURAL_FEATURES], axis: &[f32]) {
    let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
    if norm2 <= 1.0e-12 {
        return;
    }
    let dot = vector
        .iter()
        .zip(axis.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f32>();
    let scale = dot / norm2;
    for (value, &axis_value) in vector.iter_mut().zip(axis.iter()) {
        *value -= scale * axis_value;
    }
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

pub fn corrupt_dataset(data: &Dataset, noise_std: f32, missing_rate: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = data.clone();
    for sample in 0..out.samples {
        for point in 0..out.points {
            let base = (sample * out.points + point) * 2;
            if rng.random::<f32>() < missing_rate {
                out.x[base] = 0.0;
                out.x[base + 1] = 0.0;
            } else if noise_std > 0.0 {
                out.x[base] += normalish(&mut rng) * noise_std;
                out.x[base + 1] += normalish(&mut rng) * noise_std;
            }
        }
    }
    out
}

fn forward_timing(model: &EntropyPoolLocalLearner, data: &Dataset, repeats: usize) -> Timing {
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

#[derive(Clone, Debug)]
struct CeOut {
    loss: f32,
    acc: f32,
    entropy: f32,
}

fn cross_entropy(logits: &[f32], labels: &[usize]) -> CeOut {
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut entropy = 0.0;
    for i in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * i], logits[2 * i + 1]);
        loss += if labels[i] == 0 { -p0.ln() } else { -p1.ln() };
        correct += usize::from((p1 > p0) == (labels[i] == 1));
        entropy += entropy2(p0, p1);
    }
    CeOut {
        loss: loss / labels.len() as f32,
        acc: correct as f32 / labels.len() as f32,
        entropy: entropy / labels.len() as f32,
    }
}

fn binary_entropy(p: f32) -> f32 {
    let q = 1.0 - p;
    -(p * p.max(1.0e-12).ln() + q * q.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

fn normalish(rng: &mut StdRng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..6 {
        sum += rng.random::<f32>() - 0.5;
    }
    sum * 0.816_496_6
}

fn entropy2(p0: f32, p1: f32) -> f32 {
    -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    (ea / (ea + eb), eb / (ea + eb))
}

fn task_name(task: Task) -> &'static str {
    match task {
        Task::Moons => "moons",
        Task::Spiral => "spiral",
        Task::Xor => "xor",
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
        writeln!(file, "      \"task\": \"{}\",", task_name(row.task))?;
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
        writeln!(file, "      \"task\": \"{}\",", task_name(row.task))?;
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
            task_name(row.task),
            row.local.acc,
            row.local_timing.median_us_per_sample,
            row.backprop_acc,
            row.backprop_timing_median_us
        );
    }
    for row in &stress_rows {
        println!(
            "{} {} entropy acc={:.3}; constant acc={:.3}; projection acc={:.3}",
            task_name(row.task),
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
