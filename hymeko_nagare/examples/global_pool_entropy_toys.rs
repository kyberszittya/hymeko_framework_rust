//! Nagare-side point-cloud toys with simultaneous global pooling and entropy feedback.
//!
//! The example intentionally uses Nagare's explicit `LinearLayer`, `linear_backward`,
//! and `AdamState` kernels instead of PyTorch/autograd. It is a small CPU harness,
//! not the final fused substrate kernel.

use std::{
    env,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
};

use hymeko_nagare::{AdamState, LinearLayer, adam_step, linear_backward, linear_forward};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    Moons,
    Rings,
    Xor,
}

impl Task {
    fn as_str(self) -> &'static str {
        match self {
            Self::Moons => "moons",
            Self::Rings => "rings",
            Self::Xor => "xor",
        }
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "moons" => Ok(Self::Moons),
            "rings" => Ok(Self::Rings),
            "xor" => Ok(Self::Xor),
            other => Err(format!("unknown task '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x: Vec<f32>,
    pub y: Vec<usize>,
    pub samples: usize,
    pub points: usize,
}

#[derive(Clone, Debug)]
pub struct SplitDataset {
    pub train: Dataset,
    pub test: Dataset,
}

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
            tasks: vec![Task::Moons, Task::Rings, Task::Xor],
            n_train: 192,
            n_test: 96,
            n_points: 48,
            hidden: 32,
            epochs: 80,
            batch_size: 32,
            lr: 2.0e-3,
            seed: 11,
            out: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub acc: f32,
    pub loss: f32,
    pub entropy: f32,
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub mean_us_per_sample: f64,
    pub median_us_per_sample: f64,
    pub min_us_per_sample: f64,
    pub max_us_per_sample: f64,
}

#[derive(Clone, Debug)]
pub struct ModelResult {
    pub train: Metrics,
    pub test: Metrics,
    pub timing: Timing,
    pub n_params: usize,
    pub param_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct TaskResult {
    pub task: Task,
    pub baseline: ModelResult,
    pub entropy_feedback: ModelResult,
}

pub fn make_dataset(
    task: Task,
    n_train: usize,
    n_test: usize,
    n_points: usize,
    seed: u64,
) -> SplitDataset {
    assert!(n_train > 0);
    assert!(n_test > 0);
    assert!(n_points > 1);
    let n = n_train + n_test;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(n);
    for sample in 0..n {
        let label = sample % 2;
        let mut x = vec![0.0; n_points * 2];
        match task {
            Task::Moons => sample_moons(label, n_points, &mut rng, &mut x),
            Task::Rings => sample_rings(label, n_points, &mut rng, &mut x),
            Task::Xor => sample_xor(label, n_points, &mut rng, &mut x),
        }
        rows.push((x, label));
    }
    rows.shuffle(&mut rng);

    let mut train_x = Vec::with_capacity(n_train * n_points * 2);
    let mut train_y = Vec::with_capacity(n_train);
    let mut test_x = Vec::with_capacity(n_test * n_points * 2);
    let mut test_y = Vec::with_capacity(n_test);
    for (i, (x, y)) in rows.into_iter().enumerate() {
        if i < n_train {
            train_x.extend(x);
            train_y.push(y);
        } else {
            test_x.extend(x);
            test_y.push(y);
        }
    }
    SplitDataset {
        train: Dataset {
            x: train_x,
            y: train_y,
            samples: n_train,
            points: n_points,
        },
        test: Dataset {
            x: test_x,
            y: test_y,
            samples: n_test,
            points: n_points,
        },
    }
}

fn sample_moons(label: usize, points: usize, rng: &mut StdRng, out: &mut [f32]) {
    for point in 0..points {
        let theta = rng.random::<f32>() * std::f32::consts::PI;
        let nx = normalish(rng) * 0.055;
        let ny = normalish(rng) * 0.055;
        let (x, y) = if label == 0 {
            (theta.cos(), theta.sin())
        } else {
            (1.0 - theta.cos(), 0.45 - theta.sin())
        };
        out[2 * point] = x + nx;
        out[2 * point + 1] = y + ny;
    }
}

fn sample_rings(label: usize, points: usize, rng: &mut StdRng, out: &mut [f32]) {
    for point in 0..points {
        let theta = rng.random::<f32>() * std::f32::consts::TAU;
        let radius = if label == 0 { 0.55 } else { 1.05 } + normalish(rng) * 0.055;
        out[2 * point] = radius * theta.cos();
        out[2 * point + 1] = radius * theta.sin();
    }
}

fn sample_xor(label: usize, points: usize, rng: &mut StdRng, out: &mut [f32]) {
    let centers = if label == 0 {
        [(-0.75, -0.75), (0.75, 0.75)]
    } else {
        [(-0.75, 0.75), (0.75, -0.75)]
    };
    for point in 0..points {
        let (cx, cy) = centers[rng.random_range(0..2)];
        out[2 * point] = cx + normalish(rng) * 0.12;
        out[2 * point + 1] = cy + normalish(rng) * 0.12;
    }
}

fn normalish(rng: &mut StdRng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..6 {
        sum += rng.random::<f32>() - 0.5;
    }
    sum * 0.816_496_6
}

#[derive(Clone, Debug)]
struct DenseRelu {
    layer: LinearLayer,
    w_state: AdamState,
    b_state: AdamState,
}

impl DenseRelu {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let layer = LinearLayer::new(in_dim, out_dim, seed);
        Self {
            w_state: AdamState::new(layer.w.len()),
            b_state: AdamState::new(layer.b.len()),
            layer,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let pre = linear_forward(&self.layer, x);
        let act = pre.iter().map(|v| v.max(0.0)).collect();
        (pre, act)
    }

    fn backward(&self, x: &[f32], pre: &[f32], grad: &[f32]) -> (Vec<f32>, LinearLayer) {
        let grad_pre: Vec<f32> = grad
            .iter()
            .zip(pre.iter())
            .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
            .collect();
        linear_backward(&self.layer, x, &grad_pre)
    }

    fn step(&mut self, grad: &LinearLayer, lr: f32) {
        adam_step(&mut self.layer.w, &grad.w, &mut self.w_state, lr);
        adam_step(&mut self.layer.b, &grad.b, &mut self.b_state, lr);
    }

    fn n_params(&self) -> usize {
        self.layer.w.len() + self.layer.b.len()
    }
}

#[derive(Clone, Debug)]
struct Dense {
    layer: LinearLayer,
    w_state: AdamState,
    b_state: AdamState,
}

impl Dense {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let layer = LinearLayer::new(in_dim, out_dim, seed);
        Self {
            w_state: AdamState::new(layer.w.len()),
            b_state: AdamState::new(layer.b.len()),
            layer,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        linear_forward(&self.layer, x)
    }

    fn backward(&self, x: &[f32], grad: &[f32]) -> (Vec<f32>, LinearLayer) {
        linear_backward(&self.layer, x, grad)
    }

    fn step(&mut self, grad: &LinearLayer, lr: f32) {
        adam_step(&mut self.layer.w, &grad.w, &mut self.w_state, lr);
        adam_step(&mut self.layer.b, &grad.b, &mut self.b_state, lr);
    }

    fn n_params(&self) -> usize {
        self.layer.w.len() + self.layer.b.len()
    }
}

#[derive(Clone, Debug)]
struct PoolCache {
    input: Vec<f32>,
    mean: Vec<f32>,
    std: Vec<f32>,
    max_index: Vec<usize>,
    batch: usize,
    points: usize,
    hidden: usize,
}

pub fn global_pool(input: &[f32], batch: usize, points: usize, hidden: usize) -> Vec<f32> {
    global_pool_with_cache(input, batch, points, hidden).0
}

fn global_pool_with_cache(
    input: &[f32],
    batch: usize,
    points: usize,
    hidden: usize,
) -> (Vec<f32>, PoolCache) {
    assert_eq!(input.len(), batch * points * hidden);
    let mut out = vec![0.0; batch * hidden * 3];
    let mut mean = vec![0.0; batch * hidden];
    let mut std = vec![0.0; batch * hidden];
    let mut max_index = vec![0; batch * hidden];
    let inv_points = 1.0 / points as f32;
    for b in 0..batch {
        for h in 0..hidden {
            let mut sum = 0.0;
            let mut max_val = f32::NEG_INFINITY;
            let mut argmax = 0;
            for p in 0..points {
                let v = input[(b * points + p) * hidden + h];
                sum += v;
                if v > max_val {
                    max_val = v;
                    argmax = p;
                }
            }
            let mu = sum * inv_points;
            let mut var = 0.0;
            for p in 0..points {
                let d = input[(b * points + p) * hidden + h] - mu;
                var += d * d;
            }
            let sigma = (var * inv_points + 1.0e-6).sqrt();
            mean[b * hidden + h] = mu;
            std[b * hidden + h] = sigma;
            max_index[b * hidden + h] = argmax;
            out[b * hidden * 3 + h] = mu;
            out[b * hidden * 3 + hidden + h] = sigma;
            out[b * hidden * 3 + 2 * hidden + h] = max_val;
        }
    }
    let cache = PoolCache {
        input: input.to_vec(),
        mean,
        std,
        max_index,
        batch,
        points,
        hidden,
    };
    (out, cache)
}

fn global_pool_backward(cache: &PoolCache, grad_pool: &[f32]) -> Vec<f32> {
    assert_eq!(grad_pool.len(), cache.batch * cache.hidden * 3);
    let mut grad = vec![0.0; cache.input.len()];
    let inv_points = 1.0 / cache.points as f32;
    for b in 0..cache.batch {
        for h in 0..cache.hidden {
            let base = b * cache.hidden * 3;
            let grad_mean = grad_pool[base + h];
            let grad_std = grad_pool[base + cache.hidden + h];
            let grad_max = grad_pool[base + 2 * cache.hidden + h];
            let mu = cache.mean[b * cache.hidden + h];
            let sigma = cache.std[b * cache.hidden + h].max(1.0e-6);
            let argmax = cache.max_index[b * cache.hidden + h];
            for p in 0..cache.points {
                let idx = (b * cache.points + p) * cache.hidden + h;
                let centered = cache.input[idx] - mu;
                grad[idx] += grad_mean * inv_points;
                grad[idx] += grad_std * centered * inv_points / sigma;
                if p == argmax {
                    grad[idx] += grad_max;
                }
            }
        }
    }
    grad
}

#[derive(Clone, Debug)]
struct CeOut {
    loss: f32,
    grad_logits: Vec<f32>,
    acc: f32,
    entropy: f32,
}

fn cross_entropy(logits: &[f32], labels: &[usize]) -> CeOut {
    assert_eq!(logits.len(), labels.len() * 2);
    let batch = labels.len();
    let mut grad = vec![0.0; logits.len()];
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut entropy = 0.0;
    for b in 0..batch {
        let a = logits[2 * b];
        let c = logits[2 * b + 1];
        let m = a.max(c);
        let ea = (a - m).exp();
        let ec = (c - m).exp();
        let z = ea + ec;
        let p0 = ea / z;
        let p1 = ec / z;
        let y = labels[b];
        loss += if y == 0 { -p0.ln() } else { -p1.ln() };
        correct += usize::from((p1 > p0) == (y == 1));
        entropy +=
            -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
        grad[2 * b] = (p0 - f32::from(y == 0)) / batch as f32;
        grad[2 * b + 1] = (p1 - f32::from(y == 1)) / batch as f32;
    }
    CeOut {
        loss: loss / batch as f32,
        grad_logits: grad,
        acc: correct as f32 / batch as f32,
        entropy: entropy / batch as f32,
    }
}

fn entropy_feature(logits: &[f32]) -> Vec<f32> {
    let batch = logits.len() / 2;
    let mut ent = vec![0.0; batch];
    for b in 0..batch {
        let a = logits[2 * b];
        let c = logits[2 * b + 1];
        let m = a.max(c);
        let ea = (a - m).exp();
        let ec = (c - m).exp();
        let p0 = ea / (ea + ec);
        let p1 = ec / (ea + ec);
        ent[b] = -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
    }
    ent
}

trait ToyModel {
    fn train_batch(&mut self, x: &[f32], y: &[usize], batch: usize, points: usize, lr: f32) -> f32;
    fn logits(&self, x: &[f32], batch: usize, points: usize) -> Vec<f32>;
    fn n_params(&self) -> usize;
}

#[derive(Clone, Debug)]
struct BaselineSetNet {
    embed: DenseRelu,
    head: Dense,
    hidden: usize,
}

impl BaselineSetNet {
    fn new(hidden: usize, seed: u64) -> Self {
        Self {
            embed: DenseRelu::new(2, hidden, seed),
            head: Dense::new(3 * hidden, 2, seed + 1),
            hidden,
        }
    }
}

impl ToyModel for BaselineSetNet {
    fn train_batch(&mut self, x: &[f32], y: &[usize], batch: usize, points: usize, lr: f32) -> f32 {
        let (embed_pre, h) = self.embed.forward(x);
        let (pooled, pool_cache) = global_pool_with_cache(&h, batch, points, self.hidden);
        let logits = self.head.forward(&pooled);
        let ce = cross_entropy(&logits, y);
        let (grad_pool, grad_head) = self.head.backward(&pooled, &ce.grad_logits);
        let grad_h = global_pool_backward(&pool_cache, &grad_pool);
        let (_grad_x, grad_embed) = self.embed.backward(x, &embed_pre, &grad_h);
        self.head.step(&grad_head, lr);
        self.embed.step(&grad_embed, lr);
        ce.loss
    }

    fn logits(&self, x: &[f32], batch: usize, points: usize) -> Vec<f32> {
        let (_, h) = self.embed.forward(x);
        self.head
            .forward(&global_pool(&h, batch, points, self.hidden))
    }

    fn n_params(&self) -> usize {
        self.embed.n_params() + self.head.n_params()
    }
}

#[derive(Clone, Debug)]
struct EntropyFeedbackSetNet {
    embed: DenseRelu,
    first: Dense,
    update: DenseRelu,
    head: Dense,
    hidden: usize,
}

impl EntropyFeedbackSetNet {
    fn new(hidden: usize, seed: u64) -> Self {
        Self {
            embed: DenseRelu::new(2, hidden, seed),
            first: Dense::new(3 * hidden, 2, seed + 1),
            update: DenseRelu::new(4 * hidden + 1, hidden, seed + 2),
            head: Dense::new(3 * hidden, 2, seed + 3),
            hidden,
        }
    }

    fn update_input(
        &self,
        h: &[f32],
        pooled: &[f32],
        entropy: &[f32],
        batch: usize,
        points: usize,
    ) -> Vec<f32> {
        let in_dim = 4 * self.hidden + 1;
        let mut out = vec![0.0; batch * points * in_dim];
        for b in 0..batch {
            for p in 0..points {
                let dst = (b * points + p) * in_dim;
                let src_h = (b * points + p) * self.hidden;
                out[dst..dst + self.hidden].copy_from_slice(&h[src_h..src_h + self.hidden]);
                out[dst + self.hidden..dst + 4 * self.hidden]
                    .copy_from_slice(&pooled[b * 3 * self.hidden..(b + 1) * 3 * self.hidden]);
                out[dst + 4 * self.hidden] = entropy[b];
            }
        }
        out
    }
}

impl ToyModel for EntropyFeedbackSetNet {
    fn train_batch(&mut self, x: &[f32], y: &[usize], batch: usize, points: usize, lr: f32) -> f32 {
        let (embed_pre, h) = self.embed.forward(x);
        let (pooled, pool_cache) = global_pool_with_cache(&h, batch, points, self.hidden);
        let logits_first = self.first.forward(&pooled);
        let first_ce = cross_entropy(&logits_first, y);
        let entropy = entropy_feature(&logits_first);
        let update_x = self.update_input(&h, &pooled, &entropy, batch, points);
        let (update_pre, h2) = self.update.forward(&update_x);
        let (pooled2, pool2_cache) = global_pool_with_cache(&h2, batch, points, self.hidden);
        let logits = self.head.forward(&pooled2);
        let final_ce = cross_entropy(&logits, y);

        let (grad_pool2, grad_head) = self.head.backward(&pooled2, &final_ce.grad_logits);
        let grad_h2 = global_pool_backward(&pool2_cache, &grad_pool2);
        let (grad_update_x, grad_update) = self.update.backward(&update_x, &update_pre, &grad_h2);

        let mut grad_h = vec![0.0; h.len()];
        let mut grad_pool = vec![0.0; pooled.len()];
        let update_in_dim = 4 * self.hidden + 1;
        for b in 0..batch {
            for p in 0..points {
                let src = (b * points + p) * update_in_dim;
                let dst_h = (b * points + p) * self.hidden;
                for k in 0..self.hidden {
                    grad_h[dst_h + k] += grad_update_x[src + k];
                }
                for k in 0..3 * self.hidden {
                    grad_pool[b * 3 * self.hidden + k] += grad_update_x[src + self.hidden + k];
                }
            }
        }

        let mut grad_first_logits = first_ce.grad_logits;
        for g in &mut grad_first_logits {
            *g *= 0.25;
        }
        let (grad_pool_first, grad_first) = self.first.backward(&pooled, &grad_first_logits);
        for (dst, src) in grad_pool.iter_mut().zip(grad_pool_first.iter()) {
            *dst += src;
        }
        let grad_h_from_pool = global_pool_backward(&pool_cache, &grad_pool);
        for (dst, src) in grad_h.iter_mut().zip(grad_h_from_pool.iter()) {
            *dst += src;
        }
        let (_grad_x, grad_embed) = self.embed.backward(x, &embed_pre, &grad_h);

        self.head.step(&grad_head, lr);
        self.update.step(&grad_update, lr);
        self.first.step(&grad_first, lr);
        self.embed.step(&grad_embed, lr);
        final_ce.loss + 0.25 * first_ce.loss
    }

    fn logits(&self, x: &[f32], batch: usize, points: usize) -> Vec<f32> {
        let (_, h) = self.embed.forward(x);
        let pooled = global_pool(&h, batch, points, self.hidden);
        let logits_first = self.first.forward(&pooled);
        let entropy = entropy_feature(&logits_first);
        let update_x = self.update_input(&h, &pooled, &entropy, batch, points);
        let (_, h2) = self.update.forward(&update_x);
        self.head
            .forward(&global_pool(&h2, batch, points, self.hidden))
    }

    fn n_params(&self) -> usize {
        self.embed.n_params()
            + self.first.n_params()
            + self.update.n_params()
            + self.head.n_params()
    }
}

fn train_model<M: ToyModel>(
    model: &mut M,
    train: &Dataset,
    test: &Dataset,
    cfg: &Config,
    seed: u64,
) -> ModelResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..train.samples).collect();
    for _ in 0..cfg.epochs {
        indices.shuffle(&mut rng);
        for chunk in indices.chunks(cfg.batch_size) {
            let (batch_x, batch_y) = gather_batch(train, chunk);
            model.train_batch(&batch_x, &batch_y, chunk.len(), train.points, cfg.lr);
        }
    }
    let train_metrics = evaluate(model, train);
    let test_metrics = evaluate(model, test);
    ModelResult {
        train: train_metrics,
        test: test_metrics,
        timing: forward_timing(model, test, 120),
        n_params: model.n_params(),
        param_bytes: model.n_params() * std::mem::size_of::<f32>(),
    }
}

fn gather_batch(data: &Dataset, indices: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let row = data.points * 2;
    let mut x = Vec::with_capacity(indices.len() * row);
    let mut y = Vec::with_capacity(indices.len());
    for &idx in indices {
        x.extend_from_slice(&data.x[idx * row..(idx + 1) * row]);
        y.push(data.y[idx]);
    }
    (x, y)
}

fn evaluate<M: ToyModel>(model: &M, data: &Dataset) -> Metrics {
    let logits = model.logits(&data.x, data.samples, data.points);
    let ce = cross_entropy(&logits, &data.y);
    Metrics {
        acc: ce.acc,
        loss: ce.loss,
        entropy: ce.entropy,
    }
}

fn forward_timing<M: ToyModel>(model: &M, data: &Dataset, repeats: usize) -> Timing {
    for _ in 0..20 {
        let _ = model.logits(&data.x, data.samples, data.points);
    }
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let _ = model.logits(&data.x, data.samples, data.points);
        times.push(start.elapsed().as_secs_f64() * 1.0e6 / data.samples as f64);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    Timing {
        mean_us_per_sample: mean,
        median_us_per_sample: times[times.len() / 2],
        min_us_per_sample: times[0],
        max_us_per_sample: times[times.len() - 1],
    }
}

pub fn run_suite(cfg: &Config) -> Vec<TaskResult> {
    assert!(cfg.n_train <= 4096);
    assert!(cfg.n_test <= 4096);
    assert!(cfg.n_points <= 512);
    cfg.tasks
        .iter()
        .enumerate()
        .map(|(i, &task)| {
            let split = make_dataset(
                task,
                cfg.n_train,
                cfg.n_test,
                cfg.n_points,
                cfg.seed + 100 * i as u64,
            );
            let mut baseline = BaselineSetNet::new(cfg.hidden, cfg.seed + 10 * i as u64);
            let mut feedback = EntropyFeedbackSetNet::new(cfg.hidden, cfg.seed + 10 * i as u64);
            TaskResult {
                task,
                baseline: train_model(
                    &mut baseline,
                    &split.train,
                    &split.test,
                    cfg,
                    cfg.seed + i as u64 + 1,
                ),
                entropy_feedback: train_model(
                    &mut feedback,
                    &split.train,
                    &split.test,
                    cfg,
                    cfg.seed + i as u64 + 11,
                ),
            }
        })
        .collect()
}

fn write_json(path: &PathBuf, cfg: &Config, rows: &[TaskResult]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(file, "  \"global_pool\": \"concat(mean,std,max)\",")?;
    writeln!(file, "  \"n_train\": {},", cfg.n_train)?;
    writeln!(file, "  \"n_test\": {},", cfg.n_test)?;
    writeln!(file, "  \"n_points\": {},", cfg.n_points)?;
    writeln!(file, "  \"hidden\": {},", cfg.hidden)?;
    writeln!(file, "  \"epochs\": {},", cfg.epochs)?;
    writeln!(file, "  \"tasks\": [")?;
    for (idx, row) in rows.iter().enumerate() {
        let comma = if idx + 1 == rows.len() { "" } else { "," };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", row.task.as_str())?;
        write_model_json(&mut file, "baseline", &row.baseline, true)?;
        write_model_json(&mut file, "entropy_feedback", &row.entropy_feedback, false)?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    Ok(())
}

fn write_model_json(
    file: &mut File,
    name: &str,
    model: &ModelResult,
    comma: bool,
) -> io::Result<()> {
    writeln!(file, "      \"{name}\": {{")?;
    writeln!(file, "        \"test_acc\": {:.6},", model.test.acc)?;
    writeln!(file, "        \"test_loss\": {:.6},", model.test.loss)?;
    writeln!(file, "        \"test_entropy\": {:.6},", model.test.entropy)?;
    writeln!(
        file,
        "        \"median_us_per_sample\": {:.6},",
        model.timing.median_us_per_sample
    )?;
    writeln!(
        file,
        "        \"mean_us_per_sample\": {:.6},",
        model.timing.mean_us_per_sample
    )?;
    writeln!(
        file,
        "        \"max_us_per_sample\": {:.6},",
        model.timing.max_us_per_sample
    )?;
    writeln!(file, "        \"n_params\": {},", model.n_params)?;
    writeln!(file, "        \"param_bytes\": {}", model.param_bytes)?;
    writeln!(file, "      }}{}", if comma { "," } else { "" })?;
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
                    .collect::<Result<Vec<_>, _>>()?;
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
    for row in &rows {
        println!(
            "{} baseline acc={:.3} median_us={:.2}; entropy_feedback acc={:.3} median_us={:.2}",
            row.task.as_str(),
            row.baseline.test.acc,
            row.baseline.timing.median_us_per_sample,
            row.entropy_feedback.test.acc,
            row.entropy_feedback.timing.median_us_per_sample
        );
    }
    if let Some(path) = &cfg.out {
        write_json(path, &cfg, &rows)?;
    }
    Ok(())
}
