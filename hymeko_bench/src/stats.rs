//! Aggregation helpers for the camera-ready benchmarks: robust central
//! tendency (median / IQR) and a log–log power-law fit with R² so the scaling
//! exponent is *reported*, not asserted (SMC 2026 Task 2 requires the fitted
//! exponent and its R²).

/// Median of a slice of samples. Returns `f64::NAN` for an empty slice.
#[must_use]
pub fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The `q`-quantile (0..=1) by linear interpolation on the sorted samples.
#[must_use]
pub fn quantile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

/// Inter-quartile range (Q3 − Q1).
#[must_use]
pub fn iqr(samples: &[f64]) -> f64 {
    quantile(samples, 0.75) - quantile(samples, 0.25)
}

/// Ordinary-least-squares fit of `log(y) = a + b·log(x)` returning
/// `(exponent b, r_squared)`. Points with non-positive `x` or `y` are skipped.
/// Returns `(NAN, NAN)` if fewer than two usable points remain.
#[must_use]
pub fn loglog_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let pts: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(x, y)| **x > 0.0 && **y > 0.0)
        .map(|(x, y)| (x.ln(), y.ln()))
        .collect();
    let n = pts.len();
    if n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let nf = n as f64;
    let sx: f64 = pts.iter().map(|(x, _)| *x).sum();
    let sy: f64 = pts.iter().map(|(_, y)| *y).sum();
    let sxx: f64 = pts.iter().map(|(x, _)| x * x).sum();
    let sxy: f64 = pts.iter().map(|(x, y)| x * y).sum();
    let denom = nf * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return (f64::NAN, f64::NAN);
    }
    let slope = (nf * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / nf;
    // R² on the log-transformed fit.
    let mean_y = sy / nf;
    let ss_tot: f64 = pts.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = pts
        .iter()
        .map(|(x, y)| {
            let pred = intercept + slope * x;
            (y - pred).powi(2)
        })
        .sum();
    let r2 = if ss_tot.abs() < f64::EPSILON {
        f64::NAN
    } else {
        1.0 - ss_res / ss_tot
    };
    (slope, r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn iqr_basic() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        // Q1≈25.75, Q3≈75.25 with linear interpolation → IQR≈49.5.
        let got = iqr(&data);
        assert!((got - 49.5).abs() < 1.0, "iqr {got}");
    }

    #[test]
    fn loglog_recovers_linear_exponent() {
        // y = 3 x^1 exactly → exponent 1, R² 1.
        let xs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x).collect();
        let (b, r2) = loglog_fit(&xs, &ys);
        assert!((b - 1.0).abs() < 1e-9, "exponent {b}");
        assert!((r2 - 1.0).abs() < 1e-9, "r2 {r2}");
    }

    #[test]
    fn loglog_recovers_quadratic_exponent() {
        let xs: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x * x).collect();
        let (b, r2) = loglog_fit(&xs, &ys);
        assert!((b - 2.0).abs() < 1e-9, "exponent {b}");
        assert!((r2 - 1.0).abs() < 1e-9, "r2 {r2}");
    }

    #[test]
    fn loglog_too_few_points() {
        let (b, r2) = loglog_fit(&[1.0], &[1.0]);
        assert!(b.is_nan() && r2.is_nan());
    }
}
