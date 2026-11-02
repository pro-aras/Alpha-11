//! Non-parametric empirical distributions, Mann-Whitney U / rank-biserial effect
//! sizes, deterministic permutation tests, and individuality divergence metrics.

/// Default deterministic permutation count for two-sample p-values.
pub const DEFAULT_PERMUTATIONS: usize = 1_000;

/// Default deterministic seed for permutation resampling.
pub const DEFAULT_PERMUTATION_SEED: u64 = 0xA110_B5E2_5EED_0001;

/// Deterministic `SplitMix64` pseudo-random generator for reproducible shuffles
/// and Monte Carlo permutation tests (`std`-only, zero external dependencies).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Creates a new deterministic PRNG from a 64-bit seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the exact 64-bit internal state word for deterministic checkpoint serialization.
    #[must_use]
    pub const fn raw_state(&self) -> u64 {
        self.state
    }

    /// Reconstructs a `DeterministicRng` from an exact 64-bit state word.
    #[must_use]
    pub const fn from_raw_state(state: u64) -> Self {
        Self { state }
    }

    /// Generates the next deterministic 64-bit unsigned integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Generates an unbiased index in `0..upper_exclusive` using rejection sampling.
    /// Returns `0` if `upper_exclusive <= 1`.
    pub fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        let bound = upper_exclusive as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let candidate = self.next_u64();
            if candidate >= threshold {
                return (candidate % bound) as usize;
            }
        }
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Performs an in-place Fisher-Yates shuffle on `slice`.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

/// Non-parametric quantile and moment summary of an empirical sample.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmpiricalDistribution {
    pub count: usize,
    pub non_finite_filtered: usize,
    pub min: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub iqr: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub cv: f64,
}

impl EmpiricalDistribution {
    /// Summarizes a slice of `f64` observations, filtering and counting non-finite values.
    #[must_use]
    pub fn from_f64_slice(values: &[f64]) -> Self {
        let mut finite = Vec::with_capacity(values.len());
        let mut non_finite_filtered = 0usize;
        for &v in values {
            if v.is_finite() {
                finite.push(v);
            } else {
                non_finite_filtered += 1;
            }
        }
        if finite.is_empty() {
            return Self {
                non_finite_filtered,
                ..Self::default()
            };
        }

        finite.sort_by(f64::total_cmp);
        let count = finite.len();
        let min = finite[0];
        let p10 = quantile_sorted(&finite, 0.10);
        let p25 = quantile_sorted(&finite, 0.25);
        let p50 = quantile_sorted(&finite, 0.50);
        let p75 = quantile_sorted(&finite, 0.75);
        let p90 = quantile_sorted(&finite, 0.90);
        let p95 = quantile_sorted(&finite, 0.95);
        let p99 = quantile_sorted(&finite, 0.99);
        let max = finite[count - 1];
        let iqr = p75 - p25;

        let mean = finite.iter().sum::<f64>() / (count as f64);
        let std_dev = if count >= 2 {
            let var_sum: f64 = finite
                .iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum();
            (var_sum / ((count - 1) as f64)).sqrt()
        } else {
            0.0
        };
        let cv = if mean.abs() > 1e-12 {
            std_dev / mean.abs()
        } else {
            0.0
        };

        Self {
            count,
            non_finite_filtered,
            min,
            p10,
            p25,
            p50,
            p75,
            p90,
            p95,
            p99,
            max,
            iqr,
            mean,
            std_dev,
            cv,
        }
    }

    /// Summarizes a slice of `f32` observations.
    #[must_use]
    pub fn from_f32_slice(values: &[f32]) -> Self {
        let converted: Vec<f64> = values.iter().map(|&x| f64::from(x)).collect();
        Self::from_f64_slice(&converted)
    }

    /// Summarizes a slice of `u64` observations.
    #[must_use]
    pub fn from_u64_slice(values: &[u64]) -> Self {
        let converted: Vec<f64> = values.iter().map(|&x| x as f64).collect();
        Self::from_f64_slice(&converted)
    }
}

/// Computes linear-interpolated quantile `q in [0.0, 1.0]` (Hyndman-Fan Type 7) on a sorted slice.
#[must_use]
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let clamped = q.clamp(0.0, 1.0);
    let pos = clamped * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Two-sample non-parametric Mann-Whitney U comparison between sample `A` and sample `B`.
#[derive(Clone, Debug, PartialEq)]
pub struct MannWhitneyComparison {
    pub n_a: usize,
    pub n_b: usize,
    /// Directional U statistic for sample A (`#{a_i > b_j} + 0.5 * #{a_i == b_j}`).
    pub u_a: f64,
    /// Directional U statistic for sample B (`n_a * n_b - u_a`).
    pub u_b: f64,
    /// Symmetric Mann-Whitney U statistic (`min(u_a, u_b)`).
    pub u_stat: f64,
    /// Rank-biserial correlation effect size in `[-1.0, 1.0]` (`+1.0` when all `A > B`).
    pub rank_biserial: f64,
    /// Difference in sample medians (`median(A) - median(B)`).
    pub median_diff: f64,
    /// Tie-corrected normal-approximation signed z-score.
    pub z_score: f64,
    /// Two-sided tie- and continuity-corrected normal-approximation p-value.
    pub p_normal_approx: f64,
    /// Two-sided Monte Carlo permutation p-value (`(exceedances + 1) / (permutations + 1)`).
    pub p_permutation: f64,
    pub permutation_exceedances: usize,
    pub permutations: usize,
}

impl Default for MannWhitneyComparison {
    fn default() -> Self {
        Self {
            n_a: 0,
            n_b: 0,
            u_a: 0.0,
            u_b: 0.0,
            u_stat: 0.0,
            rank_biserial: 0.0,
            median_diff: 0.0,
            z_score: 0.0,
            p_normal_approx: 1.0,
            p_permutation: 1.0,
            permutation_exceedances: 0,
            permutations: 0,
        }
    }
}

/// Runs a two-sample Mann-Whitney U comparison with `1,000` deterministic permutations.
#[must_use]
pub fn mann_whitney_u(sample_a: &[f64], sample_b: &[f64]) -> MannWhitneyComparison {
    mann_whitney_u_with_permutations(
        sample_a,
        sample_b,
        DEFAULT_PERMUTATIONS,
        DEFAULT_PERMUTATION_SEED,
    )
}

/// Runs a two-sample Mann-Whitney U comparison with explicit permutation count and RNG seed.
#[must_use]
pub fn mann_whitney_u_with_permutations(
    sample_a: &[f64],
    sample_b: &[f64],
    permutations: usize,
    seed: u64,
) -> MannWhitneyComparison {
    let mut clean_a: Vec<f64> = sample_a.iter().copied().filter(|x| x.is_finite()).collect();
    let mut clean_b: Vec<f64> = sample_b.iter().copied().filter(|x| x.is_finite()).collect();
    let n_a = clean_a.len();
    let n_b = clean_b.len();
    if n_a == 0 || n_b == 0 {
        return MannWhitneyComparison {
            n_a,
            n_b,
            permutations,
            ..MannWhitneyComparison::default()
        };
    }

    clean_a.sort_by(f64::total_cmp);
    clean_b.sort_by(f64::total_cmp);
    let median_diff = quantile_sorted(&clean_a, 0.50) - quantile_sorted(&clean_b, 0.50);

    let total_n = n_a + n_b;
    let mut pooled: Vec<(f64, bool)> = Vec::with_capacity(total_n);
    for &x in &clean_a {
        pooled.push((x, true));
    }
    for &y in &clean_b {
        pooled.push((y, false));
    }
    pooled.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));

    let mut ranks = vec![0.0_f64; total_n];
    let mut tie_sum = 0.0_f64;
    let mut i = 0usize;
    while i < total_n {
        let mut j = i + 1;
        while j < total_n && pooled[j].0.to_bits() == pooled[i].0.to_bits() {
            j += 1;
        }
        let run_len = j - i;
        let midrank = ((i + 1) + j) as f64 * 0.5;
        for slot in &mut ranks[i..j] {
            *slot = midrank;
        }
        if run_len > 1 {
            let t = run_len as f64;
            tie_sum += t * (t * t - 1.0);
        }
        i = j;
    }

    let rank_sum_a: f64 = pooled
        .iter()
        .zip(ranks.iter())
        .filter_map(|((_, is_a), &r)| if *is_a { Some(r) } else { None })
        .sum();

    let min_rank_sum_a = (n_a as f64) * ((n_a + 1) as f64) * 0.5;
    let pair_count = (n_a * n_b) as f64;
    let u_a = (rank_sum_a - min_rank_sum_a).clamp(0.0, pair_count);
    let u_b = (pair_count - u_a).clamp(0.0, pair_count);
    let u_stat = u_a.min(u_b);
    let rank_biserial = ((2.0 * u_a) / pair_count - 1.0).clamp(-1.0, 1.0);

    let mu_u = 0.5 * pair_count;
    let n_f = total_n as f64;
    let var_u = if total_n > 1 {
        (pair_count / 12.0) * (n_f + 1.0 - tie_sum / (n_f * (n_f - 1.0)))
    } else {
        0.0
    };

    let obs_dev = (u_a - mu_u).abs();
    let (z_score, p_normal_approx) = if var_u <= 1e-15 {
        if obs_dev <= 1e-12 {
            (0.0, 1.0)
        } else {
            let sign = if u_a > mu_u { 1.0 } else { -1.0 };
            (sign * f64::INFINITY, 0.0)
        }
    } else {
        let sigma = var_u.sqrt();
        let z = (u_a - mu_u) / sigma;
        let z_cc = (obs_dev - 0.5).max(0.0) / sigma;
        let p = erfc_approx(z_cc / std::f64::consts::SQRT_2).clamp(0.0, 1.0);
        (z, p)
    };

    let (permutation_exceedances, p_permutation) = if permutations == 0 {
        (0, p_normal_approx)
    } else if var_u <= 1e-15 && obs_dev <= 1e-12 {
        (permutations, 1.0)
    } else {
        let mut perm_ranks = ranks;
        let mut rng = DeterministicRng::new(seed);
        let mut exceedances = 0usize;
        for _ in 0..permutations {
            for k in 0..n_a {
                let swap_idx = k + rng.next_usize(total_n - k);
                perm_ranks.swap(k, swap_idx);
            }
            let r_a_perm: f64 = perm_ranks[..n_a].iter().sum();
            let u_a_perm = r_a_perm - min_rank_sum_a;
            if (u_a_perm - mu_u).abs() + 1e-12 >= obs_dev {
                exceedances += 1;
            }
        }
        let p_perm = ((exceedances + 1) as f64) / ((permutations + 1) as f64);
        (exceedances, p_perm.clamp(0.0, 1.0))
    };

    MannWhitneyComparison {
        n_a,
        n_b,
        u_a,
        u_b,
        u_stat,
        rank_biserial,
        median_diff,
        z_score,
        p_normal_approx,
        p_permutation,
        permutation_exceedances,
        permutations,
    }
}

/// Abramowitz & Stegun 7.1.26 rational approximation to `erfc(x)` (`max error < 1.5e-7`).
#[must_use]
pub fn erfc_approx(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    let val = poly * (-z * z).exp();
    if x >= 0.0 { val } else { 2.0 - val }
}

/// Pairwise cosine distance `1 - cos(a, b)` in `[0.0, 2.0]` (`[0.0, 1.0]` for non-negative profiles).
///
/// Two all-zero profiles have distance `0.0`; one zero and one non-zero profile have distance `1.0`.
#[must_use]
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let max_len = a.len().max(b.len());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for idx in 0..max_len {
        let va = a.get(idx).copied().unwrap_or(0.0);
        let vb = b.get(idx).copied().unwrap_or(0.0);
        if va.is_finite() && vb.is_finite() {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
    }
    if norm_a <= 1e-24 && norm_b <= 1e-24 {
        return 0.0;
    }
    if norm_a <= 1e-24 || norm_b <= 1e-24 {
        return 1.0;
    }
    let cos = (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0);
    (1.0 - cos).clamp(0.0, 2.0)
}

/// Computes `(mean_pairwise_cosine_distance, max_pairwise_cosine_distance)` across `profiles`.
#[must_use]
pub fn mean_pairwise_cosine_distance(profiles: &[Vec<f64>]) -> (f64, f64) {
    if profiles.len() < 2 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    let mut max_d = 0.0_f64;
    let mut pairs = 0usize;
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            let d = cosine_distance(&profiles[i], &profiles[j]);
            sum += d;
            if d > max_d {
                max_d = d;
            }
            pairs += 1;
        }
    }
    (sum / (pairs as f64), max_d)
}

/// Base-2 Jensen-Shannon divergence `JSD(P || Q)` bounded in `[0.0, 1.0]` bits.
///
/// Non-negative weight vectors `p` and `q` are normalized into probability distributions.
/// Two zero-mass vectors return `0.0`; one zero-mass and one positive-mass vector return `1.0`.
#[must_use]
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    let len = p.len().max(q.len());
    if len == 0 {
        return 0.0;
    }
    let sum_p: f64 = p
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .sum();
    let sum_q: f64 = q
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .sum();
    if sum_p <= 1e-24 && sum_q <= 1e-24 {
        return 0.0;
    }
    if sum_p <= 1e-24 || sum_q <= 1e-24 {
        return 1.0;
    }

    let mut jsd = 0.0_f64;
    for idx in 0..len {
        let wp = p.get(idx).copied().unwrap_or(0.0);
        let wq = q.get(idx).copied().unwrap_or(0.0);
        let pi = if wp.is_finite() && wp > 0.0 {
            wp / sum_p
        } else {
            0.0
        };
        let qi = if wq.is_finite() && wq > 0.0 {
            wq / sum_q
        } else {
            0.0
        };
        let mi = 0.5 * (pi + qi);
        if pi > 0.0 && mi > 0.0 {
            jsd += 0.5 * pi * (pi / mi).log2();
        }
        if qi > 0.0 && mi > 0.0 {
            jsd += 0.5 * qi * (qi / mi).log2();
        }
    }
    jsd.clamp(0.0, 1.0)
}

/// Computes `(mean_pairwise_jsd, max_pairwise_jsd)` across non-negative histograms.
#[must_use]
pub fn mean_pairwise_jsd(distributions: &[Vec<f64>]) -> (f64, f64) {
    if distributions.len() < 2 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    let mut max_d = 0.0_f64;
    let mut pairs = 0usize;
    for i in 0..distributions.len() {
        for j in (i + 1)..distributions.len() {
            let d = jensen_shannon_divergence(&distributions[i], &distributions[j]);
            sum += d;
            if d > max_d {
                max_d = d;
            }
            pairs += 1;
        }
    }
    (sum / (pairs as f64), max_d)
}

/// Computes `(entropy_bits, normalized_entropy_0_to_1)` for a non-negative histogram.
#[must_use]
pub fn shannon_entropy_bits(weights: &[f64]) -> (f64, f64) {
    if weights.is_empty() {
        return (0.0, 0.0);
    }
    let total: f64 = weights
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .sum();
    if total <= 1e-24 {
        return (0.0, 0.0);
    }
    let mut h = 0.0_f64;
    for &w in weights {
        if w.is_finite() && w > 0.0 {
            let p = w / total;
            h -= p * p.log2();
        }
    }
    let max_h = if weights.len() > 1 {
        (weights.len() as f64).log2()
    } else {
        0.0
    };
    let norm_h = if max_h > 0.0 {
        (h / max_h).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (h.max(0.0), norm_h)
}

/// Multi-axis Individuality / Constrained-Randomness Divergence summary across seeds or twins.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndividualityDivergence {
    pub individual_count: usize,
    /// Inter-individual coefficient of variation (`std_dev / |mean|`) of primary scalar outcome.
    pub inter_individual_cv: f64,
    /// Mean pairwise cosine distance of sensory pore-channel conductance profiles (`[0, 1]`).
    pub mean_pairwise_pore_cosine_distance: f64,
    /// Maximum pairwise cosine distance of sensory pore-channel conductance profiles (`[0, 1]`).
    pub max_pairwise_pore_cosine_distance: f64,
    /// Mean pairwise Jensen-Shannon divergence (bits, `[0, 1]`) of `tau` distributions.
    pub mean_pairwise_tau_jsd: f64,
    /// Maximum pairwise Jensen-Shannon divergence (bits, `[0, 1]`) of `tau` distributions.
    pub max_pairwise_tau_jsd: f64,
    /// Mean intra-individual behavioral trajectory Shannon entropy (bits).
    pub mean_trajectory_entropy_bits: f64,
    /// Mean intra-individual normalized trajectory entropy (`[0, 1]`).
    pub normalized_trajectory_entropy: f64,
    /// Multi-individual trajectory JSD (`H_2(mean_P) - mean(H_2(P_i))` in bits, `>= 0`).
    /// Separates genuine niche/persona divergence from identical-policy random noise.
    pub trajectory_divergence_bits: f64,
    /// Bounded `[0.0, 1.0]` composite individuality index averaging the four divergence axes.
    pub composite_individuality_index: f64,
}

/// Computes the complete `IndividualityDivergence` profile across a cohort of individuals.
#[must_use]
pub fn compute_individuality_divergence(
    scalar_outcomes: &[f64],
    pore_profiles: &[Vec<f64>],
    tau_distributions: &[Vec<f64>],
    trajectory_histograms: &[Vec<f64>],
) -> IndividualityDivergence {
    let individual_count = scalar_outcomes
        .len()
        .max(pore_profiles.len())
        .max(tau_distributions.len())
        .max(trajectory_histograms.len());
    if individual_count == 0 {
        return IndividualityDivergence::default();
    }

    let scalar_dist = EmpiricalDistribution::from_f64_slice(scalar_outcomes);
    let inter_individual_cv = scalar_dist.cv;

    let (mean_pore_cos, max_pore_cos) = mean_pairwise_cosine_distance(pore_profiles);
    let (mean_tau_jsd, max_tau_jsd) = mean_pairwise_jsd(tau_distributions);

    let (mean_traj_h, norm_traj_h, traj_div_bits, max_traj_bins) =
        compute_trajectory_entropy_metrics(trajectory_histograms);

    let cv_squashed = inter_individual_cv / (1.0 + inter_individual_cv);
    let norm_traj_div = if max_traj_bins > 1 && trajectory_histograms.len() > 1 {
        let denom = (trajectory_histograms.len().min(max_traj_bins) as f64).log2();
        if denom > 0.0 {
            (traj_div_bits / denom).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let traj_component = 0.5 * norm_traj_h.clamp(0.0, 1.0) + 0.5 * norm_traj_div;
    let composite = 0.25
        * (cv_squashed
            + mean_pore_cos.clamp(0.0, 1.0)
            + mean_tau_jsd.clamp(0.0, 1.0)
            + traj_component)
            .clamp(0.0, 4.0);

    IndividualityDivergence {
        individual_count,
        inter_individual_cv,
        mean_pairwise_pore_cosine_distance: mean_pore_cos,
        max_pairwise_pore_cosine_distance: max_pore_cos,
        mean_pairwise_tau_jsd: mean_tau_jsd,
        max_pairwise_tau_jsd: max_tau_jsd,
        mean_trajectory_entropy_bits: mean_traj_h,
        normalized_trajectory_entropy: norm_traj_h,
        trajectory_divergence_bits: traj_div_bits,
        composite_individuality_index: composite,
    }
}

fn compute_trajectory_entropy_metrics(histograms: &[Vec<f64>]) -> (f64, f64, f64, usize) {
    if histograms.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let max_bins = histograms.iter().map(Vec::len).max().unwrap_or(0);
    if max_bins == 0 {
        return (0.0, 0.0, 0.0, 0);
    }

    let mut sum_h = 0.0_f64;
    let mut sum_norm_h = 0.0_f64;
    let mut valid_individuals = 0usize;
    let mut pooled = vec![0.0_f64; max_bins];

    for hist in histograms {
        let total: f64 = hist
            .iter()
            .copied()
            .filter(|x| x.is_finite() && *x > 0.0)
            .sum();
        if total > 1e-24 {
            let (h, nh) = shannon_entropy_bits(hist);
            sum_h += h;
            sum_norm_h += nh;
            valid_individuals += 1;
            for (idx, &w) in hist.iter().enumerate() {
                if w.is_finite() && w > 0.0 {
                    pooled[idx] += w / total;
                }
            }
        }
    }

    if valid_individuals == 0 {
        return (0.0, 0.0, 0.0, max_bins);
    }

    let inv_m = 1.0 / (valid_individuals as f64);
    for slot in &mut pooled {
        *slot *= inv_m;
    }
    let mean_h = sum_h * inv_m;
    let mean_norm_h = sum_norm_h * inv_m;
    let (pooled_h, _) = shannon_entropy_bits(&pooled);
    let traj_div_bits = (pooled_h - mean_h).max(0.0);

    (mean_h, mean_norm_h, traj_div_bits, max_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_distribution_computes_exact_quantiles_and_moments() {
        let sample = [50.0, 10.0, 30.0, 20.0, 40.0, f64::NAN];
        let dist = EmpiricalDistribution::from_f64_slice(&sample);
        assert_eq!(dist.count, 5);
        assert_eq!(dist.non_finite_filtered, 1);
        assert!((dist.min - 10.0).abs() < 1e-12);
        assert!((dist.p25 - 20.0).abs() < 1e-12);
        assert!((dist.p50 - 30.0).abs() < 1e-12);
        assert!((dist.p75 - 40.0).abs() < 1e-12);
        assert!((dist.iqr - 20.0).abs() < 1e-12);
        assert!((dist.max - 50.0).abs() < 1e-12);
        assert!((dist.mean - 30.0).abs() < 1e-12);
        assert!((dist.std_dev - 250.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mann_whitney_strictly_separated_gives_plus_or_minus_one_effect_size() {
        let high = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let low = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let cmp_pos = mann_whitney_u(&high, &low);
        assert_eq!(cmp_pos.n_a, 8);
        assert_eq!(cmp_pos.n_b, 8);
        assert!((cmp_pos.u_a - 64.0).abs() < 1e-12);
        assert!((cmp_pos.u_b - 0.0).abs() < 1e-12);
        assert!((cmp_pos.u_stat - 0.0).abs() < 1e-12);
        assert!((cmp_pos.rank_biserial - 1.0).abs() < 1e-12);
        assert!(cmp_pos.p_permutation < 0.005);
        assert!(cmp_pos.p_normal_approx < 0.005);

        let cmp_neg = mann_whitney_u(&low, &high);
        assert!((cmp_neg.rank_biserial + 1.0).abs() < 1e-12);
        assert!((cmp_neg.u_stat - 0.0).abs() < 1e-12);
    }

    #[test]
    fn mann_whitney_identical_samples_gives_zero_effect_size_and_unit_p() {
        let a = [2.5, 4.0, 4.0, 7.2, 9.1, 12.0];
        let b = [2.5, 4.0, 4.0, 7.2, 9.1, 12.0];
        let cmp = mann_whitney_u(&a, &b);
        assert!((cmp.u_a - 18.0).abs() < 1e-12);
        assert!((cmp.u_b - 18.0).abs() < 1e-12);
        assert!(cmp.rank_biserial.abs() < 1e-12);
        assert!((cmp.p_normal_approx - 1.0).abs() < 1e-6);
        assert!((cmp.p_permutation - 1.0).abs() < 1e-6);
    }

    #[test]
    fn individuality_divergence_separates_clones_from_specialized_twins() {
        let identical_scalars = [10.0, 10.0, 10.0, 10.0];
        let identical_pores = vec![vec![1.0, 0.0, 0.0]; 4];
        let identical_taus = vec![vec![0.8, 0.2, 0.0]; 4];
        let identical_trajs = vec![vec![10.0, 0.0, 0.0, 0.0]; 4];

        let clone_div = compute_individuality_divergence(
            &identical_scalars,
            &identical_pores,
            &identical_taus,
            &identical_trajs,
        );
        assert!(clone_div.inter_individual_cv.abs() < 1e-12);
        assert!(clone_div.mean_pairwise_pore_cosine_distance.abs() < 1e-12);
        assert!(clone_div.mean_pairwise_tau_jsd.abs() < 1e-12);
        assert!(clone_div.trajectory_divergence_bits.abs() < 1e-12);
        assert!(clone_div.composite_individuality_index.abs() < 1e-12);

        let diverse_scalars = [4.0, 12.0, 22.0, 35.0];
        let diverse_pores = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let diverse_taus = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let diverse_trajs = vec![
            vec![20.0, 1.0, 0.0, 0.0],
            vec![0.0, 20.0, 1.0, 0.0],
            vec![0.0, 0.0, 20.0, 1.0],
            vec![1.0, 0.0, 0.0, 20.0],
        ];
        let spec_div = compute_individuality_divergence(
            &diverse_scalars,
            &diverse_pores,
            &diverse_taus,
            &diverse_trajs,
        );
        assert!(spec_div.inter_individual_cv > 0.5);
        assert!((spec_div.mean_pairwise_pore_cosine_distance - 1.0).abs() < 1e-12);
        assert!((spec_div.mean_pairwise_tau_jsd - 1.0).abs() < 1e-12);
        assert!(spec_div.trajectory_divergence_bits > 1.5);
        assert!(spec_div.composite_individuality_index > 0.65);
    }
}
