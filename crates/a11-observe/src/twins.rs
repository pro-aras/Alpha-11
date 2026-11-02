//! Multi-seed 6-control scientific twin observation engine, sense-organ band
//! conductance analyzer, mid-run environmental perturbation hooks, and
//! genetically identical twin divergence observer.

use std::ops::Range;

use a11_medium::{
    ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError, WorldMedium,
};

use crate::stats::{
    DeterministicRng, EmpiricalDistribution, IndividualityDivergence, MannWhitneyComparison,
    compute_individuality_divergence, mann_whitney_u,
};

/// Default 12 deterministic seeds (`>= 10` required by causal engineering protocol).
pub const DEFAULT_TWELVE_SEEDS: [u64; 12] = [
    101, 202, 303, 404, 505, 606, 707, 808, 909, 1_010, 1_111, 1_212,
];

const ACTION_SHUFFLE_SEED_SALT: u64 = 0x5A17_A110_C0DE_7788;
pub(crate) const POST_LIFE_WORLD_SEED_SALT: u64 = 0xE7A1_9057_11FE_4422;
const FRESH_RESTORE_SEED_SALT: u64 = 0xFACE_B00C_1234_5678;
const TEMPORAL_SHUFFLE_BUFFER_LEN: usize = 8;
const TAU_HISTOGRAM_BINS: usize = 10;

/// Physical role of an ambient channel range, known ONLY to the external observer
/// and NEVER revealed to `OrganismEngine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandRole {
    /// Predictable passive or movement-coupled physical flux (e.g., 2D photonic or thermal gradient).
    StructuredPhysical,
    /// Time-of-flight reflected wave channels returning `\Delta` ticks after actuator emission.
    ActiveEcho,
    /// High-variance un-predictable exogenous static ("Noisy TV" distractor).
    NoisyTvDistractor,
    /// Any auxiliary or unlabelled channel group.
    Unclassified,
}

impl BandRole {
    /// Human-readable role identifier for reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StructuredPhysical => "structured_physical",
            Self::ActiveEcho => "active_echo",
            Self::NoisyTvDistractor => "noisy_tv_distractor",
            Self::Unclassified => "unclassified",
        }
    }
}

/// Observer-side descriptor mapping a slice of `ambient_in` indices to a physical phenomenon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandSpec {
    pub label: String,
    pub range: Range<u32>,
    pub role: BandRole,
}

impl BandSpec {
    /// Creates a new observer-side channel band descriptor.
    #[must_use]
    pub fn new(label: impl Into<String>, range: Range<u32>, role: BandRole) -> Self {
        Self {
            label: label.into(),
            range,
            role,
        }
    }
}

/// Measured sensory pore attachment and delay tuning within one `BandSpec` at a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct BandAllocation {
    pub label: String,
    pub role: BandRole,
    pub channel_width: usize,
    pub pore_count: usize,
    pub raw_conductance_sum: f64,
    /// Fraction of the organism's total sensory pore conductance attached to this band (`[0, 1]`).
    pub conductance_share: f64,
    /// Unweighted mean delay `\Delta` (in ticks) of pores attached to this band.
    pub mean_pore_delay: f64,
    /// Conductance-weighted mean delay `\Delta` (in ticks) of pores attached to this band.
    pub conductance_weighted_delay: f64,
}

/// Returns a copy of `snapshot` with sensory and motor pore channel indices mapped through
/// observer-only decoders (e.g. `PermutedMedium::invert_sensor_channel` / `invert_motor_channel`).
#[must_use]
pub fn decode_snapshot_channels<DS, DM>(
    snapshot: &OrganismSnapshot,
    decode_sensor: DS,
    decode_motor: DM,
) -> OrganismSnapshot
where
    DS: Fn(u32) -> u32,
    DM: Fn(u32) -> u32,
{
    let mut out = snapshot.clone();
    for ch in &mut out.sensor_pore_channels {
        *ch = decode_sensor(*ch);
    }
    for m in &mut out.motor_pore_channels {
        *m = decode_motor(*m);
    }
    out
}

/// Computes per-band sensory pore conductance allocations plus `(total_conductance, noisy_tv_share)`.
#[must_use]
pub fn compute_band_allocations(
    snapshot: &OrganismSnapshot,
    bands: &[BandSpec],
) -> (Vec<BandAllocation>, f64, f64) {
    let pore_len = snapshot
        .sensor_pore_channels
        .len()
        .min(snapshot.sensor_pore_conductances.len());

    let mut total_conductance = 0.0_f64;
    for idx in 0..pore_len {
        let c = f64::from(snapshot.sensor_pore_conductances[idx]);
        if c.is_finite() && c > 0.0 {
            total_conductance += c;
        }
    }

    let mut allocations = Vec::with_capacity(bands.len());
    let mut noisy_tv_conductance = 0.0_f64;

    for band in bands {
        let width = band.range.end.saturating_sub(band.range.start) as usize;
        let mut pore_count = 0usize;
        let mut cond_sum = 0.0_f64;
        let mut delay_sum = 0.0_f64;
        let mut weighted_delay_sum = 0.0_f64;

        for idx in 0..pore_len {
            let ch = snapshot.sensor_pore_channels[idx];
            if band.range.contains(&ch) {
                let raw_c = f64::from(snapshot.sensor_pore_conductances[idx]);
                let c = if raw_c.is_finite() && raw_c > 0.0 {
                    raw_c
                } else {
                    0.0
                };
                let delay = f64::from(snapshot.sensor_pore_delays.get(idx).copied().unwrap_or(1));
                pore_count += 1;
                cond_sum += c;
                delay_sum += delay;
                weighted_delay_sum += c * delay;
            }
        }

        if band.role == BandRole::NoisyTvDistractor {
            noisy_tv_conductance += cond_sum;
        }

        let conductance_share = if total_conductance > 1e-18 {
            (cond_sum / total_conductance).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mean_pore_delay = if pore_count > 0 {
            delay_sum / (pore_count as f64)
        } else {
            0.0
        };
        let conductance_weighted_delay = if cond_sum > 1e-18 {
            weighted_delay_sum / cond_sum
        } else {
            mean_pore_delay
        };

        allocations.push(BandAllocation {
            label: band.label.clone(),
            role: band.role,
            channel_width: width,
            pore_count,
            raw_conductance_sum: cond_sum,
            conductance_share,
            mean_pore_delay,
            conductance_weighted_delay,
        });
    }

    let noisy_tv_share = if total_conductance > 1e-18 {
        (noisy_tv_conductance / total_conductance).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (allocations, total_conductance, noisy_tv_share)
}

/// Experimental arm in the multi-seed twin observation protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TwinArm {
    /// Open-ended morphogenesis and plasticity (`1..=T`).
    Natural,
    /// Strictly frozen parameters and topology (`ControlMode::FrozenPlasticity`, `1..=T`).
    FrozenPlasticity,
    /// Identical `Natural` plasticity fed the `Natural` donor's per-tick `(metabolic_influx, tissue_stress)` (`1..=T`).
    YokedMetabolism,
    /// Identical `Natural` plasticity whose emitted `FieldPulse` is temporally and channel-shuffled (`1..=T`).
    ActionShuffled,
    /// Mature `Natural` organism at tick `T` evaluated without lesion on `eval_world` (`1..=T_eval`).
    UnlesionedPostLife,
    /// Mature `Natural` organism at tick `T` after flagless `zero_learned_conductances()` on `eval_world` (`1..=T_eval`).
    FlaglessPostLifeLesion,
}

impl TwinArm {
    /// Canonical list of all arms in report display order.
    pub const ALL: [Self; 6] = [
        Self::Natural,
        Self::FrozenPlasticity,
        Self::YokedMetabolism,
        Self::ActionShuffled,
        Self::UnlesionedPostLife,
        Self::FlaglessPostLifeLesion,
    ];

    /// Identifier string for tables and JSON serialization.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "Natural",
            Self::FrozenPlasticity => "FrozenPlasticity",
            Self::YokedMetabolism => "YokedMetabolism",
            Self::ActionShuffled => "ActionShuffled",
            Self::UnlesionedPostLife => "UnlesionedPostLife",
            Self::FlaglessPostLifeLesion => "FlaglessPostLifeLesion",
        }
    }
}

/// Configuration for a multi-seed 6-control twin study.
#[derive(Clone, Debug)]
pub struct TwinStudyConfig {
    pub study_name: String,
    pub seeds: Vec<u64>,
    pub horizon_ticks: u64,
    /// Post-life evaluation window (`T_eval`) comparing unlesioned mature continuation vs flagless lesion.
    pub post_life_eval_ticks: u64,
    pub bands: Vec<BandSpec>,
}

impl TwinStudyConfig {
    /// Creates a standard 12-seed study configuration (`post_life_eval_ticks = max(horizon_ticks / 2, 50)`).
    #[must_use]
    pub fn new(study_name: impl Into<String>, horizon_ticks: u64, bands: Vec<BandSpec>) -> Self {
        let horizon = horizon_ticks.max(1);
        let eval_ticks = (horizon / 2).max(1);
        Self {
            study_name: study_name.into(),
            seeds: DEFAULT_TWELVE_SEEDS.to_vec(),
            horizon_ticks: horizon,
            post_life_eval_ticks: eval_ticks,
            bands,
        }
    }
}

/// Computes the canonical developmental checkpoint ticks (`Tick 1`, `10%`, `25%`, `50%`, `75%`, `100%`).
#[must_use]
pub fn developmental_checkpoint_ticks(horizon_ticks: u64) -> Vec<u64> {
    let t = horizon_ticks.max(1);
    let fractions = [0.10_f64, 0.25, 0.50, 0.75, 1.00];
    let mut ticks = Vec::with_capacity(6);
    ticks.push(1_u64);
    for frac in fractions {
        let raw = ((t as f64) * frac).round() as u64;
        let clamped = raw.clamp(1, t);
        if ticks.last().copied() != Some(clamped) {
            ticks.push(clamped);
        }
    }
    ticks.sort_unstable();
    ticks.dedup();
    ticks
}

/// Single developmental checkpoint sample along an individual's lifetime.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentalSample {
    pub tick: u64,
    pub progress_fraction: f64,
    pub window_ticks: u64,
    pub window_achieved_influx_per_tick: f64,
    pub window_achieved_stress_per_tick: f64,
    pub cumulative_achieved_influx: f64,
    pub cumulative_achieved_stress: f64,
    pub snapshot: OrganismSnapshot,
    pub band_allocations: Vec<BandAllocation>,
    pub noisy_tv_conductance_share: f64,
}

/// Complete telemetry recorded for one `(seed, TwinArm)` closed-loop run.
#[derive(Clone, Debug, PartialEq)]
pub struct SeedArmRun {
    pub seed: u64,
    pub arm: TwinArm,
    pub ticks_executed: u64,
    /// Real nourishment generated by this arm's own physical world `W`.
    pub total_achieved_influx: f64,
    /// Real tissue stress generated by this arm's own physical world `W`.
    pub total_achieved_stress: f64,
    /// Nourishment actually fed into `OrganismEngine::step` (equals donor stream for `YokedMetabolism`).
    pub total_experienced_influx: f64,
    /// Stress actually fed into `OrganismEngine::step` (equals donor stream for `YokedMetabolism`).
    pub total_experienced_stress: f64,
    pub achieved_influx_per_tick: f64,
    pub achieved_stress_per_tick: f64,
    pub net_achieved_energy_per_tick: f64,
    pub first_half_influx_per_tick: f64,
    pub second_half_influx_per_tick: f64,
    pub mean_emitted_pulse_l2: f64,
    pub mean_applied_pulse_l2: f64,
    pub final_snapshot: OrganismSnapshot,
    pub final_band_allocations: Vec<BandAllocation>,
    pub final_total_sensory_conductance: f64,
    pub final_noisy_tv_conductance_share: f64,
    pub developmental_trajectory: Vec<DevelopmentalSample>,
    pub pore_channel_profile: Vec<f64>,
    pub tau_octave_histogram: Vec<f64>,
    pub trajectory_action_histogram: Vec<f64>,
}

/// Aggregated empirical distributions across all seeds for a single `BandSpec`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandCohortSummary {
    pub label: String,
    pub role: BandRole,
    pub channel_width: usize,
    pub uniform_random_share_expectation: f64,
    pub conductance_share: EmpiricalDistribution,
    pub pore_count: EmpiricalDistribution,
    pub mean_pore_delay: EmpiricalDistribution,
    pub conductance_weighted_delay: EmpiricalDistribution,
}

/// Aggregated empirical summary across all seeds at one developmental checkpoint fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentalCohortPoint {
    pub tick: u64,
    pub progress_fraction: f64,
    pub window_influx_per_tick: EmpiricalDistribution,
    pub window_stress_per_tick: EmpiricalDistribution,
    pub atp: EmpiricalDistribution,
    pub cell_count: EmpiricalDistribution,
    pub tube_count: EmpiricalDistribution,
    pub mean_tau: EmpiricalDistribution,
    pub tau_octave_count: EmpiricalDistribution,
    pub mean_interventional_agency: EmpiricalDistribution,
    pub noisy_tv_conductance_share: EmpiricalDistribution,
    pub band_conductance_shares: Vec<(String, EmpiricalDistribution)>,
    pub band_weighted_delays: Vec<(String, EmpiricalDistribution)>,
}

/// Multi-seed empirical summary for one `TwinArm`.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmCohortSummary {
    pub arm: TwinArm,
    pub seed_count: usize,
    pub ticks_per_seed: u64,
    pub achieved_influx_per_tick: EmpiricalDistribution,
    pub achieved_stress_per_tick: EmpiricalDistribution,
    pub net_achieved_energy_per_tick: EmpiricalDistribution,
    pub first_half_influx_per_tick: EmpiricalDistribution,
    pub second_half_influx_per_tick: EmpiricalDistribution,
    pub mean_emitted_pulse_l2: EmpiricalDistribution,
    pub mean_applied_pulse_l2: EmpiricalDistribution,
    pub final_atp: EmpiricalDistribution,
    pub final_adenosine: EmpiricalDistribution,
    pub final_temperature: EmpiricalDistribution,
    pub cell_count: EmpiricalDistribution,
    pub tube_count: EmpiricalDistribution,
    pub stiff_highway_tubes: EmpiricalDistribution,
    pub inhibitory_tubes: EmpiricalDistribution,
    pub active_echo_delay_tubes: EmpiricalDistribution,
    pub mean_tau: EmpiricalDistribution,
    pub max_tau: EmpiricalDistribution,
    pub tau_octave_count: EmpiricalDistribution,
    pub mean_interventional_agency: EmpiricalDistribution,
    pub mean_prediction_error: EmpiricalDistribution,
    pub committed_motor_pulses: EmpiricalDistribution,
    pub subthreshold_rehearsals: EmpiricalDistribution,
    pub rest_consolidation_ticks: EmpiricalDistribution,
    pub mitosis_births: EmpiricalDistribution,
    pub contradiction_splits: EmpiricalDistribution,
    pub myelinated_shortcuts: EmpiricalDistribution,
    pub atrophied_structures: EmpiricalDistribution,
    pub used_state_bytes: EmpiricalDistribution,
    pub noisy_tv_conductance_share: EmpiricalDistribution,
    pub band_summaries: Vec<BandCohortSummary>,
    pub developmental_curve: Vec<DevelopmentalCohortPoint>,
    pub individuality: IndividualityDivergence,
    pub seed_runs: Vec<SeedArmRun>,
}

/// Named two-sample non-parametric comparison between two experimental arms on a specific metric.
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseMetricComparison {
    pub metric_name: String,
    pub arm_a: TwinArm,
    pub arm_b: TwinArm,
    pub comparison: MannWhitneyComparison,
}

/// Complete output of a multi-seed 6-control twin study.
#[derive(Clone, Debug, PartialEq)]
pub struct TwinStudyReport {
    pub study_name: String,
    pub seeds: Vec<u64>,
    pub horizon_ticks: u64,
    pub post_life_eval_ticks: u64,
    pub num_ambient_channels: usize,
    pub num_actuators: usize,
    pub bands: Vec<BandSpec>,
    pub uniform_noisy_tv_channel_fraction: f64,
    pub lesion_preserved_natural_mode_all_seeds: bool,
    pub arm_summaries: Vec<ArmCohortSummary>,
    pub pairwise_comparisons: Vec<PairwiseMetricComparison>,
}

impl TwinStudyReport {
    /// Looks up the cohort summary for a specific `TwinArm`.
    #[must_use]
    pub fn arm_summary(&self, arm: TwinArm) -> Option<&ArmCohortSummary> {
        self.arm_summaries.iter().find(|s| s.arm == arm)
    }
}

/// Runs a multi-seed 6-control twin observation study without mid-run world perturbations.
pub fn run_twin_study<O, W, FO, FW>(
    config: &TwinStudyConfig,
    make_organism: FO,
    make_world: FW,
) -> Result<TwinStudyReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    run_twin_study_with_perturbation(config, make_organism, make_world, |_, _| {})
}

/// Runs a multi-seed 6-control twin observation study with a continuous mid-run environment
/// perturbation hook `perturb(tick, &mut world)` (e.g., Day-to-Darkness Cavefish Transition
/// or Mid-Life Polarity Reversal without resetting the organism).
pub fn run_twin_study_with_perturbation<O, W, FO, FW, P>(
    config: &TwinStudyConfig,
    make_organism: FO,
    make_world: FW,
    mut perturb: P,
) -> Result<TwinStudyReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
    P: FnMut(u64, &mut W),
{
    run_twin_study_instrumented(
        config,
        make_organism,
        make_world,
        |_seed, _arm, tick, _org, world| perturb(tick, world),
        |_seed, _arm, _world| {},
        |ch, _world| ch,
        |m, _world| m,
        |_run, _org, _world| {},
    )
}

/// Runs a multi-seed 6-control twin observation study with full non-invasive observer hooks:
/// - `on_primary_tick(seed, arm, tick, &org, &mut world)`: invoked before each tick `1..=T` of the 4 primary arms.
/// - `init_eval_world(seed, arm, &mut eval_world)`: configures post-life lesion evaluation worlds (e.g. post-shift regime).
/// - `decode_sensor_channel(ch, &world)` / `decode_motor_channel(m, &world)`: observer-only wire bijections.
/// - `on_run_finished(&seed_arm_run, &org, &world)`: collects world-specific metrics after each `(seed, arm)` run.
#[allow(clippy::too_many_arguments)]
pub fn run_twin_study_instrumented<O, W, FO, FW, P, EP, DS, DM, FFin>(
    config: &TwinStudyConfig,
    make_organism: FO,
    make_world: FW,
    mut on_primary_tick: P,
    mut init_eval_world: EP,
    decode_sensor_channel: DS,
    decode_motor_channel: DM,
    mut on_run_finished: FFin,
) -> Result<TwinStudyReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
    P: FnMut(u64, TwinArm, u64, &O, &mut W),
    EP: FnMut(u64, TwinArm, &mut W),
    DS: Fn(u32, &W) -> u32,
    DM: Fn(u32, &W) -> u32,
    FFin: FnMut(&SeedArmRun, &O, &W),
{
    let horizon = config.horizon_ticks.max(1);
    let eval_ticks = config.post_life_eval_ticks.max(1);
    let checkpoints = developmental_checkpoint_ticks(horizon);
    let eval_checkpoints = developmental_checkpoint_ticks(eval_ticks);

    let probe_world = make_world(config.seeds.first().copied().unwrap_or(1));
    let num_ambient_channels = probe_world.num_ambient_channels();
    let num_actuators = probe_world.num_actuators();
    drop(probe_world);

    let noisy_channels: usize = config
        .bands
        .iter()
        .filter(|b| b.role == BandRole::NoisyTvDistractor)
        .map(|b| b.range.end.saturating_sub(b.range.start) as usize)
        .sum();
    let uniform_noisy_tv_channel_fraction = if num_ambient_channels > 0 {
        (noisy_channels as f64 / num_ambient_channels as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let n_seeds = config.seeds.len();
    let mut natural_runs = Vec::with_capacity(n_seeds);
    let mut mature_checkpoints = Vec::with_capacity(n_seeds);
    let mut donor_traces: Vec<Vec<(f32, f32)>> = Vec::with_capacity(n_seeds);
    let mut frozen_runs = Vec::with_capacity(n_seeds);
    let mut yoked_runs = Vec::with_capacity(n_seeds);
    let mut shuffled_runs = Vec::with_capacity(n_seeds);
    let mut unlesioned_eval_runs = Vec::with_capacity(n_seeds);
    let mut lesioned_eval_runs = Vec::with_capacity(n_seeds);
    let mut lesion_preserved_natural_mode_all_seeds = true;

    // Pass 1: Run all Natural twins (`1..=T`) to record mature checkpoints and donor metabolic streams.
    for &seed in &config.seeds {
        let mut natural_org = make_organism(seed);
        natural_org.set_control_mode(ControlMode::Natural);
        let mut natural_world = make_world(seed);
        let mut donor_metabolic_trace = Vec::with_capacity(horizon as usize);

        let nat_run = execute_single_run(
            seed,
            TwinArm::Natural,
            &mut natural_org,
            &mut natural_world,
            horizon,
            &checkpoints,
            &config.bands,
            Some(&mut donor_metabolic_trace),
            None,
            None,
            &mut on_primary_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&nat_run, &natural_org, &natural_world);
        let mature_checkpoint = natural_org.encode_checkpoint()?;
        natural_runs.push(nat_run);
        mature_checkpoints.push(mature_checkpoint);
        donor_traces.push(donor_metabolic_trace);
    }

    // Pass 2: Run FrozenPlasticity, cohort-deranged YokedMetabolism, ActionShuffled,
    // and Post-Life Unlesioned vs Flagless Lesion controls on each seed.
    // Pairing `Yoked[i]` (`O(seed_i), W(seed_i)`) with `Natural[(i + 1) % N]` preserves the exact
    // cohort distribution of metabolic streams while breaking deterministic same-seed replay lock.
    for (idx, &seed) in config.seeds.iter().enumerate() {
        // 2. FrozenPlasticity Twin (`1..=T`).
        let mut frozen_org = make_organism(seed);
        frozen_org.set_control_mode(ControlMode::FrozenPlasticity);
        let mut frozen_world = make_world(seed);
        let froz_run = execute_single_run(
            seed,
            TwinArm::FrozenPlasticity,
            &mut frozen_org,
            &mut frozen_world,
            horizon,
            &checkpoints,
            &config.bands,
            None,
            None,
            None,
            &mut on_primary_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&froz_run, &frozen_org, &frozen_world);
        frozen_runs.push(froz_run);

        // 3. YokedMetabolism Twin (`1..=T`) receiving a decoupled Natural donor's `(influx, stress)` stream.
        let single_seed_shifted: Vec<(f32, f32)>;
        let donor_slice: &[(f32, f32)] = if n_seeds >= 2 {
            &donor_traces[(idx + 1) % n_seeds]
        } else {
            let orig = &donor_traces[0];
            let shift = (orig.len() / 2).max(1);
            single_seed_shifted = orig
                .iter()
                .cycle()
                .skip(shift)
                .take(orig.len())
                .copied()
                .collect();
            &single_seed_shifted
        };
        let mut yoked_org = make_organism(seed);
        yoked_org.set_control_mode(ControlMode::Natural);
        let mut yoked_world = make_world(seed);
        let yok_run = execute_single_run(
            seed,
            TwinArm::YokedMetabolism,
            &mut yoked_org,
            &mut yoked_world,
            horizon,
            &checkpoints,
            &config.bands,
            None,
            Some(donor_slice),
            None,
            &mut on_primary_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&yok_run, &yoked_org, &yoked_world);
        yoked_runs.push(yok_run);

        // 4. ActionShuffled Twin (`1..=T`) with temporal + channel pulse shuffling.
        let mut shuffled_org = make_organism(seed);
        shuffled_org.set_control_mode(ControlMode::Natural);
        let mut shuffled_world = make_world(seed);
        let mut shuffler =
            PulseTemporalChannelShuffler::new(seed ^ ACTION_SHUFFLE_SEED_SALT, num_actuators);
        let shuf_run = execute_single_run(
            seed,
            TwinArm::ActionShuffled,
            &mut shuffled_org,
            &mut shuffled_world,
            horizon,
            &checkpoints,
            &config.bands,
            None,
            None,
            Some(&mut shuffler),
            &mut on_primary_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&shuf_run, &shuffled_org, &shuffled_world);
        shuffled_runs.push(shuf_run);

        // 5a. Unlesioned Post-Life Mature Continuation (`1..=T_eval` on identical eval_world).
        let mature_checkpoint = &mature_checkpoints[idx];
        let eval_world_seed = seed ^ POST_LIFE_WORLD_SEED_SALT;
        let mut unlesioned_org = make_organism(seed ^ FRESH_RESTORE_SEED_SALT);
        unlesioned_org.restore_checkpoint(mature_checkpoint)?;
        unlesioned_org.set_control_mode(ControlMode::FrozenPlasticity);
        let mut unlesioned_world = make_world(eval_world_seed);
        init_eval_world(seed, TwinArm::UnlesionedPostLife, &mut unlesioned_world);
        let mut no_op_tick = |_: u64, _: TwinArm, _: u64, _: &O, _: &mut W| {};
        let unlesioned_run = execute_single_run(
            seed,
            TwinArm::UnlesionedPostLife,
            &mut unlesioned_org,
            &mut unlesioned_world,
            eval_ticks,
            &eval_checkpoints,
            &config.bands,
            None,
            None,
            None,
            &mut no_op_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&unlesioned_run, &unlesioned_org, &unlesioned_world);
        unlesioned_eval_runs.push(unlesioned_run);

        // 5b. Flagless Post-Life Conductance Lesion (`1..=T_eval` on identical eval_world).
        let mut lesioned_org = make_organism(seed ^ FRESH_RESTORE_SEED_SALT);
        lesioned_org.restore_checkpoint(mature_checkpoint)?;
        lesioned_org.set_control_mode(ControlMode::Natural);
        lesioned_org.zero_learned_conductances();
        if lesioned_org.control_mode() != ControlMode::Natural {
            lesion_preserved_natural_mode_all_seeds = false;
        }
        lesioned_org.set_control_mode(ControlMode::FrozenPlasticity);
        let mut lesioned_world = make_world(eval_world_seed);
        init_eval_world(seed, TwinArm::FlaglessPostLifeLesion, &mut lesioned_world);
        let lesioned_run = execute_single_run(
            seed,
            TwinArm::FlaglessPostLifeLesion,
            &mut lesioned_org,
            &mut lesioned_world,
            eval_ticks,
            &eval_checkpoints,
            &config.bands,
            None,
            None,
            None,
            &mut no_op_tick,
            &decode_sensor_channel,
            &decode_motor_channel,
        )?;
        on_run_finished(&lesioned_run, &lesioned_org, &lesioned_world);
        lesioned_eval_runs.push(lesioned_run);
    }

    let arm_summaries = vec![
        summarize_arm_cohort(
            TwinArm::Natural,
            horizon,
            num_ambient_channels,
            &config.bands,
            natural_runs,
        ),
        summarize_arm_cohort(
            TwinArm::FrozenPlasticity,
            horizon,
            num_ambient_channels,
            &config.bands,
            frozen_runs,
        ),
        summarize_arm_cohort(
            TwinArm::YokedMetabolism,
            horizon,
            num_ambient_channels,
            &config.bands,
            yoked_runs,
        ),
        summarize_arm_cohort(
            TwinArm::ActionShuffled,
            horizon,
            num_ambient_channels,
            &config.bands,
            shuffled_runs,
        ),
        summarize_arm_cohort(
            TwinArm::UnlesionedPostLife,
            eval_ticks,
            num_ambient_channels,
            &config.bands,
            unlesioned_eval_runs,
        ),
        summarize_arm_cohort(
            TwinArm::FlaglessPostLifeLesion,
            eval_ticks,
            num_ambient_channels,
            &config.bands,
            lesioned_eval_runs,
        ),
    ];

    let pairwise_comparisons = build_pairwise_comparisons(&arm_summaries);

    Ok(TwinStudyReport {
        study_name: config.study_name.clone(),
        seeds: config.seeds.clone(),
        horizon_ticks: horizon,
        post_life_eval_ticks: eval_ticks,
        num_ambient_channels,
        num_actuators,
        bands: config.bands.clone(),
        uniform_noisy_tv_channel_fraction,
        lesion_preserved_natural_mode_all_seeds,
        arm_summaries,
        pairwise_comparisons,
    })
}

/// Configuration for testing emergent individuality across genetically identical twins
/// exposed only to microscopic initial sensory wave perturbations.
#[derive(Clone, Debug)]
pub struct IdenticalTwinConfig {
    pub genome_seed: u64,
    pub world_seed: u64,
    pub twin_count: usize,
    pub horizon_ticks: u64,
    pub perturbation_ticks: u64,
    pub perturbation_epsilon: f32,
    pub bands: Vec<BandSpec>,
}

impl IdenticalTwinConfig {
    /// Default 12 genetically identical twins perturbed by `1e-4` sensory jitter on ticks `1..=3`.
    #[must_use]
    pub fn default_twelve(genome_seed: u64, horizon_ticks: u64, bands: Vec<BandSpec>) -> Self {
        Self {
            genome_seed,
            world_seed: genome_seed ^ 0x1DE2_71CA_1000_0001,
            twin_count: 12,
            horizon_ticks: horizon_ticks.max(1),
            perturbation_ticks: 3,
            perturbation_epsilon: 1e-4,
            bands,
        }
    }
}

/// Result of running genetically identical twins under microscopic sensory perturbations.
#[derive(Clone, Debug, PartialEq)]
pub struct IdenticalTwinReport {
    pub genome_seed: u64,
    pub world_seed: u64,
    pub twin_count: usize,
    pub horizon_ticks: u64,
    pub perturbation_epsilon: f32,
    pub cohort_summary: ArmCohortSummary,
}

/// Observes `twin_count` genetically identical `Natural` organisms exposed to microscopic initial jitter.
pub fn observe_identical_twin_divergence<O, W, FO, FW>(
    config: &IdenticalTwinConfig,
    make_organism: FO,
    make_world: FW,
) -> Result<IdenticalTwinReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    let count = config.twin_count.max(2);
    let horizon = config.horizon_ticks.max(1);
    let checkpoints = developmental_checkpoint_ticks(horizon);

    let probe_world = make_world(config.world_seed);
    let num_ambient_channels = probe_world.num_ambient_channels();
    drop(probe_world);

    let mut runs = Vec::with_capacity(count);
    for twin_idx in 0..count {
        let mut org = make_organism(config.genome_seed);
        org.set_control_mode(ControlMode::Natural);
        let mut world = make_world(config.world_seed);
        let mut jitter_rng =
            DeterministicRng::new(config.genome_seed ^ ((twin_idx as u64 + 1) * 0x9E37_79B9));
        let mut ambient_scratch = vec![0.0_f32; num_ambient_channels];

        let run = execute_identical_twin_single_run(
            twin_idx as u64,
            &mut org,
            &mut world,
            horizon,
            config.perturbation_ticks,
            config.perturbation_epsilon,
            &mut jitter_rng,
            &mut ambient_scratch,
            &checkpoints,
            &config.bands,
        )?;
        runs.push(run);
    }

    let cohort_summary = summarize_arm_cohort(
        TwinArm::Natural,
        horizon,
        num_ambient_channels,
        &config.bands,
        runs,
    );

    Ok(IdenticalTwinReport {
        genome_seed: config.genome_seed,
        world_seed: config.world_seed,
        twin_count: count,
        horizon_ticks: horizon,
        perturbation_epsilon: config.perturbation_epsilon,
        cohort_summary,
    })
}

/// Deterministic temporal + actuator-channel `FieldPulse` shuffler preserving pulse energy
/// while breaking fine sensorimotor contingency (`ActionShuffled` control).
pub struct PulseTemporalChannelShuffler {
    rng: DeterministicRng,
    reservoir: Vec<FieldPulse>,
    perm_indices: Vec<usize>,
}

impl PulseTemporalChannelShuffler {
    /// Creates a new deterministic pulse shuffler for `num_actuators`.
    #[must_use]
    pub fn new(seed: u64, num_actuators: usize) -> Self {
        let reservoir = (0..TEMPORAL_SHUFFLE_BUFFER_LEN)
            .map(|_| FieldPulse::silent(num_actuators))
            .collect();
        let perm_indices = (0..num_actuators).collect();
        Self {
            rng: DeterministicRng::new(seed),
            reservoir,
            perm_indices,
        }
    }

    /// Permutes actuator channels and draws from a fixed-lag reservoir buffer.
    pub fn shuffle_pulse(&mut self, emitted: &FieldPulse) -> FieldPulse {
        let n = emitted.actuator_out.len();
        if self.perm_indices.len() != n {
            self.perm_indices = (0..n).collect();
            self.reservoir = (0..TEMPORAL_SHUFFLE_BUFFER_LEN)
                .map(|_| FieldPulse::silent(n))
                .collect();
        }
        self.rng.shuffle(&mut self.perm_indices);
        let mut permuted = Vec::with_capacity(n);
        for &src_idx in &self.perm_indices {
            permuted.push(emitted.actuator_out[src_idx]);
        }
        let candidate = FieldPulse {
            actuator_out: permuted,
        };
        let slot = self.rng.next_usize(self.reservoir.len());
        std::mem::replace(&mut self.reservoir[slot], candidate)
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_single_run<O, W, P, DS, DM>(
    seed: u64,
    arm: TwinArm,
    org: &mut O,
    world: &mut W,
    horizon: u64,
    checkpoints: &[u64],
    bands: &[BandSpec],
    mut record_donor_trace: Option<&mut Vec<(f32, f32)>>,
    yoked_donor_trace: Option<&[(f32, f32)]>,
    mut pulse_shuffler: Option<&mut PulseTemporalChannelShuffler>,
    on_tick: &mut P,
    decode_sensor: &DS,
    decode_motor: &DM,
) -> Result<SeedArmRun, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    P: FnMut(u64, TwinArm, u64, &O, &mut W),
    DS: Fn(u32, &W) -> u32,
    DM: Fn(u32, &W) -> u32,
{
    let num_ambient = world.num_ambient_channels();
    let num_actuators = world.num_actuators();
    let half_tick = (horizon / 2).max(1);

    let mut total_achieved_influx = 0.0_f64;
    let mut total_achieved_stress = 0.0_f64;
    let mut total_experienced_influx = 0.0_f64;
    let mut total_experienced_stress = 0.0_f64;
    let mut first_half_influx = 0.0_f64;
    let mut second_half_influx = 0.0_f64;
    let mut sum_emitted_l2 = 0.0_f64;
    let mut sum_applied_l2 = 0.0_f64;

    let mut next_cp_idx = 0usize;
    let mut prev_cp_tick = 0_u64;
    let mut prev_cp_influx = 0.0_f64;
    let mut prev_cp_stress = 0.0_f64;
    let mut trajectory = Vec::with_capacity(checkpoints.len());
    let mut action_hist = vec![0.0_f64; 1 + 2 * num_actuators];
    let mut tau_hist = vec![0.0_f64; TAU_HISTOGRAM_BINS];

    for tick in 1..=horizon {
        on_tick(seed, arm, tick, org, world);

        let (emitted_pulse, ach_in, ach_st, exp_in, exp_st) = {
            let raw_view = world.current_view();
            let ach_in = f64::from(raw_view.metabolic_influx);
            let ach_st = f64::from(raw_view.tissue_stress);

            if let Some(trace_out) = record_donor_trace.as_deref_mut() {
                trace_out.push((raw_view.metabolic_influx, raw_view.tissue_stress));
            }

            let step_view = if let Some(donor) = yoked_donor_trace {
                let (d_in, d_st) = donor
                    .get((tick - 1) as usize)
                    .copied()
                    .unwrap_or((raw_view.metabolic_influx, raw_view.tissue_stress));
                FieldView {
                    ambient_in: raw_view.ambient_in,
                    metabolic_influx: d_in,
                    tissue_stress: d_st,
                    num_actuators: raw_view.num_actuators,
                }
            } else {
                raw_view
            };

            let exp_in = f64::from(step_view.metabolic_influx);
            let exp_st = f64::from(step_view.tissue_stress);
            let pulse = org.step(step_view)?;
            (pulse, ach_in, ach_st, exp_in, exp_st)
        };

        total_achieved_influx += ach_in;
        total_achieved_stress += ach_st;
        total_experienced_influx += exp_in;
        total_experienced_stress += exp_st;
        if tick <= half_tick {
            first_half_influx += ach_in;
        } else {
            second_half_influx += ach_in;
        }

        record_action_bin(&emitted_pulse, &mut action_hist);
        sum_emitted_l2 += pulse_l2_norm(&emitted_pulse);

        if let Some(shuffler) = pulse_shuffler.as_deref_mut() {
            let applied = shuffler.shuffle_pulse(&emitted_pulse);
            sum_applied_l2 += pulse_l2_norm(&applied);
            world.apply_pulse(&applied);
        } else {
            sum_applied_l2 += pulse_l2_norm(&emitted_pulse);
            world.apply_pulse(&emitted_pulse);
        }

        if checkpoints.get(next_cp_idx).copied() == Some(tick) {
            let snap = decode_snapshot_channels(
                &org.snapshot(),
                |c| decode_sensor(c, world),
                |m| decode_motor(m, world),
            );
            accumulate_tau_octave_sample(&snap, &mut tau_hist);
            let (band_allocations, _, noisy_tv_share) = compute_band_allocations(&snap, bands);
            let window_ticks = (tick - prev_cp_tick).max(1);
            let win_in = (total_achieved_influx - prev_cp_influx) / (window_ticks as f64);
            let win_st = (total_achieved_stress - prev_cp_stress) / (window_ticks as f64);

            trajectory.push(DevelopmentalSample {
                tick,
                progress_fraction: (tick as f64) / (horizon as f64),
                window_ticks,
                window_achieved_influx_per_tick: win_in,
                window_achieved_stress_per_tick: win_st,
                cumulative_achieved_influx: total_achieved_influx,
                cumulative_achieved_stress: total_achieved_stress,
                snapshot: snap,
                band_allocations,
                noisy_tv_conductance_share: noisy_tv_share,
            });
            prev_cp_tick = tick;
            prev_cp_influx = total_achieved_influx;
            prev_cp_stress = total_achieved_stress;
            next_cp_idx += 1;
        }
    }

    let decoded_final_snap = decode_snapshot_channels(
        &org.snapshot(),
        |c| decode_sensor(c, world),
        |m| decode_motor(m, world),
    );

    finalize_seed_arm_run(
        seed,
        arm,
        horizon,
        half_tick,
        num_ambient,
        bands,
        decoded_final_snap,
        total_achieved_influx,
        total_achieved_stress,
        total_experienced_influx,
        total_experienced_stress,
        first_half_influx,
        second_half_influx,
        sum_emitted_l2,
        sum_applied_l2,
        trajectory,
        tau_hist,
        action_hist,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_identical_twin_single_run<O, W>(
    twin_id: u64,
    org: &mut O,
    world: &mut W,
    horizon: u64,
    perturbation_ticks: u64,
    perturbation_epsilon: f32,
    jitter_rng: &mut DeterministicRng,
    ambient_scratch: &mut Vec<f32>,
    checkpoints: &[u64],
    bands: &[BandSpec],
) -> Result<SeedArmRun, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
{
    let num_ambient = world.num_ambient_channels();
    let num_actuators = world.num_actuators();
    let half_tick = (horizon / 2).max(1);

    let mut total_achieved_influx = 0.0_f64;
    let mut total_achieved_stress = 0.0_f64;
    let mut first_half_influx = 0.0_f64;
    let mut second_half_influx = 0.0_f64;
    let mut sum_emitted_l2 = 0.0_f64;

    let mut next_cp_idx = 0usize;
    let mut prev_cp_tick = 0_u64;
    let mut prev_cp_influx = 0.0_f64;
    let mut prev_cp_stress = 0.0_f64;
    let mut trajectory = Vec::with_capacity(checkpoints.len());
    let mut action_hist = vec![0.0_f64; 1 + 2 * num_actuators];
    let mut tau_hist = vec![0.0_f64; TAU_HISTOGRAM_BINS];

    for tick in 1..=horizon {
        let (emitted_pulse, ach_in, ach_st) = {
            let raw_view = world.current_view();
            let ach_in = f64::from(raw_view.metabolic_influx);
            let ach_st = f64::from(raw_view.tissue_stress);

            let pulse = if tick <= perturbation_ticks && perturbation_epsilon > 0.0 {
                ambient_scratch.resize(raw_view.ambient_in.len(), 0.0);
                for (dst, &src) in ambient_scratch.iter_mut().zip(raw_view.ambient_in.iter()) {
                    let noise = ((jitter_rng.next_f64() * 2.0) - 1.0) as f32 * perturbation_epsilon;
                    *dst = (src + noise).clamp(-1.0, 1.0);
                }
                let jittered_view = FieldView {
                    ambient_in: ambient_scratch.as_slice(),
                    metabolic_influx: raw_view.metabolic_influx,
                    tissue_stress: raw_view.tissue_stress,
                    num_actuators: raw_view.num_actuators,
                };
                org.step(jittered_view)?
            } else {
                org.step(raw_view)?
            };
            (pulse, ach_in, ach_st)
        };

        total_achieved_influx += ach_in;
        total_achieved_stress += ach_st;
        if tick <= half_tick {
            first_half_influx += ach_in;
        } else {
            second_half_influx += ach_in;
        }
        record_action_bin(&emitted_pulse, &mut action_hist);
        sum_emitted_l2 += pulse_l2_norm(&emitted_pulse);
        world.apply_pulse(&emitted_pulse);

        if checkpoints.get(next_cp_idx).copied() == Some(tick) {
            let snap = org.snapshot();
            accumulate_tau_octave_sample(&snap, &mut tau_hist);
            let (band_allocations, _, noisy_tv_share) = compute_band_allocations(&snap, bands);
            let window_ticks = (tick - prev_cp_tick).max(1);
            let win_in = (total_achieved_influx - prev_cp_influx) / (window_ticks as f64);
            let win_st = (total_achieved_stress - prev_cp_stress) / (window_ticks as f64);
            trajectory.push(DevelopmentalSample {
                tick,
                progress_fraction: (tick as f64) / (horizon as f64),
                window_ticks,
                window_achieved_influx_per_tick: win_in,
                window_achieved_stress_per_tick: win_st,
                cumulative_achieved_influx: total_achieved_influx,
                cumulative_achieved_stress: total_achieved_stress,
                snapshot: snap,
                band_allocations,
                noisy_tv_conductance_share: noisy_tv_share,
            });
            prev_cp_tick = tick;
            prev_cp_influx = total_achieved_influx;
            prev_cp_stress = total_achieved_stress;
            next_cp_idx += 1;
        }
    }

    finalize_seed_arm_run(
        twin_id,
        TwinArm::Natural,
        horizon,
        half_tick,
        num_ambient,
        bands,
        org.snapshot(),
        total_achieved_influx,
        total_achieved_stress,
        total_achieved_influx,
        total_achieved_stress,
        first_half_influx,
        second_half_influx,
        sum_emitted_l2,
        sum_emitted_l2,
        trajectory,
        tau_hist,
        action_hist,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_seed_arm_run(
    seed: u64,
    arm: TwinArm,
    horizon: u64,
    half_tick: u64,
    num_ambient: usize,
    bands: &[BandSpec],
    final_snapshot: OrganismSnapshot,
    total_achieved_influx: f64,
    total_achieved_stress: f64,
    total_experienced_influx: f64,
    total_experienced_stress: f64,
    first_half_influx: f64,
    second_half_influx: f64,
    sum_emitted_l2: f64,
    sum_applied_l2: f64,
    developmental_trajectory: Vec<DevelopmentalSample>,
    mut tau_octave_histogram: Vec<f64>,
    trajectory_action_histogram: Vec<f64>,
) -> Result<SeedArmRun, PersistError> {
    let inv_t = 1.0 / (horizon.max(1) as f64);
    let second_half_ticks = horizon.saturating_sub(half_tick);
    let first_half_rate = first_half_influx / (half_tick.max(1) as f64);
    let second_half_rate = if second_half_ticks > 0 {
        second_half_influx / (second_half_ticks as f64)
    } else {
        first_half_rate
    };

    if tau_octave_histogram.iter().all(|&v| v <= 0.0) {
        accumulate_tau_octave_sample(&final_snapshot, &mut tau_octave_histogram);
    }

    let (final_band_allocations, final_total_sensory_conductance, final_noisy_tv_conductance_share) =
        compute_band_allocations(&final_snapshot, bands);
    let pore_channel_profile = build_pore_channel_profile(&final_snapshot, num_ambient);

    let achieved_influx_per_tick = total_achieved_influx * inv_t;
    let achieved_stress_per_tick = total_achieved_stress * inv_t;

    Ok(SeedArmRun {
        seed,
        arm,
        ticks_executed: horizon,
        total_achieved_influx,
        total_achieved_stress,
        total_experienced_influx,
        total_experienced_stress,
        achieved_influx_per_tick,
        achieved_stress_per_tick,
        net_achieved_energy_per_tick: achieved_influx_per_tick - achieved_stress_per_tick,
        first_half_influx_per_tick: first_half_rate,
        second_half_influx_per_tick: second_half_rate,
        mean_emitted_pulse_l2: sum_emitted_l2 * inv_t,
        mean_applied_pulse_l2: sum_applied_l2 * inv_t,
        final_snapshot,
        final_band_allocations,
        final_total_sensory_conductance,
        final_noisy_tv_conductance_share,
        developmental_trajectory,
        pore_channel_profile,
        tau_octave_histogram,
        trajectory_action_histogram,
    })
}

fn pulse_l2_norm(pulse: &FieldPulse) -> f64 {
    let sq_sum: f64 = pulse
        .actuator_out
        .iter()
        .map(|&v| {
            let vf = f64::from(v);
            if vf.is_finite() { vf * vf } else { 0.0 }
        })
        .sum();
    sq_sum.sqrt()
}

fn record_action_bin(pulse: &FieldPulse, hist: &mut [f64]) {
    if hist.is_empty() {
        return;
    }
    let norm = pulse_l2_norm(pulse);
    if norm < 1e-4 || pulse.actuator_out.is_empty() {
        hist[0] += 1.0;
        return;
    }
    for (idx, &val) in pulse.actuator_out.iter().enumerate() {
        let vf = f64::from(val);
        if !vf.is_finite() || vf.abs() < 1e-6 {
            continue;
        }
        let bin = if vf >= 0.0 {
            1 + 2 * idx
        } else {
            1 + 2 * idx + 1
        };
        if let Some(slot) = hist.get_mut(bin) {
            *slot += vf.abs();
        }
    }
}

fn accumulate_tau_octave_sample(snap: &OrganismSnapshot, hist: &mut [f64]) {
    if hist.is_empty() {
        return;
    }
    let max_idx = (hist.len() - 1) as f64;
    let mean_oct = (f64::from(snap.mean_tau).max(0.5).log2() + 1.0).clamp(0.0, max_idx);
    let max_oct = (f64::from(snap.max_tau).max(0.5).log2() + 1.0).clamp(0.0, max_idx);

    splat_linear(hist, mean_oct, 0.6);
    splat_linear(hist, max_oct, 0.4);

    if snap.tau_octave_count > 1 {
        let lo = (mean_oct.floor() as usize).min(hist.len() - 1);
        let hi = (max_oct.ceil() as usize).min(hist.len() - 1);
        if hi >= lo {
            let per_bin = 0.2 / ((hi - lo + 1) as f64);
            for slot in &mut hist[lo..=hi] {
                *slot += per_bin;
            }
        }
    }
}

fn splat_linear(hist: &mut [f64], pos: f64, weight: f64) {
    let lo = (pos.floor() as usize).min(hist.len() - 1);
    let hi = (pos.ceil() as usize).min(hist.len() - 1);
    let frac = pos - (lo as f64);
    hist[lo] += weight * (1.0 - frac);
    hist[hi] += weight * frac;
}

fn build_pore_channel_profile(snap: &OrganismSnapshot, num_ambient_channels: usize) -> Vec<f64> {
    let max_ch = snap
        .sensor_pore_channels
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let profile_len = num_ambient_channels.max(max_ch);
    let mut profile = vec![0.0_f64; profile_len];
    let n = snap
        .sensor_pore_channels
        .len()
        .min(snap.sensor_pore_conductances.len());
    for i in 0..n {
        let ch = snap.sensor_pore_channels[i] as usize;
        let c = f64::from(snap.sensor_pore_conductances[i]);
        if c.is_finite() && c > 0.0 && ch < profile.len() {
            profile[ch] += c;
        }
    }
    profile
}

fn summarize_arm_cohort(
    arm: TwinArm,
    ticks_per_seed: u64,
    num_ambient_channels: usize,
    bands: &[BandSpec],
    runs: Vec<SeedArmRun>,
) -> ArmCohortSummary {
    let seed_count = runs.len();
    let collect_f64 = |f: fn(&SeedArmRun) -> f64| -> EmpiricalDistribution {
        let vals: Vec<f64> = runs.iter().map(f).collect();
        EmpiricalDistribution::from_f64_slice(&vals)
    };

    let achieved_influx_per_tick = collect_f64(|r| r.achieved_influx_per_tick);
    let achieved_stress_per_tick = collect_f64(|r| r.achieved_stress_per_tick);
    let net_achieved_energy_per_tick = collect_f64(|r| r.net_achieved_energy_per_tick);
    let first_half_influx_per_tick = collect_f64(|r| r.first_half_influx_per_tick);
    let second_half_influx_per_tick = collect_f64(|r| r.second_half_influx_per_tick);
    let mean_emitted_pulse_l2 = collect_f64(|r| r.mean_emitted_pulse_l2);
    let mean_applied_pulse_l2 = collect_f64(|r| r.mean_applied_pulse_l2);

    let final_atp = collect_f64(|r| f64::from(r.final_snapshot.atp));
    let final_adenosine = collect_f64(|r| f64::from(r.final_snapshot.adenosine));
    let final_temperature = collect_f64(|r| f64::from(r.final_snapshot.temperature));
    let cell_count = collect_f64(|r| r.final_snapshot.cell_count as f64);
    let tube_count = collect_f64(|r| r.final_snapshot.tube_count as f64);
    let stiff_highway_tubes = collect_f64(|r| r.final_snapshot.stiff_highway_tubes as f64);
    let inhibitory_tubes = collect_f64(|r| r.final_snapshot.inhibitory_tubes as f64);
    let active_echo_delay_tubes = collect_f64(|r| r.final_snapshot.active_echo_delay_tubes as f64);
    let mean_tau = collect_f64(|r| f64::from(r.final_snapshot.mean_tau));
    let max_tau = collect_f64(|r| f64::from(r.final_snapshot.max_tau));
    let tau_octave_count = collect_f64(|r| r.final_snapshot.tau_octave_count as f64);
    let mean_interventional_agency =
        collect_f64(|r| f64::from(r.final_snapshot.mean_interventional_agency));
    let mean_prediction_error = collect_f64(|r| f64::from(r.final_snapshot.mean_prediction_error));
    let committed_motor_pulses = collect_f64(|r| r.final_snapshot.committed_motor_pulses as f64);
    let subthreshold_rehearsals = collect_f64(|r| r.final_snapshot.subthreshold_rehearsals as f64);
    let rest_consolidation_ticks =
        collect_f64(|r| r.final_snapshot.rest_consolidation_ticks as f64);
    let mitosis_births = collect_f64(|r| r.final_snapshot.mitosis_births as f64);
    let contradiction_splits = collect_f64(|r| r.final_snapshot.contradiction_splits as f64);
    let myelinated_shortcuts = collect_f64(|r| r.final_snapshot.myelinated_shortcuts as f64);
    let atrophied_structures = collect_f64(|r| r.final_snapshot.atrophied_structures as f64);
    let used_state_bytes = collect_f64(|r| r.final_snapshot.used_state_bytes as f64);
    let noisy_tv_conductance_share = collect_f64(|r| r.final_noisy_tv_conductance_share);

    let mut band_summaries = Vec::with_capacity(bands.len());
    for (b_idx, band) in bands.iter().enumerate() {
        let width = band.range.end.saturating_sub(band.range.start) as usize;
        let uniform_expectation = if num_ambient_channels > 0 {
            (width as f64 / num_ambient_channels as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shares: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.final_band_allocations.get(b_idx).map(|a| a.conductance_share))
            .collect();
        let pores: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.final_band_allocations.get(b_idx).map(|a| a.pore_count as f64))
            .collect();
        let delays: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.final_band_allocations.get(b_idx).map(|a| a.mean_pore_delay))
            .collect();
        let w_delays: Vec<f64> = runs
            .iter()
            .filter_map(|r| {
                r.final_band_allocations
                    .get(b_idx)
                    .map(|a| a.conductance_weighted_delay)
            })
            .collect();

        band_summaries.push(BandCohortSummary {
            label: band.label.clone(),
            role: band.role,
            channel_width: width,
            uniform_random_share_expectation: uniform_expectation,
            conductance_share: EmpiricalDistribution::from_f64_slice(&shares),
            pore_count: EmpiricalDistribution::from_f64_slice(&pores),
            mean_pore_delay: EmpiricalDistribution::from_f64_slice(&delays),
            conductance_weighted_delay: EmpiricalDistribution::from_f64_slice(&w_delays),
        });
    }

    let cp_len = runs
        .first()
        .map_or(0, |r| r.developmental_trajectory.len());
    let mut developmental_curve = Vec::with_capacity(cp_len);
    for cp_idx in 0..cp_len {
        let Some(ref_pt) = runs
            .first()
            .and_then(|r| r.developmental_trajectory.get(cp_idx))
        else {
            continue;
        };
        let tick = ref_pt.tick;
        let progress_fraction = ref_pt.progress_fraction;

        let cp_vals = |f: fn(&DevelopmentalSample) -> f64| -> EmpiricalDistribution {
            let vals: Vec<f64> = runs
                .iter()
                .filter_map(|r| r.developmental_trajectory.get(cp_idx).map(f))
                .collect();
            EmpiricalDistribution::from_f64_slice(&vals)
        };

        let mut band_shares = Vec::with_capacity(bands.len());
        let mut band_delays = Vec::with_capacity(bands.len());
        for (b_idx, band) in bands.iter().enumerate() {
            let bs: Vec<f64> = runs
                .iter()
                .filter_map(|r| {
                    r.developmental_trajectory
                        .get(cp_idx)?
                        .band_allocations
                        .get(b_idx)
                        .map(|a| a.conductance_share)
                })
                .collect();
            let bd: Vec<f64> = runs
                .iter()
                .filter_map(|r| {
                    r.developmental_trajectory
                        .get(cp_idx)?
                        .band_allocations
                        .get(b_idx)
                        .map(|a| a.conductance_weighted_delay)
                })
                .collect();
            band_shares.push((band.label.clone(), EmpiricalDistribution::from_f64_slice(&bs)));
            band_delays.push((band.label.clone(), EmpiricalDistribution::from_f64_slice(&bd)));
        }

        developmental_curve.push(DevelopmentalCohortPoint {
            tick,
            progress_fraction,
            window_influx_per_tick: cp_vals(|d| d.window_achieved_influx_per_tick),
            window_stress_per_tick: cp_vals(|d| d.window_achieved_stress_per_tick),
            atp: cp_vals(|d| f64::from(d.snapshot.atp)),
            cell_count: cp_vals(|d| d.snapshot.cell_count as f64),
            tube_count: cp_vals(|d| d.snapshot.tube_count as f64),
            mean_tau: cp_vals(|d| f64::from(d.snapshot.mean_tau)),
            tau_octave_count: cp_vals(|d| d.snapshot.tau_octave_count as f64),
            mean_interventional_agency: cp_vals(|d| {
                f64::from(d.snapshot.mean_interventional_agency)
            }),
            noisy_tv_conductance_share: cp_vals(|d| d.noisy_tv_conductance_share),
            band_conductance_shares: band_shares,
            band_weighted_delays: band_delays,
        });
    }

    let scalar_outcomes: Vec<f64> = runs.iter().map(|r| r.achieved_influx_per_tick).collect();
    let pore_profiles: Vec<Vec<f64>> = runs.iter().map(|r| r.pore_channel_profile.clone()).collect();
    let tau_hists: Vec<Vec<f64>> = runs.iter().map(|r| r.tau_octave_histogram.clone()).collect();
    let traj_hists: Vec<Vec<f64>> = runs
        .iter()
        .map(|r| r.trajectory_action_histogram.clone())
        .collect();
    let individuality = compute_individuality_divergence(
        &scalar_outcomes,
        &pore_profiles,
        &tau_hists,
        &traj_hists,
    );

    ArmCohortSummary {
        arm,
        seed_count,
        ticks_per_seed,
        achieved_influx_per_tick,
        achieved_stress_per_tick,
        net_achieved_energy_per_tick,
        first_half_influx_per_tick,
        second_half_influx_per_tick,
        mean_emitted_pulse_l2,
        mean_applied_pulse_l2,
        final_atp,
        final_adenosine,
        final_temperature,
        cell_count,
        tube_count,
        stiff_highway_tubes,
        inhibitory_tubes,
        active_echo_delay_tubes,
        mean_tau,
        max_tau,
        tau_octave_count,
        mean_interventional_agency,
        mean_prediction_error,
        committed_motor_pulses,
        subthreshold_rehearsals,
        rest_consolidation_ticks,
        mitosis_births,
        contradiction_splits,
        myelinated_shortcuts,
        atrophied_structures,
        used_state_bytes,
        noisy_tv_conductance_share,
        band_summaries,
        developmental_curve,
        individuality,
        seed_runs: runs,
    }
}

fn build_pairwise_comparisons(summaries: &[ArmCohortSummary]) -> Vec<PairwiseMetricComparison> {
    let find = |arm: TwinArm| summaries.iter().find(|s| s.arm == arm);
    let mut out = Vec::new();

    let Some(nat) = find(TwinArm::Natural) else {
        return out;
    };

    let primary_controls = [
        TwinArm::FrozenPlasticity,
        TwinArm::YokedMetabolism,
        TwinArm::ActionShuffled,
    ];

    for &ctrl_arm in &primary_controls {
        if let Some(ctrl) = find(ctrl_arm) {
            push_comparison(
                &mut out,
                "achieved_influx_per_tick",
                nat,
                ctrl,
                |r| r.achieved_influx_per_tick,
            );
            push_comparison(
                &mut out,
                "net_achieved_energy_per_tick",
                nat,
                ctrl,
                |r| r.net_achieved_energy_per_tick,
            );
            push_comparison(
                &mut out,
                "second_half_influx_per_tick",
                nat,
                ctrl,
                |r| r.second_half_influx_per_tick,
            );
            push_comparison(
                &mut out,
                "mean_interventional_agency",
                nat,
                ctrl,
                |r| f64::from(r.final_snapshot.mean_interventional_agency),
            );
            push_comparison(
                &mut out,
                "noisy_tv_conductance_share",
                nat,
                ctrl,
                |r| r.final_noisy_tv_conductance_share,
            );
        }
    }

    if let (Some(unlesioned), Some(lesioned)) = (
        find(TwinArm::UnlesionedPostLife),
        find(TwinArm::FlaglessPostLifeLesion),
    ) {
        push_comparison(
            &mut out,
            "post_life_achieved_influx_per_tick",
            unlesioned,
            lesioned,
            |r| r.achieved_influx_per_tick,
        );
        push_comparison(
            &mut out,
            "post_life_net_energy_per_tick",
            unlesioned,
            lesioned,
            |r| r.net_achieved_energy_per_tick,
        );
    }

    if let (Some(lesioned), Some(frozen)) = (
        find(TwinArm::FlaglessPostLifeLesion),
        find(TwinArm::FrozenPlasticity),
    ) {
        push_comparison(
            &mut out,
            "lesion_vs_frozen_influx_per_tick",
            lesioned,
            frozen,
            |r| r.achieved_influx_per_tick,
        );
    }

    out
}

fn push_comparison(
    out: &mut Vec<PairwiseMetricComparison>,
    metric_name: &str,
    cohort_a: &ArmCohortSummary,
    cohort_b: &ArmCohortSummary,
    extract: fn(&SeedArmRun) -> f64,
) {
    let vals_a: Vec<f64> = cohort_a.seed_runs.iter().map(extract).collect();
    let vals_b: Vec<f64> = cohort_b.seed_runs.iter().map(extract).collect();
    out.push(PairwiseMetricComparison {
        metric_name: metric_name.to_string(),
        arm_a: cohort_a.arm,
        arm_b: cohort_b.arm,
        comparison: mann_whitney_u(&vals_a, &vals_b),
    });
}
