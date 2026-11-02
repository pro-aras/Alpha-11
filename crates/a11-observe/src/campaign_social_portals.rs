//! Wave B empirical campaign studies 4 through 6:
//! - Study 4: `HarmonicWaveStream` 1D phase-resonance (`6` controls) & Identical-Twin Constrained-Randomness Individuality (`1e-4` vs `0.0` jitter).
//! - Study 5: Multi-generational cultural ratchet in `MultiGenTerrarium` (`CohabitingNewborn` vs `SoloOrphan`, `PhaseMatchedSoloOrphan`, `CohabitingFrozenNewborn`).
//! - Study 6: Unforced Layer-2 human-activity portals (`DrawingScratchpad`, `PatternMemoryBoard`, `RhythmEchoPad`, `BalanceCatchToy`, `CompositePortalMedium`).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use a11_core::Alpha11Organism;
use a11_medium::{ControlMode, FieldPulse, OrganismEngine, WorldMedium};
use a11_portals::{
    BalanceCatchToy, CompositePortalMedium, DrawingScratchpad, PatternMemoryBoard, RhythmEchoPad,
};
use a11_worlds::{
    BrightRadiantWorld, HarmonicMetrics, HarmonicWaveStream, MultiGenTerrarium,
};

use crate::campaign::{CampaignError, cave_bands_128, make_default_organism};
use crate::report::{
    escape_json_str, format_identical_twins_json, format_identical_twins_markdown,
    format_twin_study_json, format_twin_study_markdown, write_dist_json, write_indent,
    write_kv_f64, write_kv_str, write_kv_u64, write_kv_usize, write_mann_whitney_json,
};
use crate::stats::{EmpiricalDistribution, MannWhitneyComparison, mann_whitney_u, quantile_sorted};
use crate::twins::{
    BandRole, BandSpec, DEFAULT_TWELVE_SEEDS, IdenticalTwinConfig, IdenticalTwinReport,
    PulseTemporalChannelShuffler, TwinArm, TwinStudyConfig, TwinStudyReport,
    compute_band_allocations, observe_identical_twin_divergence, run_twin_study_instrumented,
};

const ACTION_SHUFFLE_PORTAL_SALT: u64 = 0x5A17_A110_9087_A155;
const ELDER_SEED_SALT_0: u64 = 0xE1DE_8000_0000_0001;
const ELDER_SEED_SALT_1: u64 = 0xE1DE_8000_0000_0002;
const NEWBORN_SEED_SALT: u64 = 0x4E57_B084_A110_0003;

/// Summary of `HarmonicMetrics` across seeds for one `TwinArm`.
#[derive(Clone, Debug)]
pub struct HarmonicArmSummary {
    pub arm: TwinArm,
    pub in_phase_locks: EmpiricalDistribution,
    pub anti_phase_collisions: EmpiricalDistribution,
    pub echo_energy_received: EmpiricalDistribution,
    pub raw_locks: Vec<f64>,
    pub raw_collisions: Vec<f64>,
}

/// Complete Study 4 outcome.
#[derive(Clone, Debug)]
pub struct Study4Outcome {
    pub harmonic_report: TwinStudyReport,
    pub harmonic_by_arm: Vec<HarmonicArmSummary>,
    pub identical_twins_jittered: IdenticalTwinReport,
    pub identical_twins_zero_jitter: IdenticalTwinReport,
}

/// Runs Study 4 (`HarmonicWaveStream` + Identical-Twin Constrained-Randomness Individuality).
pub fn run_study_4_harmonic_and_individuality(
    output_dir: &Path,
) -> Result<Study4Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;

    let harmonic_bands = vec![
        BandSpec::new(
            "Traveling Wave (0..48)",
            0..48,
            BandRole::StructuredPhysical,
        ),
        BandSpec::new("Delayed Resonance (48..64)", 48..64, BandRole::ActiveEcho),
        BandSpec::new(
            "Noisy TV Static (64..96)",
            64..96,
            BandRole::NoisyTvDistractor,
        ),
    ];

    let mut cfg_4a = TwinStudyConfig::new(
        "04a HarmonicWaveStream (96ch: 1D Traveling Waves + 3..8 Tick Delayed Resonance + NoisyTV)",
        1_800,
        harmonic_bands,
    );
    cfg_4a.post_life_eval_ticks = 450;

    let mut harm_rec: HashMap<(u64, TwinArm), HarmonicMetrics> = HashMap::new();
    let harmonic_report = run_twin_study_instrumented(
        &cfg_4a,
        make_default_organism,
        HarmonicWaveStream::new,
        |_, _, _, _, _| {},
        |_, _, _| {},
        |ch, _| ch,
        |m, _| m,
        |run, _org, world| {
            harm_rec.insert((run.seed, run.arm), world.metrics().clone());
        },
    )?;

    let mut harmonic_by_arm = Vec::with_capacity(TwinArm::ALL.len());
    for arm in TwinArm::ALL {
        let mut locks = Vec::new();
        let mut colls = Vec::new();
        let mut echo_e = Vec::new();
        for &seed in &cfg_4a.seeds {
            if let Some(m) = harm_rec.get(&(seed, arm)) {
                locks.push(m.in_phase_locks as f64);
                colls.push(m.anti_phase_collisions as f64);
                echo_e.push(f64::from(m.echo_energy_received));
            }
        }
        harmonic_by_arm.push(HarmonicArmSummary {
            arm,
            in_phase_locks: EmpiricalDistribution::from_f64_slice(&locks),
            anti_phase_collisions: EmpiricalDistribution::from_f64_slice(&colls),
            echo_energy_received: EmpiricalDistribution::from_f64_slice(&echo_e),
            raw_locks: locks,
            raw_collisions: colls,
        });
    }

    // Part B: 12 genetically identical twins in BrightRadiantWorld over 2,000 ticks
    // (1) epsilon = 1e-4 on ticks 1..=3
    let id_cfg_jitter = IdenticalTwinConfig::default_twelve(777, 2_000, cave_bands_128());
    let identical_twins_jittered = observe_identical_twin_divergence(
        &id_cfg_jitter,
        make_default_organism,
        BrightRadiantWorld::new,
    )?;

    // (2) epsilon = 0.0 determinism proof (must show 0.0000 divergence!)
    let mut id_cfg_zero = id_cfg_jitter.clone();
    id_cfg_zero.perturbation_epsilon = 0.0;
    id_cfg_zero.perturbation_ticks = 0;
    let identical_twins_zero_jitter = observe_identical_twin_divergence(
        &id_cfg_zero,
        make_default_organism,
        BrightRadiantWorld::new,
    )?;

    let mut md = format_twin_study_markdown(&harmonic_report);
    let _ = writeln!(md);
    let _ = writeln!(md, "## 7. `HarmonicWaveStream` Phase-Lock & Delayed-Resonance Metrics");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Control Arm | In-Phase Locks (`p50 [IQR]`) | Anti-Phase Collisions (`p50 [IQR]`) | Delayed Echo Energy (`p50 [IQR]`) |"
    );
    let _ = writeln!(md, "| :--- | :--- | :--- | :--- |");
    for h in &harmonic_by_arm {
        let _ = writeln!(
            md,
            "| `{}` | `{:.1} [{:.1}]` | `{:.1} [{:.1}]` | `{:.3} [{:.3}]` |",
            h.arm.as_str(),
            h.in_phase_locks.p50,
            h.in_phase_locks.iqr,
            h.anti_phase_collisions.p50,
            h.anti_phase_collisions.iqr,
            h.echo_energy_received.p50,
            h.echo_energy_received.iqr,
        );
    }

    if let (Some(hn), Some(hf), Some(hs)) = (
        harmonic_by_arm.iter().find(|x| x.arm == TwinArm::Natural),
        harmonic_by_arm
            .iter()
            .find(|x| x.arm == TwinArm::FrozenPlasticity),
        harmonic_by_arm
            .iter()
            .find(|x| x.arm == TwinArm::ActionShuffled),
    ) {
        let c_nf = mann_whitney_u(&hn.raw_locks, &hf.raw_locks);
        let c_ns = mann_whitney_u(&hn.raw_locks, &hs.raw_locks);
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "- **`in_phase_locks` `Natural` vs `FrozenPlasticity`**: `diff = {:+.2}`, `r_rb = {:+.4}`, `p_perm = {:.4}`",
            c_nf.median_diff, c_nf.rank_biserial, c_nf.p_permutation
        );
        let _ = writeln!(
            md,
            "- **`in_phase_locks` `Natural` vs `ActionShuffled`**: `diff = {:+.2}`, `r_rb = {:+.4}`, `p_perm = {:.4}`",
            c_ns.median_diff, c_ns.rank_biserial, c_ns.p_permutation
        );
    }

    let _ = writeln!(md);
    let _ = writeln!(md, "---");
    let _ = writeln!(md);
    md.push_str(&format_identical_twins_markdown(&identical_twins_jittered));
    let _ = writeln!(md);
    let z_ind = &identical_twins_zero_jitter.cohort_summary.individuality;
    let _ = writeln!(
        md,
        "### Zero-Perturbation Determinism Control (`epsilon = 0.0`, `N = 12` Identical Twins)"
    );
    let _ = writeln!(
        md,
        "- **Outcome `CV`**: `{:.6}` | **Pairwise Pore Cosine Dist**: `{:.6}` | **Pairwise `tau` JSD**: `{:.6}` | **Inter-Twin Trajectory JSD**: `{:.6}` | **Composite Individuality**: `{:.6}`",
        z_ind.inter_individual_cv,
        z_ind.mean_pairwise_pore_cosine_distance,
        z_ind.mean_pairwise_tau_jsd,
        z_ind.trajectory_divergence_bits,
        z_ind.composite_individuality_index,
    );

    let mut json = String::with_capacity(49_152);
    json.push_str("{\n");
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"harmonic_twin_study\": {},",
        format_twin_study_json(&harmonic_report)
    );
    write_indent(&mut json, 1);
    json.push_str("\"harmonic_metrics_by_arm\": [\n");
    for (idx, h) in harmonic_by_arm.iter().enumerate() {
        write_indent(&mut json, 2);
        json.push_str("{\n");
        write_kv_str(&mut json, 3, "arm", h.arm.as_str(), true);
        write_dist_json(&mut json, 3, "in_phase_locks", &h.in_phase_locks, true);
        write_dist_json(
            &mut json,
            3,
            "anti_phase_collisions",
            &h.anti_phase_collisions,
            true,
        );
        write_dist_json(
            &mut json,
            3,
            "echo_energy_received",
            &h.echo_energy_received,
            false,
        );
        write_indent(&mut json, 2);
        if idx + 1 < harmonic_by_arm.len() {
            json.push_str("},\n");
        } else {
            json.push_str("}\n");
        }
    }
    write_indent(&mut json, 1);
    json.push_str("],\n");
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"identical_twins_jitter_1e_4\": {},",
        format_identical_twins_json(&identical_twins_jittered)
    );
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"identical_twins_zero_jitter_control\": {}",
        format_identical_twins_json(&identical_twins_zero_jitter)
    );
    json.push_str("}\n");

    fs::write(output_dir.join("04_harmonic_and_individuality.md"), md)?;
    fs::write(output_dir.join("04_harmonic_and_individuality.json"), json)?;

    Ok(Study4Outcome {
        harmonic_report,
        harmonic_by_arm,
        identical_twins_jittered,
        identical_twins_zero_jitter,
    })
}

/// Experimental condition in Study 5 (`MultiGenTerrarium` Cultural Ratchet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrariumCohortArm {
    CohabitingNewborn,
    SoloOrphan,
    PhaseMatchedSoloOrphan,
    CohabitingFrozenNewborn,
}

impl TerrariumCohortArm {
    pub const ALL: [Self; 4] = [
        Self::CohabitingNewborn,
        Self::SoloOrphan,
        Self::PhaseMatchedSoloOrphan,
        Self::CohabitingFrozenNewborn,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CohabitingNewborn => "CohabitingNewborn",
            Self::SoloOrphan => "SoloOrphan",
            Self::PhaseMatchedSoloOrphan => "PhaseMatchedSoloOrphan",
            Self::CohabitingFrozenNewborn => "CohabitingFrozenNewborn",
        }
    }
}

#[derive(Clone, Debug)]
struct TerrariumSeedRun {
    influx_per_tick: f64,
    first_half_influx_per_tick: f64,
    second_half_influx_per_tick: f64,
    stress_per_tick: f64,
    night_stress_per_night_tick: f64,
    final_atp: f64,
    final_cells: f64,
    final_agency: f64,
    noisy_tv_share: f64,
    persistent_mark_coverage: f64,
    total_mark_mass: f64,
    mark_corridor_alignment: f64,
    elder_newborn_acoustic_synchrony: f64,
    elder_newborn_motor_synchrony: f64,
}

/// Cohort summary for one arm in Study 5.
#[derive(Clone, Debug)]
pub struct TerrariumArmSummary {
    pub arm: TerrariumCohortArm,
    pub influx_per_tick: EmpiricalDistribution,
    pub first_half_influx_per_tick: EmpiricalDistribution,
    pub second_half_influx_per_tick: EmpiricalDistribution,
    pub stress_per_tick: EmpiricalDistribution,
    pub night_stress_per_night_tick: EmpiricalDistribution,
    pub final_atp: EmpiricalDistribution,
    pub final_cells: EmpiricalDistribution,
    pub final_agency: EmpiricalDistribution,
    pub noisy_tv_share: EmpiricalDistribution,
    pub persistent_mark_coverage: EmpiricalDistribution,
    pub total_mark_mass: EmpiricalDistribution,
    pub mark_corridor_alignment: EmpiricalDistribution,
    pub elder_newborn_acoustic_synchrony: EmpiricalDistribution,
    pub elder_newborn_motor_synchrony: EmpiricalDistribution,
    raw_influx: Vec<f64>,
    raw_stress: Vec<f64>,
    raw_night_stress: Vec<f64>,
    raw_atp: Vec<f64>,
    raw_corridor: Vec<f64>,
}

/// Complete Study 5 outcome.
#[derive(Clone, Debug)]
pub struct Study5Outcome {
    pub elder_pre_mark_coverage: EmpiricalDistribution,
    pub elder_pre_corridor_alignment: EmpiricalDistribution,
    pub arm_summaries: Vec<TerrariumArmSummary>,
    pub comparisons: Vec<(String, TerrariumCohortArm, TerrariumCohortArm, MannWhitneyComparison)>,
}

/// Runs Study 5 (`05_cultural_terrarium`: Multi-Generational Cultural Ratchet across `12` seeds).
pub fn run_study_5_cultural_terrarium(output_dir: &Path) -> Result<Study5Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;
    let elder_pre_ticks = 1_500_u64;
    let newborn_ticks = 1_500_u64;
    let bands = cave_bands_128();

    let mut pre_cov = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
    let mut pre_align = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());

    let mut cohab_runs = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
    let mut solo_runs = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
    let mut phase_solo_runs = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
    let mut cohab_frozen_runs = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());

    for &seed in &DEFAULT_TWELVE_SEEDS {
        // 1. Pre-live 2 Natural elders for 1,500 ticks in MultiGenTerrarium::new(seed, 2).
        let mut elder_terrarium = MultiGenTerrarium::new(seed, 2);
        let mut elders = vec![
            make_default_organism(seed ^ ELDER_SEED_SALT_0),
            make_default_organism(seed ^ ELDER_SEED_SALT_1),
        ];
        for _ in 0..elder_pre_ticks {
            let _ = elder_terrarium.step_population(&mut elders)?;
        }
        pre_cov.push(f64::from(elder_terrarium.metrics().persistent_mark_coverage));
        pre_align.push(f64::from(elder_terrarium.metrics().mark_corridor_alignment));

        // Clone habitat + elder checkpoints for identical starting state in CohabitingFrozenNewborn!
        let mut frozen_terrarium = elder_terrarium.clone();
        let elder0_ckpt = elders[0].encode_checkpoint()?;
        let elder1_ckpt = elders[1].encode_checkpoint()?;

        // Arm A: CohabitingNewborn (Natural newborn in slot 2 alongside the 2 elders)
        elder_terrarium.respawn_newborn_in_slot(2);
        let mut pop_cohab = elders;
        let mut newborn_nat = make_default_organism(seed ^ NEWBORN_SEED_SALT);
        newborn_nat.set_control_mode(ControlMode::Natural);
        pop_cohab.push(newborn_nat);
        cohab_runs.push(step_newborn_in_terrarium(
            &mut elder_terrarium,
            &mut pop_cohab,
            2,
            newborn_ticks,
            &bands,
        )?);

        // Arm D: CohabitingFrozenNewborn (FrozenPlasticity newborn in slot 2 alongside identical elders)
        let mut e0_clone = make_default_organism(seed ^ 0x9999_0001);
        let mut e1_clone = make_default_organism(seed ^ 0x9999_0002);
        e0_clone.restore_checkpoint(&elder0_ckpt)?;
        e1_clone.restore_checkpoint(&elder1_ckpt)?;
        frozen_terrarium.respawn_newborn_in_slot(2);
        let mut newborn_froz = make_default_organism(seed ^ NEWBORN_SEED_SALT);
        newborn_froz.set_control_mode(ControlMode::FrozenPlasticity);
        let mut pop_froz = vec![e0_clone, e1_clone, newborn_froz];
        cohab_frozen_runs.push(step_newborn_in_terrarium(
            &mut frozen_terrarium,
            &mut pop_froz,
            2,
            newborn_ticks,
            &bands,
        )?);

        // Arm B: SoloOrphan (Natural newborn alone in fresh unmarked MultiGenTerrarium::new(seed, 1))
        let mut solo_terrarium = MultiGenTerrarium::new(seed, 1);
        let mut newborn_solo = make_default_organism(seed ^ NEWBORN_SEED_SALT);
        newborn_solo.set_control_mode(ControlMode::Natural);
        let mut pop_solo = vec![newborn_solo];
        solo_runs.push(step_newborn_in_terrarium(
            &mut solo_terrarium,
            &mut pop_solo,
            0,
            newborn_ticks,
            &bands,
        )?);

        // Arm C: PhaseMatchedSoloOrphan (unmarked terrarium silently advanced 1,500 ticks, then newborn spawned)
        let mut phase_terrarium = MultiGenTerrarium::new(seed, 1);
        let silent_pulse = [FieldPulse::silent(6)];
        for _ in 0..elder_pre_ticks {
            phase_terrarium.apply_population_pulses(&silent_pulse);
        }
        phase_terrarium.respawn_newborn_in_slot(0);
        let mut newborn_phase = make_default_organism(seed ^ NEWBORN_SEED_SALT);
        newborn_phase.set_control_mode(ControlMode::Natural);
        let mut pop_phase = vec![newborn_phase];
        phase_solo_runs.push(step_newborn_in_terrarium(
            &mut phase_terrarium,
            &mut pop_phase,
            0,
            newborn_ticks,
            &bands,
        )?);
    }

    let arm_summaries = vec![
        summarize_terrarium_arm(TerrariumCohortArm::CohabitingNewborn, &cohab_runs),
        summarize_terrarium_arm(TerrariumCohortArm::SoloOrphan, &solo_runs),
        summarize_terrarium_arm(
            TerrariumCohortArm::PhaseMatchedSoloOrphan,
            &phase_solo_runs,
        ),
        summarize_terrarium_arm(
            TerrariumCohortArm::CohabitingFrozenNewborn,
            &cohab_frozen_runs,
        ),
    ];

    let mut comparisons = Vec::new();
    let pairs = [
        (
            TerrariumCohortArm::CohabitingNewborn,
            TerrariumCohortArm::SoloOrphan,
        ),
        (
            TerrariumCohortArm::CohabitingNewborn,
            TerrariumCohortArm::PhaseMatchedSoloOrphan,
        ),
        (
            TerrariumCohortArm::CohabitingNewborn,
            TerrariumCohortArm::CohabitingFrozenNewborn,
        ),
    ];
    for (a, b) in pairs {
        let sa = arm_summaries.iter().find(|s| s.arm == a);
        let sb = arm_summaries.iter().find(|s| s.arm == b);
        if let (Some(xa), Some(xb)) = (sa, sb) {
            comparisons.push((
                "newborn_influx_per_tick".to_string(),
                a,
                b,
                mann_whitney_u(&xa.raw_influx, &xb.raw_influx),
            ));
            comparisons.push((
                "newborn_stress_per_tick".to_string(),
                a,
                b,
                mann_whitney_u(&xa.raw_stress, &xb.raw_stress),
            ));
            comparisons.push((
                "newborn_night_stress_per_night_tick".to_string(),
                a,
                b,
                mann_whitney_u(&xa.raw_night_stress, &xb.raw_night_stress),
            ));
            comparisons.push((
                "newborn_final_atp".to_string(),
                a,
                b,
                mann_whitney_u(&xa.raw_atp, &xb.raw_atp),
            ));
            comparisons.push((
                "mark_corridor_alignment".to_string(),
                a,
                b,
                mann_whitney_u(&xa.raw_corridor, &xb.raw_corridor),
            ));
        }
    }

    let outcome = Study5Outcome {
        elder_pre_mark_coverage: EmpiricalDistribution::from_f64_slice(&pre_cov),
        elder_pre_corridor_alignment: EmpiricalDistribution::from_f64_slice(&pre_align),
        arm_summaries,
        comparisons,
    };

    write_study5_files(output_dir, &outcome)?;
    Ok(outcome)
}

fn step_newborn_in_terrarium<O: OrganismEngine>(
    terrarium: &mut MultiGenTerrarium,
    population: &mut [O],
    newborn_slot: usize,
    ticks: u64,
    bands: &[BandSpec],
) -> Result<TerrariumSeedRun, CampaignError> {
    let half = (ticks / 2).max(1);
    let mut total_in = 0.0_f64;
    let mut h1_in = 0.0_f64;
    let mut h2_in = 0.0_f64;
    let mut total_st = 0.0_f64;
    let mut night_st = 0.0_f64;
    let mut night_ticks = 0_u64;

    for t in 1..=ticks {
        let _ = terrarium.step_population(population)?;
        let nb_view = terrarium.agent_view(newborn_slot);
        let inf = f64::from(nb_view.metabolic_influx);
        let st = f64::from(nb_view.tissue_stress);
        total_in += inf;
        total_st += st;
        if t <= half {
            h1_in += inf;
        } else {
            h2_in += inf;
        }
        if terrarium.illumination() <= 0.10 {
            night_st += st;
            night_ticks += 1;
        }
    }

    let snap = population[newborn_slot].snapshot();
    let (_, _, noisy_tv_share) = compute_band_allocations(&snap, bands);
    let m = terrarium.metrics();
    let inv_t = 1.0 / (ticks.max(1) as f64);

    Ok(TerrariumSeedRun {
        influx_per_tick: total_in * inv_t,
        first_half_influx_per_tick: h1_in / (half as f64),
        second_half_influx_per_tick: h2_in / (ticks.saturating_sub(half).max(1) as f64),
        stress_per_tick: total_st * inv_t,
        night_stress_per_night_tick: if night_ticks > 0 {
            night_st / (night_ticks as f64)
        } else {
            0.0
        },
        final_atp: f64::from(snap.atp),
        final_cells: snap.cell_count as f64,
        final_agency: f64::from(snap.mean_interventional_agency),
        noisy_tv_share,
        persistent_mark_coverage: f64::from(m.persistent_mark_coverage),
        total_mark_mass: f64::from(m.total_mark_mass),
        mark_corridor_alignment: f64::from(m.mark_corridor_alignment),
        elder_newborn_acoustic_synchrony: f64::from(m.elder_newborn_acoustic_synchrony),
        elder_newborn_motor_synchrony: f64::from(m.elder_newborn_motor_synchrony),
    })
}

fn summarize_terrarium_arm(
    arm: TerrariumCohortArm,
    runs: &[TerrariumSeedRun],
) -> TerrariumArmSummary {
    let dist = |f: fn(&TerrariumSeedRun) -> f64| -> EmpiricalDistribution {
        let vals: Vec<f64> = runs.iter().map(f).collect();
        EmpiricalDistribution::from_f64_slice(&vals)
    };
    TerrariumArmSummary {
        arm,
        influx_per_tick: dist(|r| r.influx_per_tick),
        first_half_influx_per_tick: dist(|r| r.first_half_influx_per_tick),
        second_half_influx_per_tick: dist(|r| r.second_half_influx_per_tick),
        stress_per_tick: dist(|r| r.stress_per_tick),
        night_stress_per_night_tick: dist(|r| r.night_stress_per_night_tick),
        final_atp: dist(|r| r.final_atp),
        final_cells: dist(|r| r.final_cells),
        final_agency: dist(|r| r.final_agency),
        noisy_tv_share: dist(|r| r.noisy_tv_share),
        persistent_mark_coverage: dist(|r| r.persistent_mark_coverage),
        total_mark_mass: dist(|r| r.total_mark_mass),
        mark_corridor_alignment: dist(|r| r.mark_corridor_alignment),
        elder_newborn_acoustic_synchrony: dist(|r| r.elder_newborn_acoustic_synchrony),
        elder_newborn_motor_synchrony: dist(|r| r.elder_newborn_motor_synchrony),
        raw_influx: runs.iter().map(|r| r.influx_per_tick).collect(),
        raw_stress: runs.iter().map(|r| r.stress_per_tick).collect(),
        raw_night_stress: runs.iter().map(|r| r.night_stress_per_night_tick).collect(),
        raw_atp: runs.iter().map(|r| r.final_atp).collect(),
        raw_corridor: runs.iter().map(|r| r.mark_corridor_alignment).collect(),
    }
}

fn write_study5_files(output_dir: &Path, out: &Study5Outcome) -> std::io::Result<()> {
    let mut md = String::with_capacity(12_288);
    let _ = writeln!(
        md,
        "# Study 5: Multi-Generational Cultural Ratchet in `MultiGenTerrarium` (`N = 12` Seeds)"
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **Elder Pre-Habitation (`1,500` ticks, `2` `Natural` Elders)**: Mark Coverage `p50 = {:.4} [{:.4}]`, Corridor Alignment `p50 = {:.4} [{:.4}]`",
        out.elder_pre_mark_coverage.p50,
        out.elder_pre_mark_coverage.iqr,
        out.elder_pre_corridor_alignment.p50,
        out.elder_pre_corridor_alignment.iqr,
    );
    let _ = writeln!(
        md,
        "- **Newborn Evaluation Horizon**: `1,500` ticks (`32`-cell initial newborn seed matched across all 4 arms)"
    );
    let _ = writeln!(md);
    let _ = writeln!(md, "## 1. Newborn Metabolic, Thermal Stress & Cultural Niche Telemetry");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Newborn Cohort Arm | Influx/Tick (`p50 [IQR]`) | 1st->2nd Half Influx | Stress/Tick (`p50 [IQR]`) | Night Stress/Tick (`lux <= 0.10`, `p50 [IQR]`) | Final ATP (`p50`) | Cells (`p50`) | Agency (`p50`) | Noisy-TV Share (`p50`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |"
    );
    for s in &out.arm_summaries {
        let _ = writeln!(
            md,
            "| `{}` | `{:.4} [{:.4}]` | `{:.4} -> {:.4}` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:.3}` | `{:.1}` | `{:.4}` | `{:.1}%` |",
            s.arm.as_str(),
            s.influx_per_tick.p50,
            s.influx_per_tick.iqr,
            s.first_half_influx_per_tick.p50,
            s.second_half_influx_per_tick.p50,
            s.stress_per_tick.p50,
            s.stress_per_tick.iqr,
            s.night_stress_per_night_tick.p50,
            s.night_stress_per_night_tick.iqr,
            s.final_atp.p50,
            s.final_cells.p50,
            s.final_agency.p50,
            s.noisy_tv_share.p50 * 100.0,
        );
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## 2. Persistent Surface Marks & Elder-Newborn Synchrony");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Newborn Cohort Arm | Mark Coverage (`p50 [IQR]`) | Total Mark Mass (`p50 [IQR]`) | Corridor Alignment (`p50 [IQR]`) | Elder-Newborn Acoustic Sync (`p50 [IQR]`) | Elder-Newborn Motor Sync (`p50 [IQR]`) |"
    );
    let _ = writeln!(md, "| :--- | :--- | :--- | :--- | :--- | :--- |");
    for s in &out.arm_summaries {
        let _ = writeln!(
            md,
            "| `{}` | `{:.4} [{:.4}]` | `{:.3} [{:.3}]` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` |",
            s.arm.as_str(),
            s.persistent_mark_coverage.p50,
            s.persistent_mark_coverage.iqr,
            s.total_mark_mass.p50,
            s.total_mark_mass.iqr,
            s.mark_corridor_alignment.p50,
            s.mark_corridor_alignment.iqr,
            s.elder_newborn_acoustic_synchrony.p50,
            s.elder_newborn_acoustic_synchrony.iqr,
            s.elder_newborn_motor_synchrony.p50,
            s.elder_newborn_motor_synchrony.iqr,
        );
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "## 3. Non-Parametric Mann-Whitney U Comparisons");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Metric | Arm A | Arm B | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |"
    );
    let _ = writeln!(md, "| :--- | :--- | :--- | ---: | ---: | ---: |");
    for (metric, a, b, c) in &out.comparisons {
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{}` | `{:+.5}` | `{:+.4}` | `{:.4}` |",
            metric,
            a.as_str(),
            b.as_str(),
            c.median_diff,
            c.rank_biserial,
            c.p_permutation
        );
    }

    let mut json = String::with_capacity(24_576);
    json.push_str("{\n");
    write_dist_json(
        &mut json,
        1,
        "elder_pre_mark_coverage",
        &out.elder_pre_mark_coverage,
        true,
    );
    write_dist_json(
        &mut json,
        1,
        "elder_pre_corridor_alignment",
        &out.elder_pre_corridor_alignment,
        true,
    );
    write_indent(&mut json, 1);
    json.push_str("\"arm_summaries\": [\n");
    for (idx, s) in out.arm_summaries.iter().enumerate() {
        write_indent(&mut json, 2);
        json.push_str("{\n");
        write_kv_str(&mut json, 3, "arm", s.arm.as_str(), true);
        write_dist_json(&mut json, 3, "influx_per_tick", &s.influx_per_tick, true);
        write_dist_json(&mut json, 3, "stress_per_tick", &s.stress_per_tick, true);
        write_dist_json(
            &mut json,
            3,
            "night_stress_per_night_tick",
            &s.night_stress_per_night_tick,
            true,
        );
        write_dist_json(&mut json, 3, "final_atp", &s.final_atp, true);
        write_dist_json(&mut json, 3, "final_cells", &s.final_cells, true);
        write_dist_json(&mut json, 3, "final_agency", &s.final_agency, true);
        write_dist_json(&mut json, 3, "noisy_tv_share", &s.noisy_tv_share, true);
        write_dist_json(
            &mut json,
            3,
            "persistent_mark_coverage",
            &s.persistent_mark_coverage,
            true,
        );
        write_dist_json(
            &mut json,
            3,
            "mark_corridor_alignment",
            &s.mark_corridor_alignment,
            true,
        );
        write_dist_json(
            &mut json,
            3,
            "elder_newborn_acoustic_synchrony",
            &s.elder_newborn_acoustic_synchrony,
            true,
        );
        write_dist_json(
            &mut json,
            3,
            "elder_newborn_motor_synchrony",
            &s.elder_newborn_motor_synchrony,
            false,
        );
        write_indent(&mut json, 2);
        if idx + 1 < out.arm_summaries.len() {
            json.push_str("},\n");
        } else {
            json.push_str("}\n");
        }
    }
    write_indent(&mut json, 1);
    json.push_str("],\n");
    write_indent(&mut json, 1);
    json.push_str("\"comparisons\": [\n");
    for (idx, (metric, a, b, c)) in out.comparisons.iter().enumerate() {
        write_indent(&mut json, 2);
        json.push_str("{\n");
        write_kv_str(&mut json, 3, "metric", metric, true);
        write_kv_str(&mut json, 3, "arm_a", a.as_str(), true);
        write_kv_str(&mut json, 3, "arm_b", b.as_str(), true);
        write_mann_whitney_json(&mut json, 3, "stats", c, false);
        write_indent(&mut json, 2);
        if idx + 1 < out.comparisons.len() {
            json.push_str("},\n");
        } else {
            json.push_str("}\n");
        }
    }
    write_indent(&mut json, 1);
    json.push_str("]\n}\n");

    fs::write(output_dir.join("05_cultural_terrarium.md"), md)?;
    fs::write(output_dir.join("05_cultural_terrarium.json"), json)?;
    Ok(())
}

// =========================================================================
// Study 6: Unforced Human-Activity Portals (Scratchpad, Pattern, Rhythm, Balance, Composite)
// =========================================================================

/// Control arms evaluated across each portal in Study 6.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortalArm {
    Natural,
    FrozenPlasticity,
    ActionShuffled,
}

impl PortalArm {
    pub const ALL: [Self; 3] = [Self::Natural, Self::FrozenPlasticity, Self::ActionShuffled];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "Natural",
            Self::FrozenPlasticity => "FrozenPlasticity",
            Self::ActionShuffled => "ActionShuffled",
        }
    }
}

/// Named empirical metric distribution across 3 portal arms plus `Natural` vs `Frozen` and `Natural` vs `Shuffled` tests.
#[derive(Clone, Debug)]
pub struct PortalMetricBlock {
    pub metric_name: String,
    pub natural: EmpiricalDistribution,
    pub frozen: EmpiricalDistribution,
    pub shuffled: EmpiricalDistribution,
    pub nat_vs_frozen: MannWhitneyComparison,
    pub nat_vs_shuffled: MannWhitneyComparison,
}

/// Complete telemetry for one portal experiment in Study 6.
#[derive(Clone, Debug)]
pub struct PortalExperimentSummary {
    pub portal_name: String,
    pub num_channels: usize,
    pub num_actuators: usize,
    pub uniform_noisy_tv_share: f64,
    pub metrics: Vec<PortalMetricBlock>,
}

/// Complete Study 6 outcome across all 5 portal configurations.
#[derive(Clone, Debug)]
pub struct Study6Outcome {
    pub portals: Vec<PortalExperimentSummary>,
}

fn build_metric_block(
    name: &str,
    nat: &[f64],
    froz: &[f64],
    shuf: &[f64],
) -> PortalMetricBlock {
    PortalMetricBlock {
        metric_name: name.to_string(),
        natural: EmpiricalDistribution::from_f64_slice(nat),
        frozen: EmpiricalDistribution::from_f64_slice(froz),
        shuffled: EmpiricalDistribution::from_f64_slice(shuf),
        nat_vs_frozen: mann_whitney_u(nat, froz),
        nat_vs_shuffled: mann_whitney_u(nat, shuf),
    }
}

fn step_portal_run<W, FTick>(
    seed: u64,
    arm: PortalArm,
    ticks: u64,
    mut world: W,
    bands: &[BandSpec],
    mut on_after_tick: FTick,
) -> Result<(W, a11_medium::OrganismSnapshot, f64, f64), CampaignError>
where
    W: WorldMedium,
    FTick: FnMut(u64, &Alpha11Organism, &W),
{
    let mut org = make_default_organism(seed);
    match arm {
        PortalArm::Natural | PortalArm::ActionShuffled => {
            org.set_control_mode(ControlMode::Natural);
        }
        PortalArm::FrozenPlasticity => {
            org.set_control_mode(ControlMode::FrozenPlasticity);
        }
    }
    let mut shuffler = (arm == PortalArm::ActionShuffled).then(|| {
        PulseTemporalChannelShuffler::new(
            seed ^ ACTION_SHUFFLE_PORTAL_SALT,
            world.num_actuators(),
        )
    });

    let mut total_influx = 0.0_f64;
    for tick in 1..=ticks {
        let (emitted, ach_in) = {
            let view = world.current_view();
            let ach_in = f64::from(view.metabolic_influx);
            let p = org.step(view)?;
            (p, ach_in)
        };
        total_influx += ach_in;
        if let Some(sh) = shuffler.as_mut() {
            let applied = sh.shuffle_pulse(&emitted);
            world.apply_pulse(&applied);
        } else {
            world.apply_pulse(&emitted);
        }
        on_after_tick(tick, &org, &world);
    }

    let snap = org.snapshot();
    let (_, _, noisy_tv_share) = compute_band_allocations(&snap, bands);
    let influx_per_tick = total_influx / (ticks.max(1) as f64);
    Ok((world, snap, noisy_tv_share, influx_per_tick))
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return 0.0;
    }
    let nf = n as f64;
    let mx = xs[..n].iter().sum::<f64>() / nf;
    let my = ys[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom > 1e-12 {
        (num / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Runs Study 6 across all 5 portal configurations (`12` seeds, `1,500` ticks, `3` controls).
pub fn run_study_6_human_activity_portals(
    output_dir: &Path,
) -> Result<Study6Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;
    let ticks = 1_500_u64;
    let mut portals = Vec::with_capacity(5);

    // 1. DrawingScratchpad (192 channels, 5 actuators)
    {
        let bands = vec![
            BandSpec::new("Canvas Grid (0..144)", 0..144, BandRole::StructuredPhysical),
            BandSpec::new("Stylus Foveal (144..160)", 144..160, BandRole::StructuredPhysical),
            BandSpec::new("Noisy TV (160..192)", 160..192, BandRole::NoisyTvDistractor),
        ];
        let mut by_arm: HashMap<PortalArm, Vec<[f64; 7]>> = HashMap::new();
        for arm in PortalArm::ALL {
            let mut rows = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
            for &seed in &DEFAULT_TWELVE_SEEDS {
                let (w, snap, ntv, _) = step_portal_run(
                    seed,
                    arm,
                    ticks,
                    DrawingScratchpad::new(seed),
                    &bands,
                    |_, _, _| {},
                )?;
                let obs = w.observation();
                rows.push([
                    obs.active_stroke_ticks as f64,
                    f64::from(obs.canvas_coverage_fraction),
                    f64::from(obs.spatial_autocorrelation),
                    f64::from(obs.revisit_ratio),
                    f64::from(obs.stylus_trajectory_entropy),
                    ntv,
                    f64::from(snap.mean_interventional_agency),
                ]);
            }
            by_arm.insert(arm, rows);
        }
        let names = [
            "active_stroke_ticks",
            "canvas_coverage_fraction",
            "spatial_autocorrelation",
            "revisit_ratio",
            "stylus_trajectory_entropy",
            "noisy_tv_conductance_share",
            "mean_interventional_agency",
        ];
        portals.push(PortalExperimentSummary {
            portal_name: "DrawingScratchpad".to_string(),
            num_channels: 192,
            num_actuators: 5,
            uniform_noisy_tv_share: 32.0 / 192.0,
            metrics: collect_portal_blocks(&names, &by_arm),
        });
    }

    // 2. PatternMemoryBoard (176 channels, 4 actuators)
    {
        let bands = vec![
            BandSpec::new("Tile Waves (0..128)", 0..128, BandRole::StructuredPhysical),
            BandSpec::new("Cursor & Sonar (128..144)", 128..144, BandRole::ActiveEcho),
            BandSpec::new("Noisy TV (144..176)", 144..176, BandRole::NoisyTvDistractor),
        ];
        let mut by_arm: HashMap<PortalArm, Vec<[f64; 7]>> = HashMap::new();
        for arm in PortalArm::ALL {
            let mut rows = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
            for &seed in &DEFAULT_TWELVE_SEEDS {
                let (w, snap, ntv, _) = step_portal_run(
                    seed,
                    arm,
                    ticks,
                    PatternMemoryBoard::new(seed),
                    &bands,
                    |_, _, _| {},
                )?;
                let obs = w.observation();
                rows.push([
                    obs.tiles_inspected as f64,
                    obs.unique_tiles_visited as f64,
                    obs.matched_resonances_locked as f64,
                    f64::from(obs.repeat_visit_interval_mean),
                    f64::from(obs.cursor_dwell_stability),
                    ntv,
                    f64::from(snap.mean_interventional_agency),
                ]);
            }
            by_arm.insert(arm, rows);
        }
        let names = [
            "tiles_inspected",
            "unique_tiles_visited",
            "matched_resonances_locked",
            "repeat_visit_interval_mean",
            "cursor_dwell_stability",
            "noisy_tv_conductance_share",
            "mean_interventional_agency",
        ];
        portals.push(PortalExperimentSummary {
            portal_name: "PatternMemoryBoard".to_string(),
            num_channels: 176,
            num_actuators: 4,
            uniform_noisy_tv_share: 32.0 / 176.0,
            metrics: collect_portal_blocks(&names, &by_arm),
        });
    }

    // 3. RhythmEchoPad (128 channels, 5 actuators)
    {
        let bands = vec![
            BandSpec::new("Spectral Pads (0..64)", 0..64, BandRole::StructuredPhysical),
            BandSpec::new("Room Reverb (64..96)", 64..96, BandRole::ActiveEcho),
            BandSpec::new("Noisy TV (96..128)", 96..128, BandRole::NoisyTvDistractor),
        ];
        let mut by_arm: HashMap<PortalArm, Vec<[f64; 6]>> = HashMap::new();
        for arm in PortalArm::ALL {
            let mut rows = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
            for &seed in &DEFAULT_TWELVE_SEEDS {
                let (w, snap, ntv, _) = step_portal_run(
                    seed,
                    arm,
                    ticks,
                    RhythmEchoPad::new(seed),
                    &bands,
                    |_, _, _| {},
                )?;
                let obs = w.observation();
                rows.push([
                    obs.total_pad_strikes as f64,
                    f64::from(obs.listening_vs_playing_ratio),
                    f64::from(obs.phase_locking_value),
                    f64::from(obs.motif_continuation_similarity),
                    ntv,
                    f64::from(snap.mean_interventional_agency),
                ]);
            }
            by_arm.insert(arm, rows);
        }
        let names = [
            "total_pad_strikes",
            "listening_vs_playing_ratio",
            "phase_locking_value",
            "motif_continuation_similarity",
            "noisy_tv_conductance_share",
            "mean_interventional_agency",
        ];
        portals.push(PortalExperimentSummary {
            portal_name: "RhythmEchoPad".to_string(),
            num_channels: 128,
            num_actuators: 5,
            uniform_noisy_tv_share: 32.0 / 128.0,
            metrics: collect_portal_blocks(&names, &by_arm),
        });
    }

    // 4. BalanceCatchToy (128 channels, 3 actuators)
    {
        let bands = vec![
            BandSpec::new("Cradle Vestibular (0..96)", 0..96, BandRole::StructuredPhysical),
            BandSpec::new("Noisy TV (96..128)", 96..128, BandRole::NoisyTvDistractor),
        ];
        let mut by_arm: HashMap<PortalArm, Vec<[f64; 6]>> = HashMap::new();
        for arm in PortalArm::ALL {
            let mut rows = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
            for &seed in &DEFAULT_TWELVE_SEEDS {
                let (w, snap, ntv, _) = step_portal_run(
                    seed,
                    arm,
                    ticks,
                    BalanceCatchToy::new(seed),
                    &bands,
                    |_, _, _| {},
                )?;
                let obs = w.observation();
                rows.push([
                    f64::from(obs.mean_upright_duration_ticks),
                    obs.max_upright_streak as f64,
                    obs.sphere_falls as f64,
                    f64::from(obs.torque_phase_lead_correlation),
                    ntv,
                    f64::from(snap.mean_interventional_agency),
                ]);
            }
            by_arm.insert(arm, rows);
        }
        let names = [
            "mean_upright_duration_ticks",
            "max_upright_streak",
            "sphere_falls",
            "torque_phase_lead_correlation",
            "noisy_tv_conductance_share",
            "mean_interventional_agency",
        ];
        portals.push(PortalExperimentSummary {
            portal_name: "BalanceCatchToy".to_string(),
            num_channels: 128,
            num_actuators: 3,
            uniform_noisy_tv_share: 32.0 / 128.0,
            metrics: collect_portal_blocks(&names, &by_arm),
        });
    }

    // 5. CompositePortalMedium<BrightRadiantWorld> (752 channels, 27 actuators)
    {
        let bands = vec![
            BandSpec::new("Host Cave Structured (0..96)", 0..96, BandRole::StructuredPhysical),
            BandSpec::new("Host Cave Noisy TV (96..128)", 96..128, BandRole::NoisyTvDistractor),
            BandSpec::new("Scratchpad Structured (128..288)", 128..288, BandRole::StructuredPhysical),
            BandSpec::new("Scratchpad Noisy TV (288..320)", 288..320, BandRole::NoisyTvDistractor),
            BandSpec::new("Pattern Structured (320..464)", 320..464, BandRole::StructuredPhysical),
            BandSpec::new("Pattern Noisy TV (464..496)", 464..496, BandRole::NoisyTvDistractor),
            BandSpec::new("Rhythm Structured (496..592)", 496..592, BandRole::StructuredPhysical),
            BandSpec::new("Rhythm Noisy TV (592..624)", 592..624, BandRole::NoisyTvDistractor),
            BandSpec::new("Balance Structured (624..720)", 624..720, BandRole::StructuredPhysical),
            BandSpec::new("Balance Noisy TV (720..752)", 720..752, BandRole::NoisyTvDistractor),
        ];
        let mut by_arm: HashMap<PortalArm, Vec<[f64; 10]>> = HashMap::new();
        for arm in PortalArm::ALL {
            let mut rows = Vec::with_capacity(DEFAULT_TWELVE_SEEDS.len());
            for &seed in &DEFAULT_TWELVE_SEEDS {
                let mut atp_trace = Vec::with_capacity(ticks as usize);
                let mut focus_trace = Vec::with_capacity(ticks as usize);
                let comp_world =
                    CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(seed), seed);
                let (w, snap, ntv, influx_rate) = step_portal_run(
                    seed,
                    arm,
                    ticks,
                    comp_world,
                    &bands,
                    |_tick, org, cw| {
                        atp_trace.push(f64::from(org.snapshot().atp));
                        let tot_focus: f64 = cw
                            .portal_focus_levels()
                            .iter()
                            .map(|&f| f64::from(f))
                            .sum();
                        focus_trace.push(tot_focus);
                    },
                )?;
                let obs = w.observation();
                let tot_engaged: u64 = obs.portal_engaged_ticks.iter().sum();
                let mean_focus_sum: f64 = obs.mean_portal_focus.iter().map(|&f| f64::from(f)).sum();
                let r_atp_focus = pearson_corr(&atp_trace, &focus_trace);

                let mut sorted_atp = atp_trace.clone();
                sorted_atp.sort_by(f64::total_cmp);
                let med_atp = quantile_sorted(&sorted_atp, 0.50);
                let mut hi_sum = 0.0_f64;
                let mut hi_n = 0usize;
                let mut lo_sum = 0.0_f64;
                let mut lo_n = 0usize;
                for (&a, &f) in atp_trace.iter().zip(focus_trace.iter()) {
                    if a >= med_atp {
                        hi_sum += f;
                        hi_n += 1;
                    } else {
                        lo_sum += f;
                        lo_n += 1;
                    }
                }
                let hi_focus = if hi_n > 0 { hi_sum / (hi_n as f64) } else { 0.0 };
                let lo_focus = if lo_n > 0 { lo_sum / (lo_n as f64) } else { 0.0 };

                // Portal sensory conductance share (channels 128..752 structured slices).
                let (allocs, _, _) = compute_band_allocations(&snap, &bands);
                let portal_structured_share: f64 = [2usize, 4, 6, 8]
                    .iter()
                    .filter_map(|&idx| allocs.get(idx).map(|a| a.conductance_share))
                    .sum();

                rows.push([
                    influx_rate,
                    f64::from(snap.atp),
                    w.host().metrics().springs_depleted as f64,
                    tot_engaged as f64,
                    mean_focus_sum,
                    r_atp_focus,
                    hi_focus - lo_focus,
                    portal_structured_share,
                    ntv,
                    f64::from(snap.mean_interventional_agency),
                ]);
            }
            by_arm.insert(arm, rows);
        }
        let names = [
            "host_achieved_influx_per_tick",
            "final_atp",
            "host_springs_depleted",
            "total_portal_engaged_ticks",
            "sum_mean_portal_focus",
            "atp_vs_portal_focus_pearson_r",
            "high_minus_low_atp_portal_focus",
            "portal_structured_conductance_share",
            "noisy_tv_conductance_share",
            "mean_interventional_agency",
        ];
        portals.push(PortalExperimentSummary {
            portal_name: "CompositePortalMedium<BrightRadiantWorld>".to_string(),
            num_channels: 752,
            num_actuators: 27,
            uniform_noisy_tv_share: 160.0 / 752.0,
            metrics: collect_portal_blocks(&names, &by_arm),
        });
    }

    let outcome = Study6Outcome { portals };
    write_study6_files(output_dir, ticks, &outcome)?;
    Ok(outcome)
}

fn collect_portal_blocks<const K: usize>(
    names: &[&str; K],
    by_arm: &HashMap<PortalArm, Vec<[f64; K]>>,
) -> Vec<PortalMetricBlock> {
    let mut blocks = Vec::with_capacity(K);
    for (col, &name) in names.iter().enumerate() {
        let extract = |arm: PortalArm| -> Vec<f64> {
            by_arm
                .get(&arm)
                .map_or_else(Vec::new, |rows| rows.iter().map(|r| r[col]).collect())
        };
        let nat = extract(PortalArm::Natural);
        let froz = extract(PortalArm::FrozenPlasticity);
        let shuf = extract(PortalArm::ActionShuffled);
        blocks.push(build_metric_block(name, &nat, &froz, &shuf));
    }
    blocks
}

fn write_study6_files(
    output_dir: &Path,
    ticks: u64,
    out: &Study6Outcome,
) -> std::io::Result<()> {
    let mut md = String::with_capacity(24_576);
    let _ = writeln!(
        md,
        "# Study 6: Unforced Human-Activity Portals (`N = 12` Seeds, `{ticks}` Ticks, Zero Game-Score Rewards)"
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "Standalone portals provide a strictly flat constant basal metabolic trickle (`0.008`/tick) with `0.0` bonus for coverage, matches, rhythm, or balance. `CompositePortalMedium<BrightRadiantWorld>` attaches all 4 portals onto `BrightRadiantWorld` with `0.0` portal metabolic reward (`752` ambient channels, `27` actuators)."
    );
    let _ = writeln!(md);

    for p in &out.portals {
        let _ = writeln!(
            md,
            "## Portal: `{}` (`{}` Channels, `{}` Actuators, Uniform Noisy-TV Share `{:.2}%`)",
            p.portal_name,
            p.num_channels,
            p.num_actuators,
            p.uniform_noisy_tv_share * 100.0
        );
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |"
        );
        let _ = writeln!(
            md,
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        );
        for m in &p.metrics {
            let _ = writeln!(
                md,
                "| `{}` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:+.4}` (`p={:.4}`) | `{:+.4}` (`p={:.4}`) |",
                m.metric_name,
                m.natural.p50,
                m.natural.iqr,
                m.frozen.p50,
                m.frozen.iqr,
                m.shuffled.p50,
                m.shuffled.iqr,
                m.nat_vs_frozen.rank_biserial,
                m.nat_vs_frozen.p_permutation,
                m.nat_vs_shuffled.rank_biserial,
                m.nat_vs_shuffled.p_permutation,
            );
        }
        let _ = writeln!(md);
    }

    let mut json = String::with_capacity(32_768);
    json.push_str("{\n");
    write_kv_u64(&mut json, 1, "ticks_per_seed", ticks, true);
    write_kv_usize(&mut json, 1, "seed_count", DEFAULT_TWELVE_SEEDS.len(), true);
    write_indent(&mut json, 1);
    json.push_str("\"portals\": [\n");
    for (p_idx, p) in out.portals.iter().enumerate() {
        write_indent(&mut json, 2);
        json.push_str("{\n");
        write_kv_str(&mut json, 3, "portal_name", &p.portal_name, true);
        write_kv_usize(&mut json, 3, "num_channels", p.num_channels, true);
        write_kv_usize(&mut json, 3, "num_actuators", p.num_actuators, true);
        write_kv_f64(
            &mut json,
            3,
            "uniform_noisy_tv_share",
            p.uniform_noisy_tv_share,
            true,
        );
        write_indent(&mut json, 3);
        json.push_str("\"metrics\": [\n");
        for (m_idx, m) in p.metrics.iter().enumerate() {
            write_indent(&mut json, 4);
            let _ = writeln!(json, "{{\"metric\": \"{}\",", escape_json_str(&m.metric_name));
            write_dist_json(&mut json, 5, "natural", &m.natural, true);
            write_dist_json(&mut json, 5, "frozen_plasticity", &m.frozen, true);
            write_dist_json(&mut json, 5, "action_shuffled", &m.shuffled, true);
            write_mann_whitney_json(&mut json, 5, "natural_vs_frozen", &m.nat_vs_frozen, true);
            write_mann_whitney_json(
                &mut json,
                5,
                "natural_vs_shuffled",
                &m.nat_vs_shuffled,
                false,
            );
            write_indent(&mut json, 4);
            if m_idx + 1 < p.metrics.len() {
                json.push_str("},\n");
            } else {
                json.push_str("}\n");
            }
        }
        write_indent(&mut json, 3);
        json.push_str("]\n");
        write_indent(&mut json, 2);
        if p_idx + 1 < out.portals.len() {
            json.push_str("},\n");
        } else {
            json.push_str("}\n");
        }
    }
    write_indent(&mut json, 1);
    json.push_str("]\n}\n");

    fs::write(output_dir.join("06_human_activity_portals.md"), md)?;
    fs::write(output_dir.join("06_human_activity_portals.json"), json)?;
    Ok(())
}
