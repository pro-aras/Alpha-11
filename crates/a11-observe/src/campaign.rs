//! Wave B empirical campaign studies 0 through 3:
//! - Study 0: Workspace forensics, transactional persistence, plasticity/lesion audit, and `O(1)` latency profile (`4,096` vs `128` channels).
//! - Study 1: Spontaneous sense organogenesis (`BrightRadiantWorld` vs `PitchBlackEchoCave`, `12` seeds, `6` controls).
//! - Study 2: Coordinate-free scrambled wire bijection (`PermutedMedium`, `12` seeds, `6` controls).
//! - Study 3: Mid-life regime shifts (`03a` Cavefish Day-to-Darkness, `03b` Mid-Life Causal Reversal, `12` seeds, `6` controls).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{OrganismEngine, PersistError, WorldMedium};
use a11_worlds::{
    AdaptiveCaveWorld, BrightRadiantWorld, CaveMetrics, DenseFieldWorld, PermutedMedium,
    PitchBlackEchoCave,
};

use crate::forensics::{
    LatencyMemoryProfile, PersistenceAuditConfig, PersistenceAuditReport,
    PlasticityLesionAuditReport, SourceHygieneReport, audit_persistence_with_config,
    audit_plasticity_and_lesion, profile_latency, scan_crate_source,
};
use crate::report::{
    escape_json_str, fmt_json_f64, format_forensics_json, format_forensics_markdown,
    format_twin_study_json, format_twin_study_markdown, write_dist_json, write_indent,
    write_kv_f64, write_kv_str, write_kv_usize, write_mann_whitney_json,
};
use crate::stats::{EmpiricalDistribution, MannWhitneyComparison, mann_whitney_u};
use crate::twins::{
    BandRole, BandSpec, DEFAULT_TWELVE_SEEDS, POST_LIFE_WORLD_SEED_SALT, SeedArmRun, TwinArm,
    TwinStudyConfig, TwinStudyReport, run_twin_study_instrumented,
};

const PERMUTED_WIRE_SALT: u64 = 0x5C8A_4B1E_9D2F_7103;
const OVERSIZED_AUDIT_BYTES: usize = 257 * 1024 * 1024;

/// Error wrapper for live campaign execution (`io::Error` or `PersistError`).
#[derive(Debug)]
pub enum CampaignError {
    Io(io::Error),
    Persist(PersistError),
}

impl std::fmt::Display for CampaignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Persist(e) => write!(f, "Persistence/Organism error: {e:?}"),
        }
    }
}

impl std::error::Error for CampaignError {}

impl From<io::Error> for CampaignError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<PersistError> for CampaignError {
    fn from(e: PersistError) -> Self {
        Self::Persist(e)
    }
}

/// Canonical `128`-channel band specification for `AdaptiveCaveWorld` (`0..64`, `64..96`, `96..128`).
#[must_use]
pub fn cave_bands_128() -> Vec<BandSpec> {
    vec![
        BandSpec::new("Photonic 2D (0..64)", 0..64, BandRole::StructuredPhysical),
        BandSpec::new("Wave-Echo Sonar (64..96)", 64..96, BandRole::ActiveEcho),
        BandSpec::new(
            "Noisy TV Static (96..128)",
            96..128,
            BandRole::NoisyTvDistractor,
        ),
    ]
}

#[must_use]
pub(crate) fn make_default_organism(seed: u64) -> Alpha11Organism {
    Alpha11Organism::new(seed, OrganismConfig::default())
}

/// Per-crate hygiene record in Study 0.
#[derive(Clone, Debug)]
pub struct CrateHygieneEntry {
    pub crate_name: String,
    pub report: SourceHygieneReport,
}

/// Complete Study 0 forensics & latency output.
#[derive(Clone, Debug)]
pub struct Study0Outcome {
    pub crate_hygiene: Vec<CrateHygieneEntry>,
    pub core_hygiene: SourceHygieneReport,
    pub persistence: PersistenceAuditReport,
    pub plasticity: PlasticityLesionAuditReport,
    pub dense_4096_latency: LatencyMemoryProfile,
    pub radiant_128_latency: LatencyMemoryProfile,
    pub md_path: PathBuf,
    pub json_path: PathBuf,
}

/// Executes Study 0 strictly single-threaded before any worker threads are spawned.
pub fn run_study_0_forensics_and_latency(
    workspace_root: &Path,
    output_dir: &Path,
) -> Result<Study0Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;

    let crate_targets = [
        ("a11-medium", workspace_root.join("crates/a11-medium/src")),
        ("a11-core", workspace_root.join("crates/a11-core/src")),
        ("a11-worlds", workspace_root.join("crates/a11-worlds/src")),
        ("a11-portals", workspace_root.join("crates/a11-portals/src")),
    ];

    let mut crate_hygiene = Vec::with_capacity(crate_targets.len());
    let mut core_hygiene = SourceHygieneReport::default();
    for (name, path) in &crate_targets {
        let rep = scan_crate_source(path)?;
        if *name == "a11-core" {
            core_hygiene = rep.clone();
        }
        crate_hygiene.push(CrateHygieneEntry {
            crate_name: (*name).to_string(),
            report: rep,
        });
    }

    let p_cfg = PersistenceAuditConfig {
        seed: 101,
        warmup_steps: 100,
        continuation_steps: 100,
        oversized_bytes_len: OVERSIZED_AUDIT_BYTES,
    };
    let persistence = audit_persistence_with_config(
        p_cfg,
        make_default_organism,
        BrightRadiantWorld::new,
    )?;

    let plasticity = audit_plasticity_and_lesion(
        101,
        500,
        make_default_organism,
        BrightRadiantWorld::new,
    )?;

    // Warm up instruction caches briefly, then profile 2,000 release steps on 128ch vs 4,096ch.
    let mut org_128 = make_default_organism(101);
    let mut world_128 = BrightRadiantWorld::new(101);
    let radiant_128_latency = profile_latency(&mut org_128, &mut world_128, 2_000)?;

    let mut org_4096 = make_default_organism(101);
    let mut world_4096 = DenseFieldWorld::new(101);
    let dense_4096_latency = profile_latency(&mut org_4096, &mut world_4096, 2_000)?;

    let md = format_study0_markdown(
        &crate_hygiene,
        &core_hygiene,
        &persistence,
        &plasticity,
        &radiant_128_latency,
        &dense_4096_latency,
    );
    let json = format_study0_json(
        &crate_hygiene,
        &core_hygiene,
        &persistence,
        &plasticity,
        &radiant_128_latency,
        &dense_4096_latency,
    );

    let md_path = output_dir.join("00_forensics_and_latency.md");
    let json_path = output_dir.join("00_forensics_and_latency.json");
    fs::write(&md_path, md)?;
    fs::write(&json_path, json)?;

    Ok(Study0Outcome {
        crate_hygiene,
        core_hygiene,
        persistence,
        plasticity,
        dense_4096_latency,
        radiant_128_latency,
        md_path,
        json_path,
    })
}

fn format_study0_markdown(
    crate_hygiene: &[CrateHygieneEntry],
    core_hygiene: &SourceHygieneReport,
    persistence: &PersistenceAuditReport,
    plasticity: &PlasticityLesionAuditReport,
    lat_128: &LatencyMemoryProfile,
    lat_4096: &LatencyMemoryProfile,
) -> String {
    let mut md = format_forensics_markdown(
        Some(core_hygiene),
        Some(persistence),
        Some(plasticity),
        Some(lat_4096),
    );
    let _ = writeln!(md);
    let _ = writeln!(md, "## 5. Multi-Crate Source Hygiene Breakdown (`a11-medium`, `a11-core`, `a11-worlds`, `a11-portals`)");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Crate | Files Scanned | String Literals | Forbidden Identifiers | `unsafe` Blocks | `unwrap(` Calls | Zero Forbidden/Unsafe/Unwrap |"
    );
    let _ = writeln!(
        md,
        "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |"
    );
    for entry in crate_hygiene {
        let r = &entry.report;
        let strict_non_string_clean = r.forbidden_identifier_count == 0
            && r.unsafe_block_count == 0
            && r.unwrap_call_count == 0;
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |",
            entry.crate_name,
            r.files_scanned,
            r.string_literal_count,
            r.forbidden_identifier_count,
            r.unsafe_block_count,
            r.unwrap_call_count,
            strict_non_string_clean,
        );
        if !r.findings.is_empty() {
            for finding in r.findings.iter().take(8) {
                let _ = writeln!(
                    md,
                    "  - Finding in `{}:{}` (`{:?}`): `{}`",
                    finding.file, finding.line, finding.kind, finding.token
                );
            }
        }
    }
    let _ = writeln!(md);

    let p50_ratio = if lat_128.step_latency_us.p50 > 1e-9 {
        lat_4096.step_latency_us.p50 / lat_128.step_latency_us.p50
    } else {
        0.0
    };
    let p99_ratio = if lat_128.step_latency_us.p99 > 1e-9 {
        lat_4096.step_latency_us.p99 / lat_128.step_latency_us.p99
    } else {
        0.0
    };

    let _ = writeln!(
        md,
        "## 6. Bounded `O(1)` Perceptual Scaling Comparison (`128` vs `4,096` Ambient Channels = `32x` Input Width)"
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| World Arena | Ambient Channels | Steps | `step()` `p50` (`us`) | `step()` `p95` (`us`) | `step()` `p99` (`us`) | `step()` `max` (`us`) | Full Tick `p50` (`us`) | Final Logical State (`bytes`) | Checkpoint (`bytes`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for (label, prof) in [
        ("BrightRadiantWorld (128ch)", lat_128),
        ("DenseFieldWorld (4096ch)", lat_4096),
    ] {
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{}` | `{:.2}` | `{:.2}` | `{:.2}` | `{:.2}` | `{:.2}` | `{}` | `{}` |",
            label,
            prof.num_ambient_channels,
            prof.steps_profiled,
            prof.step_latency_us.p50,
            prof.step_latency_us.p95,
            prof.step_latency_us.p99,
            prof.step_latency_us.max,
            prof.full_tick_latency_us.p50,
            prof.final_used_state_bytes,
            prof.checkpoint_bytes,
        );
    }
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **`32x` Channel Width Scaling Ratio (`4096ch / 128ch`)**: `p50_ratio = {:.3}x`, `p99_ratio = {:.3}x` (vs `32.000x` linear `O(N)` scaling).",
        p50_ratio, p99_ratio
    );
    md
}

fn format_study0_json(
    crate_hygiene: &[CrateHygieneEntry],
    core_hygiene: &SourceHygieneReport,
    persistence: &PersistenceAuditReport,
    plasticity: &PlasticityLesionAuditReport,
    lat_128: &LatencyMemoryProfile,
    lat_4096: &LatencyMemoryProfile,
) -> String {
    let mut out = String::with_capacity(8_192);
    out.push_str("{\n");
    let base_core_json = format_forensics_json(
        Some(core_hygiene),
        Some(persistence),
        Some(plasticity),
        Some(lat_4096),
    );
    write_indent(&mut out, 1);
    let _ = writeln!(out, "\"core_and_dense_4096_audit\": {},", base_core_json);

    let base_128_json = format_forensics_json(None, None, None, Some(lat_128));
    write_indent(&mut out, 1);
    let _ = writeln!(out, "\"radiant_128_latency_audit\": {},", base_128_json);

    let p50_ratio = if lat_128.step_latency_us.p50 > 1e-9 {
        lat_4096.step_latency_us.p50 / lat_128.step_latency_us.p50
    } else {
        0.0
    };
    let p99_ratio = if lat_128.step_latency_us.p99 > 1e-9 {
        lat_4096.step_latency_us.p99 / lat_128.step_latency_us.p99
    } else {
        0.0
    };
    write_kv_f64(&mut out, 1, "channel_32x_step_p50_ratio", p50_ratio, true);
    write_kv_f64(&mut out, 1, "channel_32x_step_p99_ratio", p99_ratio, true);

    write_indent(&mut out, 1);
    out.push_str("\"crate_hygiene\": [\n");
    for (idx, entry) in crate_hygiene.iter().enumerate() {
        let r = &entry.report;
        write_indent(&mut out, 2);
        out.push_str("{\n");
        write_kv_str(&mut out, 3, "crate_name", &entry.crate_name, true);
        write_kv_usize(&mut out, 3, "files_scanned", r.files_scanned, true);
        write_kv_usize(&mut out, 3, "string_literal_count", r.string_literal_count, true);
        write_kv_usize(
            &mut out,
            3,
            "forbidden_identifier_count",
            r.forbidden_identifier_count,
            true,
        );
        write_kv_usize(&mut out, 3, "unsafe_block_count", r.unsafe_block_count, true);
        write_kv_usize(&mut out, 3, "unwrap_call_count", r.unwrap_call_count, false);
        write_indent(&mut out, 2);
        if idx + 1 < crate_hygiene.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(&mut out, 1);
    out.push_str("]\n}\n");
    out
}

/// Summary of world-level `CaveMetrics` across seeds for one `TwinArm`.
#[derive(Clone, Debug)]
pub struct CaveArmMetricsSummary {
    pub arm: TwinArm,
    pub active_pings_emitted: EmpiricalDistribution,
    pub echo_energy_received: EmpiricalDistribution,
    pub springs_depleted: EmpiricalDistribution,
    pub wall_impacts: EmpiricalDistribution,
    pub distance_travelled: EmpiricalDistribution,
    pub valve_activations: EmpiricalDistribution,
    pub emit_pulse_motor_conductance: EmpiricalDistribution,
    pub raw_pings: Vec<f64>,
    pub raw_echo_energy: Vec<f64>,
    pub raw_springs: Vec<f64>,
    pub raw_emit_motor_cond: Vec<f64>,
}

/// Complete outcome for one Cave twin study (`01a`, `01b`, or `02` sub-world).
#[derive(Clone, Debug)]
pub struct CaveStudyOutcome {
    pub report: TwinStudyReport,
    pub cave_by_arm: Vec<CaveArmMetricsSummary>,
}

fn summarize_cave_by_arm(
    records: &HashMap<(u64, TwinArm), (CaveMetrics, f64)>,
    seeds: &[u64],
) -> Vec<CaveArmMetricsSummary> {
    let mut out = Vec::with_capacity(TwinArm::ALL.len());
    for arm in TwinArm::ALL {
        let mut pings = Vec::with_capacity(seeds.len());
        let mut echo_e = Vec::with_capacity(seeds.len());
        let mut springs = Vec::with_capacity(seeds.len());
        let mut walls = Vec::with_capacity(seeds.len());
        let mut dist = Vec::with_capacity(seeds.len());
        let mut valves = Vec::with_capacity(seeds.len());
        let mut emit_cond = Vec::with_capacity(seeds.len());

        for &seed in seeds {
            if let Some((m, mc)) = records.get(&(seed, arm)) {
                pings.push(m.active_pings_emitted as f64);
                echo_e.push(f64::from(m.echo_energy_received));
                springs.push(m.springs_depleted as f64);
                walls.push(m.wall_impacts as f64);
                dist.push(f64::from(m.distance_travelled));
                valves.push(m.valve_activations as f64);
                emit_cond.push(*mc);
            }
        }

        out.push(CaveArmMetricsSummary {
            arm,
            active_pings_emitted: EmpiricalDistribution::from_f64_slice(&pings),
            echo_energy_received: EmpiricalDistribution::from_f64_slice(&echo_e),
            springs_depleted: EmpiricalDistribution::from_f64_slice(&springs),
            wall_impacts: EmpiricalDistribution::from_f64_slice(&walls),
            distance_travelled: EmpiricalDistribution::from_f64_slice(&dist),
            valve_activations: EmpiricalDistribution::from_f64_slice(&valves),
            emit_pulse_motor_conductance: EmpiricalDistribution::from_f64_slice(&emit_cond),
            raw_pings: pings,
            raw_echo_energy: echo_e,
            raw_springs: springs,
            raw_emit_motor_cond: emit_cond,
        });
    }
    out
}

fn motor_conductance_for_actuator(run: &SeedArmRun, target_actuator: u32) -> f64 {
    let n = run
        .final_snapshot
        .motor_pore_channels
        .len()
        .min(run.final_snapshot.motor_pore_conductances.len());
    let mut sum = 0.0_f64;
    for i in 0..n {
        if run.final_snapshot.motor_pore_channels[i] == target_actuator {
            let g = f64::from(run.final_snapshot.motor_pore_conductances[i]);
            if g.is_finite() && g > 0.0 {
                sum += g;
            }
        }
    }
    sum
}

/// Runs Study 1a (`BrightRadiantWorld`) and Study 1b (`PitchBlackEchoCave`) and writes both `.{md,json}` pairs.
pub fn run_study_1_sense_organogenesis(
    output_dir: &Path,
) -> Result<(CaveStudyOutcome, CaveStudyOutcome), CampaignError> {
    fs::create_dir_all(output_dir)?;
    let bands = cave_bands_128();

    // Study 1a: BrightRadiantWorld (2,400 ticks + 600 lesion eval, 12 seeds)
    let mut cfg_1a = TwinStudyConfig::new(
        "01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)",
        2_400,
        bands.clone(),
    );
    cfg_1a.post_life_eval_ticks = 600;
    let mut rec_1a: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();
    let rep_1a = run_twin_study_instrumented(
        &cfg_1a,
        make_default_organism,
        BrightRadiantWorld::new,
        |_, _, _, _, _| {},
        |_, _, _| {},
        |ch, _| ch,
        |m, _| m,
        |run, _org, world| {
            let emit_g = motor_conductance_for_actuator(run, 4);
            rec_1a.insert((run.seed, run.arm), (world.metrics().clone(), emit_g));
        },
    )?;
    let cave_1a = summarize_cave_by_arm(&rec_1a, &cfg_1a.seeds);
    let out_1a = CaveStudyOutcome {
        report: rep_1a,
        cave_by_arm: cave_1a,
    };

    // Study 1b: PitchBlackEchoCave (2,400 ticks + 600 lesion eval, 12 seeds)
    let mut cfg_1b = TwinStudyConfig::new(
        "01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)",
        2_400,
        bands,
    );
    cfg_1b.post_life_eval_ticks = 600;
    let mut rec_1b: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();
    let rep_1b = run_twin_study_instrumented(
        &cfg_1b,
        make_default_organism,
        PitchBlackEchoCave::new,
        |_, _, _, _, _| {},
        |_, _, _| {},
        |ch, _| ch,
        |m, _| m,
        |run, _org, world| {
            let emit_g = motor_conductance_for_actuator(run, 4);
            rec_1b.insert((run.seed, run.arm), (world.metrics().clone(), emit_g));
        },
    )?;
    let cave_1b = summarize_cave_by_arm(&rec_1b, &cfg_1b.seeds);
    let out_1b = CaveStudyOutcome {
        report: rep_1b,
        cave_by_arm: cave_1b,
    };

    write_cave_study_files(output_dir, "01a_bright_radiant_twins", &out_1a, Some(&out_1b))?;
    write_cave_study_files(
        output_dir,
        "01b_pitch_black_echo_cave_twins",
        &out_1b,
        Some(&out_1a),
    )?;

    Ok((out_1a, out_1b))
}

fn append_cave_metrics_markdown(
    md: &mut String,
    outcome: &CaveStudyOutcome,
    cross_world_peer: Option<&CaveStudyOutcome>,
) {
    let _ = writeln!(md);
    let _ = writeln!(md, "## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |"
    );
    for c in &outcome.cave_by_arm {
        let _ = writeln!(
            md,
            "| `{}` | `{:.1} [{:.1}]` | `{:.3} [{:.3}]` | `{:.3} [{:.3}]` | `{:.1} [{:.1}]` | `{:.1}` | `{:.2}` | `{:.1}` |",
            c.arm.as_str(),
            c.active_pings_emitted.p50,
            c.active_pings_emitted.iqr,
            c.echo_energy_received.p50,
            c.echo_energy_received.iqr,
            c.emit_pulse_motor_conductance.p50,
            c.emit_pulse_motor_conductance.iqr,
            c.springs_depleted.p50,
            c.springs_depleted.iqr,
            c.wall_impacts.p50,
            c.distance_travelled.p50,
            c.valve_activations.p50,
        );
    }

    if let Some(peer) = cross_world_peer {
        let nat_self = outcome.cave_by_arm.iter().find(|x| x.arm == TwinArm::Natural);
        let nat_peer = peer.cave_by_arm.iter().find(|x| x.arm == TwinArm::Natural);
        let band_self = outcome.report.arm_summary(TwinArm::Natural);
        let band_peer = peer.report.arm_summary(TwinArm::Natural);

        if let (Some(cs), Some(cp), Some(bs), Some(bp)) = (nat_self, nat_peer, band_self, band_peer)
        {
            let cmp_pings = mann_whitney_u(&cs.raw_pings, &cp.raw_pings);
            let cmp_echo_e = mann_whitney_u(&cs.raw_echo_energy, &cp.raw_echo_energy);
            let self_phot: Vec<f64> = bs
                .seed_runs
                .iter()
                .filter_map(|r| r.final_band_allocations.first().map(|a| a.conductance_share))
                .collect();
            let peer_phot: Vec<f64> = bp
                .seed_runs
                .iter()
                .filter_map(|r| r.final_band_allocations.first().map(|a| a.conductance_share))
                .collect();
            let self_sonar: Vec<f64> = bs
                .seed_runs
                .iter()
                .filter_map(|r| r.final_band_allocations.get(1).map(|a| a.conductance_share))
                .collect();
            let peer_sonar: Vec<f64> = bp
                .seed_runs
                .iter()
                .filter_map(|r| r.final_band_allocations.get(1).map(|a| a.conductance_share))
                .collect();
            let cmp_phot = mann_whitney_u(&self_phot, &peer_phot);
            let cmp_sonar = mann_whitney_u(&self_sonar, &peer_sonar);

            let _ = writeln!(md);
            let _ = writeln!(
                md,
                "### Cross-Habitat Modality Comparison (`{}` vs `{}` — `Natural` Cohort)",
                outcome.report.study_name, peer.report.study_name
            );
            let _ = writeln!(md);
            let _ = writeln!(
                md,
                "| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |"
            );
            let _ = writeln!(md, "| :--- | ---: | ---: | ---: |");
            for (name, c) in [
                ("active_pings_emitted", &cmp_pings),
                ("echo_energy_received", &cmp_echo_e),
                ("photonic_band_conductance_share", &cmp_phot),
                ("sonar_echo_band_conductance_share", &cmp_sonar),
            ] {
                let _ = writeln!(
                    md,
                    "| `{}` | `{:+.5}` | `{:+.4}` | `{:.4}` |",
                    name, c.median_diff, c.rank_biserial, c.p_permutation
                );
            }
        }
    }
}

fn write_cave_metrics_json_array(out: &mut String, indent: usize, cave_by_arm: &[CaveArmMetricsSummary]) {
    write_indent(out, indent);
    out.push_str("\"cave_metrics_by_arm\": [\n");
    for (idx, c) in cave_by_arm.iter().enumerate() {
        write_indent(out, indent + 1);
        out.push_str("{\n");
        write_kv_str(out, indent + 2, "arm", c.arm.as_str(), true);
        write_dist_json(
            out,
            indent + 2,
            "active_pings_emitted",
            &c.active_pings_emitted,
            true,
        );
        write_dist_json(
            out,
            indent + 2,
            "echo_energy_received",
            &c.echo_energy_received,
            true,
        );
        write_dist_json(
            out,
            indent + 2,
            "emit_pulse_motor_conductance",
            &c.emit_pulse_motor_conductance,
            true,
        );
        write_dist_json(
            out,
            indent + 2,
            "springs_depleted",
            &c.springs_depleted,
            true,
        );
        write_dist_json(out, indent + 2, "wall_impacts", &c.wall_impacts, true);
        write_dist_json(
            out,
            indent + 2,
            "distance_travelled",
            &c.distance_travelled,
            true,
        );
        write_dist_json(
            out,
            indent + 2,
            "valve_activations",
            &c.valve_activations,
            false,
        );
        write_indent(out, indent + 1);
        if idx + 1 < cave_by_arm.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(out, indent);
    out.push_str("]\n");
}

fn write_cave_study_files(
    output_dir: &Path,
    slug: &str,
    outcome: &CaveStudyOutcome,
    cross_world_peer: Option<&CaveStudyOutcome>,
) -> io::Result<()> {
    let mut md = format_twin_study_markdown(&outcome.report);
    append_cave_metrics_markdown(&mut md, outcome, cross_world_peer);

    let mut json = String::with_capacity(40_960);
    json.push_str("{\n");
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"twin_study\": {},",
        format_twin_study_json(&outcome.report)
    );
    write_cave_metrics_json_array(&mut json, 1, &outcome.cave_by_arm);
    json.push_str("}\n");

    fs::write(output_dir.join(format!("{slug}.md")), md)?;
    fs::write(output_dir.join(format!("{slug}.json")), json)?;
    Ok(())
}

/// Complete Study 2 outcome (`PermutedMedium<BrightRadiantWorld>` and `PermutedMedium<PitchBlackEchoCave>`).
#[derive(Clone, Debug)]
pub struct Study2Outcome {
    pub permuted_bright: CaveStudyOutcome,
    pub permuted_dark: CaveStudyOutcome,
}

/// Runs Study 2 (`PermutedMedium` coordinate-free wire scrambling) and compares against unpermuted Study 1.
pub fn run_study_2_permuted_wire_invariance(
    output_dir: &Path,
    unpermuted_bright: &CaveStudyOutcome,
    unpermuted_dark: &CaveStudyOutcome,
) -> Result<Study2Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;
    let bands = cave_bands_128();

    let congenital_perm_seed = |w_seed: u64| -> u64 {
        let org_seed = if DEFAULT_TWELVE_SEEDS.contains(&w_seed) {
            w_seed
        } else {
            w_seed ^ POST_LIFE_WORLD_SEED_SALT
        };
        org_seed ^ PERMUTED_WIRE_SALT
    };

    let mut cfg_bright = TwinStudyConfig::new(
        "02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)",
        2_400,
        bands.clone(),
    );
    cfg_bright.post_life_eval_ticks = 600;
    let mut rec_bright: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();
    let rep_bright = run_twin_study_instrumented(
        &cfg_bright,
        make_default_organism,
        |w_seed| PermutedMedium::new(BrightRadiantWorld::new(w_seed), congenital_perm_seed(w_seed)),
        |_, _, _, _, _| {},
        |_, _, _| {},
        |ch, world| world.invert_sensor_channel(ch as usize) as u32,
        |m, world| world.invert_motor_channel(m as usize) as u32,
        |run, _org, world| {
            let emit_g = motor_conductance_for_actuator(run, 4);
            rec_bright.insert((run.seed, run.arm), (world.inner().metrics().clone(), emit_g));
        },
    )?;
    let permuted_bright = CaveStudyOutcome {
        report: rep_bright,
        cave_by_arm: summarize_cave_by_arm(&rec_bright, &cfg_bright.seeds),
    };

    let mut cfg_dark = TwinStudyConfig::new(
        "02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)",
        2_400,
        bands,
    );
    cfg_dark.post_life_eval_ticks = 600;
    let mut rec_dark: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();
    let rep_dark = run_twin_study_instrumented(
        &cfg_dark,
        make_default_organism,
        |w_seed| PermutedMedium::new(PitchBlackEchoCave::new(w_seed), congenital_perm_seed(w_seed)),
        |_, _, _, _, _| {},
        |_, _, _| {},
        |ch, world| world.invert_sensor_channel(ch as usize) as u32,
        |m, world| world.invert_motor_channel(m as usize) as u32,
        |run, _org, world| {
            let emit_g = motor_conductance_for_actuator(run, 4);
            rec_dark.insert((run.seed, run.arm), (world.inner().metrics().clone(), emit_g));
        },
    )?;
    let permuted_dark = CaveStudyOutcome {
        report: rep_dark,
        cave_by_arm: summarize_cave_by_arm(&rec_dark, &cfg_dark.seeds),
    };

    write_study2_files(
        output_dir,
        &permuted_bright,
        &permuted_dark,
        unpermuted_bright,
        unpermuted_dark,
    )?;

    Ok(Study2Outcome {
        permuted_bright,
        permuted_dark,
    })
}

fn compute_invariance_checks(
    perm: &CaveStudyOutcome,
    unperm: &CaveStudyOutcome,
) -> Vec<(String, f64, f64, MannWhitneyComparison)> {
    let mut checks = Vec::new();
    let Some(ps) = perm.report.arm_summary(TwinArm::Natural) else {
        return checks;
    };
    let Some(us) = unperm.report.arm_summary(TwinArm::Natural) else {
        return checks;
    };
    let Some(pc) = perm.cave_by_arm.iter().find(|x| x.arm == TwinArm::Natural) else {
        return checks;
    };
    let Some(uc) = unperm.cave_by_arm.iter().find(|x| x.arm == TwinArm::Natural) else {
        return checks;
    };

    let p_in: Vec<f64> = ps.seed_runs.iter().map(|r| r.achieved_influx_per_tick).collect();
    let u_in: Vec<f64> = us.seed_runs.iter().map(|r| r.achieved_influx_per_tick).collect();
    checks.push((
        "achieved_influx_per_tick".to_string(),
        ps.achieved_influx_per_tick.p50,
        us.achieved_influx_per_tick.p50,
        mann_whitney_u(&p_in, &u_in),
    ));

    for (b_idx, b_name) in [
        (0usize, "decoded_photonic_conductance_share"),
        (1usize, "decoded_sonar_conductance_share"),
        (2usize, "decoded_noisy_tv_conductance_share"),
    ] {
        let pv: Vec<f64> = ps
            .seed_runs
            .iter()
            .filter_map(|r| r.final_band_allocations.get(b_idx).map(|a| a.conductance_share))
            .collect();
        let uv: Vec<f64> = us
            .seed_runs
            .iter()
            .filter_map(|r| r.final_band_allocations.get(b_idx).map(|a| a.conductance_share))
            .collect();
        let p_med = ps
            .band_summaries
            .get(b_idx)
            .map_or(0.0, |b| b.conductance_share.p50);
        let u_med = us
            .band_summaries
            .get(b_idx)
            .map_or(0.0, |b| b.conductance_share.p50);
        checks.push((b_name.to_string(), p_med, u_med, mann_whitney_u(&pv, &uv)));
    }

    checks.push((
        "active_pings_emitted".to_string(),
        pc.active_pings_emitted.p50,
        uc.active_pings_emitted.p50,
        mann_whitney_u(&pc.raw_pings, &uc.raw_pings),
    ));
    checks.push((
        "decoded_emit_pulse_motor_conductance".to_string(),
        pc.emit_pulse_motor_conductance.p50,
        uc.emit_pulse_motor_conductance.p50,
        mann_whitney_u(&pc.raw_emit_motor_cond, &uc.raw_emit_motor_cond),
    ));
    checks
}

fn write_study2_files(
    output_dir: &Path,
    permuted_bright: &CaveStudyOutcome,
    permuted_dark: &CaveStudyOutcome,
    unpermuted_bright: &CaveStudyOutcome,
    unpermuted_dark: &CaveStudyOutcome,
) -> io::Result<()> {
    let bright_inv = compute_invariance_checks(permuted_bright, unpermuted_bright);
    let dark_inv = compute_invariance_checks(permuted_dark, unpermuted_dark);

    let mut md = String::with_capacity(32_768);
    let _ = writeln!(
        md,
        "# Study 2: Coordinate-Free Bijective Wire Permutation Invariance (`PermutedMedium`)"
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "All `128` ambient sensory channels and all `6` motor actuator channels were bijectively scrambled via `PermutedMedium::new(world, seed ^ 0x5C8A_4B1E_9D2F_7103)` before entering `Alpha11Organism`. Channel identities were inverted **exclusively inside `a11-observe`** (`invert_sensor_channel` / `invert_motor_channel`) after each snapshot."
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "## Invariance Summary (`Permuted` vs `Unpermuted` `Natural` Cohort, `N = 12` Seeds)"
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Habitat | Observable | Permuted `p50` | Unpermuted `p50` | `median(Perm) - median(Unperm)` | Rank-Biserial `r_rb` | `p_perm` |"
    );
    let _ = writeln!(
        md,
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |"
    );
    for (hab, checks) in [
        ("BrightRadiantWorld", &bright_inv),
        ("PitchBlackEchoCave", &dark_inv),
    ] {
        for (name, p_med, u_med, cmp) in checks {
            let _ = writeln!(
                md,
                "| `{}` | `{}` | `{:.4}` | `{:.4}` | `{:+.5}` | `{:+.4}` | `{:.4}` |",
                hab, name, p_med, u_med, cmp.median_diff, cmp.rank_biserial, cmp.p_permutation
            );
        }
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "---");
    let _ = writeln!(md);
    md.push_str(&format_twin_study_markdown(&permuted_bright.report));
    append_cave_metrics_markdown(&mut md, permuted_bright, Some(permuted_dark));
    let _ = writeln!(md);
    let _ = writeln!(md, "---");
    let _ = writeln!(md);
    md.push_str(&format_twin_study_markdown(&permuted_dark.report));
    append_cave_metrics_markdown(&mut md, permuted_dark, Some(permuted_bright));

    let mut json = String::with_capacity(65_536);
    json.push_str("{\n");
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"permuted_bright_twin_study\": {},",
        format_twin_study_json(&permuted_bright.report)
    );
    write_indent(&mut json, 1);
    out_named_cave_array(&mut json, 1, "permuted_bright_cave_metrics", &permuted_bright.cave_by_arm, true);
    write_indent(&mut json, 1);
    let _ = writeln!(
        json,
        "\"permuted_dark_twin_study\": {},",
        format_twin_study_json(&permuted_dark.report)
    );
    write_indent(&mut json, 1);
    out_named_cave_array(&mut json, 1, "permuted_dark_cave_metrics", &permuted_dark.cave_by_arm, true);

    write_invariance_json(&mut json, 1, "bright_invariance_vs_unpermuted", &bright_inv, true);
    write_invariance_json(&mut json, 1, "dark_invariance_vs_unpermuted", &dark_inv, false);
    json.push_str("}\n");

    fs::write(output_dir.join("02_permuted_wire_invariance.md"), md)?;
    fs::write(output_dir.join("02_permuted_wire_invariance.json"), json)?;
    Ok(())
}

fn out_named_cave_array(
    out: &mut String,
    indent: usize,
    key: &str,
    cave_by_arm: &[CaveArmMetricsSummary],
    trailing_comma: bool,
) {
    let mut tmp = String::new();
    write_cave_metrics_json_array(&mut tmp, indent, cave_by_arm);
    let replaced = tmp.replacen("\"cave_metrics_by_arm\":", &format!("\"{}\":", escape_json_str(key)), 1);
    let trimmed = replaced.trim_start().trim_end();
    out.push_str(trimmed);
    if trailing_comma {
        out.push_str(",\n");
    } else {
        out.push('\n');
    }
}

fn write_invariance_json(
    out: &mut String,
    indent: usize,
    key: &str,
    checks: &[(String, f64, f64, MannWhitneyComparison)],
    trailing_comma: bool,
) {
    write_indent(out, indent);
    let _ = writeln!(out, "\"{}\": [", escape_json_str(key));
    for (idx, (name, p_med, u_med, cmp)) in checks.iter().enumerate() {
        write_indent(out, indent + 1);
        out.push_str("{\n");
        write_kv_str(out, indent + 2, "metric", name, true);
        write_kv_f64(out, indent + 2, "permuted_p50", *p_med, true);
        write_kv_f64(out, indent + 2, "unpermuted_p50", *u_med, true);
        write_mann_whitney_json(out, indent + 2, "comparison", cmp, false);
        write_indent(out, indent + 1);
        if idx + 1 < checks.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(out, indent);
    if trailing_comma {
        out.push_str("],\n");
    } else {
        out.push_str("]\n");
    }
}

/// Phase-split pre-shift (`1..=1500`) vs post-shift (`1501..=3000`) metrics for Study 3a & 3b.
#[derive(Clone, Debug)]
pub struct Study3Outcome {
    pub cavefish_report: TwinStudyReport,
    pub cavefish_pre_pings_nat: EmpiricalDistribution,
    pub cavefish_post_pings_nat: EmpiricalDistribution,
    pub cavefish_pre_echo_e_nat: EmpiricalDistribution,
    pub cavefish_post_echo_e_nat: EmpiricalDistribution,
    pub reversal_report: TwinStudyReport,
    pub reversal_pre_splits_nat: EmpiricalDistribution,
    pub reversal_post_splits_nat: EmpiricalDistribution,
    /// Mean temperature over the equal-length pre-reversal window `1201..=1500`.
    pub reversal_mean_temp_pre_nat: EmpiricalDistribution,
    /// Mean temperature over the equal-length post-reversal window `1501..=1800`.
    pub reversal_mean_temp_post_nat: EmpiricalDistribution,
    pub reversal_early_learning_influx_nat: EmpiricalDistribution,
    pub reversal_post_shock_influx_nat: EmpiricalDistribution,
    pub reversal_late_recovery_influx_nat: EmpiricalDistribution,
}

#[derive(Clone, Copy, Debug, Default)]
struct ReversalTrace {
    splits_at_1500: u64,
    // Audit P17: compare equal-length window MEANS (300 ticks each). The former statistic compared the
    // MAXIMUM over 399 post ticks against a SINGLE pre tick, which is biased upward even for a random walk.
    temp_sum_pre_1201_1500: f64,
    temp_sum_post_1501_1800: f64,
    influx_1_300: f64,
    influx_301_600: f64,
    influx_1501_1800: f64,
    influx_1801_2100: f64,
    influx_2401_3000: f64,
}

/// Runs Study 3a (`03a_cavefish_day_to_dark`) and Study 3b (`03b_midlife_causal_reversal`).
pub fn run_study_3_regime_shifts(output_dir: &Path) -> Result<Study3Outcome, CampaignError> {
    fs::create_dir_all(output_dir)?;
    let bands = cave_bands_128();

    // --- Study 3a: Cavefish Day-to-Darkness Transition at tick 1,500 ---
    let mut cfg_3a = TwinStudyConfig::new(
        "03a Cavefish Day-to-Darkness Transition (Illumination 1.0 -> 0.0 at Tick 1500)",
        3_000,
        bands.clone(),
    );
    cfg_3a.post_life_eval_ticks = 600;

    let mut pre_cave_3a: HashMap<(u64, TwinArm), CaveMetrics> = HashMap::new();
    let mut fin_cave_3a: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();

    let rep_3a = run_twin_study_instrumented(
        &cfg_3a,
        make_default_organism,
        AdaptiveCaveWorld::bright_radiant,
        |seed, arm, tick, _org, world| {
            // Tick 1501 occurs immediately after tick 1500 (`50%` checkpoint) has completed.
            if tick == 1_501 {
                pre_cave_3a.insert((seed, arm), world.metrics().clone());
                world.set_illumination(0.0);
            }
        },
        |_seed, _arm, eval_world| {
            eval_world.set_illumination(0.0);
        },
        |ch, _| ch,
        |m, _| m,
        |run, _org, world| {
            let emit_g = motor_conductance_for_actuator(run, 4);
            fin_cave_3a.insert((run.seed, run.arm), (world.metrics().clone(), emit_g));
        },
    )?;

    let cave_3a_summary = summarize_cave_by_arm(&fin_cave_3a, &DEFAULT_TWELVE_SEEDS);
    let mut pre_pings_nat = Vec::new();
    let mut post_pings_nat = Vec::new();
    let mut pre_echo_nat = Vec::new();
    let mut post_echo_nat = Vec::new();
    let mut pre_springs_nat = Vec::new();
    let mut post_springs_nat = Vec::new();

    for &seed in &DEFAULT_TWELVE_SEEDS {
        if let (Some(pre), Some((fin, _))) = (
            pre_cave_3a.get(&(seed, TwinArm::Natural)),
            fin_cave_3a.get(&(seed, TwinArm::Natural)),
        ) {
            pre_pings_nat.push(pre.active_pings_emitted as f64);
            post_pings_nat.push(fin.active_pings_emitted.saturating_sub(pre.active_pings_emitted) as f64);
            pre_echo_nat.push(f64::from(pre.echo_energy_received));
            post_echo_nat.push((f64::from(fin.echo_energy_received) - f64::from(pre.echo_energy_received)).max(0.0));
            pre_springs_nat.push(pre.springs_depleted as f64);
            post_springs_nat.push(fin.springs_depleted.saturating_sub(pre.springs_depleted) as f64);
        }
    }

    let d_pre_pings = EmpiricalDistribution::from_f64_slice(&pre_pings_nat);
    let d_post_pings = EmpiricalDistribution::from_f64_slice(&post_pings_nat);
    let d_pre_echo = EmpiricalDistribution::from_f64_slice(&pre_echo_nat);
    let d_post_echo = EmpiricalDistribution::from_f64_slice(&post_echo_nat);
    let d_pre_springs = EmpiricalDistribution::from_f64_slice(&pre_springs_nat);
    let d_post_springs = EmpiricalDistribution::from_f64_slice(&post_springs_nat);
    let cmp_dark_pings = mann_whitney_u(&post_pings_nat, &pre_pings_nat);
    let cmp_dark_echo = mann_whitney_u(&post_echo_nat, &pre_echo_nat);

    let mut md_3a = format_twin_study_markdown(&rep_3a);
    let tmp_out_3a = CaveStudyOutcome {
        report: rep_3a.clone(),
        cave_by_arm: cave_3a_summary.clone(),
    };
    append_cave_metrics_markdown(&mut md_3a, &tmp_out_3a, None);
    let _ = writeln!(md_3a);
    let _ = writeln!(
        md_3a,
        "## 8. Pre-Darkness (`Ticks 1..=1500`) vs Post-Darkness (`Ticks 1501..=3000`) Shift Telemetry (`Natural`)"
    );
    let _ = writeln!(md_3a);
    let _ = writeln!(
        md_3a,
        "| Phase Metric | Day Half (`1..=1500`, `p50 [IQR]`) | Dark Half (`1501..=3000`, `p50 [IQR]`) | `median(Dark) - median(Day)` | Rank-Biserial `r_rb` | `p_perm` |"
    );
    let _ = writeln!(md_3a, "| :--- | :--- | :--- | ---: | ---: | ---: |");
    let _ = writeln!(
        md_3a,
        "| `active_pings_emitted` | `{:.1} [{:.1}]` | `{:.1} [{:.1}]` | `{:+.2}` | `{:+.4}` | `{:.4}` |",
        d_pre_pings.p50,
        d_pre_pings.iqr,
        d_post_pings.p50,
        d_post_pings.iqr,
        cmp_dark_pings.median_diff,
        cmp_dark_pings.rank_biserial,
        cmp_dark_pings.p_permutation,
    );
    let _ = writeln!(
        md_3a,
        "| `echo_energy_received` | `{:.3} [{:.3}]` | `{:.3} [{:.3}]` | `{:+.3}` | `{:+.4}` | `{:.4}` |",
        d_pre_echo.p50,
        d_pre_echo.iqr,
        d_post_echo.p50,
        d_post_echo.iqr,
        cmp_dark_echo.median_diff,
        cmp_dark_echo.rank_biserial,
        cmp_dark_echo.p_permutation,
    );
    let cmp_dark_springs = mann_whitney_u(&post_springs_nat, &pre_springs_nat);
    let _ = writeln!(
        md_3a,
        "| `springs_depleted` | `{:.1} [{:.1}]` | `{:.1} [{:.1}]` | `{:+.2}` | `{:+.4}` | `{:.4}` |",
        d_pre_springs.p50,
        d_pre_springs.iqr,
        d_post_springs.p50,
        d_post_springs.iqr,
        cmp_dark_springs.median_diff,
        cmp_dark_springs.rank_biserial,
        cmp_dark_springs.p_permutation,
    );

    let mut json_3a = String::with_capacity(40_960);
    json_3a.push_str("{\n");
    write_indent(&mut json_3a, 1);
    let _ = writeln!(json_3a, "\"twin_study\": {},", format_twin_study_json(&rep_3a));
    out_named_cave_array(&mut json_3a, 1, "cave_metrics_by_arm", &cave_3a_summary, true);
    write_dist_json(&mut json_3a, 1, "natural_pre_dark_pings", &d_pre_pings, true);
    write_dist_json(&mut json_3a, 1, "natural_post_dark_pings", &d_post_pings, true);
    write_dist_json(&mut json_3a, 1, "natural_pre_dark_echo_energy", &d_pre_echo, true);
    write_dist_json(&mut json_3a, 1, "natural_post_dark_echo_energy", &d_post_echo, true);
    write_mann_whitney_json(&mut json_3a, 1, "post_vs_pre_dark_pings", &cmp_dark_pings, true);
    write_mann_whitney_json(&mut json_3a, 1, "post_vs_pre_dark_echo_energy", &cmp_dark_echo, false);
    json_3a.push_str("}\n");

    fs::write(output_dir.join("03a_cavefish_day_to_dark.md"), md_3a)?;
    fs::write(output_dir.join("03a_cavefish_day_to_dark.json"), json_3a)?;

    // --- Study 3b: Mid-Life Causal Reversal at tick 1,500 ---
    let mut cfg_3b = TwinStudyConfig::new(
        "03b Mid-Life Causal Reversal (Motor + Thermal Vent Polarity Inverted at Tick 1500)",
        3_000,
        bands,
    );
    cfg_3b.post_life_eval_ticks = 600;

    let mut rev_traces: HashMap<(u64, TwinArm), ReversalTrace> = HashMap::new();
    let mut final_tick_influx: HashMap<(u64, TwinArm), f64> = HashMap::new();
    let mut fin_cave_3b: HashMap<(u64, TwinArm), (CaveMetrics, f64)> = HashMap::new();

    let rep_3b = run_twin_study_instrumented(
        &cfg_3b,
        make_default_organism,
        BrightRadiantWorld::new,
        |seed, arm, tick, org, world| {
            let entry = rev_traces.entry((seed, arm)).or_default();
            // The hook at `tick` sees the organism state after `tick - 1` completed steps.
            if (1_202..=1_501).contains(&tick) {
                entry.temp_sum_pre_1201_1500 += f64::from(org.snapshot().temperature);
            } else if (1_502..=1_801).contains(&tick) {
                entry.temp_sum_post_1501_1800 += f64::from(org.snapshot().temperature);
            }
            if tick == 1_501 {
                entry.splits_at_1500 = org.snapshot().contradiction_splits;
                world.trigger_causal_reversal();
            }

            // Accumulate achieved window influx from world.current_view() (which reflects previous tick's action).
            if tick > 1 {
                let prev_tick = tick - 1;
                let inf = f64::from(world.current_view().metabolic_influx);
                if prev_tick <= 300 {
                    entry.influx_1_300 += inf;
                } else if prev_tick <= 600 {
                    entry.influx_301_600 += inf;
                } else if (1_501..=1_800).contains(&prev_tick) {
                    entry.influx_1501_1800 += inf;
                } else if (1_801..=2_100).contains(&prev_tick) {
                    entry.influx_1801_2100 += inf;
                } else if (2_401..=3_000).contains(&prev_tick) {
                    entry.influx_2401_3000 += inf;
                }
            }
        },
        |_seed, _arm, eval_world| {
            eval_world.trigger_causal_reversal();
        },
        |ch, _| ch,
        |m, _| m,
        |run, _org, world| {
            if run.ticks_executed == 3_000 {
                final_tick_influx.insert(
                    (run.seed, run.arm),
                    f64::from(world.current_view().metabolic_influx),
                );
            }
            let emit_g = motor_conductance_for_actuator(run, 4);
            fin_cave_3b.insert((run.seed, run.arm), (world.metrics().clone(), emit_g));
        },
    )?;

    for (key, extra_inf) in final_tick_influx {
        if let Some(entry) = rev_traces.get_mut(&key) {
            entry.influx_2401_3000 += extra_inf;
        }
    }

    let cave_3b_summary = summarize_cave_by_arm(&fin_cave_3b, &DEFAULT_TWELVE_SEEDS);

    let mut pre_splits = Vec::new();
    let mut post_splits = Vec::new();
    let mut temp_1500 = Vec::new();
    let mut temp_peak = Vec::new();
    let mut w_1_300 = Vec::new();
    let mut w_301_600 = Vec::new();
    let mut w_1501_1800 = Vec::new();
    let mut w_1801_2100 = Vec::new();
    let mut w_2401_3000 = Vec::new();

    if let Some(nat_summary) = rep_3b.arm_summary(TwinArm::Natural) {
        for run in &nat_summary.seed_runs {
            if let Some(tr) = rev_traces.get(&(run.seed, TwinArm::Natural)) {
                pre_splits.push(tr.splits_at_1500 as f64);
                post_splits.push(
                    run.final_snapshot
                        .contradiction_splits
                        .saturating_sub(tr.splits_at_1500) as f64,
                );
                temp_1500.push(tr.temp_sum_pre_1201_1500 / 300.0);
                temp_peak.push(tr.temp_sum_post_1501_1800 / 300.0);
                w_1_300.push(tr.influx_1_300 / 300.0);
                w_301_600.push(tr.influx_301_600 / 300.0);
                w_1501_1800.push(tr.influx_1501_1800 / 300.0);
                w_1801_2100.push(tr.influx_1801_2100 / 300.0);
                w_2401_3000.push(tr.influx_2401_3000 / 600.0);
            }
        }
    }

    let d_pre_splits = EmpiricalDistribution::from_f64_slice(&pre_splits);
    let d_post_splits = EmpiricalDistribution::from_f64_slice(&post_splits);
    let d_t1500 = EmpiricalDistribution::from_f64_slice(&temp_1500);
    let d_tpeak = EmpiricalDistribution::from_f64_slice(&temp_peak);
    let d_w1 = EmpiricalDistribution::from_f64_slice(&w_1_300);
    let d_w2 = EmpiricalDistribution::from_f64_slice(&w_301_600);
    let d_w_shock = EmpiricalDistribution::from_f64_slice(&w_1501_1800);
    let d_w_reacq = EmpiricalDistribution::from_f64_slice(&w_1801_2100);
    let d_w_late = EmpiricalDistribution::from_f64_slice(&w_2401_3000);

    // Post-reversal half (`second_half_influx_per_tick`) comparison across controls!
    let get_second_half = |arm: TwinArm| -> Vec<f64> {
        rep_3b
            .arm_summary(arm)
            .map_or_else(Vec::new, |s| {
                s.seed_runs
                    .iter()
                    .map(|r| r.second_half_influx_per_tick)
                    .collect()
            })
    };
    let nat_h2 = get_second_half(TwinArm::Natural);
    let froz_h2 = get_second_half(TwinArm::FrozenPlasticity);
    let yok_h2 = get_second_half(TwinArm::YokedMetabolism);
    let shuf_h2 = get_second_half(TwinArm::ActionShuffled);

    let cmp_h2_froz = mann_whitney_u(&nat_h2, &froz_h2);
    let cmp_h2_yok = mann_whitney_u(&nat_h2, &yok_h2);
    let cmp_h2_shuf = mann_whitney_u(&nat_h2, &shuf_h2);
    let cmp_reacq_vs_init1 = mann_whitney_u(&w_1501_1800, &w_1_300);
    let cmp_reacq2_vs_init2 = mann_whitney_u(&w_1801_2100, &w_301_600);
    let cmp_late_vs_shock = mann_whitney_u(&w_2401_3000, &w_1501_1800);
    let cmp_temp_spike = mann_whitney_u(&temp_peak, &temp_1500);

    let mut md_3b = format_twin_study_markdown(&rep_3b);
    let tmp_out_3b = CaveStudyOutcome {
        report: rep_3b.clone(),
        cave_by_arm: cave_3b_summary.clone(),
    };
    append_cave_metrics_markdown(&mut md_3b, &tmp_out_3b, None);
    let _ = writeln!(md_3b);
    let _ = writeln!(
        md_3b,
        "## 8. Mid-Life Causal Reversal Shock, Contradiction Splitting & Re-Acquisition (`Natural`, `N = 12`)"
    );
    let _ = writeln!(md_3b);
    let _ = writeln!(
        md_3b,
        "- **Contradiction Splits (`1..=1500` vs `1501..=3000`)**: `{:.1} [{:.1}]` vs `{:.1} [{:.1}]`",
        d_pre_splits.p50, d_pre_splits.iqr, d_post_splits.p50, d_post_splits.iqr
    );
    let _ = writeln!(
        md_3b,
        "- **Mean Temperature (`1201..=1500` -> `1501..=1800`, equal 300-tick windows)**: `{:.4} [{:.4}]` -> `{:.4} [{:.4}]` (`r_rb = {:+.4}`, `p_perm = {:.4}`)",
        d_t1500.p50,
        d_t1500.iqr,
        d_tpeak.p50,
        d_tpeak.iqr,
        cmp_temp_spike.rank_biserial,
        cmp_temp_spike.p_permutation,
    );
    let _ = writeln!(
        md_3b,
        "- **Initial Learning Windows (`1..=300` -> `301..=600` Influx/Tick `p50`)**: `{:.4}` -> `{:.4}`",
        d_w1.p50, d_w2.p50
    );
    let _ = writeln!(
        md_3b,
        "- **Post-Reversal Windows (`1501..=1800` -> `1801..=2100` -> `2401..=3000` Influx/Tick `p50`)**: `{:.4}` -> `{:.4}` -> `{:.4}`",
        d_w_shock.p50, d_w_reacq.p50, d_w_late.p50
    );
    let _ = writeln!(md_3b);
    let _ = writeln!(
        md_3b,
        "| Reversal Hypothesis Test | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |"
    );
    let _ = writeln!(md_3b, "| :--- | ---: | ---: | ---: |");
    for (label, c) in [
        ("Post-Reversal Half (1501..=3000): Natural vs FrozenPlasticity", &cmp_h2_froz),
        ("Post-Reversal Half (1501..=3000): Natural vs YokedMetabolism", &cmp_h2_yok),
        ("Post-Reversal Half (1501..=3000): Natural vs ActionShuffled", &cmp_h2_shuf),
        ("Post-Reversal Late (2401..=3000) vs Shock (1501..=1800)", &cmp_late_vs_shock),
        ("Re-Acquisition (1501..=1800) vs Initial (1..=300)", &cmp_reacq_vs_init1),
        ("Re-Acquisition (1801..=2100) vs Initial (301..=600)", &cmp_reacq2_vs_init2),
    ] {
        let _ = writeln!(
            md_3b,
            "| `{}` | `{:+.5}` | `{:+.4}` | `{:.4}` |",
            label, c.median_diff, c.rank_biserial, c.p_permutation
        );
    }

    let mut json_3b = String::with_capacity(40_960);
    json_3b.push_str("{\n");
    write_indent(&mut json_3b, 1);
    let _ = writeln!(json_3b, "\"twin_study\": {},", format_twin_study_json(&rep_3b));
    out_named_cave_array(&mut json_3b, 1, "cave_metrics_by_arm", &cave_3b_summary, true);
    write_dist_json(&mut json_3b, 1, "pre_reversal_contradiction_splits", &d_pre_splits, true);
    write_dist_json(&mut json_3b, 1, "post_reversal_contradiction_splits", &d_post_splits, true);
    write_dist_json(&mut json_3b, 1, "mean_temperature_1201_1500", &d_t1500, true);
    write_dist_json(&mut json_3b, 1, "mean_temperature_1501_1800", &d_tpeak, true);
    write_dist_json(&mut json_3b, 1, "influx_1_300", &d_w1, true);
    write_dist_json(&mut json_3b, 1, "influx_301_600", &d_w2, true);
    write_dist_json(&mut json_3b, 1, "influx_1501_1800", &d_w_shock, true);
    write_dist_json(&mut json_3b, 1, "influx_1801_2100", &d_w_reacq, true);
    write_dist_json(&mut json_3b, 1, "influx_2401_3000", &d_w_late, true);
    write_mann_whitney_json(&mut json_3b, 1, "post_half_natural_vs_frozen", &cmp_h2_froz, true);
    write_mann_whitney_json(&mut json_3b, 1, "post_half_natural_vs_yoked", &cmp_h2_yok, true);
    write_mann_whitney_json(&mut json_3b, 1, "post_half_natural_vs_shuffled", &cmp_h2_shuf, true);
    write_mann_whitney_json(&mut json_3b, 1, "late_recovery_vs_shock", &cmp_late_vs_shock, true);
    write_mann_whitney_json(&mut json_3b, 1, "reacq_1501_1800_vs_init_1_300", &cmp_reacq_vs_init1, true);
    let _ = fmt_json_f64(0.0);
    write_mann_whitney_json(
        &mut json_3b,
        1,
        "reacq_1801_2100_vs_init_301_600",
        &cmp_reacq2_vs_init2,
        false,
    );
    json_3b.push_str("}\n");

    fs::write(output_dir.join("03b_midlife_causal_reversal.md"), md_3b)?;
    fs::write(output_dir.join("03b_midlife_causal_reversal.json"), json_3b)?;

    Ok(Study3Outcome {
        cavefish_report: rep_3a,
        cavefish_pre_pings_nat: d_pre_pings,
        cavefish_post_pings_nat: d_post_pings,
        cavefish_pre_echo_e_nat: d_pre_echo,
        cavefish_post_echo_e_nat: d_post_echo,
        reversal_report: rep_3b,
        reversal_pre_splits_nat: d_pre_splits,
        reversal_post_splits_nat: d_post_splits,
        reversal_mean_temp_pre_nat: d_t1500,
        reversal_mean_temp_post_nat: d_tpeak,
        reversal_early_learning_influx_nat: d_w1,
        reversal_post_shock_influx_nat: d_w_shock,
        reversal_late_recovery_influx_nat: d_w_late,
    })
}
