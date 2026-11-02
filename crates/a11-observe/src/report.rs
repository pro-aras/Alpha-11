//! Honest, non-cherry-picked Markdown and RFC-8259 JSON empirical report generators
//! for multi-seed twin studies, identical-twin divergence, and forensic audits.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::forensics::{
    CorruptionCaseOutcome, LatencyMemoryProfile, PersistenceAuditReport,
    PlasticityLesionAuditReport, SourceHygieneReport,
};
use crate::stats::{EmpiricalDistribution, IndividualityDivergence, MannWhitneyComparison};
use crate::twins::{ArmCohortSummary, IdenticalTwinReport, TwinArm, TwinStudyReport};

/// Writes `<slug>.md` and `<slug>.json` for a `TwinStudyReport` into `output_dir`.
pub fn write_twin_study_files(
    output_dir: &Path,
    slug: &str,
    report: &TwinStudyReport,
) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(output_dir)?;
    let safe_slug = sanitize_slug(slug);
    let md_path = output_dir.join(format!("{safe_slug}.md"));
    let json_path = output_dir.join(format!("{safe_slug}.json"));

    fs::write(&md_path, format_twin_study_markdown(report))?;
    fs::write(&json_path, format_twin_study_json(report))?;
    Ok((md_path, json_path))
}

/// Writes `<slug>.md` and `<slug>.json` for forensic, persistence, plasticity, and latency audits.
pub fn write_forensics_files(
    output_dir: &Path,
    slug: &str,
    hygiene: Option<&SourceHygieneReport>,
    persistence: Option<&PersistenceAuditReport>,
    plasticity: Option<&PlasticityLesionAuditReport>,
    latency: Option<&LatencyMemoryProfile>,
) -> io::Result<(PathBuf, PathBuf)> {
    fs::create_dir_all(output_dir)?;
    let safe_slug = sanitize_slug(slug);
    let md_path = output_dir.join(format!("{safe_slug}.md"));
    let json_path = output_dir.join(format!("{safe_slug}.json"));

    fs::write(
        &md_path,
        format_forensics_markdown(hygiene, persistence, plasticity, latency),
    )?;
    fs::write(
        &json_path,
        format_forensics_json(hygiene, persistence, plasticity, latency),
    )?;
    Ok((md_path, json_path))
}

fn sanitize_slug(slug: &str) -> String {
    let mut out = String::with_capacity(slug.len());
    for ch in slug.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "observation_report".to_string()
    } else {
        out
    }
}

/// Formats a complete, non-cherry-picked Markdown report for a `TwinStudyReport`.
#[must_use]
pub fn format_twin_study_markdown(report: &TwinStudyReport) -> String {
    let mut md = String::with_capacity(16_384);

    let _ = writeln!(md, "# Observational Twin Study: {}", report.study_name);
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **Seeds (`N`)**: `{}` (`{:?}`)",
        report.seeds.len(),
        report.seeds
    );
    let _ = writeln!(
        md,
        "- **Primary Life Horizon (`T`)**: `{}` ticks | **Post-Life Lesion Window (`T_eval`)**: `{}` ticks",
        report.horizon_ticks, report.post_life_eval_ticks
    );
    let _ = writeln!(
        md,
        "- **Medium Boundary**: `{}` unlabelled ambient channels, `{}` continuous actuators",
        report.num_ambient_channels, report.num_actuators
    );
    let _ = writeln!(
        md,
        "- **Uniform Noisy-TV Channel Expectation**: `{:.2}%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `{}`",
        report.uniform_noisy_tv_channel_fraction * 100.0,
        report.lesion_preserved_natural_mode_all_seeds
    );
    let _ = writeln!(md);

    // Section 1: Metabolic & Causal Telemetry Across All 6 Arms
    let _ = writeln!(md, "## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |"
    );
    for s in &report.arm_summaries {
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{:.4} [{:.4}]` (`{:.4}..{:.4}`) | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:.4} -> {:.4}` | `{:.4} [{:.4}]` | `{:.4} [{:.4}]` | `{:.3}` |",
            s.arm.as_str(),
            s.ticks_per_seed,
            s.achieved_influx_per_tick.p50,
            s.achieved_influx_per_tick.iqr,
            s.achieved_influx_per_tick.min,
            s.achieved_influx_per_tick.max,
            s.achieved_stress_per_tick.p50,
            s.achieved_stress_per_tick.iqr,
            s.net_achieved_energy_per_tick.p50,
            s.net_achieved_energy_per_tick.iqr,
            s.first_half_influx_per_tick.p50,
            s.second_half_influx_per_tick.p50,
            s.mean_interventional_agency.p50,
            s.mean_interventional_agency.iqr,
            s.mean_prediction_error.p50,
            s.mean_prediction_error.iqr,
            s.final_atp.p50,
        );
    }
    let _ = writeln!(md);

    // Section 2: Morphogenesis & Structural Dynamics
    let _ = writeln!(md, "## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |"
    );
    for s in &report.arm_summaries {
        let _ = writeln!(
            md,
            "| `{}` | `{:.1} [{:.1}]` (`p95={:.1}`) | `{:.1} [{:.1}]` | `{:.1}` | `{:.1}` | `{:.1}` | `{:.2} / {:.2} / {:.1}` | `{:.1} / {:.1} / {:.1} / {:.1}` | `{:.1} / {:.1} / {:.1}` |",
            s.arm.as_str(),
            s.cell_count.p50,
            s.cell_count.iqr,
            s.cell_count.p95,
            s.tube_count.p50,
            s.tube_count.iqr,
            s.stiff_highway_tubes.p50,
            s.inhibitory_tubes.p50,
            s.active_echo_delay_tubes.p50,
            s.mean_tau.p50,
            s.max_tau.p50,
            s.tau_octave_count.p50,
            s.mitosis_births.p50,
            s.contradiction_splits.p50,
            s.myelinated_shortcuts.p50,
            s.atrophied_structures.p50,
            s.committed_motor_pulses.p50,
            s.subthreshold_rehearsals.p50,
            s.rest_consolidation_ticks.p50,
        );
    }
    let _ = writeln!(md);

    // Section 3: Non-Parametric Mann-Whitney U & Rank-Biserial Comparisons
    let _ = writeln!(md, "## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |"
    );
    let _ = writeln!(
        md,
        "| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |"
    );
    for cmp in &report.pairwise_comparisons {
        let c = &cmp.comparison;
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{}` | `{:+.5}` | `{:.1}` (`{:.1}`) | `{:+.4}` | `{:.4}` | `{:.4}` |",
            cmp.metric_name,
            cmp.arm_a.as_str(),
            cmp.arm_b.as_str(),
            c.median_diff,
            c.u_stat,
            c.u_a,
            c.rank_biserial,
            c.p_permutation,
            c.p_normal_approx,
        );
    }
    let _ = writeln!(md);

    // Section 4: Sense-Organ Band Allocation & Noisy-TV Distractor Analysis
    if !report.bands.is_empty() {
        let _ = writeln!(md, "## 4. Sense-Organ Band Conductance Allocation & Delay Tuning");
        let _ = writeln!(md);
        let _ = writeln!(
            md,
            "| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\\Delta` (`p50 [IQR]`) |"
        );
        let _ = writeln!(
            md,
            "| :--- | :--- | :--- | ---: | :--- | ---: | :--- |"
        );
        for s in &report.arm_summaries {
            for b in &s.band_summaries {
                let _ = writeln!(
                    md,
                    "| `{}` | `{}` | `{}` | `{:.1}%` | `{:.1}% [{:.1}%]` (`p95={:.1}%`) | `{:.1}` | `{:.2} [{:.2}]` |",
                    s.arm.as_str(),
                    b.label,
                    b.role.as_str(),
                    b.uniform_random_share_expectation * 100.0,
                    b.conductance_share.p50 * 100.0,
                    b.conductance_share.iqr * 100.0,
                    b.conductance_share.p95 * 100.0,
                    b.pore_count.p50,
                    b.conductance_weighted_delay.p50,
                    b.conductance_weighted_delay.iqr,
                );
            }
        }
        let _ = writeln!(md);
    }

    // Section 5: Developmental Trajectory of Natural Arm
    if let Some(nat) = report.arm_summary(TwinArm::Natural)
        && !nat.developmental_curve.is_empty()
    {
        let _ = writeln!(md, "## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)");
        let _ = writeln!(md);
        let mut header = String::from(
            "| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share |",
        );
        let mut divider = String::from(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        );
        for b in &report.bands {
            let _ = write!(header, " `{}` Share / `\\Delta` |", b.label);
            divider.push_str(" :--- |");
        }
        let _ = writeln!(md, "{header}");
        let _ = writeln!(md, "{divider}");

        for pt in &nat.developmental_curve {
            let mut row = format!(
                "| `{}` | `{:.0}%` | `{:.4}` | `{:.3}` | `{:.1}` | `{:.1}` | `{:.2}` | `{:.1}` | `{:.4}` | `{:.1}%` |",
                pt.tick,
                pt.progress_fraction * 100.0,
                pt.window_influx_per_tick.p50,
                pt.atp.p50,
                pt.cell_count.p50,
                pt.tube_count.p50,
                pt.mean_tau.p50,
                pt.tau_octave_count.p50,
                pt.mean_interventional_agency.p50,
                pt.noisy_tv_conductance_share.p50 * 100.0,
            );
            for idx in 0..report.bands.len() {
                let sh = pt
                    .band_conductance_shares
                    .get(idx)
                    .map_or(0.0, |(_, d)| d.p50 * 100.0);
                let dl = pt
                    .band_weighted_delays
                    .get(idx)
                    .map_or(0.0, |(_, d)| d.p50);
                let _ = write!(row, " `{sh:.1}%` / `{dl:.2}` |");
            }
            let _ = writeln!(md, "{row}");
        }
        let _ = writeln!(md);
    }

    // Section 6: Individuality & Constrained-Randomness Divergence
    let _ = writeln!(md, "## 6. Emergent Individuality & Constrained-Randomness Divergence");
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |"
    );
    let _ = writeln!(
        md,
        "| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |"
    );
    for s in &report.arm_summaries {
        let ind = &s.individuality;
        let _ = writeln!(
            md,
            "| `{}` | `{}` | `{:.4}` | `{:.4} / {:.4}` | `{:.4} / {:.4}` | `{:.4} / {:.4}` | `{:.4}` | `{:.4}` |",
            s.arm.as_str(),
            ind.individual_count,
            ind.inter_individual_cv,
            ind.mean_pairwise_pore_cosine_distance,
            ind.max_pairwise_pore_cosine_distance,
            ind.mean_pairwise_tau_jsd,
            ind.max_pairwise_tau_jsd,
            ind.mean_trajectory_entropy_bits,
            ind.normalized_trajectory_entropy,
            ind.trajectory_divergence_bits,
            ind.composite_individuality_index,
        );
    }

    md
}

/// Formats an `IdenticalTwinReport` into Markdown.
#[must_use]
pub fn format_identical_twins_markdown(report: &IdenticalTwinReport) -> String {
    let mut md = String::with_capacity(2_048);
    let ind = &report.cohort_summary.individuality;
    let _ = writeln!(
        md,
        "# Identical-Genome Constrained-Randomness Divergence (`genome_seed={}`)",
        report.genome_seed
    );
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "- **Twins (`N`)**: `{}` | **World Seed**: `{}` | **Horizon**: `{}` ticks | **Initial Jitter `epsilon`**: `{:.2e}`",
        report.twin_count, report.world_seed, report.horizon_ticks, report.perturbation_epsilon
    );
    let _ = writeln!(
        md,
        "- **Influx / Tick (`p50 [IQR]`, `min..max`)**: `{:.4} [{:.4}]` (`{:.4}..{:.4}`)",
        report.cohort_summary.achieved_influx_per_tick.p50,
        report.cohort_summary.achieved_influx_per_tick.iqr,
        report.cohort_summary.achieved_influx_per_tick.min,
        report.cohort_summary.achieved_influx_per_tick.max,
    );
    let _ = writeln!(md, "- **Outcome `CV`**: `{:.4}`", ind.inter_individual_cv);
    let _ = writeln!(
        md,
        "- **Mean Pairwise Pore Cosine Distance**: `{:.4}` (`max={:.4}`)",
        ind.mean_pairwise_pore_cosine_distance, ind.max_pairwise_pore_cosine_distance
    );
    let _ = writeln!(
        md,
        "- **Mean Pairwise `tau` JSD**: `{:.4}` bits (`max={:.4}`)",
        ind.mean_pairwise_tau_jsd, ind.max_pairwise_tau_jsd
    );
    let _ = writeln!(
        md,
        "- **Mean Intra-Individual Trajectory Entropy**: `{:.4}` bits (`norm={:.4}`)",
        ind.mean_trajectory_entropy_bits, ind.normalized_trajectory_entropy
    );
    let _ = writeln!(
        md,
        "- **Inter-Individual Trajectory JSD**: `{:.4}` bits",
        ind.trajectory_divergence_bits
    );
    let _ = writeln!(
        md,
        "- **Composite Individuality Index**: `{:.4}`",
        ind.composite_individuality_index
    );
    md
}

/// Formats forensic source-hygiene, persistence, plasticity/lesion, and latency reports as Markdown.
#[must_use]
pub fn format_forensics_markdown(
    hygiene: Option<&SourceHygieneReport>,
    persistence: Option<&PersistenceAuditReport>,
    plasticity: Option<&PlasticityLesionAuditReport>,
    latency: Option<&LatencyMemoryProfile>,
) -> String {
    let mut md = String::with_capacity(4_096);
    let _ = writeln!(md, "# Alpha 11 Forensic, Persistence, Plasticity & Latency Audit");
    let _ = writeln!(md);

    if let Some(h) = hygiene {
        let _ = writeln!(md, "## 1. Production Source Hygiene");
        let _ = writeln!(md, "- **Files Scanned**: `{}` | **Clean**: `{}`", h.files_scanned, h.is_clean());
        let _ = writeln!(
            md,
            "- **String Literals**: `{}` | **Forbidden Anthropomorphic Identifiers**: `{}` | **`unsafe` Blocks**: `{}` | **`unwrap(` Calls**: `{}`",
            h.string_literal_count,
            h.forbidden_identifier_count,
            h.unsafe_block_count,
            h.unwrap_call_count
        );
        let _ = writeln!(md);
    }

    if let Some(p) = persistence {
        let _ = writeln!(md, "## 2. Transactional CRC32C Persistence & Fail-Closed Poisoning");
        let _ = writeln!(
            md,
            "- **All Checks Passed**: `{}` | **Checkpoint Size @100**: `{}` bytes",
            p.all_passed(),
            p.checkpoint_bytes_at_100
        );
        let _ = writeln!(
            md,
            "- **`@100` Restore Snapshot/Hash Exact**: `{}` / `{}`",
            p.restore_at_100_snapshot_equal, p.restore_at_100_hash_equal
        );
        let _ = writeln!(
            md,
            "- **`@200` Continuation Bit-Identical (`Pulses` / `Snapshot` / `Hash`)**: `{}` / `{}` / `{}`",
            p.continuation_200_pulses_bit_identical,
            p.continuation_200_snapshot_bit_identical,
            p.continuation_200_hash_identical
        );
        let _ = writeln!(
            md,
            "- **Payload Bit-Flip Fail-Closed**: `{}` | **Header/CRC Bit-Flips Passed**: `{}/{}` | **Truncations Passed**: `{}/{}` | **Oversized Fail-Closed**: `{}`",
            p.payload_bitflip_case.passed(),
            p.header_or_crc_bitflip_cases
                .iter()
                .filter(|c| c.passed())
                .count(),
            p.header_or_crc_bitflip_cases.len(),
            p.truncation_cases.iter().filter(|c| c.passed()).count(),
            p.truncation_cases.len(),
            p.oversized_case.passed(),
        );
        let _ = writeln!(md);
    }

    if let Some(pl) = plasticity {
        let _ = writeln!(md, "## 3. `FrozenPlasticity` Invariance & Flagless Lesion Verification");
        let _ = writeln!(
            md,
            "- **All Checks Passed**: `{}` (`{}` frozen steps tested)",
            pl.all_passed(),
            pl.frozen_steps_tested
        );
        let _ = writeln!(
            md,
            "- **Max Changed Parameter Hash Bits (`FrozenPlasticity` / Post-Growth `FrozenPlasticity`)**: `{}` / `{}`",
            pl.max_frozen_hash_changed_bits, pl.post_growth_frozen_hash_changed_bits
        );
        let _ = writeln!(
            md,
            "- **Frozen Topology Invariant**: `{}` | **Flagless Lesion Preserved `Natural` Mode**: `{}` | **Zeroed All Pore Conductances**: `{}`",
            pl.frozen_topology_invariant,
            pl.lesion_preserved_natural_mode,
            pl.lesion_zeroed_all_pore_conductances
        );
        let _ = writeln!(md);
    }

    if let Some(lat) = latency {
        let _ = writeln!(md, "## 4. Step Latency & Logical / OS Memory Profile");
        let _ = writeln!(
            md,
            "- **Steps Profiled**: `{}` (`{}` ambient channels, `{}` actuators)",
            lat.steps_profiled, lat.num_ambient_channels, lat.num_actuators
        );
        let _ = writeln!(
            md,
            "- **`OrganismEngine::step` Latency (`us`)**: `p50={:.2}`, `p95={:.2}`, `p99={:.2}`, `max={:.2}`, `mean={:.2}`",
            lat.step_latency_us.p50,
            lat.step_latency_us.p95,
            lat.step_latency_us.p99,
            lat.step_latency_us.max,
            lat.step_latency_us.mean,
        );
        let _ = writeln!(
            md,
            "- **Logical State Bytes (`final` / `peak`)**: `{}` / `{}` bytes | **Checkpoint Bytes**: `{}` bytes",
            lat.final_used_state_bytes, lat.peak_used_state_bytes, lat.checkpoint_bytes
        );
        if let Some(os) = lat.os_memory {
            let _ = writeln!(
                md,
                "- **Linux `/proc/self/status` Memory**: `VmRSS={}` bytes (`{:.2} MiB`), `VmHWM={}` bytes (`{:.2} MiB`)",
                os.vm_rss_bytes,
                (os.vm_rss_bytes as f64) / (1024.0 * 1024.0),
                os.vm_hwm_bytes,
                (os.vm_hwm_bytes as f64) / (1024.0 * 1024.0),
            );
        }
    }

    md
}

/// Formats a `TwinStudyReport` as valid, pretty-printed RFC-8259 JSON (`std`-only).
#[must_use]
pub fn format_twin_study_json(report: &TwinStudyReport) -> String {
    let mut out = String::with_capacity(32_768);
    out.push_str("{\n");
    write_kv_str(&mut out, 1, "study_name", &report.study_name, true);
    write_indent(&mut out, 1);
    out.push_str("\"seeds\": [");
    for (idx, &seed) in report.seeds.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{seed}");
    }
    out.push_str("],\n");
    write_kv_u64(&mut out, 1, "horizon_ticks", report.horizon_ticks, true);
    write_kv_u64(
        &mut out,
        1,
        "post_life_eval_ticks",
        report.post_life_eval_ticks,
        true,
    );
    write_kv_usize(
        &mut out,
        1,
        "num_ambient_channels",
        report.num_ambient_channels,
        true,
    );
    write_kv_usize(&mut out, 1, "num_actuators", report.num_actuators, true);
    write_kv_f64(
        &mut out,
        1,
        "uniform_noisy_tv_channel_fraction",
        report.uniform_noisy_tv_channel_fraction,
        true,
    );
    write_kv_bool(
        &mut out,
        1,
        "lesion_preserved_natural_mode_all_seeds",
        report.lesion_preserved_natural_mode_all_seeds,
        true,
    );

    write_indent(&mut out, 1);
    out.push_str("\"bands\": [\n");
    for (idx, b) in report.bands.iter().enumerate() {
        write_indent(&mut out, 2);
        out.push_str("{\n");
        write_kv_str(&mut out, 3, "label", &b.label, true);
        write_kv_str(&mut out, 3, "role", b.role.as_str(), true);
        write_kv_u64(&mut out, 3, "start", u64::from(b.range.start), true);
        write_kv_u64(&mut out, 3, "end", u64::from(b.range.end), false);
        write_indent(&mut out, 2);
        if idx + 1 < report.bands.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(&mut out, 1);
    out.push_str("],\n");

    write_indent(&mut out, 1);
    out.push_str("\"arm_summaries\": [\n");
    for (idx, arm_s) in report.arm_summaries.iter().enumerate() {
        write_arm_summary_json(&mut out, 2, arm_s, idx + 1 < report.arm_summaries.len());
    }
    write_indent(&mut out, 1);
    out.push_str("],\n");

    write_indent(&mut out, 1);
    out.push_str("\"pairwise_comparisons\": [\n");
    for (idx, cmp) in report.pairwise_comparisons.iter().enumerate() {
        write_indent(&mut out, 2);
        out.push_str("{\n");
        write_kv_str(&mut out, 3, "metric_name", &cmp.metric_name, true);
        write_kv_str(&mut out, 3, "arm_a", cmp.arm_a.as_str(), true);
        write_kv_str(&mut out, 3, "arm_b", cmp.arm_b.as_str(), true);
        write_mann_whitney_json(&mut out, 3, "comparison", &cmp.comparison, false);
        write_indent(&mut out, 2);
        if idx + 1 < report.pairwise_comparisons.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(&mut out, 1);
    out.push_str("]\n");
    out.push_str("}\n");
    out
}

/// Formats an `IdenticalTwinReport` as valid RFC-8259 JSON.
#[must_use]
pub fn format_identical_twins_json(report: &IdenticalTwinReport) -> String {
    let mut out = String::with_capacity(8_192);
    out.push_str("{\n");
    write_kv_u64(&mut out, 1, "genome_seed", report.genome_seed, true);
    write_kv_u64(&mut out, 1, "world_seed", report.world_seed, true);
    write_kv_usize(&mut out, 1, "twin_count", report.twin_count, true);
    write_kv_u64(&mut out, 1, "horizon_ticks", report.horizon_ticks, true);
    write_kv_f64(
        &mut out,
        1,
        "perturbation_epsilon",
        f64::from(report.perturbation_epsilon),
        true,
    );
    write_indent(&mut out, 1);
    out.push_str("\"cohort_summary\": ");
    let mut sub = String::new();
    write_arm_summary_json(&mut sub, 1, &report.cohort_summary, false);
    out.push_str(sub.trim_start());
    out.push_str("}\n");
    out
}

/// Formats forensic reports as valid RFC-8259 JSON.
#[must_use]
pub fn format_forensics_json(
    hygiene: Option<&SourceHygieneReport>,
    persistence: Option<&PersistenceAuditReport>,
    plasticity: Option<&PlasticityLesionAuditReport>,
    latency: Option<&LatencyMemoryProfile>,
) -> String {
    let mut out = String::with_capacity(4_096);
    out.push_str("{\n");

    write_indent(&mut out, 1);
    out.push_str("\"hygiene\": ");
    if let Some(h) = hygiene {
        out.push_str("{\n");
        write_kv_usize(&mut out, 2, "files_scanned", h.files_scanned, true);
        write_kv_bool(&mut out, 2, "is_clean", h.is_clean(), true);
        write_kv_usize(
            &mut out,
            2,
            "string_literal_count",
            h.string_literal_count,
            true,
        );
        write_kv_usize(
            &mut out,
            2,
            "forbidden_identifier_count",
            h.forbidden_identifier_count,
            true,
        );
        write_kv_usize(
            &mut out,
            2,
            "unsafe_block_count",
            h.unsafe_block_count,
            true,
        );
        write_kv_usize(&mut out, 2, "unwrap_call_count", h.unwrap_call_count, false);
        write_indent(&mut out, 1);
        out.push_str("},\n");
    } else {
        out.push_str("null,\n");
    }

    write_indent(&mut out, 1);
    out.push_str("\"persistence\": ");
    if let Some(p) = persistence {
        out.push_str("{\n");
        write_kv_bool(&mut out, 2, "all_passed", p.all_passed(), true);
        write_kv_usize(
            &mut out,
            2,
            "checkpoint_bytes_at_100",
            p.checkpoint_bytes_at_100,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "restore_at_100_snapshot_equal",
            p.restore_at_100_snapshot_equal,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "restore_at_100_hash_equal",
            p.restore_at_100_hash_equal,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "continuation_200_pulses_bit_identical",
            p.continuation_200_pulses_bit_identical,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "continuation_200_snapshot_bit_identical",
            p.continuation_200_snapshot_bit_identical,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "continuation_200_hash_identical",
            p.continuation_200_hash_identical,
            true,
        );
        write_corruption_case_json(
            &mut out,
            2,
            "payload_bitflip_case",
            &p.payload_bitflip_case,
            true,
        );
        write_corruption_case_json(&mut out, 2, "oversized_case", &p.oversized_case, false);
        write_indent(&mut out, 1);
        out.push_str("},\n");
    } else {
        out.push_str("null,\n");
    }

    write_indent(&mut out, 1);
    out.push_str("\"plasticity_and_lesion\": ");
    if let Some(pl) = plasticity {
        out.push_str("{\n");
        write_kv_bool(&mut out, 2, "all_passed", pl.all_passed(), true);
        write_kv_usize(
            &mut out,
            2,
            "frozen_steps_tested",
            pl.frozen_steps_tested,
            true,
        );
        write_kv_u64(
            &mut out,
            2,
            "max_frozen_hash_changed_bits",
            u64::from(pl.max_frozen_hash_changed_bits),
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "frozen_topology_invariant",
            pl.frozen_topology_invariant,
            true,
        );
        write_kv_u64(
            &mut out,
            2,
            "post_growth_frozen_hash_changed_bits",
            u64::from(pl.post_growth_frozen_hash_changed_bits),
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "lesion_preserved_natural_mode",
            pl.lesion_preserved_natural_mode,
            true,
        );
        write_kv_bool(
            &mut out,
            2,
            "lesion_zeroed_all_pore_conductances",
            pl.lesion_zeroed_all_pore_conductances,
            false,
        );
        write_indent(&mut out, 1);
        out.push_str("},\n");
    } else {
        out.push_str("null,\n");
    }

    write_indent(&mut out, 1);
    out.push_str("\"latency\": ");
    if let Some(lat) = latency {
        out.push_str("{\n");
        write_kv_usize(&mut out, 2, "steps_profiled", lat.steps_profiled, true);
        write_kv_usize(
            &mut out,
            2,
            "num_ambient_channels",
            lat.num_ambient_channels,
            true,
        );
        write_kv_usize(&mut out, 2, "num_actuators", lat.num_actuators, true);
        write_dist_json(&mut out, 2, "step_latency_us", &lat.step_latency_us, true);
        write_dist_json(
            &mut out,
            2,
            "full_tick_latency_us",
            &lat.full_tick_latency_us,
            true,
        );
        write_kv_usize(
            &mut out,
            2,
            "peak_used_state_bytes",
            lat.peak_used_state_bytes,
            true,
        );
        write_kv_usize(
            &mut out,
            2,
            "final_used_state_bytes",
            lat.final_used_state_bytes,
            true,
        );
        write_kv_usize(
            &mut out,
            2,
            "checkpoint_bytes",
            lat.checkpoint_bytes,
            lat.os_memory.is_some(),
        );
        if let Some(os) = lat.os_memory {
            write_kv_usize(&mut out, 2, "vm_rss_bytes", os.vm_rss_bytes, true);
            write_kv_usize(&mut out, 2, "vm_hwm_bytes", os.vm_hwm_bytes, false);
        }
        write_indent(&mut out, 1);
        out.push_str("}\n");
    } else {
        out.push_str("null\n");
    }

    out.push_str("}\n");
    out
}

fn write_arm_summary_json(
    out: &mut String,
    indent: usize,
    s: &ArmCohortSummary,
    trailing_comma: bool,
) {
    write_indent(out, indent);
    out.push_str("{\n");
    write_kv_str(out, indent + 1, "arm", s.arm.as_str(), true);
    write_kv_usize(out, indent + 1, "seed_count", s.seed_count, true);
    write_kv_u64(out, indent + 1, "ticks_per_seed", s.ticks_per_seed, true);
    write_dist_json(
        out,
        indent + 1,
        "achieved_influx_per_tick",
        &s.achieved_influx_per_tick,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "achieved_stress_per_tick",
        &s.achieved_stress_per_tick,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "net_achieved_energy_per_tick",
        &s.net_achieved_energy_per_tick,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "first_half_influx_per_tick",
        &s.first_half_influx_per_tick,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "second_half_influx_per_tick",
        &s.second_half_influx_per_tick,
        true,
    );
    write_dist_json(out, indent + 1, "final_atp", &s.final_atp, true);
    write_dist_json(out, indent + 1, "cell_count", &s.cell_count, true);
    write_dist_json(out, indent + 1, "tube_count", &s.tube_count, true);
    write_dist_json(
        out,
        indent + 1,
        "stiff_highway_tubes",
        &s.stiff_highway_tubes,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "inhibitory_tubes",
        &s.inhibitory_tubes,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "active_echo_delay_tubes",
        &s.active_echo_delay_tubes,
        true,
    );
    write_dist_json(out, indent + 1, "mean_tau", &s.mean_tau, true);
    write_dist_json(out, indent + 1, "max_tau", &s.max_tau, true);
    write_dist_json(
        out,
        indent + 1,
        "tau_octave_count",
        &s.tau_octave_count,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "mean_interventional_agency",
        &s.mean_interventional_agency,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "mean_prediction_error",
        &s.mean_prediction_error,
        true,
    );
    write_dist_json(
        out,
        indent + 1,
        "noisy_tv_conductance_share",
        &s.noisy_tv_conductance_share,
        true,
    );

    write_indent(out, indent + 1);
    out.push_str("\"band_summaries\": [\n");
    for (b_idx, b) in s.band_summaries.iter().enumerate() {
        write_indent(out, indent + 2);
        out.push_str("{\n");
        write_kv_str(out, indent + 3, "label", &b.label, true);
        write_kv_str(out, indent + 3, "role", b.role.as_str(), true);
        write_kv_usize(out, indent + 3, "channel_width", b.channel_width, true);
        write_kv_f64(
            out,
            indent + 3,
            "uniform_random_share_expectation",
            b.uniform_random_share_expectation,
            true,
        );
        write_dist_json(
            out,
            indent + 3,
            "conductance_share",
            &b.conductance_share,
            true,
        );
        write_dist_json(out, indent + 3, "pore_count", &b.pore_count, true);
        write_dist_json(
            out,
            indent + 3,
            "conductance_weighted_delay",
            &b.conductance_weighted_delay,
            false,
        );
        write_indent(out, indent + 2);
        if b_idx + 1 < s.band_summaries.len() {
            out.push_str("},\n");
        } else {
            out.push_str("}\n");
        }
    }
    write_indent(out, indent + 1);
    out.push_str("],\n");

    write_individuality_json(out, indent + 1, "individuality", &s.individuality, false);

    write_indent(out, indent);
    if trailing_comma {
        out.push_str("},\n");
    } else {
        out.push_str("}\n");
    }
}

pub(crate) fn write_dist_json(
    out: &mut String,
    indent: usize,
    key: &str,
    d: &EmpiricalDistribution,
    trailing_comma: bool,
) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {{", escape_json_str(key));
    let _ = write!(
        out,
        "\"count\": {}, \"min\": {}, \"p10\": {}, \"p25\": {}, \"p50\": {}, \"p75\": {}, \"p90\": {}, \"p95\": {}, \"p99\": {}, \"max\": {}, \"iqr\": {}, \"mean\": {}, \"std_dev\": {}, \"cv\": {}",
        d.count,
        fmt_json_f64(d.min),
        fmt_json_f64(d.p10),
        fmt_json_f64(d.p25),
        fmt_json_f64(d.p50),
        fmt_json_f64(d.p75),
        fmt_json_f64(d.p90),
        fmt_json_f64(d.p95),
        fmt_json_f64(d.p99),
        fmt_json_f64(d.max),
        fmt_json_f64(d.iqr),
        fmt_json_f64(d.mean),
        fmt_json_f64(d.std_dev),
        fmt_json_f64(d.cv),
    );
    if trailing_comma {
        out.push_str("},\n");
    } else {
        out.push_str("}\n");
    }
}

pub(crate) fn write_mann_whitney_json(
    out: &mut String,
    indent: usize,
    key: &str,
    c: &MannWhitneyComparison,
    trailing_comma: bool,
) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {{", escape_json_str(key));
    let _ = write!(
        out,
        "\"n_a\": {}, \"n_b\": {}, \"u_a\": {}, \"u_b\": {}, \"u_stat\": {}, \"rank_biserial\": {}, \"median_diff\": {}, \"z_score\": {}, \"p_normal_approx\": {}, \"p_permutation\": {}, \"permutations\": {}",
        c.n_a,
        c.n_b,
        fmt_json_f64(c.u_a),
        fmt_json_f64(c.u_b),
        fmt_json_f64(c.u_stat),
        fmt_json_f64(c.rank_biserial),
        fmt_json_f64(c.median_diff),
        fmt_json_f64(c.z_score),
        fmt_json_f64(c.p_normal_approx),
        fmt_json_f64(c.p_permutation),
        c.permutations,
    );
    if trailing_comma {
        out.push_str("},\n");
    } else {
        out.push_str("}\n");
    }
}

pub(crate) fn write_individuality_json(
    out: &mut String,
    indent: usize,
    key: &str,
    ind: &IndividualityDivergence,
    trailing_comma: bool,
) {
    write_indent(out, indent);
    let _ = writeln!(out, "\"{}\": {{", escape_json_str(key));
    write_kv_usize(out, indent + 1, "individual_count", ind.individual_count, true);
    write_kv_f64(
        out,
        indent + 1,
        "inter_individual_cv",
        ind.inter_individual_cv,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "mean_pairwise_pore_cosine_distance",
        ind.mean_pairwise_pore_cosine_distance,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "max_pairwise_pore_cosine_distance",
        ind.max_pairwise_pore_cosine_distance,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "mean_pairwise_tau_jsd",
        ind.mean_pairwise_tau_jsd,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "max_pairwise_tau_jsd",
        ind.max_pairwise_tau_jsd,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "mean_trajectory_entropy_bits",
        ind.mean_trajectory_entropy_bits,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "normalized_trajectory_entropy",
        ind.normalized_trajectory_entropy,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "trajectory_divergence_bits",
        ind.trajectory_divergence_bits,
        true,
    );
    write_kv_f64(
        out,
        indent + 1,
        "composite_individuality_index",
        ind.composite_individuality_index,
        false,
    );
    write_indent(out, indent);
    if trailing_comma {
        out.push_str("},\n");
    } else {
        out.push_str("}\n");
    }
}

fn write_corruption_case_json(
    out: &mut String,
    indent: usize,
    key: &str,
    c: &CorruptionCaseOutcome,
    trailing_comma: bool,
) {
    write_indent(out, indent);
    let _ = write!(
        out,
        "\"{}\": {{\"label\": \"{}\", \"passed\": {}, \"poisoned_after_reject\": {}, \"live_hash_preserved\": {}, \"step_blocked_with_poison\": {}, \"encode_blocked_with_poison\": {}}}",
        escape_json_str(key),
        escape_json_str(&c.label),
        c.passed(),
        c.poisoned_after_reject,
        c.live_hash_preserved,
        c.step_blocked_with_poison,
        c.encode_blocked_with_poison,
    );
    if trailing_comma {
        out.push_str(",\n");
    } else {
        out.push('\n');
    }
}

pub(crate) fn write_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

pub(crate) fn write_kv_str(out: &mut String, indent: usize, key: &str, val: &str, comma: bool) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": \"{}\"", escape_json_str(key), escape_json_str(val));
    out.push_str(if comma { ",\n" } else { "\n" });
}

pub(crate) fn write_kv_u64(out: &mut String, indent: usize, key: &str, val: u64, comma: bool) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {val}", escape_json_str(key));
    out.push_str(if comma { ",\n" } else { "\n" });
}

pub(crate) fn write_kv_usize(out: &mut String, indent: usize, key: &str, val: usize, comma: bool) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {val}", escape_json_str(key));
    out.push_str(if comma { ",\n" } else { "\n" });
}

pub(crate) fn write_kv_bool(out: &mut String, indent: usize, key: &str, val: bool, comma: bool) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {val}", escape_json_str(key));
    out.push_str(if comma { ",\n" } else { "\n" });
}

pub(crate) fn write_kv_f64(out: &mut String, indent: usize, key: &str, val: f64, comma: bool) {
    write_indent(out, indent);
    let _ = write!(out, "\"{}\": {}", escape_json_str(key), fmt_json_f64(val));
    out.push_str(if comma { ",\n" } else { "\n" });
}

pub(crate) fn fmt_json_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "null".to_string()
    }
}

pub(crate) fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}
