//! Live cross-crate empirical observation campaign runner for Alpha 11 (`alpha11-observe`).
//!
//! Executes Studies 0 through 6 across `a11-core`, `a11-worlds`, and `a11-portals`, plus
//! longitudinal whole-body developmental trajectory observation (`observe-development`) on
//! [`Alpha11Organism`] inside the unscripted 276-channel `OpenSocialHabitat`.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, WorldMedium};
use a11_observe::TwinArm;
use a11_observe::campaign::{
    CampaignError, run_study_0_forensics_and_latency, run_study_1_sense_organogenesis,
    run_study_2_permuted_wire_invariance, run_study_3_regime_shifts,
};
use a11_observe::campaign_social_portals::{
    TerrariumCohortArm, run_study_4_harmonic_and_individuality, run_study_5_cultural_terrarium,
    run_study_6_human_activity_portals,
};
use a11_observe::gates::{holm_adjust, paired_gate};
use a11_observe::stats::DeterministicRng;
use a11_portals::{EMBODIED_ORGAN_ACTUATORS, EMBODIED_ORGAN_CHANNELS, EmbodiedBodyOrgans};
use a11_worlds::{
    EARTH_AUDIO_FRAME_SAMPLES, MultiGenTerrarium, TERRARIUM_ACTUATORS, TERRARIUM_CHANNELS,
};

const OPEN_CHANNELS: usize = TERRARIUM_CHANNELS + EMBODIED_ORGAN_CHANNELS;
const OPEN_ACTUATORS: usize = TERRARIUM_ACTUATORS + EMBODIED_ORGAN_ACTUATORS;
const ELDER_WARMUP_TICKS: usize = 640;

fn resolve_workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn run_all() -> Result<(), CampaignError> {
    let t0 = Instant::now();
    let workspace_root = resolve_workspace_root();
    let output_dir = workspace_root.join("observations");

    println!(
        "[alpha11-observe] Workspace: {} | Output: {}",
        workspace_root.display(),
        output_dir.display()
    );

    // Phase 1: Study 0 MUST run single-threaded before spawning worker threads so latency & RSS are uncontended.
    println!("[Study 0] Running source hygiene, persistence, plasticity/lesion & O(1) latency audits...");
    let s0 = run_study_0_forensics_and_latency(&workspace_root, &output_dir)?;
    println!(
        "[Study 0] Done ({:.2}s): a11-core clean={} (strings={}, forbidden={}, unsafe={}, unwrap={}) | persistence={} | plasticity={} | 128ch p50={:.2}us p99={:.2}us | 4096ch p50={:.2}us p99={:.2}us",
        t0.elapsed().as_secs_f64(),
        s0.core_hygiene.is_clean(),
        s0.core_hygiene.string_literal_count,
        s0.core_hygiene.forbidden_identifier_count,
        s0.core_hygiene.unsafe_block_count,
        s0.core_hygiene.unwrap_call_count,
        s0.persistence.all_passed(),
        s0.plasticity.all_passed(),
        s0.radiant_128_latency.step_latency_us.p50,
        s0.radiant_128_latency.step_latency_us.p99,
        s0.dense_4096_latency.step_latency_us.p50,
        s0.dense_4096_latency.step_latency_us.p99,
    );

    // Phase 2: Run purely deterministic multi-seed twin cohorts (Studies 1..=6) across 4 balanced worker threads.
    println!("[Studies 1..6] Launching deterministic 12-seed twin & portal studies...");
    let ((s1a, s1b, s2), s3, (s4, s5), s6) = std::thread::scope(|scope| {
        let out_ref = &output_dir;

        let h_1_2 = scope.spawn(move || -> Result<_, CampaignError> {
            let (out_1a, out_1b) = run_study_1_sense_organogenesis(out_ref)?;
            let out_2 = run_study_2_permuted_wire_invariance(out_ref, &out_1a, &out_1b)?;
            Ok((out_1a, out_1b, out_2))
        });

        let h_3 = scope.spawn(move || run_study_3_regime_shifts(out_ref));

        let h_4_5 = scope.spawn(move || -> Result<_, CampaignError> {
            let out_4 = run_study_4_harmonic_and_individuality(out_ref)?;
            let out_5 = run_study_5_cultural_terrarium(out_ref)?;
            Ok((out_4, out_5))
        });

        let h_6 = scope.spawn(move || run_study_6_human_activity_portals(out_ref));

        let r_1_2 = h_1_2.join().map_err(|_| {
            CampaignError::Io(std::io::Error::other("Study 1/2 worker thread panicked"))
        })??;
        let r_3 = h_3.join().map_err(|_| {
            CampaignError::Io(std::io::Error::other("Study 3 worker thread panicked"))
        })??;
        let r_4_5 = h_4_5.join().map_err(|_| {
            CampaignError::Io(std::io::Error::other("Study 4/5 worker thread panicked"))
        })??;
        let r_6 = h_6.join().map_err(|_| {
            CampaignError::Io(std::io::Error::other("Study 6 worker thread panicked"))
        })??;

        Ok::<_, CampaignError>((r_1_2, r_3, r_4_5, r_6))
    })?;

    if let (Some(n1a), Some(f1a), Some(n1b), Some(f1b)) = (
        s1a.report.arm_summary(TwinArm::Natural),
        s1a.report.arm_summary(TwinArm::FrozenPlasticity),
        s1b.report.arm_summary(TwinArm::Natural),
        s1b.report.arm_summary(TwinArm::FrozenPlasticity),
    ) {
        println!(
            "[Study 1a Bright] Natural influx/tick p50={:.4} vs Frozen={:.4} | NoisyTV share p50={:.2}%",
            n1a.achieved_influx_per_tick.p50,
            f1a.achieved_influx_per_tick.p50,
            n1a.noisy_tv_conductance_share.p50 * 100.0,
        );
        println!(
            "[Study 1b Dark]   Natural influx/tick p50={:.4} vs Frozen={:.4} | NoisyTV share p50={:.2}%",
            n1b.achieved_influx_per_tick.p50,
            f1b.achieved_influx_per_tick.p50,
            n1b.noisy_tv_conductance_share.p50 * 100.0,
        );
    }

    if let (Some(p_b), Some(p_d)) = (
        s2.permuted_bright.report.arm_summary(TwinArm::Natural),
        s2.permuted_dark.report.arm_summary(TwinArm::Natural),
    ) {
        println!(
            "[Study 2 Permuted] Bright Natural influx/tick p50={:.4} (NoisyTV={:.2}%) | Dark Natural influx/tick p50={:.4} (NoisyTV={:.2}%)",
            p_b.achieved_influx_per_tick.p50,
            p_b.noisy_tv_conductance_share.p50 * 100.0,
            p_d.achieved_influx_per_tick.p50,
            p_d.noisy_tv_conductance_share.p50 * 100.0,
        );
    }

    println!(
        "[Study 3a Cavefish] Natural pings Day={:.1} -> Dark={:.1} | Echo energy Day={:.3} -> Dark={:.3}",
        s3.cavefish_pre_pings_nat.p50,
        s3.cavefish_post_pings_nat.p50,
        s3.cavefish_pre_echo_e_nat.p50,
        s3.cavefish_post_echo_e_nat.p50,
    );
    println!(
        "[Study 3b Reversal] Splits Pre={:.1} Post={:.1} | Temp mean 1201-1500={:.4} -> 1501-1800={:.4} | Influx Shock={:.4} -> Late={:.4}",
        s3.reversal_pre_splits_nat.p50,
        s3.reversal_post_splits_nat.p50,
        s3.reversal_mean_temp_pre_nat.p50,
        s3.reversal_mean_temp_post_nat.p50,
        s3.reversal_post_shock_influx_nat.p50,
        s3.reversal_late_recovery_influx_nat.p50,
    );

    if let Some(nh) = s4.harmonic_report.arm_summary(TwinArm::Natural) {
        let ind = &s4.identical_twins_jittered.cohort_summary.individuality;
        println!(
            "[Study 4 Harmonic & Twins] Harmonic Natural influx/tick p50={:.4} | Identical-Twin 1e-4 Composite Individuality={:.4} (0-jitter={:.6})",
            nh.achieved_influx_per_tick.p50,
            ind.composite_individuality_index,
            s4.identical_twins_zero_jitter
                .cohort_summary
                .individuality
                .composite_individuality_index,
        );
    }

    if let (Some(cohab), Some(solo)) = (
        s5.arm_summaries
            .iter()
            .find(|a| a.arm == TerrariumCohortArm::CohabitingNewborn),
        s5.arm_summaries
            .iter()
            .find(|a| a.arm == TerrariumCohortArm::SoloOrphan),
    ) {
        println!(
            "[Study 5 Terrarium] CohabitingNewborn influx p50={:.4} night_stress={:.4} vs SoloOrphan influx={:.4} night_stress={:.4}",
            cohab.influx_per_tick.p50,
            cohab.night_stress_per_night_tick.p50,
            solo.influx_per_tick.p50,
            solo.night_stress_per_night_tick.p50,
        );
    }

    println!(
        "[Study 6 Portals] Completed {} portal suites | Total campaign wall time: {:.2}s",
        s6.portals.len(),
        t0.elapsed().as_secs_f64()
    );

    Ok(())
}

// =============================================================================
// Longitudinal Whole-Body Developmental Trajectory (`observe-development`)
// =============================================================================

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
fn clamp_nonneg(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RoleBoutAccum {
    sum: [f64; 6],
    count: u64,
}

#[derive(Clone)]
struct DevOpenHabitat {
    terrarium: MultiGenTerrarium,
    earth_mode: bool,
    mute_peer_voice: bool,
    elder: Option<(Alpha11Organism, EmbodiedBodyOrgans, [f32; OPEN_CHANNELS], f32, f32)>,
    focal_slot: usize,
    focal_body: EmbodiedBodyOrgans,
    focal_ambient: [f32; OPEN_CHANNELS],
    focal_influx: f32,
    focal_stress: f32,
    shared_slab_ticks: u64,
    prev_coop_grove_pods: u64,
    prev_occluded_approaches: u64,
    window_influx_sum: f64,
    window_net_sum: f64,
    window_ticks: u64,
    prev_in_distress: bool,
    distress_from_ticks: u64,
    distress_recoveries: u64,
    drought_net_sum: f64,
    drought_ticks: u64,
    flood_net_sum: f64,
    flood_ticks: u64,
    occluded_pair_ticks: u64,
    occluded_closing_sum: f64,
    prev_elder_pos: Option<(f32, f32)>,
    last_role_tick: [[u64; 2]; 2],
    role_bout_idx: [[u64; 2]; 2],
    role_acc: [[[RoleBoutAccum; 2]; 2]; 2],
}

impl DevOpenHabitat {
    fn new(seed: u64, with_elder: bool) -> Self {
        Self::with_spawn_mode(seed, with_elder, true, false, None)
    }

    fn new_vocal_muted(seed: u64) -> Self {
        Self::with_spawn_mode(seed, true, true, true, None)
    }

    fn new_barren(seed: u64, with_elder: bool) -> Self {
        Self::with_spawn_mode(seed, with_elder, false, false, None)
    }

    fn with_spawn_mode(
        seed: u64,
        with_elder: bool,
        earth_mode: bool,
        mute_peer_voice: bool,
        open_pos: Option<(f32, f32)>,
    ) -> Self {
        let mut terrarium = if earth_mode {
            MultiGenTerrarium::new_earth_ecology(seed, 1)
        } else {
            MultiGenTerrarium::new(seed, 1)
        };
        let elder = if with_elder {
            let mut e_brain =
                Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
            let mut e_body = EmbodiedBodyOrgans::new_adult(seed ^ 0xE1DE_B0D1_2026_0002);
            let mut e_amb = [0.0_f32; OPEN_CHANNELS];
            for _ in 0..ELDER_WARMUP_TICKS {
                let tv = terrarium.agent_view(0);
                for (i, slot) in e_amb[..TERRARIUM_CHANNELS].iter_mut().enumerate() {
                    *slot = clamp_unit(tv.ambient_in.get(i).copied().unwrap_or(0.0));
                }
                e_amb[TERRARIUM_CHANNELS..].copy_from_slice(e_body.sensory_channels());
                let view = FieldView {
                    ambient_in: &e_amb,
                    metabolic_influx: clamp_nonneg(tv.metabolic_influx),
                    tissue_stress: clamp_nonneg(tv.tissue_stress + e_body.last_organ_stress()),
                    num_actuators: OPEN_ACTUATORS,
                };
                let ep = e_brain
                    .step(view)
                    .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACTUATORS));
                let mut eb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
                for (j, slot) in eb.iter_mut().enumerate() {
                    *slot = clamp_unit(
                        ep.actuator_out
                            .get(TERRARIUM_ACTUATORS + j)
                            .copied()
                            .unwrap_or(0.0),
                    );
                }
                let _ = e_body.step_limbs_and_radiate_voice(&eb);
                if earth_mode {
                    let (es, eh) = e_body.embodied_kinematics();
                    terrarium.set_agent_embodied_kinematics(0, es, eh);
                    terrarium.set_agent_embodied_voice_rms(0, e_body.vocal().last_radiated_rms());
                }

                let mut ew = FieldPulse {
                    actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
                };
                for i in 0..TERRARIUM_ACTUATORS {
                    ew.actuator_out
                        .push(clamp_unit(ep.actuator_out.get(i).copied().unwrap_or(0.0)));
                }
                terrarium.apply_population_pulses(&[ew]);
                if earth_mode {
                    e_body.apply_terrain_sway(terrarium.agent_lateral_sway(0));
                    let mut env_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                    let mut env_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                    let enc = terrarium.binaural_earth_waves_for_agent(0, &mut env_l, &mut env_r);
                    e_body.finish_binaural_ecology_transduction(
                        None,
                        Some((&env_l, &env_r)),
                        enc,
                    );
                } else {
                    e_body.finish_binaural_transduction(None);
                }
            }
            e_body.vocal_mut().reset_window_observations();
            if let Some(pos) = open_pos {
                terrarium.respawn_at_position(1, pos);
            } else {
                terrarium.respawn_newborn_in_slot(1);
            }
            Some((e_brain, e_body, e_amb, 0.0, 0.0))
        } else {
            // Advance identical day/night & seasonal clock with silent physics so birth phase matches.
            let silent = [FieldPulse::silent(TERRARIUM_ACTUATORS)];
            for _ in 0..ELDER_WARMUP_TICKS {
                terrarium.apply_population_pulses(&silent);
            }
            if let Some(pos) = open_pos {
                terrarium.respawn_at_position(0, pos);
            } else {
                terrarium.respawn_newborn_in_slot(0);
            }
            None
        };

        let focal_slot = if with_elder { 1 } else { 0 };
        let focal_body = EmbodiedBodyOrgans::new_infant(seed ^ 0xF0CA_1B0D_2026_0003);
        let init_coop = terrarium.ecology_metrics().coop_grove_pods_cracked;
        let init_occ = terrarium.ecology_metrics().occluded_vocal_approaches;
        let init_e_pos = with_elder
            .then(|| terrarium.metrics().agent_positions.first().copied())
            .flatten();
        let mut hab = Self {
            terrarium,
            earth_mode,
            mute_peer_voice,
            elder,
            focal_slot,
            focal_body,
            focal_ambient: [0.0; OPEN_CHANNELS],
            focal_influx: 0.0,
            focal_stress: 0.0,
            shared_slab_ticks: 0,
            prev_coop_grove_pods: init_coop,
            prev_occluded_approaches: init_occ,
            window_influx_sum: 0.0,
            window_net_sum: 0.0,
            window_ticks: 0,
            prev_in_distress: false,
            distress_from_ticks: 0,
            distress_recoveries: 0,
            drought_net_sum: 0.0,
            drought_ticks: 0,
            flood_net_sum: 0.0,
            flood_ticks: 0,
            occluded_pair_ticks: 0,
            occluded_closing_sum: 0.0,
            prev_elder_pos: init_e_pos,
            last_role_tick: [[0; 2]; 2],
            role_bout_idx: [[0; 2]; 2],
            role_acc: [[[RoleBoutAccum::default(); 2]; 2]; 2],
        };
        hab.refresh_views();
        hab
    }

    fn reset_stage_window(&mut self) {
        self.window_influx_sum = 0.0;
        self.window_net_sum = 0.0;
        self.window_ticks = 0;
        self.prev_in_distress = false;
        self.distress_from_ticks = 0;
        self.distress_recoveries = 0;
        self.drought_net_sum = 0.0;
        self.drought_ticks = 0;
        self.flood_net_sum = 0.0;
        self.flood_ticks = 0;
        self.occluded_pair_ticks = 0;
        self.occluded_closing_sum = 0.0;
        self.last_role_tick = [[0; 2]; 2];
        self.role_bout_idx = [[0; 2]; 2];
        self.role_acc = [[[RoleBoutAccum::default(); 2]; 2]; 2];
        self.prev_coop_grove_pods = self.terrarium.ecology_metrics().coop_grove_pods_cracked;
        self.prev_occluded_approaches = self.terrarium.ecology_metrics().occluded_vocal_approaches;
        self.focal_body.vocal_mut().reset_window_observations();
        if let Some((_, ref mut e_body, _, _, _)) = self.elder {
            e_body.vocal_mut().reset_window_observations();
        }
    }

    fn role_swap_cv_separation(&self) -> f64 {
        let mut sum_sep = 0.0_f64;
        let mut valid_bins = 0_usize;
        for bin in 0..2 {
            let a_ev = &self.role_acc[bin][0][0];
            let a_od = &self.role_acc[bin][0][1];
            let b_ev = &self.role_acc[bin][1][0];
            let b_od = &self.role_acc[bin][1][1];
            if a_ev.count >= 2 && a_od.count >= 2 && b_ev.count >= 2 && b_od.count >= 2 {
                let mut dot = 0.0_f64;
                for k in 0..6 {
                    let d_ev = (a_ev.sum[k] / (a_ev.count as f64))
                        - (b_ev.sum[k] / (b_ev.count as f64));
                    let d_od = (a_od.sum[k] / (a_od.count as f64))
                        - (b_od.sum[k] / (b_od.count as f64));
                    dot += d_ev * d_od;
                }
                sum_sep += dot;
                valid_bins += 1;
            }
        }
        if valid_bins > 0 {
            sum_sep / (valid_bins as f64)
        } else {
            0.0
        }
    }

    fn refresh_views(&mut self) {
        if let Some((_, ref e_body, ref mut e_amb, ref mut e_in, ref mut e_st)) = self.elder {
            let ev = self.terrarium.agent_view(0);
            for (i, slot) in e_amb[..TERRARIUM_CHANNELS].iter_mut().enumerate() {
                *slot = clamp_unit(ev.ambient_in.get(i).copied().unwrap_or(0.0));
            }
            e_amb[TERRARIUM_CHANNELS..].copy_from_slice(e_body.sensory_channels());
            *e_in = clamp_nonneg(ev.metabolic_influx);
            *e_st = clamp_nonneg(ev.tissue_stress + e_body.last_organ_stress());
        }

        let fv = self.terrarium.agent_view(self.focal_slot);
        for (i, slot) in self.focal_ambient[..TERRARIUM_CHANNELS]
            .iter_mut()
            .enumerate()
        {
            *slot = clamp_unit(fv.ambient_in.get(i).copied().unwrap_or(0.0));
        }
        self.focal_ambient[TERRARIUM_CHANNELS..].copy_from_slice(self.focal_body.sensory_channels());
        self.focal_influx = clamp_nonneg(fv.metabolic_influx);
        self.focal_stress = clamp_nonneg(fv.tissue_stress + self.focal_body.last_organ_stress());
    }
}

impl WorldMedium for DevOpenHabitat {
    fn num_ambient_channels(&self) -> usize {
        OPEN_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        OPEN_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.focal_ambient,
            metabolic_influx: self.focal_influx,
            tissue_stress: self.focal_stress,
            num_actuators: OPEN_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, focal_pulse: &FieldPulse) {
        let mut focal_world = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        for i in 0..TERRARIUM_ACTUATORS {
            focal_world.actuator_out.push(clamp_unit(
                focal_pulse.actuator_out.get(i).copied().unwrap_or(0.0),
            ));
        }
        let mut focal_body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        for (j, slot) in focal_body_act.iter_mut().enumerate() {
            *slot = clamp_unit(
                focal_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
        }

        let pre_positions = self.terrarium.metrics().agent_positions.clone();
        let e_prev_opt = (self.elder.is_some()).then(|| pre_positions.first().copied()).flatten();
        let f_prev_opt = pre_positions.get(self.focal_slot).copied();

        // Read-only bout-split trajectory/role-swap observation before stepping physics.
        if let (Some(e_prev), Some(f_prev), Some(e_old)) =
            (e_prev_opt, f_prev_opt, self.prev_elder_pos)
            && self.terrarium.has_line_of_sight(f_prev, e_prev)
        {
            let dx = e_prev.0 - f_prev.0;
            let dy = e_prev.1 - f_prev.1;
            let dist = dx.hypot(dy);
            if (1.4..=10.5).contains(&dist) {
                let ux = dx / dist;
                let uy = dy / dist;
                let e_vx = e_prev.0 - e_old.0;
                let e_vy = e_prev.1 - e_old.1;
                let v_peer_to_self = -(e_vx * ux + e_vy * uy);
                let class_opt = if v_peer_to_self > 0.025 {
                    Some(0_usize)
                } else if v_peer_to_self < -0.025 {
                    Some(1_usize)
                } else {
                    None
                };
                if let Some(cls) = class_opt {
                    let bin = usize::from(dist >= 5.0);
                    if self.window_ticks > self.last_role_tick[bin][cls].saturating_add(8) {
                        self.role_bout_idx[bin][cls] =
                            self.role_bout_idx[bin][cls].saturating_add(1);
                    }
                    self.last_role_tick[bin][cls] = self.window_ticks;
                    let parity = (self.role_bout_idx[bin][cls] & 1) as usize;
                    let fx = focal_world.actuator_out.first().copied().unwrap_or(0.0);
                    let fy = focal_world.actuator_out.get(1).copied().unwrap_or(0.0);
                    let gx = focal_world.actuator_out.get(2).copied().unwrap_or(0.0);
                    let gy = focal_world.actuator_out.get(3).copied().unwrap_or(0.0);
                    let ping = focal_world.actuator_out.get(4).copied().unwrap_or(0.0).abs();
                    let hand_e = 0.25
                        * (focal_body_act[0].abs()
                            + focal_body_act[1].abs()
                            + focal_body_act[2].abs()
                            + focal_body_act[3].abs());
                    let voc_e = 0.20
                        * (ping
                            + focal_body_act[6].abs()
                            + focal_body_act[7].abs()
                            + focal_body_act[8].abs()
                            + focal_body_act[9].abs());
                    let y = [
                        f64::from(fx * ux + fy * uy),
                        f64::from(-fx * uy + fy * ux),
                        f64::from(gx * ux + gy * uy),
                        f64::from(-gx * uy + gy * ux),
                        f64::from(hand_e),
                        f64::from(voc_e),
                    ];
                    let acc = &mut self.role_acc[bin][cls][parity];
                    for k in 0..6 {
                        acc.sum[k] += y[k];
                    }
                    acc.count = acc.count.saturating_add(1);
                }
            }
        }
        if let Some(e_prev) = e_prev_opt {
            self.prev_elder_pos = Some(e_prev);
        }

        if let Some((ref mut e_brain, ref mut e_body, ref e_amb, e_in, e_st)) = self.elder {
            let e_view = FieldView {
                ambient_in: e_amb,
                metabolic_influx: e_in,
                tissue_stress: e_st,
                num_actuators: OPEN_ACTUATORS,
            };
            let ep = e_brain
                .step(e_view)
                .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACTUATORS));
            let mut elder_world = FieldPulse {
                actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
            };
            for i in 0..TERRARIUM_ACTUATORS {
                elder_world
                    .actuator_out
                    .push(clamp_unit(ep.actuator_out.get(i).copied().unwrap_or(0.0)));
            }
            let mut e_body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
            for (j, slot) in e_body_act.iter_mut().enumerate() {
                *slot = clamp_unit(
                    ep.actuator_out
                        .get(TERRARIUM_ACTUATORS + j)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            let e_wave = e_body.step_limbs_and_radiate_voice(&e_body_act);
            let f_wave = self.focal_body.step_limbs_and_radiate_voice(&focal_body_act);

            if self.earth_mode {
                let (es, eh) = e_body.embodied_kinematics();
                let (fs_stab, fh) = self.focal_body.embodied_kinematics();
                self.terrarium.set_agent_embodied_kinematics(0, es, eh);
                self.terrarium.set_agent_embodied_kinematics(1, fs_stab, fh);
                let (e_vrms, f_vrms) = if self.mute_peer_voice {
                    (0.0, 0.0)
                } else {
                    (
                        e_body.vocal().last_radiated_rms(),
                        self.focal_body.vocal().last_radiated_rms(),
                    )
                };
                self.terrarium.set_agent_embodied_voice_rms(0, e_vrms);
                self.terrarium.set_agent_embodied_voice_rms(1, f_vrms);
            }

            if self.mute_peer_voice {
                if elder_world.actuator_out.len() >= 6 {
                    elder_world.actuator_out[4] = 0.0;
                    elder_world.actuator_out[5] = 0.0;
                }
                if focal_world.actuator_out.len() >= 6 {
                    focal_world.actuator_out[4] = 0.0;
                    focal_world.actuator_out[5] = 0.0;
                }
            }

            self.terrarium
                .apply_population_pulses(&[elder_world, focal_world]);

            let positions = &self.terrarium.metrics().agent_positions;
            let e_pos = positions.first().copied().unwrap_or((0.0, 0.0));
            let f_pos = positions.get(1).copied().unwrap_or((0.0, 0.0));
            let rel_e = (e_pos.0 - f_pos.0, e_pos.1 - f_pos.1);
            let rel_f = (f_pos.0 - e_pos.0, f_pos.1 - e_pos.1);
            let peer_diff = self.terrarium.acoustic_transmittance(f_pos, e_pos);
            let mut e_wave_diff = e_wave;
            let mut f_wave_diff = f_wave;
            if peer_diff < 0.999 {
                for s in &mut e_wave_diff {
                    *s *= peer_diff;
                }
                for s in &mut f_wave_diff {
                    *s *= peer_diff;
                }
            }
            let peer_for_focal = (!self.mute_peer_voice).then_some((&e_wave_diff, rel_e));
            let peer_for_elder = (!self.mute_peer_voice).then_some((&f_wave_diff, rel_f));

            if self.earth_mode {
                e_body.apply_terrain_sway(self.terrarium.agent_lateral_sway(0));
                self.focal_body
                    .apply_terrain_sway(self.terrarium.agent_lateral_sway(1));

                let dist = rel_e.0.hypot(rel_e.1);
                if dist <= 3.6 {
                    let coupling = 0.26 * (1.0 - dist / 3.6);
                    EmbodiedBodyOrgans::couple_shared_hands(e_body, &mut self.focal_body, coupling);
                    self.shared_slab_ticks = self.shared_slab_ticks.saturating_add(1);
                }

                let mut f_el = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let mut f_er = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let f_enc = self
                    .terrarium
                    .binaural_earth_waves_for_agent(1, &mut f_el, &mut f_er);
                self.focal_body.finish_binaural_ecology_transduction(
                    peer_for_focal,
                    Some((&f_el, &f_er)),
                    f_enc,
                );

                let mut e_el = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let mut e_er = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let e_enc = self
                    .terrarium
                    .binaural_earth_waves_for_agent(0, &mut e_el, &mut e_er);
                e_body.finish_binaural_ecology_transduction(
                    peer_for_elder,
                    Some((&e_el, &e_er)),
                    e_enc,
                );
            } else {
                self.focal_body.finish_binaural_transduction(peer_for_focal);
                e_body.finish_binaural_transduction(peer_for_elder);
            }
        } else {
            let _ = self.focal_body.step_limbs_and_radiate_voice(&focal_body_act);
            if self.earth_mode {
                let (fs_stab, fh) = self.focal_body.embodied_kinematics();
                self.terrarium.set_agent_embodied_kinematics(0, fs_stab, fh);
            }
            self.terrarium.apply_population_pulses(&[focal_world]);
            if self.earth_mode {
                self.focal_body
                    .apply_terrain_sway(self.terrarium.agent_lateral_sway(0));
                let mut f_el = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let mut f_er = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
                let f_enc = self
                    .terrarium
                    .binaural_earth_waves_for_agent(0, &mut f_el, &mut f_er);
                self.focal_body.finish_binaural_ecology_transduction(
                    None,
                    Some((&f_el, &f_er)),
                    f_enc,
                );
            } else {
                self.focal_body.finish_binaural_transduction(None);
            }
        }

        if let (Some(e_prev), Some(f_prev)) = (e_prev_opt, f_prev_opt)
            && !self.terrarium.has_line_of_sight(f_prev, e_prev)
        {
            let post_pos = &self.terrarium.metrics().agent_positions;
            if let (Some(&e_now), Some(&f_now)) = (post_pos.first(), post_pos.get(self.focal_slot))
            {
                let d_prev = (e_prev.0 - f_prev.0).hypot(e_prev.1 - f_prev.1);
                let d_now = (e_now.0 - f_now.0).hypot(e_now.1 - f_now.1);
                if d_prev > 1.0 {
                    self.occluded_pair_ticks = self.occluded_pair_ticks.saturating_add(1);
                    self.occluded_closing_sum += f64::from(d_prev - d_now);
                }
            }
        }

        self.refresh_views();
        let net_tick = self.focal_influx - self.focal_stress;
        self.window_influx_sum += f64::from(self.focal_influx);
        self.window_net_sum += f64::from(net_tick);
        self.window_ticks = self.window_ticks.saturating_add(1);

        let in_distress = net_tick < -0.012;
        let recovered = net_tick >= -0.003;
        if self.prev_in_distress {
            self.distress_from_ticks = self.distress_from_ticks.saturating_add(1);
            if recovered {
                self.distress_recoveries = self.distress_recoveries.saturating_add(1);
            }
        }
        self.prev_in_distress = if self.prev_in_distress {
            !recovered
        } else {
            in_distress
        };

        let season_cos =
            (std::f32::consts::TAU * self.terrarium.ecology_metrics().season_phase).cos();
        if season_cos < 0.0 {
            self.drought_net_sum += f64::from(net_tick);
            self.drought_ticks = self.drought_ticks.saturating_add(1);
        } else {
            self.flood_net_sum += f64::from(net_tick);
            self.flood_ticks = self.flood_ticks.saturating_add(1);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StageSnapshot {
    tick: u64,
    cells: f64,
    tubes: f64,
    stiff_tubes: f64,
    echo_tubes: f64,
    mean_tau: f64,
    max_tau: f64,
    tau_octaves: f64,
    mitosis_births: f64,
    contradiction_splits: f64,
    sleep_ticks: f64,

    retina_pores: f64,
    echo_pores: f64,
    hand_pores: f64,
    vestibular_pores: f64,
    cochlea_pores: f64,
    distractor_pores: f64,
    distractor_share: f64,

    motor_mastery: f64,
    motor_jerk: f64,
    loco_mastery: f64,
    hand_mastery: f64,
    vest_mastery: f64,
    vocal_mastery: f64,

    atp: f64,
    net_per_tick: f64,
    drought_net_per_tick: f64,
    flood_net_per_tick: f64,
    markov_error_recovery: f64,
    agency: f64,
    pred_err: f64,

    hand_strokes: f64,
    canvas_coverage: f64,
    canvas_morans_i: f64,
    balance_mean_upright: f64,
    balance_max_streak: f64,

    voiced_rate: f64,
    tract_length_cm: f64,
    syllable_onsets: f64,
    syllable_transitions: f64,
    plosive_bursts: f64,
    alternating_responses: f64,
    shelter_echo_ticks: f64,
    mean_f0_hz: f64,
    mean_f1_hz: f64,
    mean_f2_hz: f64,
    f1_spread_hz: f64,
    f2_spread_hz: f64,

    pods_cracked: f64,
    coop_grove_pods: f64,
    occluded_vocal_approaches: f64,
    occluded_closing_rate: f64,
    referential_coord_score: f64,
    role_swap_separation: f64,
    coop_boulder_ticks: f64,
    huddling_ticks: f64,
    shared_slab_ticks: f64,
    elder_cells: f64,
    elder_atp: f64,
}

fn group_mastery(items: &[(u32, f32, f32)], lo: u32, hi_excl: u32) -> f64 {
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;
    for &(ch, mastery, _) in items {
        if ch >= lo && ch < hi_excl {
            sum += f64::from(mastery);
            cnt += 1;
        }
    }
    if cnt > 0 { sum / (cnt as f64) } else { 0.0 }
}

fn capture_stage(org: &Alpha11Organism, hab: &DevOpenHabitat) -> StageSnapshot {
    let s = org.snapshot();
    let mut ret = 0usize;
    let mut ech = 0usize;
    let mut hnd = 0usize;
    let mut vst = 0usize;
    let mut coc = 0usize;
    let mut dst = 0usize;
    for &ch in &s.sensor_pore_channels {
        match ch {
            0..=63 => ret += 1,
            64..=95 => ech += 1,
            96..=127 => dst += 1,
            128..=179 => hnd += 1,
            180..=195 => vst += 1,
            196..=259 => coc += 1,
            _ => dst += 1,
        }
    }
    let tot_p = s.sensor_pore_channels.len().max(1) as f64;
    let m_items = org.motor_maturation_by_channel();
    let bo = hab.focal_body.observation();
    let wt = hab.window_ticks.max(1) as f64;
    let vwt = bo.vocal.window_ticks.max(1) as f64;

    StageSnapshot {
        tick: s.tick,
        cells: s.cell_count as f64,
        tubes: s.tube_count as f64,
        stiff_tubes: s.stiff_highway_tubes as f64,
        echo_tubes: s.active_echo_delay_tubes as f64,
        mean_tau: f64::from(s.mean_tau),
        max_tau: f64::from(s.max_tau),
        tau_octaves: s.tau_octave_count as f64,
        mitosis_births: s.mitosis_births as f64,
        contradiction_splits: s.contradiction_splits as f64,
        sleep_ticks: s.rest_consolidation_ticks as f64,

        retina_pores: ret as f64,
        echo_pores: ech as f64,
        hand_pores: hnd as f64,
        vestibular_pores: vst as f64,
        cochlea_pores: coc as f64,
        distractor_pores: dst as f64,
        distractor_share: (dst as f64) / tot_p,

        motor_mastery: f64::from(org.mean_motor_mastery()),
        motor_jerk: f64::from(org.mean_motor_jerk()),
        loco_mastery: group_mastery(&m_items, 0, 6),
        hand_mastery: group_mastery(&m_items, 6, 10),
        vest_mastery: group_mastery(&m_items, 10, 12),
        vocal_mastery: group_mastery(&m_items, 12, 17),

        atp: f64::from(s.atp),
        net_per_tick: hab.window_net_sum / wt,
        drought_net_per_tick: if hab.drought_ticks > 0 {
            hab.drought_net_sum / (hab.drought_ticks as f64)
        } else {
            hab.window_net_sum / wt
        },
        flood_net_per_tick: if hab.flood_ticks > 0 {
            hab.flood_net_sum / (hab.flood_ticks as f64)
        } else {
            hab.window_net_sum / wt
        },
        markov_error_recovery: if hab.distress_from_ticks > 0 {
            (hab.distress_recoveries as f64) / (hab.distress_from_ticks as f64)
        } else {
            1.0
        },
        agency: f64::from(s.mean_interventional_agency),
        pred_err: f64::from(s.mean_prediction_error),

        hand_strokes: bo.scratchpad.active_stroke_ticks as f64,
        canvas_coverage: f64::from(bo.scratchpad.canvas_coverage_fraction),
        canvas_morans_i: f64::from(bo.scratchpad.spatial_autocorrelation),
        balance_mean_upright: f64::from(bo.balance.mean_upright_duration_ticks),
        balance_max_streak: bo.balance.max_upright_streak as f64,

        voiced_rate: (bo.vocal.voiced_ticks as f64) / vwt,
        tract_length_cm: f64::from(bo.vocal.tract_length_cm),
        syllable_onsets: bo.vocal.syllable_onsets as f64,
        syllable_transitions: bo.vocal.syllable_transitions as f64,
        plosive_bursts: bo.vocal.plosive_burst_frames as f64,
        alternating_responses: bo.vocal.alternating_response_ticks as f64,
        shelter_echo_ticks: bo.vocal.shelter_echo_ticks as f64,
        mean_f0_hz: f64::from(bo.vocal.mean_f0_hz),
        mean_f1_hz: f64::from(bo.vocal.mean_f1_hz),
        mean_f2_hz: f64::from(bo.vocal.mean_f2_hz),
        f1_spread_hz: f64::from(bo.vocal.f1_spread_hz),
        f2_spread_hz: f64::from(bo.vocal.f2_spread_hz),

        pods_cracked: (hab
            .terrarium
            .ecology_metrics()
            .agent_pods_cracked
            .get(hab.focal_slot)
            .copied()
            .unwrap_or(0)
            + hab.terrarium.ecology_metrics().pods_cracked_by_stone) as f64,
        coop_grove_pods: hab
            .terrarium
            .ecology_metrics()
            .coop_grove_pods_cracked
            .saturating_sub(hab.prev_coop_grove_pods) as f64,
        occluded_vocal_approaches: hab
            .terrarium
            .ecology_metrics()
            .occluded_vocal_approaches
            .saturating_sub(hab.prev_occluded_approaches) as f64,
        occluded_closing_rate: if hab.occluded_pair_ticks > 0 {
            hab.occluded_closing_sum / (hab.occluded_pair_ticks as f64)
        } else {
            0.0
        },
        referential_coord_score: (if hab.occluded_pair_ticks > 0 {
            hab.occluded_closing_sum / (hab.occluded_pair_ticks as f64)
        } else {
            0.0
        }) + 0.015
            * (hab
                .terrarium
                .ecology_metrics()
                .coop_grove_pods_cracked
                .saturating_sub(hab.prev_coop_grove_pods) as f64),
        role_swap_separation: hab.role_swap_cv_separation(),
        coop_boulder_ticks: hab.terrarium.ecology_metrics().cooperative_boulder_ticks as f64,
        huddling_ticks: hab.terrarium.ecology_metrics().huddling_ticks as f64,
        shared_slab_ticks: hab.shared_slab_ticks as f64,
        elder_cells: hab
            .elder
            .as_ref()
            .map_or(0.0, |(eb, _, _, _, _)| eb.snapshot().cell_count as f64),
        elder_atp: hab
            .elder
            .as_ref()
            .map_or(0.0, |(eb, _, _, _, _)| f64::from(eb.snapshot().atp)),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SeedDevelopmentRecord {
    stages_nat: [StageSnapshot; 4],
    mature_frz: StageSnapshot,
    mature_shuf: StageSnapshot,
    mature_solo: StageSnapshot,
    mature_muted: StageSnapshot,
    mature_barren: StageSnapshot,
    probe_intact_net: f64,
    probe_lesioned_net: f64,
}

fn sync_dev_vocal_ontogeny(org: &Alpha11Organism, hab: &mut DevOpenHabitat) {
    let s = org.snapshot();
    let m_items = org.motor_maturation_by_channel();
    let voc_m = group_mastery(&m_items, 12, 17) as f32;
    hab.focal_body
        .advance_vocal_ontogeny(s.tick, s.mitosis_births, voc_m);
}

fn run_single_dev_seed(seed: u64, total_ticks: usize) -> Result<SeedDevelopmentRecord, CampaignError> {
    let world_seed = seed ^ 0xA110_09E7_2026_0011;
    let total_ticks = total_ticks.max(160);
    let stage_ends = [
        (total_ticks / 16).max(1),
        (total_ticks / 4).max(2),
        (total_ticks * 5 / 8).max(3),
        total_ticks,
    ];

    // 1. Natural growing Alpha11Organism in OpenSocialHabitat with autonomous Elder
    let mut nat_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut nat_hab = DevOpenHabitat::new(world_seed, true);
    let mut stages_nat = [StageSnapshot::default(); 4];
    let mut stage_idx = 0usize;

    for t in 1..=total_ticks {
        let p = nat_org.step(nat_hab.current_view())?;
        nat_hab.apply_pulse(&p);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&nat_org, &mut nat_hab);
        }
        if stage_idx < 4 && t == stage_ends[stage_idx] {
            stages_nat[stage_idx] = capture_stage(&nat_org, &nat_hab);
            stage_idx += 1;
            if stage_idx < 4 {
                nat_hab.reset_stage_window();
            }
        }
    }

    // Acute 120-tick (2.4s) post-maturation flagless connectome+weight lesion probe from exposed ridge (0.0, 6.5) before online regrowth — 0.0 passive food & 0.0 passive warmth (zero mode flags)
    let mut intact_org = nat_org.clone();
    let mut intact_hab = DevOpenHabitat::new(world_seed, false);
    intact_hab.terrarium.respawn_at_position(intact_hab.focal_slot, (0.0, 6.5));
    intact_hab.refresh_views();
    intact_hab.reset_stage_window();
    sync_dev_vocal_ontogeny(&intact_org, &mut intact_hab);

    let mut lesion_org = nat_org.clone();
    lesion_org.zero_learned_conductances();
    let mut lesion_hab = DevOpenHabitat::new(world_seed, false);
    lesion_hab.terrarium.respawn_at_position(lesion_hab.focal_slot, (0.0, 6.5));
    lesion_hab.refresh_views();
    lesion_hab.reset_stage_window();
    sync_dev_vocal_ontogeny(&lesion_org, &mut lesion_hab);

    for t_probe in 1..=120_usize {
        let pi = intact_org.step(intact_hab.current_view())?;
        intact_hab.apply_pulse(&pi);
        let pl = lesion_org.step(lesion_hab.current_view())?;
        lesion_hab.apply_pulse(&pl);
        if (t_probe & 15) == 0 {
            sync_dev_vocal_ontogeny(&intact_org, &mut intact_hab);
            sync_dev_vocal_ontogeny(&lesion_org, &mut lesion_hab);
        }
    }
    let probe_intact_net = intact_hab.window_net_sum / (intact_hab.window_ticks.max(1) as f64);
    let probe_lesioned_net = lesion_hab.window_net_sum / (lesion_hab.window_ticks.max(1) as f64);

    // 2. FrozenPlasticity twin (same birth seed & habitat, zero growth/plasticity)
    let mut frz_org = Alpha11Organism::new(seed, OrganismConfig::default());
    frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut frz_hab = DevOpenHabitat::new(world_seed, true);
    for t in 1..=total_ticks {
        let p = frz_org.step(frz_hab.current_view())?;
        frz_hab.apply_pulse(&p);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&frz_org, &mut frz_hab);
        }
        if stage_ends[..3].contains(&t) {
            frz_hab.reset_stage_window();
        }
    }
    let mature_frz = capture_stage(&frz_org, &frz_hab);

    // 3. ActionShuffled twin (64-slot motor shuffle ring)
    let mut shuf_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut shuf_hab = DevOpenHabitat::new(world_seed, true);
    let mut shuf_rng = DeterministicRng::new(seed ^ 0x540F_F1ED_09E7_2026);
    let mut pulse_pool: Vec<FieldPulse> = (0..64)
        .map(|_| FieldPulse {
            actuator_out: (0..OPEN_ACTUATORS)
                .map(|_| (shuf_rng.next_f64() as f32) * 2.0 - 1.0)
                .collect(),
        })
        .collect();
    for t in 1..=total_ticks {
        let emitted = shuf_org.step(shuf_hab.current_view())?;
        let swap_idx = shuf_rng.next_usize(pulse_pool.len());
        let delivered = std::mem::replace(&mut pulse_pool[swap_idx], emitted);
        shuf_hab.apply_pulse(&delivered);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&shuf_org, &mut shuf_hab);
        }
        if stage_ends[..3].contains(&t) {
            shuf_hab.reset_stage_window();
        }
    }
    let mature_shuf = capture_stage(&shuf_org, &shuf_hab);

    // 4. SoloIsolated twin (same phase-matched Earth ecology + hands + vestibular + vocal body, no Elder)
    let mut solo_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut solo_hab = DevOpenHabitat::new(world_seed, false);
    for t in 1..=total_ticks {
        let p = solo_org.step(solo_hab.current_view())?;
        solo_hab.apply_pulse(&p);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&solo_org, &mut solo_hab);
        }
        if stage_ends[..3].contains(&t) {
            solo_hab.reset_stage_window();
        }
    }
    let mature_solo = capture_stage(&solo_org, &solo_hab);

    // 5. VocalMuted twin (same Earth canyon + Elder companion + hands + retina, peer 16 kHz voice muted)
    let mut muted_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut muted_hab = DevOpenHabitat::new_vocal_muted(world_seed);
    for t in 1..=total_ticks {
        let p = muted_org.step(muted_hab.current_view())?;
        muted_hab.apply_pulse(&p);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&muted_org, &mut muted_hab);
        }
        if stage_ends[..3].contains(&t) {
            muted_hab.reset_stage_window();
        }
    }
    let mature_muted = capture_stage(&muted_org, &muted_hab);

    // 6. BarrenArena twin (same birth seed & Elder, without Earth ecology's 16 kHz water/impacts/pods/seasonal gradient)
    let mut bar_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut bar_hab = DevOpenHabitat::new_barren(world_seed, true);
    for t in 1..=total_ticks {
        let p = bar_org.step(bar_hab.current_view())?;
        bar_hab.apply_pulse(&p);
        if (t & 15) == 0 {
            sync_dev_vocal_ontogeny(&bar_org, &mut bar_hab);
        }
        if stage_ends[..3].contains(&t) {
            bar_hab.reset_stage_window();
        }
    }
    let mature_barren = capture_stage(&bar_org, &bar_hab);

    Ok(SeedDevelopmentRecord {
        stages_nat,
        mature_frz,
        mature_shuf,
        mature_solo,
        mature_muted,
        mature_barren,
        probe_intact_net,
        probe_lesioned_net,
    })
}

fn mean_stage(records: &[SeedDevelopmentRecord], pick: fn(&SeedDevelopmentRecord) -> StageSnapshot) -> StageSnapshot {
    let n = records.len().max(1) as f64;
    let mut acc = StageSnapshot::default();
    for r in records {
        let s = pick(r);
        acc.tick = s.tick;
        acc.cells += s.cells;
        acc.tubes += s.tubes;
        acc.stiff_tubes += s.stiff_tubes;
        acc.echo_tubes += s.echo_tubes;
        acc.mean_tau += s.mean_tau;
        acc.max_tau += s.max_tau;
        acc.tau_octaves += s.tau_octaves;
        acc.mitosis_births += s.mitosis_births;
        acc.contradiction_splits += s.contradiction_splits;
        acc.sleep_ticks += s.sleep_ticks;
        acc.retina_pores += s.retina_pores;
        acc.echo_pores += s.echo_pores;
        acc.hand_pores += s.hand_pores;
        acc.vestibular_pores += s.vestibular_pores;
        acc.cochlea_pores += s.cochlea_pores;
        acc.distractor_pores += s.distractor_pores;
        acc.distractor_share += s.distractor_share;
        acc.motor_mastery += s.motor_mastery;
        acc.motor_jerk += s.motor_jerk;
        acc.loco_mastery += s.loco_mastery;
        acc.hand_mastery += s.hand_mastery;
        acc.vest_mastery += s.vest_mastery;
        acc.vocal_mastery += s.vocal_mastery;
        acc.atp += s.atp;
        acc.net_per_tick += s.net_per_tick;
        acc.drought_net_per_tick += s.drought_net_per_tick;
        acc.flood_net_per_tick += s.flood_net_per_tick;
        acc.markov_error_recovery += s.markov_error_recovery;
        acc.agency += s.agency;
        acc.pred_err += s.pred_err;
        acc.hand_strokes += s.hand_strokes;
        acc.canvas_coverage += s.canvas_coverage;
        acc.canvas_morans_i += s.canvas_morans_i;
        acc.balance_mean_upright += s.balance_mean_upright;
        acc.balance_max_streak += s.balance_max_streak;
        acc.voiced_rate += s.voiced_rate;
        acc.tract_length_cm += s.tract_length_cm;
        acc.syllable_onsets += s.syllable_onsets;
        acc.syllable_transitions += s.syllable_transitions;
        acc.plosive_bursts += s.plosive_bursts;
        acc.alternating_responses += s.alternating_responses;
        acc.shelter_echo_ticks += s.shelter_echo_ticks;
        acc.mean_f0_hz += s.mean_f0_hz;
        acc.mean_f1_hz += s.mean_f1_hz;
        acc.mean_f2_hz += s.mean_f2_hz;
        acc.f1_spread_hz += s.f1_spread_hz;
        acc.f2_spread_hz += s.f2_spread_hz;
        acc.pods_cracked += s.pods_cracked;
        acc.coop_grove_pods += s.coop_grove_pods;
        acc.occluded_vocal_approaches += s.occluded_vocal_approaches;
        acc.occluded_closing_rate += s.occluded_closing_rate;
        acc.referential_coord_score += s.referential_coord_score;
        acc.role_swap_separation += s.role_swap_separation;
        acc.coop_boulder_ticks += s.coop_boulder_ticks;
        acc.huddling_ticks += s.huddling_ticks;
        acc.shared_slab_ticks += s.shared_slab_ticks;
        acc.elder_cells += s.elder_cells;
        acc.elder_atp += s.elder_atp;
    }
    acc.cells /= n;
    acc.tubes /= n;
    acc.stiff_tubes /= n;
    acc.echo_tubes /= n;
    acc.mean_tau /= n;
    acc.max_tau /= n;
    acc.tau_octaves /= n;
    acc.mitosis_births /= n;
    acc.contradiction_splits /= n;
    acc.sleep_ticks /= n;
    acc.retina_pores /= n;
    acc.echo_pores /= n;
    acc.hand_pores /= n;
    acc.vestibular_pores /= n;
    acc.cochlea_pores /= n;
    acc.distractor_pores /= n;
    acc.distractor_share /= n;
    acc.motor_mastery /= n;
    acc.motor_jerk /= n;
    acc.loco_mastery /= n;
    acc.hand_mastery /= n;
    acc.vest_mastery /= n;
    acc.vocal_mastery /= n;
    acc.atp /= n;
    acc.net_per_tick /= n;
    acc.drought_net_per_tick /= n;
    acc.flood_net_per_tick /= n;
    acc.markov_error_recovery /= n;
    acc.agency /= n;
    acc.pred_err /= n;
    acc.hand_strokes /= n;
    acc.canvas_coverage /= n;
    acc.canvas_morans_i /= n;
    acc.balance_mean_upright /= n;
    acc.balance_max_streak /= n;
    acc.voiced_rate /= n;
    acc.tract_length_cm /= n;
    acc.syllable_onsets /= n;
    acc.syllable_transitions /= n;
    acc.plosive_bursts /= n;
    acc.alternating_responses /= n;
    acc.shelter_echo_ticks /= n;
    acc.mean_f0_hz /= n;
    acc.mean_f1_hz /= n;
    acc.mean_f2_hz /= n;
    acc.f1_spread_hz /= n;
    acc.f2_spread_hz /= n;
    acc.pods_cracked /= n;
    acc.coop_grove_pods /= n;
    acc.occluded_vocal_approaches /= n;
    acc.occluded_closing_rate /= n;
    acc.referential_coord_score /= n;
    acc.role_swap_separation /= n;
    acc.coop_boulder_ticks /= n;
    acc.huddling_ticks /= n;
    acc.shared_slab_ticks /= n;
    acc.elder_cells /= n;
    acc.elder_atp /= n;
    acc
}

fn format_stage_json(label: &str, s: &StageSnapshot) -> String {
    let mut out = String::with_capacity(1280);
    let _ = write!(
        out,
        "{{\"stage\":\"{label}\",\"tick\":{},\"cells\":{:.2},\"tubes\":{:.2},\"stiff_tubes\":{:.2},\"echo_tubes\":{:.2},\"mean_tau\":{:.3},\"max_tau\":{:.3},\"tau_octaves\":{:.2},\"mitosis_births\":{:.2},\"contradiction_splits\":{:.2},\"sleep_ticks\":{:.2},\"pores\":{{\"retina\":{:.2},\"echo\":{:.2},\"hands\":{:.2},\"vestibular\":{:.2},\"cochlea\":{:.2},\"distractors\":{:.2},\"distractor_share\":{:.4}}},\"muscles\":{{\"mastery\":{:.4},\"jerk\":{:.4},\"loco_mastery\":{:.4},\"hand_mastery\":{:.4},\"vest_mastery\":{:.4},\"vocal_mastery\":{:.4}}},\"homeostasis\":{{\"atp\":{:.3},\"net_per_tick\":{:.5},\"drought_net_per_tick\":{:.5},\"flood_net_per_tick\":{:.5},\"markov_error_recovery\":{:.5},\"agency\":{:.5},\"pred_err\":{:.4}}},\"ecology\":{{\"pods_cracked\":{:.2},\"coop_grove_pods\":{:.2},\"occluded_vocal_approaches\":{:.1},\"occluded_closing_rate\":{:.5},\"referential_coord_score\":{:.5},\"role_swap_separation\":{:.5},\"coop_boulder_ticks\":{:.1},\"huddling_ticks\":{:.1},\"shared_slab_ticks\":{:.1},\"shelter_echo_ticks\":{:.1},\"elder_cells\":{:.2},\"elder_atp\":{:.3}}},\"embodied\":{{\"hand_strokes\":{:.1},\"canvas_coverage\":{:.4},\"canvas_morans_i\":{:.4},\"balance_mean_upright\":{:.2},\"balance_max_streak\":{:.1},\"tract_length_cm\":{:.2},\"voiced_rate\":{:.4},\"syllable_onsets\":{:.1},\"syllable_transitions\":{:.1},\"plosive_bursts\":{:.1},\"alternating_responses\":{:.1},\"mean_f0_hz\":{:.1},\"mean_f1_hz\":{:.1},\"mean_f2_hz\":{:.1},\"f1_spread_hz\":{:.1},\"f2_spread_hz\":{:.1}}}}}",
        s.tick,
        s.cells,
        s.tubes,
        s.stiff_tubes,
        s.echo_tubes,
        s.mean_tau,
        s.max_tau,
        s.tau_octaves,
        s.mitosis_births,
        s.contradiction_splits,
        s.sleep_ticks,
        s.retina_pores,
        s.echo_pores,
        s.hand_pores,
        s.vestibular_pores,
        s.cochlea_pores,
        s.distractor_pores,
        s.distractor_share,
        s.motor_mastery,
        s.motor_jerk,
        s.loco_mastery,
        s.hand_mastery,
        s.vest_mastery,
        s.vocal_mastery,
        s.atp,
        s.net_per_tick,
        s.drought_net_per_tick,
        s.flood_net_per_tick,
        s.markov_error_recovery,
        s.agency,
        s.pred_err,
        s.pods_cracked,
        s.coop_grove_pods,
        s.occluded_vocal_approaches,
        s.occluded_closing_rate,
        s.referential_coord_score,
        s.role_swap_separation,
        s.coop_boulder_ticks,
        s.huddling_ticks,
        s.shared_slab_ticks,
        s.shelter_echo_ticks,
        s.elder_cells,
        s.elder_atp,
        s.hand_strokes,
        s.canvas_coverage,
        s.canvas_morans_i,
        s.balance_mean_upright,
        s.balance_max_streak,
        s.tract_length_cm,
        s.voiced_rate,
        s.syllable_onsets,
        s.syllable_transitions,
        s.plosive_bursts,
        s.alternating_responses,
        s.mean_f0_hz,
        s.mean_f1_hz,
        s.mean_f2_hz,
        s.f1_spread_hz,
        s.f2_spread_hz,
    );
    out
}

fn run_observe_development() -> Result<(), CampaignError> {
    let t0 = Instant::now();
    let mut n_seeds = 12_usize;
    let mut total_ticks = 8_000_usize;
    let mut args = std::env::args().skip(2);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => {
                if let Some(v) = args.next()
                    && let Ok(parsed) = v.parse::<usize>()
                {
                    n_seeds = parsed.max(1);
                }
            }
            "--ticks" => {
                if let Some(v) = args.next()
                    && let Ok(parsed) = v.parse::<usize>()
                {
                    total_ticks = parsed.max(160);
                }
            }
            _ => {}
        }
    }
    let seeds: Vec<u64> = (0..(n_seeds as u64)).collect();
    let n = seeds.len();
    println!(
        "== Longitudinal Whole-Body Developmental Trajectory | Organism=Alpha11Organism | Habitat=OpenSocialHabitat (276ch, 17 muscles, ticks={total_ticks}) | n={n} seeds =="
    );

    let workers = std::thread::available_parallelism()
        .map(|p| p.get().min(n))
        .unwrap_or(4);
    let mut records = vec![SeedDevelopmentRecord::default(); n];
    std::thread::scope(|scope| -> Result<(), CampaignError> {
        let chunk_len = n.div_ceil(workers);
        let mut handles = Vec::new();
        for (c_idx, chunk_seeds) in seeds.chunks(chunk_len).enumerate() {
            let start = c_idx * chunk_len;
            handles.push(scope.spawn(
                move || -> Result<(usize, Vec<SeedDevelopmentRecord>), CampaignError> {
                    let mut out = Vec::with_capacity(chunk_seeds.len());
                    for &sd in chunk_seeds {
                        out.push(run_single_dev_seed(sd, total_ticks)?);
                    }
                    Ok((start, out))
                },
            ));
        }
        for h in handles {
            let (start, chunk_out) = h
                .join()
                .map_err(|_| CampaignError::Io(std::io::Error::other("dev worker panicked")))??;
            for (offset, rec) in chunk_out.into_iter().enumerate() {
                records[start + offset] = rec;
            }
        }
        Ok(())
    })?;

    let st_newborn = mean_stage(&records, |r| r.stages_nat[0]);
    let st_infant = mean_stage(&records, |r| r.stages_nat[1]);
    let st_juvenile = mean_stage(&records, |r| r.stages_nat[2]);
    let st_mature = mean_stage(&records, |r| r.stages_nat[3]);
    let st_frz = mean_stage(&records, |r| r.mature_frz);
    let st_shuf = mean_stage(&records, |r| r.mature_shuf);
    let st_solo = mean_stage(&records, |r| r.mature_solo);
    let st_muted = mean_stage(&records, |r| r.mature_muted);
    let st_bar = mean_stage(&records, |r| r.mature_barren);

    println!("\n--- 1. Brain Morphogenesis & Consolidation Across Life (12-seed means) ---");
    for (name, s) in [
        ("Newborn  (0..10s)", &st_newborn),
        ("Infant  (10..40s)", &st_infant),
        ("Juvenile(40..100s)", &st_juvenile),
        ("Mature (100..160s)", &st_mature),
        ("BarrenTwin(@160s)", &st_bar),
        ("FrozenTwin(@160s)", &st_frz),
    ] {
        println!(
            "  {name}: cells={:5.1} tubes={:6.1} stiff_LTM={:5.1} echo_seq={:5.1} mean_tau={:5.2} max_tau={:6.1} octaves={:3.1} mitosis={:4.1} splits={:4.1} sleep={:5.1} | elder_cells={:5.1} elder_ATP={:4.2}",
            s.cells,
            s.tubes,
            s.stiff_tubes,
            s.echo_tubes,
            s.mean_tau,
            s.max_tau,
            s.tau_octaves,
            s.mitosis_births,
            s.contradiction_splits,
            s.sleep_ticks,
            s.elder_cells,
            s.elder_atp,
        );
    }

    println!("\n--- 2. Self-Wired Sensory Organogenesis (Pore Counts by Modality vs Distractors) ---");
    for (name, s) in [
        ("Newborn  (0..10s)", &st_newborn),
        ("Infant  (10..40s)", &st_infant),
        ("Juvenile(40..100s)", &st_juvenile),
        ("Mature (100..160s)", &st_mature),
        ("BarrenTwin(@160s)", &st_bar),
    ] {
        println!(
            "  {name}: retina={:4.1} echo={:4.1} hands={:4.1} vestibular={:4.1} cochlea={:4.1} | distractors={:4.1} ({:4.1}%)",
            s.retina_pores,
            s.echo_pores,
            s.hand_pores,
            s.vestibular_pores,
            s.cochlea_pores,
            s.distractor_pores,
            s.distractor_share * 100.0,
        );
    }

    println!("\n--- 3. Neuromuscular Refinement & Whole-Body Motor Mastery ---");
    for (name, s) in [
        ("Newborn  (0..10s)", &st_newborn),
        ("Infant  (10..40s)", &st_infant),
        ("Juvenile(40..100s)", &st_juvenile),
        ("Mature (100..160s)", &st_mature),
        ("BarrenTwin(@160s)", &st_bar),
        ("FrozenTwin(@160s)", &st_frz),
    ] {
        println!(
            "  {name}: mastery={:.4} (loco={:.3} hand={:.3} vest={:.3} voice={:.3}) | kinematic_jerk={:.4} | agency={:.5}",
            s.motor_mastery,
            s.loco_mastery,
            s.hand_mastery,
            s.vest_mastery,
            s.vocal_mastery,
            s.motor_jerk,
            s.agency,
        );
    }

    println!("\n--- 4. Earth Ecology, Occluded Ridges, Cooperative Grove Pods & 16 kHz Voice ---");
    for (name, s) in [
        ("Newborn  (0..10s)", &st_newborn),
        ("Infant  (10..40s)", &st_infant),
        ("Juvenile(40..100s)", &st_juvenile),
        ("Mature (100..160s)", &st_mature),
        ("VocalMuted(@160s)", &st_muted),
        ("BarrenTwin(@160s)", &st_bar),
        ("SoloOrphan(@160s)", &st_solo),
        ("FrozenTwin(@160s)", &st_frz),
    ] {
        println!(
            "  {name}: ATP={:5.2} net/t={:+.4} (dr={:+.4} fl={:+.4}) p_rec={:.4} pods={:4.1} grove_coop={:4.2} occ_cls={:+.4} role_sep={:+.5} huddle={:5.1} | upright={:5.1} | tract={:5.2}cm voiced={:.3} syl_tr={:5.1} plos={:4.1} alt={:4.1}",
            s.atp,
            s.net_per_tick,
            s.drought_net_per_tick,
            s.flood_net_per_tick,
            s.markov_error_recovery,
            s.pods_cracked,
            s.coop_grove_pods,
            s.occluded_closing_rate,
            s.role_swap_separation,
            s.huddling_ticks,
            s.balance_mean_upright,
            s.tract_length_cm,
            s.voiced_rate,
            s.syllable_transitions,
            s.plosive_bursts,
            s.alternating_responses,
        );
    }

    // Paired causal, developmental & Earth-ecology comparisons (with Holm-Bonferroni p-values)
    let vec_of =
        |f: fn(&SeedDevelopmentRecord) -> f64| -> Vec<f64> { records.iter().map(f).collect() };
    let comparisons: [(&str, Vec<f64>, Vec<f64>); 10] = [
        (
            "Homeostatic Survival (Mature Natural vs Frozen)",
            vec_of(|r| r.stages_nat[3].net_per_tick),
            vec_of(|r| r.mature_frz.net_per_tick),
        ),
        (
            "Causal Motor Control (Mature Natural vs ActionShuffled)",
            vec_of(|r| r.stages_nat[3].net_per_tick),
            vec_of(|r| r.mature_shuf.net_per_tick),
        ),
        (
            "Post-Maturation Foraging Necessity (Intact vs Lesioned)",
            vec_of(|r| r.probe_intact_net),
            vec_of(|r| r.probe_lesioned_net),
        ),
        (
            "Counterfactual Agency Purity (Natural vs ActionShuffled)",
            vec_of(|r| r.stages_nat[3].agency),
            vec_of(|r| r.mature_shuf.agency),
        ),
        (
            "Neuromuscular Mastery Growth (Mature vs Newborn)",
            vec_of(|r| r.stages_nat[3].motor_mastery),
            vec_of(|r| r.stages_nat[0].motor_mastery),
        ),
        (
            "Kinematic Jerk Reduction (Newborn Jerk vs Mature Jerk)",
            vec_of(|r| r.stages_nat[0].motor_jerk),
            vec_of(|r| r.stages_nat[3].motor_jerk),
        ),
        (
            "Earth Ecology Vitality Scaffolding (Earth vs Barren)",
            vec_of(|r| r.stages_nat[3].net_per_tick),
            vec_of(|r| r.mature_barren.net_per_tick),
        ),
        (
            "Markov Error-Recovery p(bad->good) (Natural vs Frozen)",
            vec_of(|r| r.stages_nat[3].markov_error_recovery),
            vec_of(|r| r.mature_frz.markov_error_recovery),
        ),
        (
            "Occluded Acoustic Coordination (Natural vs VocalMuted)",
            vec_of(|r| r.stages_nat[3].referential_coord_score),
            vec_of(|r| r.mature_muted.referential_coord_score),
        ),
        (
            "Trajectory/Role-Swap Separation (Natural vs Frozen)",
            vec_of(|r| r.stages_nat[3].role_swap_separation),
            vec_of(|r| r.mature_frz.role_swap_separation),
        ),
    ];

    let gates: Vec<_> = comparisons
        .iter()
        .map(|(_, a, b)| paired_gate(a, b))
        .collect();
    let raw_p: Vec<f64> = gates.iter().map(|g| g.p_two_sided).collect();
    let holm_p = holm_adjust(&raw_p);

    println!("\n--- 5. Paired Longitudinal, Causal & Ecological Comparisons (n={n}, Holm-adjusted) ---");
    for ((label, _, _), (g, &ph)) in comparisons.iter().zip(gates.iter().zip(holm_p.iter())) {
        println!(
            "  {:<56} A={:+8.4} B={:+8.4} diff={:+8.4} t={:+6.2} p_holm={:.5} ({}/{})",
            label, g.mean_a, g.mean_b, g.mean_diff, g.t, ph, g.wins, g.n
        );
    }

    let workspace_root = resolve_workspace_root();
    let out_dir = workspace_root.join("observations");
    fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join("open_world_development.json");

    let mut json = String::with_capacity(5120);
    let _ = writeln!(
        json,
        "{{\n\
         \x20 \"seeds\": {n},\n\
         \x20 \"elapsed_seconds\": {:.3},\n\
         \x20 \"trajectory\": [\n\
         \x20   {},\n\
         \x20   {},\n\
         \x20   {},\n\
         \x20   {}\n\
         \x20 ],\n\
         \x20 \"controls_at_8000\": {{\n\
         \x20   \"frozen_plasticity\": {},\n\
         \x20   \"action_shuffled\": {},\n\
         \x20   \"solo_isolated\": {},\n\
         \x20   \"vocal_muted\": {},\n\
         \x20   \"barren_arena\": {},\n\
         \x20   \"intact_probe_net\": {:.5},\n\
         \x20   \"lesioned_probe_net\": {:.5}\n\
         \x20 }}\n\
         }}",
        t0.elapsed().as_secs_f64(),
        format_stage_json("newborn_0_500", &st_newborn),
        format_stage_json("infant_500_2000", &st_infant),
        format_stage_json("juvenile_2000_5000", &st_juvenile),
        format_stage_json("mature_5000_8000", &st_mature),
        format_stage_json("frozen_5000_8000", &st_frz),
        format_stage_json("shuffled_5000_8000", &st_shuf),
        format_stage_json("solo_5000_8000", &st_solo),
        format_stage_json("vocal_muted_5000_8000", &st_muted),
        format_stage_json("barren_5000_8000", &st_bar),
        gates[2].mean_a,
        gates[2].mean_b,
    );
    fs::write(&out_path, json.as_bytes())?;
    println!(
        "\n[alpha11-observe] Saved longitudinal report -> {} (elapsed {:.2}s)",
        out_path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    let arg1 = std::env::args().nth(1);
    if matches!(arg1.as_deref(), Some("observe-development" | "vocal-dev30")) {
        return match run_observe_development() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("[alpha11-observe] ERROR: {err}");
                ExitCode::FAILURE
            }
        };
    }

    match run_all() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[alpha11-observe] ERROR: {err}");
            ExitCode::FAILURE
        }
    }
}
