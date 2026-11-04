//! Locked long-horizon energy-homeostasis judge for Alpha 11 (`alpha11-longlife`).
//!
//! Pre-registered on 2026-11-04 (ledger entry 53) BEFORE any tuning of the starvation fix.
//! Do not edit gate definitions, seeds, tick counts, or windows after the lock commit.
//!
//! Habitat: the same whole-body `OpenSocialHabitat` the live organism inhabits (Earth ecology,
//! autonomous Elder in slot 0 warmed up for 640 ticks, infant body in slot 1).
//!
//! Arms per seed (identical birth seed and world seed):
//! - `Natural`: full plasticity and morphogenesis.
//! - `Frozen`: `ControlMode::FrozenPlasticity` from birth.
//! - `Shuffled`: emitted pulses pass through a 64-slot random swap ring before reaching the world.
//!
//! Late window = final 20% of ticks. ATP is sampled every 10 ticks.
//!
//! Gates (pass = all five):
//! - `L1` late mean ATP: Natural > Frozen (paired t, Holm over L1..L3, `p_holm < 0.05`).
//! - `L2` late net energy per tick (`influx - stress`): Natural > Frozen (paired, Holm).
//! - `L3` late net energy per tick: Natural > Shuffled (paired, Holm).
//! - `L4` absolute: mean Natural late fraction of samples with `ATP < 0.05` is `<= 0.10`.
//! - `L5` absolute: mean Natural late ATP `>= 1.5` (exit from the starvation regime of `body.rs`).
//!
//! Diagnostics (not gates): an externally reconstructed ATP ledger for the Natural arm
//! (influx, repair `0.05 * stress`, basal `0.0008`, structure rent from cell/tube/pore counts,
//! motor `0.0018 * E`, mitosis `0.40` per birth) plus a coarse time series every 5% of life.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, WorldMedium};
use a11_observe::gates::{holm_adjust, paired_gate};
use a11_observe::stats::DeterministicRng;
use a11_portals::{EMBODIED_ORGAN_ACTUATORS, EMBODIED_ORGAN_CHANNELS, EmbodiedBodyOrgans};
use a11_worlds::{
    EARTH_AUDIO_FRAME_SAMPLES, MultiGenTerrarium, TERRARIUM_ACTUATORS, TERRARIUM_CHANNELS,
};

const CH: usize = TERRARIUM_CHANNELS + EMBODIED_ORGAN_CHANNELS;
const ACT: usize = TERRARIUM_ACTUATORS + EMBODIED_ORGAN_ACTUATORS;
const ELDER_WARMUP_TICKS: usize = 640;
const WORLD_SEED_SALT: u64 = 0x10A6_11FE_2026_1104;
const ATP_SAMPLE_EVERY: usize = 10;
const STARVE_ATP: f32 = 0.05;
const SERIES_POINTS: usize = 20;

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

#[inline]
fn clamp_nonneg(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 { v } else { 0.0 }
}

/// Mirror of the live `OpenSocialHabitat` (`a11-live`), without PCM recording.
struct OpenHabitat {
    terrarium: MultiGenTerrarium,
    elder_brain: Alpha11Organism,
    elder_body: EmbodiedBodyOrgans,
    elder_ambient: [f32; CH],
    elder_influx: f32,
    elder_stress: f32,
    focal_body: EmbodiedBodyOrgans,
    focal_ambient: [f32; CH],
    focal_influx: f32,
    focal_stress: f32,
}

impl OpenHabitat {
    fn new(seed: u64) -> Self {
        let mut terrarium = MultiGenTerrarium::new_earth_ecology(seed, 1);
        let mut elder_brain =
            Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
        let mut elder_body = EmbodiedBodyOrgans::new_adult(seed ^ 0xE1DE_B0D1_2026_0002);
        let mut elder_ambient = [0.0_f32; CH];
        for _ in 0..ELDER_WARMUP_TICKS {
            let tv = terrarium.agent_view(0);
            for (i, slot) in elder_ambient[..TERRARIUM_CHANNELS].iter_mut().enumerate() {
                *slot = clamp_unit(tv.ambient_in.get(i).copied().unwrap_or(0.0));
            }
            elder_ambient[TERRARIUM_CHANNELS..].copy_from_slice(elder_body.sensory_channels());
            let view = FieldView {
                ambient_in: &elder_ambient,
                metabolic_influx: clamp_nonneg(tv.metabolic_influx),
                tissue_stress: clamp_nonneg(tv.tissue_stress + elder_body.last_organ_stress()),
                num_actuators: ACT,
            };
            let ep = elder_brain
                .step(view)
                .unwrap_or_else(|_| FieldPulse::silent(ACT));
            let mut eb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
            for (j, slot) in eb.iter_mut().enumerate() {
                *slot = clamp_unit(
                    ep.actuator_out
                        .get(TERRARIUM_ACTUATORS + j)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            let _ = elder_body.step_limbs_and_radiate_voice(&eb);
            let (es, eh) = elder_body.embodied_kinematics();
            terrarium.set_agent_embodied_kinematics(0, es, eh);
            terrarium.set_agent_embodied_voice_rms(0, elder_body.vocal().last_radiated_rms());
            let mut ew = FieldPulse {
                actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
            };
            for i in 0..TERRARIUM_ACTUATORS {
                ew.actuator_out
                    .push(clamp_unit(ep.actuator_out.get(i).copied().unwrap_or(0.0)));
            }
            terrarium.apply_population_pulses(&[ew]);
            elder_body.apply_terrain_sway(terrarium.agent_lateral_sway(0));
            let mut env_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
            let mut env_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
            let enc = terrarium.binaural_earth_waves_for_agent(0, &mut env_l, &mut env_r);
            elder_body.finish_binaural_ecology_transduction(None, Some((&env_l, &env_r)), enc);
        }
        elder_body.vocal_mut().reset_window_observations();
        terrarium.respawn_newborn_in_slot(1);
        let focal_body = EmbodiedBodyOrgans::new_infant(seed ^ 0xF0CA_1B0D_2026_0003);
        let mut hab = Self {
            terrarium,
            elder_brain,
            elder_body,
            elder_ambient,
            elder_influx: 0.0,
            elder_stress: 0.0,
            focal_body,
            focal_ambient: [0.0; CH],
            focal_influx: 0.0,
            focal_stress: 0.0,
        };
        hab.refresh_views();
        hab
    }

    fn refresh_views(&mut self) {
        let ev = self.terrarium.agent_view(0);
        for (i, slot) in self.elder_ambient[..TERRARIUM_CHANNELS]
            .iter_mut()
            .enumerate()
        {
            *slot = clamp_unit(ev.ambient_in.get(i).copied().unwrap_or(0.0));
        }
        self.elder_ambient[TERRARIUM_CHANNELS..].copy_from_slice(self.elder_body.sensory_channels());
        self.elder_influx = clamp_nonneg(ev.metabolic_influx);
        self.elder_stress = clamp_nonneg(ev.tissue_stress + self.elder_body.last_organ_stress());

        let fv = self.terrarium.agent_view(1);
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

    fn sync_vocal_ontogeny(&mut self, org: &Alpha11Organism) {
        let s = org.snapshot();
        let items = org.motor_maturation_by_channel();
        let mut sum = 0.0_f64;
        let mut n = 0_u32;
        for &(ch, mastery, _) in &items {
            if (12..17).contains(&ch) {
                sum += f64::from(mastery);
                n += 1;
            }
        }
        let voc_m = if n > 0 { (sum / f64::from(n)) as f32 } else { 0.0 };
        self.focal_body
            .advance_vocal_ontogeny(s.tick, s.mitosis_births, voc_m);
    }
}

impl WorldMedium for OpenHabitat {
    fn num_ambient_channels(&self) -> usize {
        CH
    }

    fn num_actuators(&self) -> usize {
        ACT
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.focal_ambient,
            metabolic_influx: self.focal_influx,
            tissue_stress: self.focal_stress,
            num_actuators: ACT,
        }
    }

    fn apply_pulse(&mut self, focal_pulse: &FieldPulse) {
        let elder_view = FieldView {
            ambient_in: &self.elder_ambient,
            metabolic_influx: self.elder_influx,
            tissue_stress: self.elder_stress,
            num_actuators: ACT,
        };
        let elder_pulse = self
            .elder_brain
            .step(elder_view)
            .unwrap_or_else(|_| FieldPulse::silent(ACT));

        let mut eb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        let mut fb = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        for j in 0..EMBODIED_ORGAN_ACTUATORS {
            eb[j] = clamp_unit(
                elder_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
            fb[j] = clamp_unit(
                focal_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
        }
        let elder_wave = self.elder_body.step_limbs_and_radiate_voice(&eb);
        let focal_wave = self.focal_body.step_limbs_and_radiate_voice(&fb);

        let (e_stab, e_hw) = self.elder_body.embodied_kinematics();
        let (f_stab, f_hw) = self.focal_body.embodied_kinematics();
        self.terrarium.set_agent_embodied_kinematics(0, e_stab, e_hw);
        self.terrarium.set_agent_embodied_kinematics(1, f_stab, f_hw);
        self.terrarium
            .set_agent_embodied_voice_rms(0, self.elder_body.vocal().last_radiated_rms());
        self.terrarium
            .set_agent_embodied_voice_rms(1, self.focal_body.vocal().last_radiated_rms());

        let mut ew = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        let mut fw = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        for i in 0..TERRARIUM_ACTUATORS {
            ew.actuator_out
                .push(clamp_unit(elder_pulse.actuator_out.get(i).copied().unwrap_or(0.0)));
            fw.actuator_out
                .push(clamp_unit(focal_pulse.actuator_out.get(i).copied().unwrap_or(0.0)));
        }
        self.terrarium.apply_population_pulses(&[ew, fw]);

        self.elder_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(0));
        self.focal_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(1));

        let positions = &self.terrarium.metrics().agent_positions;
        let e_pos = positions.first().copied().unwrap_or((0.0, 0.0));
        let f_pos = positions.get(1).copied().unwrap_or((0.0, 0.0));
        let rel_ef = (e_pos.0 - f_pos.0, e_pos.1 - f_pos.1);
        let rel_fe = (f_pos.0 - e_pos.0, f_pos.1 - e_pos.1);
        let pair_dist = rel_ef.0.hypot(rel_ef.1);
        if pair_dist <= 3.6 {
            let coupling = 0.26 * (1.0 - pair_dist / 3.6);
            EmbodiedBodyOrgans::couple_shared_hands(
                &mut self.elder_body,
                &mut self.focal_body,
                coupling,
            );
        }

        let peer_diff = self.terrarium.acoustic_transmittance(f_pos, e_pos);
        let mut elder_wave_diff = elder_wave;
        let mut focal_wave_diff = focal_wave;
        if peer_diff < 0.999 {
            for s in &mut elder_wave_diff {
                *s *= peer_diff;
            }
            for s in &mut focal_wave_diff {
                *s *= peer_diff;
            }
        }
        let mut f_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let mut f_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let f_enc = self
            .terrarium
            .binaural_earth_waves_for_agent(1, &mut f_l, &mut f_r);
        self.focal_body.finish_binaural_ecology_transduction(
            Some((&elder_wave_diff, rel_ef)),
            Some((&f_l, &f_r)),
            f_enc,
        );
        let mut e_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let mut e_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let e_enc = self
            .terrarium
            .binaural_earth_waves_for_agent(0, &mut e_l, &mut e_r);
        self.elder_body.finish_binaural_ecology_transduction(
            Some((&focal_wave_diff, rel_fe)),
            Some((&e_l, &e_r)),
            e_enc,
        );
        self.refresh_views();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Natural,
    Frozen,
    Shuffled,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Frozen => "frozen",
            Self::Shuffled => "shuffled",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SeriesPoint {
    tick: usize,
    atp: f64,
    cells: f64,
    tubes: f64,
    pores: f64,
    births: f64,
    pruned: f64,
    win_influx_per_tick: f64,
    win_stress_per_tick: f64,
    win_starve_frac: f64,
    elder_atp: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Ledger {
    influx: f64,
    repair: f64,
    basal: f64,
    rent: f64,
    motor: f64,
    mitosis: f64,
    ticks: f64,
}

impl Ledger {
    fn add(&mut self, o: &Self) {
        self.influx += o.influx;
        self.repair += o.repair;
        self.basal += o.basal;
        self.rent += o.rent;
        self.motor += o.motor;
        self.mitosis += o.mitosis;
        self.ticks += o.ticks;
    }

    fn json(&self) -> String {
        let t = self.ticks.max(1.0);
        format!(
            "{{\"ticks\":{:.0},\"influx_per_tick\":{:.6},\"repair_per_tick\":{:.6},\"basal_per_tick\":{:.6},\"rent_per_tick\":{:.6},\"motor_per_tick\":{:.6},\"mitosis_per_tick\":{:.6},\"balance_per_tick\":{:.6}}}",
            self.ticks,
            self.influx / t,
            self.repair / t,
            self.basal / t,
            self.rent / t,
            self.motor / t,
            self.mitosis / t,
            (self.influx - self.repair - self.basal - self.rent - self.motor - self.mitosis) / t,
        )
    }
}

#[derive(Clone, Debug, Default)]
struct ArmResult {
    late_atp: f64,
    late_net_per_tick: f64,
    late_starve_frac: f64,
    life_net_per_tick: f64,
    life_starve_frac: f64,
    final_cells: f64,
    final_births: f64,
    final_elder_atp: f64,
    ledger_life: Ledger,
    ledger_late: Ledger,
    series: Vec<SeriesPoint>,
}

fn run_arm(seed: u64, arm: Arm, ticks: usize) -> ArmResult {
    let world_seed = seed ^ WORLD_SEED_SALT;
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    if arm == Arm::Frozen {
        org.set_control_mode(ControlMode::FrozenPlasticity);
    }
    let mut hab = OpenHabitat::new(world_seed);
    let mut shuf_rng = DeterministicRng::new(seed ^ 0x540F_F1ED_1011_2026);
    let mut pool: Vec<FieldPulse> = (0..64)
        .map(|_| FieldPulse {
            actuator_out: (0..ACT)
                .map(|_| (shuf_rng.next_f64() as f32) * 2.0 - 1.0)
                .collect(),
        })
        .collect();

    let late_start = ticks - ticks / 5;
    let series_every = (ticks / SERIES_POINTS).max(1);
    let mut res = ArmResult::default();

    let mut late_atp_sum = 0.0_f64;
    let mut late_atp_n = 0_u64;
    let mut late_starve_n = 0_u64;
    let mut life_starve_n = 0_u64;
    let mut life_atp_n = 0_u64;
    let mut late_net = 0.0_f64;
    let mut life_net = 0.0_f64;

    let mut win_influx = 0.0_f64;
    let mut win_stress = 0.0_f64;
    let mut win_starve = 0_u64;
    let mut win_samples = 0_u64;

    let mut prev_births = 0_u64;
    let mut rent_now = 0.0_f64;
    let mut ledger_life = Ledger::default();
    let mut ledger_late = Ledger::default();

    for t in 1..=ticks {
        let emitted = org
            .step(hab.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(ACT));
        let mut e_sum = 0.0_f32;
        for &v in &emitted.actuator_out {
            e_sum += v * v;
        }
        let motor_e = if emitted.actuator_out.is_empty() {
            0.0
        } else {
            (e_sum / (emitted.actuator_out.len() as f32)).clamp(0.0, 4.0)
        };
        if arm == Arm::Shuffled {
            let idx = shuf_rng.next_usize(pool.len());
            let delivered = std::mem::replace(&mut pool[idx], emitted);
            hab.apply_pulse(&delivered);
        } else {
            hab.apply_pulse(&emitted);
        }
        if (t & 15) == 0 {
            hab.sync_vocal_ontogeny(&org);
        }

        let v = hab.current_view();
        let influx = f64::from(v.metabolic_influx);
        let stress = f64::from(v.tissue_stress);
        let net = influx - stress;
        life_net += net;
        win_influx += influx;
        win_stress += stress;
        let in_late = t > late_start;
        if in_late {
            late_net += net;
        }

        if t % ATP_SAMPLE_EVERY == 0 {
            let s = org.snapshot();
            let atp = s.atp;
            let starving = atp < STARVE_ATP;
            life_atp_n += 1;
            if starving {
                life_starve_n += 1;
                win_starve += 1;
            }
            win_samples += 1;
            if in_late {
                late_atp_sum += f64::from(atp);
                late_atp_n += 1;
                if starving {
                    late_starve_n += 1;
                }
            }
            rent_now = 0.000_010 * (s.cell_count as f64)
                + 0.000_000_4 * (s.tube_count as f64)
                + 0.000_004 * ((s.sensor_pore_count + s.motor_pore_count) as f64);
            let births = s.mitosis_births;
            let d_births = births.saturating_sub(prev_births);
            prev_births = births;
            let step_ledger = Ledger {
                influx: 0.0,
                repair: 0.0,
                basal: 0.0,
                rent: 0.0,
                motor: 0.0,
                mitosis: 0.40 * (d_births as f64),
                ticks: 0.0,
            };
            ledger_life.add(&step_ledger);
            if in_late {
                ledger_late.add(&step_ledger);
            }
        }

        // Next tick's burn is computed inside the organism from this tick's view; the external ledger
        // uses the same coefficients as `body.rs` and the most recent sampled structure counts.
        let tick_ledger = Ledger {
            influx,
            repair: 0.05 * stress,
            basal: 0.0008,
            rent: rent_now,
            motor: 0.0018 * f64::from(motor_e),
            mitosis: 0.0,
            ticks: 1.0,
        };
        ledger_life.add(&tick_ledger);
        if in_late {
            ledger_late.add(&tick_ledger);
        }

        if t % series_every == 0 {
            let s = org.snapshot();
            let es = hab.elder_brain.snapshot();
            let n = series_every as f64;
            res.series.push(SeriesPoint {
                tick: t,
                atp: f64::from(s.atp),
                cells: s.cell_count as f64,
                tubes: s.tube_count as f64,
                pores: (s.sensor_pore_count + s.motor_pore_count) as f64,
                births: s.mitosis_births as f64,
                pruned: s.atrophied_structures as f64,
                win_influx_per_tick: win_influx / n,
                win_stress_per_tick: win_stress / n,
                win_starve_frac: if win_samples > 0 {
                    (win_starve as f64) / (win_samples as f64)
                } else {
                    0.0
                },
                elder_atp: f64::from(es.atp),
            });
            win_influx = 0.0;
            win_stress = 0.0;
            win_starve = 0;
            win_samples = 0;
        }
    }

    let s = org.snapshot();
    let late_ticks = (ticks - late_start).max(1) as f64;
    res.late_atp = late_atp_sum / (late_atp_n.max(1) as f64);
    res.late_net_per_tick = late_net / late_ticks;
    res.late_starve_frac = (late_starve_n as f64) / (late_atp_n.max(1) as f64);
    res.life_net_per_tick = life_net / (ticks as f64);
    res.life_starve_frac = (life_starve_n as f64) / (life_atp_n.max(1) as f64);
    res.final_cells = s.cell_count as f64;
    res.final_births = s.mitosis_births as f64;
    res.final_elder_atp = f64::from(hab.elder_brain.snapshot().atp);
    res.ledger_life = ledger_life;
    res.ledger_late = ledger_late;
    res
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / (v.len() as f64)
    }
}

fn resolve_workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

struct Cli {
    seeds: usize,
    seed_offset: u64,
    ticks: usize,
    threads: usize,
    tag: String,
}

fn parse_cli() -> Result<Cli, String> {
    let mut cli = Cli {
        seeds: 10,
        seed_offset: 0,
        ticks: 100_000,
        threads: 12,
        tag: "run".to_string(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let val = args.get(i + 1).ok_or_else(|| format!("missing value after {key}"))?;
        match key {
            "--seeds" => cli.seeds = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--seed-offset" => cli.seed_offset = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--ticks" => cli.ticks = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--threads" => cli.threads = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--tag" => cli.tag.clone_from(val),
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    if cli.seed_offset >= 200 && cli.seed_offset < 212 {
        return Err("seeds 200..211 are the sealed heldout vault".to_string());
    }
    cli.seeds = cli.seeds.max(1);
    cli.ticks = cli.ticks.max(1_000);
    cli.threads = cli.threads.max(1);
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[alpha11-longlife] {e}");
            eprintln!(
                "Usage: alpha11-longlife [--seeds 10] [--seed-offset 0] [--ticks 100000] [--threads 12] [--tag name]"
            );
            return ExitCode::from(2);
        }
    };
    let t0 = Instant::now();
    let arms = [Arm::Natural, Arm::Frozen, Arm::Shuffled];
    let jobs: Vec<(usize, u64, Arm)> = (0..cli.seeds)
        .flat_map(|k| {
            let seed = cli.seed_offset + k as u64;
            arms.iter().map(move |&a| (k, seed, a))
        })
        .collect();
    let results: Mutex<Vec<Option<ArmResult>>> = Mutex::new(vec![None; jobs.len()]);
    let next = AtomicUsize::new(0);
    let ticks = cli.ticks;
    std::thread::scope(|scope| {
        for _ in 0..cli.threads.min(jobs.len()) {
            scope.spawn(|| {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= jobs.len() {
                        break;
                    }
                    let (_, seed, arm) = jobs[j];
                    let r = run_arm(seed, arm, ticks);
                    if let Ok(mut g) = results.lock() {
                        g[j] = Some(r);
                    }
                }
            });
        }
    });
    let results = match results.into_inner() {
        Ok(r) => r,
        Err(_) => {
            eprintln!("[alpha11-longlife] result lock poisoned");
            return ExitCode::from(1);
        }
    };

    let mut by_arm: [Vec<ArmResult>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (j, r) in results.into_iter().enumerate() {
        let Some(r) = r else {
            eprintln!("[alpha11-longlife] job {j} produced no result");
            return ExitCode::from(1);
        };
        let ai = arms.iter().position(|&a| a == jobs[j].2).unwrap_or(0);
        by_arm[ai].push(r);
    }
    let pick = |ai: usize, f: fn(&ArmResult) -> f64| -> Vec<f64> { by_arm[ai].iter().map(f).collect() };

    let nat_atp = pick(0, |r| r.late_atp);
    let frz_atp = pick(1, |r| r.late_atp);
    let nat_net = pick(0, |r| r.late_net_per_tick);
    let frz_net = pick(1, |r| r.late_net_per_tick);
    let shf_net = pick(2, |r| r.late_net_per_tick);
    let nat_starve = pick(0, |r| r.late_starve_frac);

    let g1 = paired_gate(&nat_atp, &frz_atp);
    let g2 = paired_gate(&nat_net, &frz_net);
    let g3 = paired_gate(&nat_net, &shf_net);
    let holm = holm_adjust(&[g1.p_two_sided, g2.p_two_sided, g3.p_two_sided]);
    let l1 = g1.mean_diff > 0.0 && holm[0] < 0.05;
    let l2 = g2.mean_diff > 0.0 && holm[1] < 0.05;
    let l3 = g3.mean_diff > 0.0 && holm[2] < 0.05;
    let mean_starve = mean(&nat_starve);
    let l4 = mean_starve <= 0.10;
    let mean_nat_atp = mean(&nat_atp);
    let l5 = mean_nat_atp >= 1.5;
    let passed = [l1, l2, l3, l4, l5].iter().filter(|&&p| p).count();

    let verdict = |p: bool| if p { "PASS" } else { "FAIL" };
    println!(
        "== alpha11-longlife | tag={} | seeds {}..{} | ticks={} | wall={:.1}s ==",
        cli.tag,
        cli.seed_offset,
        cli.seed_offset + cli.seeds as u64 - 1,
        cli.ticks,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "L1 late ATP Natural>Frozen        nat={:.3} frz={:.3} t={:+.2} wins={}/{} p_holm={:.5} {}",
        g1.mean_a, g1.mean_b, g1.t, g1.wins, g1.n, holm[0], verdict(l1)
    );
    println!(
        "L2 late net Natural>Frozen        nat={:+.5} frz={:+.5} t={:+.2} wins={}/{} p_holm={:.5} {}",
        g2.mean_a, g2.mean_b, g2.t, g2.wins, g2.n, holm[1], verdict(l2)
    );
    println!(
        "L3 late net Natural>Shuffled      nat={:+.5} shf={:+.5} t={:+.2} wins={}/{} p_holm={:.5} {}",
        g3.mean_a, g3.mean_b, g3.t, g3.wins, g3.n, holm[2], verdict(l3)
    );
    println!(
        "L4 late starvation frac (<=0.10)  nat={:.4} {}",
        mean_starve,
        verdict(l4)
    );
    println!(
        "L5 late mean ATP (>=1.5)          nat={:.3} {}",
        mean_nat_atp,
        verdict(l5)
    );
    println!("RESULT {passed}/5");

    for (ai, arm) in arms.iter().enumerate() {
        let mut life = Ledger::default();
        let mut late = Ledger::default();
        for r in &by_arm[ai] {
            life.add(&r.ledger_life);
            late.add(&r.ledger_late);
        }
        println!(
            "[{}] late_atp={:.3} late_starve={:.3} life_starve={:.3} late_net={:+.5} life_net={:+.5} cells={:.1} births={:.0} elder_atp={:.3}",
            arm.label(),
            mean(&pick(ai, |r| r.late_atp)),
            mean(&pick(ai, |r| r.late_starve_frac)),
            mean(&pick(ai, |r| r.life_starve_frac)),
            mean(&pick(ai, |r| r.late_net_per_tick)),
            mean(&pick(ai, |r| r.life_net_per_tick)),
            mean(&pick(ai, |r| r.final_cells)),
            mean(&pick(ai, |r| r.final_births)),
            mean(&pick(ai, |r| r.final_elder_atp)),
        );
        println!("  ledger_life={}", life.json());
        println!("  ledger_late={}", late.json());
    }

    // JSON report.
    let mut js = String::new();
    let _ = write!(
        js,
        "{{\"tag\":\"{}\",\"seed_offset\":{},\"seeds\":{},\"ticks\":{},\"gates\":{{\"L1\":{{\"nat\":{:.5},\"frz\":{:.5},\"t\":{:.4},\"p_holm\":{:.6},\"pass\":{}}},\"L2\":{{\"nat\":{:.6},\"frz\":{:.6},\"t\":{:.4},\"p_holm\":{:.6},\"pass\":{}}},\"L3\":{{\"nat\":{:.6},\"shf\":{:.6},\"t\":{:.4},\"p_holm\":{:.6},\"pass\":{}}},\"L4\":{{\"nat\":{:.5},\"pass\":{}}},\"L5\":{{\"nat\":{:.5},\"pass\":{}}},\"passed\":{}}},\"arms\":{{",
        cli.tag, cli.seed_offset, cli.seeds, cli.ticks,
        g1.mean_a, g1.mean_b, g1.t, holm[0], l1,
        g2.mean_a, g2.mean_b, g2.t, holm[1], l2,
        g3.mean_a, g3.mean_b, g3.t, holm[2], l3,
        mean_starve, l4, mean_nat_atp, l5, passed
    );
    for (ai, arm) in arms.iter().enumerate() {
        if ai > 0 {
            js.push(',');
        }
        let _ = write!(js, "\"{}\":[", arm.label());
        for (k, r) in by_arm[ai].iter().enumerate() {
            if k > 0 {
                js.push(',');
            }
            let _ = write!(
                js,
                "{{\"seed\":{},\"late_atp\":{:.5},\"late_net\":{:.6},\"late_starve\":{:.5},\"life_net\":{:.6},\"life_starve\":{:.5},\"cells\":{},\"births\":{},\"elder_atp\":{:.4},\"ledger_life\":{},\"ledger_late\":{},\"series\":[",
                cli.seed_offset + k as u64,
                r.late_atp, r.late_net_per_tick, r.late_starve_frac, r.life_net_per_tick,
                r.life_starve_frac, r.final_cells, r.final_births, r.final_elder_atp,
                r.ledger_life.json(), r.ledger_late.json()
            );
            for (q, p) in r.series.iter().enumerate() {
                if q > 0 {
                    js.push(',');
                }
                let _ = write!(
                    js,
                    "{{\"t\":{},\"atp\":{:.4},\"cells\":{},\"tubes\":{},\"pores\":{},\"births\":{},\"pruned\":{},\"influx\":{:.6},\"stress\":{:.6},\"starve\":{:.4},\"elder_atp\":{:.4}}}",
                    p.tick, p.atp, p.cells, p.tubes, p.pores, p.births, p.pruned,
                    p.win_influx_per_tick, p.win_stress_per_tick, p.win_starve_frac, p.elder_atp
                );
            }
            js.push_str("]}");
        }
        js.push(']');
    }
    js.push_str("}}\n");

    let out_dir = resolve_workspace_root().join("observations").join("longlife");
    let out_path = out_dir.join(format!("longlife_{}.json", cli.tag));
    if let Err(e) = fs::create_dir_all(&out_dir).and_then(|()| fs::write(&out_path, js)) {
        eprintln!("[alpha11-longlife] failed to write report: {e}");
        return ExitCode::from(1);
    }
    println!("report: observations/longlife/longlife_{}.json", cli.tag);
    ExitCode::SUCCESS
}
