//! Crash-safe live host binary (`a11-live`) for unforced, embodied Alpha 11 life on [`Alpha11Organism`].
//!
//! Supports three open-ended continuous habitats:
//! - `--habitat open` (default): Whole-body embodied open world (`OpenSocialHabitat`, `276` sensory
//!   channels, `17` continuous muscles) coupling `MultiGenTerrarium` (`8x8` retina, `2..14`-tick wave
//!   echoes, regenerating nutrient springs, thermal shelters, pushable warm mineral blocks, `16x16`
//!   terrain trails, day/night cycles) + always-attached `EmbodiedBodyOrgans` (`DrawingScratchpad`
//!   hands, `BalanceCatchToy` vestibular limb, `BinauralVocalOrgan` `16 kHz` vocal tract & binaural
//!   `ErbCochlea32` + `LSO`), cohabiting with an autonomous living Elder `Alpha11Organism` (`14.5 cm`
//!   adult vocal tract). Writes transactional `organism.ckpt` + `.prev`, `telemetry.jsonl`,
//!   `development_status.json`, and unedited `16 kHz` `.wav` recordings (`newborn_first_10s.wav`,
//!   `latest_10s_voice_mono.wav`, `latest_10s_room_stereo.wav`).
//! - `--habitat composite`: `AdaptiveCaveWorld` with smooth day/night illumination cycles wrapped
//!   with all 4 human-activity portals inside `CompositePortalMedium` via `LiveHostRunner`.
//! - `--habitat terrarium`: 4-organism elder + younger `Alpha11Organism` cohort inside
//!   `MultiGenTerrarium` with transactional CRC32C-checked atomic population checkpoints.

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;
use std::time::{Duration, Instant};

use a11_core::crc32c::crc32c;
use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{FieldPulse, FieldView, OrganismEngine, PersistError, WorldMedium};
use a11_portals::{
    CompositePortalMedium, EMBODIED_ORGAN_ACTUATORS, EMBODIED_ORGAN_CHANNELS, EmbodiedBodyOrgans,
    LiveHostConfig, LiveHostError, LiveHostRunner, RecoveryOutcome, VOCAL_RING_SAMPLES,
    encode_wav_16khz_mono, encode_wav_16khz_stereo, read_bounded_file, read_linux_rss_bytes,
};
use a11_worlds::{
    AdaptiveCaveWorld, DEFAULT_DAY_NIGHT_PERIOD, EARTH_AUDIO_FRAME_SAMPLES, MultiGenTerrarium,
    TERRARIUM_ACTUATORS, TERRARIUM_CHANNELS, compute_day_night_lux,
};

const DEFAULT_STATE_DIR: &str = "live/state";
const DEFAULT_SEED: u64 = 11;
const DEFAULT_STEPS: u64 = 5000;

const POP_MAGIC: [u8; 8] = *b"A11TPOP3";
const POP_VERSION: u32 = 3;
const MAX_POP_CHECKPOINT_BYTES: usize = 128 * 1024 * 1024;
const TERRARIUM_MAX_LOG_BYTES: usize = 8 * 1024 * 1024;
const TERRARIUM_TELEMETRY_INTERVAL: u64 = 10;
const TERRARIUM_CHECKPOINT_INTERVAL: u64 = 250;
const ELDER_HEAD_START_TICKS: u64 = 400;
const LATENCY_RING_CAP: usize = 2048;

const OPEN_ELDER_WARMUP_TICKS: u64 = 640;
const OPEN_SNAPSHOT_INTERVAL_TICKS: u64 = 250;
const OPEN_VOCAL_WINDOW_TICKS: u64 = 1_000;

/// Total sensory channels in `OpenSocialHabitat` (`128` terrarium + `148` embodied body = `276`).
const OPEN_HABITAT_CHANNELS: usize = TERRARIUM_CHANNELS + EMBODIED_ORGAN_CHANNELS;
/// Total continuous muscles in `OpenSocialHabitat` (`6` terrarium + `11` embodied body = `17`).
const OPEN_HABITAT_ACTUATORS: usize = TERRARIUM_ACTUATORS + EMBODIED_ORGAN_ACTUATORS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HabitatKind {
    Open,
    Composite,
    Terrarium,
}

impl HabitatKind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "open" | "vocal" => Ok(Self::Open),
            "composite" => Ok(Self::Composite),
            "terrarium" => Ok(Self::Terrarium),
            other => Err(format!(
                "invalid --habitat value '{other}' (expected 'open', 'composite', or 'terrarium')"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Composite => "composite",
            Self::Terrarium => "terrarium",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CliArgs {
    state_dir: PathBuf,
    seed: u64,
    steps: u64,
    pace_ms: u64,
    habitat: HabitatKind,
    show_help: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            state_dir: PathBuf::from(DEFAULT_STATE_DIR),
            seed: DEFAULT_SEED,
            steps: DEFAULT_STEPS,
            pace_ms: 0,
            habitat: HabitatKind::Open,
            show_help: false,
        }
    }
}

fn parse_cli_args<I, S>(iter: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut cfg = CliArgs::default();
    let mut args = iter.into_iter();
    while let Some(arg_raw) = args.next() {
        let arg = arg_raw.as_ref();
        match arg {
            "-h" | "--help" => {
                cfg.show_help = true;
            }
            "--state-dir" => {
                let Some(val) = args.next() else {
                    return Err("missing path value after --state-dir".to_string());
                };
                if val.as_ref().is_empty() {
                    return Err("--state-dir cannot be empty".to_string());
                }
                cfg.state_dir = PathBuf::from(val.as_ref());
            }
            "--seed" => {
                let Some(val) = args.next() else {
                    return Err("missing integer value after --seed".to_string());
                };
                cfg.seed = val
                    .as_ref()
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --seed u64 value '{}'", val.as_ref()))?;
            }
            "--steps" => {
                let Some(val) = args.next() else {
                    return Err("missing integer value after --steps".to_string());
                };
                cfg.steps = val
                    .as_ref()
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --steps u64 value '{}'", val.as_ref()))?;
            }
            "--pace-ms" => {
                let Some(val) = args.next() else {
                    return Err("missing integer value after --pace-ms".to_string());
                };
                cfg.pace_ms = val
                    .as_ref()
                    .parse::<u64>()
                    .map_err(|_| format!("invalid --pace-ms u64 value '{}'", val.as_ref()))?;
            }
            "--habitat" => {
                let Some(val) = args.next() else {
                    return Err("missing value after --habitat".to_string());
                };
                cfg.habitat = HabitatKind::parse(val.as_ref())?;
            }
            unknown => {
                return Err(format!("unrecognized flag '{unknown}' (try --help)"));
            }
        }
    }
    Ok(cfg)
}

fn print_usage() {
    println!(
        "Usage: a11-live [--state-dir <path>] [--seed <u64>] [--steps <u64>] [--pace-ms <u64>] [--habitat open|composite|terrarium]\n\
         \n\
         Options:\n\
         \x20 --state-dir <path>                      Directory for checkpoints, WAVs & JSONL telemetry (default: {DEFAULT_STATE_DIR})\n\
         \x20 --seed <u64>                            Deterministic world & genesis seed (default: {DEFAULT_SEED})\n\
         \x20 --steps <u64>                           Ticks to execute; 0 runs continuously until STOP file or signal (default: {DEFAULT_STEPS})\n\
         \x20 --pace-ms <u64>                         Minimum wall-clock milliseconds per tick (0 = unthrottled, 20 = 1.0x 16 kHz real time)\n\
         \x20 --habitat open|composite|terrarium      Primary habitat mode (default: open)"
    );
}

#[inline]
fn pace_tick(tick_start: Instant, pace_ms: u64) {
    if pace_ms > 0 {
        let budget = Duration::from_millis(pace_ms);
        let elapsed = tick_start.elapsed();
        if elapsed < budget {
            thread::sleep(budget - elapsed);
        }
    }
}

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

// =============================================================================
// Mode 0 (Default): Whole-Body Open Social Habitat (`OpenSocialHabitat`)
// =============================================================================

struct OpenSocialHabitat {
    terrarium: MultiGenTerrarium,
    elder_brain: Alpha11Organism,
    elder_body: EmbodiedBodyOrgans,
    elder_ambient: [f32; OPEN_HABITAT_CHANNELS],
    elder_influx: f32,
    elder_stress: f32,

    focal_body: EmbodiedBodyOrgans,
    focal_ambient: [f32; OPEN_HABITAT_CHANNELS],
    focal_influx: f32,
    focal_stress: f32,
    shared_slab_contact_ticks: u64,
}

impl OpenSocialHabitat {
    fn new(seed: u64, enable_pcm: bool) -> Self {
        let mut terrarium = MultiGenTerrarium::new_earth_ecology(seed, 1);
        let mut elder_brain =
            Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
        let mut elder_body = EmbodiedBodyOrgans::new_adult(seed ^ 0xE1DE_B0D1_2026_0002);
        let mut elder_ambient = [0.0_f32; OPEN_HABITAT_CHANNELS];

        // Let the autonomous Elder mature in the Earth ecology first, forging trails, cracking pods, and mastering limbs.
        for _ in 0..OPEN_ELDER_WARMUP_TICKS {
            let tv = terrarium.agent_view(0);
            for (i, slot) in elder_ambient[..TERRARIUM_CHANNELS].iter_mut().enumerate() {
                *slot = clamp_unit(tv.ambient_in.get(i).copied().unwrap_or(0.0));
            }
            elder_ambient[TERRARIUM_CHANNELS..].copy_from_slice(elder_body.sensory_channels());
            let elder_view = FieldView {
                ambient_in: &elder_ambient,
                metabolic_influx: clamp_nonneg(tv.metabolic_influx),
                tissue_stress: clamp_nonneg(tv.tissue_stress + elder_body.last_organ_stress()),
                num_actuators: OPEN_HABITAT_ACTUATORS,
            };
            let ep = elder_brain
                .step(elder_view)
                .unwrap_or_else(|_| FieldPulse::silent(OPEN_HABITAT_ACTUATORS));

            let mut e_body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
            for (j, slot) in e_body_act.iter_mut().enumerate() {
                *slot = clamp_unit(
                    ep.actuator_out
                        .get(TERRARIUM_ACTUATORS + j)
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            let _ = elder_body.step_limbs_and_radiate_voice(&e_body_act);
            let (e_stab, e_hw) = elder_body.embodied_kinematics();
            terrarium.set_agent_embodied_kinematics(0, e_stab, e_hw);
            terrarium.set_agent_embodied_voice_rms(0, elder_body.vocal().last_radiated_rms());

            let mut e_world = FieldPulse {
                actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
            };
            for i in 0..TERRARIUM_ACTUATORS {
                e_world
                    .actuator_out
                    .push(clamp_unit(ep.actuator_out.get(i).copied().unwrap_or(0.0)));
            }
            terrarium.apply_population_pulses(&[e_world]);
            elder_body.apply_terrain_sway(terrarium.agent_lateral_sway(0));

            let mut env_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
            let mut env_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
            let enc = terrarium.binaural_earth_waves_for_agent(0, &mut env_l, &mut env_r);
            elder_body.finish_binaural_ecology_transduction(None, Some((&env_l, &env_r)), enc);
        }
        elder_body.vocal_mut().reset_window_observations();

        // Spawn the newborn body in slot 1 near a warm resonant shelter.
        terrarium.respawn_newborn_in_slot(1);
        let mut focal_body = EmbodiedBodyOrgans::new_infant(seed ^ 0xF0CA_1B0D_2026_0003);
        if enable_pcm {
            focal_body.enable_pcm_recording();
        }

        let mut hab = Self {
            terrarium,
            elder_brain,
            elder_body,
            elder_ambient,
            elder_influx: 0.0,
            elder_stress: 0.0,
            focal_body,
            focal_ambient: [0.0; OPEN_HABITAT_CHANNELS],
            focal_influx: 0.0,
            focal_stress: 0.0,
            shared_slab_contact_ticks: 0,
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
        self.elder_ambient[TERRARIUM_CHANNELS..]
            .copy_from_slice(self.elder_body.sensory_channels());
        self.elder_influx = clamp_nonneg(ev.metabolic_influx);
        self.elder_stress = clamp_nonneg(ev.tissue_stress + self.elder_body.last_organ_stress());

        let fv = self.terrarium.agent_view(1);
        for (i, slot) in self.focal_ambient[..TERRARIUM_CHANNELS]
            .iter_mut()
            .enumerate()
        {
            *slot = clamp_unit(fv.ambient_in.get(i).copied().unwrap_or(0.0));
        }
        self.focal_ambient[TERRARIUM_CHANNELS..]
            .copy_from_slice(self.focal_body.sensory_channels());
        self.focal_influx = clamp_nonneg(fv.metabolic_influx);
        self.focal_stress = clamp_nonneg(fv.tissue_stress + self.focal_body.last_organ_stress());
    }
}

impl WorldMedium for OpenSocialHabitat {
    fn num_ambient_channels(&self) -> usize {
        OPEN_HABITAT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        OPEN_HABITAT_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.focal_ambient,
            metabolic_influx: self.focal_influx,
            tissue_stress: self.focal_stress,
            num_actuators: OPEN_HABITAT_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, focal_pulse: &FieldPulse) {
        // 1. Step autonomous Elder brain on its own 276-channel whole-body sensory view.
        let elder_view = FieldView {
            ambient_in: &self.elder_ambient,
            metabolic_influx: self.elder_influx,
            tissue_stress: self.elder_stress,
            num_actuators: OPEN_HABITAT_ACTUATORS,
        };
        let elder_pulse = self
            .elder_brain
            .step(elder_view)
            .unwrap_or_else(|_| FieldPulse::silent(OPEN_HABITAT_ACTUATORS));

        // 2. Step both organisms' hands, vestibular limbs, and 16 kHz vocal radiation first
        //    so real postural stability & hand work physically drive 2D traction and pod cracking.
        let mut elder_body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        let mut focal_body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        for j in 0..EMBODIED_ORGAN_ACTUATORS {
            elder_body_act[j] = clamp_unit(
                elder_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
            focal_body_act[j] = clamp_unit(
                focal_pulse
                    .actuator_out
                    .get(TERRARIUM_ACTUATORS + j)
                    .copied()
                    .unwrap_or(0.0),
            );
        }
        let elder_wave = self.elder_body.step_limbs_and_radiate_voice(&elder_body_act);
        let focal_wave = self.focal_body.step_limbs_and_radiate_voice(&focal_body_act);

        let (e_stab, e_hw) = self.elder_body.embodied_kinematics();
        let (f_stab, f_hw) = self.focal_body.embodied_kinematics();
        self.terrarium.set_agent_embodied_kinematics(0, e_stab, e_hw);
        self.terrarium.set_agent_embodied_kinematics(1, f_stab, f_hw);
        self.terrarium
            .set_agent_embodied_voice_rms(0, self.elder_body.vocal().last_radiated_rms());
        self.terrarium
            .set_agent_embodied_voice_rms(1, self.focal_body.vocal().last_radiated_rms());

        // 3. Step shared 2D Earth physics for [Elder (slot 0), Newborn (slot 1)].
        let mut elder_world = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        let mut focal_world = FieldPulse {
            actuator_out: Vec::with_capacity(TERRARIUM_ACTUATORS),
        };
        for i in 0..TERRARIUM_ACTUATORS {
            elder_world.actuator_out.push(clamp_unit(
                elder_pulse.actuator_out.get(i).copied().unwrap_or(0.0),
            ));
            focal_world.actuator_out.push(clamp_unit(
                focal_pulse.actuator_out.get(i).copied().unwrap_or(0.0),
            ));
        }
        self.terrarium
            .apply_population_pulses(&[elder_world, focal_world]);

        // 4. Couple terrain acceleration sway into vestibular limbs and shared stone hearth canvas when close.
        self.elder_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(0));
        self.focal_body
            .apply_terrain_sway(self.terrarium.agent_lateral_sway(1));

        let positions = &self.terrarium.metrics().agent_positions;
        let e_pos = positions.first().copied().unwrap_or((0.0, 0.0));
        let f_pos = positions.get(1).copied().unwrap_or((0.0, 0.0));
        let rel_elder_from_focal = (e_pos.0 - f_pos.0, e_pos.1 - f_pos.1);
        let rel_focal_from_elder = (f_pos.0 - e_pos.0, f_pos.1 - e_pos.1);
        let pair_dist = rel_elder_from_focal.0.hypot(rel_elder_from_focal.1);
        if pair_dist <= 3.6 {
            let coupling = 0.26 * (1.0 - pair_dist / 3.6);
            EmbodiedBodyOrgans::couple_shared_hands(
                &mut self.elder_body,
                &mut self.focal_body,
                coupling,
            );
            self.shared_slab_contact_ticks = self.shared_slab_contact_ticks.saturating_add(1);
        }

        // 5. Propagate 16 kHz voices (diffracting 0.68x across opaque basalt ridges) + flowing water + shell cracks + boulder rumbles + 40 ms cave reverb!
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
        let mut f_env_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let mut f_env_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let f_enclosure =
            self.terrarium
                .binaural_earth_waves_for_agent(1, &mut f_env_l, &mut f_env_r);
        self.focal_body.finish_binaural_ecology_transduction(
            Some((&elder_wave_diff, rel_elder_from_focal)),
            Some((&f_env_l, &f_env_r)),
            f_enclosure,
        );

        let mut e_env_l = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let mut e_env_r = [0.0_f32; EARTH_AUDIO_FRAME_SAMPLES];
        let e_enclosure =
            self.terrarium
                .binaural_earth_waves_for_agent(0, &mut e_env_l, &mut e_env_r);
        self.elder_body.finish_binaural_ecology_transduction(
            Some((&focal_wave_diff, rel_focal_from_elder)),
            Some((&e_env_l, &e_env_r)),
            e_enclosure,
        );

        self.refresh_views();
    }
}

/// Wraps `AdaptiveCaveWorld` with a smooth continuous day/night `set_illumination` cycle.
struct DayNightCaveWorld {
    cave: AdaptiveCaveWorld,
    period: u64,
    phase_offset: u64,
}

impl DayNightCaveWorld {
    fn new(seed: u64, period: u64) -> Self {
        let p = period.max(16);
        let init_lux = compute_day_night_lux(0, p);
        Self {
            cave: AdaptiveCaveWorld::new(seed, init_lux),
            period: p,
            phase_offset: 0,
        }
    }

    fn set_phase_offset(&mut self, offset: u64) {
        self.phase_offset = offset;
        let lux = compute_day_night_lux(
            self.phase_offset.saturating_add(self.cave.metrics().tick),
            self.period,
        );
        self.cave.set_illumination(lux);
    }

    fn cave(&self) -> &AdaptiveCaveWorld {
        &self.cave
    }
}

impl WorldMedium for DayNightCaveWorld {
    fn num_ambient_channels(&self) -> usize {
        self.cave.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.cave.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        self.cave.current_view()
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.cave.apply_pulse(pulse);
        let effective_tick = self.phase_offset.saturating_add(self.cave.metrics().tick);
        let next_lux = compute_day_night_lux(effective_tick, self.period);
        self.cave.set_illumination(next_lux);
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_cli_args(&raw_args) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("error: {msg}");
            return ExitCode::FAILURE;
        }
    };

    if cli.show_help {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let res = match cli.habitat {
        HabitatKind::Open => run_open_habitat(&cli),
        HabitatKind::Composite => run_composite_habitat(&cli),
        HabitatKind::Terrarium => run_terrarium_habitat(&cli),
    };

    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("fatal live-host error: {err:?}");
            ExitCode::FAILURE
        }
    }
}

fn recover_elder_if_present(
    state_dir: &Path,
    seed: u64,
    elder_brain: &mut Alpha11Organism,
) -> Result<(), LiveHostError> {
    let primary = state_dir.join("elder.ckpt");
    let prev = state_dir.join("elder.ckpt.prev");
    let tmp = state_dir.join("elder.ckpt.tmp");
    let has_primary = primary.exists();
    let has_prev = prev.exists();
    if !has_primary && !has_prev {
        return Ok(());
    }

    let try_decode = |bytes: &[u8]| -> Result<Alpha11Organism, PersistError> {
        let mut shadow = Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
        shadow.restore_checkpoint(bytes)?;
        Ok(shadow)
    };

    let primary_err = if has_primary {
        match read_bounded_file(&primary, MAX_POP_CHECKPOINT_BYTES) {
            Ok(bytes) => match try_decode(&bytes) {
                Ok(restored) => {
                    *elder_brain = restored;
                    return Ok(());
                }
                Err(e) => e,
            },
            Err(e) => e,
        }
    } else {
        PersistError::IoFailure
    };

    if !has_prev {
        return Err(LiveHostError::Persist(primary_err));
    }

    let prev_bytes = read_bounded_file(&prev, MAX_POP_CHECKPOINT_BYTES)?;
    let restored = try_decode(&prev_bytes)?;
    write_atomic_bytes(&tmp, &primary, &prev_bytes)?;
    *elder_brain = restored;
    Ok(())
}

fn save_elder_transactional(
    state_dir: &Path,
    seed: u64,
    elder_brain: &Alpha11Organism,
) -> Result<(), LiveHostError> {
    let primary = state_dir.join("elder.ckpt");
    let prev = state_dir.join("elder.ckpt.prev");
    let prev_tmp = state_dir.join("elder.ckpt.prev.tmp");
    let tmp = state_dir.join("elder.ckpt.tmp");

    let bytes = elder_brain.encode_checkpoint()?;
    let mut verify = Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
    verify.restore_checkpoint(&bytes)?;
    write_synced(&tmp, &bytes)?;

    if primary.exists()
        && let Ok(existing) = read_bounded_file(&primary, MAX_POP_CHECKPOINT_BYTES)
    {
        let mut check_existing =
            Alpha11Organism::new(seed ^ 0xE1DE_A110_2026_0001, OrganismConfig::default());
        if check_existing.restore_checkpoint(&existing).is_ok() {
            write_atomic_bytes(&prev_tmp, &prev, &existing)?;
        }
    }

    fs::rename(&tmp, &primary)?;
    sync_parent(&primary);
    Ok(())
}

fn count_pores_by_modality(channels: &[u32]) -> (usize, usize, usize, usize, usize, usize) {
    let mut ret = 0usize;
    let mut ech = 0usize;
    let mut hnd = 0usize;
    let mut vst = 0usize;
    let mut coc = 0usize;
    let mut dst = 0usize;
    for &ch in channels {
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
    (ret, ech, hnd, vst, coc, dst)
}

fn run_open_habitat(cli: &CliArgs) -> Result<(), LiveHostError> {
    fs::create_dir_all(&cli.state_dir)?;
    let host_cfg = LiveHostConfig::in_directory(&cli.state_dir);

    let mut open_world = OpenSocialHabitat::new(cli.seed, true);
    recover_elder_if_present(&cli.state_dir, cli.seed, &mut open_world.elder_brain)?;
    open_world.refresh_views();

    let seed = cli.seed;
    let (mut runner, recovery) = LiveHostRunner::open_or_recover(host_cfg, open_world, move || {
        Alpha11Organism::new(seed, OrganismConfig::default())
    })?;
    {
        let eng_snap = runner.engine().snapshot();
        let m_items = runner.engine().motor_maturation_by_channel();
        let (voc_m, _) = group_motor_stats(&m_items, 12, 17);
        runner.world_mut().focal_body.advance_vocal_ontogeny(
            eng_snap.tick,
            eng_snap.mitosis_births,
            voc_m,
        );
    }

    let is_genesis_birth =
        matches!(recovery, RecoveryOutcome::Genesis) && runner.engine().snapshot().tick == 0;
    let newborn_wav_path = cli.state_dir.join("newborn_first_10s.wav");
    let newborn_tmp_path = cli.state_dir.join("newborn_first_10s.wav.tmp");
    let mut newborn_saved = newborn_wav_path.exists();

    let ckpt_interval = runner.config().checkpoint_interval_ticks;
    let stop_sentinel = cli.state_dir.join("STOP");
    let mut local_steps = 0_u64;

    loop {
        if cli.steps > 0 && local_steps >= cli.steps {
            break;
        }
        if (local_steps & 15) == 0 && stop_sentinel.exists() {
            break;
        }

        let t0 = Instant::now();
        runner.step_once()?;
        local_steps = local_steps.saturating_add(1);
        if (local_steps & 15) == 0 {
            let eng_snap = runner.engine().snapshot();
            let m_items = runner.engine().motor_maturation_by_channel();
            let (voc_m, _) = group_motor_stats(&m_items, 12, 17);
            runner.world_mut().focal_body.advance_vocal_ontogeny(
                eng_snap.tick,
                eng_snap.mitosis_births,
                voc_m,
            );
        }

        if ckpt_interval > 0 && local_steps % ckpt_interval == 0 {
            save_elder_transactional(&cli.state_dir, cli.seed, &runner.world().elder_brain)?;
        }

        let newborn_pcm = runner.world().focal_body.vocal().newborn_first_10s_pcm();
        if is_genesis_birth && !newborn_saved && newborn_pcm.len() >= VOCAL_RING_SAMPLES {
            let wav = encode_wav_16khz_mono(newborn_pcm);
            write_atomic_bytes(&newborn_tmp_path, &newborn_wav_path, &wav)?;
            newborn_saved = true;
        }

        if local_steps % OPEN_SNAPSHOT_INTERVAL_TICKS == 0 {
            write_open_development_snapshots(&cli.state_dir, runner.engine(), &runner.stats(), runner.world())?;
        }

        if local_steps % OPEN_VOCAL_WINDOW_TICKS == 0 {
            runner
                .world_mut()
                .focal_body
                .vocal_mut()
                .reset_window_observations();
            runner
                .world_mut()
                .elder_body
                .vocal_mut()
                .reset_window_observations();
        }

        pace_tick(t0, cli.pace_ms);
    }

    let newborn_pcm = runner.world().focal_body.vocal().newborn_first_10s_pcm();
    if is_genesis_birth && !newborn_saved && !newborn_pcm.is_empty() {
        let wav = encode_wav_16khz_mono(newborn_pcm);
        write_atomic_bytes(&newborn_tmp_path, &newborn_wav_path, &wav)?;
    }

    if runner.stats().steps_executed > 0 {
        if ckpt_interval == 0 || runner.stats().steps_executed % ckpt_interval != 0 {
            let _ = runner.save_checkpoint()?;
        }
        save_elder_transactional(&cli.state_dir, cli.seed, &runner.world().elder_brain)?;
    }

    let stats = runner.stats();
    write_open_development_snapshots(&cli.state_dir, runner.engine(), &stats, runner.world())?;

    let snap = runner.engine().snapshot();
    let body_obs = runner.world().focal_body.observation();
    let elder_voc = runner.world().elder_body.vocal().observation();
    let tm = runner.world().terrarium.metrics();
    let eco = runner.world().terrarium.ecology_metrics();

    println!("=== a11-live summary (habitat=open, earth_ecology={}) ===", eco.enabled);
    println!("recovery_outcome: {recovery:?}");
    println!(
        "brain_growth: tick={} cells={} tubes={} stiff_tubes={} echo_tubes={} mean_tau={:.2} max_tau={:.2} octaves={} mitosis={} splits={} sleep_ticks={}",
        snap.tick,
        snap.cell_count,
        snap.tube_count,
        snap.stiff_highway_tubes,
        snap.active_echo_delay_tubes,
        snap.mean_tau,
        snap.max_tau,
        snap.tau_octave_count,
        snap.mitosis_births,
        snap.contradiction_splits,
        snap.rest_consolidation_ticks,
    );
    println!(
        "physiology_and_muscles: atp={:.3} agency={:.5} motor_mastery={:.4} motor_jerk={:.4} used_state_bytes={} os_rss_bytes={:?}",
        snap.atp,
        snap.mean_interventional_agency,
        runner.engine().mean_motor_mastery(),
        runner.engine().mean_motor_jerk(),
        stats.last_used_state_bytes,
        stats.os_rss_bytes,
    );
    println!(
        "embodied_and_ecology: strokes={} canvas_cov={:.4} shared_touch={} pods_hand={} coop_grove={} pods_stone={} coop_push={} huddling={} occ_approach={} upright_mean={:.2} voiced={} onsets={} syl_trans={} plosives={} alt_replies={} cave_echo={} F1={:.1}Hz F2={:.1}Hz (elder_F2={:.1}Hz) pairwise_dist={:.2}",
        body_obs.scratchpad.active_stroke_ticks,
        body_obs.scratchpad.canvas_coverage_fraction,
        runner.world().shared_slab_contact_ticks,
        eco.pods_cracked_by_hands,
        eco.coop_grove_pods_cracked,
        eco.pods_cracked_by_stone,
        eco.cooperative_boulder_ticks,
        eco.huddling_ticks,
        eco.occluded_vocal_approaches,
        body_obs.balance.mean_upright_duration_ticks,
        body_obs.vocal.voiced_ticks,
        body_obs.vocal.syllable_onsets,
        body_obs.vocal.syllable_transitions,
        body_obs.vocal.plosive_burst_frames,
        body_obs.vocal.alternating_response_ticks,
        body_obs.vocal.shelter_echo_ticks,
        body_obs.vocal.mean_f1_hz,
        body_obs.vocal.mean_f2_hz,
        elder_voc.mean_f2_hz,
        tm.mean_pairwise_distance,
    );

    Ok(())
}

fn group_motor_stats(items: &[(u32, f32, f32)], lo: u32, hi_excl: u32) -> (f32, f32) {
    let mut m_sum = 0.0_f32;
    let mut j_sum = 0.0_f32;
    let mut cnt = 0usize;
    for &(ch, mastery, jerk) in items {
        if ch >= lo && ch < hi_excl {
            m_sum += mastery;
            j_sum += jerk;
            cnt += 1;
        }
    }
    if cnt > 0 {
        let inv = 1.0 / (cnt as f32);
        (m_sum * inv, j_sum * inv)
    } else {
        (0.0, 0.0)
    }
}

fn write_open_development_snapshots(
    state_dir: &Path,
    org: &Alpha11Organism,
    stats: &a11_portals::LiveHostStats,
    hab: &OpenSocialHabitat,
) -> Result<(), LiveHostError> {
    let mono_pcm = hab.focal_body.vocal().latest_10s_self_mono_pcm();
    if !mono_pcm.is_empty() {
        let mono_wav = encode_wav_16khz_mono(&mono_pcm);
        write_atomic_bytes(
            &state_dir.join("latest_10s_voice_mono.wav.tmp"),
            &state_dir.join("latest_10s_voice_mono.wav"),
            &mono_wav,
        )?;
    }

    let stereo_pcm = hab.focal_body.vocal().latest_10s_room_stereo_pcm();
    if !stereo_pcm.is_empty() {
        let stereo_wav = encode_wav_16khz_stereo(&stereo_pcm);
        write_atomic_bytes(
            &state_dir.join("latest_10s_room_stereo.wav.tmp"),
            &state_dir.join("latest_10s_room_stereo.wav"),
            &stereo_wav,
        )?;
    }

    let snap = org.snapshot();
    let motor_items = org.motor_maturation_by_channel();
    let (loco_m, loco_j) = group_motor_stats(&motor_items, 0, 6);
    let (hand_m, hand_j) = group_motor_stats(&motor_items, 6, 10);
    let (vest_m, vest_j) = group_motor_stats(&motor_items, 10, 12);
    let (voc_m, voc_j) = group_motor_stats(&motor_items, 12, 17);
    let (retina_pores, echo_pores, hand_pores, vestibular_pores, cochlea_pores, distractor_pores) =
        count_pores_by_modality(&snap.sensor_pore_channels);

    let body_obs = hab.focal_body.observation();
    let elder_obs = hab.elder_body.observation();
    let elder_voc = elder_obs.vocal;
    let elder_snap = hab.elder_brain.snapshot();
    let (e_ret, e_ech, e_hnd, e_vst, e_coc, e_dst) =
        count_pores_by_modality(&elder_snap.sensor_pore_channels);
    let tm = hab.terrarium.metrics();
    let eco = hab.terrarium.ecology_metrics();

    let e_pos = tm.agent_positions.first().copied().unwrap_or((0.0, 0.0));
    let f_pos = tm.agent_positions.get(1).copied().unwrap_or((0.0, 0.0));
    let elder_cum_influx = tm.agent_metabolic_influx.first().copied().unwrap_or(0.0);
    let elder_cum_stress = tm.agent_tissue_stress.first().copied().unwrap_or(0.0);
    let focal_cum_influx = tm.agent_metabolic_influx.get(1).copied().unwrap_or(0.0);
    let focal_cum_stress = tm.agent_tissue_stress.get(1).copied().unwrap_or(0.0);
    let elder_pods = eco.agent_pods_cracked.first().copied().unwrap_or(0);
    let focal_pods = eco.agent_pods_cracked.get(1).copied().unwrap_or(0);
    let rss = stats.os_rss_bytes.unwrap_or(0);

    let mut json = String::with_capacity(4096);
    let _ = writeln!(
        json,
        "{{\n\
         \x20 \"tick\": {},\n\
         \x20 \"world_tick\": {},\n\
         \x20 \"position_xy\": [{}, {}],\n\
         \x20 \"brain\": {{\"cells\": {}, \"tubes\": {}, \"stiff_consolidated_tubes\": {}, \"inhibitory_tubes\": {}, \"echo_delay_tubes\": {}, \"mean_tau\": {}, \"max_tau\": {}, \"tau_octaves\": {}, \"mitosis_births\": {}, \"atrophied_pruned\": {}, \"contradiction_splits\": {}, \"myelinated_shortcuts\": {}, \"subthreshold_rehearsals\": {}, \"sleep_consolidation_ticks\": {}}},\n\
         \x20 \"physiology\": {{\"atp\": {}, \"adenosine\": {}, \"temperature\": {}, \"interventional_agency\": {}, \"prediction_error\": {}, \"cumulative_influx\": {}, \"cumulative_stress\": {}}},\n\
         \x20 \"sensory_pores\": {{\"total\": {}, \"retina\": {}, \"wave_echo\": {}, \"hands_canvas\": {}, \"vestibular\": {}, \"binaural_cochlea\": {}, \"distractors\": {}}},\n\
         \x20 \"muscle_maturation\": {{\"overall_mastery\": {}, \"overall_jerk\": {}, \"locomotion_mastery\": {}, \"locomotion_jerk\": {}, \"hand_mastery\": {}, \"hand_jerk\": {}, \"vestibular_mastery\": {}, \"vestibular_jerk\": {}, \"vocal_mastery\": {}, \"vocal_jerk\": {}}},\n\
         \x20 \"hands_scratchpad\": {{\"active_stroke_ticks\": {}, \"shared_slab_contact_ticks\": {}, \"pods_cracked_by_focal\": {}, \"canvas_coverage\": {}, \"spatial_autocorrelation\": {}, \"revisit_ratio\": {}, \"trajectory_entropy\": {}}},\n\
         \x20 \"vestibular_balance\": {{\"mean_upright_ticks\": {}, \"max_upright_streak\": {}, \"falls\": {}, \"phase_lead_correlation\": {}}},\n\
         \x20 \"voice_16khz\": {{\"tract_length_cm\": {}, \"window_ticks\": {}, \"voiced_ticks\": {}, \"syllable_onsets\": {}, \"syllable_transitions\": {}, \"plosive_bursts\": {}, \"alternating_responses\": {}, \"simultaneous_overlaps\": {}, \"shelter_echo_ticks\": {}, \"mean_f0_hz\": {}, \"mean_f1_hz\": {}, \"mean_f2_hz\": {}, \"f1_spread_hz\": {}, \"f2_spread_hz\": {}, \"articulator_delta\": {}}},\n\
         \x20 \"earth_ecology\": {{\"season_phase\": {}, \"illumination_lux\": {}, \"flowing_springs\": {}, \"mean_spring_flow\": {}, \"ripe_intact_pods\": {}, \"open_cracked_pods\": {}, \"pods_cracked_by_hands\": {}, \"coop_grove_pods_cracked\": {}, \"pods_cracked_by_stone\": {}, \"cooperative_boulder_ticks\": {}, \"huddling_ticks\": {}, \"acoustic_impact_events\": {}, \"visual_occlusions_active\": {}, \"occluded_vocal_approaches\": {}, \"terrain_mark_coverage\": {}, \"pairwise_distance_m\": {}}},\n\
         \x20 \"cohabiting_elder\": {{\"tick\": {}, \"position_xy\": [{}, {}], \"tract_length_cm\": {}, \"cells\": {}, \"tubes\": {}, \"stiff_tubes\": {}, \"mean_tau\": {}, \"atp\": {}, \"cumulative_influx\": {}, \"cumulative_stress\": {}, \"motor_mastery\": {}, \"motor_jerk\": {}, \"pores\": {{\"retina\": {}, \"echo\": {}, \"hands\": {}, \"vestibular\": {}, \"cochlea\": {}, \"distractors\": {}}}, \"pods_cracked\": {}, \"canvas_coverage\": {}, \"voiced_ticks\": {}, \"syllable_onsets\": {}, \"syllable_transitions\": {}, \"plosive_bursts\": {}, \"alternating_responses\": {}, \"shelter_echo_ticks\": {}, \"mean_f1_hz\": {}, \"mean_f2_hz\": {}}},\n\
         \x20 \"resources\": {{\"used_state_bytes\": {}, \"os_rss_bytes\": {}, \"p50_step_us\": {}, \"p95_step_us\": {}, \"p99_step_us\": {}}}\n\
         }}",
        snap.tick,
        tm.tick,
        fmt_f32(f_pos.0),
        fmt_f32(f_pos.1),
        snap.cell_count,
        snap.tube_count,
        snap.stiff_highway_tubes,
        snap.inhibitory_tubes,
        snap.active_echo_delay_tubes,
        fmt_f32(snap.mean_tau),
        fmt_f32(snap.max_tau),
        snap.tau_octave_count,
        snap.mitosis_births,
        snap.atrophied_structures,
        snap.contradiction_splits,
        snap.myelinated_shortcuts,
        snap.subthreshold_rehearsals,
        snap.rest_consolidation_ticks,
        fmt_f32(snap.atp),
        fmt_f32(snap.adenosine),
        fmt_f32(snap.temperature),
        fmt_f32(snap.mean_interventional_agency),
        fmt_f32(snap.mean_prediction_error),
        fmt_f32(focal_cum_influx),
        fmt_f32(focal_cum_stress),
        snap.sensor_pore_count,
        retina_pores,
        echo_pores,
        hand_pores,
        vestibular_pores,
        cochlea_pores,
        distractor_pores,
        fmt_f32(org.mean_motor_mastery()),
        fmt_f32(org.mean_motor_jerk()),
        fmt_f32(loco_m),
        fmt_f32(loco_j),
        fmt_f32(hand_m),
        fmt_f32(hand_j),
        fmt_f32(vest_m),
        fmt_f32(vest_j),
        fmt_f32(voc_m),
        fmt_f32(voc_j),
        body_obs.scratchpad.active_stroke_ticks,
        hab.shared_slab_contact_ticks,
        focal_pods,
        fmt_f32(body_obs.scratchpad.canvas_coverage_fraction),
        fmt_f32(body_obs.scratchpad.spatial_autocorrelation),
        fmt_f32(body_obs.scratchpad.revisit_ratio),
        fmt_f32(body_obs.scratchpad.stylus_trajectory_entropy),
        fmt_f32(body_obs.balance.mean_upright_duration_ticks),
        body_obs.balance.max_upright_streak,
        body_obs.balance.sphere_falls,
        fmt_f32(body_obs.balance.torque_phase_lead_correlation),
        fmt_f32(body_obs.vocal.tract_length_cm),
        body_obs.vocal.window_ticks,
        body_obs.vocal.voiced_ticks,
        body_obs.vocal.syllable_onsets,
        body_obs.vocal.syllable_transitions,
        body_obs.vocal.plosive_burst_frames,
        body_obs.vocal.alternating_response_ticks,
        body_obs.vocal.simultaneous_overlap_ticks,
        body_obs.vocal.shelter_echo_ticks,
        fmt_f32(body_obs.vocal.mean_f0_hz),
        fmt_f32(body_obs.vocal.mean_f1_hz),
        fmt_f32(body_obs.vocal.mean_f2_hz),
        fmt_f32(body_obs.vocal.f1_spread_hz),
        fmt_f32(body_obs.vocal.f2_spread_hz),
        fmt_f32(body_obs.vocal.mean_articulator_delta),
        fmt_f32(eco.season_phase),
        fmt_f32(tm.illumination_lux),
        eco.flowing_springs,
        fmt_f32(eco.mean_spring_flow),
        eco.ripe_intact_pods,
        eco.open_cracked_pods,
        eco.pods_cracked_by_hands,
        eco.coop_grove_pods_cracked,
        eco.pods_cracked_by_stone,
        eco.cooperative_boulder_ticks,
        eco.huddling_ticks,
        eco.acoustic_impact_events,
        eco.visual_occlusions_active,
        eco.occluded_vocal_approaches,
        fmt_f32(tm.persistent_mark_coverage),
        fmt_f32(tm.mean_pairwise_distance),
        elder_snap.tick,
        fmt_f32(e_pos.0),
        fmt_f32(e_pos.1),
        fmt_f32(elder_voc.tract_length_cm),
        elder_snap.cell_count,
        elder_snap.tube_count,
        elder_snap.stiff_highway_tubes,
        fmt_f32(elder_snap.mean_tau),
        fmt_f32(elder_snap.atp),
        fmt_f32(elder_cum_influx),
        fmt_f32(elder_cum_stress),
        fmt_f32(hab.elder_brain.mean_motor_mastery()),
        fmt_f32(hab.elder_brain.mean_motor_jerk()),
        e_ret,
        e_ech,
        e_hnd,
        e_vst,
        e_coc,
        e_dst,
        elder_pods,
        fmt_f32(elder_obs.scratchpad.canvas_coverage_fraction),
        elder_voc.voiced_ticks,
        elder_voc.syllable_onsets,
        elder_voc.syllable_transitions,
        elder_voc.plosive_burst_frames,
        elder_voc.alternating_response_ticks,
        elder_voc.shelter_echo_ticks,
        fmt_f32(elder_voc.mean_f1_hz),
        fmt_f32(elder_voc.mean_f2_hz),
        stats.last_used_state_bytes,
        rss,
        fmt_f64(stats.p50_step_us),
        fmt_f64(stats.p95_step_us),
        fmt_f64(stats.p99_step_us),
    );

    write_atomic_bytes(
        &state_dir.join("development_status.json.tmp"),
        &state_dir.join("development_status.json"),
        json.as_bytes(),
    )
}

// =============================================================================
// Mode 1: Composite Habitat (AdaptiveCaveWorld + 4 Human-Activity Portals)
// =============================================================================

fn run_composite_habitat(cli: &CliArgs) -> Result<(), LiveHostError> {
    fs::create_dir_all(&cli.state_dir)?;
    let host_cfg = LiveHostConfig::in_directory(&cli.state_dir);

    let cave = DayNightCaveWorld::new(cli.seed, DEFAULT_DAY_NIGHT_PERIOD);
    let composite = CompositePortalMedium::with_all_portals(cave, cli.seed ^ 0xA110_9087_55AA);
    let seed = cli.seed;

    let (mut runner, recovery) = LiveHostRunner::open_or_recover(host_cfg, composite, move || {
        Alpha11Organism::new(seed, OrganismConfig::default())
    })?;

    let start_tick = runner.engine().snapshot().tick;
    if start_tick > 0 {
        runner.world_mut().host_mut().set_phase_offset(start_tick);
        runner.world_mut().sync_view();
    }

    let stop_sentinel = cli.state_dir.join("STOP");
    if cli.steps == 0 {
        loop {
            if stop_sentinel.exists() {
                break;
            }
            let t0 = Instant::now();
            runner.step_once()?;
            pace_tick(t0, cli.pace_ms);
        }
    } else {
        for step_idx in 0..cli.steps {
            if (step_idx & 63) == 0 && stop_sentinel.exists() {
                break;
            }
            let t0 = Instant::now();
            runner.step_once()?;
            pace_tick(t0, cli.pace_ms);
        }
    }

    let ckpt_interval = runner.config().checkpoint_interval_ticks;
    if runner.stats().steps_executed > 0
        && (ckpt_interval == 0 || runner.stats().steps_executed % ckpt_interval != 0)
    {
        let _ = runner.save_checkpoint()?;
    }

    let snap = runner.engine().snapshot();
    let stats = runner.stats();
    let comp_obs = runner.world().observation();
    let cave_metrics = runner.world().host().cave().metrics();

    write_portal_observations_summary(&cli.state_dir, &snap, &stats, &comp_obs, cave_metrics)?;

    println!("=== a11-live summary (habitat={}) ===", cli.habitat.as_str());
    println!("recovery_outcome: {recovery:?}");
    println!(
        "organism: tick={} atp={:.4} adenosine={:.4} temp={:.4} cells={} tubes={} sensor_pores={} motor_pores={}",
        snap.tick,
        snap.atp,
        snap.adenosine,
        snap.temperature,
        snap.cell_count,
        snap.tube_count,
        snap.sensor_pore_count,
        snap.motor_pore_count,
    );
    Ok(())
}

fn write_portal_observations_summary(
    state_dir: &Path,
    snap: &a11_medium::OrganismSnapshot,
    stats: &a11_portals::LiveHostStats,
    obs: &a11_portals::CompositePortalObservation,
    cave: &a11_worlds::CaveMetrics,
) -> Result<(), LiveHostError> {
    let sp = obs.scratchpad.unwrap_or_default();
    let pb = obs.pattern_board.unwrap_or_default();
    let rp = obs.rhythm_pad.unwrap_or_default();
    let bt = obs.balance_toy.unwrap_or_default();
    let rss = stats.os_rss_bytes.unwrap_or(0);

    let mut json = String::with_capacity(1024);
    let _ = writeln!(
        json,
        "{{\n\
         \x20 \"tick\": {},\n\
         \x20 \"atp\": {},\n\
         \x20 \"cell_count\": {},\n\
         \x20 \"tube_count\": {},\n\
         \x20 \"sensor_pore_count\": {},\n\
         \x20 \"max_tau\": {},\n\
         \x20 \"mitosis_births\": {},\n\
         \x20 \"contradiction_splits\": {},\n\
         \x20 \"used_state_bytes\": {},\n\
         \x20 \"os_rss_bytes\": {},\n\
         \x20 \"p50_step_us\": {},\n\
         \x20 \"p95_step_us\": {},\n\
         \x20 \"p99_step_us\": {},\n\
         \x20 \"cave\": {{\"distance_travelled\": {}, \"springs_depleted\": {}, \"valve_activations\": {}, \"active_pings_emitted\": {}}},\n\
         \x20 \"portal_engaged_ticks\": [{}, {}, {}, {}],\n\
         \x20 \"mean_portal_focus\": [{}, {}, {}, {}],\n\
         \x20 \"scratchpad\": {{\"active_stroke_ticks\": {}, \"canvas_coverage_fraction\": {}, \"spatial_autocorrelation\": {}, \"revisit_ratio\": {}, \"stylus_trajectory_entropy\": {}}},\n\
         \x20 \"pattern_board\": {{\"tiles_inspected\": {}, \"unique_tiles_visited\": {}, \"matched_resonances_locked\": {}, \"repeat_visit_interval_mean\": {}, \"cursor_dwell_stability\": {}}},\n\
         \x20 \"rhythm_pad\": {{\"total_pad_strikes\": {}, \"listening_vs_playing_ratio\": {}, \"phase_locking_value\": {}, \"motif_continuation_similarity\": {}}},\n\
         \x20 \"balance_toy\": {{\"mean_upright_duration_ticks\": {}, \"max_upright_streak\": {}, \"sphere_falls\": {}, \"torque_phase_lead_correlation\": {}}}\n\
         }}",
        snap.tick,
        fmt_f32(snap.atp),
        snap.cell_count,
        snap.tube_count,
        snap.sensor_pore_count,
        fmt_f32(snap.max_tau),
        snap.mitosis_births,
        snap.contradiction_splits,
        stats.last_used_state_bytes,
        rss,
        fmt_f64(stats.p50_step_us),
        fmt_f64(stats.p95_step_us),
        fmt_f64(stats.p99_step_us),
        fmt_f32(cave.distance_travelled),
        cave.springs_depleted,
        cave.valve_activations,
        cave.active_pings_emitted,
        obs.portal_engaged_ticks.first().copied().unwrap_or(0),
        obs.portal_engaged_ticks.get(1).copied().unwrap_or(0),
        obs.portal_engaged_ticks.get(2).copied().unwrap_or(0),
        obs.portal_engaged_ticks.get(3).copied().unwrap_or(0),
        fmt_f32(obs.mean_portal_focus.first().copied().unwrap_or(0.0)),
        fmt_f32(obs.mean_portal_focus.get(1).copied().unwrap_or(0.0)),
        fmt_f32(obs.mean_portal_focus.get(2).copied().unwrap_or(0.0)),
        fmt_f32(obs.mean_portal_focus.get(3).copied().unwrap_or(0.0)),
        sp.active_stroke_ticks,
        fmt_f32(sp.canvas_coverage_fraction),
        fmt_f32(sp.spatial_autocorrelation),
        fmt_f32(sp.revisit_ratio),
        fmt_f32(sp.stylus_trajectory_entropy),
        pb.tiles_inspected,
        pb.unique_tiles_visited,
        pb.matched_resonances_locked,
        fmt_f32(pb.repeat_visit_interval_mean),
        fmt_f32(pb.cursor_dwell_stability),
        rp.total_pad_strikes,
        fmt_f32(rp.listening_vs_playing_ratio),
        fmt_f32(rp.phase_locking_value),
        fmt_f32(rp.motif_continuation_similarity),
        fmt_f32(bt.mean_upright_duration_ticks),
        bt.max_upright_streak,
        bt.sphere_falls,
        fmt_f32(bt.torque_phase_lead_correlation),
    );

    let dst = state_dir.join("portal_observations.json");
    let tmp = state_dir.join("portal_observations.json.tmp");
    write_atomic_bytes(&tmp, &dst, json.as_bytes())
}

// =============================================================================
// Mode 2: MultiGenTerrarium 4-Organism Elder + Younger Cohort (`Alpha11Organism`)
// =============================================================================

struct DecodedPopulation {
    world_tick: u64,
    spawn_ticks: Vec<u64>,
    organisms: Vec<Alpha11Organism>,
}

fn encode_population_checkpoint(
    world_tick: u64,
    spawn_ticks: &[u64],
    organisms: &[Alpha11Organism],
) -> Result<Vec<u8>, PersistError> {
    if organisms.is_empty() || organisms.len() > 16 {
        return Err(PersistError::InvalidTopology);
    }
    let mut buf = Vec::with_capacity(organisms.len() * 64_000);
    buf.extend_from_slice(&POP_MAGIC);
    buf.extend_from_slice(&POP_VERSION.to_le_bytes());
    buf.extend_from_slice(&world_tick.to_le_bytes());
    buf.extend_from_slice(&(organisms.len() as u32).to_le_bytes());

    for (i, org) in organisms.iter().enumerate() {
        if org.is_poisoned() {
            return Err(PersistError::PoisonedAfterCorruption);
        }
        let sp_tick = spawn_ticks.get(i).copied().unwrap_or(0);
        let blob = org.encode_checkpoint()?;
        let slot_start = buf.len();
        buf.extend_from_slice(&sp_tick.to_le_bytes());
        buf.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        buf.extend_from_slice(&blob);
        let slot_crc = crc32c(&buf[slot_start..]);
        buf.extend_from_slice(&slot_crc.to_le_bytes());
    }

    let file_crc = crc32c(&buf);
    buf.extend_from_slice(&file_crc.to_le_bytes());
    if buf.len() > MAX_POP_CHECKPOINT_BYTES {
        return Err(PersistError::OversizedInput);
    }
    Ok(buf)
}

fn shadow_decode_population_checkpoint(bytes: &[u8]) -> Result<DecodedPopulation, PersistError> {
    if bytes.len() > MAX_POP_CHECKPOINT_BYTES {
        return Err(PersistError::OversizedInput);
    }
    if bytes.len() < 28 {
        return Err(PersistError::TruncatedInput);
    }
    if bytes[0..8] != POP_MAGIC {
        return Err(PersistError::BadMagicOrVersion);
    }
    let version = read_u32_le(bytes, 8)?;
    if version != POP_VERSION {
        return Err(PersistError::BadMagicOrVersion);
    }

    let payload_end = bytes.len() - 4;
    let expected_file_crc = read_u32_le(bytes, payload_end)?;
    if crc32c(&bytes[..payload_end]) != expected_file_crc {
        return Err(PersistError::ChecksumMismatch);
    }

    let world_tick = read_u64_le(bytes, 12)?;
    let count = read_u32_le(bytes, 20)? as usize;
    if count == 0 || count > 16 {
        return Err(PersistError::InvalidTopology);
    }

    let mut pos = 24usize;
    let mut spawn_ticks = Vec::with_capacity(count);
    let mut organisms = Vec::with_capacity(count);

    for idx in 0..count {
        if pos.saturating_add(16) > payload_end {
            return Err(PersistError::TruncatedInput);
        }
        let slot_start = pos;
        let sp_tick = read_u64_le(bytes, pos)?;
        let blob_len = read_u32_le(bytes, pos + 8)? as usize;
        pos += 12;

        if pos.saturating_add(blob_len).saturating_add(4) > payload_end {
            return Err(PersistError::TruncatedInput);
        }
        let blob_end = pos + blob_len;
        let expected_slot_crc = read_u32_le(bytes, blob_end)?;
        if crc32c(&bytes[slot_start..blob_end]) != expected_slot_crc {
            return Err(PersistError::ChecksumMismatch);
        }

        let mut candidate = Alpha11Organism::new((idx as u64) + 1, OrganismConfig::default());
        candidate.restore_checkpoint(&bytes[pos..blob_end])?;
        if candidate.is_poisoned() {
            return Err(PersistError::PoisonedAfterCorruption);
        }

        spawn_ticks.push(sp_tick);
        organisms.push(candidate);
        pos = blob_end + 4;
    }

    if pos != payload_end {
        return Err(PersistError::InvalidTopology);
    }

    Ok(DecodedPopulation {
        world_tick,
        spawn_ticks,
        organisms,
    })
}

fn recover_or_init_terrarium_population(
    primary: &Path,
    prev: &Path,
    tmp: &Path,
    seed: u64,
) -> Result<(DecodedPopulation, RecoveryOutcome), LiveHostError> {
    let has_primary = primary.exists();
    let has_prev = prev.exists();

    if !has_primary && !has_prev {
        let elders = vec![
            Alpha11Organism::new(seed ^ 0xE1DE_0001, OrganismConfig::default()),
            Alpha11Organism::new(seed ^ 0xE1DE_0002, OrganismConfig::default()),
        ];
        return Ok((
            DecodedPopulation {
                world_tick: 0,
                spawn_ticks: vec![0, 0],
                organisms: elders,
            },
            RecoveryOutcome::Genesis,
        ));
    }

    let primary_err = if has_primary {
        match read_bounded_file(primary, MAX_POP_CHECKPOINT_BYTES) {
            Ok(bytes) => match shadow_decode_population_checkpoint(&bytes) {
                Ok(pop) => {
                    return Ok((
                        pop,
                        RecoveryOutcome::RestoredPrimary {
                            bytes_len: bytes.len(),
                        },
                    ));
                }
                Err(e) => e,
            },
            Err(e) => e,
        }
    } else {
        PersistError::IoFailure
    };

    if !has_prev {
        return Err(LiveHostError::Persist(primary_err));
    }

    let prev_bytes = read_bounded_file(prev, MAX_POP_CHECKPOINT_BYTES)?;
    let pop = shadow_decode_population_checkpoint(&prev_bytes)?;
    write_atomic_bytes(tmp, primary, &prev_bytes)?;
    Ok((
        pop,
        RecoveryOutcome::RecoveredFromPrev {
            primary_error: primary_err,
            bytes_len: prev_bytes.len(),
        },
    ))
}

fn save_population_transactional(
    primary: &Path,
    prev: &Path,
    prev_tmp: &Path,
    tmp: &Path,
    world_tick: u64,
    spawn_ticks: &[u64],
    organisms: &[Alpha11Organism],
) -> Result<usize, LiveHostError> {
    let bytes = encode_population_checkpoint(world_tick, spawn_ticks, organisms)?;
    shadow_decode_population_checkpoint(&bytes)?;
    write_synced(tmp, &bytes)?;

    if primary.exists()
        && let Ok(existing) = read_bounded_file(primary, MAX_POP_CHECKPOINT_BYTES)
        && shadow_decode_population_checkpoint(&existing).is_ok()
    {
        write_atomic_bytes(prev_tmp, prev, &existing)?;
    }

    fs::rename(tmp, primary)?;
    sync_parent(primary);
    Ok(bytes.len())
}

fn run_terrarium_habitat(cli: &CliArgs) -> Result<(), LiveHostError> {
    fs::create_dir_all(&cli.state_dir)?;

    let primary_ckpt = cli.state_dir.join("terrarium_population.ckpt");
    let prev_ckpt = cli.state_dir.join("terrarium_population.ckpt.prev");
    let prev_tmp_ckpt = cli.state_dir.join("terrarium_population.ckpt.prev.tmp");
    let tmp_ckpt = cli.state_dir.join("terrarium_population.ckpt.tmp");

    let telem_path = cli.state_dir.join("terrarium_telemetry.jsonl");
    let rotated_telem_path = cli.state_dir.join("terrarium_telemetry.jsonl.1");
    let stop_sentinel = cli.state_dir.join("STOP");

    let (decoded, recovery) =
        recover_or_init_terrarium_population(&primary_ckpt, &prev_ckpt, &tmp_ckpt, cli.seed)?;

    let mut cohort = decoded.organisms;
    let mut spawn_ticks = decoded.spawn_ticks;
    let mut terrarium = MultiGenTerrarium::new(cli.seed, 2);

    if cohort.len() > 2 {
        if terrarium.metrics().tick == 0 {
            let silent = vec![FieldPulse::silent(6); 2];
            terrarium.apply_population_pulses(&silent);
        }
        for slot_idx in 2..cohort.len() {
            terrarium.respawn_newborn_in_slot(slot_idx);
        }
    }

    let younger_join_step = if cli.steps > 0 {
        ELDER_HEAD_START_TICKS.min((cli.steps / 4).max(1))
    } else {
        ELDER_HEAD_START_TICKS
    };

    let mut current_log_bytes = fs::metadata(&telem_path).map_or(0, |m| m.len() as usize);
    let mut telemetry_lines = 0u64;
    let mut checkpoints_saved = 0u64;
    let mut latency_ring = Vec::with_capacity(LATENCY_RING_CAP);
    let mut latency_cursor = 0usize;
    let mut steps_done = 0u64;

    loop {
        if cli.steps > 0 && steps_done >= cli.steps {
            break;
        }
        if (steps_done & 63) == 0 && stop_sentinel.exists() {
            break;
        }

        if cohort.len() < 4 && steps_done == younger_join_step {
            let now_tick = decoded.world_tick.saturating_add(steps_done);
            cohort.push(Alpha11Organism::new(
                cli.seed ^ 0x9E0B_0003,
                OrganismConfig::default(),
            ));
            cohort.push(Alpha11Organism::new(
                cli.seed ^ 0x9E0B_0004,
                OrganismConfig::default(),
            ));
            spawn_ticks.push(now_tick);
            spawn_ticks.push(now_tick);
            terrarium.respawn_newborn_in_slot(2);
            terrarium.respawn_newborn_in_slot(3);
        }

        let t0 = Instant::now();
        let _pulses = terrarium.step_population(&mut cohort)?;
        let step_us = t0.elapsed().as_nanos() as f64 / 1_000.0;

        if latency_ring.len() < LATENCY_RING_CAP {
            latency_ring.push(step_us);
        } else {
            latency_ring[latency_cursor] = step_us;
            latency_cursor = (latency_cursor + 1) % LATENCY_RING_CAP;
        }

        steps_done = steps_done.saturating_add(1);
        let logical_world_tick = decoded.world_tick.saturating_add(steps_done);

        if steps_done % TERRARIUM_TELEMETRY_INTERVAL == 0 {
            let line = format_terrarium_jsonl(logical_world_tick, &terrarium, &cohort, step_us);
            let line_bytes = line.as_bytes();
            if current_log_bytes > 0
                && current_log_bytes + line_bytes.len() > TERRARIUM_MAX_LOG_BYTES
            {
                if telem_path.exists() {
                    fs::rename(&telem_path, &rotated_telem_path)?;
                }
                current_log_bytes = 0;
            }
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&telem_path)?;
            f.write_all(line_bytes)?;
            f.flush()?;
            current_log_bytes += line_bytes.len();
            telemetry_lines = telemetry_lines.saturating_add(1);
        }

        if steps_done % TERRARIUM_CHECKPOINT_INTERVAL == 0 {
            let _ = save_population_transactional(
                &primary_ckpt,
                &prev_ckpt,
                &prev_tmp_ckpt,
                &tmp_ckpt,
                logical_world_tick,
                &spawn_ticks,
                &cohort,
            )?;
            checkpoints_saved = checkpoints_saved.saturating_add(1);
        }

        pace_tick(t0, cli.pace_ms);
    }

    let final_world_tick = decoded.world_tick.saturating_add(steps_done);
    if steps_done > 0 && steps_done % TERRARIUM_CHECKPOINT_INTERVAL != 0 {
        let _ = save_population_transactional(
            &primary_ckpt,
            &prev_ckpt,
            &prev_tmp_ckpt,
            &tmp_ckpt,
            final_world_tick,
            &spawn_ticks,
            &cohort,
        )?;
        checkpoints_saved = checkpoints_saved.saturating_add(1);
    }

    let (p50, p95, p99) = percentiles(&latency_ring);
    let rss = read_linux_rss_bytes();
    let tm = terrarium.metrics();

    println!("=== a11-live summary (habitat=terrarium) ===");
    println!("recovery_outcome: {recovery:?}");
    println!(
        "terrarium: world_tick={} cohort_size={} lux={:.3} mark_cov={:.4} p50_us={p50:.2} p95_us={p95:.2} p99_us={p99:.2} rss={rss:?} ckpts={checkpoints_saved} jsonl={telemetry_lines}",
        final_world_tick,
        cohort.len(),
        tm.illumination_lux,
        tm.persistent_mark_coverage,
    );

    Ok(())
}

fn format_terrarium_jsonl(
    world_tick: u64,
    terrarium: &MultiGenTerrarium,
    cohort: &[Alpha11Organism],
    step_us: f64,
) -> String {
    let tm = terrarium.metrics();
    let n = cohort.len().max(1) as f32;

    let mut sum_atp = 0.0_f32;
    let mut total_cells = 0usize;
    let mut total_tubes = 0usize;
    let mut total_sensor_pores = 0usize;
    let mut max_tau = 0.0_f32;
    let mut sum_agency = 0.0_f32;
    let mut total_mitosis = 0u64;
    let mut total_splits = 0u64;
    let mut total_bytes = 0usize;

    for org in cohort {
        let s = org.snapshot();
        sum_atp += s.atp;
        total_cells += s.cell_count;
        total_tubes += s.tube_count;
        total_sensor_pores += s.sensor_pore_count;
        max_tau = max_tau.max(s.max_tau);
        sum_agency += s.mean_interventional_agency;
        total_mitosis = total_mitosis.saturating_add(s.mitosis_births);
        total_splits = total_splits.saturating_add(s.contradiction_splits);
        total_bytes = total_bytes.saturating_add(s.used_state_bytes);
    }

    let mut out = String::with_capacity(512);
    let _ = writeln!(
        out,
        "{{\"tick\":{},\"cohort_size\":{},\"illumination_lux\":{},\"persistent_mark_coverage\":{},\"mean_inter_organism_proximity\":{},\"mean_atp\":{},\"total_cells\":{},\"total_tubes\":{},\"total_sensor_pores\":{},\"max_tau\":{},\"mean_interventional_agency\":{},\"total_mitosis_births\":{},\"total_contradiction_splits\":{},\"used_state_bytes\":{},\"step_us\":{}}}",
        world_tick,
        cohort.len(),
        fmt_f32(tm.illumination_lux),
        fmt_f32(tm.persistent_mark_coverage),
        fmt_f32(tm.mean_inter_organism_proximity),
        fmt_f32(sum_atp / n),
        total_cells,
        total_tubes,
        total_sensor_pores,
        fmt_f32(max_tau),
        fmt_f32(sum_agency / n),
        total_mitosis,
        total_splits,
        total_bytes,
        fmt_f64(step_us),
    );
    out
}

// =============================================================================
// Safe Binary & File Helpers (Zero unwrap/expect)
// =============================================================================

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, PersistError> {
    let slice = bytes
        .get(offset..offset.saturating_add(4))
        .ok_or(PersistError::TruncatedInput)?;
    let arr: [u8; 4] = slice.try_into().map_err(|_| PersistError::TruncatedInput)?;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64, PersistError> {
    let slice = bytes
        .get(offset..offset.saturating_add(8))
        .ok_or(PersistError::TruncatedInput)?;
    let arr: [u8; 8] = slice.try_into().map_err(|_| PersistError::TruncatedInput)?;
    Ok(u64::from_le_bytes(arr))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), LiveHostError> {
    let mut f = File::create(path).map_err(|_: io::Error| LiveHostError::IoFailure)?;
    f.write_all(bytes)
        .map_err(|_: io::Error| LiveHostError::IoFailure)?;
    f.sync_all()
        .map_err(|_: io::Error| LiveHostError::IoFailure)?;
    Ok(())
}

fn write_atomic_bytes(tmp: &Path, dst: &Path, bytes: &[u8]) -> Result<(), LiveHostError> {
    write_synced(tmp, bytes)?;
    fs::rename(tmp, dst).map_err(|_: io::Error| LiveHostError::IoFailure)?;
    sync_parent(dst);
    Ok(())
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent()
        && let Ok(dir) = File::open(parent)
    {
        let _ = dir.sync_all();
    }
}

fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "0.000000".to_string()
    }
}

fn fmt_f64(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "0.000".to_string()
    }
}

fn percentiles(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let idx = |q: f64| -> usize { ((q * (n - 1) as f64).round() as usize).min(n - 1) };
    (sorted[idx(0.50)], sorted[idx(0.95)], sorted[idx(0.99)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cli_parser_handles_defaults_open_and_pace_ms_cleanly() {
        let def = parse_cli_args(Vec::<&str>::new());
        assert_eq!(def, Ok(CliArgs::default()));
        assert_eq!(CliArgs::default().habitat, HabitatKind::Open);

        let custom = parse_cli_args([
            "--state-dir",
            "/tmp/a11_custom",
            "--seed",
            "42",
            "--steps",
            "1500",
            "--pace-ms",
            "20",
            "--habitat",
            "open",
        ]);
        assert_eq!(
            custom,
            Ok(CliArgs {
                state_dir: PathBuf::from("/tmp/a11_custom"),
                seed: 42,
                steps: 1500,
                pace_ms: 20,
                habitat: HabitatKind::Open,
                show_help: false,
            })
        );

        assert!(parse_cli_args(["--habitat", "mars"]).is_err());
        assert!(parse_cli_args(["--seed", "not_a_number"]).is_err());
        assert!(parse_cli_args(["--pace-ms", "bad"]).is_err());
        assert!(parse_cli_args(["--unknown"]).is_err());
    }

    #[test]
    fn population_checkpoint_recovers_from_prev_and_fails_closed_when_both_corrupt() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(1);
        let dir =
            std::env::temp_dir().join(format!("a11_live_pop_test_{}_{nanos}", std::process::id()));
        let _ = fs::create_dir_all(&dir);

        let primary = dir.join("pop.ckpt");
        let prev = dir.join("pop.ckpt.prev");
        let prev_tmp = dir.join("pop.ckpt.prev.tmp");
        let tmp = dir.join("pop.ckpt.tmp");

        let orgs = vec![
            Alpha11Organism::new(101, OrganismConfig::default()),
            Alpha11Organism::new(102, OrganismConfig::default()),
        ];

        assert!(
            save_population_transactional(&primary, &prev, &prev_tmp, &tmp, 100, &[0, 0], &orgs)
                .is_ok()
        );
        assert!(
            save_population_transactional(&primary, &prev, &prev_tmp, &tmp, 200, &[0, 0], &orgs)
                .is_ok()
        );

        let mut raw = fs::read(&primary).unwrap_or_default();
        raw[14] ^= 0xFF;
        let _ = fs::write(&primary, &raw);

        let recovered = recover_or_init_terrarium_population(&primary, &prev, &tmp, 11);
        assert!(recovered.is_ok());
        if let Ok((pop, outcome)) = recovered {
            assert!(matches!(outcome, RecoveryOutcome::RecoveredFromPrev { .. }));
            assert_eq!(pop.world_tick, 100);
            assert_eq!(pop.organisms.len(), 2);
        }

        let _ = fs::write(&primary, b"bad");
        let _ = fs::write(&prev, b"bad_prev");
        assert!(recover_or_init_terrarium_population(&primary, &prev, &tmp, 11).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupted_primary_is_never_rotated_over_valid_prev() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(1);
        let dir =
            std::env::temp_dir().join(format!("a11_live_rot_test_{}_{nanos}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let primary = dir.join("pop.ckpt");
        let prev = dir.join("pop.ckpt.prev");
        let prev_tmp = dir.join("pop.ckpt.prev.tmp");
        let tmp = dir.join("pop.ckpt.tmp");
        let orgs = vec![
            Alpha11Organism::new(201, OrganismConfig::default()),
            Alpha11Organism::new(202, OrganismConfig::default()),
        ];

        assert!(
            save_population_transactional(&primary, &prev, &prev_tmp, &tmp, 100, &[0, 0], &orgs)
                .is_ok()
        );
        assert!(
            save_population_transactional(&primary, &prev, &prev_tmp, &tmp, 200, &[0, 0], &orgs)
                .is_ok()
        );
        let good_prev = fs::read(&prev).unwrap_or_default();

        let mut raw = fs::read(&primary).unwrap_or_default();
        raw[20] ^= 0x5A;
        let _ = fs::write(&primary, &raw);
        assert!(
            save_population_transactional(&primary, &prev, &prev_tmp, &tmp, 300, &[0, 0], &orgs)
                .is_ok()
        );

        assert_eq!(fs::read(&prev).unwrap_or_default(), good_prev);
        let prev_bytes = fs::read(&prev).unwrap_or_default();
        let decoded = shadow_decode_population_checkpoint(&prev_bytes);
        assert!(matches!(decoded, Ok(ref p) if p.world_tick == 100));
        let prim_bytes = fs::read(&primary).unwrap_or_default();
        assert!(
            matches!(shadow_decode_population_checkpoint(&prim_bytes), Ok(ref p) if p.world_tick == 300)
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
