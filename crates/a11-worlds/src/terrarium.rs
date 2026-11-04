//! Open Multi-Organism Continuous Habitat (`MultiGenTerrarium`) with Day/Night Cycles,
//! Thermal Shelters, Pushable Warm Matter, and Persistent Cultural Surface Marks (`16x16`).

use std::f32::consts::TAU;
use std::ops::Range;

use a11_medium::{FieldPulse, FieldView, OrganismEngine, PersistError, WorldMedium};

use crate::rng::{DeterministicRng, sanitize_actuator};
use crate::wavefield::{
    DEFAULT_MAX_ECHO_RANGE, DEFAULT_WAVE_SPEED, EMIT_PULSE_THRESHOLD, Reflector, ReflectorKind,
    WaveEchoRing,
};

/// Maximum simultaneous organisms supported in `MultiGenTerrarium`.
pub const MAX_TERRARIUM_ORGANISMS: usize = 16;

/// Ambient channels presented to each organism (`0..64` photonic, `64..96` echo, `96..128` Noisy TV).
pub const TERRARIUM_CHANNELS: usize = 128;

/// Continuous actuators per organism (`[fx, fy, gx, gy, emit_pulse, emit_tone]`).
pub const TERRARIUM_ACTUATORS: usize = 6;

/// Spatial resolution of the persistent externalized surface-mark grid (`16x16`).
pub const MARK_GRID_DIM: usize = 16;

/// Per-tick multiplicative retention of persistent terrain marks (`~385`-tick half-life).
pub const MARK_DECAY_PER_TICK: f32 = 0.9982;

/// Default full day/night oscillation period in ticks.
pub const DEFAULT_DAY_NIGHT_PERIOD: u64 = 320;

/// Acoustic frame length (`20 ms @ 16 kHz = 320` samples) for Earth soundscape synthesis.
pub const EARTH_AUDIO_FRAME_SAMPLES: usize = 320;

/// Seasonal watershed flood/drought cycle period in ticks (`3` day/night cycles).
pub const EARTH_SEASON_PERIOD_TICKS: u64 = 960;

const TERRARIUM_RADIUS: f32 = 20.0;
const AGENT_RADIUS: f32 = 0.58;
const MAX_THRUST: f32 = 0.22;
const DRAG: f32 = 0.82;
const PHOTONIC_ROWS: usize = 8;
const PHOTONIC_COLS: usize = 8;
const PHOTONIC_END: usize = 64;
const ECHO_END: usize = 96;
const MAX_WATER_SOURCES: usize = 8;
const MAX_IMPACT_SLOTS: usize = 4;
const MAX_EARTH_ITD_SAMPLES: usize = 6;

/// Warm geothermal shelter zone that buffers cold-night tissue stress.
#[derive(Clone, Copy, Debug)]
pub struct ThermalShelter {
    pub pos: (f32, f32),
    pub radius: f32,
}

/// Pushable warm mineral block in the shared habitat.
#[derive(Clone, Copy, Debug)]
pub struct PushableMatter {
    pub pos: (f32, f32),
    pub vel: (f32, f32),
    pub radius: f32,
    pub warmth_radius: f32,
}

/// Regenerating nutrient spring in the shared habitat.
#[derive(Clone, Copy, Debug)]
pub struct TerrariumSpring {
    pub pos: (f32, f32),
    pub reservoir: f32,
    pub core_radius: f32,
    pub halo_radius: f32,
}

/// Opaque solid basalt canyon ridge blocking visual line-of-sight (`0.0` light transmittance)
/// while permitting `16 kHz` acoustic diffraction (`0.68`) over/around the rock crest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasaltRidge {
    pub a: (f32, f32),
    pub b: (f32, f32),
    pub thickness: f32,
}

/// Hard-shelled botanical forage pod requiring manual hand dexterity or stone-anvil impact to crack.
///
/// Walking over an intact pod (`shell_integrity > 0.0`) yields strictly `0.0` `metabolic_influx`.
/// When `heavy_shell` is `true`, a single organism prying alone cannot crack the shell (`>= 0.22` floor,
/// emitting woody `16 kHz` taps); `>= 2` organisms working together within `2.8 m` (or a boulder impact)
/// crack it open into a rich `1.85x` shared nutrient pool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForagePod {
    pub pos: (f32, f32),
    pub ripeness: f32,
    pub shell_integrity: f32,
    pub open_nutrients: f32,
    pub heavy_shell: bool,
}

/// Read-only ecological & cooperative metrics for `MultiGenTerrarium` in Earth-ecology mode.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EarthEcologyMetrics {
    pub enabled: bool,
    pub season_phase: f32,
    pub flowing_springs: usize,
    pub mean_spring_flow: f32,
    pub ripe_intact_pods: usize,
    pub open_cracked_pods: usize,
    pub pods_cracked_by_hands: u64,
    pub pods_cracked_by_stone: u64,
    pub coop_grove_pods_cracked: u64,
    pub cooperative_boulder_ticks: u64,
    pub huddling_ticks: u64,
    pub acoustic_impact_events: u64,
    pub visual_occlusions_active: usize,
    pub occluded_vocal_approaches: u64,
    pub agent_pods_cracked: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct AcousticImpactSlot {
    pos: (f32, f32),
    amp_a: f32,
    amp_b: f32,
    phase_a: f32,
    phase_b: f32,
    omega_a: f32,
    omega_b: f32,
    decay_per_sample: f32,
    ticks_left: u8,
}

/// Read-only external observer metrics for `MultiGenTerrarium` (never leaked into `FieldView`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrariumMetrics {
    pub tick: u64,
    pub illumination_lux: f32,
    pub day_transitions: u64,
    pub night_transitions: u64,
    pub agent_metabolic_influx: Vec<f32>,
    pub agent_tissue_stress: Vec<f32>,
    pub agent_positions: Vec<(f32, f32)>,
    pub agent_ages: Vec<u64>,
    /// Fraction of `16x16` terrain mark cells with density `>= 0.05`.
    pub persistent_mark_coverage: f32,
    /// Sum of mark intensity across all `256` terrain cells.
    pub total_mark_mass: f32,
    /// Fraction of total mark mass aligned along nutrient-spring <-> thermal-shelter corridors (`[0.0, 1.0]`).
    pub mark_corridor_alignment: f32,
    /// Smooth pairwise proximity `mean_{i<j} 1 / (1 + 0.2 * dist_ij)` in `[0.0, 1.0]`.
    pub mean_inter_organism_proximity: f32,
    /// Mean Euclidean distance across organism pairs (`0.0` if `< 2` organisms).
    pub mean_pairwise_distance: f32,
    /// Exponentially smoothed acoustic call/echo synchrony between elders and younger peers (`[0.0, 1.0]`).
    pub elder_newborn_acoustic_synchrony: f32,
    /// Exponentially smoothed velocity/motor alignment between elders and younger peers (`[-1.0, 1.0]`).
    pub elder_newborn_motor_synchrony: f32,
}

impl TerrariumMetrics {
    /// Canonical channel band ranges `(Photonic, Echo, NoisyTV)`.
    #[must_use]
    pub fn channel_band_ranges() -> (Range<usize>, Range<usize>, Range<usize>) {
        (0..PHOTONIC_END, PHOTONIC_END..ECHO_END, ECHO_END..TERRARIUM_CHANNELS)
    }
}

#[derive(Clone, Debug)]
struct AgentSlot {
    pos: (f32, f32),
    vel: (f32, f32),
    spawn_tick: u64,
    warmth_nutrients_carried: f32,
    recent_nourishment_trace: f32,
    last_ping_strength: f32,
    recent_acoustic_activity: f32,
    echo_ring: WaveEchoRing,
    ambient: Vec<f32>,
    last_influx: f32,
    last_stress: f32,
    cumulative_influx: f32,
    cumulative_stress: f32,
    embodied_stability: f32,
    embodied_hand_work: f32,
    embodied_voice_rms: f32,
    last_lateral_sway: f32,
    pods_cracked: u64,
}

impl AgentSlot {
    fn new(pos: (f32, f32), spawn_tick: u64) -> Self {
        Self {
            pos,
            vel: (0.0, 0.0),
            spawn_tick,
            warmth_nutrients_carried: 0.30,
            recent_nourishment_trace: 0.25,
            last_ping_strength: 0.0,
            recent_acoustic_activity: 0.0,
            echo_ring: WaveEchoRing::new(
                ECHO_END - PHOTONIC_END,
                DEFAULT_WAVE_SPEED,
                DEFAULT_MAX_ECHO_RANGE,
            ),
            ambient: vec![0.0; TERRARIUM_CHANNELS],
            last_influx: 0.0,
            last_stress: 0.0,
            cumulative_influx: 0.0,
            cumulative_stress: 0.0,
            embodied_stability: 1.0,
            embodied_hand_work: 0.0,
            embodied_voice_rms: 0.0,
            last_lateral_sway: 0.0,
            pods_cracked: 0,
        }
    }
}

/// Shared multi-organism continuous 2D habitat with day/night cycles and durable terrain marks.
#[derive(Clone, Debug)]
pub struct MultiGenTerrarium {
    arena_radius: f32,
    day_night_period: u64,
    illumination_lux: f32,
    was_night: bool,
    earth_ecology: bool,

    springs: Vec<TerrariumSpring>,
    spring_phase_offsets: Vec<f32>,
    spring_max_flows: Vec<f32>,
    shelters: Vec<ThermalShelter>,
    matter: Vec<PushableMatter>,
    ridges: Vec<BasaltRidge>,
    pods: Vec<ForagePod>,

    /// Continuous `16x16` persistent surface-mark scalar field in `[0.0, 1.0]`.
    surface_marks: [[f32; MARK_GRID_DIM]; MARK_GRID_DIM],
    /// Precomputed corridor alignment weight in `[0.0, 1.0]` for each `16x16` cell.
    corridor_weights: [[f32; MARK_GRID_DIM]; MARK_GRID_DIM],

    col_cos: [f32; PHOTONIC_COLS],
    col_sin: [f32; PHOTONIC_COLS],
    row_depths: [f32; PHOTONIC_ROWS],

    water_resonators: [(f32, f32); MAX_WATER_SOURCES],
    water_frames: [[f32; EARTH_AUDIO_FRAME_SAMPLES]; MAX_WATER_SOURCES],
    impact_events: [AcousticImpactSlot; MAX_IMPACT_SLOTS],
    impact_frames: [[f32; EARTH_AUDIO_FRAME_SAMPLES]; MAX_IMPACT_SLOTS],

    agents: Vec<AgentSlot>,
    world_rng: DeterministicRng,
    noisy_rng: DeterministicRng,
    acoustic_rng: DeterministicRng,
    metrics: TerrariumMetrics,
    ecology_metrics: EarthEcologyMetrics,
}

impl MultiGenTerrarium {
    /// Creates a new `MultiGenTerrarium` initialized for `initial_agents` (`1..=16`).
    #[must_use]
    pub fn new(seed: u64, initial_agents: usize) -> Self {
        Self::with_day_night_period(seed, initial_agents, DEFAULT_DAY_NIGHT_PERIOD)
    }

    /// Creates a `MultiGenTerrarium` with a custom `day_night_period` in ticks (`>= 16`).
    #[must_use]
    pub fn with_day_night_period(seed: u64, initial_agents: usize, day_night_period: u64) -> Self {
        let mut master = DeterministicRng::new(seed);
        let mut world_rng = master.fork(0xA11_7E88);
        let noisy_rng = master.fork(0xA11_9015);
        let acoustic_rng = master.fork(0xA11_EA87);

        let shelters = vec![
            ThermalShelter {
                pos: (-4.5, 0.0),
                radius: 3.2,
            },
            ThermalShelter {
                pos: (4.5, 0.0),
                radius: 3.2,
            },
        ];

        let mut springs = Vec::with_capacity(4);
        let spring_anchors = [(-10.5, 6.5), (-10.0, -6.8), (10.5, 6.8), (10.2, -6.5)];
        for &(bx, by) in &spring_anchors {
            springs.push(TerrariumSpring {
                pos: (
                    bx + world_rng.next_f32_range(-1.0, 1.0),
                    by + world_rng.next_f32_range(-1.0, 1.0),
                ),
                reservoir: 1.0,
                core_radius: 2.0,
                halo_radius: 4.6,
            });
        }
        let spring_phase_offsets = vec![0.0; springs.len()];
        let spring_max_flows = vec![1.0; springs.len()];

        let mut matter = Vec::with_capacity(3);
        for i in 0..3 {
            let ang = (i as f32) * TAU / 3.0 + 0.5;
            matter.push(PushableMatter {
                pos: (3.2 * ang.cos(), 3.2 * ang.sin()),
                vel: (0.0, 0.0),
                radius: 0.85,
                warmth_radius: 2.6,
            });
        }

        let corridor_weights = compute_corridor_weight_grid(&shelters, &springs, TERRARIUM_RADIUS);

        let col_cos = std::array::from_fn(|c| (((c as f32) + 0.5) * TAU / 8.0).cos());
        let col_sin = std::array::from_fn(|c| (((c as f32) + 0.5) * TAU / 8.0).sin());
        let row_depths = std::array::from_fn(|r| {
            let frac = (r as f32) / 7.0;
            1.6 + 16.0 * frac * (0.65 + 0.35 * frac)
        });

        let clamped_agents = initial_agents.clamp(1, MAX_TERRARIUM_ORGANISMS);
        let mut agents = Vec::with_capacity(clamped_agents);
        for idx in 0..clamped_agents {
            let shelter = &shelters[idx % shelters.len()];
            let ang = (idx as f32) * TAU / (clamped_agents.max(1) as f32);
            let spawn_pos = (
                shelter.pos.0 + 1.1 * ang.cos(),
                shelter.pos.1 + 1.1 * ang.sin(),
            );
            agents.push(AgentSlot::new(spawn_pos, 0));
        }

        let period = day_night_period.max(16);
        let lux = compute_day_night_lux(0, period);

        let mut terrarium = Self {
            arena_radius: TERRARIUM_RADIUS,
            day_night_period: period,
            illumination_lux: lux,
            was_night: lux <= 0.01,
            earth_ecology: false,
            springs,
            spring_phase_offsets,
            spring_max_flows,
            shelters,
            matter,
            ridges: Vec::new(),
            pods: Vec::new(),
            surface_marks: [[0.0; MARK_GRID_DIM]; MARK_GRID_DIM],
            corridor_weights,
            col_cos,
            col_sin,
            row_depths,
            water_resonators: [(0.0, 0.0); MAX_WATER_SOURCES],
            water_frames: [[0.0; EARTH_AUDIO_FRAME_SAMPLES]; MAX_WATER_SOURCES],
            impact_events: [AcousticImpactSlot::default(); MAX_IMPACT_SLOTS],
            impact_frames: [[0.0; EARTH_AUDIO_FRAME_SAMPLES]; MAX_IMPACT_SLOTS],
            agents,
            world_rng,
            noisy_rng,
            acoustic_rng,
            metrics: TerrariumMetrics {
                illumination_lux: lux,
                ..TerrariumMetrics::default()
            },
            ecology_metrics: EarthEcologyMetrics::default(),
        };

        terrarium.refresh_all_agent_ambients();
        terrarium.sync_metrics_snapshot();
        terrarium
    }

    /// Creates a rich Earth-like ecological habitat (`earth_ecology = true`) with:
    /// - Seasonal hydro-watershed springs + valley meadow seeps that murmur in `16 kHz` stereo,
    /// - Hard-shelled botanical `ForagePod`s requiring manual hand work or stone-anvil impact to crack,
    /// - Cooperative heavy warm hearth boulders (`2.6x` joint mechanical advantage) & `16 kHz` impacts,
    /// - Resonant geothermal cave shelters (`40 ms` vocal/formant reflections) and postural traction.
    #[must_use]
    pub fn new_earth_ecology(seed: u64, initial_agents: usize) -> Self {
        let mut master = DeterministicRng::new(seed);
        let mut world_rng = master.fork(0xA11_7E88);
        let noisy_rng = master.fork(0xA11_9015);
        let acoustic_rng = master.fork(0xA11_EA87);

        let shelters = vec![
            ThermalShelter {
                pos: (-4.2, 0.0),
                radius: 3.5,
            },
            ThermalShelter {
                pos: (4.2, 0.0),
                radius: 3.5,
            },
        ];

        let mut springs = Vec::with_capacity(6);
        let mut spring_phase_offsets = Vec::with_capacity(6);
        let mut spring_max_flows = Vec::with_capacity(6);

        let watershed_anchors = [(-10.8, 6.8), (-10.6, -6.8), (10.8, 6.8), (10.6, -6.8)];
        for (idx, &(bx, by)) in watershed_anchors.iter().enumerate() {
            springs.push(TerrariumSpring {
                pos: (
                    bx + world_rng.next_f32_range(-0.8, 0.8),
                    by + world_rng.next_f32_range(-0.8, 0.8),
                ),
                reservoir: 0.92,
                core_radius: 2.2,
                halo_radius: 4.6,
            });
            spring_phase_offsets.push((idx as f32) * TAU / 4.0);
            spring_max_flows.push(1.0);
        }

        // Two outer foothill seeps (require genuine navigation from central shelters; deplete if camped).
        let valley_seeps = [(0.0, 11.4), (0.0, -11.4)];
        for (idx, &(vx, vy)) in valley_seeps.iter().enumerate() {
            springs.push(TerrariumSpring {
                pos: (
                    vx + world_rng.next_f32_range(-0.5, 0.5),
                    vy + world_rng.next_f32_range(-0.5, 0.5),
                ),
                reservoir: 0.68,
                core_radius: 2.0,
                halo_radius: 4.2,
            });
            spring_phase_offsets.push(std::f32::consts::FRAC_PI_2 + (idx as f32) * std::f32::consts::PI);
            spring_max_flows.push(0.72);
        }

        let mut matter = Vec::with_capacity(3);
        for i in 0..3 {
            let ang = (i as f32) * TAU / 3.0 + 0.5;
            matter.push(PushableMatter {
                pos: (4.8 * ang.cos(), 4.8 * ang.sin()),
                vel: (0.0, 0.0),
                radius: 0.85,
                warmth_radius: 3.2,
            });
        }

        // Opaque basalt canyon ridges guarding West and East hidden groves while preserving
        // clear Shelter <-> Watershed Spring highways in the central valley.
        let ridges = vec![
            BasaltRidge {
                a: (-11.6, -2.6),
                b: (-11.6, 2.6),
                thickness: 0.52,
            },
            BasaltRidge {
                a: (11.6, -2.6),
                b: (11.6, 2.6),
                thickness: 0.52,
            },
        ];

        // 6 solo valley pods + 4 heavy-shelled cooperative grove pods tucked behind the basalt ridges.
        let solo_pod_anchors = [
            (-9.2, 3.6),
            (-9.2, -3.6),
            (9.2, 3.6),
            (9.2, -3.6),
            (-3.2, 9.8),
            (3.2, -9.8),
        ];
        let mut pods = Vec::with_capacity(solo_pod_anchors.len() + 4);
        for (idx, &(px, py)) in solo_pod_anchors.iter().enumerate() {
            pods.push(ForagePod {
                pos: (
                    px + world_rng.next_f32_range(-0.45, 0.45),
                    py + world_rng.next_f32_range(-0.45, 0.45),
                ),
                ripeness: (0.48 + 0.09 * (idx as f32)).min(0.96),
                shell_integrity: 1.0,
                open_nutrients: 0.0,
                heavy_shell: false,
            });
        }
        let mut grove_rng = master.fork(0xA11_6807);
        let grove_pod_anchors = [(-14.4, -1.2), (-14.4, 1.2), (14.4, -1.2), (14.4, 1.2)];
        for (idx, &(gx, gy)) in grove_pod_anchors.iter().enumerate() {
            pods.push(ForagePod {
                pos: (
                    gx + grove_rng.next_f32_range(-0.30, 0.30),
                    gy + grove_rng.next_f32_range(-0.30, 0.30),
                ),
                ripeness: (0.68 + 0.06 * (idx as f32)).min(0.96),
                shell_integrity: 1.0,
                open_nutrients: 0.0,
                heavy_shell: true,
            });
        }

        let corridor_weights = compute_corridor_weight_grid(&shelters, &springs, TERRARIUM_RADIUS);
        let col_cos = std::array::from_fn(|c| (((c as f32) + 0.5) * TAU / 8.0).cos());
        let col_sin = std::array::from_fn(|c| (((c as f32) + 0.5) * TAU / 8.0).sin());
        let row_depths = std::array::from_fn(|r| {
            let frac = (r as f32) / 7.0;
            1.6 + 16.0 * frac * (0.65 + 0.35 * frac)
        });

        let clamped_agents = initial_agents.clamp(1, MAX_TERRARIUM_ORGANISMS);
        let mut agents = Vec::with_capacity(clamped_agents);
        for idx in 0..clamped_agents {
            let shelter = &shelters[idx % shelters.len()];
            let ang = (idx as f32) * TAU / (clamped_agents.max(1) as f32);
            let spawn_pos = (
                shelter.pos.0 + 1.0 * ang.cos(),
                shelter.pos.1 + 1.0 * ang.sin(),
            );
            agents.push(AgentSlot::new(spawn_pos, 0));
        }

        let period = DEFAULT_DAY_NIGHT_PERIOD;
        let lux = compute_day_night_lux(0, period);

        let mut terrarium = Self {
            arena_radius: TERRARIUM_RADIUS,
            day_night_period: period,
            illumination_lux: lux,
            was_night: lux <= 0.01,
            earth_ecology: true,
            springs,
            spring_phase_offsets,
            spring_max_flows,
            shelters,
            matter,
            ridges,
            pods,
            surface_marks: [[0.0; MARK_GRID_DIM]; MARK_GRID_DIM],
            corridor_weights,
            col_cos,
            col_sin,
            row_depths,
            water_resonators: [(0.0, 0.0); MAX_WATER_SOURCES],
            water_frames: [[0.0; EARTH_AUDIO_FRAME_SAMPLES]; MAX_WATER_SOURCES],
            impact_events: [AcousticImpactSlot::default(); MAX_IMPACT_SLOTS],
            impact_frames: [[0.0; EARTH_AUDIO_FRAME_SAMPLES]; MAX_IMPACT_SLOTS],
            agents,
            world_rng,
            noisy_rng,
            acoustic_rng,
            metrics: TerrariumMetrics {
                illumination_lux: lux,
                ..TerrariumMetrics::default()
            },
            ecology_metrics: EarthEcologyMetrics {
                enabled: true,
                ..EarthEcologyMetrics::default()
            },
        };

        terrarium.step_earth_acoustics_16khz();
        terrarium.refresh_all_agent_ambients();
        terrarium.sync_metrics_snapshot();
        terrarium
    }

    /// Spawns or resets an organism body slot at a thermal shelter at the current tick
    /// (representing a newborn 32-cell seed entering an elder-inhabited terrarium).
    ///
    /// Persistent `16x16` world marks, elders, and day/night phase remain intact!
    pub fn respawn_newborn_in_slot(&mut self, slot_idx: usize) {
        if slot_idx >= MAX_TERRARIUM_ORGANISMS {
            return;
        }
        let tick = self.metrics.tick;
        let shelter = self.shelters[slot_idx % self.shelters.len()];
        let ang = self.world_rng.next_f32_range(0.0, TAU);
        let pos = (
            shelter.pos.0 + 1.0 * ang.cos(),
            shelter.pos.1 + 1.0 * ang.sin(),
        );
        while self.agents.len() <= slot_idx {
            self.agents.push(AgentSlot::new(pos, tick));
        }
        self.agents[slot_idx] = AgentSlot::new(pos, tick);
        for idx in 0..self.agents.len() {
            self.refresh_agent_photonic(idx);
        }
        self.refresh_agent_noisy_tv(slot_idx);
        self.sync_metrics_snapshot();
    }

    /// Spawns or resets an organism body slot at an explicit open-arena coordinate `(x, y)`
    /// (clamped within `arena_radius - AGENT_RADIUS`) at the current tick.
    pub fn respawn_at_position(&mut self, slot_idx: usize, pos: (f32, f32)) {
        if slot_idx >= MAX_TERRARIUM_ORGANISMS {
            return;
        }
        let tick = self.metrics.tick;
        let max_r = (self.arena_radius - AGENT_RADIUS).max(1.0);
        let r = pos.0.hypot(pos.1);
        let clamped_pos = if r > max_r && r > 1.0e-5 {
            (pos.0 * max_r / r, pos.1 * max_r / r)
        } else {
            pos
        };
        while self.agents.len() <= slot_idx {
            self.agents.push(AgentSlot::new(clamped_pos, tick));
        }
        self.agents[slot_idx] = AgentSlot::new(clamped_pos, tick);
        for idx in 0..self.agents.len() {
            self.refresh_agent_photonic(idx);
        }
        self.refresh_agent_noisy_tv(slot_idx);
        self.sync_metrics_snapshot();
    }

    /// Steps a population of `1..=16` [`OrganismEngine`] instances by one shared physics tick.
    /// Environment-side provisioning (a caregiver bringing cracked food): relocates the first
    /// intact, closed pod to `agent_pos + offset` (clamped inside the arena) and opens it with
    /// `nutrients` of harvestable food. Returns `false` when the agent or a closed pod is missing.
    /// Organisms only ever experience the result through ordinary senses and `metabolic_influx`.
    pub fn provision_open_pod_near(
        &mut self,
        agent_idx: usize,
        offset: (f32, f32),
        nutrients: f32,
    ) -> bool {
        let Some(agent) = self.agents.get(agent_idx) else {
            return false;
        };
        let target = (agent.pos.0 + offset.0, agent.pos.1 + offset.1);
        let max_r = (self.arena_radius - AGENT_RADIUS).max(1.0);
        let r = target.0.hypot(target.1);
        let target = if r > max_r && r > 1.0e-5 {
            (target.0 * max_r / r, target.1 * max_r / r)
        } else {
            target
        };
        let Some(pod) = self
            .pods
            .iter_mut()
            .find(|p| p.shell_integrity > 0.0 && p.open_nutrients <= 0.02)
        else {
            return false;
        };
        pod.pos = target;
        pod.ripeness = 1.0;
        pod.shell_integrity = 0.0;
        pod.open_nutrients = nutrients.clamp(0.0, 1.0);
        true
    }

    pub fn step_population<O: OrganismEngine>(
        &mut self,
        organisms: &mut [O],
    ) -> Result<Vec<FieldPulse>, PersistError> {
        let count = organisms.len().min(MAX_TERRARIUM_ORGANISMS);
        if count == 0 {
            return Ok(Vec::new());
        }

        // Dynamically grow agent body slots if new organisms joined the habitat.
        while self.agents.len() < count {
            let new_idx = self.agents.len();
            self.respawn_newborn_in_slot(new_idx);
        }

        let mut pulses = Vec::with_capacity(count);
        for (idx, org) in organisms.iter_mut().take(count).enumerate() {
            let view = self.agent_view(idx);
            let pulse = org.step(view)?;
            pulses.push(pulse);
        }

        self.apply_population_pulses(&pulses);
        Ok(pulses)
    }

    /// Returns the current [`FieldView`] for organism `agent_idx`.
    #[must_use]
    pub fn agent_view(&self, agent_idx: usize) -> FieldView<'_> {
        let slot = &self.agents[agent_idx.min(self.agents.len().saturating_sub(1))];
        FieldView {
            ambient_in: &slot.ambient,
            metabolic_influx: slot.last_influx,
            tissue_stress: slot.last_stress,
            num_actuators: TERRARIUM_ACTUATORS,
        }
    }

    /// Advances shared habitat physics for one tick given per-agent [`FieldPulse`] outputs.
    pub fn apply_population_pulses(&mut self, pulses: &[FieldPulse]) {
        let active_count = pulses.len().min(self.agents.len()).min(MAX_TERRARIUM_ORGANISMS);
        let emit_tick = self.metrics.tick;

        // 1. Collect shared static + dynamic world reflectors.
        let base_reflectors = self.collect_world_reflectors();
        let peer_positions: Vec<(f32, f32)> =
            self.agents.iter().take(active_count).map(|a| a.pos).collect();

        // 2. Process wave emissions & peer-to-peer acoustic propagation BEFORE advancing tick.
        let mut emitted_pulses = vec![(0.0_f32, 0.0_f32); active_count];
        for i in 0..active_count {
            let p_val = sanitize_actuator(pulses[i].actuator_out.get(4).copied());
            let t_val = sanitize_actuator(pulses[i].actuator_out.get(5).copied());
            emitted_pulses[i] = (p_val, t_val);

            let mut refls = base_reflectors.clone();
            for (j, &peer_pos) in peer_positions.iter().enumerate() {
                if j != i {
                    refls.push(Reflector {
                        pos: peer_pos,
                        reflectivity: 0.9,
                        kind: ReflectorKind::PeerOrganism,
                    });
                }
            }
            let r = peer_positions[i].0.hypot(peer_positions[i].1);
            let wall_dist = (self.arena_radius - r).max(0.0);
            if wall_dist <= DEFAULT_MAX_ECHO_RANGE {
                let wall_pos = if r > 1e-4 {
                    (
                        self.arena_radius * peer_positions[i].0 / r,
                        self.arena_radius * peer_positions[i].1 / r,
                    )
                } else {
                    (self.arena_radius, 0.0)
                };
                refls.push(Reflector {
                    pos: wall_pos,
                    reflectivity: 0.88,
                    kind: ReflectorKind::ArenaWall,
                });
            }

            let did_emit =
                self.agents[i]
                    .echo_ring
                    .emit_pulse(emit_tick, peer_positions[i], p_val, t_val, &refls);
            self.agents[i].last_ping_strength = if did_emit { p_val.abs() } else { 0.0 };
            self.agents[i].recent_acoustic_activity = 0.85 * self.agents[i].recent_acoustic_activity
                + 0.15 * self.agents[i].last_ping_strength;
        }

        // Propagate one-way acoustic calls from emitter `i` to distinct peers `j != i`.
        for i in 0..active_count {
            let (p_val, t_val) = emitted_pulses[i];
            if p_val.abs() <= EMIT_PULSE_THRESHOLD {
                continue;
            }
            for j in 0..active_count {
                if i == j {
                    continue;
                }
                let diff = self.acoustic_transmittance(peer_positions[i], peer_positions[j]);
                self.agents[j].echo_ring.enqueue_peer_call(
                    emit_tick,
                    peer_positions[i],
                    peer_positions[j],
                    p_val * diff,
                    t_val,
                );
            }
        }

        // 3. Decay persistent surface marks (`0.9982` per tick).
        for row in &mut self.surface_marks {
            for cell in row.iter_mut() {
                *cell *= MARK_DECAY_PER_TICK;
            }
        }

        // 4. Move agents, push warm mineral blocks, and compute somatic fluxes & trail marks.
        let wall_limit = self.arena_radius - AGENT_RADIUS;
        let mut tick_stresses = vec![0.0_f32; active_count];
        let mut mark_forces = vec![0.0_f32; active_count];
        let mut boulder_pushers = [0_usize; 8];
        if self.earth_ecology {
            for (b_idx, block) in self.matter.iter().enumerate().take(8) {
                let reach = AGENT_RADIUS + block.radius + 1.25;
                boulder_pushers[b_idx] = self
                    .agents
                    .iter()
                    .take(active_count)
                    .filter(|a| (a.pos.0 - block.pos.0).hypot(a.pos.1 - block.pos.1) <= reach)
                    .count();
            }
        }
        let mut spawned_impacts: Vec<(f32, f32, f32, f32, f32, f32, u8)> = Vec::new();

        for i in 0..active_count {
            let fx = sanitize_actuator(pulses[i].actuator_out.first().copied());
            let fy = sanitize_actuator(pulses[i].actuator_out.get(1).copied());
            let gx = sanitize_actuator(pulses[i].actuator_out.get(2).copied());
            let gy = sanitize_actuator(pulses[i].actuator_out.get(3).copied());
            mark_forces[i] = gx.hypot(gy);

            let prev_pos = self.agents[i].pos;
            let prev_vel = self.agents[i].vel;
            let traction = if self.earth_ecology {
                0.70 + 0.30 * self.agents[i].embodied_stability.clamp(0.25, 1.0)
            } else {
                1.0
            };
            let mut vx = (prev_vel.0 + fx * MAX_THRUST * traction) * DRAG;
            let mut vy = (prev_vel.1 + fy * MAX_THRUST * traction) * DRAG;

            let mut nx = prev_pos.0 + vx;
            let mut ny = prev_pos.1 + vy;
            let r = nx.hypot(ny);
            if r > wall_limit && r > 1e-5 {
                let ux = nx / r;
                let uy = ny / r;
                let v_out = (vx * ux + vy * uy).max(0.0);
                nx = ux * wall_limit;
                ny = uy * wall_limit;
                vx -= 1.4 * v_out * ux;
                vy -= 1.4 * v_out * uy;
                if self.earth_ecology {
                    tick_stresses[i] += (0.03 + 0.18 * v_out).min(0.28);
                } else {
                    tick_stresses[i] += (0.07 + 0.30 * v_out).min(0.55);
                }
            }

            // Interact with pushable warm mineral blocks (cooperative mechanical advantage in Earth mode).
            for (b_idx, block) in self.matter.iter_mut().enumerate() {
                let dx = nx - block.pos.0;
                let dy = ny - block.pos.1;
                let dist = dx.hypot(dy);
                let min_sep = AGENT_RADIUS + block.radius;
                let coop = self.earth_ecology && boulder_pushers.get(b_idx).copied().unwrap_or(0) >= 2;
                let push_gain = if coop {
                    0.096
                } else if self.earth_ecology {
                    0.036
                } else {
                    0.04
                };
                if dist < min_sep + 0.8 {
                    block.vel.0 += push_gain * gx;
                    block.vel.1 += push_gain * gy;
                    if coop && mark_forces[i] > 0.12 {
                        self.ecology_metrics.cooperative_boulder_ticks = self
                            .ecology_metrics
                            .cooperative_boulder_ticks
                            .saturating_add(1);
                        if spawned_impacts.len() < MAX_IMPACT_SLOTS {
                            spawned_impacts.push((block.pos.0, block.pos.1, 0.055, 0.030, 260.0, 430.0, 3));
                        }
                    }
                }
                if dist < min_sep {
                    let (ux, uy) = if dist > 1e-5 {
                        (dx / dist, dy / dist)
                    } else {
                        (1.0, 0.0)
                    };
                    let overlap = min_sep - dist;
                    nx += 0.65 * overlap * ux;
                    ny += 0.65 * overlap * uy;
                    let body_push = if coop { 0.24 } else { 0.12 };
                    block.vel.0 -= body_push * overlap * ux;
                    block.vel.1 -= body_push * overlap * uy;
                }
            }

            // Solid canyon basalt ridge collision (smoothly slides along the rock wall).
            for ridge in &self.ridges {
                let cp = closest_point_on_segment((nx, ny), ridge.a, ridge.b);
                let dx = nx - cp.0;
                let dy = ny - cp.1;
                let dist = dx.hypot(dy);
                let min_r = AGENT_RADIUS + ridge.thickness;
                if dist < min_r {
                    let (ux, uy) = if dist > 1e-5 {
                        (dx / dist, dy / dist)
                    } else {
                        let rx = ridge.b.0 - ridge.a.0;
                        let ry = ridge.b.1 - ridge.a.1;
                        let rl = rx.hypot(ry).max(1e-5);
                        (-ry / rl, rx / rl)
                    };
                    let overlap = min_r - dist;
                    nx += overlap * ux;
                    ny += overlap * uy;
                    let v_into = (vx * ux + vy * uy).min(0.0);
                    vx -= v_into * ux;
                    vy -= v_into * uy;
                }
            }

            if self.earth_ecology {
                let local_mark = sample_surface_mark(&self.surface_marks, (nx, ny), self.arena_radius);
                let roughness = 1.0 - 0.55 * local_mark;
                let dv_lat = (vx - prev_vel.0) - 0.6 * (vy - prev_vel.1);
                self.agents[i].last_lateral_sway = (0.08 * dv_lat * roughness).clamp(-0.022, 0.022);
            } else {
                self.agents[i].last_lateral_sway = 0.0;
            }

            self.agents[i].pos = (nx, ny);
            self.agents[i].vel = (vx, vy);
        }

        // Advance pushable warm mineral blocks with friction, ridge collision, and wall containment.
        let block_limit = self.arena_radius - 1.0;
        for block in &mut self.matter {
            block.vel.0 *= 0.80;
            block.vel.1 *= 0.80;
            block.pos.0 += block.vel.0;
            block.pos.1 += block.vel.1;
            for ridge in &self.ridges {
                let cp = closest_point_on_segment(block.pos, ridge.a, ridge.b);
                let dx = block.pos.0 - cp.0;
                let dy = block.pos.1 - cp.1;
                let dist = dx.hypot(dy);
                let min_r = block.radius + ridge.thickness;
                if dist < min_r && dist > 1e-5 {
                    let ux = dx / dist;
                    let uy = dy / dist;
                    block.pos.0 += (min_r - dist) * ux;
                    block.pos.1 += (min_r - dist) * uy;
                    let v_into = (block.vel.0 * ux + block.vel.1 * uy).min(0.0);
                    block.vel.0 -= 1.2 * v_into * ux;
                    block.vel.1 -= 1.2 * v_into * uy;
                }
            }
            let br = block.pos.0.hypot(block.pos.1);
            if br > block_limit && br > 1e-5 {
                block.pos.0 = block.pos.0 / br * block_limit;
                block.pos.1 = block.pos.1 / br * block_limit;
                block.vel = (0.0, 0.0);
            }
        }

        // Stone-anvil & manual/cooperative hand cracking of hard-shelled botanical ForagePods.
        if self.earth_ecology {
            for pod in &mut self.pods {
                if pod.shell_integrity > 0.0 {
                    pod.ripeness = (pod.ripeness + 0.0038).min(1.0);
                    if pod.ripeness >= 0.40 {
                        // 1. Lithic stone-anvil impact from warm mineral boulders
                        for block in &self.matter {
                            let d_bp = (block.pos.0 - pod.pos.0).hypot(block.pos.1 - pod.pos.1);
                            let b_speed = block.vel.0.hypot(block.vel.1);
                            if d_bp <= block.radius + 0.85 && b_speed > 0.014 {
                                pod.shell_integrity =
                                    (pod.shell_integrity - (0.58 + 3.0 * b_speed)).max(0.0);
                                if pod.shell_integrity <= 0.0 {
                                    pod.open_nutrients = if pod.heavy_shell {
                                        1.85 * pod.ripeness
                                    } else {
                                        pod.ripeness
                                    };
                                    self.ecology_metrics.pods_cracked_by_stone = self
                                        .ecology_metrics
                                        .pods_cracked_by_stone
                                        .saturating_add(1);
                                    if spawned_impacts.len() < MAX_IMPACT_SLOTS {
                                        spawned_impacts.push((
                                            pod.pos.0, pod.pos.1, 0.11, 0.08, 390.0, 1180.0, 4,
                                        ));
                                    }
                                    break;
                                }
                            }
                        }
                        // 2. Manual hand cracking: heavy_shell grove pods require >= 2 simultaneous
                        //    organisms within 2.8 m (solo prying stops at 0.22 and radiates woody taps).
                        if pod.shell_integrity > 0.0 {
                            if pod.heavy_shell {
                                let mut workers: Vec<(usize, f32)> = Vec::new();
                                for i in 0..active_count {
                                    let d_ap = (self.agents[i].pos.0 - pod.pos.0)
                                        .hypot(self.agents[i].pos.1 - pod.pos.1);
                                    if d_ap <= 2.8 {
                                        let stab =
                                            self.agents[i].embodied_stability.clamp(0.25, 1.0);
                                        let hw = self.agents[i].embodied_hand_work;
                                        let mf = (mark_forces[i] - 0.14).max(0.0);
                                        let chip = (0.20 * hw + 0.08 * mf) * stab;
                                        if chip > 0.010 {
                                            workers.push((i, chip));
                                        }
                                    }
                                }
                                if workers.len() >= 2 {
                                    let total_chip: f32 =
                                        workers.iter().map(|&(_, c)| c).sum::<f32>() * 1.45;
                                    pod.shell_integrity =
                                        (pod.shell_integrity - total_chip).max(0.0);
                                    if pod.shell_integrity <= 0.0 {
                                        pod.open_nutrients = 1.85 * pod.ripeness;
                                        for &(wi, _) in &workers {
                                            self.agents[wi].pods_cracked =
                                                self.agents[wi].pods_cracked.saturating_add(1);
                                        }
                                        self.ecology_metrics.pods_cracked_by_hands = self
                                            .ecology_metrics
                                            .pods_cracked_by_hands
                                            .saturating_add(1);
                                        self.ecology_metrics.coop_grove_pods_cracked = self
                                            .ecology_metrics
                                            .coop_grove_pods_cracked
                                            .saturating_add(1);
                                        if spawned_impacts.len() < MAX_IMPACT_SLOTS {
                                            spawned_impacts.push((
                                                pod.pos.0, pod.pos.1, 0.13, 0.095, 480.0, 1120.0, 5,
                                            ));
                                        }
                                    }
                                } else if let Some(&(_, solo_chip)) = workers.first() {
                                    pod.shell_integrity =
                                        (pod.shell_integrity - 0.45 * solo_chip).max(0.22);
                                    if solo_chip > 0.025
                                        && spawned_impacts.len() < MAX_IMPACT_SLOTS
                                    {
                                        spawned_impacts.push((
                                            pod.pos.0, pod.pos.1, 0.055, 0.036, 540.0, 1460.0, 3,
                                        ));
                                    }
                                }
                            } else {
                                for i in 0..active_count {
                                    let d_ap = (self.agents[i].pos.0 - pod.pos.0)
                                        .hypot(self.agents[i].pos.1 - pod.pos.1);
                                    if d_ap <= 2.3 {
                                        let stab =
                                            self.agents[i].embodied_stability.clamp(0.25, 1.0);
                                        let hw = self.agents[i].embodied_hand_work;
                                        let mf = (mark_forces[i] - 0.18).max(0.0);
                                        let chip = (0.18 * hw + 0.06 * mf) * stab;
                                        if chip > 0.012 {
                                            pod.shell_integrity =
                                                (pod.shell_integrity - chip).max(0.0);
                                            if pod.shell_integrity <= 0.0 {
                                                pod.open_nutrients = 0.95 * pod.ripeness;
                                                self.agents[i].pods_cracked =
                                                    self.agents[i].pods_cracked.saturating_add(1);
                                                self.ecology_metrics.pods_cracked_by_hands = self
                                                    .ecology_metrics
                                                    .pods_cracked_by_hands
                                                    .saturating_add(1);
                                                if spawned_impacts.len() < MAX_IMPACT_SLOTS {
                                                    spawned_impacts.push((
                                                        pod.pos.0, pod.pos.1, 0.10, 0.075, 640.0,
                                                        1240.0, 4,
                                                    ));
                                                }
                                                break;
                                            } else if chip > 0.035
                                                && spawned_impacts.len() < MAX_IMPACT_SLOTS
                                            {
                                                spawned_impacts.push((
                                                    pod.pos.0, pod.pos.1, 0.038, 0.022, 820.0,
                                                    1360.0, 2,
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Track visually occluded acoustic approaches across opaque basalt ridges.
            for i in 0..active_count {
                for j in 0..active_count {
                    if i == j {
                        continue;
                    }
                    let pi = self.agents[i].pos;
                    let pj = self.agents[j].pos;
                    if !ridges_clear_sight(&self.ridges, pi, pj)
                        && (self.agents[i].embodied_voice_rms >= 0.015
                            || self.agents[i].recent_acoustic_activity >= 0.12)
                    {
                        let dx = pi.0 - pj.0;
                        let dy = pi.1 - pj.1;
                        let dist = dx.hypot(dy);
                        if dist > 0.5 {
                            let approach_v =
                                (self.agents[j].vel.0 * dx + self.agents[j].vel.1 * dy) / dist;
                            if approach_v > 0.015 {
                                self.ecology_metrics.occluded_vocal_approaches = self
                                    .ecology_metrics
                                    .occluded_vocal_approaches
                                    .saturating_add(1);
                            }
                        }
                    }
                }
            }
            for (ix, iy, aa, ab, fa, fb, tl) in spawned_impacts {
                self.enqueue_impact_sound((ix, iy), aa, ab, fa, fb, tl);
            }
        }

        // Gentle soft body separation between overlapping peer organisms.
        for i in 0..active_count {
            for j in (i + 1)..active_count {
                let dx = self.agents[j].pos.0 - self.agents[i].pos.0;
                let dy = self.agents[j].pos.1 - self.agents[i].pos.1;
                let dist = dx.hypot(dy);
                let min_sep = 2.0 * AGENT_RADIUS;
                if dist < min_sep && dist > 1e-5 {
                    let push = 0.5 * (min_sep - dist);
                    let ux = dx / dist;
                    let uy = dy / dist;
                    self.agents[i].pos.0 -= push * ux;
                    self.agents[i].pos.1 -= push * uy;
                    self.agents[j].pos.0 += push * ux;
                    self.agents[j].pos.1 += push * uy;
                }
            }
        }

        // 5. Advance tick and continuous Day/Night illumination cycle.
        let current_tick = emit_tick + 1;
        self.metrics.tick = current_tick;
        self.illumination_lux = compute_day_night_lux(current_tick, self.day_night_period);
        let is_night = self.illumination_lux <= 0.01;
        if is_night && !self.was_night {
            self.metrics.night_transitions += 1;
        } else if !is_night && self.was_night {
            self.metrics.day_transitions += 1;
        }
        self.was_night = is_night;
        self.metrics.illumination_lux = self.illumination_lux;

        // 6. Compute per-agent nutrient harvesting and cold-night shelter protection.
        let night_chill = (1.0 - self.illumination_lux).clamp(0.0, 1.0);
        let updated_positions: Vec<(f32, f32)> =
            self.agents.iter().take(active_count).map(|a| a.pos).collect();

        for i in 0..active_count {
            let pos = updated_positions[i];
            let mut carried = self.agents[i].warmth_nutrients_carried * 0.948;
            let mut nourished = self.agents[i].recent_nourishment_trace * 0.994;
            let mut influx = 0.0_f32;
            let drain_floor = if self.earth_ecology { 0.02 } else { 0.05 };
            let core_drain = if self.earth_ecology { 0.014 } else { 0.010 };
            for spring in &mut self.springs {
                let d = (pos.0 - spring.pos.0).hypot(pos.1 - spring.pos.1);
                if d <= spring.core_radius {
                    let norm_d = d / spring.core_radius.max(0.1);
                    influx += 0.22 * spring.reservoir * (1.0 - 0.4 * norm_d * norm_d);
                    spring.reservoir = (spring.reservoir - core_drain).max(drain_floor);
                    carried = (carried + 0.34).min(1.0);
                    nourished = (nourished + 0.25).min(1.0);
                } else if d <= spring.halo_radius {
                    let u = 1.0 - (d - spring.core_radius) / (spring.halo_radius - spring.core_radius);
                    influx += 0.038 * spring.reservoir * u * u;
                    spring.reservoir = (spring.reservoir - 0.005).max(drain_floor);
                    carried = (carried + 0.12 * u).min(1.0);
                    nourished = (nourished + 0.08 * u).min(1.0);
                }
            }

            // Harvest cracked open ForagePods (shared between opener and nearby peers within 2.8 m).
            if self.earth_ecology {
                for pod in &mut self.pods {
                    if pod.shell_integrity <= 0.0 && pod.open_nutrients > 0.02 {
                        let d_pod = (pos.0 - pod.pos.0).hypot(pos.1 - pod.pos.1);
                        if d_pod <= 2.8 {
                            let prox = 1.0 - 0.55 * (d_pod / 2.8);
                            influx += 0.28 * pod.open_nutrients * prox;
                            pod.open_nutrients = (pod.open_nutrients - 0.012).max(0.0);
                            carried = (carried + 0.36).min(1.0);
                            nourished = (nourished + 0.28).min(1.0);
                            if pod.open_nutrients <= 0.02 {
                                pod.open_nutrients = 0.0;
                                pod.ripeness = 0.08;
                                pod.shell_integrity = 1.0;
                            }
                        }
                    }
                }
            }

            // Evaluate thermal protection from shelters, warm blocks, huddling peers,
            // and pre-existing warm inscribed terrain paths (+0.15..0.20 * mark_strength, zero free influx).
            let mut warmth_protection: f32 = 0.0;
            let shelter_carry_cap = 0.22 + 0.72 * nourished;
            let shelter_halo = if self.earth_ecology { 2.5 } else { 2.0 };
            for shelter in &self.shelters {
                let d = (pos.0 - shelter.pos.0).hypot(pos.1 - shelter.pos.1);
                if d <= shelter.radius {
                    warmth_protection = warmth_protection.max(1.0);
                    carried = (carried + 0.20 * shelter_carry_cap).min(shelter_carry_cap);
                } else if d <= shelter.radius + shelter_halo {
                    let w = 1.0 - (d - shelter.radius) / shelter_halo;
                    warmth_protection = warmth_protection.max(w);
                }
            }
            for block in &self.matter {
                let d_blk = (pos.0 - block.pos.0).hypot(pos.1 - block.pos.1);
                if d_blk <= block.warmth_radius {
                    let w = 0.78 * (1.0 - d_blk / block.warmth_radius);
                    warmth_protection = warmth_protection.max(w);
                }
            }
            let huddle_radius = if self.earth_ecology { 3.5 } else { 2.8 };
            for (j, &peer_pos) in updated_positions.iter().enumerate() {
                if i != j {
                    let d = (pos.0 - peer_pos.0).hypot(pos.1 - peer_pos.1);
                    if d <= huddle_radius {
                        let w = (if self.earth_ecology { 0.75 } else { 0.65 }) * (1.0 - d / huddle_radius);
                        warmth_protection = warmth_protection.max(w);
                        if self.earth_ecology && i < j {
                            self.ecology_metrics.huddling_ticks =
                                self.ecology_metrics.huddling_ticks.saturating_add(1);
                        }
                    }
                }
            }

            let local_mark = sample_surface_mark(&self.surface_marks, pos, self.arena_radius);
            let mark_insul = if self.earth_ecology { 0.20 } else { 0.15 };
            let insulated_protection = (warmth_protection + mark_insul * local_mark).clamp(0.0, 1.0);
            let chill_coeff = if self.earth_ecology { 0.030 } else { 0.045 };
            let cold_stress = chill_coeff * night_chill * (1.0 - insulated_protection);
            let total_stress = (tick_stresses[i] + cold_stress).max(0.0);

            self.agents[i].warmth_nutrients_carried = carried;
            self.agents[i].recent_nourishment_trace = nourished;
            let speed = self.agents[i].vel.0.hypot(self.agents[i].vel.1);
            let mark_force = mark_forces[i];
            if carried > 0.12 && (mark_force > 0.15 || speed > 0.05) {
                let motion_gate = (speed / 0.26).clamp(0.15, 1.0);
                let drive = (0.65 * (mark_force - 0.15).max(0.0)
                    + 0.45 * (speed - 0.05).max(0.0))
                    .clamp(0.0, 1.0);
                deposit_surface_mark(
                    &mut self.surface_marks,
                    pos,
                    self.arena_radius,
                    0.044 * carried * motion_gate * drive,
                );
            }

            self.agents[i].last_influx = influx.max(0.0);
            self.agents[i].last_stress = total_stress;
            self.agents[i].cumulative_influx += self.agents[i].last_influx;
            self.agents[i].cumulative_stress += total_stress;
        }

        // Regenerate nutrient springs (hydro-seasonal watershed phase in Earth mode).
        if self.earth_ecology {
            let season_angle = TAU * ((current_tick % EARTH_SEASON_PERIOD_TICKS) as f32)
                / (EARTH_SEASON_PERIOD_TICKS as f32);
            for (s_idx, spring) in self.springs.iter_mut().enumerate() {
                let harvested = updated_positions
                    .iter()
                    .any(|&p| (p.0 - spring.pos.0).hypot(p.1 - spring.pos.1) <= spring.halo_radius);
                let phase_off = self.spring_phase_offsets.get(s_idx).copied().unwrap_or(0.0);
                let max_cap = self.spring_max_flows.get(s_idx).copied().unwrap_or(1.0);
                let seasonal = 0.25 + 0.75 * (0.5 + 0.5 * (season_angle + phase_off).cos());
                if !harvested {
                    spring.reservoir = (spring.reservoir + 0.0058 * seasonal).min(max_cap * (0.35 + 0.65 * seasonal));
                }
            }
            self.step_earth_acoustics_16khz();
        } else {
            for spring in &mut self.springs {
                let harvested = updated_positions
                    .iter()
                    .any(|&p| (p.0 - spring.pos.0).hypot(p.1 - spring.pos.1) <= spring.halo_radius);
                if !harvested {
                    spring.reservoir = (spring.reservoir + 0.0045).min(1.0);
                }
            }
        }

        // 7. Refresh all agent sensory views (Photonic, Echo arrivals at `current_tick`, Noisy TV).
        for i in 0..self.agents.len() {
            self.refresh_agent_photonic(i);
            let agent = &mut self.agents[i];
            let pos = agent.pos;
            let _ = agent.echo_ring.receive_at_tick(
                current_tick,
                pos,
                &mut agent.ambient[PHOTONIC_END..ECHO_END],
            );
            self.refresh_agent_noisy_tv(i);
        }

        self.update_synchrony_and_metrics(active_count);
    }

    /// Couples an organism's real postural stability (`[0.25, 1.0]`) and mechanical hand work
    /// (`[0.0, 1.5]`) into the 2D Earth physics before `apply_population_pulses`.
    pub fn set_agent_embodied_kinematics(&mut self, agent_idx: usize, stability: f32, hand_work: f32) {
        if let Some(slot) = self.agents.get_mut(agent_idx) {
            slot.embodied_stability = if stability.is_finite() {
                stability.clamp(0.25, 1.0)
            } else {
                1.0
            };
            slot.embodied_hand_work = if hand_work.is_finite() {
                hand_work.clamp(0.0, 1.5)
            } else {
                0.0
            };
        }
    }

    /// Records `agent_idx`'s latest radiated `16 kHz` vocal RMS (`[0.0, 1.0]`) for read-only
    /// occluded acoustic approach observation (zero effect on `metabolic_influx`).
    pub fn set_agent_embodied_voice_rms(&mut self, agent_idx: usize, voice_rms: f32) {
        if let Some(slot) = self.agents.get_mut(agent_idx) {
            slot.embodied_voice_rms = if voice_rms.is_finite() {
                voice_rms.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
    }

    /// Returns `true` if the straight line-of-sight between `p0` and `p1` is unoccluded by basalt ridges.
    #[must_use]
    pub fn has_line_of_sight(&self, p0: (f32, f32), p1: (f32, f32)) -> bool {
        ridges_clear_sight(&self.ridges, p0, p1)
    }

    /// Airborne `16 kHz` acoustic transmittance between `p0` and `p1`: `1.0` in clear sight,
    /// `0.68` diffracted over/around an opaque basalt canyon ridge (whereas visual light is `0.0`).
    #[must_use]
    pub fn acoustic_transmittance(&self, p0: (f32, f32), p1: (f32, f32)) -> f32 {
        if ridges_clear_sight(&self.ridges, p0, p1) {
            1.0
        } else {
            0.68
        }
    }

    /// Read-only view of opaque basalt canyon ridges.
    #[must_use]
    pub fn basalt_ridges(&self) -> &[BasaltRidge] {
        &self.ridges
    }

    /// Lateral terrain sway acceleration experienced by `agent_idx` on the latest 2D step.
    #[must_use]
    pub fn agent_lateral_sway(&self, agent_idx: usize) -> f32 {
        self.agents
            .get(agent_idx)
            .map_or(0.0, |a| a.last_lateral_sway)
    }

    /// Spatializes the `16 kHz` Earth soundscape (flowing watershed springs, cracked forage pods,
    /// rolling warm boulders) at `agent_idx`'s Left & Right ears with physical inverse-distance
    /// attenuation, sample-offset `ITD`, and head-shadow `ILD`, returning `shelter_enclosure` (`[0, 1]`).
    pub fn binaural_earth_waves_for_agent(
        &self,
        agent_idx: usize,
        out_left: &mut [f32; EARTH_AUDIO_FRAME_SAMPLES],
        out_right: &mut [f32; EARTH_AUDIO_FRAME_SAMPLES],
    ) -> f32 {
        out_left.fill(0.0);
        out_right.fill(0.0);
        let Some(agent) = self.agents.get(agent_idx) else {
            return 0.0;
        };
        let (lx, ly) = agent.pos;

        let mut enclosure = 0.0_f32;
        for shelter in &self.shelters {
            let d = (lx - shelter.pos.0).hypot(ly - shelter.pos.1);
            if d <= shelter.radius {
                let e = (1.0 - 0.45 * (d / shelter.radius.max(0.1))).clamp(0.0, 1.0);
                enclosure = enclosure.max(e);
            }
        }
        for ridge in &self.ridges {
            let cp = closest_point_on_segment((lx, ly), ridge.a, ridge.b);
            let d_r = (lx - cp.0).hypot(ly - cp.1);
            if d_r <= 2.4 {
                let e_r = 0.48 * (1.0 - d_r / 2.4).clamp(0.0, 1.0);
                enclosure = enclosure.max(e_r);
            }
        }
        if !self.earth_ecology {
            return enclosure;
        }

        // 1. Spatialize flowing water murmurs from active springs (diffracting 0.68x across basalt ridges).
        for (s_idx, spring) in self.springs.iter().enumerate().take(MAX_WATER_SOURCES) {
            if spring.reservoir <= 0.04 {
                continue;
            }
            let diff = self.acoustic_transmittance((lx, ly), spring.pos);
            accumulate_spatial_source(
                &self.water_frames[s_idx],
                (spring.pos.0 - lx, spring.pos.1 - ly),
                0.24,
                diff,
                out_left,
                out_right,
            );
        }

        // 2. Spatialize transient shell cracks, woody taps, and cooperative stone rumbles.
        for (e_idx, ev) in self.impact_events.iter().enumerate() {
            if ev.ticks_left == 0 {
                continue;
            }
            let diff = self.acoustic_transmittance((lx, ly), ev.pos);
            accumulate_spatial_source(
                &self.impact_frames[e_idx],
                (ev.pos.0 - lx, ev.pos.1 - ly),
                0.15,
                diff,
                out_left,
                out_right,
            );
        }

        for s in out_left.iter_mut() {
            *s = s.clamp(-0.65, 0.65);
        }
        for s in out_right.iter_mut() {
            *s = s.clamp(-0.65, 0.65);
        }
        enclosure
    }

    /// Read-only Earth-ecology observation metrics.
    #[must_use]
    pub fn ecology_metrics(&self) -> &EarthEcologyMetrics {
        &self.ecology_metrics
    }

    /// Read-only view of botanical forage pods.
    #[must_use]
    pub fn forage_pods(&self) -> &[ForagePod] {
        &self.pods
    }

    /// Current ambient illumination in `[0.0, 1.0]`.
    #[must_use]
    pub fn illumination(&self) -> f32 {
        self.illumination_lux
    }

    /// Read-only view of the `16x16` persistent surface-mark grid.
    #[must_use]
    pub fn surface_marks(&self) -> &[[f32; MARK_GRID_DIM]; MARK_GRID_DIM] {
        &self.surface_marks
    }

    /// Samples bilinearly interpolated persistent mark strength at continuous `(x, y)`.
    #[must_use]
    pub fn sample_mark_at(&self, pos: (f32, f32)) -> f32 {
        sample_surface_mark(&self.surface_marks, pos, self.arena_radius)
    }

    /// Read-only external observer metrics.
    #[must_use]
    pub fn metrics(&self) -> &TerrariumMetrics {
        &self.metrics
    }

    /// Read-only view of thermal shelters.
    #[must_use]
    pub fn shelters(&self) -> &[ThermalShelter] {
        &self.shelters
    }

    /// Read-only view of nutrient springs.
    #[must_use]
    pub fn springs(&self) -> &[TerrariumSpring] {
        &self.springs
    }

    /// Read-only view of pushable warm mineral blocks.
    #[must_use]
    pub fn pushable_matter(&self) -> &[PushableMatter] {
        &self.matter
    }

    fn collect_world_reflectors(&self) -> Vec<Reflector> {
        let mut out = Vec::with_capacity(32);
        for spring in &self.springs {
            out.push(Reflector {
                pos: spring.pos,
                reflectivity: (0.45 + 0.55 * spring.reservoir).clamp(0.4, 1.0),
                kind: ReflectorKind::NourishingMatter,
            });
        }
        for shelter in &self.shelters {
            out.push(Reflector {
                pos: shelter.pos,
                reflectivity: 0.85,
                kind: ReflectorKind::ThermalShelter,
            });
        }
        for block in &self.matter {
            out.push(Reflector {
                pos: block.pos,
                reflectivity: 0.92,
                kind: ReflectorKind::HardObstacle,
            });
        }
        for ridge in &self.ridges {
            out.push(Reflector {
                pos: (0.5 * (ridge.a.0 + ridge.b.0), 0.5 * (ridge.a.1 + ridge.b.1)),
                reflectivity: 0.45,
                kind: ReflectorKind::HardObstacle,
            });
        }
        for pod in &self.pods {
            if pod.heavy_shell && pod.shell_integrity > 0.0 {
                continue;
            }
            let refl = if pod.shell_integrity > 0.0 {
                0.55 + 0.35 * pod.ripeness
            } else {
                0.75 * pod.open_nutrients
            };
            if refl >= 0.25 {
                out.push(Reflector {
                    pos: pod.pos,
                    reflectivity: refl.min(1.0),
                    kind: ReflectorKind::NourishingMatter,
                });
            }
        }
        for r in 0..MARK_GRID_DIM {
            for c in 0..MARK_GRID_DIM {
                let mark = self.surface_marks[r][c];
                if mark >= 0.08 {
                    out.push(Reflector {
                        pos: cell_center_world(r, c, self.arena_radius),
                        reflectivity: (0.35 + 0.65 * mark).min(1.0),
                        kind: ReflectorKind::SurfaceMark,
                    });
                }
            }
        }
        out
    }

    fn refresh_all_agent_ambients(&mut self) {
        for i in 0..self.agents.len() {
            self.refresh_agent_photonic(i);
            self.agents[i].ambient[PHOTONIC_END..ECHO_END].fill(0.0);
            self.refresh_agent_noisy_tv(i);
        }
    }

    fn refresh_agent_photonic(&mut self, agent_idx: usize) {
        if self.illumination_lux <= 0.0 {
            self.agents[agent_idx].ambient[0..PHOTONIC_END].fill(0.0);
            return;
        }

        let lux = self.illumination_lux;
        let (px, py) = self.agents[agent_idx].pos;
        let r_sq = px * px + py * py;
        let arena_r_sq = self.arena_radius * self.arena_radius;
        let peer_positions: Vec<(f32, f32)> = self
            .agents
            .iter()
            .enumerate()
            .filter_map(|(j, a)| (j != agent_idx).then_some(a.pos))
            .collect();

        let mut buf = [0.0_f32; PHOTONIC_END];
        for col in 0..PHOTONIC_COLS {
            let cx = self.col_cos[col];
            let cy = self.col_sin[col];
            let b = px * cx + py * cy;
            let disc = (b * b - (r_sq - arena_r_sq)).max(0.0);
            let arena_wall_dist = (-b + disc.sqrt()).max(0.2);
            let mut ridge_hit_dist = f32::INFINITY;
            for ridge in &self.ridges {
                if let Some(hd) = ray_segment_distance((px, py), (cx, cy), ridge.a, ridge.b)
                    && hd < ridge_hit_dist
                {
                    ridge_hit_dist = hd;
                }
            }

            for row in 0..PHOTONIC_ROWS {
                let depth = self.row_depths[row];
                let sample_pt = (px + cx * depth, py + cy * depth);
                let beyond_ridge = depth > ridge_hit_dist + 0.35;
                let wall_rim = (-0.5 * ((depth - arena_wall_dist) / 2.2).powi(2)).exp();
                let mut val =
                    0.05 * (1.0 - depth / (self.arena_radius * 1.4)).max(0.0) - 0.14 * wall_rim;
                if ridge_hit_dist.is_finite() {
                    let ridge_rim = (-0.5 * ((depth - ridge_hit_dist) / 1.4).powi(2)).exp();
                    val -= 0.04 * ridge_rim;
                }

                // Surface trail mark reflectance along ray point (blocked behind opaque basalt ridges)!
                if !beyond_ridge && sample_pt.0.hypot(sample_pt.1) <= self.arena_radius {
                    let trail_mark =
                        sample_surface_mark(&self.surface_marks, sample_pt, self.arena_radius);
                    val += 0.42 * trail_mark;
                }

                // Nutrient springs cast warm radiant peaks ONLY along clear unoccluded lines of sight.
                for spring in &self.springs {
                    if !ridges_clear_sight(&self.ridges, (px, py), spring.pos) {
                        continue;
                    }
                    let dx = spring.pos.0 - px;
                    let dy = spring.pos.1 - py;
                    let dist = dx.hypot(dy).max(0.2);
                    let align = (dx * cx + dy * cy) / dist;
                    if align > 0.3 {
                        let ang_gauss = (-(2.0 * (1.0 - align)).max(0.0) / 0.32).exp();
                        let depth_gauss = (-0.5 * ((depth - dist) / 3.0).powi(2)).exp();
                        val += (0.3 + 0.65 * spring.reservoir)
                            * ang_gauss
                            * (0.35 + 0.65 * depth_gauss)
                            / (1.0 + 0.06 * dist);
                    }
                }

                // Thermal shelters cast calm amber glow ONLY along clear unoccluded lines of sight.
                for shelter in &self.shelters {
                    if !ridges_clear_sight(&self.ridges, (px, py), shelter.pos) {
                        continue;
                    }
                    let dx = shelter.pos.0 - px;
                    let dy = shelter.pos.1 - py;
                    let dist = dx.hypot(dy).max(0.2);
                    let align = (dx * cx + dy * cy) / dist;
                    if align > 0.35 {
                        let ang_gauss = (-(2.0 * (1.0 - align)).max(0.0) / 0.36).exp();
                        let depth_gauss = (-0.5 * ((depth - dist) / 3.2).powi(2)).exp();
                        val += 0.35 * ang_gauss * depth_gauss / (1.0 + 0.05 * dist);
                    }
                }

                // Botanical ForagePods cast highlights ONLY when not occluded by basalt ridges.
                for pod in &self.pods {
                    if !ridges_clear_sight(&self.ridges, (px, py), pod.pos) {
                        continue;
                    }
                    let dx = pod.pos.0 - px;
                    let dy = pod.pos.1 - py;
                    let dist = dx.hypot(dy).max(0.2);
                    let align = (dx * cx + dy * cy) / dist;
                    if align > 0.38 {
                        let ang_gauss = (-(2.0 * (1.0 - align)).max(0.0) / 0.24).exp();
                        let depth_gauss = (-0.5 * ((depth - dist) / 2.2).powi(2)).exp();
                        let pod_lum = if pod.shell_integrity > 0.0 {
                            if pod.heavy_shell {
                                0.14 * pod.ripeness
                            } else {
                                0.28 * pod.ripeness
                            }
                        } else {
                            0.62 * pod.open_nutrients
                        };
                        val += pod_lum * ang_gauss * depth_gauss / (1.0 + 0.07 * dist);
                    }
                }

                // Peer organisms cast visible silhouettes ONLY along clear unoccluded lines of sight.
                for &(qx, qy) in &peer_positions {
                    if !ridges_clear_sight(&self.ridges, (px, py), (qx, qy)) {
                        continue;
                    }
                    let dx = qx - px;
                    let dy = qy - py;
                    let dist = dx.hypot(dy).max(0.2);
                    let align = (dx * cx + dy * cy) / dist;
                    if align > 0.45 {
                        let ang_gauss = (-(2.0 * (1.0 - align)).max(0.0) / 0.22).exp();
                        let depth_gauss = (-0.5 * ((depth - dist) / 2.0).powi(2)).exp();
                        val += 0.38 * ang_gauss * depth_gauss / (1.0 + 0.08 * dist);
                    }
                }

                buf[row * PHOTONIC_COLS + col] = (val * lux).clamp(-1.0, 1.0);
            }
        }
        self.agents[agent_idx].ambient[0..PHOTONIC_END].copy_from_slice(&buf);
    }

    fn refresh_agent_noisy_tv(&mut self, agent_idx: usize) {
        let flash_gate = self.noisy_rng.next_f32_unit();
        for ch in &mut self.agents[agent_idx].ambient[ECHO_END..TERRARIUM_CHANNELS] {
            let base = self.noisy_rng.next_f32_bipolar();
            let flash = if flash_gate > 0.72 {
                if self.noisy_rng.next_u32() & 1 == 0 {
                    0.96
                } else {
                    -0.96
                }
            } else {
                base
            };
            *ch = (0.65 * base + 0.35 * flash).clamp(-1.0, 1.0);
        }
    }

    fn sync_metrics_snapshot(&mut self) {
        let current_tick = self.metrics.tick;
        self.metrics.agent_metabolic_influx =
            self.agents.iter().map(|a| a.cumulative_influx).collect();
        self.metrics.agent_tissue_stress =
            self.agents.iter().map(|a| a.cumulative_stress).collect();
        self.metrics.agent_positions = self.agents.iter().map(|a| a.pos).collect();
        self.metrics.agent_ages = self
            .agents
            .iter()
            .map(|a| current_tick.saturating_sub(a.spawn_tick))
            .collect();

        let mut covered = 0usize;
        let mut total_mass = 0.0_f32;
        let mut aligned_mass = 0.0_f32;
        for r in 0..MARK_GRID_DIM {
            for c in 0..MARK_GRID_DIM {
                let m = self.surface_marks[r][c];
                total_mass += m;
                aligned_mass += m * self.corridor_weights[r][c];
                if m >= 0.05 {
                    covered += 1;
                }
            }
        }
        self.metrics.persistent_mark_coverage =
            (covered as f32) / ((MARK_GRID_DIM * MARK_GRID_DIM) as f32);
        self.metrics.total_mark_mass = total_mass;
        self.metrics.mark_corridor_alignment = if total_mass > 1e-5 {
            (aligned_mass / total_mass).clamp(0.0, 1.0)
        } else {
            0.0
        };

        if self.earth_ecology {
            self.ecology_metrics.enabled = true;
            self.ecology_metrics.season_phase =
                ((current_tick % EARTH_SEASON_PERIOD_TICKS) as f32) / (EARTH_SEASON_PERIOD_TICKS as f32);
            self.ecology_metrics.flowing_springs =
                self.springs.iter().filter(|s| s.reservoir >= 0.20).count();
            let s_cnt = self.springs.len().max(1) as f32;
            self.ecology_metrics.mean_spring_flow =
                self.springs.iter().map(|s| s.reservoir).sum::<f32>() / s_cnt;
            self.ecology_metrics.ripe_intact_pods = self
                .pods
                .iter()
                .filter(|p| p.shell_integrity > 0.0 && p.ripeness >= 0.40)
                .count();
            self.ecology_metrics.open_cracked_pods = self
                .pods
                .iter()
                .filter(|p| p.shell_integrity <= 0.0 && p.open_nutrients > 0.02)
                .count();
            let mut occ_cnt = 0usize;
            for i in 0..self.agents.len() {
                let pi = self.agents[i].pos;
                for j in (i + 1)..self.agents.len() {
                    if !ridges_clear_sight(&self.ridges, pi, self.agents[j].pos) {
                        occ_cnt += 1;
                    }
                }
                for pod in &self.pods {
                    if (pod.ripeness >= 0.40 || pod.open_nutrients > 0.02)
                        && !ridges_clear_sight(&self.ridges, pi, pod.pos)
                    {
                        occ_cnt += 1;
                    }
                }
            }
            self.ecology_metrics.visual_occlusions_active = occ_cnt;
            self.ecology_metrics.agent_pods_cracked =
                self.agents.iter().map(|a| a.pods_cracked).collect();
        }
    }

    fn enqueue_impact_sound(
        &mut self,
        pos: (f32, f32),
        amp_a: f32,
        amp_b: f32,
        freq_a_hz: f32,
        freq_b_hz: f32,
        ticks_left: u8,
    ) {
        let slot_idx = self
            .impact_events
            .iter()
            .position(|s| s.ticks_left == 0)
            .unwrap_or_else(|| {
                self.impact_events
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, s)| s.ticks_left)
                    .map_or(0, |(idx, _)| idx)
            });
        self.impact_events[slot_idx] = AcousticImpactSlot {
            pos,
            amp_a,
            amp_b,
            phase_a: 0.0,
            phase_b: 0.0,
            omega_a: TAU * freq_a_hz / 16_000.0,
            omega_b: TAU * freq_b_hz / 16_000.0,
            decay_per_sample: 0.9965,
            ticks_left,
        };
        self.ecology_metrics.acoustic_impact_events =
            self.ecology_metrics.acoustic_impact_events.saturating_add(1);
    }

    fn step_earth_acoustics_16khz(&mut self) {
        let base_sample = (self.metrics.tick as f32) * (EARTH_AUDIO_FRAME_SAMPLES as f32);
        for (s_idx, spring) in self.springs.iter().enumerate().take(MAX_WATER_SOURCES) {
            let flow = spring.reservoir.clamp(0.0, 1.0);
            if flow <= 0.04 {
                self.water_frames[s_idx].fill(0.0);
                continue;
            }
            let f_center = 330.0 + 115.0 * (s_idx as f32);
            let r_pole = 0.982_f32;
            let a1 = 2.0 * r_pole * (TAU * f_center / 16_000.0).cos();
            let a2 = -(r_pole * r_pole);
            let ripple_omega = TAU * (6.5 + 1.3 * (s_idx as f32)) / 16_000.0;
            let drive_gain = 0.0042 * flow;
            let (mut y1, mut y2) = self.water_resonators[s_idx];

            for (k, out_s) in self.water_frames[s_idx].iter_mut().enumerate() {
                let n = self.acoustic_rng.next_f32_bipolar();
                let y = drive_gain * n + a1 * y1 + a2 * y2;
                let y_clean = if y.is_finite() { y.clamp(-0.25, 0.25) } else { 0.0 };
                y2 = y1;
                y1 = y_clean;
                let ripple = 0.68 + 0.32 * ((base_sample + (k as f32)) * ripple_omega).sin();
                *out_s = (y_clean * ripple).clamp(-0.20, 0.20);
            }
            self.water_resonators[s_idx] = (y1, y2);
        }

        for (e_idx, ev) in self.impact_events.iter_mut().enumerate() {
            if ev.ticks_left == 0 {
                self.impact_frames[e_idx].fill(0.0);
                continue;
            }
            for out_s in &mut self.impact_frames[e_idx] {
                let s = ev.amp_a * ev.phase_a.sin() + ev.amp_b * ev.phase_b.sin();
                *out_s = s.clamp(-0.35, 0.35);
                ev.phase_a = (ev.phase_a + ev.omega_a) % TAU;
                ev.phase_b = (ev.phase_b + ev.omega_b) % TAU;
                ev.amp_a *= ev.decay_per_sample;
                ev.amp_b *= ev.decay_per_sample;
            }
            ev.ticks_left = ev.ticks_left.saturating_sub(1);
        }
    }

    fn update_synchrony_and_metrics(&mut self, active_count: usize) {
        self.sync_metrics_snapshot();

        if active_count < 2 {
            self.metrics.mean_inter_organism_proximity = 0.0;
            self.metrics.mean_pairwise_distance = 0.0;
            return;
        }

        let mut prox_sum = 0.0_f32;
        let mut dist_sum = 0.0_f32;
        let mut pair_count = 0usize;

        let mut ac_sync_sum = 0.0_f32;
        let mut motor_sync_sum = 0.0_f32;
        let mut sync_pairs = 0usize;

        let has_distinct_cohorts = self
            .agents
            .iter()
            .take(active_count)
            .map(|a| a.spawn_tick)
            .min()
            != self
                .agents
                .iter()
                .take(active_count)
                .map(|a| a.spawn_tick)
                .max();

        for i in 0..active_count {
            for j in (i + 1)..active_count {
                let d = (self.agents[i].pos.0 - self.agents[j].pos.0)
                    .hypot(self.agents[i].pos.1 - self.agents[j].pos.1);
                prox_sum += 1.0 / (1.0 + 0.2 * d);
                dist_sum += d;
                pair_count += 1;

                let cohort_match = if has_distinct_cohorts {
                    self.agents[i].spawn_tick != self.agents[j].spawn_tick
                } else {
                    true
                };
                if cohort_match {
                    let ai = self.agents[i].recent_acoustic_activity;
                    let aj = self.agents[j].recent_acoustic_activity;
                    let ac_sim = if ai + aj > 0.02 {
                        (2.0 * ai.min(aj) / (ai + aj)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    ac_sync_sum += ac_sim;

                    let vi = self.agents[i].vel;
                    let vj = self.agents[j].vel;
                    let si = vi.0.hypot(vi.1);
                    let sj = vj.0.hypot(vj.1);
                    let vel_cos = if si > 1e-3 && sj > 1e-3 {
                        ((vi.0 * vj.0 + vi.1 * vj.1) / (si * sj)).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    };
                    motor_sync_sum += vel_cos;
                    sync_pairs += 1;
                }
            }
        }

        if pair_count > 0 {
            self.metrics.mean_inter_organism_proximity = prox_sum / (pair_count as f32);
            self.metrics.mean_pairwise_distance = dist_sum / (pair_count as f32);
        }
        if sync_pairs > 0 {
            let inst_ac = ac_sync_sum / (sync_pairs as f32);
            let inst_mo = motor_sync_sum / (sync_pairs as f32);
            self.metrics.elder_newborn_acoustic_synchrony =
                0.90 * self.metrics.elder_newborn_acoustic_synchrony + 0.10 * inst_ac;
            self.metrics.elder_newborn_motor_synchrony =
                0.90 * self.metrics.elder_newborn_motor_synchrony + 0.10 * inst_mo;
        }
    }
}

impl Default for MultiGenTerrarium {
    fn default() -> Self {
        Self::new(0xA11_7E88, 2)
    }
}

impl WorldMedium for MultiGenTerrarium {
    fn num_ambient_channels(&self) -> usize {
        TERRARIUM_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        TERRARIUM_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        self.agent_view(0)
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.apply_population_pulses(std::slice::from_ref(pulse));
    }
}

fn accumulate_spatial_source(
    wave: &[f32; EARTH_AUDIO_FRAME_SAMPLES],
    rel: (f32, f32),
    dist_falloff: f32,
    amp_scale: f32,
    out_left: &mut [f32; EARTH_AUDIO_FRAME_SAMPLES],
    out_right: &mut [f32; EARTH_AUDIO_FRAME_SAMPLES],
) {
    let dist = rel.0.hypot(rel.1);
    let att = amp_scale / (1.0 + dist_falloff * dist.max(0.0));
    if att <= 0.03 {
        return;
    }
    let sin_az = if dist > 0.05 {
        (rel.0 / dist).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let signed_itd = (sin_az * 5.0).round() as isize;
    let d_l = if signed_itd > 0 {
        (signed_itd as usize).min(MAX_EARTH_ITD_SAMPLES)
    } else {
        0
    };
    let d_r = if signed_itd < 0 {
        ((-signed_itd) as usize).min(MAX_EARTH_ITD_SAMPLES)
    } else {
        0
    };
    let g_r = att * (0.62 + 0.36 * sin_az);
    let g_l = att * (0.62 - 0.36 * sin_az);

    for i in 0..EARTH_AUDIO_FRAME_SAMPLES {
        let sl = wave[i.saturating_sub(d_l)];
        let sr = wave[i.saturating_sub(d_r)];
        out_left[i] += g_l * sl;
        out_right[i] += g_r * sr;
    }
}

/// Smooth day/night curve oscillating between sustained bright day (`1.0`) and sustained
/// pitch-black night (`0.0`) with smooth twilight ramps.
#[must_use]
pub fn compute_day_night_lux(tick: u64, period: u64) -> f32 {
    let phase = TAU * ((tick % period.max(1)) as f32) / (period.max(1) as f32);
    let raw = (0.5 + 0.78 * phase.cos()).clamp(0.0, 1.0);
    // Smoothstep C1 continuity across dawn/dusk boundaries.
    raw * raw * (3.0 - 2.0 * raw)
}

fn world_to_grid_coords(pos: (f32, f32), arena_radius: f32) -> (f32, f32) {
    let scale = (MARK_GRID_DIM as f32 - 1.0) / (2.0 * arena_radius.max(1.0));
    let gx = ((pos.0 + arena_radius) * scale).clamp(0.0, (MARK_GRID_DIM - 1) as f32);
    let gy = ((pos.1 + arena_radius) * scale).clamp(0.0, (MARK_GRID_DIM - 1) as f32);
    (gx, gy)
}

fn cell_center_world(row: usize, col: usize, arena_radius: f32) -> (f32, f32) {
    let cell_span = (2.0 * arena_radius) / (MARK_GRID_DIM as f32);
    let x = -arena_radius + ((col as f32) + 0.5) * cell_span;
    let y = -arena_radius + ((row as f32) + 0.5) * cell_span;
    (x, y)
}

fn deposit_surface_mark(
    grid: &mut [[f32; MARK_GRID_DIM]; MARK_GRID_DIM],
    pos: (f32, f32),
    arena_radius: f32,
    amount: f32,
) {
    if amount <= 0.0 || !amount.is_finite() {
        return;
    }
    let (gx, gy) = world_to_grid_coords(pos, arena_radius);
    let c0 = (gx.floor() as usize).min(MARK_GRID_DIM - 1);
    let r0 = (gy.floor() as usize).min(MARK_GRID_DIM - 1);
    let c1 = (c0 + 1).min(MARK_GRID_DIM - 1);
    let r1 = (r0 + 1).min(MARK_GRID_DIM - 1);
    let tx = gx - (c0 as f32);
    let ty = gy - (r0 as f32);

    let weights = [
        (r0, c0, (1.0 - tx) * (1.0 - ty)),
        (r0, c1, tx * (1.0 - ty)),
        (r1, c0, (1.0 - tx) * ty),
        (r1, c1, tx * ty),
    ];
    for (r, c, w) in weights {
        grid[r][c] = (grid[r][c] + amount * w).min(1.0);
    }
}

fn sample_surface_mark(
    grid: &[[f32; MARK_GRID_DIM]; MARK_GRID_DIM],
    pos: (f32, f32),
    arena_radius: f32,
) -> f32 {
    let (gx, gy) = world_to_grid_coords(pos, arena_radius);
    let c0 = (gx.floor() as usize).min(MARK_GRID_DIM - 1);
    let r0 = (gy.floor() as usize).min(MARK_GRID_DIM - 1);
    let c1 = (c0 + 1).min(MARK_GRID_DIM - 1);
    let r1 = (r0 + 1).min(MARK_GRID_DIM - 1);
    let tx = gx - (c0 as f32);
    let ty = gy - (r0 as f32);

    let v00 = grid[r0][c0];
    let v01 = grid[r0][c1];
    let v10 = grid[r1][c0];
    let v11 = grid[r1][c1];
    let top = v00 * (1.0 - tx) + v01 * tx;
    let bot = v10 * (1.0 - tx) + v11 * tx;
    top * (1.0 - ty) + bot * ty
}

fn point_to_segment_dist(pt: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let cp = closest_point_on_segment(pt, a, b);
    (pt.0 - cp.0).hypot(pt.1 - cp.1)
}

fn closest_point_on_segment(pt: (f32, f32), a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let abx = b.0 - a.0;
    let aby = b.1 - a.1;
    let len_sq = abx * abx + aby * aby;
    if len_sq <= 1e-6 {
        return a;
    }
    let t = (((pt.0 - a.0) * abx + (pt.1 - a.1) * aby) / len_sq).clamp(0.0, 1.0);
    (a.0 + t * abx, a.1 + t * aby)
}

fn cross_2d(u: (f32, f32), v: (f32, f32)) -> f32 {
    u.0 * v.1 - u.1 * v.0
}

fn segments_properly_intersect(
    p0: (f32, f32),
    p1: (f32, f32),
    q0: (f32, f32),
    q1: (f32, f32),
) -> bool {
    let r = (p1.0 - p0.0, p1.1 - p0.1);
    let s = (q1.0 - q0.0, q1.1 - q0.1);
    let denom = cross_2d(r, s);
    if denom.abs() <= 1e-6 {
        return false;
    }
    let qp = (q0.0 - p0.0, q0.1 - p0.1);
    let t = cross_2d(qp, s) / denom;
    let u = cross_2d(qp, r) / denom;
    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

fn segment_to_segment_dist(
    p0: (f32, f32),
    p1: (f32, f32),
    q0: (f32, f32),
    q1: (f32, f32),
) -> f32 {
    if segments_properly_intersect(p0, p1, q0, q1) {
        return 0.0;
    }
    point_to_segment_dist(p0, q0, q1)
        .min(point_to_segment_dist(p1, q0, q1))
        .min(point_to_segment_dist(q0, p0, p1))
        .min(point_to_segment_dist(q1, p0, p1))
}

fn ridges_clear_sight(ridges: &[BasaltRidge], p0: (f32, f32), p1: (f32, f32)) -> bool {
    ridges
        .iter()
        .all(|r| segment_to_segment_dist(p0, p1, r.a, r.b) > r.thickness)
}

fn ray_segment_distance(
    origin: (f32, f32),
    dir: (f32, f32),
    a: (f32, f32),
    b: (f32, f32),
) -> Option<f32> {
    let s = (b.0 - a.0, b.1 - a.1);
    let denom = cross_2d(dir, s);
    if denom.abs() <= 1e-6 {
        return None;
    }
    let w = (a.0 - origin.0, a.1 - origin.1);
    let t = cross_2d(w, s) / denom;
    if t < 0.05 {
        return None;
    }
    let u = cross_2d(w, dir) / denom;
    let pad = 0.52 / s.0.hypot(s.1).max(1e-4);
    ((-pad)..=(1.0 + pad)).contains(&u).then_some(t)
}

fn compute_corridor_weight_grid(
    shelters: &[ThermalShelter],
    springs: &[TerrariumSpring],
    arena_radius: f32,
) -> [[f32; MARK_GRID_DIM]; MARK_GRID_DIM] {
    let mut weights = [[0.0_f32; MARK_GRID_DIM]; MARK_GRID_DIM];
    for (r, row) in weights.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let pt = cell_center_world(r, c, arena_radius);
            let mut min_dist = f32::MAX;
            for spring in springs {
                // Connect each spring to its nearest shelter as a primary foraging corridor.
                if let Some(nearest_shelter) = shelters.iter().min_by(|s1, s2| {
                    let d1 = (s1.pos.0 - spring.pos.0).hypot(s1.pos.1 - spring.pos.1);
                    let d2 = (s2.pos.0 - spring.pos.0).hypot(s2.pos.1 - spring.pos.1);
                    d1.total_cmp(&d2)
                }) {
                    let d = point_to_segment_dist(pt, nearest_shelter.pos, spring.pos);
                    min_dist = min_dist.min(d);
                }
            }
            if min_dist <= 3.6 {
                let norm = min_dist / 3.6;
                *cell = (1.0 - norm * norm).clamp(0.0, 1.0);
            }
        }
    }
    weights
}
