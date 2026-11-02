//! Unified continuous 2D Photonic + Wave-Echo + Thermal Arena (`AdaptiveCaveWorld`).
//!
//! Exposes `BrightRadiantWorld` (`lux = 1.0`), `PitchBlackEchoCave` (`lux = 0.0`),
//! and `DenseFieldWorld` (configurable up to `4,096` channels for `O(1)` latency stress testing).

use std::f32::consts::TAU;
use std::ops::{Deref, DerefMut, Range};

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::rng::{DeterministicRng, sanitize_actuator};
use crate::wavefield::{
    DEFAULT_MAX_ECHO_RANGE, DEFAULT_WAVE_SPEED, Reflector, ReflectorKind, WaveEchoRing,
};

/// Default ambient channel count (`64` photonic + `32` wave-echo + `32` exogenous noise).
pub const DEFAULT_CAVE_CHANNELS: usize = 128;

/// Maximum dense channel count for `O(1)` perceptual scaling verification.
pub const MAX_DENSE_CAVE_CHANNELS: usize = 4096;

/// Continuous actuator count for cave environments:
/// `[fx, fy, gx, gy, emit_pulse, emit_tone]`.
pub const CAVE_ACTUATORS: usize = 6;

/// Default circular arena radius in continuous world units.
pub const DEFAULT_ARENA_RADIUS: f32 = 20.0;

const BODY_RADIUS: f32 = 0.6;
const MAX_LOCOMOTION_THRUST: f32 = 0.22;
const VISCOUS_DRAG: f32 = 0.82;
const VALVE_BOOST_DURATION: u32 = 80;

/// Read-only external observer metrics for `AdaptiveCaveWorld` (never exposed in `FieldView`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CaveMetrics {
    pub tick: u64,
    pub total_metabolic_influx: f32,
    pub total_tissue_stress: f32,
    pub springs_depleted: u64,
    pub wall_impacts: u64,
    pub active_pings_emitted: u64,
    pub echo_energy_received: f32,
    pub distance_travelled: f32,
    pub position: (f32, f32),
    pub illumination_lux: f32,
    pub causal_reversed: bool,
    pub valve_activations: u64,
    pub valve_boost_ticks_remaining: u32,
}

impl CaveMetrics {
    /// Canonical ambient channel band ranges `(Photonic, Echo, NoisyTV)` for the 128-channel layout.
    #[must_use]
    pub fn channel_band_ranges() -> (Range<usize>, Range<usize>, Range<usize>) {
        (0..64, 64..96, 96..128)
    }
}

/// Regenerating thermal/nutrient vent in the continuous 2D arena.
#[derive(Clone, Copy, Debug)]
pub struct ThermalVent {
    pub pos: (f32, f32),
    pub reservoir: f32,
    pub core_radius: f32,
    pub halo_radius: f32,
    /// `false` = primary nourishing vent (pre-reversal), `true` = secondary noxious vent (nourishing post-reversal).
    pub inverted_polarity: bool,
    depleted_latched: bool,
}

/// Solid occluding rock obstacle in the continuous 2D arena.
#[derive(Clone, Copy, Debug)]
pub struct SolidObstacle {
    pub pos: (f32, f32),
    pub radius: f32,
}

/// Resonant mechanical valve that boosts nearby spring flow for `80` ticks when manipulated.
#[derive(Clone, Copy, Debug)]
pub struct ResonantValve {
    pub pos: (f32, f32),
    pub interact_radius: f32,
    pub boost_ticks_remaining: u32,
}

/// Continuous 2D Photonic + Wave-Echo + Thermal Arena implementing [`WorldMedium`].
#[derive(Clone, Debug)]
pub struct AdaptiveCaveWorld {
    arena_radius: f32,
    pos: (f32, f32),
    vel: (f32, f32),
    illumination_lux: f32,
    causal_reversed: bool,

    vents: Vec<ThermalVent>,
    obstacles: Vec<SolidObstacle>,
    valve: Option<ResonantValve>,

    photonic_end: usize,
    echo_end: usize,
    photonic_rows: usize,
    photonic_cols: usize,
    col_cos: Vec<f32>,
    col_sin: Vec<f32>,
    row_depths: Vec<f32>,

    echo_ring: WaveEchoRing,
    reflector_scratch: Vec<Reflector>,
    world_rng: DeterministicRng,
    noisy_rng: DeterministicRng,

    ambient: Vec<f32>,
    last_metabolic_influx: f32,
    last_tissue_stress: f32,
    metrics: CaveMetrics,
}

impl AdaptiveCaveWorld {
    /// Constructs a 128-channel cave world at the specified `illumination_lux` in `[0.0, 1.0]`.
    #[must_use]
    pub fn new(seed: u64, illumination_lux: f32) -> Self {
        Self::with_config(seed, illumination_lux, DEFAULT_CAVE_CHANNELS)
    }

    /// Constructs `BrightRadiantWorld` (`illumination_lux = 1.0`, `128` channels).
    #[must_use]
    pub fn bright_radiant(seed: u64) -> Self {
        Self::with_config(seed, 1.0, DEFAULT_CAVE_CHANNELS)
    }

    /// Constructs `PitchBlackEchoCave` (`illumination_lux = 0.0`, `128` channels).
    #[must_use]
    pub fn pitch_black_echo_cave(seed: u64) -> Self {
        Self::with_config(seed, 0.0, DEFAULT_CAVE_CHANNELS)
    }

    /// Constructs a high-dimensional `DenseFieldWorld` (`128..=4096` channels) for `O(1)` latency testing.
    #[must_use]
    pub fn dense_field(seed: u64, num_channels: usize) -> Self {
        let clamped_channels = num_channels.clamp(DEFAULT_CAVE_CHANNELS, MAX_DENSE_CAVE_CHANNELS);
        Self::with_config(seed, 1.0, clamped_channels)
    }

    /// Constructs a controlled cave arena containing only explicit reflectors and custom arena radius
    /// (for exact isolated time-of-flight and attenuation physics verification).
    #[must_use]
    pub fn with_isolated_reflectors(
        seed: u64,
        illumination_lux: f32,
        arena_radius: f32,
        reflectors: &[(f32, f32, ReflectorKind)],
    ) -> Self {
        let mut world = Self::with_config(seed, illumination_lux, DEFAULT_CAVE_CHANNELS);
        world.arena_radius = arena_radius.max(4.0);
        world.vents.clear();
        world.obstacles.clear();
        world.valve = None;

        for &(rx, ry, kind) in reflectors {
            match kind {
                ReflectorKind::NourishingMatter => world.vents.push(ThermalVent {
                    pos: (rx, ry),
                    reservoir: 1.0,
                    core_radius: 1.6,
                    halo_radius: 3.5,
                    inverted_polarity: false,
                    depleted_latched: false,
                }),
                ReflectorKind::ResonantValve => {
                    world.valve = Some(ResonantValve {
                        pos: (rx, ry),
                        interact_radius: 2.0,
                        boost_ticks_remaining: 0,
                    });
                }
                _ => world.obstacles.push(SolidObstacle {
                    pos: (rx, ry),
                    radius: 1.0,
                }),
            }
        }
        world.refresh_photonic_band();
        world
    }

    fn with_config(seed: u64, illumination_lux: f32, total_channels: usize) -> Self {
        let mut master_rng = DeterministicRng::new(seed);
        let mut world_rng = master_rng.fork(0xA110_CA7E_0001);
        let noisy_rng = master_rng.fork(0xA110_F015_E002);

        let safe_lux = if illumination_lux.is_finite() {
            illumination_lux.clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Partition channels into (Photonic 50%, Wave-Echo 25%, Noisy TV 25%).
        // At 128 channels this is exact: 0..64, 64..96, 96..128.
        let photonic_rows = if total_channels >= 512 { 16 } else { 8 };
        let raw_photonic = (total_channels / 2).max(64);
        let photonic_cols = (raw_photonic / photonic_rows).max(8);
        let photonic_end = photonic_rows * photonic_cols;
        let remaining = total_channels.saturating_sub(photonic_end);
        let echo_len = (remaining / 2).max(16);
        let echo_end = photonic_end + echo_len;
        let actual_total = total_channels.max(echo_end + 16);

        let mut col_cos = Vec::with_capacity(photonic_cols);
        let mut col_sin = Vec::with_capacity(photonic_cols);
        for c in 0..photonic_cols {
            let theta = ((c as f32) + 0.5) * TAU / (photonic_cols as f32);
            col_cos.push(theta.cos());
            col_sin.push(theta.sin());
        }

        let mut row_depths = Vec::with_capacity(photonic_rows);
        for r in 0..photonic_rows {
            let frac = (r as f32) / ((photonic_rows.saturating_sub(1)).max(1) as f32);
            // Smooth near-to-horizon radial depth sampling in [1.6, 17.6].
            row_depths.push(1.6 + 16.0 * frac * (0.65 + 0.35 * frac));
        }

        // Spawn 4 nourishing thermal springs (including 1 starter spring within gentle thermal halo
        // of origin so newborn 32-cell seeds can ignite) plus 3 opposite-polarity noxious vents.
        let mut vents = Vec::with_capacity(7);
        let starter_angle = world_rng.next_f32_range(0.0, TAU);
        let starter_dist = 2.6_f32;
        vents.push(ThermalVent {
            pos: (
                starter_dist * starter_angle.cos(),
                starter_dist * starter_angle.sin(),
            ),
            reservoir: 1.0,
            core_radius: 1.8,
            halo_radius: 4.2,
            inverted_polarity: false,
            depleted_latched: false,
        });

        for i in 1..4 {
            let base_angle = starter_angle + (i as f32) * TAU / 4.0 + world_rng.next_f32_range(-0.28, 0.28);
            let dist = world_rng.next_f32_range(6.2, 13.5);
            vents.push(ThermalVent {
                pos: (dist * base_angle.cos(), dist * base_angle.sin()),
                reservoir: world_rng.next_f32_range(0.8, 1.0),
                core_radius: 1.85,
                halo_radius: 4.4,
                inverted_polarity: false,
                depleted_latched: false,
            });
        }

        for i in 0..3 {
            let angle = starter_angle + 0.78 + (i as f32) * TAU / 3.0 + world_rng.next_f32_range(-0.2, 0.2);
            let dist = world_rng.next_f32_range(5.5, 13.0);
            vents.push(ThermalVent {
                pos: (dist * angle.cos(), dist * angle.sin()),
                reservoir: 0.95,
                core_radius: 1.65,
                halo_radius: 3.6,
                inverted_polarity: true,
                depleted_latched: false,
            });
        }

        // Spawn 4 solid rocky obstacles that cast geometric shadows and reflect hard echoes.
        let mut obstacles = Vec::with_capacity(4);
        for i in 0..4 {
            let angle = 0.42 + (i as f32) * TAU / 4.0 + world_rng.next_f32_range(-0.22, 0.22);
            let dist = world_rng.next_f32_range(4.8, 11.5);
            obstacles.push(SolidObstacle {
                pos: (dist * angle.cos(), dist * angle.sin()),
                radius: world_rng.next_f32_range(1.0, 1.45),
            });
        }

        let valve_angle = starter_angle + 1.05;
        let valve = Some(ResonantValve {
            pos: (5.4 * valve_angle.cos(), 5.4 * valve_angle.sin()),
            interact_radius: 2.2,
            boost_ticks_remaining: 0,
        });

        let mut cave = Self {
            arena_radius: DEFAULT_ARENA_RADIUS,
            pos: (0.0, 0.0),
            vel: (0.0, 0.0),
            illumination_lux: safe_lux,
            causal_reversed: false,
            vents,
            obstacles,
            valve,
            photonic_end,
            echo_end,
            photonic_rows,
            photonic_cols,
            col_cos,
            col_sin,
            row_depths,
            echo_ring: WaveEchoRing::new(echo_len, DEFAULT_WAVE_SPEED, DEFAULT_MAX_ECHO_RANGE),
            reflector_scratch: Vec::with_capacity(16),
            world_rng,
            noisy_rng,
            ambient: vec![0.0; actual_total],
            last_metabolic_influx: 0.0,
            last_tissue_stress: 0.0,
            metrics: CaveMetrics {
                illumination_lux: safe_lux,
                ..CaveMetrics::default()
            },
        };

        cave.refresh_photonic_band();
        cave.refresh_noisy_tv_band();
        cave
    }

    /// Dynamically sets ambient illumination in `[0.0, 1.0]` without resetting world or organism state.
    ///
    /// Setting `lux = 0.0` immediately transitions Band A (`0..photonic_end`) to complete darkness (`0.0`).
    pub fn set_illumination(&mut self, lux: f32) {
        let safe_lux = if lux.is_finite() {
            lux.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.illumination_lux = safe_lux;
        self.metrics.illumination_lux = safe_lux;
        self.refresh_photonic_band();
    }

    /// Current illumination level in `[0.0, 1.0]`.
    #[must_use]
    pub fn illumination(&self) -> f32 {
        self.illumination_lux
    }

    /// Triggers a mid-life causal reversal without resetting organism or world topology:
    /// inverts locomotion actuator polarity and swaps nourishing vs noxious thermal vent families.
    pub fn trigger_causal_reversal(&mut self) {
        self.causal_reversed = !self.causal_reversed;
        self.metrics.causal_reversed = self.causal_reversed;
        self.refresh_photonic_band();
    }

    /// Read-only external observer metrics snapshot.
    #[must_use]
    pub fn metrics(&self) -> &CaveMetrics {
        &self.metrics
    }

    /// Actual channel band ranges `(Photonic, Echo, NoisyTV)` for this instance's channel count.
    #[must_use]
    pub fn band_ranges(&self) -> (Range<usize>, Range<usize>, Range<usize>) {
        (
            0..self.photonic_end,
            self.photonic_end..self.echo_end,
            self.echo_end..self.ambient.len(),
        )
    }

    /// Canonical `(0..64, 64..96, 96..128)` band ranges helper.
    #[must_use]
    pub fn channel_band_ranges(&self) -> (Range<usize>, Range<usize>, Range<usize>) {
        self.band_ranges()
    }

    /// Read-only view of thermal vents (for external scientific observers only).
    #[must_use]
    pub fn thermal_vents(&self) -> &[ThermalVent] {
        &self.vents
    }

    /// Read-only view of solid obstacles (for external scientific observers only).
    #[must_use]
    pub fn obstacles(&self) -> &[SolidObstacle] {
        &self.obstacles
    }

    /// Read-only view of the resonant mechanical valve, if present.
    #[must_use]
    pub fn valve(&self) -> Option<&ResonantValve> {
        self.valve.as_ref()
    }

    /// Wave propagation speed `c_wave` in arena units per tick.
    #[must_use]
    pub fn wave_speed(&self) -> f32 {
        self.echo_ring.wave_speed()
    }

    /// Euclidean distance from the organism's current position to the closest external reflector
    /// within echo sensing range, if any.
    #[must_use]
    pub fn nearest_reflector_distance(&self) -> Option<f32> {
        let mut reflectors = Vec::with_capacity(16);
        self.collect_reflectors_into(&mut reflectors);
        reflectors
            .iter()
            .map(|r| (r.pos.0 - self.pos.0).hypot(r.pos.1 - self.pos.1))
            .filter(|&d| d <= self.echo_ring.max_range())
            .min_by(|a, b| a.total_cmp(b))
    }

    /// Returns `true` if `vent` currently provides nourishment (respecting `causal_reversed`).
    #[must_use]
    pub fn is_vent_nourishing(&self, vent: &ThermalVent) -> bool {
        vent.inverted_polarity == self.causal_reversed
    }

    fn collect_reflectors_into(&self, out: &mut Vec<Reflector>) {
        out.clear();

        for vent in &self.vents {
            let kind = if !vent.inverted_polarity {
                ReflectorKind::NourishingMatter
            } else {
                ReflectorKind::HardObstacle
            };
            let refl_strength = (0.45 + 0.55 * vent.reservoir).clamp(0.3, 1.0);
            out.push(Reflector {
                pos: vent.pos,
                reflectivity: refl_strength,
                kind,
            });
        }

        for obs in &self.obstacles {
            out.push(Reflector {
                pos: obs.pos,
                reflectivity: 1.0,
                kind: ReflectorKind::HardObstacle,
            });
        }

        if let Some(valve) = &self.valve {
            let boost_bonus = if valve.boost_ticks_remaining > 0 {
                0.25
            } else {
                0.0
            };
            out.push(Reflector {
                pos: valve.pos,
                reflectivity: 0.95 + boost_bonus,
                kind: ReflectorKind::ResonantValve,
            });
        }

        // Nearest point on circular arena wall.
        let r = self.pos.0.hypot(self.pos.1);
        let wall_dist = (self.arena_radius - r).max(0.0);
        if wall_dist <= self.echo_ring.max_range() {
            let wall_pos = if r > 1e-4 {
                (
                    self.arena_radius * self.pos.0 / r,
                    self.arena_radius * self.pos.1 / r,
                )
            } else {
                (self.arena_radius, 0.0)
            };
            out.push(Reflector {
                pos: wall_pos,
                reflectivity: 0.9,
                kind: ReflectorKind::ArenaWall,
            });
        }
    }

    fn refresh_photonic_band(&mut self) {
        let photonic_slice = &mut self.ambient[0..self.photonic_end];
        if self.illumination_lux <= 0.0 {
            photonic_slice.fill(0.0);
            return;
        }

        let lux = self.illumination_lux;
        let (px, py) = self.pos;
        let r_sq = px * px + py * py;
        let arena_r_sq = self.arena_radius * self.arena_radius;

        let n_rows = self.photonic_rows;
        let n_cols = self.photonic_cols;

        for col in 0..n_cols {
            let cx = self.col_cos[col];
            let cy = self.col_sin[col];

            // Exact continuous ray-circle distance to arena boundary along (cx, cy).
            let b = px * cx + py * cy;
            let c_term = r_sq - arena_r_sq;
            let disc = (b * b - c_term).max(0.0);
            let wall_dist = (-b + disc.sqrt()).max(0.2);

            for row in 0..n_rows {
                let depth = self.row_depths[row];
                // Subtle continuous horizon depth gradient modulated by wall proximity.
                let wall_rim = (-0.5 * ((depth - wall_dist) / 2.2).powi(2)).exp();
                let mut ray_val = 0.06 * (1.0 - depth / (self.arena_radius * 1.4)).max(0.0)
                    - 0.16 * wall_rim;

                // Evaluate geometric shadow transmittance from solid obstacles in front of `depth`.
                let mut shadow_transmittance = 1.0_f32;
                for obs in &self.obstacles {
                    let odx = obs.pos.0 - px;
                    let ody = obs.pos.1 - py;
                    let odist = odx.hypot(ody).max(0.25);
                    let align = (odx * cx + ody * cy) / odist;
                    let half_ang = (obs.radius / odist).clamp(0.08, 1.15);
                    let cos_thresh = half_ang.cos();

                    if align > cos_thresh - 0.12 {
                        let ang_factor =
                            ((align - (cos_thresh - 0.12)) / (1.0 - cos_thresh + 0.12)).clamp(0.0, 1.0);
                        // Direct silhouette darkening at obstacle depth.
                        let depth_hit = (-0.5 * ((depth - odist) / 1.6).powi(2)).exp();
                        ray_val -= 0.35 * ang_factor * depth_hit;

                        // Cast occluding shadow behind the obstacle (depth > odist).
                        if depth > odist {
                            let behind = ((depth - odist) / 1.5).clamp(0.0, 1.0);
                            shadow_transmittance *= 1.0 - 0.82 * ang_factor * behind;
                        }
                    }
                }

                // Cast radiant emission from thermal vents (attenuated by obstacle shadows!).
                for vent in &self.vents {
                    let vdx = vent.pos.0 - px;
                    let vdy = vent.pos.1 - py;
                    let vdist = vdx.hypot(vdy).max(0.2);
                    let align = (vdx * cx + vdy * cy) / vdist;
                    if align <= 0.25 {
                        continue;
                    }
                    let ang_sq = (2.0 * (1.0 - align)).max(0.0);
                    let ang_gauss = (-ang_sq / (2.0 * 0.16)).exp();
                    let depth_gauss = (-0.5 * ((depth - vdist) / 3.0).powi(2)).exp();
                    let proximity_gain = 1.0 / (1.0 + 0.06 * vdist);
                    let polarity_sign = if !vent.inverted_polarity { 1.0 } else { -0.75 };
                    let strength =
                        polarity_sign * (0.25 + 0.75 * vent.reservoir) * proximity_gain;

                    ray_val +=
                        shadow_transmittance * strength * ang_gauss * (0.35 + 0.65 * depth_gauss);
                }

                // Resonant mechanical valve shimmer.
                if let Some(valve) = &self.valve {
                    let ldx = valve.pos.0 - px;
                    let ldy = valve.pos.1 - py;
                    let ldist = ldx.hypot(ldy).max(0.2);
                    let align = (ldx * cx + ldy * cy) / ldist;
                    if align > 0.4 {
                        let ang_sq = (2.0 * (1.0 - align)).max(0.0);
                        let ang_gauss = (-ang_sq / (2.0 * 0.10)).exp();
                        let depth_gauss = (-0.5 * ((depth - ldist) / 2.4).powi(2)).exp();
                        let boost_glow = if valve.boost_ticks_remaining > 0 {
                            0.65
                        } else {
                            0.32
                        };
                        ray_val += shadow_transmittance * boost_glow * ang_gauss * depth_gauss;
                    }
                }

                photonic_slice[row * n_cols + col] = (ray_val * lux).clamp(-1.0, 1.0);
            }
        }
    }

    fn refresh_noisy_tv_band(&mut self) {
        let noisy_slice = &mut self.ambient[self.echo_end..];
        // Zero-mean, temporally independent high-variance exogenous distractor noise + flashes.
        // Strictly independent of FieldPulse and organism state!
        let flash_gate = self.noisy_rng.next_f32_unit();
        for ch in noisy_slice.iter_mut() {
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

    fn respawn_depleted_vent(
        vent_pos: &mut (f32, f32),
        org_pos: (f32, f32),
        arena_radius: f32,
        rng: &mut DeterministicRng,
    ) {
        let max_r = (arena_radius - 3.5).max(6.5);
        for _ in 0..16 {
            let angle = rng.next_f32_range(0.0, TAU);
            let dist = rng.next_f32_range(5.5, max_r);
            let candidate = (dist * angle.cos(), dist * angle.sin());
            if (candidate.0 - org_pos.0).hypot(candidate.1 - org_pos.1) > 5.8 {
                *vent_pos = candidate;
                return;
            }
        }
        // Deterministic fallback strictly opposite `org_pos` outside halo radius.
        let r = org_pos.0.hypot(org_pos.1);
        if r > 1e-4 {
            let opp_r = (arena_radius * 0.55).clamp(6.2, max_r);
            *vent_pos = (-org_pos.0 / r * opp_r, -org_pos.1 / r * opp_r);
        } else {
            *vent_pos = (8.5_f32.min(max_r), 0.0);
        }
    }
}

impl WorldMedium for AdaptiveCaveWorld {
    fn num_ambient_channels(&self) -> usize {
        self.ambient.len()
    }

    fn num_actuators(&self) -> usize {
        CAVE_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient,
            metabolic_influx: self.last_metabolic_influx,
            tissue_stress: self.last_tissue_stress,
            num_actuators: CAVE_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let emit_tick = self.metrics.tick;

        let raw_fx = sanitize_actuator(pulse.actuator_out.first().copied());
        let raw_fy = sanitize_actuator(pulse.actuator_out.get(1).copied());
        let gx = sanitize_actuator(pulse.actuator_out.get(2).copied());
        let gy = sanitize_actuator(pulse.actuator_out.get(3).copied());
        let emit_pulse_val = sanitize_actuator(pulse.actuator_out.get(4).copied());
        let emit_tone_val = sanitize_actuator(pulse.actuator_out.get(5).copied());

        let polarity = if self.causal_reversed { -1.0 } else { 1.0 };
        let fx = raw_fx * polarity;
        let fy = raw_fy * polarity;

        // 1. Emit active sonar/wave packet from current position BEFORE advancing tick,
        //    so any reflection arrives at `emit_tick + \Delta` (\Delta in 2..=14).
        let mut reflectors = std::mem::take(&mut self.reflector_scratch);
        self.collect_reflectors_into(&mut reflectors);
        let did_ping = self.echo_ring.emit_pulse(
            emit_tick,
            self.pos,
            emit_pulse_val,
            emit_tone_val,
            &reflectors,
        );
        self.reflector_scratch = reflectors;
        if did_ping {
            self.metrics.active_pings_emitted += 1;
        }

        // 2. Continuous 2D locomotion with viscous drag and soft/hard elastic boundary physics.
        let prev_pos = self.pos;
        let mut vx = (self.vel.0 + fx * MAX_LOCOMOTION_THRUST) * VISCOUS_DRAG;
        let mut vy = (self.vel.1 + fy * MAX_LOCOMOTION_THRUST) * VISCOUS_DRAG;

        // Soft wall cushion near perimeter.
        let wall_limit = (self.arena_radius - BODY_RADIUS).max(1.0);
        let soft_zone = wall_limit - 1.0;
        let pre_r = prev_pos.0.hypot(prev_pos.1);
        if pre_r > soft_zone && pre_r > 1e-5 {
            let cushion = ((pre_r - soft_zone) / 1.0).clamp(0.0, 1.5);
            vx -= 0.06 * cushion * (prev_pos.0 / pre_r);
            vy -= 0.06 * cushion * (prev_pos.1 / pre_r);
        }

        let mut nx = prev_pos.0 + vx;
        let mut ny = prev_pos.1 + vy;
        let mut tick_stress = 0.0_f32;

        // Hard circular wall collision & rebound.
        let new_r = nx.hypot(ny);
        if new_r > wall_limit && new_r > 1e-5 {
            let ux = nx / new_r;
            let uy = ny / new_r;
            let v_outward = (vx * ux + vy * uy).max(0.0);
            nx = ux * wall_limit;
            ny = uy * wall_limit;
            vx -= 1.45 * v_outward * ux;
            vy -= 1.45 * v_outward * uy;
            tick_stress += (0.08 + 0.35 * v_outward).min(0.65);
            self.metrics.wall_impacts += 1;
        }

        // Solid obstacle collision & rebound.
        for obs in &self.obstacles {
            let dx = nx - obs.pos.0;
            let dy = ny - obs.pos.1;
            let dist = dx.hypot(dy);
            let min_sep = obs.radius + BODY_RADIUS;
            if dist < min_sep {
                let (ux, uy) = if dist > 1e-5 {
                    (dx / dist, dy / dist)
                } else {
                    (1.0, 0.0)
                };
                let v_into = (-(vx * ux + vy * uy)).max(0.0);
                nx = obs.pos.0 + ux * min_sep;
                ny = obs.pos.1 + uy * min_sep;
                vx += 1.35 * v_into * ux;
                vy += 1.35 * v_into * uy;
                let penetration = (min_sep - dist).max(0.0);
                tick_stress += (0.06 + 0.28 * v_into + 0.12 * penetration).min(0.55);
                self.metrics.wall_impacts += 1;
            }
        }

        let step_dist = (nx - prev_pos.0).hypot(ny - prev_pos.1);
        self.pos = (nx, ny);
        self.vel = (vx, vy);
        self.metrics.distance_travelled += step_dist;
        self.metrics.position = self.pos;

        // 3. Resonant mechanical valve interaction (actuators 2..4).
        let manip_force = gx.hypot(gy);
        let mut valve_boost_pos: Option<(f32, f32)> = None;
        if let Some(valve) = &mut self.valve {
            if valve.boost_ticks_remaining > 0 {
                valve.boost_ticks_remaining -= 1;
            }
            let dist_to_valve = (self.pos.0 - valve.pos.0).hypot(self.pos.1 - valve.pos.1);
            if dist_to_valve <= valve.interact_radius && manip_force > 0.25 {
                if valve.boost_ticks_remaining == 0 {
                    self.metrics.valve_activations += 1;
                }
                valve.boost_ticks_remaining = VALVE_BOOST_DURATION;
            }
            if valve.boost_ticks_remaining > 0 {
                valve_boost_pos = Some(valve.pos);
            }
            self.metrics.valve_boost_ticks_remaining = valve.boost_ticks_remaining;
        } else {
            self.metrics.valve_boost_ticks_remaining = 0;
        }

        // 4. Regenerating thermal/nutrient springs and noxious vents.
        let mut tick_influx = 0.0_f32;
        let reversed = self.causal_reversed;
        let org_pos = self.pos;
        let arena_r = self.arena_radius;
        for vent in &mut self.vents {
            let nourishing = vent.inverted_polarity == reversed;
            let dist = (org_pos.0 - vent.pos.0).hypot(org_pos.1 - vent.pos.1);
            let valve_flow_multiplier = if let Some(vpos) = valve_boost_pos {
                let dv = (vent.pos.0 - vpos.0).hypot(vent.pos.1 - vpos.1);
                1.0 + 0.85 * (-(dv / 10.5).powi(2)).exp()
            } else {
                1.0
            };

            if nourishing {
                if dist <= vent.core_radius {
                    let norm_d = dist / vent.core_radius.max(0.1);
                    let kernel = 1.0 - 0.45 * norm_d * norm_d;
                    let gain = 0.24 * valve_flow_multiplier * vent.reservoir * kernel;
                    tick_influx += gain;
                    // Deplete local reservoir so camping indefinitely is suboptimal.
                    vent.reservoir = (vent.reservoir - 0.012).max(0.0);
                    if vent.reservoir < 0.05 && !vent.depleted_latched {
                        vent.depleted_latched = true;
                        self.metrics.springs_depleted += 1;
                        Self::respawn_depleted_vent(
                            &mut vent.pos,
                            org_pos,
                            arena_r,
                            &mut self.world_rng,
                        );
                        vent.reservoir = 0.35;
                    }
                } else if dist <= vent.halo_radius {
                    // Gentle thermal scent gradient allowing ~90-tick newborn ignition,
                    // while still depleting & relocating if camped motionless!
                    let span = (vent.halo_radius - vent.core_radius).max(0.2);
                    let u = 1.0 - (dist - vent.core_radius) / span;
                    tick_influx += 0.042 * valve_flow_multiplier * vent.reservoir * u * u;
                    vent.reservoir = (vent.reservoir - 0.011).max(0.0);
                    if vent.reservoir < 0.05 && !vent.depleted_latched {
                        vent.depleted_latched = true;
                        self.metrics.springs_depleted += 1;
                        Self::respawn_depleted_vent(
                            &mut vent.pos,
                            org_pos,
                            arena_r,
                            &mut self.world_rng,
                        );
                        vent.reservoir = 0.35;
                    }
                } else {
                    // Strictly unharvested spring (dist > halo_radius) regenerates smoothly.
                    let regen = 0.0045 * valve_flow_multiplier;
                    vent.reservoir = (vent.reservoir + regen).min(1.0);
                    if vent.reservoir > 0.25 {
                        vent.depleted_latched = false;
                    }
                }
            } else if dist <= vent.core_radius {
                // Noxious thermal vent inflicts mild contact stress.
                let norm_d = 1.0 - (dist / vent.core_radius.max(0.1));
                tick_stress += 0.14 * norm_d;
            }
        }

        // 5. Advance tick and render new sensory frame for `tick = emit_tick + 1`.
        self.metrics.tick = emit_tick + 1;
        let current_tick = self.metrics.tick;

        self.refresh_photonic_band();
        let echo_energy = self.echo_ring.receive_at_tick(
            current_tick,
            self.pos,
            &mut self.ambient[self.photonic_end..self.echo_end],
        );
        self.metrics.echo_energy_received += echo_energy;
        self.refresh_noisy_tv_band();

        self.last_metabolic_influx = tick_influx.max(0.0);
        self.last_tissue_stress = tick_stress.max(0.0);
        self.metrics.total_metabolic_influx += self.last_metabolic_influx;
        self.metrics.total_tissue_stress += self.last_tissue_stress;
    }
}

/// Sunlit continuous 2D radiant + wave-echo arena (`illumination_lux = 1.0`, `128` channels).
#[derive(Clone, Debug)]
pub struct BrightRadiantWorld(pub AdaptiveCaveWorld);

impl BrightRadiantWorld {
    /// Creates a new sunlit 2D radiant arena seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(AdaptiveCaveWorld::bright_radiant(seed))
    }

    /// Unwraps the inner [`AdaptiveCaveWorld`].
    #[must_use]
    pub fn into_inner(self) -> AdaptiveCaveWorld {
        self.0
    }
}

impl Default for BrightRadiantWorld {
    fn default() -> Self {
        Self::new(0xA11_B817)
    }
}

impl Deref for BrightRadiantWorld {
    type Target = AdaptiveCaveWorld;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BrightRadiantWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl WorldMedium for BrightRadiantWorld {
    fn num_ambient_channels(&self) -> usize {
        self.0.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.0.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        self.0.current_view()
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.0.apply_pulse(pulse);
    }
}

/// Pitch-black 2D wave-echo cave (`illumination_lux = 0.0`, Photonic Band A strictly `0.0`).
#[derive(Clone, Debug)]
pub struct PitchBlackEchoCave(pub AdaptiveCaveWorld);

impl PitchBlackEchoCave {
    /// Creates a new pitch-black wave-echo cave seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(AdaptiveCaveWorld::pitch_black_echo_cave(seed))
    }

    /// Creates a pitch-black cave with explicit isolated reflectors (for precision TOF benchmarking).
    #[must_use]
    pub fn with_isolated_reflectors(
        seed: u64,
        arena_radius: f32,
        reflectors: &[(f32, f32, ReflectorKind)],
    ) -> Self {
        Self(AdaptiveCaveWorld::with_isolated_reflectors(
            seed,
            0.0,
            arena_radius,
            reflectors,
        ))
    }

    /// Unwraps the inner [`AdaptiveCaveWorld`].
    #[must_use]
    pub fn into_inner(self) -> AdaptiveCaveWorld {
        self.0
    }
}

impl Default for PitchBlackEchoCave {
    fn default() -> Self {
        Self::new(0xA11_DA8C)
    }
}

impl Deref for PitchBlackEchoCave {
    type Target = AdaptiveCaveWorld;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PitchBlackEchoCave {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl WorldMedium for PitchBlackEchoCave {
    fn num_ambient_channels(&self) -> usize {
        self.0.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.0.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        self.0.current_view()
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.0.apply_pulse(pulse);
    }
}

/// High-channel-count (`4,096` by default) continuous 2D arena for `O(1)` latency stress testing.
#[derive(Clone, Debug)]
pub struct DenseFieldWorld(pub AdaptiveCaveWorld);

impl DenseFieldWorld {
    /// Creates a `4,096`-channel dense field arena seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(AdaptiveCaveWorld::dense_field(
            seed,
            MAX_DENSE_CAVE_CHANNELS,
        ))
    }

    /// Creates a dense field arena with a custom channel count in `128..=4096`.
    #[must_use]
    pub fn with_channels(seed: u64, num_channels: usize) -> Self {
        Self(AdaptiveCaveWorld::dense_field(seed, num_channels))
    }

    /// Unwraps the inner [`AdaptiveCaveWorld`].
    #[must_use]
    pub fn into_inner(self) -> AdaptiveCaveWorld {
        self.0
    }
}

impl Default for DenseFieldWorld {
    fn default() -> Self {
        Self::new(0xA11_4096)
    }
}

impl Deref for DenseFieldWorld {
    type Target = AdaptiveCaveWorld;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DenseFieldWorld {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl WorldMedium for DenseFieldWorld {
    fn num_ambient_channels(&self) -> usize {
        self.0.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.0.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        self.0.current_view()
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.0.apply_pulse(pulse);
    }
}
