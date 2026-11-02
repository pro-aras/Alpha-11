//! 1D Multi-Frequency Traveling Wave & Delayed Phase-Resonance Medium (`HarmonicWaveStream`).
//!
//! Channel layout (`96` ambient channels, `4` continuous actuators):
//! - `0..48`: Coupled 1D multi-frequency traveling waves along a continuous spatial-spectral ribbon.
//! - `48..64`: Actuator-resonant phase feedback arriving with strict `3..=8` tick echo delays.
//! - `64..96`: Exogenous high-variance `Noisy TV` white noise decoupled from actions and nourishment.

use std::f32::consts::{PI, TAU};
use std::ops::Range;

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::rng::{DeterministicRng, sanitize_actuator};

/// Total ambient channels in `HarmonicWaveStream`.
pub const HARMONIC_CHANNELS: usize = 96;

/// Continuous actuators in `HarmonicWaveStream`.
pub const HARMONIC_ACTUATORS: usize = 4;

const TRAVELING_BAND_END: usize = 48;
const RESONANT_ECHO_END: usize = 64;
const ECHO_RING_LEN: usize = 16;
const FEEDING_LOCUS_INDEX: usize = 38;

/// Read-only external observer metrics for `HarmonicWaveStream`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HarmonicMetrics {
    pub tick: u64,
    pub total_metabolic_influx: f32,
    pub total_tissue_stress: f32,
    pub in_phase_locks: u64,
    pub anti_phase_collisions: u64,
    pub echo_energy_received: f32,
    pub causal_reversed: bool,
}

impl HarmonicMetrics {
    /// Canonical channel band ranges `(TravelingWaves, DelayedResonance, NoisyTV)`.
    #[must_use]
    pub fn channel_band_ranges() -> (Range<usize>, Range<usize>, Range<usize>) {
        (
            0..TRAVELING_BAND_END,
            TRAVELING_BAND_END..RESONANT_ECHO_END,
            RESONANT_ECHO_END..HARMONIC_CHANNELS,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct DelayedResonancePacket {
    arrival_tick: u64,
    sub_channel: usize,
    amplitude: f32,
}

/// 1D multi-frequency traveling wave stream implementing [`WorldMedium`].
#[derive(Clone, Debug)]
pub struct HarmonicWaveStream {
    phases: [f32; 3],
    omegas: [f32; 3],
    wavenumbers: [f32; 3],
    mode_weights: [f32; 3],
    coupled_ribbon: [f32; TRAVELING_BAND_END],

    echo_ring: [Vec<DelayedResonancePacket>; ECHO_RING_LEN],
    noisy_rng: DeterministicRng,
    causal_reversed: bool,

    ambient: Vec<f32>,
    last_metabolic_influx: f32,
    last_tissue_stress: f32,
    metrics: HarmonicMetrics,
}

impl HarmonicWaveStream {
    /// Creates a new deterministic `HarmonicWaveStream` from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut master = DeterministicRng::new(seed);
        let mut wave_rng = master.fork(0xA11_FA7E);
        let noisy_rng = master.fork(0xA11_D157);

        let phases = [
            wave_rng.next_f32_range(0.0, TAU),
            wave_rng.next_f32_range(0.0, TAU),
            wave_rng.next_f32_range(0.0, TAU),
        ];
        // Three incommensurate traveling wave frequencies so phase-locking requires true sensory tracking.
        let omegas = [
            0.18 + wave_rng.next_f32_range(-0.015, 0.015),
            0.31 + wave_rng.next_f32_range(-0.02, 0.02),
            0.49 + wave_rng.next_f32_range(-0.025, 0.025),
        ];
        let wavenumbers = [0.135, 0.24, 0.39];
        let mode_weights = [0.55, 0.30, 0.15];

        let mut stream = Self {
            phases,
            omegas,
            wavenumbers,
            mode_weights,
            coupled_ribbon: [0.0; TRAVELING_BAND_END],
            echo_ring: std::array::from_fn(|_| Vec::with_capacity(8)),
            noisy_rng,
            causal_reversed: false,
            ambient: vec![0.0; HARMONIC_CHANNELS],
            last_metabolic_influx: 0.0,
            last_tissue_stress: 0.0,
            metrics: HarmonicMetrics::default(),
        };

        stream.refresh_traveling_ribbon();
        stream.refresh_noisy_tv();
        stream
    }

    /// Reverses traveling wave propagation direction (`0 -> 47` becomes `47 -> 0`) and flips
    /// feeding crest polarity by `\pi` mid-life without resetting state.
    pub fn trigger_causal_reversal(&mut self) {
        self.causal_reversed = !self.causal_reversed;
        self.metrics.causal_reversed = self.causal_reversed;
        self.refresh_traveling_ribbon();
    }

    /// Read-only external observer metrics.
    #[must_use]
    pub fn metrics(&self) -> &HarmonicMetrics {
        &self.metrics
    }

    /// Canonical `(0..48, 48..64, 64..96)` channel band ranges.
    #[must_use]
    pub fn channel_band_ranges(&self) -> (Range<usize>, Range<usize>, Range<usize>) {
        HarmonicMetrics::channel_band_ranges()
    }

    /// Computes instantaneous quadrature crest vector `(cos_crest, sin_crest, harmonic_env)`
    /// at the organism's feeding zone.
    #[must_use]
    pub fn current_feeding_crest(&self) -> (f32, f32, f32) {
        let dir_sign = if self.causal_reversed { -1.0 } else { 1.0 };
        let x = FEEDING_LOCUS_INDEX as f32;
        let mut c_sum = 0.0_f32;
        let mut s_sum = 0.0_f32;
        for m in 0..3 {
            let arg = self.phases[m] - dir_sign * self.wavenumbers[m] * x;
            c_sum += self.mode_weights[m] * arg.cos();
            s_sum += self.mode_weights[m] * arg.sin();
        }
        let polarity = if self.causal_reversed { -1.0 } else { 1.0 };
        let env = (self.phases[1] - dir_sign * self.wavenumbers[1] * x).cos() * polarity;
        (c_sum * polarity, s_sum * polarity, env)
    }

    fn refresh_traveling_ribbon(&mut self) {
        let dir_sign = if self.causal_reversed { -1.0 } else { 1.0 };
        let mut raw = [0.0_f32; TRAVELING_BAND_END];
        for (i, slot) in raw.iter_mut().enumerate() {
            let x = i as f32;
            let mut val = 0.0_f32;
            for m in 0..3 {
                let arg = self.phases[m] - dir_sign * self.wavenumbers[m] * x;
                val += self.mode_weights[m] * arg.cos();
            }
            *slot = val;
        }

        // Gentle local spatial coupling + temporal inertia across the 48-channel 1D ribbon.
        let prev = self.coupled_ribbon;
        for i in 0..TRAVELING_BAND_END {
            let left = prev[i.saturating_sub(1)];
            let right = prev[(i + 1).min(TRAVELING_BAND_END - 1)];
            let laplacian = 0.5 * (left + right) - prev[i];
            let updated = 0.78 * raw[i] + 0.22 * (prev[i] + 0.25 * laplacian);
            self.coupled_ribbon[i] = updated.clamp(-1.0, 1.0);
            self.ambient[i] = self.coupled_ribbon[i];
        }
    }

    fn refresh_noisy_tv(&mut self) {
        let flash_gate = self.noisy_rng.next_f32_unit();
        for ch in &mut self.ambient[RESONANT_ECHO_END..HARMONIC_CHANNELS] {
            let base = self.noisy_rng.next_f32_bipolar();
            let flash = if flash_gate > 0.70 {
                if self.noisy_rng.next_u32() & 1 == 0 {
                    0.95
                } else {
                    -0.95
                }
            } else {
                base
            };
            *ch = (0.65 * base + 0.35 * flash).clamp(-1.0, 1.0);
        }
    }
}

impl Default for HarmonicWaveStream {
    fn default() -> Self {
        Self::new(0xA11_FA7E)
    }
}

impl WorldMedium for HarmonicWaveStream {
    fn num_ambient_channels(&self) -> usize {
        HARMONIC_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        HARMONIC_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient,
            metabolic_influx: self.last_metabolic_influx,
            tissue_stress: self.last_tissue_stress,
            num_actuators: HARMONIC_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let emit_tick = self.metrics.tick;

        let a0 = sanitize_actuator(pulse.actuator_out.first().copied());
        let a1 = sanitize_actuator(pulse.actuator_out.get(1).copied());
        let a2 = sanitize_actuator(pulse.actuator_out.get(2).copied());
        let a3 = sanitize_actuator(pulse.actuator_out.get(3).copied());
        let acts = [a0, a1, a2, a3];

        // 1. Evaluate phase-locking between organism actuators and current traveling crest.
        let (crest_cos, crest_sin, harmonic_env) = self.current_feeding_crest();
        let crest_norm = crest_cos.hypot(crest_sin).max(1e-5);
        let drive_mag = a0.hypot(a1).clamp(0.0, 1.0);

        let mut tick_influx = 0.0_f32;
        let mut tick_stress = 0.0_f32;
        if drive_mag > 0.05 {
            let phase_cos = (a0 * crest_cos + a1 * crest_sin) / (drive_mag * crest_norm);
            // Audit P27: the former `+ 0.5 * a3.abs()` term paid out regardless of wave state (free bonus).
            // Only aperture actuation matched to the real harmonic envelope now modulates intake.
            let aperture_bonus = 1.0 + 0.25 * (a2 * harmonic_env).clamp(-0.3, 0.5);
            if phase_cos > 0.15 {
                tick_influx = 0.24 * drive_mag * (phase_cos - 0.15) * aperture_bonus;
                self.metrics.in_phase_locks += 1;
            } else if phase_cos < -0.25 {
                tick_stress = 0.14 * drive_mag * (-phase_cos - 0.25);
                self.metrics.anti_phase_collisions += 1;
            }
        }

        // 2. Schedule delayed actuator-resonant echoes across delays [3, 4, 6, 8] ticks
        //    into Band B (48..64). Strictly zero t+0 or t+1 self-copy!
        const SUB_DELAYS: [u64; 4] = [3, 4, 6, 8];
        const SUB_ATTEN: [f32; 4] = [0.78, 0.64, 0.50, 0.38];
        let medium_impedance = 0.75 + 0.25 * (crest_cos * PI * 0.5).cos();

        for (act_idx, &act_val) in acts.iter().enumerate() {
            if act_val.abs() <= 0.08 {
                continue;
            }
            for (tap_idx, &delay) in SUB_DELAYS.iter().enumerate() {
                let arrival_tick = emit_tick + delay;
                let slot_idx = (arrival_tick as usize) & (ECHO_RING_LEN - 1);
                let phase_twist = if tap_idx & 1 == 1 { -0.85 } else { 1.0 };
                self.echo_ring[slot_idx].push(DelayedResonancePacket {
                    arrival_tick,
                    sub_channel: act_idx * 4 + tap_idx,
                    amplitude: act_val * SUB_ATTEN[tap_idx] * medium_impedance * phase_twist,
                });
            }
        }

        // 3. Advance traveling wave phases by 1 tick.
        for m in 0..3 {
            self.phases[m] = (self.phases[m] + self.omegas[m]).rem_euclid(TAU);
        }
        self.metrics.tick = emit_tick + 1;
        let current_tick = self.metrics.tick;

        self.refresh_traveling_ribbon();

        // 4. Render arriving delayed resonant feedback in 48..64 for `current_tick`.
        let echo_slice = &mut self.ambient[TRAVELING_BAND_END..RESONANT_ECHO_END];
        echo_slice.fill(0.0);
        let slot_idx = (current_tick as usize) & (ECHO_RING_LEN - 1);
        let slot = &mut self.echo_ring[slot_idx];
        let mut idx = 0;
        while idx < slot.len() {
            if slot[idx].arrival_tick == current_tick {
                let pkt = slot.swap_remove(idx);
                if pkt.sub_channel < echo_slice.len() {
                    echo_slice[pkt.sub_channel] += pkt.amplitude;
                }
            } else if slot[idx].arrival_tick < current_tick {
                slot.swap_remove(idx);
            } else {
                idx += 1;
            }
        }

        let mut echo_energy = 0.0_f32;
        for ch in echo_slice.iter_mut() {
            *ch = ch.clamp(-1.0, 1.0);
            echo_energy += (*ch) * (*ch);
        }
        self.metrics.echo_energy_received += echo_energy;

        // 5. Refresh exogenous Noisy TV channels (64..96).
        self.refresh_noisy_tv();

        self.last_metabolic_influx = tick_influx.max(0.0);
        self.last_tissue_stress = tick_stress.max(0.0);
        self.metrics.total_metabolic_influx += self.last_metabolic_influx;
        self.metrics.total_tissue_stress += self.last_tissue_stress;
    }
}
