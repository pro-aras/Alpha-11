//! Continuous 2D expanding wave-ring and directional time-of-flight echo physics.
//!
//! Enforces strict causal purity: emitted wave packets NEVER inject immediate `t+0`
//! or `t+1` motor self-copy into the receiver array. Signals appear in the directional
//! echo channels ONLY when real external matter reflects a wave packet back `2..=14`
//! ticks later (or when a distinct peer organism's call arrives after finite travel time).

use std::f32::consts::TAU;

/// Default finite acoustic / wave propagation speed in arena units per tick.
pub const DEFAULT_WAVE_SPEED: f32 = 2.2;

/// Minimum actuator magnitude on the pulse emitter channel required to launch a wave packet.
pub const EMIT_PULSE_THRESHOLD: f32 = 0.12;

/// Minimum round-trip echo delay in ticks (strictly `>= 2`, preventing `t+0`/`t+1` self-copy).
pub const MIN_ECHO_DELAY_TICKS: u64 = 2;

/// Maximum round-trip echo delay in ticks.
pub const MAX_ECHO_DELAY_TICKS: u64 = 14;

/// Default maximum one-way sensing range for wave reflections.
pub const DEFAULT_MAX_ECHO_RANGE: f32 = 16.0;

/// Distance attenuation coefficient in `1.0 / (1.0 + 0.18 * dist)`.
pub const DISTANCE_ATTENUATION_COEFF: f32 = 0.18;

/// Ring buffer slot count (power of two `>= MAX_ECHO_DELAY_TICKS + 2`).
const RING_SLOTS: usize = 16;

/// Physical material class of an external wave reflector, determining reflection phase
/// polarity and angular harmonic timbre without exposing symbolic labels to the organism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReflectorKind {
    /// Soft nourishing thermal/nutrient matter: positive phase, warm broad lobe.
    NourishingMatter,
    /// Hard rocky obstacle/hazard: inverted phase, crisp high-frequency edge ripple.
    HardObstacle,
    /// Hard circular arena wall: inverted phase, broad firm boundary reflection.
    ArenaWall,
    /// Resonant mechanical valve: tone-sensitive harmonic reflection.
    ResonantValve,
    /// Warm thermal shelter structure: gentle positive phase reflection.
    ThermalShelter,
    /// Persistent physical/chemical terrain mark: textured positive reflection.
    SurfaceMark,
    /// Soft biological body of a peer organism: warm biological echo.
    PeerOrganism,
    /// Direct one-way acoustic wave arriving from a distinct peer organism (`peer != self`).
    DirectPeerCall,
}

/// Point/surface reflector descriptor sampled during wave emission.
#[derive(Clone, Copy, Debug)]
pub struct Reflector {
    /// Continuous 2D world coordinates `(x, y)` of the reflecting surface/body.
    pub pos: (f32, f32),
    /// Intrinsic reflectivity scalar in `[0.0, 1.5]`.
    pub reflectivity: f32,
    /// Physical material impedance/timbre class.
    pub kind: ReflectorKind,
}

#[derive(Clone, Copy, Debug)]
struct ScheduledArrival {
    arrival_tick: u64,
    source_pos: (f32, f32),
    signed_amplitude: f32,
    tone: f32,
    kind: ReflectorKind,
}

/// Computes exact round-trip time-of-flight delay `\Delta = clamp(round(2 * dist / c_wave), 2, 14)`.
#[inline]
#[must_use]
pub fn compute_echo_delay_ticks(dist: f32, c_wave: f32) -> u64 {
    let safe_dist = if dist.is_finite() { dist.max(0.0) } else { 0.0 };
    let safe_c = if c_wave.is_finite() {
        c_wave.max(1e-4)
    } else {
        DEFAULT_WAVE_SPEED
    };
    let raw = (2.0 * safe_dist / safe_c).round() as u64;
    raw.clamp(MIN_ECHO_DELAY_TICKS, MAX_ECHO_DELAY_TICKS)
}

/// Computes one-way acoustic propagation delay between two distinct organisms (`1..=14` ticks).
#[inline]
#[must_use]
pub fn compute_one_way_delay_ticks(dist: f32, c_wave: f32) -> u64 {
    let safe_dist = if dist.is_finite() { dist.max(0.0) } else { 0.0 };
    let safe_c = if c_wave.is_finite() {
        c_wave.max(1e-4)
    } else {
        DEFAULT_WAVE_SPEED
    };
    let raw = (safe_dist / safe_c).round() as u64;
    raw.clamp(1, MAX_ECHO_DELAY_TICKS)
}

/// Computes radial wave amplitude attenuation `1.0 / (1.0 + 0.18 * dist)`.
#[inline]
#[must_use]
pub fn compute_distance_attenuation(dist: f32) -> f32 {
    let safe_dist = if dist.is_finite() { dist.max(0.0) } else { 0.0 };
    1.0 / (1.0 + DISTANCE_ATTENUATION_COEFF * safe_dist)
}

/// Per-receiver wave-echo arrival ring and directional hydrophone/sonar array.
#[derive(Clone, Debug)]
pub struct WaveEchoRing {
    c_wave: f32,
    max_range: f32,
    dir_cos: Vec<f32>,
    dir_sin: Vec<f32>,
    slots: [Vec<ScheduledArrival>; RING_SLOTS],
}

impl WaveEchoRing {
    /// Creates a directional receiver array with `num_channels` evenly spaced over `[0, 2\pi)`.
    #[must_use]
    pub fn new(num_channels: usize, c_wave: f32, max_range: f32) -> Self {
        let n = num_channels.max(1);
        let mut dir_cos = Vec::with_capacity(n);
        let mut dir_sin = Vec::with_capacity(n);
        for k in 0..n {
            let theta = (k as f32) * TAU / (n as f32);
            dir_cos.push(theta.cos());
            dir_sin.push(theta.sin());
        }
        Self {
            c_wave: c_wave.max(0.1),
            max_range: max_range.max(1.0),
            dir_cos,
            dir_sin,
            slots: std::array::from_fn(|_| Vec::with_capacity(16)),
        }
    }

    /// Wave propagation speed in units per tick.
    #[must_use]
    pub fn wave_speed(&self) -> f32 {
        self.c_wave
    }

    /// Maximum one-way sensing range for reflections.
    #[must_use]
    pub fn max_range(&self) -> f32 {
        self.max_range
    }

    /// Number of directional receiver channels.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.dir_cos.len()
    }

    /// Emits an outgoing wave packet at `emit_tick` from `emitter_pos` if `|emit_pulse| > 0.12`.
    ///
    /// Returns `true` if a pulse was physically emitted (`|emit_pulse| > EMIT_PULSE_THRESHOLD`).
    /// If `reflectors` has no entries within `max_range`, zero arrivals are enqueued (empty void
    /// produces strictly `0.0` across all future ticks).
    pub fn emit_pulse(
        &mut self,
        emit_tick: u64,
        emitter_pos: (f32, f32),
        emit_pulse: f32,
        emit_tone: f32,
        reflectors: &[Reflector],
    ) -> bool {
        if !emit_pulse.is_finite() || emit_pulse.abs() <= EMIT_PULSE_THRESHOLD {
            return false;
        }
        let clamped_pulse = emit_pulse.clamp(-1.0, 1.0);
        let clamped_tone = if emit_tone.is_finite() {
            emit_tone.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        for refl in reflectors {
            if refl.reflectivity <= 0.0 || !refl.reflectivity.is_finite() {
                continue;
            }
            let dx = refl.pos.0 - emitter_pos.0;
            let dy = refl.pos.1 - emitter_pos.1;
            let dist = dx.hypot(dy);
            if !dist.is_finite() || dist > self.max_range {
                continue;
            }

            let delay = compute_echo_delay_ticks(dist, self.c_wave);
            let arrival_tick = emit_tick + delay;
            let attenuation = compute_distance_attenuation(dist);
            let signed_amplitude = clamped_pulse * refl.reflectivity.min(1.5) * attenuation;

            let slot_idx = (arrival_tick as usize) & (RING_SLOTS - 1);
            self.slots[slot_idx].push(ScheduledArrival {
                arrival_tick,
                source_pos: refl.pos,
                signed_amplitude,
                tone: clamped_tone,
                kind: refl.kind,
            });
        }

        true
    }

    /// Enqueues a one-way acoustic call emitted by a distinct peer organism (`peer != receiver`).
    pub fn enqueue_peer_call(
        &mut self,
        emit_tick: u64,
        sender_pos: (f32, f32),
        receiver_pos_approx: (f32, f32),
        emit_pulse: f32,
        emit_tone: f32,
    ) {
        if !emit_pulse.is_finite() || emit_pulse.abs() <= EMIT_PULSE_THRESHOLD {
            return;
        }
        let dx = sender_pos.0 - receiver_pos_approx.0;
        let dy = sender_pos.1 - receiver_pos_approx.1;
        let dist = dx.hypot(dy);
        if !dist.is_finite() || dist > self.max_range * 1.25 {
            return;
        }
        let delay = compute_one_way_delay_ticks(dist, self.c_wave);
        let arrival_tick = emit_tick + delay;
        let attenuation = compute_distance_attenuation(dist);
        let slot_idx = (arrival_tick as usize) & (RING_SLOTS - 1);
        self.slots[slot_idx].push(ScheduledArrival {
            arrival_tick,
            source_pos: sender_pos,
            signed_amplitude: emit_pulse.clamp(-1.0, 1.0) * 0.9 * attenuation,
            tone: if emit_tone.is_finite() {
                emit_tone.clamp(-1.0, 1.0)
            } else {
                0.0
            },
            kind: ReflectorKind::DirectPeerCall,
        });
    }

    /// Renders all wave arrivals scheduled for `current_tick` onto `out_channels` around
    /// `receiver_pos`, clears consumed arrivals, and returns total `L2` echo energy received.
    pub fn receive_at_tick(
        &mut self,
        current_tick: u64,
        receiver_pos: (f32, f32),
        out_channels: &mut [f32],
    ) -> f32 {
        out_channels.fill(0.0);
        let n = out_channels.len().min(self.dir_cos.len());
        if n == 0 {
            return 0.0;
        }

        let slot_idx = (current_tick as usize) & (RING_SLOTS - 1);
        let slot = &mut self.slots[slot_idx];
        if slot.is_empty() {
            return 0.0;
        }

        let mut idx = 0;
        while idx < slot.len() {
            if slot[idx].arrival_tick == current_tick {
                let arrival = slot.swap_remove(idx);
                accumulate_directional_arrival(
                    &arrival,
                    receiver_pos,
                    &self.dir_cos[..n],
                    &self.dir_sin[..n],
                    &mut out_channels[..n],
                );
            } else if slot[idx].arrival_tick < current_tick {
                slot.swap_remove(idx);
            } else {
                idx += 1;
            }
        }

        let mut energy = 0.0_f32;
        for ch in &mut out_channels[..n] {
            *ch = ch.clamp(-1.0, 1.0);
            energy += (*ch) * (*ch);
        }
        energy
    }
}

fn accumulate_directional_arrival(
    arrival: &ScheduledArrival,
    receiver_pos: (f32, f32),
    dir_cos: &[f32],
    dir_sin: &[f32],
    out_channels: &mut [f32],
) {
    let dx = arrival.source_pos.0 - receiver_pos.0;
    let dy = arrival.source_pos.1 - receiver_pos.1;
    let dist = dx.hypot(dy);
    let (ux, uy, isotropic) = if dist > 1e-5 {
        (dx / dist, dy / dist, false)
    } else {
        (1.0, 0.0, true)
    };

    let amp = arrival.signed_amplitude;
    let tone = arrival.tone;

    for k in 0..out_channels.len() {
        let cos_diff = if isotropic {
            0.6
        } else {
            (ux * dir_cos[k] + uy * dir_sin[k]).clamp(-1.0, 1.0)
        };
        let cos_pos = cos_diff.max(0.0);
        if cos_pos <= 0.0 {
            continue;
        }

        // Chebyshev harmonics: cos(2x) = 2 cos^2(x) - 1, cos(3x) = 4 cos^3(x) - 3 cos(x).
        let cos_2 = 2.0 * cos_diff * cos_diff - 1.0;
        let cos_3 = 4.0 * cos_diff * cos_diff * cos_diff - 3.0 * cos_diff;

        let gain = match arrival.kind {
            ReflectorKind::NourishingMatter => {
                // Warm in-phase fundamental with smooth wide lobe.
                cos_pos * cos_pos * (0.88 + 0.12 * tone)
            }
            ReflectorKind::HardObstacle => {
                // Phase-inverted sharp reflection with second-harmonic edge contrast.
                -cos_pos * cos_pos * cos_pos * (0.78 + 0.22 * cos_2)
            }
            ReflectorKind::ArenaWall => {
                // Phase-inverted firm boundary reflection.
                -0.85 * cos_pos * cos_pos * (0.82 + 0.18 * cos_2)
            }
            ReflectorKind::ResonantValve => {
                // Tone-sensitive metallic third-harmonic resonance.
                let tone_res = 0.72 + 0.28 * (tone * std::f32::consts::PI).cos();
                cos_pos * cos_pos * cos_pos * tone_res * (0.75 + 0.25 * cos_3)
            }
            ReflectorKind::ThermalShelter => 0.72 * cos_pos * cos_pos,
            ReflectorKind::SurfaceMark => 0.58 * cos_pos * cos_pos * cos_pos,
            ReflectorKind::PeerOrganism => {
                0.82 * cos_pos * cos_pos * (0.86 + 0.14 * (tone * 1.5).sin())
            }
            ReflectorKind::DirectPeerCall => 0.92 * cos_pos * cos_pos * (0.82 + 0.18 * tone),
        };

        out_channels[k] += amp * gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_void_emits_zero_self_copy() {
        let mut ring = WaveEchoRing::new(32, DEFAULT_WAVE_SPEED, DEFAULT_MAX_ECHO_RANGE);
        let mut buf = [0.0_f32; 32];
        let emitted = ring.emit_pulse(0, (0.0, 0.0), 1.0, 0.5, &[]);
        assert!(emitted);
        for tick in 1..=20 {
            let energy = ring.receive_at_tick(tick, (0.0, 0.0), &mut buf);
            assert_eq!(energy, 0.0);
            assert!(buf.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn exact_time_of_flight_and_attenuation() {
        let mut ring = WaveEchoRing::new(32, 2.2, 16.0);
        let mut buf = [0.0_f32; 32];
        let d = 5.5_f32; // 2 * 5.5 / 2.2 = 5 ticks
        let expected_delay = compute_echo_delay_ticks(d, 2.2);
        assert_eq!(expected_delay, 5);

        let reflectors = [Reflector {
            pos: (d, 0.0),
            reflectivity: 1.0,
            kind: ReflectorKind::NourishingMatter,
        }];
        assert!(ring.emit_pulse(10, (0.0, 0.0), 0.9, 0.0, &reflectors));

        for tick in 11..15 {
            let e = ring.receive_at_tick(tick, (0.0, 0.0), &mut buf);
            assert_eq!(e, 0.0, "must be silent before arrival tick 15");
        }
        let arrival_energy = ring.receive_at_tick(15, (0.0, 0.0), &mut buf);
        assert!(arrival_energy > 0.0, "echo must arrive at t0 + delta");
        assert!(buf[0] > 0.2, "channel 0 points along +x toward (d, 0)");
        assert_eq!(buf[16], 0.0, "opposite channel 16 (-x) must be silent");
    }
}
