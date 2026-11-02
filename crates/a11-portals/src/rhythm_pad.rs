//! Temporal call-and-response acoustic medium (`RhythmEchoPad`) for unforced rhythmic observation.
//!
//! Exposes `128` unlabelled ambient channels (`64` coupled spectral pad channels across `4`
//! acoustic membranes + `32` multi-tap `3..7` tick room-reverberation channels + `32` exogenous
//! Noisy-TV static channels) and `5` continuous actuators (`4` pad strike forces + `1` dampener).
//!
//! Periodically plays a 3-note external rhythmic wave motif (`A -> B -> C`) followed by a quiet
//! listening/response window. Striking in temporal beat phase or motif order excites constructive
//! harmonic interference with strictly zero artificial score `metabolic_influx`.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::{SplitMix64, actuator_at, sanitize_nonneg, sanitize_unit_signed};

/// Number of physical acoustic pads on the circle.
pub const NUM_PADS: usize = 4;
/// Spectral wave channels per pad (`4 * 16 = 64`).
pub const SPECTRAL_PER_PAD: usize = 16;
/// Total coupled spectral channels across all 4 pads.
pub const RHYTHM_SPECTRAL_CHANNELS: usize = NUM_PADS * SPECTRAL_PER_PAD;
/// Delayed room-reverberation echo channels (`3..=7` tick acoustic reflections).
pub const ROOM_REVERB_CHANNELS: usize = 32;
/// Exogenous static distractor channels (`Noisy TV`).
pub const RHYTHM_NOISY_TV_CHANNELS: usize = 32;
/// Total ambient channels exposed by [`RhythmEchoPad`] (`64 + 32 + 32 = 128`).
pub const RHYTHM_PAD_AMBIENT_CHANNELS: usize =
    RHYTHM_SPECTRAL_CHANNELS + ROOM_REVERB_CHANNELS + RHYTHM_NOISY_TV_CHANNELS;
/// Number of continuous actuators (`4` pad strikes + `1` palm dampener).
pub const RHYTHM_PAD_ACTUATORS: usize = 5;

/// Beat period in ticks (`6` ticks per beat).
pub const BEAT_TICKS: u64 = 6;
/// Length of a full call-and-response bar (`3` call beats + `3` quiet response beats = `36` ticks).
pub const CYCLE_TICKS: u64 = BEAT_TICKS * 6;
/// Ticks per bar dedicated to the external 3-note motif (`0..18`).
pub const CALL_WINDOW_TICKS: u64 = BEAT_TICKS * 3;
/// Default flat basal metabolic trickle in standalone portal mode (`>= 0.0`).
pub const DEFAULT_RHYTHM_BASAL_INFLUX: f32 = 0.008;

const STRIKE_THRESHOLD: f32 = 0.20;
const ECHO_RING_LEN: usize = 8;

/// Read-only behavioral observation metrics for [`RhythmEchoPad`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RhythmPadObservation {
    /// Total organism pad strike onsets (`force > 0.20` rising triggers).
    pub total_pad_strikes: u64,
    /// Turn-taking index in `[0.0, 1.0]`: peaks near `1.0` when the organism pauses during
    /// external call motifs and strikes during quiet response windows; drops toward `0.0`
    /// under constant mashing or total silence.
    pub listening_vs_playing_ratio: f32,
    /// Circular Phase Locking Value (PLV) of strike activity relative to the beat period (`[0.0, 1.0]`).
    pub phase_locking_value: f32,
    /// Similarity in `[0.0, 1.0]` between organism response strikes and the preceding `A -> B -> C` motif.
    pub motif_continuation_similarity: f32,
}

/// 4-pad coupled acoustic resonator with `3..7` tick room echoes and call-and-response motifs.
#[derive(Clone, Debug)]
pub struct RhythmEchoPad {
    tick: u64,
    basal_influx: f32,
    rng: SplitMix64,

    /// Repertoire of 4 deterministic 3-note motifs rotated smoothly every 4 bars (`144` ticks).
    motif_bank: [[usize; 3]; 4],
    current_motif: [usize; 3],

    /// Acoustic membrane amplitudes and velocities per pad.
    pad_energy: [f32; NUM_PADS],
    pad_phase: [f32; NUM_PADS],
    /// Sympathetic harmonic priming for constructive motif continuation.
    sympathetic_prime: [f32; NUM_PADS],
    constructive_bloom: f32,
    prev_strike_held: [bool; NUM_PADS],

    /// Circular history buffer of 8 room acoustic bands over the past 8 ticks (for 3..=7 tick echoes).
    room_ring: [[f32; 8]; ECHO_RING_LEN],
    ring_head: usize,

    // Read-only behavioral observation accumulators
    total_pad_strikes: u64,
    response_window_strikes: u64,
    call_ticks_total: u64,
    call_ticks_with_press: u64,

    plv_cos_sum: f64,
    plv_sin_sum: f64,
    plv_samples: u64,

    slot_pad_energy: [[f32; NUM_PADS]; 3],
    completed_motif_score_sum: f64,
    completed_motif_slots: u64,

    ambient_buf: Vec<f32>,
}

impl Default for RhythmEchoPad {
    fn default() -> Self {
        Self::new(0x8974_EC40_0011_A110)
    }
}

impl RhythmEchoPad {
    /// Creates a new `RhythmEchoPad` seeded deterministically by `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_basal_influx(seed, DEFAULT_RHYTHM_BASAL_INFLUX)
    }

    /// Creates a new `RhythmEchoPad` with an explicit constant `basal_influx` (`>= 0.0`).
    #[must_use]
    pub fn with_basal_influx(seed: u64, basal_influx: f32) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut motif_bank = [[0_usize; 3]; 4];
        for (bar_idx, motif) in motif_bank.iter_mut().enumerate() {
            let p0 = ((rng.next_u64() as usize) + bar_idx) % NUM_PADS;
            let p1 = (p0 + 1 + ((rng.next_u64() as usize) % (NUM_PADS - 1))) % NUM_PADS;
            let p2 = (p1 + 1 + ((rng.next_u64() as usize) % (NUM_PADS - 1))) % NUM_PADS;
            *motif = [p0, p1, p2];
        }
        let current_motif = motif_bank[0];

        let mut pad = Self {
            tick: 0,
            basal_influx: sanitize_nonneg(basal_influx),
            rng,
            motif_bank,
            current_motif,
            pad_energy: [0.0; NUM_PADS],
            pad_phase: [0.0; NUM_PADS],
            sympathetic_prime: [0.0; NUM_PADS],
            constructive_bloom: 0.0,
            prev_strike_held: [false; NUM_PADS],
            room_ring: [[0.0; 8]; ECHO_RING_LEN],
            ring_head: 0,
            total_pad_strikes: 0,
            response_window_strikes: 0,
            call_ticks_total: 0,
            call_ticks_with_press: 0,
            plv_cos_sum: 0.0,
            plv_sin_sum: 0.0,
            plv_samples: 0,
            slot_pad_energy: [[0.0; NUM_PADS]; 3],
            completed_motif_score_sum: 0.0,
            completed_motif_slots: 0,
            ambient_buf: vec![0.0; RHYTHM_PAD_AMBIENT_CHANNELS],
        };

        // Trigger the opening note of bar 0 at tick 0 so view_0 immediately presents call note A.
        pad.trigger_external_call_if_scheduled();
        pad.record_room_snapshot();
        pad.refresh_ambient();
        pad
    }

    /// Current simulation tick of this rhythm pad.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Currently active 3-note motif `[A, B, C]` (`0..4` pad indices) for external analysis/tests.
    #[must_use]
    pub fn active_motif(&self) -> [usize; 3] {
        self.current_motif
    }

    /// Returns `true` if the medium is currently in the external motif call window (`phase < 18`).
    #[must_use]
    pub fn is_in_call_window(&self) -> bool {
        (self.tick % CYCLE_TICKS) < CALL_WINDOW_TICKS
    }

    /// Computes read-only behavioral observation metrics without mutating world state.
    #[must_use]
    pub fn observation(&self) -> RhythmPadObservation {
        let listening_vs_playing_ratio = if self.total_pad_strikes == 0 {
            0.0
        } else {
            let response_share =
                (self.response_window_strikes as f32 / self.total_pad_strikes as f32).clamp(0.0, 1.0);
            let call_silence = if self.call_ticks_total == 0 {
                1.0
            } else {
                (1.0 - (self.call_ticks_with_press as f32 / self.call_ticks_total as f32))
                    .clamp(0.0, 1.0)
            };
            (response_share * call_silence).clamp(0.0, 1.0)
        };

        let phase_locking_value = if self.plv_samples == 0 {
            0.0
        } else {
            let norm = (self.plv_cos_sum.powi(2) + self.plv_sin_sum.powi(2)).sqrt()
                / self.plv_samples as f64;
            (norm as f32).clamp(0.0, 1.0)
        };

        let motif_continuation_similarity = if self.completed_motif_slots == 0 {
            0.0
        } else {
            (self.completed_motif_score_sum / self.completed_motif_slots as f64)
                .clamp(0.0, 1.0) as f32
        };

        RhythmPadObservation {
            total_pad_strikes: self.total_pad_strikes,
            listening_vs_playing_ratio,
            phase_locking_value,
            motif_continuation_similarity,
        }
    }

    fn trigger_external_call_if_scheduled(&mut self) {
        let bar_number = self.tick / CYCLE_TICKS;
        self.current_motif = self.motif_bank[((bar_number / 4) as usize) % self.motif_bank.len()];

        let phase = self.tick % CYCLE_TICKS;
        if phase == 0 {
            let p = self.current_motif[0];
            self.pad_energy[p] = (self.pad_energy[p] + 0.78).min(1.0);
            self.sympathetic_prime[self.current_motif[0]] = 0.85;
        } else if phase == BEAT_TICKS {
            let p = self.current_motif[1];
            self.pad_energy[p] = (self.pad_energy[p] + 0.78).min(1.0);
            self.sympathetic_prime[self.current_motif[1]] = 0.85;
        } else if phase == BEAT_TICKS * 2 {
            let p = self.current_motif[2];
            self.pad_energy[p] = (self.pad_energy[p] + 0.78).min(1.0);
            self.sympathetic_prime[self.current_motif[2]] = 0.85;
        } else if phase == CALL_WINDOW_TICKS {
            // Prime expected continuation note A at start of quiet response window.
            self.sympathetic_prime = [0.0; NUM_PADS];
            self.sympathetic_prime[self.current_motif[0]] = 0.90;
        } else if phase == CALL_WINDOW_TICKS + BEAT_TICKS {
            self.sympathetic_prime[self.current_motif[1]] =
                self.sympathetic_prime[self.current_motif[1]].max(0.75);
        } else if phase == CALL_WINDOW_TICKS + BEAT_TICKS * 2 {
            self.sympathetic_prime[self.current_motif[2]] =
                self.sympathetic_prime[self.current_motif[2]].max(0.75);
        }
    }

    fn finalize_response_bar_metrics(&mut self) {
        for slot in 0..3 {
            let energies = self.slot_pad_energy[slot];
            let total_e: f32 = energies.iter().sum();
            let target_pad = self.current_motif[slot];
            let score = if total_e <= 1e-5 {
                0.0
            } else {
                let target_share = energies[target_pad] / total_e;
                let non_target_struck = (0..NUM_PADS)
                    .filter(|&p| p != target_pad && energies[p] > 0.05)
                    .count();
                let selectivity = 1.0 - (non_target_struck as f32 / (NUM_PADS - 1) as f32);
                (target_share * selectivity).clamp(0.0, 1.0)
            };
            self.completed_motif_score_sum += f64::from(score);
            self.completed_motif_slots = self.completed_motif_slots.saturating_add(1);
        }
        self.slot_pad_energy = [[0.0; NUM_PADS]; 3];
    }

    fn record_room_snapshot(&mut self) {
        let mut snap = [0.0_f32; 8];
        for p in 0..NUM_PADS {
            let e = self.pad_energy[p];
            let ph = self.pad_phase[p];
            snap[p * 2] = sanitize_unit_signed(e * ph.cos());
            snap[p * 2 + 1] = sanitize_unit_signed(e * (2.0 * ph).sin() * (0.6 + 0.4 * self.constructive_bloom));
        }
        self.room_ring[self.ring_head] = snap;
        self.ring_head = (self.ring_head + 1) % ECHO_RING_LEN;
    }

    fn refresh_ambient(&mut self) {
        let beat_theta =
            2.0 * std::f32::consts::PI * ((self.tick % BEAT_TICKS) as f32 / BEAT_TICKS as f32);
        let in_call = self.is_in_call_window();

        // 0..64: 4 pads * 16 spectral harmonic / envelope / beat-resonator channels
        for p in 0..NUM_PADS {
            let base = p * SPECTRAL_PER_PAD;
            let e = self.pad_energy[p];
            let ph = self.pad_phase[p];

            // Fundamental + harmonic overtones enriched by constructive phase/motif bloom
            for h in 0..12 {
                let harmonic_mult = 1.0 + 0.35 * h as f32;
                let decay_h = 1.0 / (1.0 + 0.18 * h as f32);
                let overtone = (ph * harmonic_mult).sin() * decay_h;
                let bloom_boost = 1.0 + 0.45 * self.constructive_bloom;
                self.ambient_buf[base + h] = sanitize_unit_signed(e * overtone * bloom_boost);
            }
            // Envelope, sympathetic string hum, beat clock carrier, and call/quiet room air
            self.ambient_buf[base + 12] = sanitize_unit_signed(e);
            // Channel `base + 13` previously exposed `sympathetic_prime`, i.e. the EXPECTED next motif pad
            // before the organism acted (teacher cue, audit P26). The prime now stays hidden world physics:
            // it only shapes the *consequence* (constructive bloom) after a strike, never a pre-cue.
            self.ambient_buf[base + 13] = 0.0;
            self.ambient_buf[base + 14] = sanitize_unit_signed(0.35 * beat_theta.cos());
            self.ambient_buf[base + 15] = sanitize_unit_signed(if in_call { 0.30 } else { -0.30 });
        }

        // 64..96: Delayed room-reverberation channels across exact 3..=7 tick echoes
        // Tap layout: delay 3 (8 ch), delay 4 (6 ch), delay 5 (6 ch), delay 6 (6 ch), delay 7 (6 ch) = 32 ch
        let tap_specs: [(usize, usize, f32); 5] = [
            (3, 8, 0.75),
            (4, 6, 0.60),
            (5, 6, 0.48),
            (6, 6, 0.38),
            (7, 6, 0.28),
        ];
        let mut out_idx = RHYTHM_SPECTRAL_CHANNELS;
        for (delay, count, gain) in tap_specs {
            // ring_head was already incremented after recording current tick (`delay == 0` would be ring_head - 1)
            let slot_idx = (self.ring_head + ECHO_RING_LEN - 1 - delay) % ECHO_RING_LEN;
            for c in 0..count {
                self.ambient_buf[out_idx] = sanitize_unit_signed(self.room_ring[slot_idx][c] * gain);
                out_idx += 1;
            }
        }

        // 96..128: Exogenous Noisy-TV static channels
        let tv_start = RHYTHM_SPECTRAL_CHANNELS + ROOM_REVERB_CHANNELS;
        for ch in &mut self.ambient_buf[tv_start..RHYTHM_PAD_AMBIENT_CHANNELS] {
            *ch = sanitize_unit_signed(self.rng.next_f32_signed() * 0.85);
        }
    }
}

impl WorldMedium for RhythmEchoPad {
    fn num_ambient_channels(&self) -> usize {
        RHYTHM_PAD_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        RHYTHM_PAD_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.basal_influx,
            tissue_stress: 0.0,
            num_actuators: RHYTHM_PAD_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        // Evaluate pulse against the phase visible in current_view() BEFORE advancing self.tick.
        let phase_before = self.tick % CYCLE_TICKS;
        let in_call_before = phase_before < CALL_WINDOW_TICKS;
        let beat_phase_before = self.tick % BEAT_TICKS;
        let beat_theta = 2.0 * std::f64::consts::PI * (beat_phase_before as f64 / BEAT_TICKS as f64);

        let dampener = actuator_at(pulse, 4).max(0.0);
        let decay = 0.82 * (1.0 - 0.45 * dampener);

        for p in 0..NUM_PADS {
            self.pad_energy[p] *= decay;
            self.sympathetic_prime[p] *= 0.90;
            let omega_p = 0.55 + 0.27 * p as f32;
            self.pad_phase[p] = (self.pad_phase[p] + omega_p).rem_euclid(2.0 * std::f32::consts::PI);
        }
        self.constructive_bloom *= 0.84;

        let mut any_pressed = false;
        for p in 0..NUM_PADS {
            let force = actuator_at(pulse, p);
            let pressed = force > STRIKE_THRESHOLD;
            if pressed {
                any_pressed = true;
            }
            let onset = pressed && !self.prev_strike_held[p];
            self.prev_strike_held[p] = pressed;

            if onset {
                let amp = ((force - STRIKE_THRESHOLD) / (1.0 - STRIKE_THRESHOLD)).clamp(0.0, 1.0);
                self.total_pad_strikes = self.total_pad_strikes.saturating_add(1);

                // Constructive interference peaks when struck in beat phase and matching sympathetic motif order.
                let beat_align = (beat_theta.cos() as f32).max(0.0);
                let motif_align = self.sympathetic_prime[p];
                let constructive = 0.50 * beat_align + 0.50 * motif_align;

                if in_call_before {
                    // Clashing over the external call dampens coherent bloom.
                    self.constructive_bloom = (self.constructive_bloom * 0.5).clamp(0.0, 1.0);
                } else {
                    self.response_window_strikes = self.response_window_strikes.saturating_add(1);
                    self.constructive_bloom =
                        (self.constructive_bloom + 0.45 * amp * constructive).clamp(0.0, 1.0);

                    let slot = ((phase_before - CALL_WINDOW_TICKS) / BEAT_TICKS).min(2) as usize;
                    self.slot_pad_energy[slot][p] += amp;

                    // Chain sympathetic priming to the next note in the motif when played in sequence!
                    if slot < 2 && p == self.current_motif[slot] {
                        let next_pad = self.current_motif[slot + 1];
                        self.sympathetic_prime[next_pad] =
                            (self.sympathetic_prime[next_pad] + 0.65 * amp).min(1.0);
                    }
                }

                self.pad_energy[p] =
                    (self.pad_energy[p] + amp * (0.65 + 0.35 * constructive)).min(1.0);
            }
        }

        if in_call_before {
            self.call_ticks_total = self.call_ticks_total.saturating_add(1);
            if any_pressed {
                self.call_ticks_with_press = self.call_ticks_with_press.saturating_add(1);
            }
        }

        // Record PLV on every active press tick so constant DC mashing cancels across all beat phases,
        // while rhythmic strikes lock cleanly at their beat phase.
        if any_pressed {
            self.plv_cos_sum += beat_theta.cos();
            self.plv_sin_sum += beat_theta.sin();
            self.plv_samples = self.plv_samples.saturating_add(1);
        }

        if phase_before == CYCLE_TICKS - 1 {
            self.finalize_response_bar_metrics();
        }

        // Advance world clock to next tick and trigger scheduled external call notes if any.
        self.tick = self.tick.saturating_add(1);
        self.trigger_external_call_if_scheduled();
        self.record_room_snapshot();
        self.refresh_ambient();
    }
}

// =============================================================================
// Continuous 16 kHz Physical Vocal Tract, 32-Band ERB Cochlea & Binaural Organ
// =============================================================================

/// Physical acoustic sample rate (`16,000 Hz`).
pub const VOCAL_SR_HZ: u32 = 16_000;
/// Acoustic samples synthesized and transduced per organism tick (`20 ms @ 16 kHz = 320 samples`).
pub const VOCAL_FRAME_SAMPLES: usize = 320;
/// Power-of-two radix-2 FFT window length (`512` samples -> `31.25 Hz` bin spacing).
pub const VOCAL_FFT_SIZE: usize = 512;
/// Positive-frequency Nyquist bin count (`0..=256`).
pub const VOCAL_FFT_BINS: usize = (VOCAL_FFT_SIZE / 2) + 1;
/// Number of Equivalent Rectangular Bandwidth (`ERB`) cochlear channels (`80..7,600 Hz`).
pub const VOCAL_ERB_BANDS: usize = 32;
/// Unlabeled continuous auditory + distractor channels exposed by [`BinauralVocalOrgan`].
/// Layout: `0..32` binaural diotic hair-cell sum, `32..64` brainstem `LSO` interaural level contrast,
/// `64..80` smooth `OU(0.97)` Brownian distractor channels.
pub const VOCAL_ORGAN_CHANNELS: usize = 80;
/// Continuous vocal-tract muscle actuators (`lung, larynx, jaw, tongue_body, lips`).
pub const VOCAL_ACTUATORS: usize = 5;

/// Rolling live PCM capture duration in seconds (`10 s = 500 ticks = 160,000 samples`).
pub const VOCAL_RING_SECONDS: usize = 10;
/// Rolling mono PCM sample capacity (`160,000` samples).
pub const VOCAL_RING_SAMPLES: usize = (VOCAL_SR_HZ as usize) * VOCAL_RING_SECONDS;

const VOCAL_SUBSTEPS: usize = 8;
const SAMPLES_PER_SUBSTEP: usize = VOCAL_FRAME_SAMPLES / VOCAL_SUBSTEPS;
const VOICED_RMS_THRESHOLD: f32 = 0.022;
/// Maximum sample-accurate interaural delay (`~0.5 ms = 8 samples @ 16 kHz`).
const MAX_HEAD_ITD_SAMPLES: usize = 8;

const WAV_RIFF: [u8; 4] = [0x52, 0x49, 0x46, 0x46];
const WAV_WAVE: [u8; 4] = [0x57, 0x41, 0x56, 0x45];
const WAV_FMT: [u8; 4] = [0x66, 0x6D, 0x74, 0x20];
const WAV_DATA: [u8; 4] = [0x64, 0x61, 0x74, 0x61];

/// Newborn vocal-tract acoustic length (`8.8 cm`, Fitch & Giedd 1999).
pub const INFANT_VOCAL_TRACT_CM: f32 = 8.8;
/// Mature descended adult vocal-tract acoustic length (`14.5 cm`, Fitch & Giedd 1999).
pub const ADULT_VOCAL_TRACT_CM: f32 = 14.5;

/// Continuous 5-articulator (`lung, larynx, jaw, tongue_body, lips`) 2nd-order muscle
/// and 3-formant (`F1, F2, F3`) `16 kHz` Klatt/Rosenberg physical vocal tract with
/// ontogenetic laryngeal descent (`8.8 cm -> 14.5 cm`).
#[derive(Clone, Debug)]
pub struct PhysicalVocalTract {
    pos: [f32; VOCAL_ACTUATORS],
    vel: [f32; VOCAL_ACTUATORS],
    maturity: f32,
    tract_length_cm: f32,
    pitch_base_hz: f32,
    pitch_span_hz: f32,
    glottal_phase: f32,
    oral_pressure: f32,
    res_y1: [f32; 3],
    res_y2: [f32; 3],
    last_f0_hz: f32,
    last_f1_hz: f32,
    last_f2_hz: f32,
    last_f3_hz: f32,
    last_plosive_burst: f32,
    rng: SplitMix64,
}

impl PhysicalVocalTract {
    /// Creates an infant-scale (`8.8 cm`, `maturity = 0.0`) vocal tract (`F0 ~ 285..415 Hz`).
    #[must_use]
    pub fn new_infant(seed: u64) -> Self {
        Self::with_ontogeny(seed ^ 0x1F4A_7700_A110_0001, 0.0)
    }

    /// Creates an adult-scale (`14.5 cm`, `maturity = 1.0`) vocal tract (`F0 ~ 205..290 Hz`).
    #[must_use]
    pub fn new_adult(seed: u64) -> Self {
        Self::with_ontogeny(seed ^ 0xCA2E_617E_A110_0002, 1.0)
    }

    #[must_use]
    fn with_ontogeny(seed: u64, maturity: f32) -> Self {
        let m = sanitize_nonneg(maturity).min(1.0);
        let pitch_base_hz = 285.0 - 80.0 * m;
        let pitch_span_hz = 130.0 - 45.0 * m;
        let tract_length_cm =
            INFANT_VOCAL_TRACT_CM + (ADULT_VOCAL_TRACT_CM - INFANT_VOCAL_TRACT_CM) * m;
        let mut tract = Self {
            pos: [0.0; VOCAL_ACTUATORS],
            vel: [0.0; VOCAL_ACTUATORS],
            maturity: m,
            tract_length_cm,
            pitch_base_hz,
            pitch_span_hz,
            glottal_phase: 0.0,
            oral_pressure: 0.0,
            res_y1: [0.0; 3],
            res_y2: [0.0; 3],
            last_f0_hz: pitch_base_hz,
            last_f1_hz: 580.0,
            last_f2_hz: 1605.0,
            last_f3_hz: 2880.0,
            last_plosive_burst: 0.0,
            rng: SplitMix64::new(seed),
        };
        tract.update_formant_targets(0.0);
        tract
    }

    /// Current ontogenetic maturity in `[0.0, 1.0]` (`0.0 = 8.8 cm` newborn, `1.0 = 14.5 cm` adult).
    #[must_use]
    pub fn maturity(&self) -> f32 {
        self.maturity
    }

    /// Current physical vocal-tract length in centimeters (`8.8..=14.5 cm`).
    #[must_use]
    pub fn tract_length_cm(&self) -> f32 {
        self.tract_length_cm
    }

    /// Monotonically advances vocal-tract length and laryngeal descent (`8.8 cm -> 14.5 cm`)
    /// from physiological age (`age_ticks`), endogenous cellular growth (`mitosis_births`), and vocal mastery.
    pub fn advance_ontogeny(&mut self, age_ticks: u64, mitosis_births: u64, vocal_mastery: f32) {
        let age_s = (age_ticks.min(1_000_000) as f32) / 6_400.0;
        let age_factor = (1.0 - (-age_s).exp()).clamp(0.0, 1.0);
        let soma = ((mitosis_births.min(10_000) as f32) / 80.0).clamp(0.0, 1.0);
        let mast = (sanitize_nonneg(vocal_mastery) / 0.38).clamp(0.0, 1.0);
        let trophic = (0.50 * soma + 0.50 * mast).clamp(0.0, 1.0);
        let target = (age_factor * trophic).clamp(0.0, 1.0);
        if target > self.maturity {
            self.maturity = target;
            self.tract_length_cm =
                INFANT_VOCAL_TRACT_CM + (ADULT_VOCAL_TRACT_CM - INFANT_VOCAL_TRACT_CM) * target;
            self.pitch_base_hz = 285.0 - 80.0 * target;
            self.pitch_span_hz = 130.0 - 45.0 * target;
        }
    }

    /// Current 2nd-order spring-mass muscle positions in `[-1.0, 1.0]`.
    #[must_use]
    pub fn articulator_positions(&self) -> [f32; VOCAL_ACTUATORS] {
        self.pos
    }

    /// Instantaneous physical chamber resonances `(f0_hz, f1_hz, f2_hz, f3_hz)`.
    #[must_use]
    pub fn current_formants_hz(&self) -> (f32, f32, f32, f32) {
        (
            self.last_f0_hz,
            self.last_f1_hz,
            self.last_f2_hz,
            self.last_f3_hz,
        )
    }

    fn update_formant_targets(&mut self, p_sub: f32) {
        let u_pitch = 0.5 * (self.pos[1] + 1.0);
        let openness = 0.5 * (self.pos[2] + 1.0);
        let frontness = 0.5 * (self.pos[3] + 1.0);
        let tip = 0.5 * (self.pos[4] + 1.0);

        // Laryngeal descent (Fitch & Giedd 1999; Story 2005): lengthening the pharyngeal cavity
        // lowers baseline tube resonances while widening two-tube articulatory formant range.
        let tube_scale = 1.0 - 0.08 * self.maturity;
        let pharyngeal_span = 1.0 + 0.10 * self.maturity;

        let f0 = self.pitch_base_hz + self.pitch_span_hz * u_pitch + 24.0 * p_sub;
        let f1 = ((300.0 + 500.0 * pharyngeal_span * openness) * tube_scale).clamp(280.0, 810.0);
        let f2 = ((940.0 + 1460.0 * pharyngeal_span * frontness - 80.0 * openness) * tube_scale)
            .max(f1 + 240.0)
            .clamp(890.0, 2420.0);
        let f3 = ((2800.0 + 500.0 * tip + 150.0 * frontness) * (1.0 - 0.06 * self.maturity))
            .max(f2 + 380.0)
            .clamp(2680.0, 3550.0);

        self.last_f0_hz = f0;
        self.last_f1_hz = f1;
        self.last_f2_hz = f2;
        self.last_f3_hz = f3;
    }

    /// Integrates 2nd-order overdamped muscle dynamics and synthesizes `320` samples (`20 ms`)
    /// of `16,000 Hz` acoustic pressure bounded in `[-1.0, 1.0]`.
    pub fn synthesize_frame(
        &mut self,
        motor_cmd: &[f32; VOCAL_ACTUATORS],
        out_pcm: &mut [f32; VOCAL_FRAME_SAMPLES],
    ) {
        // Biomechanical 2nd-order overdamped muscle parameters:
        // Respiratory/laryngeal drive (`d = 0, 1`): `omega = 75 rad/s` (~15 ms onset);
        // Supralaryngeal jaw/tongue (`d = 2, 3, 4`): `omega = 48 rad/s` (~25-35 ms glide).
        let dt_sub = 0.020_f32 / (VOCAL_SUBSTEPS as f32);
        let sr = VOCAL_SR_HZ as f32;
        let two_pi_over_sr = 2.0 * std::f32::consts::PI / sr;
        let bws = [75.0_f32, 110.0_f32, 165.0_f32];

        let mut peak_burst = 0.0_f32;
        for sub in 0..VOCAL_SUBSTEPS {
            for d in 0..VOCAL_ACTUATORS {
                let omega = if d < 2 { 75.0_f32 } else { 48.0_f32 };
                let damp = 2.0 * 1.05 * omega;
                let omega_sq = omega * omega;
                let target = sanitize_unit_signed(motor_cmd[d]);
                let acc = omega_sq * (target - self.pos[d]) - damp * self.vel[d];
                self.vel[d] = sanitize_bounded(self.vel[d] + dt_sub * acc, 60.0);
                self.pos[d] = sanitize_unit_signed(self.pos[d] + dt_sub * self.vel[d]);
            }

            // Subglottal phonation threshold (Titze 1988): exact zero drive below threshold.
            let p_sub = if self.pos[0] > 0.06 {
                ((self.pos[0] - 0.06) / 0.94).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.update_formant_targets(p_sub);

            // Consonant constriction & intra-oral pressure buildup/plosive release (`CV` syllables):
            // mandibular/labial seal leverage and pharyngeal cavity compliance grow with laryngeal descent.
            let closure_leverage = 1.0 + 0.28 * self.maturity;
            let jaw_closed = ((-0.12 - self.pos[2]) / 0.58).clamp(0.0, 1.0);
            let lip_constr = ((self.pos[4] - 0.12) / 0.58).clamp(0.0, 1.0);
            let closure =
                (closure_leverage * (0.65 * jaw_closed + 0.55 * lip_constr)).clamp(0.0, 1.0);

            let prev_oral = self.oral_pressure;
            if closure > 0.32 && p_sub > 0.02 {
                let charge_rate = 0.32 + 0.10 * self.maturity;
                self.oral_pressure = (self.oral_pressure + charge_rate * p_sub * closure).min(1.0);
            } else {
                self.oral_pressure *= 0.35;
            }
            let plosive_burst = (prev_oral - self.oral_pressure).max(0.0) * (1.0 - closure);
            if plosive_burst > peak_burst {
                peak_burst = plosive_burst;
            }
            let radiation_open = 1.0 - 0.86 * closure;

            // Equal-peak-gain 2-pole formant resonators so front (high-F2) and back (low-F2)
            // vowels carry unbiased acoustic power at equal subglottal pressure.
            let freqs = [self.last_f1_hz, self.last_f2_hz, self.last_f3_hz];
            let mut c0 = [0.0_f32; 3];
            let mut c1 = [0.0_f32; 3];
            let mut c2 = [0.0_f32; 3];
            for k in 0..3 {
                let w0 = two_pi_over_sr * freqs[k];
                let r = (-std::f32::consts::PI * bws[k] / sr).exp();
                c1[k] = 2.0 * r * w0.cos();
                c2[k] = -(r * r);
                let pole_skirt = (1.0 - 2.0 * r * (2.0 * w0).cos() + r * r).max(1e-6).sqrt();
                c0[k] = 5.2 * (1.0 - r) * pole_skirt;
            }

            let dphi = self.last_f0_hz / sr;
            let base_idx = sub * SAMPLES_PER_SUBSTEP;
            for s in 0..SAMPLES_PER_SUBSTEP {
                let excitation = if p_sub > 0.0 || plosive_burst > 1e-4 {
                    self.glottal_phase += dphi;
                    if self.glottal_phase >= 1.0 {
                        self.glottal_phase -= self.glottal_phase.floor();
                    }
                    let g_deriv = rosenberg_glottal_derivative(self.glottal_phase);
                    let hiss = self.rng.next_f32_signed();
                    p_sub * (0.90 * g_deriv + 0.035 * hiss) + 0.85 * plosive_burst * hiss
                } else {
                    0.0
                };

                // Parallel + cascade 3-formant transfer driven by glottal derivative source.
                let mut par = [0.0_f32; 3];
                for k in 0..3 {
                    let y = c0[k] * excitation + c1[k] * self.res_y1[k] + c2[k] * self.res_y2[k];
                    let y_clean = sanitize_bounded(y, 8.0);
                    self.res_y2[k] = self.res_y1[k];
                    self.res_y1[k] = y_clean;
                    par[k] = y_clean;
                }

                let raw_pressure =
                    radiation_open * (0.56 * par[0] + 0.48 * par[1] + 0.22 * par[2]);
                out_pcm[base_idx + s] = if raw_pressure.abs() < 1e-9 && p_sub == 0.0 {
                    0.0
                } else {
                    sanitize_unit_signed((1.15 * raw_pressure).tanh())
                };
            }
        }
        self.last_plosive_burst = peak_burst;
    }

    /// Peak consonant plosive release burst recorded in the latest `20 ms` frame.
    #[must_use]
    pub fn last_plosive_burst(&self) -> f32 {
        self.last_plosive_burst
    }
}

#[inline]
fn sanitize_bounded(v: f32, limit: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-limit, limit)
    } else {
        0.0
    }
}

/// Exact zero-DC-mean Rosenberg glottal volume-velocity derivative (`int_0^1 g'(phi) dphi == 0`).
#[inline]
fn rosenberg_glottal_derivative(phi: f32) -> f32 {
    const TP: f32 = 0.42;
    const OQ: f32 = 0.65;
    if phi < TP {
        let tau = phi / TP;
        // d/dphi [3*tau^2 - 2*tau^3] = (6*tau - 6*tau^2) / TP
        (6.0 * tau * (1.0 - tau)) / TP * 0.28
    } else if phi < OQ {
        let u = (phi - TP) / (OQ - TP);
        // d/dphi [1 - u^2] = -2*u / (OQ - TP)
        (-2.0 * u) / (OQ - TP) * 0.28
    } else {
        0.0
    }
}

/// Pure-Rust stack-allocated `512`-point in-place Cooley-Tukey radix-2 FFT.
pub fn fft_radix2_512(re: &mut [f32; VOCAL_FFT_SIZE], im: &mut [f32; VOCAL_FFT_SIZE]) {
    // 9-bit bit-reversal permutation (`2^9 = 512`).
    let shift = usize::BITS - 9;
    for i in 0..VOCAL_FFT_SIZE {
        let j = i.reverse_bits() >> shift;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= VOCAL_FFT_SIZE {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / (len as f64);
        let (w_step_sin, w_step_cos) = angle.sin_cos();
        let mut base = 0usize;
        while base < VOCAL_FFT_SIZE {
            let mut wr = 1.0_f64;
            let mut wi = 0.0_f64;
            for k in 0..half {
                let e_idx = base + k;
                let o_idx = e_idx + half;
                let or_val = f64::from(re[o_idx]);
                let oi_val = f64::from(im[o_idx]);
                let tr = (wr * or_val - wi * oi_val) as f32;
                let ti = (wr * oi_val + wi * or_val) as f32;

                let er_val = re[e_idx];
                let ei_val = im[e_idx];
                re[o_idx] = er_val - tr;
                im[o_idx] = ei_val - ti;
                re[e_idx] = er_val + tr;
                im[e_idx] = ei_val + ti;

                let next_wr = wr * w_step_cos - wi * w_step_sin;
                wi = wr * w_step_sin + wi * w_step_cos;
                wr = next_wr;
            }
            base += len;
        }
        len <<= 1;
    }
}

#[inline]
fn hann_fft_power_512_preemph(
    pcm: &[f32; VOCAL_FRAME_SAMPLES],
    preemph: f32,
    power_out: &mut [f32; VOCAL_FFT_BINS],
) {
    let mut re = [0.0_f32; VOCAL_FFT_SIZE];
    let mut im = [0.0_f32; VOCAL_FFT_SIZE];
    let denom = (VOCAL_FRAME_SAMPLES - 1) as f32;
    let mut prev = sanitize_unit_signed(pcm[0]);
    for (i, &sample) in pcm.iter().enumerate() {
        let cur = sanitize_unit_signed(sample);
        let hp = if i == 0 { cur * (1.0 - preemph) } else { cur - preemph * prev };
        prev = cur;
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / denom).cos());
        re[i] = hp * w;
    }
    fft_radix2_512(&mut re, &mut im);
    let inv_n = 1.0 / (VOCAL_FFT_SIZE as f32);
    for k in 0..VOCAL_FFT_BINS {
        let p = (re[k] * re[k] + im[k] * im[k]) * inv_n;
        power_out[k] = if p.is_finite() && p > 0.0 { p } else { 0.0 };
    }
}

/// Measures `(rms, f0_hz, f1_hz, f2_hz)` directly from a raw `16 kHz` `320`-sample audio frame.
#[must_use]
pub fn measure_frame_acoustics(pcm: &[f32; VOCAL_FRAME_SAMPLES]) -> (f32, f32, f32, f32) {
    let mut sum_sq = 0.0_f32;
    for &s in pcm {
        let x = sanitize_unit_signed(s);
        sum_sq += x * x;
    }
    let rms = (sum_sq / (VOCAL_FRAME_SAMPLES as f32)).sqrt();
    if rms < 1e-4 {
        return (rms, 0.0, 0.0, 0.0);
    }

    let mut pow_pre = [0.0_f32; VOCAL_FFT_BINS];
    // Moderate +6 dB/oct acoustic radiation pre-emphasis flattens glottal tilt without over-boosting F3.
    hann_fft_power_512_preemph(pcm, 0.82, &mut pow_pre);
    let bin_hz = (VOCAL_SR_HZ as f32) / (VOCAL_FFT_SIZE as f32); // 31.25 Hz

    // Smooth harmonic comb across +/-4 bins (~250 Hz envelope) to locate true formant envelopes.
    let mut env = [0.0_f32; VOCAL_FFT_BINS];
    for k in 2..(VOCAL_FFT_BINS - 4) {
        let mut s = 0.0_f32;
        let mut wsum = 0.0_f32;
        for d in -4_isize..=4_isize {
            let idx = (k as isize + d).clamp(1, (VOCAL_FFT_BINS - 1) as isize) as usize;
            let w = 5.0 - (d.unsigned_abs() as f32);
            s += w * pow_pre[idx];
            wsum += w;
        }
        env[k] = s / wsum;
    }

    // Fundamental F0 via YIN cumulative mean normalized difference (de Cheveigne & Kawahara 2002, JASA):
    // immune to F1 resonator octave-doubling (`2*F0` cancels in both `d(T0/2)` and `d(T0)`, while odd
    // glottal harmonics keep `d(T0/2) >> d(T0)`). Search lags `34..=88` (`181.8..470.6 Hz` @ `16 kHz`).
    let f0_hz = yin_fundamental_hz(pcm, VOCAL_SR_HZ as f32);

    // Peterson-Barney disjoint formant bands: F1 (`280..812 Hz`), F2 (`875..2,438 Hz`), F3 (`>= 2,780 Hz`).
    let f1_hz = band_peak_centroid_hz(&env, 9, 26, bin_hz);
    let f2_hz = band_peak_centroid_hz(&env, 28, 78, bin_hz);

    (rms, f0_hz, f1_hz, f2_hz)
}

#[inline]
fn yin_fundamental_hz(pcm: &[f32; VOCAL_FRAME_SAMPLES], sr: f32) -> f32 {
    const WIN: usize = 200;
    const MIN_TAU: usize = 34; // 470.6 Hz
    const MAX_TAU: usize = 88; // 181.8 Hz
    const MAX_LAG: usize = MAX_TAU + 2;

    let mut clean = [0.0_f32; VOCAL_FRAME_SAMPLES];
    for (dst, &src) in clean.iter_mut().zip(pcm.iter()) {
        *dst = sanitize_unit_signed(src);
    }

    let mut cmnd = [1.0_f32; MAX_LAG + 1];
    let mut running_sum = 0.0_f32;
    for tau in 1..=MAX_LAG {
        let mut diff = 0.0_f32;
        for j in 0..WIN {
            let d = clean[j] - clean[j + tau];
            diff += d * d;
        }
        running_sum += diff;
        cmnd[tau] = if running_sum > 1e-12 {
            diff * (tau as f32) / running_sum
        } else {
            1.0
        };
    }

    let mut chosen_tau = None;
    let mut tau = MIN_TAU;
    while tau <= MAX_TAU {
        if cmnd[tau] < 0.18 {
            while tau < MAX_TAU && cmnd[tau + 1] < cmnd[tau] {
                tau += 1;
            }
            chosen_tau = Some(tau);
            break;
        }
        tau += 1;
    }

    let best_tau = match chosen_tau {
        Some(t) => t,
        None => {
            let mut min_t = MIN_TAU;
            let mut min_v = cmnd[MIN_TAU];
            for (t, &v) in cmnd.iter().enumerate().take(MAX_TAU + 1).skip(MIN_TAU + 1) {
                if v < min_v {
                    min_v = v;
                    min_t = t;
                }
            }
            min_t
        }
    };

    let a = cmnd[best_tau - 1];
    let b = cmnd[best_tau];
    let c = cmnd[best_tau + 1];
    let denom = a - 2.0 * b + c;
    let delta = if denom.abs() > 1e-9 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    sr / (((best_tau as f32) + delta).max(1.0))
}

#[inline]
fn band_peak_centroid_hz(
    env: &[f32; VOCAL_FFT_BINS],
    lo_bin: usize,
    hi_bin: usize,
    bin_hz: f32,
) -> f32 {
    let mut peak_k = lo_bin;
    let mut peak_v = -1.0_f32;
    for k in lo_bin..=hi_bin {
        if env[k] > peak_v {
            peak_v = env[k];
            peak_k = k;
        }
    }
    if peak_v <= 1e-10 {
        return 0.5 * ((lo_bin + hi_bin) as f32) * bin_hz;
    }
    // Power-weighted local formant centroid within +/-5 bins of the spectral envelope peak.
    let c_lo = peak_k.saturating_sub(5).max(lo_bin);
    let c_hi = (peak_k + 5).min(hi_bin);
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for k in c_lo..=c_hi {
        let w = env[k] * env[k];
        num += w * (k as f32) * bin_hz;
        den += w;
    }
    if den > 1e-12 {
        num / den
    } else {
        (peak_k as f32) * bin_hz
    }
}

/// 32-band Glasberg-Moore Equivalent Rectangular Bandwidth (`ERB`, `80..7,600 Hz`) cochlea
/// with logarithmic outer-hair-cell compression and 2nd-order critically damped hair-cell
/// & echoic adaptation filters.
#[derive(Clone, Debug)]
pub struct ErbCochlea32 {
    low_hz: [f32; VOCAL_ERB_BANDS],
    center_hz: [f32; VOCAL_ERB_BANDS],
    high_hz: [f32; VOCAL_ERB_BANDS],
    log_env: [f32; VOCAL_ERB_BANDS],
    raw_comp: [f32; VOCAL_ERB_BANDS],
    fast_pos: [f32; VOCAL_ERB_BANDS],
    fast_vel: [f32; VOCAL_ERB_BANDS],
    echo_pos: [f32; VOCAL_ERB_BANDS],
    echo_vel: [f32; VOCAL_ERB_BANDS],
    flux_pos: [f32; 16],
    flux_vel: [f32; 16],
}

impl Default for ErbCochlea32 {
    fn default() -> Self {
        Self::new()
    }
}

impl ErbCochlea32 {
    /// Constructs a `32`-band ERB filterbank spanning `80..7,600 Hz`.
    #[must_use]
    pub fn new() -> Self {
        let erb_min = hz_to_erb(80.0);
        let erb_max = hz_to_erb(7_600.0);
        let mut edges = [0.0_f32; VOCAL_ERB_BANDS + 2];
        for (i, slot) in edges.iter_mut().enumerate() {
            let frac = (i as f32) / ((VOCAL_ERB_BANDS + 1) as f32);
            *slot = erb_to_hz(erb_min + frac * (erb_max - erb_min));
        }
        let mut low_hz = [0.0_f32; VOCAL_ERB_BANDS];
        let mut center_hz = [0.0_f32; VOCAL_ERB_BANDS];
        let mut high_hz = [0.0_f32; VOCAL_ERB_BANDS];
        for b in 0..VOCAL_ERB_BANDS {
            low_hz[b] = edges[b];
            center_hz[b] = edges[b + 1];
            high_hz[b] = edges[b + 2];
        }
        Self {
            low_hz,
            center_hz,
            high_hz,
            log_env: [0.0; VOCAL_ERB_BANDS],
            raw_comp: [-0.35; VOCAL_ERB_BANDS],
            fast_pos: [-0.35; VOCAL_ERB_BANDS],
            fast_vel: [0.0; VOCAL_ERB_BANDS],
            echo_pos: [-0.35; VOCAL_ERB_BANDS],
            echo_vel: [0.0; VOCAL_ERB_BANDS],
            flux_pos: [0.0; 16],
            flux_vel: [0.0; 16],
        }
    }

    /// Center frequencies (`Hz`) of the `32` tonotopic ERB channels.
    #[must_use]
    pub fn center_frequencies_hz(&self) -> &[f32; VOCAL_ERB_BANDS] {
        &self.center_hz
    }

    /// Latest pre-inhibition logarithmic hair-cell excitation in `[0.0, 1.0]` (for brainstem `LSO` ILD).
    #[must_use]
    pub fn log_excitation_spectrum(&self) -> &[f32; VOCAL_ERB_BANDS] {
        &self.log_env
    }

    /// Latest instantaneous log-compressed ERB excitation in `[-1.0, 1.0]`.
    #[must_use]
    pub fn raw_compressed_spectrum(&self) -> &[f32; VOCAL_ERB_BANDS] {
        &self.raw_comp
    }

    /// Transduces one `320`-sample (`20 ms @ 16 kHz`) acoustic frame into:
    /// - `32` fast inner-hair-cell channels (middle-ear stapes pre-emphasized + lateral-inhibited),
    /// - `32` slow echoic sensory-adaptation channels (`tau ~ 32 ticks`), and
    /// - `16` tonotopic onset/offset spectral-flux channels.
    pub fn transduce_frame(
        &mut self,
        air_pcm: &[f32; VOCAL_FRAME_SAMPLES],
    ) -> ([f32; VOCAL_ERB_BANDS], [f32; VOCAL_ERB_BANDS], [f32; 16]) {
        let mut pow = [0.0_f32; VOCAL_FFT_BINS];
        // Middle-ear stapes transfer (`+6 dB/oct`, Rosowski 1991) equalizes F2/F3 against F0/F1.
        hann_fft_power_512_preemph(air_pcm, 0.92, &mut pow);
        let bin_hz = (VOCAL_SR_HZ as f32) / (VOCAL_FFT_SIZE as f32);
        let log_norm = (1201.0_f32).ln();

        let mut log_bands = [0.0_f32; VOCAL_ERB_BANDS];
        for b in 0..VOCAL_ERB_BANDS {
            let lo = self.low_hz[b];
            let mid = self.center_hz[b];
            let hi = self.high_hz[b];
            let k_start = ((lo / bin_hz).floor() as usize).clamp(1, VOCAL_FFT_BINS - 1);
            let k_end = ((hi / bin_hz).ceil() as usize).clamp(k_start, VOCAL_FFT_BINS - 1);

            let mut band_energy = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for k in k_start..=k_end {
                let f = (k as f32) * bin_hz;
                let w = if f <= mid {
                    ((f - lo) / (mid - lo).max(1.0)).clamp(0.0, 1.0)
                } else {
                    ((hi - f) / (hi - mid).max(1.0)).clamp(0.0, 1.0)
                };
                band_energy += w * pow[k];
                weight_sum += w;
            }
            let mean_e = if weight_sum > 1e-6 {
                band_energy / weight_sum
            } else {
                0.0
            };
            log_bands[b] = ((1.0 + 1200.0 * mean_e).ln() / log_norm).clamp(0.0, 1.0);
        }
        self.log_env = log_bands;

        // Cochlear lateral inhibition (Shamma 1985): enhances formant peaks against broad spectral tilt.
        for b in 0..VOCAL_ERB_BANDS {
            let lo_n = b.saturating_sub(4);
            let hi_n = (b + 4).min(VOCAL_ERB_BANDS - 1);
            let mut neigh = 0.0_f32;
            for &v in &log_bands[lo_n..=hi_n] {
                neigh += v;
            }
            neigh /= ((hi_n - lo_n + 1) as f32).max(1.0);
            let prom = log_bands[b] - 0.72 * neigh;
            let u = sanitize_unit_signed(2.4 * prom - 0.35);
            self.raw_comp[b] = u;

            // 4 substeps of critically damped 2nd-order hair-cell adaptation so cochlear velocity is C^1 smooth.
            for _ in 0..4 {
                // Fast inner-hair-cell ribbon (`omega_f = 0.24` per quarter-tick -> ~1.5 tick rise).
                let wf = 0.24_f32;
                let af = (wf * wf) * (u - self.fast_pos[b]) - (2.0 * wf) * self.fast_vel[b];
                self.fast_vel[b] = sanitize_bounded(self.fast_vel[b] + af, 4.0);
                self.fast_pos[b] = sanitize_unit_signed(self.fast_pos[b] + self.fast_vel[b]);

                // Slow auditory-cortex echoic trace (`omega_e = 0.017` per quarter-tick -> ~30 tick memory).
                let we = 0.017_f32;
                let ae = (we * we) * (u - self.echo_pos[b]) - (2.0 * we) * self.echo_vel[b];
                self.echo_vel[b] = sanitize_bounded(self.echo_vel[b] + ae, 2.0);
                self.echo_pos[b] = sanitize_unit_signed(self.echo_pos[b] + self.echo_vel[b]);
            }
        }

        for j in 0..16 {
            let b0 = 2 * j;
            let b1 = b0 + 1;
            let diff = 0.5
                * ((self.fast_pos[b0] - self.echo_pos[b0])
                    + (self.fast_pos[b1] - self.echo_pos[b1]));
            let target = sanitize_unit_signed(1.6 * diff);
            for _ in 0..4 {
                let wx = 0.18_f32;
                let ax = (wx * wx) * (target - self.flux_pos[j]) - (2.0 * wx) * self.flux_vel[j];
                self.flux_vel[j] = sanitize_bounded(self.flux_vel[j] + ax, 4.0);
                self.flux_pos[j] = sanitize_unit_signed(self.flux_pos[j] + self.flux_vel[j]);
            }
        }

        (self.fast_pos, self.echo_pos, self.flux_pos)
    }
}

#[inline]
fn hz_to_erb(hz: f32) -> f32 {
    21.4 * (1.0 + 0.00437 * hz).log10()
}

#[inline]
fn erb_to_hz(erb: f32) -> f32 {
    (10.0_f32.powf(erb / 21.4) - 1.0) / 0.00437
}

/// Encodes normalized `[-1.0, 1.0]` mono samples into a standard 44-byte header `16,000 Hz`
/// 16-bit PCM `.wav` byte vector (zero string literals).
#[must_use]
pub fn encode_wav_16khz_mono(samples: &[f32]) -> Vec<u8> {
    encode_wav_pcm16(samples, 1, VOCAL_SR_HZ)
}

/// Encodes interleaved `[L0, R0, L1, R1, ...]` stereo samples into a standard 44-byte header
/// `16,000 Hz` 16-bit PCM `.wav` byte vector (zero string literals).
#[must_use]
pub fn encode_wav_16khz_stereo(interleaved_lr: &[f32]) -> Vec<u8> {
    encode_wav_pcm16(interleaved_lr, 2, VOCAL_SR_HZ)
}

fn encode_wav_pcm16(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<u8> {
    let ch = channels.max(1);
    let data_bytes_len = (samples.len().min((u32::MAX as usize) / 2 - 44)) * 2;
    let riff_chunk_size = (36 + data_bytes_len) as u32;
    let byte_rate = sample_rate * u32::from(ch) * 2;
    let block_align = ch * 2;

    let mut out = Vec::with_capacity(44 + data_bytes_len);
    out.extend_from_slice(&WAV_RIFF);
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(&WAV_WAVE);
    out.extend_from_slice(&WAV_FMT);
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes()); // PCM = 1
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&16_u16.to_le_bytes()); // 16 bits/sample
    out.extend_from_slice(&WAV_DATA);
    out.extend_from_slice(&(data_bytes_len as u32).to_le_bytes());

    for &s in samples.iter().take(data_bytes_len / 2) {
        let clamped = sanitize_unit_signed(s);
        let pcm = (clamped * 32_767.0).round() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

/// Read-only longitudinal acoustic & articulatory metrics measured directly from `16 kHz` waveforms.
/// Zero rewards or synthetic targets ever depend on this struct.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VocalOrganObservation {
    pub tick: u64,
    pub window_ticks: u64,
    pub voiced_ticks: u64,
    pub syllable_onsets: u64,
    /// Multi-syllable / diphthong formant contour transitions within `<= 20` ticks (`400 ms`)
    /// shifting `(F1, F2)` by `>= 115 Hz` Euclidean distance (`0` on steady monotone hums).
    pub syllable_transitions: u64,
    pub plosive_burst_frames: u64,
    /// Voiced ticks occurring while an external peer wave was simultaneously audible.
    pub simultaneous_overlap_ticks: u64,
    /// Voiced ticks occurring within `1..=25` ticks (`20..500 ms`) after a peer utterance ended.
    pub alternating_response_ticks: u64,
    /// Ticks where stone-cave shelter walls returned an audible `40 ms` vocal/acoustic reflection.
    pub shelter_echo_ticks: u64,
    /// Current anatomical vocal-tract length in centimeters (`8.8..=14.5 cm`).
    pub tract_length_cm: f32,
    pub mean_rms: f32,
    pub mean_f0_hz: f32,
    pub mean_f1_hz: f32,
    pub mean_f2_hz: f32,
    /// Standard deviation of `F1` (`Hz`) across voiced frames (jaw openness range explored).
    pub f1_spread_hz: f32,
    /// Standard deviation of `F2` (`Hz`) across voiced frames (front-back tongue range explored).
    pub f2_spread_hz: f32,
    /// Mean frame-to-frame articulatory muscle change `mean_d |pos_t[d] - pos_{t-1}[d]|`.
    pub mean_articulator_delta: f32,
}

/// Pure `16 kHz` physical vocal-tract & binaural `32`-band cochlear organ (zero scripted caregiver,
/// zero phase clocks, zero cosine-similarity or vocal metabolic rewards).
#[derive(Clone, Debug)]
pub struct BinauralVocalOrgan {
    tick: u64,
    rng: SplitMix64,

    tract: PhysicalVocalTract,
    left_cochlea: ErbCochlea32,
    right_cochlea: ErbCochlea32,

    last_self_wave: [f32; VOCAL_FRAME_SAMPLES],
    last_self_rms: f32,
    cave_reverb_buf: [[f32; VOCAL_FRAME_SAMPLES]; 2],
    cave_reverb_cursor: usize,
    peer_itd_tail_l: [f32; MAX_HEAD_ITD_SAMPLES],
    peer_itd_tail_r: [f32; MAX_HEAD_ITD_SAMPLES],
    lso_fast_pos: [f32; VOCAL_ERB_BANDS],
    lso_fast_vel: [f32; VOCAL_ERB_BANDS],

    wind_l_pos: f32,
    wind_l_vel: f32,
    wind_r_pos: f32,
    wind_r_vel: f32,
    ou_distractors: [f32; 16],

    prev_self_voiced: bool,
    prev_artic_pos: [f32; VOCAL_ACTUATORS],
    ticks_since_peer_voiced: u32,
    ticks_since_self_voiced: u32,
    voiced_since_transition: u32,
    anchor_f1_hz: f32,
    anchor_f2_hz: f32,

    channels_out: [f32; VOCAL_ORGAN_CHANNELS],

    // Read-only longitudinal acoustic accumulators
    window_ticks: u64,
    voiced_ticks: u64,
    syllable_onsets: u64,
    syllable_transitions: u64,
    plosive_burst_frames: u64,
    simultaneous_overlap_ticks: u64,
    alternating_response_ticks: u64,
    shelter_echo_ticks: u64,
    sum_rms: f64,
    sum_f0_hz: f64,
    sum_f1_hz: f64,
    sum_f2_hz: f64,
    sum_f1_sq: f64,
    sum_f2_sq: f64,
    sum_artic_delta: f64,

    // Optional rolling 16 kHz PCM capture for live `.wav` observation
    record_pcm: bool,
    newborn_pcm: Vec<f32>,
    self_ring: Vec<f32>,
    room_stereo_ring: Vec<f32>,
    pcm_ring_cursor: usize,
    pcm_ring_filled: bool,
}

impl BinauralVocalOrgan {
    /// Creates an infant-scale (`8.8 cm`) `16 kHz` vocal & binaural cochlear organ.
    #[must_use]
    pub fn new_infant(seed: u64) -> Self {
        Self::with_tract(seed, PhysicalVocalTract::new_infant(seed))
    }

    /// Creates an adult-scale (`14.5 cm`) `16 kHz` vocal & binaural cochlear organ.
    #[must_use]
    pub fn new_adult(seed: u64) -> Self {
        Self::with_tract(seed, PhysicalVocalTract::new_adult(seed))
    }

    #[must_use]
    fn with_tract(seed: u64, tract: PhysicalVocalTract) -> Self {
        let mut organ = Self {
            tick: 0,
            rng: SplitMix64::new(seed ^ 0x50C1_A170_CA11_2026),
            tract,
            left_cochlea: ErbCochlea32::new(),
            right_cochlea: ErbCochlea32::new(),
            last_self_wave: [0.0; VOCAL_FRAME_SAMPLES],
            last_self_rms: 0.0,
            cave_reverb_buf: [[0.0; VOCAL_FRAME_SAMPLES]; 2],
            cave_reverb_cursor: 0,
            peer_itd_tail_l: [0.0; MAX_HEAD_ITD_SAMPLES],
            peer_itd_tail_r: [0.0; MAX_HEAD_ITD_SAMPLES],
            lso_fast_pos: [-0.35; VOCAL_ERB_BANDS],
            lso_fast_vel: [0.0; VOCAL_ERB_BANDS],
            wind_l_pos: 0.0,
            wind_l_vel: 0.0,
            wind_r_pos: 0.0,
            wind_r_vel: 0.0,
            ou_distractors: [0.0; 16],
            prev_self_voiced: false,
            prev_artic_pos: [0.0; VOCAL_ACTUATORS],
            ticks_since_peer_voiced: 1_000,
            ticks_since_self_voiced: 1_000,
            voiced_since_transition: 0,
            anchor_f1_hz: 0.0,
            anchor_f2_hz: 0.0,
            channels_out: [-0.35; VOCAL_ORGAN_CHANNELS],
            window_ticks: 0,
            voiced_ticks: 0,
            syllable_onsets: 0,
            syllable_transitions: 0,
            plosive_burst_frames: 0,
            simultaneous_overlap_ticks: 0,
            alternating_response_ticks: 0,
            shelter_echo_ticks: 0,
            sum_rms: 0.0,
            sum_f0_hz: 0.0,
            sum_f1_hz: 0.0,
            sum_f2_hz: 0.0,
            sum_f1_sq: 0.0,
            sum_f2_sq: 0.0,
            sum_artic_delta: 0.0,
            record_pcm: false,
            newborn_pcm: Vec::new(),
            self_ring: Vec::new(),
            room_stereo_ring: Vec::new(),
            pcm_ring_cursor: 0,
            pcm_ring_filled: false,
        };
        organ.transduce_binaural_air(None, false);
        organ
    }

    /// Enables live `16 kHz` rolling `10 s` mono/stereo + newborn-first-`10 s` PCM capture.
    pub fn enable_pcm_recording(&mut self) {
        if !self.record_pcm {
            self.record_pcm = true;
            self.newborn_pcm.reserve_exact(VOCAL_RING_SAMPLES);
            self.self_ring = vec![0.0; VOCAL_RING_SAMPLES];
            self.room_stereo_ring = vec![0.0; VOCAL_RING_SAMPLES * 2];
            self.pcm_ring_cursor = 0;
            self.pcm_ring_filled = false;
        }
    }

    /// Current simulation tick (`20 ms` per tick).
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Advances anatomical vocal-tract length & laryngeal descent (`8.8 cm -> 14.5 cm`).
    pub fn advance_ontogeny(&mut self, age_ticks: u64, mitosis_births: u64, vocal_mastery: f32) {
        self.tract
            .advance_ontogeny(age_ticks, mitosis_births, vocal_mastery);
    }

    /// Current anatomical vocal-tract length in centimeters (`8.8..=14.5 cm`).
    #[must_use]
    pub fn tract_length_cm(&self) -> f32 {
        self.tract.tract_length_cm()
    }

    /// Latest `320`-sample (`20 ms @ 16 kHz`) acoustic pressure wave radiated from this organism's mouth.
    #[must_use]
    pub fn last_radiated_wave(&self) -> &[f32; VOCAL_FRAME_SAMPLES] {
        &self.last_self_wave
    }

    /// RMS amplitude (`[0.0, 1.0]`) of the latest `320`-sample radiated vocal pressure wave.
    #[must_use]
    pub fn last_radiated_rms(&self) -> f32 {
        self.last_self_rms
    }

    /// Latest `80` unlabeled continuous cochlear + `LSO` + `OU(0.97)` channels in `[-1.0, 1.0]`.
    #[must_use]
    pub fn sensory_channels(&self) -> &[f32; VOCAL_ORGAN_CHANNELS] {
        &self.channels_out
    }

    /// Resets read-only developmental window accumulators without mutating any physical state.
    pub fn reset_window_observations(&mut self) {
        self.window_ticks = 0;
        self.voiced_ticks = 0;
        self.syllable_onsets = 0;
        self.syllable_transitions = 0;
        self.plosive_burst_frames = 0;
        self.simultaneous_overlap_ticks = 0;
        self.alternating_response_ticks = 0;
        self.shelter_echo_ticks = 0;
        self.ticks_since_self_voiced = 1_000;
        self.voiced_since_transition = 0;
        self.anchor_f1_hz = 0.0;
        self.anchor_f2_hz = 0.0;
        self.sum_rms = 0.0;
        self.sum_f0_hz = 0.0;
        self.sum_f1_hz = 0.0;
        self.sum_f2_hz = 0.0;
        self.sum_f1_sq = 0.0;
        self.sum_f2_sq = 0.0;
        self.sum_artic_delta = 0.0;
    }

    /// Synthesizes `320` samples (`20 ms @ 16 kHz`) from the organism's 5 articulatory muscles
    /// (`lung, larynx, jaw, tongue_body, lips`) without yet stepping cochlear reception, so multiple
    /// cohabiting organisms can radiate into the shared air simultaneously on the same tick.
    pub fn radiate_voice(&mut self, vocal_cmd: &[f32; VOCAL_ACTUATORS]) -> [f32; VOCAL_FRAME_SAMPLES] {
        let mut pcm = [0.0_f32; VOCAL_FRAME_SAMPLES];
        self.tract.synthesize_frame(vocal_cmd, &mut pcm);
        let mut sq = 0.0_f32;
        for &s in &pcm {
            sq += s * s;
        }
        self.last_self_rms = (sq / (VOCAL_FRAME_SAMPLES as f32)).sqrt().clamp(0.0, 1.0);
        self.last_self_wave = pcm;
        pcm
    }

    /// Transduces binaural `16 kHz` air at Left & Right ears from `last_self_wave` plus an optional
    /// peer wave arriving from relative 2D world offset `(dx, dy)` (`peer_pos - self_pos`).
    pub fn transduce_binaural_air(
        &mut self,
        peer_source: Option<(&[f32; VOCAL_FRAME_SAMPLES], (f32, f32))>,
        advance_clock: bool,
    ) {
        self.transduce_binaural_ecology(peer_source, None, 0.0, advance_clock);
    }

    /// Transduces binaural `16 kHz` air including peer vocal waves, spatialized environmental
    /// soundscapes (`env_ears`: flowing water, cracked forage shells, rolling boulders), and
    /// `40 ms` physical stone-cave wall reverberation (`shelter_enclosure` in `[0.0, 1.0]`).
    pub fn transduce_binaural_ecology(
        &mut self,
        peer_source: Option<(&[f32; VOCAL_FRAME_SAMPLES], (f32, f32))>,
        env_ears: Option<(&[f32; VOCAL_FRAME_SAMPLES], &[f32; VOCAL_FRAME_SAMPLES])>,
        shelter_enclosure: f32,
        advance_clock: bool,
    ) {
        let mut p_left = [0.0_f32; VOCAL_FRAME_SAMPLES];
        let mut p_right = [0.0_f32; VOCAL_FRAME_SAMPLES];
        let enclosure = if shelter_enclosure.is_finite() {
            shelter_enclosure.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let reverb_gain = 0.34 * enclosure;
        let delayed_cave = self.cave_reverb_buf[self.cave_reverb_cursor];
        let mut echo_sq = 0.0_f32;

        let (peer_opt, gain_l, gain_r, delay_l, delay_r, peer_voiced) = match peer_source {
            Some((p_peer, (dx, dy))) => {
                let dist = dx.hypot(dy);
                let att = 1.0 / (1.0 + 0.14 * dist.max(0.0));
                let sin_az = if dist > 0.05 {
                    (dx / dist).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                // Positive sin_az (peer to the right, +x) arrives earlier & stronger at Right ear.
                let signed_itd = (sin_az * 6.0).round() as isize;
                let d_l = if signed_itd > 0 {
                    (signed_itd as usize).min(MAX_HEAD_ITD_SAMPLES)
                } else {
                    0
                };
                let d_r = if signed_itd < 0 {
                    ((-signed_itd) as usize).min(MAX_HEAD_ITD_SAMPLES)
                } else {
                    0
                };
                let g_r = att * (0.64 + 0.34 * sin_az);
                let g_l = att * (0.64 - 0.34 * sin_az);

                let mut peer_sq = 0.0_f32;
                for &s in p_peer {
                    peer_sq += s * s;
                }
                let p_rms = (peer_sq / (VOCAL_FRAME_SAMPLES as f32)).sqrt() * att;
                (Some(p_peer), g_l, g_r, d_l, d_r, p_rms >= 0.015)
            }
            None => (None, 0.0, 0.0, 0, 0, false),
        };

        let mut next_reverb = [0.0_f32; VOCAL_FRAME_SAMPLES];
        for i in 0..VOCAL_FRAME_SAMPLES {
            let gust_l = self.rng.next_f32_signed() * 0.0020;
            let aw_l = gust_l - 0.05 * self.wind_l_vel - 0.004 * self.wind_l_pos;
            self.wind_l_vel = sanitize_bounded(self.wind_l_vel + aw_l, 0.1);
            self.wind_l_pos = sanitize_bounded(self.wind_l_pos + self.wind_l_vel, 0.05);

            let gust_r = self.rng.next_f32_signed() * 0.0020;
            let aw_r = gust_r - 0.05 * self.wind_r_vel - 0.004 * self.wind_r_pos;
            self.wind_r_vel = sanitize_bounded(self.wind_r_vel + aw_r, 0.1);
            self.wind_r_pos = sanitize_bounded(self.wind_r_pos + self.wind_r_vel, 0.05);

            let (pw_l, pw_r) = if let Some(p_peer) = peer_opt {
                let s_l = if i < delay_l {
                    self.peer_itd_tail_l[MAX_HEAD_ITD_SAMPLES - delay_l + i]
                } else {
                    p_peer[i - delay_l]
                };
                let s_r = if i < delay_r {
                    self.peer_itd_tail_r[MAX_HEAD_ITD_SAMPLES - delay_r + i]
                } else {
                    p_peer[i - delay_r]
                };
                (gain_l * s_l, gain_r * s_r)
            } else {
                (0.0, 0.0)
            };

            let (env_l, env_r) = if let Some((el, er)) = env_ears {
                (el[i], er[i])
            } else {
                (0.0, 0.0)
            };

            let self_s = if advance_clock {
                0.70 * self.last_self_wave[i]
            } else {
                0.0
            };
            let cave_reflected = reverb_gain * delayed_cave[i];
            echo_sq += cave_reflected * cave_reflected;
            next_reverb[i] = sanitize_unit_signed(
                self_s + 0.5 * (pw_l + pw_r + env_l + env_r) + 0.22 * cave_reflected,
            );

            p_left[i] = sanitize_unit_signed(self_s + pw_l + env_l + cave_reflected + self.wind_l_pos);
            p_right[i] = sanitize_unit_signed(self_s + pw_r + env_r + cave_reflected + self.wind_r_pos);
        }

        if advance_clock {
            self.cave_reverb_buf[self.cave_reverb_cursor] = next_reverb;
            self.cave_reverb_cursor ^= 1;
            let echo_rms = (echo_sq / (VOCAL_FRAME_SAMPLES as f32)).sqrt();
            if echo_rms >= 0.008 {
                self.shelter_echo_ticks = self.shelter_echo_ticks.saturating_add(1);
            }
        }

        if let Some(p_peer) = peer_opt {
            self.peer_itd_tail_l
                .copy_from_slice(&p_peer[(VOCAL_FRAME_SAMPLES - MAX_HEAD_ITD_SAMPLES)..]);
            self.peer_itd_tail_r
                .copy_from_slice(&p_peer[(VOCAL_FRAME_SAMPLES - MAX_HEAD_ITD_SAMPLES)..]);
        } else {
            self.peer_itd_tail_l = [0.0; MAX_HEAD_ITD_SAMPLES];
            self.peer_itd_tail_r = [0.0; MAX_HEAD_ITD_SAMPLES];
        }

        if advance_clock && self.record_pcm {
            if self.newborn_pcm.len() < VOCAL_RING_SAMPLES {
                let rem = VOCAL_RING_SAMPLES - self.newborn_pcm.len();
                let n_copy = rem.min(VOCAL_FRAME_SAMPLES);
                self.newborn_pcm
                    .extend_from_slice(&self.last_self_wave[..n_copy]);
            }
            for i in 0..VOCAL_FRAME_SAMPLES {
                let idx = self.pcm_ring_cursor;
                self.self_ring[idx] = self.last_self_wave[i];
                self.room_stereo_ring[idx * 2] = p_left[i];
                self.room_stereo_ring[idx * 2 + 1] = p_right[i];
                self.pcm_ring_cursor += 1;
                if self.pcm_ring_cursor >= VOCAL_RING_SAMPLES {
                    self.pcm_ring_cursor = 0;
                    self.pcm_ring_filled = true;
                }
            }
        }

        let (fast_l, _echo_l, _flux_l) = self.left_cochlea.transduce_frame(&p_left);
        let (fast_r, _echo_r, _flux_r) = self.right_cochlea.transduce_frame(&p_right);
        let log_l = *self.left_cochlea.log_excitation_spectrum();
        let log_r = *self.right_cochlea.log_excitation_spectrum();

        for b in 0..VOCAL_ERB_BANDS {
            self.channels_out[b] = sanitize_unit_signed(0.5 * (fast_r[b] + fast_l[b]));
            let u_ild = sanitize_unit_signed(-0.35 + 4.8 * (log_r[b] - log_l[b]));
            for _ in 0..4 {
                let wf = 0.24_f32;
                let af =
                    (wf * wf) * (u_ild - self.lso_fast_pos[b]) - (2.0 * wf) * self.lso_fast_vel[b];
                self.lso_fast_vel[b] = sanitize_bounded(self.lso_fast_vel[b] + af, 4.0);
                self.lso_fast_pos[b] =
                    sanitize_unit_signed(self.lso_fast_pos[b] + self.lso_fast_vel[b]);
            }
            self.channels_out[32 + b] = self.lso_fast_pos[b];
        }

        let ou_rho = 0.97_f32;
        let ou_scale = (1.0 - ou_rho * ou_rho).sqrt() * 0.85;
        for (i, ch) in self.channels_out[64..80].iter_mut().enumerate() {
            let z = self.rng.next_f32_signed();
            self.ou_distractors[i] =
                sanitize_unit_signed(ou_rho * self.ou_distractors[i] + ou_scale * z);
            *ch = self.ou_distractors[i];
        }

        if advance_clock {
            let (rms, f0_hz, f1_hz, f2_hz) = measure_frame_acoustics(&self.last_self_wave);
            let voiced = rms >= VOICED_RMS_THRESHOLD;
            self.window_ticks = self.window_ticks.saturating_add(1);
            self.sum_rms += f64::from(rms);

            let cur_pos = self.tract.articulator_positions();
            let mut artic_d = 0.0_f32;
            for d in 0..VOCAL_ACTUATORS {
                artic_d += (cur_pos[d] - self.prev_artic_pos[d]).abs();
            }
            self.prev_artic_pos = cur_pos;
            self.sum_artic_delta += f64::from(artic_d / (VOCAL_ACTUATORS as f32));

            if self.tract.last_plosive_burst() >= 0.018 {
                self.plosive_burst_frames = self.plosive_burst_frames.saturating_add(1);
            }
            if voiced && !self.prev_self_voiced {
                self.syllable_onsets = self.syllable_onsets.saturating_add(1);
            }
            self.prev_self_voiced = voiced;

            if voiced {
                self.voiced_ticks = self.voiced_ticks.saturating_add(1);
                self.sum_f0_hz += f64::from(f0_hz);
                self.sum_f1_hz += f64::from(f1_hz);
                self.sum_f2_hz += f64::from(f2_hz);
                self.sum_f1_sq += f64::from(f1_hz) * f64::from(f1_hz);
                self.sum_f2_sq += f64::from(f2_hz) * f64::from(f2_hz);

                if self.ticks_since_self_voiced > 4 || self.anchor_f1_hz <= 0.0 {
                    self.anchor_f1_hz = f1_hz;
                    self.anchor_f2_hz = f2_hz;
                    self.voiced_since_transition = 1;
                } else {
                    self.voiced_since_transition = self.voiced_since_transition.saturating_add(1);
                    let formant_shift =
                        (f1_hz - self.anchor_f1_hz).hypot(f2_hz - self.anchor_f2_hz);
                    if self.voiced_since_transition >= 3 && formant_shift >= 140.0 {
                        self.syllable_transitions = self.syllable_transitions.saturating_add(1);
                        self.anchor_f1_hz = f1_hz;
                        self.anchor_f2_hz = f2_hz;
                        self.voiced_since_transition = 0;
                    }
                }
                self.ticks_since_self_voiced = 0;

                if peer_voiced {
                    self.simultaneous_overlap_ticks =
                        self.simultaneous_overlap_ticks.saturating_add(1);
                } else if (1..=25).contains(&self.ticks_since_peer_voiced) {
                    self.alternating_response_ticks =
                        self.alternating_response_ticks.saturating_add(1);
                }
            } else {
                self.ticks_since_self_voiced = self.ticks_since_self_voiced.saturating_add(1);
            }

            if peer_voiced {
                self.ticks_since_peer_voiced = 0;
            } else {
                self.ticks_since_peer_voiced = self.ticks_since_peer_voiced.saturating_add(1);
            }
            self.tick = self.tick.saturating_add(1);
        }
    }

    /// Convenience single-step update (`radiate_voice` + `transduce_binaural_air`).
    pub fn step_organ(
        &mut self,
        vocal_cmd: &[f32; VOCAL_ACTUATORS],
        peer_source: Option<(&[f32; VOCAL_FRAME_SAMPLES], (f32, f32))>,
    ) {
        let _ = self.radiate_voice(vocal_cmd);
        self.transduce_binaural_air(peer_source, true);
    }

    /// Unedited `16 kHz` mono PCM samples from the organism's very first `10.0 s` (`ticks 1..=500`).
    #[must_use]
    pub fn newborn_first_10s_pcm(&self) -> &[f32] {
        &self.newborn_pcm
    }

    /// Chronologically ordered latest `10.0 s` (`<= 160,000` samples) of pure self voice mono PCM.
    #[must_use]
    pub fn latest_10s_self_mono_pcm(&self) -> Vec<f32> {
        if !self.record_pcm {
            return Vec::new();
        }
        if !self.pcm_ring_filled {
            return self.self_ring[..self.pcm_ring_cursor].to_vec();
        }
        let mut out = Vec::with_capacity(VOCAL_RING_SAMPLES);
        out.extend_from_slice(&self.self_ring[self.pcm_ring_cursor..]);
        out.extend_from_slice(&self.self_ring[..self.pcm_ring_cursor]);
        out
    }

    /// Chronologically ordered latest `10.0 s` interleaved stereo PCM (`Left` & `Right` ear waves).
    #[must_use]
    pub fn latest_10s_room_stereo_pcm(&self) -> Vec<f32> {
        if !self.record_pcm {
            return Vec::new();
        }
        let cur2 = self.pcm_ring_cursor * 2;
        if !self.pcm_ring_filled {
            return self.room_stereo_ring[..cur2].to_vec();
        }
        let mut out = Vec::with_capacity(VOCAL_RING_SAMPLES * 2);
        out.extend_from_slice(&self.room_stereo_ring[cur2..]);
        out.extend_from_slice(&self.room_stereo_ring[..cur2]);
        out
    }

    /// Computes read-only longitudinal `16 kHz` acoustic and articulatory metrics.
    #[must_use]
    pub fn observation(&self) -> VocalOrganObservation {
        let wt = self.window_ticks.max(1) as f64;
        let mean_rms = (self.sum_rms / wt) as f32;
        let mean_articulator_delta = (self.sum_artic_delta / wt) as f32;

        let (mean_f0_hz, mean_f1_hz, mean_f2_hz, f1_spread_hz, f2_spread_hz) =
            if self.voiced_ticks > 0 {
                let inv = 1.0 / (self.voiced_ticks as f64);
                let mf0 = self.sum_f0_hz * inv;
                let mf1 = self.sum_f1_hz * inv;
                let mf2 = self.sum_f2_hz * inv;
                let vf1 = (self.sum_f1_sq * inv - mf1 * mf1).max(0.0).sqrt();
                let vf2 = (self.sum_f2_sq * inv - mf2 * mf2).max(0.0).sqrt();
                (mf0 as f32, mf1 as f32, mf2 as f32, vf1 as f32, vf2 as f32)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };

        VocalOrganObservation {
            tick: self.tick,
            window_ticks: self.window_ticks,
            voiced_ticks: self.voiced_ticks,
            syllable_onsets: self.syllable_onsets,
            syllable_transitions: self.syllable_transitions,
            plosive_burst_frames: self.plosive_burst_frames,
            simultaneous_overlap_ticks: self.simultaneous_overlap_ticks,
            alternating_response_ticks: self.alternating_response_ticks,
            shelter_echo_ticks: self.shelter_echo_ticks,
            tract_length_cm: self.tract.tract_length_cm(),
            mean_rms,
            mean_f0_hz,
            mean_f1_hz,
            mean_f2_hz,
            f1_spread_hz,
            f2_spread_hz,
            mean_articulator_delta,
        }
    }
}
