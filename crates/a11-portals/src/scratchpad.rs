//! Persistent 2D pigment canvas (`DrawingScratchpad`) for unforced sensorimotor observation.
//!
//! Exposes `192` unlabelled ambient channels (`144` canvas wave channels + `16` stylus
//! foveal/tactile/velocity channels + `32` exogenous Noisy-TV static channels) and `5`
//! continuous actuators (`dx`, `dy`, `pressure`, `pigment_tone`, `wipe_pulse`).
//!
//! Somatic reward purity: `metabolic_influx` is strictly equal to a flat constant
//! `basal_influx` regardless of canvas coverage, patterns, or actuator output, and
//! `tissue_stress` is `0.0`.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::{SplitMix64, actuator_at, sanitize_nonneg, sanitize_unit_signed};

/// Side length of the continuous square pigment grid (`12 x 12 = 144` cells).
pub const CANVAS_SIDE: usize = 12;
/// Number of canvas wave channels (`12 * 12`).
pub const CANVAS_CHANNELS: usize = CANVAS_SIDE * CANVAS_SIDE;
/// Number of stylus tactile, velocity, and local `3x3` foveal feedback channels.
pub const STYLUS_CHANNELS: usize = 16;
/// Number of exogenous static distractor channels (`Noisy TV`).
pub const NOISY_TV_CHANNELS: usize = 32;
/// Total ambient channels exposed by [`DrawingScratchpad`] (`144 + 16 + 32 = 192`).
pub const SCRATCHPAD_AMBIENT_CHANNELS: usize =
    CANVAS_CHANNELS + STYLUS_CHANNELS + NOISY_TV_CHANNELS;
/// Number of continuous actuators accepted by [`DrawingScratchpad`].
pub const SCRATCHPAD_ACTUATORS: usize = 5;

/// Default flat basal metabolic trickle in standalone portal mode (`>= 0.0`).
pub const DEFAULT_BASAL_INFLUX: f32 = 0.008;
/// Slow natural pigment drying factor per tick.
pub const NATURAL_DRYING_FACTOR: f32 = 0.9998;

const MAX_COORD: f32 = (CANVAS_SIDE - 1) as f32;
const INKED_CELL_THRESHOLD: f32 = 0.05;
const REVISIT_MIN_GAP_TICKS: u64 = 6;

/// Read-only behavioral observation metrics for [`DrawingScratchpad`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScratchpadObservation {
    /// Ticks where stylus moved while `pressure > 0.1`.
    pub active_stroke_ticks: u64,
    /// Fraction of the `12x12` canvas cells with settled pigment `>= 0.05` (`[0.0, 1.0]`).
    pub canvas_coverage_fraction: f32,
    /// Moran's I spatial autocorrelation over the `12x12` settled pigment grid (`[-1.0, 1.0]`).
    /// Distinguishes continuous connected lines/curves/clusters from scattered confetti dots.
    pub spatial_autocorrelation: f32,
    /// Fraction of active stylus cell visits that return to prior settled marks after leaving (`[0.0, 1.0]`).
    pub revisit_ratio: f32,
    /// Normalized Shannon entropy of stylus trajectory across headings and grid cells (`[0.0, 1.0]`).
    pub stylus_trajectory_entropy: f32,
}

/// Persistent 2D pigment medium with `2`-tick wet-to-settled ink physics.
#[derive(Clone, Debug)]
pub struct DrawingScratchpad {
    tick: u64,
    basal_influx: f32,
    rng: SplitMix64,

    stylus_x: f32,
    stylus_y: f32,
    stylus_vx: f32,
    stylus_vy: f32,
    last_eff_pressure: f32,

    /// Settled durable pigment density on `[0.0, 1.0]`.
    pigment: [f32; CANVAS_CHANNELS],
    /// Settled signed pigment tone modulation on `[-1.0, 1.0]`.
    pigment_tone: [f32; CANVAS_CHANNELS],

    /// Staging buffer deposited at `t`, settling into `pigment` on the following tick (`t + 2` view).
    wet_pending: [f32; CANVAS_CHANNELS],
    wet_tone_pending: [f32; CANVAS_CHANNELS],

    /// Last tick the stylus neighborhood occupied each cell (for genuine revisit detection).
    last_neighborhood_tick: [Option<u64>; CANVAS_CHANNELS],

    // Behavioral observation accumulators (strictly read-only relative to world physics)
    active_stroke_ticks: u64,
    active_visit_ticks: u64,
    revisit_ticks: u64,
    cell_visit_counts: [u64; CANVAS_CHANNELS],
    heading_octant_counts: [u64; 8],
    total_heading_samples: u64,

    ambient_buf: Vec<f32>,
}

impl Default for DrawingScratchpad {
    fn default() -> Self {
        Self::new(0x5C8A_7C40_0011_A110)
    }
}

impl DrawingScratchpad {
    /// Creates a new `DrawingScratchpad` with default `basal_influx` (`0.008`/tick).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_basal_influx(seed, DEFAULT_BASAL_INFLUX)
    }

    /// Creates a new `DrawingScratchpad` with an explicit constant `basal_influx` (`>= 0.0`).
    #[must_use]
    pub fn with_basal_influx(seed: u64, basal_influx: f32) -> Self {
        let mut pad = Self {
            tick: 0,
            basal_influx: sanitize_nonneg(basal_influx),
            rng: SplitMix64::new(seed),
            stylus_x: MAX_COORD * 0.5,
            stylus_y: MAX_COORD * 0.5,
            stylus_vx: 0.0,
            stylus_vy: 0.0,
            last_eff_pressure: 0.0,
            pigment: [0.0; CANVAS_CHANNELS],
            pigment_tone: [0.0; CANVAS_CHANNELS],
            wet_pending: [0.0; CANVAS_CHANNELS],
            wet_tone_pending: [0.0; CANVAS_CHANNELS],
            last_neighborhood_tick: [None; CANVAS_CHANNELS],
            active_stroke_ticks: 0,
            active_visit_ticks: 0,
            revisit_ticks: 0,
            cell_visit_counts: [0; CANVAS_CHANNELS],
            heading_octant_counts: [0; 8],
            total_heading_samples: 0,
            ambient_buf: vec![0.0; SCRATCHPAD_AMBIENT_CHANNELS],
        };
        pad.refresh_ambient();
        pad
    }

    /// Current simulation tick of this scratchpad.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Current stylus coordinates `(x, y)` in `[0.0, 11.0]`.
    #[must_use]
    pub fn stylus_position(&self) -> (f32, f32) {
        (self.stylus_x, self.stylus_y)
    }

    /// Settled pigment grid (`12 x 12 = 144` cells, each in `[0.0, 1.0]`).
    #[must_use]
    pub fn settled_pigment(&self) -> &[f32; CANVAS_CHANNELS] {
        &self.pigment
    }

    /// Physical mechanical work exerted by the hand/stylus on the latest tick (`[0.0, 1.5]`),
    /// used by external world objects (such as shelled forage pods) without altering pad rewards.
    #[must_use]
    pub fn mechanical_hand_work(&self) -> f32 {
        let speed = self.stylus_vx.hypot(self.stylus_vy);
        let contact_pressure = ((self.last_eff_pressure - 0.08) / 0.92).clamp(0.0, 1.0);
        (1.25 * speed * contact_pressure).clamp(0.0, 1.5)
    }

    /// Couples two organisms' tactile pads onto a shared physical stone hearth surface when they
    /// are close together in the habitat (`coupling in [0.0, 0.35]`), allowing each organism to
    /// feel and see the other's hand strokes directly.
    pub fn couple_shared_surface(&mut self, peer: &mut Self, coupling: f32) {
        let k = if coupling.is_finite() {
            coupling.clamp(0.0, 0.35)
        } else {
            0.0
        };
        if k <= 1.0e-5 {
            return;
        }
        for idx in 0..CANVAS_CHANNELS {
            let pa = self.pigment[idx];
            let pb = peer.pigment[idx];
            let shared_peak = pa.max(pb);
            self.pigment[idx] = (pa + k * (shared_peak - pa)).clamp(0.0, 1.0);
            peer.pigment[idx] = (pb + k * (shared_peak - pb)).clamp(0.0, 1.0);

            let ta = self.pigment_tone[idx];
            let tb = peer.pigment_tone[idx];
            let mid_t = 0.5 * (ta + tb);
            self.pigment_tone[idx] = (ta + k * (mid_t - ta)).clamp(-1.0, 1.0);
            peer.pigment_tone[idx] = (tb + k * (mid_t - tb)).clamp(-1.0, 1.0);
        }
        self.refresh_ambient();
        peer.refresh_ambient();
    }

    /// Computes read-only behavioral observation metrics without mutating world state.
    #[must_use]
    pub fn observation(&self) -> ScratchpadObservation {
        let inked_count = self
            .pigment
            .iter()
            .filter(|&&p| p >= INKED_CELL_THRESHOLD)
            .count();
        let canvas_coverage_fraction = inked_count as f32 / CANVAS_CHANNELS as f32;
        let spatial_autocorrelation = compute_morans_i(&self.pigment);
        let revisit_ratio = if self.active_visit_ticks == 0 {
            0.0
        } else {
            (self.revisit_ticks as f32 / self.active_visit_ticks as f32).clamp(0.0, 1.0)
        };
        let stylus_trajectory_entropy =
            compute_trajectory_entropy(&self.cell_visit_counts, &self.heading_octant_counts);

        ScratchpadObservation {
            active_stroke_ticks: self.active_stroke_ticks,
            canvas_coverage_fraction,
            spatial_autocorrelation,
            revisit_ratio,
            stylus_trajectory_entropy,
        }
    }

    fn deposit_stroke_segment(&mut self, seg: StrokeSegment) {
        let strength_norm = ((seg.eff_pressure - 0.1) / 0.9).clamp(0.0, 1.0);
        let flow_gain = 0.45 + 0.55 * (seg.speed / 0.35).min(1.0);
        let deposit_scale = 0.55 * strength_norm * flow_gain;
        if deposit_scale <= 0.0 {
            return;
        }

        let dist = ((seg.x1 - seg.x0).powi(2) + (seg.y1 - seg.y0).powi(2)).sqrt();
        let steps = ((dist / 0.18).ceil() as usize).clamp(1, 28);
        let inv_steps = 1.0 / steps as f32;

        // Fine nib brush falloff (sigma = 0.38): isolated taps leave crisp point marks while
        // sub-stepped continuous motion merges into connected lines, curves, and clusters.
        let inv_two_sigma_sq = 1.0 / (2.0 * 0.38 * 0.38);

        for s in 0..=steps {
            let frac = if steps == 0 { 1.0 } else { s as f32 * inv_steps };
            let px = seg.x0 + (seg.x1 - seg.x0) * frac;
            let py = seg.y0 + (seg.y1 - seg.y0) * frac;

            let min_c = (px.floor() as isize - 1).max(0) as usize;
            let max_c = (px.ceil() as isize + 1).min(CANVAS_SIDE as isize - 1) as usize;
            let min_r = (py.floor() as isize - 1).max(0) as usize;
            let max_r = (py.ceil() as isize + 1).min(CANVAS_SIDE as isize - 1) as usize;

            let sub_weight = if steps > 1 {
                deposit_scale * (1.8 / (steps as f32 + 1.0)).min(1.0)
            } else {
                deposit_scale
            };

            for r in min_r..=max_r {
                for c in min_c..=max_c {
                    let dx = c as f32 - px;
                    let dy = r as f32 - py;
                    let d2 = dx * dx + dy * dy;
                    if d2 > 1.10 {
                        continue;
                    }
                    let kernel = (-d2 * inv_two_sigma_sq).exp();
                    let delta = sub_weight * kernel;
                    if delta <= 1e-5 {
                        continue;
                    }
                    let idx = r * CANVAS_SIDE + c;
                    let prev_w = self.wet_pending[idx];
                    let next_w = (prev_w + delta).min(1.0);
                    if next_w > 1e-6 {
                        let blend = delta / (prev_w + delta);
                        self.wet_tone_pending[idx] =
                            (self.wet_tone_pending[idx] * (1.0 - blend) + seg.tone * blend)
                                .clamp(-1.0, 1.0);
                    }
                    self.wet_pending[idx] = next_w;
                }
            }
        }
    }

    fn sample_pigment_bilinear(&self, x: f32, y: f32) -> (f32, f32) {
        let cx = x.clamp(0.0, MAX_COORD);
        let cy = y.clamp(0.0, MAX_COORD);
        let c0 = (cx.floor() as usize).min(CANVAS_SIDE - 1);
        let r0 = (cy.floor() as usize).min(CANVAS_SIDE - 1);
        let c1 = (c0 + 1).min(CANVAS_SIDE - 1);
        let r1 = (r0 + 1).min(CANVAS_SIDE - 1);
        let tx = cx - c0 as f32;
        let ty = cy - r0 as f32;

        let p00 = self.pigment[r0 * CANVAS_SIDE + c0];
        let p10 = self.pigment[r0 * CANVAS_SIDE + c1];
        let p01 = self.pigment[r1 * CANVAS_SIDE + c0];
        let p11 = self.pigment[r1 * CANVAS_SIDE + c1];

        let t00 = self.pigment_tone[r0 * CANVAS_SIDE + c0];
        let t10 = self.pigment_tone[r0 * CANVAS_SIDE + c1];
        let t01 = self.pigment_tone[r1 * CANVAS_SIDE + c0];
        let t11 = self.pigment_tone[r1 * CANVAS_SIDE + c1];

        let p_top = p00 * (1.0 - tx) + p10 * tx;
        let p_bot = p01 * (1.0 - tx) + p11 * tx;
        let p = p_top * (1.0 - ty) + p_bot * ty;

        let t_top = t00 * (1.0 - tx) + t10 * tx;
        let t_bot = t01 * (1.0 - tx) + t11 * tx;
        let t = t_top * (1.0 - ty) + t_bot * ty;
        (p.clamp(0.0, 1.0), t.clamp(-1.0, 1.0))
    }

    fn refresh_ambient(&mut self) {
        let inv_fovea_var = 1.0 / (2.0 * 1.75 * 1.75);

        // 0..144: Coarse + foveally boosted 12x12 canvas wave channels
        for r in 0..CANVAS_SIDE {
            for c in 0..CANVAS_SIDE {
                let idx = r * CANVAS_SIDE + c;
                let dx = c as f32 - self.stylus_x;
                let dy = r as f32 - self.stylus_y;
                let fovea = (-(dx * dx + dy * dy) * inv_fovea_var).exp();

                // Settled pigment is dominant; wet_pending contributes a faint 0.15 pre-settling glint.
                let ink = (self.pigment[idx] + 0.15 * self.wet_pending[idx]).min(1.0);
                let tone_carrier = 0.65 + 0.35 * self.pigment_tone[idx];
                let mark_wave = ink * tone_carrier;
                let stylus_shadow = -0.08 * fovea * (1.0 - ink);
                let val = (0.35 + 0.65 * fovea) * mark_wave + stylus_shadow;
                self.ambient_buf[idx] = sanitize_unit_signed(val);
            }
        }

        // 144..153: Local 3x3 stylus-centered foveal patch of settled pigment
        let mut local_idx = CANVAS_CHANNELS;
        for dr in [-1.0_f32, 0.0, 1.0] {
            for dc in [-1.0_f32, 0.0, 1.0] {
                let (p, t) = self.sample_pigment_bilinear(self.stylus_x + dc, self.stylus_y + dr);
                let wave = p * (0.65 + 0.35 * t);
                self.ambient_buf[local_idx] = sanitize_unit_signed(wave);
                local_idx += 1;
            }
        }

        // 153..160: Stylus velocity, position, tactile drag, and local pigment gradient
        let (under_p, _) = self.sample_pigment_bilinear(self.stylus_x, self.stylus_y);
        let (px_pos, _) = self.sample_pigment_bilinear(self.stylus_x + 0.75, self.stylus_y);
        let (px_neg, _) = self.sample_pigment_bilinear(self.stylus_x - 0.75, self.stylus_y);
        let (py_pos, _) = self.sample_pigment_bilinear(self.stylus_x, self.stylus_y + 0.75);
        let (py_neg, _) = self.sample_pigment_bilinear(self.stylus_x, self.stylus_y - 0.75);

        let half = MAX_COORD * 0.5;
        self.ambient_buf[CANVAS_CHANNELS + 9] = sanitize_unit_signed(self.stylus_vx / 2.5);
        self.ambient_buf[CANVAS_CHANNELS + 10] = sanitize_unit_signed(self.stylus_vy / 2.5);
        self.ambient_buf[CANVAS_CHANNELS + 11] =
            sanitize_unit_signed((self.stylus_x - half) / half);
        self.ambient_buf[CANVAS_CHANNELS + 12] =
            sanitize_unit_signed((self.stylus_y - half) / half);
        self.ambient_buf[CANVAS_CHANNELS + 13] =
            sanitize_unit_signed(self.last_eff_pressure * (0.30 + 0.70 * under_p));
        self.ambient_buf[CANVAS_CHANNELS + 14] = sanitize_unit_signed(px_pos - px_neg);
        self.ambient_buf[CANVAS_CHANNELS + 15] = sanitize_unit_signed(py_pos - py_neg);

        // 160..192: Exogenous Noisy-TV static channels (strictly decoupled from organism actions)
        let tv_start = CANVAS_CHANNELS + STYLUS_CHANNELS;
        for ch in &mut self.ambient_buf[tv_start..SCRATCHPAD_AMBIENT_CHANNELS] {
            *ch = sanitize_unit_signed(self.rng.next_f32_signed() * 0.85);
        }
    }
}

impl WorldMedium for DrawingScratchpad {
    fn num_ambient_channels(&self) -> usize {
        SCRATCHPAD_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        SCRATCHPAD_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.basal_influx,
            tissue_stress: 0.0,
            num_actuators: SCRATCHPAD_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.tick = self.tick.saturating_add(1);

        let dx = actuator_at(pulse, 0);
        let dy = actuator_at(pulse, 1);
        let eff_pressure = actuator_at(pulse, 2).max(0.0);
        let tone = actuator_at(pulse, 3);
        let wipe_act = actuator_at(pulse, 4);

        // 1. Slow natural drying (0.9998/tick) + optional smooth sponge wipe when wipe_pulse > 0.5.
        let wipe_strength = ((wipe_act - 0.5) / 0.5).clamp(0.0, 1.0);
        let retention = NATURAL_DRYING_FACTOR * (1.0 - 0.35 * wipe_strength);
        for i in 0..CANVAS_CHANNELS {
            self.pigment[i] *= retention;
            if self.pigment[i] < 1e-6 {
                self.pigment[i] = 0.0;
                self.pigment_tone[i] = 0.0;
            }
        }

        // 2. Complete the 2-tick stroke-to-pigment settling stage:
        //    Strokes deposited on the previous tick (wet_pending) now become durable settled pigment.
        for i in 0..CANVAS_CHANNELS {
            let wet = self.wet_pending[i] * (1.0 - 0.35 * wipe_strength);
            if wet > 1e-6 {
                let prev_p = self.pigment[i];
                let next_p = (prev_p + wet).min(1.0);
                let blend = wet / (prev_p + wet).max(1e-6);
                self.pigment_tone[i] = (self.pigment_tone[i] * (1.0 - blend)
                    + self.wet_tone_pending[i] * blend)
                    .clamp(-1.0, 1.0);
                self.pigment[i] = next_p;
            }
            self.wet_pending[i] = 0.0;
            self.wet_tone_pending[i] = 0.0;
        }

        // 3. Stylus kinematics: lifted stylus glides freely; initial pen-down contact arrests
        //    air-glide momentum, and sustained pressed strokes experience smooth paper drag.
        let prev_x = self.stylus_x;
        let prev_y = self.stylus_y;
        if eff_pressure > 0.1 {
            let drag = if self.last_eff_pressure <= 0.1 {
                0.08
            } else {
                0.45 - 0.10 * eff_pressure
            };
            self.stylus_vx = self.stylus_vx * drag + 0.72 * dx;
            self.stylus_vy = self.stylus_vy * drag + 0.72 * dy;
        } else {
            self.stylus_vx = self.stylus_vx * 0.45 + 3.40 * dx;
            self.stylus_vy = self.stylus_vy * 0.45 + 3.40 * dy;
        }

        let unclamped_x = prev_x + self.stylus_vx;
        let unclamped_y = prev_y + self.stylus_vy;
        self.stylus_x = unclamped_x.clamp(0.0, MAX_COORD);
        self.stylus_y = unclamped_y.clamp(0.0, MAX_COORD);
        if (self.stylus_x - unclamped_x).abs() > 1e-5 {
            self.stylus_vx = 0.0;
        }
        if (self.stylus_y - unclamped_y).abs() > 1e-5 {
            self.stylus_vy = 0.0;
        }

        let step_dx = self.stylus_x - prev_x;
        let step_dy = self.stylus_y - prev_y;
        let speed = (step_dx * step_dx + step_dy * step_dy).sqrt();
        self.last_eff_pressure = eff_pressure;

        // 4. Behavioral observation bookkeeping (before depositing new ink at current locus).
        let cur_c = (self.stylus_x.round() as usize).min(CANVAS_SIDE - 1);
        let cur_r = (self.stylus_y.round() as usize).min(CANVAS_SIDE - 1);
        let cur_idx = cur_r * CANVAS_SIDE + cur_c;

        if speed > 0.01 || eff_pressure > 0.1 {
            self.active_visit_ticks = self.active_visit_ticks.saturating_add(1);
            self.cell_visit_counts[cur_idx] = self.cell_visit_counts[cur_idx].saturating_add(1);

            if self.pigment[cur_idx] >= INKED_CELL_THRESHOLD
                && matches!(
                    self.last_neighborhood_tick[cur_idx],
                    Some(last_t) if self.tick.saturating_sub(last_t) >= REVISIT_MIN_GAP_TICKS
                )
            {
                self.revisit_ticks = self.revisit_ticks.saturating_add(1);
            }
        }

        // Record exact cell occupancy tick so consecutive dwell ticks (dt < 6) are not counted as
        // revisits, while returning to a cell >= 6 ticks after departing registers cleanly.
        self.last_neighborhood_tick[cur_idx] = Some(self.tick);

        if speed > 0.01 {
            let angle = step_dy.atan2(step_dx); // [-pi, pi]
            let norm = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let octant = ((norm * 8.0).floor() as usize).min(7);
            self.heading_octant_counts[octant] =
                self.heading_octant_counts[octant].saturating_add(1);
            self.total_heading_samples = self.total_heading_samples.saturating_add(1);
        }

        // 5. Deposit new wet stroke if pressure > 0.1.
        if eff_pressure > 0.1 {
            if speed > 0.01 {
                self.active_stroke_ticks = self.active_stroke_ticks.saturating_add(1);
            }
            self.deposit_stroke_segment(StrokeSegment {
                x0: prev_x,
                y0: prev_y,
                x1: self.stylus_x,
                y1: self.stylus_y,
                eff_pressure,
                tone,
                speed,
            });
        }

        self.refresh_ambient();
    }
}

#[derive(Clone, Copy)]
struct StrokeSegment {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    eff_pressure: f32,
    tone: f32,
    speed: f32,
}

/// Computes Moran's I spatial autocorrelation over the `12 x 12` pigment grid using 8-connected neighbors.
fn compute_morans_i(grid: &[f32; CANVAS_CHANNELS]) -> f32 {
    let n = CANVAS_CHANNELS as f32;
    let mean = grid.iter().copied().sum::<f32>() / n;

    let mut denom = 0.0_f32;
    for &v in grid {
        let diff = v - mean;
        denom += diff * diff;
    }
    if denom <= 1e-8 {
        return 0.0;
    }

    let mut numer = 0.0_f32;
    let mut weight_sum = 0.0_f32;

    for r in 0..CANVAS_SIDE {
        for c in 0..CANVAS_SIDE {
            let xi = grid[r * CANVAS_SIDE + c] - mean;
            let r_lo = r.saturating_sub(1);
            let r_hi = (r + 1).min(CANVAS_SIDE - 1);
            let c_lo = c.saturating_sub(1);
            let c_hi = (c + 1).min(CANVAS_SIDE - 1);
            for nr in r_lo..=r_hi {
                for nc in c_lo..=c_hi {
                    if nr == r && nc == c {
                        continue;
                    }
                    let w = if nr == r || nc == c {
                        1.0
                    } else {
                        std::f32::consts::FRAC_1_SQRT_2
                    };
                    let xj = grid[nr * CANVAS_SIDE + nc] - mean;
                    numer += w * xi * xj;
                    weight_sum += w;
                }
            }
        }
    }

    if weight_sum <= 1e-8 {
        return 0.0;
    }
    ((n / weight_sum) * (numer / denom)).clamp(-1.0, 1.0)
}

fn compute_trajectory_entropy(
    cell_counts: &[u64; CANVAS_CHANNELS],
    octant_counts: &[u64; 8],
) -> f32 {
    let cell_total: u64 = cell_counts.iter().sum();
    let oct_total: u64 = octant_counts.iter().sum();
    if cell_total == 0 && oct_total == 0 {
        return 0.0;
    }

    let cell_h = if cell_total > 1 {
        let inv = 1.0 / cell_total as f32;
        let mut h = 0.0_f32;
        for &c in cell_counts {
            if c > 0 {
                let p = c as f32 * inv;
                h -= p * p.ln();
            }
        }
        (h / (CANVAS_CHANNELS as f32).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let oct_h = if oct_total > 1 {
        let inv = 1.0 / oct_total as f32;
        let mut h = 0.0_f32;
        for &c in octant_counts {
            if c > 0 {
                let p = c as f32 * inv;
                h -= p * p.ln();
            }
        }
        (h / 8.0_f32.ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (0.5 * cell_h + 0.5 * oct_h).clamp(0.0, 1.0)
}
