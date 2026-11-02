//! Continuous Newtonian rocker-beam & rolling-sphere cradle (`BalanceCatchToy`).
//!
//! Exposes `128` unlabelled ambient channels (`96` continuous spatial/tilt/vestibular wave
//! channels + `32` exogenous Noisy-TV static channels) and `3` continuous actuators
//! (`torque`, `damping_grip`, `nudge`).
//!
//! If the sphere rolls past the cradle edge (`|ball_pos| >= 1.0`), it softly re-centers with
//! a fresh gentle velocity. Zero game-score `metabolic_influx` is ever injected.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::{SplitMix64, actuator_at, sanitize_nonneg, sanitize_unit_signed};

/// Spatial/tilt/vestibular wave channels along the physical cradle.
pub const CRADLE_WAVE_CHANNELS: usize = 96;
/// Exogenous static distractor channels (`Noisy TV`).
pub const BALANCE_NOISY_TV_CHANNELS: usize = 32;
/// Total ambient channels exposed by [`BalanceCatchToy`] (`96 + 32 = 128`).
pub const BALANCE_TOY_AMBIENT_CHANNELS: usize = CRADLE_WAVE_CHANNELS + BALANCE_NOISY_TV_CHANNELS;
/// Number of continuous actuators (`torque`, `damping_grip`, `nudge`).
pub const BALANCE_TOY_ACTUATORS: usize = 3;

/// Default flat basal metabolic trickle in standalone portal mode (`>= 0.0`).
pub const DEFAULT_BALANCE_BASAL_INFLUX: f32 = 0.008;
/// Predictive phase-lead horizon (`tau` ticks) used in `torque_phase_lead_correlation`.
pub const PHASE_LEAD_TAU_TICKS: f32 = 4.0;

const MAX_BEAM_ANGLE: f32 = 0.65;

/// Read-only behavioral observation metrics for [`BalanceCatchToy`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BalanceToyObservation {
    /// Mean consecutive upright ticks across completed and ongoing balance streaks.
    pub mean_upright_duration_ticks: f32,
    /// Longest unbroken upright streak observed so far.
    pub max_upright_streak: u64,
    /// Number of times the sphere rolled off either edge (`|ball_pos| >= 1.0`).
    pub sphere_falls: u64,
    /// Pearson correlation (`[-1.0, 1.0]`) between opposing torque (`-torque`) and predictive
    /// edge-approach state (`ball_pos + tau * ball_vel`) evaluated strictly before the torque step.
    pub torque_phase_lead_correlation: f32,
}

/// Newtonian rocker-beam and rolling-sphere cradle under gravity and gentle wind gusts.
#[derive(Clone, Debug)]
pub struct BalanceCatchToy {
    tick: u64,
    basal_influx: f32,
    crash_jolt_stress: f32,
    last_tick_stress: f32,
    rng: SplitMix64,

    angle: f32,
    angular_vel: f32,
    ball_pos: f32,
    ball_vel: f32,
    wind: f32,

    // Read-only behavioral observation accumulators
    current_upright_streak: u64,
    max_upright_streak: u64,
    completed_streaks_sum: u64,
    completed_streaks_count: u64,
    sphere_falls: u64,

    corr_n: u64,
    sum_lead: f64,
    sum_opp_torque: f64,
    sum_lead_sq: f64,
    sum_opp_torque_sq: f64,
    sum_cross: f64,

    ambient_buf: Vec<f32>,
}

impl Default for BalanceCatchToy {
    fn default() -> Self {
        Self::new(0xBA1A_9CE7_0011_A110)
    }
}

impl BalanceCatchToy {
    /// Creates a new `BalanceCatchToy` in pure zero-stress play mode (`crash_jolt_stress = 0.0`).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_crash_jolt(seed, DEFAULT_BALANCE_BASAL_INFLUX, 0.0)
    }

    /// Creates a new `BalanceCatchToy` with explicit `basal_influx` and optional somatic crash jolt
    /// (e.g. `0.02` on hard edge drop, or `0.0` in pure play mode).
    #[must_use]
    pub fn with_crash_jolt(seed: u64, basal_influx: f32, crash_jolt_stress: f32) -> Self {
        let mut rng = SplitMix64::new(seed);
        let sign = if (rng.next_u64() & 1) == 0 {
            1.0_f32
        } else {
            -1.0_f32
        };
        let init_pos = sign * (0.10 + 0.04 * rng.next_f32_unit());
        let init_vel = sign * (0.015 + 0.008 * rng.next_f32_unit());

        let mut toy = Self {
            tick: 0,
            basal_influx: sanitize_nonneg(basal_influx),
            crash_jolt_stress: sanitize_nonneg(crash_jolt_stress),
            last_tick_stress: 0.0,
            rng,
            angle: 0.02 * sign,
            angular_vel: 0.0,
            ball_pos: init_pos,
            ball_vel: init_vel,
            wind: 0.0,
            current_upright_streak: 0,
            max_upright_streak: 0,
            completed_streaks_sum: 0,
            completed_streaks_count: 0,
            sphere_falls: 0,
            corr_n: 0,
            sum_lead: 0.0,
            sum_opp_torque: 0.0,
            sum_lead_sq: 0.0,
            sum_opp_torque_sq: 0.0,
            sum_cross: 0.0,
            ambient_buf: vec![0.0; BALANCE_TOY_AMBIENT_CHANNELS],
        };
        toy.refresh_ambient();
        toy
    }

    /// Current simulation tick of this cradle.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Current physical state `(angle, angular_vel, ball_pos, ball_vel)` for external inspection/tests.
    #[must_use]
    pub fn physical_state(&self) -> (f32, f32, f32, f32) {
        (self.angle, self.angular_vel, self.ball_pos, self.ball_vel)
    }

    /// Instantaneous postural stability (`[0.25, 1.0]`), governing locomotive traction and hand
    /// precision when coupled to a physical 2D terrain.
    #[must_use]
    pub fn upright_stability(&self) -> f32 {
        let ball_dev = self.ball_pos.abs().clamp(0.0, 1.0);
        let tilt_dev = (self.angle.abs() / MAX_BEAM_ANGLE).clamp(0.0, 1.0);
        (1.0 - 0.55 * ball_dev - 0.30 * tilt_dev).clamp(0.25, 1.0)
    }

    /// Applies real 2D terrain acceleration/slope sway into the vestibular pendulum and refreshes
    /// postural receptors.
    pub fn apply_terrain_sway(&mut self, lateral_accel: f32) {
        if !lateral_accel.is_finite() || lateral_accel.abs() <= 1.0e-6 {
            return;
        }
        let bounded = lateral_accel.clamp(-0.022, 0.022);
        self.angular_vel = (self.angular_vel + bounded).clamp(-0.30, 0.30);
        self.refresh_ambient();
    }

    /// Computes read-only behavioral observation metrics without mutating world state.
    #[must_use]
    pub fn observation(&self) -> BalanceToyObservation {
        let total_streaks =
            self.completed_streaks_count + u64::from(self.current_upright_streak > 0);
        let mean_upright_duration_ticks = if total_streaks == 0 {
            0.0
        } else {
            (self.completed_streaks_sum + self.current_upright_streak) as f32
                / total_streaks as f32
        };

        let torque_phase_lead_correlation = compute_pearson(
            self.corr_n,
            self.sum_lead,
            self.sum_opp_torque,
            self.sum_lead_sq,
            self.sum_opp_torque_sq,
            self.sum_cross,
        );

        BalanceToyObservation {
            mean_upright_duration_ticks,
            max_upright_streak: self.max_upright_streak,
            sphere_falls: self.sphere_falls,
            torque_phase_lead_correlation,
        }
    }

    fn refresh_ambient(&mut self) {
        // 0..48: Continuous spatial receptor array tracking rolling sphere position along [-1.0, 1.0]
        let inv_ball_var = 1.0 / (2.0 * 0.10 * 0.10);
        let roll_phase = self.ball_pos * 12.0;
        for i in 0..48 {
            let x_i = -1.0 + 2.0 * (i as f32 / 47.0);
            let diff = x_i - self.ball_pos;
            let env = (-diff * diff * inv_ball_var).exp();
            let tex = 0.80 + 0.20 * (roll_phase + i as f32 * 0.3).cos();
            self.ambient_buf[i] = sanitize_unit_signed(env * tex);
        }

        // 48..64: Population-coded beam tilt angle across [-MAX_BEAM_ANGLE, MAX_BEAM_ANGLE]
        let norm_angle = (self.angle / MAX_BEAM_ANGLE).clamp(-1.0, 1.0);
        let inv_tilt_var = 1.0 / (2.0 * 0.18 * 0.18);
        for i in 0..16 {
            let center = -1.0 + 2.0 * (i as f32 / 15.0);
            let d = center - norm_angle;
            self.ambient_buf[48 + i] = sanitize_unit_signed((-d * d * inv_tilt_var).exp());
        }

        // 64..80: Population-coded angular & sphere velocity wave receptors
        let norm_bvel = (self.ball_vel / 0.18).clamp(-1.0, 1.0);
        for i in 0..16 {
            let center = -1.0 + 2.0 * (i as f32 / 15.0);
            let d = center - norm_bvel;
            self.ambient_buf[64 + i] = sanitize_unit_signed((-d * d * inv_tilt_var).exp());
        }

        // 80..96: Direct vestibular, rocker-contact, wind, and edge-vibration summary channels
        let norm_avel = (self.angular_vel / 0.20).clamp(-1.0, 1.0);
        let edge_prox = self.ball_pos.abs().powi(3).clamp(0.0, 1.0);
        let t = self.tick as f32;

        self.ambient_buf[80] = sanitize_unit_signed(self.ball_pos);
        self.ambient_buf[81] = sanitize_unit_signed(norm_bvel);
        self.ambient_buf[82] = sanitize_unit_signed(norm_angle);
        self.ambient_buf[83] = sanitize_unit_signed(norm_avel);
        // Channel 84 previously broadcast `lead_wave = ball_pos + PHASE_LEAD_TAU_TICKS * ball_vel`, the exact
        // quantity the phase-lead metric scores (answer leak, audit P26). The organism must infer it.
        self.ambient_buf[84] = 0.0;
        self.ambient_buf[85] = sanitize_unit_signed(edge_prox * (t * 0.9).sin());
        self.ambient_buf[86] = sanitize_unit_signed(edge_prox * self.ball_pos.signum());
        self.ambient_buf[87] = sanitize_unit_signed((self.wind * 40.0).clamp(-1.0, 1.0));
        self.ambient_buf[88] = sanitize_unit_signed((-self.ball_pos).max(0.0));
        self.ambient_buf[89] = sanitize_unit_signed(self.ball_pos.max(0.0));
        self.ambient_buf[90] = sanitize_unit_signed((-norm_angle).max(0.0));
        self.ambient_buf[91] = sanitize_unit_signed(norm_angle.max(0.0));
        self.ambient_buf[92] = sanitize_unit_signed((self.ball_pos * self.ball_vel * 8.0).clamp(-1.0, 1.0));
        self.ambient_buf[93] = sanitize_unit_signed((self.angle.sin() * 1.5).clamp(-1.0, 1.0));
        self.ambient_buf[94] = sanitize_unit_signed((self.angle.cos() * 2.0 - 1.0).clamp(-1.0, 1.0));
        self.ambient_buf[95] = sanitize_unit_signed(1.0 - edge_prox);

        // 96..128: Exogenous Noisy-TV static channels
        for ch in &mut self.ambient_buf[CRADLE_WAVE_CHANNELS..BALANCE_TOY_AMBIENT_CHANNELS] {
            *ch = sanitize_unit_signed(self.rng.next_f32_signed() * 0.85);
        }
    }
}

impl WorldMedium for BalanceCatchToy {
    fn num_ambient_channels(&self) -> usize {
        BALANCE_TOY_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        BALANCE_TOY_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.basal_influx,
            tissue_stress: self.last_tick_stress,
            num_actuators: BALANCE_TOY_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.tick = self.tick.saturating_add(1);

        let torque = actuator_at(pulse, 0);
        let grip = actuator_at(pulse, 1).max(0.0);
        let nudge = actuator_at(pulse, 2);

        // 1. Evaluate phase-lead correlation strictly against pre-step state (t-1 -> t alignment).
        let lead_state = f64::from(self.ball_pos + PHASE_LEAD_TAU_TICKS * self.ball_vel);
        let opp_torque = f64::from(-torque);
        self.corr_n = self.corr_n.saturating_add(1);
        self.sum_lead += lead_state;
        self.sum_opp_torque += opp_torque;
        self.sum_lead_sq += lead_state * lead_state;
        self.sum_opp_torque_sq += opp_torque * opp_torque;
        self.sum_cross += lead_state * opp_torque;

        // 2. Smooth Ornstein-Uhlenbeck gentle wind gust.
        self.wind = 0.90 * self.wind + 0.0025 * self.rng.next_f32_signed();

        // 3. Coupled Newtonian rocker-beam + rolling-sphere semi-implicit Euler step.
        let ang_acc = 0.095 * torque
            + 0.045 * self.ball_pos * self.angle.cos()
            - 0.14 * self.angle
            - (0.34 + 0.26 * grip) * self.angular_vel
            + 0.25 * self.wind;
        self.angular_vel = (self.angular_vel + ang_acc).clamp(-0.35, 0.35);
        self.angle = (self.angle + self.angular_vel).clamp(-MAX_BEAM_ANGLE, MAX_BEAM_ANGLE);
        if self.angle.abs() >= MAX_BEAM_ANGLE - 1e-5 {
            self.angular_vel *= 0.2;
        }

        let ball_acc = 0.052 * self.angle.sin()
            + 0.02 * self.ball_pos * self.angular_vel * self.angular_vel
            + self.wind
            - 0.035 * self.ball_vel
            - 0.018 * nudge;
        self.ball_vel = (self.ball_vel + ball_acc).clamp(-0.45, 0.45);
        self.ball_pos += self.ball_vel;

        // 4. Edge-fall check and soft re-centering.
        if !self.ball_pos.is_finite() || self.ball_pos.abs() >= 1.0 {
            self.sphere_falls = self.sphere_falls.saturating_add(1);
            self.completed_streaks_sum =
                self.completed_streaks_sum.saturating_add(self.current_upright_streak);
            self.completed_streaks_count = self.completed_streaks_count.saturating_add(1);
            self.current_upright_streak = 0;
            self.last_tick_stress = self.crash_jolt_stress;

            let dir = if (self.rng.next_u64() & 1) == 0 {
                1.0_f32
            } else {
                -1.0_f32
            };
            self.ball_pos = self.rng.next_f32_signed() * 0.10;
            self.ball_vel = dir * (0.015 + 0.015 * self.rng.next_f32_unit());
            self.angle *= 0.25;
            self.angular_vel = 0.0;
        } else {
            self.current_upright_streak = self.current_upright_streak.saturating_add(1);
            if self.current_upright_streak > self.max_upright_streak {
                self.max_upright_streak = self.current_upright_streak;
            }
            self.last_tick_stress = 0.0;
        }

        self.refresh_ambient();
    }
}

fn compute_pearson(
    n: u64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
) -> f32 {
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let var_x = (sum_xx - (sum_x * sum_x) / nf).max(0.0);
    let var_y = (sum_yy - (sum_y * sum_y) / nf).max(0.0);
    let denom = (var_x * var_y).sqrt();
    if denom <= 1e-10 {
        return 0.0;
    }
    let cov = sum_xy - (sum_x * sum_y) / nf;
    ((cov / denom) as f32).clamp(-1.0, 1.0)
}
