//! Metabolic homeostasis, constrained exploratory temperature, and adenosine rest bouts.

use a11_medium::ControlMode;

use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp, smooth_saturate};

pub const EXPLORE_CHANNELS: usize = 32;

#[derive(Clone, Debug, PartialEq)]
pub struct BodyCore {
    pub atp: f32,
    pub max_atp: f32,
    pub initial_atp: f32,
    pub adenosine: f32,
    pub temperature: f32,
    pub motor_clutch_threshold: f32,
    pub baseline_valence_ema: f32,
    pub prev_raw_valence: f32,
    pub valence_scale_ema: f32,
    pub valence_min_ema: f32,
    pub valence_max_ema: f32,
    pub ers_scarcity: f32,
    pub fast_valence: f32,
    pub rest_ticks_remaining: u16,
    pub rest_consolidation_ticks: u64,
    pub explore_momentum: [f32; EXPLORE_CHANNELS],
    pub explore_dc_accum: [f32; EXPLORE_CHANNELS],
    pub explore_target: [f32; EXPLORE_CHANNELS],
    pub explore_hold_ticks: [u8; EXPLORE_CHANNELS],
}

impl BodyCore {
    pub const LOGICAL_BYTES: usize = 4 * 13 + 2 + 8 + 4 * EXPLORE_CHANNELS * 3 + EXPLORE_CHANNELS;

    #[must_use]
    pub fn new(initial_atp: f32, max_atp: f32) -> Self {
        let clean_max = sanitize(max_atp).max(1.0);
        let clean_init = sanitize_clamp(initial_atp, 0.0, clean_max);
        Self {
            atp: clean_init,
            max_atp: clean_max,
            initial_atp: clean_init,
            adenosine: 0.0,
            temperature: 0.28,
            motor_clutch_threshold: 0.05,
            baseline_valence_ema: 0.0,
            prev_raw_valence: 0.0,
            valence_scale_ema: 0.0015,
            valence_min_ema: -0.04,
            valence_max_ema: 0.04,
            ers_scarcity: 0.0,
            fast_valence: 0.0,
            rest_ticks_remaining: 0,
            rest_consolidation_ticks: 0,
            explore_momentum: [0.0; EXPLORE_CHANNELS],
            explore_dc_accum: [0.0; EXPLORE_CHANNELS],
            explore_target: [0.0; EXPLORE_CHANNELS],
            explore_hold_ticks: [0; EXPLORE_CHANNELS],
        }
    }

    #[must_use]
    pub fn context_valence_span(&self) -> f32 {
        sanitize_clamp(self.valence_max_ema - self.valence_min_ema, 0.008, 4.0)
    }

    #[must_use]
    pub fn is_resting(&self) -> bool {
        self.rest_ticks_remaining > 0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_metabolism(
        &mut self,
        metabolic_influx: f32,
        tissue_stress: f32,
        alive_cells: usize,
        alive_tubes: usize,
        alive_pores: usize,
        last_motor_energy: f32,
        mean_surprise: f32,
        mode: ControlMode,
    ) -> f32 {
        let influx = sanitize(metabolic_influx).max(0.0);
        let stress = sanitize(tissue_stress).max(0.0);
        let motor_e = sanitize_clamp(last_motor_energy, 0.0, 4.0);
        let surprise = sanitize_clamp(mean_surprise, 0.0, 2.0);

        let basal_cost = 0.0008;
        let structure_rent = 0.000_010 * (alive_cells as f32)
            + 0.000_000_4 * (alive_tubes as f32)
            + 0.000_004 * (alive_pores as f32);
        let motor_cost = 0.0018 * motor_e;
        let repair_cost = 0.05 * stress;
        let total_burn = basal_cost + structure_rent + motor_cost + repair_cost;

        self.atp = sanitize_clamp(self.atp + influx - total_burn, 0.0, self.max_atp);

        let raw_valence = sanitize(2.6 * influx - 3.2 * stress);
        let delta_valence = sanitize(raw_valence - self.prev_raw_valence);
        self.prev_raw_valence = raw_valence;

        let level_adv = sanitize(raw_valence - self.baseline_valence_ema);
        let combined_signal = sanitize(0.25 * level_adv + 2.2 * delta_valence);
        let span = self.context_valence_span();
        let scale = (0.65 * self.valence_scale_ema + 0.12 * span).max(2.0e-3);
        let normalized_advantage = if combined_signal.abs() > 1.0e-7 {
            sanitize_clamp(0.45 * (combined_signal / scale), -2.0, 2.0)
        } else {
            0.0
        };

        self.fast_valence = raw_valence;
        if mode == ControlMode::Natural {
            self.baseline_valence_ema = sanitize_clamp(
                0.98 * self.baseline_valence_ema + 0.02 * raw_valence,
                -5.0,
                5.0,
            );
            if delta_valence.abs() > 1.0e-7 && delta_valence.abs() < 0.12 {
                self.valence_scale_ema = sanitize_clamp(
                    0.94 * self.valence_scale_ema + 0.06 * delta_valence.abs(),
                    1.0e-4,
                    1.0,
                );
            }
            let min_relax = self.baseline_valence_ema - 0.015;
            if raw_valence < self.valence_min_ema {
                self.valence_min_ema =
                    sanitize_clamp(0.86 * self.valence_min_ema + 0.14 * raw_valence, -5.0, 5.0);
            } else {
                self.valence_min_ema =
                    sanitize_clamp(0.996 * self.valence_min_ema + 0.004 * min_relax, -5.0, 5.0);
            }
            let max_relax = (self.baseline_valence_ema + 0.015).max(self.valence_min_ema + 0.008);
            if raw_valence > self.valence_max_ema {
                self.valence_max_ema = sanitize_clamp(
                    0.86 * self.valence_max_ema + 0.14 * raw_valence,
                    self.valence_min_ema + 0.008,
                    5.0,
                );
            } else {
                self.valence_max_ema = sanitize_clamp(
                    0.996 * self.valence_max_ema + 0.004 * max_relax,
                    self.valence_min_ema + 0.008,
                    5.0,
                );
            }

            let reserve_thresh = (0.50 * self.initial_atp).clamp(2.0, 12.0);
            let atp_deficit = ((reserve_thresh - self.atp) / reserve_thresh).clamp(0.0, 1.0);
            let chronic_stress = (-14.0 * self.baseline_valence_ema).clamp(0.0, 0.60);
            let net_gain_relief = if influx > total_burn { 0.35 } else { 0.0 };
            let ers_target =
                (0.70 * atp_deficit + 0.50 * chronic_stress - net_gain_relief).clamp(0.0, 1.0);
            self.ers_scarcity =
                sanitize_clamp(0.985 * self.ers_scarcity + 0.015 * ers_target, 0.0, 1.0);
        }

        let comfort_atp = 0.65 * self.max_atp;
        let hunger = if comfort_atp > 1.0e-4 {
            ((comfort_atp - self.atp) / comfort_atp).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let starvation_boost = if self.atp < 1.5 { 0.22 } else { 0.0 };
        let valence_drop = (self.baseline_valence_ema - raw_valence - 0.01).max(0.0)
            + (-delta_valence - 0.02).max(0.0);
        let somatic_surprise = sanitize_clamp(2.2 * valence_drop, 0.0, 0.75);
        let temp_target = sanitize_clamp(
            0.05 + 0.45 * hunger
                + 0.30 * surprise.min(1.0)
                + 0.26 * stress.min(1.0)
                + 0.45 * somatic_surprise
                + starvation_boost,
            0.03,
            1.0,
        );
        self.temperature = sanitize_clamp(0.90 * self.temperature + 0.10 * temp_target, 0.03, 1.0);

        if self.rest_ticks_remaining > 0 {
            if stress >= 0.035 || somatic_surprise > 0.12 || surprise > 0.16 {
                self.rest_ticks_remaining = 0;
                self.motor_clutch_threshold = 0.05;
            } else {
                self.rest_ticks_remaining -= 1;
                self.rest_consolidation_ticks = self.rest_consolidation_ticks.saturating_add(1);
                self.motor_clutch_threshold = 0.78;
                self.adenosine = sanitize_clamp(self.adenosine - 0.042, 0.0, 1.0);
            }
        } else {
            self.adenosine = sanitize_clamp(self.adenosine + 0.0017 + 0.045 * total_burn, 0.0, 1.0);
            if self.adenosine > 0.72 && stress < 0.05 {
                self.rest_ticks_remaining = 20;
                self.rest_consolidation_ticks = self.rest_consolidation_ticks.saturating_add(1);
                self.motor_clutch_threshold = 0.78;
                self.adenosine = sanitize_clamp(self.adenosine - 0.042, 0.0, 1.0);
            } else {
                let awake_clutch = 0.045;
                self.motor_clutch_threshold = sanitize_clamp(
                    0.80 * self.motor_clutch_threshold + 0.20 * awake_clutch,
                    0.04,
                    0.85,
                );
            }
        }

        normalized_advantage
    }

    pub fn next_exploratory_wave(&mut self, rng: &mut Rng64, out: &mut [f32]) {
        let amp = self.temperature;
        for (idx, dst) in out.iter_mut().enumerate() {
            let slot = idx % EXPLORE_CHANNELS;
            if self.explore_hold_ticks[slot] == 0 {
                let raw = rng.next_f32_signed();
                let anti_dc = sanitize_clamp(-0.35 * self.explore_dc_accum[slot], -0.65, 0.65);
                let flip_nudge = if self.explore_dc_accum[slot].abs() > 0.45 {
                    -0.35 * self.explore_dc_accum[slot].signum()
                } else {
                    0.0
                };
                self.explore_target[slot] = smooth_saturate(0.95 * raw + anti_dc + flip_nudge);
                self.explore_hold_ticks[slot] = 6 + (rng.next_u32() % 7) as u8;
            } else {
                self.explore_hold_ticks[slot] -= 1;
            }

            let target = self.explore_target[slot];
            let m = sanitize_clamp(
                0.80 * self.explore_momentum[slot] + 0.20 * target,
                -1.0,
                1.0,
            );
            self.explore_momentum[slot] = m;
            self.explore_dc_accum[slot] =
                sanitize_clamp(0.97 * self.explore_dc_accum[slot] + 0.03 * m, -1.5, 1.5);
            *dst = sanitize_clamp(m * (0.28 + 0.72 * amp), -1.0, 1.0);
        }
    }

    pub fn zero_learned_state(&mut self) {
        self.adenosine = 0.0;
        self.temperature = 0.28;
        self.motor_clutch_threshold = 0.05;
        self.baseline_valence_ema = 0.0;
        self.prev_raw_valence = 0.0;
        self.valence_scale_ema = 0.0015;
        self.valence_min_ema = -0.04;
        self.valence_max_ema = 0.04;
        self.ers_scarcity = 0.0;
        self.fast_valence = 0.0;
        self.rest_ticks_remaining = 0;
        self.rest_consolidation_ticks = 0;
        self.explore_momentum = [0.0; EXPLORE_CHANNELS];
        self.explore_dc_accum = [0.0; EXPLORE_CHANNELS];
        self.explore_target = [0.0; EXPLORE_CHANNELS];
        self.explore_hold_ticks = [0; EXPLORE_CHANNELS];
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_f32(self.baseline_valence_ema);
        hasher.write_f32(self.valence_scale_ema);
        hasher.write_f32(self.valence_min_ema);
        hasher.write_f32(self.valence_max_ema);
        hasher.write_f32(self.ers_scarcity);
    }
}
