//! Membrane sensor and motor pores with bounded O(1) rolling cohort scouting,
//! Newtonian C1 / sparse-echo / motor-contingent organogenesis, and receptor adaptation.

use a11_medium::ControlMode;

use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp};

pub const SCOUT_COHORT_WIDTH: usize = 32;
pub const MAX_SCOUT_CHANNELS: usize = 16_384;
pub const MAX_ACTIVE_SENSOR_PORES_PER_TICK: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SensorPore {
    pub alive: bool,
    pub channel: u32,
    pub delay: u8,
    pub target_cell: u32,
    pub delta_target_cell: u32,
    pub conductance: f32,
    pub trophic_ema: f32,
    pub last_val: f32,
    pub adapt_mean: f32,
    pub adapt_init: bool,
    pub history: [f32; 16],
    pub hist_head: u8,
    pub age_ticks: u32,
}

impl SensorPore {
    pub const LOGICAL_BYTES: usize = 1 + 4 + 1 + 4 + 4 + 4 * 4 + 1 + 4 * 16 + 1 + 4;

    #[must_use]
    pub fn newborn(
        channel: u32,
        delay: u8,
        target_cell: u32,
        delta_target_cell: u32,
        conductance: f32,
    ) -> Self {
        Self {
            alive: true,
            channel,
            delay: delay.clamp(1, 16),
            target_cell,
            delta_target_cell,
            conductance: sanitize_clamp(conductance, 0.0, 2.0),
            trophic_ema: 0.04,
            last_val: 0.0,
            adapt_mean: 0.0,
            adapt_init: false,
            history: [0.0; 16],
            hist_head: 0,
            age_ticks: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotorPore {
    pub alive: bool,
    pub actuator_channel: u32,
    pub conductance: f32,
    pub klinokinesis_gain: f32,
    pub motor_mastery: f32,
    pub jerk_ema: f32,
    pub last_out: f32,
}

impl MotorPore {
    pub const LOGICAL_BYTES: usize = 1 + 4 + 4 * 5;

    #[must_use]
    pub fn newborn(actuator_channel: u32, conductance: f32) -> Self {
        Self {
            alive: true,
            actuator_channel,
            conductance: sanitize_clamp(conductance, 0.0, 2.0),
            klinokinesis_gain: 0.0,
            motor_mastery: 0.0,
            jerk_ema: 0.42,
            last_out: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ChannelScoutStat {
    pub initialized: bool,
    pub mean: f32,
    pub var_ema: f32,
    pub diff_sq_ema: f32,
    pub prev_val: f32,
    pub ar1_w: f32,
    pub ar1_err_ema: f32,
    pub prev_dx: f32,
    pub cov_dx_ema: f32,
    pub fourth_ema: f32,
    pub motor_cov_ema: f32,
}

impl ChannelScoutStat {
    pub const LOGICAL_BYTES: usize = 1 + 4 * 10;

    #[inline]
    #[must_use]
    pub fn passive_quality(&self) -> f32 {
        if !self.initialized || self.var_ema < 4.0e-6 {
            return 0.0;
        }
        let roughness = self.diff_sq_ema / self.var_ema.max(1.0e-8);
        let ar1_ratio = self.ar1_err_ema / self.var_ema.max(1.0e-8);
        let vel_corr = self.cov_dx_ema / self.diff_sq_ema.max(1.0e-8);
        let kurtosis = self.fourth_ema / (self.var_ema * self.var_ema).max(1.0e-10);
        let motor_r = self.motor_cov_ema.abs();

        // Stationary 1st-order OU(rho) drift has vel_corr = -(1 - rho)/2 <= 0, kurtosis ~ 2.95,
        // and motor_r ~ 0; white NoisyTV has roughness ~ 2.0 and vel_corr ~ -0.50.
        // Real physical waves, cochlear hair cells, viscoelastic limbs, and sparse wave echoes
        // exhibit C1 inertial flow (vel_corr > 0.12), non-Gaussian wave arrivals (kurtosis > 5.4),
        // or direct efference contingency (motor_r > 0.16).
        let is_c1_wave =
            self.var_ema >= 1.5e-5 && roughness < 0.72 && ar1_ratio < 0.78 && vel_corr > 0.12;
        let is_sparse_echo = !self.is_smooth_spatial_field()
            && roughness > 0.72
            && roughness < 2.25
            && vel_corr < 0.06
            && kurtosis > 4.5
            && self.var_ema < 0.050;
        let is_motor_contingent = self.var_ema >= 1.5e-5 && roughness < 1.35 && motor_r > 0.16;

        if !(is_c1_wave || is_sparse_echo || is_motor_contingent) {
            return 0.0;
        }

        let wave_q = if is_c1_wave {
            let smooth_gain = ((0.72 - roughness) / 0.72).clamp(0.0, 1.0);
            let pred_gain = ((0.78 - ar1_ratio) / 0.78).clamp(0.0, 1.0);
            let inertia_gain = ((vel_corr - 0.10) / 0.55).clamp(0.18, 1.0);
            0.5 * (smooth_gain + pred_gain) * inertia_gain
        } else {
            0.0
        };
        let echo_q = if is_sparse_echo {
            ((kurtosis - 4.0) / 9.5).clamp(0.28, 0.64)
        } else {
            0.0
        };
        let motor_q = if is_motor_contingent {
            ((motor_r - 0.14) / 0.45).clamp(0.18, 0.90)
        } else {
            0.0
        };

        let raw_q = wave_q.max(echo_q).max(motor_q);
        let amp_mult = if is_sparse_echo { 10.5 } else { 3.2 };
        let amp_scale = (self.var_ema.sqrt() * amp_mult).clamp(0.12, 1.0);
        sanitize_clamp(raw_q * amp_scale, 0.0, 1.0)
    }

    #[inline]
    #[must_use]
    pub fn is_sparse_impulse(&self) -> bool {
        if !self.initialized
            || self.var_ema < 4.0e-6
            || self.var_ema >= 0.050
            || self.is_smooth_spatial_field()
        {
            return false;
        }
        let roughness = self.diff_sq_ema / self.var_ema.max(1.0e-8);
        let vel_corr = self.cov_dx_ema / self.diff_sq_ema.max(1.0e-8);
        let kurtosis = self.fourth_ema / (self.var_ema * self.var_ema).max(1.0e-10);
        roughness > 0.72 && roughness < 2.25 && vel_corr < 0.06 && kurtosis > 4.5
    }

    #[inline]
    #[must_use]
    pub fn is_smooth_spatial_field(&self) -> bool {
        if !self.initialized || self.var_ema < 4.0e-6 {
            return false;
        }
        let roughness = self.diff_sq_ema / self.var_ema.max(1.0e-8);
        let ar1_ratio = self.ar1_err_ema / self.var_ema.max(1.0e-8);
        let vel_corr = self.cov_dx_ema / self.diff_sq_ema.max(1.0e-8);
        roughness < 0.42 && ar1_ratio < 0.45 && vel_corr > 0.26
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PoreMembrane {
    pub sensor_pores: Vec<SensorPore>,
    pub sensor_free_list: Vec<u32>,
    pub alive_sensor_count: usize,
    pub motor_pores: Vec<MotorPore>,
    pub motor_free_list: Vec<u32>,
    pub alive_motor_count: usize,
    pub scout_stats: Vec<ChannelScoutStat>,
    pub scout_cohort_idx: u32,
    pub passive_health_ema: f32,
    pub motor_env_fast: f32,
    pub motor_env_mean: f32,
    pub motor_env_var: f32,
}

impl PoreMembrane {
    #[must_use]
    pub fn new(initial_cells: usize, max_pores: usize, rng: &mut Rng64) -> Self {
        let cell_cap = initial_cells.max(4) as u32;
        let half = (cell_cap / 2).max(1);
        let init_sensors = 20usize.min(max_pores.max(1));
        let mut ch_pool: [u32; 32] = std::array::from_fn(|k| k as u32);
        for i in (1..32).rev() {
            let j = rng.next_index(i + 1);
            ch_pool.swap(i, j);
        }
        let mut sensor_pores =
            Vec::with_capacity(init_sensors.saturating_mul(2).min(max_pores.max(1)));
        for i in 0..init_sensors {
            let ch = ch_pool[i % 32];
            let tgt = (rng.next_index(half as usize)) as u32;
            let rem = (cell_cap - half).max(1) as usize;
            let dtgt = (half + (rng.next_index(rem) as u32)) % cell_cap;
            sensor_pores.push(SensorPore::newborn(ch, 1, tgt, dtgt, 0.16));
        }

        let init_motors = 8usize.min(max_pores.max(1));
        let mut motor_pores =
            Vec::with_capacity(init_motors.saturating_mul(2).min(max_pores.max(1)));
        for m in 0..init_motors {
            motor_pores.push(MotorPore::newborn(m as u32, 0.45));
        }

        Self {
            sensor_pores,
            sensor_free_list: Vec::new(),
            alive_sensor_count: init_sensors,
            motor_pores,
            motor_free_list: Vec::new(),
            alive_motor_count: init_motors,
            scout_stats: Vec::new(),
            scout_cohort_idx: 0,
            passive_health_ema: 0.28,
            motor_env_fast: 0.0,
            motor_env_mean: 0.0,
            motor_env_var: 0.01,
        }
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        self.sensor_pores.len() * SensorPore::LOGICAL_BYTES
            + self.sensor_free_list.len() * 4
            + self.motor_pores.len() * MotorPore::LOGICAL_BYTES
            + self.motor_free_list.len() * 4
            + self.scout_stats.len() * ChannelScoutStat::LOGICAL_BYTES
            + 36
    }

    #[must_use]
    pub fn total_alive_pores(&self) -> usize {
        self.alive_sensor_count + self.alive_motor_count
    }

    pub fn ensure_actuators_bound(
        &mut self,
        num_actuators: usize,
        max_pores: usize,
        mode: ControlMode,
    ) {
        if mode != ControlMode::Natural {
            return;
        }
        let target = num_actuators.min(32);
        for act in 0..target {
            let exists = self
                .motor_pores
                .iter()
                .any(|mp| mp.alive && mp.actuator_channel as usize == act);
            if !exists && self.total_alive_pores() < max_pores {
                let pore = MotorPore::newborn(act as u32, 0.45);
                if let Some(slot) = self.motor_free_list.pop() {
                    self.motor_pores[slot as usize] = pore;
                } else {
                    self.motor_pores.push(pore);
                }
                self.alive_motor_count += 1;
            }
        }
    }

    pub fn prepare_scout_cycle(
        &mut self,
        n_ambient: usize,
        _num_actuators: usize,
        last_motor_energy: f32,
        mode: ControlMode,
        rng: &mut Rng64,
    ) {
        if mode != ControlMode::Natural {
            return;
        }
        let me = sanitize_clamp(last_motor_energy.sqrt(), 0.0, 2.0);
        self.motor_env_fast = sanitize_clamp(0.76 * self.motor_env_fast + 0.24 * me, 0.0, 2.0);
        let dm = sanitize(self.motor_env_fast - self.motor_env_mean);
        self.motor_env_mean = sanitize_clamp(
            0.98 * self.motor_env_mean + 0.02 * self.motor_env_fast,
            0.0,
            2.0,
        );
        self.motor_env_var =
            sanitize_clamp(0.96 * self.motor_env_var + 0.04 * dm * dm, 1.0e-5, 4.0);

        let capped_ambient = n_ambient.min(MAX_SCOUT_CHANNELS);
        if self.scout_stats.is_empty() && capped_ambient > 32 {
            let mut chosen_chs: Vec<u32> = Vec::with_capacity(self.sensor_pores.len());
            for sp in &mut self.sensor_pores {
                if !sp.alive || sp.age_ticks > 0 {
                    continue;
                }
                let mut cand = rng.next_index(capped_ambient) as u32;
                for _ in 0..8 {
                    if !chosen_chs.contains(&cand) {
                        break;
                    }
                    cand = rng.next_index(capped_ambient) as u32;
                }
                sp.channel = cand;
                chosen_chs.push(cand);
            }
        }
        if self.scout_stats.len() < capped_ambient {
            self.scout_stats
                .resize(capped_ambient, ChannelScoutStat::default());
        }
        if capped_ambient == 0 {
            return;
        }
        let num_cohorts = capped_ambient.div_ceil(SCOUT_COHORT_WIDTH).max(1) as u32;
        self.scout_cohort_idx = (self.scout_cohort_idx + 1) % num_cohorts;
    }

    pub fn absorb_and_scout(
        &mut self,
        ambient_in: &[f32],
        cell_drive: &mut [f32],
        sensory_trophic_out: &mut [f32],
        mode: ControlMode,
    ) {
        let n_amb = ambient_in.len();
        let capped_amb = n_amb.min(MAX_SCOUT_CHANNELS);
        let motor_dev = sanitize(self.motor_env_fast - self.motor_env_mean);
        let motor_std = self.motor_env_var.max(1.0e-5).sqrt();

        if mode == ControlMode::Natural && capped_amb > 0 {
            let start = ((self.scout_cohort_idx as usize) * SCOUT_COHORT_WIDTH).min(capped_amb);
            let end = (start + SCOUT_COHORT_WIDTH).min(capped_amb);
            for (offset, st) in self.scout_stats[start..end].iter_mut().enumerate() {
                let c = start + offset;
                let x = sanitize_clamp(ambient_in[c], -4.0, 4.0);
                update_channel_stat(st, x, motor_dev, motor_std);
            }
        }

        // First pass over alive sensor pores: receptor delay line, calcium adaptation, and lateral surround mean.
        let mut active_signals: [(u32, u32, f32, f32, f32, f32); MAX_ACTIVE_SENSOR_PORES_PER_TICK] =
            [(0, 0, 0.0, 0.0, 0.0, 0.0); MAX_ACTIVE_SENSOR_PORES_PER_TICK];
        let mut active_budget = 0usize;
        let mut surround_sum = 0.0_f32;
        let mut passive_sum = 0.0_f32;

        for sp in &mut self.sensor_pores {
            if !sp.alive {
                continue;
            }
            sp.age_ticks = sp.age_ticks.saturating_add(1);
            let c = sp.channel as usize;
            let raw_x = if c < n_amb {
                sanitize_clamp(ambient_in[c], -4.0, 4.0)
            } else {
                0.0
            };

            if mode == ControlMode::Natural
                && c < capped_amb
                && !in_current_cohort(c, self.scout_cohort_idx as usize)
            {
                update_channel_stat(&mut self.scout_stats[c], raw_x, motor_dev, motor_std);
            }

            let head = (sp.hist_head as usize) & 15;
            sp.history[head] = raw_x;
            sp.hist_head = ((head + 1) & 15) as u8;

            let d_idx = (head + 16 - ((sp.delay.saturating_sub(1) as usize) & 15)) & 15;
            let delayed_x = sp.history[d_idx];
            let dsig = sanitize_clamp(delayed_x - sp.last_val, -4.0, 4.0);
            sp.last_val = delayed_x;

            if !sp.adapt_init {
                sp.adapt_init = true;
                sp.adapt_mean = delayed_x;
            } else {
                sp.adapt_mean =
                    sanitize_clamp(0.985 * sp.adapt_mean + 0.015 * delayed_x, -4.0, 4.0);
            }
            let centered = sanitize_clamp(delayed_x - sp.adapt_mean, -4.0, 4.0);

            if active_budget < MAX_ACTIVE_SENSOR_PORES_PER_TICK {
                active_signals[active_budget] = (
                    sp.target_cell,
                    sp.delta_target_cell,
                    sp.conductance,
                    delayed_x,
                    centered,
                    dsig,
                );
                surround_sum += centered;
                active_budget += 1;
            }

            if mode == ControlMode::Natural {
                let (q_pass, is_sparse_imp) = if c < capped_amb {
                    let st = &self.scout_stats[c];
                    (st.passive_quality(), st.is_sparse_impulse())
                } else {
                    (0.0, false)
                };
                passive_sum += q_pass;

                let trophic_in = if is_sparse_imp && raw_x.abs() >= 0.015 {
                    2.25 * q_pass + 0.55 * raw_x.abs()
                } else {
                    1.55 * q_pass
                };
                let tr_decay = if is_sparse_imp && raw_x.abs() < 0.015 {
                    0.996
                } else {
                    0.95
                };
                sp.trophic_ema = sanitize_clamp(
                    tr_decay * sp.trophic_ema + (1.0 - tr_decay) * trophic_in,
                    0.0,
                    3.0,
                );

                if trophic_in > 0.02 || sp.trophic_ema > 0.03 {
                    let target_g = sanitize_clamp(0.18 + 1.8 * sp.trophic_ema, 0.05, 1.6);
                    sp.conductance =
                        sanitize_clamp(0.94 * sp.conductance + 0.06 * target_g, 0.0, 1.8);
                } else {
                    let g_decay = if is_sparse_imp { 0.993 } else { 0.92 };
                    sp.conductance = sanitize(sp.conductance * g_decay);
                }

                let cell_trophic = sanitize(sp.conductance * q_pass);
                if let Some(ct) = sensory_trophic_out.get_mut(sp.target_cell as usize) {
                    *ct = sanitize(*ct + cell_trophic);
                }
                if let Some(dct) = sensory_trophic_out.get_mut(sp.delta_target_cell as usize) {
                    *dct = sanitize(*dct + 0.7 * cell_trophic);
                }
            }
        }

        // Second pass: horizontal-cell ON/OFF lateral center-surround contrast + sustained receptor luminance.
        let surround_mean = if active_budget > 0 {
            surround_sum / (active_budget as f32)
        } else {
            0.0
        };
        for &(tgt, dtgt, g, delayed_x, centered, dsig) in &active_signals[..active_budget] {
            let cs_contrast = sanitize_clamp(centered - 0.22 * surround_mean, -4.0, 4.0);
            let sig = sanitize_clamp(0.38 * delayed_x + 0.82 * cs_contrast, -4.0, 4.0);
            if let Some(dst) = cell_drive.get_mut(tgt as usize) {
                *dst = sanitize(*dst + g * sig);
            }
            if let Some(ddst) = cell_drive.get_mut(dtgt as usize) {
                *ddst = sanitize(*ddst + 0.88 * g * dsig);
            }
        }

        if mode == ControlMode::Natural {
            self.passive_health_ema = sanitize_clamp(
                0.96 * self.passive_health_ema + 0.04 * passive_sum,
                0.0,
                4.0,
            );
        }
    }

    pub fn morph_pores(
        &mut self,
        max_pores: usize,
        alive_cell_indices: &[u32],
        rng: &mut Rng64,
    ) -> u64 {
        let mut atrophied = 0u64;
        for (i, sp) in self.sensor_pores.iter_mut().enumerate() {
            if !sp.alive {
                continue;
            }
            if sp.age_ticks > 40 && sp.conductance < 0.024 && sp.trophic_ema < 0.020 {
                sp.alive = false;
                sp.conductance = 0.0;
                self.sensor_free_list.push(i as u32);
                self.alive_sensor_count = self.alive_sensor_count.saturating_sub(1);
                atrophied += 1;
            }
        }

        let sensor_cap = max_pores.saturating_sub(self.alive_motor_count).min(64);
        if self.alive_sensor_count >= sensor_cap || alive_cell_indices.is_empty() {
            return atrophied;
        }

        let n_stats = self.scout_stats.len();
        if n_stats == 0 {
            return atrophied;
        }

        let mut best_candidate: Option<(f32, usize)> = None;
        let start = ((self.scout_cohort_idx as usize) * SCOUT_COHORT_WIDTH).min(n_stats);
        let end = (start + SCOUT_COHORT_WIDTH).min(n_stats);

        for (offset, st) in self.scout_stats[start..end].iter().enumerate() {
            let c = start + offset;
            if self.has_sensor_on_channel(c as u32) {
                continue;
            }
            let score = st.passive_quality();
            if score > 0.045 {
                match best_candidate {
                    None => best_candidate = Some((score, c)),
                    Some((b, _)) if score > b => best_candidate = Some((score, c)),
                    _ => {}
                }
            }
        }

        if best_candidate.is_none() {
            for _ in 0..SCOUT_COHORT_WIDTH {
                let c = rng.next_index(n_stats);
                if self.has_sensor_on_channel(c as u32) {
                    continue;
                }
                let score = self.scout_stats[c].passive_quality();
                if score > 0.04 {
                    match best_candidate {
                        None => best_candidate = Some((score, c)),
                        Some((b, _)) if score > b => best_candidate = Some((score, c)),
                        _ => {}
                    }
                }
            }
        }

        if let Some((score, ch)) = best_candidate {
            let n_alive = alive_cell_indices.len();
            let half = (n_alive / 2).max(1);
            let mut load = vec![0u16; n_alive];
            for sp in &self.sensor_pores {
                if !sp.alive {
                    continue;
                }
                for (pos, &cid) in alive_cell_indices.iter().enumerate() {
                    if cid == sp.target_cell || cid == sp.delta_target_cell {
                        load[pos] = load[pos].saturating_add(1);
                    }
                }
            }
            let salt_tgt = rng.next_u32();
            let tgt_pos = (0..half)
                .min_by_key(|&p| (load[p], (p as u32).wrapping_mul(0x9E37_79B9) ^ salt_tgt))
                .unwrap_or(0);
            let tgt = alive_cell_indices[tgt_pos];
            let dtgt_pos = if n_alive > half {
                let salt_dtgt = rng.next_u32();
                (half..n_alive)
                    .min_by_key(|&p| (load[p], (p as u32).wrapping_mul(0x85EB_CA6B) ^ salt_dtgt))
                    .unwrap_or(half)
            } else {
                rng.next_index(n_alive)
            };
            let dtgt = alive_cell_indices[dtgt_pos];
            let mut pore = SensorPore::newborn(
                ch as u32,
                1,
                tgt,
                dtgt,
                sanitize_clamp(0.22 + 0.5 * score, 0.15, 0.85),
            );
            pore.trophic_ema = sanitize_clamp(0.12 + 0.4 * score, 0.05, 1.0);
            if let Some(slot) = self.sensor_free_list.pop() {
                self.sensor_pores[slot as usize] = pore;
            } else {
                self.sensor_pores.push(pore);
            }
            self.alive_sensor_count += 1;
        }

        atrophied
    }

    #[must_use]
    pub fn has_sensor_on_channel(&self, channel: u32) -> bool {
        self.sensor_pores
            .iter()
            .any(|sp| sp.alive && sp.channel == channel)
    }

    pub fn zero_learned_conductances(&mut self) {
        for sp in &mut self.sensor_pores {
            sp.conductance = 0.0;
            sp.trophic_ema = 0.0;
        }
        for mp in &mut self.motor_pores {
            mp.conductance = 0.0;
            mp.klinokinesis_gain = 0.0;
            mp.motor_mastery = 0.0;
        }
        self.scout_stats.fill(ChannelScoutStat::default());
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_usize(self.sensor_pores.len());
        hasher.write_usize(self.alive_sensor_count);
        for sp in &self.sensor_pores {
            hasher.write_bool(sp.alive);
            if sp.alive {
                hasher.write_u32(sp.channel);
                hasher.write_u8(sp.delay);
                hasher.write_u32(sp.target_cell);
                hasher.write_u32(sp.delta_target_cell);
                hasher.write_f32(sp.conductance);
                hasher.write_f32(sp.trophic_ema);
            }
        }
        hasher.write_usize(self.motor_pores.len());
        hasher.write_usize(self.alive_motor_count);
        for mp in &self.motor_pores {
            hasher.write_bool(mp.alive);
            if mp.alive {
                hasher.write_u32(mp.actuator_channel);
                hasher.write_f32(mp.conductance);
                hasher.write_f32(mp.klinokinesis_gain);
                hasher.write_f32(mp.motor_mastery);
            }
        }
        hasher.write_usize(self.scout_stats.len());
        for st in &self.scout_stats {
            hasher.write_bool(st.initialized);
            if st.initialized {
                hasher.write_f32(st.ar1_w);
            }
        }
    }
}

#[inline]
fn in_current_cohort(channel: usize, cohort_idx: usize) -> bool {
    let start = cohort_idx * SCOUT_COHORT_WIDTH;
    channel >= start && channel < start + SCOUT_COHORT_WIDTH
}

fn update_channel_stat(st: &mut ChannelScoutStat, x: f32, motor_dev: f32, motor_std: f32) {
    if !st.initialized {
        st.initialized = true;
        st.mean = x;
        st.prev_val = x;
        st.prev_dx = 0.0;
        return;
    }

    let step_diff = sanitize(x - st.prev_val);
    st.cov_dx_ema = sanitize_clamp(
        0.93 * st.cov_dx_ema + 0.07 * step_diff * st.prev_dx,
        -16.0,
        16.0,
    );
    st.diff_sq_ema = sanitize_clamp(
        0.93 * st.diff_sq_ema + 0.07 * step_diff * step_diff,
        0.0,
        16.0,
    );
    st.prev_dx = step_diff;

    let dx_prev = sanitize(st.prev_val - st.mean);
    let dx = sanitize(x - st.mean);
    let pred_dx = sanitize(st.ar1_w * dx_prev);
    let err = sanitize(dx - pred_dx);
    let err2 = sanitize(err * err);
    let dx2 = sanitize(dx * dx);
    let dx4 = sanitize_clamp(dx2 * dx2, 0.0, 256.0);

    st.ar1_err_ema = sanitize_clamp(0.93 * st.ar1_err_ema + 0.07 * err2, 0.0, 16.0);
    st.var_ema = sanitize_clamp(0.93 * st.var_ema + 0.07 * dx2, 0.0, 16.0);
    st.fourth_ema = sanitize_clamp(0.95 * st.fourth_ema + 0.05 * dx4, 0.0, 256.0);

    let x_std = st.var_ema.max(1.0e-5).sqrt();
    let norm_excursion = sanitize_clamp((dx.abs() - 0.7 * x_std) / x_std, -3.0, 3.0);
    let norm_motor = sanitize_clamp(motor_dev / motor_std.max(1.0e-3), -3.0, 3.0);
    st.motor_cov_ema = sanitize_clamp(
        0.96 * st.motor_cov_ema + 0.04 * norm_excursion * norm_motor,
        -1.5,
        1.5,
    );

    let denom = 0.02 + dx_prev * dx_prev;
    st.ar1_w = sanitize_clamp(st.ar1_w + 0.06 * err * dx_prev / denom, -1.2, 1.2);
    st.mean = sanitize_clamp(0.985 * st.mean + 0.015 * x, -4.0, 4.0);
    st.prev_val = x;
}
