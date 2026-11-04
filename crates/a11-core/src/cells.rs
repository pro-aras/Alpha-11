//! Continuous-tau oscillatory cell arena with relational step-difference coding and sparse k-WTA.

use std::cmp::Ordering;
use std::f32::consts::TAU;

use a11_medium::ControlMode;

use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp, smooth_saturate};

#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub alive: bool,
    pub v_real: f32,
    pub v_prev: f32,
    pub delta_v: f32,
    pub v_pred: f32,
    pub pred_err: f32,
    pub phi: f32,
    pub tau: f32,
    pub firing_rate_ema: f32,
    pub excitability_bias: f32,
    pub v_homeo: f32,
    pub v_delta_homeo: f32,
    pub atp_credit_ema: f32,
    pub surprise_ema: f32,
    pub sensory_trophic_ema: f32,
    pub mitosis_cooldown: u16,
    pub age_ticks: u32,
}

impl Cell {
    pub const LOGICAL_BYTES: usize = 1 + 4 * 14 + 2 + 4;

    #[must_use]
    pub fn newborn(tau: f32, phi: f32) -> Self {
        Self {
            alive: true,
            v_real: 0.0,
            v_prev: 0.0,
            delta_v: 0.0,
            v_pred: 0.0,
            pred_err: 0.0,
            phi: sanitize_clamp(phi, 0.0, TAU - 1.0e-5),
            tau: sanitize_clamp(tau, 1.0, 2048.0),
            firing_rate_ema: 0.06,
            excitability_bias: 0.0,
            v_homeo: 0.0,
            v_delta_homeo: 0.0,
            atp_credit_ema: 0.02,
            surprise_ema: 0.0,
            sensory_trophic_ema: 0.0,
            mitosis_cooldown: 40,
            age_ticks: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellArena {
    pub slots: Vec<Cell>,
    pub free_list: Vec<u32>,
    pub alive_count: usize,
    pub active_indices: Vec<u32>,
    pub min_cells_floor: usize,
    pub theta_phase: f32,
    pub error_reversal_gate: f32,
}

impl CellArena {
    #[must_use]
    pub fn new(initial_cells: usize, max_cells: usize, rng: &mut Rng64) -> Self {
        let count = initial_cells.max(4).min(max_cells.max(4));
        let mut slots = Vec::with_capacity(count.saturating_mul(2).min(max_cells.max(4)));
        let denom = (count.saturating_sub(1)).max(1) as f32;
        for i in 0..count {
            let frac = (i as f32) / denom;
            let base_tau = 1.5 * (8.0_f32 / 1.5_f32).powf(frac);
            let jitter = 1.0 + 0.06 * rng.next_f32_signed();
            let tau = sanitize_clamp(base_tau * jitter, 1.5, 8.0);
            let phi = ((i as f32) * 0.618_034 * TAU + 0.25 * rng.next_f32_unit()) % TAU;
            slots.push(Cell::newborn(tau, phi));
        }
        let init_theta = sanitize_clamp((slots[0].phi * 4.0).rem_euclid(TAU), 0.0, TAU - 1.0e-5);
        Self {
            slots,
            free_list: Vec::new(),
            alive_count: count,
            active_indices: Vec::with_capacity(96),
            min_cells_floor: count,
            theta_phase: init_theta,
            error_reversal_gate: 0.0,
        }
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        self.slots.len() * Cell::LOGICAL_BYTES
            + self.free_list.len() * 4
            + self.active_indices.len() * 4
            + 32
    }

    #[must_use]
    pub fn k_winners_target(&self) -> usize {
        let base = (self.alive_count / 2).max(self.alive_count.min(24));
        base.clamp(4, 96).min(self.alive_count.max(1))
    }

    pub fn integrate_step(
        &mut self,
        ext_drive: &[f32],
        pred_drive: &[f32],
        sensory_trophic_in: &[f32],
        ne_network_reset: f32,
        mode: ControlMode,
    ) {
        let ne_reset = sanitize_clamp(ne_network_reset, 0.0, 0.90);
        self.error_reversal_gate =
            sanitize_clamp((0.80 * self.error_reversal_gate).max(ne_reset), 0.0, 0.90);
        let err_rev = self.error_reversal_gate;

        let n = self.slots.len();
        let mut sum_abs = 0.0_f32;
        let mut alive_seen = 0usize;
        let mut kuramoto_sin = 0.0_f32;
        let mut kuramoto_w = 0.0_f32;

        for i in 0..n {
            if !self.slots[i].alive {
                continue;
            }
            let d = ext_drive.get(i).copied().unwrap_or(0.0);
            sum_abs += sanitize(d).abs();
            alive_seen += 1;
            let st_in = sanitize(sensory_trophic_in.get(i).copied().unwrap_or(0.0)).max(0.0);
            if st_in > 0.01 {
                let w = st_in.min(1.0);
                kuramoto_sin += w * (self.slots[i].phi - self.theta_phase).sin();
                kuramoto_w += w;
            }
        }
        let phase_pull = if kuramoto_w > 1.0e-4 {
            0.06 * (kuramoto_sin / kuramoto_w)
        } else {
            0.0
        };
        self.theta_phase = sanitize_clamp(
            (self.theta_phase + (TAU / 10.0) + phase_pull).rem_euclid(TAU),
            0.0,
            TAU - 1.0e-5,
        );
        let theta_cos = self.theta_phase.cos();
        let beta_falling_flank = (1.40..=2.44).contains(&self.theta_phase);
        let mean_abs = if alive_seen > 0 {
            sum_abs / (alive_seen as f32)
        } else {
            0.0
        };
        let div_norm = 1.0 / (1.0 + 0.75 * mean_abs);

        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(self.alive_count);

        for (i, cell) in self.slots.iter_mut().enumerate() {
            if !cell.alive {
                continue;
            }
            cell.age_ticks = cell.age_ticks.saturating_add(1);
            if cell.mitosis_cooldown > 0 && mode == ControlMode::Natural {
                cell.mitosis_cooldown -= 1;
            }

            let omega = TAU / (4.0 * cell.tau).clamp(3.0, 256.0);
            let theta_lock = 0.025 * (self.theta_phase - cell.phi).sin();
            cell.phi = (cell.phi + omega + theta_lock).rem_euclid(TAU);
            let pacemaking = 0.012 * cell.phi.cos();

            let top_pred = smooth_saturate(pred_drive.get(i).copied().unwrap_or(0.0));
            let s_trophic = sanitize(sensory_trophic_in.get(i).copied().unwrap_or(0.0)).max(0.0);
            let bottom_up_gain = if cell.tau >= 4.0 {
                1.0 + 0.30 * err_rev
            } else {
                1.0 + 0.12 * err_rev
            };
            let top_down_beta = if cell.tau >= 6.0 {
                0.055 * (1.0 - err_rev) * theta_cos.max(0.0) * top_pred
            } else {
                0.0
            };

            let gain = sanitize_clamp(1.0 + cell.excitability_bias, 0.45, 2.20);
            let raw_in = sanitize(ext_drive.get(i).copied().unwrap_or(0.0))
                * div_norm
                * gain
                * bottom_up_gain
                + top_down_beta
                + pacemaking;
            let eff_tau = if ne_reset > 0.01 && cell.tau > 6.0 {
                6.0 + (cell.tau - 6.0) * (1.0 - ne_reset)
            } else {
                cell.tau
            };
            let alpha = sanitize_clamp((-1.0 / eff_tau.max(1.0)).exp(), 0.0, 0.9995);

            cell.v_prev = cell.v_real;
            let integrated = alpha * cell.v_real + (1.0 - alpha) * (2.8 * raw_in);
            cell.v_real = smooth_saturate(integrated);
            cell.delta_v = sanitize_clamp(cell.v_real - cell.v_prev, -2.0, 2.0);

            cell.v_pred = top_pred;
            cell.pred_err = sanitize_clamp(cell.v_real - cell.v_pred, -2.0, 2.0);

            let err_salience = if err_rev > 0.05 && s_trophic > 0.008 {
                0.22 * err_rev * cell.pred_err.abs()
            } else {
                0.0
            };
            let salience = sanitize(cell.v_real.abs() + 0.55 * cell.delta_v.abs() + err_salience);
            if salience > 1.0e-4 {
                candidates.push((salience, i as u32));
            }

            if mode == ControlMode::Natural {
                cell.sensory_trophic_ema =
                    sanitize_clamp(0.95 * cell.sensory_trophic_ema + 0.05 * s_trophic, 0.0, 5.0);
                let surprise_sample = cell.pred_err.abs();
                cell.surprise_ema =
                    sanitize_clamp(0.94 * cell.surprise_ema + 0.06 * surprise_sample, 0.0, 2.0);
                if s_trophic > 0.003 {
                    cell.atp_credit_ema = sanitize_clamp(
                        0.96 * cell.atp_credit_ema + 0.04 * (0.85 * s_trophic),
                        0.0,
                        4.0,
                    );
                } else {
                    cell.atp_credit_ema = sanitize(0.994 * cell.atp_credit_ema);
                }
            }
        }

        let base_k = self.k_winners_target();
        let k = if beta_falling_flank && err_rev < 0.25 {
            (((base_k as f32) * 0.86).round() as usize)
                .max(4)
                .min(self.alive_count.max(1))
        } else {
            base_k
        };
        self.active_indices.clear();
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k - 1, |a, b| compare_candidate(b, a));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| compare_candidate(b, a));
        for &(_, idx) in &candidates {
            self.active_indices.push(idx);
        }

        if mode == ControlMode::Natural {
            let mut is_winner = vec![false; n];
            for &idx in &self.active_indices {
                if let Some(flag) = is_winner.get_mut(idx as usize) {
                    *flag = true;
                }
            }
            let target_rate = (k as f32) / (self.alive_count.max(1) as f32);
            for (i, cell) in self.slots.iter_mut().enumerate() {
                if !cell.alive {
                    continue;
                }
                let fired = if is_winner[i] { 1.0 } else { 0.0 };
                cell.firing_rate_ema =
                    sanitize_clamp(0.97 * cell.firing_rate_ema + 0.03 * fired, 0.0, 1.0);
                let rate_err = target_rate - cell.firing_rate_ema;
                cell.excitability_bias =
                    sanitize_clamp(cell.excitability_bias + 0.004 * rate_err, -0.55, 1.20);
            }
        }
    }

    #[must_use]
    pub fn somatic_value_estimate(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut weight_norm = 1.0_f32;
        for cell in &self.slots {
            if !cell.alive {
                continue;
            }
            if cell.v_homeo.abs() > 1.0e-5 || cell.v_delta_homeo.abs() > 1.0e-5 {
                sum += cell.v_homeo * cell.v_real + 0.5 * cell.v_delta_homeo * cell.delta_v;
                weight_norm += cell.v_homeo.abs() + 0.5 * cell.v_delta_homeo.abs();
            }
        }
        sanitize_clamp(sum / weight_norm.sqrt(), -4.0, 4.0)
    }

    pub fn find_mitosis_candidate(&self, max_cells: usize) -> Option<u32> {
        if self.alive_count >= max_cells {
            return None;
        }
        let mut best: Option<(f32, u32)> = None;
        for (i, cell) in self.slots.iter().enumerate() {
            if !cell.alive || cell.mitosis_cooldown > 0 || cell.tau >= 900.0 {
                continue;
            }
            let timescale_gain = (cell.tau / 3.0).clamp(1.0, 5.0).sqrt();
            let intrinsic_surprise = sanitize(cell.surprise_ema * timescale_gain);
            if cell.atp_credit_ema > 0.030 && intrinsic_surprise > 0.046 {
                let score = sanitize(cell.atp_credit_ema * intrinsic_surprise);
                match best {
                    None => best = Some((score, i as u32)),
                    Some((b_score, _)) if score > b_score => best = Some((score, i as u32)),
                    _ => {}
                }
            }
        }
        best.map(|(_, idx)| idx)
    }

    pub fn spawn_mitosis_parent(
        &mut self,
        child_idx: u32,
        max_cells: usize,
        rng: &mut Rng64,
    ) -> Option<u32> {
        if self.alive_count >= max_cells {
            return None;
        }
        let c_idx = child_idx as usize;
        if c_idx >= self.slots.len() || !self.slots[c_idx].alive {
            return None;
        }
        let child_tau = self.slots[c_idx].tau;
        let child_homeo = self.slots[c_idx].v_homeo;
        let parent_tau = sanitize_clamp(child_tau * 2.5, 2.5, 1024.0);
        let parent_phi = rng.next_f32_unit() * TAU;

        self.slots[c_idx].mitosis_cooldown = 140;
        self.slots[c_idx].surprise_ema *= 0.65;

        let mut parent = Cell::newborn(parent_tau, parent_phi);
        parent.mitosis_cooldown = 160;
        parent.v_homeo = sanitize_clamp(0.5 * child_homeo, -2.0, 2.0);
        parent.atp_credit_ema = 0.04;

        let new_idx = if let Some(slot_idx) = self.free_list.pop() {
            self.slots[slot_idx as usize] = parent;
            slot_idx
        } else {
            if self.slots.len() >= max_cells {
                return None;
            }
            let idx = self.slots.len() as u32;
            self.slots.push(parent);
            idx
        };
        self.alive_count += 1;
        Some(new_idx)
    }

    #[must_use]
    pub fn find_slow_context_cell(&self) -> Option<u32> {
        let mut best: Option<(f32, u32)> = None;
        for (i, cell) in self.slots.iter().enumerate() {
            if !cell.alive {
                continue;
            }
            let score = cell.tau * (0.2 + cell.v_real.abs() + cell.surprise_ema);
            match best {
                None => best = Some((score, i as u32)),
                Some((b_score, _)) if score > b_score => best = Some((score, i as u32)),
                _ => {}
            }
        }
        best.map(|(_, idx)| idx)
    }

    pub fn prune_starved_cells(&mut self) -> Vec<u32> {
        let mut pruned = Vec::new();
        if self.alive_count <= self.min_cells_floor {
            return pruned;
        }
        for (i, cell) in self.slots.iter_mut().enumerate() {
            if self.alive_count <= self.min_cells_floor {
                break;
            }
            if i < self.min_cells_floor || !cell.alive {
                continue;
            }
            if cell.age_ticks > 450 && cell.atp_credit_ema < 0.002 && cell.firing_rate_ema < 0.002 {
                cell.alive = false;
                cell.v_real = 0.0;
                cell.v_prev = 0.0;
                cell.delta_v = 0.0;
                self.free_list.push(i as u32);
                self.alive_count -= 1;
                pruned.push(i as u32);
            }
        }
        pruned
    }

    #[must_use]
    pub fn tau_stats(&self) -> (f32, f32, usize) {
        if self.alive_count == 0 {
            return (0.0, 0.0, 0);
        }
        let mut sum_tau = 0.0_f32;
        let mut max_tau = 0.0_f32;
        let mut octave_mask = 0u32;
        let mut seen = 0usize;
        for cell in &self.slots {
            if !cell.alive {
                continue;
            }
            sum_tau += cell.tau;
            if cell.tau > max_tau {
                max_tau = cell.tau;
            }
            let oct = (cell.tau.max(1.0).log2().floor() as u32).min(15);
            octave_mask |= 1u32 << oct;
            seen += 1;
        }
        let mean_tau = if seen > 0 {
            sum_tau / (seen as f32)
        } else {
            0.0
        };
        (
            sanitize(mean_tau),
            sanitize(max_tau),
            octave_mask.count_ones() as usize,
        )
    }

    #[must_use]
    pub fn mean_prediction_error(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut seen = 0usize;
        for cell in &self.slots {
            if !cell.alive {
                continue;
            }
            sum += cell.pred_err.abs();
            seen += 1;
        }
        if seen > 0 {
            sanitize(sum / (seen as f32))
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn mean_surprise(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut seen = 0usize;
        for cell in &self.slots {
            if !cell.alive {
                continue;
            }
            sum += cell.surprise_ema;
            seen += 1;
        }
        if seen > 0 {
            sanitize(sum / (seen as f32))
        } else {
            0.0
        }
    }

    pub fn zero_learned_conductances(&mut self) {
        self.theta_phase = 0.0;
        self.error_reversal_gate = 0.0;
        for cell in &mut self.slots {
            cell.excitability_bias = 0.0;
            cell.v_homeo = 0.0;
            cell.v_delta_homeo = 0.0;
            cell.atp_credit_ema = 0.0;
        }
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_usize(self.slots.len());
        hasher.write_usize(self.alive_count);
        for cell in &self.slots {
            hasher.write_bool(cell.alive);
            if cell.alive {
                hasher.write_f32(cell.tau);
                hasher.write_f32(cell.excitability_bias);
                hasher.write_f32(cell.v_homeo);
                hasher.write_f32(cell.v_delta_homeo);
                hasher.write_f32(cell.atp_credit_ema);
                hasher.write_f32(cell.surprise_ema);
            }
        }
    }
}

#[inline]
fn compare_candidate(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    match a.0.total_cmp(&b.0) {
        Ordering::Equal => b.1.cmp(&a.1),
        other => other,
    }
}
