//! Directed Benna-Fusi cascade tubes, 16-tick efference ring, agency, and rollout clutch.

use a11_medium::{ControlMode, FieldPulse};

use crate::body::BodyCore;
use crate::cells::{Cell, CellArena};
use crate::pores::PoreMembrane;
use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp, smooth_saturate};

pub const RING_LEN: usize = 16;
pub const TRACKED_ACTUATORS: usize = 32;
pub const PRED_BUCKETS: usize = 64;
pub const MAX_OUTGOING_PER_CELL: usize = 72;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TubeKind {
    CellToCell = 0,
    CellToMotor = 1,
}

impl TubeKind {
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CellToCell),
            1 => Some(Self::CellToMotor),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tube {
    pub alive: bool,
    pub kind: TubeKind,
    pub pre: u32,
    pub post: u32,
    pub delay: u8,
    pub u: [f32; 4],
    pub w_pred: f32,
    pub context_gate_cell: Option<u32>,
    pub gate_invert: bool,
    pub contradiction_accum: f32,
    pub chain_score: f32,
    pub last_upstream_cell: u32,
    pub usage_ema: f32,
    pub age_ticks: u32,
}

impl Tube {
    pub const LOGICAL_BYTES: usize = 1 + 1 + 4 + 4 + 1 + 4 * 4 + 4 + 5 + 1 + 4 * 3 + 4 + 4;

    #[must_use]
    pub fn newborn(kind: TubeKind, pre: u32, post: u32, delay: u8, init_u0: f32) -> Self {
        let u0 = sanitize_clamp(init_u0, -2.5, 2.5);
        Self {
            alive: true,
            kind,
            pre,
            post,
            delay: delay.clamp(1, 16),
            u: [u0, 0.25 * u0, 0.0, 0.0],
            w_pred: 0.0,
            context_gate_cell: None,
            gate_invert: false,
            contradiction_accum: 0.0,
            chain_score: 0.0,
            last_upstream_cell: u32::MAX,
            usage_ema: 0.02,
            age_ticks: 0,
        }
    }

    #[inline]
    #[must_use]
    pub fn effective_g(&self, cells: &[Cell]) -> f32 {
        let base = self.u[0];
        let gate = match self.context_gate_cell {
            None => 1.0,
            Some(g_idx) => {
                let gv = cells
                    .get(g_idx as usize)
                    .filter(|c| c.alive)
                    .map(|c| (c.v_real.abs() + 0.85 * c.surprise_ema).clamp(0.0, 1.0))
                    .unwrap_or(0.0);
                if self.gate_invert {
                    (1.0 - 1.45 * gv).clamp(0.0, 1.0)
                } else {
                    (0.25 + 1.15 * gv).clamp(0.0, 1.0)
                }
            }
        };
        sanitize(base * gate)
    }

    #[inline]
    pub fn apply_benna_fusi_impulse(&mut self, du: f32, consolidation_boost: f32) {
        self.u[0] = sanitize_clamp(self.u[0] + du, -2.5, 2.5);
        let boost = sanitize_clamp(consolidation_boost, 1.0, 6.0);
        let g01 = (0.045 * boost).min(0.35);
        let g12 = (0.009 * boost).min(0.12);
        let g23 = (0.002 * boost).min(0.04);

        let f01 = g01 * (self.u[0] - self.u[1]);
        let f12 = g12 * (self.u[1] - self.u[2]);
        let f23 = g23 * (self.u[2] - self.u[3]);

        self.u[0] = sanitize_clamp(self.u[0] - 0.5 * f01, -2.5, 2.5);
        self.u[1] = sanitize_clamp(self.u[1] + 0.65 * f01 - 0.5 * f12, -2.5, 2.5);
        self.u[2] = sanitize_clamp(self.u[2] + 0.65 * f12 - 0.5 * f23, -2.5, 2.5);
        self.u[3] = sanitize_clamp(self.u[3] + 0.65 * f23, -2.5, 2.5);
    }

    #[inline]
    #[must_use]
    pub fn stiff_strength(&self) -> f32 {
        sanitize(self.u[2].abs() + self.u[3].abs())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EfferenceEntry {
    pub valid: bool,
    pub tick: u64,
    pub committed: bool,
    pub motor_norm: f32,
    pub motor_out: [f32; TRACKED_ACTUATORS],
    pub active_cells: Vec<(u32, f32, f32)>,
    pub pred_with_action: Vec<(u32, f32)>,
    pub pred_null_action: Vec<(u32, f32)>,
    pub somatic_value_est: f32,
    pub raw_valence_at_commit: f32,
}

impl Default for EfferenceEntry {
    fn default() -> Self {
        Self {
            valid: false,
            tick: 0,
            committed: false,
            motor_norm: 0.0,
            motor_out: [0.0; TRACKED_ACTUATORS],
            active_cells: Vec::new(),
            pred_with_action: Vec::new(),
            pred_null_action: Vec::new(),
            somatic_value_est: 0.0,
            raw_valence_at_commit: 0.0,
        }
    }
}

impl EfferenceEntry {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        1 + 8 + 1 + 4 + 4 * TRACKED_ACTUATORS
            + self.active_cells.len() * 12
            + self.pred_with_action.len() * 8
            + self.pred_null_action.len() * 8
            + 8
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EfferenceRing {
    pub entries: [EfferenceEntry; RING_LEN],
    pub head: usize,
}

impl Default for EfferenceRing {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| EfferenceEntry::default()),
            head: 0,
        }
    }
}

impl EfferenceRing {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let sum: usize = self.entries.iter().map(EfferenceEntry::logical_bytes).sum();
        sum + 8
    }

    pub fn push(&mut self, entry: EfferenceEntry) {
        let idx = self.head % RING_LEN;
        self.entries[idx] = entry;
        self.head = (idx + 1) % RING_LEN;
    }

    #[must_use]
    pub fn at_lag(&self, lag: usize) -> Option<&EfferenceEntry> {
        if !(1..=RING_LEN).contains(&lag) {
            return None;
        }
        let idx = (self.head + RING_LEN - lag) % RING_LEN;
        let e = &self.entries[idx];
        if e.valid {
            Some(e)
        } else {
            None
        }
    }
}

pub struct RolloutOutcome {
    pub pulse: FieldPulse,
    pub committed: bool,
    pub rehearsals: u64,
    pub pred_with: Vec<(u32, f32)>,
    pub pred_null: Vec<(u32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TubeGraph {
    pub slots: Vec<Tube>,
    pub free_list: Vec<u32>,
    pub alive_count: usize,
    pub outgoing_by_cell: Vec<Vec<u32>>,
    pub action_sensory_w: [[f32; PRED_BUCKETS]; TRACKED_ACTUATORS],
    pub sensory_self_w: [f32; PRED_BUCKETS],
    pub act_mean: [f32; TRACKED_ACTUATORS],
    pub mean_interventional_agency: f32,
}

impl TubeGraph {
    #[must_use]
    pub fn new(
        initial_cells: usize,
        initial_motors: usize,
        max_tubes: usize,
        rng: &mut Rng64,
    ) -> Self {
        let n_cells = initial_cells.max(4);
        let n_motors = initial_motors.max(1);
        let mut graph = Self {
            slots: Vec::with_capacity((n_cells * (n_motors + 3)).min(max_tubes.max(16))),
            free_list: Vec::new(),
            alive_count: 0,
            outgoing_by_cell: vec![Vec::new(); n_cells],
            action_sensory_w: [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS],
            sensory_self_w: [0.0; PRED_BUCKETS],
            act_mean: [0.0; TRACKED_ACTUATORS],
            mean_interventional_agency: 0.0,
        };

        for c in 0..n_cells {
            for hop in 0..2 {
                let dst = (c + 1 + hop * 5 + rng.next_index(n_cells.saturating_sub(1).max(1))) % n_cells;
                if dst != c && graph.alive_count < max_tubes {
                    let delay = 1 + (rng.next_u32() % 3) as u8;
                    let init_w = 0.03 * rng.next_f32_signed();
                    let _ = graph.add_tube(
                        Tube::newborn(TubeKind::CellToCell, c as u32, dst as u32, delay, init_w),
                        max_tubes,
                    );
                }
            }
            for m in 0..n_motors.min(TRACKED_ACTUATORS) {
                if graph.alive_count < max_tubes {
                    let init_w = 0.012 * rng.next_f32_signed();
                    let _ = graph.add_tube(
                        Tube::newborn(TubeKind::CellToMotor, c as u32, m as u32, 1, init_w),
                        max_tubes,
                    );
                }
            }
        }

        graph
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        self.slots.len() * Tube::LOGICAL_BYTES
            + self.free_list.len() * 4
            + 4 * PRED_BUCKETS * TRACKED_ACTUATORS
            + 4 * PRED_BUCKETS
            + 4 * TRACKED_ACTUATORS
            + 32
    }

    pub fn ensure_cell_capacity(&mut self, cell_slots_len: usize) {
        if self.outgoing_by_cell.len() < cell_slots_len {
            self.outgoing_by_cell.resize_with(cell_slots_len, Vec::new);
        }
    }

    pub fn rebuild_adjacency(&mut self, cell_slots_len: usize) {
        self.outgoing_by_cell.clear();
        self.outgoing_by_cell.resize_with(cell_slots_len, Vec::new);
        let mut alive = 0usize;
        for (idx, t) in self.slots.iter().enumerate() {
            if !t.alive {
                continue;
            }
            alive += 1;
            if let Some(list) = self.outgoing_by_cell.get_mut(t.pre as usize)
                && list.len() < MAX_OUTGOING_PER_CELL
            {
                list.push(idx as u32);
            }
        }
        self.alive_count = alive;
    }

    pub fn add_tube(&mut self, tube: Tube, max_tubes: usize) -> Option<u32> {
        if self.alive_count >= max_tubes {
            return None;
        }
        let pre = tube.pre as usize;
        self.ensure_cell_capacity(pre + 1);
        if self.outgoing_by_cell[pre].len() >= MAX_OUTGOING_PER_CELL {
            return None;
        }
        let idx = if let Some(slot) = self.free_list.pop() {
            self.slots[slot as usize] = tube;
            slot
        } else {
            if self.slots.len() >= max_tubes {
                return None;
            }
            let s = self.slots.len() as u32;
            self.slots.push(tube);
            s
        };
        self.outgoing_by_cell[pre].push(idx);
        self.alive_count += 1;
        Some(idx)
    }

    #[must_use]
    pub fn has_direct_tube(&self, kind: TubeKind, pre: u32, post: u32) -> bool {
        let Some(list) = self.outgoing_by_cell.get(pre as usize) else {
            return false;
        };
        for &t_idx in list {
            if let Some(t) = self.slots.get(t_idx as usize)
                && t.alive
                && t.kind == kind
                && t.post == post
            {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn has_cell_tube_at_delay(&self, pre: u32, post: u32, delay: u8) -> bool {
        let Some(list) = self.outgoing_by_cell.get(pre as usize) else {
            return false;
        };
        for &t_idx in list {
            if let Some(t) = self.slots.get(t_idx as usize)
                && t.alive
                && t.kind == TubeKind::CellToCell
                && t.post == post
                && t.delay == delay
            {
                return true;
            }
        }
        false
    }

    pub fn accumulate_recurrent_and_predictive(
        &self,
        ring: &EfferenceRing,
        cells: &[Cell],
        ext_drive: &mut [f32],
        pred_drive: &mut [f32],
        sensory_trophic: &mut [f32],
        mode: ControlMode,
    ) {
        for lag in 1..=RING_LEN {
            let Some(entry) = ring.at_lag(lag) else {
                continue;
            };
            let lag_u8 = lag as u8;
            for &(c_idx, v_past, dv_past) in &entry.active_cells {
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                    continue;
                };
                let src_trophic = cells
                    .get(c_idx as usize)
                    .filter(|c| c.alive)
                    .map_or(0.0, |c| c.sensory_trophic_ema);
                let signal = sanitize(v_past + 0.45 * dv_past);
                for &t_idx in out_list {
                    let Some(tube) = self.slots.get(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive || tube.kind != TubeKind::CellToCell || tube.delay != lag_u8 {
                        continue;
                    }
                    let dst = tube.post as usize;
                    let g = tube.effective_g(cells);
                    if let Some(d_ref) = ext_drive.get_mut(dst) {
                        *d_ref = sanitize(*d_ref + g * signal);
                    }
                    if let Some(p_ref) = pred_drive.get_mut(dst) {
                        *p_ref = sanitize(*p_ref + tube.w_pred * signal);
                    }
                    if mode == ControlMode::Natural && src_trophic > 0.008 {
                        let syn_strength = (g.abs() + 0.5 * tube.w_pred.abs()).min(1.0);
                        if syn_strength > 0.05
                            && let Some(t_ref) = sensory_trophic.get_mut(dst)
                        {
                            *t_ref = sanitize(
                                *t_ref + 0.65 * syn_strength * src_trophic * (0.25 + signal.abs()),
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn evaluate_interventional_agency_and_update_predictors(
        &mut self,
        ring: &EfferenceRing,
        cells: &CellArena,
        membrane: &mut PoreMembrane,
        mode: ControlMode,
    ) {
        let Some(prev) = ring.at_lag(1) else {
            return;
        };

        if !prev.committed {
            self.mean_interventional_agency = sanitize_clamp(
                0.998 * self.mean_interventional_agency,
                0.0,
                1.0,
            );
        } else {
            let mut err_act_sq = 0.0_f32;
            let mut err_null_sq = 0.0_f32;
            let mut evaluated = 0usize;

            for (k, &(c_idx, p_act)) in prev.pred_with_action.iter().enumerate() {
                let p_null = prev
                    .pred_null_action
                    .get(k)
                    .map(|&(_, v)| v)
                    .unwrap_or(0.0);
                let Some(cell) = cells.slots.get(c_idx as usize) else {
                    continue;
                };
                if !cell.alive || cell.sensory_trophic_ema < 0.008 {
                    continue;
                }
                let actual = cell.v_real;
                let da = sanitize(actual - p_act);
                let dn = sanitize(actual - p_null);
                err_act_sq += da * da;
                err_null_sq += dn * dn;
                evaluated += 1;
            }

            let raw_gain = if evaluated > 0 && err_null_sq > 1.0e-5 {
                let improvement = err_null_sq - err_act_sq - 0.008 * err_null_sq;
                sanitize_clamp(improvement / (err_null_sq + 5.0e-4), -0.25, 1.0)
            } else {
                0.0
            };
            self.mean_interventional_agency = sanitize_clamp(
                0.985 * self.mean_interventional_agency + 0.015 * raw_gain,
                0.0,
                1.0,
            );
        }

        if mode != ControlMode::Natural {
            return;
        }

        let mut centered_act = [0.0_f32; TRACKED_ACTUATORS];
        let mut da_norm_sq = 0.0_f32;
        if prev.committed {
            for (m, &a_val) in prev.motor_out.iter().enumerate() {
                let da_m = sanitize_clamp(a_val - self.act_mean[m], -2.0, 2.0);
                centered_act[m] = da_m;
                da_norm_sq += da_m * da_m;
                self.act_mean[m] =
                    sanitize_clamp(0.985 * self.act_mean[m] + 0.015 * a_val, -1.0, 1.0);
            }
        }
        let nlms_scale = 1.0 / (1.0 + 0.18 * da_norm_sq);

        let mut muscle_err_gain = [0.0_f32; TRACKED_ACTUATORS];
        let mut muscle_null_sq = [0.0_f32; TRACKED_ACTUATORS];
        let mut muscle_sum_jac = [0.0_f32; TRACKED_ACTUATORS];
        let mut motor_myelin = [0.0_f32; TRACKED_ACTUATORS];

        for &(c_idx, _, _) in &prev.active_cells {
            let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                continue;
            };
            for &t_idx in out_list {
                let Some(tube) = self.slots.get(t_idx as usize) else {
                    continue;
                };
                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                    continue;
                }
                let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                    continue;
                };
                if mp.alive {
                    let act_ch = mp.actuator_channel as usize;
                    if act_ch < TRACKED_ACTUATORS {
                        motor_myelin[act_ch] =
                            sanitize_clamp(motor_myelin[act_ch] + tube.stiff_strength(), 0.0, 2.0);
                    }
                }
            }
        }

        for &(c_idx, v_past, dv_past) in &prev.active_cells {
            let Some(cell_now) = cells.slots.get(c_idx as usize) else {
                continue;
            };
            if !cell_now.alive || cell_now.sensory_trophic_ema < 0.005 {
                continue;
            }
            let b = (c_idx as usize) % PRED_BUCKETS;
            let actual_dv = sanitize_clamp(cell_now.v_real - v_past, -2.0, 2.0);
            let self_pred = sanitize(self.sensory_self_w[b] * dv_past);
            let mut act_pred = 0.0_f32;
            if prev.committed {
                for (m, &da_m) in centered_act.iter().enumerate() {
                    let w_mb = self.action_sensory_w[m][b];
                    if w_mb.abs() >= 0.018 {
                        act_pred += w_mb * da_m;
                    }
                    if w_mb.abs() > 0.012 {
                        muscle_sum_jac[m] += w_mb.abs();
                    }
                }
            }
            let total_err = sanitize_clamp(actual_dv - (self_pred + act_pred), -2.0, 2.0);
            self.sensory_self_w[b] =
                sanitize_clamp(0.998 * self.sensory_self_w[b] + 0.035 * total_err * dv_past, -1.2, 1.2);
            if prev.committed {
                let clean_err = total_err.clamp(-0.45, 0.45);
                for (m, &da_m) in centered_act.iter().enumerate() {
                    let w_old = self.action_sensory_w[m][b];
                    let contrib_m = w_old * da_m;
                    let w_couple = w_old.abs();
                    if da_m.abs() > 0.03 && w_couple >= 0.012 {
                        let err_without_m = sanitize(total_err + contrib_m);
                        muscle_err_gain[m] +=
                            w_couple * (err_without_m * err_without_m - total_err * total_err);
                        muscle_null_sq[m] +=
                            w_couple * (err_without_m * err_without_m) + 1.0e-5;
                    }
                    let sign_flip = da_m.abs() > 0.12 && actual_dv * contrib_m < -1.0e-3;
                    let base_decay = if sign_flip { 0.90 } else { 0.993 };
                    let eff_decay =
                        1.0 - (1.0 - base_decay) * nlms_scale * da_m.abs().clamp(0.03, 1.0);
                    let lr = (if sign_flip { 0.085 } else { 0.045 }) * nlms_scale;
                    self.action_sensory_w[m][b] = sanitize_clamp(
                        eff_decay * w_old + lr * clean_err * da_m,
                        -1.5,
                        1.5,
                    );
                }
            }
        }

        if prev.committed {
            for mp in &mut membrane.motor_pores {
                if !mp.alive {
                    continue;
                }
                let ch = mp.actuator_channel as usize;
                if ch >= TRACKED_ACTUATORS {
                    continue;
                }
                let myelin_m = (motor_myelin[ch] * 1.8).clamp(0.0, 1.0);
                if centered_act[ch].abs() > 0.03 {
                    let r2_m = if muscle_null_sq[ch] > 1.0e-5 {
                        (muscle_err_gain[ch] / muscle_null_sq[ch]).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let jac_m = (muscle_sum_jac[ch] * 2.2).clamp(0.0, 1.0);
                    let target_m =
                        sanitize_clamp(0.35 * r2_m + 0.35 * jac_m + 0.30 * myelin_m, 0.0, 1.0);
                    let alpha = if target_m >= mp.motor_mastery {
                        0.0012
                    } else {
                        0.00025
                    };
                    mp.motor_mastery = sanitize_clamp(
                        (1.0 - alpha) * mp.motor_mastery + alpha * target_m,
                        0.0,
                        1.0,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_delayed_credit_and_morphogenesis(
        &mut self,
        ring: &EfferenceRing,
        cells: &mut CellArena,
        membrane: &mut PoreMembrane,
        valence_scale: f32,
        raw_somatic_valence: f32,
        max_tubes: usize,
        rng: &mut Rng64,
    ) -> (u64, u64, u64) {
        let mut splits = 0u64;
        let mut shortcuts = 0u64;
        let mut atrophied = 0u64;
        let mut polychronous_sprouted = false;

        let now_seq_target = cells.active_indices.iter().copied().find_map(|ci| {
            let c = cells.slots.get(ci as usize)?;
            if c.alive && c.sensory_trophic_ema > 0.008 && c.v_real.abs() > 0.16 {
                Some((ci, c.v_real))
            } else {
                None
            }
        });

        let scale = valence_scale.max(1.5e-4);

        for lag in 1..=RING_LEN {
            let Some(entry) = ring.at_lag(lag) else {
                continue;
            };
            if entry.tick == 0 {
                continue;
            }
            let kernel = (-((lag.saturating_sub(1)) as f32) / 2.2).exp();
            let raw_diff = sanitize(raw_somatic_valence - entry.raw_valence_at_commit);
            let post_harvest_refractory =
                (raw_somatic_valence >= 0.0 && entry.raw_valence_at_commit > 0.12 && raw_diff < -0.08)
                    || entry.tick == 0;
            let interval_delta = if post_harvest_refractory {
                0.0
            } else {
                raw_diff
            };
            let local_level =
                0.25 * (raw_somatic_valence.abs() + entry.raw_valence_at_commit.abs()) + 1.0e-4;
            let denom = interval_delta.abs() + local_level.max(0.15 * scale);
            let weber_diff = interval_delta / denom;
            let pain_term = if raw_somatic_valence < 0.0 && raw_diff < 0.0 {
                2.0 * raw_diff
            } else {
                0.0
            };
            let td_adv = sanitize_clamp(1.75 * weber_diff + pain_term, -2.0, 2.0);

            if raw_somatic_valence.abs() > 1.0e-4 || td_adv.abs() > 0.02 {
                for &(c_idx, v_past, dv_past) in &entry.active_cells {
                    let Some(cell) = cells.slots.get_mut(c_idx as usize) else {
                        continue;
                    };
                    if !cell.alive || cell.sensory_trophic_ema < 0.005 {
                        continue;
                    }
                    let lr_v = 0.04 * kernel;
                    let target_vh = td_adv * v_past;
                    let decay_vh = if cell.v_homeo * target_vh < -1.0e-4 {
                        0.12
                    } else {
                        0.015 * (td_adv.abs() + 0.02)
                    };
                    cell.v_homeo = sanitize_clamp(
                        (1.0 - decay_vh) * cell.v_homeo + lr_v * target_vh,
                        -2.0,
                        2.0,
                    );
                    let target_vdh = td_adv * dv_past;
                    let decay_vdh = if cell.v_delta_homeo * target_vdh < -1.0e-4 {
                        0.12
                    } else {
                        0.015 * (td_adv.abs() + 0.02)
                    };
                    cell.v_delta_homeo = sanitize_clamp(
                        (1.0 - decay_vdh) * cell.v_delta_homeo + 1.4 * lr_v * target_vdh,
                        -2.0,
                        2.0,
                    );
                    if raw_somatic_valence > 0.0 {
                        cell.atp_credit_ema = sanitize_clamp(
                            cell.atp_credit_ema + 0.05 * kernel * raw_somatic_valence * v_past.abs(),
                            0.0,
                            4.0,
                        );
                    }
                }
            }

            let lag_u8 = lag as u8;
            let seq_kernel = (-((lag.saturating_sub(1)) as f32) / 5.5).exp();

            // Multi-syllable / long-horizon polychronous CellToCell sequence learning across 100..320 ms (lag 5..=16),
            // leaving fast sensorimotor lags 1..=4 to the committed cortical-motor loop below.
            if lag >= 5 {
                for &(c_idx, v_past, dv_past) in &entry.active_cells {
                    let pre_sig = sanitize(v_past + 0.25 * dv_past);
                    if pre_sig.abs() < 1.0e-3 {
                        continue;
                    }
                    let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize).cloned() else {
                        continue;
                    };
                    for t_idx in out_list {
                        let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                            continue;
                        };
                        if !tube.alive || tube.kind != TubeKind::CellToCell || tube.delay != lag_u8 {
                            continue;
                        }
                        tube.age_ticks = tube.age_ticks.saturating_add(1);
                        let Some(dst_cell) = cells.slots.get(tube.post as usize) else {
                            continue;
                        };
                        if !dst_cell.alive {
                            continue;
                        }
                        let post_now = dst_cell.v_real;
                        let pred_err = sanitize_clamp(dst_cell.pred_err, -1.5, 1.5);
                        tube.w_pred = sanitize_clamp(
                            0.9995 * tube.w_pred + 0.015 * pred_err * pre_sig,
                            -0.30,
                            0.30,
                        );
                        if entry.committed || raw_somatic_valence.abs() > 0.01 {
                            let du = 0.010 * seq_kernel * td_adv * pre_sig * post_now
                                - 0.002 * (td_adv.abs() + 0.02) * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                        }
                        tube.usage_ema = sanitize_clamp(
                            0.985 * tube.usage_ema + 0.015 * (pre_sig * post_now).abs(),
                            0.0,
                            2.0,
                        );
                    }
                }

                if !polychronous_sprouted
                    && (lag == 6 || lag == 10 || lag == 14)
                    && raw_somatic_valence > 0.06
                    && td_adv > 0.05
                    && ((entry.tick.wrapping_add(lag as u64)) & 15) == 0
                    && self.alive_count + 16 <= max_tubes
                    && !entry.active_cells.is_empty()
                    && let Some((dst_cell, dst_v)) = now_seq_target
                {
                    let n_act = entry.active_cells.len();
                    let rot = (entry.tick as usize) % n_act;
                    for off in 0..n_act {
                        let (up_cell, up_v, _) = entry.active_cells[(rot + off) % n_act];
                        if up_cell == dst_cell || up_v.abs() <= 0.18 {
                            continue;
                        }
                        if self
                            .outgoing_by_cell
                            .get(up_cell as usize)
                            .map_or(0, Vec::len)
                            >= 36
                        {
                            continue;
                        }
                        if !self.has_cell_tube_at_delay(up_cell, dst_cell, lag_u8) {
                            let sign = up_v.signum() * dst_v.signum();
                            let mut seq_tube = Tube::newborn(
                                TubeKind::CellToCell,
                                up_cell,
                                dst_cell,
                                lag_u8,
                                0.022 * sign,
                            );
                            seq_tube.w_pred = 0.08 * sign;
                            if self.add_tube(seq_tube, max_tubes).is_some() {
                                shortcuts += 1;
                                polychronous_sprouted = true;
                            }
                            break;
                        }
                    }
                }
            }

            if !entry.committed || lag > 4 {
                continue;
            }

            if lag == 1
                && let Some(prev_entry) = ring.at_lag(2)
                && prev_entry.committed
            {
                let mut dot = 0.0_f32;
                for m in 0..TRACKED_ACTUATORS {
                    dot += entry.motor_out[m] * prev_entry.motor_out[m];
                }
                if dot.abs() > 0.02 && td_adv.abs() > 0.02 {
                    for mp in &mut membrane.motor_pores {
                        if mp.alive {
                            mp.klinokinesis_gain = sanitize_clamp(
                                0.985 * mp.klinokinesis_gain + 0.038 * td_adv * dot.signum(),
                                -0.75,
                                0.75,
                            );
                        }
                    }
                }
            }

            let mut td_past = [0.0_f32; TRACKED_ACTUATORS];
            let mut max_reward_jac = [0.0_f32; TRACKED_ACTUATORS];
            for &(c_idx, v_past, dv_past) in &entry.active_cells {
                if let Some(cell) = cells.slots.get(c_idx as usize)
                    && cell.alive
                    && (cell.v_homeo.abs() + cell.v_delta_homeo.abs()) > 0.12
                {
                    let b = (c_idx as usize) % PRED_BUCKETS;
                    for (m, dst) in max_reward_jac.iter_mut().enumerate() {
                        let j = self.action_sensory_w[m][b].abs();
                        if j > *dst {
                            *dst = j;
                        }
                    }
                }
                let pre_sig = sanitize(v_past + 0.25 * dv_past);
                if pre_sig.abs() < 1.0e-3 {
                    continue;
                }
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                    continue;
                };
                for &t_idx in out_list {
                    let Some(tube) = self.slots.get(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive || tube.kind != TubeKind::CellToMotor || tube.delay != lag_u8 {
                        continue;
                    }
                    let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                        continue;
                    };
                    if !mp.alive {
                        continue;
                    }
                    let act_ch = mp.actuator_channel as usize;
                    if act_ch < TRACKED_ACTUATORS {
                        let g = tube.effective_g(&cells.slots);
                        td_past[act_ch] = sanitize(td_past[act_ch] + g * pre_sig * mp.conductance);
                    }
                }
            }

            let mut split_requests: Vec<(u32, f32)> = Vec::new();
            let mut shortcut_requests: Vec<(u32, u32, f32)> = Vec::new();

            for &(c_idx, v_past, dv_past) in &entry.active_cells {
                let pre_sig = sanitize(v_past + 0.25 * dv_past);
                if pre_sig.abs() < 1.0e-3 {
                    continue;
                }
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize).cloned() else {
                    continue;
                };

                for t_idx in out_list {
                    let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive {
                        continue;
                    }
                    tube.age_ticks = tube.age_ticks.saturating_add(1);

                    match tube.kind {
                        TubeKind::CellToCell if tube.delay == lag_u8 => {
                            let Some(dst_cell) = cells.slots.get(tube.post as usize) else {
                                continue;
                            };
                            if !dst_cell.alive {
                                continue;
                            }
                            let post_now = dst_cell.v_real;
                            let pred_err = sanitize_clamp(post_now - tube.w_pred * pre_sig, -2.0, 2.0);
                            tube.w_pred = sanitize_clamp(
                                tube.w_pred + 0.035 * pred_err * pre_sig,
                                -2.0,
                                2.0,
                            );
                            let du = 0.02 * kernel * td_adv * pre_sig * post_now
                                - 0.004 * (td_adv.abs() + 0.02) * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                            tube.usage_ema =
                                sanitize_clamp(0.98 * tube.usage_ema + 0.02 * (pre_sig * post_now).abs(), 0.0, 2.0);
                        }
                        TubeKind::CellToMotor if lag == tube.delay as usize => {
                            let m_slot = tube.post as usize;
                            let (act_ch, mp_alive) = match membrane.motor_pores.get(m_slot) {
                                Some(mp) => (mp.actuator_channel as usize, mp.alive),
                                None => continue,
                            };
                            if !mp_alive || act_ch >= TRACKED_ACTUATORS {
                                continue;
                            }
                            let a_emitted = entry.motor_out[act_ch];
                            if a_emitted.abs() < 1.0e-3 {
                                continue;
                            }
                            let a_det = smooth_saturate(2.35 * td_past[act_ch]);
                            let a_innov = a_emitted - a_det;
                            if td_adv > 0.03
                                && a_innov.abs() > 0.02
                                && let Some(mp_mut) = membrane.motor_pores.get_mut(m_slot)
                            {
                                mp_mut.conductance =
                                    sanitize_clamp(0.994 * mp_mut.conductance + 0.006 * 0.95, 0.1, 1.5);
                            }

                            let elig = pre_sig * a_emitted;
                            let raw_target = elig * td_adv;
                            let reward_coupled = max_reward_jac[act_ch] > 0.009;
                            let boot_thresh = if reward_coupled { 0.028 } else { 0.12 };
                            let bootstrap =
                                ((tube.u[0].abs() - boot_thresh) / 0.07).clamp(0.0, 0.65);
                            let eff_act = if td_adv >= 0.0 {
                                bootstrap * a_emitted + (1.0 - bootstrap) * a_innov
                            } else {
                                a_emitted
                            };
                            let credit_target = pre_sig * eff_act * td_adv;
                            let g_eff = tube.effective_g(&cells.slots);
                            let stiff = tube.u[2] + tube.u[3];
                            let active_branch =
                                g_eff.abs() >= 0.02 || tube.context_gate_cell.is_none();
                            if stiff * raw_target < -0.025 {
                                if active_branch {
                                    let lability = if td_adv < -0.10 { 0.86 } else { 0.93 };
                                    tube.u[1] = sanitize(tube.u[1] * lability);
                                    tube.u[2] = sanitize(tube.u[2] * lability);
                                    tube.u[3] = sanitize(tube.u[3] * (0.5 + 0.5 * lability));
                                }
                                if active_branch && tube.stiff_strength() > 0.045 {
                                    tube.contradiction_accum =
                                        sanitize_clamp(tube.contradiction_accum + 0.28, 0.0, 2.0);
                                    if tube.contradiction_accum >= 1.0 {
                                        split_requests.push((t_idx, raw_target.signum() * 0.58));
                                        tube.contradiction_accum = 0.0;
                                    }
                                }
                                if let Some(ctx_id) = tube.context_gate_cell
                                    && tube.gate_invert
                                    && let Some(ctx_c) = cells.slots.get_mut(ctx_id as usize)
                                    && ctx_c.alive
                                {
                                    ctx_c.v_real = smooth_saturate(ctx_c.v_real + 0.30);
                                }
                            } else if stiff * raw_target > 0.01 {
                                tube.contradiction_accum =
                                    sanitize_clamp(tube.contradiction_accum - 0.32, 0.0, 2.0);
                            } else {
                                tube.contradiction_accum = sanitize(tube.contradiction_accum * 0.92);
                            }

                            let gate_scale = if tube.context_gate_cell.is_some() {
                                if g_eff.abs() < 1.0e-4 && tube.u[0].abs() > 1.0e-3 {
                                    0.0
                                } else {
                                    1.35
                                }
                            } else {
                                1.0
                            };
                            if gate_scale > 0.0 {
                                let oja_decay = 0.065 * (td_adv.abs() + 0.025) * tube.u[0];
                                let lr_tube = if reward_coupled { 0.34 } else { 0.18 };
                                let du = gate_scale * lr_tube * kernel * credit_target - oja_decay;
                                tube.apply_benna_fusi_impulse(du, 1.0);
                                tube.w_pred = sanitize_clamp(
                                    (1.0 - 0.06 * (td_adv.abs() + 0.025)) * tube.w_pred
                                        + gate_scale * 0.24 * kernel * credit_target,
                                    -1.5,
                                    1.5,
                                );
                            }
                            tube.usage_ema = sanitize_clamp(
                                0.97 * tube.usage_ema + 0.03 * (pre_sig * a_emitted).abs(),
                                0.0,
                                2.0,
                            );

                            if td_adv > 0.02
                                && let Some(prev_e) = ring.at_lag(lag + 1)
                                && let Some(&(up_cell, up_v, _)) = prev_e.active_cells.first()
                                && up_cell != c_idx
                                && up_v.abs() > 0.08
                                && cells
                                    .slots
                                    .get(c_idx as usize)
                                    .is_some_and(|cc| cc.surprise_ema < 0.25)
                            {
                                if tube.last_upstream_cell == up_cell {
                                    tube.chain_score = sanitize_clamp(tube.chain_score + 0.28, 0.0, 2.0);
                                    if tube.chain_score >= 1.0 {
                                        shortcut_requests.push((
                                            up_cell,
                                            tube.post,
                                            0.18 * up_v.signum() * a_emitted.signum(),
                                        ));
                                        tube.chain_score = 0.0;
                                    }
                                } else {
                                    tube.last_upstream_cell = up_cell;
                                    tube.chain_score = 0.25;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            if !split_requests.is_empty() {
                let ctx_cell = cells.find_slow_context_cell().unwrap_or(0);
                if let Some(ctx_c) = cells.slots.get_mut(ctx_cell as usize)
                    && ctx_c.alive
                {
                    ctx_c.v_real = smooth_saturate(ctx_c.v_real + 0.75);
                }
                for (old_t_idx, new_u0) in split_requests {
                    let (kind, pre, post, delay, target_sib_invert) = {
                        let old = &mut self.slots[old_t_idx as usize];
                        let sib_inv = if let Some(existing_ctx) = old.context_gate_cell {
                            if existing_ctx == ctx_cell {
                                !old.gate_invert
                            } else {
                                old.context_gate_cell = Some(ctx_cell);
                                old.gate_invert = true;
                                false
                            }
                        } else {
                            old.context_gate_cell = Some(ctx_cell);
                            old.gate_invert = true;
                            false
                        };
                        old.u[0] = sanitize(old.u[0] * 0.35);
                        (old.kind, old.pre, old.post, old.delay, sib_inv)
                    };
                    let mut reused = false;
                    if let Some(out_list) = self.outgoing_by_cell.get(pre as usize).cloned() {
                        for tid in out_list {
                            if tid != old_t_idx
                                && let Some(sib) = self.slots.get_mut(tid as usize)
                                && sib.alive
                                && sib.kind == kind
                                && sib.post == post
                                && sib.delay == delay
                                && sib.gate_invert == target_sib_invert
                            {
                                sib.context_gate_cell = Some(ctx_cell);
                                sib.u[0] = new_u0;
                                sib.w_pred = 0.8 * new_u0;
                                splits += 1;
                                reused = true;
                                break;
                            }
                        }
                    }
                    if !reused && self.alive_count + 2 <= max_tubes {
                        let mut sibling = Tube::newborn(kind, pre, post, delay, new_u0);
                        sibling.w_pred = 0.8 * new_u0;
                        sibling.context_gate_cell = Some(ctx_cell);
                        sibling.gate_invert = target_sib_invert;
                        if self.add_tube(sibling, max_tubes).is_some() {
                            splits += 1;
                        }
                    }
                    if pre != ctx_cell
                        && !self.has_direct_tube(TubeKind::CellToCell, pre, ctx_cell)
                    {
                        let ctx_drive =
                            Tube::newborn(TubeKind::CellToCell, pre, ctx_cell, 1, 0.42);
                        let _ = self.add_tube(ctx_drive, max_tubes);
                    }
                    if kind == TubeKind::CellToMotor
                        && !self.has_direct_tube(TubeKind::CellToMotor, ctx_cell, post)
                    {
                        let brake = Tube::newborn(
                            TubeKind::CellToMotor,
                            ctx_cell,
                            post,
                            1,
                            -0.16,
                        );
                        let _ = self.add_tube(brake, max_tubes);
                    }
                }
            }

            for (up_cell, motor_post, init_u0) in shortcut_requests {
                if !self.has_direct_tube(TubeKind::CellToMotor, up_cell, motor_post) {
                    let shortcut =
                        Tube::newborn(TubeKind::CellToMotor, up_cell, motor_post, 1, init_u0);
                    if self.add_tube(shortcut, max_tubes).is_some() {
                        shortcuts += 1;
                    }
                }
            }
        }

        if (rng.next_u32() & 15) == 0 && self.alive_count > 32 {
            for idx in 0..self.slots.len() {
                if !self.slots[idx].alive {
                    continue;
                }
                let t = &self.slots[idx];
                if t.age_ticks > 600
                    && t.u[0].abs() < 0.004
                    && t.stiff_strength() < 0.005
                    && t.usage_ema < 0.002
                {
                    let pre = t.pre as usize;
                    self.slots[idx].alive = false;
                    self.free_list.push(idx as u32);
                    self.alive_count = self.alive_count.saturating_sub(1);
                    if let Some(list) = self.outgoing_by_cell.get_mut(pre) {
                        list.retain(|&x| x != idx as u32);
                    }
                    atrophied += 1;
                    break;
                }
            }
        }

        (splits, shortcuts, atrophied)
    }

    pub fn consolidate_rest_step(&mut self) {
        for tube in &mut self.slots {
            if !tube.alive {
                continue;
            }
            if tube.u[0].abs() >= 0.045 {
                tube.apply_benna_fusi_impulse(0.0, 4.0);
            } else if tube.stiff_strength() < 0.025 {
                tube.u[0] = sanitize(tube.u[0] * 0.995);
                tube.u[1] = sanitize(tube.u[1] * 0.995);
            }
        }
    }

    pub fn prune_tubes_referencing_cells(&mut self, dead_cells: &[u32]) -> u64 {
        if dead_cells.is_empty() {
            return 0;
        }
        let mut removed = 0u64;
        for idx in 0..self.slots.len() {
            if !self.slots[idx].alive {
                continue;
            }
            let pre = self.slots[idx].pre;
            let post = self.slots[idx].post;
            let kind = self.slots[idx].kind;
            let dead_pre = dead_cells.contains(&pre);
            let dead_post = kind == TubeKind::CellToCell && dead_cells.contains(&post);
            if dead_pre || dead_post {
                self.slots[idx].alive = false;
                self.free_list.push(idx as u32);
                self.alive_count = self.alive_count.saturating_sub(1);
                if let Some(list) = self.outgoing_by_cell.get_mut(pre as usize) {
                    list.retain(|&x| x != idx as u32);
                }
                removed += 1;
            } else if let Some(g) = self.slots[idx].context_gate_cell
                && dead_cells.contains(&g)
            {
                self.slots[idx].context_gate_cell = None;
            }
        }
        removed
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rollout_and_select_pulse(
        &self,
        num_actuators: usize,
        cells: &CellArena,
        membrane: &PoreMembrane,
        body: &mut BodyCore,
        ring: &EfferenceRing,
        _mode: ControlMode,
        rng: &mut Rng64,
    ) -> RolloutOutcome {
        if num_actuators == 0 {
            return RolloutOutcome {
                pulse: FieldPulse::silent(0),
                committed: false,
                rehearsals: 0,
                pred_with: Vec::new(),
                pred_null: Vec::new(),
            };
        }

        let mut tube_drive = vec![0.0_f32; num_actuators];
        let mut tube_pred_action_val = vec![0.0_f32; num_actuators];
        let mut slow_inhib_brake = vec![0.0_f32; num_actuators];
        let mut stiff_motor_conf = vec![0.0_f32; num_actuators];
        let mut mastery_by_act = vec![0.0_f32; num_actuators];

        for mp in &membrane.motor_pores {
            if mp.alive {
                let ch = mp.actuator_channel as usize;
                if ch < num_actuators {
                    mastery_by_act[ch] = mp.motor_mastery.clamp(0.0, 1.0);
                }
            }
        }

        for &c_idx in &cells.active_indices {
            let Some(cell) = cells.slots.get(c_idx as usize) else {
                continue;
            };
            if !cell.alive {
                continue;
            }
            let sig = sanitize(cell.v_real + 0.25 * cell.delta_v);
            let is_slow = cell.tau >= 5.0;
            let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                continue;
            };
            for &t_idx in out_list {
                let Some(tube) = self.slots.get(t_idx as usize) else {
                    continue;
                };
                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                    continue;
                }
                let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                    continue;
                };
                if !mp.alive {
                    continue;
                }
                let act_ch = mp.actuator_channel as usize;
                if act_ch >= num_actuators {
                    continue;
                }
                let g = tube.effective_g(&cells.slots);
                let contrib = sanitize(g * sig * mp.conductance);
                tube_drive[act_ch] = sanitize(tube_drive[act_ch] + contrib);
                let pred_contrib = sanitize((0.65 * g + 0.85 * tube.w_pred) * sig * mp.conductance);
                tube_pred_action_val[act_ch] = sanitize(tube_pred_action_val[act_ch] + pred_contrib);
                stiff_motor_conf[act_ch] =
                    sanitize_clamp(stiff_motor_conf[act_ch] + tube.stiff_strength(), 0.0, 2.0);
                if is_slow && g < -0.01 {
                    slow_inhib_brake[act_ch] = sanitize(slow_inhib_brake[act_ch] + (-g) * sig.abs());
                }
            }
        }

        let mut explore_wave = vec![0.0_f32; num_actuators];
        body.next_exploratory_wave(rng, &mut explore_wave);
        let mut explore_alt = vec![0.0_f32; num_actuators];
        for dst in &mut explore_alt {
            *dst = rng.next_f32_signed();
        }

        let mut prev_act = vec![0.0_f32; num_actuators];
        let mut recent_val_delta = 0.0_f32;
        if let Some(p1) = ring.at_lag(1)
            && p1.committed
        {
            for (m, dst) in prev_act.iter_mut().enumerate().take(TRACKED_ACTUATORS) {
                *dst = p1.motor_out[m];
            }
            recent_val_delta = sanitize_clamp(
                body.fast_valence + cells.somatic_value_estimate() - p1.somatic_value_est,
                -2.0,
                2.0,
            );
        }

        let mut klino_gain_mean = 0.0_f32;
        let mut klino_cnt = 0usize;
        for mp in &membrane.motor_pores {
            if mp.alive {
                klino_gain_mean += mp.klinokinesis_gain;
                klino_cnt += 1;
            }
        }
        if klino_cnt > 0 {
            klino_gain_mean /= klino_cnt as f32;
        }

        let mut sensory_cells: Vec<(u32, f32, f32)> = Vec::with_capacity(32);
        for &c_idx in &cells.active_indices {
            if sensory_cells.len() >= 32 {
                break;
            }
            if let Some(c) = cells.slots.get(c_idx as usize)
                && c.alive
                && c.sensory_trophic_ema > 0.005
            {
                sensory_cells.push((c_idx, c.v_real, c.delta_v));
            }
        }

        let thermal_need = (-body.fast_valence * 6.0).clamp(0.0, 1.0);
        let atp_surplus = ((body.atp - 6.0) / 10.0).clamp(0.0, 1.5) * (1.0 - thermal_need);
        let last_tick = ring.at_lag(1).map_or(0, |e| e.tick);
        let probe_window = atp_surplus > 0.20 && (last_tick & 15) < 3;
        let mut model_grad = vec![0.0_f32; num_actuators];
        let mut epistemic_drive = vec![0.0_f32; num_actuators];
        for &(c_idx, _v_now, _dv_now) in &sensory_cells {
            let b = (c_idx as usize) % PRED_BUCKETS;
            let cell = &cells.slots[c_idx as usize];
            let homeo_desire = sanitize(0.45 * cell.v_homeo + 0.85 * cell.v_delta_homeo);
            let predictable_neutral = cell.surprise_ema < 0.45
                && (cell.v_homeo.abs() + cell.v_delta_homeo.abs()) < 0.18;
            for m in 0..num_actuators.min(TRACKED_ACTUATORS) {
                let jac = self.action_sensory_w[m][b];
                if homeo_desire.abs() > 1.0e-4 {
                    model_grad[m] = sanitize(model_grad[m] + homeo_desire * jac);
                }
                if predictable_neutral && atp_surplus > 0.01 && jac.abs() > 0.026 {
                    let wave_dir = if explore_wave[m] >= 0.0 { 1.0 } else { -1.0 };
                    epistemic_drive[m] =
                        sanitize(epistemic_drive[m] + 2.10 * atp_surplus * jac.abs() * wave_dir);
                }
            }
        }

        let ex_scale = (0.12 + 0.68 * body.temperature).clamp(0.08, 0.90);
        let hunger = ((6.0 - body.atp) / 6.0).clamp(0.0, 1.0).max(thermal_need);
        let vigor = 1.0 + 0.90 * hunger;
        let mut has_survival_vec = vec![false; num_actuators];
        let mut candidates: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0_f32; num_actuators]);
        for m in 0..num_actuators {
            let td = tube_drive[m] * vigor;
            let mg = model_grad[m] * vigor;
            let mastery = mastery_by_act[m];
            let stiff_c = stiff_motor_conf[m].min(1.0);
            let babble_maturity = (1.0 - 0.35 * mastery) * (1.0 - 0.20 * stiff_c);
            let has_survival_drive = td.abs() + mg.abs() >= 0.14;
            has_survival_vec[m] = has_survival_drive;
            let ep = if !has_survival_drive {
                epistemic_drive[m]
            } else {
                0.0
            };
            let newborn_spontaneity = 0.28 * (1.0 - mastery);
            let idle_ex_gate = if ep.abs() > 0.05 {
                0.38 * babble_maturity + 0.12
            } else {
                (0.08 + newborn_spontaneity + 0.20 * atp_surplus.min(1.0)) * babble_maturity
            };
            let local_ex_scale = if has_survival_drive {
                ex_scale * (0.45 + 0.55 * babble_maturity)
            } else {
                ex_scale * idle_ex_gate
            };
            let ex = explore_wave[m];
            let ex2 = explore_alt[m];
            let pa = prev_act[m];

            candidates[0][m] =
                smooth_saturate(2.35 * td + 1.45 * mg + ep + 0.14 * local_ex_scale * ex);
            candidates[1][m] = smooth_saturate(
                1.75 * td
                    + 1.15 * mg
                    + ep
                    + 0.65 * klino_gain_mean.abs().max(0.15) * pa
                    + 0.14 * local_ex_scale * ex,
            );
            candidates[2][m] =
                smooth_saturate(2.25 * mg + 0.95 * td + ep + 0.26 * local_ex_scale * ex2);
            candidates[3][m] = smooth_saturate(
                0.80 * td + 0.80 * mg + ep + (0.42 + 0.48 * ex_scale) * babble_maturity * ex,
            );
            candidates[4][m] =
                smooth_saturate(0.55 * td + 0.55 * mg + 0.82 * ex_scale * babble_maturity * ex2);
        }

        let pain_bar = 0.045;
        let mut policy_conf = 0.0_f32;
        for (m, &pv) in tube_pred_action_val.iter().enumerate() {
            policy_conf += pv.abs() + 0.8 * model_grad[m].abs();
        }
        let mean_policy_conf = policy_conf / (num_actuators.max(1) as f32);
        let forage_thresh = 0.0067 + 0.0065 * hunger;
        let forage_search_w = if mean_policy_conf < forage_thresh {
            0.035 * body.temperature
        } else if probe_window {
            0.12 * atp_surplus
        } else {
            0.0
        };
        let effort_w = if mean_policy_conf < forage_thresh || probe_window {
            0.0
        } else {
            0.065 * (1.35 - 0.65 * atp_surplus.min(1.5)) * (1.0 - 0.70 * hunger)
        };
        let epistemic_w = 1.85 * atp_surplus;
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut rehearsals = 0u64;

        for (k_idx, cand) in candidates.iter().enumerate() {
            let mut pred_val = 0.0_f32;
            let mut epistemic_val = 0.0_f32;
            let mut act_energy = 0.0_f32;
            let mut brake_pen = 0.0_f32;
            let mut cont_dot = 0.0_f32;

            for m in 0..num_actuators {
                act_energy += cand[m] * cand[m];
                brake_pen += slow_inhib_brake[m] * cand[m].abs();
                cont_dot += cand[m] * prev_act[m];
                pred_val += tube_pred_action_val[m] * cand[m];
            }

            for &(c_idx, v_now, dv_now) in &sensory_cells {
                let b = (c_idx as usize) % PRED_BUCKETS;
                let self_dv = self.sensory_self_w[b] * dv_now;
                let mut act_dv = 0.0_f32;
                let mut ep_act_dv = 0.0_f32;
                for (m, &am) in cand.iter().enumerate().take(TRACKED_ACTUATORS) {
                    let jac = self.action_sensory_w[m][b];
                    let da_m = am - self.act_mean[m];
                    act_dv += jac * da_m;
                    if jac.abs() > 0.026 && !has_survival_vec[m] {
                        ep_act_dv += (jac * da_m).abs();
                    }
                }
                let p_null = smooth_saturate(v_now + self_dv);
                let v_step1 = smooth_saturate(v_now + self_dv + act_dv);
                let delta_from_act = v_step1 - p_null;
                let cell = &cells.slots[c_idx as usize];
                pred_val += 0.50 * cell.v_homeo * delta_from_act
                    + 0.75 * cell.v_delta_homeo * act_dv;
                if cell.surprise_ema < 0.45
                    && (cell.v_homeo.abs() + cell.v_delta_homeo.abs()) < 0.18
                {
                    epistemic_val += ep_act_dv;
                }
            }

            pred_val += klino_gain_mean * recent_val_delta * cont_dot;
            pred_val -= 0.75 * brake_pen;
            pred_val = sanitize(pred_val);

            let vetoed = pred_val < -pain_bar;
            if vetoed {
                rehearsals += 1;
            }
            let mean_sq_energy = act_energy / (num_actuators as f32);
            let rms = mean_sq_energy.sqrt();
            let mut total_score = pred_val
                + epistemic_w * epistemic_val
                + forage_search_w * rms
                - effort_w * mean_sq_energy;
            if vetoed {
                total_score -= 1.5;
            }

            if total_score > best_score {
                best_score = total_score;
                best_idx = k_idx;
            }
        }

        let chosen = candidates[best_idx].clone();

        let mut max_abs = 0.0_f32;
        for &v in &chosen {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }

        let committed = max_abs >= body.motor_clutch_threshold;
        let final_out = if committed {
            let mut smoothed = chosen;
            for m in 0..num_actuators {
                let maturity = (0.65 * mastery_by_act[m] + 0.35 * stiff_motor_conf[m].min(1.0))
                    .clamp(0.0, 1.0);
                let spindle_alpha = if has_survival_vec[m] {
                    (0.92 - 0.26 * maturity).clamp(0.58, 0.92)
                } else {
                    (0.95 - 0.50 * maturity).clamp(0.40, 0.95)
                };
                smoothed[m] = sanitize_clamp(
                    (1.0 - spindle_alpha) * prev_act[m] + spindle_alpha * smoothed[m],
                    -1.0,
                    1.0,
                );
            }
            smoothed
        } else {
            rehearsals += 1;
            vec![0.0_f32; num_actuators]
        };

        let mut pred_with = Vec::with_capacity(sensory_cells.len());
        let mut pred_null = Vec::with_capacity(sensory_cells.len());
        for &(c_idx, v_now, dv_now) in &sensory_cells {
            let b = (c_idx as usize) % PRED_BUCKETS;
            let base_dv = self.sensory_self_w[b] * dv_now;
            let p_null = smooth_saturate(v_now + base_dv);
            let mut act_dv = 0.0_f32;
            if committed {
                for (m, &am) in final_out.iter().enumerate().take(TRACKED_ACTUATORS) {
                    let w_mb = self.action_sensory_w[m][b];
                    if w_mb.abs() >= 0.022 {
                        act_dv += w_mb * (am - self.act_mean[m]);
                    }
                }
            }
            let p_act = smooth_saturate(v_now + base_dv + act_dv);
            pred_with.push((c_idx, p_act));
            pred_null.push((c_idx, p_null));
        }

        RolloutOutcome {
            pulse: FieldPulse {
                actuator_out: final_out,
            },
            committed,
            rehearsals,
            pred_with,
            pred_null,
        }
    }

    #[must_use]
    pub fn topology_counts(&self, membrane: &PoreMembrane) -> (usize, usize, usize) {
        let mut stiff = 0usize;
        let mut inhib = 0usize;
        let mut delay_tubes = 0usize;

        for t in &self.slots {
            if !t.alive {
                continue;
            }
            if t.stiff_strength() > 0.06 {
                stiff += 1;
            }
            if t.u[0] < -0.015 {
                inhib += 1;
            }
            if t.delay >= 2 && t.u[0].abs() > 0.02 {
                delay_tubes += 1;
            }
        }
        for sp in &membrane.sensor_pores {
            if sp.alive && sp.delay >= 2 && sp.conductance > 0.03 {
                delay_tubes += 1;
            }
        }
        (stiff, inhib, delay_tubes)
    }

    pub fn zero_learned_conductances(&mut self) {
        for t in &mut self.slots {
            t.u = [0.0; 4];
            t.w_pred = 0.0;
            t.contradiction_accum = 0.0;
            t.chain_score = 0.0;
        }
        self.action_sensory_w = [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS];
        self.sensory_self_w = [0.0; PRED_BUCKETS];
        self.act_mean = [0.0; TRACKED_ACTUATORS];
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_usize(self.slots.len());
        hasher.write_usize(self.alive_count);
        for t in &self.slots {
            hasher.write_bool(t.alive);
            if t.alive {
                hasher.write_u8(t.kind as u8);
                hasher.write_u32(t.pre);
                hasher.write_u32(t.post);
                hasher.write_u8(t.delay);
                for &uk in &t.u {
                    hasher.write_f32(uk);
                }
                hasher.write_f32(t.w_pred);
                match t.context_gate_cell {
                    None => hasher.write_u32(u32::MAX),
                    Some(g) => hasher.write_u32(g),
                }
                hasher.write_bool(t.gate_invert);
            }
        }
        for row in &self.action_sensory_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for &w in &self.sensory_self_w {
            hasher.write_f32(w);
        }
        for &am in &self.act_mean {
            hasher.write_f32(am);
        }
    }
}
