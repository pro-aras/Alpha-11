//! Directed Benna-Fusi cascade tubes, 16-tick efference ring, agency, and rollout clutch.

use a11_medium::{ControlMode, FieldPulse};

use crate::body::BodyCore;
use crate::cells::{Cell, CellArena};
use crate::pores::{ChannelScoutStat, PoreMembrane, SensorPore};
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
    pub myelin_plasticity_accum: f32,
    pub last_upstream_cell: u32,
    pub usage_ema: f32,
    pub age_ticks: u32,
}

impl Tube {
    pub const LOGICAL_BYTES: usize = 1 + 1 + 4 + 4 + 1 + 4 * 4 + 4 + 5 + 1 + 4 * 4 + 4 + 4;

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
            myelin_plasticity_accum: 0.0,
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
    pub active_cells: Vec<(u32, f32, f32, f32)>,
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
        1 + 8
            + 1
            + 4
            + 4 * TRACKED_ACTUATORS
            + self.active_cells.len() * 16
            + self.pred_with_action.len() * 8
            + self.pred_null_action.len() * 8
            + 8
    }

    #[inline]
    #[must_use]
    pub fn cell_signal(&self, target_cell: u32) -> f32 {
        for &(ci, vr, dv, _) in &self.active_cells {
            if ci == target_cell {
                return sanitize(vr + 0.25 * dv);
            }
        }
        0.0
    }

    #[inline]
    #[must_use]
    pub fn cell_v_real(&self, target_cell: u32) -> Option<f32> {
        for &(ci, vr, _, _) in &self.active_cells {
            if ci == target_cell {
                return Some(vr);
            }
        }
        None
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
        if e.valid { Some(e) } else { None }
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
    pub pore_action_w: [[f32; PRED_BUCKETS]; TRACKED_ACTUATORS],
    pub tectal_map_w: [[f32; PRED_BUCKETS]; TRACKED_ACTUATORS],
    pub sensory_self_w: [f32; PRED_BUCKETS],
    pub pore_self_w: [f32; PRED_BUCKETS],
    pub pore_mean_v: [f32; PRED_BUCKETS],
    pub pore_scale_v: [f32; PRED_BUCKETS],
    pub pore_agency: [f32; PRED_BUCKETS],
    pub pore_dsq_ema: [f32; PRED_BUCKETS],
    pub pore_var_ema: [f32; PRED_BUCKETS],
    pub pore_bg_ild: [f32; PRED_BUCKETS],
    pub social_loom_ema: f32,
    pub spatial_prox_ring: [f32; RING_LEN],
    pub spatial_prox_cursor: u8,
    pub echo_emit_w: [f32; TRACKED_ACTUATORS],
    pub vocal_artic_w: [f32; TRACKED_ACTUATORS],
    pub somatic_limb_w: [f32; TRACKED_ACTUATORS],
    pub peer_coch_fast: f32,
    pub peer_coch_slow: f32,
    pub peer_echo_fast: f32,
    pub peer_echo_slow: f32,
    pub peer_disp_coch: f32,
    pub peer_disp_echo: f32,
    pub peer_vocal_hold: f32,
    pub peer_coch_age: u8,
    pub peer_echo_age: u8,
    pub peer_bearing_ema: [f32; TRACKED_ACTUATORS],
    pub act_mean: [f32; TRACKED_ACTUATORS],
    pub mean_interventional_agency: f32,
}

impl TubeGraph {
    #[inline]
    fn deconv_hair_cell_35(history: &[f32; 16], h: usize, lag: usize) -> f32 {
        let y0 = history[(h + 15 - lag) & 15] + 0.35;
        let y1 = history[(h + 14 - lag) & 15] + 0.35;
        let y2 = history[(h + 13 - lag) & 15] + 0.35;
        sanitize_clamp((y0 - 0.66615 * y1 + 0.07312 * y2) * 2.4572, -1.65, 1.65)
    }

    #[inline]
    fn sym_deconv_ild(history: &[f32; 16], h: usize, lag: usize) -> f32 {
        let u_rel = Self::deconv_hair_cell_35(history, h, lag);
        let sym = if u_rel > 0.60 {
            0.60 + 0.0667 * (u_rel - 0.60)
        } else {
            u_rel
        };
        sanitize_clamp(sym, -0.66, 0.66)
    }

    #[inline]
    fn laryngeal_voiced_lags(ring: &EfferenceRing, mg: usize) -> [bool; 18] {
        let mut voiced = [false; 18];
        if mg >= TRACKED_ACTUATORS {
            return voiced;
        }
        let dt_sub = 0.020_f32 / 16.0_f32;
        let omega = 75.0_f32;
        let omega_sq = omega * omega;
        let damp = 2.0_f32 * 1.05_f32 * omega;
        let mut pos = ring.at_lag(RING_LEN).map_or(0.0_f32, |e| {
            if e.committed {
                sanitize_clamp(e.motor_out[mg], -1.0, 1.0)
            } else {
                0.0
            }
        });
        let mut vel = 0.0_f32;
        for lg in (1..=RING_LEN).rev() {
            let cmd = ring.at_lag(lg).map_or(0.0_f32, |e| {
                if e.committed {
                    sanitize_clamp(e.motor_out[mg], -1.0, 1.0)
                } else {
                    0.0
                }
            });
            let mut p_sum = 0.0_f32;
            for _ in 0..16 {
                let acc = omega_sq * (cmd - pos) - damp * vel;
                vel = sanitize_clamp(vel + dt_sub * acc, -60.0, 60.0);
                pos = sanitize_clamp(pos + dt_sub * vel, -1.0, 1.0);
                if pos > 0.06 {
                    p_sum += ((pos - 0.06) / 0.94).clamp(0.0, 1.0);
                }
            }
            voiced[lg] = (p_sum * (1.0 / 16.0)) >= 0.025;
        }
        voiced
    }

    #[inline]
    fn insert_top4(top4: &mut [f32; 4], val: f32) {
        if val > top4[0] {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = top4[0];
            top4[0] = val;
        } else if val > top4[1] {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = val;
        } else if val > top4[2] {
            top4[3] = top4[2];
            top4[2] = val;
        } else if val > top4[3] {
            top4[3] = val;
        }
    }

    #[inline]
    fn is_retinal_field_pore(
        &self,
        p: usize,
        sp: &SensorPore,
        scout_stats: &[ChannelScoutStat],
    ) -> bool {
        let ch_u = sp.channel as usize;
        ch_u < scout_stats.len()
            && scout_stats[ch_u].is_smooth_spatial_field()
            && self.pore_self_w[p] >= 0.35
            && (0.010..=0.62).contains(&self.pore_agency[p])
            && self.pore_mean_v[p] > -0.02
            && sp.history.iter().all(|&v| v >= -0.10)
    }

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
            pore_action_w: [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS],
            tectal_map_w: [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS],
            sensory_self_w: [0.0; PRED_BUCKETS],
            pore_self_w: [0.0; PRED_BUCKETS],
            pore_mean_v: [0.0; PRED_BUCKETS],
            pore_scale_v: [0.05; PRED_BUCKETS],
            pore_agency: [0.0; PRED_BUCKETS],
            pore_dsq_ema: [1.0e-4; PRED_BUCKETS],
            pore_var_ema: [5.0e-3; PRED_BUCKETS],
            pore_bg_ild: [0.0; PRED_BUCKETS],
            social_loom_ema: 0.0,
            spatial_prox_ring: [0.0; RING_LEN],
            spatial_prox_cursor: 0,
            echo_emit_w: [0.0; TRACKED_ACTUATORS],
            vocal_artic_w: [0.0; TRACKED_ACTUATORS],
            somatic_limb_w: [0.0; TRACKED_ACTUATORS],
            peer_coch_fast: 0.0,
            peer_coch_slow: 0.0,
            peer_echo_fast: 0.0,
            peer_echo_slow: 0.0,
            peer_disp_coch: 0.0,
            peer_disp_echo: 0.0,
            peer_vocal_hold: 0.0,
            peer_coch_age: 255,
            peer_echo_age: 255,
            peer_bearing_ema: [0.0; TRACKED_ACTUATORS],
            act_mean: [0.0; TRACKED_ACTUATORS],
            mean_interventional_agency: 0.0,
        };

        for c in 0..n_cells {
            for hop in 0..2 {
                let dst =
                    (c + 1 + hop * 5 + rng.next_index(n_cells.saturating_sub(1).max(1))) % n_cells;
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
            + 3 * 4 * PRED_BUCKETS * TRACKED_ACTUATORS
            + 8 * 4 * PRED_BUCKETS
            + 4 * RING_LEN
            + 5 * 4 * TRACKED_ACTUATORS
            + 70
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
            for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
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

        let n_act_ag = membrane.alive_motor_count.clamp(1, TRACKED_ACTUATORS);
        if !prev.committed || mode != ControlMode::Natural {
            let idle_ret = if n_act_ag <= 8 { 0.9995 } else { 0.9996 };
            self.mean_interventional_agency =
                sanitize_clamp(idle_ret * self.mean_interventional_agency, 0.0, 1.0);
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

        for &(c_idx, _, _, _) in &prev.active_cells {
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

        for &(c_idx, v_past, dv_past, _) in &prev.active_cells {
            if (c_idx as usize) >= PRED_BUCKETS {
                continue;
            }
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
            self.sensory_self_w[b] = sanitize_clamp(
                0.998 * self.sensory_self_w[b] + 0.035 * total_err * dv_past,
                -1.2,
                1.2,
            );
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
                        muscle_null_sq[m] += w_couple * (err_without_m * err_without_m) + 1.0e-5;
                    }
                    let sign_flip = da_m.abs() > 0.12 && actual_dv * contrib_m < -1.0e-3;
                    let base_decay = if sign_flip { 0.90 } else { 0.993 };
                    let eff_decay =
                        1.0 - (1.0 - base_decay) * nlms_scale * da_m.abs().clamp(0.03, 1.0);
                    let lr = (if sign_flip { 0.085 } else { 0.045 }) * nlms_scale;
                    self.action_sensory_w[m][b] =
                        sanitize_clamp(eff_decay * w_old + lr * clean_err * da_m, -1.5, 1.5);
                }
            }
        }

        let pore_cap = PRED_BUCKETS.min(membrane.sensor_pores.len());
        let n_scout = membrane.scout_stats.len();
        let mut shuf_act = [0.0_f32; TRACKED_ACTUATORS];
        if prev.committed
            && let Some(lag9) = ring.at_lag(9)
            && lag9.committed
            && lag9.tick > 0
        {
            for m in 0..n_act_ag {
                let src_m = (m + 1) % n_act_ag;
                shuf_act[m] =
                    sanitize_clamp(lag9.motor_out[src_m] - self.act_mean[src_m], -2.0, 2.0);
            }
        }
        let mut ag_null_sq = 0.0_f32;
        let mut ag_act_sq = 0.0_f32;
        let mut ag_shuf_sq = 0.0_f32;
        let mut ag_contrast_sq = 0.0_f32;
        let mut ag_eval = 0usize;
        let mut spatial_prox_sum = 0.0_f32;
        let mut spatial_prox_cnt = 0usize;
        let mut loco_auth = [0.0_f32; TRACKED_ACTUATORS];
        let mut pore_err1 = [0.0_f32; PRED_BUCKETS];
        let adult_pruned_coch = if n_act_ag > 8
            && cells.slots.len() >= 255
            && (8 * n_act_ag..150).contains(&cells.alive_count)
        {
            let young_cortex = cells.slots.iter().all(|c| !c.alive || c.age_ticks < 60_000);
            young_cortex
                || self.slots.iter().any(|t| {
                    t.alive
                        && t.kind == TubeKind::CellToMotor
                        && t.myelin_plasticity_accum.abs() > 1.0e-4
                })
        } else {
            false
        };

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive {
                continue;
            }
            let h = sp.hist_head as usize;
            let x_now = sp.history[(h + 15) & 15];
            if sp.age_ticks <= 1 {
                for m in 0..TRACKED_ACTUATORS {
                    self.pore_action_w[m][p] = 0.0;
                    self.tectal_map_w[m][p] = 0.0;
                }
                self.pore_self_w[p] = 0.0;
                self.pore_mean_v[p] = x_now;
                self.pore_scale_v[p] = 0.05;
                self.pore_agency[p] = 0.0;
                self.pore_dsq_ema[p] = 1.0e-4;
                self.pore_var_ema[p] = 5.0e-3;
                self.pore_bg_ild[p] = 0.0;
                continue;
            }
            let x_1 = sp.history[(h + 14) & 15];
            let dx1 = sanitize_clamp(x_now - x_1, -2.0, 2.0);
            let pre_mean_p = self.pore_mean_v[p];
            let pre_scale_p = self.pore_scale_v[p].max(0.015);
            self.pore_mean_v[p] =
                sanitize_clamp(0.9985 * self.pore_mean_v[p] + 0.0015 * x_now, -2.0, 2.0);
            if x_now.abs() > 1.0e-4 || x_1.abs() > 1.0e-4 {
                let dm_p = sanitize_clamp(x_now - self.pore_mean_v[p], -2.0, 2.0);
                self.pore_dsq_ema[p] = sanitize_clamp(
                    0.9980 * self.pore_dsq_ema[p] + 0.0020 * (dx1 * dx1),
                    1.0e-6,
                    2.0,
                );
                self.pore_var_ema[p] = sanitize_clamp(
                    0.9980 * self.pore_var_ema[p] + 0.0020 * (dm_p * dm_p),
                    1.0e-5,
                    2.0,
                );
            }
            self.pore_scale_v[p] = sanitize_clamp(
                0.9960 * self.pore_scale_v[p] + 0.0040 * (2.8 * dx1.abs().min(0.18)).max(0.015),
                0.015,
                0.65,
            );
            let scale_p = self.pore_scale_v[p].max(0.015);
            if sp.age_ticks >= 3 && sp.trophic_ema >= 0.025 {
                let x_2 = sp.history[(h + 13) & 15];
                let ag_pore_cap = if n_act_ag <= 8 { 0.75 } else { 0.65 };
                let ag_da_ok = n_act_ag <= 8 || da_norm_sq >= 0.015;
                let ag_delta_min = if n_act_ag <= 8 { 0.012 } else { 0.020 };
                if prev.committed
                    && ag_da_ok
                    && sp.age_ticks >= 16
                    && self.pore_agency[p] <= ag_pore_cap
                    && !(-0.38..=-0.04).contains(&pre_mean_p)
                    && (x_1 - x_2).abs() > 1.0e-3
                {
                    let mut act_delta_o = 0.0_f32;
                    let mut shuf_delta_o = 0.0_f32;
                    for m in 0..n_act_ag {
                        let w_mp = self.pore_action_w[m][p];
                        act_delta_o += w_mp * centered_act[m];
                        shuf_delta_o += w_mp * shuf_act[m];
                    }
                    if act_delta_o.abs() > ag_delta_min {
                        let self_w_p = self.pore_self_w[p];
                        let ch_u = sp.channel as usize;
                        let is_pure_optical = n_act_ag <= 8
                            && ch_u < n_scout
                            && membrane.scout_stats[ch_u].is_smooth_spatial_field()
                            && self_w_p >= 0.35
                            && pre_mean_p > -0.02
                            && sp.history.iter().all(|&v| v >= -0.05);
                        let dz_now_o = sanitize_clamp(dx1 / pre_scale_p, -3.0, 3.0);
                        let dz_prev_o = sanitize_clamp((x_1 - x_2) / pre_scale_p, -3.0, 3.0);
                        let p_null_o = if is_pure_optical {
                            sanitize(self_w_p * dz_prev_o)
                        } else {
                            let mut num_lvl = 0.0_f32;
                            let mut den_lvl = 0.0_f32;
                            for j in 1..=12usize {
                                let x_j0 = sp.history[(h + 15 - j) & 15];
                                let x_j1 = sp.history[(h + 14 - j) & 15];
                                let x_j2 = sp.history[(h + 13 - j) & 15];
                                let dz_j = sanitize_clamp((x_j0 - x_j1) / pre_scale_p, -3.0, 3.0);
                                let dz_j_prev =
                                    sanitize_clamp((x_j1 - x_j2) / pre_scale_p, -3.0, 3.0);
                                let zlvl_j =
                                    sanitize_clamp((x_j1 - pre_mean_p) / pre_scale_p, -3.0, 3.0);
                                let res_j = dz_j - self_w_p * dz_j_prev;
                                num_lvl += res_j * zlvl_j;
                                den_lvl += zlvl_j * zlvl_j;
                            }
                            let beta_hi = if n_act_ag <= 8 { 0.0 } else { 0.95 };
                            let beta_lvl =
                                sanitize_clamp(num_lvl / (den_lvl + 0.50), -0.95, beta_hi);
                            let zlvl_0 =
                                sanitize_clamp((x_1 - pre_mean_p) / pre_scale_p, -3.0, 3.0);
                            sanitize_clamp(self_w_p * dz_prev_o + beta_lvl * zlvl_0, -3.0, 3.0)
                        };
                        let e_null = sanitize_clamp(dz_now_o - p_null_o, -3.0, 3.0);
                        let e_act = sanitize_clamp(dz_now_o - (p_null_o + act_delta_o), -3.0, 3.0);
                        let e_shuf =
                            sanitize_clamp(dz_now_o - (p_null_o + shuf_delta_o), -3.0, 3.0);
                        ag_null_sq += e_null * e_null;
                        ag_act_sq += e_act * e_act;
                        ag_shuf_sq += e_shuf * e_shuf;
                        ag_contrast_sq += act_delta_o * act_delta_o;
                        ag_eval += 1;
                    }
                }
                let dz_now = sanitize_clamp(dx1 / scale_p, -3.0, 3.0);
                let dz_prev = sanitize_clamp((x_1 - x_2) / scale_p, -3.0, 3.0);
                let pred_null = sanitize(self.pore_self_w[p] * dz_prev);
                let mut pred_act = pred_null;
                if prev.committed {
                    for (m, &da_m) in centered_act.iter().enumerate() {
                        pred_act += self.pore_action_w[m][p] * da_m;
                    }
                }
                let err_null = sanitize_clamp(dz_now - pred_null, -3.0, 3.0);
                let err_act = sanitize_clamp(dz_now - pred_act, -3.0, 3.0);
                pore_err1[p] = err_act;
                if prev.committed && da_norm_sq > 1.5e-3 {
                    let inst_gain = sanitize_clamp(
                        (err_null * err_null - err_act * err_act) / (err_null * err_null + 0.04),
                        -0.40,
                        1.0,
                    );
                    self.pore_agency[p] =
                        sanitize_clamp(0.990 * self.pore_agency[p] + 0.010 * inst_gain, -0.25, 1.0);
                }
                self.pore_self_w[p] = sanitize_clamp(
                    0.998 * self.pore_self_w[p] + 0.024 * err_act.clamp(-1.5, 1.5) * dz_prev,
                    -0.95,
                    0.95,
                );
                if prev.committed {
                    let lr_p = 0.032 * nlms_scale;
                    let clean_e = err_act.clamp(-1.2, 1.2);
                    for (m, &da_m) in centered_act.iter().enumerate() {
                        let w_old = self.pore_action_w[m][p];
                        let sign_flip = da_m.abs() > 0.14 && dz_now * (w_old * da_m) < -5.0e-3;
                        let dec = if sign_flip {
                            0.94
                        } else if adult_pruned_coch
                            && da_m.abs() < 0.03
                            && w_old >= 0.012
                            && (-0.356..=-0.16).contains(&pre_mean_p)
                        {
                            0.99995
                        } else {
                            0.9990
                        };
                        self.pore_action_w[m][p] =
                            sanitize_clamp(dec * w_old + lr_p * clean_e * da_m, -1.8, 1.8);
                    }
                }
            }
        }

        if prev.committed {
            let ag_da_ok = n_act_ag <= 8 || da_norm_sq >= 0.015;
            if ag_da_ok && ag_eval > 0 {
                let base_sq = ag_null_sq.min(ag_shuf_sq);
                if n_act_ag <= 8 {
                    let imp = base_sq - ag_act_sq - 0.010 * ag_null_sq;
                    let raw_gain =
                        sanitize_clamp(imp / (ag_null_sq + ag_contrast_sq + 0.02), -0.45, 0.45);
                    self.mean_interventional_agency = sanitize_clamp(
                        0.995 * self.mean_interventional_agency + 0.005 * raw_gain,
                        0.0,
                        1.0,
                    );
                } else {
                    let imp = base_sq - ag_act_sq - 0.012 * ag_null_sq;
                    let raw_gain =
                        sanitize_clamp(imp / (ag_null_sq + ag_contrast_sq + 0.015), -1.0, 1.0);
                    self.mean_interventional_agency = sanitize_clamp(
                        0.992 * self.mean_interventional_agency + 0.008 * raw_gain,
                        0.0,
                        1.0,
                    );
                }
            } else {
                let quiet_ret = if n_act_ag <= 8 { 0.99985 } else { 0.9996 };
                self.mean_interventional_agency =
                    sanitize_clamp(quiet_ret * self.mean_interventional_agency, 0.0, 1.0);
            }
        }

        let mut field_breadth_cal = [0u16; TRACKED_ACTUATORS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if sp.alive
                && sp.age_ticks >= 16
                && sp.trophic_ema >= 0.025
                && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
            {
                for (m, fb) in field_breadth_cal.iter_mut().enumerate() {
                    if self.pore_action_w[m][p].abs() >= 0.022 {
                        *fb = fb.saturating_add(1);
                    }
                }
            }
        }
        let mut max_breadth_cal = 0u16;
        for &fb in &field_breadth_cal {
            if fb > max_breadth_cal {
                max_breadth_cal = fb;
            }
        }
        let mut is_world_loco_cal = [false; TRACKED_ACTUATORS];
        if max_breadth_cal >= 4 {
            for m in 0..TRACKED_ACTUATORS {
                if field_breadth_cal[m].saturating_mul(2) >= max_breadth_cal {
                    is_world_loco_cal[m] = true;
                }
            }
        }
        let mut is_world_ret_cal = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                continue;
            }
            if max_breadth_cal < 4 {
                is_world_ret_cal[p] = true;
            } else {
                let mut loco_w_p = 0.0_f32;
                let mut nonloco_w_p = 0.0_f32;
                for m in 0..TRACKED_ACTUATORS {
                    let wa = self.pore_action_w[m][p].abs();
                    if is_world_loco_cal[m] {
                        if wa > loco_w_p {
                            loco_w_p = wa;
                        }
                    } else if wa > nonloco_w_p {
                        nonloco_w_p = wa;
                    }
                }
                if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                    is_world_ret_cal[p] = true;
                }
            }
            if is_world_ret_cal[p] && self.pore_agency[p] >= 0.020 {
                let h = sp.hist_head as usize;
                let x_now = sp.history[(h + 15) & 15];
                spatial_prox_sum += (x_now - self.pore_mean_v[p]).max(0.0);
                spatial_prox_cnt += 1;
                for (m, la) in loco_auth.iter_mut().enumerate() {
                    *la += self.pore_action_w[m][p].abs();
                }
            }
        }

        let spatial_prox_now = if spatial_prox_cnt > 0 {
            sanitize_clamp(spatial_prox_sum / (spatial_prox_cnt as f32), 0.0, 2.0)
        } else {
            0.0
        };
        let prox_slot = (self.spatial_prox_cursor as usize) & (RING_LEN - 1);
        self.spatial_prox_ring[prox_slot] = spatial_prox_now;
        self.spatial_prox_cursor = ((prox_slot + 1) & (RING_LEN - 1)) as u8;

        let mut max_loco_auth = 0.08_f32;
        for &la in &loco_auth {
            if la > max_loco_auth {
                max_loco_auth = la;
            }
        }
        let mut max_2_6 = [0.0_f32; TRACKED_ACTUATORS];
        let mut sum_abs15 = [0.0_f32; TRACKED_ACTUATORS];
        let mut committed_15 = 0usize;
        for lag in 1..=15 {
            if let Some(ent) = ring.at_lag(lag)
                && ent.committed
            {
                committed_15 += 1;
                for m in 0..TRACKED_ACTUATORS {
                    let am = ent.motor_out[m];
                    let a_abs = am.abs();
                    sum_abs15[m] += a_abs;
                    if (2..=6).contains(&lag) && a_abs > max_2_6[m] {
                        max_2_6[m] = a_abs;
                    }
                }
            }
        }

        let mut has_echo_pore = false;
        let mut max_echo_exc = 0.0_f32;
        let mut top_echo_cal_0 = 0.0_f32;
        let mut top_echo_cal_1 = 0.0_f32;
        let mut min_h_cal = [1.0_f32; PRED_BUCKETS];
        let mut pos_coch_w = [0.0_f32; TRACKED_ACTUATORS];
        let mut pos_coch_cnt = [0u16; TRACKED_ACTUATORS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                continue;
            }
            let h = sp.hist_head as usize;
            let x_now = sp.history[(h + 15) & 15];
            let mean_p = self.pore_mean_v[p];
            if membrane.scout_stats[ch_u].is_sparse_impulse() {
                let abs_now = (x_now - mean_p).abs();
                if self.pore_agency[p] < 0.035 {
                    let pos_now = (x_now - mean_p).max(0.0);
                    if pos_now > top_echo_cal_0 {
                        top_echo_cal_1 = top_echo_cal_0;
                        top_echo_cal_0 = pos_now;
                    } else if pos_now > top_echo_cal_1 {
                        top_echo_cal_1 = pos_now;
                    }
                }
                let mut sum_abs16 = 0.0_f32;
                for &hv in &sp.history {
                    sum_abs16 += (hv - mean_p).abs();
                }
                let exc = abs_now - sum_abs16 * (1.0 / 16.0);
                if !has_echo_pore || exc > max_echo_exc {
                    max_echo_exc = exc;
                    has_echo_pore = true;
                }
                continue;
            }
            let mut min_h = 1.0_f32;
            for &hv in &sp.history {
                if hv < min_h {
                    min_h = hv;
                }
            }
            min_h_cal[p] = min_h;
            if is_world_ret_cal[p] || self.pore_agency[p] < 0.006 {
                continue;
            }
            if (-0.38..=-0.16).contains(&min_h) && (-0.345..=-0.04).contains(&mean_p) {
                for m in 0..TRACKED_ACTUATORS {
                    let pw = self.pore_action_w[m][p];
                    if pw >= 0.015 {
                        pos_coch_w[m] += pw;
                        pos_coch_cnt[m] = pos_coch_cnt[m].saturating_add(1);
                    }
                }
            }
        }

        let mut mg_opt: Option<usize> = None;
        let mut best_mg_w = 0.05_f32;
        for m in 0..TRACKED_ACTUATORS {
            if loco_auth[m] < 0.22 * max_loco_auth
                && pos_coch_cnt[m] >= 2
                && pos_coch_w[m] > best_mg_w
            {
                best_mg_w = pos_coch_w[m];
                mg_opt = Some(m);
            }
        }

        let mut coch_dx1_sum = 0.0_f32;
        let mut coch_dx_prev_sum = 0.0_f32;
        let mut coch_cnt = 0usize;
        let mut canvas_err_sum = 0.0_f32;
        let mut canvas_cnt = 0usize;
        let mut som_inst = [0.0_f32; TRACKED_ACTUATORS];
        let mut coch_inst = [0.0_f32; TRACKED_ACTUATORS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
                || is_world_ret_cal[p]
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            let is_monaural = if let Some(mg) = mg_opt {
                self.pore_action_w[mg][p] >= 0.016 && mean_p >= -0.342
            } else {
                self.pore_agency[p] >= 0.010
                    && (-0.38..=-0.16).contains(&min_h_cal[p])
                    && (-0.345..=-0.04).contains(&mean_p)
            };
            let h = sp.hist_head as usize;
            let x_now = sp.history[(h + 15) & 15];
            let x_1 = sp.history[(h + 14) & 15];
            let x_2 = sp.history[(h + 13) & 15];
            if is_monaural {
                coch_dx1_sum += (x_now - x_1).abs();
                coch_dx_prev_sum += (x_1 - x_2).abs();
                coch_cnt += 1;
                for m in 0..TRACKED_ACTUATORS {
                    let pwa = self.pore_action_w[m][p].abs();
                    if pwa >= 0.0045 {
                        coch_inst[m] += pwa;
                    }
                }
            } else if (mean_p >= -0.22 || min_h_cal[p] > -0.18) && self.pore_agency[p] >= 0.010 {
                let loco_coupled = max_breadth_cal >= 4
                    && is_world_loco_cal
                        .iter()
                        .enumerate()
                        .any(|(ml, &il)| il && self.pore_action_w[ml][p].abs() >= 0.013);
                if !loco_coupled {
                    if mean_p >= 0.02 {
                        canvas_err_sum += pore_err1[p];
                        canvas_cnt += 1;
                    }
                    for m in 0..TRACKED_ACTUATORS {
                        let pwa = self.pore_action_w[m][p].abs();
                        if pwa >= 0.018 {
                            som_inst[m] += pwa;
                        }
                    }
                }
            }
        }

        if has_echo_pore && committed_15 >= 8 {
            let inv_c15 = 1.0 / (committed_15 as f32);
            let echo_surp = sanitize_clamp(max_echo_exc, -0.15, 0.45);
            for m in 0..TRACKED_ACTUATORS {
                if loco_auth[m] >= 0.22 * max_loco_auth || Some(m) == mg_opt {
                    self.echo_emit_w[m] = sanitize(0.992 * self.echo_emit_w[m]);
                } else {
                    let mean_15 = sum_abs15[m] * inv_c15;
                    let lead_p = sanitize_clamp(max_2_6[m] - mean_15, -0.5, 0.8);
                    let dec = if max_2_6[m] > 0.08 || mean_15 > 0.08 {
                        0.9980
                    } else {
                        0.9996
                    };
                    self.echo_emit_w[m] = sanitize_clamp(
                        dec * self.echo_emit_w[m] + 0.048 * echo_surp * lead_p,
                        -0.4,
                        1.0,
                    );
                }
            }
        }

        if let Some(mg) = mg_opt
            && coch_cnt >= 3
            && committed_15 >= 10
            && let (Some(e1), Some(e2), Some(e3)) = (ring.at_lag(1), ring.at_lag(2), ring.at_lag(3))
            && e1.committed
            && e2.committed
            && e3.committed
            && e1.motor_out[mg] > 0.12
            && e2.motor_out[mg] > 0.12
            && e3.motor_out[mg] > 0.12
            && (e1.motor_out[mg] - e2.motor_out[mg]).abs() < 0.10
            && (e2.motor_out[mg] - e3.motor_out[mg]).abs() < 0.10
        {
            let c_flux_surp = sanitize_clamp(
                (coch_dx1_sum - coch_dx_prev_sum) / (coch_cnt as f32),
                -0.25,
                0.35,
            );
            for m in 0..TRACKED_ACTUATORS {
                if loco_auth[m] >= 0.22 * max_loco_auth || m == mg {
                    self.vocal_artic_w[m] = sanitize(0.992 * self.vocal_artic_w[m]);
                } else {
                    let step_now = (e1.motor_out[m] - e2.motor_out[m]).abs();
                    let step_prev = (e2.motor_out[m] - e3.motor_out[m]).abs();
                    let step_surp = sanitize_clamp(step_now - step_prev, -0.6, 0.8);
                    self.vocal_artic_w[m] = sanitize_clamp(
                        0.9985 * self.vocal_artic_w[m] + 0.085 * c_flux_surp * step_surp,
                        -0.4,
                        1.0,
                    );
                }
            }
        }

        let canvas_err_mean = if canvas_cnt >= 2 {
            sanitize_clamp(canvas_err_sum / (canvas_cnt as f32), -0.60, 0.60)
        } else {
            0.0
        };
        let lag2_ent = ring.at_lag(2).filter(|e| e.committed);
        for (m, &is_loco_c) in is_world_loco_cal.iter().enumerate() {
            if is_loco_c
                || loco_auth[m] >= 0.06 * max_loco_auth
                || Some(m) == mg_opt
                || (self.vocal_artic_w[m] >= 0.008
                    && coch_inst[m] > som_inst[m]
                    && self.somatic_limb_w[m] < 0.08)
                || self.echo_emit_w[m] >= 0.010
            {
                self.somatic_limb_w[m] = sanitize(0.980 * self.somatic_limb_w[m]);
            } else {
                let inst_drive = if som_inst[m] >= 0.022 && som_inst[m] >= coch_inst[m] {
                    som_inst[m].min(0.60)
                } else {
                    0.0
                };
                let mut w_som = 0.99982 * self.somatic_limb_w[m] + 0.0014 * inst_drive;
                if canvas_cnt >= 2
                    && committed_15 >= 8
                    && let Some(e2) = lag2_ent
                {
                    let da2_m = sanitize_clamp(e2.motor_out[m] - self.act_mean[m], -1.0, 1.0);
                    w_som += 0.0020 * canvas_err_mean * da2_m;
                }
                self.somatic_limb_w[m] = sanitize_clamp(w_som, -0.20, 1.0);
            }
        }

        let mut emit_opt: Option<usize> = None;
        let mut best_ew = 0.010_f32;
        for (m, &ew) in self.echo_emit_w.iter().enumerate() {
            if loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt && ew > best_ew {
                best_ew = ew;
                emit_opt = Some(m);
            }
        }
        let mut quiet_echo_before_6 = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut last_emit_pulse = 0.0_f32;
        if quiet_echo_before_6 {
            for lag in 8..=16 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..TRACKED_ACTUATORS {
                        let is_emit_cand = if let Some(em) = emit_opt {
                            m == em
                        } else {
                            loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt
                        };
                        let am = ent.motor_out[m];
                        if is_emit_cand && am.abs() > 0.12 {
                            quiet_echo_before_6 = false;
                            if (8..=13).contains(&lag) && am.abs() > last_emit_pulse.abs() {
                                last_emit_pulse = am;
                            }
                        }
                    }
                }
            }
        }

        let max_echo_raw_now = top_echo_cal_0.hypot(top_echo_cal_1);
        let mut quiet_echo_cal_14 = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        if quiet_echo_cal_14 {
            for lag in 1..=14 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..TRACKED_ACTUATORS {
                        let is_emit_cand = if let Some(em) = emit_opt {
                            m == em
                        } else {
                            loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt
                        };
                        if is_emit_cand && ent.motor_out[m].abs() > 0.12 {
                            quiet_echo_cal_14 = false;
                        }
                    }
                }
            }
        }
        let span_e = (self.peer_echo_age as usize).saturating_add(1);
        let d_echo_cal = if quiet_echo_cal_14
            && max_echo_raw_now > 0.018
            && self.peer_echo_fast > 0.016
            && (3..=34).contains(&span_e)
        {
            (max_echo_raw_now - self.peer_echo_fast).clamp(-0.22, 0.22)
        } else {
            0.0
        };
        let ring_warm_cal = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut q_cal_lag = [ring_warm_cal; 17];
        if let Some(mg) = mg_opt
            && ring_warm_cal
        {
            for (d, ql) in q_cal_lag.iter_mut().enumerate().skip(1) {
                *ql = ring
                    .at_lag(d)
                    .is_some_and(|e| !e.committed || e.motor_out[mg] <= -0.05);
            }
        }
        let quiet_self_cal_7 = ring_warm_cal
            && q_cal_lag[1]
            && q_cal_lag[2]
            && q_cal_lag[3]
            && q_cal_lag[4]
            && q_cal_lag[5]
            && q_cal_lag[6]
            && q_cal_lag[7];

        let peer_call_at_lag = |stim_lag: usize| -> (Option<f32>, f32) {
            let mut top4_cal = [0.0_f32; 4];
            let mut seen_coch_ch = [u32::MAX; PRED_BUCKETS];
            let mut seen_coch_n = 0usize;
            let mut coch_strong_cnt = 0usize;
            let mut coch_weak_cnt = 0usize;
            let mut cnt_c = 0usize;

            let mut top4_lso_pos = [0.0_f32; 4];
            let mut top4_lso_neg = [0.0_f32; 4];
            let mut top4_clean_pos = [0.0_f32; 4];
            let mut top4_clean_neg = [0.0_f32; 4];
            let mut lso_pos_cnt = 0usize;
            let mut lso_neg_cnt = 0usize;
            let mut clean_pos_cnt = 0usize;
            let mut clean_neg_cnt = 0usize;
            let mut lso_signed_sum = 0.0_f32;
            let mut seen_lso_ch = [u32::MAX; PRED_BUCKETS];
            let mut seen_lso_n = 0usize;

            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                    || is_world_ret_cal[p]
                {
                    continue;
                }
                let mean_p = self.pore_mean_v[p];
                let is_monaural_coch = if let Some(mg) = mg_opt {
                    self.pore_action_w[mg][p] >= 0.016 && mean_p >= -0.342
                } else {
                    self.pore_agency[p] >= 0.010
                        && (-0.38..=-0.16).contains(&min_h_cal[p])
                        && (-0.345..=-0.04).contains(&mean_p)
                };
                let h = sp.hist_head as usize;
                if is_monaural_coch {
                    cnt_c += 1;
                    if seen_coch_ch[..seen_coch_n].contains(&sp.channel) {
                        continue;
                    }
                    seen_coch_ch[seen_coch_n] = sp.channel;
                    seen_coch_n += 1;
                    let u0 = Self::deconv_hair_cell_35(&sp.history, h, stim_lag).max(0.0);
                    let u1 = Self::deconv_hair_cell_35(&sp.history, h, stim_lag + 1).max(0.0);
                    let u2 = Self::deconv_hair_cell_35(&sp.history, h, stim_lag + 2).max(0.0);
                    let u_sus3 = if u0 > 0.0 && u1 > 0.0 && u2 > 0.0 {
                        u0.min(1.30 * u1 + 0.004).min(1.65 * u2 + 0.006)
                    } else {
                        0.0
                    };
                    if u_sus3 > 0.038 {
                        coch_strong_cnt += 1;
                    }
                    if u_sus3 > 0.016 {
                        coch_weak_cnt += 1;
                    }
                    Self::insert_top4(&mut top4_cal, u_sus3);
                    continue;
                }
                if sp.age_ticks < 40 {
                    continue;
                }
                let mut somatic_other_w = 0.0_f32;
                for m in 0..TRACKED_ACTUATORS {
                    if loco_auth[m] < 0.22 * max_loco_auth
                        && Some(m) != mg_opt
                        && self.vocal_artic_w[m] < 0.005
                    {
                        somatic_other_w += self.pore_action_w[m][p].abs();
                    }
                }
                let is_coch_trough = (-0.36..=-0.33).contains(&mean_p)
                    && mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                let is_glottal_pos = mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
                if is_coch_trough
                    || is_glottal_pos
                    || min_h_cal[p] > -0.20
                    || mean_p >= -0.18
                    || somatic_other_w >= 0.038
                    || seen_lso_ch[..seen_lso_n].contains(&sp.channel)
                {
                    continue;
                }
                seen_lso_ch[seen_lso_n] = sp.channel;
                seen_lso_n += 1;
                let bg = self.pore_bg_ild[p];
                let d0 = Self::sym_deconv_ild(&sp.history, h, stim_lag) - bg;
                let d1 = Self::sym_deconv_ild(&sp.history, h, stim_lag + 1) - bg;
                let d2 = Self::sym_deconv_ild(&sp.history, h, stim_lag + 2) - bg;
                let d3 = Self::sym_deconv_ild(&sp.history, h, stim_lag + 3) - bg;
                if d0.abs() > 0.015 {
                    lso_signed_sum += d0;
                }
                if d0 * d1 > 0.0 && d1 * d2 > 0.0 && d2 * d3 > 0.0 {
                    let sus_d = d0.abs().min(d1.abs()).min(d2.abs()).min(d3.abs());
                    if sus_d > 0.018 {
                        if d0 > 0.0 {
                            lso_pos_cnt += 1;
                            Self::insert_top4(&mut top4_lso_pos, sus_d);
                        } else {
                            lso_neg_cnt += 1;
                            Self::insert_top4(&mut top4_lso_neg, sus_d);
                        }
                    }
                    if sp.age_ticks >= 48 && bg.abs() < 0.018 && sus_d > 0.022 {
                        if d0 > 0.0 {
                            clean_pos_cnt += 1;
                            Self::insert_top4(&mut top4_clean_pos, sus_d);
                        } else {
                            clean_neg_cnt += 1;
                            Self::insert_top4(&mut top4_clean_neg, sus_d);
                        }
                    }
                }
            }
            let (dom_sign, lso_coh_cnt, top4_lso_dom, lso_clean_cnt, top4_lso_clean) =
                if lso_pos_cnt > lso_neg_cnt {
                    (
                        1.0_f32,
                        lso_pos_cnt,
                        top4_lso_pos,
                        clean_pos_cnt,
                        top4_clean_pos,
                    )
                } else if lso_neg_cnt > lso_pos_cnt {
                    (
                        -1.0_f32,
                        lso_neg_cnt,
                        top4_lso_neg,
                        clean_neg_cnt,
                        top4_clean_neg,
                    )
                } else if lso_signed_sum >= 0.0 {
                    (
                        1.0_f32,
                        lso_pos_cnt,
                        top4_lso_pos,
                        clean_pos_cnt,
                        top4_clean_pos,
                    )
                } else {
                    (
                        -1.0_f32,
                        lso_neg_cnt,
                        top4_lso_neg,
                        clean_neg_cnt,
                        top4_clean_neg,
                    )
                };
            let harm_opt = if cnt_c >= 2 {
                let lvl = 0.5 * (top4_cal[0] + top4_cal[1]);
                let ok = (coch_strong_cnt >= 2
                    && lvl > 0.040
                    && lso_coh_cnt >= 3
                    && top4_lso_dom[2] > 0.018)
                    || (coch_strong_cnt >= 1
                        && top4_cal[0] > 0.058
                        && lso_coh_cnt >= 4
                        && top4_lso_dom[3] > 0.020)
                    || (coch_weak_cnt >= 2
                        && lvl > 0.018
                        && lso_clean_cnt >= 5
                        && top4_lso_clean[3] > 0.022);
                if ok { Some(lvl) } else { None }
            } else {
                None
            };
            (harm_opt, dom_sign)
        };

        let span_c = (self.peer_coch_age as usize).saturating_add(1);
        let (peer_lvl_cal_opt, dom_sign_cal) =
            if quiet_self_cal_7 && self.peer_coch_fast > 0.016 && (3..=15).contains(&span_c) {
                peer_call_at_lag(0)
            } else {
                (None, 0.0)
            };
        let d_coch_cal =
            peer_lvl_cal_opt.map_or(0.0, |lvl| (lvl - self.peer_coch_fast).clamp(-0.22, 0.22));

        let mda_over_span = |span: usize| -> ([f32; TRACKED_ACTUATORS], usize, f32) {
            let mut mda = [0.0_f32; TRACKED_ACTUATORS];
            let mut cnt = 0usize;
            let s_clamped = span.clamp(1, 15);
            let inv_s = 1.0 / (s_clamped as f32);
            for lag in 1..=s_clamped {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    cnt += 1;
                    for m in 0..TRACKED_ACTUATORS {
                        mda[m] = sanitize(mda[m] + (ent.motor_out[m] - self.act_mean[m]) * inv_s);
                    }
                }
            }
            let dp = if spatial_prox_cnt > 0 {
                let past_p =
                    self.spatial_prox_ring[(prox_slot + RING_LEN - s_clamped) & (RING_LEN - 1)];
                spatial_prox_now - past_p
            } else {
                0.0
            };
            (mda, cnt, dp)
        };

        if d_coch_cal.abs() > 0.0015 {
            let (cal_mda_c, cnt_mc, d_prox_c) = mda_over_span(span_c);
            let app_sig_c = sanitize_clamp(
                6.2 * d_coch_cal
                    + if d_coch_cal * d_prox_c > 0.0 {
                        2.0 * d_prox_c
                    } else {
                        0.0
                    },
                -1.2,
                1.2,
            );
            if cnt_mc >= 3 && app_sig_c.abs() >= 0.008 {
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 40
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || is_world_ret_cal[p]
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                    {
                        continue;
                    }
                    let mean_p = self.pore_mean_v[p];
                    let is_monaural_coch = mg_opt
                        .is_some_and(|mg| self.pore_action_w[mg][p] >= 0.016 && mean_p >= -0.342);
                    let is_coch_trough = (-0.36..=-0.33).contains(&mean_p)
                        && mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                    let mut somatic_other_w = 0.0_f32;
                    for m in 0..TRACKED_ACTUATORS {
                        if loco_auth[m] < 0.22 * max_loco_auth
                            && Some(m) != mg_opt
                            && self.vocal_artic_w[m] < 0.005
                        {
                            somatic_other_w += self.pore_action_w[m][p].abs();
                        }
                    }
                    if is_monaural_coch
                        || is_coch_trough
                        || mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008)
                        || min_h_cal[p] > -0.20
                        || mean_p >= -0.18
                        || somatic_other_w >= 0.038
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let bg = self.pore_bg_ild[p];
                    let d_now = Self::sym_deconv_ild(&sp.history, h, 0) - bg;
                    let d_past = if span_c <= 13 {
                        Self::sym_deconv_ild(&sp.history, h, span_c) - bg
                    } else if q_cal_lag[span_c.min(16)] {
                        sp.history[(h + 15 - span_c.min(15)) & 15] - (-0.35 + bg)
                    } else {
                        0.0
                    };
                    if d_now * dom_sign_cal <= 0.018
                        || (d_past.abs() > 0.012 && d_now * d_past <= 0.0)
                    {
                        continue;
                    }
                    let d_mid = if d_past.abs() > 0.012 {
                        0.5 * (d_now + d_past)
                    } else {
                        d_now
                    };
                    let s_lso = sanitize_clamp(6.5 * d_mid, -2.5, 2.5);
                    if s_lso.abs() < 0.14 {
                        continue;
                    }
                    for (m, &mda) in cal_mda_c.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9994 * self.tectal_map_w[m][p] + 0.022 * app_sig_c * s_lso * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
                }
            }
        }

        if d_echo_cal.abs() > 0.0015 {
            let (cal_mda_e, cnt_me, d_prox_e) = mda_over_span(span_e);
            let app_sig_e = sanitize_clamp(
                5.4 * d_echo_cal
                    + if d_echo_cal * d_prox_e > 0.0 {
                        2.0 * d_prox_e
                    } else {
                        0.0
                    },
                -1.2,
                1.2,
            );
            if cnt_me >= 3 && app_sig_e.abs() >= 0.008 {
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || !membrane.scout_stats[ch_u].is_sparse_impulse()
                        || self.pore_agency[p] >= 0.035
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let mean_p = self.pore_mean_v[p];
                    let scale_p = self.pore_scale_v[p].max(0.015);
                    let dev_now = (sp.history[(h + 15) & 15] - mean_p).max(0.0);
                    let dev_lag1 = (sp.history[(h + 14) & 15] - mean_p).max(0.0);
                    let dev_use = dev_now.max(0.75 * dev_lag1);
                    if dev_use <= 0.018 {
                        continue;
                    }
                    let s_echo = sanitize_clamp(dev_use / scale_p, 0.0, 2.5);
                    if s_echo < 0.14 {
                        continue;
                    }
                    for (m, &mda) in cal_mda_e.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9994 * self.tectal_map_w[m][p] + 0.026 * app_sig_e * s_echo * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
                }
            }
        } else if !quiet_echo_before_6 && last_emit_pulse > 0.12 {
            let (cal_mda_s, cnt_ms, d_prox_s) = mda_over_span(10);
            if cnt_ms >= 4 && d_prox_s > 0.008 {
                let app_sig_s = sanitize_clamp(3.2 * d_prox_s, 0.0, 0.85);
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || !membrane.scout_stats[ch_u].is_sparse_impulse()
                        || self.pore_agency[p] >= 0.035
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let mean_p = self.pore_mean_v[p];
                    let scale_p = self.pore_scale_v[p].max(0.015);
                    let mut peak_dev = 0.0_f32;
                    for stim_lag in 6..=11 {
                        let dv = (sp.history[(h + 15 - stim_lag) & 15] - mean_p).max(0.0);
                        if dv > peak_dev {
                            peak_dev = dv;
                        }
                    }
                    if peak_dev <= 0.022 {
                        continue;
                    }
                    let s_echo = sanitize_clamp(peak_dev / scale_p, 0.0, 2.5);
                    for (m, &mda) in cal_mda_s.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9995 * self.tectal_map_w[m][p] + 0.014 * app_sig_s * s_echo * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
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

    pub fn apply_delayed_credit_and_morphogenesis(
        &mut self,
        ring: &EfferenceRing,
        cells: &mut CellArena,
        membrane: &mut PoreMembrane,
        body: &BodyCore,
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

        let scale = body.valence_scale_ema.max(1.5e-4);
        let raw_somatic_valence = body.fast_valence;
        let ctx_span = body.context_valence_span();

        let mut embodied_m_glottal: Option<usize> = None;
        let mut is_monaural_cr = [false; PRED_BUCKETS];
        let mut is_lso_cr = [false; PRED_BUCKETS];
        if membrane.alive_motor_count > 8 && body.rest_consolidation_ticks >= 480 {
            let pore_cap_cr = PRED_BUCKETS.min(membrane.sensor_pores.len());
            let n_scout_cr = membrane.scout_stats.len();
            let act_cap_cr = membrane.motor_pores.len().min(TRACKED_ACTUATORS);
            let mut field_breadth_cr = [0u16; TRACKED_ACTUATORS];
            for p in 0..pore_cap_cr {
                let sp = &membrane.sensor_pores[p];
                if sp.alive
                    && sp.age_ticks >= 16
                    && sp.trophic_ema >= 0.025
                    && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
                {
                    for (m, fb) in field_breadth_cr.iter_mut().enumerate().take(act_cap_cr) {
                        if self.pore_action_w[m][p].abs() >= 0.022 {
                            *fb = fb.saturating_add(1);
                        }
                    }
                }
            }
            let mut max_breadth_cr = 0u16;
            for &fb in &field_breadth_cr[..act_cap_cr] {
                if fb > max_breadth_cr {
                    max_breadth_cr = fb;
                }
            }
            let mut is_world_loco_cr = [false; TRACKED_ACTUATORS];
            if max_breadth_cr >= 4 {
                for m in 0..act_cap_cr {
                    if field_breadth_cr[m].saturating_mul(2) >= max_breadth_cr {
                        is_world_loco_cr[m] = true;
                    }
                }
            }
            let mut is_world_retina_cr = [false; PRED_BUCKETS];
            for (p, ret_flag) in is_world_retina_cr.iter_mut().enumerate().take(pore_cap_cr) {
                let sp = &membrane.sensor_pores[p];
                if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                    continue;
                }
                if max_breadth_cr < 4 {
                    *ret_flag = true;
                } else {
                    let mut loco_w_p = 0.0_f32;
                    let mut nonloco_w_p = 0.0_f32;
                    for m in 0..act_cap_cr {
                        let wa = self.pore_action_w[m][p].abs();
                        if is_world_loco_cr[m] {
                            if wa > loco_w_p {
                                loco_w_p = wa;
                            }
                        } else if wa > nonloco_w_p {
                            nonloco_w_p = wa;
                        }
                    }
                    if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                        *ret_flag = true;
                    }
                }
            }
            let mut is_glottal_cand_cr = [false; PRED_BUCKETS];
            for (p, gc_flag) in is_glottal_cand_cr.iter_mut().enumerate().take(pore_cap_cr) {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout_cr
                    || is_world_retina_cr[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                let mut min_h = 1.0_f32;
                for &hv in &sp.history {
                    if hv < min_h {
                        min_h = hv;
                    }
                }
                if self.pore_agency[p] >= 0.006
                    && (-0.38..=-0.16).contains(&min_h)
                    && (-0.345..=-0.04).contains(&self.pore_mean_v[p])
                {
                    *gc_flag = true;
                }
            }
            let mut max_asw_cr = [0.0_f32; TRACKED_ACTUATORS];
            let mut tot_reaff_cr = [0.0_f32; TRACKED_ACTUATORS];
            let mut pos_coch_w_cr = [0.0_f32; TRACKED_ACTUATORS];
            let mut pos_coch_cnt_cr = [0u16; TRACKED_ACTUATORS];
            let mut soft_coch_w_cr = [0.0_f32; TRACKED_ACTUATORS];
            let mut soft_coch_cnt_cr = [0u16; TRACKED_ACTUATORS];
            let mut is_pure_coch_cr = [false; PRED_BUCKETS];
            let mut is_quiet_coch_cr = [false; PRED_BUCKETS];
            let mut min_h_cr = [1.0_f32; PRED_BUCKETS];
            for p in 0..pore_cap_cr {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout_cr
                    || is_world_retina_cr[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                let mut min_h = 1.0_f32;
                for &hv in &sp.history {
                    if hv < min_h {
                        min_h = hv;
                    }
                }
                min_h_cr[p] = min_h;
                if (-0.356..=-0.16).contains(&min_h) {
                    if (-0.349..=-0.05).contains(&self.pore_mean_v[p]) {
                        is_pure_coch_cr[p] = true;
                    }
                    if (-0.354..=-0.05).contains(&self.pore_mean_v[p]) {
                        is_quiet_coch_cr[p] = true;
                    }
                }
            }
            for m in 0..act_cap_cr {
                for &w in &self.action_sensory_w[m] {
                    let wa = w.abs();
                    if wa > max_asw_cr[m] {
                        max_asw_cr[m] = wa;
                    }
                }
                for p in 0..pore_cap_cr {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout_cr
                    {
                        continue;
                    }
                    let pw = self.pore_action_w[m][p];
                    let pwa = pw.abs();
                    if self.pore_agency[p] >= 0.010 && pwa >= 0.015 {
                        tot_reaff_cr[m] += pwa;
                    }
                    if is_glottal_cand_cr[p] && pw >= 0.015 {
                        pos_coch_w_cr[m] += pw;
                        pos_coch_cnt_cr[m] = pos_coch_cnt_cr[m].saturating_add(1);
                    }
                    if (is_pure_coch_cr[p] && pw >= 0.0045) || (is_quiet_coch_cr[p] && pw >= 0.0120)
                    {
                        soft_coch_w_cr[m] += pw;
                        soft_coch_cnt_cr[m] = soft_coch_cnt_cr[m].saturating_add(1);
                    }
                }
            }
            let mut best_gw = 0.05_f32;
            for m in 0..act_cap_cr {
                if !is_world_loco_cr[m]
                    && (tot_reaff_cr[m] >= 0.045 || max_asw_cr[m] >= 0.065)
                    && pos_coch_cnt_cr[m] >= 2
                    && pos_coch_w_cr[m] > best_gw
                {
                    best_gw = pos_coch_w_cr[m];
                    embodied_m_glottal = Some(m);
                }
            }
            if embodied_m_glottal.is_none() {
                let mut best_myelin = 0.14_f32;
                for m in 0..act_cap_cr {
                    if is_world_loco_cr[m] || soft_coch_w_cr[m] < 0.008 {
                        continue;
                    }
                    let mut es = 0.0_f32;
                    let mut ec = 0usize;
                    let mut os = 0.0_f32;
                    let mut oc = 0usize;
                    for tube in &self.slots {
                        if tube.alive
                            && tube.kind == TubeKind::CellToMotor
                            && membrane
                                .motor_pores
                                .get(tube.post as usize)
                                .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == m)
                        {
                            if (tube.pre & 1) == 0 {
                                es += tube.myelin_plasticity_accum;
                                ec += 1;
                            } else {
                                os += tube.myelin_plasticity_accum;
                                oc += 1;
                            }
                        }
                    }
                    let em = if ec >= 3 { es / (ec as f32) } else { 0.0 };
                    let om = if oc >= 3 { os / (oc as f32) } else { 0.0 };
                    let peak_m = em.max(om);
                    if peak_m > best_myelin {
                        best_myelin = peak_m;
                        embodied_m_glottal = Some(m);
                    }
                }
            }
            if embodied_m_glottal.is_none() {
                let mut best_sw = 0.018_f32;
                for m in 0..act_cap_cr {
                    if !is_world_loco_cr[m]
                        && (tot_reaff_cr[m] >= 0.020
                            || max_asw_cr[m] >= 0.038
                            || (soft_coch_cnt_cr[m] >= 4 && soft_coch_w_cr[m] >= 0.048))
                        && soft_coch_cnt_cr[m] >= 2
                        && soft_coch_w_cr[m] > best_sw
                    {
                        best_sw = soft_coch_w_cr[m];
                        embodied_m_glottal = Some(m);
                    }
                }
            }
            if let Some(mg) = embodied_m_glottal {
                for p in 0..pore_cap_cr {
                    is_monaural_cr[p] = (is_pure_coch_cr[p] && self.pore_action_w[mg][p] >= 0.0050)
                        || (is_quiet_coch_cr[p] && self.pore_action_w[mg][p] >= 0.0120);
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 40
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout_cr
                        || is_world_retina_cr[p]
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                        || is_monaural_cr[p]
                    {
                        continue;
                    }
                    let mean_p = self.pore_mean_v[p];
                    let is_coch_trough =
                        (-0.36..=-0.33).contains(&mean_p) && self.pore_action_w[mg][p] < -0.016;
                    let is_glottal_pos = self.pore_action_w[mg][p] >= 0.008;
                    if is_coch_trough || is_glottal_pos || min_h_cr[p] > -0.20 || mean_p >= -0.18 {
                        continue;
                    }
                    let mut somatic_other_w = 0.0_f32;
                    for m in 0..act_cap_cr {
                        if !is_world_loco_cr[m] && m != mg && self.vocal_artic_w[m] < 0.005 {
                            somatic_other_w += self.pore_action_w[m][p].abs();
                        }
                    }
                    if somatic_other_w < 0.038 {
                        is_lso_cr[p] = true;
                    }
                }
            }
        }

        for lag in 1..=RING_LEN {
            let Some(entry) = ring.at_lag(lag) else {
                continue;
            };
            if entry.tick == 0 {
                continue;
            }
            let kernel = (-((lag.saturating_sub(1)) as f32) / 2.2).exp();
            let raw_diff = sanitize(raw_somatic_valence - entry.raw_valence_at_commit);
            let post_harvest_refractory = (raw_somatic_valence >= 0.0
                && entry.raw_valence_at_commit > 0.12
                && raw_diff < -0.08)
                || entry.tick == 0;
            let interval_delta = if post_harvest_refractory {
                0.0
            } else {
                raw_diff
            };
            let local_level =
                0.25 * (raw_somatic_valence.abs() + entry.raw_valence_at_commit.abs()) + 1.0e-4;
            let base_level = if interval_delta > 0.0 && entry.raw_valence_at_commit < 0.0 {
                (0.22 * ctx_span).max(0.15 * scale)
            } else {
                local_level.max(0.18 * ctx_span).max(0.15 * scale)
            };
            let denom = interval_delta.abs() + base_level;
            let weber_diff = interval_delta / denom;
            let pain_term = if raw_somatic_valence < 0.0 && raw_diff < 0.0 {
                2.0 * raw_diff
            } else {
                0.0
            };
            let td_adv = sanitize_clamp(1.75 * weber_diff + pain_term, -2.0, 2.0);

            if raw_somatic_valence.abs() > 1.0e-4 || td_adv.abs() > 0.02 {
                for &(c_idx, v_past, dv_past, pre_attn) in &entry.active_cells {
                    let Some(cell) = cells.slots.get_mut(c_idx as usize) else {
                        continue;
                    };
                    if !cell.alive || cell.sensory_trophic_ema < 0.005 {
                        continue;
                    }
                    let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                    let lr_v = 0.04 * kernel * attn;
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
                            cell.atp_credit_ema
                                + 0.05 * kernel * attn * raw_somatic_valence * v_past.abs(),
                            0.0,
                            4.0,
                        );
                    }
                }
            }

            let lag_u8 = lag as u8;
            let seq_kernel = (-((lag.saturating_sub(1)) as f32) / 5.5).exp();
            let check_myelin = (entry.tick & 7) == 0;
            let faster_entry = (check_myelin && lag > 1)
                .then(|| ring.at_lag(lag - 1))
                .flatten();
            let slower_entry = (check_myelin && lag < RING_LEN)
                .then(|| ring.at_lag(lag + 1))
                .flatten();
            let mut myelin_shift_requests: Vec<(u32, u8)> = Vec::new();

            // Multi-syllable / long-horizon polychronous CellToCell sequence learning across 100..320 ms (lag 5..=16),
            // leaving fast sensorimotor lags 1..=4 to the committed cortical-motor loop below.
            if lag >= 5 {
                for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
                    let pre_sig = sanitize(v_past + 0.25 * dv_past);
                    if pre_sig.abs() < 1.0e-3 {
                        continue;
                    }
                    let v_faster = faster_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                    let v_slower = slower_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                    let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize).cloned() else {
                        continue;
                    };
                    for t_idx in out_list {
                        let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                            continue;
                        };
                        if !tube.alive || tube.kind != TubeKind::CellToCell || tube.delay != lag_u8
                        {
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
                        if check_myelin
                            && tube.usage_ema > 0.025
                            && (tube.u[0].abs() + tube.w_pred.abs()) > 0.055
                        {
                            let pol = if tube.u[0].abs() >= 0.02 {
                                tube.u[0].signum()
                            } else {
                                tube.w_pred.signum()
                            };
                            let c_curr = pol * pre_sig * post_now;
                            let c_fast = pol * v_faster * post_now;
                            let c_slow = pol * v_slower * post_now;
                            if c_fast > c_curr + 0.015 || c_slow > c_curr + 0.015 {
                                let d_mu = 0.22 * tube.usage_ema.min(1.0) * (c_fast - c_slow);
                                tube.myelin_plasticity_accum = sanitize_clamp(
                                    0.95 * tube.myelin_plasticity_accum + d_mu,
                                    -1.5,
                                    1.5,
                                );
                                if tube.myelin_plasticity_accum >= 1.0 && tube.delay > 1 {
                                    myelin_shift_requests.push((t_idx, tube.delay - 1));
                                    tube.myelin_plasticity_accum = 0.0;
                                } else if tube.myelin_plasticity_accum <= -1.0
                                    && (tube.delay as usize) < RING_LEN
                                {
                                    myelin_shift_requests.push((t_idx, tube.delay + 1));
                                    tube.myelin_plasticity_accum = 0.0;
                                }
                            } else {
                                tube.myelin_plasticity_accum =
                                    sanitize(0.98 * tube.myelin_plasticity_accum);
                            }
                        }
                    }
                }
                for (t_idx, new_d) in myelin_shift_requests.drain(..) {
                    let (pre, post, old_d) = {
                        let t = &self.slots[t_idx as usize];
                        (t.pre, t.post, t.delay)
                    };
                    if new_d != old_d && !self.has_cell_tube_at_delay(pre, post, new_d) {
                        self.slots[t_idx as usize].delay = new_d;
                        if new_d < old_d {
                            shortcuts += 1;
                        }
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
                        let (up_cell, up_v, _, _) = entry.active_cells[(rot + off) % n_act];
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
            for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
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

            for &(c_idx, v_past, dv_past, pre_attn) in &entry.active_cells {
                let pre_sig = sanitize(v_past + 0.25 * dv_past);
                if pre_sig.abs() < 1.0e-3 {
                    continue;
                }
                let v_faster = faster_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                let v_slower = slower_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
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
                            let pred_err =
                                sanitize_clamp(post_now - tube.w_pred * pre_sig, -2.0, 2.0);
                            tube.w_pred =
                                sanitize_clamp(tube.w_pred + 0.035 * pred_err * pre_sig, -2.0, 2.0);
                            let du = 0.02 * kernel * attn * td_adv * pre_sig * post_now
                                - 0.004 * (td_adv.abs() + 0.02) * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                            tube.usage_ema = sanitize_clamp(
                                0.98 * tube.usage_ema + 0.02 * (pre_sig * post_now).abs(),
                                0.0,
                                2.0,
                            );
                            if check_myelin
                                && tube.usage_ema > 0.025
                                && (tube.u[0].abs() + tube.w_pred.abs()) > 0.055
                            {
                                let pol = if tube.u[0].abs() >= 0.02 {
                                    tube.u[0].signum()
                                } else {
                                    tube.w_pred.signum()
                                };
                                let c_curr = pol * pre_sig * post_now;
                                let c_fast = pol * v_faster * post_now;
                                let c_slow = pol * v_slower * post_now;
                                if c_fast > c_curr + 0.015 || c_slow > c_curr + 0.015 {
                                    let d_mu = 0.22 * tube.usage_ema.min(1.0) * (c_fast - c_slow);
                                    tube.myelin_plasticity_accum = sanitize_clamp(
                                        0.95 * tube.myelin_plasticity_accum + d_mu,
                                        -1.5,
                                        1.5,
                                    );
                                    if tube.myelin_plasticity_accum >= 1.0 && tube.delay > 1 {
                                        myelin_shift_requests.push((t_idx, tube.delay - 1));
                                        tube.myelin_plasticity_accum = 0.0;
                                    } else if tube.myelin_plasticity_accum <= -1.0
                                        && (tube.delay as usize) < RING_LEN
                                    {
                                        myelin_shift_requests.push((t_idx, tube.delay + 1));
                                        tube.myelin_plasticity_accum = 0.0;
                                    }
                                } else {
                                    tube.myelin_plasticity_accum =
                                        sanitize(0.98 * tube.myelin_plasticity_accum);
                                }
                            }
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
                                mp_mut.conductance = sanitize_clamp(
                                    0.994 * mp_mut.conductance + 0.006 * 0.95,
                                    0.1,
                                    1.5,
                                );
                            }

                            let elig = pre_sig * a_emitted;
                            let eff_td_adv = if Some(act_ch) == embodied_m_glottal
                                && a_emitted > 0.05
                                && tube.myelin_plasticity_accum >= 0.18
                                && (entry.raw_valence_at_commit - body.baseline_valence_ema) > 0.015
                                && (raw_somatic_valence - body.baseline_valence_ema) >= -0.015
                                && raw_diff < 0.0
                            {
                                0.0
                            } else {
                                td_adv
                            };
                            let raw_target = elig * eff_td_adv;
                            let reward_coupled = max_reward_jac[act_ch] > 0.009;
                            let boot_thresh = if reward_coupled { 0.028 } else { 0.12 };
                            let bootstrap =
                                ((tube.u[0].abs() - boot_thresh) / 0.07).clamp(0.0, 0.65);
                            let eff_act = if eff_td_adv >= 0.0 {
                                bootstrap * a_emitted + (1.0 - bootstrap) * a_innov
                            } else {
                                a_emitted
                            };
                            let credit_target = attn * pre_sig * eff_act * eff_td_adv;
                            let g_eff = tube.effective_g(&cells.slots);
                            let stiff = tube.u[2] + tube.u[3];
                            let active_branch =
                                g_eff.abs() >= 0.02 || tube.context_gate_cell.is_none();
                            if stiff * raw_target < -0.025 {
                                if active_branch {
                                    let lability = if eff_td_adv < -0.10 { 0.86 } else { 0.93 };
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
                                tube.contradiction_accum =
                                    sanitize(tube.contradiction_accum * 0.92);
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
                                let oja_decay = 0.065 * (eff_td_adv.abs() + 0.025) * tube.u[0];
                                let lr_tube = if reward_coupled { 0.34 } else { 0.18 };
                                let du = gate_scale * lr_tube * kernel * credit_target - oja_decay;
                                tube.apply_benna_fusi_impulse(du, 1.0);
                                tube.w_pred = sanitize_clamp(
                                    (1.0 - 0.06 * (eff_td_adv.abs() + 0.025)) * tube.w_pred
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

                            if eff_td_adv > 0.02
                                && let Some(prev_e) = ring.at_lag(lag + 1)
                                && let Some(&(up_cell, up_v, _, _)) = prev_e.active_cells.first()
                                && up_cell != c_idx
                                && up_v.abs() > 0.08
                                && cells
                                    .slots
                                    .get(c_idx as usize)
                                    .is_some_and(|cc| cc.surprise_ema < 0.25)
                            {
                                if tube.last_upstream_cell == up_cell {
                                    tube.chain_score =
                                        sanitize_clamp(tube.chain_score + 0.28, 0.0, 2.0);
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

            for (t_idx, new_d) in myelin_shift_requests {
                let (pre, post, old_d) = {
                    let t = &self.slots[t_idx as usize];
                    (t.pre, t.post, t.delay)
                };
                if new_d != old_d && !self.has_cell_tube_at_delay(pre, post, new_d) {
                    self.slots[t_idx as usize].delay = new_d;
                    if new_d < old_d {
                        shortcuts += 1;
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
                    if pre != ctx_cell && !self.has_direct_tube(TubeKind::CellToCell, pre, ctx_cell)
                    {
                        let ctx_drive = Tube::newborn(TubeKind::CellToCell, pre, ctx_cell, 1, 0.42);
                        let _ = self.add_tube(ctx_drive, max_tubes);
                    }
                    if kind == TubeKind::CellToMotor
                        && !self.has_direct_tube(TubeKind::CellToMotor, ctx_cell, post)
                    {
                        let brake = Tube::newborn(TubeKind::CellToMotor, ctx_cell, post, 1, -0.16);
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

        let mut vocal_sensitive_open = body.rest_consolidation_ticks < 9_000;
        if !vocal_sensitive_open && membrane.alive_motor_count > 8 {
            if let Some(mg) = embodied_m_glottal {
                let mut es = 0.0_f32;
                let mut ec = 0usize;
                let mut os = 0.0_f32;
                let mut oc = 0usize;
                for tube in &self.slots {
                    if tube.alive
                        && tube.kind == TubeKind::CellToMotor
                        && membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                    {
                        if (tube.pre & 1) == 0 {
                            es += tube.myelin_plasticity_accum;
                            ec += 1;
                        } else {
                            os += tube.myelin_plasticity_accum;
                            oc += 1;
                        }
                    }
                }
                let em = if ec > 0 { es / (ec as f32) } else { 0.0 };
                let om = if oc > 0 { os / (oc as f32) } else { 0.0 };
                if em >= 0.72 || (om >= 0.72 && em > -0.20) {
                    vocal_sensitive_open = true;
                }
            }
            if !vocal_sensitive_open {
                for tube in &mut self.slots {
                    if tube.alive
                        && tube.kind == TubeKind::CellToMotor
                        && tube.myelin_plasticity_accum != 0.0
                    {
                        let decayed = sanitize(0.99 * tube.myelin_plasticity_accum);
                        tube.myelin_plasticity_accum =
                            if decayed.abs() < 1.0e-3 { 0.0 } else { decayed };
                    }
                }
            }
        }

        if vocal_sensitive_open
            && let Some(mg) = embodied_m_glottal
            && ring.at_lag(16).is_some_and(|e| e.tick > 0)
        {
            let eff_v = Self::laryngeal_voiced_lags(ring, mg);
            let fwd_on = |lg: usize| -> bool { eff_v[lg] && !eff_v[lg + 1] };

            let val_prev1 = ring
                .at_lag(1)
                .map_or(raw_somatic_valence, |e| e.raw_valence_at_commit);
            let val_prev2 = ring
                .at_lag(2)
                .map_or(val_prev1, |e| e.raw_valence_at_commit);
            let d_val_1 = raw_somatic_valence - val_prev1;
            let level_jump = raw_somatic_valence - body.baseline_valence_ema;
            let trophic_burst = if ((level_jump > 0.032 && d_val_1 > 0.042)
                || (raw_somatic_valence > 0.18 && d_val_1 > 0.068))
                && val_prev1 >= -0.025
                && val_prev2 < raw_somatic_valence - 0.042
            {
                sanitize_clamp(9.2 * d_val_1, 0.0, 1.0)
            } else {
                0.0
            };

            if trophic_burst > 0.05 {
                if let Some(lag_on) = (2..=5).find(|&lg| fwd_on(lg)) {
                    let mut even_sum = 0.0_f32;
                    let mut even_cnt = 0usize;
                    for tube in &mut self.slots {
                        if !tube.alive || tube.kind != TubeKind::CellToMotor || (tube.pre & 1) != 0
                        {
                            continue;
                        }
                        if membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                        {
                            tube.myelin_plasticity_accum = sanitize_clamp(
                                tube.myelin_plasticity_accum + 0.105 * trophic_burst,
                                -0.30,
                                1.0,
                            );
                            even_sum += tube.myelin_plasticity_accum;
                            even_cnt += 1;
                        }
                    }
                    let even_mean = if even_cnt > 0 {
                        even_sum / (even_cnt as f32)
                    } else {
                        0.0
                    };
                    if even_mean > 0.14
                        && let Some(on_entry) = ring.at_lag(lag_on)
                    {
                        for &(c_idx, v_past, dv_past, pre_attn) in &on_entry.active_cells {
                            if (c_idx & 1) != 0 {
                                continue;
                            }
                            let pre_sig = sanitize(v_past + 0.25 * dv_past);
                            if pre_sig <= 0.02 {
                                continue;
                            }
                            let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                            let out_len = self
                                .outgoing_by_cell
                                .get(c_idx as usize)
                                .map_or(0, Vec::len);
                            for k in 0..out_len {
                                let t_idx = self.outgoing_by_cell[c_idx as usize][k];
                                let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                                    continue;
                                };
                                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                                    continue;
                                }
                                let is_mg =
                                    membrane.motor_pores.get(tube.post as usize).is_some_and(
                                        |mp| mp.alive && mp.actuator_channel as usize == mg,
                                    );
                                if !is_mg {
                                    continue;
                                }
                                let du = 0.055 * attn * trophic_burst * pre_sig - 0.070 * tube.u[0];
                                tube.apply_benna_fusi_impulse(du, 1.0);
                            }
                        }
                    }
                } else if !(1..=6).any(&fwd_on) {
                    for tube in &mut self.slots {
                        if !tube.alive || tube.kind != TubeKind::CellToMotor || (tube.pre & 1) != 0
                        {
                            continue;
                        }
                        if membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                        {
                            tube.myelin_plasticity_accum = sanitize_clamp(
                                tube.myelin_plasticity_accum - 0.082 * trophic_burst,
                                -0.30,
                                1.0,
                            );
                        }
                    }
                }
            }

            let q4 = (1..=4).all(|d| !eff_v[d]);
            let post_syl_unveil = q4 && eff_v[5];
            let unprompted_bg = q4 && !eff_v[5];
            let unprompted_q6 = (1..=12).all(|d| !eff_v[d]);
            let in_syl_contig =
                eff_v[1] && eff_v[2] && eff_v[3] && eff_v[4] && (4..=7).any(&fwd_on);
            let unveil_lg_on = if post_syl_unveil {
                (5..=15).find(|&lg| fwd_on(lg))
            } else {
                None
            };
            let pre_quiet_lag = unveil_lg_on.filter(|&lg| lg <= 12).and_then(|lg_on| {
                ((lg_on + 1)..=13)
                    .find(|&q| !eff_v[q] && !eff_v[q + 1] && !eff_v[q + 2])
                    .map(|q| q - 1)
            });
            let mut odd_myelin_delta = 0.0_f32;
            let mut odd_silent_miss = false;
            let mut odd_quiet_homeo = false;
            let mut odd_impulse_lag: Option<usize> = None;

            if post_syl_unveil || unprompted_q6 {
                let pore_cap_cr = PRED_BUCKETS.min(membrane.sensor_pores.len());
                let mut seen_cr = [u32::MAX; PRED_BUCKETS];
                let mut seen_cr_n = 0usize;
                let mut c_det = 0usize;
                let mut m_det = 0.0_f32;
                let mut c_pre = 0usize;
                let mut m_pre = 0.0_f32;
                let mut max_u2_all = 0.0_f32;
                let mut max_ub_all = 0.0_f32;
                for (p, &is_mc) in is_monaural_cr.iter().enumerate().take(pore_cap_cr) {
                    if !is_mc {
                        continue;
                    }
                    let sp = &membrane.sensor_pores[p];
                    if !sp.alive || seen_cr[..seen_cr_n].contains(&sp.channel) {
                        continue;
                    }
                    seen_cr[seen_cr_n] = sp.channel;
                    seen_cr_n += 1;
                    let h = sp.hist_head as usize;
                    let u0 = Self::deconv_hair_cell_35(&sp.history, h, 0).max(0.0);
                    let u1 = Self::deconv_hair_cell_35(&sp.history, h, 1).max(0.0);
                    if post_syl_unveil {
                        let u_pre = if let Some(lag_pre) = pre_quiet_lag {
                            let up0 = Self::deconv_hair_cell_35(&sp.history, h, lag_pre).max(0.0);
                            let up1 =
                                Self::deconv_hair_cell_35(&sp.history, h, (lag_pre + 1).min(13))
                                    .max(0.0);
                            let up = up0.max(up1);
                            if up > 0.068 {
                                c_pre += 1;
                            }
                            if up > m_pre {
                                m_pre = up;
                            }
                            up
                        } else {
                            0.0
                        };
                        let d_u0 = (u0 - 0.90 * u_pre).max(0.0);
                        let d_u1 = (u1 - 0.90 * u_pre).max(0.0);
                        let sus2 = d_u0.min(1.25 * d_u1);
                        if sus2 > 0.068 {
                            c_det += 1;
                        }
                        if sus2 > m_det {
                            m_det = sus2;
                        }
                    } else {
                        let u2 = Self::deconv_hair_cell_35(&sp.history, h, 2).max(0.0);
                        let u3 = Self::deconv_hair_cell_35(&sp.history, h, 3).max(0.0);
                        let u4 = Self::deconv_hair_cell_35(&sp.history, h, 4).max(0.0);
                        if u2 > max_u2_all {
                            max_u2_all = u2;
                        }
                        let ub = u3.max(u4);
                        if ub > max_ub_all {
                            max_ub_all = ub;
                        }
                        let sus2 = (u0 - ub).max(0.0).min(1.25 * (u1 - ub).max(0.0));
                        if sus2 > 0.062 {
                            c_det += 1;
                        }
                        if sus2 > m_det {
                            m_det = sus2;
                        }
                    }
                }
                let m_bar = if post_syl_unveil { 0.100 } else { 0.092 };
                let peer_voice_det = seen_cr_n > 0 && (c_det >= 2 || m_det > m_bar);
                let pre_voice_busy = seen_cr_n > 0 && (c_pre >= 2 || m_pre > 0.100);
                if post_syl_unveil {
                    if peer_voice_det {
                        odd_myelin_delta += 0.115;
                        odd_impulse_lag = Some(unveil_lg_on.unwrap_or(15));
                    } else if !pre_voice_busy {
                        odd_silent_miss = true;
                    }
                } else if peer_voice_det && max_ub_all < 0.050 && max_u2_all < 0.068 {
                    odd_myelin_delta -= 0.175;
                } else if !peer_voice_det {
                    odd_quiet_homeo = true;
                }
            }

            if in_syl_contig || unprompted_q6 {
                let pore_cap_cr = PRED_BUCKETS.min(membrane.sensor_pores.len());
                let mut seen_lso = [u32::MAX; PRED_BUCKETS];
                let mut seen_lso_n = 0usize;
                let mut lso_jump_cnt = 0usize;
                let mut max_lso_d = 0.0_f32;
                let mut max_lso_pre = 0.0_f32;
                for (p, &is_lso) in is_lso_cr.iter().enumerate().take(pore_cap_cr) {
                    if !is_lso {
                        continue;
                    }
                    let sp = &membrane.sensor_pores[p];
                    if !sp.alive || seen_lso[..seen_lso_n].contains(&sp.channel) {
                        continue;
                    }
                    seen_lso[seen_lso_n] = sp.channel;
                    seen_lso_n += 1;
                    let h = sp.hist_head as usize;
                    let u0 = Self::sym_deconv_ild(&sp.history, h, 0);
                    let u1 = Self::sym_deconv_ild(&sp.history, h, 1);
                    let u2 = Self::sym_deconv_ild(&sp.history, h, 2);
                    let u3 = Self::sym_deconv_ild(&sp.history, h, 3);
                    let u4 = Self::sym_deconv_ild(&sp.history, h, 4);
                    let pre_mag = u3.abs().max(u4.abs());
                    if pre_mag > max_lso_pre {
                        max_lso_pre = pre_mag;
                    }
                    let u_pre = 0.5 * (u3 + u4);
                    let d0 = u0 - u_pre;
                    let d1 = u1 - u_pre;
                    let d2 = u2 - u_pre;
                    if (u3 - u4).abs() < 0.026
                        && (u2 - u3).abs() > 0.024
                        && d0 * d1 > 0.0
                        && d1 * d2 > 0.0
                        && d2.abs() > 0.024
                        && u0.abs() > u_pre.abs() + 0.026
                    {
                        let sus_d = d0.abs().min(d1.abs());
                        if sus_d > 0.038 {
                            lso_jump_cnt += 1;
                        }
                        if sus_d > max_lso_d {
                            max_lso_d = sus_d;
                        }
                    }
                }
                if seen_lso_n > 0 && (lso_jump_cnt >= 2 || max_lso_d > 0.068) {
                    if in_syl_contig {
                        odd_quiet_homeo = false;
                        odd_myelin_delta += 0.078;
                        if odd_impulse_lag.is_none() {
                            odd_impulse_lag = Some((4..=7).find(|&lg| fwd_on(lg)).unwrap_or(5));
                        }
                    } else if max_lso_pre < 0.036 {
                        odd_quiet_homeo = false;
                        odd_myelin_delta -= 0.078;
                    }
                }
            }

            if odd_myelin_delta != 0.0 || odd_silent_miss || odd_quiet_homeo {
                let mut odd_sum = 0.0_f32;
                let mut odd_cnt = 0usize;
                for tube in &mut self.slots {
                    if !tube.alive || tube.kind != TubeKind::CellToMotor || (tube.pre & 1) == 0 {
                        continue;
                    }
                    if membrane
                        .motor_pores
                        .get(tube.post as usize)
                        .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                    {
                        let mut next_m = tube.myelin_plasticity_accum + odd_myelin_delta;
                        if odd_silent_miss && odd_myelin_delta == 0.0 && next_m > 0.0 {
                            next_m = (next_m - 0.004).max(0.0);
                        } else if odd_quiet_homeo && odd_myelin_delta == 0.0 && next_m < 0.0 {
                            next_m = (next_m + 0.00006).min(0.0);
                        }
                        tube.myelin_plasticity_accum = sanitize_clamp(next_m, -0.30, 1.0);
                        odd_sum += tube.myelin_plasticity_accum;
                        odd_cnt += 1;
                    }
                }
                let odd_mean = if odd_cnt > 0 {
                    odd_sum / (odd_cnt as f32)
                } else {
                    0.0
                };
                if odd_mean > 0.14
                    && let Some(lag_on) = odd_impulse_lag
                    && let Some(on_entry) = ring.at_lag(lag_on)
                {
                    for &(c_idx, v_past, dv_past, pre_attn) in &on_entry.active_cells {
                        if (c_idx & 1) == 0 {
                            continue;
                        }
                        let pre_sig = sanitize(v_past + 0.25 * dv_past);
                        if pre_sig <= 0.02 {
                            continue;
                        }
                        let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                        let out_len = self
                            .outgoing_by_cell
                            .get(c_idx as usize)
                            .map_or(0, Vec::len);
                        for k in 0..out_len {
                            let t_idx = self.outgoing_by_cell[c_idx as usize][k];
                            let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                                continue;
                            };
                            if !tube.alive || tube.kind != TubeKind::CellToMotor {
                                continue;
                            }
                            let is_mg = membrane
                                .motor_pores
                                .get(tube.post as usize)
                                .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg);
                            if !is_mg {
                                continue;
                            }
                            let du = 0.048 * attn * pre_sig - 0.070 * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                        }
                    }
                }
            }

            if unprompted_bg {
                let mut es = 0.0_f32;
                let mut ec = 0usize;
                let mut os = 0.0_f32;
                let mut oc = 0usize;
                for tube in &self.slots {
                    if tube.alive
                        && tube.kind == TubeKind::CellToMotor
                        && membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                    {
                        if (tube.pre & 1) == 0 {
                            es += tube.myelin_plasticity_accum;
                            ec += 1;
                        } else {
                            os += tube.myelin_plasticity_accum;
                            oc += 1;
                        }
                    }
                }
                let em = if ec > 0 { es / (ec as f32) } else { 0.0 };
                let om = if oc > 0 { os / (oc as f32) } else { 0.0 };
                if em > 0.28 || om > 0.28 {
                    for tube in &mut self.slots {
                        if !tube.alive || tube.kind != TubeKind::CellToMotor {
                            continue;
                        }
                        let is_mg = membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg);
                        if !is_mg {
                            continue;
                        }
                        if (tube.pre & 1) == 0 && em > 0.28 {
                            let du = 0.008 * em - 0.055 * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                        } else if (tube.pre & 1) != 0 && om > 0.28 {
                            let du = 0.008 * om - 0.055 * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                        }
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
                    && !(t.kind == TubeKind::CellToMotor && t.myelin_plasticity_accum > 0.14)
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
            if tube.u[0].abs() >= 0.045
                || (tube.kind == TubeKind::CellToMotor
                    && tube.myelin_plasticity_accum > 0.24
                    && tube.u[0].abs() >= 0.018)
            {
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
        &mut self,
        num_actuators: usize,
        cells: &CellArena,
        membrane: &PoreMembrane,
        body: &mut BodyCore,
        ring: &EfferenceRing,
        mode: ControlMode,
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

        let mut basal_drive = vec![0.0_f32; num_actuators];
        let mut apical_drive = vec![0.0_f32; num_actuators];
        let mut tube_drive = vec![0.0_f32; num_actuators];
        let mut tube_pred_action_val = vec![0.0_f32; num_actuators];
        let mut slow_inhib_brake = vec![0.0_f32; num_actuators];
        let mut stiff_motor_conf = vec![0.0_f32; num_actuators];
        let mut mastery_by_act = vec![0.0_f32; num_actuators];
        let mut mp_cond_by_act = vec![0.0_f32; num_actuators];
        let mut pyr_env_sum = [0.0_f32; TRACKED_ACTUATORS];
        let mut pyr_env_cnt = [0u16; TRACKED_ACTUATORS];

        for mp in &membrane.motor_pores {
            if mp.alive {
                let ch = mp.actuator_channel as usize;
                if ch < num_actuators {
                    mastery_by_act[ch] = mp.motor_mastery.clamp(0.0, 1.0);
                    mp_cond_by_act[ch] = mp.conductance.clamp(0.0, 2.0);
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
            let is_apical = cell.tau >= 16.0;
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
                if act_ch < TRACKED_ACTUATORS {
                    pyr_env_sum[act_ch] =
                        sanitize(pyr_env_sum[act_ch] + sig.abs() * mp.conductance);
                    pyr_env_cnt[act_ch] = pyr_env_cnt[act_ch].saturating_add(1);
                }
                let g = tube.effective_g(&cells.slots);
                let contrib = sanitize(g * sig * mp.conductance);
                if is_apical {
                    apical_drive[act_ch] = sanitize(apical_drive[act_ch] + contrib);
                } else {
                    basal_drive[act_ch] = sanitize(basal_drive[act_ch] + contrib);
                }
                let pred_contrib = sanitize((0.65 * g + 0.85 * tube.w_pred) * sig * mp.conductance);
                tube_pred_action_val[act_ch] =
                    sanitize(tube_pred_action_val[act_ch] + pred_contrib);
                stiff_motor_conf[act_ch] =
                    sanitize_clamp(stiff_motor_conf[act_ch] + tube.stiff_strength(), 0.0, 2.0);
                if is_slow && g < -0.01 {
                    slow_inhib_brake[act_ch] =
                        sanitize(slow_inhib_brake[act_ch] + (-g) * sig.abs());
                }
            }
        }
        for m in 0..num_actuators {
            tube_drive[m] = sanitize(
                1.28 * smooth_saturate(0.78 * basal_drive[m])
                    + 0.96 * smooth_saturate(0.78 * apical_drive[m]),
            );
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
        let acute_distress = (-body.fast_valence * 22.0).clamp(0.0, 1.0);
        let atp_surplus = ((body.atp - 6.0) / 10.0).clamp(0.0, 1.5) * (1.0 - thermal_need);
        let last_tick = ring.at_lag(1).map_or(0, |e| e.tick);
        let probe_window = atp_surplus > 0.20 && (last_tick & 15) < 3;
        let pore_cap = PRED_BUCKETS.min(membrane.sensor_pores.len());
        let n_scout = membrane.scout_stats.len();
        let mut field_breadth = [0u16; TRACKED_ACTUATORS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if sp.alive
                && sp.age_ticks >= 16
                && sp.trophic_ema >= 0.025
                && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
            {
                for m in 0..num_actuators.min(TRACKED_ACTUATORS) {
                    if self.pore_action_w[m][p].abs() >= 0.022 {
                        field_breadth[m] = field_breadth[m].saturating_add(1);
                    }
                }
            }
        }
        let mut max_breadth = 0u16;
        for &fb in &field_breadth[..num_actuators.min(TRACKED_ACTUATORS)] {
            if fb > max_breadth {
                max_breadth = fb;
            }
        }
        let mut is_world_loco = [false; TRACKED_ACTUATORS];
        if max_breadth >= 4 {
            for m in 0..num_actuators.min(TRACKED_ACTUATORS) {
                if field_breadth[m].saturating_mul(2) >= max_breadth {
                    is_world_loco[m] = true;
                }
            }
        }

        let act_cap = num_actuators.min(TRACKED_ACTUATORS);
        let mut is_world_retina = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                continue;
            }
            if max_breadth < 4 {
                is_world_retina[p] = true;
            } else {
                let mut loco_w_p = 0.0_f32;
                let mut nonloco_w_p = 0.0_f32;
                for m in 0..act_cap {
                    let wa = self.pore_action_w[m][p].abs();
                    if is_world_loco[m] {
                        if wa > loco_w_p {
                            loco_w_p = wa;
                        }
                    } else if wa > nonloco_w_p {
                        nonloco_w_p = wa;
                    }
                }
                if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                    is_world_retina[p] = true;
                }
            }
        }

        let mut min_h_ro = [1.0_f32; PRED_BUCKETS];
        let mut is_glottal_cand_pore = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
            {
                continue;
            }
            let mut min_h = 1.0_f32;
            for &hv in &sp.history {
                if hv < min_h {
                    min_h = hv;
                }
            }
            min_h_ro[p] = min_h;
            if is_world_retina[p] {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            if self.pore_agency[p] >= 0.006
                && (-0.38..=-0.16).contains(&min_h)
                && (-0.345..=-0.04).contains(&mean_p)
            {
                is_glottal_cand_pore[p] = true;
            }
        }

        let mut max_asw_jac = [0.0_f32; TRACKED_ACTUATORS];
        let mut tot_pore_reaff = [0.0_f32; TRACKED_ACTUATORS];
        let mut pos_cochlea_w = [0.0_f32; TRACKED_ACTUATORS];
        let mut pos_cochlea_cnt = [0u16; TRACKED_ACTUATORS];
        for m in 0..act_cap {
            for &w in &self.action_sensory_w[m] {
                let wa = w.abs();
                if wa > max_asw_jac[m] {
                    max_asw_jac[m] = wa;
                }
            }
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                    continue;
                }
                let pw = self.pore_action_w[m][p];
                let pwa = pw.abs();
                if self.pore_agency[p] >= 0.010 && pwa >= 0.015 {
                    tot_pore_reaff[m] += pwa;
                }
                if is_glottal_cand_pore[p] && pw >= 0.015 {
                    pos_cochlea_w[m] += pw;
                    pos_cochlea_cnt[m] = pos_cochlea_cnt[m].saturating_add(1);
                }
            }
        }

        let mut m_glottal: Option<usize> = None;
        let mut best_glottal_w = 0.05_f32;
        for m in 0..act_cap {
            if !is_world_loco[m]
                && (tot_pore_reaff[m] >= 0.045 || max_asw_jac[m] >= 0.065)
                && pos_cochlea_cnt[m] >= 2
                && pos_cochlea_w[m] > best_glottal_w
            {
                best_glottal_w = pos_cochlea_w[m];
                m_glottal = Some(m);
            }
        }

        let mut is_monaural_coch = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || is_world_retina[p]
                || membrane.scout_stats[ch_u].is_sparse_impulse()
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            is_monaural_coch[p] = if let Some(mg) = m_glottal {
                self.pore_action_w[mg][p] >= 0.016 && mean_p >= -0.342
            } else {
                self.pore_agency[p] >= 0.010
                    && (-0.38..=-0.16).contains(&min_h_ro[p])
                    && (-0.345..=-0.04).contains(&mean_p)
            };
        }

        let mut somatic_prop_reaff = [0.0_f32; TRACKED_ACTUATORS];
        let mut abs_cochlea_reaff = [0.0_f32; TRACKED_ACTUATORS];
        let mut coch_artic_cnt = [0u16; TRACKED_ACTUATORS];
        for m in 0..act_cap {
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                    continue;
                }
                let pwa = self.pore_action_w[m][p].abs();
                let is_echo = membrane.scout_stats[ch_u].is_sparse_impulse();
                if is_monaural_coch[p] {
                    if pwa >= 0.0045 {
                        abs_cochlea_reaff[m] += pwa;
                        coch_artic_cnt[m] = coch_artic_cnt[m].saturating_add(1);
                    }
                } else if !is_world_retina[p]
                    && !is_echo
                    && (self.pore_mean_v[p] >= -0.22 || min_h_ro[p] > -0.18)
                    && self.pore_agency[p] >= 0.012
                    && pwa >= 0.020
                    && !(max_breadth >= 4
                        && is_world_loco
                            .iter()
                            .enumerate()
                            .any(|(ml, &il)| il && self.pore_action_w[ml][p].abs() >= 0.013))
                {
                    somatic_prop_reaff[m] += pwa;
                }
            }
        }

        let mut quiet_self_7 = false;
        let mut has_recent_breath_13 = true;
        let mut has_recent_breath_14 = true;
        let mut q_lag = [false; 17];
        if let Some(mg) = m_glottal
            && ring.at_lag(16).is_some_and(|e| e.tick > 0)
        {
            q_lag[0] = true;
            for (d, ql) in q_lag.iter_mut().enumerate().skip(1) {
                *ql = ring
                    .at_lag(d)
                    .is_some_and(|e| !e.committed || e.motor_out[mg] <= -0.05);
            }
            quiet_self_7 =
                q_lag[1] && q_lag[2] && q_lag[3] && q_lag[4] && q_lag[5] && q_lag[6] && q_lag[7];
            has_recent_breath_13 = false;
            has_recent_breath_14 = false;
            for k in 1..=6 {
                if q_lag[k]
                    && q_lag[k + 1]
                    && q_lag[k + 2]
                    && q_lag[k + 3]
                    && q_lag[k + 4]
                    && q_lag[k + 5]
                    && q_lag[k + 6]
                    && q_lag[k + 7]
                {
                    has_recent_breath_14 = true;
                    if k <= 5 {
                        has_recent_breath_13 = true;
                    }
                    break;
                }
            }
        }

        let mut loco_nov_sum = 0.0_f32;
        let mut loco_mast_sum = 0.0_f32;
        let mut loco_nov_cnt = 0usize;
        for m in 0..act_cap {
            if is_world_loco[m] {
                let nov = (1.0 - 10.0 * (mastery_by_act[m] - 0.30).max(0.0)).clamp(0.0, 1.0);
                loco_nov_sum += nov;
                loco_mast_sum += mastery_by_act[m];
                loco_nov_cnt += 1;
            }
        }
        let loco_ep_nov = if loco_nov_cnt > 0 {
            loco_nov_sum / (loco_nov_cnt as f32)
        } else {
            1.0
        };
        let loco_mat_emb = if loco_nov_cnt > 0 {
            let mean_lm = loco_mast_sum / (loco_nov_cnt as f32);
            ((mean_lm - 0.12) * 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let active_embodied = if num_actuators <= 8 {
            max_breadth >= 4 && m_glottal.is_some()
        } else {
            max_breadth >= 4 && m_glottal.is_some() && loco_mat_emb >= 0.85 && body.atp > 6.2
        };
        let loco_mat = if active_embodied {
            loco_mat_emb
        } else if num_actuators <= 8 {
            (1.0 - loco_ep_nov).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let emb_ep_nov = (1.0 - loco_mat_emb).clamp(0.0, 1.0);
        let eff_ep_surplus = atp_surplus * (1.0 - acute_distress);
        let mvt_patch_depleted = num_actuators > 8
            && active_embodied
            && body.atp > 22.0
            && body.fast_valence < 0.015
            && acute_distress < 0.05;
        let mvt_wave = if mvt_patch_depleted { 0.24 } else { 0.0 };
        let mvt_probe = if mvt_patch_depleted { 0.32 } else { 0.0 };
        let wave_ep_surplus = if active_embodied {
            if num_actuators <= 8 {
                eff_ep_surplus.min(0.06 + 0.44 * emb_ep_nov)
            } else {
                eff_ep_surplus.min((0.55 * emb_ep_nov).max(mvt_wave))
            }
        } else if num_actuators <= 8 {
            atp_surplus.min(0.28 + 1.23 * loco_ep_nov)
        } else {
            atp_surplus
        };
        let noise_ep_surplus = if active_embodied {
            if num_actuators <= 8 {
                eff_ep_surplus.min(0.02 + 0.38 * emb_ep_nov)
            } else {
                eff_ep_surplus.min(0.45 * emb_ep_nov)
            }
        } else if num_actuators <= 8 {
            atp_surplus.min(0.10 + 1.41 * loco_ep_nov)
        } else {
            atp_surplus
        };
        let probe_surplus = if active_embodied {
            if num_actuators <= 8 {
                eff_ep_surplus.min(0.04 + 0.40 * emb_ep_nov)
            } else {
                eff_ep_surplus.min((0.50 * emb_ep_nov).max(mvt_probe))
            }
        } else if num_actuators <= 8 {
            atp_surplus.min(0.45 + 1.06 * loco_ep_nov)
        } else {
            atp_surplus
        };

        let mut homeo_grad = vec![0.0_f32; num_actuators];
        let mut model_grad = vec![0.0_f32; num_actuators];
        let mut v_align = vec![0.0_f32; num_actuators];
        let mut epistemic_drive = vec![0.0_f32; num_actuators];
        for &(c_idx, _, _) in &sensory_cells {
            let b = (c_idx as usize) % PRED_BUCKETS;
            let cell = &cells.slots[c_idx as usize];
            let homeo_desire = sanitize(0.45 * cell.v_homeo + 0.85 * cell.v_delta_homeo);
            let valence_mag = cell.v_homeo.abs() + cell.v_delta_homeo.abs();
            let predictable_neutral = cell.surprise_ema < 0.45 && valence_mag < 0.18;
            for m in 0..act_cap {
                let jac = self.action_sensory_w[m][b];
                if homeo_desire.abs() > 1.0e-4 {
                    homeo_grad[m] = sanitize(homeo_grad[m] + homeo_desire * jac);
                }
                if predictable_neutral
                    && wave_ep_surplus > 0.01
                    && jac.abs() > 0.026
                    && (!active_embodied || is_world_loco[m])
                {
                    let wave_dir = if explore_wave[m] >= 0.0 { 1.0 } else { -1.0 };
                    let next_ep = sanitize(
                        epistemic_drive[m] + 2.10 * wave_ep_surplus * jac.abs() * wave_dir,
                    );
                    epistemic_drive[m] = if active_embodied {
                        next_ep.clamp(-0.26, 0.26)
                    } else {
                        next_ep
                    };
                }
            }
        }

        let mut m_echo_emitter: Option<usize> = None;
        let mut best_emit_w = 0.010_f32;
        for m in 0..act_cap {
            if !is_world_loco[m]
                && Some(m) != m_glottal
                && somatic_prop_reaff[m] < 0.030
                && self.vocal_artic_w[m] < 0.005
                && self.echo_emit_w[m] > best_emit_w
            {
                best_emit_w = self.echo_emit_w[m];
                m_echo_emitter = Some(m);
            }
        }
        let ring_warm = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut quiet_echo_self = ring_warm;
        if quiet_echo_self {
            for lag in 1..=14 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..act_cap {
                        let is_emit_cand = if let Some(em) = m_echo_emitter {
                            m == em
                        } else {
                            !is_world_loco[m]
                                && somatic_prop_reaff[m] < 0.030
                                && self.vocal_artic_w[m] < 0.005
                                && Some(m) != m_glottal
                        };
                        if is_emit_cand && ent.motor_out[m].abs() > 0.12 {
                            quiet_echo_self = false;
                        }
                    }
                }
            }
        }

        let mut vis_prox_sum = 0.0_f32;
        let mut vis_prox_cnt = 0usize;
        let mut ret_jump_sum = 0.0_f32;
        let mut ret_jump_cnt = 0usize;
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            if is_world_retina[p] {
                let h = sp.hist_head as usize;
                let x_now = sp.history[(h + 15) & 15];
                let x_prev1 = sp.history[(h + 14) & 15];
                ret_jump_sum += (x_now - x_prev1).abs();
                ret_jump_cnt += 1;
                if self.pore_agency[p] >= 0.020 {
                    vis_prox_sum += (x_now - self.pore_mean_v[p]).max(0.0);
                    vis_prox_cnt += 1;
                }
            }
        }
        let ret_jump_mean = if ret_jump_cnt >= 4 {
            ret_jump_sum / (ret_jump_cnt as f32)
        } else {
            0.0
        };
        if mode == ControlMode::Natural && (ret_jump_mean > 0.085 || acute_distress > 0.48) {
            self.peer_vocal_hold = 0.0;
            self.social_loom_ema = 0.0;
            self.peer_coch_age = 255;
            self.peer_echo_age = 255;
            self.peer_disp_coch = 0.0;
            self.peer_disp_echo = 0.0;
            self.peer_bearing_ema = [0.0; TRACKED_ACTUATORS];
        }
        let vis_prox_now = if vis_prox_cnt > 0 {
            sanitize_clamp(vis_prox_sum / (vis_prox_cnt as f32), 0.0, 2.0)
        } else {
            0.0
        };

        let mut acoustic_taxis = [0.0_f32; TRACKED_ACTUATORS];
        let mut top4_coch = [0.0_f32; 4];
        let mut seen_coch_ro = [u32::MAX; PRED_BUCKETS];
        let mut seen_coch_ro_n = 0usize;
        let mut coch_strong_cnt = 0usize;
        let mut coch_weak_cnt = 0usize;
        let mut sym_coch_cnt = 0usize;
        let mut top_echo_0 = 0.0_f32;
        let mut top_echo_1 = 0.0_f32;
        let mut echo_pore_cnt = 0usize;

        let mut top4_lso_pos = [0.0_f32; 4];
        let mut top4_lso_neg = [0.0_f32; 4];
        let mut top4_clean_pos = [0.0_f32; 4];
        let mut top4_clean_neg = [0.0_f32; 4];
        let mut lso_pos_cnt = 0usize;
        let mut lso_neg_cnt = 0usize;
        let mut clean_pos_cnt = 0usize;
        let mut clean_neg_cnt = 0usize;
        let mut lso_signed_sum = 0.0_f32;
        let mut seen_lso_ro = [u32::MAX; PRED_BUCKETS];
        let mut seen_lso_ro_n = 0usize;

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            if is_monaural_coch[p] {
                sym_coch_cnt += 1;
                if mode == ControlMode::Natural && self.pore_bg_ild[p] != 0.0 {
                    self.pore_bg_ild[p] = 0.0;
                }
                if !seen_coch_ro[..seen_coch_ro_n].contains(&sp.channel) {
                    seen_coch_ro[seen_coch_ro_n] = sp.channel;
                    seen_coch_ro_n += 1;
                    let h = sp.hist_head as usize;
                    let u0 = Self::deconv_hair_cell_35(&sp.history, h, 0).max(0.0);
                    let u1 = Self::deconv_hair_cell_35(&sp.history, h, 1).max(0.0);
                    let u2 = Self::deconv_hair_cell_35(&sp.history, h, 2).max(0.0);
                    let u_sus3 = if u0 > 0.0 && u1 > 0.0 && u2 > 0.0 {
                        u0.min(1.30 * u1 + 0.004).min(1.65 * u2 + 0.006)
                    } else {
                        0.0
                    };
                    if u_sus3 > 0.038 {
                        coch_strong_cnt += 1;
                    }
                    if u_sus3 > 0.016 {
                        coch_weak_cnt += 1;
                    }
                    Self::insert_top4(&mut top4_coch, u_sus3);
                }
            }
            if ch_u < n_scout
                && membrane.scout_stats[ch_u].is_sparse_impulse()
                && !is_world_retina[p]
                && self.pore_agency[p] < 0.035
            {
                echo_pore_cnt += 1;
                let h = sp.hist_head as usize;
                let dev_now = (sp.history[(h + 15) & 15] - self.pore_mean_v[p]).max(0.0);
                if dev_now > top_echo_0 {
                    top_echo_1 = top_echo_0;
                    top_echo_0 = dev_now;
                } else if dev_now > top_echo_1 {
                    top_echo_1 = dev_now;
                }
            } else if quiet_self_7
                && sp.age_ticks >= 40
                && ch_u < n_scout
                && !membrane.scout_stats[ch_u].is_sparse_impulse()
                && !is_monaural_coch[p]
                && !is_world_retina[p]
                && !seen_lso_ro[..seen_lso_ro_n].contains(&sp.channel)
            {
                let mean_p = self.pore_mean_v[p];
                let mut somatic_other_w = 0.0_f32;
                for m in 0..act_cap {
                    if !is_world_loco[m] && Some(m) != m_glottal && self.vocal_artic_w[m] < 0.005 {
                        somatic_other_w += self.pore_action_w[m][p].abs();
                    }
                }
                let is_coch_trough = (-0.36..=-0.33).contains(&mean_p)
                    && m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                let is_glottal_pos = m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
                if !is_coch_trough
                    && !is_glottal_pos
                    && min_h_ro[p] <= -0.20
                    && mean_p < -0.18
                    && somatic_other_w < 0.038
                {
                    seen_lso_ro[seen_lso_ro_n] = sp.channel;
                    seen_lso_ro_n += 1;
                    let h = sp.hist_head as usize;
                    let bg = self.pore_bg_ild[p];
                    let d0 = Self::sym_deconv_ild(&sp.history, h, 0) - bg;
                    let d1 = Self::sym_deconv_ild(&sp.history, h, 1) - bg;
                    let d2 = Self::sym_deconv_ild(&sp.history, h, 2) - bg;
                    let d3 = Self::sym_deconv_ild(&sp.history, h, 3) - bg;
                    if d0.abs() > 0.015 {
                        lso_signed_sum += d0;
                    }
                    if d0 * d1 > 0.0 && d1 * d2 > 0.0 && d2 * d3 > 0.0 {
                        let sus_d = d0.abs().min(d1.abs()).min(d2.abs()).min(d3.abs());
                        if sus_d > 0.018 {
                            if d0 > 0.0 {
                                lso_pos_cnt += 1;
                                Self::insert_top4(&mut top4_lso_pos, sus_d);
                            } else {
                                lso_neg_cnt += 1;
                                Self::insert_top4(&mut top4_lso_neg, sus_d);
                            }
                        }
                        if sp.age_ticks >= 48 && bg.abs() < 0.018 && sus_d > 0.022 {
                            if d0 > 0.0 {
                                clean_pos_cnt += 1;
                                Self::insert_top4(&mut top4_clean_pos, sus_d);
                            } else {
                                clean_neg_cnt += 1;
                                Self::insert_top4(&mut top4_clean_neg, sus_d);
                            }
                        }
                    }
                }
            }
        }
        let max_echo_peer_now = top_echo_0.hypot(top_echo_1);
        let (dom_sign_ro, lso_coh_cnt, top4_lso_dom, lso_clean_cnt, top4_lso_clean) =
            if lso_pos_cnt > lso_neg_cnt {
                (
                    1.0_f32,
                    lso_pos_cnt,
                    top4_lso_pos,
                    clean_pos_cnt,
                    top4_clean_pos,
                )
            } else if lso_neg_cnt > lso_pos_cnt {
                (
                    -1.0_f32,
                    lso_neg_cnt,
                    top4_lso_neg,
                    clean_neg_cnt,
                    top4_clean_neg,
                )
            } else if lso_signed_sum >= 0.0 {
                (
                    1.0_f32,
                    lso_pos_cnt,
                    top4_lso_pos,
                    clean_pos_cnt,
                    top4_clean_pos,
                )
            } else {
                (
                    -1.0_f32,
                    lso_neg_cnt,
                    top4_lso_neg,
                    clean_neg_cnt,
                    top4_clean_neg,
                )
            };
        let mut coch_lvl_now = 0.0_f32;
        let peer_harmonic_now = if quiet_self_7 && sym_coch_cnt >= 2 {
            coch_lvl_now = sanitize_clamp(0.5 * (top4_coch[0] + top4_coch[1]), 0.0, 2.0);
            (coch_strong_cnt >= 2
                && coch_lvl_now > 0.040
                && lso_coh_cnt >= 3
                && top4_lso_dom[2] > 0.018)
                || (coch_strong_cnt >= 1
                    && top4_coch[0] > 0.058
                    && lso_coh_cnt >= 4
                    && top4_lso_dom[3] > 0.020)
                || (coch_weak_cnt >= 2
                    && coch_lvl_now > 0.018
                    && lso_clean_cnt >= 5
                    && top4_lso_clean[3] > 0.022)
        } else {
            false
        };
        let peer_echo_now = echo_pore_cnt >= 1 && quiet_echo_self && top_echo_0 > 0.018;

        if mode == ControlMode::Natural {
            self.peer_coch_age = self.peer_coch_age.saturating_add(1);
            self.peer_echo_age = self.peer_echo_age.saturating_add(1);
            if peer_harmonic_now || peer_echo_now {
                self.peer_vocal_hold = 1.0;
            } else {
                self.peer_vocal_hold = sanitize(0.970 * self.peer_vocal_hold);
            }
        }
        let active_peer_hold = if mode == ControlMode::Natural && self.peer_vocal_hold > 0.22 {
            self.peer_vocal_hold
        } else {
            0.0
        };

        let mut lso_norm_now = [0.0_f32; PRED_BUCKETS];
        let mut lso_elder_ok = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
                || is_monaural_coch[p]
                || is_world_retina[p]
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            let mut somatic_other_w = 0.0_f32;
            for m in 0..act_cap {
                if !is_world_loco[m] && Some(m) != m_glottal && self.vocal_artic_w[m] < 0.005 {
                    somatic_other_w += self.pore_action_w[m][p].abs();
                }
            }
            let is_coch_trough = (-0.36..=-0.33).contains(&mean_p)
                && m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
            let is_glottal_pos = m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
            if is_coch_trough
                || is_glottal_pos
                || min_h_ro[p] > -0.20
                || mean_p >= -0.18
                || somatic_other_w >= 0.038
            {
                continue;
            }
            let h = sp.hist_head as usize;
            if quiet_self_7 {
                let u_0 = Self::sym_deconv_ild(&sp.history, h, 0);
                let d0 = u_0 - self.pore_bg_ild[p];
                if peer_harmonic_now
                    && sp.age_ticks >= 40
                    && lso_coh_cnt >= 3
                    && d0 * dom_sign_ro > 0.020
                {
                    lso_norm_now[p] = d0;
                    lso_elder_ok[p] = true;
                }
                if mode == ControlMode::Natural {
                    if sp.age_ticks < 40 {
                        self.pore_bg_ild[p] = if self.pore_bg_ild[p] == 0.0 {
                            sanitize_clamp(u_0, -0.70, 0.70)
                        } else {
                            sanitize_clamp(0.85 * self.pore_bg_ild[p] + 0.15 * u_0, -0.70, 0.70)
                        };
                    } else {
                        let bg_err = (u_0 - self.pore_bg_ild[p]).clamp(-0.045, 0.045);
                        let lr_bg = if !peer_harmonic_now && !peer_echo_now {
                            0.14
                        } else {
                            0.035
                        };
                        self.pore_bg_ild[p] =
                            sanitize_clamp(self.pore_bg_ild[p] + lr_bg * bg_err, -0.70, 0.70);
                    }
                }
            }
        }

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            let h = sp.hist_head as usize;
            let scale_p = self.pore_scale_v[p].max(0.015);
            let mean_p = self.pore_mean_v[p];
            let ch_u = sp.channel as usize;

            if !is_world_retina[p] && ch_u < n_scout {
                let is_echo = membrane.scout_stats[ch_u].is_sparse_impulse();
                if is_echo && self.pore_agency[p] < 0.035 {
                    let mut peak_echo_dev = 0.0_f32;
                    for lag in 0..=2 {
                        let dev_l = (sp.history[(h + 15 - lag) & 15] - mean_p).max(0.0);
                        if dev_l > peak_echo_dev {
                            peak_echo_dev = dev_l;
                        }
                    }
                    if quiet_echo_self && peak_echo_dev > 0.018 {
                        let s_call = sanitize_clamp(peak_echo_dev / scale_p, 0.0, 2.5);
                        for m in 0..act_cap {
                            let tw = self.tectal_map_w[m][p];
                            if tw.abs() >= 0.010 {
                                acoustic_taxis[m] =
                                    sanitize(acoustic_taxis[m] + 2.95 * tw * s_call);
                            }
                        }
                    }
                } else if !is_echo && lso_elder_ok[p] {
                    let d0 = lso_norm_now[p];
                    let s_call = sanitize_clamp(6.5 * d0, -2.5, 2.5);
                    for m in 0..act_cap {
                        let tw = self.tectal_map_w[m][p];
                        if tw.abs() >= 0.010 {
                            acoustic_taxis[m] = sanitize(acoustic_taxis[m] + 3.25 * tw * s_call);
                        }
                    }
                }
            }
        }

        let mut taxis_sq = 0.0_f32;
        for m in 0..act_cap {
            if max_breadth < 4 || is_world_loco[m] {
                taxis_sq += acoustic_taxis[m] * acoustic_taxis[m];
            }
        }
        let raw_taxis_norm = taxis_sq.sqrt();
        let mut cpa_flipped = false;
        if mode == ControlMode::Natural {
            if raw_taxis_norm > 0.030 {
                let mut old_sq = 0.0_f32;
                let mut dot_bear = 0.0_f32;
                for m in 0..act_cap {
                    if max_breadth < 4 || is_world_loco[m] {
                        let b_m = self.peer_bearing_ema[m];
                        let unit_m = acoustic_taxis[m] / raw_taxis_norm;
                        old_sq += b_m * b_m;
                        dot_bear += b_m * unit_m;
                    }
                }
                let old_norm = old_sq.sqrt();
                if old_norm > 0.14 && dot_bear < 0.0 {
                    cpa_flipped = true;
                    for m in 0..act_cap {
                        let unit_m = if max_breadth < 4 || is_world_loco[m] {
                            acoustic_taxis[m] / raw_taxis_norm
                        } else {
                            0.0
                        };
                        self.peer_bearing_ema[m] = sanitize_clamp(0.80 * unit_m, -1.0, 1.0);
                    }
                    if peer_harmonic_now {
                        self.peer_coch_fast = sanitize_clamp(coch_lvl_now, 0.0, 2.0);
                        self.peer_coch_age = 0;
                        self.peer_disp_coch = 0.0;
                    } else {
                        self.peer_coch_age = 255;
                        self.peer_disp_coch = 0.0;
                    }
                    if peer_echo_now {
                        self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                        self.peer_echo_age = 0;
                        self.peer_disp_echo = 0.0;
                    } else {
                        self.peer_echo_age = 255;
                        self.peer_disp_echo = 0.0;
                    }
                    self.social_loom_ema = sanitize(0.25 * self.social_loom_ema);
                } else {
                    for m in 0..act_cap {
                        let unit_m = if max_breadth < 4 || is_world_loco[m] {
                            acoustic_taxis[m] / raw_taxis_norm
                        } else {
                            0.0
                        };
                        self.peer_bearing_ema[m] = sanitize_clamp(
                            0.58 * self.peer_bearing_ema[m] + 0.42 * unit_m,
                            -1.0,
                            1.0,
                        );
                    }
                }
            } else {
                for m in 0..act_cap {
                    self.peer_bearing_ema[m] = sanitize(0.982 * self.peer_bearing_ema[m]);
                }
            }
        }
        let mut bear_sq = 0.0_f32;
        for m in 0..act_cap {
            bear_sq += self.peer_bearing_ema[m] * self.peer_bearing_ema[m];
        }
        let bear_norm = bear_sq.sqrt();
        if raw_taxis_norm < 0.030 && active_peer_hold > 0.25 && bear_norm > 0.16 {
            for m in 0..act_cap {
                if max_breadth < 4 || is_world_loco[m] {
                    acoustic_taxis[m] =
                        sanitize(0.88 * active_peer_hold * (self.peer_bearing_ema[m] / bear_norm));
                }
            }
        }
        let mut v_inst = [0.0_f32; TRACKED_ACTUATORS];
        if let Some(ent15) = ring.at_lag(15)
            && ent15.committed
        {
            for m in 0..act_cap {
                v_inst[m] = sanitize(0.84 * ent15.motor_out[m]);
            }
        }
        let mut v_by_lag = [[0.0_f32; TRACKED_ACTUATORS]; 16];
        for lag in (1..=15).rev() {
            if let Some(ent) = ring.at_lag(lag)
                && ent.committed
            {
                for m in 0..act_cap {
                    v_inst[m] = sanitize((v_inst[m] + 0.185 * ent.motor_out[m]) * 0.82);
                }
            } else {
                for m in 0..act_cap {
                    v_inst[m] = sanitize(v_inst[m] * 0.82);
                }
            }
            v_by_lag[lag] = v_inst;
        }
        let v_ego_now = if bear_norm > 0.12 {
            let mut proj = 0.0_f32;
            for (m, &vm) in v_by_lag[1].iter().enumerate().take(act_cap) {
                if max_breadth < 4 || is_world_loco[m] {
                    proj += (self.peer_bearing_ema[m] / bear_norm) * vm;
                }
            }
            let bear_conf = 0.62 * (bear_norm / 0.36).clamp(0.0, 1.0);
            sanitize_clamp(bear_conf * proj, -0.55, 0.55)
        } else {
            0.0
        };

        let mut loom_num = 0.0_f32;
        let mut loom_den = 0.0_f32;
        if mode == ControlMode::Natural && !cpa_flipped {
            self.peer_disp_coch = sanitize_clamp(self.peer_disp_coch + v_ego_now, -7.5, 7.5);
            self.peer_disp_echo = sanitize_clamp(self.peer_disp_echo + v_ego_now, -7.5, 7.5);

            if peer_harmonic_now {
                let dt_c = self.peer_coch_age;
                let drifted_c = dt_c > 16;
                if drifted_c || self.peer_coch_slow <= 0.014 {
                    self.peer_coch_fast = coch_lvl_now;
                    self.peer_coch_slow = coch_lvl_now;
                    self.peer_coch_age = 0;
                    self.peer_disp_coch = 0.0;
                } else if dt_c >= 3 {
                    let prev_fast = self.peer_coch_fast;
                    let u_mid = 0.5 * (prev_fast + coch_lvl_now);
                    let sens_c = ((0.10 + 0.48 * u_mid) * u_mid).clamp(0.0038, 0.045);
                    let disp_close_c = ((coch_lvl_now - prev_fast) / sens_c).clamp(-7.5, 7.5);
                    let v_peer_c = sanitize_clamp(
                        (disp_close_c - self.peer_disp_coch) / (dt_c as f32),
                        -0.75,
                        0.75,
                    );
                    self.peer_coch_fast = sanitize_clamp(coch_lvl_now, 0.0, 2.0);
                    self.peer_coch_slow =
                        sanitize_clamp(0.82 * self.peer_coch_slow + 0.18 * coch_lvl_now, 0.0, 2.0);
                    self.peer_coch_age = 0;
                    self.peer_disp_coch = 0.0;
                    loom_num = sanitize(loom_num + 1.35 * (v_peer_c / 0.085));
                    loom_den = sanitize(loom_den + 1.35);
                }
            }

            if peer_echo_now {
                let dt_e = self.peer_echo_age;
                let drifted_e = dt_e > 28 || (dt_e > 14 && self.peer_disp_echo.abs() > 2.4);
                if drifted_e || self.peer_echo_slow <= 0.014 {
                    self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_slow = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_age = 0;
                    self.peer_disp_echo = 0.0;
                } else if dt_e >= 3 {
                    let prev_fast = self.peer_echo_fast;
                    let a_mid = 0.5 * (prev_fast + max_echo_peer_now);
                    let sens_e = ((0.020 + 0.22 * a_mid) * a_mid).clamp(0.0035, 0.075);
                    let disp_close_e = ((max_echo_peer_now - prev_fast) / sens_e).clamp(-7.5, 7.5);
                    let v_peer_e = sanitize_clamp(
                        (disp_close_e - self.peer_disp_echo) / (dt_e as f32),
                        -0.75,
                        0.75,
                    );
                    self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_slow = sanitize_clamp(
                        0.82 * self.peer_echo_slow + 0.18 * max_echo_peer_now,
                        0.0,
                        2.0,
                    );
                    self.peer_echo_age = 0;
                    self.peer_disp_echo = 0.0;
                    loom_num = sanitize(loom_num + 1.80 * (v_peer_e / 0.085));
                    loom_den = sanitize(loom_den + 1.80);
                }
            }
        }

        let min_sample_age = self.peer_coch_age.min(self.peer_echo_age);
        if loom_den > 0.024 {
            let inst_loom = smooth_saturate(2.8 * loom_num / (loom_den + 0.05));
            if mode == ControlMode::Natural {
                self.social_loom_ema =
                    sanitize_clamp(0.28 * self.social_loom_ema + 0.72 * inst_loom, -1.0, 1.0);
            }
        } else if mode == ControlMode::Natural && !cpa_flipped {
            let decay = if min_sample_age <= 15 && active_peer_hold > 0.25 {
                0.945
            } else {
                0.62
            };
            self.social_loom_ema = sanitize(decay * self.social_loom_ema);
        }
        let peer_present_gate = (active_peer_hold * 1.6).clamp(0.0, 1.0);
        let foveated_loom = if mode == ControlMode::Natural {
            self.social_loom_ema * peer_present_gate
        } else {
            0.0
        };

        let ers = body.ers_scarcity.clamp(0.0, 1.0);
        let ex_scale = (0.12 + 0.68 * body.temperature).clamp(0.08, 0.90);
        let hunger = ((6.0 - body.atp) / 6.0).clamp(0.0, 1.0).max(thermal_need);
        let comfort_atp = (0.65 * body.max_atp).max(6.0);
        let unsated = ((comfort_atp - body.atp) / comfort_atp).clamp(0.0, 1.0);
        let eff_hunger = if num_actuators <= 8 {
            hunger.max(loco_mat * (0.28 + 0.48 * unsated))
        } else if active_embodied {
            let adult_cruise =
                loco_mat * (0.30 * unsated).max(0.18) * ((body.atp - 6.0) / 6.0).clamp(0.0, 1.0);
            hunger.max(adult_cruise)
        } else {
            hunger
        };
        let vigor = if active_embodied {
            if num_actuators <= 8 {
                1.0 + 0.90 * hunger + 0.35 * (eff_hunger - hunger).max(0.0) + 0.30 * ers
            } else {
                1.0 + 0.90 * eff_hunger + 0.30 * ers
            }
        } else {
            1.0 + 0.90 * hunger + 0.30 * ers
        };
        let starvation_guard = ((body.atp - 2.0) / 1.8).clamp(0.0, 1.0);
        let close_peer_suppress = (1.0
            - 5.0 * (coch_lvl_now - 0.115).max(0.0)
            - 4.0 * (max_echo_peer_now - 0.165).max(0.0))
        .clamp(0.0, 1.0);
        let taxis_gate =
            (starvation_guard * close_peer_suppress * (1.0 - 1.5 * acute_distress)).clamp(0.0, 1.0);
        let express_gate = ((1.0 - 0.14 * hunger) * (1.0 - acute_distress)).clamp(0.0, 1.0);
        let social_loom = express_gate * foveated_loom;
        let engage_sig = smooth_saturate(4.6 * social_loom);
        let approach_engage = engage_sig.clamp(0.0, 0.98);
        let recede_call = (-engage_sig).clamp(0.0, 0.98);
        let limb_loom_scale = (1.0 + 0.94 * approach_engage - 0.96 * recede_call).clamp(0.04, 1.94);
        let mut homeo_sq = 0.0_f32;
        for m in 0..act_cap {
            if max_breadth < 4 || is_world_loco[m] {
                homeo_sq += homeo_grad[m] * homeo_grad[m];
            }
        }
        let homeo_vec_norm = homeo_sq.sqrt();
        let vis_target_suppress = (1.0 - 1.35 * homeo_vec_norm).clamp(0.38, 1.0);
        let vis_occ_gate = if vis_prox_now < 0.180 {
            (1.0 - 2.6 * (vis_prox_now - 0.05).max(0.0)).clamp(0.26, 1.0)
        } else {
            (0.26 * (1.0 - (vis_prox_now - 0.180) / 0.030)).clamp(0.0, 0.26)
        };
        let social_drive_mag = 0.64 * vis_occ_gate;
        for m in 0..num_actuators {
            if m < TRACKED_ACTUATORS {
                let social_va = if active_peer_hold > 0.25
                    && bear_norm > 0.16
                    && (max_breadth < 4 || is_world_loco[m])
                {
                    let u_peer_m = if raw_taxis_norm > 0.040 {
                        0.60 * (acoustic_taxis[m] / raw_taxis_norm)
                            + 0.40 * (self.peer_bearing_ema[m] / bear_norm)
                    } else {
                        self.peer_bearing_ema[m] / bear_norm
                    };
                    taxis_gate * vis_target_suppress * social_drive_mag * u_peer_m
                } else {
                    0.0
                };
                v_align[m] = sanitize_clamp(social_va, -0.78, 0.78);
            }
            model_grad[m] = sanitize_clamp(homeo_grad[m] + v_align[m], -1.5, 1.5);
        }
        let mut glottal_urge_gain = 0.0_f32;
        let mut glottal_listen_gain = 0.0_f32;
        let mut glottal_syl_exp = false;
        let mut glottal_syl_inhale = false;
        let mut mg_urge_active: Option<usize> = None;
        let mut mg_listen_active: Option<usize> = None;
        let mut sated_listen_mode = false;
        if num_actuators > 8
            && mode == ControlMode::Natural
            && body.rest_consolidation_ticks >= 480
            && body.atp > 1.2
            && acute_distress < 0.96
            && ring_warm
        {
            let mut soft_coch_w_ro = [0.0_f32; TRACKED_ACTUATORS];
            let mut soft_coch_cnt_ro = [0u16; TRACKED_ACTUATORS];
            let mut sat_coch_cnt_ro = [0u16; TRACKED_ACTUATORS];
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout
                    || is_world_retina[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                if (-0.356..=-0.16).contains(&min_h_ro[p])
                    && (-0.354..=0.10).contains(&self.pore_mean_v[p])
                {
                    let pw_bar = if self.pore_mean_v[p] >= -0.349 {
                        0.0045
                    } else {
                        0.0120
                    };
                    for m in 0..act_cap {
                        let pw = self.pore_action_w[m][p];
                        if pw >= pw_bar {
                            soft_coch_w_ro[m] += pw;
                            soft_coch_cnt_ro[m] = soft_coch_cnt_ro[m].saturating_add(1);
                        }
                    }
                } else if (min_h_ro[p] > -0.16 || self.pore_mean_v[p] > 0.10)
                    && self.pore_mean_v[p] >= -0.354
                {
                    for m in 0..act_cap {
                        if self.pore_action_w[m][p] >= 0.006 {
                            sat_coch_cnt_ro[m] = sat_coch_cnt_ro[m].saturating_add(1);
                        }
                    }
                }
            }
            let mut mg_urge_opt = m_glottal;
            if mg_urge_opt.is_none() {
                let mut best_my = 0.14_f32;
                for m in 0..act_cap {
                    if is_world_loco[m] {
                        continue;
                    }
                    let sat_apnea = cells.alive_count < 150
                        && mastery_by_act[m] >= 0.52
                        && soft_coch_w_ro[m] < 0.008
                        && sat_coch_cnt_ro[m] >= 2
                        && (1..=6).all(|d| {
                            ring.at_lag(d)
                                .is_some_and(|e| e.committed && e.motor_out[m] > 0.05)
                        });
                    if soft_coch_w_ro[m] < 0.008 && !sat_apnea {
                        continue;
                    }
                    let mut es = 0.0_f32;
                    let mut ec = 0usize;
                    let mut os = 0.0_f32;
                    let mut oc = 0usize;
                    for tube in &self.slots {
                        if tube.alive
                            && tube.kind == TubeKind::CellToMotor
                            && membrane
                                .motor_pores
                                .get(tube.post as usize)
                                .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == m)
                        {
                            if (tube.pre & 1) == 0 {
                                es += tube.myelin_plasticity_accum;
                                ec += 1;
                            } else {
                                os += tube.myelin_plasticity_accum;
                                oc += 1;
                            }
                        }
                    }
                    let em = if ec >= 3 { es / (ec as f32) } else { 0.0 };
                    let om = if oc >= 3 { os / (oc as f32) } else { 0.0 };
                    let pk = if sat_apnea {
                        if em.max(om) >= 0.65 { em.max(om) } else { 0.0 }
                    } else {
                        em.max(om)
                    };
                    if pk > best_my {
                        best_my = pk;
                        mg_urge_opt = Some(m);
                    }
                }
            }
            if mg_urge_opt.is_none() {
                let mut best_sw = 0.018_f32;
                for m in 0..act_cap {
                    if !is_world_loco[m]
                        && (tot_pore_reaff[m] >= 0.020
                            || max_asw_jac[m] >= 0.038
                            || (soft_coch_cnt_ro[m] >= 4 && soft_coch_w_ro[m] >= 0.048))
                        && soft_coch_cnt_ro[m] >= 2
                        && soft_coch_w_ro[m] > best_sw
                    {
                        best_sw = soft_coch_w_ro[m];
                        mg_urge_opt = Some(m);
                    }
                }
            }
            if let Some(mg_act) = mg_urge_opt {
                let mut even_sum = 0.0_f32;
                let mut even_cnt = 0usize;
                let mut odd_sum = 0.0_f32;
                let mut odd_cnt = 0usize;
                for tube in &self.slots {
                    if !tube.alive || tube.kind != TubeKind::CellToMotor {
                        continue;
                    }
                    if membrane
                        .motor_pores
                        .get(tube.post as usize)
                        .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg_act)
                    {
                        if (tube.pre & 1) == 0 {
                            even_sum += tube.myelin_plasticity_accum;
                            even_cnt += 1;
                        } else {
                            odd_sum += tube.myelin_plasticity_accum;
                            odd_cnt += 1;
                        }
                    }
                }
                let u_even_raw = if even_cnt > 0 {
                    even_sum / (even_cnt as f32)
                } else {
                    0.0
                };
                let u_odd_raw = if odd_cnt > 0 {
                    odd_sum / (odd_cnt as f32)
                } else {
                    0.0
                };
                let u_even = u_even_raw.max(0.0);
                let u_odd = u_odd_raw.max(0.0);
                let vocal_sensitive_ro = body.rest_consolidation_ticks < 9_000
                    || u_even >= 0.72
                    || (u_odd >= 0.72 && u_even_raw > -0.20);
                let contingent_urge = if vocal_sensitive_ro {
                    u_even.max(u_odd).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                glottal_urge_gain = ((contingent_urge - 0.28) / 0.28).clamp(0.0, 1.0);
                if glottal_urge_gain >= 0.25 {
                    mg_urge_active = Some(mg_act);
                    let mut q_urge = [false; 17];
                    q_urge[0] = true;
                    for (d, ql) in q_urge.iter_mut().enumerate().skip(1) {
                        *ql = ring
                            .at_lag(d)
                            .is_some_and(|e| !e.committed || e.motor_out[mg_act] <= -0.05);
                    }
                    let q_8_now = (1..=8).all(|d| q_urge[d]);
                    let q_8_s1 = !q_urge[1] && (2..=9).all(|d| q_urge[d]);
                    let q_8_s2 = !q_urge[1] && !q_urge[2] && (3..=10).all(|d| q_urge[d]);
                    let q_8_s3 =
                        !q_urge[1] && !q_urge[2] && !q_urge[3] && (4..=11).all(|d| q_urge[d]);
                    glottal_syl_exp = q_8_now || q_8_s1 || q_8_s2 || q_8_s3;
                    glottal_syl_inhale = !glottal_syl_exp;
                    if glottal_syl_exp {
                        let pyr_mean = if pyr_env_cnt[mg_act] > 0 {
                            pyr_env_sum[mg_act] / (pyr_env_cnt[mg_act] as f32)
                        } else {
                            0.25
                        };
                        let corticobulbar_drive =
                            glottal_urge_gain * (0.58 + 1.05 * pyr_mean.clamp(0.12, 0.75));
                        tube_drive[mg_act] = sanitize(
                            (tube_drive[mg_act] + corticobulbar_drive)
                                .max(0.78 * glottal_urge_gain),
                        );
                    }
                } else {
                    let adult_pruned = (8 * num_actuators..150).contains(&cells.alive_count);
                    let sated_adult_listen = adult_pruned
                        && vocal_sensitive_ro
                        && body.atp > 24.0
                        && u_odd_raw <= -0.25
                        && u_even_raw <= -0.22
                        && soft_coch_cnt_ro[mg_act] >= 4;
                    let dev_listen = cells.alive_count >= 152
                        && vocal_sensitive_ro
                        && soft_coch_cnt_ro[mg_act] >= 6
                        && u_odd_raw <= -0.20
                        && u_even_raw < 0.02;
                    if (sated_adult_listen || dev_listen) && body.atp > 2.0 && acute_distress < 0.85
                    {
                        mg_listen_active = Some(mg_act);
                        sated_listen_mode = sated_adult_listen;
                        glottal_listen_gain = if sated_adult_listen {
                            ((-u_odd_raw - 0.22) / 0.07).clamp(0.0, 1.0)
                        } else {
                            ((-u_odd_raw - 0.18) / 0.10).clamp(0.0, 1.0)
                        };
                        let listen_pull = if sated_adult_listen { 0.42 } else { 0.14 };
                        tube_drive[mg_act] =
                            sanitize(tube_drive[mg_act] - listen_pull * glottal_listen_gain);
                    }
                }
            }
        }

        let idle_ers_damp = 1.0 - 0.45 * ers;
        let mut has_survival_vec = vec![false; num_actuators];
        let mut is_glottal_vec = vec![false; num_actuators];
        let mut limb_share_vec = vec![0.0_f32; num_actuators];
        let mut emit_share_vec = vec![0.0_f32; num_actuators];
        let mut candidates: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0_f32; num_actuators]);
        for m in 0..num_actuators {
            let is_loco_m = m < TRACKED_ACTUATORS && is_world_loco[m];
            let is_glottal_act =
                Some(m) == m_glottal || Some(m) == mg_urge_active || Some(m) == mg_listen_active;
            let is_urge_glottal = Some(m) == mg_urge_active;
            let is_urge_vocal = is_urge_glottal;
            is_glottal_vec[m] = is_glottal_act;
            let is_echo_emitter = Some(m) != mg_urge_active
                && Some(m) != mg_listen_active
                && Some(m) == m_echo_emitter;
            let som_dom = somatic_prop_reaff[m].max(0.5 * self.somatic_limb_w[m]);
            let is_vocal_artic = m < TRACKED_ACTUATORS
                && !is_loco_m
                && !is_glottal_act
                && !is_echo_emitter
                && ((self.vocal_artic_w[m] >= 0.005 && abs_cochlea_reaff[m] > som_dom)
                    || (coch_artic_cnt[m] >= 2
                        && abs_cochlea_reaff[m] >= 0.011
                        && abs_cochlea_reaff[m] > 1.2 * som_dom));
            let is_somatic_limb = m < TRACKED_ACTUATORS
                && !is_loco_m
                && !is_glottal_act
                && !is_echo_emitter
                && !is_vocal_artic
                && ((somatic_prop_reaff[m] >= 0.022
                    && somatic_prop_reaff[m] >= abs_cochlea_reaff[m])
                    || self.somatic_limb_w[m] >= 0.022);
            let homeo_m_abs = (homeo_grad[m] * vigor).abs();
            let raw_td = tube_drive[m] * vigor;
            let surv_thresh = if active_embodied && is_loco_m {
                0.065
            } else {
                0.14
            };
            let has_survival_drive = raw_td.abs() + homeo_m_abs >= surv_thresh
                || (active_embodied && is_loco_m && v_align[m].abs() >= 0.06);
            has_survival_vec[m] = has_survival_drive;
            let nmj_gate = ((mp_cond_by_act[m] - 0.16) * 3.5).clamp(0.0, 1.0);
            let limb_share = if is_somatic_limb && body.atp > 1.3 {
                (1.0 - 2.0 * homeo_grad[m].abs() - 2.0 * acute_distress).clamp(0.0, 1.0) * nmj_gate
            } else {
                0.0
            };
            let emit_share = if is_echo_emitter {
                (1.0 - 2.0 * acute_distress).clamp(0.0, 1.0) * nmj_gate
            } else {
                0.0
            };
            limb_share_vec[m] = limb_share;
            emit_share_vec[m] = emit_share;
            let mod_scale = 1.0 + limb_share * (limb_loom_scale - 1.0);
            let td = raw_td * mod_scale;
            let mg = model_grad[m] * vigor;
            let mastery = mastery_by_act[m];
            let stiff_c = stiff_motor_conf[m].min(1.0);
            let urge_babble_atten = if is_urge_vocal && glottal_syl_exp {
                1.0 - 0.75 * glottal_urge_gain
            } else if is_vocal_artic && mg_urge_active.is_some() {
                1.0 - 0.68 * glottal_urge_gain
            } else if Some(m) == mg_listen_active {
                1.0 - (if sated_listen_mode { 0.88 } else { 0.52 }) * glottal_listen_gain
            } else {
                1.0
            };
            let babble_maturity =
                (1.0 - 0.35 * mastery) * (1.0 - 0.20 * stiff_c) * urge_babble_atten;
            let ep_listen_atten = if Some(m) == mg_listen_active && sated_listen_mode {
                1.0 - 0.85 * glottal_listen_gain
            } else {
                1.0
            };
            let ep = if !has_survival_drive {
                epistemic_drive[m] * idle_ers_damp * mod_scale * ep_listen_atten
            } else {
                0.0
            };
            let newborn_spontaneity = 0.28 * (1.0 - mastery);
            let idle_ex_gate = (if ep.abs() > 0.05 {
                0.38 * babble_maturity + 0.12
            } else {
                (0.08 + newborn_spontaneity + 0.20 * noise_ep_surplus.min(1.0)) * babble_maturity
            }) * idle_ers_damp
                * mod_scale;
            let surv_mod = if homeo_m_abs < 0.04 { mod_scale } else { 1.0 };
            let local_ex_scale = if has_survival_drive {
                ex_scale * (0.45 + 0.55 * babble_maturity) * surv_mod
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
                    + (0.65 * klino_gain_mean.abs().max(0.15) * pa) * mod_scale.min(1.15)
                    + 0.14 * local_ex_scale * ex,
            );
            let td_c2 = if is_urge_vocal && glottal_syl_exp {
                1.60 * td
            } else {
                0.95 * td
            };
            candidates[2][m] =
                smooth_saturate(2.25 * mg + td_c2 + ep + 0.28 * local_ex_scale * ex2);
            let td_c3 = if is_urge_vocal && glottal_syl_exp {
                1.55 * td
            } else {
                0.80 * td
            };
            let td_c4 = if is_urge_vocal && glottal_syl_exp {
                1.50 * td
            } else {
                0.55 * td
            };
            candidates[3][m] = smooth_saturate(
                td_c3
                    + 0.80 * mg
                    + ep
                    + (0.42 + 0.48 * ex_scale) * babble_maturity * idle_ers_damp * mod_scale * ex,
            );
            candidates[4][m] = smooth_saturate(
                td_c4
                    + 0.55 * mg
                    + 0.82 * ex_scale * babble_maturity * idle_ers_damp * mod_scale * ex2,
            );
        }

        let pain_bar = 0.045;
        let mut policy_conf = 0.0_f32;
        let mut max_act_conf = 0.0_f32;
        for (m, &pv) in tube_pred_action_val.iter().enumerate() {
            let ac = pv.abs() + 0.8 * homeo_grad[m].abs();
            policy_conf += ac;
            if ac > max_act_conf {
                max_act_conf = ac;
            }
        }
        let mean_policy_conf = policy_conf / (num_actuators.max(1) as f32);
        let eff_policy_conf = mean_policy_conf.max(0.42 * max_act_conf);
        let eff_probe = probe_window && (!active_embodied || probe_surplus > 0.08);
        let forage_thresh = 0.0067 + 0.0065 * eff_hunger;
        let forage_search_w = if eff_policy_conf < forage_thresh {
            0.035 * body.temperature.max(0.28 * loco_mat)
        } else if eff_probe {
            0.12 * probe_surplus
        } else {
            0.0
        };
        let effort_w = if eff_policy_conf < forage_thresh || eff_probe {
            0.0
        } else {
            0.065 * (1.35 - 0.65 * atp_surplus.min(1.5)) * (1.0 - 0.70 * eff_hunger)
        };
        let epistemic_w = 1.85 * noise_ep_surplus;
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
                let urge_exp_m = if is_glottal_vec[m] && glottal_syl_exp {
                    0.085 * glottal_urge_gain
                } else {
                    0.0
                };
                pred_val += (tube_pred_action_val[m] + 0.55 * v_align[m] + urge_exp_m) * cand[m];
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
                    if jac.abs() > 0.026
                        && !has_survival_vec[m]
                        && (!active_embodied || is_world_loco[m])
                    {
                        ep_act_dv += (jac * da_m).abs();
                    }
                }
                let p_null = smooth_saturate(v_now + self_dv);
                let v_step1 = smooth_saturate(v_now + self_dv + act_dv);
                let delta_from_act = v_step1 - p_null;
                let cell = &cells.slots[c_idx as usize];
                pred_val +=
                    0.50 * cell.v_homeo * delta_from_act + 0.75 * cell.v_delta_homeo * act_dv;
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
            let mut total_score = pred_val + epistemic_w * epistemic_val + forage_search_w * rms
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
                let is_loco_m = m < TRACKED_ACTUATORS && is_world_loco[m];
                let ls = limb_share_vec[m];
                let emit_s = emit_share_vec[m];
                let social_active = mode == ControlMode::Natural && active_peer_hold > 0.25;
                let recent_breath_ok = if explore_wave[m] + 0.65 * explore_alt[m] < -0.15 {
                    has_recent_breath_13
                } else {
                    has_recent_breath_14
                };
                let urge_inhale_m = Some(m) == mg_urge_active
                    && mode == ControlMode::Natural
                    && glottal_syl_inhale
                    && acute_distress < 0.96;
                let soft_inhale_m = urge_inhale_m;
                let hb_inhale = is_glottal_vec[m]
                    && mode == ControlMode::Natural
                    && ((!recent_breath_ok && acute_distress < 0.40) || soft_inhale_m);
                let spindle_alpha = if emit_s > 0.05 || hb_inhale {
                    if soft_inhale_m && recent_breath_ok {
                        0.94
                    } else {
                        0.92
                    }
                } else if ls > 0.05 && social_active {
                    0.93
                } else if is_glottal_vec[m] {
                    if glottal_syl_exp && glottal_urge_gain >= 0.25 {
                        0.88
                    } else {
                        (0.86 - 0.18 * maturity).clamp(0.68, 0.86)
                    }
                } else if has_survival_vec[m] || is_loco_m {
                    (0.92 - 0.30 * maturity).clamp(0.58, 0.92)
                } else {
                    (0.95 - 0.80 * maturity).clamp(0.38, 0.95)
                };
                let urge_open_artic = mg_urge_active.is_some()
                    && m < TRACKED_ACTUATORS
                    && !is_loco_m
                    && !is_glottal_vec[m]
                    && (self.vocal_artic_w[m] >= 0.004 || abs_cochlea_reaff[m] >= 0.008);
                let raw_t = if urge_open_artic {
                    smoothed[m] * (1.0 - 0.64 * glottal_urge_gain)
                } else if emit_s > 0.05 {
                    let ping_bout = mode == ControlMode::Natural
                        && body.atp > 1.2
                        && quiet_echo_self
                        && explore_alt[m] > 0.78;
                    if ping_bout {
                        (0.86 * emit_s).clamp(0.0, 0.90)
                    } else {
                        0.0
                    }
                } else if hb_inhale {
                    if soft_inhale_m && recent_breath_ok {
                        -0.28
                    } else {
                        -0.38
                    }
                } else if Some(m) == mg_urge_active && glottal_syl_exp {
                    smoothed[m].max(0.74 * glottal_urge_gain)
                } else if Some(m) == mg_listen_active {
                    if sated_listen_mode {
                        smoothed[m] * (1.0 - 0.78 * glottal_listen_gain)
                            - 0.24 * glottal_listen_gain
                    } else {
                        smoothed[m] - 0.12 * glottal_listen_gain
                    }
                } else if ls > 0.05 && social_active {
                    let base_s = smoothed[m] * (1.0 - 0.28 * ls * peer_present_gate);
                    let shunted = base_s * (1.0 - 0.96 * ls * recede_call);
                    let unit_dir = if smoothed[m].abs() >= 0.04 {
                        smoothed[m].signum()
                    } else if prev_act[m].abs() >= 0.04 {
                        prev_act[m].signum()
                    } else if explore_wave[m] >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let headroom = (0.96 - shunted.abs()).max(0.0);
                    shunted + ls * approach_engage * headroom * unit_dir
                } else if is_loco_m
                    && active_peer_hold > 0.25
                    && bear_norm > 0.16
                    && acute_distress < 0.35
                {
                    let brake_safe = (1.0 - 2.0 * slow_inhib_brake[m]).clamp(0.0, 1.0);
                    if vis_prox_now < 0.180 && body.atp > 2.0 {
                        smoothed[m] + 0.62 * brake_safe * v_align[m]
                    } else {
                        smoothed[m]
                    }
                } else {
                    smoothed[m]
                };
                let target_act = sanitize_clamp(raw_t, -1.0, 1.0);
                smoothed[m] = sanitize_clamp(
                    (1.0 - spindle_alpha) * prev_act[m] + spindle_alpha * target_act,
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
            t.myelin_plasticity_accum = 0.0;
        }
        self.action_sensory_w = [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS];
        self.pore_action_w = [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS];
        self.tectal_map_w = [[0.0; PRED_BUCKETS]; TRACKED_ACTUATORS];
        self.sensory_self_w = [0.0; PRED_BUCKETS];
        self.pore_self_w = [0.0; PRED_BUCKETS];
        self.pore_mean_v = [0.0; PRED_BUCKETS];
        self.pore_scale_v = [0.05; PRED_BUCKETS];
        self.pore_agency = [0.0; PRED_BUCKETS];
        self.pore_dsq_ema = [1.0e-4; PRED_BUCKETS];
        self.pore_var_ema = [5.0e-3; PRED_BUCKETS];
        self.pore_bg_ild = [0.0; PRED_BUCKETS];
        self.social_loom_ema = 0.0;
        self.spatial_prox_ring = [0.0; RING_LEN];
        self.spatial_prox_cursor = 0;
        self.echo_emit_w = [0.0; TRACKED_ACTUATORS];
        self.vocal_artic_w = [0.0; TRACKED_ACTUATORS];
        self.somatic_limb_w = [0.0; TRACKED_ACTUATORS];
        self.peer_coch_fast = 0.0;
        self.peer_coch_slow = 0.0;
        self.peer_echo_fast = 0.0;
        self.peer_echo_slow = 0.0;
        self.peer_disp_coch = 0.0;
        self.peer_disp_echo = 0.0;
        self.peer_vocal_hold = 0.0;
        self.peer_coch_age = 255;
        self.peer_echo_age = 255;
        self.peer_bearing_ema = [0.0; TRACKED_ACTUATORS];
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
                hasher.write_f32(t.myelin_plasticity_accum);
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
        for row in &self.pore_action_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for row in &self.tectal_map_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for &w in &self.sensory_self_w {
            hasher.write_f32(w);
        }
        for &w in &self.pore_self_w {
            hasher.write_f32(w);
        }
        for &mv in &self.pore_mean_v {
            hasher.write_f32(mv);
        }
        for &sv in &self.pore_scale_v {
            hasher.write_f32(sv);
        }
        for &pa in &self.pore_agency {
            hasher.write_f32(pa);
        }
        for &dsq in &self.pore_dsq_ema {
            hasher.write_f32(dsq);
        }
        for &pvr in &self.pore_var_ema {
            hasher.write_f32(pvr);
        }
        for &pbg in &self.pore_bg_ild {
            hasher.write_f32(pbg);
        }
        hasher.write_f32(self.social_loom_ema);
        for &spv in &self.spatial_prox_ring {
            hasher.write_f32(spv);
        }
        hasher.write_u8(self.spatial_prox_cursor);
        for &ew in &self.echo_emit_w {
            hasher.write_f32(ew);
        }
        for &vw in &self.vocal_artic_w {
            hasher.write_f32(vw);
        }
        for &sw in &self.somatic_limb_w {
            hasher.write_f32(sw);
        }
        hasher.write_f32(self.peer_coch_fast);
        hasher.write_f32(self.peer_coch_slow);
        hasher.write_f32(self.peer_echo_fast);
        hasher.write_f32(self.peer_echo_slow);
        hasher.write_f32(self.peer_disp_coch);
        hasher.write_f32(self.peer_disp_echo);
        hasher.write_f32(self.peer_vocal_hold);
        hasher.write_u8(self.peer_coch_age);
        hasher.write_u8(self.peer_echo_age);
        for &pb in &self.peer_bearing_ema {
            hasher.write_f32(pb);
        }
        for &am in &self.act_mean {
            hasher.write_f32(am);
        }
    }
}
