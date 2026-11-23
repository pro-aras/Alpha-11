//! General open-ended procedural operator bank and causal falsifier VM (`TAG_OPER`).
//!
//! Synthesizes sparse cue-triggered multi-step motor subroutines from mature,
//! non-quarantined, rising non-negative sensory pores, unlocks only on passive
//! early cue-contingent reinforcement, falsifies each candidate interventionally
//! against counterfactual null holds, archives contradicted skills (`ABAB`), and
//! enforces an STN preparatory motor hold during cue bursts. Zero hardcoded world
//! constants, zero string literals, zero `.unwrap()`, and strict `O(k)` per tick.

use a11_medium::ControlMode;

use crate::pores::PoreMembrane;
use crate::rng::{sanitize, sanitize_clamp, ParamHasher};
use crate::tubes::STRIATAL_PORES;

pub const OP_MAX_SLOTS: usize = 16;
pub const OP_MAX_ARCHIVE: usize = 24;
pub const OP_MAX_KEY_FEATS: usize = 24;
pub const OP_MAX_STEPS: usize = 5;
pub const OP_ACT_MAX: usize = 64;
pub const OP_MOTOR_DRIVE: f32 = 1.0;

const CUE_COS_MATCH: f32 = 0.62;
const CUE_DEDUP_COS: f32 = 0.72;
const CUE_RISING_DX_MIN: f32 = 0.22;
const CUE_RISING_NORM_MIN: f32 = 0.48;
const CUE_MIN_PEAK_NORM: f32 = 0.95;
const CUE_CH_MIN_VAL: f32 = 0.04;
const STEP_CONFIRM_INFLUX: f32 = 0.20;
const FALSIFY_MARGIN: f32 = 1.20;
const STRESS_REVERSAL_THRESH: f32 = 0.02;
const STRESS_OCCLUDED_THRESH: f32 = 0.08;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpStep {
    pub act: u16,
    pub dir: i8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorSlot {
    pub status: u8,
    pub key: Vec<(u32, f32)>,
    pub steps: Vec<OpStep>,
    pub exec_trials: u16,
    pub null_trials: u16,
    pub exec_rew_sum: f32,
    pub null_rew_sum: f32,
    pub confirmed_rew: f32,
    pub probe_step_cursor: u16,
    pub archive_eligible: bool,
    pub occluded_strikes: u8,
}

impl Default for OperatorSlot {
    fn default() -> Self {
        Self {
            status: 0,
            key: Vec::new(),
            steps: Vec::new(),
            exec_trials: 0,
            null_trials: 0,
            exec_rew_sum: 0.0,
            null_rew_sum: 0.0,
            confirmed_rew: 0.0,
            probe_step_cursor: 0,
            archive_eligible: false,
            occluded_strikes: 0,
        }
    }
}

impl OperatorSlot {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        36 + self.key.len() * 8 + self.steps.len() * 4
    }

    #[must_use]
    pub fn cosine_similarity(&self, obs_key: &[(u32, f32)]) -> f32 {
        cosine_sparse(&self.key, obs_key)
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.status);
        hasher.write_usize(self.key.len());
        for &(ch, w) in &self.key {
            hasher.write_u32(ch);
            hasher.write_f32(w);
        }
        hasher.write_usize(self.steps.len());
        for st in &self.steps {
            hasher.write_u32(st.act as u32);
            hasher.write_u8(st.dir as u8);
        }
        hasher.write_u32(self.exec_trials as u32);
        hasher.write_u32(self.null_trials as u32);
        hasher.write_f32(self.exec_rew_sum);
        hasher.write_f32(self.null_rew_sum);
        hasher.write_f32(self.confirmed_rew);
        hasher.write_u32(self.probe_step_cursor as u32);
        hasher.write_bool(self.archive_eligible);
        hasher.write_u8(self.occluded_strikes);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OperatorBank {
    pub slots: Vec<OperatorSlot>,
    pub archive: Vec<OperatorSlot>,
    pub op_domain_unlocked: bool,
    pub passive_early_hits: u16,

    // Transient per-bout runtime registers (excluded from hash_learned so
    // ControlMode::FrozenPlasticity executes promoted slots with 0-bit drift).
    pub in_burst: bool,
    pub b_tick: u16,
    pub halted: bool,
    pub bout_had_stress: bool,
    pub peak_cue_norm: f32,
    pub cue_accum: Vec<(u32, f32)>,
    pub burst_chs: Vec<u32>,
    pub active_slot: Option<u8>,
    pub step_pc: u8,
    pub active_mode: u8,
    pub pending_probe_act: u16,
    pub pending_probe_dir: i8,
    pub pending_probe_valid: bool,
    pub bout_rew: f32,
    pub post_burst_ticks: u16,
    pub settle_hold: bool,
    pub last_emitted_by_op: bool,
    pub prev2_emitted_by_op: bool,
}

impl Default for OperatorBank {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            archive: Vec::new(),
            op_domain_unlocked: false,
            passive_early_hits: 0,
            in_burst: false,
            b_tick: 0,
            halted: false,
            bout_had_stress: false,
            peak_cue_norm: 0.0,
            cue_accum: Vec::new(),
            burst_chs: Vec::new(),
            active_slot: None,
            step_pc: 0,
            active_mode: 0,
            pending_probe_act: 0,
            pending_probe_dir: 1,
            pending_probe_valid: false,
            bout_rew: 0.0,
            post_burst_ticks: 0,
            settle_hold: false,
            last_emitted_by_op: false,
            prev2_emitted_by_op: false,
        }
    }
}

impl OperatorBank {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let slot_b: usize = self.slots.iter().map(OperatorSlot::logical_bytes).sum();
        let arch_b: usize = self.archive.iter().map(OperatorSlot::logical_bytes).sum();
        let accum_b = self.cue_accum.len() * 8 + self.burst_chs.len() * 4;
        56 + slot_b + arch_b + accum_b
    }

    pub fn zero_learned(&mut self) {
        *self = Self::default();
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_usize(self.slots.len());
        for s in &self.slots {
            s.hash_learned(hasher);
        }
        hasher.write_usize(self.archive.len());
        for a in &self.archive {
            a.hash_learned(hasher);
        }
        hasher.write_bool(self.op_domain_unlocked);
        hasher.write_u32(self.passive_early_hits as u32);
    }

    pub fn scrub_quarantined(&mut self, quarantine_ch: &[u16]) {
        if quarantine_ch.is_empty() {
            return;
        }
        let mut active_removed = false;
        let active_idx = self.active_slot.map(|v| v as usize);
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if scrub_slot_key(slot, quarantine_ch) && Some(i) == active_idx {
                active_removed = true;
            }
        }
        if active_removed {
            self.active_slot = None;
            self.active_mode = 0;
            self.pending_probe_valid = false;
            self.halted = true;
        } else if let Some(aidx) = self.active_slot {
            let mut shift = 0usize;
            for (i, s) in self.slots.iter().enumerate() {
                if i < (aidx as usize) && s.status == 0 {
                    shift += 1;
                }
            }
            let new_idx = (aidx as usize).saturating_sub(shift);
            if new_idx < self.slots.len() && self.slots[new_idx].status >= 1 {
                self.active_slot = Some(new_idx as u8);
            } else {
                self.active_slot = None;
            }
        }
        self.slots.retain(|s| s.status >= 1);
        for arch in &mut self.archive {
            let _ = scrub_slot_key(arch, quarantine_ch);
        }
        self.archive.retain(|a| a.status >= 1);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_operator_vm(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        num_actuators: usize,
        membrane: &mut PoreMembrane,
        quarantine_ch: &[u16],
        pore_ch: &[u32; STRIATAL_PORES],
        pore_age: &[u32; STRIATAL_PORES],
        pore_var_raw: &[f32; STRIATAL_PORES],
        _pore_var_exaff: &[f32; STRIATAL_PORES],
        mode: ControlMode,
        seed: u64,
    ) -> Option<Vec<f32>> {
        if num_actuators == 0 {
            return None;
        }

        if mode == ControlMode::Natural {
            self.scrub_quarantined(quarantine_ch);
        }

        let clean_influx = if metabolic_influx.is_finite() {
            metabolic_influx.max(0.0)
        } else {
            0.0
        };
        let clean_stress = if tissue_stress.is_finite() {
            tissue_stress.max(0.0)
        } else {
            0.0
        };

        let na = num_actuators.min(OP_ACT_MAX).max(1);
        let min_pore_age = 46u32.max((3 * na) as u32 + 2);
        let has_any_op = self.op_domain_unlocked || self.slots.iter().any(|s| s.status >= 2);

        if !has_any_op && !self.in_burst && self.post_burst_ticks > 0 {
            if mode == ControlMode::Natural
                && (1..=5).contains(&self.post_burst_ticks)
                && clean_influx >= 0.25
                && clean_stress <= 0.01
            {
                self.passive_early_hits = self.passive_early_hits.saturating_add(1);
                self.post_burst_ticks = 0;
                if self.passive_early_hits >= 2 {
                    self.op_domain_unlocked = true;
                }
            } else {
                self.post_burst_ticks = self.post_burst_ticks.saturating_add(1);
                if self.post_burst_ticks > 5 {
                    self.post_burst_ticks = 0;
                }
            }
        } else if self.active_slot.is_some() && !self.in_burst {
            self.post_burst_ticks = self.post_burst_ticks.saturating_add(1);
            self.bout_rew = sanitize(self.bout_rew + clean_influx - clean_stress);

            let mut handled_stress = false;
            if let Some(sidx) = self.active_slot.map(|v| v as usize)
                && sidx < self.slots.len()
            {
                if self.active_mode == 1 && self.pending_probe_valid {
                    self.pending_probe_valid = false;
                    let slot = &mut self.slots[sidx];
                    if clean_influx >= STEP_CONFIRM_INFLUX && clean_stress <= 0.01 {
                        if slot.steps.len() < OP_MAX_STEPS {
                            slot.steps.push(OpStep {
                                act: self.pending_probe_act,
                                dir: self.pending_probe_dir,
                            });
                        }
                        slot.confirmed_rew = self.bout_rew;
                        slot.probe_step_cursor = 0;
                        slot.occluded_strikes = 0;
                        self.step_pc = slot.steps.len() as u8;
                        if slot.steps.len() >= OP_MAX_STEPS || clean_influx >= 1.5 {
                            self.halted = true;
                        }
                    } else if clean_stress >= STRESS_OCCLUDED_THRESH {
                        slot.probe_step_cursor = slot.probe_step_cursor.saturating_add(1);
                        if !self.bout_had_stress {
                            slot.occluded_strikes = slot.occluded_strikes.saturating_add(1);
                            self.bout_had_stress = true;
                        }
                        handled_stress = true;
                        self.halted = true;
                    } else {
                        slot.probe_step_cursor = slot.probe_step_cursor.saturating_add(1);
                        if (slot.probe_step_cursor as usize) >= 2 * na {
                            slot.status = 0;
                            self.halted = true;
                        } else {
                            self.settle_hold = true;
                        }
                    }
                }
                if !handled_stress
                    && clean_stress >= STRESS_OCCLUDED_THRESH
                    && self.active_mode == 1
                    && !self.bout_had_stress
                    && (self.last_emitted_by_op || self.prev2_emitted_by_op)
                {
                    self.bout_had_stress = true;
                    self.halted = true;
                    self.slots[sidx].occluded_strikes =
                        self.slots[sidx].occluded_strikes.saturating_add(1);
                }
            }

            if self.active_mode == 2
                && clean_stress > STRESS_REVERSAL_THRESH
                && (self.last_emitted_by_op || self.prev2_emitted_by_op)
            {
                self.bout_had_stress = true;
                self.halted = true;
            }
        }

        let has_any_op_now = self.op_domain_unlocked || self.slots.iter().any(|s| s.status >= 2);
        let result = self.step_state_machine(
            ambient_in,
            membrane,
            quarantine_ch,
            pore_ch,
            pore_age,
            pore_var_raw,
            min_pore_age,
            clean_influx,
            num_actuators,
            na,
            has_any_op_now,
            mode,
            seed,
        );

        self.prev2_emitted_by_op = self.last_emitted_by_op;
        self.last_emitted_by_op = result
            .as_ref()
            .is_some_and(|v| v.iter().any(|&x| x.abs() > 1.0e-4));
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn step_state_machine(
        &mut self,
        ambient_in: &[f32],
        membrane: &PoreMembrane,
        quarantine_ch: &[u16],
        pore_ch: &[u32; STRIATAL_PORES],
        pore_age: &[u32; STRIATAL_PORES],
        pore_var_raw: &[f32; STRIATAL_PORES],
        min_pore_age: u32,
        clean_influx: f32,
        num_actuators: usize,
        na: usize,
        has_any_op: bool,
        mode: ControlMode,
        seed: u64,
    ) -> Option<Vec<f32>> {
        if !self.in_burst {
            let blocked_onset =
                self.last_emitted_by_op || self.prev2_emitted_by_op || clean_influx > 0.05;
            if !blocked_onset {
                let (rising_feats, rising_norm, inst_norm) = extract_rising_nonneg_candidates(
                    ambient_in,
                    membrane,
                    quarantine_ch,
                    pore_ch,
                    pore_age,
                    pore_var_raw,
                    min_pore_age,
                );
                if rising_feats.len() >= 2 && rising_norm >= CUE_RISING_NORM_MIN {
                    if mode == ControlMode::Natural && has_any_op {
                        self.finalize_previous_bout();
                    } else {
                        self.active_slot = None;
                        self.active_mode = 0;
                        self.pending_probe_valid = false;
                    }
                    self.in_burst = true;
                    self.b_tick = 1;
                    self.halted = false;
                    self.bout_had_stress = false;
                    self.settle_hold = false;
                    self.post_burst_ticks = 0;
                    self.peak_cue_norm = inst_norm;
                    self.cue_accum.clear();
                    self.burst_chs.clear();
                    for &(ch, v) in &rising_feats {
                        if self.burst_chs.len() < OP_MAX_KEY_FEATS && !self.burst_chs.contains(&ch)
                        {
                            self.burst_chs.push(ch);
                        }
                        accumulate_one(&mut self.cue_accum, ch, v);
                    }
                    self.step_pc = 0;
                    self.bout_rew = 0.0;

                    return if has_any_op {
                        Some(vec![0.0_f32; num_actuators])
                    } else {
                        None
                    };
                }
            }

            if !has_any_op {
                return None;
            }
            return self.continue_bout(num_actuators, na, seed);
        }

        if self.b_tick == 1 {
            let (rising_feats, _, _) = extract_rising_nonneg_candidates(
                ambient_in,
                membrane,
                quarantine_ch,
                pore_ch,
                pore_age,
                pore_var_raw,
                min_pore_age,
            );
            for &(ch, _) in &rising_feats {
                if self.burst_chs.len() < OP_MAX_KEY_FEATS && !self.burst_chs.contains(&ch) {
                    self.burst_chs.push(ch);
                }
            }
        }

        let burst_inst_norm = measure_burst_channels(
            ambient_in,
            &self.burst_chs,
            &mut self.cue_accum,
        );
        if burst_inst_norm > self.peak_cue_norm {
            self.peak_cue_norm = burst_inst_norm;
        }

        let burst_ended =
            self.b_tick >= 3 && burst_inst_norm < (0.03 * self.peak_cue_norm).max(0.02);
        let tonic_abort = self.b_tick >= 12 && burst_inst_norm >= 0.15;

        if tonic_abort {
            self.in_burst = false;
            self.b_tick = 0;
            self.cue_accum.clear();
            self.burst_chs.clear();
            self.active_slot = None;
            self.active_mode = 0;
            self.post_burst_ticks = 0;
            return None;
        }

        if burst_ended {
            self.in_burst = false;
            self.b_tick = 0;
            self.burst_chs.clear();
            let obs_key = normalize_cue_key(&self.cue_accum);
            let valid_cue = obs_key.len() >= 2 && self.peak_cue_norm >= CUE_MIN_PEAK_NORM;
            self.cue_accum.clear();

            if !valid_cue {
                self.active_slot = None;
                self.active_mode = 0;
                self.post_burst_ticks = 0;
                return None;
            }

            if !has_any_op {
                self.post_burst_ticks = 1;
                return None;
            }

            self.post_burst_ticks = 0;
            self.bind_slot_for_cue(&obs_key, mode);
            return self.continue_bout(num_actuators, na, seed);
        }

        self.b_tick = self.b_tick.saturating_add(1);
        if has_any_op {
            Some(vec![0.0_f32; num_actuators])
        } else {
            None
        }
    }

    fn bind_slot_for_cue(&mut self, obs_key: &[(u32, f32)], mode: ControlMode) {
        self.active_slot = None;
        self.active_mode = 0;
        self.step_pc = 0;
        self.halted = false;
        self.bout_had_stress = false;
        self.bout_rew = 0.0;
        self.pending_probe_valid = false;
        self.settle_hold = false;

        let mut best_promo: Option<(usize, f32)> = None;
        let mut best_cand: Option<(usize, f32)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.status == 0 {
                continue;
            }
            let sim = slot.cosine_similarity(obs_key);
            if sim >= CUE_COS_MATCH {
                if slot.status >= 2 {
                    if best_promo.is_none_or(|(_, b)| sim > b) {
                        best_promo = Some((i, sim));
                    }
                } else if mode == ControlMode::Natural && best_cand.is_none_or(|(_, b)| sim > b) {
                    best_cand = Some((i, sim));
                }
            }
        }

        if let Some((idx, _)) = best_promo {
            if mode == ControlMode::Natural {
                blend_sparse_key(&mut self.slots[idx].key, obs_key, 0.10);
                let promo_key = self.slots[idx].key.clone();
                for (j, other) in self.slots.iter_mut().enumerate() {
                    if j != idx
                        && other.status == 1
                        && other.cosine_similarity(&promo_key) >= CUE_DEDUP_COS
                    {
                        other.status = 0;
                    }
                }
                if self.slots.iter().any(|s| s.status == 0) {
                    let mut shift = 0usize;
                    for (j, s) in self.slots.iter().enumerate() {
                        if j < idx && s.status == 0 {
                            shift += 1;
                        }
                    }
                    self.slots.retain(|s| s.status >= 1);
                    let new_idx = idx.saturating_sub(shift);
                    self.active_slot = Some(new_idx as u8);
                } else {
                    self.active_slot = Some(idx as u8);
                }
            } else {
                self.active_slot = Some(idx as u8);
            }
            self.active_mode = 2;
            return;
        }

        if mode != ControlMode::Natural {
            return;
        }

        if let Some((idx, _)) = best_cand {
            blend_sparse_key(&mut self.slots[idx].key, obs_key, 0.15);
            let slot = &self.slots[idx];
            if slot.steps.len() < OP_MAX_STEPS {
                if slot.occluded_strikes >= 2 {
                    return;
                }
                self.active_slot = Some(idx as u8);
                self.active_mode = 1;
            } else {
                self.active_slot = Some(idx as u8);
                self.active_mode = if slot.exec_trials <= slot.null_trials {
                    2
                } else {
                    3
                };
            }
            return;
        }

        let mut best_arch: Option<(usize, f32)> = None;
        for (i, arch) in self.archive.iter().enumerate() {
            if arch.archive_eligible && arch.status >= 2 {
                let sim = arch.cosine_similarity(obs_key);
                if sim >= CUE_COS_MATCH && best_arch.is_none_or(|(_, b)| sim > b) {
                    best_arch = Some((i, sim));
                }
            }
        }
        if let Some((aidx, _)) = best_arch
            && self.slots.len() < OP_MAX_SLOTS
        {
            let mut restored = self.archive.remove(aidx);
            restored.archive_eligible = false;
            restored.status = 2;
            blend_sparse_key(&mut restored.key, obs_key, 0.10);
            let new_idx = self.slots.len();
            self.slots.push(restored);
            self.active_slot = Some(new_idx as u8);
            self.active_mode = 2;
            return;
        }

        if self.slots.len() < OP_MAX_SLOTS {
            let slot = OperatorSlot {
                status: 1,
                key: obs_key.to_vec(),
                ..OperatorSlot::default()
            };
            let new_idx = self.slots.len();
            self.slots.push(slot);
            self.active_slot = Some(new_idx as u8);
            self.active_mode = 1;
        }
    }

    fn continue_bout(&mut self, num_actuators: usize, na: usize, seed: u64) -> Option<Vec<f32>> {
        let Some(sidx) = self.active_slot.map(|v| v as usize) else {
            return None;
        };
        if sidx >= self.slots.len() {
            self.active_slot = None;
            return None;
        }

        if self.halted {
            if self.bout_had_stress {
                return Some(vec![0.0_f32; num_actuators]);
            }
            if self.slots[sidx].status >= 2 {
                return None;
            }
            return Some(vec![0.0_f32; num_actuators]);
        }

        if self.active_mode == 3 {
            self.step_pc = self.step_pc.saturating_add(1);
            if (self.step_pc as usize) >= OP_MAX_STEPS {
                self.halted = true;
            }
            return Some(vec![0.0_f32; num_actuators]);
        }

        let pc = self.step_pc as usize;
        if self.active_mode == 2 {
            let steps_len = self.slots[sidx].steps.len();
            if pc < steps_len {
                let st = self.slots[sidx].steps[pc];
                let mut out = vec![0.0_f32; num_actuators];
                if (st.act as usize) < num_actuators {
                    out[st.act as usize] = (st.dir as f32) * OP_MOTOR_DRIVE;
                }
                self.step_pc = self.step_pc.saturating_add(1);
                if (self.step_pc as usize) >= steps_len {
                    self.halted = true;
                }
                return Some(out);
            }
            self.halted = true;
            return None;
        }

        if self.active_mode == 1 {
            if self.post_burst_ticks > 8 {
                self.settle_hold = false;
                self.halted = true;
                return Some(vec![0.0_f32; num_actuators]);
            }
            if self.settle_hold {
                self.settle_hold = false;
                return Some(vec![0.0_f32; num_actuators]);
            }
            let steps_len = self.slots[sidx].steps.len();
            if pc < steps_len {
                let st = self.slots[sidx].steps[pc];
                let mut out = vec![0.0_f32; num_actuators];
                if (st.act as usize) < num_actuators {
                    out[st.act as usize] = (st.dir as f32) * OP_MOTOR_DRIVE;
                }
                self.step_pc = self.step_pc.saturating_add(1);
                return Some(out);
            }
            if steps_len < OP_MAX_STEPS {
                let cursor = self.slots[sidx].probe_step_cursor as usize;
                let key_salt = self.slots[sidx].key.first().map_or(0u64, |&(c, _)| c as u64);
                let (act, dir) = probe_action_for_cursor(cursor, na, seed, key_salt);
                self.pending_probe_act = act;
                self.pending_probe_dir = dir;
                self.pending_probe_valid = true;
                let mut out = vec![0.0_f32; num_actuators];
                if (act as usize) < num_actuators {
                    out[act as usize] = (dir as f32) * OP_MOTOR_DRIVE;
                }
                return Some(out);
            }
            self.halted = true;
            return Some(vec![0.0_f32; num_actuators]);
        }

        None
    }

    fn finalize_previous_bout(&mut self) {
        let Some(sidx) = self.active_slot.take().map(|v| v as usize) else {
            self.active_mode = 0;
            self.pending_probe_valid = false;
            self.settle_hold = false;
            return;
        };
        if sidx >= self.slots.len() {
            self.active_mode = 0;
            self.pending_probe_valid = false;
            self.settle_hold = false;
            return;
        }

        let mode_was = self.active_mode;
        self.active_mode = 0;
        self.pending_probe_valid = false;
        self.settle_hold = false;

        if mode_was == 2 {
            if self.slots[sidx].status >= 2 {
                if self.bout_had_stress {
                    let mut archived = self.slots[sidx].clone();
                    archived.archive_eligible = false;
                    self.slots[sidx].status = 0;
                    for a in &mut self.archive {
                        if a.cosine_similarity(&archived.key) >= CUE_COS_MATCH {
                            a.archive_eligible = true;
                        }
                    }
                    if self.archive.len() >= OP_MAX_ARCHIVE {
                        self.archive.remove(0);
                    }
                    self.archive.push(archived);
                }
            } else if self.slots[sidx].status == 1 {
                let slot = &mut self.slots[sidx];
                if self.bout_had_stress {
                    slot.status = 0;
                } else {
                    slot.exec_trials = slot.exec_trials.saturating_add(1);
                    slot.exec_rew_sum = sanitize(slot.exec_rew_sum + self.bout_rew);
                    if slot.exec_trials >= 2 && slot.null_trials >= 2 {
                        let mean_exec = slot.exec_rew_sum / (slot.exec_trials as f32);
                        let mean_null = slot.null_rew_sum / (slot.null_trials as f32);
                        if mean_exec > mean_null + FALSIFY_MARGIN {
                            slot.status = 2;
                            self.op_domain_unlocked = true;
                        } else {
                            slot.status = 0;
                        }
                    }
                }
            }
        } else if mode_was == 3 && self.slots[sidx].status == 1 {
            let slot = &mut self.slots[sidx];
            slot.null_trials = slot.null_trials.saturating_add(1);
            slot.null_rew_sum = sanitize(slot.null_rew_sum + self.bout_rew);
            if slot.exec_trials >= 2 && slot.null_trials >= 2 {
                let mean_exec = slot.exec_rew_sum / (slot.exec_trials as f32);
                let mean_null = slot.null_rew_sum / (slot.null_trials as f32);
                if mean_exec > mean_null + FALSIFY_MARGIN {
                    slot.status = 2;
                    self.op_domain_unlocked = true;
                } else {
                    slot.status = 0;
                }
            }
        }

        self.slots.retain(|s| s.status >= 1);
    }
}

fn extract_rising_nonneg_candidates(
    ambient_in: &[f32],
    membrane: &PoreMembrane,
    quarantine_ch: &[u16],
    pore_ch: &[u32; STRIATAL_PORES],
    pore_age: &[u32; STRIATAL_PORES],
    pore_var_raw: &[f32; STRIATAL_PORES],
    min_pore_age: u32,
) -> (Vec<(u32, f32)>, f32, f32) {
    let mut feats: Vec<(u32, f32)> = Vec::new();
    let mut rise_sq = 0.0_f32;
    let mut inst_sq = 0.0_f32;
    let amb_len = ambient_in.len();

    for p in 0..STRIATAL_PORES {
        let ch = pore_ch[p];
        let ch_u = ch as usize;
        if ch == u32::MAX || ch_u >= amb_len {
            continue;
        }
        if pore_age[p] < min_pore_age || pore_var_raw[p] <= 1.0e-5 {
            continue;
        }
        if is_ch_quarantined(ch, quarantine_ch) {
            continue;
        }
        let raw_x = ambient_in[ch_u];
        if !raw_x.is_finite() || raw_x <= CUE_RISING_DX_MIN {
            continue;
        }

        let mut sp_dx = 0.0_f32;
        let mut ok = false;
        for sp in &membrane.sensor_pores {
            if sp.alive && sp.channel == ch {
                if sp.history.iter().all(|&hv| hv >= -1.0e-4) {
                    let mut min_hist = f32::INFINITY;
                    for &hv in &sp.history {
                        if hv < min_hist {
                            min_hist = hv;
                        }
                    }
                    if min_hist < 0.05 {
                        let idx_prev = ((sp.hist_head as usize) + 14) & 15;
                        sp_dx = raw_x - sp.history[idx_prev];
                        if sp_dx > CUE_RISING_DX_MIN {
                            ok = true;
                        }
                    }
                }
                break;
            }
        }
        if !ok {
            continue;
        }

        if let Some(pos) = feats.iter().position(|(c, _)| *c == ch) {
            if raw_x > feats[pos].1 {
                feats[pos].1 = raw_x;
            }
        } else if feats.len() < OP_MAX_KEY_FEATS {
            feats.push((ch, raw_x));
            rise_sq += sp_dx * sp_dx;
            inst_sq += raw_x * raw_x;
        }
    }

    (feats, sanitize(rise_sq.sqrt()), sanitize(inst_sq.sqrt()))
}

fn measure_burst_channels(
    ambient_in: &[f32],
    burst_chs: &[u32],
    cue_accum: &mut Vec<(u32, f32)>,
) -> f32 {
    let mut sq = 0.0_f32;
    for &ch in burst_chs {
        let ch_u = ch as usize;
        if ch_u >= ambient_in.len() {
            continue;
        }
        let x = ambient_in[ch_u];
        if x.is_finite() && x > CUE_CH_MIN_VAL {
            sq += x * x;
            accumulate_one(cue_accum, ch, x);
        }
    }
    sanitize(sq.sqrt())
}

#[inline]
fn is_ch_quarantined(ch: u32, quarantine_ch: &[u16]) -> bool {
    if ch > u16::MAX as u32 {
        return false;
    }
    let c16 = ch as u16;
    quarantine_ch.contains(&c16)
}

fn scrub_slot_key(slot: &mut OperatorSlot, quarantine_ch: &[u16]) -> bool {
    let prev_len = slot.key.len();
    slot.key.retain(|&(ch, _)| !is_ch_quarantined(ch, quarantine_ch));
    if slot.key.len() == prev_len {
        return false;
    }
    let norm = l2_norm_sparse(&slot.key);
    if slot.key.len() < 2 || norm < 1.0e-4 {
        slot.status = 0;
        slot.key.clear();
        return true;
    }
    let inv = 1.0 / norm;
    for (_, w) in &mut slot.key {
        *w = sanitize(*w * inv);
    }
    false
}

fn accumulate_one(accum: &mut Vec<(u32, f32)>, ch: u32, val: f32) {
    if let Some(pos) = accum.iter().position(|(c, _)| *c == ch) {
        accum[pos].1 = sanitize(accum[pos].1 + val);
    } else if accum.len() < OP_MAX_KEY_FEATS {
        accum.push((ch, val));
    }
}

fn normalize_cue_key(accum: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut peak = 0.0_f32;
    for &(_, v) in accum {
        if v > peak {
            peak = v;
        }
    }
    if peak <= 1.0e-5 {
        return Vec::new();
    }
    let cut = 0.08 * peak;
    let mut out: Vec<(u32, f32)> = accum
        .iter()
        .copied()
        .filter(|&(_, v)| v > cut && v.is_finite())
        .collect();
    out.sort_unstable_by_key(|&(ch, _)| ch);
    if out.len() > OP_MAX_KEY_FEATS {
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(OP_MAX_KEY_FEATS);
        out.sort_unstable_by_key(|&(ch, _)| ch);
    }
    let norm = l2_norm_sparse(&out);
    if norm <= 1.0e-5 {
        return Vec::new();
    }
    let inv = 1.0 / norm;
    for (_, w) in &mut out {
        *w = sanitize(*w * inv);
    }
    out
}

fn blend_sparse_key(dst: &mut Vec<(u32, f32)>, obs: &[(u32, f32)], alpha: f32) {
    let keep = 1.0 - alpha;
    for (_, w) in dst.iter_mut() {
        *w = sanitize(*w * keep);
    }
    for &(ch, ow) in obs {
        if let Some(pos) = dst.iter().position(|(c, _)| *c == ch) {
            dst[pos].1 = sanitize(dst[pos].1 + alpha * ow);
        } else if dst.len() < OP_MAX_KEY_FEATS {
            dst.push((ch, sanitize(alpha * ow)));
        }
    }
    dst.sort_unstable_by_key(|&(ch, _)| ch);
    let norm = l2_norm_sparse(dst);
    if norm > 1.0e-5 {
        let inv = 1.0 / norm;
        for (_, w) in dst.iter_mut() {
            *w = sanitize(*w * inv);
        }
    }
}

#[inline]
fn l2_norm_sparse(v: &[(u32, f32)]) -> f32 {
    let mut s = 0.0_f32;
    for &(_, x) in v {
        s += x * x;
    }
    sanitize(s.sqrt())
}

fn cosine_sparse(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut dot = 0.0_f32;
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let ca = a[i].0;
        let cb = b[j].0;
        if ca == cb {
            dot += a[i].1 * b[j].1;
            i += 1;
            j += 1;
        } else if ca < cb {
            i += 1;
        } else {
            j += 1;
        }
    }
    sanitize_clamp(dot, 0.0, 1.0)
}

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn coprime_stride(n: usize) -> usize {
    if n <= 2 {
        return 1;
    }
    let target = ((n as f64) * 0.381_966_011_f64).round() as usize;
    let s = target.clamp(1, n - 1);
    for offset in 0..n {
        if s > offset {
            let cand_dn = s - offset;
            if cand_dn > 0 && gcd(cand_dn, n) == 1 {
                return cand_dn;
            }
        }
        let cand_up = (s + offset) % n;
        if cand_up > 0 && gcd(cand_up, n) == 1 {
            return cand_up;
        }
    }
    1
}

#[inline]
fn probe_action_for_cursor(cursor: usize, na: usize, seed: u64, key_salt: u64) -> (u16, i8) {
    let stride = coprime_stride(na);
    let idx = cursor % na;
    let start = ((seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ key_salt.wrapping_mul(0x517C_C1B7_2722_0A95))
        .rotate_left(17) as usize)
        % na;
    let act = ((start + idx * stride) % na) as u16;
    let dir = if (cursor / na) & 1 == 0 { 1i8 } else { -1i8 };
    (act, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pores::SensorPore;
    use crate::rng::Rng64;

    #[test]
    fn subset_cosine_and_quarantine_scrub_work() {
        let a = normalize_cue_key(&[(0, 1.0), (1, 1.0)]);
        let b = normalize_cue_key(&[(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]);
        let sim = cosine_sparse(&a, &b);
        assert!(sim >= CUE_COS_MATCH);

        let mut bank = OperatorBank::default();
        bank.slots.push(OperatorSlot {
            status: 1,
            key: vec![(0, 0.707), (99, 0.707)],
            ..OperatorSlot::default()
        });
        bank.scrub_quarantined(&[99]);
        assert!(bank.slots.is_empty());
    }

    #[test]
    fn stn_hold_activates_only_when_op_domain_unlocked_or_promoted() {
        let mut bank = OperatorBank::default();
        let mut rng = Rng64::from_seed(1);
        let mut membrane = PoreMembrane::new(8, 32, &mut rng);
        membrane.sensor_pores.clear();
        let mut amb = vec![0.0_f32; 8];
        amb[0] = 0.45;
        amb[1] = 0.45;
        for ch in 0..2u32 {
            let mut sp = SensorPore::newborn(ch, 1, 0, 0, 1.0);
            sp.age_ticks = 60;
            sp.history[0] = 0.06;
            sp.history[1] = 0.45;
            sp.hist_head = 2;
            membrane.sensor_pores.push(sp);
        }
        let mut pore_ch = [u32::MAX; STRIATAL_PORES];
        let mut pore_age = [0u32; STRIATAL_PORES];
        let mut pore_var_raw = [0.0f32; STRIATAL_PORES];
        let pore_var_exaff = [0.0f32; STRIATAL_PORES];
        pore_ch[0] = 0;
        pore_ch[1] = 1;
        pore_age[0] = 60;
        pore_age[1] = 60;
        pore_var_raw[0] = 0.1;
        pore_var_raw[1] = 0.1;

        let out_locked = bank.step_operator_vm(
            &amb,
            0.0,
            0.0,
            4,
            &mut membrane,
            &[],
            &pore_ch,
            &pore_age,
            &pore_var_raw,
            &pore_var_exaff,
            ControlMode::Natural,
            1,
        );
        assert!(bank.in_burst);
        assert!(out_locked.is_none());

        bank.in_burst = false;
        bank.op_domain_unlocked = true;
        let out_unlocked = bank.step_operator_vm(
            &amb,
            0.0,
            0.0,
            4,
            &mut membrane,
            &[],
            &pore_ch,
            &pore_age,
            &pore_var_raw,
            &pore_var_exaff,
            ControlMode::FrozenPlasticity,
            1,
        );
        assert!(bank.in_burst);
        assert_eq!(out_unlocked, Some(vec![0.0_f32; 4]));
    }
}
