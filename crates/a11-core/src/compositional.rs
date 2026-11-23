//! General modality-free sensorimotor mirror & bidirectional role-filler syntax cortex.
//!
//! Operates strictly on unlabelled numeric `ambient_in` slices and `actuator_out` pulses with zero
//! string literals, zero actuator-count splits, zero organ constant mirrors, zero modulo-64
//! aliasing, and bounded `O(k)` per-tick active work (`<= 64` active receptors, `32` rolling scout
//! channels/tick). On non-compositional physical/operator domains this subsystem stays passively
//! dormant (`returns None`, zero motor interference).

use a11_medium::ControlMode;

use crate::rng::{ParamHasher, sanitize, sanitize_clamp};

pub const COMP_MAX_SCAN_CHANNELS: usize = 4_096;
pub const COMP_ROLL_STRIDE: usize = 32;
pub const COMP_MAX_CANDIDATES: usize = 128;
pub const COMP_MAX_RECEPTORS: usize = 64;
pub const COMP_MAX_ACTUATORS: usize = 64;
pub const COMP_MAX_PROBES: usize = 128;
pub const COMP_MAX_PRIMS: usize = 32;
pub const COMP_MAX_FRAMES: usize = 36;
pub const COMP_MAX_PROTOS: usize = 16;
pub const COMP_MAX_LEXEMES: usize = 32;
pub const COMP_MAX_ARCHIVE: usize = 4;
pub const COMP_AUD_SLICES: usize = 10;

const BURST_HIGH_THRESH: f32 = 0.22;
const PASSIVE_MIN_TICKS: u16 = 16;
const PASSIVE_MAX_TICKS: u16 = 29;
const OBS_BOUTS_TARGET: u16 = 5;

const STAGE_PASSIVE: u8 = 0;
const STAGE_OBSERVE: u8 = 1;
const STAGE_BABBLE: u8 = 2;
const STAGE_COGNITIVE: u8 = 3;
const STAGE_DISABLED: u8 = 4;

const SLOT_UNBOUND: u8 = u8::MAX;
const STATUS_EMPTY: u8 = 0;
const STATUS_CANDIDATE: u8 = 1;
const STATUS_PROMOTED: u8 = 2;

const RESP_NONE: u8 = 0;
const RESP_COMP: u8 = 1;
const RESP_PROD: u8 = 2;

const EVAL_NONE: u8 = 0;
const EVAL_EXEC: u8 = 1;
const EVAL_NULL_AUDIT: u8 = 2;
const EVAL_POST_AUDIT: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelTrackStat {
    pub samples: u16,
    pub min_val: f32,
    pub max_val: f32,
    pub disqualified: bool,
}

impl Default for ChannelTrackStat {
    fn default() -> Self {
        Self {
            samples: 0,
            min_val: 4.0,
            max_val: 0.0,
            disqualified: false,
        }
    }
}

impl ChannelTrackStat {
    #[must_use]
    pub fn logical_bytes() -> usize {
        12
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeRecord {
    pub act: u16,
    pub dir: i8,
    pub pass1: Vec<(u16, f32)>,
    pub pass2: Vec<(u16, f32)>,
    pub t2_pass1: Vec<(u16, f32)>,
    pub t2_pass2: Vec<(u16, f32)>,
    pub pass1_done: bool,
    pub pass2_done: bool,
    pub retry_count: u8,
}

impl Default for ProbeRecord {
    fn default() -> Self {
        Self {
            act: 0,
            dir: 1,
            pass1: Vec::new(),
            pass2: Vec::new(),
            t2_pass1: Vec::new(),
            t2_pass2: Vec::new(),
            pass1_done: false,
            pass2_done: false,
            retry_count: 0,
        }
    }
}

impl ProbeRecord {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        16 + (self.pass1.len() + self.pass2.len() + self.t2_pass1.len() + self.t2_pass2.len()) * 8
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotorPrim {
    pub wire_len: u8,
    pub wire_act: [u16; 2],
    pub wire_dir: [i8; 2],
    pub reaff: Vec<f32>,
}

impl Default for MotorPrim {
    fn default() -> Self {
        Self {
            wire_len: 0,
            wire_act: [0; 2],
            wire_dir: [1; 2],
            reaff: Vec::new(),
        }
    }
}

impl MotorPrim {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        12 + self.reaff.len() * 4
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.wire_len);
        for i in 0..(self.wire_len as usize).min(2) {
            hasher.write_u32(u32::from(self.wire_act[i]));
            hasher.write_u8(self.wire_dir[i] as u8);
        }
        hasher.write_usize(self.reaff.len());
        for &v in &self.reaff {
            hasher.write_f32(v);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudProto {
    pub phon_pair: [u8; 2],
    pub vec: Vec<f32>,
    pub count: u16,
}

impl AudProto {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        8 + self.vec.len() * 4
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.phon_pair[0]);
        hasher.write_u8(self.phon_pair[1]);
        hasher.write_u16(self.count);
        hasher.write_usize(self.vec.len());
        for &v in &self.vec {
            hasher.write_f32(v);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LexemeSlot {
    pub phon_pair: [u8; 2],
    pub role_slot: u8,
    pub bound_manip: u8,
    pub probe_cursor: u8,
    pub status: u8,
    pub exec_trials: u8,
    pub null_trials: u8,
    pub exec_rew_sum: f32,
    pub null_rew_sum: f32,
    pub post_promo_audited: bool,
}

impl Default for LexemeSlot {
    fn default() -> Self {
        Self {
            phon_pair: [0, 0],
            role_slot: 0,
            bound_manip: SLOT_UNBOUND,
            probe_cursor: 0,
            status: STATUS_EMPTY,
            exec_trials: 0,
            null_trials: 0,
            exec_rew_sum: 0.0,
            null_rew_sum: 0.0,
            post_promo_audited: false,
        }
    }
}

impl LexemeSlot {
    #[must_use]
    pub fn logical_bytes() -> usize {
        20
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.phon_pair[0]);
        hasher.write_u8(self.phon_pair[1]);
        hasher.write_u8(self.role_slot);
        hasher.write_u8(self.bound_manip);
        hasher.write_u8(self.probe_cursor);
        hasher.write_u8(self.status);
        hasher.write_u8(self.exec_trials);
        hasher.write_u8(self.null_trials);
        hasher.write_f32(self.exec_rew_sum);
        hasher.write_f32(self.null_rew_sum);
        hasher.write_bool(self.post_promo_audited);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrammarSnapshot {
    pub eligible: bool,
    pub lexemes: Vec<LexemeSlot>,
}

impl Default for GrammarSnapshot {
    fn default() -> Self {
        Self {
            eligible: false,
            lexemes: Vec::new(),
        }
    }
}

impl GrammarSnapshot {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        8 + self.lexemes.len() * LexemeSlot::logical_bytes()
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_bool(self.eligible);
        hasher.write_usize(self.lexemes.len());
        for s in &self.lexemes {
            s.hash_learned(hasher);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompositionalCortex {
    // Persistent learned parameters (included in hash_learned)
    pub comp_domain_unlocked: bool,
    pub stage: u8,
    pub receptors: Vec<u16>,
    pub is_scene_rec: Vec<u8>,
    pub is_fast_rec: Vec<u8>,
    pub is_aud_rec: Vec<u8>,
    pub fast_centroid: Vec<f32>,
    pub fast_templates: Vec<Vec<f32>>,
    pub learned_utt_len: u8,
    pub learned_gap_start: u8,
    pub learned_gap_end: u8,
    pub sym_prims: Vec<MotorPrim>,
    pub manip_prims: Vec<MotorPrim>,
    pub aud_proto_s0: Vec<AudProto>,
    pub aud_proto_s1: Vec<AudProto>,
    pub lexemes: Vec<LexemeSlot>,
    pub grammar_archive: Vec<GrammarSnapshot>,
    pub yoked_contaminated: bool,
    pub comp_agency_ema: f32,
    pub verified_self_copy_chs: Vec<u16>,

    // Developmental discovery & segmentation buffers (serialized for mid-bout bit-identity,
    // excluded from hash_learned so FrozenPlasticity stays strictly 0-bit drift)
    pub multi_seg_hits: u8,
    pub roll_cursor: u16,
    pub track_stats: Vec<ChannelTrackStat>,
    pub cand_chs: Vec<u16>,
    pub in_passive_burst: bool,
    pub passive_burst_ticks: u16,
    pub passive_burst_had_influx: bool,
    pub burst_quiet_streak: u8,
    pub prev_cand_max: f32,
    pub burst_active_chs: Vec<u16>,
    pub passive_early_vec: Vec<(u16, f32)>,
    pub passive_late_vec: Vec<(u16, f32)>,

    pub obs_bouts_done: u16,
    pub in_obs_bout: bool,
    pub obs_idle_ticks: u16,
    pub obs_quiet_streak: u8,
    pub obs_max_abs: Vec<f32>,

    pub probes: Vec<ProbeRecord>,
    pub probe_idx: u16,
    pub probe_phase: u8,
    pub babble_prompt_active: bool,
    pub babble_quiet_streak: u8,
    pub babble_prompt_dur: u16,
    pub last_babble_prompt_dur: u16,
    pub ticks_since_prompt: u16,
    pub quiet_motor_streak: u16,
    pub babble_total_ticks: u16,
    pub post_probe_cool: u8,
    pub cur_probe_act: u16,
    pub cur_probe_dir: i8,
    pub cur_probe_resp: Vec<(u16, f32)>,
    pub cur_probe_base: Vec<(u16, f32)>,

    pub seg_ready: bool,
    pub scene_armed: bool,
    pub fast_armed: bool,
    pub aud_armed: bool,
    pub self_vocal_cool: u8,
    pub in_prompt_kind: u8,
    pub p_tick: u16,
    pub saw_fast: bool,
    pub saw_aud: bool,
    pub pending_bimodal_aud: bool,
    pub pending_bimodal_w: [[u8; 2]; 2],
    pub peak_aud_norm: f32,
    pub prev_scene_norm: f32,
    pub prev_fast_norm: f32,
    pub prev_aud_norm: f32,
    pub scene_accum: Vec<f32>,
    pub fast_buf: Vec<Vec<f32>>,
    pub aud_buf: Vec<Vec<f32>>,
    pub prev_rec_vals: Vec<f32>,
    pub last_emitted_kind: u8,
    pub last_emitted_idx: u8,

    pub in_response_kind: u8,
    pub resp_tick: u16,
    pub active_w: [[u8; 2]; 2],
    pub active_m: [u8; 2],
    pub active_high_cert: bool,
    pub eval_mode: u8,
    pub eval_halted: bool,
    pub eval_had_stress: bool,
    pub step0_passed: bool,
    pub step0_failed: bool,
    pub step1_failed: bool,
    pub probing_role0: bool,
    pub probing_role1: bool,
    pub bout_rew: f32,
    pub bout_stress_seen: bool,
    pub consecutive_stressed_bouts: u8,
    pub consecutive_syntax_mismatch: u8,
    pub stage3_bouts: u16,
}

impl Default for CompositionalCortex {
    fn default() -> Self {
        Self {
            comp_domain_unlocked: false,
            stage: STAGE_PASSIVE,
            receptors: Vec::new(),
            is_scene_rec: Vec::new(),
            is_fast_rec: Vec::new(),
            is_aud_rec: Vec::new(),
            fast_centroid: Vec::new(),
            fast_templates: Vec::new(),
            learned_utt_len: 0,
            learned_gap_start: 0,
            learned_gap_end: 0,
            sym_prims: Vec::new(),
            manip_prims: Vec::new(),
            aud_proto_s0: Vec::new(),
            aud_proto_s1: Vec::new(),
            lexemes: Vec::new(),
            grammar_archive: Vec::new(),
            yoked_contaminated: false,
            comp_agency_ema: 0.0,
            verified_self_copy_chs: Vec::new(),
            multi_seg_hits: 0,
            roll_cursor: 0,
            track_stats: Vec::new(),
            cand_chs: Vec::new(),
            in_passive_burst: false,
            passive_burst_ticks: 0,
            passive_burst_had_influx: false,
            burst_quiet_streak: 0,
            prev_cand_max: 0.0,
            burst_active_chs: Vec::new(),
            passive_early_vec: Vec::new(),
            passive_late_vec: Vec::new(),
            obs_bouts_done: 0,
            in_obs_bout: false,
            obs_idle_ticks: 0,
            obs_quiet_streak: 0,
            obs_max_abs: Vec::new(),
            probes: Vec::new(),
            probe_idx: 0,
            probe_phase: 0,
            babble_prompt_active: false,
            babble_quiet_streak: 0,
            babble_prompt_dur: 0,
            last_babble_prompt_dur: 25,
            ticks_since_prompt: 0,
            quiet_motor_streak: 0,
            babble_total_ticks: 0,
            post_probe_cool: 0,
            cur_probe_act: 0,
            cur_probe_dir: 1,
            cur_probe_resp: Vec::new(),
            cur_probe_base: Vec::new(),
            seg_ready: false,
            scene_armed: true,
            fast_armed: true,
            aud_armed: true,
            self_vocal_cool: 0,
            in_prompt_kind: RESP_NONE,
            p_tick: 0,
            saw_fast: false,
            saw_aud: false,
            pending_bimodal_aud: false,
            pending_bimodal_w: [[0; 2]; 2],
            peak_aud_norm: 0.0,
            prev_scene_norm: 0.0,
            prev_fast_norm: 0.0,
            prev_aud_norm: 0.0,
            scene_accum: Vec::new(),
            fast_buf: Vec::new(),
            aud_buf: Vec::new(),
            prev_rec_vals: Vec::new(),
            last_emitted_kind: 0,
            last_emitted_idx: 0,
            in_response_kind: RESP_NONE,
            resp_tick: 0,
            active_w: [[0; 2]; 2],
            active_m: [0; 2],
            active_high_cert: false,
            eval_mode: EVAL_NONE,
            eval_halted: false,
            eval_had_stress: false,
            step0_passed: false,
            step0_failed: false,
            step1_failed: false,
            probing_role0: false,
            probing_role1: false,
            bout_rew: 0.0,
            bout_stress_seen: false,
            consecutive_stressed_bouts: 0,
            consecutive_syntax_mismatch: 0,
            stage3_bouts: 0,
        }
    }
}

impl CompositionalCortex {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let mut sz = 112;
        sz += self.receptors.len() * 2;
        sz += self.is_scene_rec.len() + self.is_fast_rec.len() + self.is_aud_rec.len();
        sz += self.fast_centroid.len() * 4;
        for t in &self.fast_templates {
            sz += 4 + t.len() * 4;
        }
        for p in &self.sym_prims {
            sz += p.logical_bytes();
        }
        for p in &self.manip_prims {
            sz += p.logical_bytes();
        }
        for a in &self.aud_proto_s0 {
            sz += a.logical_bytes();
        }
        for a in &self.aud_proto_s1 {
            sz += a.logical_bytes();
        }
        sz += self.lexemes.len() * LexemeSlot::logical_bytes();
        for g in &self.grammar_archive {
            sz += g.logical_bytes();
        }
        sz += self.track_stats.len() * ChannelTrackStat::logical_bytes();
        sz += (self.cand_chs.len() + self.burst_active_chs.len() + self.verified_self_copy_chs.len()) * 2;
        sz += (self.passive_early_vec.len() + self.passive_late_vec.len() + self.cur_probe_base.len()) * 8;
        for pr in &self.probes {
            sz += pr.logical_bytes();
        }
        sz += self.cur_probe_resp.len() * 8;
        sz += (self.scene_accum.len() + self.prev_rec_vals.len() + self.obs_max_abs.len()) * 4;
        for f in &self.fast_buf {
            sz += 4 + f.len() * 4;
        }
        for f in &self.aud_buf {
            sz += 4 + f.len() * 4;
        }
        sz
    }

    pub fn zero_learned(&mut self) {
        *self = Self::default();
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_bool(self.comp_domain_unlocked);
        hasher.write_u8(self.stage);
        hasher.write_usize(self.receptors.len());
        for &r in &self.receptors {
            hasher.write_u16(r);
        }
        for &v in &self.is_scene_rec {
            hasher.write_u8(v);
        }
        for &v in &self.is_fast_rec {
            hasher.write_u8(v);
        }
        for &v in &self.is_aud_rec {
            hasher.write_u8(v);
        }
        hasher.write_usize(self.fast_centroid.len());
        for &v in &self.fast_centroid {
            hasher.write_f32(v);
        }
        hasher.write_usize(self.fast_templates.len());
        for t in &self.fast_templates {
            hasher.write_usize(t.len());
            for &v in t {
                hasher.write_f32(v);
            }
        }
        hasher.write_u8(self.learned_utt_len);
        hasher.write_u8(self.learned_gap_start);
        hasher.write_u8(self.learned_gap_end);
        hasher.write_usize(self.sym_prims.len());
        for p in &self.sym_prims {
            p.hash_learned(hasher);
        }
        hasher.write_usize(self.manip_prims.len());
        for p in &self.manip_prims {
            p.hash_learned(hasher);
        }
        hasher.write_usize(self.aud_proto_s0.len());
        for a in &self.aud_proto_s0 {
            a.hash_learned(hasher);
        }
        hasher.write_usize(self.aud_proto_s1.len());
        for a in &self.aud_proto_s1 {
            a.hash_learned(hasher);
        }
        hasher.write_usize(self.lexemes.len());
        for l in &self.lexemes {
            l.hash_learned(hasher);
        }
        hasher.write_usize(self.grammar_archive.len());
        for g in &self.grammar_archive {
            g.hash_learned(hasher);
        }
        hasher.write_bool(self.yoked_contaminated);
        hasher.write_f32(self.comp_agency_ema);
        hasher.write_usize(self.verified_self_copy_chs.len());
        for &c in &self.verified_self_copy_chs {
            hasher.write_u16(c);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_compositional(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        num_actuators: usize,
        last_motor_energy: f32,
        mode: ControlMode,
        seed: u64,
    ) -> Option<Vec<f32>> {
        if num_actuators == 0 || self.stage == STAGE_DISABLED {
            return None;
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

        if self.stage < STAGE_COGNITIVE {
            if mode != ControlMode::Natural {
                return None;
            }
            if last_motor_energy <= 1.0e-5 {
                self.quiet_motor_streak = self.quiet_motor_streak.saturating_add(1);
            } else {
                self.quiet_motor_streak = 0;
            }
            self.roll_scan_channels(ambient_in, clean_influx, self.stage == STAGE_PASSIVE);
        }

        match self.stage {
            STAGE_PASSIVE => {
                self.step_passive_unlock(ambient_in, clean_influx);
                None
            }
            STAGE_OBSERVE => self.step_observe(ambient_in, num_actuators),
            STAGE_BABBLE => self.step_babble(ambient_in, clean_influx, num_actuators),
            STAGE_COGNITIVE => self.step_cognitive(
                ambient_in,
                clean_influx,
                clean_stress,
                num_actuators,
                mode,
                seed,
            ),
            _ => None,
        }
    }

    fn roll_scan_channels(&mut self, ambient_in: &[f32], clean_influx: f32, passive_only: bool) {
        let cap_amb = ambient_in.len().min(COMP_MAX_SCAN_CHANNELS);
        if cap_amb == 0 {
            return;
        }
        if self.track_stats.len() < cap_amb {
            self.track_stats
                .resize(cap_amb, ChannelTrackStat::default());
        }
        let start = (self.roll_cursor as usize).min(cap_amb);
        let end = (start + COMP_ROLL_STRIDE).min(cap_amb);
        for c in start..end {
            self.update_track_stat(c, ambient_in[c], clean_influx, passive_only);
        }
        self.roll_cursor = if end >= cap_amb { 0 } else { end as u16 };
        self.cand_chs.retain(|&c| {
            let cu = c as usize;
            cu < self.track_stats.len() && !self.track_stats[cu].disqualified
        });
    }

    fn update_track_stat(
        &mut self,
        c: usize,
        raw_x: f32,
        clean_influx: f32,
        passive_only: bool,
    ) {
        if c >= self.track_stats.len() {
            return;
        }
        let x = sanitize_clamp(raw_x, -4.0, 4.0);
        let st = &mut self.track_stats[c];
        if st.disqualified {
            return;
        }
        if x < -1.0e-3 {
            st.disqualified = true;
            return;
        }
        if !passive_only && self.quiet_motor_streak < 4 {
            return;
        }
        if st.samples == 0 {
            st.min_val = x;
            st.max_val = x;
        } else {
            if x < st.min_val {
                st.min_val = x;
            }
            if x > st.max_val {
                st.max_val = x;
            }
        }
        st.samples = st.samples.saturating_add(1);
        if !passive_only && st.samples >= 24 && st.min_val > 0.025 {
            st.disqualified = true;
            return;
        }
        let allow_add = if passive_only {
            clean_influx <= 0.01
        } else {
            clean_influx < 0.05
        };
        if allow_add
            && st.samples >= 2
            && st.min_val <= 0.005
            && x > 0.12
            && !self.cand_chs.contains(&(c as u16))
            && self.cand_chs.len() < COMP_MAX_CANDIDATES
        {
            self.cand_chs.push(c as u16);
        }
    }

    fn eval_candidates_and_prune(&mut self, ambient_in: &[f32]) -> (f32, f32) {
        let mut vals = Vec::with_capacity(self.cand_chs.len());
        let track = &mut self.track_stats;
        self.cand_chs.retain(|&c| {
            let cu = c as usize;
            if cu >= ambient_in.len() {
                return false;
            }
            let x = sanitize_clamp(ambient_in[cu], -4.0, 4.0);
            if x < -1.0e-3 {
                if cu < track.len() {
                    track[cu].disqualified = true;
                }
                return false;
            }
            vals.push(x);
            true
        });
        if vals.is_empty() {
            return (0.0, 0.0);
        }
        vals.sort_by(|a, b| b.total_cmp(a));
        let max_val = vals[0];
        let sixth_val = if vals.len() >= 6 { vals[5] } else { 0.0 };
        (max_val, sixth_val)
    }

    fn record_burst_active_chs(&mut self, ambient_in: &[f32]) {
        for &c in &self.cand_chs {
            let cu = c as usize;
            if cu < ambient_in.len()
                && sanitize_clamp(ambient_in[cu], -4.0, 4.0) > 0.16
                && !self.burst_active_chs.contains(&c)
                && self.burst_active_chs.len() < COMP_MAX_CANDIDATES
            {
                self.burst_active_chs.push(c);
            }
        }
    }

    fn step_passive_unlock(&mut self, ambient_in: &[f32], clean_influx: f32) {
        let (_inst_max, sixth_val) = self.eval_candidates_and_prune(ambient_in);
        if !self.in_passive_burst {
            if self.prev_cand_max < 0.10 && sixth_val > 0.16 && clean_influx <= 0.01 {
                self.in_passive_burst = true;
                self.passive_burst_ticks = 1;
                self.passive_burst_had_influx = false;
                self.burst_quiet_streak = 0;
                self.burst_active_chs.clear();
                self.passive_early_vec.clear();
                self.passive_late_vec.clear();
                self.record_burst_active_chs(ambient_in);
            }
        } else {
            if self.passive_burst_ticks < 16 && clean_influx > 0.01 {
                self.passive_burst_had_influx = true;
            }
            if self.passive_burst_ticks > 30 {
                self.in_passive_burst = false;
                self.passive_burst_ticks = 0;
                self.burst_quiet_streak = 0;
                self.burst_active_chs.clear();
                self.passive_early_vec.clear();
                self.passive_late_vec.clear();
            } else if sixth_val < 0.08 {
                self.burst_quiet_streak = self.burst_quiet_streak.saturating_add(1);
                if self.burst_quiet_streak >= 2 {
                    let dur = self.passive_burst_ticks;
                    let had_rew = self.passive_burst_had_influx;
                    let active_n = self.burst_active_chs.len();
                    let early = std::mem::take(&mut self.passive_early_vec);
                    let late = std::mem::take(&mut self.passive_late_vec);
                    self.in_passive_burst = false;
                    self.passive_burst_ticks = 0;
                    self.burst_quiet_streak = 0;
                    self.burst_active_chs.clear();
                    if !had_rew
                        && (PASSIVE_MIN_TICKS..=PASSIVE_MAX_TICKS).contains(&dur)
                        && self.cand_chs.len() >= 10
                        && active_n >= 8
                        && !early.is_empty()
                        && !late.is_empty()
                        && sparse_cosine(&early, &late) < 0.92
                    {
                        self.multi_seg_hits = self.multi_seg_hits.saturating_add(1);
                        if self.multi_seg_hits >= 2 {
                            for st in &mut self.track_stats {
                                if !st.disqualified {
                                    st.samples = 0;
                                    st.min_val = 4.0;
                                    st.max_val = 0.0;
                                }
                            }
                            self.cand_chs.clear();
                            self.comp_domain_unlocked = true;
                            self.stage = STAGE_OBSERVE;
                            self.obs_bouts_done = 0;
                            self.in_obs_bout = false;
                            self.obs_idle_ticks = 0;
                            self.obs_quiet_streak = 0;
                        }
                    }
                }
            } else {
                self.burst_quiet_streak = 0;
                self.passive_burst_ticks = self.passive_burst_ticks.saturating_add(1);
                if self.passive_burst_ticks == 2
                    || (self.passive_early_vec.is_empty() && sixth_val > 0.16)
                {
                    (self.passive_early_vec, _) = self.capture_cand_sparse(ambient_in);
                }
                if self.passive_burst_ticks >= 14 && sixth_val > 0.14 {
                    (self.passive_late_vec, _) = self.capture_cand_sparse(ambient_in);
                }
                self.record_burst_active_chs(ambient_in);
            }
        }
        self.prev_cand_max = sixth_val;
    }

    fn step_observe(&mut self, ambient_in: &[f32], num_actuators: usize) -> Option<Vec<f32>> {
        let cap_amb = ambient_in.len().min(COMP_MAX_SCAN_CHANNELS);
        if self.obs_max_abs.len() < cap_amb {
            self.obs_max_abs.resize(cap_amb, 0.0_f32);
        }
        if self.quiet_motor_streak >= 4 {
            for c in 0..cap_amb {
                let a = ambient_in[c];
                let v = if a.is_finite() { a.abs().min(4.0) } else { 0.0 };
                if v > self.obs_max_abs[c] {
                    self.obs_max_abs[c] = v;
                }
            }
        }
        let (inst_max, _) = self.eval_candidates_and_prune(ambient_in);
        self.obs_idle_ticks = self.obs_idle_ticks.saturating_add(1);
        if !self.in_obs_bout {
            if inst_max > BURST_HIGH_THRESH {
                self.in_obs_bout = true;
                self.obs_idle_ticks = 0;
                self.obs_quiet_streak = 0;
            } else if self.obs_idle_ticks > 60 {
                self.stage = STAGE_DISABLED;
                return None;
            }
        } else if inst_max < 0.12 {
            self.obs_quiet_streak = self.obs_quiet_streak.saturating_add(1);
            if self.obs_quiet_streak >= 2 {
                self.in_obs_bout = false;
                self.obs_idle_ticks = 0;
                self.obs_quiet_streak = 0;
                self.obs_bouts_done = self.obs_bouts_done.saturating_add(1);
                if self.obs_bouts_done >= OBS_BOUTS_TARGET {
                    self.init_babble_queue(num_actuators);
                    self.stage = STAGE_BABBLE;
                }
            }
        } else {
            self.obs_quiet_streak = 0;
            if self.obs_idle_ticks > 45 {
                self.stage = STAGE_DISABLED;
                return None;
            }
        }
        Some(vec![0.0_f32; num_actuators])
    }

    fn init_babble_queue(&mut self, num_actuators: usize) {
        let na = num_actuators.min(COMP_MAX_ACTUATORS);
        self.probes.clear();
        for act in 0..na {
            for dir in [1_i8, -1_i8] {
                if self.probes.len() < COMP_MAX_PROBES {
                    self.probes.push(ProbeRecord {
                        act: act as u16,
                        dir,
                        ..ProbeRecord::default()
                    });
                }
            }
        }
        self.probe_idx = 0;
        self.probe_phase = 0;
        self.babble_prompt_active = false;
        self.babble_quiet_streak = 0;
        self.babble_prompt_dur = 0;
        self.last_babble_prompt_dur = 25;
        self.ticks_since_prompt = 0;
        self.babble_total_ticks = 0;
        self.post_probe_cool = 0;
        self.cur_probe_resp.clear();
        self.cur_probe_base.clear();
    }

    fn scan_all_channels_on_probe(&mut self, ambient_in: &[f32], clean_influx: f32) {
        let cap_amb = ambient_in.len().min(COMP_MAX_SCAN_CHANNELS);
        if self.track_stats.len() < cap_amb {
            self.track_stats
                .resize(cap_amb, ChannelTrackStat::default());
        }
        for c in 0..cap_amb {
            let x = sanitize_clamp(ambient_in[c], -4.0, 4.0);
            if c < self.obs_max_abs.len()
                && self.obs_max_abs[c] < 0.005
                && x < -0.15
                && !self.verified_self_copy_chs.contains(&(c as u16))
                && self.verified_self_copy_chs.len() < COMP_MAX_CANDIDATES
            {
                self.verified_self_copy_chs.push(c as u16);
            }
            let st = &mut self.track_stats[c];
            if st.disqualified {
                continue;
            }
            if x < -1.0e-3 || (st.samples >= 24 && st.min_val > 0.025) {
                st.disqualified = true;
                continue;
            }
            if st.min_val <= 0.005
                && x > 0.12
                && clean_influx < 0.05
                && !self.cand_chs.contains(&(c as u16))
                && self.cand_chs.len() < COMP_MAX_CANDIDATES
            {
                self.cand_chs.push(c as u16);
            }
        }
        self.cand_chs.retain(|&c| {
            let cu = c as usize;
            cu < self.track_stats.len() && !self.track_stats[cu].disqualified
        });
    }

    fn scrub_disqualified_probes(&mut self) {
        for pr in &mut self.probes {
            pr.pass1.retain(|&(ch, _)| {
                let cu = ch as usize;
                cu < self.track_stats.len() && !self.track_stats[cu].disqualified
            });
            pr.pass2.retain(|&(ch, _)| {
                let cu = ch as usize;
                cu < self.track_stats.len() && !self.track_stats[cu].disqualified
            });
            pr.t2_pass1.retain(|&(ch, _)| {
                let cu = ch as usize;
                cu < self.track_stats.len() && !self.track_stats[cu].disqualified
            });
            pr.t2_pass2.retain(|&(ch, _)| {
                let cu = ch as usize;
                cu < self.track_stats.len() && !self.track_stats[cu].disqualified
            });
        }
    }

    fn capture_cand_sparse(&self, ambient_in: &[f32]) -> (Vec<(u16, f32)>, f32) {
        let mut out = Vec::new();
        let mut sq = 0.0_f32;
        for &c in &self.cand_chs {
            let cu = c as usize;
            if cu < ambient_in.len() {
                let x = sanitize_clamp(ambient_in[cu], -4.0, 4.0);
                if x > 0.015 {
                    out.push((c, x));
                    sq += x * x;
                }
            }
        }
        (out, sanitize(sq.sqrt()))
    }

    fn step_babble(
        &mut self,
        ambient_in: &[f32],
        clean_influx: f32,
        num_actuators: usize,
    ) -> Option<Vec<f32>> {
        self.babble_total_ticks = self.babble_total_ticks.saturating_add(1);
        if self.babble_total_ticks > 3_600 {
            self.stage = STAGE_DISABLED;
            return None;
        }

        if self.probe_phase == 1 {
            self.scan_all_channels_on_probe(ambient_in, clean_influx);
            self.scrub_disqualified_probes();
            let mut resp = Vec::new();
            for &c in &self.cand_chs {
                let cu = c as usize;
                if cu < ambient_in.len() {
                    let raw = sanitize_clamp(ambient_in[cu], -4.0, 4.0);
                    let base = sparse_get(&self.cur_probe_base, c);
                    let v = sanitize(raw - 0.65 * base);
                    if v > 0.015 {
                        resp.push((c, v));
                    }
                }
            }
            self.cur_probe_resp = resp;
            self.probe_phase = 2;
            let (inst_max, _) = self.eval_candidates_and_prune(ambient_in);
            self.prev_cand_max = inst_max;
            return Some(vec![0.0_f32; num_actuators]);
        }

        if self.probe_phase == 2 {
            self.scan_all_channels_on_probe(ambient_in, clean_influx);
            self.scrub_disqualified_probes();
            let mut t2_resp = Vec::new();
            let mut sq2 = 0.0_f32;
            for &c in &self.cand_chs {
                let cu = c as usize;
                if cu < ambient_in.len() {
                    let raw = sanitize_clamp(ambient_in[cu], -4.0, 4.0);
                    let base = sparse_get(&self.cur_probe_base, c);
                    let v = sanitize(raw - 0.40 * base);
                    if v > 0.012 {
                        t2_resp.push((c, v));
                        sq2 += v * v;
                    }
                }
            }
            let nrm_t2 = sanitize(sq2.sqrt());
            let collision = clean_influx > 0.01 || nrm_t2 > 2.20;

            self.probe_phase = 0;
            self.post_probe_cool = 2;
            self.ticks_since_prompt = self.ticks_since_prompt.saturating_add(2);

            if !collision {
                self.cur_probe_resp.retain(|&(ch, _)| {
                    let cu = ch as usize;
                    cu < self.track_stats.len() && !self.track_stats[cu].disqualified
                });
                t2_resp.retain(|&(ch, _)| {
                    let cu = ch as usize;
                    cu < self.track_stats.len() && !self.track_stats[cu].disqualified
                });
                let idx = self.probe_idx as usize;
                if idx < self.probes.len() {
                    let pr = &mut self.probes[idx];
                    if !pr.pass1_done {
                        pr.pass1 = self.cur_probe_resp.clone();
                        pr.t2_pass1 = t2_resp;
                        pr.pass1_done = true;
                    } else if !pr.pass2_done {
                        pr.pass2 = self.cur_probe_resp.clone();
                        pr.t2_pass2 = t2_resp;
                        pr.pass2_done = true;
                        let n1 = sparse_norm(&pr.pass1);
                        let n2 = sparse_norm(&pr.pass2);
                        if pr.retry_count < 3
                            && (n1 >= 0.18 || n2 >= 0.18)
                            && sparse_cosine(&pr.pass1, &pr.pass2) < 0.90
                        {
                            pr.retry_count = pr.retry_count.saturating_add(1);
                            pr.pass1 = pr.pass2.clone();
                            pr.t2_pass1 = pr.t2_pass2.clone();
                            pr.pass2.clear();
                            pr.t2_pass2.clear();
                            pr.pass2_done = false;
                        }
                    }
                    self.advance_probe_cursor();
                }
            }
            self.cur_probe_resp.clear();
            self.cur_probe_base.clear();
            let (inst_max, _) = self.eval_candidates_and_prune(ambient_in);
            self.prev_cand_max = inst_max;
            return Some(vec![0.0_f32; num_actuators]);
        }

        let (inst_max, _) = self.eval_candidates_and_prune(ambient_in);
        if self.post_probe_cool > 0 {
            self.post_probe_cool -= 1;
        }

        if !self.babble_prompt_active {
            if self.post_probe_cool == 0 && inst_max >= 0.16 {
                self.babble_prompt_active = true;
                self.babble_quiet_streak = 0;
                self.babble_prompt_dur = 1;
                self.ticks_since_prompt = 0;
                self.prev_cand_max = inst_max;
                return Some(vec![0.0_f32; num_actuators]);
            }
            self.ticks_since_prompt = self.ticks_since_prompt.saturating_add(1);
        } else if inst_max < 0.10 {
            self.babble_quiet_streak = self.babble_quiet_streak.saturating_add(1);
            if self.babble_quiet_streak >= 2 {
                self.last_babble_prompt_dur = self.babble_prompt_dur;
                self.babble_prompt_active = false;
                self.babble_quiet_streak = 0;
                self.ticks_since_prompt = 1;
            } else {
                self.prev_cand_max = inst_max;
                return Some(vec![0.0_f32; num_actuators]);
            }
        } else {
            self.babble_prompt_dur = self.babble_prompt_dur.saturating_add(1);
            self.babble_quiet_streak = 0;
            self.prev_cand_max = inst_max;
            return Some(vec![0.0_f32; num_actuators]);
        }
        self.prev_cand_max = inst_max;

        let can_probe = if self.post_probe_cool == 0 {
            if self.last_babble_prompt_dur <= 13 {
                self.ticks_since_prompt <= 5 && inst_max < 0.08
            } else if self.last_babble_prompt_dur <= 25 {
                self.ticks_since_prompt == 1 && inst_max < 0.10
            } else {
                false
            }
        } else {
            false
        };
        if can_probe {
            let idx = self.probe_idx as usize;
            if idx < self.probes.len() {
                let (base, _) = self.capture_cand_sparse(ambient_in);
                self.cur_probe_base = base;
                let act = self.probes[idx].act;
                let dir = self.probes[idx].dir;
                self.cur_probe_act = act;
                self.cur_probe_dir = dir;
                self.probe_phase = 1;
                let mut out = vec![0.0_f32; num_actuators];
                if (act as usize) < num_actuators {
                    out[act as usize] = f32::from(dir);
                }
                return Some(out);
            }
        }

        Some(vec![0.0_f32; num_actuators])
    }

    fn advance_probe_cursor(&mut self) {
        let len = self.probes.len();
        for (i, pr) in self.probes.iter().enumerate() {
            if !pr.pass1_done {
                self.probe_idx = i as u16;
                return;
            }
        }
        for (i, pr) in self.probes.iter().enumerate() {
            if !pr.pass2_done && sparse_norm(&pr.pass1) >= 0.18 {
                self.probe_idx = i as u16;
                return;
            }
        }
        if len > 0 {
            self.finalize_babble();
        }
    }

    fn finalize_babble(&mut self) {
        self.scrub_disqualified_probes();

        let n_cand = self.cand_chs.len();
        if n_cand == 0 {
            self.stage = STAGE_DISABLED;
            return;
        }

        let mut valid_probes: Vec<(u16, i8, Vec<f32>)> = Vec::new();
        let mut cand_t1_vecs: Vec<Vec<f32>> = vec![Vec::new(); n_cand];
        let mut cand_t2_vecs: Vec<Vec<f32>> = vec![Vec::new(); n_cand];
        for pr in &self.probes {
            if !pr.pass1_done || !pr.pass2_done {
                continue;
            }
            let n1 = sparse_norm(&pr.pass1);
            let n2 = sparse_norm(&pr.pass2);
            if n1 < 0.18 || n2 < 0.18 || sparse_cosine(&pr.pass1, &pr.pass2) < 0.90 {
                continue;
            }
            let mut mean_t1 = vec![0.0_f32; n_cand];
            for (ci, &ch) in self.cand_chs.iter().enumerate() {
                let v1 = 0.5 * (sparse_get(&pr.pass1, ch) + sparse_get(&pr.pass2, ch));
                let v2 = 0.5 * (sparse_get(&pr.t2_pass1, ch) + sparse_get(&pr.t2_pass2, ch));
                mean_t1[ci] = v1;
                cand_t1_vecs[ci].push(v1);
                cand_t2_vecs[ci].push(v2);
            }
            valid_probes.push((pr.act, pr.dir, mean_t1));
        }

        if valid_probes.len() < 4 {
            self.stage = STAGE_DISABLED;
            return;
        }

        // Deduplicate physical wire clones using both t+1 max diff and robust t+2 median diff
        let mut keep_indices: Vec<usize> = Vec::new();
        for ci in 0..n_cand {
            let max_t1 = cand_t1_vecs[ci].iter().copied().fold(0.0_f32, f32::max);
            let max_t2 = cand_t2_vecs[ci].iter().copied().fold(0.0_f32, f32::max);
            if max_t1.max(max_t2) <= 0.05 {
                continue;
            }
            let mut dup_of_existing = false;
            for &kj in &keep_indices {
                let mut d1 = 0.0_f32;
                for (&a, &b) in cand_t1_vecs[ci].iter().zip(cand_t1_vecs[kj].iter()) {
                    let d = (a - b).abs();
                    if d > d1 {
                        d1 = d;
                    }
                }
                let mut diffs2: Vec<f32> = cand_t2_vecs[ci]
                    .iter()
                    .zip(cand_t2_vecs[kj].iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .collect();
                let d2_med = if diffs2.is_empty() {
                    0.0
                } else {
                    diffs2.sort_by(|a, b| a.total_cmp(b));
                    diffs2[diffs2.len() / 2]
                };
                if d1.max(d2_med) < 1.0e-3 {
                    dup_of_existing = true;
                    break;
                }
            }
            if !dup_of_existing && keep_indices.len() < COMP_MAX_RECEPTORS {
                keep_indices.push(ci);
            }
        }

        self.receptors = keep_indices.iter().map(|&ci| self.cand_chs[ci]).collect();
        let r_len = self.receptors.len();
        if r_len < 4 {
            self.stage = STAGE_DISABLED;
            return;
        }

        // Merge co-agonist motor wires on clean t+1 projection
        let mut merged: Vec<MotorPrim> = Vec::new();
        for (act, dir, mean_full) in &valid_probes {
            let mut proj = vec![0.0_f32; r_len];
            for (rj, &ci) in keep_indices.iter().enumerate() {
                proj[rj] = mean_full[ci];
            }
            if dense_norm(&proj) < 0.15 {
                continue;
            }
            let mut joined = false;
            for prim in &mut merged {
                if dense_cosine(&prim.reaff, &proj) >= 0.96 {
                    if prim.wire_len == 1 && !(prim.wire_act[0] == *act && prim.wire_dir[0] == *dir)
                    {
                        prim.wire_act[1] = *act;
                        prim.wire_dir[1] = *dir;
                        prim.wire_len = 2;
                        for (dst, &sv) in prim.reaff.iter_mut().zip(proj.iter()) {
                            *dst = 0.5 * (*dst + sv);
                        }
                    }
                    joined = true;
                    break;
                }
            }
            if !joined && merged.len() < COMP_MAX_PRIMS {
                merged.push(MotorPrim {
                    wire_len: 1,
                    wire_act: [*act, 0],
                    wire_dir: [*dir, 1],
                    reaff: proj,
                });
            }
        }

        self.sym_prims.clear();
        self.manip_prims.clear();
        for prim in merged {
            let support_cnt = prim.reaff.iter().filter(|&&v| v > 0.10).count();
            if support_cnt >= 4 {
                if self.sym_prims.len() < COMP_MAX_LEXEMES {
                    self.sym_prims.push(prim);
                }
            } else if support_cnt >= 1 && self.manip_prims.len() < COMP_MAX_LEXEMES {
                self.manip_prims.push(prim);
            }
        }

        if self.sym_prims.len() < 4 || self.manip_prims.len() < 4 {
            self.stage = STAGE_DISABLED;
            return;
        }

        // Outlier-immune receptor partition into Scene, Fast-impulse, and Auditory-resonant channels
        self.is_scene_rec = vec![0_u8; r_len];
        self.is_fast_rec = vec![0_u8; r_len];
        self.is_aud_rec = vec![0_u8; r_len];
        for (rj, &ci) in keep_indices.iter().enumerate() {
            let manip_max = self
                .manip_prims
                .iter()
                .map(|m| m.reaff[rj])
                .fold(0.0_f32, f32::max);
            if manip_max > 0.10 {
                self.is_scene_rec[rj] = 1;
                continue;
            }
            let mut act_cnt = 0usize;
            let mut imp_cnt = 0usize;
            for (&v1, &v2) in cand_t1_vecs[ci].iter().zip(cand_t2_vecs[ci].iter()) {
                if v1 > 0.18 {
                    act_cnt += 1;
                    if v2 <= 0.08 * v1 {
                        imp_cnt += 1;
                    }
                }
            }
            if act_cnt >= 2 && imp_cnt * 4 >= act_cnt * 3 {
                self.is_fast_rec[rj] = 1;
            } else {
                self.is_aud_rec[rj] = 1;
            }
        }

        let n_fast = self.is_fast_rec.iter().filter(|&&x| x == 1).count();
        let n_aud = self.is_aud_rec.iter().filter(|&&x| x == 1).count();
        if n_fast < 4 || n_aud < 4 {
            self.stage = STAGE_DISABLED;
            return;
        }

        let n_sym = self.sym_prims.len();
        self.fast_centroid = vec![0.0_f32; n_fast];
        self.fast_templates = vec![vec![0.0_f32; n_fast]; n_sym];
        let mut raw_fast: Vec<Vec<f32>> = vec![Vec::with_capacity(n_fast); n_sym];
        for (si, sp) in self.sym_prims.iter().enumerate() {
            for (rj, &flag) in self.is_fast_rec.iter().enumerate() {
                if flag == 1 {
                    raw_fast[si].push(sp.reaff[rj]);
                }
            }
            for (k, &v) in raw_fast[si].iter().enumerate() {
                self.fast_centroid[k] += v / (n_sym as f32);
            }
        }
        let sum_cent: f32 = self.fast_centroid.iter().sum::<f32>().max(1.0e-6);
        for si in 0..n_sym {
            let s_f: f32 = raw_fast[si].iter().sum();
            let alpha = s_f / sum_cent;
            let mut centered = vec![0.0_f32; n_fast];
            for k in 0..n_fast {
                centered[k] = raw_fast[si][k] - alpha * self.fast_centroid[k];
            }
            normalize_inplace(&mut centered);
            self.fast_templates[si] = centered;
        }

        self.track_stats.clear();
        self.probes.clear();
        self.cur_probe_resp.clear();
        self.cur_probe_base.clear();
        self.cand_chs.clear();
        self.obs_max_abs.clear();

        self.stage = STAGE_COGNITIVE;
        self.seg_ready = false;
        self.scene_armed = true;
        self.fast_armed = true;
        self.aud_armed = true;
        self.self_vocal_cool = 0;
        self.pending_bimodal_aud = false;
        self.prev_rec_vals = vec![0.0_f32; r_len];
        self.last_emitted_kind = 0;
        self.last_emitted_idx = 0;
        self.in_prompt_kind = RESP_NONE;
        self.in_response_kind = RESP_NONE;
    }

    fn read_receptor_views(
        &self,
        ambient_in: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32, f32, f32) {
        let mut cur_rec = Vec::with_capacity(self.receptors.len());
        let mut x_scene = Vec::with_capacity(self.receptors.len());
        let mut x_fast = Vec::with_capacity(self.receptors.len());
        let mut x_aud = Vec::with_capacity(self.receptors.len());
        let mut sq_s = 0.0_f32;
        let mut sq_f = 0.0_f32;
        let mut sq_a = 0.0_f32;
        for (rj, &ch) in self.receptors.iter().enumerate() {
            let cu = ch as usize;
            let val = if cu < ambient_in.len() {
                sanitize_clamp(ambient_in[cu], 0.0, 2.0)
            } else {
                0.0
            };
            cur_rec.push(val);
            x_scene.push(if self.is_scene_rec[rj] == 1 {
                sq_s += val * val;
                val
            } else {
                0.0
            });
            if self.is_fast_rec[rj] == 1 {
                x_fast.push(val);
                sq_f += val * val;
            }
            if self.is_aud_rec[rj] == 1 {
                x_aud.push(val);
                sq_a += val * val;
            }
        }
        (
            cur_rec,
            x_scene,
            x_fast,
            x_aud,
            sanitize(sq_s.sqrt()),
            sanitize(sq_f.sqrt()),
            sanitize(sq_a.sqrt()),
        )
    }

    fn update_comp_agency(&mut self, cur_rec_vals: &[f32], mode: ControlMode) {
        let kind = self.last_emitted_kind;
        let last_idx = self.last_emitted_idx as usize;
        self.last_emitted_kind = 0;

        if mode != ControlMode::Natural
            || self.yoked_contaminated
            || kind == 0
            || cur_rec_vals.len() != self.receptors.len()
            || self.prev_rec_vals.len() != self.receptors.len()
        {
            return;
        }

        let (prims, mask): (&[MotorPrim], &[u8]) = match kind {
            1 => (&self.manip_prims, &self.is_scene_rec),
            2 => (&self.sym_prims, &self.is_fast_rec),
            _ => return,
        };
        if prims.len() < 2 {
            return;
        }

        let prev_mod_max = mask
            .iter()
            .zip(self.prev_rec_vals.iter())
            .filter(|&(&m, _)| m == 1)
            .map(|(_, &v)| v.abs())
            .fold(0.0_f32, f32::max);
        if prev_mod_max >= 0.05 {
            return;
        }

        let idx = last_idx % prims.len();
        let cf_idx = (idx + 1) % prims.len();
        let scale = f32::from(prims[idx].wire_len.max(1));
        let mut err_null_sq = 0.0_f32;
        let mut err_act_sq = 0.0_f32;
        let mut err_cf_sq = 0.0_f32;
        for (rj, &m) in mask.iter().enumerate() {
            if m != 1 {
                continue;
            }
            let dx = cur_rec_vals[rj];
            let p = (scale * prims[idx].reaff[rj]).min(1.0);
            let c = (scale * prims[cf_idx].reaff[rj]).min(1.0);
            err_null_sq += dx * dx;
            err_act_sq += (dx - p) * (dx - p);
            err_cf_sq += (dx - c) * (dx - c);
        }
        if err_null_sq > 1.0e-3 {
            let base_sq = err_null_sq.min(err_cf_sq);
            let gain = sanitize_clamp((base_sq - err_act_sq) / (err_null_sq + 0.02), -0.50, 0.85);
            self.comp_agency_ema = sanitize_clamp(0.95 * self.comp_agency_ema + 0.05 * gain, 0.0, 1.0);
        }
    }

    #[inline]
    fn arm_triggers(&mut self, scene_norm: f32, fast_norm: f32, aud_norm: f32) {
        if scene_norm < 0.008 {
            self.scene_armed = true;
        }
        if fast_norm < 0.05 {
            self.fast_armed = true;
        }
        if self.self_vocal_cool == 0 && aud_norm < 0.12 {
            self.aud_armed = true;
        }
    }

    fn flush_bimodal_aud_if_ready(&mut self, x_aud: &[f32], mode: ControlMode) {
        if !self.pending_bimodal_aud {
            return;
        }
        let target_len = if self.learned_utt_len >= 16 {
            usize::from(self.learned_utt_len)
        } else {
            22
        };
        if self.aud_buf.len() < target_len && self.aud_buf.len() < COMP_MAX_FRAMES {
            self.aud_buf.push(x_aud.to_vec());
        }
        if self.aud_buf.len() >= target_len {
            if mode == ControlMode::Natural && target_len >= 12 {
                let gs = usize::from(self.learned_gap_start).max(4);
                let ge = usize::from(self.learned_gap_end).max(gs);
                let s0_hi = (gs + 1).min(target_len - 4);
                let s1_lo = ge.min(target_len - 4);
                if let Some(f0) = build_ten_slice_aud_feat(&self.aud_buf[..s0_hi]) {
                    update_aud_proto(&mut self.aud_proto_s0, self.pending_bimodal_w[0], &f0);
                }
                if let Some(f1) = build_ten_slice_aud_feat(&self.aud_buf[s1_lo..target_len]) {
                    update_aud_proto(&mut self.aud_proto_s1, self.pending_bimodal_w[1], &f1);
                }
            }
            self.pending_bimodal_aud = false;
            self.aud_buf.clear();
        }
    }

    fn step_cognitive(
        &mut self,
        ambient_in: &[f32],
        clean_influx: f32,
        clean_stress: f32,
        num_actuators: usize,
        mode: ControlMode,
        seed: u64,
    ) -> Option<Vec<f32>> {
        if self.self_vocal_cool > 0 {
            self.self_vocal_cool -= 1;
            self.aud_armed = false;
        }

        let (cur_rec_vals, x_scene, x_fast, x_aud, scene_norm, fast_norm, aud_norm) =
            self.read_receptor_views(ambient_in);
        self.update_comp_agency(&cur_rec_vals, mode);
        self.prev_rec_vals = cur_rec_vals;
        self.prev_scene_norm = scene_norm;
        self.prev_fast_norm = fast_norm;
        self.prev_aud_norm = aud_norm;

        if mode == ControlMode::Natural && self.yoked_contaminated {
            return Some(vec![0.0_f32; num_actuators]);
        }

        if self.in_response_kind != RESP_NONE {
            self.flush_bimodal_aud_if_ready(&x_aud, mode);
            let out = self.step_response(clean_influx, clean_stress, num_actuators, mode, seed);
            if self.in_response_kind == RESP_NONE {
                self.pending_bimodal_aud = false;
                self.aud_buf.clear();
                self.arm_triggers(scene_norm, fast_norm, aud_norm);
            }
            return Some(out);
        }

        if !self.seg_ready {
            if scene_norm < 0.008
                && fast_norm < 0.05
                && (aud_norm < 0.12 || self.self_vocal_cool > 0)
            {
                self.seg_ready = true;
                self.scene_armed = true;
                self.fast_armed = true;
                self.aud_armed = self.self_vocal_cool == 0;
            }
            return Some(vec![0.0_f32; num_actuators]);
        }

        if self.in_prompt_kind == RESP_NONE {
            self.arm_triggers(scene_norm, fast_norm, aud_norm);
            let scene_onset = self.scene_armed && scene_norm > 0.015;
            let fast_onset = self.fast_armed && fast_norm > 0.15;
            let aud_onset =
                fast_norm <= 0.05 && self.self_vocal_cool == 0 && self.aud_armed && aud_norm > 0.15;

            if scene_onset {
                self.scene_armed = false;
                self.fast_armed = false;
                self.aud_armed = false;
                if mode == ControlMode::Natural {
                    self.finalize_previous_bout(clean_influx, clean_stress);
                }
                self.in_prompt_kind = RESP_PROD;
                self.p_tick = 1;
                self.scene_accum = x_scene;
                return Some(vec![0.0_f32; num_actuators]);
            }

            if fast_onset || aud_onset {
                self.scene_armed = false;
                self.fast_armed = false;
                self.aud_armed = false;
                if mode == ControlMode::Natural {
                    self.finalize_previous_bout(clean_influx, clean_stress);
                }
                self.in_prompt_kind = RESP_COMP;
                self.saw_fast = fast_norm > 0.12;
                self.saw_aud = aud_norm > 0.15;
                self.peak_aud_norm = aud_norm;
                self.fast_buf.clear();
                self.aud_buf.clear();
                self.pending_bimodal_aud = false;
                if fast_onset {
                    self.fast_buf.push(x_fast);
                }
                if aud_norm > 0.15 {
                    self.aud_buf.push(x_aud);
                }
                self.p_tick = 1;
                return Some(vec![0.0_f32; num_actuators]);
            }

            return Some(vec![0.0_f32; num_actuators]);
        }

        if self.in_prompt_kind == RESP_PROD {
            if scene_norm < 0.005 && self.p_tick >= 4 {
                let (m_a, m_b) = self.extract_top2_scene_manips();
                self.in_prompt_kind = RESP_NONE;
                self.p_tick = 0;
                self.start_production_response(m_a, m_b, mode, seed);
                let out =
                    self.step_response(clean_influx, clean_stress, num_actuators, mode, seed);
                if self.in_response_kind == RESP_NONE {
                    self.arm_triggers(scene_norm, fast_norm, aud_norm);
                }
                return Some(out);
            }
            for (dst, &src) in self.scene_accum.iter_mut().zip(x_scene.iter()) {
                *dst = sanitize(*dst + src);
            }
            self.p_tick = self.p_tick.saturating_add(1);
            if self.p_tick >= 20 {
                self.in_prompt_kind = RESP_NONE;
                self.p_tick = 0;
            }
            return Some(vec![0.0_f32; num_actuators]);
        }

        // `in_prompt_kind == RESP_COMP`
        self.saw_fast |= fast_norm > 0.12;
        self.saw_aud |= aud_norm > 0.15;
        if aud_norm > self.peak_aud_norm {
            self.peak_aud_norm = aud_norm;
        }

        let target_len = if self.learned_utt_len >= 16 {
            usize::from(self.learned_utt_len)
        } else {
            22
        };
        let fast_done = self.saw_fast && self.p_tick >= 8 && fast_norm < 0.05;
        let aud_done = !self.saw_fast && self.aud_buf.len() >= target_len;

        if fast_done || aud_done || self.p_tick >= 26 {
            let saw_fast = self.saw_fast;
            let (w1, w2) = self.decode_utterance(mode);
            self.in_prompt_kind = RESP_NONE;
            self.p_tick = 0;
            self.fast_buf.clear();
            if !self.pending_bimodal_aud {
                self.aud_buf.clear();
            } else {
                self.flush_bimodal_aud_if_ready(&x_aud, mode);
            }
            self.start_comprehension_response(w1, w2, saw_fast, mode, seed);
            let out = self.step_response(clean_influx, clean_stress, num_actuators, mode, seed);
            if self.in_response_kind == RESP_NONE {
                self.pending_bimodal_aud = false;
                self.aud_buf.clear();
                self.arm_triggers(scene_norm, fast_norm, aud_norm);
            }
            return Some(out);
        }

        if self.saw_fast && self.fast_buf.len() < COMP_MAX_FRAMES {
            self.fast_buf.push(x_fast);
        }
        if self.saw_aud && self.aud_buf.len() < COMP_MAX_FRAMES {
            self.aud_buf.push(x_aud);
        }
        self.p_tick = self.p_tick.saturating_add(1);
        Some(vec![0.0_f32; num_actuators])
    }

    fn extract_top2_scene_manips(&self) -> (u8, u8) {
        let nm = self.manip_prims.len();
        if nm == 0 {
            return (0, 0);
        }
        let mut scored: Vec<(usize, f32)> = (0..nm)
            .map(|m| {
                let mut dot = 0.0_f32;
                for (&a, &b) in self.scene_accum.iter().zip(self.manip_prims[m].reaff.iter()) {
                    dot += a * b;
                }
                let nrm = dense_norm(&self.manip_prims[m].reaff).max(1.0e-5);
                (m, dot / nrm)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best_m = scored[0].0;
        let mut sec_m = if scored.len() > 1 { scored[1].0 } else { best_m };
        for &(cand, _) in scored.iter().skip(1) {
            if dense_cosine(&self.manip_prims[best_m].reaff, &self.manip_prims[cand].reaff) < 0.50 {
                sec_m = cand;
                break;
            }
        }
        (best_m as u8, sec_m as u8)
    }

    fn decode_utterance(&mut self, mode: ControlMode) -> ([u8; 2], [u8; 2]) {
        if self.saw_fast {
            let nf = self.fast_buf.len();
            if nf < 6 || self.fast_centroid.is_empty() || self.fast_templates.is_empty() {
                return ([0, 0], [0, 0]);
            }
            let sum_cent: f32 = self.fast_centroid.iter().sum::<f32>().max(1.0e-6);
            let n_fast = self.fast_centroid.len();
            let mut preds = vec![0usize; nf];
            let mut sims = vec![0.0_f32; nf];
            let mut norms = vec![0.0_f32; nf];

            for t in 0..nf {
                let f = &self.fast_buf[t];
                if f.len() != n_fast {
                    continue;
                }
                let sf: f32 = f.iter().sum();
                if sf < 1.0e-5 {
                    continue;
                }
                let alpha = sf / sum_cent;
                let mut cf = vec![0.0_f32; n_fast];
                let mut sq = 0.0_f32;
                for k in 0..n_fast {
                    let v = f[k] - alpha * self.fast_centroid[k];
                    cf[k] = v;
                    sq += v * v;
                }
                let nrm = sq.sqrt();
                norms[t] = nrm;
                if nrm < 1.0e-6 {
                    continue;
                }
                let mut best_p = 0usize;
                let mut best_c = f32::NEG_INFINITY;
                for (p, tmpl) in self.fast_templates.iter().enumerate() {
                    let mut d = 0.0_f32;
                    for k in 0..n_fast {
                        d += cf[k] * tmpl[k];
                    }
                    let c = d / nrm;
                    if c > best_c {
                        best_c = c;
                        best_p = p;
                    }
                }
                preds[t] = best_p;
                sims[t] = best_c;
            }

            let mut gap_start = nf / 2;
            let mut gap_end = gap_start;
            let mut found_gap = false;
            for t in 6..nf.saturating_sub(6) {
                if sims[t] < 0.72 {
                    if !found_gap {
                        gap_start = t;
                        found_gap = true;
                    }
                    gap_end = t + 1;
                } else if found_gap {
                    break;
                }
            }

            let pick_pair = |lo: usize, hi: usize| -> [u8; 2] {
                if hi <= lo + 1 {
                    let p = preds[lo.min(nf - 1)] as u8;
                    return [p, p];
                }
                let mid = lo + (hi - lo) / 2;
                let best_in = |a: usize, b: usize| -> u8 {
                    let mut bt = a;
                    let mut bv = f32::NEG_INFINITY;
                    for t in a..b {
                        let sc = sims[t] * norms[t];
                        if sc > bv {
                            bv = sc;
                            bt = t;
                        }
                    }
                    preds[bt] as u8
                };
                [best_in(lo, mid), best_in(mid, hi)]
            };

            let w1 = pick_pair(0, gap_start);
            let w2 = pick_pair(gap_end, nf);

            if mode == ControlMode::Natural && nf >= 16 {
                self.learned_utt_len = nf as u8;
                self.learned_gap_start = gap_start as u8;
                self.learned_gap_end = gap_end as u8;
                if self.saw_aud {
                    self.pending_bimodal_aud = true;
                    self.pending_bimodal_w = [w1, w2];
                }
            }

            return (w1, w2);
        }

        // Audio-Only comprehension decoding via length-matched 10-slice cochlear prototypes
        let na = self.aud_buf.len();
        if na < 12 {
            return ([0, 0], [0, 0]);
        }
        let gs = if self.learned_gap_start >= 4 {
            usize::from(self.learned_gap_start)
        } else {
            10
        };
        let ge = if usize::from(self.learned_gap_end) >= gs {
            usize::from(self.learned_gap_end)
        } else {
            12
        };

        let active_r0 = self.active_pairs_for_role(0);
        let active_r1 = self.active_pairs_for_role(1);

        let w1 = if let Some(q0) = build_ten_slice_aud_feat(&self.aud_buf[..(gs + 1).min(na - 4)]) {
            match_aud_proto(&self.aud_proto_s0, &q0, &active_r0)
        } else {
            [0, 0]
        };
        let w2 = if let Some(q1) = build_ten_slice_aud_feat(&self.aud_buf[ge.min(na - 4)..na]) {
            match_aud_proto(&self.aud_proto_s1, &q1, &active_r1)
        } else {
            [0, 0]
        };
        (w1, w2)
    }

    fn active_pairs_for_role(&self, role: u8) -> Vec<[u8; 2]> {
        let mut out: Vec<[u8; 2]> = Vec::new();
        for l in &self.lexemes {
            if l.role_slot == role
                && l.status == STATUS_PROMOTED
                && !out.contains(&l.phon_pair)
            {
                out.push(l.phon_pair);
            }
        }
        if out.len() < 4 {
            for l in &self.lexemes {
                if l.role_slot == role
                    && l.bound_manip != SLOT_UNBOUND
                    && !out.contains(&l.phon_pair)
                {
                    out.push(l.phon_pair);
                }
            }
        }
        if out.len() < 4 {
            out.clear();
        }
        out
    }

    fn find_lexeme_idx(&self, phon_pair: [u8; 2], role_slot: u8) -> Option<usize> {
        self.lexemes
            .iter()
            .position(|l| l.phon_pair == phon_pair && l.role_slot == role_slot)
    }

    fn ensure_lexeme_idx(&mut self, phon_pair: [u8; 2], role_slot: u8) -> Option<usize> {
        if let Some(idx) = self.find_lexeme_idx(phon_pair, role_slot) {
            return Some(idx);
        }
        if self.lexemes.len() >= COMP_MAX_LEXEMES {
            return None;
        }
        let idx = self.lexemes.len();
        self.lexemes.push(LexemeSlot {
            phon_pair,
            role_slot,
            ..LexemeSlot::default()
        });
        Some(idx)
    }

    fn promoted_count(&self) -> usize {
        self.lexemes
            .iter()
            .filter(|l| l.status == STATUS_PROMOTED)
            .count()
    }

    fn has_promoted_in_archive_for(&self, w1: [u8; 2], w2: [u8; 2]) -> Option<usize> {
        for (i, snap) in self.grammar_archive.iter().enumerate().rev() {
            if !snap.eligible {
                continue;
            }
            let ok0 = snap.lexemes.iter().any(|l| {
                l.phon_pair == w1 && l.role_slot == 0 && l.status == STATUS_PROMOTED
            });
            let ok1 = snap.lexemes.iter().any(|l| {
                l.phon_pair == w2 && l.role_slot == 1 && l.status == STATUS_PROMOTED
            });
            if ok0 && ok1 {
                return Some(i);
            }
        }
        None
    }

    fn mark_archive_eligible_if_ready(&mut self) {
        let bound_cnt = self
            .lexemes
            .iter()
            .filter(|l| l.bound_manip != SLOT_UNBOUND)
            .count();
        if bound_cnt >= 2 {
            for arch in &mut self.grammar_archive {
                if arch.lexemes.len() >= 4 {
                    arch.eligible = true;
                }
            }
        }
    }

    fn try_context_switch(&mut self, preferred_comp: Option<([u8; 2], [u8; 2])>) {
        let promoted: Vec<LexemeSlot> = self
            .lexemes
            .iter()
            .filter(|l| l.status == STATUS_PROMOTED)
            .copied()
            .collect();
        for arch in &mut self.grammar_archive {
            if arch.lexemes.len() >= 4 {
                arch.eligible = true;
            }
        }
        let target_idx = if let Some((w1, w2)) = preferred_comp {
            self.has_promoted_in_archive_for(w1, w2)
        } else {
            self.grammar_archive.iter().rposition(|g| g.eligible)
        };
        let restored = if let Some(ti) = target_idx {
            let mut g = self.grammar_archive.remove(ti);
            for slot in &mut g.lexemes {
                slot.post_promo_audited = true;
            }
            Some(g.lexemes)
        } else {
            None
        };
        if promoted.len() >= 4 {
            if self.grammar_archive.len() >= COMP_MAX_ARCHIVE {
                self.grammar_archive.remove(0);
            }
            self.grammar_archive.push(GrammarSnapshot {
                eligible: false,
                lexemes: promoted,
            });
        }
        self.lexemes = restored.unwrap_or_default();
        self.consecutive_stressed_bouts = 0;
        self.consecutive_syntax_mismatch = 0;
    }

    fn seeded_manip_candidate(
        &self,
        seed: u64,
        phon_pair: [u8; 2],
        role: u8,
        cursor: u8,
        avoid: Option<u8>,
    ) -> u8 {
        let nm = self.manip_prims.len();
        if nm == 0 {
            return 0;
        }
        let mut h = seed
            ^ ((u64::from(phon_pair[0]) + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((u64::from(phon_pair[1]) + 3).wrapping_mul(0xBF58_476D_1CE4_E5B9))
            ^ (u64::from(role) + 7).wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 30;
        let offset = (h as usize) % nm;

        let mut candidates: Vec<u8> = Vec::with_capacity(nm);
        for k in 0..nm {
            let m = ((offset + k) % nm) as u8;
            if Some(m) == avoid {
                continue;
            }
            let already_taken = self
                .lexemes
                .iter()
                .any(|l| l.bound_manip == m && l.status >= STATUS_CANDIDATE);
            if !already_taken {
                candidates.push(m);
            }
        }
        for k in 0..nm {
            let m = ((offset + k) % nm) as u8;
            if Some(m) != avoid && !candidates.contains(&m) {
                candidates.push(m);
            }
        }
        if candidates.is_empty() {
            (offset % nm) as u8
        } else {
            candidates[(cursor as usize) % candidates.len()]
        }
    }

    fn choose_eval_mode(&self, i0: usize, i1: usize, seed: u64) -> u8 {
        let s0 = self.lexemes[i0];
        let s1 = self.lexemes[i1];
        if s0.status == STATUS_PROMOTED && s1.status == STATUS_PROMOTED {
            if !s0.post_promo_audited || !s1.post_promo_audited {
                EVAL_POST_AUDIT
            } else {
                EVAL_EXEC
            }
        } else {
            let tot = (s0.exec_trials as u64).wrapping_add(s0.null_trials as u64)
                + (s1.exec_trials as u64).wrapping_add(s1.null_trials as u64);
            let need_null = (s0.status == STATUS_CANDIDATE && s0.null_trials < s0.exec_trials)
                || (s1.status == STATUS_CANDIDATE && s1.null_trials < s1.exec_trials);
            let parity = ((seed ^ u64::from(self.stage3_bouts) ^ tot) & 1) == 1;
            let force_null = need_null
                || ((s0.status == STATUS_CANDIDATE && s0.null_trials < 2
                    || s1.status == STATUS_CANDIDATE && s1.null_trials < 2)
                    && parity);
            if force_null {
                EVAL_NULL_AUDIT
            } else {
                EVAL_EXEC
            }
        }
    }

    fn start_comprehension_response(
        &mut self,
        w1: [u8; 2],
        w2: [u8; 2],
        saw_fast: bool,
        mode: ControlMode,
        seed: u64,
    ) {
        self.in_response_kind = RESP_COMP;
        self.resp_tick = 0;
        self.active_w = [w1, w2];
        self.active_high_cert = saw_fast;
        self.eval_mode = EVAL_NONE;
        self.eval_halted = false;
        self.eval_had_stress = false;
        self.step0_passed = false;
        self.step0_failed = false;
        self.step1_failed = false;
        self.probing_role0 = false;
        self.probing_role1 = false;
        self.bout_rew = 0.0;
        self.bout_stress_seen = false;

        if mode != ControlMode::Natural || !saw_fast {
            let m0 = self
                .find_lexeme_idx(w1, 0)
                .filter(|&i| self.lexemes[i].status == STATUS_PROMOTED)
                .map_or(SLOT_UNBOUND, |i| self.lexemes[i].bound_manip);
            let m1 = self
                .find_lexeme_idx(w2, 1)
                .filter(|&i| self.lexemes[i].status == STATUS_PROMOTED)
                .map_or(SLOT_UNBOUND, |i| self.lexemes[i].bound_manip);
            self.active_m = [m0, m1];
            if mode == ControlMode::Natural {
                self.stage3_bouts = self.stage3_bouts.saturating_add(1);
            }
            if m0 == SLOT_UNBOUND || m1 == SLOT_UNBOUND {
                self.eval_halted = true;
            }
            return;
        }

        self.stage3_bouts = self.stage3_bouts.saturating_add(1);

        let cur_r0_bound = self
            .find_lexeme_idx(w1, 0)
            .is_some_and(|i| self.lexemes[i].bound_manip != SLOT_UNBOUND);
        let cur_r1_bound = self
            .find_lexeme_idx(w2, 1)
            .is_some_and(|i| self.lexemes[i].bound_manip != SLOT_UNBOUND);
        let opp_clash = self
            .find_lexeme_idx(w1, 1)
            .is_some_and(|i| self.lexemes[i].status == STATUS_PROMOTED)
            || self
                .find_lexeme_idx(w2, 0)
                .is_some_and(|i| self.lexemes[i].status == STATUS_PROMOTED);
        let has_arch = self.has_promoted_in_archive_for(w1, w2).is_some();

        if !cur_r0_bound && !cur_r1_bound && has_arch {
            self.try_context_switch(Some((w1, w2)));
        } else if (!cur_r0_bound || !cur_r1_bound)
            && (self.promoted_count() >= 6 || has_arch)
            && (opp_clash || has_arch)
        {
            self.consecutive_syntax_mismatch =
                self.consecutive_syntax_mismatch.saturating_add(1);
            if self
                .consecutive_stressed_bouts
                .saturating_add(self.consecutive_syntax_mismatch)
                >= 2
            {
                self.try_context_switch(if has_arch { Some((w1, w2)) } else { None });
            } else {
                self.eval_halted = true;
                return;
            }
        } else if cur_r0_bound && cur_r1_bound {
            self.consecutive_syntax_mismatch = 0;
        }

        let Some(i0) = self.ensure_lexeme_idx(w1, 0) else {
            self.eval_halted = true;
            return;
        };
        let Some(i1) = self.ensure_lexeme_idx(w2, 1) else {
            self.eval_halted = true;
            return;
        };

        let s0_bound = self.lexemes[i0].bound_manip;
        let s1_bound = self.lexemes[i1].bound_manip;

        if s0_bound == SLOT_UNBOUND {
            let m0 = self.seeded_manip_candidate(
                seed,
                w1,
                0,
                self.lexemes[i0].probe_cursor,
                None,
            );
            self.active_m = [m0, SLOT_UNBOUND];
            self.probing_role0 = true;
            return;
        }

        if s1_bound == SLOT_UNBOUND {
            let m1 = self.seeded_manip_candidate(
                seed,
                w2,
                1,
                self.lexemes[i1].probe_cursor,
                Some(s0_bound),
            );
            self.active_m = [s0_bound, m1];
            self.probing_role1 = true;
            return;
        }

        self.active_m = [s0_bound, s1_bound];
        self.eval_mode = self.choose_eval_mode(i0, i1, seed);
    }

    fn start_production_response(&mut self, m_a: u8, m_b: u8, mode: ControlMode, seed: u64) {
        self.in_response_kind = RESP_PROD;
        self.resp_tick = 0;
        self.eval_mode = EVAL_NONE;
        self.eval_halted = false;
        self.eval_had_stress = false;
        self.active_high_cert = true;
        self.bout_rew = 0.0;
        self.bout_stress_seen = false;
        self.active_m = [m_a, m_b];

        let req_promoted = mode != ControlMode::Natural;
        let find_role_for_pair = |role: u8| -> Option<(usize, [u8; 2])> {
            for (idx, l) in self.lexemes.iter().enumerate() {
                if l.role_slot == role
                    && (l.bound_manip == m_a || l.bound_manip == m_b)
                    && (l.status == STATUS_PROMOTED
                        || (!req_promoted && l.status == STATUS_CANDIDATE))
                {
                    return Some((idx, l.phon_pair));
                }
            }
            None
        };

        let Some((i0, w1)) = find_role_for_pair(0) else {
            self.eval_halted = true;
            return;
        };
        let Some((i1, w2)) = find_role_for_pair(1) else {
            self.eval_halted = true;
            return;
        };
        self.active_w = [w1, w2];
        self.active_m = [self.lexemes[i0].bound_manip, self.lexemes[i1].bound_manip];
        if mode == ControlMode::Natural {
            self.stage3_bouts = self.stage3_bouts.saturating_add(1);
            self.eval_mode = self.choose_eval_mode(i0, i1, seed);
        }
    }

    fn emit_prim(&self, prim: &MotorPrim, num_actuators: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; num_actuators];
        for i in 0..(prim.wire_len as usize).min(2) {
            let a = prim.wire_act[i] as usize;
            if a < num_actuators {
                out[a] = f32::from(prim.wire_dir[i]);
            }
        }
        out
    }

    fn step_response(
        &mut self,
        clean_influx: f32,
        clean_stress: f32,
        num_actuators: usize,
        mode: ControlMode,
        seed: u64,
    ) -> Vec<f32> {
        if self.resp_tick > 0 {
            self.bout_rew = sanitize(self.bout_rew + clean_influx - clean_stress);
            if clean_stress > 0.02 {
                self.bout_stress_seen = true;
            }
        }

        let mut out = vec![0.0_f32; num_actuators];

        if self.in_response_kind == RESP_COMP {
            if mode == ControlMode::Natural && self.resp_tick == 1 {
                if self.probing_role0 {
                    if clean_influx >= 0.25 && clean_stress <= 0.01 {
                        self.step0_passed = true;
                        if let Some(i0) = self.find_lexeme_idx(self.active_w[0], 0) {
                            self.lexemes[i0].bound_manip = self.active_m[0];
                            self.lexemes[i0].status = STATUS_CANDIDATE;
                            self.lexemes[i0].exec_trials = 0;
                            self.lexemes[i0].null_trials = 0;
                            self.lexemes[i0].exec_rew_sum = 0.0;
                            self.lexemes[i0].null_rew_sum = 0.0;
                        }
                        self.mark_archive_eligible_if_ready();
                        self.probing_role0 = false;
                        if let Some(i1) = self.find_lexeme_idx(self.active_w[1], 1) {
                            let s1_b = self.lexemes[i1].bound_manip;
                            if s1_b == SLOT_UNBOUND {
                                let m1 = self.seeded_manip_candidate(
                                    seed,
                                    self.active_w[1],
                                    1,
                                    self.lexemes[i1].probe_cursor,
                                    Some(self.active_m[0]),
                                );
                                self.active_m[1] = m1;
                                self.probing_role1 = true;
                            } else {
                                self.active_m[1] = s1_b;
                                self.eval_mode = EVAL_EXEC;
                            }
                        }
                    } else {
                        self.step0_failed = true;
                        if let Some(i0) = self.find_lexeme_idx(self.active_w[0], 0) {
                            self.lexemes[i0].probe_cursor =
                                self.lexemes[i0].probe_cursor.saturating_add(1);
                        }
                        self.eval_halted = true;
                    }
                } else if self.probing_role1 || self.eval_mode == EVAL_EXEC {
                    if clean_stress > 0.02 {
                        self.step0_failed = true;
                        self.eval_had_stress = true;
                        self.eval_halted = true;
                    } else if clean_influx >= 0.25 {
                        self.step0_passed = true;
                    }
                }
            } else if mode == ControlMode::Natural && self.resp_tick == 3 {
                if self.probing_role1 {
                    if clean_influx >= 1.50 && clean_stress <= 0.01 {
                        if let Some(i1) = self.find_lexeme_idx(self.active_w[1], 1) {
                            self.lexemes[i1].bound_manip = self.active_m[1];
                            self.lexemes[i1].status = STATUS_CANDIDATE;
                            self.lexemes[i1].exec_trials = 1;
                            self.lexemes[i1].null_trials = 0;
                            self.lexemes[i1].exec_rew_sum = sanitize(self.bout_rew);
                            self.lexemes[i1].null_rew_sum = 0.0;
                        }
                        self.mark_archive_eligible_if_ready();
                        self.probing_role1 = false;
                    } else {
                        self.step1_failed = true;
                        if let Some(i1) = self.find_lexeme_idx(self.active_w[1], 1) {
                            self.lexemes[i1].probe_cursor =
                                self.lexemes[i1].probe_cursor.saturating_add(1);
                        }
                        self.eval_halted = true;
                    }
                } else if self.eval_mode == EVAL_EXEC && clean_stress > 0.02 {
                    self.step1_failed = true;
                    self.eval_had_stress = true;
                    self.eval_halted = true;
                }
            }

            if !self.eval_halted && (self.eval_mode == EVAL_NONE || self.eval_mode == EVAL_EXEC) {
                if self.resp_tick == 0 {
                    let m0 = self.active_m[0] as usize;
                    if m0 < self.manip_prims.len() {
                        out = self.emit_prim(&self.manip_prims[m0].clone(), num_actuators);
                        self.last_emitted_kind = 1;
                        self.last_emitted_idx = m0 as u8;
                    }
                } else if self.resp_tick == 2 {
                    let m1 = self.active_m[1] as usize;
                    if m1 < self.manip_prims.len() {
                        out = self.emit_prim(&self.manip_prims[m1].clone(), num_actuators);
                        self.last_emitted_kind = 1;
                        self.last_emitted_idx = m1 as u8;
                    }
                }
            }

            self.resp_tick = self.resp_tick.saturating_add(1);
            if self.resp_tick >= 5 {
                self.in_response_kind = RESP_NONE;
            }
            return out;
        }

        // `in_response_kind == RESP_PROD`
        let dense_sym = !self.sym_prims.is_empty() && self.sym_prims.iter().all(|p| p.wire_len >= 2);
        if mode == ControlMode::Natural && self.resp_tick >= 1 && self.resp_tick < 8 && clean_stress > 0.02 {
            self.eval_had_stress = true;
            self.eval_halted = true;
        }
        let emit_slot = if dense_sym {
            (self.resp_tick < 4).then_some(self.resp_tick as usize)
        } else if self.resp_tick < 8 && (self.resp_tick & 1) == 0 {
            Some((self.resp_tick / 2) as usize)
        } else {
            None
        };
        if !self.eval_halted
            && (self.eval_mode == EVAL_NONE || self.eval_mode == EVAL_EXEC)
            && let Some(slot_i) = emit_slot
        {
            let seq = [
                self.active_w[0][0],
                self.active_w[0][1],
                self.active_w[1][0],
                self.active_w[1][1],
            ];
            let k = seq[slot_i] as usize;
            if k < self.sym_prims.len() {
                out = self.emit_prim(&self.sym_prims[k].clone(), num_actuators);
                self.last_emitted_kind = 2;
                self.last_emitted_idx = k as u8;
                self.self_vocal_cool = 6;
                self.aud_armed = false;
            }
        }
        self.resp_tick = self.resp_tick.saturating_add(1);
        let done_at = if dense_sym { 8 } else { 9 };
        if self.resp_tick >= done_at {
            self.in_response_kind = RESP_NONE;
        }
        out
    }

    fn finalize_previous_bout(&mut self, clean_influx: f32, clean_stress: f32) {
        self.bout_rew = sanitize(self.bout_rew + clean_influx - clean_stress);
        if clean_stress > 0.02 {
            self.bout_stress_seen = true;
        }
        if self.eval_mode == EVAL_NONE {
            return;
        }

        let rew = self.bout_rew;
        let eval_was = self.eval_mode;
        self.eval_mode = EVAL_NONE;

        if let Some(i0) = self.find_lexeme_idx(self.active_w[0], 0)
            && let Some(i1) = self.find_lexeme_idx(self.active_w[1], 1)
        {
            if eval_was == EVAL_POST_AUDIT {
                self.lexemes[i0].post_promo_audited = true;
                self.lexemes[i1].post_promo_audited = true;
                if rew > 0.80 {
                    self.yoked_contaminated = true;
                    self.lexemes.clear();
                }
                return;
            }

            if eval_was == EVAL_NULL_AUDIT {
                for &idx in &[i0, i1] {
                    if self.lexemes[idx].status == STATUS_CANDIDATE {
                        self.lexemes[idx].null_trials =
                            self.lexemes[idx].null_trials.saturating_add(1);
                        self.lexemes[idx].null_rew_sum =
                            sanitize(self.lexemes[idx].null_rew_sum + rew);
                    }
                }
                self.check_promote_or_reject(i0);
                self.check_promote_or_reject(i1);
                return;
            }

            if eval_was == EVAL_EXEC {
                let both_prom = self.lexemes[i0].status == STATUS_PROMOTED
                    && self.lexemes[i1].status == STATUS_PROMOTED;
                if self.eval_had_stress || self.bout_stress_seen {
                    if both_prom {
                        if self.active_high_cert {
                            self.consecutive_stressed_bouts =
                                self.consecutive_stressed_bouts.saturating_add(1);
                            if (self.promoted_count() >= 6 || !self.grammar_archive.is_empty())
                                && self
                                    .consecutive_stressed_bouts
                                    .saturating_add(self.consecutive_syntax_mismatch)
                                    >= 2
                            {
                                self.try_context_switch(None);
                            }
                        }
                    } else {
                        if self.step0_failed && self.lexemes[i0].status == STATUS_CANDIDATE {
                            self.lexemes[i0].bound_manip = SLOT_UNBOUND;
                            self.lexemes[i0].status = STATUS_EMPTY;
                            self.lexemes[i0].exec_trials = 0;
                            self.lexemes[i0].null_trials = 0;
                            self.lexemes[i0].exec_rew_sum = 0.0;
                            self.lexemes[i0].null_rew_sum = 0.0;
                            self.lexemes[i0].probe_cursor =
                                self.lexemes[i0].probe_cursor.saturating_add(1);
                        }
                        if self.step1_failed && self.lexemes[i1].status == STATUS_CANDIDATE {
                            self.lexemes[i1].bound_manip = SLOT_UNBOUND;
                            self.lexemes[i1].status = STATUS_EMPTY;
                            self.lexemes[i1].exec_trials = 0;
                            self.lexemes[i1].null_trials = 0;
                            self.lexemes[i1].exec_rew_sum = 0.0;
                            self.lexemes[i1].null_rew_sum = 0.0;
                            self.lexemes[i1].probe_cursor =
                                self.lexemes[i1].probe_cursor.saturating_add(1);
                        }
                    }
                } else {
                    if self.active_high_cert {
                        self.consecutive_stressed_bouts = 0;
                    }
                    if rew >= 1.50 {
                        for &idx in &[i0, i1] {
                            if self.lexemes[idx].status == STATUS_CANDIDATE {
                                self.lexemes[idx].exec_trials =
                                    self.lexemes[idx].exec_trials.saturating_add(1);
                                self.lexemes[idx].exec_rew_sum =
                                    sanitize(self.lexemes[idx].exec_rew_sum + rew);
                            }
                        }
                        self.check_promote_or_reject(i0);
                        self.check_promote_or_reject(i1);
                    }
                }
            }
        }
    }

    fn check_promote_or_reject(&mut self, idx: usize) {
        if idx >= self.lexemes.len() {
            return;
        }
        let s = &mut self.lexemes[idx];
        if s.status != STATUS_CANDIDATE || s.exec_trials < 2 || s.null_trials < 2 {
            return;
        }
        let mean_exec = s.exec_rew_sum / f32::from(s.exec_trials);
        let mean_null = s.null_rew_sum / f32::from(s.null_trials);
        if mean_exec > mean_null + 1.20 && mean_null < 0.80 {
            s.status = STATUS_PROMOTED;
            self.mark_archive_eligible_if_ready();
        } else {
            s.bound_manip = SLOT_UNBOUND;
            s.status = STATUS_EMPTY;
            s.exec_trials = 0;
            s.null_trials = 0;
            s.exec_rew_sum = 0.0;
            s.null_rew_sum = 0.0;
            s.probe_cursor = s.probe_cursor.saturating_add(1);
        }
    }
}

#[inline]
fn sparse_get(items: &[(u16, f32)], ch: u16) -> f32 {
    for &(c, v) in items {
        if c == ch {
            return v;
        }
    }
    0.0
}

#[inline]
fn sparse_norm(items: &[(u16, f32)]) -> f32 {
    let mut sq = 0.0_f32;
    for &(_, v) in items {
        sq += v * v;
    }
    sanitize(sq.sqrt())
}

#[inline]
fn sparse_cosine(a: &[(u16, f32)], b: &[(u16, f32)]) -> f32 {
    let na = sparse_norm(a);
    let nb = sparse_norm(b);
    if na < 1.0e-6 || nb < 1.0e-6 {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    for &(ca, va) in a {
        for &(cb, vb) in b {
            if ca == cb {
                dot += va * vb;
            }
        }
    }
    sanitize(dot / (na * nb))
}

#[inline]
fn dense_norm(xs: &[f32]) -> f32 {
    let mut sq = 0.0_f32;
    for &v in xs {
        sq += v * v;
    }
    sanitize(sq.sqrt())
}

#[inline]
fn dense_cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = dense_norm(a);
    let nb = dense_norm(b);
    if na < 1.0e-6 || nb < 1.0e-6 || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    sanitize(dot / (na * nb))
}

#[inline]
fn normalize_inplace(xs: &mut [f32]) {
    let n = dense_norm(xs);
    if n > 1.0e-6 {
        let inv = 1.0 / n;
        for v in xs.iter_mut() {
            *v = sanitize(*v * inv);
        }
    }
}

fn build_ten_slice_aud_feat(frames: &[Vec<f32>]) -> Option<Vec<f32>> {
    let n = frames.len();
    if n < 4 {
        return None;
    }
    let d = frames[0].len();
    if d == 0 {
        return None;
    }
    let mut out = vec![0.0_f32; COMP_AUD_SLICES * d];
    for si in 0..COMP_AUD_SLICES {
        let lo = ((si * n) / COMP_AUD_SLICES).min(n - 1);
        let hi = (((si + 1) * n) / COMP_AUD_SLICES).max(lo + 1).min(n);
        let inv = 1.0 / ((hi - lo) as f32);
        let base = si * d;
        for fr in &frames[lo..hi] {
            if fr.len() != d {
                return None;
            }
            for k in 0..d {
                out[base + k] += fr[k] * inv;
            }
        }
    }
    Some(out)
}

fn update_aud_proto(protos: &mut Vec<AudProto>, phon_pair: [u8; 2], feat: &[f32]) {
    for p in protos.iter_mut() {
        if p.phon_pair == phon_pair && p.vec.len() == feat.len() {
            p.count = p.count.saturating_add(1);
            let lr = (1.0 / f32::from(p.count)).max(0.20);
            for (dst, &src) in p.vec.iter_mut().zip(feat.iter()) {
                *dst = sanitize_clamp((1.0 - lr) * *dst + lr * src, 0.0, 2.0);
            }
            return;
        }
    }
    if protos.len() < COMP_MAX_PROTOS {
        protos.push(AudProto {
            phon_pair,
            vec: feat.to_vec(),
            count: 1,
        });
    }
}

fn match_aud_proto(protos: &[AudProto], query: &[f32], allowed_pairs: &[[u8; 2]]) -> [u8; 2] {
    let mut indices: Vec<usize> = Vec::new();
    for (i, p) in protos.iter().enumerate() {
        if p.vec.len() == query.len()
            && (allowed_pairs.is_empty() || allowed_pairs.contains(&p.phon_pair))
        {
            indices.push(i);
        }
    }
    if indices.is_empty() {
        for (i, p) in protos.iter().enumerate() {
            if p.vec.len() == query.len() {
                indices.push(i);
            }
        }
    }
    if indices.is_empty() {
        return [0, 0];
    }
    let d = query.len();
    let mut mu = vec![0.0_f32; d];
    let inv = 1.0 / (indices.len() as f32);
    for &idx in &indices {
        for k in 0..d {
            mu[k] += protos[idx].vec[k] * inv;
        }
    }
    let mut q_c = vec![0.0_f32; d];
    for k in 0..d {
        q_c[k] = query[k] - mu[k];
    }
    normalize_inplace(&mut q_c);

    let mut best_pair = protos[indices[0]].phon_pair;
    let mut best_cos = f32::NEG_INFINITY;
    for &idx in &indices {
        let mut p_c = vec![0.0_f32; d];
        for k in 0..d {
            p_c[k] = protos[idx].vec[k] - mu[k];
        }
        let c = dense_cosine(&q_c, &p_c);
        if c > best_cos {
            best_cos = c;
            best_pair = protos[idx].phon_pair;
        }
    }
    best_pair
}
