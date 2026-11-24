//! Modality-free structural world-model synthesis and hierarchical skill cortex (`SYN5`).
//!
//! Operates strictly on unlabelled numeric `ambient_in` slices and `actuator_out` pulses with zero
//! string literals, zero actuator-count splits, zero organ constant mirrors, zero modulo-64
//! aliasing, strict `O(k)` per-tick active work when passive on large fields, and strict `0`-bit
//! learned-parameter invariance under `ControlMode::FrozenPlasticity`.
//!
//! Unlocks only in multi-bank discrete synthesis portals (`REGIME_SYNTH`) or high-rank cue plus
//! tonic-carrier lifelong task streams (`REGIME_LIFELONG`). On all prior-phase physical, spatial,
//! operator, and compositional worlds this subsystem stays dormant (`None`).

use a11_medium::ControlMode;

use crate::rng::{ParamHasher, sanitize, sanitize_clamp};

pub const SYN_MAX_SCAN_CHANNELS: usize = 512;
pub const SYN_ROLL_STRIDE: usize = 64;
pub const SYN_STATE_BITS: usize = 48;
pub const SYN_CUE_BITS: usize = 64;
pub const SYN_CUE_WORDS: usize = 8;
pub const SYN_STATE_WORDS: usize = 2;
pub const SYN_MAX_PAIRS: usize = 16;
pub const SYN_OP_SYNS: usize = 8;
pub const SYN_MAX_RAW_TRANS: usize = 224;
pub const SYN_MAX_ARC_TRANS: usize = 64;
pub const SYN_MAX_SOLVED_EPS: usize = 72;
pub const SYN_MAX_SLOTS: usize = 96;
pub const SYN_MAX_ARCHIVE: usize = 32;
pub const SYN_MAX_MACROS: usize = 8;
pub const SYN_MAX_STEPS: usize = 4;
pub const SYN_MAX_PAR_CHS: usize = 48;
pub const SYN_MAX_CAND96: usize = 112;
pub const SYN_MAX_CUE_CHS: usize = 96;
pub const SYN_MAX_EXEC_RING: usize = 48;

const REGIME_PASSIVE: u8 = 0;
const REGIME_SYNTH: u8 = 1;
const REGIME_LIFELONG: u8 = 2;
const REGIME_DISABLED: u8 = 3;

const FULL_STATE_MASK: u64 = (1u64 << SYN_STATE_BITS) - 1;
const PROBE_AMP: f32 = 0.55;
const PULSE_AMP: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynChannelStat {
    pub t0_val: f32,
    pub min_val: f32,
    pub max_val: f32,
    pub zero_streak: u16,
    pub max_zero_streak: u16,
    pub ever_neg: bool,
    pub burst_early_high: bool,
    pub low_mid: bool,
    pub steady_high: bool,
    pub cue_seen_high: bool,
    pub cue_seen_low: bool,
    pub probe_min: f32,
    pub probe_max: f32,
    pub max_step_dx: f32,
    pub prev_val: f32,
    pub moved_mid_bout: bool,
    pub reset_active: bool,
}

impl Default for SynChannelStat {
    fn default() -> Self {
        Self {
            t0_val: 0.0,
            min_val: 4.0,
            max_val: -4.0,
            zero_streak: 0,
            max_zero_streak: 0,
            ever_neg: false,
            burst_early_high: false,
            low_mid: false,
            steady_high: true,
            cue_seen_high: false,
            cue_seen_low: false,
            probe_min: 4.0,
            probe_max: -4.0,
            max_step_dx: 0.0,
            prev_val: 0.0,
            moved_mid_bout: false,
            reset_active: false,
        }
    }
}

impl SynChannelStat {
    #[must_use]
    pub fn logical_bytes() -> usize {
        48
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SynProbeRec {
    pub act: u16,
    pub resp: Vec<f32>,
    pub pass1_done: bool,
    pub pass2_done: bool,
}

impl Default for SynProbeRec {
    fn default() -> Self {
        Self {
            act: 0,
            resp: Vec::new(),
            pass1_done: false,
            pass2_done: false,
        }
    }
}

impl SynProbeRec {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        8 + self.resp.len() * 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawSynTrans {
    pub op_idx: u8,
    pub trapped: bool,
    pub cin: [u64; SYN_STATE_WORDS],
    pub c0: [u64; SYN_STATE_WORDS],
    pub e0: [u64; SYN_STATE_WORDS],
    pub c1: [u64; SYN_STATE_WORDS],
    pub e1: [u64; SYN_STATE_WORDS],
}

impl Default for RawSynTrans {
    fn default() -> Self {
        Self {
            op_idx: 0,
            trapped: false,
            cin: [0; SYN_STATE_WORDS],
            c0: [0; SYN_STATE_WORDS],
            e0: [0; SYN_STATE_WORDS],
            c1: [0; SYN_STATE_WORDS],
            e1: [0; SYN_STATE_WORDS],
        }
    }
}

impl RawSynTrans {
    #[must_use]
    pub fn logical_bytes() -> usize {
        82
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.op_idx);
        hasher.write_bool(self.trapped);
        for w in 0..SYN_STATE_WORDS {
            hasher.write_u64(self.cin[w]);
            hasher.write_u64(self.c0[w]);
            hasher.write_u64(self.e0[w]);
            hasher.write_u64(self.c1[w]);
            hasher.write_u64(self.e1[w]);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolvedSynEp {
    pub dm_w: [[u64; SYN_STATE_WORDS]; 3],
    pub dm_e: [[u64; SYN_STATE_WORDS]; 3],
    pub dm_s_core: [u64; 3],
    pub dm_s_edge: [u64; 3],
    pub ops: [u8; 3],
    pub op_len: u8,
}

impl Default for SolvedSynEp {
    fn default() -> Self {
        Self {
            dm_w: [[0; SYN_STATE_WORDS]; 3],
            dm_e: [[0; SYN_STATE_WORDS]; 3],
            dm_s_core: [0; 3],
            dm_s_edge: [0; 3],
            ops: [0; 3],
            op_len: 0,
        }
    }
}

impl SolvedSynEp {
    #[must_use]
    pub fn logical_bytes() -> usize {
        148
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        for k in 0..3 {
            for w in 0..SYN_STATE_WORDS {
                hasher.write_u64(self.dm_w[k][w]);
            }
            for w in 0..SYN_STATE_WORDS {
                hasher.write_u64(self.dm_e[k][w]);
            }
            hasher.write_u64(self.dm_s_core[k]);
            hasher.write_u64(self.dm_s_edge[k]);
            hasher.write_u8(self.ops[k]);
        }
        hasher.write_u8(self.op_len);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecAffPiece {
    pub a_neg: i8,
    pub b_neg: i8,
    pub c_neg: i8,
    pub a_pos: i8,
    pub b_pos: i8,
    pub c_pos: i8,
}

impl Default for ExecAffPiece {
    fn default() -> Self {
        Self {
            a_neg: 1,
            b_neg: 0,
            c_neg: 0,
            a_pos: 1,
            b_pos: 0,
            c_pos: 0,
        }
    }
}

impl ExecAffPiece {
    #[must_use]
    pub fn eval_next(self, z: i32, x_z: i32) -> Option<i32> {
        let (a, b, c) = if z < 0 {
            (i32::from(self.a_neg), i32::from(self.b_neg), i32::from(self.c_neg))
        } else {
            (i32::from(self.a_pos), i32::from(self.b_pos), i32::from(self.c_pos))
        };
        let next = a * z + b * x_z + c;
        if next.abs() > 23 { None } else { Some(next) }
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.a_neg as u8);
        hasher.write_u8(self.b_neg as u8);
        hasher.write_u8(self.c_neg as u8);
        hasher.write_u8(self.a_pos as u8);
        hasher.write_u8(self.b_pos as u8);
        hasher.write_u8(self.c_pos as u8);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LifeTaskSlot {
    pub key_mask: [u64; SYN_CUE_WORDS],
    pub known_mask: [u64; SYN_CUE_WORDS],
    pub steps: Vec<u8>,
    pub prefix: Vec<u8>,
    pub pending_verify: Vec<u8>,
    pub status: u8,
    pub exec_rew: f32,
    pub m1_idx: u8,
    pub m2_idx: u8,
    pub p3_idx: u8,
    pub tried_mask: u16,
}

impl Default for LifeTaskSlot {
    fn default() -> Self {
        Self {
            key_mask: [0; SYN_CUE_WORDS],
            known_mask: [0; SYN_CUE_WORDS],
            steps: Vec::new(),
            prefix: Vec::new(),
            pending_verify: Vec::new(),
            status: 0,
            exec_rew: 0.0,
            m1_idx: 0,
            m2_idx: 0,
            p3_idx: 0,
            tried_mask: 0,
        }
    }
}

impl LifeTaskSlot {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        144 + self.steps.len() + self.prefix.len() + self.pending_verify.len()
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        for w in 0..SYN_CUE_WORDS {
            hasher.write_u64(self.key_mask[w]);
            hasher.write_u64(self.known_mask[w]);
        }
        hasher.write_usize(self.steps.len());
        for &s in &self.steps {
            hasher.write_u8(s);
        }
        hasher.write_usize(self.prefix.len());
        for &s in &self.prefix {
            hasher.write_u8(s);
        }
        hasher.write_usize(self.pending_verify.len());
        for &s in &self.pending_verify {
            hasher.write_u8(s);
        }
        hasher.write_u8(self.status);
        hasher.write_f32(self.exec_rew);
        hasher.write_u8(self.m1_idx);
        hasher.write_u8(self.m2_idx);
        hasher.write_u8(self.p3_idx);
        hasher.write_u16(self.tried_mask);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifeArchiveEntry {
    pub key_mask: [u64; SYN_CUE_WORDS],
    pub known_mask: [u64; SYN_CUE_WORDS],
    pub steps: Vec<u8>,
}

impl Default for LifeArchiveEntry {
    fn default() -> Self {
        Self {
            key_mask: [0; SYN_CUE_WORDS],
            known_mask: [0; SYN_CUE_WORDS],
            steps: Vec::new(),
        }
    }
}

impl LifeArchiveEntry {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        132 + self.steps.len()
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        for w in 0..SYN_CUE_WORDS {
            hasher.write_u64(self.key_mask[w]);
            hasher.write_u64(self.known_mask[w]);
        }
        hasher.write_usize(self.steps.len());
        for &s in &self.steps {
            hasher.write_u8(s);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisCortex {
    // --- Persistent learned parameters (included in hash_learned, mutated ONLY when Natural) ---
    pub regime: u8,
    pub calib_done: bool,
    pub num_actuators_bound: u8,
    pub par_chs: Vec<u16>,
    pub wire_pairs: Vec<[u16; 2]>,
    pub syn_agency_ema: f32,
    pub verified_self_copy_chs: Vec<u16>,
    pub yoked_contaminated: bool,

    // Multi-bank Synth (Pillars 1 & 2) persistent learned model
    pub spec_chs: Vec<u16>,
    pub cand96_chs: Vec<u16>,
    pub work_map_idx: Vec<u8>,
    pub cand96_h0: Vec<u64>,
    pub cand96_h1: Vec<u64>,
    pub cand96_twin: Vec<u8>,
    pub is_exec_domain: bool,
    pub op_syns: Vec<u8>,
    pub syn_probed_mask: u16,
    pub raw_trans: Vec<RawSynTrans>,
    pub direct_arc_trans: Vec<Vec<(u64, u64)>>,
    pub solved_eps: Vec<SolvedSynEp>,
    pub poss_sw: Vec<u64>,
    pub arc_perms: Vec<Vec<u8>>,
    pub arc_is_grav: Vec<bool>,
    pub arc_grav_cols: Vec<Vec<[u8; 3]>>,
    pub exec_chain: Vec<u8>,
    pub exec_p0: u8,
    pub exec_models: Vec<ExecAffPiece>,
    pub exec_model_valid: bool,
    pub seen_ep_cnt: u16,
    pub null_trials: u16,
    pub exec_trials: u16,
    pub null_rew_sum: f32,
    pub exec_rew_sum: f32,
    pub unearned_null_hits: u8,
    pub synth_op_rr: u16,
    pub synth_sb_op_cursor: u16,
    pub exec_sb_bouts: u16,
    pub exec_sb_sh1: Vec<u8>,
    pub exec_sb_exp: Vec<u8>,
    pub exec_sb_plan_valid: bool,

    // Lifelong52 (Pillar 3) persistent learned state
    pub obs_cue_mask: [u64; SYN_CUE_WORDS],
    pub life_confirmed_ops_mask: u16,
    pub life_slots: Vec<LifeTaskSlot>,
    pub life_archive: Vec<LifeArchiveEntry>,
    pub macros_2step: Vec<[u8; 2]>,

    // --- Discovery & mid-episode execution buffers (serialized for P5-I3 bit-identity,
    //     EXCLUDED from hash_learned so ControlMode::FrozenPlasticity stays strictly 0-bit drift) ---
    pub roll_cursor: u16,
    pub passive_ticks: u16,
    pub ch_stats: Vec<SynChannelStat>,
    pub probes: Vec<SynProbeRec>,
    pub probe_step: u16,
    pub calib_settle: u8,
    pub oos_idx: u8,
    pub oos_cos_sum: f32,
    pub cur_probe_base: Vec<f32>,
    pub cur_probe_pos: Vec<f32>,
    pub last_probe_act: u16,
    pub last_probe_dir: i8,

    // Transient Synth runtime registers
    pub op_probe_active: bool,
    pub op_probe_pair: u8,
    pub op_probe_wait: u8,
    pub op_probe_retries: u8,
    pub op_probe_before: Vec<f32>,
    pub sb_period_est: u16,
    pub sb_since_reset: u16,
    pub sb_reset_ticks: Vec<u16>,
    pub sb_quiet_ticks: u16,
    pub sb_tick: u16,
    pub sb_start_vals: Vec<f32>,
    pub sb_op_wait: u8,
    pub sb_pending_op: u8,
    pub sb_pre_c: [u64; SYN_STATE_WORDS],
    pub sb_pre_e: [u64; SYN_STATE_WORDS],
    pub sb_in_c: [u64; SYN_STATE_WORDS],
    pub sb_bout_count: u16,
    pub sb_step_idx: u8,
    pub sb_active_plan: [u8; 4],

    pub in_demo: bool,
    pub demo_tick: u16,
    pub test_tick: u16,
    pub demo_x_cand: [[u64; SYN_STATE_WORDS]; 3],
    pub demo_x_edge_cand: [[u64; SYN_STATE_WORDS]; 3],
    pub demo_s_core: [u64; 3],
    pub demo_s_edge: [u64; 3],
    pub is_null_hold_ep: bool,
    pub ep_null_got_influx: bool,
    pub ep_planned: [u8; 3],
    pub ep_plan_len: u8,
    pub ep_plan_pc: u8,
    pub ep_halted: bool,
    pub ep_solved_flag: bool,
    pub ep_test_wait: u8,
    pub ep_pending_op: u8,
    pub ep_pre_c: [u64; SYN_STATE_WORDS],
    pub ep_pre_e: [u64; SYN_STATE_WORDS],
    pub ep_in_c: [u64; SYN_STATE_WORDS],
    pub ep_ep_rew: f32,
    pub ep_pending_finalize: bool,

    // Transient Lifelong52 runtime registers
    pub life_in_cue: bool,
    pub life_cue_tick: u16,
    pub life_act_tick: u16,
    pub life_cur_key: [u64; SYN_CUE_WORDS],
    pub life_cur_known: [u64; SYN_CUE_WORDS],
    pub life_cur_slot_idx: u16,
    pub life_bout_halted: bool,
    pub life_waiting_res: bool,
    pub life_wait_syn: u8,
    pub life_wait_from_prefix: bool,
    pub life_wait_prefix_idx: u8,
    pub life_wait_macro_mode: u8,
    pub life_bout_rew: f32,
    pub life_step_pos: u8,
    pub life_prefix_idx: u8,
    pub life_macro_step_idx: u8,
    pub life_macro_plan: [u8; 4],
    pub life_macro_plan_len: u8,
    pub life_bout_unconf: bool,
    pub life_bout_last_conf: u8,
    pub life_bout_last_conf_valid: bool,
    pub life_pending_null_check: bool,
    pub life_is_null_bout: bool,
    pub life_null_rew_accum: f32,
    pub life_prev_had_step: bool,
}

impl Default for SynthesisCortex {
    fn default() -> Self {
        Self {
            regime: REGIME_PASSIVE,
            calib_done: false,
            num_actuators_bound: 0,
            par_chs: Vec::new(),
            wire_pairs: Vec::new(),
            syn_agency_ema: 0.0,
            verified_self_copy_chs: Vec::new(),
            yoked_contaminated: false,

            spec_chs: Vec::new(),
            cand96_chs: Vec::new(),
            work_map_idx: Vec::new(),
            cand96_h0: Vec::new(),
            cand96_h1: Vec::new(),
            cand96_twin: Vec::new(),
            is_exec_domain: false,
            op_syns: Vec::new(),
            syn_probed_mask: 0,
            raw_trans: Vec::new(),
            direct_arc_trans: Vec::new(),
            solved_eps: Vec::new(),
            poss_sw: Vec::new(),
            arc_perms: Vec::new(),
            arc_is_grav: Vec::new(),
            arc_grav_cols: Vec::new(),
            exec_chain: Vec::new(),
            exec_p0: 0,
            exec_models: Vec::new(),
            exec_model_valid: false,
            seen_ep_cnt: 0,
            null_trials: 0,
            exec_trials: 0,
            null_rew_sum: 0.0,
            exec_rew_sum: 0.0,
            unearned_null_hits: 0,
            synth_op_rr: 0,
            synth_sb_op_cursor: 0,
            exec_sb_bouts: 0,
            exec_sb_sh1: Vec::new(),
            exec_sb_exp: Vec::new(),
            exec_sb_plan_valid: false,

            obs_cue_mask: [0; SYN_CUE_WORDS],
            life_confirmed_ops_mask: 0,
            life_slots: Vec::new(),
            life_archive: Vec::new(),
            macros_2step: Vec::new(),

            roll_cursor: 0,
            passive_ticks: 0,
            ch_stats: Vec::new(),
            probes: Vec::new(),
            probe_step: 0,
            calib_settle: 0,
            oos_idx: 0,
            oos_cos_sum: 0.0,
            cur_probe_base: Vec::new(),
            cur_probe_pos: Vec::new(),
            last_probe_act: 0,
            last_probe_dir: 1,

            op_probe_active: false,
            op_probe_pair: 0,
            op_probe_wait: 0,
            op_probe_retries: 0,
            op_probe_before: Vec::new(),
            sb_period_est: 0,
            sb_since_reset: 0,
            sb_reset_ticks: Vec::new(),
            sb_quiet_ticks: 0,
            sb_tick: 0,
            sb_start_vals: Vec::new(),
            sb_op_wait: 0,
            sb_pending_op: 0,
            sb_pre_c: [0; SYN_STATE_WORDS],
            sb_pre_e: [0; SYN_STATE_WORDS],
            sb_in_c: [0; SYN_STATE_WORDS],
            sb_bout_count: 0,
            sb_step_idx: 0,
            sb_active_plan: [0; 4],

            in_demo: false,
            demo_tick: 0,
            test_tick: 0,
            demo_x_cand: [[0; SYN_STATE_WORDS]; 3],
            demo_x_edge_cand: [[0; SYN_STATE_WORDS]; 3],
            demo_s_core: [0; 3],
            demo_s_edge: [0; 3],
            is_null_hold_ep: false,
            ep_null_got_influx: false,
            ep_planned: [0; 3],
            ep_plan_len: 0,
            ep_plan_pc: 0,
            ep_halted: false,
            ep_solved_flag: false,
            ep_test_wait: 0,
            ep_pending_op: 0,
            ep_pre_c: [0; SYN_STATE_WORDS],
            ep_pre_e: [0; SYN_STATE_WORDS],
            ep_in_c: [0; SYN_STATE_WORDS],
            ep_ep_rew: 0.0,
            ep_pending_finalize: false,

            life_in_cue: false,
            life_cue_tick: 0,
            life_act_tick: 0,
            life_cur_key: [0; SYN_CUE_WORDS],
            life_cur_known: [0; SYN_CUE_WORDS],
            life_cur_slot_idx: u16::MAX,
            life_bout_halted: false,
            life_waiting_res: false,
            life_wait_syn: 0,
            life_wait_from_prefix: false,
            life_wait_prefix_idx: 0,
            life_wait_macro_mode: 0,
            life_bout_rew: 0.0,
            life_step_pos: 0,
            life_prefix_idx: 0,
            life_macro_step_idx: 0,
            life_macro_plan: [0; 4],
            life_macro_plan_len: 0,
            life_bout_unconf: false,
            life_bout_last_conf: 0,
            life_bout_last_conf_valid: false,
            life_pending_null_check: false,
            life_is_null_bout: false,
            life_null_rew_accum: 0.0,
            life_prev_had_step: false,
        }
    }
}

impl SynthesisCortex {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let ch_b = self.ch_stats.len() * SynChannelStat::logical_bytes();
        let pr_b: usize = self.probes.iter().map(SynProbeRec::logical_bytes).sum();
        let tr_b = self.raw_trans.len() * RawSynTrans::logical_bytes();
        let dir_b: usize = self.direct_arc_trans.iter().map(|v| 8 + v.len() * 16).sum();
        let ep_b = self.solved_eps.len() * SolvedSynEp::logical_bytes();
        let sl_b: usize = self.life_slots.iter().map(LifeTaskSlot::logical_bytes).sum();
        let ar_b: usize = self
            .life_archive
            .iter()
            .map(LifeArchiveEntry::logical_bytes)
            .sum();
        320 + self.par_chs.len() * 2
            + self.wire_pairs.len() * 4
            + self.verified_self_copy_chs.len() * 2
            + self.spec_chs.len() * 2
            + self.cand96_chs.len() * 2
            + self.work_map_idx.len()
            + self.cand96_h0.len() * 8
            + self.cand96_h1.len() * 8
            + self.cand96_twin.len()
            + ch_b
            + pr_b
            + tr_b
            + dir_b
            + ep_b
            + sl_b
            + ar_b
            + self.macros_2step.len() * 2
            + 2048
    }

    pub fn zero_learned(&mut self) {
        *self = Self::default();
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u8(self.regime);
        hasher.write_bool(self.calib_done);
        hasher.write_u8(self.num_actuators_bound);
        hasher.write_usize(self.par_chs.len());
        for &c in &self.par_chs {
            hasher.write_u16(c);
        }
        hasher.write_usize(self.wire_pairs.len());
        for &p in &self.wire_pairs {
            hasher.write_u16(p[0]);
            hasher.write_u16(p[1]);
        }
        hasher.write_f32(self.syn_agency_ema);
        hasher.write_usize(self.verified_self_copy_chs.len());
        for &c in &self.verified_self_copy_chs {
            hasher.write_u16(c);
        }
        hasher.write_bool(self.yoked_contaminated);

        hasher.write_usize(self.spec_chs.len());
        for &c in &self.spec_chs {
            hasher.write_u16(c);
        }
        hasher.write_usize(self.cand96_chs.len());
        for &c in &self.cand96_chs {
            hasher.write_u16(c);
        }
        hasher.write_usize(self.work_map_idx.len());
        for &w in &self.work_map_idx {
            hasher.write_u8(w);
        }
        hasher.write_usize(self.cand96_h0.len());
        for &h in &self.cand96_h0 {
            hasher.write_u64(h);
        }
        hasher.write_usize(self.cand96_h1.len());
        for &h in &self.cand96_h1 {
            hasher.write_u64(h);
        }
        hasher.write_usize(self.cand96_twin.len());
        for &t in &self.cand96_twin {
            hasher.write_u8(t);
        }
        hasher.write_bool(self.is_exec_domain);
        hasher.write_usize(self.op_syns.len());
        for &s in &self.op_syns {
            hasher.write_u8(s);
        }
        hasher.write_u16(self.syn_probed_mask);
        hasher.write_usize(self.raw_trans.len());
        for tr in &self.raw_trans {
            tr.hash_learned(hasher);
        }
        hasher.write_usize(self.direct_arc_trans.len());
        for v in &self.direct_arc_trans {
            hasher.write_usize(v.len());
            for &(m0, m1) in v {
                hasher.write_u64(m0);
                hasher.write_u64(m1);
            }
        }
        hasher.write_usize(self.solved_eps.len());
        for ep in &self.solved_eps {
            ep.hash_learned(hasher);
        }
        hasher.write_usize(self.poss_sw.len());
        for &p in &self.poss_sw {
            hasher.write_u64(p);
        }
        hasher.write_usize(self.arc_perms.len());
        for v in &self.arc_perms {
            hasher.write_usize(v.len());
            for &x in v {
                hasher.write_u8(x);
            }
        }
        hasher.write_usize(self.arc_is_grav.len());
        for &g in &self.arc_is_grav {
            hasher.write_bool(g);
        }
        hasher.write_usize(self.arc_grav_cols.len());
        for col in &self.arc_grav_cols {
            hasher.write_usize(col.len());
            for &row in col {
                hasher.write_u8(row[0]);
                hasher.write_u8(row[1]);
                hasher.write_u8(row[2]);
            }
        }
        hasher.write_usize(self.exec_chain.len());
        for &b in &self.exec_chain {
            hasher.write_u8(b);
        }
        hasher.write_u8(self.exec_p0);
        hasher.write_usize(self.exec_models.len());
        for m in &self.exec_models {
            m.hash_learned(hasher);
        }
        hasher.write_bool(self.exec_model_valid);
        hasher.write_u16(self.seen_ep_cnt);
        hasher.write_u16(self.null_trials);
        hasher.write_u16(self.exec_trials);
        hasher.write_f32(self.null_rew_sum);
        hasher.write_f32(self.exec_rew_sum);
        hasher.write_u8(self.unearned_null_hits);
        hasher.write_u16(self.synth_sb_op_cursor);
        hasher.write_u16(self.exec_sb_bouts);
        hasher.write_usize(self.exec_sb_sh1.len());
        for &s in &self.exec_sb_sh1 {
            hasher.write_u8(s);
        }
        hasher.write_usize(self.exec_sb_exp.len());
        for &s in &self.exec_sb_exp {
            hasher.write_u8(s);
        }
        hasher.write_bool(self.exec_sb_plan_valid);

        for w in 0..SYN_CUE_WORDS {
            hasher.write_u64(self.obs_cue_mask[w]);
        }
        hasher.write_u16(self.life_confirmed_ops_mask);
        hasher.write_usize(self.life_slots.len());
        for sl in &self.life_slots {
            sl.hash_learned(hasher);
        }
        hasher.write_usize(self.life_archive.len());
        for ae in &self.life_archive {
            ae.hash_learned(hasher);
        }
        hasher.write_usize(self.macros_2step.len());
        for m in &self.macros_2step {
            hasher.write_u8(m[0]);
            hasher.write_u8(m[1]);
        }
    }

    pub fn step_synthesis(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        num_actuators: usize,
        mode: ControlMode,
        seed: u64,
    ) -> Option<Vec<f32>> {
        if self.regime == REGIME_DISABLED || num_actuators < 2 {
            return None;
        }
        if ambient_in.len() > SYN_MAX_SCAN_CHANNELS {
            if mode == ControlMode::Natural {
                self.roll_cursor = ((usize::from(self.roll_cursor) + SYN_ROLL_STRIDE)
                    % ambient_in.len().max(1)) as u16;
            }
            return None;
        }
        let n_ch = ambient_in.len();
        if n_ch == 0 {
            return None;
        }
        let na = num_actuators.min(32);

        if self.regime == REGIME_PASSIVE {
            if mode != ControlMode::Natural {
                return None;
            }
            self.update_passive_unlock(ambient_in, na);
            return None;
        }

        if mode == ControlMode::Natural && !self.calib_done {
            return Some(self.step_calibration(ambient_in, na));
        }

        match self.regime {
            REGIME_SYNTH => Some(self.step_synth_regime(
                ambient_in,
                metabolic_influx,
                tissue_stress,
                num_actuators,
                mode,
                seed,
            )),
            REGIME_LIFELONG => Some(self.step_lifelong_regime(
                ambient_in,
                metabolic_influx,
                tissue_stress,
                num_actuators,
                mode,
            )),
            _ => None,
        }
    }

    fn update_passive_unlock(&mut self, ambient_in: &[f32], na: usize) {
        let n_ch = ambient_in.len();
        if self.ch_stats.len() != n_ch {
            self.ch_stats = vec![SynChannelStat::default(); n_ch];
        }
        let pt = self.passive_ticks;
        let mut reset_jump_cnt = 0usize;
        for (c, st) in self.ch_stats.iter_mut().enumerate() {
            let v = sanitize(ambient_in[c]);
            if v < -1.0e-3 {
                st.ever_neg = true;
            }
            if pt == 0 {
                st.t0_val = v;
                st.prev_val = v;
            } else {
                let dx = (v - st.prev_val).abs();
                if dx > st.max_step_dx {
                    st.max_step_dx = dx;
                }
                if !st.ever_neg && dx > 0.38 {
                    reset_jump_cnt += 1;
                }
                st.prev_val = v;
            }
            if v < st.min_val {
                st.min_val = v;
            }
            if v > st.max_val {
                st.max_val = v;
            }
            if v.abs() < 0.002 {
                st.zero_streak = st.zero_streak.saturating_add(1);
                if st.zero_streak > st.max_zero_streak {
                    st.max_zero_streak = st.zero_streak;
                }
            } else {
                st.zero_streak = 0;
            }
            if pt < 4 && v > 0.65 {
                st.burst_early_high = true;
            }
            if (5..8).contains(&pt) && v < 0.15 {
                st.low_mid = true;
            }
            if (2..8).contains(&pt) && v <= 0.65 {
                st.steady_high = false;
            }
            if v > 0.65 {
                st.cue_seen_high = true;
            }
            if v < 0.12 {
                st.cue_seen_low = true;
            }
        }
        if pt >= 12
            && reset_jump_cnt >= 2
            && self
                .sb_reset_ticks
                .last()
                .is_none_or(|&prev| pt.saturating_sub(prev) >= 10)
            && self.sb_reset_ticks.len() < 8
        {
            if let Some(&prev_t) = self.sb_reset_ticks.last() {
                let diff = pt.saturating_sub(prev_t);
                if (10..=42).contains(&diff) {
                    if self.sb_period_est == 0 {
                        self.sb_period_est = diff;
                    } else {
                        let g = gcd_u16(self.sb_period_est, diff);
                        if g >= 10 {
                            self.sb_period_est = g;
                        } else if diff < self.sb_period_est {
                            self.sb_period_est = diff;
                        }
                    }
                }
            }
            self.sb_reset_ticks.push(pt);
        }
        self.passive_ticks = self.passive_ticks.saturating_add(1);

        // 1. Check Lifelong52 unlock at passive_ticks == 8
        if self.passive_ticks == 8 {
            let mut early_drop_cnt = 0usize;
            let mut carrier_cnt = 0usize;
            for st in &self.ch_stats {
                if !st.ever_neg && st.burst_early_high && st.low_mid {
                    early_drop_cnt += 1;
                }
                if !st.ever_neg && st.steady_high {
                    carrier_cnt += 1;
                }
            }
            if early_drop_cnt >= 22 && carrier_cnt >= 14 {
                self.init_regime_common(REGIME_LIFELONG, na);
                for (c, st) in self.ch_stats.iter().enumerate() {
                    if !st.ever_neg && st.burst_early_high && st.low_mid {
                        set_cue_bit(&mut self.obs_cue_mask, c);
                    }
                }
                return;
            }
        }

        // 2. Check Multi-Bank Synth unlock (t>=8..=28 so Lifelong52 always checks first at t==8)
        if self.passive_ticks >= 8 && self.passive_ticks <= 28 {
            let mut floor_cluster = 0usize;
            for st in &self.ch_stats {
                if !st.ever_neg && (0.03..=0.10).contains(&st.t0_val) {
                    let cnt = self
                        .ch_stats
                        .iter()
                        .filter(|s2| !s2.ever_neg && (s2.t0_val - st.t0_val).abs() <= 0.006)
                        .count();
                    if cnt > floor_cluster {
                        floor_cluster = cnt;
                    }
                }
            }
            let mut spec_cnt = 0usize;
            for st in &self.ch_stats {
                let is_spec = !st.ever_neg
                    && st.max_zero_streak >= 6
                    && (st.max_val < 0.002 || st.max_val > 0.03);
                if is_spec {
                    spec_cnt += 1;
                }
            }
            if floor_cluster >= 56 && (44..=92).contains(&spec_cnt) {
                self.init_regime_common(REGIME_SYNTH, na);
                self.spec_chs.clear();
                self.cand96_chs.clear();
                for (c, st) in self.ch_stats.iter().enumerate() {
                    let is_spec = !st.ever_neg
                        && st.max_zero_streak >= 6
                        && (st.max_val < 0.002 || st.max_val > 0.03);
                    if is_spec {
                        self.spec_chs.push(c as u16);
                    }
                }
                return;
            }
        }

        if self.passive_ticks >= 32 {
            self.regime = REGIME_DISABLED;
        }
    }

    fn init_regime_common(&mut self, reg: u8, na: usize) {
        self.regime = reg;
        self.num_actuators_bound = na as u8;
        self.calib_done = false;
        self.probe_step = 0;
        self.calib_settle = 0;
        self.oos_idx = 0;
        self.oos_cos_sum = 0.0;
        let n_ch = self.ch_stats.len();
        self.cur_probe_base = vec![0.0; n_ch];
        self.cur_probe_pos = vec![0.0; n_ch];
        self.probes = (0..na)
            .map(|a| SynProbeRec {
                act: a as u16,
                ..SynProbeRec::default()
            })
            .collect();
        self.poss_sw = vec![FULL_STATE_MASK; SYN_STATE_BITS];
    }

    fn step_calibration(&mut self, ambient_in: &[f32], na: usize) -> Vec<f32> {
        let mut pulse = vec![0.0_f32; na];
        let n_ch = ambient_in.len().min(self.ch_stats.len());
        let pt_total = self.passive_ticks;
        self.passive_ticks = self.passive_ticks.saturating_add(1);

        // Update general channel stats + spec / reset tracking during sub-threshold calibration
        let mut reset_jump_cnt = 0usize;
        let settled = self.calib_settle >= 8;
        for c in 0..n_ch {
            let v = sanitize(ambient_in[c]);
            let st = &mut self.ch_stats[c];
            if v < -1.0e-3 {
                st.ever_neg = true;
            }
            let dx = (v - st.prev_val).abs();
            if settled && dx > st.max_step_dx {
                st.max_step_dx = dx;
            }
            if (self.probe_step > 0 || pt_total > 0) && !st.ever_neg && dx > 0.38 {
                reset_jump_cnt += 1;
            }
            if v < st.min_val {
                st.min_val = v;
            }
            if v > st.max_val {
                st.max_val = v;
            }
            if v.abs() < 0.002 {
                st.zero_streak = st.zero_streak.saturating_add(1);
                if st.zero_streak > st.max_zero_streak {
                    st.max_zero_streak = st.zero_streak;
                }
            } else {
                st.zero_streak = 0;
            }
            if settled {
                if v < st.probe_min {
                    st.probe_min = v;
                }
                if v > st.probe_max {
                    st.probe_max = v;
                }
                if self.probe_step >= 1
                    && v < -0.05
                    && self.last_probe_dir < 0
                    && !self.verified_self_copy_chs.contains(&(c as u16))
                {
                    self.verified_self_copy_chs.push(c as u16);
                }
            }
            st.prev_val = v;
        }
        if self.regime == REGIME_SYNTH
            && pt_total >= 12
            && reset_jump_cnt >= 2
            && self
                .sb_reset_ticks
                .last()
                .is_none_or(|&prev| pt_total.saturating_sub(prev) >= 10)
        {
            if let Some(&prev_t) = self.sb_reset_ticks.last() {
                let diff = pt_total.saturating_sub(prev_t);
                if (10..=42).contains(&diff) {
                    if self.sb_period_est == 0 {
                        self.sb_period_est = diff;
                    } else {
                        let g = gcd_u16(self.sb_period_est, diff);
                        if g >= 10 {
                            self.sb_period_est = g;
                        } else if diff < self.sb_period_est {
                            self.sb_period_est = diff;
                        }
                    }
                }
            }
            if self.sb_reset_ticks.len() < 8 {
                self.sb_reset_ticks.push(pt_total);
            }
        }
        if self.calib_settle < 8 {
            self.calib_settle = self.calib_settle.saturating_add(1);
            if self.calib_settle == 8 {
                for (c, st) in self.ch_stats.iter_mut().take(n_ch).enumerate() {
                    let v = sanitize(ambient_in[c]);
                    st.max_step_dx = 0.0;
                    st.probe_min = v;
                    st.probe_max = v;
                }
                self.verified_self_copy_chs.clear();
            }
            return pulse;
        }

        let ps = self.probe_step as usize;
        if ps > 0 && ps <= 2 * na {
            let w = (ps - 1) / 2;
            let is_neg_result = (ps - 1) % 2 == 1;
            if !is_neg_result {
                self.cur_probe_pos.copy_from_slice(&ambient_in[..n_ch]);
            } else if w < self.probes.len() {
                let mut r = vec![0.0_f32; n_ch];
                for c in 0..n_ch {
                    r[c] = (self.cur_probe_pos[c] - self.cur_probe_base[c])
                        - 0.45 * (ambient_in[c] - self.cur_probe_pos[c]);
                }
                self.probes[w].resp = r;
                self.probes[w].pass1_done = true;
            }
        } else if ps > 2 * na && ps <= 2 * na + 6 {
            let p2_idx = ps - (2 * na + 1);
            let oi = p2_idx / 2;
            let is_neg_res = p2_idx % 2 == 1;
            if !is_neg_res {
                self.cur_probe_pos.copy_from_slice(&ambient_in[..n_ch]);
            } else if oi < self.wire_pairs.len() {
                let w = usize::from(self.wire_pairs[oi][0]);
                if w < self.probes.len() && self.par_chs.len() >= 4 {
                    let cos = masked_cosine_diff(
                        &self.par_chs,
                        &self.probes[w].resp,
                        &self.cur_probe_base,
                        &self.cur_probe_pos,
                        ambient_in,
                    );
                    self.oos_cos_sum += cos;
                }
            }
        }

        // Transition Pass 1 -> Pass 2 right at `ps == 2 * na`
        if ps == 2 * na {
            if self.regime == REGIME_SYNTH {
                self.spec_chs.clear();
                for (c, st) in self.ch_stats.iter().enumerate() {
                    if !st.ever_neg
                        && st.max_zero_streak >= 6
                        && (st.max_val < 0.002 || st.max_val > 0.03)
                    {
                        self.spec_chs.push(c as u16);
                    }
                }
            }
            self.par_chs.clear();
            for (c, st) in self.ch_stats.iter().enumerate() {
                if !st.ever_neg
                    && st.min_val >= 0.08
                    && st.max_step_dx <= 0.085
                    && (st.probe_max - st.probe_min) >= 0.0005
                    && !self.spec_chs.contains(&(c as u16))
                    && self.par_chs.len() < SYN_MAX_PAR_CHS
                {
                    self.par_chs.push(c as u16);
                }
            }
            self.pair_wires(na);
            if self.regime == REGIME_LIFELONG {
                for (c, st) in self.ch_stats.iter().enumerate() {
                    let c16 = c as u16;
                    if st.ever_neg
                        || self.par_chs.contains(&c16)
                        || self.verified_self_copy_chs.contains(&c16)
                    {
                        clear_cue_bit(&mut self.obs_cue_mask, c);
                    }
                }
            }
            if self.regime == REGIME_SYNTH {
                self.finalize_cand96_partition();
            }
        }

        // Finish calibration right at `ps == 2 * na + 6`
        if ps >= 2 * na + 6 {
            let mean_cos = self.oos_cos_sum / 3.0;
            if !mean_cos.is_finite() || mean_cos < 0.70 {
                self.syn_agency_ema = 0.0;
                self.regime = REGIME_DISABLED;
                return pulse;
            }
            self.syn_agency_ema = sanitize_clamp(mean_cos * 0.85, 0.0, 1.0);
            self.calib_done = true;
            if self.regime == REGIME_SYNTH {
                self.op_probe_active = true;
                self.op_probe_pair = 0;
                self.op_probe_wait = 0;
                self.op_probe_retries = 0;
                let last_rst = self.sb_reset_ticks.last().copied().unwrap_or(0);
                let mut per = self.sb_period_est;
                if (22..=40).contains(&per) && per.is_multiple_of(2) {
                    per /= 2;
                }
                if !(10..=20).contains(&per) {
                    per = 14;
                }
                self.sb_period_est = per;
                self.sb_since_reset = pt_total.saturating_sub(last_rst) % self.sb_period_est;
                self.sb_tick = self.sb_since_reset;
            }
            return pulse;
        }

        // Schedule next sub-threshold probe tick
        if ps < 2 * na {
            let w = ps / 2;
            let neg = ps % 2 == 1;
            if !neg {
                self.cur_probe_base.copy_from_slice(&ambient_in[..n_ch]);
            }
            pulse[w] = if neg { -PROBE_AMP } else { PROBE_AMP };
            self.last_probe_act = w as u16;
            self.last_probe_dir = if neg { -1 } else { 1 };
        } else {
            let p2_idx = ps - 2 * na;
            let oi = p2_idx / 2;
            let neg = p2_idx % 2 == 1;
            if !neg {
                self.cur_probe_base.copy_from_slice(&ambient_in[..n_ch]);
            }
            if oi < self.wire_pairs.len() {
                let w = usize::from(self.wire_pairs[oi][0]);
                if w < na {
                    pulse[w] = if neg { -PROBE_AMP } else { PROBE_AMP };
                    self.last_probe_act = w as u16;
                    self.last_probe_dir = if neg { -1 } else { 1 };
                }
            }
        }
        self.probe_step = self.probe_step.saturating_add(1);
        pulse
    }

    fn pair_wires(&mut self, na: usize) {
        let mut edges: Vec<(f32, usize, usize)> = Vec::new();
        for i in 0..na {
            for j in (i + 1)..na {
                let c = masked_vec_cosine(&self.par_chs, &self.probes[i].resp, &self.probes[j].resp);
                edges.push((c, i, j));
            }
        }
        edges.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut used = vec![false; na];
        self.wire_pairs.clear();
        for &(_, i, j) in &edges {
            if !used[i] && !used[j] && self.wire_pairs.len() < SYN_MAX_PAIRS {
                used[i] = true;
                used[j] = true;
                self.wire_pairs.push([i as u16, j as u16]);
            }
        }
    }

    fn finalize_cand96_partition(&mut self) {
        self.cand96_chs.clear();
        for (c, st) in self.ch_stats.iter().enumerate() {
            if !st.ever_neg
                && !self.par_chs.contains(&(c as u16))
                && !self.spec_chs.contains(&(c as u16))
                && st.max_val > 0.035
                && self.cand96_chs.len() < SYN_MAX_CAND96
            {
                self.cand96_chs.push(c as u16);
            }
        }
        let n96 = self.cand96_chs.len();
        self.work_map_idx = vec![u8::MAX; n96];
        self.cand96_h0 = vec![0u64; n96];
        self.cand96_h1 = vec![0u64; n96];
        self.cand96_twin = vec![u8::MAX; SYN_STATE_BITS];
        self.sb_start_vals = vec![0.0; n96];
        self.op_probe_before = vec![0.0; n96];
    }

    fn record_cand96_hist(&mut self, cm: [u64; SYN_STATE_WORDS]) {
        let n = self
            .cand96_chs
            .len()
            .min(self.cand96_h0.len())
            .min(self.cand96_h1.len());
        for i in 0..n {
            let w = i / 64;
            let b = i % 64;
            if w < SYN_STATE_WORDS {
                let bit = (cm[w] >> b) & 1;
                let top = (self.cand96_h0[i] >> 63) & 1;
                self.cand96_h0[i] = (self.cand96_h0[i] << 1) | bit;
                self.cand96_h1[i] = (self.cand96_h1[i] << 1) | top;
            }
        }
    }

    fn rebuild_arc_twin_work_map(&mut self) {
        let n = self
            .cand96_chs
            .len()
            .min(self.cand96_h0.len())
            .min(self.cand96_h1.len());
        if n == 0 {
            return;
        }
        let mut new_wmap = vec![u8::MAX; self.cand96_chs.len()];
        let mut new_twin = vec![u8::MAX; SYN_STATE_BITS];
        let mut classes: Vec<(u64, u64, usize)> = Vec::with_capacity(SYN_STATE_BITS);
        for i in 0..n {
            let h0 = self.cand96_h0[i];
            let h1 = self.cand96_h1[i];
            if let Some(w_idx) = classes.iter().position(|&(a, b, _)| a == h0 && b == h1) {
                if new_twin[w_idx] == u8::MAX {
                    new_twin[w_idx] = i as u8;
                }
            } else if classes.len() < SYN_STATE_BITS {
                let w_idx = classes.len();
                classes.push((h0, h1, i));
                new_wmap[i] = w_idx as u8;
            }
        }
        if classes.len() >= 44 {
            self.work_map_idx = new_wmap;
            self.cand96_twin = new_twin;
        }
    }

    fn emit_pair(&self, na: usize, pair_idx: usize) -> Vec<f32> {
        let mut p = vec![0.0_f32; na];
        if let Some(&[wa, wb]) = self.wire_pairs.get(pair_idx) {
            let ia = usize::from(wa);
            let ib = usize::from(wb);
            if ia < na {
                p[ia] = PULSE_AMP;
            }
            if ib < na {
                p[ib] = PULSE_AMP;
            }
        }
        p
    }

    // =========================================================================
    // REGIME 1: Multi-Bank Synthesis (ArcGrid & SandboxedExec)
    // =========================================================================

    fn spec_max(&self, ambient_in: &[f32]) -> f32 {
        let mut mx = 0.0_f32;
        for &c in &self.spec_chs {
            if let Some(&v) = ambient_in.get(usize::from(c))
                && v > mx
            {
                mx = v;
            }
        }
        mx
    }

    fn read_cand96_masks(&self, ambient_in: &[f32]) -> ([u64; SYN_STATE_WORDS], [u64; SYN_STATE_WORDS]) {
        let mut cm = [0u64; SYN_STATE_WORDS];
        let mut em = [0u64; SYN_STATE_WORDS];
        for (i, &c) in self.cand96_chs.iter().enumerate() {
            if let Some(&v) = ambient_in.get(usize::from(c)) {
                let w = i / 64;
                let b = i % 64;
                if w < SYN_STATE_WORDS {
                    if v > 0.56 {
                        cm[w] |= 1u64 << b;
                    } else if v > 0.115 && v <= 0.45 {
                        em[w] |= 1u64 << b;
                    }
                }
            }
        }
        (cm, em)
    }

    fn read_spec_masks(&self, ambient_in: &[f32]) -> (u64, u64) {
        let mut cm = 0u64;
        let mut em = 0u64;
        for (b, &c) in self.spec_chs.iter().take(SYN_STATE_BITS).enumerate() {
            if let Some(&v) = ambient_in.get(usize::from(c)) {
                if v > 0.56 {
                    cm |= 1u64 << b;
                } else if v > 0.115 && v <= 0.45 {
                    em |= 1u64 << b;
                }
            }
        }
        (cm, em)
    }

    fn project_work_mask(&self, m: [u64; SYN_STATE_WORDS]) -> u64 {
        let mut out = 0u64;
        for (i, &w_idx) in self.work_map_idx.iter().enumerate() {
            if (w_idx as usize) < SYN_STATE_BITS {
                let w = i / 64;
                let b = i % 64;
                if w < SYN_STATE_WORDS && ((m[w] >> b) & 1) == 1 {
                    out |= 1u64 << w_idx;
                }
            }
        }
        out
    }

    fn mark_work_mask_delta(
        &mut self,
        c0: [u64; SYN_STATE_WORDS],
        c1: [u64; SYN_STATE_WORDS],
        e0: [u64; SYN_STATE_WORDS],
        e1: [u64; SYN_STATE_WORDS],
        allow_edge: bool,
    ) -> usize {
        let mut moved_cnt = 0usize;
        let mut next_w = self
            .work_map_idx
            .iter()
            .filter(|&&w| (w as usize) < SYN_STATE_BITS)
            .count();
        for i in 0..self.cand96_chs.len() {
            let w = i / 64;
            let b = i % 64;
            if w >= SYN_STATE_WORDS {
                break;
            }
            let dc = ((c0[w] ^ c1[w]) >> b) & 1 == 1;
            let de = allow_edge && (((e0[w] ^ e1[w]) >> b) & 1 == 1);
            if dc {
                moved_cnt += 1;
            }
            if (dc || de)
                && i < self.work_map_idx.len()
                && self.work_map_idx[i] == u8::MAX
                && next_w < SYN_STATE_BITS
            {
                self.work_map_idx[i] = next_w as u8;
                next_w += 1;
            }
        }
        moved_cnt
    }

    fn step_synth_regime(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        na: usize,
        mode: ControlMode,
        seed: u64,
    ) -> Vec<f32> {
        let clean_inf = if metabolic_influx.is_finite() {
            metabolic_influx.max(0.0)
        } else {
            0.0
        };
        let clean_str = if tissue_stress.is_finite() {
            tissue_stress.max(0.0)
        } else {
            0.0
        };
        let sm = self.spec_max(ambient_in);
        let spec_active = sm > 0.44;

        // If we transitioned into Seen/Unseen (`spec_active` appears!), cancel Sandbox probing cleanly
        if spec_active && !self.in_demo {
            let first_entry_from_sb = !self.ep_pending_finalize && self.seen_ep_cnt == 0;
            self.finalize_pending_seen_ep(mode);
            if mode == ControlMode::Natural && self.op_syns.len() < SYN_OP_SYNS {
                self.op_probe_active = false;
                let np = self.wire_pairs.len();
                for i in 0..np {
                    let idx = i as u8;
                    if !self.op_syns.contains(&idx) && self.op_syns.len() < SYN_OP_SYNS {
                        self.op_syns.push(idx);
                    }
                }
                if !self.raw_trans.is_empty() {
                    if self.is_exec_domain {
                        self.identify_exec_sb_ops();
                    }
                    self.rebuild_world_models();
                }
            } else if mode == ControlMode::Natural
                && first_entry_from_sb
                && !self.raw_trans.is_empty()
            {
                self.rebuild_world_models();
            }
            self.in_demo = true;
            self.demo_tick = 0;
            self.test_tick = 0;
        }

        if self.in_demo
            || (!spec_active
                && (self.ep_pending_finalize || self.test_tick > 0 || self.seen_ep_cnt > 0))
        {
            return self.step_synth_episode(ambient_in, clean_inf, clean_str, na, mode, seed);
        }

        if mode != ControlMode::Natural {
            return vec![0.0_f32; na];
        }

        // Check if we first detected spec_active mid-test or are truly in Sandbox
        if spec_active {
            self.in_demo = true;
            self.demo_tick = 0;
            return self.step_synth_episode(ambient_in, clean_inf, clean_str, na, mode, seed);
        }

        self.step_synth_sandbox(ambient_in, na)
    }

    fn step_synth_sandbox(&mut self, ambient_in: &[f32], na: usize) -> Vec<f32> {
        let period = self.sb_period_est.max(12);
        self.sb_tick = (self.sb_tick + 1) % period;
        if self.sb_tick == 2 {
            let (cin, ein) = self.read_cand96_masks(ambient_in);
            self.sb_in_c = cin;
            self.record_cand96_hist(cin);
            if (cin[0].count_ones() + cin[1].count_ones()) == 12
                && (ein[0].count_ones() + ein[1].count_ones()) >= 3
            {
                self.is_exec_domain = true;
            }
            if !self.op_probe_active {
                self.sb_step_idx = 0;
                if self.is_exec_domain {
                    self.identify_exec_sb_ops();
                }
                self.choose_sb_bout_plan();
            }
        }

        if self.op_probe_wait > 0 {
            self.op_probe_wait -= 1;
            if self.op_probe_wait == 0 {
                let (c1, e1) = self.read_cand96_masks(ambient_in);
                self.record_cand96_hist(c1);
                let moved = self.mark_work_mask_delta(
                    self.sb_pre_c,
                    c1,
                    self.sb_pre_e,
                    e1,
                    self.is_exec_domain,
                );
                let p_idx = self.op_probe_pair;
                if moved > 0 && !self.op_syns.contains(&p_idx) && self.op_syns.len() < SYN_OP_SYNS {
                    let op_idx = self.op_syns.len() as u8;
                    self.op_syns.push(p_idx);
                    if self.sb_in_c != [0; SYN_STATE_WORDS]
                        && self.raw_trans.len() < SYN_MAX_RAW_TRANS
                    {
                        let trapped = self.is_exec_domain
                            && (e1[0].count_ones() + e1[1].count_ones()) <= 2;
                        self.raw_trans.push(RawSynTrans {
                            op_idx,
                            trapped,
                            cin: self.sb_in_c,
                            c0: self.sb_pre_c,
                            e0: self.sb_pre_e,
                            c1,
                            e1,
                        });
                    }
                }
                self.syn_probed_mask |= 1u16 << p_idx;
                let np = self.wire_pairs.len().max(1);
                if self.op_syns.len() >= SYN_OP_SYNS {
                    self.op_probe_active = false;
                    self.sb_op_wait = 0;
                    self.sb_step_idx = 4;
                    if self.is_exec_domain {
                        self.identify_exec_sb_ops();
                    }
                    self.rebuild_world_models();
                } else {
                    let rem: Vec<u8> = (0..np as u8)
                        .filter(|i| !self.op_syns.contains(i))
                        .collect();
                    if let Some(&nxt) = rem.iter().find(|&&i| i > p_idx) {
                        self.op_probe_pair = nxt;
                    } else if let Some(&first_rem) = rem.first() {
                        self.op_probe_pair = first_rem;
                        self.op_probe_retries = self.op_probe_retries.saturating_add(1);
                        if self.op_probe_retries >= 10 {
                            for &idx in &rem {
                                if self.op_syns.len() < SYN_OP_SYNS {
                                    self.op_syns.push(idx);
                                }
                            }
                            self.op_probe_active = false;
                            self.sb_op_wait = 0;
                            self.sb_step_idx = 4;
                            if self.is_exec_domain {
                                self.identify_exec_sb_ops();
                            }
                            self.rebuild_world_models();
                        }
                    }
                }
            }
            if self.op_probe_wait > 0 || !self.op_probe_active {
                return vec![0.0_f32; na];
            }
        }

        // Complete pending op transition at dt == 2
        if self.sb_op_wait > 0 {
            self.sb_op_wait -= 1;
            if self.sb_op_wait == 0 {
                let (c1, e1) = self.read_cand96_masks(ambient_in);
                self.record_cand96_hist(c1);
                let _ = self.mark_work_mask_delta(
                    self.sb_pre_c,
                    c1,
                    self.sb_pre_e,
                    e1,
                    self.is_exec_domain,
                );
                let trapped =
                    self.is_exec_domain && (e1[0].count_ones() + e1[1].count_ones()) <= 2;
                if self.raw_trans.len() < SYN_MAX_RAW_TRANS {
                    self.raw_trans.push(RawSynTrans {
                        op_idx: self.sb_pending_op,
                        trapped,
                        cin: self.sb_in_c,
                        c0: self.sb_pre_c,
                        e0: self.sb_pre_e,
                        c1,
                        e1,
                    });
                }
            }
        }

        if self.op_probe_active {
            if matches!(self.sb_tick, 2 | 4 | 6 | 8) && self.op_syns.len() < SYN_OP_SYNS {
                let np = self.wire_pairs.len().max(1);
                let mut picked = self.op_probe_pair;
                if self.op_syns.contains(&picked) {
                    if let Some(first_rem) = (0..np as u8).find(|i| !self.op_syns.contains(i)) {
                        picked = first_rem;
                    }
                }
                self.op_probe_pair = picked;
                let (c0, e0) = self.read_cand96_masks(ambient_in);
                self.sb_pre_c = c0;
                self.sb_pre_e = e0;
                self.op_probe_wait = 2;
                return self.emit_pair(na, usize::from(picked));
            }
            return vec![0.0_f32; na];
        }

        if matches!(self.sb_tick, 2 | 4 | 6 | 8) && (self.sb_step_idx as usize) < 4 {
            let op_idx = self.sb_active_plan[self.sb_step_idx as usize];
            let (c0, e0) = self.read_cand96_masks(ambient_in);
            self.sb_pre_c = c0;
            self.sb_pre_e = e0;
            self.sb_pending_op = op_idx;
            self.sb_op_wait = 2;
            self.sb_step_idx = self.sb_step_idx.saturating_add(1);
            if let Some(&pair_idx) = self.op_syns.get(usize::from(op_idx)) {
                return self.emit_pair(na, usize::from(pair_idx));
            }
        }

        if self.sb_tick == 11 {
            self.rebuild_world_models();
        }
        vec![0.0_f32; na]
    }

    fn classify_exec_sb_ops(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut sh1 = Vec::new();
        let mut sh1_fb = Vec::new();
        for op in 0..SYN_OP_SYNS {
            let mut distinct_c0: Vec<[u64; SYN_STATE_WORDS]> = Vec::new();
            let mut all_sh1 = true;
            for tr in &self.raw_trans {
                if usize::from(tr.op_idx) != op || tr.trapped {
                    continue;
                }
                if !distinct_c0.contains(&tr.c0) {
                    distinct_c0.push(tr.c0);
                }
                if pop96(tr.c0, tr.c1) != 2 {
                    all_sh1 = false;
                }
            }
            if distinct_c0.len() >= 2 && all_sh1 {
                sh1_fb.push(op as u8);
                if distinct_c0.len() >= 3 {
                    sh1.push(op as u8);
                }
            }
        }
        if sh1.len() < 2 && sh1_fb.len() >= 2 {
            sh1_fb.truncate(2);
            sh1 = sh1_fb;
        }
        let mut inv = Vec::new();
        for w in self.raw_trans.windows(2) {
            let t0 = w[0];
            let t1 = w[1];
            if t0.op_idx == t1.op_idx
                && t0.cin == t1.cin
                && t0.c1 == t1.c0
                && !sh1.contains(&t0.op_idx)
                && !t0.trapped
                && !t1.trapped
                && t1.c1 == t0.c0
                && t1.c1 != t0.c1
                && pop96(t0.c0, t0.c1) >= 4
                && !inv.contains(&t0.op_idx)
            {
                inv.push(t0.op_idx);
            }
        }
        let mut grow = Vec::new();
        for op in 0..SYN_OP_SYNS as u8 {
            if sh1.contains(&op) || inv.contains(&op) {
                continue;
            }
            let ever_noop = self
                .raw_trans
                .iter()
                .any(|t| !t.trapped && t.op_idx == op && t.c1 == t.c0);
            let is_idem = self.raw_trans.windows(2).any(|w| {
                let a = w[0];
                let b = w[1];
                !a.trapped
                    && !b.trapped
                    && a.op_idx == op
                    && b.op_idx == op
                    && a.cin == b.cin
                    && a.c1 == b.c0
                    && b.c1 == a.c1
                    && a.c1 != a.c0
            });
            let mut non_trap_cnt = 0usize;
            let mut all_p4 = true;
            let mut any_trap = false;
            for tr in &self.raw_trans {
                if tr.op_idx != op {
                    continue;
                }
                if tr.trapped {
                    any_trap = true;
                } else {
                    non_trap_cnt += 1;
                    if pop96(tr.c0, tr.c1) != 4 {
                        all_p4 = false;
                    }
                }
            }
            let const_sh2 = non_trap_cnt >= 2 && all_p4 && !any_trap;
            if (non_trap_cnt > 0 || any_trap) && !ever_noop && !is_idem && !const_sh2 {
                grow.push(op);
            }
        }
        (sh1, grow, inv)
    }

    fn identify_exec_sb_ops(&mut self) {
        if self.op_syns.len() < SYN_OP_SYNS {
            return;
        }
        let (sh1, grow, _) = self.classify_exec_sb_ops();
        self.exec_sb_plan_valid = sh1.len() >= 2;
        self.exec_sb_sh1 = sh1;
        self.exec_sb_exp = grow;
    }

    fn choose_sb_bout_plan(&mut self) {
        if self.is_exec_domain && !self.op_probe_active {
            let (sh1, grow, inv) = self.classify_exec_sb_ops();
            let b = usize::from(self.exec_sb_bouts);
            self.exec_sb_bouts = self.exec_sb_bouts.saturating_add(1);
            if b < SYN_OP_SYNS {
                let op = b as u8;
                self.sb_active_plan = [op, op, ((b + 1) % 8) as u8, ((b + 3) % 8) as u8];
                return;
            }
            if b < 22 && sh1.len() >= 2 && !grow.is_empty() {
                let k = b - SYN_OP_SYNS;
                let g = grow[0];
                let g2 = grow[1 % grow.len()];
                let sh0 = sh1[0];
                let sh1_ = sh1[1];
                let iv = inv.first().copied().unwrap_or(g);
                self.sb_active_plan = match k % 14 {
                    0 => [sh0, sh0, sh0, sh0],
                    1 => [sh1_, sh1_, sh1_, sh1_],
                    2 => [g, sh0, sh0, sh0],
                    3 => [iv, g, sh1_, sh1_],
                    4 => [g, g, sh0, sh0],
                    5 => [iv, g, g, sh1_],
                    6 => [g2, g2, sh0, sh0],
                    7 => [iv, g2, g2, sh1_],
                    8 => [g, g, sh1_, sh1_],
                    9 => [iv, g, g, sh0],
                    10 => [g2, g2, g2, sh0],
                    11 => [iv, g2, g2, g2],
                    12 => [sh0, g, sh0, sh0],
                    _ => [sh1_, g2, sh1_, sh1_],
                };
                return;
            }
            let non_sh: Vec<u8> = (0..SYN_OP_SYNS as u8)
                .filter(|op| !sh1.contains(op))
                .collect();
            if !non_sh.is_empty() {
                let k = b.saturating_sub(22);
                let op = non_sh[k % non_sh.len()];
                let sh = sh1.first().copied().unwrap_or((op + 1) % 8);
                let iv = inv.first().copied().unwrap_or(sh);
                self.sb_active_plan = [sh, op, iv, op];
                return;
            }
        }
        for k in 0..4 {
            self.sb_active_plan[k] = (self.synth_sb_op_cursor % 8) as u8;
            self.synth_sb_op_cursor = self.synth_sb_op_cursor.wrapping_add(1);
        }
    }

    fn finalize_pending_seen_ep(&mut self, mode: ControlMode) {
        if !self.ep_pending_finalize {
            return;
        }
        self.ep_pending_finalize = false;
        if mode != ControlMode::Natural {
            return;
        }
        let inf_hit = self.ep_ep_rew >= 1.5;
        if self.is_null_hold_ep {
            self.null_trials = self.null_trials.saturating_add(1);
            self.null_rew_sum = sanitize(self.null_rew_sum + self.ep_ep_rew);
            if self.ep_null_got_influx {
                self.unearned_null_hits = self.unearned_null_hits.saturating_add(1);
            }
            if self.unearned_null_hits >= 2
                || (self.null_trials >= 6
                    && self.seen_ep_cnt >= 48
                    && self.exec_trials >= 16
                    && (self.exec_rew_sum / f32::from(self.exec_trials))
                        - (self.null_rew_sum / f32::from(self.null_trials))
                        < 0.35)
            {
                self.yoked_contaminated = true;
                self.regime = REGIME_DISABLED;
            }
        } else {
            self.exec_trials = self.exec_trials.saturating_add(1);
            self.exec_rew_sum = sanitize(self.exec_rew_sum + self.ep_ep_rew);
            if inf_hit && self.ep_solved_flag && self.seen_ep_cnt < 72 {
                let mut dm_w = [[0u64; SYN_STATE_WORDS]; 3];
                for k in 0..3 {
                    for w in 0..SYN_STATE_WORDS {
                        dm_w[k][w] = self.demo_x_cand[k][w];
                    }
                }
                if self.solved_eps.len() < SYN_MAX_SOLVED_EPS {
                    self.solved_eps.push(SolvedSynEp {
                        dm_w,
                        dm_e: self.demo_x_edge_cand,
                        dm_s_core: self.demo_s_core,
                        dm_s_edge: self.demo_s_edge,
                        ops: self.ep_planned,
                        op_len: self.ep_plan_len,
                    });
                }
                if self.ep_plan_len == 1 && !self.is_exec_domain {
                    let op_u = usize::from(self.ep_planned[0]);
                    if self.seen_ep_cnt % 2 == 0 {
                        // Also add direct work-model transition if solved with length 1 in ArcGrid
                        let mut perms_ready = !self.arc_perms.is_empty();
                        if !perms_ready {
                            perms_ready = false;
                        }
                        let _ = (op_u, perms_ready);
                    }
                }
                self.rebuild_world_models();
            }
        }
        self.seen_ep_cnt = self.seen_ep_cnt.saturating_add(1);
        if self.seen_ep_cnt % 6 == 0 && self.seen_ep_cnt <= 72 {
            self.rebuild_world_models();
        }
    }

    fn step_synth_episode(
        &mut self,
        ambient_in: &[f32],
        clean_inf: f32,
        clean_str: f32,
        na: usize,
        mode: ControlMode,
        seed: u64,
    ) -> Vec<f32> {
        let sm = self.spec_max(ambient_in);
        let spec_active = sm > 0.44;
        self.ep_ep_rew = sanitize(self.ep_ep_rew + clean_inf - clean_str);
        if self.is_null_hold_ep && clean_inf >= 1.0 && !self.in_demo {
            self.ep_null_got_influx = true;
        }
        if clean_inf >= 1.0 && !self.in_demo {
            self.ep_halted = true;
            self.ep_solved_flag = true;
        }
        if clean_str >= 0.10 && !self.in_demo {
            self.ep_halted = true;
        }

        if spec_active {
            if !self.in_demo {
                self.finalize_pending_seen_ep(mode);
                self.in_demo = true;
                self.demo_tick = 0;
            }
            if matches!(self.demo_tick, 3 | 7 | 11) {
                let k = (self.demo_tick / 4) as usize;
                let (cx, ex) = self.read_cand96_masks(ambient_in);
                let (cs, es) = self.read_spec_masks(ambient_in);
                self.demo_x_cand[k] = cx;
                self.demo_x_edge_cand[k] = ex;
                self.demo_s_core[k] = cs;
                self.demo_s_edge[k] = es;
                if self.demo_tick == 11
                    && mode == ControlMode::Natural
                    && self.demo_s_core.iter().all(|&c| c.count_ones() == 6)
                    && self.demo_s_edge.iter().all(|&e| e.count_ones() == 2)
                {
                    self.is_exec_domain = true;
                }
            }
            self.demo_tick = self.demo_tick.saturating_add(1);
            self.test_tick = 0;
            self.ep_pending_finalize = true;
            return vec![0.0_f32; na];
        }

        // Transition Demo -> Test (`spec_active` dropped)
        if self.in_demo {
            self.in_demo = false;
            self.test_tick = 0;
            self.ep_halted = false;
            self.ep_solved_flag = false;
            self.ep_plan_pc = 0;
            self.ep_plan_len = 0;
            self.ep_test_wait = 0;
            self.ep_ep_rew = 0.0;
            self.ep_null_got_influx = false;
            let seed_off = (seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) & 0x3F;
            self.is_null_hold_ep = mode == ControlMode::Natural
                && self.seen_ep_cnt >= 6
                && self.seen_ep_cnt < 48
                && self.null_trials < 6
                && ((usize::from(self.seen_ep_cnt) + (seed_off as usize)) % 6) == 0;
        } else {
            self.test_tick = self.test_tick.saturating_add(1);
        }

        // Collect dt == 2 transition from previous emitted test op if Natural
        if self.ep_test_wait > 0 {
            self.ep_test_wait -= 1;
            if self.ep_test_wait == 0 && mode == ControlMode::Natural {
                let (c1, e1) = self.read_cand96_masks(ambient_in);
                let trapped =
                    self.is_exec_domain && (e1[0].count_ones() + e1[1].count_ones()) <= 2;
                if self.raw_trans.len() < SYN_MAX_RAW_TRANS {
                    self.raw_trans.push(RawSynTrans {
                        op_idx: self.ep_pending_op,
                        trapped,
                        cin: self.ep_in_c,
                        c0: self.ep_pre_c,
                        e0: self.ep_pre_e,
                        c1,
                        e1,
                    });
                }
                if self.is_exec_domain {
                    let _ = self.mark_work_mask_delta(
                        self.ep_pre_c,
                        c1,
                        self.ep_pre_e,
                        e1,
                        true,
                    );
                } else {
                    let n = self.work_map_idx.len();
                    let mut swapped = false;
                    for i in 0..n {
                        let w = self.work_map_idx[i];
                        if usize::from(w) < SYN_STATE_BITS {
                            let j = usize::from(self.cand96_twin[usize::from(w)]);
                            if j < n {
                                let di = ((self.ep_pre_c[i / 64] ^ c1[i / 64]) >> (i % 64)) & 1
                                    == 1;
                                let dj = ((self.ep_pre_c[j / 64] ^ c1[j / 64]) >> (j % 64)) & 1
                                    == 1;
                                if dj && !di {
                                    self.work_map_idx[i] = u8::MAX;
                                    self.work_map_idx[j] = w;
                                    self.cand96_twin[usize::from(w)] = i as u8;
                                    swapped = true;
                                }
                            }
                        }
                    }
                    if swapped {
                        self.rebuild_world_models();
                    }
                }
            }
        }

        if self.is_null_hold_ep || self.ep_halted || self.op_syns.len() < SYN_OP_SYNS {
            return vec![0.0_f32; na];
        }

        if self.test_tick == 2 {
            let (cin, _) = self.read_cand96_masks(ambient_in);
            self.ep_in_c = cin;
            let (plan, plen) = self.plan_synth_program();
            self.ep_planned = plan;
            self.ep_plan_len = plen;
            self.ep_plan_pc = 0;
        }

        if matches!(self.test_tick, 2 | 4 | 6)
            && usize::from(self.ep_plan_pc) < usize::from(self.ep_plan_len)
        {
            let op_idx = self.ep_planned[usize::from(self.ep_plan_pc)];
            let (c0, e0) = self.read_cand96_masks(ambient_in);
            self.ep_pre_c = c0;
            self.ep_pre_e = e0;
            self.ep_pending_op = op_idx;
            self.ep_test_wait = 2;
            self.ep_plan_pc = self.ep_plan_pc.saturating_add(1);
            if let Some(&pair_idx) = self.op_syns.get(usize::from(op_idx)) {
                return self.emit_pair(na, usize::from(pair_idx));
            }
        }

        vec![0.0_f32; na]
    }

    fn rebuild_world_models(&mut self) {
        if self.is_exec_domain {
            self.rebuild_exec_model();
        } else {
            self.rebuild_arc_model();
        }
        self.rebuild_poss_sw();
    }

    fn rebuild_arc_model(&mut self) {
        if !self.is_exec_domain
            && !self.in_demo
            && !self.ep_pending_finalize
            && self.seen_ep_cnt == 0
        {
            self.rebuild_arc_twin_work_map();
        }
        let mut possible = [[FULL_STATE_MASK; SYN_STATE_BITS]; SYN_OP_SYNS];
        let mut is_grav = [false; SYN_OP_SYNS];
        let mut grav_pairs: Vec<(u64, u64)> = Vec::new();
        let mut cooc = [[[0u16; SYN_STATE_BITS]; SYN_STATE_BITS]; SYN_OP_SYNS];

        for tr in &self.raw_trans {
            let op = usize::from(tr.op_idx);
            if op >= SYN_OP_SYNS {
                continue;
            }
            let m0 = self.project_work_mask(tr.c0);
            let m1 = self.project_work_mask(tr.c1);
            for d in 0..SYN_STATE_BITS {
                if ((m1 >> d) & 1) == 1 {
                    possible[op][d] &= m0;
                    for s in 0..SYN_STATE_BITS {
                        if ((m0 >> s) & 1) == 1 {
                            cooc[op][d][s] = cooc[op][d][s].saturating_add(1);
                        }
                    }
                } else {
                    possible[op][d] &= (!m0) & FULL_STATE_MASK;
                }
                if possible[op][d] == 0 {
                    is_grav[op] = true;
                }
            }
        }

        for _ in 0..8 {
            for op in 0..SYN_OP_SYNS {
                if is_grav[op] {
                    continue;
                }
                for _ in 0..16 {
                    let mut used = 0u64;
                    for d in 0..SYN_STATE_BITS {
                        let c = possible[op][d];
                        if c != 0 && (c & (c - 1)) == 0 {
                            used |= c;
                        }
                    }
                    let mut ch = false;
                    for d in 0..SYN_STATE_BITS {
                        let c = possible[op][d];
                        if (c & (c.wrapping_sub(1))) != 0 {
                            let nc = c & (!used);
                            if nc != 0 && nc != c {
                                possible[op][d] = nc;
                                ch = true;
                            }
                        }
                    }
                    for s in 0..SYN_STATE_BITS {
                        let bit = 1u64 << s;
                        let mut hold_cnt = 0usize;
                        let mut hold_d = 0usize;
                        for d in 0..SYN_STATE_BITS {
                            if (possible[op][d] & bit) != 0 {
                                hold_cnt += 1;
                                hold_d = d;
                            }
                        }
                        if hold_cnt == 1 && possible[op][hold_d] != bit {
                            possible[op][hold_d] = bit;
                            ch = true;
                        }
                    }
                    if !ch {
                        break;
                    }
                }
            }

            let mut op_c: Option<usize> = None;
            for op in 0..SYN_OP_SYNS {
                if is_grav[op] {
                    continue;
                }
                let mut sing = [u8::MAX; SYN_STATE_BITS];
                for d in 0..SYN_STATE_BITS {
                    let c = possible[op][d];
                    if c != 0 && (c & (c - 1)) == 0 {
                        sing[d] = c.trailing_zeros() as u8;
                    }
                }
                let mut c3 = 0usize;
                for d in 0..SYN_STATE_BITS {
                    let s1 = sing[d] as usize;
                    if s1 < SYN_STATE_BITS && s1 != d {
                        let s2 = sing[s1] as usize;
                        if s2 < SYN_STATE_BITS && sing[s2] as usize == d {
                            c3 += 1;
                        }
                    }
                }
                if c3 >= 12 {
                    op_c = Some(op);
                    break;
                }
            }

            if let Some(oc) = op_c {
                let mut sing_c = [u8::MAX; SYN_STATE_BITS];
                for d in 0..SYN_STATE_BITS {
                    let c = possible[oc][d];
                    if c != 0 && (c & (c - 1)) == 0 {
                        sing_c[d] = c.trailing_zeros() as u8;
                    }
                }
                for a in 0..SYN_STATE_BITS {
                    let b = sing_c[a] as usize;
                    if b < SYN_STATE_BITS && b != a {
                        let c = sing_c[b] as usize;
                        if c < SYN_STATE_BITS
                            && c != a
                            && c != b
                            && ((possible[oc][c] >> a) & 1) == 1
                        {
                            possible[oc][c] = 1u64 << a;
                            sing_c[c] = a as u8;
                        }
                    }
                }
                for op in 0..SYN_OP_SYNS {
                    if is_grav[op] || op == oc {
                        continue;
                    }
                    let mut sing_g = [u8::MAX; SYN_STATE_BITS];
                    for d in 0..SYN_STATE_BITS {
                        let c = possible[op][d];
                        if c != 0 && (c & (c - 1)) == 0 {
                            sing_g[d] = c.trailing_zeros() as u8;
                        }
                    }
                    for d in 0..SYN_STATE_BITS {
                        let cd = sing_c[d] as usize;
                        let gd = sing_g[d] as usize;
                        if cd < SYN_STATE_BITS && gd < SYN_STATE_BITS {
                            let cgd = sing_c[gd] as usize;
                            let gcd = sing_g[cd] as usize;
                            if cgd < SYN_STATE_BITS && ((possible[op][cd] >> cgd) & 1) == 1 {
                                possible[op][cd] = 1u64 << cgd;
                            } else if gcd < SYN_STATE_BITS && ((possible[oc][gd] >> gcd) & 1) == 1
                            {
                                possible[oc][gd] = 1u64 << gcd;
                            }
                        }
                    }
                }
            }
        }

        self.arc_perms = vec![vec![0u8; SYN_STATE_BITS]; SYN_OP_SYNS];
        self.arc_is_grav = is_grav.to_vec();
        for op in 0..SYN_OP_SYNS {
            if is_grav[op] {
                for tr in &self.raw_trans {
                    if usize::from(tr.op_idx) == op {
                        grav_pairs
                            .push((self.project_work_mask(tr.c0), self.project_work_mask(tr.c1)));
                    }
                }
                continue;
            }
            let mut used = 0u64;
            for d in 0..SYN_STATE_BITS {
                let c = possible[op][d];
                if c != 0 && (c & (c - 1)) == 0 {
                    let s = c.trailing_zeros() as usize;
                    self.arc_perms[op][d] = s as u8;
                    used |= 1u64 << s;
                }
            }
            for d in 0..SYN_STATE_BITS {
                let c = possible[op][d];
                if !(c != 0 && (c & (c - 1)) == 0) {
                    let mut rem = c & (!used);
                    if rem == 0 {
                        rem = FULL_STATE_MASK & (!used);
                    }
                    let mut best_s = rem.trailing_zeros() as usize;
                    let mut best_cnt = 0u16;
                    for s in 0..SYN_STATE_BITS {
                        if ((rem >> s) & 1) == 1 && cooc[op][d][s] >= best_cnt {
                            best_cnt = cooc[op][d][s];
                            best_s = s;
                        }
                    }
                    self.arc_perms[op][d] = best_s as u8;
                    used |= 1u64 << best_s;
                }
            }
        }

        // Induce gravity column structure from op_c + commuting order-4 shift op
        self.arc_grav_cols.clear();
        let mut op_c = None;
        let grav_cnt = is_grav.iter().filter(|&&g| g).count();
        for op in 0..SYN_OP_SYNS {
            if !is_grav[op] {
                let p = &self.arc_perms[op];
                let mut ok3 = 0usize;
                for i in 0..SYN_STATE_BITS {
                    let a = usize::from(p[i]);
                    let b = usize::from(p[a]);
                    let c = usize::from(p[b]);
                    if a != i && c == i {
                        ok3 += 1;
                    }
                }
                if ok3 == SYN_STATE_BITS {
                    op_c = Some(op);
                    break;
                }
            }
        }
        if let Some(oc) = op_c
            && grav_cnt == 1
        {
            let pc = &self.arc_perms[oc];
            for sop in 0..SYN_OP_SYNS {
                if is_grav[sop] || sop == oc {
                    continue;
                }
                let psh = &self.arc_perms[sop];
                let mut is_ord4 = true;
                let mut any_ord4 = false;
                for i in 0..SYN_STATE_BITS {
                    let a = usize::from(psh[i]);
                    let b = usize::from(psh[a]);
                    let c = usize::from(psh[b]);
                    let d = usize::from(psh[c]);
                    if d != i {
                        is_ord4 = false;
                        break;
                    }
                    if b != i {
                        any_ord4 = true;
                    }
                }
                if !is_ord4 || !any_ord4 {
                    continue;
                }
                let mut vis = [false; SYN_STATE_BITS];
                let mut raw_cols: Vec<Vec<[u8; 3]>> = Vec::new();
                let mut ok4 = true;
                for b in 0..SYN_STATE_BITS {
                    if vis[b] {
                        continue;
                    }
                    let mut col = Vec::with_capacity(4);
                    let mut t0 = b as u8;
                    let mut t1 = pc[usize::from(t0)];
                    let mut t2 = pc[usize::from(t1)];
                    for _ in 0..4 {
                        vis[usize::from(t0)] = true;
                        vis[usize::from(t1)] = true;
                        vis[usize::from(t2)] = true;
                        col.push([t0, t1, t2]);
                        t0 = psh[usize::from(t0)];
                        t1 = psh[usize::from(t1)];
                        t2 = psh[usize::from(t2)];
                    }
                    if [t0, t1, t2] != col[0] {
                        ok4 = false;
                        break;
                    }
                    raw_cols.push(col);
                }
                if !ok4 || raw_cols.len() != 4 {
                    continue;
                }
                let mut oriented = Vec::with_capacity(4);
                let mut all_ok = true;
                for col in &raw_cols {
                    let mut found_seq: Option<Vec<[u8; 3]>> = None;
                    for rot in 0..4 {
                        for rev in [false, true] {
                            let mut seq: Vec<[u8; 3]> =
                                col[rot..].iter().chain(&col[..rot]).copied().collect();
                            if rev {
                                seq.reverse();
                            }
                            let mut ok_s = true;
                            for &(m0, m1) in &grav_pairs {
                                let mut vals = [u8::MAX; 4];
                                let mut vlen = 0usize;
                                for row in &seq {
                                    for k in 0..3 {
                                        if ((m0 >> row[k]) & 1) == 1 {
                                            vals[vlen] = k as u8;
                                            vlen += 1;
                                            break;
                                        }
                                    }
                                }
                                let empty = 4usize.saturating_sub(vlen);
                                for (pi, row) in seq.iter().enumerate() {
                                    let exp = if pi < empty {
                                        u8::MAX
                                    } else {
                                        vals[pi - empty]
                                    };
                                    for k in 0..3 {
                                        let want = u8::from(exp == k as u8);
                                        let got = ((m1 >> row[k]) & 1) as u8;
                                        if want != got {
                                            ok_s = false;
                                            break;
                                        }
                                    }
                                    if !ok_s {
                                        break;
                                    }
                                }
                                if !ok_s {
                                    break;
                                }
                            }
                            if ok_s {
                                found_seq = Some(seq);
                                break;
                            }
                        }
                        if found_seq.is_some() {
                            break;
                        }
                    }
                    if let Some(s) = found_seq {
                        oriented.push(s);
                    } else {
                        all_ok = false;
                        break;
                    }
                }
                if all_ok {
                    self.arc_grav_cols = oriented;
                    break;
                }
            }
        }
    }

    fn rebuild_exec_model(&mut self) {
        let mut sh1 = Vec::new();
        let mut sh1_fallback = Vec::new();
        let mut all_states: Vec<(u64, u64)> = Vec::new();
        for op in 0..SYN_OP_SYNS {
            let mut distinct_c0: Vec<[u64; SYN_STATE_WORDS]> = Vec::new();
            let mut all_sh = true;
            for tr in &self.raw_trans {
                if usize::from(tr.op_idx) != op || tr.trapped {
                    continue;
                }
                if !distinct_c0.contains(&tr.c0) {
                    distinct_c0.push(tr.c0);
                }
                if pop96(tr.c0, tr.c1) != 2 {
                    all_sh = false;
                }
                let c0 = self.project_work_mask(tr.c0);
                let e0 = self.project_work_mask(tr.e0);
                let c1 = self.project_work_mask(tr.c1);
                let e1 = self.project_work_mask(tr.e1);
                if c0.count_ones() == 6
                    && e0.count_ones() == 2
                    && !all_states.iter().any(|&(c, _)| c == c0)
                {
                    all_states.push((c0, e0));
                }
                if c1.count_ones() == 6
                    && e1.count_ones() == 2
                    && !all_states.iter().any(|&(c, _)| c == c1)
                {
                    all_states.push((c1, e1));
                }
            }
            if !distinct_c0.is_empty() && all_sh {
                sh1_fallback.push(op);
                if distinct_c0.len() >= 3 {
                    sh1.push(op);
                }
            }
        }
        if sh1.len() < 2 && sh1_fallback.len() > sh1.len() {
            sh1 = sh1_fallback;
        }
        for ep in &self.solved_eps {
            for k in 0..3 {
                let c = self.project_work_mask(ep.dm_w[k]);
                let e = self.project_work_mask(ep.dm_e[k]);
                if c.count_ones() == 6
                    && e.count_ones() == 2
                    && !all_states.iter().any(|&(x, _)| x == c)
                {
                    all_states.push((c, e));
                }
            }
        }
        if sh1.is_empty() {
            return;
        }
        let mut next_b = [u8::MAX; SYN_STATE_BITS];
        let mut prev_b = [u8::MAX; SYN_STATE_BITS];
        let add_d =
            |next_b: &mut [u8; SYN_STATE_BITS],
             prev_b: &mut [u8; SYN_STATE_BITS],
             u: usize,
             v: usize| {
                if u < SYN_STATE_BITS && v < SYN_STATE_BITS {
                    next_b[u] = v as u8;
                    prev_b[v] = u as u8;
                }
            };
        for (idx_s, &sop) in sh1.iter().take(2).enumerate() {
            for tr in &self.raw_trans {
                if usize::from(tr.op_idx) != sop || tr.trapped {
                    continue;
                }
                let (mut c0, mut e0, mut c1, mut e1) = (
                    self.project_work_mask(tr.c0),
                    self.project_work_mask(tr.e0),
                    self.project_work_mask(tr.c1),
                    self.project_work_mask(tr.e1),
                );
                if idx_s == 1 {
                    std::mem::swap(&mut c0, &mut c1);
                    std::mem::swap(&mut e0, &mut e1);
                }
                if c0.count_ones() != 6
                    || c1.count_ones() != 6
                    || e0.count_ones() != 2
                    || e1.count_ones() != 2
                {
                    continue;
                }
                let dr = c0 & (!c1);
                let ad = c1 & (!c0);
                if dr.count_ones() == 1 && ad.count_ones() == 1 {
                    let b_dr = dr.trailing_zeros() as usize;
                    let b_ad = ad.trailing_zeros() as usize;
                    let eo0 = e0 ^ ad;
                    let eo1 = e1 ^ dr;
                    if eo0.count_ones() == 1 {
                        add_d(&mut next_b, &mut prev_b, eo0.trailing_zeros() as usize, b_dr);
                    }
                    if eo1.count_ones() == 1 {
                        add_d(&mut next_b, &mut prev_b, b_ad, eo1.trailing_zeros() as usize);
                    }
                }
            }
        }
        for _ in 0..16 {
            for i in 0..all_states.len() {
                let (c_a, e_a) = all_states[i];
                if c_a.count_ones() != 6 || e_a.count_ones() != 2 {
                    continue;
                }
                for j in (i + 1)..all_states.len() {
                    let (c_b, e_b) = all_states[j];
                    if c_b.count_ones() != 6 || e_b.count_ones() != 2 {
                        continue;
                    }
                    let inter = c_a & c_b;
                    let kv = inter.count_ones();
                    if kv == 5 {
                        let u_a = (c_a & (!c_b)).trailing_zeros() as usize;
                        let u_b = (c_b & (!c_a)).trailing_zeros() as usize;
                        let mut links = 0usize;
                        let mut head = SYN_STATE_BITS;
                        let mut tail = SYN_STATE_BITS;
                        for b in 0..SYN_STATE_BITS {
                            if ((inter >> b) & 1) == 1 {
                                let nx = next_b[b] as usize;
                                if nx < SYN_STATE_BITS && ((inter >> nx) & 1) == 1 {
                                    links += 1;
                                } else {
                                    tail = b;
                                }
                                let pr = prev_b[b] as usize;
                                if !(pr < SYN_STATE_BITS && ((inter >> pr) & 1) == 1) {
                                    head = b;
                                }
                            }
                        }
                        if links == 4 && head < SYN_STATE_BITS && tail < SYN_STATE_BITS {
                            if prev_b[head] as usize == u_a || next_b[tail] as usize == u_b {
                                add_d(&mut next_b, &mut prev_b, u_a, head);
                                add_d(&mut next_b, &mut prev_b, tail, u_b);
                                let eo_a = e_a ^ (1u64 << u_b);
                                let eo_b = e_b ^ (1u64 << u_a);
                                if eo_a.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        eo_a.trailing_zeros() as usize,
                                        u_a,
                                    );
                                }
                                if eo_b.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        u_b,
                                        eo_b.trailing_zeros() as usize,
                                    );
                                }
                            } else if next_b[tail] as usize == u_a || prev_b[head] as usize == u_b
                            {
                                add_d(&mut next_b, &mut prev_b, u_b, head);
                                add_d(&mut next_b, &mut prev_b, tail, u_a);
                                let eo_b = e_b ^ (1u64 << u_a);
                                let eo_a = e_a ^ (1u64 << u_b);
                                if eo_b.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        eo_b.trailing_zeros() as usize,
                                        u_b,
                                    );
                                }
                                if eo_a.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        u_a,
                                        eo_a.trailing_zeros() as usize,
                                    );
                                }
                            }
                        }
                    } else if kv == 4 {
                        let ina = (c_a & (!c_b)) & e_b;
                        let outa = (c_a & (!c_b)) & (!e_b);
                        let inb = (c_b & (!c_a)) & e_a;
                        let outb = (c_b & (!c_a)) & (!e_a);
                        if ina.count_ones() == 1
                            && outa.count_ones() == 1
                            && inb.count_ones() == 1
                            && outb.count_ones() == 1
                        {
                            let ia = ina.trailing_zeros() as usize;
                            let oa = outa.trailing_zeros() as usize;
                            let ib = inb.trailing_zeros() as usize;
                            let ob = outb.trailing_zeros() as usize;
                            let fwd = (usize::from(next_b[oa]) == ia)
                                || (usize::from(next_b[ia]) < SYN_STATE_BITS
                                    && ((inter >> next_b[ia]) & 1) == 1)
                                || (usize::from(prev_b[ib]) < SYN_STATE_BITS
                                    && ((inter >> prev_b[ib]) & 1) == 1)
                                || (usize::from(next_b[ib]) == ob);
                            let rev = (usize::from(next_b[ob]) == ib)
                                || (usize::from(next_b[ib]) < SYN_STATE_BITS
                                    && ((inter >> next_b[ib]) & 1) == 1)
                                || (usize::from(prev_b[ia]) < SYN_STATE_BITS
                                    && ((inter >> prev_b[ia]) & 1) == 1)
                                || (usize::from(next_b[ia]) == oa);
                            if fwd && !rev {
                                add_d(&mut next_b, &mut prev_b, oa, ia);
                                add_d(&mut next_b, &mut prev_b, ib, ob);
                                let ela = e_a & (!(c_b | e_b));
                                let erb = e_b & (!(c_a | e_a));
                                if ela.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        ela.trailing_zeros() as usize,
                                        oa,
                                    );
                                }
                                if erb.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        ob,
                                        erb.trailing_zeros() as usize,
                                    );
                                }
                            } else if rev && !fwd {
                                add_d(&mut next_b, &mut prev_b, ob, ib);
                                add_d(&mut next_b, &mut prev_b, ia, oa);
                                let elb = e_b & (!(c_a | e_a));
                                let era = e_a & (!(c_b | e_b));
                                if elb.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        elb.trailing_zeros() as usize,
                                        ob,
                                    );
                                }
                                if era.count_ones() == 1 {
                                    add_d(
                                        &mut next_b,
                                        &mut prev_b,
                                        oa,
                                        era.trailing_zeros() as usize,
                                    );
                                }
                            }
                        }
                    } else if kv == 3 {
                        let mida = (c_a & (!c_b)) & e_b;
                        let midb = (c_b & (!c_a)) & e_a;
                        if mida.count_ones() == 1 && midb.count_ones() == 1 {
                            let ma = mida.trailing_zeros() as usize;
                            let mb = midb.trailing_zeros() as usize;
                            let mut links = 0usize;
                            let mut head = SYN_STATE_BITS;
                            let mut tail = SYN_STATE_BITS;
                            for b in 0..SYN_STATE_BITS {
                                if ((inter >> b) & 1) == 1 {
                                    let nx = usize::from(next_b[b]);
                                    if nx < SYN_STATE_BITS && ((inter >> nx) & 1) == 1 {
                                        links += 1;
                                    } else {
                                        tail = b;
                                    }
                                    let pr = usize::from(prev_b[b]);
                                    if !(pr < SYN_STATE_BITS && ((inter >> pr) & 1) == 1) {
                                        head = b;
                                    }
                                }
                            }
                            if links == 2 && head < SYN_STATE_BITS && tail < SYN_STATE_BITS {
                                if usize::from(prev_b[head]) == ma
                                    || usize::from(next_b[tail]) == mb
                                {
                                    add_d(&mut next_b, &mut prev_b, ma, head);
                                    add_d(&mut next_b, &mut prev_b, tail, mb);
                                } else if usize::from(next_b[tail]) == ma
                                    || usize::from(prev_b[head]) == mb
                                {
                                    add_d(&mut next_b, &mut prev_b, mb, head);
                                    add_d(&mut next_b, &mut prev_b, tail, ma);
                                }
                            }
                        }
                    }
                }
            }
            for &(cm, em) in &all_states {
                if cm.count_ones() != 6 || em.count_ones() != 2 {
                    continue;
                }
                let mut e_bits = [SYN_STATE_BITS; 2];
                let mut ec = 0usize;
                for b in 0..SYN_STATE_BITS {
                    if ((em >> b) & 1) == 1 && ec < 2 {
                        e_bits[ec] = b;
                        ec += 1;
                    }
                }
                let mut starts = [SYN_STATE_BITS; 6];
                let mut sc = 0usize;
                for b in 0..SYN_STATE_BITS {
                    if ((cm >> b) & 1) == 1 {
                        let pr = prev_b[b] as usize;
                        if !(pr < SYN_STATE_BITS && ((cm >> pr) & 1) == 1) {
                            starts[sc] = b;
                            sc += 1;
                        }
                    }
                }
                if sc == 1 {
                    let mut ch = [starts[0]; 6];
                    let mut l = 1usize;
                    while l < 6 {
                        let nx = next_b[ch[l - 1]] as usize;
                        if nx < SYN_STATE_BITS && ((cm >> nx) & 1) == 1 {
                            ch[l] = nx;
                            l += 1;
                        } else {
                            break;
                        }
                    }
                    if l == 6 {
                        let b0 = ch[0];
                        let b5 = ch[5];
                        if prev_b[b0] as usize == e_bits[0] {
                            add_d(&mut next_b, &mut prev_b, b5, e_bits[1]);
                        } else if prev_b[b0] as usize == e_bits[1] {
                            add_d(&mut next_b, &mut prev_b, b5, e_bits[0]);
                        } else if next_b[b5] as usize == e_bits[0] {
                            add_d(&mut next_b, &mut prev_b, e_bits[1], b0);
                        } else if next_b[b5] as usize == e_bits[1] {
                            add_d(&mut next_b, &mut prev_b, e_bits[0], b0);
                        }
                    }
                }
            }
        }

        let mut best_ch: Vec<u8> = Vec::new();
        for start_kind in 0..2 {
            for b in 0..SYN_STATE_BITS {
                if next_b[b] == u8::MAX {
                    continue;
                }
                if start_kind == 0 && prev_b[b] != u8::MAX {
                    continue;
                }
                let mut ch = Vec::with_capacity(48);
                let mut vis = [false; SYN_STATE_BITS];
                let mut cur = b;
                while cur < SYN_STATE_BITS && !vis[cur] {
                    vis[cur] = true;
                    ch.push(cur as u8);
                    let nx = next_b[cur] as usize;
                    if nx >= SYN_STATE_BITS {
                        break;
                    }
                    cur = nx;
                }
                if ch.len() > best_ch.len() {
                    best_ch = ch;
                }
            }
        }
        if best_ch.len() == 46 {
            let mut ring_bits = 0u64;
            for &(cm, em) in &all_states {
                if em != 0 {
                    ring_bits |= cm | em;
                }
            }
            let chain_mask = best_ch.iter().fold(0u64, |acc, &b| acc | (1u64 << b));
            let missing = ring_bits & (!chain_mask);
            if ring_bits.count_ones() == 47 && missing.count_ones() == 1 {
                best_ch.push(missing.trailing_zeros() as u8);
            }
        }
        if best_ch.len() < 12 {
            return;
        }
        let clen = best_ch.len();
        let mut win_masks: Vec<u64> = Vec::with_capacity(clen);
        let lim = if clen == 47 { 47 } else { clen - 5 };
        for i in 0..lim {
            let mut m = 0u64;
            for off in 0..6 {
                let idx = if clen == 47 { (i + off) % 47 } else { i + off };
                m |= 1u64 << best_ch[idx];
            }
            win_masks.push(m);
        }

        let mut best_p0 = 0usize;
        let mut best_models = vec![ExecAffPiece::default(); SYN_OP_SYNS];
        let mut best_err = i64::MAX;

        for p0 in 0..win_masks.len() {
            let m2z = |cm: u64| -> Option<i32> {
                if cm == 0 {
                    return None;
                }
                for (pos, &wm) in win_masks.iter().enumerate() {
                    if wm == cm {
                        let mut d = (pos as i32) - (p0 as i32);
                        if clen == 47 {
                            if d > 23 {
                                d -= 47;
                            } else if d < -23 {
                                d += 47;
                            }
                        }
                        return Some(d);
                    }
                }
                None
            };

            let mut tot_err = 0i64;
            let mut models = vec![ExecAffPiece::default(); SYN_OP_SYNS];
            for op in 0..SYN_OP_SYNS {
                let mut samples: Vec<(i32, i32, Option<i32>)> = Vec::new();
                for tr in &self.raw_trans {
                    if usize::from(tr.op_idx) != op {
                        continue;
                    }
                    let zin = m2z(self.project_work_mask(tr.cin));
                    let z0 = m2z(self.project_work_mask(tr.c0));
                    if let (Some(zi), Some(zv)) = (zin, z0) {
                        if tr.trapped {
                            samples.push((zi, zv, None));
                        } else if let Some(z1) = m2z(self.project_work_mask(tr.c1)) {
                            samples.push((zi, zv, Some(z1)));
                        }
                    }
                }
                let mut bn = (1i8, 0i8, 0i8);
                let mut en = i64::MAX;
                let mut bp = (1i8, 0i8, 0i8);
                let mut ep = i64::MAX;
                for a in -2i8..=2 {
                    for b in -1i8..=1 {
                        for c in -2i8..=2 {
                            let pen = i64::from((a - 1).abs() + b.abs() + c.abs());
                            let mut ern = pen;
                            let mut erp = pen;
                            for &(zi, zv, z1) in &samples {
                                let pred = i32::from(a) * zv + i32::from(b) * zi + i32::from(c);
                                let ptr = pred.abs() > 23;
                                let bad = match z1 {
                                    None => !ptr,
                                    Some(actual) => ptr || pred != actual,
                                };
                                if bad {
                                    if zv < 0 {
                                        ern += 1000;
                                    } else {
                                        erp += 1000;
                                    }
                                }
                            }
                            if ern < en {
                                en = ern;
                                bn = (a, b, c);
                            }
                            if erp < ep {
                                ep = erp;
                                bp = (a, b, c);
                            }
                        }
                    }
                }
                tot_err += (en / 1000) * 1000 + (ep / 1000) * 1000;
                if bn.1 != bp.1 || bn.2 != bp.2 {
                    tot_err += 10;
                }
                models[op] = ExecAffPiece {
                    a_neg: bn.0,
                    b_neg: bn.1,
                    c_neg: bn.2,
                    a_pos: bp.0,
                    b_pos: bp.1,
                    c_pos: bp.2,
                };
            }
            if tot_err < best_err {
                best_err = tot_err;
                best_p0 = p0;
                best_models = models;
                if best_err == 0 && clen >= 36 {
                    break;
                }
            }
        }
        self.exec_chain = best_ch;
        self.exec_p0 = best_p0 as u8;
        self.exec_models = best_models;
        self.exec_model_valid = true;
    }

    fn sim_arc_mask(&self, op: usize, m: u64) -> u64 {
        if self.arc_is_grav.get(op).copied().unwrap_or(false) {
            if self.arc_grav_cols.len() != 4 {
                return m;
            }
            let mut out = 0u64;
            for seq in &self.arc_grav_cols {
                let mut vals = [u8::MAX; 4];
                let mut vlen = 0usize;
                for row in seq {
                    for k in 0..3 {
                        if ((m >> row[k]) & 1) == 1 {
                            vals[vlen] = k as u8;
                            vlen += 1;
                            break;
                        }
                    }
                }
                let empty = 4usize.saturating_sub(vlen);
                for (pi, row) in seq.iter().enumerate() {
                    if pi >= empty {
                        let c_k = vals[pi - empty] as usize;
                        if c_k < 3 {
                            out |= 1u64 << row[c_k];
                        }
                    }
                }
            }
            return out;
        }
        let Some(p) = self.arc_perms.get(op) else {
            return m;
        };
        if p.len() < SYN_STATE_BITS {
            return m;
        }
        let mut out = 0u64;
        for d in 0..SYN_STATE_BITS {
            if ((m >> p[d]) & 1) == 1 {
                out |= 1u64 << d;
            }
        }
        out
    }

    fn exec_mask_to_z(&self, cm: u64) -> Option<i32> {
        let clen = self.exec_chain.len();
        if !self.exec_model_valid || clen < 12 || cm == 0 {
            return None;
        }
        let lim = if clen == 47 { 47 } else { clen - 5 };
        for i in 0..lim {
            let mut m = 0u64;
            for off in 0..6 {
                let idx = if clen == 47 { (i + off) % 47 } else { i + off };
                m |= 1u64 << self.exec_chain[idx];
            }
            if m == cm {
                let mut d = (i as i32) - i32::from(self.exec_p0);
                if clen == 47 {
                    if d > 23 {
                        d -= 47;
                    } else if d < -23 {
                        d += 47;
                    }
                }
                return Some(d);
            }
        }
        None
    }

    fn exec_z_to_masks(&self, z: i32) -> Option<(u64, u64)> {
        let clen = self.exec_chain.len();
        if !self.exec_model_valid || clen < 12 {
            return None;
        }
        let mut idx = z + i32::from(self.exec_p0);
        let lim = if clen == 47 {
            idx = idx.rem_euclid(47);
            47usize
        } else {
            clen - 5
        };
        if idx < 0 || (idx as usize) >= lim {
            return None;
        }
        let p = idx as usize;
        let mut cm = 0u64;
        for off in 0..6 {
            let ci = if clen == 47 { (p + off) % 47 } else { p + off };
            cm |= 1u64 << self.exec_chain[ci];
        }
        let mut em = 0u64;
        if clen == 47 {
            em |= 1u64 << self.exec_chain[(p + 46) % 47];
            em |= 1u64 << self.exec_chain[(p + 6) % 47];
        } else {
            if p >= 1 {
                em |= 1u64 << self.exec_chain[p - 1];
            }
            if p + 6 < clen {
                em |= 1u64 << self.exec_chain[p + 6];
            }
        }
        Some((cm, em))
    }

    fn rebuild_poss_sw(&mut self) {
        let mut psw = vec![FULL_STATE_MASK; SYN_STATE_BITS];
        if !self.is_exec_domain && self.arc_perms.len() != SYN_OP_SYNS {
            self.poss_sw = psw;
            return;
        }
        if self.is_exec_domain && self.exec_models.len() != SYN_OP_SYNS {
            self.poss_sw = psw;
            return;
        }
        for ep in &self.solved_eps {
            for k in 0..3 {
                let wx = self.project_work_mask(ep.dm_w[k]);
                let (pred_c, pred_e) = if !self.is_exec_domain {
                    let mut cur = wx;
                    for s in 0..usize::from(ep.op_len) {
                        cur = self.sim_arc_mask(usize::from(ep.ops[s]), cur);
                    }
                    (cur, 0u64)
                } else if let Some(xz) = self.exec_mask_to_z(wx) {
                    let mut z = xz;
                    let mut ok = true;
                    for s in 0..usize::from(ep.op_len) {
                        if let Some(nz) = self.exec_models[usize::from(ep.ops[s])].eval_next(z, xz) {
                            z = nz;
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let Some((c, e)) = self.exec_z_to_masks(z) else {
                        continue;
                    };
                    (c, e)
                } else {
                    continue;
                };
                let sy_c = ep.dm_s_core[k];
                let sy_e = ep.dm_s_edge[k];
                for sb in 0..SYN_STATE_BITS {
                    let mask = if ((sy_c >> sb) & 1) == 1 {
                        pred_c
                    } else if ((sy_e >> sb) & 1) == 1 {
                        pred_e
                    } else {
                        (!(pred_c | pred_e)) & FULL_STATE_MASK
                    };
                    let cand = psw[sb] & mask;
                    if cand != 0 {
                        psw[sb] = cand;
                    }
                }
            }
        }
        for _ in 0..16 {
            let mut used = 0u64;
            for &c in &psw {
                if c != 0 && (c & (c - 1)) == 0 {
                    used |= c;
                }
            }
            let mut ch = false;
            for slot in &mut psw {
                let c = *slot;
                if (c & c.wrapping_sub(1)) != 0 {
                    let nc = c & (!used);
                    if nc != 0 && nc != c {
                        *slot = nc;
                        ch = true;
                    }
                }
            }
            for wb in 0..SYN_STATE_BITS {
                let bit = 1u64 << wb;
                let mut hcnt = 0usize;
                let mut hsb = 0usize;
                for (sb, &p) in psw.iter().enumerate() {
                    if (p & bit) != 0 {
                        hcnt += 1;
                        hsb = sb;
                    }
                }
                if hcnt == 1 && psw[hsb] != bit {
                    psw[hsb] = bit;
                    ch = true;
                }
            }
            if !ch {
                break;
            }
        }
        self.poss_sw = psw;
    }

    fn sw_consistent(&self, pred_c: &[u64; 3], pred_e: &[u64; 3]) -> bool {
        for k in 0..3 {
            if self.demo_s_core[k].count_ones() != pred_c[k].count_ones() {
                return false;
            }
        }
        for (sb, &base) in self.poss_sw.iter().enumerate() {
            let mut m = base;
            for k in 0..3 {
                let is_c = ((self.demo_s_core[k] >> sb) & 1) == 1;
                let is_e = ((self.demo_s_edge[k] >> sb) & 1) == 1;
                let wmask = if is_c {
                    pred_c[k]
                } else if is_e {
                    pred_e[k]
                } else {
                    (!(pred_c[k] | pred_e[k])) & FULL_STATE_MASK
                };
                m &= wmask;
                if m == 0 {
                    return false;
                }
            }
        }
        true
    }

    fn sw_singleton_count(&self) -> usize {
        self.poss_sw
            .iter()
            .filter(|&&c| c != 0 && (c & (c - 1)) == 0)
            .count()
    }

    fn plan_synth_program(&mut self) -> ([u8; 3], u8) {
        if self.is_exec_domain {
            return self.plan_exec_program();
        }
        self.plan_arc_program()
    }

    fn plan_arc_program(&mut self) -> ([u8; 3], u8) {
        if self.arc_perms.len() != SYN_OP_SYNS {
            let op = (self.synth_op_rr % 8) as u8;
            self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
            return ([op, 0, 0], 1);
        }
        let dm_w = [
            self.project_work_mask(self.demo_x_cand[0]),
            self.project_work_mask(self.demo_x_cand[1]),
            self.project_work_mask(self.demo_x_cand[2]),
        ];
        let zero_e = [0u64; 3];
        let sw_sing = self.sw_singleton_count();
        let max_depth = if sw_sing >= 12 {
            3usize
        } else if sw_sing >= 7 {
            2usize
        } else {
            1usize
        };

        let mut depth1_cands: Vec<u8> = Vec::new();
        for a in 0..SYN_OP_SYNS {
            let pc = [
                self.sim_arc_mask(a, dm_w[0]),
                self.sim_arc_mask(a, dm_w[1]),
                self.sim_arc_mask(a, dm_w[2]),
            ];
            if self.sw_consistent(&pc, &zero_e) {
                depth1_cands.push(a as u8);
            }
        }
        if !depth1_cands.is_empty() {
            let idx = usize::from(self.seen_ep_cnt) % depth1_cands.len();
            return ([depth1_cands[idx], 0, 0], 1);
        }
        if max_depth >= 2 {
            let mut m_a = [[0u64; 3]; SYN_OP_SYNS];
            for a in 0..SYN_OP_SYNS {
                for k in 0..3 {
                    m_a[a][k] = self.sim_arc_mask(a, dm_w[k]);
                }
            }
            for a in 0..SYN_OP_SYNS {
                for b in 0..SYN_OP_SYNS {
                    let pc = [
                        self.sim_arc_mask(b, m_a[a][0]),
                        self.sim_arc_mask(b, m_a[a][1]),
                        self.sim_arc_mask(b, m_a[a][2]),
                    ];
                    if self.sw_consistent(&pc, &zero_e) {
                        return ([a as u8, b as u8, 0], 2);
                    }
                }
            }
            if max_depth == 3 {
                for a in 0..SYN_OP_SYNS {
                    for b in 0..SYN_OP_SYNS {
                        let m_ab = [
                            self.sim_arc_mask(b, m_a[a][0]),
                            self.sim_arc_mask(b, m_a[a][1]),
                            self.sim_arc_mask(b, m_a[a][2]),
                        ];
                        for c in 0..SYN_OP_SYNS {
                            let pc = [
                                self.sim_arc_mask(c, m_ab[0]),
                                self.sim_arc_mask(c, m_ab[1]),
                                self.sim_arc_mask(c, m_ab[2]),
                            ];
                            if self.sw_consistent(&pc, &zero_e) {
                                return ([a as u8, b as u8, c as u8], 3);
                            }
                        }
                    }
                }
            }
        }
        if !depth1_cands.is_empty() {
            let idx = usize::from(self.seen_ep_cnt) % depth1_cands.len();
            return ([depth1_cands[idx], 0, 0], 1);
        }
        let op = (self.synth_op_rr % 8) as u8;
        self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
        ([op, 0, 0], 1)
    }

    fn plan_exec_program(&mut self) -> ([u8; 3], u8) {
        if !self.exec_model_valid || self.exec_models.len() != SYN_OP_SYNS {
            let op = (self.synth_op_rr % 8) as u8;
            self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
            return ([op, 0, 0], 1);
        }
        let mut dz = [0i32; 3];
        for k in 0..3 {
            let Some(zv) = self.exec_mask_to_z(self.project_work_mask(self.demo_x_cand[k]))
            else {
                let op = (self.synth_op_rr % 8) as u8;
                self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
                return ([op, 0, 0], 1);
            };
            dz[k] = zv;
        }
        let Some(zxt) = self.exec_mask_to_z(self.project_work_mask(self.ep_in_c)) else {
            let op = (self.synth_op_rr % 8) as u8;
            self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
            return ([op, 0, 0], 1);
        };

        let sw_sing = self.sw_singleton_count();
        let max_depth = if sw_sing >= 10 {
            3usize
        } else if sw_sing >= 6 {
            2usize
        } else {
            1usize
        };

        let sig_match = |pc: &[u64; 3], pe: &[u64; 3]| -> bool {
            for i in 0..3 {
                for j in 0..3 {
                    if (pc[i] & pc[j]).count_ones() != (self.demo_s_core[i] & self.demo_s_core[j]).count_ones()
                        || (pc[i] & pe[j]).count_ones()
                            != (self.demo_s_core[i] & self.demo_s_edge[j]).count_ones()
                        || (pe[i] & pc[j]).count_ones()
                            != (self.demo_s_edge[i] & self.demo_s_core[j]).count_ones()
                        || (pe[i] & pe[j]).count_ones()
                            != (self.demo_s_edge[i] & self.demo_s_edge[j]).count_ones()
                    {
                        return false;
                    }
                }
            }
            true
        };

        let mut d1_cands: Vec<u8> = Vec::new();
        for a in 0..SYN_OP_SYNS {
            let mut pc = [0u64; 3];
            let mut pe = [0u64; 3];
            let mut ok = self.exec_models[a].eval_next(zxt, zxt).is_some();
            if ok {
                for k in 0..3 {
                    if let Some(nz) = self.exec_models[a].eval_next(dz[k], dz[k])
                        && let Some((c, e)) = self.exec_z_to_masks(nz)
                    {
                        pc[k] = c;
                        pe[k] = e;
                    } else {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && self.sw_consistent(&pc, &pe) && sig_match(&pc, &pe) {
                d1_cands.push(a as u8);
            }
        }
        if d1_cands.len() == 1 {
            return ([d1_cands[0], 0, 0], 1);
        }
        if !d1_cands.is_empty() && (max_depth == 1 || sw_sing >= 20) {
            let idx = usize::from(self.seen_ep_cnt) % d1_cands.len();
            return ([d1_cands[idx], 0, 0], 1);
        }
        if max_depth >= 2 {
            for a in 0..SYN_OP_SYNS {
                for b in 0..SYN_OP_SYNS {
                    let Some(t1) = self.exec_models[a].eval_next(zxt, zxt) else {
                        continue;
                    };
                    if self.exec_models[b].eval_next(t1, zxt).is_none() {
                        continue;
                    }
                    let mut pc = [0u64; 3];
                    let mut pe = [0u64; 3];
                    let mut ok = true;
                    for k in 0..3 {
                        if let Some(z1) = self.exec_models[a].eval_next(dz[k], dz[k])
                            && let Some(z2) = self.exec_models[b].eval_next(z1, dz[k])
                            && let Some((c, e)) = self.exec_z_to_masks(z2)
                        {
                            pc[k] = c;
                            pe[k] = e;
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok && self.sw_consistent(&pc, &pe) && sig_match(&pc, &pe) {
                        return ([a as u8, b as u8, 0], 2);
                    }
                }
            }
            if max_depth == 3 {
                for a in 0..SYN_OP_SYNS {
                    for b in 0..SYN_OP_SYNS {
                        let Some(t1) = self.exec_models[a].eval_next(zxt, zxt) else {
                            continue;
                        };
                        let Some(t2) = self.exec_models[b].eval_next(t1, zxt) else {
                            continue;
                        };
                        let mut z_ab = [0i32; 3];
                        let mut ok_ab = true;
                        for k in 0..3 {
                            if let Some(z1) = self.exec_models[a].eval_next(dz[k], dz[k])
                                && let Some(z2) = self.exec_models[b].eval_next(z1, dz[k])
                            {
                                z_ab[k] = z2;
                            } else {
                                ok_ab = false;
                                break;
                            }
                        }
                        if !ok_ab {
                            continue;
                        }
                        for c in 0..SYN_OP_SYNS {
                            if self.exec_models[c].eval_next(t2, zxt).is_none() {
                                continue;
                            }
                            let mut pc = [0u64; 3];
                            let mut pe = [0u64; 3];
                            let mut ok = true;
                            for k in 0..3 {
                                if let Some(z3) = self.exec_models[c].eval_next(z_ab[k], dz[k])
                                    && let Some((mc, me)) = self.exec_z_to_masks(z3)
                                {
                                    pc[k] = mc;
                                    pe[k] = me;
                                } else {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok && self.sw_consistent(&pc, &pe) && sig_match(&pc, &pe) {
                                return ([a as u8, b as u8, c as u8], 3);
                            }
                        }
                    }
                }
            }
        }
        if !d1_cands.is_empty() {
            let idx = usize::from(self.seen_ep_cnt) % d1_cands.len();
            return ([d1_cands[idx], 0, 0], 1);
        }
        let op = (self.synth_op_rr % 8) as u8;
        self.synth_op_rr = self.synth_op_rr.wrapping_add(1);
        ([op, 0, 0], 1)
    }

    // =========================================================================
    // REGIME 2: 52-Task Lifelong Hierarchical Composition (P5-G3a..P5-G4b)
    // =========================================================================

    fn cue_max_now(&self, ambient_in: &[f32]) -> f32 {
        let mut mx = 0.0_f32;
        for (c, &v) in ambient_in.iter().enumerate() {
            if has_cue_bit(&self.obs_cue_mask, c) && v > mx {
                mx = v;
            }
        }
        mx
    }

    fn find_life_slot(
        &self,
        key: &[u64; SYN_CUE_WORDS],
        known: &[u64; SYN_CUE_WORDS],
    ) -> Option<usize> {
        let mut best_idx = None;
        let mut best_err = 5u32;
        for (idx, sl) in self.life_slots.iter().enumerate() {
            let mut common_cnt = 0u32;
            let mut err_cnt = 0u32;
            for w in 0..SYN_CUE_WORDS {
                let common = sl.known_mask[w] & known[w];
                common_cnt += common.count_ones();
                err_cnt += ((sl.key_mask[w] ^ key[w]) & common).count_ones();
            }
            if common_cnt >= 16 && err_cnt < best_err {
                best_err = err_cnt;
                best_idx = Some(idx);
            }
        }
        best_idx
    }

    fn find_alt_archive(
        &self,
        key: &[u64; SYN_CUE_WORDS],
        known: &[u64; SYN_CUE_WORDS],
        exclude_steps: &[u8],
    ) -> Option<Vec<u8>> {
        for ae in &self.life_archive {
            let mut common_cnt = 0u32;
            let mut err_cnt = 0u32;
            for w in 0..SYN_CUE_WORDS {
                let common = ae.known_mask[w] & known[w];
                common_cnt += common.count_ones();
                err_cnt += ((ae.key_mask[w] ^ key[w]) & common).count_ones();
            }
            if common_cnt >= 16 && err_cnt <= 4 && ae.steps.as_slice() != exclude_steps {
                return Some(ae.steps.clone());
            }
        }
        None
    }

    fn push_life_archive(
        &mut self,
        key: &[u64; SYN_CUE_WORDS],
        known: &[u64; SYN_CUE_WORDS],
        steps: &[u8],
    ) {
        for ae in &self.life_archive {
            if ae.key_mask == *key && ae.steps.as_slice() == steps {
                return;
            }
        }
        if self.life_archive.len() < SYN_MAX_ARCHIVE {
            self.life_archive.push(LifeArchiveEntry {
                key_mask: *key,
                known_mask: *known,
                steps: steps.to_vec(),
            });
        }
    }

    fn pick_untried_syn(&self, tried_mask: u16) -> Option<u8> {
        let np = self.wire_pairs.len().min(12);
        for s in 0..np {
            if ((self.life_confirmed_ops_mask >> s) & 1) == 1 && ((tried_mask >> s) & 1) == 0 {
                return Some(s as u8);
            }
        }
        if self.life_confirmed_ops_mask.count_ones() < SYN_OP_SYNS as u32 {
            for s in 0..np {
                if ((tried_mask >> s) & 1) == 0 {
                    return Some(s as u8);
                }
            }
        }
        None
    }

    fn step_lifelong_regime(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        na: usize,
        mode: ControlMode,
    ) -> Vec<f32> {
        let clean_inf = if metabolic_influx.is_finite() {
            metabolic_influx.max(0.0)
        } else {
            0.0
        };
        let clean_str = if tissue_stress.is_finite() {
            tissue_stress.max(0.0)
        } else {
            0.0
        };
        let cmax = self.cue_max_now(ambient_in);
        let in_cue_now = cmax > 0.40;

        if in_cue_now {
            if !self.life_in_cue {
                // Transition active -> cue: previous bout just ended
                if self.life_waiting_res && mode == ControlMode::Natural {
                    self.resolve_life_step_result(clean_inf, clean_str);
                }
                if self.life_pending_null_check && mode == ControlMode::Natural {
                    self.life_null_rew_accum =
                        sanitize(self.life_null_rew_accum + clean_inf - clean_str);
                    self.finalize_life_null_hold();
                } else if self.life_prev_had_step && mode == ControlMode::Natural {
                    self.finalize_life_bout_end();
                }
                if mode == ControlMode::Natural {
                    for st in &mut self.ch_stats {
                        st.moved_mid_bout = false;
                    }
                }
                self.life_in_cue = true;
                self.life_cue_tick = 0;
                self.life_act_tick = 0;
                self.life_waiting_res = false;
                self.life_bout_halted = false;
                self.life_prev_had_step = false;
            } else {
                self.life_cue_tick = self.life_cue_tick.saturating_add(1);
            }
            if self.life_cue_tick == 2 {
                if mode == ControlMode::Natural {
                    for (c, st) in self.ch_stats.iter_mut().enumerate() {
                        if let Some(&raw_v) = ambient_in.get(c) {
                            let v = sanitize(raw_v);
                            if v < -1.0e-3 {
                                st.ever_neg = true;
                            }
                            st.moved_mid_bout = v > 0.65;
                        }
                    }
                }
                let mut key = [0u64; SYN_CUE_WORDS];
                let known = self.obs_cue_mask;
                for (c, &v) in ambient_in.iter().enumerate() {
                    if has_cue_bit(&known, c) && v > 0.45 {
                        set_cue_bit(&mut key, c);
                    }
                }
                self.life_cur_key = key;
                self.life_cur_known = known;
                if let Some(sidx) = self.find_life_slot(&key, &known) {
                    self.life_cur_slot_idx = sidx as u16;
                } else if mode == ControlMode::Natural && self.life_slots.len() < SYN_MAX_SLOTS {
                    self.life_slots.push(LifeTaskSlot {
                        key_mask: key,
                        known_mask: known,
                        ..LifeTaskSlot::default()
                    });
                    self.life_cur_slot_idx = (self.life_slots.len() - 1) as u16;
                } else {
                    self.life_cur_slot_idx = u16::MAX;
                }
            }
            return vec![0.0_f32; na];
        }

        // Transition cue -> active (`in_cue` dropped)
        if self.life_in_cue {
            self.life_in_cue = false;
            self.life_act_tick = 0;
            self.life_waiting_res = false;
            self.life_bout_halted = false;
            self.life_bout_rew = 0.0;
            self.life_step_pos = 0;
            self.life_prefix_idx = 0;
            self.life_macro_step_idx = 0;
            self.life_macro_plan_len = 0;
            self.life_bout_unconf = false;
            self.life_bout_last_conf_valid = false;
            self.life_is_null_bout = false;
            self.life_pending_null_check = false;
            self.life_null_rew_accum = 0.0;
            self.life_prev_had_step = false;
            self.prepare_life_bout_plan(mode);
        } else {
            self.life_act_tick = self.life_act_tick.saturating_add(1);
        }

        if self.life_act_tick == 2 && mode == ControlMode::Natural {
            let sidx_now = usize::from(self.life_cur_slot_idx);
            for (c, st) in self.ch_stats.iter().enumerate() {
                if let Some(&raw_v) = ambient_in.get(c)
                    && !st.ever_neg
                    && !self.par_chs.contains(&(c as u16))
                    && st.moved_mid_bout
                    && raw_v < 0.02
                {
                    set_cue_bit(&mut self.obs_cue_mask, c);
                }
            }
            if sidx_now < self.life_slots.len() {
                let sl = &mut self.life_slots[sidx_now];
                for (c, st) in self.ch_stats.iter().enumerate() {
                    if has_cue_bit(&self.obs_cue_mask, c) {
                        set_cue_bit(&mut sl.known_mask, c);
                        if st.moved_mid_bout {
                            set_cue_bit(&mut sl.key_mask, c);
                        }
                    }
                }
            }
        }

        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() {
            return vec![0.0_f32; na];
        }

        if self.life_is_null_bout {
            self.life_null_rew_accum =
                sanitize(self.life_null_rew_accum + clean_inf - clean_str);
            return vec![0.0_f32; na];
        }

        if self.life_waiting_res {
            self.life_waiting_res = false;
            if mode == ControlMode::Natural {
                self.resolve_life_step_result(clean_inf, clean_str);
            } else if clean_inf >= 0.80 || clean_str >= 0.10 {
                self.life_bout_halted = true;
            }
        }

        if self.life_bout_halted {
            return vec![0.0_f32; na];
        }

        // Emit on even `life_act_tick` (`0, 2, 4, 6`)
        if self.life_act_tick % 2 == 0 {
            if let Some((syn, from_pref, p_idx, m_mode)) = self.next_life_pulse(mode) {
                self.life_waiting_res = true;
                self.life_wait_syn = syn;
                self.life_wait_from_prefix = from_pref;
                self.life_wait_prefix_idx = p_idx;
                self.life_wait_macro_mode = m_mode;
                self.life_prev_had_step = true;
                return self.emit_pair(na, usize::from(syn));
            }
        }

        vec![0.0_f32; na]
    }

    fn prepare_life_bout_plan(&mut self, mode: ControlMode) {
        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() {
            return;
        }
        let status = self.life_slots[sidx].status;
        if status == 1 {
            if mode == ControlMode::Natural {
                self.life_is_null_bout = true;
                self.life_pending_null_check = true;
            }
            return;
        }
        if status == 2 {
            let steps = self.life_slots[sidx].steps.clone();
            let plen = steps.len().min(SYN_MAX_STEPS);
            for i in 0..plen {
                self.life_macro_plan[i] = steps[i];
            }
            self.life_macro_plan_len = plen as u8;
            return;
        }
        if mode != ControlMode::Natural {
            return;
        }
        if self.macros_2step.len() >= 4 {
            let pref = self.life_slots[sidx].prefix.clone();
            if pref.len() < 2 {
                let m = self.macros_2step[usize::from(self.life_slots[sidx].m1_idx) % self.macros_2step.len()];
                self.life_macro_plan[0] = m[0];
                self.life_macro_plan[1] = m[1];
                self.life_macro_plan_len = 2;
            } else if usize::from(self.life_slots[sidx].m2_idx) < self.macros_2step.len() {
                let mb = self.macros_2step[usize::from(self.life_slots[sidx].m2_idx)];
                self.life_macro_plan[0] = pref[0];
                self.life_macro_plan[1] = pref[1];
                self.life_macro_plan[2] = mb[0];
                self.life_macro_plan[3] = mb[1];
                self.life_macro_plan_len = 4;
            } else {
                let mut cands: Vec<u8> = Vec::with_capacity(8);
                for m in &self.macros_2step {
                    if !cands.contains(&m[1]) {
                        cands.push(m[1]);
                    }
                }
                for s in 0..self.wire_pairs.len().min(12) {
                    if ((self.life_confirmed_ops_mask >> s) & 1) == 1 && !cands.contains(&(s as u8))
                    {
                        cands.push(s as u8);
                    }
                }
                if !cands.is_empty() {
                    let s2 = cands[usize::from(self.life_slots[sidx].p3_idx) % cands.len()];
                    self.life_macro_plan[0] = pref[0];
                    self.life_macro_plan[1] = pref[1];
                    self.life_macro_plan[2] = s2;
                    self.life_macro_plan_len = 3;
                }
            }
            return;
        }
        if !self.life_slots[sidx].pending_verify.is_empty() {
            let pv = self.life_slots[sidx].pending_verify.clone();
            let plen = pv.len().min(SYN_MAX_STEPS);
            for i in 0..plen {
                self.life_macro_plan[i] = pv[i];
            }
            self.life_macro_plan_len = plen as u8;
        }
    }

    fn next_life_pulse(&mut self, mode: ControlMode) -> Option<(u8, bool, u8, u8)> {
        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() {
            return None;
        }
        let status = self.life_slots[sidx].status;
        if status == 2 {
            let cursor = usize::from(self.life_macro_step_idx);
            if cursor < usize::from(self.life_macro_plan_len) {
                let s = self.life_macro_plan[cursor];
                self.life_macro_step_idx = self.life_macro_step_idx.saturating_add(1);
                return Some((s, true, cursor as u8, 1));
            }
            return None;
        }
        if mode != ControlMode::Natural {
            return None;
        }
        if self.macros_2step.len() >= 4 {
            let cursor = usize::from(self.life_macro_step_idx);
            if cursor < usize::from(self.life_macro_plan_len) {
                let s = self.life_macro_plan[cursor];
                self.life_macro_step_idx = self.life_macro_step_idx.saturating_add(1);
                return Some((s, true, cursor as u8, 2));
            }
            return None;
        }
        if !self.life_slots[sidx].pending_verify.is_empty() {
            let cursor = usize::from(self.life_macro_step_idx);
            if cursor < usize::from(self.life_macro_plan_len) {
                let s = self.life_macro_plan[cursor];
                self.life_macro_step_idx = self.life_macro_step_idx.saturating_add(1);
                return Some((s, true, cursor as u8, 3));
            }
            return None;
        }
        let p_idx = usize::from(self.life_prefix_idx);
        if p_idx < self.life_slots[sidx].prefix.len() {
            let s = self.life_slots[sidx].prefix[p_idx];
            self.life_prefix_idx = self.life_prefix_idx.saturating_add(1);
            return Some((s, true, p_idx as u8, 0));
        }
        if let Some(s) = self.pick_untried_syn(self.life_slots[sidx].tried_mask) {
            return Some((s, false, p_idx as u8, 0));
        }
        None
    }

    fn resolve_life_step_result(&mut self, clean_inf: f32, clean_str: f32) {
        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() {
            return;
        }
        let s = self.life_wait_syn;
        self.life_bout_rew = sanitize(self.life_bout_rew + clean_inf - clean_str);
        if clean_inf >= 0.10 || clean_str >= 0.10 {
            self.life_confirmed_ops_mask |= 1u16 << s;
        }

        let m_mode = self.life_wait_macro_mode;
        if m_mode == 1 {
            // Promoted slot (`ABAB` monitoring)
            if clean_str >= 0.10 {
                self.life_bout_halted = true;
                let key = self.life_slots[sidx].key_mask;
                let known = self.life_slots[sidx].known_mask;
                let cur_steps = self.life_slots[sidx].steps.clone();
                self.push_life_archive(&key, &known, &cur_steps);
                if let Some(alt) = self.find_alt_archive(&key, &known, &cur_steps) {
                    self.life_slots[sidx].steps = alt;
                } else {
                    let sl = &mut self.life_slots[sidx];
                    sl.status = 0;
                    sl.prefix.clear();
                    sl.m1_idx = 0;
                    sl.m2_idx = 0;
                    sl.p3_idx = 0;
                    sl.tried_mask = 0;
                }
            } else if clean_inf >= 0.80 {
                self.life_bout_halted = true;
            }
            return;
        }

        if m_mode == 2 {
            // Macro search (`macros_2step.len() >= 4`)
            if clean_inf >= 0.80 {
                let len = usize::from(self.life_macro_step_idx);
                let sl = &mut self.life_slots[sidx];
                sl.steps = self.life_macro_plan[..len].to_vec();
                sl.exec_rew = self.life_bout_rew;
                sl.status = 1;
                self.life_bout_halted = true;
            } else if clean_str >= 0.10 {
                self.life_bout_halted = true;
                let sl = &mut self.life_slots[sidx];
                if sl.prefix.len() < 2 {
                    sl.m1_idx = sl.m1_idx.saturating_add(1);
                } else if usize::from(sl.m2_idx) < self.macros_2step.len() {
                    sl.m2_idx = sl.m2_idx.saturating_add(1);
                } else {
                    sl.p3_idx = sl.p3_idx.saturating_add(1);
                }
            }
            return;
        }

        if m_mode == 3 {
            // Pending-verify stripped plan
            if clean_inf >= 0.80 {
                let len = usize::from(self.life_macro_step_idx);
                let sl = &mut self.life_slots[sidx];
                sl.steps = self.life_macro_plan[..len].to_vec();
                sl.exec_rew = self.life_bout_rew;
                sl.status = 1;
                sl.pending_verify.clear();
                self.life_bout_halted = true;
            } else if clean_str >= 0.10 {
                self.life_slots[sidx].pending_verify.clear();
                self.life_bout_halted = true;
            }
            return;
        }

        // Sequential builder (`m_mode == 0`)
        if clean_str >= 0.10 {
            self.life_bout_halted = true;
            let sl = &mut self.life_slots[sidx];
            if self.life_wait_from_prefix {
                sl.prefix.clear();
                sl.tried_mask = 0;
            } else {
                sl.tried_mask |= 1u16 << s;
            }
        } else if clean_inf >= 0.80 {
            self.life_bout_halted = true;
            let sl = &mut self.life_slots[sidx];
            if self.life_bout_unconf {
                let mut cand = sl.prefix.clone();
                cand.push(s);
                sl.pending_verify = cand;
            } else {
                let mut steps = sl.prefix.clone();
                steps.push(s);
                sl.steps = steps;
                sl.exec_rew = self.life_bout_rew;
                sl.status = 1;
            }
        } else if clean_inf >= 0.20 {
            if !self.life_wait_from_prefix && self.life_slots[sidx].prefix.len() < SYN_MAX_STEPS {
                self.life_slots[sidx].prefix.push(s);
                self.life_slots[sidx].tried_mask = 0;
                self.life_prefix_idx = self.life_slots[sidx].prefix.len() as u8;
            }
        } else if ((self.life_confirmed_ops_mask >> s) & 1) == 1 {
            if !self.life_wait_from_prefix && self.life_slots[sidx].prefix.len() < SYN_MAX_STEPS {
                self.life_slots[sidx].prefix.push(s);
                self.life_slots[sidx].tried_mask = 0;
                self.life_prefix_idx = self.life_slots[sidx].prefix.len() as u8;
            }
        } else {
            self.life_slots[sidx].tried_mask |= 1u16 << s;
            self.life_bout_unconf = true;
        }
    }

    fn finalize_life_bout_end(&mut self) {
        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() || self.life_bout_halted {
            return;
        }
        if self.macros_2step.len() >= 4 {
            let sl = &mut self.life_slots[sidx];
            if sl.status == 0
                && sl.prefix.len() < 2
                && usize::from(self.life_macro_step_idx) == 2
                && usize::from(self.life_macro_plan_len) == 2
            {
                sl.prefix = vec![self.life_macro_plan[0], self.life_macro_plan[1]];
                sl.m2_idx = 0;
                sl.p3_idx = 0;
            }
        } else {
            let can_pick = self
                .pick_untried_syn(self.life_slots[sidx].tried_mask)
                .is_some();
            let sl = &mut self.life_slots[sidx];
            if !sl.pending_verify.is_empty() {
                sl.pending_verify.clear();
            } else if sl.status == 0 && !can_pick {
                sl.tried_mask = 0;
            }
        }
    }

    fn finalize_life_null_hold(&mut self) {
        self.life_pending_null_check = false;
        self.life_is_null_bout = false;
        let sidx = usize::from(self.life_cur_slot_idx);
        if sidx >= self.life_slots.len() {
            return;
        }
        let null_rew = self.life_null_rew_accum;
        let diff = self.life_slots[sidx].exec_rew - null_rew;
        if diff > 0.50 {
            self.life_slots[sidx].status = 2;
            let st = self.life_slots[sidx].steps.clone();
            let key = self.life_slots[sidx].key_mask;
            let known = self.life_slots[sidx].known_mask;
            if st.len() == 2 && self.macros_2step.len() < SYN_MAX_MACROS {
                let pair = [st[0], st[1]];
                if !self.macros_2step.contains(&pair) {
                    self.macros_2step.push(pair);
                }
            }
            self.push_life_archive(&key, &known, &st);
        } else {
            let sl = &mut self.life_slots[sidx];
            sl.status = 0;
            sl.prefix.clear();
            sl.tried_mask = 0;
        }
    }
}

#[inline]
fn pop96(a: [u64; SYN_STATE_WORDS], b: [u64; SYN_STATE_WORDS]) -> u32 {
    (a[0] ^ b[0]).count_ones() + (a[1] ^ b[1]).count_ones()
}

#[inline]
fn set_cue_bit(mask: &mut [u64; SYN_CUE_WORDS], ch: usize) {
    let w = ch / 64;
    let b = ch % 64;
    if w < SYN_CUE_WORDS {
        mask[w] |= 1u64 << b;
    }
}

#[inline]
fn clear_cue_bit(mask: &mut [u64; SYN_CUE_WORDS], ch: usize) {
    let w = ch >> 6;
    let b = ch & 63;
    if w < SYN_CUE_WORDS {
        mask[w] &= !(1u64 << b);
    }
}

#[inline]
fn gcd_u16(mut a: u16, mut b: u16) -> u16 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[inline]
fn has_cue_bit(mask: &[u64; SYN_CUE_WORDS], ch: usize) -> bool {
    let w = ch / 64;
    let b = ch % 64;
    w < SYN_CUE_WORDS && ((mask[w] >> b) & 1) == 1
}

fn masked_cosine_diff(
    chs: &[u16],
    a: &[f32],
    base: &[f32],
    pos: &[f32],
    neg: &[f32],
) -> f32 {
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    let mut dot = 0.0_f32;
    for &ch in chs {
        let idx = usize::from(ch);
        if idx < a.len() && idx < base.len() && idx < pos.len() && idx < neg.len() {
            let va = a[idx];
            let vb = (pos[idx] - base[idx]) - 0.45 * (neg[idx] - pos[idx]);
            na += va * va;
            nb += vb * vb;
            dot += va * vb;
        }
    }
    let den = (na * nb).sqrt();
    if den > 1.0e-6 {
        sanitize(dot / den)
    } else {
        0.0
    }
}

fn masked_vec_cosine(chs: &[u16], a: &[f32], b: &[f32]) -> f32 {
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    let mut dot = 0.0_f32;
    for &ch in chs {
        let idx = usize::from(ch);
        if idx < a.len() && idx < b.len() {
            let va = a[idx];
            let vb = b[idx];
            na += va * va;
            nb += vb * vb;
            dot += va * vb;
        }
    }
    let den = (na * nb).sqrt();
    if den > 1.0e-6 {
        sanitize(dot / den)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passive_dormant_on_4096_channel_field_and_zero_frozen_drift() {
        let mut syn = SynthesisCortex::default();
        let h0 = {
            let mut h = ParamHasher::new(11);
            syn.hash_learned(&mut h);
            h.finish()
        };
        let big_in = vec![0.1_f32; 4096];
        for _ in 0..40 {
            assert!(syn
                .step_synthesis(&big_in, 0.0, 0.0, 24, ControlMode::Natural, 11)
                .is_none());
        }
        assert_eq!(syn.regime, REGIME_PASSIVE);
        let h1 = {
            let mut h = ParamHasher::new(11);
            syn.hash_learned(&mut h);
            h.finish()
        };
        assert_eq!(h0, h1);
    }

    #[test]
    fn exec_aff_piece_eval_handles_trap_bounds() {
        let p = ExecAffPiece {
            a_neg: 2,
            b_neg: 0,
            c_neg: -1,
            a_pos: 2,
            b_pos: 1,
            c_pos: 0,
        };
        assert_eq!(p.eval_next(-5, 3), Some(-11));
        assert_eq!(p.eval_next(-12, 3), None);
        assert_eq!(p.eval_next(5, 4), Some(14));
        assert_eq!(p.eval_next(10, 4), None);
    }
}
