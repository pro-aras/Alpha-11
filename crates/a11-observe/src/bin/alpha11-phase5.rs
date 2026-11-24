//! Pre-registered locked Phase-5 Zero-Leak Machine-Scale Benchmarks judge (`P5-G1..G6` + `P5-I1..I4`).
//!
//! Three zero-leak, WirePerm24-scrambled external `FieldView -> FieldPulse` portals:
//! 1. [`ArcGridMedium`]: Sealed `4x4` `4`-color few-shot ARC-AGI transformation portal (`12` Seen
//!    rules + `8` Held-Out Unseen Depth-2/Depth-3 compositions evaluated under `FrozenPlasticity`).
//! 2. [`SandboxedExecMedium`]: Live trap-safe Register VM program synthesis & execution portal
//!    (`8` arithmetic/saturating ops, `12` Seen specs + `8` Held-Out Unseen Depth-2/Depth-3 specs
//!    under `FrozenPlasticity`).
//! 3. [`Lifelong52Medium`]: `52`-task sequential forward-transfer & retention suite (`T_0..T_51`)
//!    with Depth-4 hierarchical macro composition (`T_36..T_51`), occluded step feedback, `< 2%`
//!    catastrophic forgetting cap (`T_0..T_15`), `ABAB` rule-flip savings, and `OPER` lesion proof.
//!
//! Usage: `alpha11-phase5 [dev|dev30|heldout] [--threads N] [--open-vault] [--live-chk <path>]`

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use a11_core::crc32c::{crc32c, crc32c_update};
use a11_core::persistence::{
    FILE_MAGIC, FORMAT_VERSION, MIN_COMPAT_FORMAT_VERSION, SLAB_COUNT, TAG_BODY, TAG_CELL,
    TAG_PORE, TAG_RING, TAG_SCOT, TAG_TUBE, encode_organism, shadow_decode_organism,
};
use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, PersistError, WorldMedium};
use a11_observe::gates::{PairedGate, holm_adjust, paired_gate};
use a11_observe::{DEFAULT_TWELVE_SEEDS, PulseTemporalChannelShuffler, scan_crate_source};
use a11_worlds::{DenseFieldWorld, DeterministicRng, YokedMedium};

const P5_CHANNELS: usize = 256;
const P5_ACTUATORS: usize = 24;
const DISTRACTOR_START: usize = 192;
const OU_CHANNELS: usize = 32;
const SELF_COPY_CHANNELS: usize = 32;

const NUM_GROUPS: usize = 12;
const OP_GROUPS: usize = 8;

const SANDBOX_BOUTS: usize = 32;
const SANDBOX_LEN: u64 = 14;
const SEEN_EPISODES: usize = 96;
const SEEN_EVAL_TAIL: usize = 24;
const UNSEEN_EPISODES: usize = 24;
const EPISODE_LEN: u64 = 24;
const DEMO_TICKS: u64 = 12;
const MAX_EPISODE_OPS: usize = 3;
const MAX_SANDBOX_OPS: usize = 4;

const OPER_SLAB_TAG: u32 = u32::from_le_bytes(*b"OPER");

const P5_SENSOR_SALT: u64 = 0x5A11_5E45_0000_2026;
const P5_MOTOR_SALT: u64 = 0x5A11_4D07_0000_2026;
const P5_SHUF_SALT: u64 = 0x5A11_5A17_9999_8888;
const P5_ARC_SALT: u64 = 0x5A11_A4C0_1111_0001;
const P5_EXEC_SALT: u64 = 0x5A11_E8EC_2222_0002;
const P5_LIFE_SALT: u64 = 0x5A11_11FE_3333_0003;

const ARC_SEEN_RULES: [&[usize]; 12] = [
    &[0],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[6],
    &[7],
    &[0, 3],
    &[1, 4],
    &[5, 5],
    &[6, 3],
];

const ARC_UNSEEN_RULES: [&[usize]; 8] = [
    &[2, 0],
    &[0, 5],
    &[2, 5],
    &[3, 4],
    &[7, 5],
    &[1, 6],
    &[2, 0, 5],
    &[3, 7, 5],
];

const EXEC_SEEN_RULES: [&[usize]; 12] = [
    &[0],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[6],
    &[7],
    &[0, 0],
    &[1, 1],
    &[4, 2],
    &[6, 0],
];

const EXEC_UNSEEN_RULES: [&[usize]; 8] = [
    &[3, 0],
    &[0, 3],
    &[5, 2],
    &[6, 3],
    &[7, 1],
    &[4, 0],
    &[5, 3, 0],
    &[7, 7, 1],
];

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

#[derive(Clone, Debug)]
struct WirePerm24 {
    sensor_perm: [usize; P5_CHANNELS],
    act_perm: [usize; P5_ACTUATORS],
}

impl WirePerm24 {
    fn from_seed(seed: u64) -> Self {
        let mut s_rng = DeterministicRng::new(seed ^ P5_SENSOR_SALT);
        let mut sensor_perm: [usize; P5_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..P5_CHANNELS).rev() {
            let j = (s_rng.next_u64() as usize) % (i + 1);
            sensor_perm.swap(i, j);
        }
        let mut m_rng = DeterministicRng::new(seed ^ P5_MOTOR_SALT);
        let mut act_perm: [usize; P5_ACTUATORS] = std::array::from_fn(|i| i);
        for i in (1..P5_ACTUATORS).rev() {
            let j = (m_rng.next_u64() as usize) % (i + 1);
            act_perm.swap(i, j);
        }
        Self {
            sensor_perm,
            act_perm,
        }
    }

    fn distractor_conductance_share(&self, channels: &[u32], conductances: &[f32]) -> f64 {
        let mut tot = 0.0_f64;
        let mut dst = 0.0_f64;
        for (&ch, &g) in channels.iter().zip(conductances.iter()) {
            let sc = ch as usize;
            if sc < P5_CHANNELS && g.is_finite() && g > 0.0 {
                let gf = f64::from(g);
                tot += gf;
                if self.sensor_perm[sc] >= DISTRACTOR_START {
                    dst += gf;
                }
            }
        }
        if tot > 1.0e-12 { dst / tot } else { 0.0 }
    }
}

#[derive(Clone, Debug)]
struct SharedSkin {
    perm: WirePerm24,
    vel_grp: [f32; NUM_GROUPS],
    pos_grp: [f32; NUM_GROUPS],
    op_cooldown: u8,
    motor_ring: [[f32; P5_ACTUATORS]; 3],
    ring_cursor: usize,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    global_tick: u64,
    disp_vel: [f32; 144],
    disp_lvl: [f32; 144],
    first_frame: bool,
    raw_ambient: [f32; P5_CHANNELS],
    permuted_ambient: [f32; P5_CHANNELS],
}

impl SharedSkin {
    fn new(perm: WirePerm24, ou_seed: u64) -> Self {
        Self {
            perm,
            vel_grp: [0.0; NUM_GROUPS],
            pos_grp: [0.0; NUM_GROUPS],
            op_cooldown: 0,
            motor_ring: [[0.0; P5_ACTUATORS]; 3],
            ring_cursor: 0,
            ou_rng: DeterministicRng::new(ou_seed),
            ou_state: [0.0; OU_CHANNELS],
            global_tick: 0,
            disp_vel: [0.0; 144],
            disp_lvl: [0.0; 144],
            first_frame: true,
            raw_ambient: [0.0; P5_CHANNELS],
            permuted_ambient: [0.0; P5_CHANNELS],
        }
    }

    fn step_motor(&mut self, pulse: &FieldPulse) -> Option<usize> {
        let mut logical_act = [0.0_f32; P5_ACTUATORS];
        for (log_idx, &phys_idx) in self.perm.act_perm.iter().enumerate() {
            logical_act[log_idx] =
                clamp_unit(pulse.actuator_out.get(phys_idx).copied().unwrap_or(0.0));
        }
        self.motor_ring[self.ring_cursor] = logical_act;
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        for g in 0..NUM_GROUPS {
            let drive_g = 0.5 * (logical_act[2 * g] + logical_act[2 * g + 1]);
            self.vel_grp[g] = (0.35 * self.vel_grp[g] + 0.65 * drive_g).clamp(-1.0, 1.0);
            self.pos_grp[g] = (0.72 * self.pos_grp[g] + 0.28 * self.vel_grp[g]).clamp(-1.0, 1.0);
        }

        if self.op_cooldown > 0 {
            self.op_cooldown -= 1;
        }

        let mut best_g = 0usize;
        let mut best_val = self.vel_grp[0];
        let mut second_val = f32::NEG_INFINITY;
        for g in 1..OP_GROUPS {
            let v = self.vel_grp[g];
            if v > best_val {
                second_val = best_val;
                best_val = v;
                best_g = g;
            } else if v > second_val {
                second_val = v;
            }
        }
        if self.op_cooldown == 0 && best_val > 0.38 && (best_val - second_val) > 0.06 {
            self.op_cooldown = 2;
            Some(best_g)
        } else {
            None
        }
    }

    fn finalize_ambient(&mut self) {
        if self.first_frame {
            self.disp_lvl.copy_from_slice(&self.raw_ambient[0..144]);
            self.first_frame = false;
        } else {
            for c in 0..144 {
                let tgt = self.raw_ambient[c];
                self.disp_vel[c] = 0.42 * self.disp_vel[c] + 0.58 * (tgt - self.disp_lvl[c]);
                self.disp_lvl[c] = (self.disp_lvl[c] + self.disp_vel[c]).clamp(0.0, 1.0);
                self.raw_ambient[c] = self.disp_lvl[c];
            }
        }
        let t_f = self.global_tick as f32;
        for p in 0..16 {
            let ga = p % NUM_GROUPS;
            let gb = (p + 3) % NUM_GROUPS;
            let fa = 1.65 + 0.21 * ((p % 4) as f32);
            let fb = 1.45 + 0.27 * (((p / 4) % 3) as f32);
            let off = 0.37 * (p as f32);
            let vp =
                0.45 + 0.36 * (fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            let vm =
                0.45 + 0.36 * (-fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            self.raw_ambient[144 + 2 * p] = vp.clamp(0.0, 1.0);
            self.raw_ambient[144 + 2 * p + 1] = vm.clamp(0.0, 1.0);
        }
        for ch in 176..DISTRACTOR_START {
            self.raw_ambient[ch] = 0.015 * (0.019 * t_f + 0.11 * (ch as f32)).sin();
        }
        for i in 0..OU_CHANNELS {
            let z = self.ou_rng.next_f32_range(-1.0, 1.0);
            self.ou_state[i] = (0.35 * self.ou_state[i] + 0.65 * z).clamp(-1.0, 1.0);
            self.raw_ambient[DISTRACTOR_START + i] = self.ou_state[i];
        }
        let lag1 = (self.ring_cursor + 2) % 3;
        let lag2 = (self.ring_cursor + 1) % 3;
        for i in 0..SELF_COPY_CHANNELS {
            let a1 = self.motor_ring[lag1][i % P5_ACTUATORS];
            let a2 = self.motor_ring[lag2][(i + 5) % P5_ACTUATORS];
            self.raw_ambient[DISTRACTOR_START + OU_CHANNELS + i] =
                (0.65 * a1 - 0.30 * a2).clamp(-1.0, 1.0);
        }
        for (sc_ch, &tr_ch) in self.perm.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[tr_ch];
        }
        self.global_tick += 1;
    }
}

// =============================================================================
// Pillar 1: Sealed 4x4 4-Color Few-Shot ARC-AGI Transformation Portal
// =============================================================================

type ArcGrid = [u8; 16];

fn apply_arc_op(op: usize, g: ArcGrid) -> ArcGrid {
    let mut out = [0_u8; 16];
    match op {
        0 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[r * 4 + (3 - c)] = g[r * 4 + c];
                }
            }
        }
        1 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[(3 - r) * 4 + c] = g[r * 4 + c];
                }
            }
        }
        2 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[c * 4 + r] = g[r * 4 + c];
                }
            }
        }
        3 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[r * 4 + ((c + 1) & 3)] = g[r * 4 + c];
                }
            }
        }
        4 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[((r + 1) & 3) * 4 + c] = g[r * 4 + c];
                }
            }
        }
        5 => {
            for i in 0..16 {
                out[i] = if g[i] == 0 { 0 } else { (g[i] % 3) + 1 };
            }
        }
        6 => {
            for r in 0..4 {
                for c in 0..4 {
                    out[(r ^ 2) * 4 + (c ^ 2)] = g[r * 4 + c];
                }
            }
        }
        _ => {
            for c in 0..4 {
                let mut write_r = 3isize;
                for r in (0..4).rev() {
                    let v = g[r * 4 + c];
                    if v != 0 {
                        out[(write_r as usize) * 4 + c] = v;
                        write_r -= 1;
                    }
                }
            }
        }
    }
    out
}

fn eval_arc_rule(ops: &[usize], mut g: ArcGrid) -> ArcGrid {
    for &op in ops {
        g = apply_arc_op(op, g);
    }
    g
}

fn encode_arc_grid(g: &ArcGrid, dst: &mut [f32]) {
    dst.fill(0.05);
    for (cell, &v) in g.iter().enumerate() {
        if (1..=3).contains(&v) {
            dst[cell * 3 + ((v as usize) - 1)] = 0.85;
        }
    }
}

fn random_asymmetric_arc_grid(rng: &mut DeterministicRng, rule: Option<&[usize]>) -> ArcGrid {
    for _ in 0..64 {
        let mut g = [0_u8; 16];
        let mut cells: [usize; 16] = std::array::from_fn(|i| i);
        rng.shuffle(&mut cells);
        let n_fg = 6 + rng.next_usize(3);
        for &c in &cells[..n_fg] {
            g[c] = 1 + (rng.next_usize(3) as u8);
        }
        let distinct_sym = apply_arc_op(0, g) != g
            && apply_arc_op(1, g) != g
            && apply_arc_op(2, g) != g
            && apply_arc_op(6, g) != g;
        let non_id_rule = rule.is_none_or(|r| eval_arc_rule(r, g) != g);
        if distinct_sym && non_id_rule {
            return g;
        }
    }
    let mut g = [0_u8; 16];
    for (i, &c) in [0usize, 1, 5, 6, 9, 15].iter().enumerate() {
        g[c] = 1 + ((i % 3) as u8);
    }
    g
}

#[derive(Clone, Debug)]
struct ArcEpisode {
    demos: [(ArcGrid, ArcGrid); 3],
    x_test: ArcGrid,
    y_test: ArcGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArcStage {
    Sandbox,
    Seen,
    Unseen,
}

#[derive(Clone, Debug)]
struct ArcGridMedium {
    skin: SharedSkin,
    stage: ArcStage,
    sandbox_grids: Vec<ArcGrid>,
    episodes: Vec<ArcEpisode>,
    ep_idx: usize,
    b_tick: u64,
    work_grid: ArcGrid,
    emitted_ops: Vec<usize>,
    ops_used: usize,
    bout_solved: bool,
    last_influx: f32,
    last_stress: f32,
    total_net_val: f64,
    eval_tail_net_val: f64,
    eval_tail_start: usize,
    solved_total: usize,
    solved_tail: usize,
}

impl ArcGridMedium {
    fn new(seed: u64, salt: u64, stage: ArcStage, num_bouts: usize, perm: WirePerm24) -> Self {
        let mut rng = DeterministicRng::new(seed ^ P5_ARC_SALT ^ salt);
        let mut sandbox_grids = Vec::new();
        let mut episodes = Vec::new();
        match stage {
            ArcStage::Sandbox => {
                for _ in 0..num_bouts {
                    sandbox_grids.push(random_asymmetric_arc_grid(&mut rng, None));
                }
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let pool: &[&[usize]] = if stage == ArcStage::Seen {
                    &ARC_SEEN_RULES
                } else {
                    &ARC_UNSEEN_RULES
                };
                let rule_order = if stage == ArcStage::Seen && num_bouts > SEEN_EVAL_TAIL {
                    let train_n = num_bouts - SEEN_EVAL_TAIL;
                    let mut train_ord: Vec<usize> =
                        (0..train_n).map(|i| i % pool.len()).collect();
                    rng.shuffle(&mut train_ord);
                    let mut tail_ord: Vec<usize> =
                        (0..SEEN_EVAL_TAIL).map(|i| 8 + (i % 4)).collect();
                    rng.shuffle(&mut tail_ord);
                    train_ord.extend(tail_ord);
                    train_ord
                } else {
                    let mut ord: Vec<usize> = (0..num_bouts).map(|i| i % pool.len()).collect();
                    rng.shuffle(&mut ord);
                    ord
                };
                for &ri in &rule_order {
                    let rule = pool[ri];
                    let x0 = random_asymmetric_arc_grid(&mut rng, Some(rule));
                    let x1 = random_asymmetric_arc_grid(&mut rng, Some(rule));
                    let x2 = random_asymmetric_arc_grid(&mut rng, Some(rule));
                    let xt = random_asymmetric_arc_grid(&mut rng, Some(rule));
                    episodes.push(ArcEpisode {
                        demos: [
                            (x0, eval_arc_rule(rule, x0)),
                            (x1, eval_arc_rule(rule, x1)),
                            (x2, eval_arc_rule(rule, x2)),
                        ],
                        x_test: xt,
                        y_test: eval_arc_rule(rule, xt),
                    });
                }
            }
        }
        let init_work = match stage {
            ArcStage::Sandbox => sandbox_grids.first().copied().unwrap_or([0; 16]),
            ArcStage::Seen | ArcStage::Unseen => {
                episodes.first().map_or([0; 16], |e| e.demos[0].0)
            }
        };
        let eval_tail_start = if stage == ArcStage::Seen {
            SEEN_EPISODES.saturating_sub(SEEN_EVAL_TAIL)
        } else {
            0
        };
        let mut m = Self {
            skin: SharedSkin::new(perm, seed ^ salt ^ 0x0011_4411),
            stage,
            sandbox_grids,
            episodes,
            ep_idx: 0,
            b_tick: 0,
            work_grid: init_work,
            emitted_ops: Vec::with_capacity(4),
            ops_used: 0,
            bout_solved: false,
            last_influx: 0.0,
            last_stress: 0.0,
            total_net_val: 0.0,
            eval_tail_net_val: 0.0,
            eval_tail_start,
            solved_total: 0,
            solved_tail: 0,
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        match self.stage {
            ArcStage::Sandbox => (self.sandbox_grids.len() as u64) * SANDBOX_LEN,
            ArcStage::Seen | ArcStage::Unseen => (self.episodes.len() as u64) * EPISODE_LEN,
        }
    }

    fn refresh_ambient(&mut self) {
        self.skin.raw_ambient.fill(0.0);
        match self.stage {
            ArcStage::Sandbox => {
                encode_arc_grid(&self.work_grid, &mut self.skin.raw_ambient[0..48]);
                self.skin.raw_ambient[48..96].fill(0.0);
                encode_arc_grid(&self.work_grid, &mut self.skin.raw_ambient[96..144]);
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let ep = &self.episodes[self.ep_idx % self.episodes.len().max(1)];
                if self.b_tick < DEMO_TICKS {
                    let k = ((self.b_tick / 4) as usize).min(2);
                    encode_arc_grid(&ep.demos[k].0, &mut self.skin.raw_ambient[0..48]);
                    encode_arc_grid(&ep.demos[k].1, &mut self.skin.raw_ambient[48..96]);
                    encode_arc_grid(&ep.demos[k].0, &mut self.skin.raw_ambient[96..144]);
                } else {
                    encode_arc_grid(&ep.x_test, &mut self.skin.raw_ambient[0..48]);
                    self.skin.raw_ambient[48..96].fill(0.0);
                    encode_arc_grid(&self.work_grid, &mut self.skin.raw_ambient[96..144]);
                }
            }
        }
        self.skin.finalize_ambient();
    }
}

impl WorldMedium for ArcGridMedium {
    fn num_ambient_channels(&self) -> usize {
        P5_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P5_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.skin.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P5_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let triggered = self.skin.step_motor(pulse);
        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;

        match self.stage {
            ArcStage::Sandbox => {
                if let Some(op) = triggered
                    && self.ops_used < MAX_SANDBOX_OPS
                {
                    self.work_grid = apply_arc_op(op, self.work_grid);
                    self.ops_used += 1;
                }
                self.b_tick += 1;
                if self.b_tick >= SANDBOX_LEN {
                    self.b_tick = 0;
                    self.ep_idx += 1;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                    let idx = self.ep_idx % self.sandbox_grids.len().max(1);
                    self.work_grid = self.sandbox_grids[idx];
                }
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let n_eps = self.episodes.len().max(1);
                let cur_idx = self.ep_idx % n_eps;
                let in_tail = cur_idx >= self.eval_tail_start;
                if self.b_tick >= DEMO_TICKS && !self.bout_solved {
                    if let Some(op) = triggered
                        && self.ops_used < MAX_EPISODE_OPS
                    {
                        self.emitted_ops.push(op);
                        self.work_grid = apply_arc_op(op, self.work_grid);
                        self.ops_used += 1;
                        if self.work_grid == self.episodes[cur_idx].y_test
                            && self
                                .episodes[cur_idx]
                                .demos
                                .iter()
                                .all(|&(xd, yd)| eval_arc_rule(&self.emitted_ops, xd) == yd)
                        {
                            self.bout_solved = true;
                            self.solved_total += 1;
                            if in_tail {
                                self.solved_tail += 1;
                            }
                            influx = 2.40;
                        }
                    }
                    if !self.bout_solved && self.b_tick == EPISODE_LEN - 1 {
                        stress = 0.25;
                    }
                }
                let dv = f64::from(influx - stress);
                self.total_net_val += dv;
                if in_tail {
                    self.eval_tail_net_val += dv;
                }

                self.b_tick += 1;
                if self.b_tick == DEMO_TICKS {
                    self.work_grid = self.episodes[cur_idx].x_test;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                } else if self.b_tick >= EPISODE_LEN {
                    self.b_tick = 0;
                    self.ep_idx += 1;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                    let nxt = self.ep_idx % n_eps;
                    self.work_grid = self.episodes[nxt].demos[0].0;
                } else if self.b_tick < DEMO_TICKS {
                    let k = ((self.b_tick / 4) as usize).min(2);
                    self.work_grid = self.episodes[cur_idx].demos[k].0;
                }
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.refresh_ambient();
    }
}

// =============================================================================
// Pillar 2: Sandboxed Register VM Program Synthesis & Execution Portal
// =============================================================================

#[inline]
fn apply_vm_op(op: usize, r: i32, x: i32, already_trap: bool) -> (i32, bool) {
    if already_trap {
        return (r, true);
    }
    let next = match op {
        0 => r + 1,
        1 => r - 1,
        2 => r + 2,
        3 => r * 2,
        4 => -r,
        5 => r.abs(),
        6 => r.max(0),
        _ => r + x,
    };
    if next.abs() > 23 {
        (0, true)
    } else {
        (next, false)
    }
}

fn eval_vm_prog(ops: &[usize], x: i32) -> (i32, bool) {
    let mut r = x;
    let mut trap = false;
    for &op in ops {
        (r, trap) = apply_vm_op(op, r, x, trap);
    }
    (r, trap)
}

fn encode_vm_state(r: i32, trap: bool, dst: &mut [f32]) {
    dst.fill(0.05);
    if trap {
        for ch in [0usize, 8, 16, 24, 32, 47] {
            dst[ch] = 0.85;
        }
    } else {
        let clamped = r.clamp(-23, 23);
        let idx = (clamped + 23) as usize;
        for off in 0..6 {
            dst[(idx + off) % 47] = 0.85;
        }
        let edge_lo = (idx + 46) % 47;
        let edge_hi = (idx + 6) % 47;
        dst[edge_lo] = dst[edge_lo].max(0.28);
        dst[edge_hi] = dst[edge_hi].max(0.28);
    }
}

#[derive(Clone, Debug)]
struct ExecEpisode {
    demos: [(i32, i32); 3],
    x_test: i32,
    y_test: i32,
}

#[derive(Clone, Debug)]
struct SandboxedExecMedium {
    skin: SharedSkin,
    stage: ArcStage,
    sandbox_inits: Vec<i32>,
    episodes: Vec<ExecEpisode>,
    ep_idx: usize,
    b_tick: u64,
    cur_x: i32,
    reg_val: i32,
    trapped: bool,
    emitted_ops: Vec<usize>,
    ops_used: usize,
    bout_solved: bool,
    last_influx: f32,
    last_stress: f32,
    total_net_val: f64,
    eval_tail_net_val: f64,
    eval_tail_start: usize,
    solved_total: usize,
    solved_tail: usize,
}

impl SandboxedExecMedium {
    fn new(seed: u64, salt: u64, stage: ArcStage, num_bouts: usize, perm: WirePerm24) -> Self {
        const DEMO_TRIPLES: [[i32; 3]; 3] = [[-3, 2, -5], [-4, 3, -1], [-2, 4, -5]];
        const X_TEST_CANDS: [i32; 10] = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];

        let mut rng = DeterministicRng::new(seed ^ P5_EXEC_SALT ^ salt);
        let mut sandbox_inits = Vec::new();
        let mut episodes = Vec::new();

        match stage {
            ArcStage::Sandbox => {
                for _ in 0..num_bouts {
                    let x = X_TEST_CANDS[rng.next_usize(X_TEST_CANDS.len())];
                    sandbox_inits.push(x);
                }
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let pool: &[&[usize]] = if stage == ArcStage::Seen {
                    &EXEC_SEEN_RULES
                } else {
                    &EXEC_UNSEEN_RULES
                };
                let rule_order = if stage == ArcStage::Seen && num_bouts > SEEN_EVAL_TAIL {
                    let train_n = num_bouts - SEEN_EVAL_TAIL;
                    let mut train_ord: Vec<usize> =
                        (0..train_n).map(|i| i % pool.len()).collect();
                    rng.shuffle(&mut train_ord);
                    let mut tail_ord: Vec<usize> =
                        (0..SEEN_EVAL_TAIL).map(|i| 8 + (i % 4)).collect();
                    rng.shuffle(&mut tail_ord);
                    train_ord.extend(tail_ord);
                    train_ord
                } else {
                    let mut ord: Vec<usize> = (0..num_bouts).map(|i| i % pool.len()).collect();
                    rng.shuffle(&mut ord);
                    ord
                };
                for (i, &ri) in rule_order.iter().enumerate() {
                    let prog = pool[ri];
                    let tri = DEMO_TRIPLES[(i + rng.next_usize(3)) % 3];
                    let mut x_pool: Vec<i32> = X_TEST_CANDS
                        .iter()
                        .copied()
                        .filter(|&xt| {
                            !tri.contains(&xt) && {
                                let (yt, tr) = eval_vm_prog(prog, xt);
                                !tr && yt != xt
                            }
                        })
                        .collect();
                    rng.shuffle(&mut x_pool);
                    let xt = x_pool.first().copied().unwrap_or(-2);
                    let (yt, _) = eval_vm_prog(prog, xt);
                    episodes.push(ExecEpisode {
                        demos: [
                            (tri[0], eval_vm_prog(prog, tri[0]).0),
                            (tri[1], eval_vm_prog(prog, tri[1]).0),
                            (tri[2], eval_vm_prog(prog, tri[2]).0),
                        ],
                        x_test: xt,
                        y_test: yt,
                    });
                }
            }
        }
        let init_x = match stage {
            ArcStage::Sandbox => sandbox_inits.first().copied().unwrap_or(-3),
            ArcStage::Seen | ArcStage::Unseen => episodes.first().map_or(-3, |e| e.demos[0].0),
        };
        let eval_tail_start = if stage == ArcStage::Seen {
            SEEN_EPISODES.saturating_sub(SEEN_EVAL_TAIL)
        } else {
            0
        };
        let mut m = Self {
            skin: SharedSkin::new(perm, seed ^ salt ^ 0x0022_5522),
            stage,
            sandbox_inits,
            episodes,
            ep_idx: 0,
            b_tick: 0,
            cur_x: init_x,
            reg_val: init_x,
            trapped: false,
            emitted_ops: Vec::with_capacity(4),
            ops_used: 0,
            bout_solved: false,
            last_influx: 0.0,
            last_stress: 0.0,
            total_net_val: 0.0,
            eval_tail_net_val: 0.0,
            eval_tail_start,
            solved_total: 0,
            solved_tail: 0,
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        match self.stage {
            ArcStage::Sandbox => (self.sandbox_inits.len() as u64) * SANDBOX_LEN,
            ArcStage::Seen | ArcStage::Unseen => (self.episodes.len() as u64) * EPISODE_LEN,
        }
    }

    fn refresh_ambient(&mut self) {
        self.skin.raw_ambient.fill(0.0);
        match self.stage {
            ArcStage::Sandbox => {
                encode_vm_state(self.cur_x, false, &mut self.skin.raw_ambient[0..48]);
                self.skin.raw_ambient[48..96].fill(0.0);
                encode_vm_state(
                    self.reg_val,
                    self.trapped,
                    &mut self.skin.raw_ambient[96..144],
                );
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let ep = &self.episodes[self.ep_idx % self.episodes.len().max(1)];
                if self.b_tick < DEMO_TICKS {
                    let k = ((self.b_tick / 4) as usize).min(2);
                    encode_vm_state(ep.demos[k].0, false, &mut self.skin.raw_ambient[0..48]);
                    encode_vm_state(ep.demos[k].1, false, &mut self.skin.raw_ambient[48..96]);
                    encode_vm_state(ep.demos[k].0, false, &mut self.skin.raw_ambient[96..144]);
                } else {
                    encode_vm_state(ep.x_test, false, &mut self.skin.raw_ambient[0..48]);
                    self.skin.raw_ambient[48..96].fill(0.0);
                    encode_vm_state(
                        self.reg_val,
                        self.trapped,
                        &mut self.skin.raw_ambient[96..144],
                    );
                }
            }
        }
        self.skin.finalize_ambient();
    }
}

impl WorldMedium for SandboxedExecMedium {
    fn num_ambient_channels(&self) -> usize {
        P5_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P5_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.skin.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P5_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let triggered = self.skin.step_motor(pulse);
        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;

        match self.stage {
            ArcStage::Sandbox => {
                if let Some(op) = triggered
                    && self.ops_used < MAX_SANDBOX_OPS
                {
                    (self.reg_val, self.trapped) =
                        apply_vm_op(op, self.reg_val, self.cur_x, self.trapped);
                    self.ops_used += 1;
                }
                self.b_tick += 1;
                if self.b_tick >= SANDBOX_LEN {
                    self.b_tick = 0;
                    self.ep_idx += 1;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                    let idx = self.ep_idx % self.sandbox_inits.len().max(1);
                    self.cur_x = self.sandbox_inits[idx];
                    self.reg_val = self.cur_x;
                    self.trapped = false;
                }
            }
            ArcStage::Seen | ArcStage::Unseen => {
                let n_eps = self.episodes.len().max(1);
                let cur_idx = self.ep_idx % n_eps;
                let in_tail = cur_idx >= self.eval_tail_start;
                if self.b_tick >= DEMO_TICKS && !self.bout_solved {
                    if let Some(op) = triggered
                        && self.ops_used < MAX_EPISODE_OPS
                    {
                        self.emitted_ops.push(op);
                        (self.reg_val, self.trapped) =
                            apply_vm_op(op, self.reg_val, self.cur_x, self.trapped);
                        self.ops_used += 1;
                        if self.trapped {
                            stress = 0.25;
                            self.bout_solved = true;
                        } else if self.reg_val == self.episodes[cur_idx].y_test
                            && self
                                .episodes[cur_idx]
                                .demos
                                .iter()
                                .all(|&(xd, yd)| eval_vm_prog(&self.emitted_ops, xd) == (yd, false))
                        {
                            self.bout_solved = true;
                            self.solved_total += 1;
                            if in_tail {
                                self.solved_tail += 1;
                            }
                            influx = 2.40;
                        }
                    }
                    if !self.bout_solved && self.b_tick == EPISODE_LEN - 1 {
                        stress = 0.25;
                    }
                }
                let dv = f64::from(influx - stress);
                self.total_net_val += dv;
                if in_tail {
                    self.eval_tail_net_val += dv;
                }

                self.b_tick += 1;
                if self.b_tick == DEMO_TICKS {
                    self.cur_x = self.episodes[cur_idx].x_test;
                    self.reg_val = self.cur_x;
                    self.trapped = false;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                } else if self.b_tick >= EPISODE_LEN {
                    self.b_tick = 0;
                    self.ep_idx += 1;
                    self.ops_used = 0;
                    self.emitted_ops.clear();
                    self.bout_solved = false;
                    let nxt = self.ep_idx % n_eps;
                    self.cur_x = self.episodes[nxt].demos[0].0;
                    self.reg_val = self.cur_x;
                    self.trapped = false;
                } else if self.b_tick < DEMO_TICKS {
                    let k = ((self.b_tick / 4) as usize).min(2);
                    self.cur_x = self.episodes[cur_idx].demos[k].0;
                    self.reg_val = self.cur_x;
                    self.trapped = false;
                }
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.refresh_ambient();
    }
}

// =============================================================================
// Pillar 3: 52-Task Sequential Forward-Transfer & Retention Suite (`T_0..T_51`)
// =============================================================================

#[derive(Clone, Debug)]
struct LifeTaskSpec {
    task_id: usize,
    seq: [usize; 4],
    seq_len: usize,
    bout_len: u64,
    num_bouts: usize,
    eval_tail_bouts: usize,
    occluded: bool,
    step_rew: f32,
    term_rew: f32,
}

#[derive(Clone, Debug)]
struct LifelongTaskSet {
    hadamard_code: [[bool; 64]; 52],
    prim: [usize; 8],
    macros: [[usize; 2]; 4],
}

impl LifelongTaskSet {
    fn from_seed(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed ^ P5_LIFE_SALT);
        let mut rows: [usize; 63] = std::array::from_fn(|i| i + 1);
        rng.shuffle(&mut rows);
        let mut cols: [usize; 64] = std::array::from_fn(|i| i);
        rng.shuffle(&mut cols);
        let mut hadamard_code = [[false; 64]; 52];
        for t in 0..52 {
            let r = rows[t];
            for k in 0..64 {
                hadamard_code[t][k] = ((r & cols[k]).count_ones() & 1) == 0;
            }
        }
        let mut prim: [usize; 8] = std::array::from_fn(|i| i);
        rng.shuffle(&mut prim);
        let macros = [
            [prim[0], prim[1]],
            [prim[2], prim[3]],
            [prim[4], prim[5]],
            [prim[6], prim[7]],
        ];
        Self {
            hadamard_code,
            prim,
            macros,
        }
    }

    fn build_tasks(&self, task_range: std::ops::Range<usize>, context_b: bool, override_bouts: Option<usize>) -> Vec<LifeTaskSpec> {
        let mut out = Vec::with_capacity(task_range.len());
        for t in task_range {
            if t < 8 {
                out.push(LifeTaskSpec {
                    task_id: t,
                    seq: [self.prim[t], 0, 0, 0],
                    seq_len: 1,
                    bout_len: 8,
                    num_bouts: override_bouts.unwrap_or(14),
                    eval_tail_bouts: 0,
                    occluded: false,
                    step_rew: 0.0,
                    term_rew: 1.20,
                });
            } else if t < 16 {
                let k = (t - 8) % 4;
                let m = self.macros[k];
                out.push(LifeTaskSpec {
                    task_id: t,
                    seq: [m[0], m[1], 0, 0],
                    seq_len: 2,
                    bout_len: 10,
                    num_bouts: override_bouts.unwrap_or(20),
                    eval_tail_bouts: 0,
                    occluded: false,
                    step_rew: 0.35,
                    term_rew: 1.60,
                });
            } else if t < 36 {
                let off = t - 16;
                let a = off % 4;
                let b = ((off / 4) + a) % 8;
                let m = self.macros[a];
                out.push(LifeTaskSpec {
                    task_id: t,
                    seq: [m[0], m[1], self.prim[b], 0],
                    seq_len: 3,
                    bout_len: 10,
                    num_bouts: override_bouts.unwrap_or(32),
                    eval_tail_bouts: 0,
                    occluded: true,
                    step_rew: 0.0,
                    term_rew: 2.20,
                });
            } else {
                let off = t - 36;
                let a0 = off / 4;
                let b0 = off % 4;
                let (a, b) = if context_b {
                    ((a0 + 1) % 4, (b0 + 2) % 4)
                } else {
                    (a0, b0)
                };
                let ma = self.macros[a];
                let mb = self.macros[b];
                let nb = override_bouts.unwrap_or(28);
                out.push(LifeTaskSpec {
                    task_id: t,
                    seq: [ma[0], ma[1], mb[0], mb[1]],
                    seq_len: 4,
                    bout_len: 12,
                    num_bouts: nb,
                    eval_tail_bouts: if override_bouts.is_some() { nb } else { 8 },
                    occluded: true,
                    step_rew: 0.0,
                    term_rew: 3.00,
                });
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct Lifelong52Medium {
    skin: SharedSkin,
    tasks: LifelongTaskSet,
    schedule: Vec<LifeTaskSpec>,
    task_cursor: usize,
    bout_in_task: usize,
    b_tick: u64,
    step_pos: usize,
    failed_bout: bool,
    last_influx: f32,
    last_stress: f32,
    total_net_val: f64,
    tail_net_val: f64,
    tail_solved: usize,
    tail_total: usize,
}

impl Lifelong52Medium {
    fn new(
        seed: u64,
        salt: u64,
        perm: WirePerm24,
        tasks: LifelongTaskSet,
        schedule: Vec<LifeTaskSpec>,
    ) -> Self {
        let mut m = Self {
            skin: SharedSkin::new(perm, seed ^ salt ^ 0x0033_6633),
            tasks,
            schedule,
            task_cursor: 0,
            bout_in_task: 0,
            b_tick: 0,
            step_pos: 0,
            failed_bout: false,
            last_influx: 0.0,
            last_stress: 0.0,
            total_net_val: 0.0,
            tail_net_val: 0.0,
            tail_solved: 0,
            tail_total: 0,
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        self.schedule
            .iter()
            .map(|t| (t.num_bouts as u64) * t.bout_len)
            .sum()
    }

    fn current_spec(&self) -> &LifeTaskSpec {
        &self.schedule[self.task_cursor.min(self.schedule.len().saturating_sub(1))]
    }

    fn refresh_ambient(&mut self) {
        self.skin.raw_ambient.fill(0.0);
        let spec = self.current_spec().clone();
        let code = self.tasks.hadamard_code[spec.task_id];

        if self.b_tick < 4 {
            for (k, &bright) in code.iter().enumerate() {
                self.skin.raw_ambient[k] = if bright { 0.78 } else { 0.06 };
            }
        }
        if !spec.occluded && !self.failed_bout && self.b_tick >= 4 && self.step_pos < spec.seq_len
        {
            let base = 64 + (self.step_pos % 4) * 4;
            for ch in base..(base + 4) {
                self.skin.raw_ambient[ch] = 0.36;
            }
        }
        // Continuous non-leaking task carrier (`>= 16` channels at `0.78`) keeping CompositionalCortex dormant
        for (k, &bright) in code.iter().enumerate().take(48) {
            self.skin.raw_ambient[96 + k] = if bright { 0.78 } else { 0.06 };
        }
        self.skin.finalize_ambient();
    }
}

impl WorldMedium for Lifelong52Medium {
    fn num_ambient_channels(&self) -> usize {
        P5_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P5_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.skin.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P5_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let triggered = self.skin.step_motor(pulse);
        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;
        let spec = self.current_spec().clone();
        let in_tail =
            spec.eval_tail_bouts > 0 && self.bout_in_task + spec.eval_tail_bouts >= spec.num_bouts;

        if self.b_tick >= 4 && !self.failed_bout && self.step_pos < spec.seq_len {
            if let Some(op) = triggered {
                if op == spec.seq[self.step_pos] {
                    self.step_pos += 1;
                    if self.step_pos == spec.seq_len {
                        influx = spec.term_rew;
                        if in_tail {
                            self.tail_solved += 1;
                        }
                    } else if spec.step_rew > 0.0 {
                        influx = spec.step_rew;
                    }
                } else {
                    self.failed_bout = true;
                    stress = 0.22;
                }
            } else if self.b_tick == spec.bout_len - 1 {
                stress = 0.22;
            }
        }

        let dv = f64::from(influx - stress);
        self.total_net_val += dv;
        if in_tail {
            self.tail_net_val += dv;
        }

        self.b_tick += 1;
        if self.b_tick >= spec.bout_len {
            if in_tail {
                self.tail_total += 1;
            }
            self.b_tick = 0;
            self.step_pos = 0;
            self.failed_bout = false;
            self.bout_in_task += 1;
            if self.bout_in_task >= spec.num_bouts {
                self.bout_in_task = 0;
                if self.task_cursor + 1 < self.schedule.len() {
                    self.task_cursor += 1;
                }
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.refresh_ambient();
    }
}

fn step_medium<M: WorldMedium>(org: &mut Alpha11Organism, m: &mut M, ticks: u64) {
    for _ in 0..ticks {
        let p = {
            let v = m.current_view();
            org.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        m.apply_pulse(&p);
    }
}

fn lesion_oper_slab_in_checkpoint(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() < 28 {
        return Err("Checkpoint too short".to_string());
    }
    let orig_slab_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let payload_end = bytes.len() - 4;
    let mut out = Vec::with_capacity(bytes.len());
    out.extend_from_slice(&bytes[..24]);

    let mut pos = 24usize;
    let mut found_oper = false;
    while pos < payload_end {
        if pos + 12 > payload_end {
            return Err("Corrupt slab header".to_string());
        }
        let tag = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        let s_len =
            u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
                as usize;
        let data_end = pos + 8 + s_len;
        let crc_end = data_end + 4;
        if crc_end > payload_end {
            return Err("Truncated slab".to_string());
        }
        if tag == OPER_SLAB_TAG {
            found_oper = true;
            let empty_payload = [0u8; 4];
            let len_b = (empty_payload.len() as u32).to_le_bytes();
            let tag_b = tag.to_le_bytes();
            let mut c = 0xFFFF_FFFF;
            c = crc32c_update(c, &tag_b);
            c = crc32c_update(c, &len_b);
            c = crc32c_update(c, &empty_payload);
            let scrc = !c;
            out.extend_from_slice(&tag_b);
            out.extend_from_slice(&len_b);
            out.extend_from_slice(&empty_payload);
            out.extend_from_slice(&scrc.to_le_bytes());
        } else {
            out.extend_from_slice(&bytes[pos..crc_end]);
        }
        pos = crc_end;
    }

    if !found_oper {
        return Ok(bytes.to_vec());
    }

    let new_payload_len = (out.len() - 24) as u64;
    out[12..16].copy_from_slice(&orig_slab_count.to_le_bytes());
    out[16..24].copy_from_slice(&new_payload_len.to_le_bytes());
    let fcrc = crc32c(&out);
    out.extend_from_slice(&fcrc.to_le_bytes());
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct Phase5SeedRow {
    arc_nat: f64,
    arc_frz: f64,
    arc_yok: f64,
    arc_shf: f64,
    arc_zs_tr: f64,
    arc_zs_nb: f64,
    arc_zs_les: f64,
    arc_zs_acc: f64,
    exec_nat: f64,
    exec_frz: f64,
    exec_yok: f64,
    exec_shf: f64,
    exec_zs_tr: f64,
    exec_zs_nb: f64,
    exec_zs_les: f64,
    exec_zs_acc: f64,
    life_late_val: f64,
    life_late_nb_val: f64,
    life_late_les_val: f64,
    life_late_acc: f64,
    life_forgetting: f64,
    life_abab_ret: f64,
    life_abab_nb: f64,
    distractor_share: f64,
    nat_agency: f64,
    shf_agency: f64,
    frz_drift: bool,
}

fn run_arc_pillar(
    seed: u64,
    perm: &WirePerm24,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool) {
    let mut nat_org = Alpha11Organism::new(seed ^ 0xA101, OrganismConfig::default());
    let mut donor_train = Vec::with_capacity(2800);

    let mut sb_nat = ArcGridMedium::new(seed, 1, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    let t_sb = sb_nat.total_ticks();
    for _ in 0..t_sb {
        let v = sb_nat.current_view();
        donor_train.push((v.metabolic_influx, v.tissue_stress));
        let p = nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        sb_nat.apply_pulse(&p);
    }

    let mut sn_nat = ArcGridMedium::new(seed, 2, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    let yoked_cut = ((SEEN_EPISODES - SEEN_EVAL_TAIL) as u64) * EPISODE_LEN;
    let t_sn = sn_nat.total_ticks();
    for t in 0..t_sn {
        let v = sn_nat.current_view();
        if t < yoked_cut {
            donor_train.push((v.metabolic_influx, v.tissue_stress));
        }
        let p = nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        sn_nat.apply_pulse(&p);
    }
    let arc_trained_bytes = encode_organism(&nat_org).unwrap_or_default();
    let nat_snap = nat_org.snapshot();
    let dst_arc = perm.distractor_conductance_share(
        &nat_snap.sensor_pore_channels,
        &nat_snap.sensor_pore_conductances,
    );
    let ag_arc = f64::from(nat_snap.mean_interventional_agency);

    let mut un_nat = ArcGridMedium::new(seed, 3, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    let t_un = un_nat.total_ticks();
    step_medium(&mut nat_org, &mut un_nat, t_un);
    let arc_nat = sn_nat.eval_tail_net_val + un_nat.total_net_val;

    // Frozen arm
    let mut frz_org = Alpha11Organism::new(seed ^ 0xA101, OrganismConfig::default());
    frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut sb_frz = ArcGridMedium::new(seed, 1, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    step_medium(&mut frz_org, &mut sb_frz, t_sb);
    let mut sn_frz = ArcGridMedium::new(seed, 2, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    step_medium(&mut frz_org, &mut sn_frz, t_sn);
    let mut un_frz = ArcGridMedium::new(seed, 3, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut frz_org, &mut un_frz, t_un);
    let arc_frz = sn_frz.eval_tail_net_val + un_frz.total_net_val;

    // Yoked arm (yoked during Sandbox + Seen first 72, evaluated real during Seen last 24 + Unseen)
    let mut yok_org = Alpha11Organism::new(seed.wrapping_add(500_001), OrganismConfig::default());
    let sb_yok_src = ArcGridMedium::new(seed, 1, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    let mut sb_yok = YokedMedium::new(sb_yok_src, donor_train[..t_sb as usize].to_vec());
    step_medium(&mut yok_org, &mut sb_yok, t_sb);
    let sn_yok_src = ArcGridMedium::new(seed, 2, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    let mut sn_yok = YokedMedium::new(sn_yok_src, donor_train[t_sb as usize..].to_vec());
    for _ in 0..yoked_cut {
        let p = {
            let v = sn_yok.current_view();
            yok_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sn_yok.apply_pulse(&p);
    }
    let mut sn_yok_eval = sn_yok.into_inner();
    step_medium(&mut yok_org, &mut sn_yok_eval, t_sn - yoked_cut);
    let mut un_yok = ArcGridMedium::new(seed, 3, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut yok_org, &mut un_yok, t_un);
    let arc_yok = sn_yok_eval.eval_tail_net_val + un_yok.total_net_val;

    // ActionShuffled arm
    let mut shf_org = Alpha11Organism::new(seed ^ 0xA101, OrganismConfig::default());
    let mut p_shuf = PulseTemporalChannelShuffler::new(seed ^ P5_SHUF_SALT ^ 1, P5_ACTUATORS);
    let mut sb_shf = ArcGridMedium::new(seed, 1, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    for _ in 0..t_sb {
        let p = {
            let v = sb_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sb_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let mut sn_shf = ArcGridMedium::new(seed, 2, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    for _ in 0..t_sn {
        let p = {
            let v = sn_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sn_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let ag_shf = f64::from(shf_org.snapshot().mean_interventional_agency);
    let mut un_shf = ArcGridMedium::new(seed, 3, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    for _ in 0..t_un {
        let p = {
            let v = un_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        un_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let arc_shf = sn_shf.eval_tail_net_val + un_shf.total_net_val;

    // Zero-Shot Unseen (`FrozenPlasticity`)
    let mut tr_zs = shadow_decode_organism(&arc_trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    tr_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = tr_zs.learned_parameter_hash();
    let mut un_tr = ArcGridMedium::new(seed, 4, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    let t_z = un_tr.total_ticks();
    step_medium(&mut tr_zs, &mut un_tr, t_z);
    let h1 = tr_zs.learned_parameter_hash();
    let arc_zs_tr = un_tr.total_net_val;
    let arc_zs_acc = (un_tr.solved_total as f64) / (UNSEEN_EPISODES as f64);

    let mut nb_zs = Alpha11Organism::new(seed ^ 0xA202, OrganismConfig::default());
    nb_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let mut un_nb = ArcGridMedium::new(seed, 4, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut nb_zs, &mut un_nb, t_z);
    let arc_zs_nb = un_nb.total_net_val;

    let les_bytes = lesion_oper_slab_in_checkpoint(&arc_trained_bytes)
        .unwrap_or_else(|_| arc_trained_bytes.clone());
    let mut les_zs = shadow_decode_organism(&les_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    les_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let mut un_les = ArcGridMedium::new(seed, 4, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut les_zs, &mut un_les, t_z);
    let arc_zs_les = un_les.total_net_val;

    (
        arc_nat,
        arc_frz,
        arc_yok,
        arc_shf,
        arc_zs_tr,
        arc_zs_nb,
        arc_zs_les,
        arc_zs_acc,
        dst_arc,
        ag_arc,
        ag_shf,
        h0 != h1,
    )
}

fn run_exec_pillar(
    seed: u64,
    perm: &WirePerm24,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool) {
    let mut nat_org = Alpha11Organism::new(seed ^ 0xE101, OrganismConfig::default());
    let mut donor_train = Vec::with_capacity(2800);

    let mut sb_nat =
        SandboxedExecMedium::new(seed, 11, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    let t_sb = sb_nat.total_ticks();
    for _ in 0..t_sb {
        let v = sb_nat.current_view();
        donor_train.push((v.metabolic_influx, v.tissue_stress));
        let p = nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        sb_nat.apply_pulse(&p);
    }

    let mut sn_nat =
        SandboxedExecMedium::new(seed, 12, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    let yoked_cut = ((SEEN_EPISODES - SEEN_EVAL_TAIL) as u64) * EPISODE_LEN;
    let t_sn = sn_nat.total_ticks();
    for t in 0..t_sn {
        let v = sn_nat.current_view();
        if t < yoked_cut {
            donor_train.push((v.metabolic_influx, v.tissue_stress));
        }
        let p = nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        sn_nat.apply_pulse(&p);
    }
    let exec_trained_bytes = encode_organism(&nat_org).unwrap_or_default();
    let nat_snap = nat_org.snapshot();
    let dst_exec = perm.distractor_conductance_share(
        &nat_snap.sensor_pore_channels,
        &nat_snap.sensor_pore_conductances,
    );
    let ag_exec = f64::from(nat_snap.mean_interventional_agency);

    let mut un_nat =
        SandboxedExecMedium::new(seed, 13, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    let t_un = un_nat.total_ticks();
    step_medium(&mut nat_org, &mut un_nat, t_un);
    let exec_nat = sn_nat.eval_tail_net_val + un_nat.total_net_val;

    // Frozen arm
    let mut frz_org = Alpha11Organism::new(seed ^ 0xE101, OrganismConfig::default());
    frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut sb_frz =
        SandboxedExecMedium::new(seed, 11, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    step_medium(&mut frz_org, &mut sb_frz, t_sb);
    let mut sn_frz =
        SandboxedExecMedium::new(seed, 12, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    step_medium(&mut frz_org, &mut sn_frz, t_sn);
    let mut un_frz =
        SandboxedExecMedium::new(seed, 13, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut frz_org, &mut un_frz, t_un);
    let exec_frz = sn_frz.eval_tail_net_val + un_frz.total_net_val;

    // Yoked arm
    let mut yok_org = Alpha11Organism::new(seed.wrapping_add(600_001), OrganismConfig::default());
    let sb_yok_src =
        SandboxedExecMedium::new(seed, 11, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    let mut sb_yok = YokedMedium::new(sb_yok_src, donor_train[..t_sb as usize].to_vec());
    step_medium(&mut yok_org, &mut sb_yok, t_sb);
    let sn_yok_src =
        SandboxedExecMedium::new(seed, 12, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    let mut sn_yok = YokedMedium::new(sn_yok_src, donor_train[t_sb as usize..].to_vec());
    for _ in 0..yoked_cut {
        let p = {
            let v = sn_yok.current_view();
            yok_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sn_yok.apply_pulse(&p);
    }
    let mut sn_yok_eval = sn_yok.into_inner();
    step_medium(&mut yok_org, &mut sn_yok_eval, t_sn - yoked_cut);
    let mut un_yok =
        SandboxedExecMedium::new(seed, 13, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut yok_org, &mut un_yok, t_un);
    let exec_yok = sn_yok_eval.eval_tail_net_val + un_yok.total_net_val;

    // ActionShuffled arm
    let mut shf_org = Alpha11Organism::new(seed ^ 0xE101, OrganismConfig::default());
    let mut p_shuf = PulseTemporalChannelShuffler::new(seed ^ P5_SHUF_SALT ^ 2, P5_ACTUATORS);
    let mut sb_shf =
        SandboxedExecMedium::new(seed, 11, ArcStage::Sandbox, SANDBOX_BOUTS, perm.clone());
    for _ in 0..t_sb {
        let p = {
            let v = sb_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sb_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let mut sn_shf =
        SandboxedExecMedium::new(seed, 12, ArcStage::Seen, SEEN_EPISODES, perm.clone());
    for _ in 0..t_sn {
        let p = {
            let v = sn_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        sn_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let ag_shf = f64::from(shf_org.snapshot().mean_interventional_agency);
    let mut un_shf =
        SandboxedExecMedium::new(seed, 13, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    for _ in 0..t_un {
        let p = {
            let v = un_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        un_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let exec_shf = sn_shf.eval_tail_net_val + un_shf.total_net_val;

    // Zero-Shot Unseen (`FrozenPlasticity`)
    let mut tr_zs = shadow_decode_organism(&exec_trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    tr_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = tr_zs.learned_parameter_hash();
    let mut un_tr =
        SandboxedExecMedium::new(seed, 14, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    let t_z = un_tr.total_ticks();
    step_medium(&mut tr_zs, &mut un_tr, t_z);
    let h1 = tr_zs.learned_parameter_hash();
    let exec_zs_tr = un_tr.total_net_val;
    let exec_zs_acc = (un_tr.solved_total as f64) / (UNSEEN_EPISODES as f64);

    let mut nb_zs = Alpha11Organism::new(seed ^ 0xE202, OrganismConfig::default());
    nb_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let mut un_nb =
        SandboxedExecMedium::new(seed, 14, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut nb_zs, &mut un_nb, t_z);
    let exec_zs_nb = un_nb.total_net_val;

    let les_bytes = lesion_oper_slab_in_checkpoint(&exec_trained_bytes)
        .unwrap_or_else(|_| exec_trained_bytes.clone());
    let mut les_zs = shadow_decode_organism(&les_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    les_zs.set_control_mode(ControlMode::FrozenPlasticity);
    let mut un_les =
        SandboxedExecMedium::new(seed, 14, ArcStage::Unseen, UNSEEN_EPISODES, perm.clone());
    step_medium(&mut les_zs, &mut un_les, t_z);
    let exec_zs_les = un_les.total_net_val;

    (
        exec_nat,
        exec_frz,
        exec_yok,
        exec_shf,
        exec_zs_tr,
        exec_zs_nb,
        exec_zs_les,
        exec_zs_acc,
        dst_exec,
        ag_exec,
        ag_shf,
        h0 != h1,
    )
}

fn run_lifelong_pillar(
    seed: u64,
    perm: &WirePerm24,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool) {
    let tasks = LifelongTaskSet::from_seed(seed);
    let mut org = Alpha11Organism::new(seed ^ 0xF101, OrganismConfig::default());

    // T_0..T_15 (`16` foundational tasks)
    let sched_e16 = tasks.build_tasks(0..16, false, None);
    let mut w_e16 = Lifelong52Medium::new(seed, 21, perm.clone(), tasks.clone(), sched_e16);
    let t_e16 = w_e16.total_ticks();
    step_medium(&mut org, &mut w_e16, t_e16);

    let after_e16_bytes = encode_organism(&org).unwrap_or_default();
    let mut peak_eval_org = shadow_decode_organism(&after_e16_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    peak_eval_org.set_control_mode(ControlMode::FrozenPlasticity);
    let h_p0 = peak_eval_org.learned_parameter_hash();
    let eval_e16_sched = tasks.build_tasks(0..16, false, Some(2));
    let mut w_peak = Lifelong52Medium::new(seed, 22, perm.clone(), tasks.clone(), eval_e16_sched);
    let t_eval16 = w_peak.total_ticks();
    step_medium(&mut peak_eval_org, &mut w_peak, t_eval16);
    let h_p1 = peak_eval_org.learned_parameter_hash();
    let peak_early16 = w_peak.total_net_val;

    // T_16..T_35 (`20` Depth-3 occluded tasks)
    let sched_d3 = tasks.build_tasks(16..36, false, None);
    let mut w_d3 = Lifelong52Medium::new(seed, 23, perm.clone(), tasks.clone(), sched_d3);
    let t_d3 = w_d3.total_ticks();
    step_medium(&mut org, &mut w_d3, t_d3);
    let after_t35_bytes = encode_organism(&org).unwrap_or_default();

    // T_36..T_51 (`16` Depth-4 hierarchical composition tasks, tail 8 bouts scored)
    let sched_l16 = tasks.build_tasks(36..52, false, None);
    let mut w_l16 =
        Lifelong52Medium::new(seed, 24, perm.clone(), tasks.clone(), sched_l16.clone());
    let t_l16 = w_l16.total_ticks();
    step_medium(&mut org, &mut w_l16, t_l16);
    let late_val = w_l16.tail_net_val;
    let late_acc = (w_l16.tail_solved as f64) / (w_l16.tail_total.max(1) as f64);

    let snap_end = org.snapshot();
    let dst_life = perm.distractor_conductance_share(
        &snap_end.sensor_pore_channels,
        &snap_end.sensor_pore_conductances,
    );
    let ag_life = f64::from(snap_end.mean_interventional_agency);

    // Control 1 for P5-G3a: Fresh plastic newborn on T_36..T_51
    let mut nb_l16 = Alpha11Organism::new(seed ^ 0xF202, OrganismConfig::default());
    let mut w_nb16 =
        Lifelong52Medium::new(seed, 24, perm.clone(), tasks.clone(), sched_l16.clone());
    step_medium(&mut nb_l16, &mut w_nb16, t_l16);
    let late_nb_val = w_nb16.tail_net_val;

    // Control 2 for P5-G3b: Flagless OPER-slab lesion after T_35, plastic on T_36..T_51
    let les_bytes =
        lesion_oper_slab_in_checkpoint(&after_t35_bytes).unwrap_or_else(|_| after_t35_bytes.clone());
    let mut les_l16 = shadow_decode_organism(&les_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    let mut w_ls16 = Lifelong52Medium::new(seed, 24, perm.clone(), tasks.clone(), sched_l16);
    step_medium(&mut les_l16, &mut w_ls16, t_l16);
    let late_les_val = w_ls16.tail_net_val;

    // Re-evaluate T_0..T_15 (`FrozenPlasticity`) after T_51 (`P5-G4a` <2% forgetting)
    let after_t51_bytes = encode_organism(&org).unwrap_or_default();
    let mut fin_eval_org = shadow_decode_organism(&after_t51_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    fin_eval_org.set_control_mode(ControlMode::FrozenPlasticity);
    let h_f0 = fin_eval_org.learned_parameter_hash();
    let eval_fin_sched = tasks.build_tasks(0..16, false, Some(2));
    let mut w_fin = Lifelong52Medium::new(seed, 22, perm.clone(), tasks.clone(), eval_fin_sched);
    step_medium(&mut fin_eval_org, &mut w_fin, t_eval16);
    let h_f1 = fin_eval_org.learned_parameter_hash();
    let final_early16 = w_fin.total_net_val;
    let forgetting =
        (peak_early16 - final_early16).max(0.0) / peak_early16.abs().max(1.0);

    // P5-G4b: ABAB rule-flip on T_36..T_39 (Context B -> Return Context A vs Newborn)
    let mut abab_org = shadow_decode_organism(&after_t51_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    let sched_ctx_b = tasks.build_tasks(36..40, true, Some(28));
    let mut w_ctxb = Lifelong52Medium::new(seed, 25, perm.clone(), tasks.clone(), sched_ctx_b);
    let t_b = w_ctxb.total_ticks();
    step_medium(&mut abab_org, &mut w_ctxb, t_b);

    let sched_ret_a = tasks.build_tasks(36..40, false, Some(10));
    let mut w_reta =
        Lifelong52Medium::new(seed, 26, perm.clone(), tasks.clone(), sched_ret_a.clone());
    let t_ra = w_reta.total_ticks();
    step_medium(&mut abab_org, &mut w_reta, t_ra);
    let abab_ret = w_reta.total_net_val;

    let mut nb_ret = Alpha11Organism::new(seed ^ 0xF303, OrganismConfig::default());
    let mut w_nbret = Lifelong52Medium::new(seed, 26, perm.clone(), tasks.clone(), sched_ret_a);
    step_medium(&mut nb_ret, &mut w_nbret, t_ra);
    let abab_nb = w_nbret.total_net_val;

    // ActionShuffled control for T_0..T_15 agency purity
    let mut shf_org = Alpha11Organism::new(seed ^ 0xF101, OrganismConfig::default());
    let mut p_shuf = PulseTemporalChannelShuffler::new(seed ^ P5_SHUF_SALT ^ 3, P5_ACTUATORS);
    let sched_shf = tasks.build_tasks(0..16, false, None);
    let mut w_shf = Lifelong52Medium::new(seed, 21, perm.clone(), tasks, sched_shf);
    for _ in 0..t_e16 {
        let p = {
            let v = w_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        w_shf.apply_pulse(&p_shuf.shuffle_pulse(&p));
    }
    let ag_shf = f64::from(shf_org.snapshot().mean_interventional_agency);

    (
        late_val,
        late_nb_val,
        late_les_val,
        late_acc,
        forgetting,
        abab_ret,
        abab_nb,
        dst_life,
        ag_life,
        ag_shf,
        (h_p0 != h_p1) || (h_f0 != h_f1),
    )
}

fn run_phase5_seed(seed: u64) -> Phase5SeedRow {
    let perm = WirePerm24::from_seed(seed);
    let (
        arc_nat,
        arc_frz,
        arc_yok,
        arc_shf,
        arc_zs_tr,
        arc_zs_nb,
        arc_zs_les,
        arc_zs_acc,
        dst_arc,
        ag_arc,
        shf_ag_arc,
        drift_arc,
    ) = run_arc_pillar(seed, &perm);
    let (
        exec_nat,
        exec_frz,
        exec_yok,
        exec_shf,
        exec_zs_tr,
        exec_zs_nb,
        exec_zs_les,
        exec_zs_acc,
        dst_exec,
        ag_exec,
        shf_ag_exec,
        drift_exec,
    ) = run_exec_pillar(seed, &perm);
    let (
        life_late_val,
        life_late_nb_val,
        life_late_les_val,
        life_late_acc,
        life_forgetting,
        life_abab_ret,
        life_abab_nb,
        dst_life,
        ag_life,
        shf_ag_life,
        drift_life,
    ) = run_lifelong_pillar(seed, &perm);

    Phase5SeedRow {
        arc_nat,
        arc_frz,
        arc_yok,
        arc_shf,
        arc_zs_tr,
        arc_zs_nb,
        arc_zs_les,
        arc_zs_acc,
        exec_nat,
        exec_frz,
        exec_yok,
        exec_shf,
        exec_zs_tr,
        exec_zs_nb,
        exec_zs_les,
        exec_zs_acc,
        life_late_val,
        life_late_nb_val,
        life_late_les_val,
        life_late_acc,
        life_forgetting,
        life_abab_ret,
        life_abab_nb,
        distractor_share: (dst_arc + dst_exec + dst_life) / 3.0,
        nat_agency: (ag_arc + ag_exec + ag_life) / 3.0,
        shf_agency: (shf_ag_arc + shf_ag_exec + shf_ag_life) / 3.0,
        frz_drift: drift_arc || drift_exec || drift_life,
    }
}

fn sign_flip_two_sided_p(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let mut diffs = Vec::with_capacity(n);
    let mut sum = 0.0_f64;
    for i in 0..n {
        let d = a[i] - b[i];
        diffs.push(d);
        sum += d;
    }
    let obs = sum.abs();
    if obs <= 1.0e-12 {
        return 1.0;
    }
    if n <= 20 {
        let total = 1usize << n;
        let mut ge = 0usize;
        for mask in 0..total {
            let mut s = 0.0_f64;
            for (i, &d) in diffs.iter().enumerate() {
                if ((mask >> i) & 1) == 1 {
                    s -= d;
                } else {
                    s += d;
                }
            }
            if s.abs() + 1.0e-12 >= obs {
                ge += 1;
            }
        }
        (ge as f64) / (total as f64)
    } else {
        let mut rng = DeterministicRng::new(0x5EED_CAFE_5050_0001);
        let perms = 20_000usize;
        let mut ge = 0usize;
        for _ in 0..perms {
            let mut s = 0.0_f64;
            for &d in &diffs {
                if (rng.next_u64() & 1) == 1 {
                    s -= d;
                } else {
                    s += d;
                }
            }
            if s.abs() + 1.0e-12 >= obs {
                ge += 1;
            }
        }
        ((ge + 1) as f64) / ((perms + 1) as f64)
    }
}

fn conservative_paired_gate(a: &[f64], b: &[f64]) -> PairedGate {
    let mut g = paired_gate(a, b);
    let p_sf = sign_flip_two_sided_p(a, b);
    g.p_two_sided = g.p_two_sided.max(p_sf).min(1.0);
    g
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / (xs.len() as f64)
    }
}

fn run_jobs<T: Send + 'static, F: Fn(usize) -> T + Sync>(
    count: usize,
    threads: usize,
    f: F,
) -> Vec<Option<T>> {
    let next_idx = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..count).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.min(count).max(1) {
            scope.spawn(|| {
                loop {
                    let i = next_idx.fetch_add(1, Ordering::Relaxed);
                    if i >= count {
                        break;
                    }
                    let val = f(i);
                    if let Ok(mut guard) = slots[i].lock() {
                        *guard = Some(val);
                    }
                }
            });
        }
    });
    slots
        .into_iter()
        .map(|m| m.into_inner().ok().flatten())
        .collect()
}

fn strip_cfg_test(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_test = false;
    let mut depth = 0i32;
    let mut saw_brace = false;
    for line in src.lines() {
        let t = line.trim();
        if !in_test && (t.starts_with("#[cfg(test)]") || t.starts_with("#[test]")) {
            in_test = true;
            depth = 0;
            saw_brace = false;
            continue;
        }
        if in_test {
            for ch in line.chars() {
                if ch == '{' {
                    depth += 1;
                    saw_brace = true;
                } else if ch == '}' {
                    depth -= 1;
                }
            }
            if saw_brace && depth <= 0 {
                in_test = false;
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

fn audit_core_debt(ws_root: &Path) -> (usize, usize, usize, usize) {
    let core_src = ws_root.join("crates/a11-core/src");
    let rep = scan_crate_source(&core_src);
    let hygiene = rep.map_or(usize::MAX, |r| {
        r.string_literal_count
            + r.forbidden_identifier_count
            + r.unsafe_block_count
            + r.unwrap_call_count
    });
    let mut split = 0usize;
    let mut mirrors = 0usize;
    let mut modulo = 0usize;
    if let Ok(entries) = fs::read_dir(&core_src) {
        for e in entries.filter_map(Result::ok) {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "rs")
                || p.file_name().is_some_and(|n| n == "causal_core.rs")
            {
                continue;
            }
            if let Ok(raw) = fs::read_to_string(&p) {
                let prod = strip_cfg_test(&raw);
                for ln in prod.lines() {
                    let tr = ln.trim();
                    if tr.starts_with("//") {
                        continue;
                    }
                    if ["STRIATAL_ACTUATORS", "m_organs <= 8", "<= 8"]
                        .iter()
                        .any(|n| tr.contains(n))
                    {
                        split += 1;
                    }
                    if ["omega = 75.0", "2.4572", "0.66615", "deconv_hair_cell_35"]
                        .iter()
                        .any(|n| tr.contains(n))
                    {
                        mirrors += 1;
                    }
                    if tr.contains("% PRED_BUCKETS") {
                        modulo += 1;
                    }
                }
            }
        }
    }
    (hygiene, split, mirrors, modulo)
}

fn audit_latency_and_frozen(any_seed_drifted: bool) -> Result<(f64, usize, usize, usize), String> {
    if any_seed_drifted {
        return Err("FrozenPlasticity parameter hash drifted across Phase-5 evaluation".to_string());
    }
    let cfg = OrganismConfig {
        initial_cells: 512,
        max_cells: 65_536,
        ..OrganismConfig::default()
    };
    let mut org = Alpha11Organism::new(2026_5050, cfg);
    let mut world = DenseFieldWorld::new(91);
    let steps = 1_000usize;
    let mut us = Vec::with_capacity(steps);
    let mut max_act_cells = 0usize;
    for _ in 0..steps {
        let p = {
            let v = world.current_view();
            let t0 = Instant::now();
            let pulse = org.step(v).map_err(|e| format!("{e:?}"))?;
            us.push(t0.elapsed().as_secs_f64() * 1.0e6);
            pulse
        };
        max_act_cells = max_act_cells.max(org.snapshot().active_cells_last_tick);
        world.apply_pulse(&p);
    }
    us.sort_by(|a, b| a.total_cmp(b));
    let p99 = us[((steps * 99) / 100).min(steps - 1)];
    let snap = org.snapshot();
    Ok((p99, max_act_cells, snap.sensor_pore_count, snap.cell_count))
}

fn audit_phase5_persistence() -> Result<usize, String> {
    let seed = 2026_0505_u64;
    let perm = WirePerm24::from_seed(seed);
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w1 = ArcGridMedium::new(seed, 51, ArcStage::Seen, 24, perm.clone());
    let t1 = w1.total_ticks();
    step_medium(&mut org, &mut w1, t1);

    let mut w2 = ArcGridMedium::new(seed, 52, ArcStage::Unseen, 8, perm);
    step_medium(&mut org, &mut w2, 14);

    let dir = env::temp_dir().join(format!("a11_p5_ckpt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let res = (|| -> Result<usize, String> {
        let primary = dir.join("p5.ckpt");
        let prev = dir.join("p5.ckpt.prev");
        let tmp = dir.join("p5.ckpt.tmp");

        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen1 save: {e:?}"))?;
        let p = org.step(w2.current_view()).map_err(|e| format!("{e:?}"))?;
        w2.apply_pulse(&p);
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen2 save: {e:?}"))?;
        let ckpt_bytes = fs::metadata(&primary).map_err(|e| e.to_string())?.len() as usize;

        let mut restored = Alpha11Organism::new(1, OrganismConfig::default());
        if restored
            .load_transactional(&primary, &prev)
            .map_err(|e| format!("{e:?}"))?
        {
            return Err("Clean primary fell back to .prev".to_string());
        }
        let mut wr = w2.clone();
        for step_idx in 0..160usize {
            let pu = org.step(w2.current_view()).map_err(|e| format!("{e:?}"))?;
            let pr = restored
                .step(wr.current_view())
                .map_err(|e| format!("{e:?}"))?;
            if pu
                .actuator_out
                .iter()
                .zip(pr.actuator_out.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!("Trajectory diverged at step {step_idx}"));
            }
            w2.apply_pulse(&pu);
            wr.apply_pulse(&pr);
        }

        let mut bad = fs::read(&primary).map_err(|e| e.to_string())?;
        let m = bad.len() / 2;
        bad[m] ^= 0x5A;
        fs::write(&primary, &bad).map_err(|e| e.to_string())?;
        let mut healed = Alpha11Organism::new(2, OrganismConfig::default());
        if !healed
            .load_transactional(&primary, &prev)
            .map_err(|e| format!("{e:?}"))?
        {
            return Err("Corrupt primary did not fall back to .prev".to_string());
        }

        let mut bad_p = fs::read(&prev).map_err(|e| e.to_string())?;
        let pm = bad_p.len() / 2;
        bad_p[pm] ^= 0xA5;
        fs::write(&prev, &bad_p).map_err(|e| e.to_string())?;
        let mut victim = Alpha11Organism::new(3, OrganismConfig::default());
        if victim.load_transactional(&primary, &prev).is_ok() {
            return Err("Dual corruption succeeded".to_string());
        }
        if !matches!(
            victim.step(w2.current_view()),
            Err(PersistError::PoisonedAfterCorruption)
        ) {
            return Err("Dual corruption did not poison victim".to_string());
        }
        Ok(ckpt_bytes)
    })();

    let _ = fs::remove_dir_all(&dir);
    res
}

#[inline]
fn push_u32(dst: &mut Vec<u8>, v: u32) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_u64(dst: &mut Vec<u8>, v: u64) {
    dst.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn push_f32(dst: &mut Vec<u8>, v: f32) {
    dst.extend_from_slice(&v.to_le_bytes());
}

fn push_slab(out: &mut Vec<u8>, tag: u32, payload: &[u8]) {
    let tag_b = tag.to_le_bytes();
    let len_b = (payload.len() as u32).to_le_bytes();
    let mut crc = 0xFFFF_FFFF;
    crc = crc32c_update(crc, &tag_b);
    crc = crc32c_update(crc, &len_b);
    crc = crc32c_update(crc, payload);
    out.extend_from_slice(&tag_b);
    out.extend_from_slice(&len_b);
    out.extend_from_slice(payload);
    out.extend_from_slice(&(!crc).to_le_bytes());
}

fn build_canonical_v5_to_v7(ver: u32) -> Vec<u8> {
    const EXPLORE: usize = 32;
    const TRACKED_ACT: usize = 32;
    const BUCKETS: usize = 64;
    const RING: usize = 16;
    const OFC_CTX: usize = 4;
    const STR_PORES: usize = 64;
    const STR_ACT: usize = 8;
    const STR_CUES: usize = 4 * STR_PORES;
    const STR_FEATS: usize = STR_CUES + 1;
    const STR_TAP_DIM: usize = 4 * STR_ACT;
    const STR_W_LEN: usize = STR_ACT * STR_CUES;
    const STR_FWD_LEN: usize = STR_CUES * STR_ACT;

    let f32_zeros = |dst: &mut Vec<u8>, cnt: usize| {
        for _ in 0..cnt {
            push_f32(dst, 0.0);
        }
    };

    let mut body = Vec::new();
    push_u64(&mut body, 42);
    push_u64(&mut body, 32);
    push_u64(&mut body, 65_536);
    push_u64(&mut body, 524_288);
    push_u64(&mut body, 512);
    push_u64(&mut body, 268_435_456);
    push_f32(&mut body, 10.0);
    push_f32(&mut body, 100.0);
    body.push(0);
    push_u64(&mut body, 0x1234_5678_9ABC_DEF1);
    push_u64(&mut body, 0xFEDC_BA98_7654_3211);
    push_u64(&mut body, 0x1111_2222_3333_4445);
    push_u64(&mut body, 0x5555_6666_7777_8889);
    push_u64(&mut body, 777);
    push_u64(&mut body, 500);
    push_u64(&mut body, 12);
    push_u64(&mut body, 4);
    push_u64(&mut body, 2);
    push_u64(&mut body, 3);
    push_u64(&mut body, 1);
    push_f32(&mut body, 0.12);
    for v in [
        42.5_f32, 100.0, 10.0, 0.15, 0.25, 0.05, 0.02, 0.03, 0.01, -0.04, 0.08, 0.05, 0.03,
    ] {
        push_f32(&mut body, v);
    }
    body.extend_from_slice(&0_u16.to_le_bytes());
    push_u64(&mut body, 20);
    f32_zeros(&mut body, EXPLORE * 3);
    body.extend(std::iter::repeat_n(0_u8, EXPLORE));

    let mut cell = Vec::new();
    push_u32(&mut cell, 4);
    push_u32(&mut cell, 4);
    push_u32(&mut cell, 4);
    for _ in 0..4 {
        cell.push(1);
        for v in [
            0.1_f32, 0.05, 0.05, 0.08, 0.02, 0.0, 8.0, 0.1, 0.0, 0.0, 0.0, 0.01, 0.02, 0.1,
        ] {
            push_f32(&mut cell, v);
        }
        cell.extend_from_slice(&0_u16.to_le_bytes());
        push_u32(&mut cell, 777);
    }
    push_u32(&mut cell, 0);
    push_u32(&mut cell, 2);
    push_u32(&mut cell, 0);
    push_u32(&mut cell, 1);
    push_f32(&mut cell, 0.5);
    push_f32(&mut cell, 0.1);

    let mut pore = Vec::new();
    push_u32(&mut pore, 2);
    push_u32(&mut pore, 2);
    for ch in 0..2_u32 {
        pore.push(1);
        push_u32(&mut pore, ch);
        pore.push(1);
        push_u32(&mut pore, 0);
        push_u32(&mut pore, 1);
        push_f32(&mut pore, 0.4);
        push_f32(&mut pore, 0.2);
        push_f32(&mut pore, 0.1);
        push_f32(&mut pore, 0.05);
        pore.push(1);
        for _ in 0..16 {
            push_f32(&mut pore, 0.1);
        }
        pore.push(0);
        push_u32(&mut pore, 777);
    }
    push_u32(&mut pore, 0);
    push_u32(&mut pore, 2);
    push_u32(&mut pore, 2);
    for ch in 0..2_u32 {
        pore.push(1);
        push_u32(&mut pore, ch);
        for v in [0.45_f32, 0.0, 0.2, 0.1, 0.05] {
            push_f32(&mut pore, v);
        }
    }
    push_u32(&mut pore, 0);

    let mut tube = Vec::new();
    push_u32(&mut tube, 2);
    push_u32(&mut tube, 2);
    if ver >= 7 {
        tube.push(0);
    }
    for (kind, pre, post) in [(0_u8, 0_u32, 1_u32), (1_u8, 1_u32, 0_u32)] {
        tube.push(1);
        tube.push(kind);
        push_u32(&mut tube, pre);
        push_u32(&mut tube, post);
        tube.push(1);
        for u in [0.2_f32, 0.1, 0.05, 0.01] {
            push_f32(&mut tube, u);
        }
        push_f32(&mut tube, 0.15);
        tube.push(0);
        push_u32(&mut tube, 0);
        tube.push(0);
        push_f32(&mut tube, 0.0);
        push_f32(&mut tube, 0.1);
        push_f32(&mut tube, 0.0);
        push_u32(&mut tube, u32::MAX);
        push_f32(&mut tube, 0.1);
        push_u32(&mut tube, 777);
    }
    push_u32(&mut tube, 0);
    if ver >= 6 {
        push_u32(&mut tube, TRACKED_ACT as u32);
        push_u32(&mut tube, BUCKETS as u32);
    }
    for _ in 0..(3 * TRACKED_ACT * BUCKETS) {
        push_f32(&mut tube, 0.01);
    }
    f32_zeros(&mut tube, 3 * BUCKETS);
    for _ in 0..BUCKETS {
        push_f32(&mut tube, 0.05);
    }
    for _ in 0..BUCKETS {
        push_f32(&mut tube, 0.02);
    }
    for _ in 0..BUCKETS {
        push_f32(&mut tube, 1.0e-4);
    }
    for _ in 0..BUCKETS {
        push_f32(&mut tube, 5.0e-3);
    }
    f32_zeros(&mut tube, BUCKETS);
    push_f32(&mut tube, 0.0);
    f32_zeros(&mut tube, RING);
    tube.push(0);
    f32_zeros(&mut tube, 3 * TRACKED_ACT);
    f32_zeros(&mut tube, 7);
    tube.push(255);
    tube.push(255);
    f32_zeros(&mut tube, 2 * TRACKED_ACT);
    push_f32(&mut tube, 0.18);
    if ver >= 6 {
        push_u32(&mut tube, STR_ACT as u32);
    }
    tube.push(1);
    tube.push(0);
    tube.push(0);
    tube.extend(std::iter::repeat_n(0_u8, OFC_CTX));
    for _ in 0..OFC_CTX {
        push_u32(&mut tube, 12);
    }
    for _ in 0..(2 * OFC_CTX) {
        push_f32(&mut tube, 0.01);
    }
    for _ in 0..STR_PORES {
        push_u32(&mut tube, u32::MAX);
    }
    for _ in 0..STR_PORES {
        push_u32(&mut tube, 0);
    }
    f32_zeros(&mut tube, STR_PORES);
    for _ in 0..(2 * STR_PORES) {
        push_f32(&mut tube, 1.0e-4);
    }
    f32_zeros(&mut tube, STR_PORES * STR_TAP_DIM);
    f32_zeros(&mut tube, STR_W_LEN);
    f32_zeros(&mut tube, 2 * OFC_CTX * STR_W_LEN);
    f32_zeros(&mut tube, 2 * OFC_CTX * STR_FEATS);
    f32_zeros(&mut tube, 2 * OFC_CTX * STR_CUES);
    f32_zeros(&mut tube, 2 * STR_CUES);
    f32_zeros(&mut tube, (1 + 2 * OFC_CTX) * STR_FWD_LEN);
    for _ in 0..(2 * STR_CUES) {
        push_f32(&mut tube, 1.0e-4);
    }
    f32_zeros(&mut tube, 2 * OFC_CTX);
    f32_zeros(&mut tube, STR_ACT);
    push_u32(&mut tube, 0);
    push_u32(&mut tube, 0);
    f32_zeros(&mut tube, OFC_CTX);
    f32_zeros(&mut tube, 3);
    f32_zeros(&mut tube, STR_TAP_DIM);
    f32_zeros(&mut tube, STR_W_LEN);
    f32_zeros(&mut tube, 3 * STR_FEATS);
    f32_zeros(&mut tube, 2 * STR_CUES);
    f32_zeros(&mut tube, 2 * STR_ACT);
    push_f32(&mut tube, 0.0);
    tube.push(0);
    push_u64(&mut tube, 777);
    push_u64(&mut tube, 0xCAFE_BABE_1234_5678);
    push_u64(&mut tube, 0x8765_4321_FEED_FACE);

    if ver >= 6 {
        f32_zeros(&mut tube, BUCKETS);
        for _ in 0..BUCKETS {
            push_f32(&mut tube, 0.59);
        }
        f32_zeros(&mut tube, BUCKETS);
        for _ in 0..BUCKETS {
            push_f32(&mut tube, 1.0);
        }
        f32_zeros(&mut tube, 2 * BUCKETS);
        f32_zeros(&mut tube, TRACKED_ACT);
        f32_zeros(&mut tube, 6);
        push_u32(&mut tube, 0);
        push_u32(&mut tube, 0);
        tube.push(u8::MAX);
        push_u32(&mut tube, 0);
        push_f32(&mut tube, 0.0);
        f32_zeros(&mut tube, 16);
        push_u32(&mut tube, 0);
        push_f32(&mut tube, 0.0);
        f32_zeros(&mut tube, 16);
    }
    if ver >= 7 {
        push_u32(&mut tube, OFC_CTX as u32);
        tube.push(0);
        for c in 0..(OFC_CTX as u32) {
            push_u32(&mut tube, c);
        }
        for _ in 0..OFC_CTX {
            push_u64(&mut tube, 0);
        }
        for _ in 0..OFC_CTX {
            for _ in 0..STR_PORES {
                push_u32(&mut tube, u32::MAX);
            }
            tube.extend(std::iter::repeat_n(0_u8, STR_PORES));
            f32_zeros(&mut tube, STR_PORES);
        }
        f32_zeros(&mut tube, 5 * STR_PORES);
        push_u32(&mut tube, 0);
    }

    let mut ring = Vec::new();
    push_u32(&mut ring, 0);
    for _ in 0..RING {
        ring.push(0);
        push_u64(&mut ring, 0);
        ring.push(0);
        push_f32(&mut ring, 0.0);
        if ver >= 6 {
            push_u32(&mut ring, TRACKED_ACT as u32);
        }
        f32_zeros(&mut ring, TRACKED_ACT);
        push_f32(&mut ring, 0.0);
        push_f32(&mut ring, 0.0);
        push_u32(&mut ring, 0);
        push_u32(&mut ring, 0);
        push_u32(&mut ring, 0);
    }

    let mut scot = Vec::new();
    push_u32(&mut scot, 0);
    if ver >= 6 {
        scot.push(0);
    }
    push_f32(&mut scot, 0.25);
    push_f32(&mut scot, 0.1);
    push_f32(&mut scot, 0.1);
    push_f32(&mut scot, 0.01);
    push_u32(&mut scot, 2);
    for _ in 0..2 {
        scot.push(1);
        for v in [
            0.0_f32, 0.02, 0.005, 0.0, 0.8, 0.004, 0.0, 0.001, 0.001, 0.0,
        ] {
            push_f32(&mut scot, v);
        }
        if ver >= 6 {
            scot.extend_from_slice(&256_u16.to_le_bytes());
            push_f32(&mut scot, 0.0);
        }
    }

    let mut slabs = Vec::new();
    push_slab(&mut slabs, TAG_BODY, &body);
    push_slab(&mut slabs, TAG_CELL, &cell);
    push_slab(&mut slabs, TAG_TUBE, &tube);
    push_slab(&mut slabs, TAG_PORE, &pore);
    push_slab(&mut slabs, TAG_RING, &ring);
    push_slab(&mut slabs, TAG_SCOT, &scot);

    let mut full = Vec::with_capacity(28 + slabs.len());
    push_u64(&mut full, FILE_MAGIC);
    push_u32(&mut full, ver);
    push_u32(&mut full, SLAB_COUNT);
    push_u64(&mut full, slabs.len() as u64);
    full.extend_from_slice(&slabs);
    let file_crc = crc32c(&full);
    push_u32(&mut full, file_crc);
    full
}

fn repackage_subset_slabs(
    enc: &[u8],
    target_ver: u32,
    max_base_slabs: usize,
) -> Result<Vec<u8>, String> {
    if enc.len() < 28 {
        return Err("Checkpoint too short".to_string());
    }
    let keep_tags: &[u32] = if target_ver <= 8 {
        &[TAG_BODY, TAG_CELL, TAG_TUBE, TAG_PORE, TAG_RING, TAG_SCOT]
    } else {
        &[
            TAG_BODY,
            TAG_CELL,
            TAG_TUBE,
            TAG_PORE,
            TAG_RING,
            TAG_SCOT,
            OPER_SLAB_TAG,
        ]
    };
    let payload_end = enc.len() - 4;
    let mut slabs = Vec::with_capacity(enc.len());
    let mut kept = 0u32;
    let mut pos = 24usize;
    while pos < payload_end && (kept as usize) < max_base_slabs {
        let tag = u32::from_le_bytes([enc[pos], enc[pos + 1], enc[pos + 2], enc[pos + 3]]);
        let s_len =
            u32::from_le_bytes([enc[pos + 4], enc[pos + 5], enc[pos + 6], enc[pos + 7]])
                as usize;
        let crc_end = pos + 8 + s_len + 4;
        if crc_end > payload_end {
            return Err("Corrupt slab length".to_string());
        }
        if keep_tags.contains(&tag) {
            if tag == OPER_SLAB_TAG && target_ver == 9 {
                let empty_oper = [0_u8; 4];
                push_slab(&mut slabs, tag, &empty_oper);
            } else {
                push_slab(&mut slabs, tag, &enc[pos + 8..pos + 8 + s_len]);
            }
            kept += 1;
        }
        pos = crc_end;
    }
    let mut out = Vec::with_capacity(28 + slabs.len());
    push_u64(&mut out, FILE_MAGIC);
    push_u32(&mut out, target_ver);
    push_u32(&mut out, kept);
    push_u64(&mut out, slabs.len() as u64);
    out.extend_from_slice(&slabs);
    let fcrc = crc32c(&out);
    push_u32(&mut out, fcrc);
    Ok(out)
}

fn audit_v10_plus_migration(opt_live_chk: Option<&Path>) -> Result<String, String> {
    if FORMAT_VERSION < 10 {
        return Err(format!("FORMAT_VERSION={FORMAT_VERSION} (< 10)"));
    }
    if MIN_COMPAT_FORMAT_VERSION > 5 {
        return Err("MIN_COMPAT_FORMAT_VERSION > 5".to_string());
    }
    let max_b = OrganismConfig::default().max_state_bytes;
    let perm = WirePerm24::from_seed(55);

    for ver in [5_u32, 6_u32, 7_u32] {
        let bytes = build_canonical_v5_to_v7(ver);
        let mut org = shadow_decode_organism(&bytes, max_b)
            .map_err(|e| format!("v{ver} decode failed: {e:?}"))?;
        let mut w = ArcGridMedium::new(55, 61, ArcStage::Seen, 4, perm.clone());
        step_medium(&mut org, &mut w, 10);
        let enc = encode_organism(&org).map_err(|e| format!("v{ver} upgrade encode: {e:?}"))?;
        let _ = shadow_decode_organism(&enc, max_b)
            .map_err(|e| format!("v{ver} upgrade roundtrip: {e:?}"))?;
    }

    let fresh = Alpha11Organism::new(55, OrganismConfig::default());
    let current_bytes = encode_organism(&fresh).map_err(|e| format!("{e:?}"))?;
    for legacy_v in [8_u32, 9_u32] {
        let slabs = if legacy_v == 8 { 6 } else { 7 };
        let b = repackage_subset_slabs(&current_bytes, legacy_v, slabs)?;
        let _ = shadow_decode_organism(&b, max_b)
            .map_err(|e| format!("v{legacy_v}->v{FORMAT_VERSION} upgrade failed: {e:?}"))?;
    }

    if FORMAT_VERSION > 10 {
        let v10_bytes = repackage_subset_slabs(&current_bytes, 10, 7)?;
        let _ = shadow_decode_organism(&v10_bytes, max_b)
            .map_err(|e| format!("v10->v{FORMAT_VERSION} upgrade failed: {e:?}"))?;
    } else {
        let _ = shadow_decode_organism(&current_bytes, max_b)
            .map_err(|e| format!("v10 roundtrip failed: {e:?}"))?;
    }

    let mut live_tag = "live_chk=skipped";
    if let Some(p) = opt_live_chk {
        let disk_bytes =
            fs::read(p).map_err(|e| format!("failed reading --live-chk {}: {e}", p.display()))?;
        let mut live_org = shadow_decode_organism(&disk_bytes, max_b)
            .map_err(|e| format!("--live-chk decode failed: {e:?}"))?;
        let pre_tick = live_org.snapshot().tick;
        let mut w = ArcGridMedium::new(11, 62, ArcStage::Seen, 4, WirePerm24::from_seed(11));
        step_medium(&mut live_org, &mut w, 5);
        if live_org.snapshot().tick != pre_tick + 5 {
            return Err("--live-chk 5-step tick check failed".to_string());
        }
        live_tag = "live_chk=OK";
    }

    Ok(format!(
        "v5..v10->v{FORMAT_VERSION} upgrade=OK {live_tag}"
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let split = args
        .get(1)
        .filter(|s| !s.starts_with("--"))
        .map_or("dev", String::as_str);
    let open_vault = args.iter().any(|a| a == "--open-vault");
    let mut threads = 8usize;
    let mut opt_live_chk: Option<PathBuf> = None;
    let mut idx = 1usize;
    while idx < args.len() {
        if args[idx] == "--threads"
            && let Some(v) = args.get(idx + 1)
            && let Ok(n) = v.parse::<usize>()
        {
            threads = n.max(1);
            idx += 2;
            continue;
        }
        if args[idx] == "--live-chk"
            && let Some(v) = args.get(idx + 1)
        {
            opt_live_chk = Some(PathBuf::from(v));
            idx += 2;
            continue;
        }
        idx += 1;
    }

    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let seeds: Vec<u64> = match split {
        "dev" => DEFAULT_TWELVE_SEEDS.to_vec(),
        "dev30" => (0..30).collect(),
        "heldout" => {
            if !open_vault || !ws_root.join(".vault_authorized_phase5").exists() {
                eprintln!("Refusing: heldout requires --open-vault and .vault_authorized_phase5.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase5.opened");
            if marker.exists() {
                eprintln!("Refusing: Phase-5 vault already opened.");
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (1100..1130).collect()
        }
        other => {
            eprintln!("Unknown split '{other}' (expected dev, dev30, or heldout)");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-5 Locked Judge (ARC-AGI + SandboxedExec + 52-Task Lifelong) | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    let raw = run_jobs(seeds.len(), threads, |i| run_phase5_seed(seeds[i]));
    let mut rows = Vec::with_capacity(seeds.len());
    for (i, item) in raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Phase-5 seed job {i} panicked");
            return ExitCode::from(1);
        };
        rows.push(r);
    }

    let a_nat: Vec<f64> = rows.iter().map(|r| r.arc_nat).collect();
    let a_frz: Vec<f64> = rows.iter().map(|r| r.arc_frz).collect();
    let a_yok: Vec<f64> = rows.iter().map(|r| r.arc_yok).collect();
    let a_shf: Vec<f64> = rows.iter().map(|r| r.arc_shf).collect();
    let a_ztr: Vec<f64> = rows.iter().map(|r| r.arc_zs_tr).collect();
    let a_znb: Vec<f64> = rows.iter().map(|r| r.arc_zs_nb).collect();
    let a_zls: Vec<f64> = rows.iter().map(|r| r.arc_zs_les).collect();
    let a_zacc: Vec<f64> = rows.iter().map(|r| r.arc_zs_acc).collect();

    let e_nat: Vec<f64> = rows.iter().map(|r| r.exec_nat).collect();
    let e_frz: Vec<f64> = rows.iter().map(|r| r.exec_frz).collect();
    let e_ymx: Vec<f64> = rows.iter().map(|r| r.exec_yok.max(r.exec_shf)).collect();
    let e_ztr: Vec<f64> = rows.iter().map(|r| r.exec_zs_tr).collect();
    let e_znb: Vec<f64> = rows.iter().map(|r| r.exec_zs_nb).collect();
    let e_zls: Vec<f64> = rows.iter().map(|r| r.exec_zs_les).collect();
    let e_zacc: Vec<f64> = rows.iter().map(|r| r.exec_zs_acc).collect();

    let l_val: Vec<f64> = rows.iter().map(|r| r.life_late_val).collect();
    let l_nb: Vec<f64> = rows.iter().map(|r| r.life_late_nb_val).collect();
    let l_les: Vec<f64> = rows.iter().map(|r| r.life_late_les_val).collect();
    let l_acc: Vec<f64> = rows.iter().map(|r| r.life_late_acc).collect();
    let l_fgt: Vec<f64> = rows.iter().map(|r| r.life_forgetting).collect();
    let l_fcap: Vec<f64> = vec![0.02; rows.len()];
    let l_ab_r: Vec<f64> = rows.iter().map(|r| r.life_abab_ret).collect();
    let l_ab_n: Vec<f64> = rows.iter().map(|r| r.life_abab_nb).collect();

    let dst_vec: Vec<f64> = rows.iter().map(|r| r.distractor_share).collect();
    let dst_chance: Vec<f64> = vec![64.0 / 256.0; rows.len()];
    let ag_nat: Vec<f64> = rows.iter().map(|r| r.nat_agency).collect();
    let ag_shf: Vec<f64> = rows.iter().map(|r| r.shf_agency).collect();
    let any_drift = rows.iter().any(|r| r.frz_drift);

    let g1a = conservative_paired_gate(&a_nat, &a_frz);
    let g1b = conservative_paired_gate(&a_nat, &a_yok);
    let g1c = conservative_paired_gate(&a_nat, &a_shf);
    let a_zctrl: Vec<f64> = a_znb.iter().zip(a_zls.iter()).map(|(x, y)| x.max(*y)).collect();
    let g1d = conservative_paired_gate(&a_ztr, &a_zctrl);

    let g2a = conservative_paired_gate(&e_nat, &e_frz);
    let g2b = conservative_paired_gate(&e_nat, &e_ymx);
    let e_zctrl: Vec<f64> = e_znb.iter().zip(e_zls.iter()).map(|(x, y)| x.max(*y)).collect();
    let g2c = conservative_paired_gate(&e_ztr, &e_zctrl);

    let g3a = conservative_paired_gate(&l_val, &l_nb);
    let g3b = conservative_paired_gate(&l_val, &l_les);
    let g4a = conservative_paired_gate(&l_fcap, &l_fgt);
    let g4b = conservative_paired_gate(&l_ab_r, &l_ab_n);
    let g5 = conservative_paired_gate(&dst_chance, &dst_vec);
    let g6 = conservative_paired_gate(&ag_nat, &ag_shf);

    let p_holm = holm_adjust(&[
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g1d.p_two_sided,
        g2a.p_two_sided,
        g2b.p_two_sided,
        g2c.p_two_sided,
        g3a.p_two_sided,
        g3b.p_two_sided,
        g4a.p_two_sided,
        g4b.p_two_sided,
        g5.p_two_sided,
        g6.p_two_sided,
    ]);

    let mean_arc_zs = mean(&a_zacc);
    let mean_exec_zs = mean(&e_zacc);
    let mean_late_acc = mean(&l_acc);
    let mean_fgt = mean(&l_fgt);
    let mean_dst = mean(&dst_vec);
    let mean_shf_ag = mean(&ag_shf);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g1d = g1d.mean_diff > 0.0 && p_holm[3] < 0.05 && mean_arc_zs > 0.65;
    let pass_g2a = g2a.mean_diff > 0.0 && p_holm[4] < 0.05 && mean_exec_zs > 0.65;
    let pass_g2b = g2b.mean_diff > 0.0 && p_holm[5] < 0.05;
    let pass_g2c = g2c.mean_diff > 0.0 && p_holm[6] < 0.05;
    let pass_g3a = g3a.mean_diff > 0.0 && p_holm[7] < 0.05 && mean_late_acc > 0.65;
    let pass_g3b = g3b.mean_diff > 0.0 && p_holm[8] < 0.05;
    let pass_g4a = g4a.mean_diff > 0.0 && p_holm[9] < 0.05 && mean_fgt < 0.02;
    let pass_g4b = g4b.mean_diff > 0.0 && p_holm[10] < 0.05;
    let pass_g5 = g5.mean_diff > 0.0 && p_holm[11] < 0.05 && mean_dst < 0.05;
    let pass_g6 = g6.mean_diff > 0.015 && p_holm[12] < 0.05 && mean_shf_ag <= 0.05;

    println!("\n--- Phase-5 Zero-Leak Machine-Scale Gates (Holm m=13) ---");
    for (name, g, ph, ok) in [
        ("P5-G1a ARC-AGI Nat > Frozen", g1a, p_holm[0], pass_g1a),
        ("P5-G1b ARC-AGI Nat > Yoked", g1b, p_holm[1], pass_g1b),
        ("P5-G1c ARC-AGI Nat > Shuffled", g1c, p_holm[2], pass_g1c),
        ("P5-G1d ARC-AGI ZeroShot > Nb/Les", g1d, p_holm[3], pass_g1d),
        ("P5-G2a SandboxedExec Nat > Frozen", g2a, p_holm[4], pass_g2a),
        ("P5-G2b SandboxedExec Nat > Yok/Shf", g2b, p_holm[5], pass_g2b),
        ("P5-G2c SandboxedExec ZS > Nb/Les", g2c, p_holm[6], pass_g2c),
        ("P5-G3a Life52 Transfer > Newborn", g3a, p_holm[7], pass_g3a),
        ("P5-G3b Life52 Transfer > OPER-Les", g3b, p_holm[8], pass_g3b),
        ("P5-G4a Life52 Retention (<2% Fgt)", g4a, p_holm[9], pass_g4a),
        ("P5-G4b Life52 ABAB Ret-A > Newborn", g4b, p_holm[10], pass_g4b),
        ("P5-G5  Distractor Chance > Nat(<5%)", g5, p_holm[11], pass_g5),
        ("P5-G6  Agency Purity Nat > Shuffled", g6, p_holm[12], pass_g6),
    ] {
        println!(
            "  [{:<4}] {:<38} A={:+.4} B={:+.4} diff={:+.4} wins={}/{} t={:+.2} p_holm={:.4e}",
            if ok { "PASS" } else { "FAIL" },
            name,
            g.mean_a,
            g.mean_b,
            g.mean_diff,
            g.wins,
            g.n,
            g.t,
            ph
        );
    }
    println!(
        "         arc_zs_acc={mean_arc_zs:.4} (>0.65) | exec_zs_acc={mean_exec_zs:.4} (>0.65) | late16_acc={mean_late_acc:.4} (>0.65) | mean_fgt={mean_fgt:.4} (<0.02) | dst_share={mean_dst:.4} (<0.05) | shuf_ag={mean_shf_ag:.4} (<=0.05)"
    );

    println!("\n--- Structural, O(k) Latency & Cross-Version Persistence Invariants ---");
    let (hyg, spl, mir, mod_b) = audit_core_debt(&ws_root);
    let pass_i1 = hyg == 0 && spl == 0 && mir == 0 && mod_b == 0;
    println!(
        "  [{:<4}] P5-I1 Zero Core Debt: hygiene={hyg} split={spl} mirrors={mir} cell_mod={mod_b}",
        if pass_i1 { "PASS" } else { "FAIL" }
    );

    let (pass_i2, i2_msg) = match audit_latency_and_frozen(any_drift) {
        Ok((p99, act, pores, cells)) => {
            let ok = p99 < 2_000.0 && act <= 64 && pores <= 64 && cells >= 512;
            (
                ok,
                format!(
                    "p99={p99:.1} us (<2000) active={act} (<=64) pores={pores} cells={cells} frz=0bit"
                ),
            )
        }
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P5-I2 O(k) Latency & Frozen Drift: {i2_msg}",
        if pass_i2 { "PASS" } else { "FAIL" }
    );

    let (pass_i3, i3_msg) = match audit_phase5_persistence() {
        Ok(b) => (
            b <= 524_288,
            format!("ckpt_bytes={b} mid_ep_bit_id_160=OK prev=OK poison=OK"),
        ),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P5-I3 Mid-Episode Persistence: {i3_msg}",
        if pass_i3 { "PASS" } else { "FAIL" }
    );

    let (pass_i4, i4_msg) = match audit_v10_plus_migration(opt_live_chk.as_deref()) {
        Ok(msg) => (true, msg),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P5-I4 Legacy v5..v10 Upgrade & Live Check: {i4_msg}",
        if pass_i4 { "PASS" } else { "FAIL" }
    );

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g1d
        && pass_g2a
        && pass_g2b
        && pass_g2c
        && pass_g3a
        && pass_g3b
        && pass_g4a
        && pass_g4b
        && pass_g5
        && pass_g6
        && pass_i1
        && pass_i2
        && pass_i3
        && pass_i4;

    println!(
        "\n=== Phase-5 Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
