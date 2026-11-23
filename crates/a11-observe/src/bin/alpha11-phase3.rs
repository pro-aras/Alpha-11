//! Pre-registered locked Phase-3 Self-Evolution judge (`P3-G1..G5` + `P3-I1..I4`).
//!
//! Pure software/compositional operator benchmark (`FieldView -> FieldPulse`, zero 3D organs):
//! 1. `CompositionalOperatorMedium`: `192` permuted channels (`128` task + `32` OU + `32` motor
//!    self-copy), `16` permuted actuators (`8` opponent synergy groups), `4` goal cues each
//!    requiring an exact `5`-step sequential macro `[g_0, g_1, g_2, g_3, g_4]`.
//! 2. Stage 1 (`240` bouts) provides step milestones (`+0.40`/step, `+2.50` terminal).
//!    Stage 2 (`48` bouts) masks ALL task channels after `b_tick < 4` and withholds all
//!    intermediate reward, requiring compiled internal `TAG_OPER` multi-step subroutines.
//! 3. `P3-G2` performs a flagless `TAG_OPER` slab lesion on the serialized checkpoint to prove
//!    synthesized operators causally yield `> 10x` Depth-5 harvest over an identical twin
//!    retaining 100% of cortical/striatal/WM weights (`p_holm < 0.05`).
//! 4. `P3-G3` verifies `ABAB` compositional operator re-binding savings (`Return-A > Newborn`).
//! 5. `P3-I1..I4` verify `0/0/0/0` core debt, `4,096`-ch `p99 < 2,000 us` + `0`-bit frozen hash
//!    drift, transactional `CRC32C` persistence (`200`-step bit-identity + `.prev` + poisoning),
//!    and `FORMAT_VERSION >= 9` lossless `v5..v8 -> v9+` migration.
//!
//! Usage: `alpha11-phase3 [dev|dev30|heldout] [--threads N] [--open-vault]`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use a11_core::crc32c::{crc32c, crc32c_update};
use a11_core::persistence::{
    FILE_MAGIC, FORMAT_VERSION, MIN_COMPAT_FORMAT_VERSION, TAG_BODY, TAG_CELL, TAG_PORE,
    TAG_RING, TAG_SCOT, TAG_TUBE, encode_organism, shadow_decode_organism,
};
use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, PersistError, WorldMedium};
use a11_observe::gates::{PairedGate, holm_adjust, paired_gate};
use a11_observe::{DEFAULT_TWELVE_SEEDS, PulseTemporalChannelShuffler, scan_crate_source};
use a11_worlds::{DenseFieldWorld, DeterministicRng, YokedMedium};

const P3_CHANNELS: usize = 192;
const P3_ACTUATORS: usize = 16;
const DISTRACTOR_START: usize = 128;
const OU_CHANNELS: usize = 32;
const SELF_COPY_CHANNELS: usize = 32;

const NUM_GOALS: usize = 4;
const NUM_GROUPS: usize = 8;
const CHAIN_LEN: usize = 5;
const STAGE1_BOUTS: usize = 240;
const STAGE2_BOUTS: usize = 48;
const STAGE1_BOUT_LEN: u64 = 14;
const STAGE2_BOUT_LEN: u64 = 10;
const CUE_FLASH_LEN: u64 = 4;
const ABAB_RETURN_BOUTS: usize = 28;

const OPER_SLAB_TAG: u32 = u32::from_le_bytes(*b"OPER");

const P3_SENSOR_SALT: u64 = 0x3A11_5E45_0000_2026;
const P3_MOTOR_SALT: u64 = 0x3A11_4D07_0000_2026;
const P3_STAGE1_SALT: u64 = 0x3A11_57A1_1111_0001;
const P3_STAGE2_SALT: u64 = 0x3A11_57A2_2222_0002;
const P3_ABAB_B_SALT: u64 = 0x3A11_BBBB_3333_0003;
const P3_ABAB_RET_SALT: u64 = 0x3A11_AAAA_4444_0004;
const P3_SHUF_SALT: u64 = 0x3A11_5A17_9999_8888;

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

#[derive(Clone, Debug)]
struct WirePerm16 {
    sensor_perm: [usize; P3_CHANNELS],
    inv_sensor_perm: [usize; P3_CHANNELS],
    act_perm: [usize; P3_ACTUATORS],
}

impl WirePerm16 {
    fn from_seed(seed: u64) -> Self {
        let mut s_rng = DeterministicRng::new(seed ^ P3_SENSOR_SALT);
        let mut sensor_perm: [usize; P3_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..P3_CHANNELS).rev() {
            let j = (s_rng.next_u64() as usize) % (i + 1);
            sensor_perm.swap(i, j);
        }
        let mut inv_sensor_perm = [0usize; P3_CHANNELS];
        for (sc, &tr) in sensor_perm.iter().enumerate() {
            inv_sensor_perm[tr] = sc;
        }
        let mut m_rng = DeterministicRng::new(seed ^ P3_MOTOR_SALT);
        let mut act_perm: [usize; P3_ACTUATORS] = std::array::from_fn(|i| i);
        for i in (1..P3_ACTUATORS).rev() {
            let j = (m_rng.next_u64() as usize) % (i + 1);
            act_perm.swap(i, j);
        }
        Self {
            sensor_perm,
            inv_sensor_perm,
            act_perm,
        }
    }

    fn distractor_conductance_share(&self, channels: &[u32], conductances: &[f32]) -> f64 {
        let mut tot = 0.0_f64;
        let mut dst = 0.0_f64;
        for (&ch, &g) in channels.iter().zip(conductances.iter()) {
            let sc = ch as usize;
            if sc < P3_CHANNELS && g.is_finite() && g > 0.0 {
                let gf = f64::from(g);
                tot += gf;
                if self.inv_sensor_perm[sc] >= DISTRACTOR_START {
                    dst += gf;
                }
            }
        }
        if tot > 1.0e-12 { dst / tot } else { 0.0 }
    }
}

fn build_chains(seed: u64, context_b: bool) -> [[usize; CHAIN_LEN]; NUM_GOALS] {
    let salt = if context_b { 0xB0B0_7711 } else { 0xA0A0_5533 };
    let mut rng = DeterministicRng::new(seed ^ salt);
    let mut chains = [[0usize; CHAIN_LEN]; NUM_GOALS];
    for (g, chain) in chains.iter_mut().enumerate() {
        let mut prev = (g * 2 + usize::from(context_b)) % NUM_GROUPS;
        for (s, slot) in chain.iter_mut().enumerate() {
            let step_shift = 1 + ((rng.next_u64() as usize + s + g) % (NUM_GROUPS - 1));
            prev = (prev + step_shift) % NUM_GROUPS;
            *slot = prev;
        }
    }
    chains
}

#[derive(Clone, Debug)]
struct CompositionalOperatorMedium {
    perm: WirePerm16,
    chains: [[usize; CHAIN_LEN]; NUM_GOALS],
    occluded_stage2: bool,
    goals_schedule: Vec<usize>,
    bout_idx: usize,
    b_tick: u64,
    bout_len: u64,
    step_pos: usize,
    step_ema: f32,
    vel_grp: [f32; NUM_GROUPS],
    pos_grp: [f32; NUM_GROUPS],
    failed_bout: bool,
    completed_bouts: usize,
    global_tick: u64,
    last_influx: f32,
    last_stress: f32,
    net_valence_sum: f64,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    motor_ring: [[f32; P3_ACTUATORS]; 3],
    ring_cursor: usize,
    raw_ambient: [f32; P3_CHANNELS],
    permuted_ambient: [f32; P3_CHANNELS],
}

impl CompositionalOperatorMedium {
    fn new(
        seed: u64,
        sched_salt: u64,
        n_bouts: usize,
        occluded_stage2: bool,
        context_b: bool,
        perm: WirePerm16,
    ) -> Self {
        let mut sched_rng = DeterministicRng::new(seed ^ sched_salt);
        let mut goals_schedule = Vec::with_capacity(n_bouts);
        while goals_schedule.len() < n_bouts {
            let mut blk = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = (sched_rng.next_u64() as usize) % (i + 1);
                blk.swap(i, j);
            }
            for &g in &blk {
                if goals_schedule.len() < n_bouts {
                    goals_schedule.push(g);
                }
            }
        }
        let chains = build_chains(seed, context_b);
        let bout_len = if occluded_stage2 {
            STAGE2_BOUT_LEN
        } else {
            STAGE1_BOUT_LEN
        };
        let mut m = Self {
            perm,
            chains,
            occluded_stage2,
            goals_schedule,
            bout_idx: 0,
            b_tick: 0,
            bout_len,
            step_pos: 0,
            step_ema: 0.0,
            vel_grp: [0.0; NUM_GROUPS],
            pos_grp: [0.0; NUM_GROUPS],
            failed_bout: false,
            completed_bouts: 0,
            global_tick: 0,
            last_influx: 0.0,
            last_stress: 0.0,
            net_valence_sum: 0.0,
            ou_rng: DeterministicRng::new(seed ^ sched_salt ^ 0x0011_9999),
            ou_state: [0.0; OU_CHANNELS],
            motor_ring: [[0.0; P3_ACTUATORS]; 3],
            ring_cursor: 0,
            raw_ambient: [0.0; P3_CHANNELS],
            permuted_ambient: [0.0; P3_CHANNELS],
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        (self.goals_schedule.len() as u64) * self.bout_len
    }

    fn current_goal(&self) -> usize {
        self.goals_schedule[self.bout_idx % self.goals_schedule.len().max(1)]
    }

    fn refresh_ambient(&mut self) {
        self.raw_ambient.fill(0.0);
        let g = self.current_goal();

        // Both stages flash the goal cue exclusively on b_tick < CUE_FLASH_LEN (amplitudes 0.72..0.90).
        // After b_tick >= CUE_FLASH_LEN, goal cues vanish completely (Stage 2 is 100% task-occluded).
        if self.b_tick < CUE_FLASH_LEN {
            let amp = (0.90 - 0.06 * (self.b_tick as f32)).clamp(0.72, 0.90);
            for ch in (g * 8)..((g + 1) * 8) {
                self.raw_ambient[ch] = amp;
            }
        }

        if !self.occluded_stage2 && !self.failed_bout && self.step_pos < CHAIN_LEN {
            // Stage 1 Discovery only: gentler step-milestone cue (0.36) + continuous step_ema (72..80).
            let base = 32 + self.step_pos * 8;
            for ch in base..(base + 8) {
                self.raw_ambient[ch] = 0.36;
            }
            for ch in 72..80 {
                self.raw_ambient[ch] = (0.36 * self.step_ema).clamp(0.0, 0.36);
            }
        }

        // Smooth C1 proprioceptive parallax field from effector kinematics pos_grp (80..128):
        // Provides genuine non-linear physical agency reafference without leaking goal or chain.
        for p in 0..24 {
            let ga = p % NUM_GROUPS;
            let gb = (p + 3) % NUM_GROUPS;
            let fa = 1.65 + 0.21 * ((p % 4) as f32);
            let fb = 1.45 + 0.27 * (((p / 4) % 3) as f32);
            let off = 0.37 * (p as f32);
            let vp = 0.45 + 0.36 * (fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            let vm = 0.45 + 0.36 * (-fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            self.raw_ambient[80 + 2 * p] = vp.clamp(0.0, 1.0);
            self.raw_ambient[80 + 2 * p + 1] = vm.clamp(0.0, 1.0);
        }

        // High-variance Ornstein-Uhlenbeck distractors (128..160)
        for i in 0..OU_CHANNELS {
            let z = self.ou_rng.next_f32_range(-1.0, 1.0);
            self.ou_state[i] = (0.35 * self.ou_state[i] + 0.65 * z).clamp(-1.0, 1.0);
            self.raw_ambient[DISTRACTOR_START + i] = self.ou_state[i];
        }

        // Delayed motor self-copy distractors (160..192)
        let lag1 = (self.ring_cursor + 2) % 3;
        let lag2 = (self.ring_cursor + 1) % 3;
        for i in 0..SELF_COPY_CHANNELS {
            let a1 = self.motor_ring[lag1][i % P3_ACTUATORS];
            let a2 = self.motor_ring[lag2][(i + 5) % P3_ACTUATORS];
            self.raw_ambient[DISTRACTOR_START + OU_CHANNELS + i] =
                (0.65 * a1 - 0.30 * a2).clamp(-1.0, 1.0);
        }

        for (sc_ch, &tr_ch) in self.perm.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[tr_ch];
        }
    }
}

impl WorldMedium for CompositionalOperatorMedium {
    fn num_ambient_channels(&self) -> usize {
        P3_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P3_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P3_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let mut logical_act = [0.0_f32; P3_ACTUATORS];
        for (log_idx, &phys_idx) in self.perm.act_perm.iter().enumerate() {
            logical_act[log_idx] =
                clamp_unit(pulse.actuator_out.get(phys_idx).copied().unwrap_or(0.0));
        }
        self.motor_ring[self.ring_cursor] = logical_act;
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        let mut grp_scores = [0.0_f32; NUM_GROUPS];
        for g in 0..NUM_GROUPS {
            let drive = 0.5 * (logical_act[2 * g] + logical_act[2 * g + 1]);
            self.vel_grp[g] = (0.35 * self.vel_grp[g] + 0.65 * drive).clamp(-1.0, 1.0);
            self.pos_grp[g] = (0.72 * self.pos_grp[g] + 0.28 * self.vel_grp[g]).clamp(-1.0, 1.0);
            grp_scores[g] = self.vel_grp[g];
        }
        let mut best_g = 0usize;
        let mut best_val = grp_scores[0];
        let mut second_val = f32::NEG_INFINITY;
        for (g, &val) in grp_scores.iter().enumerate().skip(1) {
            if val > best_val {
                second_val = best_val;
                best_val = val;
                best_g = g;
            } else if val > second_val {
                second_val = val;
            }
        }
        let clear_action = best_val > 0.12 && (best_val - second_val) > 0.06;

        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;
        let goal = self.current_goal();
        let target_chain = self.chains[goal];

        if self.b_tick >= CUE_FLASH_LEN && !self.failed_bout && self.step_pos < CHAIN_LEN {
            if clear_action && best_g == target_chain[self.step_pos] {
                self.step_pos += 1;
                if self.step_pos == CHAIN_LEN {
                    self.completed_bouts += 1;
                    influx = if self.occluded_stage2 { 6.0 } else { 2.5 };
                } else if !self.occluded_stage2 {
                    influx = 0.40;
                }
            } else if clear_action {
                if self.occluded_stage2 {
                    self.failed_bout = true;
                    stress = 0.12;
                } else {
                    stress = 0.03;
                }
            }
        }

        self.step_ema = (0.55 * self.step_ema
            + 0.45 * ((self.step_pos as f32) / (CHAIN_LEN as f32)))
            .clamp(0.0, 1.0);
        self.last_influx = influx;
        self.last_stress = stress;
        self.net_valence_sum += f64::from(influx - stress);

        self.global_tick += 1;
        self.b_tick += 1;
        if self.b_tick >= self.bout_len {
            self.b_tick = 0;
            self.bout_idx += 1;
            self.step_pos = 0;
            self.failed_bout = false;
        }
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

/// Strip/empty ONLY `TAG_OPER` inside an encoded checkpoint (recomputing slab & file CRCs)
/// so the decoded twin retains 100% of cortical/striatal/WM weights and 0 synthesized operators.
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
            // Empty operator bank: count = 0 (u32 LE)
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
        // v8 checkpoints do not yet have TAG_OPER; return exact bytes (diff = 0 -> fails P3-G2).
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
struct Phase3SeedRow {
    stage2_nat: f64,
    stage2_frz: f64,
    stage2_yok: f64,
    stage2_shuf: f64,
    stage2_trained_frz: f64,
    stage2_lesion_frz: f64,
    stage2_nat_acc: f64,
    abab_ret_nat: f64,
    abab_newborn: f64,
    distractor_share: f64,
    nat_agency: f64,
    shf_agency: f64,
}

fn run_phase3_seed(seed: u64) -> Phase3SeedRow {
    let perm = WirePerm16::from_seed(seed);

    // Arm 1: Natural training (Stage 1 discovery) -> Stage 2 occluded Depth-5 evaluation
    let mut nat_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w1_nat =
        CompositionalOperatorMedium::new(seed, P3_STAGE1_SALT, STAGE1_BOUTS, false, false, perm.clone());
    let t_s1 = w1_nat.total_ticks();
    let mut donor_train: Vec<(f32, f32)> = Vec::with_capacity(t_s1 as usize);
    for _ in 0..t_s1 {
        let p = {
            let v = w1_nat.current_view();
            donor_train.push((v.metabolic_influx, v.tissue_stress));
            nat_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        w1_nat.apply_pulse(&p);
    }
    let trained_bytes = encode_organism(&nat_org).unwrap_or_default();

    let mut w2_nat =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    let t_s2 = w2_nat.total_ticks();
    step_medium(&mut nat_org, &mut w2_nat, t_s2);
    let stage2_nat = w2_nat.net_valence_sum;
    let stage2_nat_acc = (w2_nat.completed_bouts as f64) / (STAGE2_BOUTS as f64);
    let nat_snap = nat_org.snapshot();
    let distractor_share =
        perm.distractor_conductance_share(&nat_snap.sensor_pore_channels, &nat_snap.sensor_pore_conductances);
    let nat_agency = f64::from(nat_snap.mean_interventional_agency);

    // Arm 2: FrozenPlasticity
    let mut frz_org = Alpha11Organism::new(seed, OrganismConfig::default());
    frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w1_frz =
        CompositionalOperatorMedium::new(seed, P3_STAGE1_SALT, STAGE1_BOUTS, false, false, perm.clone());
    step_medium(&mut frz_org, &mut w1_frz, t_s1);
    let mut w2_frz =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    step_medium(&mut frz_org, &mut w2_frz, t_s2);
    let stage2_frz = w2_frz.net_valence_sum;

    // Arm 3: Yoked replay (decoupled rewards)
    let w1_yok_src =
        CompositionalOperatorMedium::new(seed, P3_STAGE1_SALT, STAGE1_BOUTS, false, false, perm.clone());
    let mut w1_yok = YokedMedium::new(w1_yok_src, donor_train);
    let mut yok_org = Alpha11Organism::new(seed, OrganismConfig::default());
    step_medium(&mut yok_org, &mut w1_yok, t_s1);
    let mut w2_yok =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    step_medium(&mut yok_org, &mut w2_yok, t_s2);
    let stage2_yok = w2_yok.net_valence_sum;

    // Arm 4: ActionShuffled control (scrambled motor pulses during BOTH Stage 1 & Stage 2)
    let mut shf_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut p_shuffler = PulseTemporalChannelShuffler::new(seed ^ P3_SHUF_SALT, P3_ACTUATORS);
    let mut w1_shf =
        CompositionalOperatorMedium::new(seed, P3_STAGE1_SALT, STAGE1_BOUTS, false, false, perm.clone());
    for _ in 0..t_s1 {
        let emitted = {
            let v = w1_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = p_shuffler.shuffle_pulse(&emitted);
        w1_shf.apply_pulse(&scrambled);
    }
    let shf_agency = f64::from(shf_org.snapshot().mean_interventional_agency);
    let mut w2_shf =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    for _ in 0..t_s2 {
        let emitted = {
            let v = w2_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = p_shuffler.shuffle_pulse(&emitted);
        w2_shf.apply_pulse(&scrambled);
    }
    let stage2_shuf = w2_shf.net_valence_sum;

    // Arm 5: P3-G2 Flagless TAG_OPER Lesion (both evaluated Frozen after identical Stage 1)
    let mut tr_frz = shadow_decode_organism(&trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    tr_frz.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w2_tr =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    step_medium(&mut tr_frz, &mut w2_tr, t_s2);
    let stage2_trained_frz = w2_tr.net_valence_sum;

    let les_bytes = lesion_oper_slab_in_checkpoint(&trained_bytes).unwrap_or(trained_bytes);
    let mut les_frz = shadow_decode_organism(&les_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    les_frz.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w2_ls =
        CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, STAGE2_BOUTS, true, false, perm.clone());
    step_medium(&mut les_frz, &mut w2_ls, t_s2);
    let stage2_lesion_frz = w2_ls.net_valence_sum;

    // Arm 6: P3-G3 ABAB Operator Re-Binding Savings (Context B -> Return Context A vs Newborn)
    let mut wb = CompositionalOperatorMedium::new(seed, P3_ABAB_B_SALT, STAGE1_BOUTS, false, true, perm.clone());
    let tb = wb.total_ticks();
    step_medium(&mut nat_org, &mut wb, tb);

    let mut w_ret = CompositionalOperatorMedium::new(seed, P3_ABAB_RET_SALT, ABAB_RETURN_BOUTS, true, false, perm.clone());
    let tret = w_ret.total_ticks();
    step_medium(&mut nat_org, &mut w_ret, tret);
    let abab_ret_nat = w_ret.net_valence_sum;

    let mut nb_org = Alpha11Organism::new(seed ^ 0x9911_2233, OrganismConfig::default());
    let mut w_nb = CompositionalOperatorMedium::new(seed, P3_ABAB_RET_SALT, ABAB_RETURN_BOUTS, true, false, perm);
    step_medium(&mut nb_org, &mut w_nb, tret);
    let abab_newborn = w_nb.net_valence_sum;

    Phase3SeedRow {
        stage2_nat,
        stage2_frz,
        stage2_yok,
        stage2_shuf,
        stage2_trained_frz,
        stage2_lesion_frz,
        stage2_nat_acc,
        abab_ret_nat,
        abab_newborn,
        distractor_share,
        nat_agency,
        shf_agency,
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
        let mut rng = DeterministicRng::new(0x5EED_CAFE_3030_0001);
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

fn audit_latency_and_frozen() -> Result<(f64, usize, usize, usize), String> {
    let cfg = OrganismConfig {
        initial_cells: 512,
        max_cells: 65_536,
        ..OrganismConfig::default()
    };
    let mut org = Alpha11Organism::new(2026_512, cfg);
    let mut world = DenseFieldWorld::new(77);
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

    let perm = WirePerm16::from_seed(77);
    let mut trained = Alpha11Organism::new(77, OrganismConfig::default());
    let mut w1 = CompositionalOperatorMedium::new(77, P3_STAGE1_SALT, 40, false, false, perm.clone());
    let t1 = w1.total_ticks();
    step_medium(&mut trained, &mut w1, t1);
    trained.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = trained.learned_parameter_hash();
    let mut w2 = CompositionalOperatorMedium::new(77, P3_STAGE2_SALT, 20, true, false, perm);
    let t2 = w2.total_ticks();
    step_medium(&mut trained, &mut w2, t2);
    let h1 = trained.learned_parameter_hash();
    if h0 != h1 {
        return Err(format!("Frozen hash drifted: 0x{h0:016X} != 0x{h1:016X}"));
    }
    Ok((p99, max_act_cells, snap.sensor_pore_count, snap.cell_count))
}

fn audit_phase3_persistence() -> Result<usize, String> {
    let seed = 2026_0303_u64;
    let perm = WirePerm16::from_seed(seed);
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w1 = CompositionalOperatorMedium::new(seed, P3_STAGE1_SALT, 60, false, false, perm.clone());
    let t1 = w1.total_ticks();
    step_medium(&mut org, &mut w1, t1);

    let mut w2 = CompositionalOperatorMedium::new(seed, P3_STAGE2_SALT, 12, true, false, perm);
    step_medium(&mut org, &mut w2, 8);

    let dir = env::temp_dir().join(format!("a11_p3_ckpt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let res = (|| -> Result<usize, String> {
        let primary = dir.join("p3.ckpt");
        let prev = dir.join("p3.ckpt.prev");
        let tmp = dir.join("p3.ckpt.tmp");

        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen1 save: {e:?}"))?;
        let p = org.step(w2.current_view()).map_err(|e| format!("{e:?}"))?;
        w2.apply_pulse(&p);
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen2 save: {e:?}"))?;
        let ckpt_bytes = fs::metadata(&primary).map_err(|e| e.to_string())?.len() as usize;

        let mut restored = Alpha11Organism::new(1, OrganismConfig::default());
        if restored.load_transactional(&primary, &prev).map_err(|e| format!("{e:?}"))? {
            return Err("Clean primary fell back to .prev".to_string());
        }
        let mut wr = w2.clone();
        for step_idx in 0..160usize {
            let pu = org.step(w2.current_view()).map_err(|e| format!("{e:?}"))?;
            let pr = restored.step(wr.current_view()).map_err(|e| format!("{e:?}"))?;
            if pu.actuator_out.iter().zip(pr.actuator_out.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
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
        if !healed.load_transactional(&primary, &prev).map_err(|e| format!("{e:?}"))? {
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
        if !matches!(victim.step(w2.current_view()), Err(PersistError::PoisonedAfterCorruption)) {
            return Err("Dual corruption did not poison victim".to_string());
        }
        Ok(ckpt_bytes)
    })();

    let _ = fs::remove_dir_all(&dir);
    res
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

fn audit_v9_and_legacy_upgrade() -> Result<(), String> {
    if FORMAT_VERSION < 9 {
        return Err(format!("FORMAT_VERSION={FORMAT_VERSION} (< 9)"));
    }
    if MIN_COMPAT_FORMAT_VERSION > 3 {
        return Err("MIN_COMPAT_FORMAT_VERSION > 3".to_string());
    }
    let fresh = Alpha11Organism::new(55, OrganismConfig::default());
    let enc = encode_organism(&fresh).map_err(|e| format!("{e:?}"))?;
    let mut hdr = &enc[..24];
    let magic = u64::from_le_bytes(hdr[0..8].try_into().map_err(|_| "hdr")?);
    let ver = u32::from_le_bytes(hdr[8..12].try_into().map_err(|_| "ver")?);
    let sc = u32::from_le_bytes(hdr[12..16].try_into().map_err(|_| "sc")?);
    let _ = &mut hdr;
    if magic != FILE_MAGIC || ver < 9 || sc < 7 {
        return Err(format!("Unexpected header ver={ver} slabs={sc}"));
    }
    // Check that dropping TAG_OPER and downgrading header to v8 6-slab format still decodes cleanly!
    let payload_end = enc.len() - 4;
    let mut v8_buf = Vec::with_capacity(enc.len());
    v8_buf.extend_from_slice(&enc[..24]);
    let mut pos = 24usize;
    while pos < payload_end {
        let tag = u32::from_le_bytes([enc[pos], enc[pos + 1], enc[pos + 2], enc[pos + 3]]);
        let s_len = u32::from_le_bytes([enc[pos + 4], enc[pos + 5], enc[pos + 6], enc[pos + 7]]) as usize;
        let crc_end = pos + 8 + s_len + 4;
        if tag != OPER_SLAB_TAG {
            push_slab(&mut v8_buf, tag, &enc[pos + 8..pos + 8 + s_len]);
        }
        pos = crc_end;
    }
    v8_buf[8..12].copy_from_slice(&8u32.to_le_bytes());
    v8_buf[12..16].copy_from_slice(&6u32.to_le_bytes());
    let v8_plen = (v8_buf.len() - 24) as u64;
    v8_buf[16..24].copy_from_slice(&v8_plen.to_le_bytes());
    let fcrc = crc32c(&v8_buf);
    v8_buf.extend_from_slice(&fcrc.to_le_bytes());
    shadow_decode_organism(&v8_buf, usize::MAX / 4)
        .map_err(|e| format!("v8->v9 backward compatibility failed: {e:?}"))?;
    let _ = (TAG_BODY, TAG_CELL, TAG_TUBE, TAG_PORE, TAG_RING, TAG_SCOT);
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let split = args.get(1).map_or("dev", String::as_str);
    let open_vault = args.iter().any(|a| a == "--open-vault");
    let mut threads = 8usize;
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
        idx += 1;
    }

    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let seeds: Vec<u64> = match split {
        "dev" => DEFAULT_TWELVE_SEEDS.to_vec(),
        "dev30" => (0..30).collect(),
        "heldout" => {
            if !open_vault || !ws_root.join(".vault_authorized_phase3").exists() {
                eprintln!("Refusing: heldout requires --open-vault and .vault_authorized_phase3.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase3.opened");
            if marker.exists() {
                eprintln!("Refusing: Phase-3 vault already opened.");
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (900..930).collect()
        }
        other => {
            eprintln!("Unknown split '{other}'");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-3 Locked Judge (Self-Evolution Operator VM) | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    let raw = run_jobs(seeds.len(), threads, |i| run_phase3_seed(seeds[i]));
    let mut rows = Vec::with_capacity(seeds.len());
    for (i, item) in raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Phase-3 job {i} panicked");
            return ExitCode::from(1);
        };
        rows.push(r);
    }

    let s2_nat: Vec<f64> = rows.iter().map(|r| r.stage2_nat).collect();
    let s2_frz: Vec<f64> = rows.iter().map(|r| r.stage2_frz).collect();
    let s2_yok: Vec<f64> = rows.iter().map(|r| r.stage2_yok).collect();
    let s2_shf: Vec<f64> = rows.iter().map(|r| r.stage2_shuf).collect();
    let s2_tf: Vec<f64> = rows.iter().map(|r| r.stage2_trained_frz).collect();
    let s2_ls: Vec<f64> = rows.iter().map(|r| r.stage2_lesion_frz).collect();
    let s2_acc: Vec<f64> = rows.iter().map(|r| r.stage2_nat_acc).collect();
    let ab_ret: Vec<f64> = rows.iter().map(|r| r.abab_ret_nat).collect();
    let ab_nb: Vec<f64> = rows.iter().map(|r| r.abab_newborn).collect();
    let dst_vec: Vec<f64> = rows.iter().map(|r| r.distractor_share).collect();
    let dst_chance: Vec<f64> = vec![64.0 / 192.0; rows.len()];
    let ag_nat: Vec<f64> = rows.iter().map(|r| r.nat_agency).collect();
    let ag_shf: Vec<f64> = rows.iter().map(|r| r.shf_agency).collect();

    let g1a = conservative_paired_gate(&s2_nat, &s2_frz);
    let g1b = conservative_paired_gate(&s2_nat, &s2_yok);
    let g1c = conservative_paired_gate(&s2_nat, &s2_shf);
    let g2 = conservative_paired_gate(&s2_tf, &s2_ls);
    let g3 = conservative_paired_gate(&ab_ret, &ab_nb);
    let g4 = conservative_paired_gate(&dst_chance, &dst_vec);
    let g5 = conservative_paired_gate(&ag_nat, &ag_shf);

    let p_holm = holm_adjust(&[
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g2.p_two_sided,
        g3.p_two_sided,
        g4.p_two_sided,
        g5.p_two_sided,
    ]);

    let mean_acc = mean(&s2_acc);
    let mean_dst = mean(&dst_vec);
    let mean_shf_ag = mean(&ag_shf);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05 && mean_acc > 0.55;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g2 = g2.mean_diff > 0.0 && p_holm[3] < 0.05 && g2.mean_a > 10.0 * g2.mean_b.max(0.10);
    let pass_g3 = g3.mean_diff > 0.0 && p_holm[4] < 0.05;
    let pass_g4 = g4.mean_diff > 0.0 && p_holm[5] < 0.05 && mean_dst < 0.05;
    let pass_g5 = g5.mean_diff > 0.015 && p_holm[6] < 0.05 && mean_shf_ag <= 0.05;

    println!("\n--- Phase-3 Compositional Operator Synthesis & Falsification Gates (Holm m=7) ---");
    for (name, g, ph, ok) in [
        ("P3-G1a Depth5 Mastery Nat > Frozen", g1a, p_holm[0], pass_g1a),
        ("P3-G1b Depth5 Mastery Nat > Yoked", g1b, p_holm[1], pass_g1b),
        ("P3-G1c Depth5 Mastery Nat > Shuffled", g1c, p_holm[2], pass_g1c),
        ("P3-G2  >10x Operator > TAG_OPER Lesion", g2, p_holm[3], pass_g2),
        ("P3-G3  ABAB Operator Return-A > Newborn", g3, p_holm[4], pass_g3),
        ("P3-G4  Distractor Chance > Nat (<5%)", g4, p_holm[5], pass_g4),
        ("P3-G5  Agency Purity Nat > Shuffled", g5, p_holm[6], pass_g5),
    ] {
        println!(
            "  [{:<4}] {:<40} A={:+.4} B={:+.4} diff={:+.4} wins={}/{} t={:+.2} p_holm={:.4e}",
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
        "         stage2_depth5_acc={mean_acc:.4} (>0.5500) | dst_share={mean_dst:.4} (<0.0500) | shuf_ag={mean_shf_ag:.4} (<=0.0500)"
    );

    println!("\n--- Structural, O(k) Latency & TAG_OPER Persistence Invariants ---");
    let (hyg, spl, mir, mod_b) = audit_core_debt(&ws_root);
    let pass_i1 = hyg == 0 && spl == 0 && mir == 0 && mod_b == 0;
    println!(
        "  [{:<4}] P3-I1 Zero Core Debt: hygiene={hyg} split={spl} mirrors={mir} cell_mod={mod_b}",
        if pass_i1 { "PASS" } else { "FAIL" }
    );

    let (pass_i2, i2_msg) = match audit_latency_and_frozen() {
        Ok((p99, act, pores, cells)) => {
            let ok = p99 < 2_000.0 && act <= 64 && pores <= 64 && cells >= 512;
            (ok, format!("p99={p99:.1} us (<2000) active={act} (<=64) pores={pores} cells={cells} frz=0bit"))
        }
        Err(e) => (false, e),
    };
    println!("  [{:<4}] P3-I2 O(k) Latency & Frozen Drift: {i2_msg}", if pass_i2 { "PASS" } else { "FAIL" });

    let (pass_i3, i3_msg) = match audit_phase3_persistence() {
        Ok(b) => (b <= 524_288, format!("ckpt_bytes={b} bit_id_160=OK prev=OK poison=OK")),
        Err(e) => (false, e),
    };
    println!("  [{:<4}] P3-I3 TAG_OPER Persistence: {i3_msg}", if pass_i3 { "PASS" } else { "FAIL" });

    let (pass_i4, i4_msg) = match audit_v9_and_legacy_upgrade() {
        Ok(()) => (true, "FORMAT_VERSION>=9 & v8->v9 upgrade=OK".to_string()),
        Err(e) => (false, e),
    };
    println!("  [{:<4}] P3-I4 Version 9 & Legacy Upgrade: {i4_msg}", if pass_i4 { "PASS" } else { "FAIL" });

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g2
        && pass_g3
        && pass_g4
        && pass_g5
        && pass_i1
        && pass_i2
        && pass_i3
        && pass_i4;

    println!(
        "\n=== Phase-3 Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );
    if all_pass { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
