//! Pre-registered locked Phase-2 Pure Brain judge (`P2-G1..G5` + `P2-I1..I4`, 13/13 Brain Gates; 3D organs decoupled to `alpha11-embodied-organs`).
//!
//! Enforces Prefrontal Relational Working Memory (`slot_0..slot_K` content-addressable
//! variable binding), Surprise-Prioritized Hippocampal Episodic Replay (`EVB = Gain * Need`),
//! and Adaptive-Depth System-2 Counterfactual World-Model Search across:
//! 1. `OccludedDetourMedium` (`P2-G1a..d`): `192` permuted channels, `20` permuted actuators,
//!    `18+`-tick complete sensory occlusion (`exceeding 16-tick EfferenceRing`) + mandatory
//!    `0`-reward Stage-1 central bridge unlatch detour before Stage-2 Left/Right alcove harvest,
//!    with 6-arm + flagless lesion null controls (`Natural > Frozen, Yoked, Shuffled, Lesion`).
//! 2. `SymmetricRoleSwapMedium` (`P2-G2a, P2-G2b, P2-G3`): `192` permuted channels, `20` permuted
//!    actuators, symmetric `50%/50%` entity-role assignment across a `6`-tick working-memory delay,
//!    **zero-shot `ControlMode::FrozenPlasticity` generalization to unseen entities `(E_6, E_7)`**
//!    (`proving content-addressable variable binding rather than lookup tables`), and `ABAB`
//!    relational rule reversal (`Match-Role -> NonMatch-Role -> Match-Role`).
//! 3. `P2-G4` & `P2-G5`: `< 5%` distractor pore conductance share & counterfactual agency purity
//!    (`nat_agency - shuf_agency > 0.015`, `shuf_agency <= 0.05`).
//! 4. Embodied 2-agent `OpenSocialHabitat` (`P2-G6` = `Gate 9` Occluded Acoustic Coordination
//!    `Natural > VocalMuted`; `P2-G7` = `Gate 10` Cross-Validated Trajectory/Role-Swap Separation
//!    `Natural > FrozenPlasticity`, `276` channels, `17` muscles).
//! 5. Hard invariants `P2-I1..I4`: Zero core debt (`0/0/0/0`), `M=512`/`4,096`-ch `p99 < 2,000 us`
//!    + `FrozenPlasticity` `0`-bit hash invariance, mid-occlusion transactional paged checkpoint
//!    bit-identity + `.prev` healing + fail-closed poisoning, and lossless `v5/v6/v7 -> v8+`
//!    checkpoint upgrade (`including live --v7-chk`).
//!
//! Usage: `alpha11-phase2 [dev|dev30|dev60|heldout] [--threads N] [--open-vault] [--v7-chk PATH]`

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
use a11_portals::CompositePortalMedium;
use a11_worlds::{BrightRadiantWorld, DenseFieldWorld, DeterministicRng, YokedMedium};

const P2_AMBIENT_CHANNELS: usize = 192;
const P2_ACTUATORS: usize = 20;
const DISTRACTOR_START: usize = 128;
const OU_CHANNELS: usize = 32;
const SELF_COPY_CHANNELS: usize = 32;

const DETOUR_TRAIN_BOUTS: usize = 110;
const DETOUR_EVAL_BOUTS: usize = 40;

const ROLE_NUM_ENTITIES: usize = 8;
const ROLE_TRAIN_ENTITIES: usize = 6;
const ROLE_ENTITY_DIM: usize = 8;
const ROLE_TRAIN_BOUTS: usize = 130;
const ROLE_EVAL_BOUTS: usize = 48;
const ROLE_ZEROSHOT_BOUTS: usize = 48;
const ROLE_REV_B_BOUTS: usize = 48;
const ROLE_REV_RETURN_BOUTS: usize = 16;


const DETOUR_SENSOR_SALT: u64 = 0xD201_5E45_0A9E_2026;
const DETOUR_MOTOR_SALT: u64 = 0xD201_4D07_0A9E_2026;
const DETOUR_TRAIN_SALT: u64 = 0xD201_7A41_4E00_5EED;
const DETOUR_EVAL_SALT: u64 = 0xD201_E7A1_9988_7766;

const ROLE_SENSOR_SALT: u64 = 0xA01E_5E45_0A9E_2026;
const ROLE_MOTOR_SALT: u64 = 0xA01E_4D07_0A9E_2026;
const ROLE_CODE_SALT: u64 = 0xA01E_C0DE_8844_2026;
const ROLE_TRAIN_SALT: u64 = 0xA01E_7A41_4E00_5EED;
const ROLE_EVAL_SALT: u64 = 0xA01E_E7A1_1122_3344;
const ROLE_ZS_SALT: u64 = 0xA01E_2500_6677_8899;
const ROLE_ABAB_SALT: u64 = 0xA01E_ABAB_5544_3322;

const SHUFFLE_SALT_DETOUR: u64 = 0x5A17_D201_20AC_2026;
const SHUFFLE_SALT_ROLE: u64 = 0x5A17_A01E_20AC_2026;

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

/// Seed-specific sensor/motor permutation (`PermutedWire` invariance).
#[derive(Clone, Debug)]
struct WirePerm20 {
    sensor_perm: [usize; P2_AMBIENT_CHANNELS],
    inv_sensor_perm: [usize; P2_AMBIENT_CHANNELS],
    act_perm: [usize; P2_ACTUATORS],
}

impl WirePerm20 {
    fn from_seed(seed: u64, sensor_salt: u64, motor_salt: u64) -> Self {
        let mut s_rng = DeterministicRng::new(seed ^ sensor_salt);
        let mut sensor_perm: [usize; P2_AMBIENT_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..P2_AMBIENT_CHANNELS).rev() {
            let j = (s_rng.next_u64() as usize) % (i + 1);
            sensor_perm.swap(i, j);
        }
        let mut inv_sensor_perm = [0usize; P2_AMBIENT_CHANNELS];
        for (sc_ch, &true_ch) in sensor_perm.iter().enumerate() {
            inv_sensor_perm[sc_ch] = true_ch;
        }

        let mut m_rng = DeterministicRng::new(seed ^ motor_salt);
        let mut act_perm: [usize; P2_ACTUATORS] = std::array::from_fn(|i| i);
        for i in (1..P2_ACTUATORS).rev() {
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
            if sc < P2_AMBIENT_CHANNELS && g.is_finite() && g > 0.0 {
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

/// Builds a balanced `±1.0` schedule in blocks of 8 (`4` negative + `4` positive per block).
fn build_balanced_signs(seed: u64, n_bouts: usize) -> Vec<f32> {
    let mut rng = DeterministicRng::new(seed);
    let mut out = Vec::with_capacity(n_bouts + 8);
    while out.len() < n_bouts {
        let mut blk = [-1.0_f32, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
        for i in (1..8).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            blk.swap(i, j);
        }
        out.extend_from_slice(&blk);
    }
    out.truncate(n_bouts);
    out
}

// =============================================================================
// Suite 1: OccludedDetourMedium (P2-G1a..d)
// 18+ ticks of total visual occlusion (> 16-tick EfferenceRing) + mandatory
// 0-reward Stage-1 central bridge unlatch detour before Stage-2 alcove harvest.
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct DetourBoutSpec {
    goal_side: f32,
    beacon_len: u64,
    harvest_start: u64,
    bout_len: u64,
}

#[derive(Clone, Debug)]
struct OccludedDetourMedium {
    perm: WirePerm20,
    schedule: Vec<DetourBoutSpec>,
    bout_idx: usize,
    b_tick: u64,
    global_tick: u64,
    pos_x: f32,
    pos_y: f32,
    latched: bool,
    latch_ema: f32,
    gate_ema: f32,
    bridge_dwell: u8,
    bout_harvested: bool,
    bout_wrong: bool,
    correct_bouts: usize,
    wrong_bouts: usize,
    unlatched_bouts: usize,
    last_influx: f32,
    last_stress: f32,
    net_valence_sum: f64,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    motor_ring: [[f32; P2_ACTUATORS]; 3],
    ring_cursor: usize,
    raw_ambient: [f32; P2_AMBIENT_CHANNELS],
    permuted_ambient: [f32; P2_AMBIENT_CHANNELS],
}

impl OccludedDetourMedium {
    fn new(seed: u64, world_salt: u64, n_bouts: usize, perm: WirePerm20) -> Self {
        let sched_seed = seed ^ world_salt;
        let signs = build_balanced_signs(sched_seed ^ 0x60A1_51DE, n_bouts.max(1));
        let mut dur_rng = DeterministicRng::new(sched_seed ^ 0xD0A7_1177);
        let mut schedule = Vec::with_capacity(signs.len());
        for &goal_side in &signs {
            let beacon_len = 5 + (dur_rng.next_u64() % 3); // 5..=7 ticks
            let harvest_start = beacon_len + 18 + (dur_rng.next_u64() % 3); // >= 18 ticks occlusion!
            let bout_len = harvest_start + 7 + (dur_rng.next_u64() % 4); // jittered bout length
            schedule.push(DetourBoutSpec {
                goal_side,
                beacon_len,
                harvest_start,
                bout_len,
            });
        }
        let ou_rng = DeterministicRng::new(sched_seed ^ 0x0011_4F15_E777);
        let mut m = Self {
            perm,
            schedule,
            bout_idx: 0,
            b_tick: 0,
            global_tick: 0,
            pos_x: 0.0,
            pos_y: -0.60,
            latched: false,
            latch_ema: 0.0,
            gate_ema: 0.0,
            bridge_dwell: 0,
            bout_harvested: false,
            bout_wrong: false,
            correct_bouts: 0,
            wrong_bouts: 0,
            unlatched_bouts: 0,
            last_influx: 0.0,
            last_stress: 0.0,
            net_valence_sum: 0.0,
            ou_rng,
            ou_state: [0.0; OU_CHANNELS],
            motor_ring: [[0.0; P2_ACTUATORS]; 3],
            ring_cursor: 0,
            raw_ambient: [0.0; P2_AMBIENT_CHANNELS],
            permuted_ambient: [0.0; P2_AMBIENT_CHANNELS],
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        self.schedule.iter().map(|s| s.bout_len).sum()
    }

    #[inline]
    fn current_spec(&self) -> DetourBoutSpec {
        self.schedule[self.bout_idx % self.schedule.len().max(1)]
    }

    fn refresh_ambient(&mut self) {
        self.raw_ambient.fill(0.0);
        let t_f = self.global_tick as f32;
        let spec = self.current_spec();
        let tau = std::f32::consts::TAU;

        // Smooth raised-cosine beacon flash ONLY during b_tick < spec.beacon_len.
        // During b_tick in beacon_len .. bout_len (>= 25 ticks, with >= 18 ticks before harvest_start),
        // channels 0..16 are identically 0.000!
        if self.b_tick < spec.beacon_len {
            let phase_u = ((self.b_tick as f32) + 0.5) / (spec.beacon_len as f32);
            let env = 0.5 * (1.0 - (tau * phase_u).cos());
            for k in 0..8 {
                let carrier = env * (0.64 + 0.22 * (0.31 * t_f + 0.55 * (k as f32)).sin());
                if spec.goal_side < 0.0 {
                    self.raw_ambient[k] = carrier;
                    self.raw_ambient[8 + k] = 0.03 * env;
                } else {
                    self.raw_ambient[k] = 0.03 * env;
                    self.raw_ambient[8 + k] = carrier;
                }
            }
        }

        // Mirror-paired continuous 2D spatial parallax field (16..48, 32 channels):
        // Gives rich continuous action-driven optic flow while remaining 100% Left/Right symmetric (zero goal leak).
        for p in 0..16 {
            let freq_x = 1.6 + 0.22 * ((p % 5) as f32);
            let freq_y = 1.4 + 0.27 * ((p / 4) as f32);
            let ph_off = 0.39 * (p as f32);
            let v_pos = 0.45 + 0.38 * (freq_x * self.pos_x + freq_y * self.pos_y + ph_off).cos();
            let v_mir = 0.45 + 0.38 * (-freq_x * self.pos_x + freq_y * self.pos_y + ph_off).cos();
            self.raw_ambient[16 + 2 * p] = v_pos.clamp(0.0, 1.0);
            self.raw_ambient[16 + 2 * p + 1] = v_mir.clamp(0.0, 1.0);
        }

        // Smooth central bridge latch & alcove gate tactile resonance (48..56, zero goal leak).
        let target_latch = if self.latched { 0.78 } else { 0.0 };
        self.latch_ema = 0.65 * self.latch_ema + 0.35 * target_latch;
        let target_gate = if self.b_tick >= spec.harvest_start {
            0.72
        } else {
            0.10
        };
        self.gate_ema = 0.65 * self.gate_ema + 0.35 * target_gate;
        for k in 0..4 {
            self.raw_ambient[48 + k] = self.latch_ema.clamp(0.0, 1.0);
            self.raw_ambient[52 + k] = self.gate_ema.clamp(0.0, 1.0);
        }

        // Quiet thermal background (56..128)
        for ch in 56..DISTRACTOR_START {
            self.raw_ambient[ch] = 0.02 * (0.017 * t_f + 0.13 * (ch as f32)).sin();
        }

        // High-variance Ornstein-Uhlenbeck Noisy-TV distractors (128..160)
        for i in 0..OU_CHANNELS {
            let z = self.ou_rng.next_f32_range(-1.0, 1.0);
            self.ou_state[i] = (0.32 * self.ou_state[i] + 0.68 * z).clamp(-1.0, 1.0);
            self.raw_ambient[DISTRACTOR_START + i] = self.ou_state[i];
        }

        // Delayed motor self-copy distractors (160..192)
        let lag1 = (self.ring_cursor + 2) % 3;
        let lag2 = (self.ring_cursor + 1) % 3;
        for i in 0..SELF_COPY_CHANNELS {
            let a1 = self.motor_ring[lag1][i % P2_ACTUATORS];
            let a2 = self.motor_ring[lag2][(i + 7) % P2_ACTUATORS];
            self.raw_ambient[DISTRACTOR_START + OU_CHANNELS + i] =
                (0.65 * a1 - 0.30 * a2).clamp(-1.0, 1.0);
        }

        for (sc_ch, &true_ch) in self.perm.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[true_ch];
        }
    }
}

impl WorldMedium for OccludedDetourMedium {
    fn num_ambient_channels(&self) -> usize {
        P2_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P2_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P2_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let mut logical_act = [0.0_f32; P2_ACTUATORS];
        for (log_idx, &phys_idx) in self.perm.act_perm.iter().enumerate() {
            logical_act[log_idx] = clamp_unit(pulse.actuator_out.get(phys_idx).copied().unwrap_or(0.0));
        }
        self.motor_ring[self.ring_cursor] = logical_act;
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        let mean_slice = |s: &[f32]| -> f32 { s.iter().copied().sum::<f32>() / (s.len() as f32) };
        let u_px = mean_slice(&logical_act[0..4]);
        let u_nx = mean_slice(&logical_act[4..8]);
        let u_py = mean_slice(&logical_act[8..12]);
        let u_ny = mean_slice(&logical_act[12..16]);
        let u_latch = mean_slice(&logical_act[16..20]).abs();

        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;
        let spec = self.current_spec();

        let vx = 0.115 * (u_px - u_nx);
        let vy = 0.115 * (u_py - u_ny);
        let mut nx = (self.pos_x + vx).clamp(-0.95, 0.95);
        let mut ny = if self.b_tick < spec.beacon_len {
            // Foyer observation zone before corridor opens
            (self.pos_y + 0.35 * vy).clamp(-0.85, -0.35)
        } else {
            (self.pos_y + vy).clamp(-0.90, 0.92)
        };

        if self.b_tick >= spec.beacon_len {
            // Solid basalt barrier at y = +0.20 until central bridge latch is opened!
            if !self.latched && ny > 0.20 {
                ny = 0.20;
                stress += 0.035;
            }

            // North alcove splitter wall at y > 0.42, |x| < 0.24 prevents sliding horizontally between alcoves.
            if ny > 0.42 && nx.abs() < 0.24 {
                if self.pos_x <= -0.24 {
                    nx = -0.24;
                } else if self.pos_x >= 0.24 {
                    nx = 0.24;
                } else {
                    ny = 0.42;
                }
            }
        }

        self.pos_x = nx;
        self.pos_y = ny;

        if self.b_tick >= spec.beacon_len {
            // Stage-1 Central Bridge Latch (0.0 immediate metabolic_influx — pure instrumental detour!).
            if !self.latched && self.pos_x.abs() <= 0.34 && (-0.06..=0.20).contains(&self.pos_y) {
                self.bridge_dwell = self.bridge_dwell.saturating_add(1);
                if u_latch >= 0.08 || self.bridge_dwell >= 2 {
                    self.latched = true;
                    self.unlatched_bouts = self.unlatched_bouts.saturating_add(1);
                }
            }

            // Stage-2 Occluded Harvest (strictly b_tick >= spec.harvest_start -> >= 18 ticks after beacon!).
            if self.b_tick >= spec.harvest_start && self.latched && self.pos_y >= 0.48 {
                if self.pos_x * spec.goal_side >= 0.28 {
                    influx += 0.46;
                    if !self.bout_harvested && !self.bout_wrong {
                        self.bout_harvested = true;
                        self.correct_bouts = self.correct_bouts.saturating_add(1);
                    }
                } else if self.pos_x * spec.goal_side <= -0.28 {
                    stress += 0.38;
                    if !self.bout_wrong && !self.bout_harvested {
                        self.bout_wrong = true;
                        self.wrong_bouts = self.wrong_bouts.saturating_add(1);
                    }
                }
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.net_valence_sum += f64::from(2.6 * influx - 3.2 * stress);

        self.global_tick = self.global_tick.saturating_add(1);
        self.b_tick = self.b_tick.saturating_add(1);
        if self.b_tick >= spec.bout_len {
            self.b_tick = 0;
            self.bout_idx = self.bout_idx.saturating_add(1);
            self.pos_x = 0.0;
            self.pos_y = -0.60;
            self.latched = false;
            self.bridge_dwell = 0;
            self.bout_harvested = false;
            self.bout_wrong = false;
        }
        self.refresh_ambient();
    }
}

// =============================================================================
// Suite 2: SymmetricRoleSwapMedium (P2-G2a, P2-G2b, P2-G3)
// Symmetric 50/50 entity-role binding across a 6-tick occlusion delay +
// zero-shot FrozenPlasticity transfer to unseen entities (E_6, E_7) + ABAB rule flip.
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct RoleBoutSpec {
    ref_entity: usize,
    other_entity: usize,
    target_role: f32,
    ref_len: u64,
    gap_len: u64,
    bout_len: u64,
}

#[derive(Clone, Debug)]
struct SymmetricRoleSwapMedium {
    perm: WirePerm20,
    entities: [[f32; ROLE_ENTITY_DIM]; ROLE_NUM_ENTITIES],
    schedule: Vec<RoleBoutSpec>,
    rule_sign: f32,
    bout_idx: usize,
    b_tick: u64,
    global_tick: u64,
    bout_align_sum: f32,
    correct_bouts: usize,
    completed_bouts: usize,
    cur_x: f32,
    cur_y: f32,
    last_influx: f32,
    last_stress: f32,
    net_valence_sum: f64,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    motor_ring: [[f32; P2_ACTUATORS]; 3],
    ring_cursor: usize,
    raw_ambient: [f32; P2_AMBIENT_CHANNELS],
    permuted_ambient: [f32; P2_AMBIENT_CHANNELS],
}

fn build_balanced_entities(seed: u64) -> [[f32; ROLE_ENTITY_DIM]; ROLE_NUM_ENTITIES] {
    // Rows 1..=7 of an 8x8 Sylvester-Hadamard matrix + bitwise complement of row 1:
    // Every single entity 0..7 has EXACTLY 4 bright (0.76) and 4 dim (0.14) channels!
    let mut rng = DeterministicRng::new(seed ^ ROLE_CODE_SALT);
    let mut col_perm: [usize; ROLE_ENTITY_DIM] = std::array::from_fn(|i| i);
    for i in (1..ROLE_ENTITY_DIM).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        col_perm.swap(i, j);
    }
    let mut ent_perm: [usize; ROLE_NUM_ENTITIES] = std::array::from_fn(|i| i);
    for i in (1..ROLE_TRAIN_ENTITIES).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        ent_perm.swap(i, j);
    }
    if (rng.next_u64() & 1) == 1 {
        ent_perm.swap(6, 7);
    }

    let mut out = [[0.0_f32; ROLE_ENTITY_DIM]; ROLE_NUM_ENTITIES];
    for e_slot in 0..ROLE_NUM_ENTITIES {
        let h_idx = ent_perm[e_slot];
        for k in 0..ROLE_ENTITY_DIM {
            let c = col_perm[k];
            let bit = if h_idx < 7 {
                ((h_idx + 1) & c).count_ones() & 1 == 0
            } else {
                (1usize & c).count_ones() & 1 != 0
            };
            out[e_slot][k] = if bit { 0.76 } else { 0.14 };
        }
    }
    out
}

fn build_role_schedule(seed: u64, n_bouts: usize, unseen_pair_only: bool) -> Vec<RoleBoutSpec> {
    let mut rng = DeterministicRng::new(seed);
    let roles = build_balanced_signs(seed ^ 0xA01E_BA1A, n_bouts.max(1));
    let mut out = Vec::with_capacity(n_bouts.max(1));
    for (i, &r_sign) in roles.iter().enumerate() {
        let (ref_entity, other_entity) = if unseen_pair_only {
            let swap_pair = ((i / 2) + ((rng.next_u64() & 1) as usize)) & 1 == 1;
            if swap_pair { (7, 6) } else { (6, 7) }
        } else {
            let re = (i + ((rng.next_u64() as usize) % ROLE_TRAIN_ENTITIES)) % ROLE_TRAIN_ENTITIES;
            let offset = 1 + ((rng.next_u64() as usize) % (ROLE_TRAIN_ENTITIES - 1));
            (re, (re + offset) % ROLE_TRAIN_ENTITIES)
        };
        let ref_len = 5 + (rng.next_u64() % 3); // 5..=7 ticks
        let gap_len = 6 + (rng.next_u64() % 3); // 6..=8 ticks total WM silence
        let resp_len = 10 + (rng.next_u64() % 3); // 10..=12 ticks role presentation
        out.push(RoleBoutSpec {
            ref_entity,
            other_entity,
            target_role: r_sign,
            ref_len,
            gap_len,
            bout_len: ref_len + gap_len + resp_len,
        });
    }
    out
}

impl SymmetricRoleSwapMedium {
    fn new(
        seed: u64,
        world_salt: u64,
        n_bouts: usize,
        unseen_pair_only: bool,
        rule_sign: f32,
        perm: WirePerm20,
        entities: [[f32; ROLE_ENTITY_DIM]; ROLE_NUM_ENTITIES],
    ) -> Self {
        let s_seed = seed ^ world_salt;
        let schedule = build_role_schedule(s_seed, n_bouts, unseen_pair_only);
        let ou_rng = DeterministicRng::new(s_seed ^ 0x9988_7766_5544);
        let mut m = Self {
            perm,
            entities,
            schedule,
            rule_sign,
            bout_idx: 0,
            b_tick: 0,
            global_tick: 0,
            bout_align_sum: 0.0,
            correct_bouts: 0,
            completed_bouts: 0,
            cur_x: 0.0,
            cur_y: 0.0,
            last_influx: 0.0,
            last_stress: 0.0,
            net_valence_sum: 0.0,
            ou_rng,
            ou_state: [0.0; OU_CHANNELS],
            motor_ring: [[0.0; P2_ACTUATORS]; 3],
            ring_cursor: 0,
            raw_ambient: [0.0; P2_AMBIENT_CHANNELS],
            permuted_ambient: [0.0; P2_AMBIENT_CHANNELS],
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        self.schedule.iter().map(|s| s.bout_len).sum()
    }

    #[inline]
    fn current_spec(&self) -> RoleBoutSpec {
        self.schedule[self.bout_idx % self.schedule.len().max(1)]
    }

    fn refresh_ambient(&mut self) {
        self.raw_ambient.fill(0.0);
        let t_f = self.global_tick as f32;
        let spec = self.current_spec();
        let tau = std::f32::consts::TAU;
        let role_start = spec.ref_len + spec.gap_len;

        // Smooth raised-cosine envelopes (zero instantaneous square-wave cliffs):
        // b_tick < ref_len: Reference Entity on 0..8
        // ref_len <= b_tick < role_start: Total 0.000 working-memory silence gap
        // b_tick >= role_start: Symmetric Role A (8..16) & Role B (16..24)
        if self.b_tick < spec.ref_len {
            let u = ((self.b_tick as f32) + 0.5) / (spec.ref_len as f32);
            let env = 0.5 * (1.0 - (tau * u).cos());
            for k in 0..ROLE_ENTITY_DIM {
                self.raw_ambient[k] = (self.entities[spec.ref_entity][k] * env).clamp(0.0, 1.0);
            }
        } else if self.b_tick >= role_start {
            let resp_len = (spec.bout_len - role_start).max(1) as f32;
            let u = (((self.b_tick - role_start) as f32) + 0.5) / resp_len;
            let env = 0.5 * (1.0 - (tau * u).cos());
            let (ent_a, ent_b) = if spec.target_role < 0.0 {
                (spec.ref_entity, spec.other_entity)
            } else {
                (spec.other_entity, spec.ref_entity)
            };
            for k in 0..ROLE_ENTITY_DIM {
                self.raw_ambient[8 + k] = (self.entities[ent_a][k] * env).clamp(0.0, 1.0);
                self.raw_ambient[16 + k] = (self.entities[ent_b][k] * env).clamp(0.0, 1.0);
            }
        }

        // Continuous relational rule cue (24..32: Match +1 vs NonMatch -1)
        for k in 0..4 {
            let ph = 0.03 * (0.23 * t_f + (k as f32)).cos();
            if self.rule_sign > 0.0 {
                self.raw_ambient[24 + k] = (0.74 + ph).clamp(0.0, 1.0);
                self.raw_ambient[28 + k] = (0.08 + ph).clamp(0.0, 1.0);
            } else {
                self.raw_ambient[24 + k] = (0.08 + ph).clamp(0.0, 1.0);
                self.raw_ambient[28 + k] = (0.74 + ph).clamp(0.0, 1.0);
            }
        }

        // Mirror-paired continuous 2D effector-cursor parallax field (32..56, 24 channels):
        // Provides continuous action-driven reafference/agency without leaking target_role.
        for p in 0..12 {
            let fx = 1.7 + 0.25 * ((p % 4) as f32);
            let fy = 1.5 + 0.29 * ((p / 4) as f32);
            let off = 0.43 * (p as f32);
            let vp = 0.45 + 0.36 * (fx * self.cur_x + fy * self.cur_y + off).cos();
            let vm = 0.45 + 0.36 * (-fx * self.cur_x + fy * self.cur_y + off).cos();
            self.raw_ambient[32 + 2 * p] = vp.clamp(0.0, 1.0);
            self.raw_ambient[32 + 2 * p + 1] = vm.clamp(0.0, 1.0);
        }

        for ch in 56..DISTRACTOR_START {
            self.raw_ambient[ch] = 0.02 * (0.019 * t_f + 0.11 * (ch as f32)).sin();
        }
        for i in 0..OU_CHANNELS {
            let z = self.ou_rng.next_f32_range(-1.0, 1.0);
            self.ou_state[i] = (0.32 * self.ou_state[i] + 0.68 * z).clamp(-1.0, 1.0);
            self.raw_ambient[DISTRACTOR_START + i] = self.ou_state[i];
        }
        let lag1 = (self.ring_cursor + 2) % 3;
        let lag2 = (self.ring_cursor + 1) % 3;
        for i in 0..SELF_COPY_CHANNELS {
            let a1 = self.motor_ring[lag1][i % P2_ACTUATORS];
            let a2 = self.motor_ring[lag2][(i + 5) % P2_ACTUATORS];
            self.raw_ambient[DISTRACTOR_START + OU_CHANNELS + i] =
                (0.65 * a1 - 0.30 * a2).clamp(-1.0, 1.0);
        }

        for (sc_ch, &true_ch) in self.perm.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[true_ch];
        }
    }
}

impl WorldMedium for SymmetricRoleSwapMedium {
    fn num_ambient_channels(&self) -> usize {
        P2_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P2_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P2_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let mut logical_act = [0.0_f32; P2_ACTUATORS];
        for (log_idx, &phys_idx) in self.perm.act_perm.iter().enumerate() {
            logical_act[log_idx] = clamp_unit(pulse.actuator_out.get(phys_idx).copied().unwrap_or(0.0));
        }
        self.motor_ring[self.ring_cursor] = logical_act;
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        let u_a = logical_act[0..5].iter().copied().sum::<f32>() / 5.0;
        let u_b = logical_act[5..10].iter().copied().sum::<f32>() / 5.0;
        let u_py = logical_act[10..15].iter().copied().sum::<f32>() / 5.0;
        let u_ny = logical_act[15..20].iter().copied().sum::<f32>() / 5.0;
        let u_diff = u_b - u_a;
        self.cur_x = (0.86 * self.cur_x + 0.16 * u_diff).clamp(-1.0, 1.0);
        self.cur_y = (0.86 * self.cur_y + 0.16 * (u_py - u_ny)).clamp(-1.0, 1.0);

        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;
        let spec = self.current_spec();
        let role_start = spec.ref_len + spec.gap_len;

        if self.b_tick >= role_start + 1 {
            let desired_sign = self.rule_sign * spec.target_role;
            let align = u_diff * desired_sign;
            self.bout_align_sum += align;
            if align > 0.06 {
                influx = 0.34 * align.min(1.0);
            } else if align < -0.06 {
                stress = 0.30 * (-align).min(1.0);
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.net_valence_sum += f64::from(2.6 * influx - 3.2 * stress);

        self.global_tick = self.global_tick.saturating_add(1);
        self.b_tick = self.b_tick.saturating_add(1);
        if self.b_tick >= spec.bout_len {
            self.completed_bouts = self.completed_bouts.saturating_add(1);
            if self.bout_align_sum > 0.45 {
                self.correct_bouts = self.correct_bouts.saturating_add(1);
            }
            self.bout_align_sum = 0.0;
            self.cur_x = 0.0;
            self.b_tick = 0;
            self.bout_idx = self.bout_idx.saturating_add(1);
        }
        self.refresh_ambient();
    }
}

// =============================================================================
// Per-Seed Full 6-Arm + Zero-Shot + ABAB Phase-2 Brain Runner
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Phase2SeedRow {
    detour_nat: f64,
    detour_frz: f64,
    detour_yok: f64,
    detour_shuf: f64,
    detour_trained_frz: f64,
    detour_lesion_frz: f64,
    detour_nat_acc: f64,

    role_nat: f64,
    role_yok: f64,
    role_frz: f64,
    role_nat_acc: f64,

    zs_trained_frz: f64,
    zs_newborn_frz: f64,
    zs_lesion_frz: f64,
    zs_trained_acc: f64,

    abab_return_nat: f64,
    abab_newborn: f64,

    distractor_share: f64,
    nat_agency: f64,
    shf_agency: f64,

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

fn run_phase2_seed(seed: u64) -> Phase2SeedRow {
    // --- Part 1: OccludedDetourMedium (P2-G1a..d) ---
    let d_perm = WirePerm20::from_seed(seed, DETOUR_SENSOR_SALT, DETOUR_MOTOR_SALT);

    let mut d_nat_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut d_train_w =
        OccludedDetourMedium::new(seed, DETOUR_TRAIN_SALT, DETOUR_TRAIN_BOUTS, d_perm.clone());
    let d_train_ticks = d_train_w.total_ticks();
    let mut donor_trace: Vec<(f32, f32)> = Vec::with_capacity(d_train_ticks as usize);
    for _ in 0..d_train_ticks {
        let v = d_train_w.current_view();
        donor_trace.push((v.metabolic_influx, v.tissue_stress));
        let p = d_nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        d_train_w.apply_pulse(&p);
    }
    let snap_d_nat = d_nat_org.snapshot();
    let dst_detour = d_perm.distractor_conductance_share(
        &snap_d_nat.sensor_pore_channels,
        &snap_d_nat.sensor_pore_conductances,
    );
    let ag_detour_nat = f64::from(snap_d_nat.mean_interventional_agency);

    // Eval Natural (continued) & FrozenPlasticity Intact vs Flagless Lesion on fresh eval schedule
    let mut d_trained_frz_org = d_nat_org.clone();
    d_trained_frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut d_lesion_org = d_nat_org.clone();
    d_lesion_org.zero_learned_conductances();
    d_lesion_org.set_control_mode(ControlMode::FrozenPlasticity);

    let mut d_eval_nat_w =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    let d_eval_ticks = d_eval_nat_w.total_ticks();
    let mut donor_eval_trace: Vec<(f32, f32)> = Vec::with_capacity(d_eval_ticks as usize);
    for _ in 0..d_eval_ticks {
        let v = d_eval_nat_w.current_view();
        donor_eval_trace.push((v.metabolic_influx, v.tissue_stress));
        let p = d_nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        d_eval_nat_w.apply_pulse(&p);
    }
    let detour_nat = d_eval_nat_w.net_valence_sum;
    let detour_nat_acc = (d_eval_nat_w.correct_bouts as f64) / (DETOUR_EVAL_BOUTS as f64);

    let mut d_eval_tf_w =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    step_medium(&mut d_trained_frz_org, &mut d_eval_tf_w, d_eval_ticks);
    let detour_trained_frz = d_eval_tf_w.net_valence_sum;

    let mut d_eval_les_w =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    step_medium(&mut d_lesion_org, &mut d_eval_les_w, d_eval_ticks);
    let detour_lesion_frz = d_eval_les_w.net_valence_sum;

    // Newborn FrozenPlasticity control
    let mut d_frz_org = Alpha11Organism::new(seed, OrganismConfig::default());
    d_frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut d_frz_train_w =
        OccludedDetourMedium::new(seed, DETOUR_TRAIN_SALT, DETOUR_TRAIN_BOUTS, d_perm.clone());
    step_medium(&mut d_frz_org, &mut d_frz_train_w, d_train_ticks);
    let mut d_frz_eval_w =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    step_medium(&mut d_frz_org, &mut d_frz_eval_w, d_eval_ticks);
    let detour_frz = d_frz_eval_w.net_valence_sum;

    // YokedReward control (distinct birth RNG seed so recipient trajectory is decoupled from donor)
    let mut d_yok_org = Alpha11Organism::new(seed.wrapping_add(500_000), OrganismConfig::default());
    let d_yok_inner =
        OccludedDetourMedium::new(seed, DETOUR_TRAIN_SALT, DETOUR_TRAIN_BOUTS, d_perm.clone());
    let mut d_yok_train_w = YokedMedium::new(d_yok_inner, donor_trace);
    step_medium(&mut d_yok_org, &mut d_yok_train_w, d_train_ticks);
    let d_yok_eval_inner =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    let mut d_yok_eval_w = YokedMedium::new(d_yok_eval_inner, donor_eval_trace);
    step_medium(&mut d_yok_org, &mut d_yok_eval_w, d_eval_ticks);
    let detour_yok = d_yok_eval_w.inner().net_valence_sum;

    // ActionShuffled control
    let mut d_shf_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut d_shf_train_w =
        OccludedDetourMedium::new(seed, DETOUR_TRAIN_SALT, DETOUR_TRAIN_BOUTS, d_perm.clone());
    let mut d_shuffler =
        PulseTemporalChannelShuffler::new(seed ^ SHUFFLE_SALT_DETOUR, P2_ACTUATORS);
    for _ in 0..d_train_ticks {
        let emitted = {
            let v = d_shf_train_w.current_view();
            d_shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = d_shuffler.shuffle_pulse(&emitted);
        d_shf_train_w.apply_pulse(&scrambled);
    }
    let ag_detour_shf = f64::from(d_shf_org.snapshot().mean_interventional_agency);
    let mut d_shf_eval_w =
        OccludedDetourMedium::new(seed, DETOUR_EVAL_SALT, DETOUR_EVAL_BOUTS, d_perm.clone());
    for _ in 0..d_eval_ticks {
        let emitted = {
            let v = d_shf_eval_w.current_view();
            d_shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = d_shuffler.shuffle_pulse(&emitted);
        d_shf_eval_w.apply_pulse(&scrambled);
    }
    let detour_shuf = d_shf_eval_w.net_valence_sum;

    // --- Part 2: SymmetricRoleSwapMedium (P2-G2a, P2-G2b Zero-Shot Unseen E_6/E_7, P2-G3 ABAB) ---
    let r_seed = seed ^ 0xA01E_0002_2026_0002;
    let r_perm = WirePerm20::from_seed(r_seed, ROLE_SENSOR_SALT, ROLE_MOTOR_SALT);
    let entities = build_balanced_entities(r_seed);

    let mut r_nat_org = Alpha11Organism::new(r_seed, OrganismConfig::default());
    let mut r_train_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_TRAIN_SALT,
        ROLE_TRAIN_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let r_train_ticks = r_train_w.total_ticks();
    let mut r_donor_train: Vec<(f32, f32)> = Vec::with_capacity(r_train_ticks as usize);
    for _ in 0..r_train_ticks {
        let v = r_train_w.current_view();
        r_donor_train.push((v.metabolic_influx, v.tissue_stress));
        let p = r_nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        r_train_w.apply_pulse(&p);
    }
    let snap_r_nat = r_nat_org.snapshot();
    let dst_role = r_perm.distractor_conductance_share(
        &snap_r_nat.sensor_pore_channels,
        &snap_r_nat.sensor_pore_conductances,
    );
    let ag_role_nat = f64::from(snap_r_nat.mean_interventional_agency);

    // P2-G2b: Strictly FrozenPlasticity zero-shot generalization to unseen novel entities (E_6, E_7)!
    let mut r_zs_trained = r_nat_org.clone();
    r_zs_trained.set_control_mode(ControlMode::FrozenPlasticity);
    let mut r_zs_lesion = r_nat_org.clone();
    r_zs_lesion.zero_learned_conductances();
    r_zs_lesion.set_control_mode(ControlMode::FrozenPlasticity);
    let mut r_zs_newborn = Alpha11Organism::new(r_seed, OrganismConfig::default());
    r_zs_newborn.set_control_mode(ControlMode::FrozenPlasticity);

    let mut zs_w_tr = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ZS_SALT,
        ROLE_ZEROSHOT_BOUTS,
        true,
        1.0,
        r_perm.clone(),
        entities,
    );
    let r_zs_ticks = zs_w_tr.total_ticks();
    step_medium(&mut r_zs_trained, &mut zs_w_tr, r_zs_ticks);
    let zs_trained_frz = zs_w_tr.net_valence_sum;
    let zs_trained_acc = (zs_w_tr.correct_bouts as f64) / (ROLE_ZEROSHOT_BOUTS as f64);

    let mut zs_w_nb = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ZS_SALT,
        ROLE_ZEROSHOT_BOUTS,
        true,
        1.0,
        r_perm.clone(),
        entities,
    );
    step_medium(&mut r_zs_newborn, &mut zs_w_nb, r_zs_ticks);
    let zs_newborn_frz = zs_w_nb.net_valence_sum;

    let mut zs_w_les = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ZS_SALT,
        ROLE_ZEROSHOT_BOUTS,
        true,
        1.0,
        r_perm.clone(),
        entities,
    );
    step_medium(&mut r_zs_lesion, &mut zs_w_les, r_zs_ticks);
    let zs_lesion_frz = zs_w_les.net_valence_sum;

    // P2-G2a: Symmetric role-swap evaluation (Natural vs Yoked & Frozen)
    let mut r_eval_nat_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_EVAL_SALT,
        ROLE_EVAL_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let r_eval_ticks = r_eval_nat_w.total_ticks();
    let mut r_donor_eval: Vec<(f32, f32)> = Vec::with_capacity(r_eval_ticks as usize);
    for _ in 0..r_eval_ticks {
        let v = r_eval_nat_w.current_view();
        r_donor_eval.push((v.metabolic_influx, v.tissue_stress));
        let p = r_nat_org
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
        r_eval_nat_w.apply_pulse(&p);
    }
    let role_nat = r_eval_nat_w.net_valence_sum;
    let role_nat_acc = (r_eval_nat_w.correct_bouts as f64) / (ROLE_EVAL_BOUTS as f64);

    let mut r_yok_org =
        Alpha11Organism::new(r_seed.wrapping_add(500_000), OrganismConfig::default());
    let r_yok_tr_in = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_TRAIN_SALT,
        ROLE_TRAIN_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let mut r_yok_tr = YokedMedium::new(r_yok_tr_in, r_donor_train);
    step_medium(&mut r_yok_org, &mut r_yok_tr, r_train_ticks);
    let r_yok_ev_in = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_EVAL_SALT,
        ROLE_EVAL_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let mut r_yok_ev = YokedMedium::new(r_yok_ev_in, r_donor_eval);
    step_medium(&mut r_yok_org, &mut r_yok_ev, r_eval_ticks);
    let role_yok = r_yok_ev.inner().net_valence_sum;

    let mut r_frz_org = Alpha11Organism::new(r_seed, OrganismConfig::default());
    r_frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut r_frz_ev = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_EVAL_SALT,
        ROLE_EVAL_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    step_medium(&mut r_frz_org, &mut r_frz_ev, r_eval_ticks);
    let role_frz = r_frz_ev.net_valence_sum;

    // ActionShuffled agency check on SymmetricRoleSwapMedium
    let mut r_shf_org = Alpha11Organism::new(r_seed, OrganismConfig::default());
    let mut r_shf_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_TRAIN_SALT,
        ROLE_TRAIN_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let mut r_shuffler = PulseTemporalChannelShuffler::new(r_seed ^ SHUFFLE_SALT_ROLE, P2_ACTUATORS);
    for _ in 0..r_train_ticks {
        let emitted = {
            let v = r_shf_w.current_view();
            r_shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = r_shuffler.shuffle_pulse(&emitted);
        r_shf_w.apply_pulse(&scrambled);
    }
    let ag_role_shf = f64::from(r_shf_org.snapshot().mean_interventional_agency);

    // P2-G3: ABAB Relational Rule Reversal (B = NonMatch -1.0 -> Return-A = Match +1.0 vs Plastic Newborn)
    let mut r_b_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ABAB_SALT ^ 0xBBBB,
        ROLE_REV_B_BOUTS,
        false,
        -1.0,
        r_perm.clone(),
        entities,
    );
    let r_b_ticks = r_b_w.total_ticks();
    step_medium(&mut r_nat_org, &mut r_b_w, r_b_ticks);

    let mut r_ret_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ABAB_SALT ^ 0xAAAA,
        ROLE_REV_RETURN_BOUTS,
        false,
        1.0,
        r_perm.clone(),
        entities,
    );
    let ret_ticks = r_ret_w.total_ticks();
    step_medium(&mut r_nat_org, &mut r_ret_w, ret_ticks);
    let abab_return_nat = r_ret_w.net_valence_sum;

    let mut r_nb_plastic = Alpha11Organism::new(r_seed ^ 0x4E7B_0001, OrganismConfig::default());
    let mut r_nb_ret_w = SymmetricRoleSwapMedium::new(
        r_seed,
        ROLE_ABAB_SALT ^ 0xAAAA,
        ROLE_REV_RETURN_BOUTS,
        false,
        1.0,
        r_perm,
        entities,
    );
    step_medium(&mut r_nb_plastic, &mut r_nb_ret_w, ret_ticks);
    let abab_newborn = r_nb_ret_w.net_valence_sum;

    Phase2SeedRow {
        detour_nat,
        detour_frz,
        detour_yok,
        detour_shuf,
        detour_trained_frz,
        detour_lesion_frz,
        detour_nat_acc,
        role_nat,
        role_yok,
        role_frz,
        role_nat_acc,
        zs_trained_frz,
        zs_newborn_frz,
        zs_lesion_frz,
        zs_trained_acc,
        abab_return_nat,
        abab_newborn,
        distractor_share: 0.5 * (dst_detour + dst_role),
        nat_agency: 0.5 * (ag_detour_nat + ag_role_nat),
        shf_agency: 0.5 * (ag_detour_shf + ag_role_shf),
    }
}

// =============================================================================
// Exact Paired Sign-Flip Permutation + Student-t (Conservative Maximum p-Value)
// =============================================================================

fn sign_flip_two_sided_p(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let mut diffs = Vec::with_capacity(n);
    let mut sum = 0.0_f64;
    for i in 0..n {
        let d = a[i] - b[i];
        if d.is_finite() {
            diffs.push(d);
            sum += d;
        }
    }
    let n = diffs.len();
    if n == 0 {
        return 1.0;
    }
    let nf = n as f64;
    let obs = (sum / nf).abs();
    if obs == 0.0 {
        return 1.0;
    }
    if n <= 20 {
        let total = 1usize << n;
        let mut ge = 0usize;
        for mask in 0..total {
            let mut s = 0.0_f64;
            for (i, &d) in diffs.iter().enumerate() {
                if ((mask >> i) & 1) == 0 {
                    s += d;
                } else {
                    s -= d;
                }
            }
            if (s / nf).abs() >= obs - 1.0e-15 {
                ge += 1;
            }
        }
        return (ge as f64) / (total as f64);
    }
    let mc = 20_000usize;
    let mut rng = DeterministicRng::new(0xA112_516F_5EED_2026);
    let mut ge = 1usize;
    for _ in 0..mc {
        let mut s = 0.0_f64;
        for &d in &diffs {
            if (rng.next_u64() & 1) == 0 {
                s += d;
            } else {
                s -= d;
            }
        }
        if (s / nf).abs() >= obs - 1.0e-15 {
            ge += 1;
        }
    }
    (ge as f64) / ((mc + 1) as f64)
}

fn conservative_paired_gate(a: &[f64], b: &[f64]) -> PairedGate {
    let mut g = paired_gate(a, b);
    let p_perm = sign_flip_two_sided_p(a, b);
    g.p_two_sided = g.p_two_sided.max(p_perm);
    g
}

// =============================================================================
// P2-I1: Core Source Debt Audit (0 hygiene, 0 split-brain, 0 mirrors, 0 % 64)
// =============================================================================

#[derive(Default, Debug)]
struct CoreDebtReport {
    hygiene_violations: usize,
    split_brain_sites: Vec<String>,
    organ_mirror_sites: Vec<String>,
    cell_modulo_sites: Vec<String>,
}

fn strip_cfg_test(src: &str) -> String {
    if let Some(idx) = src.find("#[cfg(test)]") {
        src[..idx].to_string()
    } else {
        src.to_string()
    }
}

fn audit_core_debt(ws_root: &Path) -> CoreDebtReport {
    let core_src = ws_root.join("crates/a11-core/src");
    let mut rep = CoreDebtReport::default();
    if let Ok(hyg) = scan_crate_source(&core_src) {
        rep.hygiene_violations = hyg.findings.len();
    }

    let split_needles = [
        "num_actuators <= 8",
        "num_actuators > 8",
        "num_actuators < 8",
        "num_actuators >= 8",
        "n_act_ag <= 8",
        "n_act_ag > 8",
        "m_organs <= 8",
        "m_organs > 8",
        "STRIATAL_ACTUATORS",
    ];
    let mirror_needles = [
        "deconv_hair_cell_35",
        "sym_deconv_ild",
        "laryngeal_voiced_lags",
        "0.66615",
        "0.07312",
        "2.4572",
        "75.0_f32",
        "-0.38..=-0.16",
        "-0.356..=-0.16",
        "-0.345..=-0.04",
    ];

    let Ok(entries) = fs::read_dir(&core_src) else {
        rep.hygiene_violations += 1;
        return rep;
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "rs"))
        .collect();
    files.sort();

    for path in files {
        let fname = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.rs");
        if fname == "causal_core.rs" {
            continue;
        }
        let Ok(raw) = fs::read_to_string(&path) else {
            continue;
        };
        let prod = strip_cfg_test(&raw);
        for (idx, line) in prod.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") {
                continue;
            }
            let loc = format!("{fname}:{}", idx + 1);
            if split_needles.iter().any(|n| trimmed.contains(n)) {
                rep.split_brain_sites.push(format!("{loc}: {trimmed}"));
            }
            if mirror_needles.iter().any(|n| trimmed.contains(n)) {
                rep.organ_mirror_sites.push(format!("{loc}: {trimmed}"));
            }
            if trimmed.contains("% PRED_BUCKETS") {
                rep.cell_modulo_sites.push(format!("{loc}: {trimmed}"));
            }
        }
    }
    rep
}

// =============================================================================
// P2-I2: O(k) Latency at M=512 / 4,096ch + FrozenPlasticity 0-Bit Hash Invariance
// =============================================================================

fn audit_latency_and_frozen_invariance() -> Result<(f64, usize, usize, usize), String> {
    let cfg = OrganismConfig {
        initial_cells: 512,
        max_cells: 65_536,
        ..OrganismConfig::default()
    };
    let mut org = Alpha11Organism::new(2026_512, cfg);
    let mut world = DenseFieldWorld::new(77);
    let steps = 1_000usize;
    let mut us = Vec::with_capacity(steps);
    let mut max_active_cells = 0usize;
    for _ in 0..steps {
        let pulse = {
            let v = world.current_view();
            let t0 = Instant::now();
            let p = org
                .step(v)
                .map_err(|e| format!("Dense4096 step failed: {e:?}"))?;
            us.push(t0.elapsed().as_secs_f64() * 1.0e6);
            p
        };
        let snap = org.snapshot();
        max_active_cells = max_active_cells.max(snap.active_cells_last_tick);
        world.apply_pulse(&pulse);
    }
    us.sort_by(|a, b| a.total_cmp(b));
    let p99 = us[((steps * 99) / 100).min(steps - 1)];
    let final_snap = org.snapshot();

    // Verify 0-bit learned_parameter_hash drift under FrozenPlasticity on BOTH new Phase-2 arenas
    // (both from birth AND after 600 ticks of Natural maturation!).
    let perm = WirePerm20::from_seed(909, DETOUR_SENSOR_SALT, DETOUR_MOTOR_SALT);
    let mut trained = Alpha11Organism::new(909, OrganismConfig::default());
    let mut w_det = OccludedDetourMedium::new(909, DETOUR_TRAIN_SALT, 30, perm.clone());
    step_medium(&mut trained, &mut w_det, 600);
    trained.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = trained.learned_parameter_hash();
    step_medium(&mut trained, &mut w_det, 300);
    let h1 = trained.learned_parameter_hash();
    if h0 != h1 {
        return Err(format!(
            "FrozenPlasticity mutated learned_parameter_hash on OccludedDetourMedium (0x{h0:016X} != 0x{h1:016X})"
        ));
    }

    let entities = build_balanced_entities(909);
    let mut w_role = SymmetricRoleSwapMedium::new(
        909,
        ROLE_ZS_SALT,
        20,
        true,
        1.0,
        perm,
        entities,
    );
    step_medium(&mut trained, &mut w_role, 300);
    let h2 = trained.learned_parameter_hash();
    if h0 != h2 {
        return Err(format!(
            "FrozenPlasticity mutated learned_parameter_hash on SymmetricRoleSwapMedium (0x{h0:016X} != 0x{h2:016X})"
        ));
    }

    Ok((
        p99,
        max_active_cells,
        final_snap.sensor_pore_count,
        final_snap.cell_count,
    ))
}

// =============================================================================
// P2-I3: Mid-Occlusion Transactional Paged Checkpoint Bit-Identity, .prev Heal & Poisoning
// =============================================================================

fn audit_mid_occlusion_persistence() -> Result<usize, String> {
    let seed = 2026_0202_u64;
    let perm = WirePerm20::from_seed(seed, DETOUR_SENSOR_SALT, DETOUR_MOTOR_SALT);
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w = OccludedDetourMedium::new(seed, DETOUR_TRAIN_SALT, 60, perm);

    // Advance 35 bouts and stop strictly mid-occlusion (b_tick == 12 of bout 35).
    while w.bout_idx < 35 || w.b_tick < 12 {
        let p = org
            .step(w.current_view())
            .map_err(|e| format!("pre-ckpt step failed: {e:?}"))?;
        w.apply_pulse(&p);
    }

    let dir = env::temp_dir().join(format!("a11_p2_ckpt_audit_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| format!("create temp dir failed: {e}"))?;

    let res = (|| -> Result<usize, String> {
        let primary = dir.join("organism.ckpt");
        let prev = dir.join("organism.ckpt.prev");
        let tmp = dir.join("organism.ckpt.tmp");

        // Gen 1 save
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen-1 save failed: {e:?}"))?;
        // Step once (b_tick == 13, still mid-occlusion!) and Gen 2 save -> populates both primary and prev
        let p1 = org
            .step(w.current_view())
            .map_err(|e| format!("gen-2 step failed: {e:?}"))?;
        w.apply_pulse(&p1);
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen-2 save failed: {e:?}"))?;

        let ckpt_bytes = fs::metadata(&primary)
            .map_err(|e| format!("stat primary failed: {e}"))?
            .len() as usize;

        // 1. Exact 200-step mid-occlusion trajectory bit-identity
        let mut restored = Alpha11Organism::new(1, OrganismConfig::default());
        let used_prev = restored
            .load_transactional(&primary, &prev)
            .map_err(|e| format!("load_transactional failed: {e:?}"))?;
        if used_prev {
            return Err("clean primary unexpectedly fell back to .prev".to_string());
        }
        let mut w_ref = w.clone();
        let mut w_rst = w.clone();
        for step_idx in 0..200usize {
            let pu = org
                .step(w_ref.current_view())
                .map_err(|e| format!("uninterrupted step {step_idx} failed: {e:?}"))?;
            let pr = restored
                .step(w_rst.current_view())
                .map_err(|e| format!("restored step {step_idx} failed: {e:?}"))?;
            if pu.actuator_out.len() != pr.actuator_out.len()
                || pu
                    .actuator_out
                    .iter()
                    .zip(pr.actuator_out.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(format!(
                    "mid-occlusion restored trajectory diverged at step {step_idx} (WM / replay state not persisted losslessly)"
                ));
            }
            w_ref.apply_pulse(&pu);
            w_rst.apply_pulse(&pr);
        }
        let enc_u = org
            .encode_checkpoint()
            .map_err(|e| format!("encode uninterrupted failed: {e:?}"))?;
        let enc_r = restored
            .encode_checkpoint()
            .map_err(|e| format!("encode restored failed: {e:?}"))?;
        if enc_u != enc_r {
            return Err("post-200-step encoded checkpoints differ".to_string());
        }

        // 2. Corrupt primary -> must heal from .prev (Ok(true))
        let mut bad_primary = fs::read(&primary).map_err(|e| e.to_string())?;
        let mid = bad_primary.len() / 2;
        bad_primary[mid] ^= 0x5A;
        fs::write(&primary, &bad_primary).map_err(|e| e.to_string())?;
        let mut healed = Alpha11Organism::new(2, OrganismConfig::default());
        let fell_back = healed
            .load_transactional(&primary, &prev)
            .map_err(|e| format!(".prev fallback failed: {e:?}"))?;
        if !fell_back || healed.is_poisoned() {
            return Err("corrupted primary did not heal cleanly from .prev".to_string());
        }

        // 3. Corrupt both primary and .prev -> must fail closed and poison organism
        let mut bad_prev = fs::read(&prev).map_err(|e| e.to_string())?;
        let pmid = bad_prev.len() / 2;
        bad_prev[pmid] ^= 0xA5;
        fs::write(&prev, &bad_prev).map_err(|e| e.to_string())?;
        let mut victim = Alpha11Organism::new(3, OrganismConfig::default());
        if victim.load_transactional(&primary, &prev).is_ok() {
            return Err("dual corruption unexpectedly succeeded".to_string());
        }
        if !matches!(
            victim.step(w.current_view()),
            Err(PersistError::PoisonedAfterCorruption)
        ) {
            return Err("dual corruption did not poison victim organism".to_string());
        }

        Ok(ckpt_bytes)
    })();

    let _ = fs::remove_dir_all(&dir);
    res
}

// =============================================================================
// P2-I4: Lossless FORMAT_VERSION >= 8 Upgrade from Canonical v5, v6, v7 & Live --v7-chk
// =============================================================================

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn push_slab(out: &mut Vec<u8>, tag: u32, payload: &[u8]) {
    let tag_b = tag.to_le_bytes();
    let len_b = (payload.len() as u32).to_le_bytes();
    let mut crc = 0xFFFF_FFFF;
    crc = crc32c_update(crc, &tag_b);
    crc = crc32c_update(crc, &len_b);
    crc = crc32c_update(crc, payload);
    let slab_crc = !crc;
    out.extend_from_slice(&tag_b);
    out.extend_from_slice(&len_b);
    out.extend_from_slice(payload);
    out.extend_from_slice(&slab_crc.to_le_bytes());
}

fn build_canonical_checkpoint(ver: u32) -> Vec<u8> {
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
    for _ in 0..(EXPLORE * 3) {
        push_f32(&mut body, 0.0);
    }
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

    let f32_zeros = |dst: &mut Vec<u8>, cnt: usize| {
        for _ in 0..cnt {
            push_f32(dst, 0.0);
        }
    };

    let mut tube = Vec::new();
    push_u32(&mut tube, 2);
    push_u32(&mut tube, 2);
    if ver >= 7 {
        tube.push(0); // external_slabs = false
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

fn audit_v8_migration(opt_v7_path: Option<&Path>) -> Result<(), String> {
    if MIN_COMPAT_FORMAT_VERSION > 5 {
        return Err(format!(
            "MIN_COMPAT_FORMAT_VERSION is {MIN_COMPAT_FORMAT_VERSION} (must remain <= 5)"
        ));
    }
    let max_b = OrganismConfig::default().max_state_bytes;
    for ver in [5_u32, 6_u32, 7_u32] {
        let legacy_bytes = build_canonical_checkpoint(ver);
        let mut org = shadow_decode_organism(&legacy_bytes, max_b)
            .map_err(|e| format!("canonical v{ver} decode failed: {e:?}"))?;
        let snap = org.snapshot();
        if snap.tick != 777 || (snap.atp - 42.5).abs() > 1.0e-3 || snap.cell_count != 4 {
            return Err(format!(
                "canonical v{ver} state mismatch: tick={} atp={} cells={}",
                snap.tick, snap.atp, snap.cell_count
            ));
        }
        let mut w = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(42), 42);
        for _ in 0..20 {
            let p = org
                .step(w.current_view())
                .map_err(|e| format!("upgraded v{ver} step failed: {e:?}"))?;
            w.apply_pulse(&p);
        }
        let upgraded = encode_organism(&org)
            .map_err(|e| format!("upgraded v{ver}->v8+ encode failed: {e:?}"))?;
        let encoded_ver =
            u32::from_le_bytes([upgraded[8], upgraded[9], upgraded[10], upgraded[11]]);
        if encoded_ver != FORMAT_VERSION {
            return Err(format!(
                "re-encoded version {encoded_ver} != {FORMAT_VERSION}"
            ));
        }
        let _ = shadow_decode_organism(&upgraded, max_b)
            .map_err(|e| format!("upgraded v{ver}->v8+ roundtrip decode failed: {e:?}"))?;
    }

    if let Some(p) = opt_v7_path {
        let disk_bytes =
            fs::read(p).map_err(|e| format!("failed reading --v7-chk {}: {e}", p.display()))?;
        let mut live_org = shadow_decode_organism(&disk_bytes, max_b)
            .map_err(|e| format!("live --v7-chk decode failed: {e:?}"))?;
        let pre_tick = live_org.snapshot().tick;
        let w = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(11), 11);
        let pulse = live_org
            .step(w.current_view())
            .map_err(|e| format!("live --v7-chk post-upgrade step failed: {e:?}"))?;
        if pulse.actuator_out.len() != 27 || live_org.snapshot().tick != pre_tick + 1 {
            return Err("live --v7-chk post-upgrade step invariant failed".to_string());
        }
    }

    if FORMAT_VERSION < 8 {
        return Err(format!(
            "v5/v6/v7 fixtures decoded OK, but FORMAT_VERSION is {FORMAT_VERSION} (Phase 2 WM + Episodic Hippocampus requires >= 8)"
        ));
    }
    Ok(())
}

// =============================================================================
// Thread Pool & Main CLI Entrypoint
// =============================================================================

fn run_jobs<T, F>(n: usize, threads: usize, f: F) -> Vec<Option<T>>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let out: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.min(n).max(1) {
            scope.spawn(|| {
                loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= n {
                        break;
                    }
                    let r = f(j);
                    if let Ok(mut g) = out.lock() {
                        g[j] = Some(r);
                    }
                }
            });
        }
    });
    out.into_inner().unwrap_or_default()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / (xs.len() as f64)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let split = args
        .get(1)
        .filter(|s| !s.starts_with("--"))
        .map_or("dev", String::as_str);
    let open_vault = args.iter().any(|a| a == "--open-vault");

    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let mut opt_v7_chk: Option<PathBuf> = None;
    let mut idx = 1;
    while idx < args.len() {
        if args[idx] == "--threads"
            && let Some(val) = args.get(idx + 1)
            && let Ok(n) = val.parse::<usize>()
        {
            threads = n.max(1);
            idx += 2;
            continue;
        }
        if args[idx] == "--v7-chk"
            && let Some(val) = args.get(idx + 1)
        {
            opt_v7_chk = Some(PathBuf::from(val));
            idx += 2;
            continue;
        }
        idx += 1;
    }

    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let seeds: Vec<u64> = match split {
        "dev" => DEFAULT_TWELVE_SEEDS.to_vec(),
        "dev30" => (0..30).collect(),
        "dev60" => (0..60).collect(),
        "heldout" => {
            if !open_vault {
                eprintln!("Refusing: heldout requires --open-vault.");
                return ExitCode::from(2);
            }
            if !ws_root.join(".vault_authorized_phase2").exists() {
                eprintln!("Refusing: .vault_authorized_phase2 missing in workspace root.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase2.opened");
            if marker.exists() {
                eprintln!(
                    "Refusing: Phase-2 held-out vault already opened ({}).",
                    marker.display()
                );
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (800..830).collect()
        }
        other => {
            eprintln!("Unknown split '{other}' (expected dev, dev30, dev60, or heldout)");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-2 Locked Judge | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    let raw = run_jobs(seeds.len(), threads, |i| run_phase2_seed(seeds[i]));
    let mut rows = Vec::with_capacity(seeds.len());
    for (i, item) in raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Phase-2 seed job {i} panicked");
            return ExitCode::from(1);
        };
        rows.push(r);
    }

    let d_nat: Vec<f64> = rows.iter().map(|r| r.detour_nat).collect();
    let d_frz: Vec<f64> = rows.iter().map(|r| r.detour_frz).collect();
    let d_yok: Vec<f64> = rows.iter().map(|r| r.detour_yok).collect();
    let d_shf: Vec<f64> = rows.iter().map(|r| r.detour_shuf).collect();
    let d_tf: Vec<f64> = rows.iter().map(|r| r.detour_trained_frz).collect();
    let d_les: Vec<f64> = rows.iter().map(|r| r.detour_lesion_frz).collect();
    let d_acc: Vec<f64> = rows.iter().map(|r| r.detour_nat_acc).collect();

    let r_nat: Vec<f64> = rows.iter().map(|r| r.role_nat).collect();
    let r_yok: Vec<f64> = rows.iter().map(|r| r.role_yok).collect();
    let r_frz: Vec<f64> = rows.iter().map(|r| r.role_frz).collect();
    let r_acc: Vec<f64> = rows.iter().map(|r| r.role_nat_acc).collect();

    let zs_tr: Vec<f64> = rows.iter().map(|r| r.zs_trained_frz).collect();
    let zs_nb: Vec<f64> = rows.iter().map(|r| r.zs_newborn_frz).collect();
    let zs_ls: Vec<f64> = rows.iter().map(|r| r.zs_lesion_frz).collect();
    let zs_acc: Vec<f64> = rows.iter().map(|r| r.zs_trained_acc).collect();

    let ab_ret: Vec<f64> = rows.iter().map(|r| r.abab_return_nat).collect();
    let ab_nb: Vec<f64> = rows.iter().map(|r| r.abab_newborn).collect();

    let dst_vec: Vec<f64> = rows.iter().map(|r| r.distractor_share).collect();
    let dst_chance: Vec<f64> = vec![64.0 / 192.0; rows.len()];
    let ag_nat: Vec<f64> = rows.iter().map(|r| r.nat_agency).collect();
    let ag_shf: Vec<f64> = rows.iter().map(|r| r.shf_agency).collect();

    let g1a = conservative_paired_gate(&d_nat, &d_frz);
    let g1b = conservative_paired_gate(&d_nat, &d_yok);
    let g1c = conservative_paired_gate(&d_nat, &d_shf);
    let g1d = conservative_paired_gate(&d_tf, &d_les);
    let g2a = conservative_paired_gate(&r_nat, &r_yok);
    let g2a_frz = conservative_paired_gate(&r_nat, &r_frz);
    let g2b = conservative_paired_gate(&zs_tr, &zs_nb);
    let g2b_les = conservative_paired_gate(&zs_tr, &zs_ls);
    let g3 = conservative_paired_gate(&ab_ret, &ab_nb);
    let g4 = conservative_paired_gate(&dst_chance, &dst_vec);
    let g5 = conservative_paired_gate(&ag_nat, &ag_shf);


    let raw_p = [
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g1d.p_two_sided,
        g2a.p_two_sided,
        g2b.p_two_sided,
        g3.p_two_sided,
        g4.p_two_sided,
        g5.p_two_sided,
    ];
    let p_holm = holm_adjust(&raw_p);

    let mean_detour_acc = mean(&d_acc);
    let mean_role_acc = mean(&r_acc);
    let mean_zs_acc = mean(&zs_acc);
    let mean_dst = mean(&dst_vec);
    let mean_shf_ag = mean(&ag_shf);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05 && mean_detour_acc > 0.58;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g1d = g1d.mean_diff > 0.0 && p_holm[3] < 0.05;
    let pass_g2a = g2a.mean_diff > 0.0
        && p_holm[4] < 0.05
        && g2a_frz.mean_diff > 0.0
        && g2a_frz.p_two_sided < 0.05
        && mean_role_acc > 0.60;
    let pass_g2b = g2b.mean_diff > 0.0
        && p_holm[5] < 0.05
        && g2b_les.mean_diff > 0.0
        && g2b_les.p_two_sided < 0.05
        && mean_zs_acc > 0.58;
    let pass_g3 = g3.mean_diff > 0.0 && p_holm[6] < 0.05;
    let pass_g4 = g4.mean_diff > 0.0 && p_holm[7] < 0.05 && mean_dst < 0.05;
    let pass_g5 = g5.mean_diff > 0.015 && p_holm[8] < 0.05 && mean_shf_ag <= 0.05;


    let gates_print = [
        ("P2-G1a OccludedDetour20 Nat > Frozen", g1a, p_holm[0], pass_g1a),
        ("P2-G1b OccludedDetour20 Nat > Yoked", g1b, p_holm[1], pass_g1b),
        ("P2-G1c OccludedDetour20 Nat > Shuffled", g1c, p_holm[2], pass_g1c),
        ("P2-G1d OccludedDetour20 Trained > Lesion", g1d, p_holm[3], pass_g1d),
        ("P2-G2a SymRoleSwap20 Nat > Yoked(&Frz)", g2a, p_holm[4], pass_g2a),
        ("P2-G2b ZeroShot(E6,E7) TrainedFrz > NbFrz", g2b, p_holm[5], pass_g2b),
        ("P2-G3  RoleSwap ABAB Return-A > Newborn", g3, p_holm[6], pass_g3),
        ("P2-G4  Distractor Chance(33.3%) > Nat(<5%)", g4, p_holm[7], pass_g4),
        ("P2-G5  Agency Purity Nat > ActionShuffled", g5, p_holm[8], pass_g5),
    ];

    println!("\n--- Behavioral, Working-Memory, System-2 & Embodied Gates (Exact Sign-Flip + Student-t, Holm m=9, Pure Brain) ---");
    for (name, g, ph, ok) in gates_print {
        println!(
            "  [{:<4}] {:<45} A={:+.4} B={:+.4} diff={:+.4} wins={}/{} t={:+.2} p_holm={:.4e}",
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
        "         detour_acc={:.4} (>0.5800) | role_acc={:.4} (>0.6000) | zeroshot_acc={:.4} (>0.5800) | dst_share={:.4} (<0.0500) | shuf_ag={:.4} (<=0.0500)",
        mean_detour_acc, mean_role_acc, mean_zs_acc, mean_dst, mean_shf_ag
    );

    println!("\n--- Structural, O(k) Search Latency & Mid-Occlusion Persistence Invariants ---");
    let debt = audit_core_debt(&ws_root);
    let pass_i1 = debt.hygiene_violations == 0
        && debt.split_brain_sites.is_empty()
        && debt.organ_mirror_sites.is_empty()
        && debt.cell_modulo_sites.is_empty();
    println!(
        "  [{:<4}] P2-I1 Zero Core Debt: hygiene={} split_brain={} mirrors={} cell_mod={}",
        if pass_i1 { "PASS" } else { "FAIL" },
        debt.hygiene_violations,
        debt.split_brain_sites.len(),
        debt.organ_mirror_sites.len(),
        debt.cell_modulo_sites.len(),
    );

    let (pass_i2, i2_msg) = match audit_latency_and_frozen_invariance() {
        Ok((p99, max_act, pores, cells)) => {
            let ok = p99 < 2_000.0 && max_act <= 64 && pores <= 64 && cells >= 512;
            (
                ok,
                format!(
                    "p99={p99:.1} us (<2000) max_active={max_act} (<=64) pores={pores} (<=64) cells={cells} frz_hash=0bit_drift"
                ),
            )
        }
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P2-I2 O(k) Latency & Frozen 0-Bit Drift: {i2_msg}",
        if pass_i2 { "PASS" } else { "FAIL" }
    );

    let (pass_i3, i3_msg) = match audit_mid_occlusion_persistence() {
        Ok(ckpt_bytes) => (
            ckpt_bytes <= 524_288,
            format!(
                "ckpt_bytes={ckpt_bytes} (<=524288) mid_occlusion_bit_id_200=OK prev_heal=OK dual_poison=OK"
            ),
        ),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P2-I3 Mid-Occlusion Paged Persistence: {i3_msg}",
        if pass_i3 { "PASS" } else { "FAIL" }
    );

    let (pass_i4, i4_msg) = match audit_v8_migration(opt_v7_chk.as_deref()) {
        Ok(()) => (
            true,
            "v5_valid=true v6_valid=true v7_valid=true status=OK".to_string(),
        ),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P2-I4 Legacy v5/v6/v7 -> v8+ Migration: {i4_msg}",
        if pass_i4 { "PASS" } else { "FAIL" }
    );

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g1d
        && pass_g2a
        && pass_g2b
        && pass_g3
        && pass_g4
        && pass_g5
        && pass_i1
        && pass_i2
        && pass_i3
        && pass_i4;

    println!(
        "\n=== Phase-2 Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );

    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
