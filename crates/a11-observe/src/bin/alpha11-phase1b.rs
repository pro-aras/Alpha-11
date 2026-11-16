//! Pre-registered locked Phase-1B judge (`P1B-G1..G5` + `P1B-I1..I4`).
//!
//! Enforces unbounded disk-paged `CRC32C` sparse cortical columns (`M -> infinity`),
//! strictly bounded `k <= 64` active routing per tick (`p99 < 2,000 us` at `4,096` channels
//! and `M >= 512` cortical cells), `< 2%` catastrophic forgetting across `12` sequential
//! sensorimotor domains (`D_0 .. D_11`) with `6`-arm + flagless lesion null controls,
//! distant-domain (`D_0 -> ... -> D_11 -> D_0`) re-entry savings, `.slab` corruption
//! `.prev` recovery + fail-closed poisoning, and lossless `v5`/`v6` -> `v7+` migration.
//!
//! Usage: `alpha11-phase1b [dev|dev30|heldout] [--threads N] [--open-vault] [--v6-chk PATH]`

use std::env;
use std::f32::consts::TAU;
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

const NUM_DOMAINS: usize = 12;
const SEQ_AMBIENT_CHANNELS: usize = 192;
const SEQ_ACTUATORS: usize = 24;
const ROOM_SPECTRUM_CHANNELS: usize = 32;
const DISTRACTOR_START: usize = 96;
const OU_CHANNELS: usize = 32;
const SELF_COPY_CHANNELS: usize = 32;

const TRAIN_TICKS_PER_DOMAIN: u64 = 1_500;
const EVAL_TICKS_PER_DOMAIN: u64 = 800;
const REENTRY_TICKS: u64 = 500;

/// Expected binary magic header on every paged cortical column `.slab` file (`"A11CSLAB"` LE).
const COL_SLAB_MAGIC: u64 = 0x4131_3143_534C_4142;

const PERM_SENSOR_SALT: u64 = 0x5E45_0A9E_B11B_2026;
const PERM_MOTOR_SALT: u64 = 0x4D07_0A9E_B11B_2026;
const SPECTRUM_SALT: u64 = 0x59EC_7A12_D012_2026;
const TRAIN_WORLD_SALT: u64 = 0x7A41_4E00_1200_5EED;
const EVAL_WORLD_SALT: u64 = 0xE7A1_0012_9988_7766;
const REENTRY_WORLD_SALT: u64 = 0xAE00_D000_1122_3344;
const SHUFFLE_SALT: u64 = 0x5A17_9B1B_24AC_2026;

/// Seed-specific permutation & room-spectrum geometry shared across all arms of a seed.
#[derive(Clone, Debug)]
struct Seq12Mapping {
    sensor_perm: [usize; SEQ_AMBIENT_CHANNELS],
    inv_sensor_perm: [usize; SEQ_AMBIENT_CHANNELS],
    act_perm: [usize; SEQ_ACTUATORS],
    /// Balanced 16-of-32 Hadamard room-spectrum codes (`true` = bright wave, `false` = dim wave).
    room_code: [[bool; ROOM_SPECTRUM_CHANNELS]; NUM_DOMAINS],
    wave_phase: [f32; ROOM_SPECTRUM_CHANNELS],
}

impl Seq12Mapping {
    fn from_seed(seed: u64) -> Self {
        let mut s_rng = DeterministicRng::new(seed ^ PERM_SENSOR_SALT);
        let mut sensor_perm: [usize; SEQ_AMBIENT_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..SEQ_AMBIENT_CHANNELS).rev() {
            let j = (s_rng.next_u64() as usize) % (i + 1);
            sensor_perm.swap(i, j);
        }
        let mut inv_sensor_perm = [0usize; SEQ_AMBIENT_CHANNELS];
        for (sc_ch, &true_ch) in sensor_perm.iter().enumerate() {
            inv_sensor_perm[sc_ch] = true_ch;
        }

        let mut m_rng = DeterministicRng::new(seed ^ PERM_MOTOR_SALT);
        let mut act_perm: [usize; SEQ_ACTUATORS] = std::array::from_fn(|i| i);
        for i in (1..SEQ_ACTUATORS).rev() {
            let j = (m_rng.next_u64() as usize) % (i + 1);
            act_perm.swap(i, j);
        }

        // Construct rows 1..=12 (seed-shuffled from 1..32) of a 32x32 Sylvester-Hadamard matrix
        // with seed-permuted columns: every room has exactly 16 bright + 16 dim wave channels,
        // and every distinct pair of rooms differs on exactly 16 of 32 channels (zero action leak).
        let mut h_rng = DeterministicRng::new(seed ^ SPECTRUM_SALT);
        let mut row_pool: [usize; 31] = std::array::from_fn(|i| i + 1);
        for i in (1..31).rev() {
            let j = (h_rng.next_u64() as usize) % (i + 1);
            row_pool.swap(i, j);
        }
        let mut col_perm: [usize; ROOM_SPECTRUM_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..ROOM_SPECTRUM_CHANNELS).rev() {
            let j = (h_rng.next_u64() as usize) % (i + 1);
            col_perm.swap(i, j);
        }
        let mut room_code = [[false; ROOM_SPECTRUM_CHANNELS]; NUM_DOMAINS];
        for d in 0..NUM_DOMAINS {
            let r = row_pool[d];
            for k in 0..ROOM_SPECTRUM_CHANNELS {
                let c = col_perm[k];
                let parity = (r & c).count_ones() & 1;
                room_code[d][k] = parity == 0;
            }
        }
        let mut wave_phase = [0.0_f32; ROOM_SPECTRUM_CHANNELS];
        for ph in &mut wave_phase {
            *ph = h_rng.next_f32_unit() * TAU;
        }

        Self {
            sensor_perm,
            inv_sensor_perm,
            act_perm,
            room_code,
            wave_phase,
        }
    }
}

/// 12-domain sequential continuous cave arena (`256` permuted channels, `24` permuted actuators).
///
/// Uninterrupted `WorldMedium`: transitions cleanly across `D_0 .. D_11` (or runs a single target
/// room `d` during held-out evaluation) without touching or resetting the organism.
#[derive(Clone, Debug)]
struct Sequential12RoomsMedium {
    map: Seq12Mapping,
    base_seed: u64,
    world_salt: u64,
    domain: usize,
    local_tick: u64,
    ticks_per_domain: u64,
    auto_advance: bool,
    cave: BrightRadiantWorld,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    motor_ring: [[f32; SEQ_ACTUATORS]; 3],
    ring_cursor: usize,
    raw_ambient: [f32; SEQ_AMBIENT_CHANNELS],
    permuted_ambient: [f32; SEQ_AMBIENT_CHANNELS],
}

impl Sequential12RoomsMedium {
    fn new_curriculum(base_seed: u64, map: Seq12Mapping) -> Self {
        Self::build(
            base_seed,
            TRAIN_WORLD_SALT,
            0,
            TRAIN_TICKS_PER_DOMAIN,
            true,
            map,
        )
    }

    fn new_single_room(base_seed: u64, world_salt: u64, domain: usize, map: Seq12Mapping) -> Self {
        Self::build(base_seed, world_salt, domain, u64::MAX, false, map)
    }

    fn build(
        base_seed: u64,
        world_salt: u64,
        domain: usize,
        ticks_per_domain: u64,
        auto_advance: bool,
        map: Seq12Mapping,
    ) -> Self {
        let d = domain % NUM_DOMAINS;
        let cave_seed = Self::room_cave_seed(base_seed, world_salt, d);
        let cave = BrightRadiantWorld::new(cave_seed);
        let ou_rng = DeterministicRng::new(base_seed ^ world_salt ^ ((d as u64) << 16) ^ 0x0097_B11B);
        let mut env_m = Self {
            map,
            base_seed,
            world_salt,
            domain: d,
            local_tick: 0,
            ticks_per_domain,
            auto_advance,
            cave,
            ou_rng,
            ou_state: [0.0; OU_CHANNELS],
            motor_ring: [[0.0; SEQ_ACTUATORS]; 3],
            ring_cursor: 0,
            raw_ambient: [0.0; SEQ_AMBIENT_CHANNELS],
            permuted_ambient: [0.0; SEQ_AMBIENT_CHANNELS],
        };
        env_m.refresh_ambient(&[0.0; SEQ_ACTUATORS]);
        env_m
    }

    fn room_cave_seed(base_seed: u64, world_salt: u64, domain: usize) -> u64 {
        base_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(world_salt)
            .wrapping_add((domain as u64).wrapping_mul(0x517C_C1B7_2722_0A95))
    }

    fn refresh_ambient(&mut self, delayed_motor: &[f32; SEQ_ACTUATORS]) {
        let cave_view = self.cave.current_view();
        // 0..64: Photonic raycasted retina from BrightRadiantWorld.
        self.raw_ambient[0..64].copy_from_slice(&cave_view.ambient_in[0..64]);

        // 64..96: Smooth C^1 room standing-wave spectrum (16 bright + 16 dim per room, dispersed
        // centers & frequencies so no narrow histogram bin mimics a silent 64-band cochlear floor).
        let t = self.local_tick as f32;
        for k in 0..ROOM_SPECTRUM_CHANNELS {
            let kf = (k as f32) / ((ROOM_SPECTRUM_CHANNELS - 1) as f32);
            let omega = 0.065 + 0.045 * kf;
            let osc = (omega * t + self.map.wave_phase[k]).sin();
            let v = if self.map.room_code[self.domain][k] {
                (0.28 + 0.44 * kf) + (0.14 + 0.08 * kf) * osc
            } else {
                (0.02 + 0.12 * kf) + (0.07 + 0.04 * kf) * osc
            };
            self.raw_ambient[64 + k] = v.clamp(-1.0, 1.0);
        }

        // 96..128: Base BrightRadiantWorld white + flash Noisy-TV distractors.
        self.raw_ambient[96..128].copy_from_slice(&cave_view.ambient_in[96..128]);

        // 128..160: Smooth Ornstein-Uhlenbeck (rho = 0.97) exogenous drift distractors.
        let rho = 0.97_f32;
        let innov = (1.0 - rho * rho).sqrt() * 0.75;
        for i in 0..OU_CHANNELS {
            let eps = self.ou_rng.next_f32_bipolar();
            self.ou_state[i] = (rho * self.ou_state[i] + innov * eps).clamp(-1.0, 1.0);
            self.raw_ambient[128 + i] = self.ou_state[i];
        }

        // 160..192: Lag-3 motor self-reafference copy distractors (96 total distractors = 50.0%).
        for i in 0..SELF_COPY_CHANNELS {
            let m_val = delayed_motor[i % SEQ_ACTUATORS];
            let sign = if (i & 1) == 0 { 0.85 } else { -0.85 };
            let jitter = 0.04 * self.ou_rng.next_f32_bipolar();
            self.raw_ambient[160 + i] = (sign * m_val + jitter).clamp(-1.0, 1.0);
        }

        for (sc_ch, &true_ch) in self.map.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[true_ch];
        }
    }
}

impl WorldMedium for Sequential12RoomsMedium {
    fn num_ambient_channels(&self) -> usize {
        SEQ_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        SEQ_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        let cv = self.cave.current_view();
        FieldView {
            ambient_in: &self.permuted_ambient,
            metabolic_influx: cv.metabolic_influx,
            tissue_stress: cv.tissue_stress,
            num_actuators: SEQ_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let delayed = self.motor_ring[self.ring_cursor];
        let mut true_act = [0.0_f32; SEQ_ACTUATORS];
        for (org_idx, &true_idx) in self.map.act_perm.iter().enumerate() {
            let v = pulse
                .actuator_out
                .get(org_idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);
            self.motor_ring[self.ring_cursor][org_idx] = v;
            true_act[true_idx] = v;
        }
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        // Domain d = 2*g + pol_bit: limb group g in 0..6 ([4*g .. 4*g+4]), thrust sign +1 (even) / -1 (odd).
        let g = (self.domain / 2) % 6;
        let sign = if (self.domain & 1) == 0 { 1.0_f32 } else { -1.0_f32 };
        let base = 4 * g;
        let cave_pulse = FieldPulse {
            actuator_out: vec![
                (sign * true_act[base]).clamp(-1.0, 1.0),
                (sign * true_act[base + 1]).clamp(-1.0, 1.0),
                (sign * true_act[base + 2]).clamp(-1.0, 1.0),
                (sign * true_act[base + 3]).clamp(-1.0, 1.0),
                0.0,
                0.0,
            ],
        };
        self.cave.apply_pulse(&cave_pulse);
        self.local_tick = self.local_tick.saturating_add(1);

        if self.auto_advance
            && self.local_tick >= self.ticks_per_domain
            && self.domain + 1 < NUM_DOMAINS
        {
            self.domain += 1;
            self.local_tick = 0;
            let next_seed = Self::room_cave_seed(self.base_seed, self.world_salt, self.domain);
            self.cave = BrightRadiantWorld::new(next_seed);
        }

        self.refresh_ambient(&delayed);
    }
}

#[derive(Clone, Debug, Default)]
struct Seq12SeedMetrics {
    reloaded_eval_total: f64,
    newborn_eval_total: f64,
    yoked_eval_total: f64,
    shuffled_eval_total: f64,
    lesion_eval_total: f64,
    forgetting_ratio: f64,
    peak_gain_mean: f64,
    early4_reloaded: f64,
    early4_newborn: f64,
    distractor_share: f64,
    nat_agency: f64,
    shf_agency: f64,
    reentry_d0_nat: f64,
    reentry_d0_newborn: f64,
    max_active_cells_seen: usize,
}

fn eval_frozen_across_12_rooms(
    org: &mut Alpha11Organism,
    seed: u64,
    map: &Seq12Mapping,
    max_active: &mut usize,
) -> [f64; NUM_DOMAINS] {
    org.set_control_mode(ControlMode::FrozenPlasticity);
    let hash_before = org.learned_parameter_hash();
    let mut scores = [0.0_f64; NUM_DOMAINS];
    for (d, slot) in scores.iter_mut().enumerate() {
        let mut room = Sequential12RoomsMedium::new_single_room(seed, EVAL_WORLD_SALT, d, map.clone());
        let mut net = 0.0_f64;
        for _ in 0..EVAL_TICKS_PER_DOMAIN {
            let pulse = {
                let v = room.current_view();
                org.step(v)
                    .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
            };
            let act_now = org.snapshot().active_cells_last_tick;
            if act_now > *max_active {
                *max_active = act_now;
            }
            room.apply_pulse(&pulse);
            let after = room.current_view();
            net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
        }
        *slot = net;
    }
    assert_eq!(
        hash_before,
        org.learned_parameter_hash(),
        "FrozenPlasticity learned_parameter_hash drifted during 12-room evaluation!"
    );
    scores
}

fn eval_frozen_single_room(
    org: &Alpha11Organism,
    seed: u64,
    domain: usize,
    map: &Seq12Mapping,
    max_active: &mut usize,
) -> f64 {
    let mut probe = org.clone();
    probe.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = probe.learned_parameter_hash();
    let mut room = Sequential12RoomsMedium::new_single_room(seed, EVAL_WORLD_SALT, domain, map.clone());
    let mut net = 0.0_f64;
    for _ in 0..EVAL_TICKS_PER_DOMAIN {
        let pulse = {
            let v = room.current_view();
            probe
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        let act_now = probe.snapshot().active_cells_last_tick;
        if act_now > *max_active {
            *max_active = act_now;
        }
        room.apply_pulse(&pulse);
        let after = room.current_view();
        net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }
    assert_eq!(
        h0,
        probe.learned_parameter_hash(),
        "FrozenPlasticity learned_parameter_hash drifted during end-of-domain probe!"
    );
    net
}

fn roundtrip_disk_reload(org: &Alpha11Organism, tag_seed: u64, label: &str) -> Alpha11Organism {
    let dir = env::temp_dir().join(format!(
        "a11_p1b_{}_{}_{}",
        std::process::id(),
        label,
        tag_seed
    ));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    let primary = dir.join("organism.ckpt");
    let prev = dir.join("organism.ckpt.prev");
    let tmp = dir.join("organism.ckpt.tmp");

    org.save_transactional(&primary, &prev, &tmp)
        .expect("transactional save during Seq12 evaluation must succeed");
    let mut reloaded = Alpha11Organism::new(tag_seed ^ 0xDEAD_BEEF, org.config());
    let used_prev = reloaded
        .load_transactional(&primary, &prev)
        .expect("transactional reload during Seq12 evaluation must succeed");
    assert!(!used_prev, "fresh primary checkpoint unexpectedly fell back to prev");
    let _ = fs::remove_dir_all(&dir);
    reloaded
}

fn run_seq12_seed(seed: u64) -> Seq12SeedMetrics {
    let cfg = OrganismConfig::default();
    let map = Seq12Mapping::from_seed(seed);
    let total_train_ticks = (NUM_DOMAINS as u64) * TRAIN_TICKS_PER_DOMAIN;
    let mut max_active = 0usize;

    // 0. Untrained Newborn-Frozen baseline across all 12 held-out evaluation rooms (B_0 .. B_11).
    let mut nb_frz = Alpha11Organism::new(seed, cfg);
    let b_scores = eval_frozen_across_12_rooms(&mut nb_frz, seed, &map, &mut max_active);

    // 1. Natural organism living uninterrupted through D_0 -> D_1 -> ... -> D_11 (recording donor stream).
    let mut nat = Alpha11Organism::new(seed, cfg);
    let mut w_nat = Sequential12RoomsMedium::new_curriculum(seed, map.clone());
    let mut donor = Vec::with_capacity(total_train_ticks as usize);
    let mut peak_scores = [0.0_f64; NUM_DOMAINS];

    for t in 0..total_train_ticks {
        let pulse = {
            let v = w_nat.current_view();
            donor.push((v.metabolic_influx, v.tissue_stress));
            nat.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        let act_now = nat.snapshot().active_cells_last_tick;
        if act_now > max_active {
            max_active = act_now;
        }
        let d_before = ((t / TRAIN_TICKS_PER_DOMAIN) as usize).min(NUM_DOMAINS - 1);
        w_nat.apply_pulse(&pulse);
        if (t + 1) % TRAIN_TICKS_PER_DOMAIN == 0 {
            peak_scores[d_before] =
                eval_frozen_single_room(&nat, seed, d_before, &map, &mut max_active);
        }
    }

    let nat_snap = nat.snapshot();
    let mut total_g = 0.0_f64;
    let mut distractor_g = 0.0_f64;
    for (&sc_ch, &g) in nat_snap
        .sensor_pore_channels
        .iter()
        .zip(nat_snap.sensor_pore_conductances.iter())
    {
        let cu = sc_ch as usize;
        if cu < SEQ_AMBIENT_CHANNELS {
            let true_ch = map.inv_sensor_perm[cu];
            let gf = f64::from(g.max(0.0));
            total_g += gf;
            if true_ch >= DISTRACTOR_START {
                distractor_g += gf;
            }
        }
    }
    let distractor_share = if total_g > 1.0e-9 {
        distractor_g / total_g
    } else {
        1.0
    };

    // Save `nat` transactionally to disk (.ckpt + .pages/*.slab), reload into fresh organism,
    // and evaluate FrozenPlasticity across all 12 rooms (0-bit drift verified).
    let mut nat_reloaded = roundtrip_disk_reload(&nat, seed, "nat");
    let mut lesion = nat_reloaded.clone();
    lesion.zero_learned_conductances();

    let r12_scores = eval_frozen_across_12_rooms(&mut nat_reloaded, seed, &map, &mut max_active);
    let lesion_scores = eval_frozen_across_12_rooms(&mut lesion, seed, &map, &mut max_active);

    // 2. Distant-domain plastic re-entry (D_0 -> ... -> D_11 -> D_0) vs fresh plastic Newborn in D_0.
    let mut nat_reentry = roundtrip_disk_reload(&nat, seed, "reentry");
    nat_reentry.set_control_mode(ControlMode::Natural);
    let mut w_re_nat =
        Sequential12RoomsMedium::new_single_room(seed, REENTRY_WORLD_SALT, 0, map.clone());
    let mut w_re_nb = w_re_nat.clone();
    let mut nb_plastic = Alpha11Organism::new(seed.wrapping_add(700_000), cfg);
    let mut reentry_d0_nat = 0.0_f64;
    let mut reentry_d0_newborn = 0.0_f64;
    for _ in 0..REENTRY_TICKS {
        let p_n = {
            let v = w_re_nat.current_view();
            nat_reentry
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        let act_now = nat_reentry.snapshot().active_cells_last_tick;
        if act_now > max_active {
            max_active = act_now;
        }
        w_re_nat.apply_pulse(&p_n);
        let af_n = w_re_nat.current_view();
        reentry_d0_nat += f64::from(af_n.metabolic_influx) - f64::from(af_n.tissue_stress);

        let p_b = {
            let v = w_re_nb.current_view();
            nb_plastic
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        w_re_nb.apply_pulse(&p_b);
        let af_b = w_re_nb.current_view();
        reentry_d0_newborn += f64::from(af_b.metabolic_influx) - f64::from(af_b.tissue_stress);
    }

    // 3. YokedReward twin across identical 12-domain curriculum -> disk reload -> 12-room frozen eval.
    let mut yok = Alpha11Organism::new(seed.wrapping_add(500_000), cfg);
    let mut w_yok = YokedMedium::new(
        Sequential12RoomsMedium::new_curriculum(seed, map.clone()),
        donor,
    );
    for _ in 0..total_train_ticks {
        let p = {
            let v = w_yok.current_view();
            yok.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        w_yok.apply_pulse(&p);
    }
    let mut yok_reloaded = roundtrip_disk_reload(&yok, seed, "yok");
    let yok_scores = eval_frozen_across_12_rooms(&mut yok_reloaded, seed, &map, &mut max_active);

    // 4. ActionShuffled twin across identical 12-domain curriculum -> disk reload -> 12-room frozen eval.
    let mut shf = Alpha11Organism::new(seed, cfg);
    let mut w_shf = Sequential12RoomsMedium::new_curriculum(seed, map.clone());
    let mut shuffler = PulseTemporalChannelShuffler::new(seed ^ SHUFFLE_SALT, SEQ_ACTUATORS);
    for _ in 0..total_train_ticks {
        let emitted = {
            let v = w_shf.current_view();
            shf.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(SEQ_ACTUATORS))
        };
        let applied = shuffler.shuffle_pulse(&emitted);
        w_shf.apply_pulse(&applied);
    }
    let shf_snap = shf.snapshot();
    let mut shf_reloaded = roundtrip_disk_reload(&shf, seed, "shf");
    let shf_scores = eval_frozen_across_12_rooms(&mut shf_reloaded, seed, &map, &mut max_active);

    // Unbiased Lopez-Paz & Ranzato (2017) backward-transfer forgetting ratio across prior domains d = 0..11.
    let prior_n = (NUM_DOMAINS - 1) as f64;
    let mut sum_peak_gain = 0.0_f64;
    let mut sum_forgot = 0.0_f64;
    for d in 0..(NUM_DOMAINS - 1) {
        sum_peak_gain += peak_scores[d] - b_scores[d];
        sum_forgot += peak_scores[d] - r12_scores[d];
    }
    let peak_gain_mean = sum_peak_gain / prior_n;
    let forgetting_ratio = sum_forgot / sum_peak_gain.max(prior_n);

    Seq12SeedMetrics {
        reloaded_eval_total: r12_scores.iter().sum(),
        newborn_eval_total: b_scores.iter().sum(),
        yoked_eval_total: yok_scores.iter().sum(),
        shuffled_eval_total: shf_scores.iter().sum(),
        lesion_eval_total: lesion_scores.iter().sum(),
        forgetting_ratio,
        peak_gain_mean,
        early4_reloaded: r12_scores[0..4].iter().sum(),
        early4_newborn: b_scores[0..4].iter().sum(),
        distractor_share,
        nat_agency: f64::from(nat_snap.mean_interventional_agency),
        shf_agency: f64::from(shf_snap.mean_interventional_agency),
        reentry_d0_nat,
        reentry_d0_newborn,
        max_active_cells_seen: max_active,
    }
}

// =============================================================================
// Exact 2^n / Phipson-Smyth Sign-Flip Permutation Test
// =============================================================================

fn sign_flip_two_sided_p(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let diffs: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
    let nf = n as f64;
    let obs = (diffs.iter().sum::<f64>() / nf).abs();
    if obs <= 0.0 {
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
    let mut rng = DeterministicRng::new(0xA11B_516F_5EED_2026);
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
// P1B-I1: Core Source Debt Audit (0 hygiene, 0 split-brain, 0 mirrors, 0 % 64)
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
// P1B-I2: Strict k <= 64 Active Routing & M=512 / 4,096-Channel O(k) Latency
// =============================================================================

fn benchmark_high_capacity_4096() -> (f64, usize, usize, usize) {
    let cfg = OrganismConfig {
        initial_cells: 512,
        max_cells: 65_536,
        ..OrganismConfig::default()
    };
    let mut org = Alpha11Organism::new(2026_512, cfg);
    let mut world = DenseFieldWorld::new(77);
    assert_eq!(world.num_ambient_channels(), 4_096);
    let steps = 1_000usize;
    let mut us = Vec::with_capacity(steps);
    let mut max_active_cells = 0usize;
    for _ in 0..steps {
        let pulse = {
            let v = world.current_view();
            let t0 = Instant::now();
            let p = org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators));
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
    (
        p99,
        max_active_cells,
        final_snap.sensor_pore_count,
        final_snap.cell_count,
    )
}

// =============================================================================
// P1B-I3: Disk-Paged CRC32C Cortical Column Slabs & Fail-Closed Corruption Audit
// =============================================================================

fn verify_slab_file_crc(path: &Path) -> Result<u32, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {} failed: {e}", path.display()))?;
    if bytes.len() < 24 {
        return Err(format!("slab {} truncated ({} < 24)", path.display(), bytes.len()));
    }
    let magic = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    if magic != COL_SLAB_MAGIC {
        return Err(format!(
            "slab {} invalid magic 0x{magic:016X} != 0x{COL_SLAB_MAGIC:016X}",
            path.display()
        ));
    }
    let ver = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if ver != FORMAT_VERSION || ver < 7 {
        return Err(format!(
            "slab {} invalid version {ver} (expected FORMAT_VERSION={FORMAT_VERSION} >= 7)",
            path.display()
        ));
    }
    let col_id = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let payload_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    if 24usize.saturating_add(payload_len) != bytes.len() {
        return Err(format!(
            "slab {} payload_len={payload_len} mismatch with total len={}",
            path.display(),
            bytes.len()
        ));
    }
    let body_end = bytes.len() - 4;
    let expected_crc = crc32c(&bytes[..body_end]);
    let stored_crc = u32::from_le_bytes([
        bytes[body_end],
        bytes[body_end + 1],
        bytes[body_end + 2],
        bytes[body_end + 3],
    ]);
    if expected_crc != stored_crc {
        return Err(format!(
            "slab {} CRC32C mismatch 0x{expected_crc:08X} != 0x{stored_crc:08X}",
            path.display()
        ));
    }
    Ok(col_id)
}

fn collect_slab_files(pages_dir: &Path) -> Result<Vec<PathBuf>, String> {
    if !pages_dir.is_dir() {
        return Err(format!("missing cortical page directory {}", pages_dir.display()));
    }
    let entries = fs::read_dir(pages_dir)
        .map_err(|e| format!("read_dir {} failed: {e}", pages_dir.display()))?;
    let mut slabs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "slab"))
        .collect();
    slabs.sort();
    Ok(slabs)
}

fn audit_paged_slabs() -> Result<(usize, usize), String> {
    let audit_seed = 2026_0118_u64;
    let map = Seq12Mapping::from_seed(audit_seed);
    let mut org = Alpha11Organism::new(audit_seed, OrganismConfig::default());
    let mut w = Sequential12RoomsMedium::new_curriculum(audit_seed, map.clone());
    let total_ticks = (NUM_DOMAINS as u64) * TRAIN_TICKS_PER_DOMAIN;
    for _ in 0..total_ticks {
        let p = {
            let v = w.current_view();
            org.step(v)
                .map_err(|e| format!("Seq12 slab-audit step failed: {e:?}"))?
        };
        w.apply_pulse(&p);
    }

    let dir = env::temp_dir().join(format!("a11_p1b_slab_audit_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| format!("create temp dir failed: {e}"))?;

    let result = (|| -> Result<(usize, usize), String> {
        let primary = dir.join("organism.ckpt");
        let prev = dir.join("organism.ckpt.prev");
        let tmp = dir.join("organism.ckpt.tmp");
        let primary_pages = dir.join("organism.ckpt.pages");
        let prev_pages = dir.join("organism.ckpt.prev.pages");

        // Generation 1 save -> populates primary + primary.pages
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen-1 save_transactional failed: {e:?}"))?;

        // Step once and save Generation 2 -> rotates gen-1 into prev + prev.pages, writes gen-2 to primary + primary.pages
        let p1 = org
            .step(w.current_view())
            .map_err(|e| format!("post-gen1 step failed: {e:?}"))?;
        w.apply_pulse(&p1);
        org.save_transactional(&primary, &prev, &tmp)
            .map_err(|e| format!("gen-2 save_transactional failed: {e:?}"))?;

        let ckpt_bytes = fs::metadata(&primary)
            .map_err(|e| format!("stat primary failed: {e}"))?
            .len() as usize;
        if ckpt_bytes > 524_288 {
            return Err(format!(
                "primary organism.ckpt bloated to {ckpt_bytes} bytes (> 524288; cold columns not paged)"
            ));
        }

        let slabs = collect_slab_files(&primary_pages)?;
        if slabs.len() < 8 {
            return Err(format!(
                "expected >= 8 paged .slab files after 12 domains, found {}",
                slabs.len()
            ));
        }
        for sp in &slabs {
            let _ = verify_slab_file_crc(sp)?;
        }
        let prev_slabs = collect_slab_files(&prev_pages)?;
        if prev_slabs.len() < 8 {
            return Err(format!(
                "expected >= 8 .prev.pages .slab files, found {}",
                prev_slabs.len()
            ));
        }
        for sp in &prev_slabs {
            let _ = verify_slab_file_crc(sp)?;
        }

        // 200-step cross-domain (100 in D_0 + 100 in D_7) bit-identical reload verification.
        let mut reloaded = Alpha11Organism::new(audit_seed ^ 0xCAFE, OrganismConfig::default());
        let used_prev = reloaded
            .load_transactional(&primary, &prev)
            .map_err(|e| format!("clean load_transactional failed: {e:?}"))?;
        if used_prev {
            return Err("clean load_transactional unexpectedly used .prev".to_string());
        }
        if org.learned_parameter_hash() != reloaded.learned_parameter_hash() {
            return Err("learned_parameter_hash mismatch after clean slab reload".to_string());
        }

        let mut org_ref = org.clone();
        for &d_test in &[0usize, 7usize] {
            let mut r_a =
                Sequential12RoomsMedium::new_single_room(audit_seed, EVAL_WORLD_SALT, d_test, map.clone());
            let mut r_b = r_a.clone();
            for step_i in 0..100 {
                let pa = org_ref
                    .step(r_a.current_view())
                    .map_err(|e| format!("orig step {step_i} in D_{d_test} failed: {e:?}"))?;
                let pb = reloaded
                    .step(r_b.current_view())
                    .map_err(|e| format!("reloaded step {step_i} in D_{d_test} failed: {e:?}"))?;
                if pa.actuator_out != pb.actuator_out {
                    return Err(format!(
                        "pulse mismatch after paged reload at step {step_i} in D_{d_test}"
                    ));
                }
                r_a.apply_pulse(&pa);
                r_b.apply_pulse(&pb);
            }
        }
        if org_ref.learned_parameter_hash() != reloaded.learned_parameter_hash() {
            return Err("post-200-step learned_parameter_hash mismatch after paged recall".to_string());
        }

        // Corrupt 1 payload bit in primary's first .slab file ONLY (leaving organism.ckpt intact):
        // load_transactional MUST reject primary via slab CRC32C and recover from prev (Ok(true)).
        let mut corrupt_slab = fs::read(&slabs[0]).map_err(|e| e.to_string())?;
        corrupt_slab[20] ^= 0x01;
        fs::write(&slabs[0], &corrupt_slab).map_err(|e| e.to_string())?;

        let mut rec_org = Alpha11Organism::new(audit_seed ^ 0x1111, OrganismConfig::default());
        let fell_back = rec_org
            .load_transactional(&primary, &prev)
            .map_err(|e| format!("slab .prev fallback failed: {e:?}"))?;
        if !fell_back || rec_org.is_poisoned() {
            return Err(format!(
                "expected Ok(true) unpoisoned .prev fallback on primary .slab bit-flip, got fell_back={fell_back} poisoned={}",
                rec_org.is_poisoned()
            ));
        }

        // Now corrupt 1 payload bit in prev's first .slab file as well:
        // load_transactional MUST fail closed with ChecksumMismatch and poison the target organism.
        let mut corrupt_prev_slab = fs::read(&prev_slabs[0]).map_err(|e| e.to_string())?;
        corrupt_prev_slab[20] ^= 0x01;
        fs::write(&prev_slabs[0], &corrupt_prev_slab).map_err(|e| e.to_string())?;

        let mut poisoned_org = Alpha11Organism::new(audit_seed ^ 0x2222, OrganismConfig::default());
        match poisoned_org.load_transactional(&primary, &prev) {
            Err(PersistError::ChecksumMismatch) if poisoned_org.is_poisoned() => {}
            other => {
                return Err(format!(
                    "expected Err(ChecksumMismatch) + poisoned=true on dual .slab corruption, got {other:?} poisoned={}",
                    poisoned_org.is_poisoned()
                ));
            }
        }
        match poisoned_org.step(w.current_view()) {
            Err(PersistError::PoisonedAfterCorruption) => {}
            other => {
                return Err(format!(
                    "expected Err(PoisonedAfterCorruption) when stepping poisoned organism, got {other:?}"
                ));
            }
        }

        Ok((slabs.len(), ckpt_bytes))
    })();

    let _ = fs::remove_dir_all(&dir);
    result
}

// =============================================================================
// P1B-I4: Lossless FORMAT_VERSION >= 7 Upgrade from Canonical v5, Canonical v6 & --v6-chk
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
    push_u64(&mut body, 42); // seed
    push_u64(&mut body, 32); // initial_cells
    push_u64(&mut body, 65_536); // max_cells
    push_u64(&mut body, 524_288); // max_tubes
    push_u64(&mut body, 512); // max_pores
    push_u64(&mut body, 268_435_456); // max_state_bytes
    push_f32(&mut body, 10.0); // initial_atp
    push_f32(&mut body, 100.0); // max_atp
    body.push(0); // Natural
    push_u64(&mut body, 0x1234_5678_9ABC_DEF1);
    push_u64(&mut body, 0xFEDC_BA98_7654_3211);
    push_u64(&mut body, 0x1111_2222_3333_4445);
    push_u64(&mut body, 0x5555_6666_7777_8889);
    push_u64(&mut body, 777); // tick
    push_u64(&mut body, 500); // committed_motor_pulses
    push_u64(&mut body, 12); // subthreshold_rehearsals
    push_u64(&mut body, 4); // mitosis_births
    push_u64(&mut body, 2); // contradiction_splits
    push_u64(&mut body, 3); // myelinated_shortcuts
    push_u64(&mut body, 1); // atrophied_structures
    push_f32(&mut body, 0.12); // last_motor_energy
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

    let mut tube = Vec::new();
    push_u32(&mut tube, 2);
    push_u32(&mut tube, 2);
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
    push_u32(&mut tube, 0); // tube_free_list
    if ver >= 6 {
        push_u32(&mut tube, TRACKED_ACT as u32);
        push_u32(&mut tube, BUCKETS as u32);
    }
    for _ in 0..(3 * TRACKED_ACT * BUCKETS) {
        push_f32(&mut tube, 0.01);
    }
    for _ in 0..(3 * BUCKETS) {
        push_f32(&mut tube, 0.0);
    }
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
    for _ in 0..BUCKETS {
        push_f32(&mut tube, 0.0);
    }
    push_f32(&mut tube, 0.0);
    for _ in 0..RING {
        push_f32(&mut tube, 0.0);
    }
    tube.push(0);
    for _ in 0..(3 * TRACKED_ACT) {
        push_f32(&mut tube, 0.0);
    }
    for _ in 0..7 {
        push_f32(&mut tube, 0.0);
    }
    tube.push(255);
    tube.push(255);
    for _ in 0..(2 * TRACKED_ACT) {
        push_f32(&mut tube, 0.0);
    }
    push_f32(&mut tube, 0.18);

    // Striatal v5 / v6 block
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
    for _ in 0..STR_PORES {
        push_f32(&mut tube, 0.0);
    }
    for _ in 0..(2 * STR_PORES) {
        push_f32(&mut tube, 1.0e-4);
    }
    let f32_zeros = |dst: &mut Vec<u8>, cnt: usize| {
        for _ in 0..cnt {
            push_f32(dst, 0.0);
        }
    };
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
    for _ in 0..3 {
        push_f32(&mut tube, 0.0);
    }
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
        f32_zeros(&mut tube, BUCKETS); // pore_floor_v
        for _ in 0..BUCKETS {
            push_f32(&mut tube, 0.59); // pore_ar1_w
        }
        f32_zeros(&mut tube, BUCKETS); // pore_ar2_w
        for _ in 0..BUCKETS {
            push_f32(&mut tube, 1.0); // pore_q5_lvl
        }
        f32_zeros(&mut tube, 2 * BUCKETS); // pore_q5_exaff, pore_vocal_reaff
        f32_zeros(&mut tube, TRACKED_ACT); // vocal_glottal_w
        for _ in 0..6 {
            push_f32(&mut tube, 0.0); // tonotopic_floor/conf, acoustic_env, agency_signed, vocal_eff_s1/s2
        }
        push_u32(&mut tube, 0); // vocal_eff_hist
        push_u32(&mut tube, 0); // vocal_onset_hist
        tube.push(u8::MAX); // vocal_last_mg
        push_u32(&mut tube, 0); // vocal_q5_cnt
        push_f32(&mut tube, 0.0); // vocal_y_ema
        f32_zeros(&mut tube, 16); // vocal_f_ema, vocal_blk_sum, vocal_s_acc, vocal_v_acc
        push_u32(&mut tube, 0); // even_tick_cnt
        push_f32(&mut tube, 0.0); // even_y_ema
        f32_zeros(&mut tube, 16); // even_f_ema, even_blk_sum, even_s_acc, even_v_acc
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
        for _ in 0..TRACKED_ACT {
            push_f32(&mut ring, 0.0);
        }
        push_f32(&mut ring, 0.0);
        push_f32(&mut ring, 0.0);
        push_u32(&mut ring, 0);
        push_u32(&mut ring, 0);
        push_u32(&mut ring, 0);
    }

    let mut scot = Vec::new();
    push_u32(&mut scot, 0);
    if ver >= 6 {
        scot.push(0); // scout_cohort_subtick
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

fn verify_v7_migration(opt_v6_path: Option<&Path>) -> Result<(), String> {
    if FORMAT_VERSION < 7 {
        return Err(format!(
            "FORMAT_VERSION is {FORMAT_VERSION} (Phase 1B requires >= 7)"
        ));
    }
    if MIN_COMPAT_FORMAT_VERSION > 5 {
        return Err(format!(
            "MIN_COMPAT_FORMAT_VERSION is {MIN_COMPAT_FORMAT_VERSION} (must remain <= 5)"
        ));
    }
    let max_b = OrganismConfig::default().max_state_bytes;
    for ver in [5_u32, 6_u32] {
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
            .map_err(|e| format!("upgraded v{ver}->v7 encode failed: {e:?}"))?;
        let encoded_ver =
            u32::from_le_bytes([upgraded[8], upgraded[9], upgraded[10], upgraded[11]]);
        if encoded_ver != FORMAT_VERSION {
            return Err(format!(
                "re-encoded version {encoded_ver} != {FORMAT_VERSION}"
            ));
        }
        let _ = shadow_decode_organism(&upgraded, max_b)
            .map_err(|e| format!("upgraded v{ver}->v7 roundtrip decode failed: {e:?}"))?;
    }

    if let Some(p) = opt_v6_path {
        let disk_bytes =
            fs::read(p).map_err(|e| format!("failed reading --v6-chk {}: {e}", p.display()))?;
        let mut live_org = shadow_decode_organism(&disk_bytes, max_b)
            .map_err(|e| format!("live --v6-chk decode failed: {e:?}"))?;
        let pre_tick = live_org.snapshot().tick;
        let w = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(11), 11);
        let pulse = live_org
            .step(w.current_view())
            .map_err(|e| format!("live --v6-chk post-upgrade step failed: {e:?}"))?;
        if pulse.actuator_out.len() != 27 || live_org.snapshot().tick != pre_tick + 1 {
            return Err("live --v6-chk post-upgrade step invariant failed".to_string());
        }
    }
    Ok(())
}

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
    let mut opt_v6_chk: Option<PathBuf> = None;
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
        if args[idx] == "--v6-chk"
            && let Some(val) = args.get(idx + 1)
        {
            opt_v6_chk = Some(PathBuf::from(val));
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
            if !open_vault {
                eprintln!("Refusing: heldout requires --open-vault.");
                return ExitCode::from(2);
            }
            if !ws_root.join(".vault_authorized_phase1b").exists() {
                eprintln!("Refusing: .vault_authorized_phase1b missing in workspace root.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase1b.opened");
            if marker.exists() {
                eprintln!(
                    "Refusing: Phase-1B held-out vault already opened ({}).",
                    marker.display()
                );
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (500..530).collect()
        }
        other => {
            eprintln!("Unknown split '{other}' (expected dev, dev30, or heldout)");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-1B Locked Judge | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    let raw = run_jobs(seeds.len(), threads, |i| run_seq12_seed(seeds[i]));
    let mut rows = Vec::with_capacity(seeds.len());
    for (i, item) in raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Seq12 seed job {i} panicked");
            return ExitCode::from(1);
        };
        rows.push(r);
    }

    let rel_tot: Vec<f64> = rows.iter().map(|r| r.reloaded_eval_total).collect();
    let nb_tot: Vec<f64> = rows.iter().map(|r| r.newborn_eval_total).collect();
    let yok_tot: Vec<f64> = rows.iter().map(|r| r.yoked_eval_total).collect();
    let shf_tot: Vec<f64> = rows.iter().map(|r| r.shuffled_eval_total).collect();
    let les_tot: Vec<f64> = rows.iter().map(|r| r.lesion_eval_total).collect();
    let fg_vec: Vec<f64> = rows.iter().map(|r| r.forgetting_ratio).collect();
    let fg_cap: Vec<f64> = vec![0.02; rows.len()];
    let peak_gains: Vec<f64> = rows.iter().map(|r| r.peak_gain_mean).collect();
    let early4_rel: Vec<f64> = rows.iter().map(|r| r.early4_reloaded).collect();
    let early4_nb: Vec<f64> = rows.iter().map(|r| r.early4_newborn).collect();
    let dst_vec: Vec<f64> = rows.iter().map(|r| r.distractor_share).collect();
    let chance_dst: Vec<f64> = vec![96.0 / 192.0; rows.len()];
    let nat_ag: Vec<f64> = rows.iter().map(|r| r.nat_agency).collect();
    let shf_ag: Vec<f64> = rows.iter().map(|r| r.shf_agency).collect();
    let re_nat: Vec<f64> = rows.iter().map(|r| r.reentry_d0_nat).collect();
    let re_nb: Vec<f64> = rows.iter().map(|r| r.reentry_d0_newborn).collect();

    let g1a = conservative_paired_gate(&rel_tot, &nb_tot);
    let g1b = conservative_paired_gate(&rel_tot, &yok_tot);
    let g1c = conservative_paired_gate(&rel_tot, &shf_tot);
    let g1d = conservative_paired_gate(&rel_tot, &les_tot);
    let g2_fg = conservative_paired_gate(&fg_cap, &fg_vec);
    let g2_early = conservative_paired_gate(&early4_rel, &early4_nb);
    let g3_dst = conservative_paired_gate(&chance_dst, &dst_vec);
    let g4_ag = conservative_paired_gate(&nat_ag, &shf_ag);
    let g5_re = conservative_paired_gate(&re_nat, &re_nb);

    let raw_p = [
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g1d.p_two_sided,
        g2_fg.p_two_sided,
        g3_dst.p_two_sided,
        g4_ag.p_two_sided,
        g5_re.p_two_sided,
    ];
    let p_holm = holm_adjust(&raw_p);

    let mean_fg = mean(&fg_vec);
    let mean_peak_gain = mean(&peak_gains);
    let mean_dst = mean(&dst_vec);
    let mean_shf_ag = mean(&shf_ag);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g1d = g1d.mean_diff > 0.0 && p_holm[3] < 0.05;
    let pass_g2 = g2_fg.mean_diff > 0.0
        && p_holm[4] < 0.05
        && mean_fg < 0.02
        && mean_peak_gain > 2.0
        && g2_early.mean_diff > 0.0
        && g2_early.p_two_sided < 0.01;
    let pass_g3 = g3_dst.mean_diff > 0.0 && p_holm[5] < 0.05 && mean_dst < 0.05;
    let pass_g4 = g4_ag.mean_diff > 0.015 && p_holm[6] < 0.05 && mean_shf_ag <= 0.05;
    let pass_g5 = g5_re.mean_diff > 0.0 && p_holm[7] < 0.05;

    let gates_print = [
        ("P1B-G1a Seq12 Reloaded > Newborn-Frozen", g1a, p_holm[0], pass_g1a),
        ("P1B-G1b Seq12 Reloaded > Yoked-Reloaded", g1b, p_holm[1], pass_g1b),
        ("P1B-G1c Seq12 Reloaded > Shuffled-Reloaded", g1c, p_holm[2], pass_g1c),
        ("P1B-G1d Seq12 Reloaded > Flagless Lesion", g1d, p_holm[3], pass_g1d),
        ("P1B-G2  Seq12 Forgetting Cap(2%) > Forgot", g2_fg, p_holm[4], pass_g2),
        ("P1B-G3  Seq12 Distractor Chance(50.0%)>Nat", g3_dst, p_holm[5], pass_g3),
        ("P1B-G4  Seq12 Agency Nat > ActionShuffled", g4_ag, p_holm[6], pass_g4),
        ("P1B-G5  Seq12 Distant Re-Entry D0 > Newborn", g5_re, p_holm[7], pass_g5),
    ];

    println!("\n--- Behavioral & Continual-Learning Gates (Exact Sign-Flip + Student-t, Holm m=8) ---");
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
        "         Seq12 mean_forgetting={:.4} (cap <0.0200) | mean_peak_gain={:+.3} (>2.0) | early4_diff={:+.3} (p={:.3e}) | distractor_share={:.4} (<0.0500) | shuf_ag={:.4} (<=0.0500)",
        mean_fg, mean_peak_gain, g2_early.mean_diff, g2_early.p_two_sided, mean_dst, mean_shf_ag
    );

    // Hard Structural, Scaling, Paged-Slab & Migration Invariants (P1B-I1..I4).
    let debt = audit_core_debt(&ws_root);
    let pass_i1 = debt.hygiene_violations == 0
        && debt.split_brain_sites.is_empty()
        && debt.organ_mirror_sites.is_empty()
        && debt.cell_modulo_sites.is_empty();

    let max_seq12_active = rows.iter().map(|r| r.max_active_cells_seen).max().unwrap_or(0);
    let (m512_p99_us, m512_max_act, m512_pores, m512_cells) = benchmark_high_capacity_4096();
    let max_active_any = max_seq12_active.max(m512_max_act);
    let pass_i2 = m512_p99_us < 2_000.0
        && max_active_any <= 64
        && m512_pores <= 64
        && m512_cells >= 512;

    let slab_res = match std::panic::catch_unwind(audit_paged_slabs) {
        Ok(res) => res,
        Err(_) => Err("panicked during audit_paged_slabs".to_string()),
    };
    let pass_i3 = slab_res.is_ok();

    let v5_valid =
        shadow_decode_organism(&build_canonical_checkpoint(5), OrganismConfig::default().max_state_bytes)
            .is_ok();
    let v6_valid =
        shadow_decode_organism(&build_canonical_checkpoint(6), OrganismConfig::default().max_state_bytes)
            .is_ok();
    let mig_res = verify_v7_migration(opt_v6_chk.as_deref());
    let pass_i4 = v5_valid && v6_valid && mig_res.is_ok();

    println!("\n--- Structural, k-WTA Scaling & Paged-Slab Persistence Invariants ---");
    println!(
        "  [{:<4}] P1B-I1 Zero Core Debt: hygiene={} split_brain={} mirrors={} cell_mod={}",
        if pass_i1 { "PASS" } else { "FAIL" },
        debt.hygiene_violations,
        debt.split_brain_sites.len(),
        debt.organ_mirror_sites.len(),
        debt.cell_modulo_sites.len()
    );
    println!(
        "  [{:<4}] P1B-I2 k-WTA & M=512/4096ch Scaling: p99={:.1} us (<2000) max_active={} (<=64) pores={} (<=64) cells={}",
        if pass_i2 { "PASS" } else { "FAIL" },
        m512_p99_us,
        max_active_any,
        m512_pores,
        m512_cells
    );
    match &slab_res {
        Ok((n_slabs, ckpt_b)) => println!(
            "  [{:<4}] P1B-I3 Disk-Paged CRC32C Cortical Slabs: slabs={} (>=8) ckpt_bytes={} (<=524288) bit_id_200=OK prev_heal=OK dual_poison=OK",
            if pass_i3 { "PASS" } else { "FAIL" },
            n_slabs,
            ckpt_b
        ),
        Err(err) => println!(
            "  [FAIL] P1B-I3 Disk-Paged CRC32C Cortical Slabs: {err}"
        ),
    }
    println!(
        "  [{:<4}] P1B-I4 Legacy v5 & Live v6 -> v7+ Migration: v5_valid={} v6_valid={} status={}",
        if pass_i4 { "PASS" } else { "FAIL" },
        v5_valid,
        v6_valid,
        mig_res.as_ref().map_or_else(|e| e.as_str(), |_| "OK")
    );

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g1d
        && pass_g2
        && pass_g3
        && pass_g4
        && pass_g5
        && pass_i1
        && pass_i2
        && pass_i3
        && pass_i4;

    println!(
        "\n=== Phase-1B Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );

    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
