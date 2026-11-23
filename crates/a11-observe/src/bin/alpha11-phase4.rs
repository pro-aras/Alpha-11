//! Pre-registered locked Phase-4 Grounded Compositional Language judge (`P4-G1..G6` + `P4-I1..I4`).
//!
//! Pure multimodal zero-leak `FieldView -> FieldPulse` portal (no tokenizers, no string vocabularies,
//! no reward hints inside `a11-core`, no 3D terrarium organs):
//! - `192` `WirePerm24`-permuted channels (`0..16` raw 8-bit UTF-8 differential bit lanes,
//!   `16..48` 16 kHz `PhysicalVocalTract` + `ErbCochlea32` cochlea channels, `48..64` grounded event
//!   manners, `64..80` grounded object referents, `80..96` nonlinear proprioceptive parallax,
//!   `96..128` thermal floor, `128..160` OU Noisy-TV distractors, `160..192` delayed self-copy).
//! - `24` `WirePerm24`-permuted actuators (`12` opponent pairs: `G_0..G_3` = `Manner_0..3`,
//!   `G_4..G_7` = `Target_0..3`, `G_8..G_11` `+/-` = `8` articulatory/UTF-8 phoneme keys).
//! - `SCAN`/`COGS` systematic recombination split (`4` Verbs x `4` Nouns = `16` clauses; `12` Seen vs
//!   `4` Unseen held-out diagonal pairs) across Comprehension (`Bimodal`, `UTF-8 Byte-Only`, `16 kHz Audio-Only`)
//!   and Grounded Production (`scene event -> 4-phoneme/byte UTF-8 + 16 kHz spoken clause`).
//! - Gates (`Holm m=10`, conservative exact sign-flip / MC + Student-t `p_holm < 0.05`):
//!   `P4-G1a..c` Seen occluded mastery, `P4-G2a..b` Zero-Shot Compositional Comprehension,
//!   `P4-G3a..b` Zero-Shot Compositional Production, `P4-G4` ABAB syntactic/lexical reversal savings,
//!   `P4-G5` Distractor immunity (<5%), `P4-G6` Interventional agency purity, and `P4-I1..I4` invariants.
//!
//! Usage: `alpha11-phase4 [dev|dev30|heldout] [--threads N] [--open-vault] [--live-chk <path>]`

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
use a11_portals::{ErbCochlea32, PhysicalVocalTract, VOCAL_ACTUATORS, VOCAL_ERB_BANDS, VOCAL_FRAME_SAMPLES};
use a11_worlds::{DenseFieldWorld, DeterministicRng, YokedMedium};

const P4_CHANNELS: usize = 192;
const P4_ACTUATORS: usize = 24;
const DISTRACTOR_START: usize = 128;
const OU_CHANNELS: usize = 32;
const SELF_COPY_CHANNELS: usize = 32;

const NUM_GROUPS: usize = 12;
const MANIP_GROUPS: usize = 8;
const NUM_VERBS: usize = 4;
const NUM_NOUNS: usize = 4;

const COMP_BOUT_LEN: u64 = 30;
const PROD_BOUT_LEN: u64 = 22;

const STAGE1_SEEN_BOUTS: usize = 288;
const STAGE2_SEEN_BOUTS: usize = 36;
const UNSEEN_COMP_BOUTS: usize = 24;
const UNSEEN_PROD_BOUTS: usize = 16;
const ABAB_B_BOUTS: usize = 192;
const ABAB_RET_BOUTS: usize = 36;

const PHONEME_BYTES: [u8; 8] = [b'p', b't', b'k', b's', b'a', b'i', b'u', b'o'];

const P4_SENSOR_SALT: u64 = 0x4A11_5E45_0000_2026;
const P4_MOTOR_SALT: u64 = 0x4A11_4D07_0000_2026;
const P4_LEX_SALT: u64 = 0x4A11_1E3C_1111_0001;
const P4_STAGE1_SALT: u64 = 0x4A11_57A1_1111_0001;
const P4_SEEN2_SALT: u64 = 0x4A11_57A2_2222_0002;
const P4_ZCOMP_SALT: u64 = 0x4A11_2C03_3333_0003;
const P4_ZPROD_SALT: u64 = 0x4A11_2B04_4444_0004;
const P4_ABAB_B_SALT: u64 = 0x4A11_BBBB_5555_0005;
const P4_ABAB_RET_SALT: u64 = 0x4A11_AAAA_6666_0006;
const P4_SHUF_SALT: u64 = 0x4A11_5A17_9999_8888;

#[inline]
fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

#[derive(Clone, Debug)]
struct WirePerm24 {
    sensor_perm: [usize; P4_CHANNELS],
    act_perm: [usize; P4_ACTUATORS],
}

impl WirePerm24 {
    fn from_seed(seed: u64) -> Self {
        let mut s_rng = DeterministicRng::new(seed ^ P4_SENSOR_SALT);
        let mut sensor_perm: [usize; P4_CHANNELS] = std::array::from_fn(|i| i);
        for i in (1..P4_CHANNELS).rev() {
            let j = (s_rng.next_u64() as usize) % (i + 1);
            sensor_perm.swap(i, j);
        }
        let mut m_rng = DeterministicRng::new(seed ^ P4_MOTOR_SALT);
        let mut act_perm: [usize; P4_ACTUATORS] = std::array::from_fn(|i| i);
        for i in (1..P4_ACTUATORS).rev() {
            let j = (m_rng.next_u64() as usize) % (i + 1);
            act_perm.swap(i, j);
        }
        Self { sensor_perm, act_perm }
    }

    fn distractor_conductance_share(&self, channels: &[u32], conductances: &[f32]) -> f64 {
        let mut tot = 0.0_f64;
        let mut dst = 0.0_f64;
        for (&ch, &g) in channels.iter().zip(conductances.iter()) {
            let sc = ch as usize;
            if sc < P4_CHANNELS && g.is_finite() && g > 0.0 {
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

#[inline]
fn phoneme_vocal_target(phoneme_idx: usize, sub: u64) -> [f32; VOCAL_ACTUATORS] {
    let release = sub >= 2;
    match phoneme_idx & 7 {
        0 => {
            if release {
                [0.85, -0.70, 0.20, -0.75, -0.30]
            } else {
                [0.85, -0.70, -0.45, -0.75, 0.55]
            }
        }
        1 => {
            if release {
                [0.85, 0.00, 0.15, 0.35, -0.10]
            } else {
                [0.85, 0.00, -0.55, 0.35, -0.35]
            }
        }
        2 => {
            if release {
                [0.85, 0.60, 0.20, -0.20, 0.05]
            } else {
                [0.85, 0.60, -0.55, -0.20, 0.45]
            }
        }
        3 => {
            if release {
                [0.85, 0.90, 0.05, 0.85, 0.10]
            } else {
                [0.85, 0.90, -0.30, 0.85, 0.55]
            }
        }
        4 => [0.85, -0.80, 0.95, -0.10, -0.50],
        5 => [0.85, 0.85, -0.05, 0.95, 0.10],
        6 => [0.85, -0.25, -0.05, -0.95, -0.85],
        _ => [0.85, 0.35, 0.50, -0.80, -0.20],
    }
}

#[derive(Clone, Copy, Debug)]
struct LexiconSpec {
    verbs: [[usize; 2]; NUM_VERBS],
    nouns: [[usize; 2]; NUM_NOUNS],
    diag_shift: usize,
}

impl LexiconSpec {
    fn from_seed(seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed ^ P4_LEX_SALT);
        let base_verbs = [[0usize, 4], [1, 5], [2, 6], [3, 7]]; // pa, ti, ku, so
        let base_nouns = [[0usize, 7], [1, 6], [2, 4], [3, 5]]; // po, tu, ka, si
        let mut vp = [0usize, 1, 2, 3];
        let mut np = [0usize, 1, 2, 3];
        for i in (1..4).rev() {
            vp.swap(i, (rng.next_u64() as usize) % (i + 1));
            np.swap(i, (rng.next_u64() as usize) % (i + 1));
        }
        let diag_shift = (rng.next_u64() as usize) % 4;
        Self {
            verbs: std::array::from_fn(|i| base_verbs[vp[i]]),
            nouns: std::array::from_fn(|i| base_nouns[np[i]]),
            diag_shift,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Modality {
    Bimodal,
    ByteOnly,
    AudioOnly,
}

#[derive(Clone, Copy, Debug)]
struct LangBout {
    is_production: bool,
    modality: Modality,
    verb: usize,
    noun: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoutPool {
    SeenInterleaved,
    UnseenCompOnly,
    UnseenProdOnly,
}

fn build_schedule(
    seed: u64,
    salt: u64,
    lex: LexiconSpec,
    n_bouts: usize,
    pool: BoutPool,
) -> Vec<LangBout> {
    let mut rng = DeterministicRng::new(seed ^ salt);
    let mut seen_pairs: Vec<(usize, usize)> = Vec::with_capacity(12);
    let mut unseen_pairs: Vec<(usize, usize)> = Vec::with_capacity(4);
    for v in 0..NUM_VERBS {
        let held = (v + lex.diag_shift) % NUM_NOUNS;
        for n in 0..NUM_NOUNS {
            if n == held {
                unseen_pairs.push((v, n));
            } else {
                seen_pairs.push((v, n));
            }
        }
    }

    let shuffle_pairs = |arr: &mut Vec<(usize, usize)>, r: &mut DeterministicRng| {
        for i in (1..arr.len()).rev() {
            let j = (r.next_u64() as usize) % (i + 1);
            arr.swap(i, j);
        }
    };

    let modalities = [Modality::Bimodal, Modality::ByteOnly, Modality::AudioOnly];
    let mut out = Vec::with_capacity(n_bouts);

    match pool {
        BoutPool::SeenInterleaved => {
            let mut pair_bag = seen_pairs.clone();
            shuffle_pairs(&mut pair_bag, &mut rng);
            let mut p_cur = 0usize;
            for i in 0..n_bouts {
                if p_cur >= pair_bag.len() {
                    shuffle_pairs(&mut pair_bag, &mut rng);
                    p_cur = 0;
                }
                let (verb, noun) = pair_bag[p_cur];
                p_cur += 1;
                let is_production = (i % 3) == 2;
                let comp_idx = i - (i / 3);
                let modality = if is_production {
                    Modality::Bimodal
                } else {
                    modalities[comp_idx % 3]
                };
                out.push(LangBout {
                    is_production,
                    modality,
                    verb,
                    noun,
                });
            }
        }
        BoutPool::UnseenCompOnly => {
            // Balance every (unseen_pair x modality) across Unseen Comprehension
            let mut items: Vec<(usize, usize, Modality)> = Vec::with_capacity(12);
            for &(v, n) in &unseen_pairs {
                for &m in &modalities {
                    items.push((v, n, m));
                }
            }
            for i in 0..n_bouts {
                if i % items.len() == 0 {
                    for k in (1..items.len()).rev() {
                        let j = (rng.next_u64() as usize) % (k + 1);
                        items.swap(k, j);
                    }
                }
                let (verb, noun, modality) = items[i % items.len()];
                out.push(LangBout {
                    is_production: false,
                    modality,
                    verb,
                    noun,
                });
            }
        }
        BoutPool::UnseenProdOnly => {
            let mut bag = unseen_pairs.clone();
            for i in 0..n_bouts {
                if i % bag.len() == 0 {
                    shuffle_pairs(&mut bag, &mut rng);
                }
                let (verb, noun) = bag[i % bag.len()];
                out.push(LangBout {
                    is_production: true,
                    modality: Modality::Bimodal,
                    verb,
                    noun,
                });
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
struct GroundedCompositionalLanguageMedium {
    perm: WirePerm24,
    lex: LexiconSpec,
    occluded_stage2: bool,
    context_b: bool,
    schedule: Vec<LangBout>,
    bout_idx: usize,
    b_tick: u64,
    step_pos: usize,
    failed_bout: bool,
    completed_bouts: usize,
    completed_by_mod: [usize; 3],
    total_by_mod: [usize; 3],
    global_tick: u64,
    vel_grp: [f32; NUM_GROUPS],
    pos_grp: [f32; NUM_GROUPS],
    active_self_phoneme: Option<(usize, f32)>,
    self_art_ticks: u64,
    ext_vocal: PhysicalVocalTract,
    self_vocal: PhysicalVocalTract,
    cochlea: ErbCochlea32,
    last_influx: f32,
    last_stress: f32,
    net_valence_sum: f64,
    ou_rng: DeterministicRng,
    ou_state: [f32; OU_CHANNELS],
    motor_ring: [[f32; P4_ACTUATORS]; 3],
    ring_cursor: usize,
    raw_ambient: [f32; P4_CHANNELS],
    permuted_ambient: [f32; P4_CHANNELS],
}

impl GroundedCompositionalLanguageMedium {
    fn new(
        seed: u64,
        salt: u64,
        n_bouts: usize,
        pool: BoutPool,
        occluded_stage2: bool,
        context_b: bool,
        perm: WirePerm24,
    ) -> Self {
        let lex = LexiconSpec::from_seed(seed);
        let schedule = build_schedule(seed, salt, lex, n_bouts, pool);
        let mut total_by_mod = [0usize; 3];
        for b in &schedule {
            if !b.is_production {
                let midx = match b.modality {
                    Modality::Bimodal => 0,
                    Modality::ByteOnly => 1,
                    Modality::AudioOnly => 2,
                };
                total_by_mod[midx] += 1;
            }
        }
        let mut m = Self {
            perm,
            lex,
            occluded_stage2,
            context_b,
            schedule,
            bout_idx: 0,
            b_tick: 0,
            step_pos: 0,
            failed_bout: false,
            completed_bouts: 0,
            completed_by_mod: [0; 3],
            total_by_mod,
            global_tick: 0,
            vel_grp: [0.0; NUM_GROUPS],
            pos_grp: [0.0; NUM_GROUPS],
            active_self_phoneme: None,
            self_art_ticks: 0,
            ext_vocal: PhysicalVocalTract::new_adult(seed ^ salt ^ 0xE770_C001),
            self_vocal: PhysicalVocalTract::new_infant(seed ^ salt ^ 0x5E1F_C002),
            cochlea: ErbCochlea32::new(),
            last_influx: 0.0,
            last_stress: 0.0,
            net_valence_sum: 0.0,
            ou_rng: DeterministicRng::new(seed ^ salt ^ 0x0011_9999),
            ou_state: [0.0; OU_CHANNELS],
            motor_ring: [[0.0; P4_ACTUATORS]; 3],
            ring_cursor: 0,
            raw_ambient: [0.0; P4_CHANNELS],
            permuted_ambient: [0.0; P4_CHANNELS],
        };
        m.refresh_ambient();
        m
    }

    fn total_ticks(&self) -> u64 {
        self.schedule
            .iter()
            .map(|b| if b.is_production { PROD_BOUT_LEN } else { COMP_BOUT_LEN })
            .sum()
    }

    fn current_bout(&self) -> LangBout {
        self.schedule[self.bout_idx % self.schedule.len().max(1)]
    }

    fn current_bout_len(&self) -> u64 {
        if self.current_bout().is_production {
            PROD_BOUT_LEN
        } else {
            COMP_BOUT_LEN
        }
    }

    fn clause_syllables(&self, bout: LangBout) -> ([usize; 2], [usize; 2], usize, usize) {
        let (v_eff, n_eff) = if self.context_b {
            ((bout.verb + 1) % NUM_VERBS, (bout.noun + 2) % NUM_NOUNS)
        } else {
            (bout.verb, bout.noun)
        };
        let vsym = self.lex.verbs[v_eff];
        let nsym = self.lex.nouns[n_eff];
        if self.context_b {
            (nsym, vsym, bout.verb, bout.noun)
        } else {
            (vsym, nsym, bout.verb, bout.noun)
        }
    }

    fn refresh_ambient(&mut self) {
        self.raw_ambient.fill(0.0);
        let bout = self.current_bout();
        let (w1, w2, eff_v, eff_n) = self.clause_syllables(bout);
        let tau = std::f32::consts::TAU;
        let t_f = self.global_tick as f32;

        // 1. Determine external UTF-8 byte + external 16 kHz vocal command (Comprehension only).
        let mut ext_byte_drive: Option<(u8, f32)> = None;
        let mut ext_vocal_target: [f32; VOCAL_ACTUATORS] = [0.0; VOCAL_ACTUATORS];

        if !bout.is_production && self.b_tick < 22 {
            let allow_bytes = !matches!(bout.modality, Modality::AudioOnly);
            let allow_audio = !matches!(bout.modality, Modality::ByteOnly);

            let (opt_ph, sub) = if self.b_tick < 5 {
                (Some(w1[0]), self.b_tick)
            } else if self.b_tick < 10 {
                (Some(w1[1]), self.b_tick - 5)
            } else if self.b_tick < 12 {
                (None, self.b_tick - 10)
            } else if self.b_tick < 17 {
                (Some(w2[0]), self.b_tick - 12)
            } else {
                (Some(w2[1]), self.b_tick - 17)
            };

            if let Some(ph_idx) = opt_ph {
                let u = ((sub as f32) + 0.5) / 5.0;
                let env = (0.5 * (1.0 - (tau * u).cos())).clamp(0.0, 1.0);
                if allow_bytes {
                    ext_byte_drive = Some((PHONEME_BYTES[ph_idx], env));
                }
                if allow_audio {
                    ext_vocal_target = phoneme_vocal_target(ph_idx, sub);
                }
            } else if allow_bytes {
                // Smooth prosodic whitespace delimiter (`0x20`) between Constituency 1 and 2.
                ext_byte_drive = Some((b' ', 0.35));
            }
        }

        // 2. Self-reafference from organism articulatory keys G_8..G_{11}.
        let mut self_byte_drive: Option<(u8, f32)> = None;
        let mut self_vocal_target: [f32; VOCAL_ACTUATORS] = [0.0; VOCAL_ACTUATORS];
        if let Some((ph_idx, drive_amp)) = self.active_self_phoneme {
            let amp = (0.85 * drive_amp).clamp(0.0, 1.0);
            self_byte_drive = Some((PHONEME_BYTES[ph_idx], amp));
            self_vocal_target = phoneme_vocal_target(ph_idx, self.self_art_ticks % 5);
        }

        // Apply UTF-8 8-bit differential lanes (0..16)
        let mut emit_byte = |b: u8, env: f32| {
            for k in 0..8 {
                let bit = ((b >> k) & 1) as f32;
                let v1 = (env * bit).clamp(0.0, 1.0);
                let v0 = (env * (1.0 - bit)).clamp(0.0, 1.0);
                self.raw_ambient[2 * k] = (self.raw_ambient[2 * k] + v1).clamp(0.0, 1.0);
                self.raw_ambient[2 * k + 1] = (self.raw_ambient[2 * k + 1] + v0).clamp(0.0, 1.0);
            }
        };
        if let Some((eb, ee)) = ext_byte_drive {
            emit_byte(eb, ee);
        }
        if let Some((sb, se)) = self_byte_drive {
            emit_byte(sb, se);
        }

        // Synthesize 16 kHz room air PCM (external + self) and transduce through ErbCochlea32 (16..48)
        let mut ext_pcm = [0.0_f32; VOCAL_FRAME_SAMPLES];
        let mut self_pcm = [0.0_f32; VOCAL_FRAME_SAMPLES];
        self.ext_vocal.synthesize_frame(&ext_vocal_target, &mut ext_pcm);
        self.self_vocal.synthesize_frame(&self_vocal_target, &mut self_pcm);
        let mut air_pcm = [0.0_f32; VOCAL_FRAME_SAMPLES];
        for i in 0..VOCAL_FRAME_SAMPLES {
            air_pcm[i] = clamp_unit(ext_pcm[i] + self_pcm[i]);
        }
        let (fast_erb, _, _) = self.cochlea.transduce_frame(&air_pcm);
        for b in 0..VOCAL_ERB_BANDS {
            self.raw_ambient[16 + b] = (fast_erb[b] + 0.35).clamp(0.0, 1.0);
        }

        // 3. Grounded Physical Event-Manner (48..64) & Object-Referent (64..80) channels
        let prompt_env = if bout.is_production && self.b_tick < 10 {
            let u = ((self.b_tick as f32) + 0.5) / 10.0;
            (0.85 * 0.5 * (1.0 - (tau * u).cos())).clamp(0.0, 0.90)
        } else {
            0.0
        };

        for m in 0..NUM_VERBS {
            let reaf = self.vel_grp[m].max(0.0);
            let ext = if m == eff_v { prompt_env } else { 0.0 };
            let val = (ext + 0.65 * reaf).clamp(0.0, 1.0);
            let base = 48 + m * 4;
            for ch in base..(base + 4) {
                self.raw_ambient[ch] = val;
            }
        }
        for n in 0..NUM_NOUNS {
            let reaf = self.vel_grp[4 + n].max(0.0);
            let ext = if n == eff_n { prompt_env } else { 0.0 };
            let val = (ext + 0.65 * reaf).clamp(0.0, 1.0);
            let base = 64 + n * 4;
            for ch in base..(base + 4) {
                self.raw_ambient[ch] = val;
            }
        }

        // 4. Mirror-paired smooth nonlinear parallax reafference (80..96, 16 ch)
        for p in 0..8 {
            let ga = p % NUM_GROUPS;
            let gb = (p + 5) % NUM_GROUPS;
            let fa = 1.65 + 0.21 * ((p % 4) as f32);
            let fb = 1.45 + 0.27 * (((p / 4) % 3) as f32);
            let off = 0.37 * (p as f32);
            let vp = 0.45 + 0.36 * (fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            let vm = 0.45 + 0.36 * (-fa * self.pos_grp[ga] + fb * self.pos_grp[gb] + off).cos();
            self.raw_ambient[80 + 2 * p] = vp.clamp(0.0, 1.0);
            self.raw_ambient[80 + 2 * p + 1] = vm.clamp(0.0, 1.0);
        }

        // 5. Quiet thermal background (96..128)
        for ch in 96..DISTRACTOR_START {
            self.raw_ambient[ch] = 0.015 * (0.019 * t_f + 0.11 * (ch as f32)).sin();
        }

        // 6. High-variance Ornstein-Uhlenbeck distractors (128..160)
        for i in 0..OU_CHANNELS {
            let z = self.ou_rng.next_f32_range(-1.0, 1.0);
            self.ou_state[i] = (0.35 * self.ou_state[i] + 0.65 * z).clamp(-1.0, 1.0);
            self.raw_ambient[DISTRACTOR_START + i] = self.ou_state[i];
        }

        // 7. Delayed motor self-copy distractors (160..192)
        let lag1 = (self.ring_cursor + 2) % 3;
        let lag2 = (self.ring_cursor + 1) % 3;
        for i in 0..SELF_COPY_CHANNELS {
            let a1 = self.motor_ring[lag1][i % P4_ACTUATORS];
            let a2 = self.motor_ring[lag2][(i + 5) % P4_ACTUATORS];
            self.raw_ambient[DISTRACTOR_START + OU_CHANNELS + i] =
                (0.65 * a1 - 0.30 * a2).clamp(-1.0, 1.0);
        }

        for (sc_ch, &tr_ch) in self.perm.sensor_perm.iter().enumerate() {
            self.permuted_ambient[sc_ch] = self.raw_ambient[tr_ch];
        }
    }
}

impl WorldMedium for GroundedCompositionalLanguageMedium {
    fn num_ambient_channels(&self) -> usize {
        P4_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        P4_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.permuted_ambient,
            metabolic_influx: self.last_influx,
            tissue_stress: self.last_stress,
            num_actuators: P4_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let mut logical_act = [0.0_f32; P4_ACTUATORS];
        for (log_idx, &phys_idx) in self.perm.act_perm.iter().enumerate() {
            logical_act[log_idx] = clamp_unit(pulse.actuator_out.get(phys_idx).copied().unwrap_or(0.0));
        }
        self.motor_ring[self.ring_cursor] = logical_act;
        self.ring_cursor = (self.ring_cursor + 1) % 3;

        for g in 0..NUM_GROUPS {
            let u_g = 0.5 * (logical_act[2 * g] + logical_act[2 * g + 1]);
            self.vel_grp[g] = (0.25 * self.vel_grp[g] + 0.75 * u_g).clamp(-1.0, 1.0);
            self.pos_grp[g] = (0.72 * self.pos_grp[g] + 0.28 * self.vel_grp[g]).clamp(-1.0, 1.0);
        }

        // Winner-take-all manipulation channel G_0..G_7
        let mut best_manip_g = 0usize;
        let mut best_manip_val = self.vel_grp[0];
        let mut second_manip_val = f32::NEG_INFINITY;
        for g in 1..MANIP_GROUPS {
            let v = self.vel_grp[g];
            if v > best_manip_val {
                second_manip_val = best_manip_val;
                best_manip_val = v;
                best_manip_g = g;
            } else if v > second_manip_val {
                second_manip_val = v;
            }
        }
        let clear_manip = best_manip_val > 0.14 && (best_manip_val - second_manip_val) > 0.06;

        // Winner-take-all bipolar articulatory phoneme key G_8..G_{11}
        let mut best_art_k = 0usize;
        let mut best_art_abs = 0.0_f32;
        let mut second_art_abs = 0.0_f32;
        for g in MANIP_GROUPS..NUM_GROUPS {
            let u = self.vel_grp[g];
            let ua = u.abs();
            let key_idx = (g - MANIP_GROUPS) * 2 + usize::from(u < 0.0);
            if ua > best_art_abs {
                second_art_abs = best_art_abs;
                best_art_abs = ua;
                best_art_k = key_idx;
            } else if ua > second_art_abs {
                second_art_abs = ua;
            }
        }
        let clear_art = best_art_abs > 0.35 && (best_art_abs - second_art_abs) > 0.06;
        if clear_art {
            if self.active_self_phoneme.is_some_and(|(p, _)| p == best_art_k) {
                self.self_art_ticks = self.self_art_ticks.saturating_add(1);
            } else {
                self.self_art_ticks = 0;
            }
            self.active_self_phoneme = Some((best_art_k, best_art_abs));
        } else {
            self.active_self_phoneme = None;
            self.self_art_ticks = 0;
        }

        let bout = self.current_bout();
        let (w1, w2, eff_v, eff_n) = self.clause_syllables(bout);
        let mut influx = 0.0_f32;
        let mut stress = 0.0_f32;

        if !self.failed_bout {
            if !bout.is_production {
                // Comprehension response window (`b_tick >= 22`)
                if self.b_tick >= 22 && self.step_pos < 2 {
                    let target_seq = if self.context_b {
                        [4 + eff_n, eff_v]
                    } else {
                        [eff_v, 4 + eff_n]
                    };
                    if clear_manip {
                        if best_manip_g == target_seq[self.step_pos] {
                            self.step_pos += 1;
                            if self.step_pos == 2 {
                                self.completed_bouts += 1;
                                let midx = match bout.modality {
                                    Modality::Bimodal => 0,
                                    Modality::ByteOnly => 1,
                                    Modality::AudioOnly => 2,
                                };
                                self.completed_by_mod[midx] += 1;
                                influx = if self.occluded_stage2 { 3.0 } else { 2.4 };
                            } else if !self.occluded_stage2 {
                                influx = 0.40;
                            }
                        } else if !(self.step_pos > 0 && best_manip_g == target_seq[self.step_pos - 1]) {
                            if self.occluded_stage2 {
                                self.failed_bout = true;
                                stress = 0.25;
                            } else {
                                stress = 0.12;
                            }
                        }
                    } else if clear_art {
                        if self.occluded_stage2 {
                            self.failed_bout = true;
                            stress = 0.25;
                        } else {
                            stress = 0.12;
                        }
                    }
                }
            } else {
                // Production response window (`b_tick >= 10`)
                if self.b_tick >= 10 && self.step_pos < 4 {
                    let target_seq = [w1[0], w1[1], w2[0], w2[1]];
                    if clear_art {
                        if best_art_k == target_seq[self.step_pos] {
                            self.step_pos += 1;
                            if self.step_pos == 4 {
                                self.completed_bouts += 1;
                                influx = if self.occluded_stage2 { 3.0 } else { 2.4 };
                            } else if !self.occluded_stage2 {
                                influx = 0.40;
                            }
                        } else if !(self.step_pos > 0 && best_art_k == target_seq[self.step_pos - 1]) {
                            if self.occluded_stage2 {
                                self.failed_bout = true;
                                stress = 0.25;
                            } else {
                                stress = 0.12;
                            }
                        }
                    } else if clear_manip {
                        if self.occluded_stage2 {
                            self.failed_bout = true;
                            stress = 0.25;
                        } else {
                            stress = 0.12;
                        }
                    }
                }
            }
        }

        self.last_influx = influx;
        self.last_stress = stress;
        self.net_valence_sum += f64::from(influx - stress);

        let blen = self.current_bout_len();
        self.global_tick += 1;
        self.b_tick += 1;
        if self.b_tick >= blen {
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

#[derive(Clone, Copy, Debug)]
struct Phase4SeedRow {
    seen_nat: f64,
    seen_frz: f64,
    seen_yok: f64,
    seen_shuf: f64,
    seen_nat_acc: f64,
    comp_zs_tr: f64,
    comp_zs_nb: f64,
    comp_zs_les: f64,
    comp_zs_acc: f64,
    byte_only_zs_acc: f64,
    audio_only_zs_acc: f64,
    prod_zs_tr: f64,
    prod_zs_nb: f64,
    prod_zs_les: f64,
    prod_zs_acc: f64,
    abab_ret_nat: f64,
    abab_newborn: f64,
    distractor_share: f64,
    nat_agency: f64,
    shf_agency: f64,
    stage2_frz_drift: bool,
}

fn run_phase4_seed(seed: u64) -> Phase4SeedRow {
    let perm = WirePerm24::from_seed(seed);

    // Arm 1: Natural interactive development on Seen 12 pairs only
    let mut nat_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w1_nat = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_STAGE1_SALT,
        STAGE1_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        false,
        false,
        perm.clone(),
    );
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
    let nat_snap = nat_org.snapshot();
    let distractor_share = perm.distractor_conductance_share(
        &nat_snap.sensor_pore_channels,
        &nat_snap.sensor_pore_conductances,
    );
    let nat_agency = f64::from(nat_snap.mean_interventional_agency);

    // Stage 2a Seen Occluded Mastery (`P4-G1a..c`)
    let mut w2_nat = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_SEEN2_SALT,
        STAGE2_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm.clone(),
    );
    let t_s2 = w2_nat.total_ticks();
    step_medium(&mut nat_org, &mut w2_nat, t_s2);
    let seen_nat = w2_nat.net_valence_sum;
    let seen_nat_acc = (w2_nat.completed_bouts as f64) / (STAGE2_SEEN_BOUTS as f64);

    // Arm 2: FrozenPlasticity control
    let mut frz_org = Alpha11Organism::new(seed, OrganismConfig::default());
    frz_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w1_frz = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_STAGE1_SALT,
        STAGE1_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        false,
        false,
        perm.clone(),
    );
    step_medium(&mut frz_org, &mut w1_frz, t_s1);
    let mut w2_frz = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_SEEN2_SALT,
        STAGE2_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut frz_org, &mut w2_frz, t_s2);
    let seen_frz = w2_frz.net_valence_sum;

    // Arm 3: Yoked replay control
    let w1_yok_src = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_STAGE1_SALT,
        STAGE1_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        false,
        false,
        perm.clone(),
    );
    let mut w1_yok = YokedMedium::new(w1_yok_src, donor_train);
    let mut yok_org = Alpha11Organism::new(seed.wrapping_add(500_000), OrganismConfig::default());
    step_medium(&mut yok_org, &mut w1_yok, t_s1);
    let mut w2_yok = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_SEEN2_SALT,
        STAGE2_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut yok_org, &mut w2_yok, t_s2);
    let seen_yok = w2_yok.net_valence_sum;

    // Arm 4: ActionShuffled control
    let mut shf_org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut p_shuf = PulseTemporalChannelShuffler::new(seed ^ P4_SHUF_SALT, P4_ACTUATORS);
    let mut w1_shf = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_STAGE1_SALT,
        STAGE1_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        false,
        false,
        perm.clone(),
    );
    for _ in 0..t_s1 {
        let emitted = {
            let v = w1_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = p_shuf.shuffle_pulse(&emitted);
        w1_shf.apply_pulse(&scrambled);
    }
    let shf_agency = f64::from(shf_org.snapshot().mean_interventional_agency);
    let mut w2_shf = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_SEEN2_SALT,
        STAGE2_SEEN_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm.clone(),
    );
    for _ in 0..t_s2 {
        let emitted = {
            let v = w2_shf.current_view();
            shf_org
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(v.num_actuators))
        };
        let scrambled = p_shuf.shuffle_pulse(&emitted);
        w2_shf.apply_pulse(&scrambled);
    }
    let seen_shuf = w2_shf.net_valence_sum;

    // Arm 5: Zero-Shot Unseen Diagonal Comprehension (`P4-G2a..b`) in FrozenPlasticity
    let mut tr_comp = shadow_decode_organism(&trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    tr_comp.set_control_mode(ControlMode::FrozenPlasticity);
    let h_comp0 = tr_comp.learned_parameter_hash();
    let mut wz_comp_tr = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZCOMP_SALT,
        UNSEEN_COMP_BOUTS,
        BoutPool::UnseenCompOnly,
        true,
        false,
        perm.clone(),
    );
    let t_zc = wz_comp_tr.total_ticks();
    step_medium(&mut tr_comp, &mut wz_comp_tr, t_zc);
    let h_comp1 = tr_comp.learned_parameter_hash();
    let comp_zs_tr = wz_comp_tr.net_valence_sum;
    let comp_zs_acc = (wz_comp_tr.completed_bouts as f64) / (UNSEEN_COMP_BOUTS as f64);
    let byte_only_zs_acc =
        (wz_comp_tr.completed_by_mod[1] as f64) / (wz_comp_tr.total_by_mod[1].max(1) as f64);
    let audio_only_zs_acc =
        (wz_comp_tr.completed_by_mod[2] as f64) / (wz_comp_tr.total_by_mod[2].max(1) as f64);

    let mut nb_comp = Alpha11Organism::new(seed ^ 0x4444_1111, OrganismConfig::default());
    nb_comp.set_control_mode(ControlMode::FrozenPlasticity);
    let mut wz_comp_nb = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZCOMP_SALT,
        UNSEEN_COMP_BOUTS,
        BoutPool::UnseenCompOnly,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut nb_comp, &mut wz_comp_nb, t_zc);
    let comp_zs_nb = wz_comp_nb.net_valence_sum;

    let mut les_comp = shadow_decode_organism(&trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    les_comp.zero_learned_conductances();
    les_comp.set_control_mode(ControlMode::FrozenPlasticity);
    let mut wz_comp_ls = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZCOMP_SALT,
        UNSEEN_COMP_BOUTS,
        BoutPool::UnseenCompOnly,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut les_comp, &mut wz_comp_ls, t_zc);
    let comp_zs_les = wz_comp_ls.net_valence_sum;

    // Arm 6: Zero-Shot Unseen Diagonal Production (`P4-G3a..b`) in FrozenPlasticity
    let mut tr_prod = shadow_decode_organism(&trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    tr_prod.set_control_mode(ControlMode::FrozenPlasticity);
    let h_prod0 = tr_prod.learned_parameter_hash();
    let mut wz_prod_tr = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZPROD_SALT,
        UNSEEN_PROD_BOUTS,
        BoutPool::UnseenProdOnly,
        true,
        false,
        perm.clone(),
    );
    let t_zp = wz_prod_tr.total_ticks();
    step_medium(&mut tr_prod, &mut wz_prod_tr, t_zp);
    let h_prod1 = tr_prod.learned_parameter_hash();
    let prod_zs_tr = wz_prod_tr.net_valence_sum;
    let prod_zs_acc = (wz_prod_tr.completed_bouts as f64) / (UNSEEN_PROD_BOUTS as f64);

    let mut nb_prod = Alpha11Organism::new(seed ^ 0x4444_2222, OrganismConfig::default());
    nb_prod.set_control_mode(ControlMode::FrozenPlasticity);
    let mut wz_prod_nb = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZPROD_SALT,
        UNSEEN_PROD_BOUTS,
        BoutPool::UnseenProdOnly,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut nb_prod, &mut wz_prod_nb, t_zp);
    let prod_zs_nb = wz_prod_nb.net_valence_sum;

    let mut les_prod = shadow_decode_organism(&trained_bytes, usize::MAX / 4)
        .unwrap_or_else(|_| Alpha11Organism::new(seed, OrganismConfig::default()));
    les_prod.zero_learned_conductances();
    les_prod.set_control_mode(ControlMode::FrozenPlasticity);
    let mut wz_prod_ls = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZPROD_SALT,
        UNSEEN_PROD_BOUTS,
        BoutPool::UnseenProdOnly,
        true,
        false,
        perm.clone(),
    );
    step_medium(&mut les_prod, &mut wz_prod_ls, t_zp);
    let prod_zs_les = wz_prod_ls.net_valence_sum;

    // ABAB Syntactic/Lexical Remapping Savings (`P4-G4`): Context B -> Return Context A vs Newborn
    let mut wb = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ABAB_B_SALT,
        ABAB_B_BOUTS,
        BoutPool::SeenInterleaved,
        false,
        true,
        perm.clone(),
    );
    let tb = wb.total_ticks();
    step_medium(&mut nat_org, &mut wb, tb);

    let mut w_ret = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ABAB_RET_SALT,
        ABAB_RET_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm.clone(),
    );
    let tret = w_ret.total_ticks();
    step_medium(&mut nat_org, &mut w_ret, tret);
    let abab_ret_nat = w_ret.net_valence_sum;

    let mut nb_ret = Alpha11Organism::new(seed ^ 0x9911_4455, OrganismConfig::default());
    let mut w_nb_ret = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ABAB_RET_SALT,
        ABAB_RET_BOUTS,
        BoutPool::SeenInterleaved,
        true,
        false,
        perm,
    );
    step_medium(&mut nb_ret, &mut w_nb_ret, tret);
    let abab_newborn = w_nb_ret.net_valence_sum;

    Phase4SeedRow {
        seen_nat,
        seen_frz,
        seen_yok,
        seen_shuf,
        seen_nat_acc,
        comp_zs_tr,
        comp_zs_nb,
        comp_zs_les,
        comp_zs_acc,
        byte_only_zs_acc,
        audio_only_zs_acc,
        prod_zs_tr,
        prod_zs_nb,
        prod_zs_les,
        prod_zs_acc,
        abab_ret_nat,
        abab_newborn,
        distractor_share,
        nat_agency,
        shf_agency,
        stage2_frz_drift: (h_comp0 != h_comp1) || (h_prod0 != h_prod1),
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
        let mut rng = DeterministicRng::new(0x5EED_CAFE_4040_0001);
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
        return Err("FrozenPlasticity parameter hash drifted across Stage 2 Unseen evaluation".to_string());
    }
    let cfg = OrganismConfig {
        initial_cells: 512,
        max_cells: 65_536,
        ..OrganismConfig::default()
    };
    let mut org = Alpha11Organism::new(2026_4096, cfg);
    let mut world = DenseFieldWorld::new(88);
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

fn audit_phase4_persistence() -> Result<usize, String> {
    let seed = 2026_0404_u64;
    let perm = WirePerm24::from_seed(seed);
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    let mut w1 = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_STAGE1_SALT,
        36,
        BoutPool::SeenInterleaved,
        false,
        false,
        perm.clone(),
    );
    let t1 = w1.total_ticks();
    step_medium(&mut org, &mut w1, t1);

    // Pause strictly mid-bout (14 ticks into an Unseen Comprehension bout)
    let mut w2 = GroundedCompositionalLanguageMedium::new(
        seed,
        P4_ZCOMP_SALT,
        12,
        BoutPool::UnseenCompOnly,
        true,
        false,
        perm,
    );
    step_medium(&mut org, &mut w2, 14);

    let dir = env::temp_dir().join(format!("a11_p4_ckpt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let res = (|| -> Result<usize, String> {
        let primary = dir.join("p4.ckpt");
        let prev = dir.join("p4.ckpt.prev");
        let tmp = dir.join("p4.ckpt.tmp");

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
            let pr = restored.step(wr.current_view()).map_err(|e| format!("{e:?}"))?;
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

fn repackage_subset_slabs(enc: &[u8], target_ver: u32, max_base_slabs: usize) -> Result<Vec<u8>, String> {
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
            u32::from_le_bytes(*b"OPER"),
        ]
    };
    let payload_end = enc.len() - 4;
    let mut slabs = Vec::with_capacity(enc.len());
    let mut kept = 0u32;
    let mut pos = 24usize;
    while pos < payload_end && (kept as usize) < max_base_slabs {
        let tag = u32::from_le_bytes([enc[pos], enc[pos + 1], enc[pos + 2], enc[pos + 3]]);
        let s_len =
            u32::from_le_bytes([enc[pos + 4], enc[pos + 5], enc[pos + 6], enc[pos + 7]]) as usize;
        let crc_end = pos + 8 + s_len + 4;
        if crc_end > payload_end {
            return Err("Corrupt slab length".to_string());
        }
        if keep_tags.contains(&tag) {
            push_slab(&mut slabs, tag, &enc[pos + 8..pos + 8 + s_len]);
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

fn audit_v9_plus_migration(opt_live_chk: Option<&Path>) -> Result<String, String> {
    if FORMAT_VERSION < 9 {
        return Err(format!("FORMAT_VERSION={FORMAT_VERSION} (< 9)"));
    }
    if MIN_COMPAT_FORMAT_VERSION > 5 {
        return Err("MIN_COMPAT_FORMAT_VERSION > 5".to_string());
    }
    let max_b = OrganismConfig::default().max_state_bytes;
    let perm = WirePerm24::from_seed(55);

    for ver in [5_u32, 6_u32, 7_u32] {
        let bytes = build_v5_v6_v7(ver);
        let mut org = shadow_decode_organism(&bytes, max_b)
            .map_err(|e| format!("v{ver} decode failed: {e:?}"))?;
        let mut w = GroundedCompositionalLanguageMedium::new(
            55,
            P4_STAGE1_SALT,
            8,
            BoutPool::SeenInterleaved,
            false,
            false,
            perm.clone(),
        );
        step_medium(&mut org, &mut w, 10);
        let enc = encode_organism(&org).map_err(|e| format!("v{ver} upgrade encode: {e:?}"))?;
        let _ = shadow_decode_organism(&enc, max_b)
            .map_err(|e| format!("v{ver} upgrade roundtrip: {e:?}"))?;
    }

    let fresh = Alpha11Organism::new(55, OrganismConfig::default());
    let current_bytes = encode_organism(&fresh).map_err(|e| format!("{e:?}"))?;
    let v8_bytes = repackage_subset_slabs(&current_bytes, 8, 6)?;
    let _ = shadow_decode_organism(&v8_bytes, max_b)
        .map_err(|e| format!("v8->v{FORMAT_VERSION} upgrade failed: {e:?}"))?;

    if FORMAT_VERSION > 9 {
        let v9_bytes = repackage_subset_slabs(&current_bytes, 9, 7)?;
        let _ = shadow_decode_organism(&v9_bytes, max_b)
            .map_err(|e| format!("v9->v{FORMAT_VERSION} upgrade failed: {e:?}"))?;
    } else {
        let _ = shadow_decode_organism(&current_bytes, max_b)
            .map_err(|e| format!("v9 roundtrip failed: {e:?}"))?;
    }

    let mut live_tag = "live_chk=skipped";
    if let Some(p) = opt_live_chk {
        let disk_bytes =
            fs::read(p).map_err(|e| format!("failed reading --live-chk {}: {e}", p.display()))?;
        let mut live_org = shadow_decode_organism(&disk_bytes, max_b)
            .map_err(|e| format!("--live-chk decode failed: {e:?}"))?;
        let pre_tick = live_org.snapshot().tick;
        let mut w = GroundedCompositionalLanguageMedium::new(
            11,
            P4_STAGE1_SALT,
            8,
            BoutPool::SeenInterleaved,
            false,
            false,
            WirePerm24::from_seed(11),
        );
        step_medium(&mut live_org, &mut w, 5);
        if live_org.snapshot().tick != pre_tick + 5 {
            return Err("--live-chk 5-step tick check failed".to_string());
        }
        live_tag = "live_chk=OK";
    }

    Ok(format!(
        "v5..v9->v{FORMAT_VERSION} upgrade=OK {live_tag}"
    ))
}

#[inline]
fn build_v5_v6_v7(ver: u32) -> Vec<u8> {
    build_canonical_v5_to_v7(ver)
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
            if !open_vault || !ws_root.join(".vault_authorized_phase4").exists() {
                eprintln!("Refusing: heldout requires --open-vault and .vault_authorized_phase4.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase4.opened");
            if marker.exists() {
                eprintln!("Refusing: Phase-4 vault already opened.");
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (1000..1030).collect()
        }
        other => {
            eprintln!("Unknown split '{other}' (expected dev, dev30, or heldout)");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-4 Locked Judge (Grounded Compositional Language) | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    let raw = run_jobs(seeds.len(), threads, |i| run_phase4_seed(seeds[i]));
    let mut rows = Vec::with_capacity(seeds.len());
    for (i, item) in raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Phase-4 seed job {i} panicked");
            return ExitCode::from(1);
        };
        rows.push(r);
    }

    let s_nat: Vec<f64> = rows.iter().map(|r| r.seen_nat).collect();
    let s_frz: Vec<f64> = rows.iter().map(|r| r.seen_frz).collect();
    let s_yok: Vec<f64> = rows.iter().map(|r| r.seen_yok).collect();
    let s_shf: Vec<f64> = rows.iter().map(|r| r.seen_shuf).collect();
    let s_acc: Vec<f64> = rows.iter().map(|r| r.seen_nat_acc).collect();

    let cz_tr: Vec<f64> = rows.iter().map(|r| r.comp_zs_tr).collect();
    let cz_nb: Vec<f64> = rows.iter().map(|r| r.comp_zs_nb).collect();
    let cz_ls: Vec<f64> = rows.iter().map(|r| r.comp_zs_les).collect();
    let cz_acc: Vec<f64> = rows.iter().map(|r| r.comp_zs_acc).collect();
    let bo_acc: Vec<f64> = rows.iter().map(|r| r.byte_only_zs_acc).collect();
    let ao_acc: Vec<f64> = rows.iter().map(|r| r.audio_only_zs_acc).collect();

    let pz_tr: Vec<f64> = rows.iter().map(|r| r.prod_zs_tr).collect();
    let pz_nb: Vec<f64> = rows.iter().map(|r| r.prod_zs_nb).collect();
    let pz_ls: Vec<f64> = rows.iter().map(|r| r.prod_zs_les).collect();
    let pz_acc: Vec<f64> = rows.iter().map(|r| r.prod_zs_acc).collect();

    let ab_ret: Vec<f64> = rows.iter().map(|r| r.abab_ret_nat).collect();
    let ab_nb: Vec<f64> = rows.iter().map(|r| r.abab_newborn).collect();

    let dst_vec: Vec<f64> = rows.iter().map(|r| r.distractor_share).collect();
    let dst_chance: Vec<f64> = vec![64.0 / 192.0; rows.len()];
    let ag_nat: Vec<f64> = rows.iter().map(|r| r.nat_agency).collect();
    let ag_shf: Vec<f64> = rows.iter().map(|r| r.shf_agency).collect();
    let any_drift = rows.iter().any(|r| r.stage2_frz_drift);

    let g1a = conservative_paired_gate(&s_nat, &s_frz);
    let g1b = conservative_paired_gate(&s_nat, &s_yok);
    let g1c = conservative_paired_gate(&s_nat, &s_shf);
    let g2a = conservative_paired_gate(&cz_tr, &cz_nb);
    let g2b = conservative_paired_gate(&cz_tr, &cz_ls);
    let g3a = conservative_paired_gate(&pz_tr, &pz_nb);
    let g3b = conservative_paired_gate(&pz_tr, &pz_ls);
    let g4 = conservative_paired_gate(&ab_ret, &ab_nb);
    let g5 = conservative_paired_gate(&dst_chance, &dst_vec);
    let g6 = conservative_paired_gate(&ag_nat, &ag_shf);

    let p_holm = holm_adjust(&[
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g2a.p_two_sided,
        g2b.p_two_sided,
        g3a.p_two_sided,
        g3b.p_two_sided,
        g4.p_two_sided,
        g5.p_two_sided,
        g6.p_two_sided,
    ]);

    let mean_seen_acc = mean(&s_acc);
    let mean_comp_zs_acc = mean(&cz_acc);
    let mean_byte_zs_acc = mean(&bo_acc);
    let mean_audio_zs_acc = mean(&ao_acc);
    let mean_prod_zs_acc = mean(&pz_acc);
    let mean_dst = mean(&dst_vec);
    let mean_shf_ag = mean(&ag_shf);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05 && mean_seen_acc > 0.65;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g2a = g2a.mean_diff > 0.0
        && p_holm[3] < 0.05
        && mean_comp_zs_acc > 0.65
        && mean_byte_zs_acc > 0.55
        && mean_audio_zs_acc > 0.55;
    let pass_g2b = g2b.mean_diff > 0.0 && p_holm[4] < 0.05;
    let pass_g3a = g3a.mean_diff > 0.0 && p_holm[5] < 0.05 && mean_prod_zs_acc > 0.60;
    let pass_g3b = g3b.mean_diff > 0.0 && p_holm[6] < 0.05;
    let pass_g4 = g4.mean_diff > 0.0 && p_holm[7] < 0.05;
    let pass_g5 = g5.mean_diff > 0.0 && p_holm[8] < 0.05 && mean_dst < 0.05;
    let pass_g6 = g6.mean_diff > 0.015 && p_holm[9] < 0.05 && mean_shf_ag <= 0.05;

    println!("\n--- Phase-4 Grounded Compositional Language Gates (Holm m=10) ---");
    for (name, g, ph, ok) in [
        ("P4-G1a Seen Occluded Nat > Frozen", g1a, p_holm[0], pass_g1a),
        ("P4-G1b Seen Occluded Nat > Yoked", g1b, p_holm[1], pass_g1b),
        ("P4-G1c Seen Occluded Nat > Shuffled", g1c, p_holm[2], pass_g1c),
        ("P4-G2a Zero-Shot Comp TrFrz > NbFrz", g2a, p_holm[3], pass_g2a),
        ("P4-G2b Zero-Shot Comp TrFrz > Lesion", g2b, p_holm[4], pass_g2b),
        ("P4-G3a Zero-Shot Prod TrFrz > NbFrz", g3a, p_holm[5], pass_g3a),
        ("P4-G3b Zero-Shot Prod TrFrz > Lesion", g3b, p_holm[6], pass_g3b),
        ("P4-G4  ABAB Syntax/Lex Return-A > Nb", g4, p_holm[7], pass_g4),
        ("P4-G5  Distractor Chance > Nat (<5%)", g5, p_holm[8], pass_g5),
        ("P4-G6  Agency Purity Nat > Shuffled", g6, p_holm[9], pass_g6),
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
        "         seen_acc={mean_seen_acc:.4} (>0.65) | comp_zs_acc={mean_comp_zs_acc:.4} (>0.65) [byte={mean_byte_zs_acc:.4} >0.55, audio={mean_audio_zs_acc:.4} >0.55] | prod_zs_acc={mean_prod_zs_acc:.4} (>0.60) | dst_share={mean_dst:.4} (<0.05) | shuf_ag={mean_shf_ag:.4} (<=0.05)"
    );

    println!("\n--- Structural, O(k) Latency & Cross-Version Persistence Invariants ---");
    let (hyg, spl, mir, mod_b) = audit_core_debt(&ws_root);
    let pass_i1 = hyg == 0 && spl == 0 && mir == 0 && mod_b == 0;
    println!(
        "  [{:<4}] P4-I1 Zero Core Debt: hygiene={hyg} split={spl} mirrors={mir} cell_mod={mod_b}",
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
        "  [{:<4}] P4-I2 O(k) Latency & Frozen Drift: {i2_msg}",
        if pass_i2 { "PASS" } else { "FAIL" }
    );

    let (pass_i3, i3_msg) = match audit_phase4_persistence() {
        Ok(b) => (
            b <= 524_288,
            format!("ckpt_bytes={b} mid_bout_bit_id_160=OK prev=OK poison=OK"),
        ),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P4-I3 Mid-Bout Paged Persistence: {i3_msg}",
        if pass_i3 { "PASS" } else { "FAIL" }
    );

    let (pass_i4, i4_msg) = match audit_v9_plus_migration(opt_live_chk.as_deref()) {
        Ok(msg) => (true, msg),
        Err(e) => (false, e),
    };
    println!(
        "  [{:<4}] P4-I4 Legacy v5..v9 Upgrade & Live Check: {i4_msg}",
        if pass_i4 { "PASS" } else { "FAIL" }
    );

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g2a
        && pass_g2b
        && pass_g3a
        && pass_g3b
        && pass_g4
        && pass_g5
        && pass_g6
        && pass_i1
        && pass_i2
        && pass_i3
        && pass_i4;

    println!(
        "\n=== Phase-4 Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );
    if all_pass { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
