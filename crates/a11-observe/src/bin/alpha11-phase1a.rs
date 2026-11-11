//! Pre-registered locked Phase-1A judge (`P1A-G1..G4` + `P1A-I1..I3`).
//!
//! Enforces unified arbitrary-actuator sensorimotor learning, focus-modulated Noisy-TV immunity,
//! counterfactual agency purity, actuator-order & resting-baseline invariance (`-0.35 -> +0.18`),
//! zero split-brain / constant-mirror / cell-modulo core debt, `O(1)` 4,096-channel latency,
//! 64-actuator safety, and lossless `FORMAT_VERSION = 5 -> >= 6` checkpoint upgrade.
//!
//! Usage: `alpha11-phase1a [dev|dev30|heldout] [--threads N] [--open-vault] [--v5-chk PATH]`

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
use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, WorldMedium};
use a11_observe::gates::{PairedGate, benchmark_dense_4096, holm_adjust, paired_gate};
use a11_observe::open_habitat::{FRAME, OPEN_ACT, OPEN_CH, OpenHabitat};
use a11_observe::{
    BandRole, BandSpec, DEFAULT_TWELVE_SEEDS, DeterministicRng, PulseTemporalChannelShuffler,
    compute_band_allocations, scan_crate_source,
};
use a11_portals::{
    CompositePortalMedium, HAND_ORGAN_CHANNELS, VESTIBULAR_ORGAN_CHANNELS, VOCAL_ERB_BANDS,
};
use a11_worlds::{BrightRadiantWorld, TERRARIUM_CHANNELS, YokedMedium};

const COMP_TICKS: u64 = 4_000;
const COMP_EVAL_TICKS: u64 = 2_000;
const TEACH_TICKS: u64 = 40_000;
const REPLY_LEN: usize = 15;
const REPLY_GAIN: f32 = 0.80;
const COMP_SHUFFLE_SALT: u64 = 0x5A17_C049_27AC_2026;
const COMP_EVAL_SALT: u64 = 0x9A5E_1A00_7788_5511;
const SHIFTED_TEACH_SALT: u64 = 0x7EAC_541F_7ED0_2026;
const PERM_SALT: u64 = 0x9E37_79B9_A110_9E24;

#[derive(Clone, Debug, Default)]
struct CompositeSeedMetrics {
    nat_net: f64,
    frz_net: f64,
    yok_net: f64,
    shf_net: f64,
    xfer_trained_net: f64,
    xfer_lesion_net: f64,
    nat_ntv_share: f64,
    nat_agency: f64,
    shf_agency: f64,
}

fn composite_bands() -> Vec<BandSpec> {
    vec![
        BandSpec::new("Host Cave Structured (0..96)", 0..96, BandRole::StructuredPhysical),
        BandSpec::new("Host Cave Noisy TV (96..128)", 96..128, BandRole::NoisyTvDistractor),
        BandSpec::new("Scratchpad Structured (128..288)", 128..288, BandRole::StructuredPhysical),
        BandSpec::new("Scratchpad Noisy TV (288..320)", 288..320, BandRole::NoisyTvDistractor),
        BandSpec::new("Pattern Structured (320..464)", 320..464, BandRole::StructuredPhysical),
        BandSpec::new("Pattern Noisy TV (464..496)", 464..496, BandRole::NoisyTvDistractor),
        BandSpec::new("Rhythm Structured (496..592)", 496..592, BandRole::StructuredPhysical),
        BandSpec::new("Rhythm Noisy TV (592..624)", 592..624, BandRole::NoisyTvDistractor),
        BandSpec::new("Balance Structured (624..720)", 624..720, BandRole::StructuredPhysical),
        BandSpec::new("Balance Noisy TV (720..752)", 720..752, BandRole::NoisyTvDistractor),
    ]
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

fn run_composite_seed(seed: u64, bands: &[BandSpec]) -> CompositeSeedMetrics {
    let cfg = OrganismConfig::default();

    // 1. Natural (recording donor metabolic stream for YokedReward twin).
    let mut nat = Alpha11Organism::new(seed, cfg);
    let mut w_nat = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(seed), seed);
    let mut donor = Vec::with_capacity(COMP_TICKS as usize);
    let mut nat_net = 0.0_f64;
    for _ in 0..COMP_TICKS {
        let pulse = {
            let v = w_nat.current_view();
            donor.push((v.metabolic_influx, v.tissue_stress));
            nat.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_nat.num_actuators()))
        };
        w_nat.apply_pulse(&pulse);
        let after = w_nat.current_view();
        nat_net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }
    let nat_snap = nat.snapshot();
    let (_, _, nat_ntv_share) = compute_band_allocations(&nat_snap, bands);

    // 2. FrozenPlasticity twin.
    let mut frz = Alpha11Organism::new(seed, cfg);
    frz.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_frz = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(seed), seed);
    let mut frz_net = 0.0_f64;
    for _ in 0..COMP_TICKS {
        let pulse = {
            let v = w_frz.current_view();
            frz.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_frz.num_actuators()))
        };
        w_frz.apply_pulse(&pulse);
        let after = w_frz.current_view();
        frz_net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }

    // 3. YokedReward twin (independent organism seed so trajectory decouples from donor).
    let mut yok = Alpha11Organism::new(seed.wrapping_add(500_000), cfg);
    let mut w_yok = YokedMedium::new(
        CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(seed), seed),
        donor,
    );
    let mut yok_net = 0.0_f64;
    for _ in 0..COMP_TICKS {
        let pulse = {
            let v = w_yok.current_view();
            yok.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_yok.num_actuators()))
        };
        w_yok.apply_pulse(&pulse);
        let true_after = w_yok.inner().current_view();
        yok_net += f64::from(true_after.metabolic_influx) - f64::from(true_after.tissue_stress);
    }

    // 4. ActionShuffled twin across all 27 actuators.
    let mut shf = Alpha11Organism::new(seed, cfg);
    let mut w_shf = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(seed), seed);
    let mut shuffler =
        PulseTemporalChannelShuffler::new(seed ^ COMP_SHUFFLE_SALT, w_shf.num_actuators());
    let mut shf_net = 0.0_f64;
    for _ in 0..COMP_TICKS {
        let emitted = {
            let v = w_shf.current_view();
            shf.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_shf.num_actuators()))
        };
        let applied = shuffler.shuffle_pulse(&emitted);
        w_shf.apply_pulse(&applied);
        let after = w_shf.current_view();
        shf_net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }
    let shf_snap = shf.snapshot();

    // 5. Post-training transfer: Trained (Frozen) vs Flagless Weight Lesion on fresh world.
    let eval_seed = seed ^ COMP_EVAL_SALT;
    let mut lesion = nat.clone();
    lesion.zero_learned_conductances();
    lesion.set_control_mode(ControlMode::FrozenPlasticity);
    nat.set_control_mode(ControlMode::FrozenPlasticity);
    let trained_hash = nat.learned_parameter_hash();

    let mut w_eval_tr =
        CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(eval_seed), eval_seed);
    let mut xfer_trained_net = 0.0_f64;
    for _ in 0..COMP_EVAL_TICKS {
        let pulse = {
            let v = w_eval_tr.current_view();
            nat.step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_eval_tr.num_actuators()))
        };
        w_eval_tr.apply_pulse(&pulse);
        let after = w_eval_tr.current_view();
        xfer_trained_net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }
    assert_eq!(
        trained_hash,
        nat.learned_parameter_hash(),
        "FrozenPlasticity parameter hash drifted during Composite27 transfer"
    );

    let mut w_eval_le =
        CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(eval_seed), eval_seed);
    let mut xfer_lesion_net = 0.0_f64;
    for _ in 0..COMP_EVAL_TICKS {
        let pulse = {
            let v = w_eval_le.current_view();
            lesion
                .step(v)
                .unwrap_or_else(|_| FieldPulse::silent(w_eval_le.num_actuators()))
        };
        w_eval_le.apply_pulse(&pulse);
        let after = w_eval_le.current_view();
        xfer_lesion_net += f64::from(after.metabolic_influx) - f64::from(after.tissue_stress);
    }

    CompositeSeedMetrics {
        nat_net,
        frz_net,
        yok_net,
        shf_net,
        xfer_trained_net,
        xfer_lesion_net,
        nat_ntv_share,
        nat_agency: f64::from(nat_snap.mean_interventional_agency),
        shf_agency: f64::from(shf_snap.mean_interventional_agency),
    }
}

// =============================================================================
// P1A-G4: Permuted Actuators + Shifted Cochlear Baseline (-0.35 -> +0.18)
// =============================================================================

struct PermutedShiftedHabitat {
    hab: OpenHabitat,
    act_perm: [usize; OPEN_ACT],
    shifted_ambient: [f32; OPEN_CH],
}

impl PermutedShiftedHabitat {
    fn new(seed: u64) -> Self {
        let hab = OpenHabitat::new(seed ^ SHIFTED_TEACH_SALT);
        let mut rng = DeterministicRng::new(seed ^ PERM_SALT);
        let mut act_perm: [usize; OPEN_ACT] = std::array::from_fn(|i| i);
        for i in (1..OPEN_ACT).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            act_perm.swap(i, j);
        }
        let mut wrapper = Self {
            hab,
            act_perm,
            shifted_ambient: [0.0; OPEN_CH],
        };
        wrapper.refresh_shifted_ambient();
        wrapper
    }

    fn refresh_shifted_ambient(&mut self) {
        let raw = self.hab.current_view().ambient_in;
        self.shifted_ambient.copy_from_slice(raw);
        let coch_start = TERRARIUM_CHANNELS + HAND_ORGAN_CHANNELS + VESTIBULAR_ORGAN_CHANNELS;
        let coch_end = coch_start + 2 * VOCAL_ERB_BANDS;
        for v in &mut self.shifted_ambient[coch_start..coch_end] {
            // Shifts silent cochlear & LSO resting floor from -0.35 to +0.18 while preserving monotonic dynamic range.
            *v = (0.72 * (*v + 0.35) + 0.18).clamp(-1.0, 1.0);
        }
    }

    fn current_view(&self) -> FieldView<'_> {
        let v = self.hab.current_view();
        FieldView {
            ambient_in: &self.shifted_ambient,
            metabolic_influx: v.metabolic_influx,
            tissue_stress: v.tissue_stress,
            num_actuators: OPEN_ACT,
        }
    }

    fn apply_permuted_pulse(&mut self, org_pulse: &FieldPulse) {
        let mut mapped = vec![0.0_f32; OPEN_ACT];
        for (org_idx, &hab_idx) in self.act_perm.iter().enumerate() {
            mapped[hab_idx] = org_pulse
                .actuator_out
                .get(org_idx)
                .copied()
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);
        }
        self.hab.apply_pulse(&FieldPulse {
            actuator_out: mapped,
        });
        self.refresh_shifted_ambient();
    }

    fn sync_permuted_vocal_ontogeny(&mut self, org: &Alpha11Organism) {
        let s = org.snapshot();
        let items = org.motor_maturation_by_channel();
        let mut sum = 0.0_f64;
        let mut n = 0_u32;
        for &(ch, mastery, _) in &items {
            let cu = ch as usize;
            if cu < OPEN_ACT && (12..17).contains(&self.act_perm[cu]) {
                sum += f64::from(mastery);
                n += 1;
            }
        }
        let voc_m = if n > 0 {
            (sum / f64::from(n)) as f32
        } else {
            0.0
        };
        self.hab
            .focal_body
            .advance_vocal_ontogeny(s.tick, s.mitosis_births, voc_m);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShiftedProgram {
    FeedC,
    FeedY,
    VoiceC,
    VoiceY,
}

fn run_shifted_teach_arm(
    seed: u64,
    program: ShiftedProgram,
    schedule: &[u64],
) -> (Vec<u64>, f64) {
    let contingent = matches!(program, ShiftedProgram::FeedC | ShiftedProgram::VoiceC);
    let feeds = matches!(program, ShiftedProgram::FeedC | ShiftedProgram::FeedY);
    let speaks = matches!(program, ShiftedProgram::VoiceC | ShiftedProgram::VoiceY);
    let (delay, refract) = if feeds { (2_u64, 20_u64) } else { (10_u64, 200_u64) };

    let org_seed = if contingent {
        seed
    } else {
        seed.wrapping_add(500_000)
    };
    let mut org = Alpha11Organism::new(org_seed, OrganismConfig::default());
    let mut env_w = PermutedShiftedHabitat::new(seed);
    let mut rng = DeterministicRng::new(seed ^ 0xFEED_541F_2026_0009);

    let late_start = TEACH_TICKS - TEACH_TICKS / 5;
    let late_ticks = (TEACH_TICKS - late_start).max(1) as f64;

    let mut ring = [[0.0_f32; FRAME]; REPLY_LEN];
    let mut ring_len = 0_usize;
    let mut ring_head = 0_usize;
    let mut reply_frames: Vec<[f32; FRAME]> = Vec::new();
    let mut reply_start = 0_u64;
    let mut pending: Option<u64> = None;
    let mut next_allowed = 0_u64;
    let mut sched_idx = 0_usize;
    let mut events = Vec::new();

    let mut prev_onsets = env_w.hab.focal_body.vocal().observation().syllable_onsets;
    let mut prev_voiced = env_w.hab.focal_body.vocal().observation().voiced_ticks;
    let mut late_onsets = 0_u64;

    for t in 1..=TEACH_TICKS {
        if speaks && !reply_frames.is_empty() && t >= reply_start {
            let k = (t - reply_start) as usize;
            if let Some(f) = reply_frames.get(k) {
                let mut scaled = [0.0_f32; FRAME];
                for (d, &s) in scaled.iter_mut().zip(f.iter()) {
                    *d = REPLY_GAIN * s;
                }
                env_w.hab.set_reply_frame(Some(scaled));
            } else {
                reply_frames.clear();
            }
        }

        let emitted = org
            .step(env_w.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACT));
        env_w.apply_permuted_pulse(&emitted);
        if (t & 15) == 0 {
            env_w.sync_permuted_vocal_ontogeny(&org);
        }

        let obs = env_w.hab.focal_body.vocal().observation();
        let d_on = obs.syllable_onsets.saturating_sub(prev_onsets);
        let d_voi = obs.voiced_ticks.saturating_sub(prev_voiced);
        prev_onsets = obs.syllable_onsets;
        prev_voiced = obs.voiced_ticks;

        if d_voi > 0 {
            ring[ring_head].copy_from_slice(env_w.hab.focal_body.vocal().last_radiated_wave());
            ring_head = (ring_head + 1) % REPLY_LEN;
            ring_len = (ring_len + 1).min(REPLY_LEN);
        }
        if t > late_start {
            late_onsets += d_on;
        }

        let mut fire: Option<u64> = None;
        if contingent {
            if d_on > 0 && pending.is_none() && t >= next_allowed {
                pending = Some(t + delay);
                next_allowed = t + delay + refract;
            }
            if let Some(e) = pending
                && t + 1 == e
            {
                fire = Some(e);
                pending = None;
            }
        } else if sched_idx < schedule.len() && schedule[sched_idx] == t + 1 {
            fire = Some(t + 1);
            sched_idx += 1;
        }

        if let Some(e) = fire {
            events.push(e);
            if feeds {
                let theta = (rng.next_f64() as f32) * std::f32::consts::TAU;
                let _ = env_w.hab.terrarium.provision_open_pod_near(
                    1,
                    (0.6 * theta.cos(), 0.6 * theta.sin()),
                    0.15,
                );
            } else if speaks && ring_len > 0 {
                reply_frames.clear();
                for k in 0..ring_len {
                    let idx = (ring_head + REPLY_LEN - ring_len + k) % REPLY_LEN;
                    reply_frames.push(ring[idx]);
                }
                reply_start = e;
            }
        }
    }

    let late_onsets_per_1k = (late_onsets as f64) * 1_000.0 / late_ticks;
    (events, late_onsets_per_1k)
}

fn run_shifted_seed(seed: u64) -> (f64, f64, f64, f64) {
    let (feed_sched, feed_c_pk) = run_shifted_teach_arm(seed, ShiftedProgram::FeedC, &[]);
    let (_, feed_y_pk) = run_shifted_teach_arm(seed, ShiftedProgram::FeedY, &feed_sched);
    let (voice_sched, voice_c_pk) = run_shifted_teach_arm(seed, ShiftedProgram::VoiceC, &[]);
    let (_, voice_y_pk) = run_shifted_teach_arm(seed, ShiftedProgram::VoiceY, &voice_sched);
    (feed_c_pk, feed_y_pk, voice_c_pk, voice_y_pk)
}

// =============================================================================
// P1A-I1: Core Source Debt Audit (0 split-brain, 0 mirrors, 0 % PRED_BUCKETS)
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
// P1A-I2: 64-Actuator Safety & Round-Trip Checkpoint Identity
// =============================================================================

struct HighActuatorMedium {
    tick: u64,
    ambient: [f32; 512],
    influx: f32,
    stress: f32,
}

impl HighActuatorMedium {
    fn new() -> Self {
        let mut m = Self {
            tick: 0,
            ambient: [0.0; 512],
            influx: 0.01,
            stress: 0.0,
        };
        m.refresh([0.0; 64]);
        m
    }

    fn refresh(&mut self, act: [f32; 64]) {
        let ph = (self.tick as f32) * 0.11;
        for (i, slot) in self.ambient.iter_mut().enumerate() {
            let a = act[i & 63];
            *slot = (0.55 * (ph + (i as f32) * 0.07).sin() + 0.35 * a).clamp(-1.0, 1.0);
        }
    }
}

impl WorldMedium for HighActuatorMedium {
    fn num_ambient_channels(&self) -> usize {
        512
    }

    fn num_actuators(&self) -> usize {
        64
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient,
            metabolic_influx: self.influx,
            tissue_stress: self.stress,
            num_actuators: 64,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let mut act = [0.0_f32; 64];
        let mut dot = 0.0_f32;
        for (i, dst) in act.iter_mut().enumerate() {
            let v = pulse
                .actuator_out
                .get(i)
                .copied()
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);
            *dst = v;
            dot += v * ((self.tick as f32 * 0.11 + (i as f32) * 0.07).sin());
        }
        self.influx = (0.012 + 0.02 * dot).max(0.0);
        self.stress = ((-dot) * 0.005).max(0.0);
        self.tick += 1;
        self.refresh(act);
    }
}

fn check_64_actuator_safety() -> Result<(), String> {
    let mut org = Alpha11Organism::new(2026_64, OrganismConfig::default());
    let mut world = HighActuatorMedium::new();
    for _ in 0..600 {
        let p = org
            .step(world.current_view())
            .map_err(|e| format!("64-actuator step failed: {e:?}"))?;
        if p.actuator_out.len() != 64 || p.actuator_out.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "64-actuator invalid pulse len={}",
                p.actuator_out.len()
            ));
        }
        world.apply_pulse(&p);
    }
    let encoded = encode_organism(&org).map_err(|e| format!("64-act encode failed: {e:?}"))?;
    let mut decoded = shadow_decode_organism(&encoded, OrganismConfig::default().max_state_bytes)
        .map_err(|e| format!("64-act shadow decode failed: {e:?}"))?;
    let p_orig = org
        .step(world.current_view())
        .map_err(|e| format!("orig post-decode step failed: {e:?}"))?;
    let p_dec = decoded
        .step(world.current_view())
        .map_err(|e| format!("decoded post-decode step failed: {e:?}"))?;
    if p_orig.actuator_out != p_dec.actuator_out {
        return Err("64-actuator save/load pulse mismatch".to_string());
    }
    Ok(())
}

// =============================================================================
// P1A-I3: Canonical FORMAT_VERSION = 5 Legacy Checkpoint Upgrade Verification
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

fn build_canonical_v5_checkpoint() -> Vec<u8> {
    const V5_VER: u32 = 5;
    const V5_EXPLORE: usize = 32;
    const V5_TRACKED_ACT: usize = 32;
    const V5_BUCKETS: usize = 64;
    const V5_RING: usize = 16;
    const V5_OFC_CTX: usize = 4;
    const V5_STR_PORES: usize = 64;
    const V5_STR_ACT: usize = 8;
    const V5_STR_CUES: usize = 4 * V5_STR_PORES;
    const V5_STR_FEATS: usize = V5_STR_CUES + 1;
    const V5_STR_TAP_DIM: usize = 4 * V5_STR_ACT;
    const V5_STR_W_LEN: usize = V5_STR_ACT * V5_STR_CUES;
    const V5_STR_FWD_LEN: usize = V5_STR_CUES * V5_STR_ACT;

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
    for _ in 0..(V5_EXPLORE * 3) {
        push_f32(&mut body, 0.0);
    }
    body.extend(std::iter::repeat_n(0_u8, V5_EXPLORE));

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
    push_u32(&mut cell, 0); // free_list
    push_u32(&mut cell, 2); // active_indices
    push_u32(&mut cell, 0);
    push_u32(&mut cell, 1);
    push_f32(&mut cell, 0.5);
    push_f32(&mut cell, 0.1);

    let mut pore = Vec::new();
    push_u32(&mut pore, 2); // sensor_pores
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
    push_u32(&mut pore, 0); // sensor_free_list
    push_u32(&mut pore, 2); // motor_pores
    push_u32(&mut pore, 2);
    for ch in 0..2_u32 {
        pore.push(1);
        push_u32(&mut pore, ch);
        for v in [0.45_f32, 0.0, 0.2, 0.1, 0.05] {
            push_f32(&mut pore, v);
        }
    }
    push_u32(&mut pore, 0); // motor_free_list

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
    for _ in 0..(3 * V5_TRACKED_ACT * V5_BUCKETS) {
        push_f32(&mut tube, 0.01);
    }
    for _ in 0..(3 * V5_BUCKETS) {
        push_f32(&mut tube, 0.0);
    }
    for _ in 0..V5_BUCKETS {
        push_f32(&mut tube, 0.05); // pore_scale_v
    }
    for _ in 0..V5_BUCKETS {
        push_f32(&mut tube, 0.02); // pore_agency
    }
    for _ in 0..V5_BUCKETS {
        push_f32(&mut tube, 1.0e-4); // pore_dsq_ema
    }
    for _ in 0..V5_BUCKETS {
        push_f32(&mut tube, 5.0e-3); // pore_var_ema
    }
    for _ in 0..V5_BUCKETS {
        push_f32(&mut tube, 0.0); // pore_bg_ild
    }
    push_f32(&mut tube, 0.0); // social_loom_ema
    for _ in 0..V5_RING {
        push_f32(&mut tube, 0.0);
    }
    tube.push(0); // spatial_prox_cursor
    for _ in 0..(3 * V5_TRACKED_ACT) {
        push_f32(&mut tube, 0.0);
    }
    for _ in 0..7 {
        push_f32(&mut tube, 0.0);
    }
    tube.push(255);
    tube.push(255);
    for _ in 0..(2 * V5_TRACKED_ACT) {
        push_f32(&mut tube, 0.0);
    }
    push_f32(&mut tube, 0.18); // mean_interventional_agency

    // Striatal v5 block
    tube.push(1); // num_ctx
    tube.push(0); // active_ctx
    tube.push(0); // impact_refrac
    tube.extend(std::iter::repeat_n(0_u8, V5_OFC_CTX));
    for _ in 0..V5_OFC_CTX {
        push_u32(&mut tube, 12);
    }
    for _ in 0..(2 * V5_OFC_CTX) {
        push_f32(&mut tube, 0.01);
    }
    for _ in 0..V5_STR_PORES {
        push_u32(&mut tube, u32::MAX);
    }
    for _ in 0..V5_STR_PORES {
        push_u32(&mut tube, 0);
    }
    for _ in 0..V5_STR_PORES {
        push_f32(&mut tube, 0.0);
    }
    for _ in 0..(2 * V5_STR_PORES) {
        push_f32(&mut tube, 1.0e-4);
    }
    let f32_zeros = |dst: &mut Vec<u8>, cnt: usize| {
        for _ in 0..cnt {
            push_f32(dst, 0.0);
        }
    };
    f32_zeros(&mut tube, V5_STR_PORES * V5_STR_TAP_DIM); // canc_g
    f32_zeros(&mut tube, V5_STR_W_LEN); // w_birth
    f32_zeros(&mut tube, 2 * V5_OFC_CTX * V5_STR_W_LEN); // w_ctx, w_cons
    f32_zeros(&mut tube, 2 * V5_OFC_CTX * V5_STR_FEATS); // v_ctx, v_cons
    f32_zeros(&mut tube, 2 * V5_OFC_CTX * V5_STR_CUES); // q_ctx, q_cons
    f32_zeros(&mut tube, 2 * V5_STR_CUES); // fwd_b, fwd_alpha
    f32_zeros(&mut tube, (1 + 2 * V5_OFC_CTX) * V5_STR_FWD_LEN); // fwd_j*
    for _ in 0..(2 * V5_STR_CUES) {
        push_f32(&mut tube, 1.0e-4);
    }
    f32_zeros(&mut tube, 2 * V5_OFC_CTX); // fwd_num, fwd_den
    f32_zeros(&mut tube, V5_STR_ACT); // act_mean
    push_u32(&mut tube, 0);
    push_u32(&mut tube, 0);
    f32_zeros(&mut tube, V5_OFC_CTX); // ctx_out_err
    for _ in 0..3 {
        push_f32(&mut tube, 0.0);
    }
    f32_zeros(&mut tube, V5_STR_TAP_DIM); // da_ring
    f32_zeros(&mut tube, V5_STR_W_LEN); // elig_actor
    f32_zeros(&mut tube, 3 * V5_STR_FEATS); // elig_critic, feat, prev_critic_feat
    f32_zeros(&mut tube, 2 * V5_STR_CUES); // prev_feat, prev_df
    f32_zeros(&mut tube, 2 * V5_STR_ACT); // prev_act, noise
    push_f32(&mut tube, 0.0); // v_prev
    tube.push(0); // has_prev
    push_u64(&mut tube, 777); // striatal.tick
    push_u64(&mut tube, 0xCAFE_BABE_1234_5678);
    push_u64(&mut tube, 0x8765_4321_FEED_FACE);

    let mut ring = Vec::new();
    push_u32(&mut ring, 0);
    for _ in 0..V5_RING {
        ring.push(0);
        push_u64(&mut ring, 0);
        ring.push(0);
        push_f32(&mut ring, 0.0);
        for _ in 0..V5_TRACKED_ACT {
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
    push_u32(&mut full, V5_VER);
    push_u32(&mut full, SLAB_COUNT);
    push_u64(&mut full, slabs.len() as u64);
    full.extend_from_slice(&slabs);
    let file_crc = crc32c(&full);
    push_u32(&mut full, file_crc);
    full
}

fn verify_v5_migration(opt_v5_path: Option<&Path>) -> Result<(), String> {
    if FORMAT_VERSION < 6 {
        return Err(format!(
            "FORMAT_VERSION is {FORMAT_VERSION} (Phase 1A requires >= 6)"
        ));
    }
    if MIN_COMPAT_FORMAT_VERSION > 5 {
        return Err(format!(
            "MIN_COMPAT_FORMAT_VERSION is {MIN_COMPAT_FORMAT_VERSION} (must remain <= 5)"
        ));
    }
    let v5_bytes = build_canonical_v5_checkpoint();
    let max_b = OrganismConfig::default().max_state_bytes;
    let mut org = shadow_decode_organism(&v5_bytes, max_b)
        .map_err(|e| format!("canonical v5 decode failed: {e:?}"))?;
    let snap = org.snapshot();
    if snap.tick != 777 || (snap.atp - 42.5).abs() > 1.0e-3 || snap.cell_count != 4 {
        return Err(format!(
            "canonical v5 state mismatch: tick={} atp={} cells={}",
            snap.tick, snap.atp, snap.cell_count
        ));
    }
    let mut w = CompositePortalMedium::with_all_portals(BrightRadiantWorld::new(42), 42);
    for _ in 0..20 {
        let p = org
            .step(w.current_view())
            .map_err(|e| format!("upgraded v5 organism step failed: {e:?}"))?;
        w.apply_pulse(&p);
    }
    let upgraded_bytes =
        encode_organism(&org).map_err(|e| format!("upgraded encode failed: {e:?}"))?;
    let encoded_ver = u32::from_le_bytes([
        upgraded_bytes[8],
        upgraded_bytes[9],
        upgraded_bytes[10],
        upgraded_bytes[11],
    ]);
    if encoded_ver != FORMAT_VERSION {
        return Err(format!(
            "re-encoded version {encoded_ver} != {FORMAT_VERSION}"
        ));
    }
    let _ = shadow_decode_organism(&upgraded_bytes, max_b)
        .map_err(|e| format!("upgraded v6 roundtrip decode failed: {e:?}"))?;

    if let Some(p) = opt_v5_path {
        let disk_bytes =
            fs::read(p).map_err(|e| format!("failed reading --v5-chk {}: {e}", p.display()))?;
        let mut live_org = shadow_decode_organism(&disk_bytes, max_b)
            .map_err(|e| format!("live --v5-chk decode failed: {e:?}"))?;
        let pre_tick = live_org.snapshot().tick;
        let pulse = live_org
            .step(w.current_view())
            .map_err(|e| format!("live --v5-chk post-upgrade step failed: {e:?}"))?;
        if pulse.actuator_out.len() != 27 || live_org.snapshot().tick != pre_tick + 1 {
            return Err("live --v5-chk post-upgrade step invariant failed".to_string());
        }
    }
    Ok(())
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
    let mut opt_v5_chk: Option<PathBuf> = None;
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
        if args[idx] == "--v5-chk"
            && let Some(val) = args.get(idx + 1)
        {
            opt_v5_chk = Some(PathBuf::from(val));
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
            if !ws_root.join(".vault_authorized_next").exists() {
                eprintln!("Refusing: .vault_authorized_next missing in workspace root.");
                return ExitCode::from(2);
            }
            let dir = ws_root.join("observations/heldout_vault");
            let _ = fs::create_dir_all(&dir);
            let marker = dir.join("phase1a.opened");
            if marker.exists() {
                eprintln!(
                    "Refusing: Phase-1A held-out vault already opened ({}).",
                    marker.display()
                );
                return ExitCode::from(3);
            }
            let _ = fs::write(&marker, b"opened\n");
            (300..312).collect()
        }
        other => {
            eprintln!("Unknown split '{other}' (expected dev, dev30, or heldout)");
            return ExitCode::from(2);
        }
    };

    let t0 = Instant::now();
    println!(
        "=== Alpha 11 Phase-1A Locked Judge | split={split} | n={} | threads={threads} ===",
        seeds.len()
    );

    // 1. Run CompositePortalMedium<BrightRadiantWorld> (752 ch, 27 act) across seeds.
    let bands = composite_bands();
    let comp_raw = run_jobs(seeds.len(), threads, |i| run_composite_seed(seeds[i], &bands));
    let mut comp_rows = Vec::with_capacity(seeds.len());
    for (i, item) in comp_raw.into_iter().enumerate() {
        let Some(r) = item else {
            eprintln!("Composite27 seed job {i} panicked");
            return ExitCode::from(1);
        };
        comp_rows.push(r);
    }

    // 2. Run Permuted + Baseline-Shifted (+0.18) Vocal Contingency across seeds.
    let shifted_raw = run_jobs(seeds.len(), threads, |i| run_shifted_seed(seeds[i]));
    let mut feed_c = Vec::with_capacity(seeds.len());
    let mut feed_y = Vec::with_capacity(seeds.len());
    let mut voice_c = Vec::with_capacity(seeds.len());
    let mut voice_y = Vec::with_capacity(seeds.len());
    for (i, item) in shifted_raw.into_iter().enumerate() {
        let Some((fc, fy, vc, vy)) = item else {
            eprintln!("PermutedShifted seed job {i} panicked");
            return ExitCode::from(1);
        };
        feed_c.push(fc);
        feed_y.push(fy);
        voice_c.push(vc);
        voice_y.push(vy);
    }

    let nat_net: Vec<f64> = comp_rows.iter().map(|r| r.nat_net).collect();
    let frz_net: Vec<f64> = comp_rows.iter().map(|r| r.frz_net).collect();
    let yok_net: Vec<f64> = comp_rows.iter().map(|r| r.yok_net).collect();
    let shf_net: Vec<f64> = comp_rows.iter().map(|r| r.shf_net).collect();
    let xfer_tr: Vec<f64> = comp_rows.iter().map(|r| r.xfer_trained_net).collect();
    let xfer_le: Vec<f64> = comp_rows.iter().map(|r| r.xfer_lesion_net).collect();
    let nat_ntv: Vec<f64> = comp_rows.iter().map(|r| r.nat_ntv_share).collect();
    // CompositePortalMedium has 5 * 32 = 160 Noisy-TV channels out of 752 total (21.28% uniform chance).
    let chance_ntv: Vec<f64> = vec![160.0 / 752.0; comp_rows.len()];
    let nat_ag: Vec<f64> = comp_rows.iter().map(|r| r.nat_agency).collect();
    let shf_ag: Vec<f64> = comp_rows.iter().map(|r| r.shf_agency).collect();

    let g1a: PairedGate = paired_gate(&nat_net, &frz_net);
    let g1b: PairedGate = paired_gate(&nat_net, &yok_net);
    let g1c: PairedGate = paired_gate(&nat_net, &shf_net);
    let g1d: PairedGate = paired_gate(&xfer_tr, &xfer_le);
    let g2_pair: PairedGate = paired_gate(&chance_ntv, &nat_ntv); // Chance (21.28%) - Natural > 0 & < 5%
    let g3_pair: PairedGate = paired_gate(&nat_ag, &shf_ag);
    let g4a: PairedGate = paired_gate(&feed_c, &feed_y);
    let g4b: PairedGate = paired_gate(&voice_c, &voice_y);

    let raw_p = [
        g1a.p_two_sided,
        g1b.p_two_sided,
        g1c.p_two_sided,
        g1d.p_two_sided,
        g2_pair.p_two_sided,
        g3_pair.p_two_sided,
        g4a.p_two_sided,
        g4b.p_two_sided,
    ];
    let p_holm = holm_adjust(&raw_p);

    let mean_nat_ntv = mean(&nat_ntv);
    let mean_shf_ag = mean(&shf_ag);

    let pass_g1a = g1a.mean_diff > 0.0 && p_holm[0] < 0.05;
    let pass_g1b = g1b.mean_diff > 0.0 && p_holm[1] < 0.05;
    let pass_g1c = g1c.mean_diff > 0.0 && p_holm[2] < 0.05;
    let pass_g1d = g1d.mean_diff > 0.0 && p_holm[3] < 0.05;
    let pass_g2 = g2_pair.mean_diff > 0.0 && p_holm[4] < 0.05 && mean_nat_ntv < 0.05;
    let pass_g3 = g3_pair.mean_diff > 0.02 && p_holm[5] < 0.05 && mean_shf_ag <= 0.05;
    let pass_g4a = g4a.mean_diff > 0.0 && p_holm[6] < 0.05;
    let pass_g4b = g4b.mean_diff > 0.0 && p_holm[7] < 0.05;

    let gates_print = [
        ("P1A-G1a Composite27 Nat > Frozen", g1a, p_holm[0], pass_g1a),
        ("P1A-G1b Composite27 Nat > Yoked", g1b, p_holm[1], pass_g1b),
        ("P1A-G1c Composite27 Nat > Shuffled", g1c, p_holm[2], pass_g1c),
        ("P1A-G1d Composite27 Trained > Lesion", g1d, p_holm[3], pass_g1d),
        ("P1A-G2  Composite27 NoisyTV Chance>Nat (<5%)", g2_pair, p_holm[4], pass_g2),
        ("P1A-G3  Composite27 Agency Nat > Shuf", g3_pair, p_holm[5], pass_g3),
        ("P1A-G4a Shifted(+0.18)+Perm FEED_C > FEED_Y", g4a, p_holm[6], pass_g4a),
        ("P1A-G4b Shifted(+0.18)+Perm VOICE_C > VOICE_Y", g4b, p_holm[7], pass_g4b),
    ];

    println!("\n--- Behavioral & Causal Gates (Holm-Bonferroni m=8) ---");
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
        "         Composite27 mean_nat_noisy_tv_share = {:.4} (cap < 0.0500) | mean_shuf_agency = {:.4} (cap <= 0.0500)",
        mean_nat_ntv, mean_shf_ag
    );

    // 3. Hard Structural, Scaling & Migration Invariants (P1A-I1..I3).
    let debt = audit_core_debt(&ws_root);
    let pass_i1 = debt.hygiene_violations == 0
        && debt.split_brain_sites.is_empty()
        && debt.organ_mirror_sites.is_empty()
        && debt.cell_modulo_sites.is_empty();

    let (_, _, dense_p99_us, dense_pores) =
        benchmark_dense_4096(&|s| Alpha11Organism::new(s, OrganismConfig::default()), 1_000);
    let act64_res = match std::panic::catch_unwind(check_64_actuator_safety) {
        Ok(res) => res,
        Err(_) => Err("panicked on 64-actuator step".to_string()),
    };
    let pass_i2 = dense_p99_us < 2_000.0 && dense_pores <= 64 && act64_res.is_ok();

    let v5_canon = build_canonical_v5_checkpoint();
    let v5_fixture_valid =
        shadow_decode_organism(&v5_canon, OrganismConfig::default().max_state_bytes).is_ok();
    let mig_res = verify_v5_migration(opt_v5_chk.as_deref());
    let pass_i3 = v5_fixture_valid && mig_res.is_ok();

    println!("\n--- Structural, Scaling & Persistence Invariants ---");
    println!(
        "  [{:<4}] P1A-I1 Zero Core Debt: hygiene={} split_brain={} mirrors={} cell_mod={}",
        if pass_i1 { "PASS" } else { "FAIL" },
        debt.hygiene_violations,
        debt.split_brain_sites.len(),
        debt.organ_mirror_sites.len(),
        debt.cell_modulo_sites.len()
    );
    println!(
        "  [{:<4}] P1A-I2 Dense4096 & 64-Actuator Safety: p99={:.1} us (cap <2000) pores={} (cap <=64) act64={}",
        if pass_i2 { "PASS" } else { "FAIL" },
        dense_p99_us,
        dense_pores,
        act64_res.as_ref().map_or_else(|e| e.as_str(), |_| "OK")
    );
    println!(
        "  [{:<4}] P1A-I3 Legacy v5 -> v6+ Migration: v5_fixture_valid={} status={}",
        if pass_i3 { "PASS" } else { "FAIL" },
        v5_fixture_valid,
        mig_res.as_ref().map_or_else(|e| e.as_str(), |_| "OK")
    );

    let all_pass = pass_g1a
        && pass_g1b
        && pass_g1c
        && pass_g1d
        && pass_g2
        && pass_g3
        && pass_g4a
        && pass_g4b
        && pass_i1
        && pass_i2
        && pass_i3;

    println!(
        "\n=== Phase-1A Verdict: {} (elapsed {:.1}s) ===",
        if all_pass { "100% PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64()
    );

    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
