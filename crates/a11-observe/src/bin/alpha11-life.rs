//! Pre-registered `alpha11-life` judges (ledger entry 57). Locked before any tuning.
//!
//! `energy` — long-horizon energy homeostasis v2 (supersedes the confounded `alpha11-longlife`):
//! seeds DEV `0..9`, `500,000` ticks, frozen Elder, arms Natural / Frozen / Shuffled(64-ring).
//! Windows: early = ticks `(10%, 30%]`, late = final 20%. ATP sampled every 10 ticks.
//! - `E1` late net (`influx - stress`) per tick Natural > Frozen (paired t, Holm m=3, `< 0.05`).
//! - `E2` late net per tick Natural > Shuffled (paired, Holm).
//! - `E3` late mean ATP Natural > Frozen (paired, Holm) — parity with `alpha11-gates I3`.
//! - `E4` absolute: `>= 8/10` Natural seeds keep `late_influx / early_influx >= 0.80` (no income collapse).
//! - `E5` absolute: mean Natural late fraction of samples with `ATP < 0.05` `<= 0.10`.
//!
//! `teach` — caregiver programs, observed from outside (no staged exams, no internal access):
//! seeds DEV `0..9`, `40,000` ticks, frozen Elder. A voiced onset is the infant vocal organ's own
//! `syllable_onsets` counter incrementing; nothing is labelled or rewarded inside the organism.
//! - `SILENT`: no caregiver program.
//! - `FEED_C`: after an infant onset, `10` ticks later the environment opens a food pod `1.2 m`
//!   away (nutrients `0.6`); refractory `200` ticks after each delivery.
//! - `FEED_Y`: identical deliveries at the tick schedule recorded in the paired `FEED_C` run.
//! - `VOICE_C`: after an infant onset, `10` ticks later the infant hears its own `15` most recent
//!   voiced frames replayed at gain `0.8` from the Elder's direction (contingent imitation); refractory `200`.
//! - `VOICE_Y`: identical replies at the paired `VOICE_C` schedule (non-contingent imitation).
//! - Frozen-plasticity organisms also live in `FEED_C` and `FEED_Y`.
//!
//! Gates (Holm m=3, paired t, `p_holm < 0.05`, direction required):
//! - `T1` late onset rate Natural `FEED_C` > Natural `FEED_Y`.
//! - `T2` late onset rate Natural `VOICE_C` > Natural `VOICE_Y`.
//! - `T3` learning interaction `(Nat FEED_C - Nat FEED_Y) > (Frz FEED_C - Frz FEED_Y)`.
//!
//! Pre-committed interpretation: passing shows learned contingent vocal behaviour (calling that
//! is shaped by its consequences), not language and not reference.
//!
//! `teach2` (ledger entry 59, added after the `teach` v1 null): identical programs, gates, seeds and
//! ticks, with caregiver timing fitted to the organism's 2–5 tick credit window: delay `2`,
//! refractory `20`, food `0.15` at `0.6 m`.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use a11_core::{Alpha11Organism, OrganismConfig};
use a11_medium::{ControlMode, FieldPulse, OrganismEngine, WorldMedium};
use a11_observe::gates::{PairedGate, holm_adjust, paired_gate};
use a11_observe::open_habitat::{FRAME, OPEN_ACT, OpenHabitat, elder_reference_hash};
use a11_observe::stats::DeterministicRng;

const ENERGY_SALT: u64 = 0x10A6_11FE_2026_2002;
const TEACH_SALT: u64 = 0x7EAC_4C41_2026_2003;
const STARVE_ATP: f32 = 0.05;
const SERIES_POINTS: usize = 20;
/// Caregiver timing for a teach study.
#[derive(Clone, Copy, Debug)]
struct TeachParams {
    study: &'static str,
    delay: u64,
    refract: u64,
    nutrients: f32,
    dist: f32,
}

/// `teach` v1 (locked at `bebcc10`).
const TEACH_V1: TeachParams = TeachParams {
    study: "teach",
    delay: 10,
    refract: 200,
    nutrients: 0.6,
    dist: 1.2,
};

/// `teach2` (ledger entry 59): fits the organism's 2–5 tick credit window and removes schedule
/// saturation so the delivery rate scales with the calling rate. Same programs, gates and seeds.
const TEACH_V2: TeachParams = TeachParams {
    study: "teach2",
    delay: 2,
    refract: 20,
    nutrients: 0.15,
    dist: 0.6,
};
const REPLY_LEN: usize = 15;
const REPLY_GAIN: f32 = 0.8;
const HUNGRY_ATP: f32 = 6.0;

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / (v.len() as f64) }
}

fn motor_energy(p: &FieldPulse) -> f32 {
    if p.actuator_out.is_empty() {
        return 0.0;
    }
    let s: f32 = p.actuator_out.iter().map(|v| v * v).sum();
    (s / (p.actuator_out.len() as f32)).clamp(0.0, 4.0)
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

// =============================================================================
// energy v2
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EArm {
    Natural,
    Frozen,
    Shuffled,
}

impl EArm {
    fn label(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Frozen => "frozen",
            Self::Shuffled => "shuffled",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EResult {
    late_atp: f64,
    late_net: f64,
    late_starve: f64,
    early_influx: f64,
    late_influx: f64,
    cells: f64,
    births: f64,
    mitosis_per_tick: f64,
    rent_per_tick: f64,
    motor_per_tick: f64,
    repair_per_tick: f64,
    series: Vec<[f64; 8]>,
}

fn energy_arm(seed: u64, arm: EArm, ticks: usize) -> EResult {
    let mut org = Alpha11Organism::new(seed, OrganismConfig::default());
    if arm == EArm::Frozen {
        org.set_control_mode(ControlMode::FrozenPlasticity);
    }
    let mut hab = OpenHabitat::new(seed ^ ENERGY_SALT);
    let mut rng = DeterministicRng::new(seed ^ 0x540F_F1ED_2002_2026);
    let mut pool: Vec<FieldPulse> = (0..64)
        .map(|_| FieldPulse {
            actuator_out: (0..OPEN_ACT)
                .map(|_| (rng.next_f64() as f32) * 2.0 - 1.0)
                .collect(),
        })
        .collect();
    let early_lo = ticks / 10;
    let early_hi = ticks * 3 / 10;
    let late_start = ticks - ticks / 5;
    let series_every = (ticks / SERIES_POINTS).max(1);

    let mut r = EResult::default();
    let (mut late_atp, mut late_n, mut late_starve) = (0.0_f64, 0_u64, 0_u64);
    let (mut late_net, mut late_influx, mut early_influx) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut mitosis, mut rent, mut motor, mut repair) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut win_in, mut win_st, mut win_starve, mut win_n) = (0.0_f64, 0.0_f64, 0_u64, 0_u64);
    let mut prev_births = 0_u64;
    let mut rent_now = 0.0_f64;

    for t in 1..=ticks {
        let emitted = org
            .step(hab.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACT));
        motor += 0.0018 * f64::from(motor_energy(&emitted));
        if arm == EArm::Shuffled {
            let idx = rng.next_usize(pool.len());
            let delivered = std::mem::replace(&mut pool[idx], emitted);
            hab.apply_pulse(&delivered);
        } else {
            hab.apply_pulse(&emitted);
        }
        if (t & 15) == 0 {
            hab.sync_vocal_ontogeny(&org);
        }
        let v = hab.current_view();
        let influx = f64::from(v.metabolic_influx);
        let stress = f64::from(v.tissue_stress);
        repair += 0.05 * stress;
        rent += rent_now;
        win_in += influx;
        win_st += stress;
        if t > early_lo && t <= early_hi {
            early_influx += influx;
        }
        let in_late = t > late_start;
        if in_late {
            late_net += influx - stress;
            late_influx += influx;
        }
        if t % 10 == 0 {
            let s = org.snapshot();
            let starving = s.atp < STARVE_ATP;
            win_n += 1;
            if starving {
                win_starve += 1;
            }
            if in_late {
                late_atp += f64::from(s.atp);
                late_n += 1;
                if starving {
                    late_starve += 1;
                }
            }
            rent_now = 0.000_010 * (s.cell_count as f64)
                + 0.000_000_4 * (s.tube_count as f64)
                + 0.000_004 * ((s.sensor_pore_count + s.motor_pore_count) as f64);
            mitosis += 0.40 * (s.mitosis_births.saturating_sub(prev_births) as f64);
            prev_births = s.mitosis_births;
        }
        if t % series_every == 0 {
            let s = org.snapshot();
            let n = series_every as f64;
            r.series.push([
                t as f64,
                f64::from(s.atp),
                s.cell_count as f64,
                s.mitosis_births as f64,
                win_in / n,
                win_st / n,
                if win_n > 0 { (win_starve as f64) / (win_n as f64) } else { 0.0 },
                f64::from(s.mean_interventional_agency),
            ]);
            win_in = 0.0;
            win_st = 0.0;
            win_starve = 0;
            win_n = 0;
        }
    }
    let s = org.snapshot();
    let tf = ticks as f64;
    let late_t = (ticks - late_start).max(1) as f64;
    r.late_atp = late_atp / (late_n.max(1) as f64);
    r.late_net = late_net / late_t;
    r.late_starve = (late_starve as f64) / (late_n.max(1) as f64);
    r.early_influx = early_influx / ((early_hi - early_lo).max(1) as f64);
    r.late_influx = late_influx / late_t;
    r.cells = s.cell_count as f64;
    r.births = s.mitosis_births as f64;
    r.mitosis_per_tick = mitosis / tf;
    r.rent_per_tick = rent / tf;
    r.motor_per_tick = motor / tf;
    r.repair_per_tick = repair / tf;
    r
}

fn gate_line(name: &str, g: &PairedGate, p_holm: f64, pass: bool) -> String {
    format!(
        "{name:<40} a={:+.5} b={:+.5} t={:+.2} wins={}/{} p_holm={:.5} {}",
        g.mean_a,
        g.mean_b,
        g.t,
        g.wins,
        g.n,
        p_holm,
        if pass { "PASS" } else { "FAIL" }
    )
}

fn run_energy(cli: &Cli) -> Result<String, String> {
    let arms = [EArm::Natural, EArm::Frozen, EArm::Shuffled];
    let jobs: Vec<(u64, EArm)> = (0..cli.seeds as u64)
        .flat_map(|k| arms.iter().map(move |&a| (cli.seed_offset + k, a)))
        .collect();
    let ticks = cli.ticks.unwrap_or(500_000);
    let elder_hash = elder_reference_hash(cli.seed_offset ^ ENERGY_SALT, 2_000);
    let res = run_jobs(jobs.len(), cli.threads, |j| energy_arm(jobs[j].0, jobs[j].1, ticks));
    let mut by: [Vec<EResult>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (j, r) in res.into_iter().enumerate() {
        let r = r.ok_or_else(|| format!("energy job {j} failed"))?;
        let ai = arms.iter().position(|&a| a == jobs[j].1).unwrap_or(0);
        by[ai].push(r);
    }
    let col = |ai: usize, f: fn(&EResult) -> f64| -> Vec<f64> { by[ai].iter().map(f).collect() };
    let g1 = paired_gate(&col(0, |r| r.late_net), &col(1, |r| r.late_net));
    let g2 = paired_gate(&col(0, |r| r.late_net), &col(2, |r| r.late_net));
    let g3 = paired_gate(&col(0, |r| r.late_atp), &col(1, |r| r.late_atp));
    let h = holm_adjust(&[g1.p_two_sided, g2.p_two_sided, g3.p_two_sided]);
    let e1 = g1.mean_diff > 0.0 && h[0] < 0.05;
    let e2 = g2.mean_diff > 0.0 && h[1] < 0.05;
    let e3 = g3.mean_diff > 0.0 && h[2] < 0.05;
    let ratios: Vec<f64> = by[0]
        .iter()
        .map(|r| r.late_influx / r.early_influx.max(1.0e-9))
        .collect();
    let ratio = mean(&ratios);
    let kept = ratios.iter().filter(|&&x| x >= 0.80).count();
    let e4 = kept * 10 >= 8 * ratios.len();
    let starve = mean(&col(0, |r| r.late_starve));
    let e5 = starve <= 0.10;
    let passed = [e1, e2, e3, e4, e5].iter().filter(|&&p| p).count();

    let mut out = String::new();
    let _ = writeln!(out, "elder_reference_hash={elder_hash:016x}");
    let _ = writeln!(out, "{}", gate_line("E1 late net Natural>Frozen", &g1, h[0], e1));
    let _ = writeln!(out, "{}", gate_line("E2 late net Natural>Shuffled", &g2, h[1], e2));
    let _ = writeln!(out, "{}", gate_line("E3 late ATP Natural>Frozen", &g3, h[2], e3));
    let _ = writeln!(
        out,
        "E4 income kept late/early>=0.80 in >=80% seeds  kept={kept}/{} mean={ratio:.4} per-seed={:?} {}",
        ratios.len(),
        ratios.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(),
        if e4 { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(
        out,
        "E5 late starvation frac (<=0.10)          nat={starve:.4} {}",
        if e5 { "PASS" } else { "FAIL" }
    );
    let _ = writeln!(out, "RESULT {passed}/5");
    for (ai, a) in arms.iter().enumerate() {
        let _ = writeln!(
            out,
            "[{}] late_atp={:.3} late_starve={:.3} early_influx={:.5} late_influx={:.5} late_net={:+.5} cells={:.1} births={:.0} | per-tick mitosis={:.5} rent={:.5} motor={:.5} repair={:.5}",
            a.label(),
            mean(&col(ai, |r| r.late_atp)),
            mean(&col(ai, |r| r.late_starve)),
            mean(&col(ai, |r| r.early_influx)),
            mean(&col(ai, |r| r.late_influx)),
            mean(&col(ai, |r| r.late_net)),
            mean(&col(ai, |r| r.cells)),
            mean(&col(ai, |r| r.births)),
            mean(&col(ai, |r| r.mitosis_per_tick)),
            mean(&col(ai, |r| r.rent_per_tick)),
            mean(&col(ai, |r| r.motor_per_tick)),
            mean(&col(ai, |r| r.repair_per_tick)),
        );
    }
    let _ = writeln!(out, "natural mean series [t, atp, cells, births, influx, stress, starve, agency]:");
    let npts = by[0].first().map_or(0, |r| r.series.len());
    for q in 0..npts {
        let mut row = [0.0_f64; 8];
        for r in &by[0] {
            for (k, v) in r.series[q].iter().enumerate() {
                row[k] += v / (by[0].len() as f64);
            }
        }
        let _ = writeln!(
            out,
            "  {:>7.0} atp={:7.3} cells={:6.1} births={:7.0} influx={:.5} stress={:.5} starve={:.3} agency={:.4}",
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]
        );
    }
    let mut js = String::new();
    let _ = write!(js, "{{\"study\":\"energy_v2\",\"ticks\":{ticks},\"elder_hash\":\"{elder_hash:016x}\",\"passed\":{passed},\"arms\":{{");
    for (ai, a) in arms.iter().enumerate() {
        if ai > 0 {
            js.push(',');
        }
        let _ = write!(js, "\"{}\":[", a.label());
        for (k, r) in by[ai].iter().enumerate() {
            if k > 0 {
                js.push(',');
            }
            let _ = write!(
                js,
                "{{\"late_atp\":{:.5},\"late_net\":{:.6},\"late_starve\":{:.5},\"early_influx\":{:.6},\"late_influx\":{:.6},\"cells\":{},\"births\":{},\"series\":{:?}}}",
                r.late_atp, r.late_net, r.late_starve, r.early_influx, r.late_influx, r.cells, r.births, r.series
            );
        }
        js.push(']');
    }
    js.push_str("}}\n");
    write_report(cli, "energy", &js)?;
    Ok(out)
}

// =============================================================================
// teach
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Program {
    Silent,
    FeedC,
    FeedY,
    VoiceC,
    VoiceY,
}

impl Program {
    fn label(self) -> &'static str {
        match self {
            Self::Silent => "SILENT",
            Self::FeedC => "FEED_C",
            Self::FeedY => "FEED_Y",
            Self::VoiceC => "VOICE_C",
            Self::VoiceY => "VOICE_Y",
        }
    }
    fn contingent(self) -> bool {
        matches!(self, Self::FeedC | Self::VoiceC)
    }
    fn feeds(self) -> bool {
        matches!(self, Self::FeedC | Self::FeedY)
    }
    fn speaks(self) -> bool {
        matches!(self, Self::VoiceC | Self::VoiceY)
    }
}

#[derive(Clone, Debug, Default)]
struct TResult {
    events: Vec<u64>,
    early_onsets_pk: f64,
    late_onsets_pk: f64,
    late_voiced_frac: f64,
    late_transitions_pk: f64,
    late_alt_resp_pk: f64,
    onsets_pk_hungry: f64,
    onsets_pk_fed: f64,
    late_net: f64,
    delivered: u64,
    final_atp: f64,
}

fn teach_arm(
    tp: &TeachParams,
    seed: u64,
    program: Program,
    frozen: bool,
    ticks: usize,
    schedule: &[u64],
) -> TResult {
    // Yoked twins get an independent birth seed (same rule as the locked `alpha11-gates` Yoked arm);
    // with the same seed a deterministic twin would replay the contingent life exactly.
    let org_seed = if program.contingent() || program == Program::Silent {
        seed
    } else {
        seed + 500_000
    };
    let mut org = Alpha11Organism::new(org_seed, OrganismConfig::default());
    if frozen {
        org.set_control_mode(ControlMode::FrozenPlasticity);
    }
    let mut hab = OpenHabitat::new(seed ^ TEACH_SALT);
    let mut rng = DeterministicRng::new(seed ^ 0xFEED_C0DE_2026_2003);
    let ticks_u = ticks as u64;
    let early_lo = ticks_u / 10;
    let early_hi = ticks_u * 3 / 10;
    let late_start = ticks_u - ticks_u / 5;

    let mut ring = [[0.0_f32; FRAME]; REPLY_LEN];
    let mut ring_len = 0_usize;
    let mut ring_head = 0_usize;
    let mut reply_frames: Vec<[f32; FRAME]> = Vec::new();
    let mut reply_start = 0_u64;
    let mut pending: Option<u64> = None;
    let mut next_allowed = 0_u64;
    let mut sched_idx = 0_usize;

    let obs0 = hab.focal_body.vocal().observation();
    let (mut p_on, mut p_voi, mut p_tr, mut p_alt) = (
        obs0.syllable_onsets,
        obs0.voiced_ticks,
        obs0.syllable_transitions,
        obs0.alternating_response_ticks,
    );
    let mut r = TResult::default();
    let (mut early_on, mut late_on, mut late_voi, mut late_tr, mut late_alt) = (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
    let (mut hun_t, mut hun_on, mut fed_t, mut fed_on) = (0_u64, 0_u64, 0_u64, 0_u64);
    let mut late_net = 0.0_f64;

    for t in 1..=ticks_u {
        // Caregiver reply frame for this tick (VOICE programs), heard during this step.
        if program.speaks() && !reply_frames.is_empty() && t >= reply_start {
            let k = (t - reply_start) as usize;
            if let Some(f) = reply_frames.get(k) {
                let mut scaled = [0.0_f32; FRAME];
                for (d, &s) in scaled.iter_mut().zip(f.iter()) {
                    *d = REPLY_GAIN * s;
                }
                hab.set_reply_frame(Some(scaled));
            } else {
                reply_frames.clear();
            }
        }
        let atp_before = org.snapshot().atp;
        let emitted = org
            .step(hab.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(OPEN_ACT));
        hab.apply_pulse(&emitted);
        if (t & 15) == 0 {
            hab.sync_vocal_ontogeny(&org);
        }

        let o = hab.focal_body.vocal().observation();
        let d_on = o.syllable_onsets.saturating_sub(p_on);
        let d_voi = o.voiced_ticks.saturating_sub(p_voi);
        let d_tr = o.syllable_transitions.saturating_sub(p_tr);
        let d_alt = o.alternating_response_ticks.saturating_sub(p_alt);
        p_on = o.syllable_onsets;
        p_voi = o.voiced_ticks;
        p_tr = o.syllable_transitions;
        p_alt = o.alternating_response_ticks;
        if d_voi > 0 {
            ring[ring_head].copy_from_slice(hab.focal_body.vocal().last_radiated_wave());
            ring_head = (ring_head + 1) % REPLY_LEN;
            ring_len = (ring_len + 1).min(REPLY_LEN);
        }
        if atp_before < HUNGRY_ATP {
            hun_t += 1;
            hun_on += d_on;
        } else {
            fed_t += 1;
            fed_on += d_on;
        }
        if t > early_lo && t <= early_hi {
            early_on += d_on;
        }
        if t > late_start {
            late_on += d_on;
            late_voi += d_voi;
            late_tr += d_tr;
            late_alt += d_alt;
            let v = hab.current_view();
            late_net += f64::from(v.metabolic_influx) - f64::from(v.tissue_stress);
        }

        // Event scheduling: an event stamped `e` is applied at the end of tick `e - 1`,
        // so its consequences are first sensed on tick `e`.
        let mut fire: Option<u64> = None;
        if program.contingent() {
            if d_on > 0 && pending.is_none() && t >= next_allowed {
                pending = Some(t + tp.delay);
                next_allowed = t + tp.delay + tp.refract;
            }
            if let Some(e) = pending
                && t + 1 == e
            {
                fire = Some(e);
                pending = None;
            }
        } else if program != Program::Silent
            && sched_idx < schedule.len()
            && schedule[sched_idx] == t + 1
        {
            fire = Some(t + 1);
            sched_idx += 1;
        }
        if let Some(e) = fire {
            r.events.push(e);
            if program.feeds() {
                let theta = (rng.next_f64() as f32) * std::f32::consts::TAU;
                if hab.terrarium.provision_open_pod_near(
                    1,
                    (tp.dist * theta.cos(), tp.dist * theta.sin()),
                    tp.nutrients,
                ) {
                    r.delivered += 1;
                }
            } else if program.speaks() && ring_len > 0 {
                reply_frames.clear();
                for k in 0..ring_len {
                    let idx = (ring_head + REPLY_LEN - ring_len + k) % REPLY_LEN;
                    reply_frames.push(ring[idx]);
                }
                reply_start = e;
                r.delivered += 1;
            }
        }
    }
    let late_t = (ticks_u - late_start).max(1) as f64;
    r.early_onsets_pk = 1000.0 * (early_on as f64) / ((early_hi - early_lo).max(1) as f64);
    r.late_onsets_pk = 1000.0 * (late_on as f64) / late_t;
    r.late_voiced_frac = (late_voi as f64) / late_t;
    r.late_transitions_pk = 1000.0 * (late_tr as f64) / late_t;
    r.late_alt_resp_pk = 1000.0 * (late_alt as f64) / late_t;
    r.onsets_pk_hungry = 1000.0 * (hun_on as f64) / (hun_t.max(1) as f64);
    r.onsets_pk_fed = 1000.0 * (fed_on as f64) / (fed_t.max(1) as f64);
    r.late_net = late_net / late_t;
    r.final_atp = f64::from(org.snapshot().atp);
    r
}

fn run_teach(cli: &Cli, tp: &TeachParams) -> Result<String, String> {
    let ticks = cli.ticks.unwrap_or(40_000);
    let seeds: Vec<u64> = (0..cli.seeds as u64).map(|k| cli.seed_offset + k).collect();
    // Phase 1: SILENT, FEED_C, VOICE_C (Natural) and FEED_C (Frozen).
    let p1: Vec<(Program, bool)> = vec![
        (Program::Silent, false),
        (Program::FeedC, false),
        (Program::VoiceC, false),
        (Program::FeedC, true),
    ];
    let jobs1: Vec<(u64, Program, bool)> = seeds
        .iter()
        .flat_map(|&s| p1.iter().map(move |&(p, f)| (s, p, f)))
        .collect();
    let r1 = run_jobs(jobs1.len(), cli.threads, |j| {
        teach_arm(tp, jobs1[j].0, jobs1[j].1, jobs1[j].2, ticks, &[])
    });
    let r1: Vec<TResult> = r1
        .into_iter()
        .enumerate()
        .map(|(j, r)| r.ok_or_else(|| format!("teach phase-1 job {j} failed")))
        .collect::<Result<_, _>>()?;
    let find1 = |s: u64, p: Program, f: bool| -> Option<&TResult> {
        jobs1
            .iter()
            .position(|&(js, jp, jf)| js == s && jp == p && jf == f)
            .map(|i| &r1[i])
    };
    // Phase 2: yoked arms replay the paired contingent schedule.
    let p2: Vec<(Program, bool, Program)> = vec![
        (Program::FeedY, false, Program::FeedC),
        (Program::VoiceY, false, Program::VoiceC),
        (Program::FeedY, true, Program::FeedC),
    ];
    let mut jobs2: Vec<(u64, Program, bool, Vec<u64>)> = Vec::new();
    for &s in &seeds {
        for &(p, f, src) in &p2 {
            let sched = find1(s, src, f).map(|r| r.events.clone()).unwrap_or_default();
            jobs2.push((s, p, f, sched));
        }
    }
    let r2 = run_jobs(jobs2.len(), cli.threads, |j| {
        teach_arm(tp, jobs2[j].0, jobs2[j].1, jobs2[j].2, ticks, &jobs2[j].3)
    });
    let r2: Vec<TResult> = r2
        .into_iter()
        .enumerate()
        .map(|(j, r)| r.ok_or_else(|| format!("teach phase-2 job {j} failed")))
        .collect::<Result<_, _>>()?;
    let get = |s: u64, p: Program, f: bool| -> Option<&TResult> {
        find1(s, p, f).or_else(|| {
            jobs2
                .iter()
                .position(|(js, jp, jf, _)| *js == s && *jp == p && *jf == f)
                .map(|i| &r2[i])
        })
    };
    let col = |p: Program, f: bool, m: fn(&TResult) -> f64| -> Vec<f64> {
        seeds.iter().filter_map(|&s| get(s, p, f).map(m)).collect()
    };
    let on = |r: &TResult| r.late_onsets_pk;
    let nfc = col(Program::FeedC, false, on);
    let nfy = col(Program::FeedY, false, on);
    let nvc = col(Program::VoiceC, false, on);
    let nvy = col(Program::VoiceY, false, on);
    let ffc = col(Program::FeedC, true, on);
    let ffy = col(Program::FeedY, true, on);
    let nat_diff: Vec<f64> = nfc.iter().zip(&nfy).map(|(a, b)| a - b).collect();
    let frz_diff: Vec<f64> = ffc.iter().zip(&ffy).map(|(a, b)| a - b).collect();
    let g1 = paired_gate(&nfc, &nfy);
    let g2 = paired_gate(&nvc, &nvy);
    let g3 = paired_gate(&nat_diff, &frz_diff);
    let h = holm_adjust(&[g1.p_two_sided, g2.p_two_sided, g3.p_two_sided]);
    let t1 = g1.mean_diff > 0.0 && h[0] < 0.05;
    let t2 = g2.mean_diff > 0.0 && h[1] < 0.05;
    let t3 = g3.mean_diff > 0.0 && h[2] < 0.05;
    let passed = [t1, t2, t3].iter().filter(|&&p| p).count();

    let mut out = String::new();
    let _ = writeln!(out, "{}", gate_line("T1 late onsets/k Nat FEED_C>FEED_Y", &g1, h[0], t1));
    let _ = writeln!(out, "{}", gate_line("T2 late onsets/k Nat VOICE_C>VOICE_Y", &g2, h[1], t2));
    let _ = writeln!(out, "{}", gate_line("T3 (NatC-NatY)>(FrzC-FrzY) FEED", &g3, h[2], t3));
    let _ = writeln!(out, "RESULT {passed}/3");
    let arms_all: [(Program, bool); 7] = [
        (Program::Silent, false),
        (Program::FeedC, false),
        (Program::FeedY, false),
        (Program::VoiceC, false),
        (Program::VoiceY, false),
        (Program::FeedC, true),
        (Program::FeedY, true),
    ];
    let mut js = String::new();
    let _ = write!(js, "{{\"study\":\"{}\",\"ticks\":{ticks},\"passed\":{passed},\"arms\":{{", tp.study);
    for (ai, &(p, f)) in arms_all.iter().enumerate() {
        let m = |g: fn(&TResult) -> f64| mean(&col(p, f, g));
        let label = format!("{}{}", if f { "frz_" } else { "nat_" }, p.label());
        let _ = writeln!(
            out,
            "[{label:<12}] onsets/k early={:.2} late={:.2} | hungry={:.2} fed={:.2} | voiced={:.3} transitions/k={:.2} alt_resp/k={:.2} | events={:.0} delivered={:.0} late_net={:+.5} atp={:.2}",
            m(|r| r.early_onsets_pk),
            m(|r| r.late_onsets_pk),
            m(|r| r.onsets_pk_hungry),
            m(|r| r.onsets_pk_fed),
            m(|r| r.late_voiced_frac),
            m(|r| r.late_transitions_pk),
            m(|r| r.late_alt_resp_pk),
            m(|r| r.events.len() as f64),
            m(|r| r.delivered as f64),
            m(|r| r.late_net),
            m(|r| r.final_atp),
        );
        if ai > 0 {
            js.push(',');
        }
        let _ = write!(js, "\"{label}\":[");
        for (k, &s) in seeds.iter().enumerate() {
            if let Some(r) = get(s, p, f) {
                if k > 0 {
                    js.push(',');
                }
                let _ = write!(
                    js,
                    "{{\"seed\":{s},\"early_onsets_pk\":{:.4},\"late_onsets_pk\":{:.4},\"hungry_pk\":{:.4},\"fed_pk\":{:.4},\"voiced\":{:.5},\"transitions_pk\":{:.4},\"alt_resp_pk\":{:.4},\"events\":{},\"delivered\":{},\"late_net\":{:.6},\"atp\":{:.4}}}",
                    r.early_onsets_pk, r.late_onsets_pk, r.onsets_pk_hungry, r.onsets_pk_fed,
                    r.late_voiced_frac, r.late_transitions_pk, r.late_alt_resp_pk,
                    r.events.len(), r.delivered, r.late_net, r.final_atp
                );
            }
        }
        js.push(']');
    }
    js.push_str("}}\n");
    write_report(cli, tp.study, &js)?;
    Ok(out)
}

// =============================================================================
// CLI
// =============================================================================

struct Cli {
    cmd: String,
    seeds: usize,
    seed_offset: u64,
    ticks: Option<usize>,
    threads: usize,
    tag: String,
}

fn resolve_workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn write_report(cli: &Cli, study: &str, js: &str) -> Result<(), String> {
    let dir = resolve_workspace_root().join("observations").join("life");
    let path = dir.join(format!("{study}_{}.json", cli.tag));
    fs::create_dir_all(&dir)
        .and_then(|()| fs::write(&path, js))
        .map_err(|e| format!("write report: {e}"))
}

fn parse_cli() -> Result<Cli, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().ok_or("missing subcommand (energy | teach | teach2)")?;
    if cmd != "energy" && cmd != "teach" && cmd != "teach2" {
        return Err(format!("unknown subcommand {cmd}"));
    }
    let mut cli = Cli {
        cmd,
        seeds: 10,
        seed_offset: 0,
        ticks: None,
        threads: 12,
        tag: "run".to_string(),
    };
    let mut i = 1;
    while i < args.len() {
        let key = args[i].as_str();
        let val = args.get(i + 1).ok_or_else(|| format!("missing value after {key}"))?;
        match key {
            "--seeds" => cli.seeds = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--seed-offset" => cli.seed_offset = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--ticks" => cli.ticks = Some(val.parse().map_err(|e| format!("{key}: {e}"))?),
            "--threads" => cli.threads = val.parse().map_err(|e| format!("{key}: {e}"))?,
            "--tag" => cli.tag.clone_from(val),
            other => return Err(format!("unknown argument {other}")),
        }
        i += 2;
    }
    let lo = cli.seed_offset;
    let hi = cli.seed_offset + cli.seeds as u64;
    if lo < 212 && hi > 200 {
        return Err("seeds 200..211 are the sealed heldout vault".to_string());
    }
    cli.seeds = cli.seeds.max(1);
    cli.threads = cli.threads.max(1);
    if let Some(t) = cli.ticks {
        cli.ticks = Some(t.max(2_000));
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = match parse_cli() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[alpha11-life] {e}");
            eprintln!(
                "Usage: alpha11-life energy|teach|teach2 [--seeds 10] [--seed-offset 0] [--ticks N] [--threads 12] [--tag name]"
            );
            return ExitCode::from(2);
        }
    };
    let t0 = Instant::now();
    let res = match cli.cmd.as_str() {
        "energy" => run_energy(&cli),
        "teach2" => run_teach(&cli, &TEACH_V2),
        _ => run_teach(&cli, &TEACH_V1),
    };
    match res {
        Ok(text) => {
            println!(
                "== alpha11-life {} | tag={} | seeds {}..{} | wall={:.1}s ==",
                cli.cmd,
                cli.tag,
                cli.seed_offset,
                cli.seed_offset + cli.seeds as u64 - 1,
                t0.elapsed().as_secs_f64()
            );
            print!("{text}");
            println!("report: observations/life/{}_{}.json", cli.cmd, cli.tag);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[alpha11-life] {e}");
            ExitCode::from(1)
        }
    }
}
