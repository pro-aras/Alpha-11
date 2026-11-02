//! Locked pre-registered closed-loop scientific gate harness (`G1..G9` + `I1..I3`).
//!
//! Every behavioral gate compares paired per-seed `net = influx - stress` achieved in the real
//! world (never yoked replay) across `DEV-30` (`0..29`), applying exact two-sided Student-t tests
//! and Holm-Bonferroni family-wise error control (`alpha = 0.05`).
//!
//! Additional hard biophysical/causal invariants (`I1..I3`) enforce:
//! - `I1`: Counterfactual agency purity (`Natural > Shuffled` & `Natural > Exogenous`, with
//!   `mean(shuffled_agency) <= 0.05` and `mean(exogenous_agency) <= 0.05`).
//! - `I2`: Bounded `O(1)` scaling at `4,096` channels (`p99 < 2,000 us` and `sensor_pore_count <= 64`).
//! - `I3`: Honest `ATP` homeostasis (`0` mitoses while `ATP <= 0.01` under starvation, and
//!   `Natural final ATP > Frozen final ATP` in `BrightRadiantWorld`).

use std::time::Instant;

use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, WorldMedium};
use a11_worlds::{
    ActionShuffledMedium, BrightRadiantWorld, DenseFieldWorld, DeterministicRng, PermutedMedium,
    YokedMedium,
};

/// Paired comparison of `a` (candidate) against `b` (control).
#[derive(Clone, Copy, Debug, Default)]
pub struct PairedGate {
    pub n: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    pub mean_diff: f64,
    pub t: f64,
    pub p_two_sided: f64,
    pub wins: usize,
}

#[must_use]
pub fn paired_gate(a: &[f64], b: &[f64]) -> PairedGate {
    let n = a.len().min(b.len());
    if n == 0 {
        return PairedGate::default();
    }
    let nf = n as f64;
    let d: Vec<f64> = (0..n).map(|i| a[i] - b[i]).collect();
    let mean_diff = d.iter().sum::<f64>() / nf;
    let var = if n > 1 {
        d.iter().map(|x| (x - mean_diff).powi(2)).sum::<f64>() / (nf - 1.0)
    } else {
        0.0
    };
    let se = (var / nf).sqrt();
    let t = if se > 1e-12 { mean_diff / se } else { 0.0 };
    let p = if n > 1 && se > 1e-12 {
        student_t_two_sided_p(t, nf - 1.0)
    } else {
        1.0
    };
    PairedGate {
        n,
        mean_a: a[..n].iter().sum::<f64>() / nf,
        mean_b: b[..n].iter().sum::<f64>() / nf,
        mean_diff,
        t,
        p_two_sided: p,
        wins: d.iter().filter(|&&x| x > 0.0).count(),
    }
}

/// Exact two-sided p-value of Student's t with `df` degrees of freedom.
#[must_use]
pub fn student_t_two_sided_p(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    regularized_incomplete_beta(0.5 * df, 0.5, x).clamp(0.0, 1.0)
}

fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`.
#[must_use]
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Holm-Bonferroni step-down adjusted p-values (same order as input).
#[must_use]
pub fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| p[i].total_cmp(&p[j]));
    let mut adj = vec![1.0; m];
    let mut running = 0.0_f64;
    for (rank, &i) in idx.iter().enumerate() {
        let v = ((m - rank) as f64 * p[i]).min(1.0);
        running = running.max(v);
        adj[i] = running;
    }
    adj
}

/// Adversarial distractor wrapper appending 32 Smooth Ornstein-Uhlenbeck (`rho = 0.97`)
/// exogenous channels and 32 Delayed Motor Self-Copy (`lag = 3`) reafference channels.
#[derive(Clone, Debug)]
pub struct TortureMedium<W: WorldMedium> {
    inner: W,
    ou_rng: DeterministicRng,
    ou_state: [f32; 32],
    motor_ring: [Vec<f32>; 3],
    ring_cursor: usize,
    combined_ambient: Vec<f32>,
}

impl<W: WorldMedium> TortureMedium<W> {
    pub const OU_CHANNELS: usize = 32;
    pub const SELF_COPY_CHANNELS: usize = 32;

    #[must_use]
    pub fn new(inner: W, seed: u64) -> Self {
        let n_base = inner.num_ambient_channels();
        let n_act = inner.num_actuators().max(1);
        let mut m = Self {
            inner,
            ou_rng: DeterministicRng::new(seed ^ 0x0097_D157_EA01),
            ou_state: [0.0; 32],
            motor_ring: std::array::from_fn(|_| vec![0.0; n_act]),
            ring_cursor: 0,
            combined_ambient: vec![0.0; n_base + Self::OU_CHANNELS + Self::SELF_COPY_CHANNELS],
        };
        m.sync_ambient(&[0.0; 8]);
        m
    }

    fn sync_ambient(&mut self, delayed_motor: &[f32]) {
        let raw = self.inner.current_view().ambient_in;
        let n_base = raw.len();
        self.combined_ambient[..n_base].copy_from_slice(raw);

        // Smooth OU(rho = 0.97): defeats naive passive AR(1)/roughness sieves.
        let rho = 0.97_f32;
        let innov = (1.0 - rho * rho).sqrt() * 0.75;
        for i in 0..Self::OU_CHANNELS {
            let eps = self.ou_rng.next_f32_bipolar();
            self.ou_state[i] = (rho * self.ou_state[i] + innov * eps).clamp(-1.0, 1.0);
            self.combined_ambient[n_base + i] = self.ou_state[i];
        }

        // Delayed motor self-copy (lag = 3): 100% action-correlated self-echo with zero world info.
        let n_act = delayed_motor.len().max(1);
        for i in 0..Self::SELF_COPY_CHANNELS {
            let m_val = delayed_motor.get(i % n_act).copied().unwrap_or(0.0);
            let sign = if (i & 1) == 0 { 0.85 } else { -0.85 };
            let jitter = 0.04 * self.ou_rng.next_f32_bipolar();
            self.combined_ambient[n_base + Self::OU_CHANNELS + i] =
                (sign * m_val + jitter).clamp(-1.0, 1.0);
        }
    }
}

impl<W: WorldMedium> WorldMedium for TortureMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.combined_ambient.len()
    }

    fn num_actuators(&self) -> usize {
        self.inner.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        let iv = self.inner.current_view();
        FieldView {
            ambient_in: &self.combined_ambient,
            metabolic_influx: iv.metabolic_influx,
            tissue_stress: iv.tissue_stress,
            num_actuators: iv.num_actuators,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let delayed = self.motor_ring[self.ring_cursor].clone();
        let n_act = self.motor_ring[self.ring_cursor].len();
        for i in 0..n_act {
            self.motor_ring[self.ring_cursor][i] = pulse
                .actuator_out
                .get(i)
                .copied()
                .unwrap_or(0.0)
                .clamp(-1.0, 1.0);
        }
        self.ring_cursor = (self.ring_cursor + 1) % 3;
        self.inner.apply_pulse(pulse);
        self.sync_ambient(&delayed);
    }
}

/// Exogenous-only control wrapper: blocks all motor effects on the inner world (`FieldPulse::silent`),
/// while exogenous smooth `OU(0.97)` and `NoisyTV` channels continue evolving normally.
#[derive(Clone, Debug)]
pub struct ExogenousOnlyMedium {
    inner: BrightRadiantWorld,
    ou_rng: DeterministicRng,
    ou_state: [f32; 32],
    ambient: Vec<f32>,
    silent: FieldPulse,
    starve_stress: f32,
}

impl ExogenousOnlyMedium {
    #[must_use]
    pub fn new(seed: u64, starve_stress: f32) -> Self {
        let inner = BrightRadiantWorld::new(seed);
        let n_base = inner.num_ambient_channels();
        let n_act = inner.num_actuators();
        let mut m = Self {
            inner,
            ou_rng: DeterministicRng::new(seed ^ 0xE706_E400_5EED),
            ou_state: [0.0; 32],
            ambient: vec![0.0; n_base + 32],
            silent: FieldPulse::silent(n_act),
            starve_stress,
        };
        m.refresh();
        m
    }

    fn refresh(&mut self) {
        let raw = self.inner.current_view().ambient_in;
        let n_base = raw.len();
        self.ambient[..n_base].copy_from_slice(raw);
        let rho = 0.97_f32;
        let innov = (1.0 - rho * rho).sqrt() * 0.75;
        for i in 0..32 {
            let eps = self.ou_rng.next_f32_bipolar();
            self.ou_state[i] = (rho * self.ou_state[i] + innov * eps).clamp(-1.0, 1.0);
            self.ambient[n_base + i] = self.ou_state[i];
        }
    }
}

impl WorldMedium for ExogenousOnlyMedium {
    fn num_ambient_channels(&self) -> usize {
        self.ambient.len()
    }

    fn num_actuators(&self) -> usize {
        self.inner.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient,
            metabolic_influx: 0.0,
            tissue_stress: self.starve_stress,
            num_actuators: self.inner.num_actuators(),
        }
    }

    fn apply_pulse(&mut self, _pulse: &FieldPulse) {
        self.inner.apply_pulse(&self.silent);
        self.refresh();
    }
}

/// Per-tick achieved outcome accumulator.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    pub influx: f64,
    pub stress: f64,
}

impl Outcome {
    #[must_use]
    pub fn net(&self) -> f64 {
        self.influx - self.stress
    }
}

/// Runs `ticks` closed-loop steps; achieved outcome is read from `achieved(world)` AFTER each pulse.
pub fn run_closed_loop<O, W, F>(org: &mut O, world: &mut W, ticks: u64, achieved: F) -> Outcome
where
    O: OrganismEngine,
    W: WorldMedium,
    F: Fn(&W) -> (f32, f32),
{
    let mut out = Outcome::default();
    for _ in 0..ticks {
        let pulse = org
            .step(world.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(world.num_actuators()));
        world.apply_pulse(&pulse);
        let (i, s) = achieved(world);
        out.influx += f64::from(i);
        out.stress += f64::from(s);
    }
    out
}

fn own<W: WorldMedium>(w: &W) -> (f32, f32) {
    let v = w.current_view();
    (v.metabolic_influx, v.tissue_stress)
}

/// One named gate result after Holm correction.
#[derive(Clone, Debug)]
pub struct GateRow {
    pub name: &'static str,
    pub stats: PairedGate,
    pub p_holm: f64,
    pub pass: bool,
}

/// Hard biophysical & causal invariant check (`I1..I3`).
#[derive(Clone, Debug)]
pub struct InvariantRow {
    pub name: &'static str,
    pub detail: String,
    pub pass: bool,
}

/// Seed schedule: organism `1000 + s`, training world `5000 + s`, fresh evaluation world `9000 + s`.
#[derive(Clone, Copy, Debug)]
pub struct GateTicks {
    pub train: u64,
    pub eval: u64,
    pub reversal_pre: u64,
    pub reversal_post: u64,
    pub reacq_window: u64,
}

impl Default for GateTicks {
    fn default() -> Self {
        Self {
            train: 6_000,
            eval: 2_000,
            reversal_pre: 1_500,
            reversal_post: 1_500,
            reacq_window: 500,
        }
    }
}

/// Raw per-seed vectors for every arm of the locked gate suite.
#[derive(Clone, Debug, Default)]
pub struct BrightArms {
    pub natural: Vec<f64>,
    pub frozen: Vec<f64>,
    pub yoked: Vec<f64>,
    pub shuffled: Vec<f64>,
    pub xfer_trained: Vec<f64>,
    pub xfer_newborn: Vec<f64>,
    pub xfer_lesion: Vec<f64>,
    pub rev_post_natural: Vec<f64>,
    pub rev_post_frozen: Vec<f64>,
    pub reacq_first_post: Vec<f64>,
    pub reacq_first_life: Vec<f64>,
    pub rev_post_frozen_clone: Vec<f64>,
    pub reacq_gain_post: Vec<f64>,
    pub acq_gain_life: Vec<f64>,
    /// Repeated reversal `A -> B -> A` (500 ticks after returning to A): experienced organism.
    pub abab_return_natural: Vec<f64>,
    /// Repeated reversal `A -> B -> A`: plastic newborn placed in the exact cloned world state at `B -> A`.
    pub abab_return_newborn: Vec<f64>,
    /// Permuted + Smooth OU(0.97) + Delayed Motor Self-Copy torture world: Natural vs Frozen.
    pub torture_natural: Vec<f64>,
    pub torture_frozen: Vec<f64>,
    /// Counterfactual agency snapshots (`Natural`, `ActionShuffled`, `ExogenousOnly`).
    pub agency_natural: Vec<f64>,
    pub agency_shuffled: Vec<f64>,
    pub agency_exogenous: Vec<f64>,
    /// Final ATP in `BrightRadiantWorld` (`Natural` vs `Frozen`) and famine mitoses when `ATP <= 0.01`.
    pub atp_natural: Vec<f64>,
    pub atp_frozen: Vec<f64>,
    pub famine_mitoses: Vec<u64>,
    /// Share of sensor pore conductance on distractor channels (`NoisyTV + OU + MotorSelfCopy`) in torture world.
    pub torture_distractor_share: Vec<f64>,
}

/// Runs every arm of the locked gate family for one seed. `make(org_seed)` builds a newborn.
pub fn run_bright_seed<O, M>(make: &M, s: u64, ticks: GateTicks, arms: &mut BrightArms)
where
    O: OrganismEngine + Clone,
    M: Fn(u64) -> O,
{
    let org_seed = 1_000 + s;
    let w_seed = 5_000 + s;
    let e_seed = 9_000 + s;

    // 1. Natural, recording the donor reward stream for the Yoked twin.
    let mut nat = make(org_seed);
    let mut w_nat = BrightRadiantWorld::new(w_seed);
    let mut donor = Vec::with_capacity(ticks.train as usize);
    let mut out_nat = Outcome::default();
    for _ in 0..ticks.train {
        let v = w_nat.current_view();
        donor.push((v.metabolic_influx, v.tissue_stress));
        let pulse = nat
            .step(v)
            .unwrap_or_else(|_| FieldPulse::silent(w_nat.num_actuators()));
        w_nat.apply_pulse(&pulse);
        let (i, st) = own(&w_nat);
        out_nat.influx += f64::from(i);
        out_nat.stress += f64::from(st);
    }
    let nat_snap = nat.snapshot();
    arms.natural.push(out_nat.net());
    arms.agency_natural
        .push(f64::from(nat_snap.mean_interventional_agency));
    arms.atp_natural.push(f64::from(nat_snap.atp));

    // 2. Frozen plasticity twin.
    let mut frz = make(org_seed);
    frz.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_frz = BrightRadiantWorld::new(w_seed);
    arms.frozen
        .push(run_closed_loop(&mut frz, &mut w_frz, ticks.train, own).net());
    arms.atp_frozen.push(f64::from(frz.snapshot().atp));

    // 3. Yoked reward twin (independent seed so trajectory decouples from donor).
    let mut yok = make(org_seed + 500_000);
    let mut w_yok = YokedMedium::new(BrightRadiantWorld::new(w_seed), donor);
    arms.yoked
        .push(run_closed_loop(&mut yok, &mut w_yok, ticks.train, |w| own(w.inner())).net());

    // 4. Action-shuffled twin.
    let mut shf = make(org_seed);
    let mut w_shf = ActionShuffledMedium::new(BrightRadiantWorld::new(w_seed), 88_000 + s);
    arms.shuffled
        .push(run_closed_loop(&mut shf, &mut w_shf, ticks.train, own).net());
    arms.agency_shuffled
        .push(f64::from(shf.snapshot().mean_interventional_agency));

    // 5. Exogenous-only agency & starvation mitosis check.
    let mut exo = make(org_seed);
    let mut w_exo = ExogenousOnlyMedium::new(w_seed, 0.30);
    run_closed_loop(&mut exo, &mut w_exo, 400, own);
    let snap_drained = exo.snapshot();
    let births_before = snap_drained.mitosis_births;
    run_closed_loop(&mut exo, &mut w_exo, 600, own);
    let snap_exo = exo.snapshot();
    let famine_births = if snap_drained.atp <= 0.01 && snap_exo.atp <= 0.01 {
        snap_exo.mitosis_births.saturating_sub(births_before)
    } else {
        0
    };
    arms.agency_exogenous
        .push(f64::from(snap_exo.mean_interventional_agency));
    arms.famine_mitoses.push(famine_births);

    // 6. Transfer & flagless BirthReset lesion in fresh world seed.
    let mut lesion = nat.clone();
    lesion.zero_learned_conductances();
    lesion.set_control_mode(ControlMode::FrozenPlasticity);
    nat.set_control_mode(ControlMode::FrozenPlasticity);
    let h0 = nat.learned_parameter_hash();
    let mut w_e1 = BrightRadiantWorld::new(e_seed);
    arms.xfer_trained
        .push(run_closed_loop(&mut nat, &mut w_e1, ticks.eval, own).net());
    assert_eq!(h0, nat.learned_parameter_hash(), "frozen hash drifted");
    let mut nb = make(org_seed);
    nb.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_e2 = BrightRadiantWorld::new(e_seed);
    arms.xfer_newborn
        .push(run_closed_loop(&mut nb, &mut w_e2, ticks.eval, own).net());
    let mut w_e3 = BrightRadiantWorld::new(e_seed);
    arms.xfer_lesion
        .push(run_closed_loop(&mut lesion, &mut w_e3, ticks.eval, own).net());

    // 7. Mid-life causal reversal (A -> B) AND repeated reversal savings (A -> B -> A).
    let mut rn = make(org_seed);
    let mut w_rn = BrightRadiantWorld::new(w_seed);
    let first_life = run_closed_loop(&mut rn, &mut w_rn, ticks.reacq_window, own);
    run_closed_loop(
        &mut rn,
        &mut w_rn,
        ticks.reversal_pre - ticks.reacq_window,
        own,
    );
    // Flip A -> B at t = 1500
    w_rn.0.trigger_causal_reversal();
    let mut clone_org = rn.clone();
    clone_org.set_control_mode(ControlMode::FrozenPlasticity);
    let mut clone_world = w_rn.clone();
    let first_post = run_closed_loop(&mut rn, &mut w_rn, ticks.reacq_window, own);
    let rest_post = run_closed_loop(
        &mut rn,
        &mut w_rn,
        ticks.reversal_post - ticks.reacq_window,
        own,
    );
    let clone_first = run_closed_loop(&mut clone_org, &mut clone_world, ticks.reacq_window, own);
    let clone_rest = run_closed_loop(
        &mut clone_org,
        &mut clone_world,
        ticks.reversal_post - ticks.reacq_window,
        own,
    );
    arms.rev_post_natural.push(first_post.net() + rest_post.net());
    arms.rev_post_frozen_clone
        .push(clone_first.net() + clone_rest.net());
    arms.reacq_first_post.push(first_post.net());
    arms.reacq_first_life.push(first_life.net());
    arms.reacq_gain_post
        .push(first_post.net() - clone_first.net());

    let mut nb_life = make(org_seed);
    nb_life.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_nb_life = BrightRadiantWorld::new(w_seed);
    let nb_first = run_closed_loop(&mut nb_life, &mut w_nb_life, ticks.reacq_window, own);
    arms.acq_gain_life.push(first_life.net() - nb_first.net());

    // Flip B -> A at t = 3000: compare experienced `rn` vs a fresh plastic newborn in the EXACT
    // cloned world state at t = 3000 (zero position/vent-halo confound; pure savings test).
    w_rn.0.trigger_causal_reversal();
    let mut w_ba_nb = w_rn.clone();
    let mut nb_ba = make(org_seed + 700_000);
    let ret_a = run_closed_loop(&mut rn, &mut w_rn, ticks.reacq_window, own);
    let nb_ret_a = run_closed_loop(&mut nb_ba, &mut w_ba_nb, ticks.reacq_window, own);
    arms.abab_return_natural.push(ret_a.net());
    arms.abab_return_newborn.push(nb_ret_a.net());

    let mut rf = make(org_seed);
    rf.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_rf = BrightRadiantWorld::new(w_seed);
    run_closed_loop(&mut rf, &mut w_rf, ticks.reversal_pre, own);
    w_rf.0.trigger_causal_reversal();
    arms.rev_post_frozen
        .push(run_closed_loop(&mut rf, &mut w_rf, ticks.reversal_post, own).net());

    // 8. Permuted + Smooth OU(0.97) + Delayed Motor Self-Copy Torture Gate (G9).
    let perm_seed = 66_000 + s;
    let mut t_nat = make(org_seed);
    let mut w_tnat = PermutedMedium::new(
        TortureMedium::new(BrightRadiantWorld::new(w_seed), perm_seed),
        perm_seed ^ 0x5C8A_B1ED,
    );
    arms.torture_natural
        .push(run_closed_loop(&mut t_nat, &mut w_tnat, ticks.train, own).net());

    let t_snap = t_nat.snapshot();
    let mut total_g = 0.0_f64;
    let mut distractor_g = 0.0_f64;
    for (&sc_ch, &g) in t_snap
        .sensor_pore_channels
        .iter()
        .zip(t_snap.sensor_pore_conductances.iter())
    {
        let true_ch = w_tnat.invert_sensor_channel(sc_ch as usize);
        let gf = f64::from(g.max(0.0));
        total_g += gf;
        // True photonic channels in BrightRadiantWorld are 0..64; echo 64..96 (silent unless pinged);
        // 96..192 are NoisyTV (96..128) + Smooth OU (128..160) + Delayed Motor Self-Copy (160..192).
        if true_ch >= 96 {
            distractor_g += gf;
        }
    }
    arms.torture_distractor_share.push(if total_g > 1e-9 {
        distractor_g / total_g
    } else {
        1.0
    });

    let mut t_frz = make(org_seed);
    t_frz.set_control_mode(ControlMode::FrozenPlasticity);
    let mut w_tfrz = PermutedMedium::new(
        TortureMedium::new(BrightRadiantWorld::new(w_seed), perm_seed),
        perm_seed ^ 0x5C8A_B1ED,
    );
    arms.torture_frozen
        .push(run_closed_loop(&mut t_frz, &mut w_tfrz, ticks.train, own).net());
}

/// Benchmarks `4,096`-channel step latency (`p50`, `p95`, `p99` in microseconds) and active pore count.
pub fn benchmark_dense_4096<O, M>(make: &M, seed: u64) -> (f64, f64, f64, usize)
where
    O: OrganismEngine,
    M: Fn(u64) -> O,
{
    let mut org = make(seed);
    let mut world = DenseFieldWorld::new(77_000 + seed);
    for _ in 0..400 {
        let p = org
            .step(world.current_view())
            .unwrap_or_else(|_| FieldPulse::silent(world.num_actuators()));
        world.apply_pulse(&p);
    }
    let mut samples_us = Vec::with_capacity(1_000);
    for _ in 0..1_000 {
        let view = world.current_view();
        let t0 = Instant::now();
        let p = org
            .step(view)
            .unwrap_or_else(|_| FieldPulse::silent(world.num_actuators()));
        let dt_us = t0.elapsed().as_nanos() as f64 / 1_000.0;
        samples_us.push(dt_us);
        world.apply_pulse(&p);
    }
    samples_us.sort_by(|a, b| a.total_cmp(b));
    let p50 = samples_us[500];
    let p95 = samples_us[950];
    let p99 = samples_us[990];
    let pores = org.snapshot().sensor_pore_count;
    (p50, p95, p99, pores)
}

/// Original (v1) reversal definitions, reported for disclosure only (not part of the Holm family).
#[must_use]
pub fn legacy_reversal_rows(arms: &BrightArms) -> [(&'static str, PairedGate); 2] {
    [
        (
            "v1-G6 post Natural > post Frozen-newborn",
            paired_gate(&arms.rev_post_natural, &arms.rev_post_frozen),
        ),
        (
            "v1-G7 first post window > first life window",
            paired_gate(&arms.reacq_first_post, &arms.reacq_first_life),
        ),
    ]
}

/// Evaluates all 9 paired closed-loop gates (`G1..G9`) with Holm-Bonferroni correction at `alpha`.
#[must_use]
pub fn evaluate_bright_gates(arms: &BrightArms, alpha: f64) -> Vec<GateRow> {
    let raw: [(&'static str, PairedGate); 9] = [
        ("G1 Natural > Frozen", paired_gate(&arms.natural, &arms.frozen)),
        ("G2 Natural > Yoked", paired_gate(&arms.natural, &arms.yoked)),
        (
            "G3 Natural > ActionShuffled",
            paired_gate(&arms.natural, &arms.shuffled),
        ),
        (
            "G4 Transfer: Trained-Frozen > Newborn-Frozen",
            paired_gate(&arms.xfer_trained, &arms.xfer_newborn),
        ),
        (
            "G5 Transfer: Trained-Frozen > BirthReset-Lesion",
            paired_gate(&arms.xfer_trained, &arms.xfer_lesion),
        ),
        (
            "G6 Reversal (A->B): post Natural > Frozen Clone",
            paired_gate(&arms.rev_post_natural, &arms.rev_post_frozen_clone),
        ),
        (
            "G7 Reversal Speed (A->B): post gain > life gain",
            paired_gate(&arms.reacq_gain_post, &arms.acq_gain_life),
        ),
        (
            "G8 Savings (A->B->A): return-A > Plastic Newborn",
            paired_gate(&arms.abab_return_natural, &arms.abab_return_newborn),
        ),
        (
            "G9 Torture (Perm+OU97+SelfCopy): Natural > Frozen",
            paired_gate(&arms.torture_natural, &arms.torture_frozen),
        ),
    ];
    let ps: Vec<f64> = raw.iter().map(|(_, g)| g.p_two_sided).collect();
    let adj = holm_adjust(&ps);
    raw.iter()
        .zip(adj)
        .map(|(&(name, stats), p_holm)| GateRow {
            name,
            stats,
            p_holm,
            pass: stats.mean_diff > 0.0 && p_holm < alpha,
        })
        .collect()
}

/// Evaluates hard biophysical & causal invariants (`I1..I3`).
#[must_use]
pub fn evaluate_invariants(
    arms: &BrightArms,
    dense_latency: (f64, f64, f64, usize),
) -> Vec<InvariantRow> {
    let ag_shuf = paired_gate(&arms.agency_natural, &arms.agency_shuffled);
    let ag_exo = paired_gate(&arms.agency_natural, &arms.agency_exogenous);
    let mean_nat_ag = ag_shuf.mean_a;
    let mean_shuf_ag = ag_shuf.mean_b;
    let mean_exo_ag = ag_exo.mean_b;
    let i1_pass = ag_shuf.mean_diff > 0.0
        && ag_shuf.p_two_sided < 0.01
        && ag_exo.mean_diff > 0.0
        && ag_exo.p_two_sided < 0.01
        && mean_shuf_ag <= 0.05
        && mean_exo_ag <= 0.05;

    let (p50_us, p95_us, p99_us, pores_4096) = dense_latency;
    let mean_dist_share = if arms.torture_distractor_share.is_empty() {
        1.0
    } else {
        arms.torture_distractor_share.iter().sum::<f64>()
            / (arms.torture_distractor_share.len() as f64)
    };
    let i2_pass = p99_us < 2_000.0 && pores_4096 <= 64 && mean_dist_share < 0.35;

    let total_famine_births: u64 = arms.famine_mitoses.iter().sum();
    let atp_cmp = paired_gate(&arms.atp_natural, &arms.atp_frozen);
    let i3_pass = total_famine_births == 0 && atp_cmp.mean_diff > 0.0 && atp_cmp.p_two_sided < 0.05;

    vec![
        InvariantRow {
            name: "I1 Counterfactual Agency Purity",
            detail: format!(
                "nat={mean_nat_ag:.4} shuf={mean_shuf_ag:.4} (<=0.05) exo={mean_exo_ag:.4} (<=0.05) p_shuf={:.5} p_exo={:.5}",
                ag_shuf.p_two_sided, ag_exo.p_two_sided
            ),
            pass: i1_pass,
        },
        InvariantRow {
            name: "I2 Bounded O(1) 4096-Ch & Pore Sieve",
            detail: format!(
                "4096ch p50={p50_us:.1}us p95={p95_us:.1}us p99={p99_us:.1}us (<2000us) pores={pores_4096} (<=64) torture_distractor_share={mean_dist_share:.3} (<0.35)"
            ),
            pass: i2_pass,
        },
        InvariantRow {
            name: "I3 Honest ATP Homeostasis & Zero Famine Mitosis",
            detail: format!(
                "famine_mitoses={total_famine_births} (==0) atp_nat={:.2} atp_frz={:.2} p={:.5}",
                atp_cmp.mean_a, atp_cmp.mean_b, atp_cmp.p_two_sided
            ),
            pass: i3_pass,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_p_matches_reference_values() {
        assert!((student_t_two_sided_p(2.262, 9.0) - 0.05).abs() < 1e-3);
        assert!((student_t_two_sided_p(4.809, 9.0) - 0.000_961_5).abs() < 2e-5);
        assert!((student_t_two_sided_p(0.0, 9.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn holm_is_monotone_and_bounded() {
        let adj = holm_adjust(&[0.01, 0.04, 0.03, 0.2]);
        assert!((adj[0] - 0.04).abs() < 1e-12);
        assert!((adj[2] - 0.09).abs() < 1e-12);
        assert!((adj[1] - 0.09).abs() < 1e-12);
        assert!((adj[3] - 0.2).abs() < 1e-12);
    }
}
