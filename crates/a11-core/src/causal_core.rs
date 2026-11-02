//! `CausalCore`: minimal biologically grounded closed-loop learning core (`Alpha 11`).
//!
//! Built strictly one mechanism at a time against the locked 12-gate `DEV-30` judge:
//! - Base (`E0`): Spatial `ON/OFF` center-surround with divisive normalization + `t-1` `TD(lambda)`
//!   three-factor actor-critic (`Fremaux & Gerstner 2016`).
//! - `E1` (`Body` + `Forward Agency`): Honest somatic `ATP` homeostasis + precision-weighted
//!   out-of-sample zero-mean action-centered sensory predictor (`Schneider et al. 2014`,
//!   `Wilmes & Senn 2025`).
//! - `E2` (`Eyes/Senses`): Bounded `O(1)` (`K <= 64`) sensory pore frontier with 4-tap corollary
//!   reafference cancellation (`Singla & Sawtell 2017`) and $C^1$ kinematic optical-flow sieve.
//! - `E3` (`Brain`): Dormant-retaining latent context bank (`K_ctx = 4`, `OFC` hidden-state
//!   inference, `Blanco-Pozo et al. 2024`, `Lopatina & Schoenbaum 2018`, `Alpha 2c/3 Rule 4`)
//!   gated by joint negative homeostatic/TD surprise and forward sensorimotor inversion.

use a11_medium::{
    ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError,
};

use crate::crc32c::crc32c;
use crate::rng::{ParamHasher, Rng64, sanitize};

const W_CLAMP: f32 = 8.0;
const TD_CLAMP: f32 = 2.0;

/// Free constants of the learning core.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CausalParams {
    pub eta_actor: f32,
    pub eta_critic: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub noise_sigma: f32,
    pub noise_rho: f32,
    pub norm_c: f32,
    pub birth_w_scale: f32,
}

impl Default for CausalParams {
    fn default() -> Self {
        Self {
            eta_actor: 0.05,
            eta_critic: 0.05,
            gamma: 0.97,
            lambda: 0.9,
            noise_sigma: 0.6,
            noise_rho: 0.85,
            norm_c: 0.05,
            birth_w_scale: 0.10,
        }
    }
}

/// Hard upper bounds (bounded RAM and bounded `O(1)` per-tick work).
pub const CAUSAL_MAX_CHANNELS: usize = 4096;
pub const CAUSAL_MAX_ACTUATORS: usize = 64;
pub const MAX_ACTIVE_PORES: usize = 64;
pub const MAX_CONTEXTS: usize = 4;
const SCOUT_SLOTS: usize = 32;
const MOTOR_TAPS: usize = 4;
const FWD_MAX_FEATURES: usize = 256;
const NEWBORN_ATP: f32 = 1.5;
const MAX_ATP: f32 = 20.0;

const CKPT_MAGIC: u64 = 0x4131_3143_4155_5331;
const CKPT_VERSION: u32 = 9;

/// Ablation switches used ONLY by external gate harnesses (never by the organism itself).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CausalAblation {
    /// Replace the spatial surround + ON/OFF + divisive normalization with raw `x` features.
    pub no_center_surround: bool,
    /// Replace the TD critic with a running-mean reward baseline (`delta = r - r_bar`).
    pub no_td_critic: bool,
    /// Disable the out-of-sample zero-mean action-centered forward sensory model (`J_act`).
    pub no_forward_agency: bool,
    /// Disable the bounded `O(1)` corollary-cancelled sensory pore frontier (`K <= 64`).
    pub no_pore_frontier: bool,
    /// Disable dormant-retaining `OFC` latent context inference (`K_ctx = 1`).
    pub no_latent_context: bool,
}

#[derive(Clone, Debug)]
pub struct CausalCore {
    seed: u64,
    mode: ControlMode,
    poisoned: bool,
    ablation: CausalAblation,
    params: CausalParams,
    tick: u64,

    n_in: usize,
    k_pores: usize,
    n_scout: usize,
    n_mon: usize,
    n_act: usize,
    n_feat: usize,
    n_fwd: usize,
    scout_cursor: usize,

    num_ctx: usize,
    active_ctx: usize,
    ctx_cooldown: u32,
    ctx_out_events: u32,
    context_switches: u64,
    ctx_ticks: [u32; MAX_CONTEXTS],
    ctx_has_cons: [u8; MAX_CONTEXTS],
    ctx_r_peak: [f32; MAX_CONTEXTS],
    ctx_out_err: [f32; MAX_CONTEXTS],
    r_bar_ctx: [f32; MAX_CONTEXTS],
    r_fast_ema: f32,
    r_med200: f32,
    valence_flip_ema: f32,

    pore_ch_birth: Vec<u32>,
    mon_ch: Vec<u32>,
    mon_age: Vec<u32>,
    mon_mean_x: Vec<f32>,
    mon_var_raw: Vec<f32>,
    mon_var_exaff: Vec<f32>,
    mon_prev_xt: Vec<f32>,
    mon_prev_dx: Vec<f32>,
    mon_var_dx: Vec<f32>,
    mon_cov_dx: Vec<f32>,
    motor_canc_g: Vec<f32>,
    da_ring: Vec<f32>,
    pore_signal: Vec<f32>,

    w_birth: Vec<f32>,
    w_ctx: Vec<f32>,
    v_ctx: Vec<f32>,
    q_ctx: Vec<f32>,
    w_cons: Vec<f32>,
    v_cons: Vec<f32>,
    q_cons: Vec<f32>,
    elig_actor: Vec<f32>,
    elig_critic: Vec<f32>,

    feat: Vec<f32>,
    prev_critic_feat: Vec<f32>,
    prev_feat: Vec<f32>,
    prev_df: Vec<f32>,
    prev_act: Vec<f32>,
    act_mean: Vec<f32>,
    fwd_b: Vec<f32>,
    fwd_alpha: Vec<f32>,
    fwd_j: Vec<f32>,
    fwd_var_null: Vec<f32>,
    fwd_var_act: Vec<f32>,
    atp: f32,

    v_prev: f32,
    has_prev: bool,
    noise: Vec<f32>,
    td_abs_ema: f32,

    rng: Rng64,
}

impl CausalCore {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_ablation(seed, CausalAblation::default())
    }

    #[must_use]
    pub fn with_ablation(seed: u64, ablation: CausalAblation) -> Self {
        Self::with_config(seed, ablation, CausalParams::default())
    }

    #[must_use]
    pub fn with_config(seed: u64, ablation: CausalAblation, params: CausalParams) -> Self {
        Self {
            seed,
            mode: ControlMode::Natural,
            poisoned: false,
            ablation,
            params,
            tick: 0,
            n_in: 0,
            k_pores: 0,
            n_scout: 0,
            n_mon: 0,
            n_act: 0,
            n_feat: 0,
            n_fwd: 0,
            scout_cursor: 0,
            num_ctx: 1,
            active_ctx: 0,
            ctx_cooldown: 0,
            ctx_out_events: 0,
            context_switches: 0,
            ctx_ticks: [0; MAX_CONTEXTS],
            ctx_has_cons: [0; MAX_CONTEXTS],
            ctx_r_peak: [0.0; MAX_CONTEXTS],
            ctx_out_err: [0.0; MAX_CONTEXTS],
            r_bar_ctx: [0.0; MAX_CONTEXTS],
            r_fast_ema: 0.0,
            r_med200: 0.0,
            valence_flip_ema: 0.0,
            pore_ch_birth: Vec::new(),
            mon_ch: Vec::new(),
            mon_age: Vec::new(),
            mon_mean_x: Vec::new(),
            mon_var_raw: Vec::new(),
            mon_var_exaff: Vec::new(),
            mon_prev_xt: Vec::new(),
            mon_prev_dx: Vec::new(),
            mon_var_dx: Vec::new(),
            mon_cov_dx: Vec::new(),
            motor_canc_g: Vec::new(),
            da_ring: Vec::new(),
            pore_signal: Vec::new(),
            w_birth: Vec::new(),
            w_ctx: Vec::new(),
            v_ctx: Vec::new(),
            q_ctx: Vec::new(),
            w_cons: Vec::new(),
            v_cons: Vec::new(),
            q_cons: Vec::new(),
            elig_actor: Vec::new(),
            elig_critic: Vec::new(),
            feat: Vec::new(),
            prev_critic_feat: Vec::new(),
            prev_feat: Vec::new(),
            prev_df: Vec::new(),
            prev_act: Vec::new(),
            act_mean: Vec::new(),
            fwd_b: Vec::new(),
            fwd_alpha: Vec::new(),
            fwd_j: Vec::new(),
            fwd_var_null: Vec::new(),
            fwd_var_act: Vec::new(),
            atp: NEWBORN_ATP,
            v_prev: 0.0,
            has_prev: false,
            noise: Vec::new(),
            td_abs_ema: 0.0,
            rng: Rng64::from_seed(seed ^ 0xCA05_A1C0_0E00_0001),
        }
    }

    #[must_use]
    pub fn ablation(&self) -> CausalAblation {
        self.ablation
    }

    fn bind(&mut self, n_in: usize, n_act: usize) {
        self.n_in = n_in.min(CAUSAL_MAX_CHANNELS);
        self.n_act = n_act.min(CAUSAL_MAX_ACTUATORS);
        self.k_pores = if self.ablation.no_pore_frontier {
            self.n_in
        } else {
            self.n_in.min(MAX_ACTIVE_PORES)
        };
        self.n_scout = if self.ablation.no_pore_frontier {
            0
        } else {
            self.n_in.saturating_sub(self.k_pores).min(SCOUT_SLOTS)
        };
        self.n_mon = self.k_pores + self.n_scout;
        self.n_feat = 2 * self.k_pores + 1;
        self.n_fwd = (2 * self.k_pores).min(FWD_MAX_FEATURES);

        let mut birth_rng = Rng64::from_seed(self.seed ^ 0xB127_0000_5EED_0002);
        let scale = self.params.birth_w_scale / ((self.n_feat as f32).sqrt().max(1.0));
        let w_len = self.n_act * self.n_feat;
        self.w_birth = (0..w_len)
            .map(|_| scale * birth_rng.next_f32_signed())
            .collect();
        self.w_ctx = Vec::with_capacity(MAX_CONTEXTS * w_len);
        for _ in 0..MAX_CONTEXTS {
            self.w_ctx.extend_from_slice(&self.w_birth);
        }
        self.v_ctx = vec![0.0; MAX_CONTEXTS * self.n_feat];
        self.q_ctx = vec![0.0; MAX_CONTEXTS * self.n_feat];
        self.w_cons = self.w_ctx.clone();
        self.v_cons = self.v_ctx.clone();
        self.q_cons = self.q_ctx.clone();

        let mut perm: Vec<u32> = (0..self.n_in as u32).collect();
        if !self.ablation.no_pore_frontier && self.n_in > 1 {
            for i in 0..self.n_mon.min(self.n_in) {
                let rem = self.n_in - i;
                let j = i + ((birth_rng.next_u64() as usize) % rem);
                perm.swap(i, j);
            }
        }
        self.pore_ch_birth = perm[..self.k_pores].to_vec();
        self.mon_ch = perm[..self.n_mon].to_vec();
        self.scout_cursor = self.n_mon % self.n_in.max(1);

        self.mon_age = vec![0; self.n_mon];
        self.mon_mean_x = vec![0.0; self.n_mon];
        self.mon_var_raw = vec![1e-4; self.n_mon];
        self.mon_var_exaff = vec![1e-4; self.n_mon];
        self.mon_prev_xt = vec![0.0; self.n_mon];
        self.mon_prev_dx = vec![0.0; self.n_mon];
        self.mon_var_dx = vec![1e-4; self.n_mon];
        self.mon_cov_dx = vec![0.0; self.n_mon];
        self.motor_canc_g = vec![0.0; self.n_mon * MOTOR_TAPS * self.n_act];
        self.da_ring = vec![0.0; MOTOR_TAPS * self.n_act];
        self.pore_signal = vec![0.0; self.k_pores];

        self.elig_actor = vec![0.0; w_len];
        self.elig_critic = vec![0.0; self.n_feat];
        self.feat = vec![0.0; self.n_feat];
        self.prev_critic_feat = vec![0.0; self.n_feat];
        self.prev_feat = vec![0.0; self.n_fwd];
        self.prev_df = vec![0.0; self.n_fwd];
        self.prev_act = vec![0.0; self.n_act];
        self.act_mean = vec![0.0; self.n_act];
        self.fwd_b = vec![0.0; self.n_fwd];
        self.fwd_alpha = vec![0.0; self.n_fwd];
        self.fwd_j = vec![0.0; self.n_fwd * self.n_act];
        self.fwd_var_null = vec![1e-4; self.n_fwd];
        self.fwd_var_act = vec![1e-4; self.n_fwd];
        self.noise = vec![0.0; self.n_act];
    }

    fn gaussian(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.rng.next_f32_unit()).sum();
        (s - 2.0) * 1.732_050_8
    }

    fn slot_fitness(&self, idx: usize) -> f32 {
        if self.mon_age[idx] < 48 {
            return 0.0;
        }
        let vr = self.mon_var_raw[idx];
        let ve = self.mon_var_exaff[idx];
        let exaff_ratio = ve / (vr + 1e-6);
        let exaff_gate = ((exaff_ratio - 0.35) / 0.45).clamp(0.0, 1.0);
        let amp_gate = ve / (ve + 0.0002);
        let corr_dx = (self.mon_cov_dx[idx] / (self.mon_var_dx[idx] + 1e-7)).clamp(-1.0, 1.0);
        let c1_gate = (corr_dx - 0.22).clamp(0.0, 1.0);
        let base = exaff_gate * amp_gate * c1_gate;
        if idx < self.k_pores && base > 0.02 {
            let mut max_w_mass = 0.0_f32;
            for c in 0..self.num_ctx {
                let w_off = c * self.n_act * self.n_feat;
                let mut wm_live = 0.0_f32;
                let mut wm_cons = 0.0_f32;
                for m in 0..self.n_act {
                    wm_live += self.w_ctx[w_off + m * self.n_feat + idx].abs()
                        + self.w_ctx[w_off + m * self.n_feat + self.k_pores + idx].abs();
                    if self.ctx_has_cons[c] != 0 {
                        wm_cons += self.w_cons[w_off + m * self.n_feat + idx].abs()
                            + self.w_cons[w_off + m * self.n_feat + self.k_pores + idx].abs();
                    }
                }
                max_w_mass = max_w_mass.max(wm_live).max(wm_cons);
            }
            base + 0.04 * max_w_mass.min(1.0)
        } else {
            base
        }
    }

    fn reset_monitor_slot(&mut self, idx: usize, new_ch: u32) {
        let tap_len = MOTOR_TAPS * self.n_act;
        self.mon_ch[idx] = new_ch;
        self.mon_age[idx] = 0;
        self.mon_mean_x[idx] = 0.0;
        self.mon_var_raw[idx] = 1e-4;
        self.mon_var_exaff[idx] = 1e-4;
        self.mon_prev_xt[idx] = 0.0;
        self.mon_prev_dx[idx] = 0.0;
        self.mon_var_dx[idx] = 1e-4;
        self.mon_cov_dx[idx] = 0.0;
        self.motor_canc_g[idx * tap_len..(idx + 1) * tap_len].fill(0.0);
    }

    fn copy_monitor_slot(&mut self, src: usize, dst: usize) {
        let tap_len = MOTOR_TAPS * self.n_act;
        self.mon_ch[dst] = self.mon_ch[src];
        self.mon_age[dst] = self.mon_age[src];
        self.mon_mean_x[dst] = self.mon_mean_x[src];
        self.mon_var_raw[dst] = self.mon_var_raw[src];
        self.mon_var_exaff[dst] = self.mon_var_exaff[src];
        self.mon_prev_xt[dst] = self.mon_prev_xt[src];
        self.mon_prev_dx[dst] = self.mon_prev_dx[src];
        self.mon_var_dx[dst] = self.mon_var_dx[src];
        self.mon_cov_dx[dst] = self.mon_cov_dx[src];
        for k in 0..tap_len {
            self.motor_canc_g[dst * tap_len + k] = self.motor_canc_g[src * tap_len + k];
        }
    }

    fn is_active_pore(&self, ch: u32) -> bool {
        self.mon_ch[..self.k_pores].contains(&ch)
    }

    fn morph_pore_frontier(&mut self) {
        if self.n_scout == 0 || self.ablation.no_pore_frontier {
            return;
        }
        let w_len = self.n_act * self.n_feat;
        for _ in 0..8 {
            let mut best_s = None;
            let mut best_fit = 0.0_f32;
            for s in self.k_pores..self.n_mon {
                let sf = self.slot_fitness(s);
                if sf > best_fit && !self.is_active_pore(self.mon_ch[s]) {
                    best_fit = sf;
                    best_s = Some(s);
                }
            }
            let Some(s_idx) = best_s else {
                break;
            };
            let mut worst_p = 0_usize;
            let mut worst_fit = f32::MAX;
            for p in 0..self.k_pores {
                if self.mon_age[p] < 48 {
                    continue;
                }
                let pf = self.slot_fitness(p);
                if pf < worst_fit {
                    worst_fit = pf;
                    worst_p = p;
                }
            }
            if worst_fit < f32::MAX && best_fit > worst_fit + 0.04 {
                self.copy_monitor_slot(s_idx, worst_p);
                self.mon_age[s_idx] = 0;
                let k = self.k_pores;
                let nf = self.n_feat;
                for c in 0..MAX_CONTEXTS {
                    let w_off = c * w_len;
                    let v_off = c * nf;
                    for m in 0..self.n_act {
                        let wb0 = self.w_birth[m * nf + worst_p];
                        let wb1 = self.w_birth[m * nf + k + worst_p];
                        self.w_ctx[w_off + m * nf + worst_p] = wb0;
                        self.w_ctx[w_off + m * nf + k + worst_p] = wb1;
                        self.w_cons[w_off + m * nf + worst_p] = wb0;
                        self.w_cons[w_off + m * nf + k + worst_p] = wb1;
                    }
                    self.v_ctx[v_off + worst_p] = 0.0;
                    self.v_ctx[v_off + k + worst_p] = 0.0;
                    self.v_cons[v_off + worst_p] = 0.0;
                    self.v_cons[v_off + k + worst_p] = 0.0;
                    self.q_ctx[v_off + worst_p] = 0.0;
                    self.q_ctx[v_off + k + worst_p] = 0.0;
                    self.q_cons[v_off + worst_p] = 0.0;
                    self.q_cons[v_off + k + worst_p] = 0.0;
                }
                for m in 0..self.n_act {
                    self.elig_actor[m * nf + worst_p] = 0.0;
                    self.elig_actor[m * nf + k + worst_p] = 0.0;
                }
                self.elig_critic[worst_p] = 0.0;
                self.elig_critic[k + worst_p] = 0.0;
                for fi in [worst_p, k + worst_p] {
                    if fi < self.n_fwd {
                        self.prev_feat[fi] = 0.0;
                        self.prev_df[fi] = 0.0;
                        self.fwd_b[fi] = 0.0;
                        self.fwd_alpha[fi] = 0.0;
                        self.fwd_var_null[fi] = 1e-4;
                        self.fwd_var_act[fi] = 1e-4;
                        let j_off = fi * self.n_act;
                        self.fwd_j[j_off..j_off + self.n_act].fill(0.0);
                    }
                }
            } else {
                break;
            }
        }

        let mut cursor = self.scout_cursor;
        for s in self.k_pores..self.n_mon {
            let mut chosen = self.mon_ch[s];
            for _ in 0..self.n_in {
                let cand = cursor as u32;
                cursor = (cursor + 1) % self.n_in;
                if !self.is_active_pore(cand) && !self.mon_ch[self.k_pores..s].contains(&cand) {
                    chosen = cand;
                    break;
                }
            }
            self.reset_monitor_slot(s, chosen);
        }
        self.scout_cursor = cursor;
    }

    fn update_pores_and_features(&mut self, x: &[f32]) {
        let k = self.k_pores;
        let n_act = self.n_act;
        let tap_len = MOTOR_TAPS * n_act;
        let is_plastic = self.plastic();
        let no_frontier = self.ablation.no_pore_frontier;

        if no_frontier {
            for p in 0..k {
                let ch = self.mon_ch[p] as usize;
                self.pore_signal[p] = sanitize(x.get(ch).copied().unwrap_or(0.0)).clamp(-1.0, 1.0);
            }
        } else {
            let mut ring_norm_sq = 0.0_f32;
            for &d in &self.da_ring {
                ring_norm_sq += d * d;
            }
            let eta_g = 0.08 / (ring_norm_sq + 0.15);
            for idx in 0..self.n_mon {
                let ch = self.mon_ch[idx] as usize;
                let raw_x = sanitize(x.get(ch).copied().unwrap_or(0.0)).clamp(-1.0, 1.0);
                let xc = raw_x - self.mon_mean_x[idx];
                let g_slice = &mut self.motor_canc_g[idx * tap_len..(idx + 1) * tap_len];
                let mut reaff = 0.0_f32;
                for t_idx in 0..tap_len {
                    reaff += g_slice[t_idx] * self.da_ring[t_idx];
                }
                let x_tilde = xc - reaff;
                let age = self.mon_age[idx];
                let dx = if age > 0 {
                    x_tilde - self.mon_prev_xt[idx]
                } else {
                    0.0
                };
                if is_plastic {
                    let beta_m = if age < 48 { 0.06 } else { 0.015 };
                    self.mon_mean_x[idx] =
                        sanitize(self.mon_mean_x[idx] + beta_m * xc).clamp(-1.0, 1.0);
                    self.mon_var_raw[idx] =
                        sanitize((1.0 - beta_m) * self.mon_var_raw[idx] + beta_m * xc * xc);
                    self.mon_var_exaff[idx] = sanitize(
                        (1.0 - beta_m) * self.mon_var_exaff[idx] + beta_m * x_tilde * x_tilde,
                    );
                    if age >= 2 {
                        self.mon_var_dx[idx] =
                            sanitize((1.0 - beta_m) * self.mon_var_dx[idx] + beta_m * dx * dx);
                        self.mon_cov_dx[idx] = sanitize(
                            (1.0 - beta_m) * self.mon_cov_dx[idx]
                                + beta_m * dx * self.mon_prev_dx[idx],
                        );
                    }
                    for t_idx in 0..tap_len {
                        g_slice[t_idx] =
                            sanitize(0.999 * g_slice[t_idx] + eta_g * x_tilde * self.da_ring[t_idx])
                                .clamp(-2.0, 2.0);
                    }
                    self.mon_age[idx] = age.saturating_add(1);
                    self.mon_prev_xt[idx] = x_tilde;
                    self.mon_prev_dx[idx] = dx;
                }
                if idx < k {
                    self.pore_signal[idx] = sanitize(raw_x - reaff).clamp(-1.0, 1.0);
                }
            }
        }

        let bias_idx = 2 * k;
        if self.ablation.no_center_surround {
            for p in 0..k {
                let xi = self.pore_signal[p];
                self.feat[p] = xi.max(0.0);
                self.feat[k + p] = (-xi).max(0.0);
            }
        } else {
            let mut mean = 0.0_f32;
            for p in 0..k {
                mean += self.pore_signal[p];
            }
            mean /= (k.max(1)) as f32;
            let mut energy = 0.0_f32;
            for p in 0..k {
                let z = self.pore_signal[p] - mean;
                let on = z.max(0.0);
                let off = (-z).max(0.0);
                self.feat[p] = on;
                self.feat[k + p] = off;
                energy += on * on + off * off;
            }
            let inv = 1.0 / (self.params.norm_c + energy).sqrt();
            for f in &mut self.feat[..2 * k] {
                *f *= inv;
            }
        }
        self.feat[bias_idx] = 1.0;
    }

    fn critic_value_on(&self, ctx: usize, f: &[f32]) -> f32 {
        let off = ctx * self.n_feat;
        self.v_ctx[off..off + self.n_feat]
            .iter()
            .zip(f.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn outcome_pred_live_on(&self, ctx: usize, f: &[f32]) -> f32 {
        let off = ctx * self.n_feat;
        let n_cue = 2 * self.k_pores;
        self.q_ctx[off..off + n_cue]
            .iter()
            .zip(f[..n_cue].iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn outcome_pred_cons_on(&self, ctx: usize, f: &[f32]) -> f32 {
        let off = ctx * self.n_feat;
        let n_cue = 2 * self.k_pores;
        self.q_cons[off..off + n_cue]
            .iter()
            .zip(f[..n_cue].iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn consolidate_active_context(&mut self) {
        let c = self.active_ctx;
        let w_len = self.n_act * self.n_feat;
        let nf = self.n_feat;
        self.w_cons[c * w_len..(c + 1) * w_len]
            .copy_from_slice(&self.w_ctx[c * w_len..(c + 1) * w_len]);
        self.v_cons[c * nf..(c + 1) * nf].copy_from_slice(&self.v_ctx[c * nf..(c + 1) * nf]);
        self.q_cons[c * nf..(c + 1) * nf].copy_from_slice(&self.q_ctx[c * nf..(c + 1) * nf]);
        self.ctx_has_cons[c] = 1;
    }

    fn restore_consolidated_context(&mut self, c: usize) {
        if self.ctx_has_cons[c] == 0 {
            return;
        }
        let w_len = self.n_act * self.n_feat;
        let nf = self.n_feat;
        self.w_ctx[c * w_len..(c + 1) * w_len]
            .copy_from_slice(&self.w_cons[c * w_len..(c + 1) * w_len]);
        self.v_ctx[c * nf..(c + 1) * nf].copy_from_slice(&self.v_cons[c * nf..(c + 1) * nf]);
        self.q_ctx[c * nf..(c + 1) * nf].copy_from_slice(&self.q_cons[c * nf..(c + 1) * nf]);
    }

    fn plastic(&self) -> bool {
        self.mode == ControlMode::Natural
    }
}

impl OrganismEngine for CausalCore {
    fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError> {
        if self.poisoned {
            return Err(PersistError::PoisonedAfterCorruption);
        }
        if self.n_feat == 0 {
            self.bind(view.ambient_in.len(), view.num_actuators);
        }
        let k = self.k_pores;
        let n_act = self.n_act;
        let n_feat = self.n_feat;
        let w_len = n_act * n_feat;
        let n_fwd = self.n_fwd;
        let is_plastic = self.plastic();
        let no_fwd = self.ablation.no_forward_agency;
        let no_ctx = self.ablation.no_latent_context;

        // 1. Somatic ATP homeostasis.
        let influx = sanitize(view.metabolic_influx).max(0.0);
        let stress = sanitize(view.tissue_stress).max(0.0);
        let reward = influx - stress;
        let mean_motor_sq = if n_act > 0 {
            self.prev_act.iter().map(|a| a * a).sum::<f32>() / (n_act as f32)
        } else {
            0.0
        };
        let atp_burn = 0.0012 + 0.0010 * mean_motor_sq + 0.45 * stress;
        self.atp = sanitize(self.atp + influx - atp_burn).clamp(0.0, MAX_ATP);

        // 1a. Peripheral O(1) sensing & corollary cancellation.
        self.update_pores_and_features(view.ambient_in);

        // 1b. Out-of-sample zero-mean action-centered forward model (I1 agency purity).
        if self.has_prev && n_fwd > 0 {
            let mut da = [0.0_f32; CAUSAL_MAX_ACTUATORS];
            let mut da_norm_sq = 0.0_f32;
            for m in 0..n_act {
                let d = self.prev_act[m] - self.act_mean[m];
                da[m] = d;
                da_norm_sq += d * d;
            }
            let beta_v = 0.004_f32;
            let eta_j = 0.015 / (da_norm_sq + 0.25);
            for j in 0..n_fwd {
                let df = self.feat[j] - self.prev_feat[j];
                let pdf = self.prev_df[j];
                let pred_null = self.fwd_b[j] + self.fwd_alpha[j] * pdf;
                let e_null = df - pred_null;
                let mut act_pred = 0.0_f32;
                if !no_fwd {
                    let row_j = &self.fwd_j[j * n_act..(j + 1) * n_act];
                    for m in 0..n_act {
                        act_pred += row_j[m] * da[m];
                    }
                }
                let e_act = e_null - act_pred;
                if is_plastic {
                    self.fwd_var_null[j] =
                        sanitize((1.0 - beta_v) * self.fwd_var_null[j] + beta_v * e_null * e_null);
                    self.fwd_var_act[j] =
                        sanitize((1.0 - beta_v) * self.fwd_var_act[j] + beta_v * e_act * e_act);
                    self.fwd_b[j] = sanitize(self.fwd_b[j] + 0.02 * e_null).clamp(-1.0, 1.0);
                    let denom_p = pdf * pdf + 0.01;
                    self.fwd_alpha[j] =
                        sanitize(self.fwd_alpha[j] + 0.04 * e_null * pdf / denom_p)
                            .clamp(-0.98, 0.98);
                    if !no_fwd {
                        let row_j = &mut self.fwd_j[j * n_act..(j + 1) * n_act];
                        for m in 0..n_act {
                            row_j[m] =
                                sanitize(0.9995 * row_j[m] + eta_j * e_act * da[m]).clamp(-2.0, 2.0);
                        }
                    }
                }
                self.prev_df[j] = df;
            }
            if is_plastic {
                for m in 0..n_act {
                    self.act_mean[m] =
                        sanitize(0.99 * self.act_mean[m] + 0.01 * self.prev_act[m]).clamp(-1.0, 1.0);
                }
            }
        }
        self.prev_feat.copy_from_slice(&self.feat[..n_fwd]);

        // 1c. Lateral OFC latent context inference & Rule 4 dormant consolidation.
        let mut switched_context = false;
        if is_plastic && self.has_prev {
            let act_c = self.active_ctx;
            self.ctx_ticks[act_c] = self.ctx_ticks[act_c].saturating_add(1);
            if self.ctx_cooldown > 0 {
                self.ctx_cooldown -= 1;
            }
            self.r_fast_ema = sanitize(0.97 * self.r_fast_ema + 0.03 * reward);
            self.r_med200 = sanitize(0.994 * self.r_med200 + 0.006 * reward);

            let rh_act_live = self.outcome_pred_live_on(act_c, &self.prev_critic_feat);
            let mut app_rec_sig = 0.0_f32;
            if reward > 0.005 {
                for c in 0..self.num_ctx {
                    if c != act_c && self.ctx_has_cons[c] != 0 {
                        let rh_c = self.outcome_pred_cons_on(c, &self.prev_critic_feat);
                        if rh_c > 0.006 && rh_act_live < 0.003 {
                            app_rec_sig = app_rec_sig.max(reward * (rh_c - rh_act_live));
                        }
                    }
                }
            }
            if app_rec_sig > 0.0 {
                self.valence_flip_ema = sanitize(0.85 * self.valence_flip_ema + 0.15 * app_rec_sig);
            } else {
                self.valence_flip_ema = sanitize(0.96 * self.valence_flip_ema);
            }

            if reward.abs() > 0.005 {
                self.ctx_out_events = self.ctx_out_events.saturating_add(1);
                for c in 0..self.num_ctx {
                    let rh = if c != act_c && self.ctx_has_cons[c] != 0 {
                        self.outcome_pred_cons_on(c, &self.prev_critic_feat)
                    } else {
                        self.outcome_pred_live_on(c, &self.prev_critic_feat)
                    };
                    let err_sq = (reward - rh) * (reward - rh);
                    self.ctx_out_err[c] = sanitize(0.88 * self.ctx_out_err[c] + 0.12 * err_sq);
                }
            }

            if self.ctx_ticks[act_c] >= 350
                && self.r_med200 > 0.002
                && self.r_med200 >= self.ctx_r_peak[act_c] - 0.003
            {
                self.ctx_r_peak[act_c] = self.ctx_r_peak[act_c].max(self.r_med200);
                self.consolidate_active_context();
            } else if act_c > 0
                && self.ctx_ticks[act_c] >= 750
                && self.r_med200 > -0.009
                && (self.ctx_has_cons[act_c] == 0
                    || self.r_med200 >= self.ctx_r_peak[act_c] - 0.002)
            {
                if self.ctx_has_cons[act_c] == 0 {
                    self.ctx_r_peak[act_c] = self.r_med200;
                } else {
                    self.ctx_r_peak[act_c] = self.ctx_r_peak[act_c].max(self.r_med200);
                }
                self.consolidate_active_context();
            }

            if !no_ctx && self.ctx_cooldown == 0 && self.ctx_ticks[act_c] >= 220 {
                // Path A: Recall an existing consolidated context when cue-outcome contingencies
                // match the dormant consolidated model or mature secondary policy hits reversal wall-slam.
                let mut recall_target = None;
                let mut best_err = f32::MAX;
                for c in 0..self.num_ctx {
                    if c != act_c && self.ctx_has_cons[c] != 0 {
                        let app_match = self.valence_flip_ema > 0.00012;
                        let out_match = self.ctx_out_events >= 5
                            && self.ctx_out_err[act_c] > self.ctx_out_err[c] + 0.0008
                            && self.ctx_out_err[c] < 0.0025;
                        let mature_collapse_recall = self.ctx_ticks[act_c] >= 800
                            && (self.r_med200 < -0.0145 || self.r_fast_ema < -0.028)
                            && self.ctx_out_err[c] <= self.ctx_out_err[act_c] + 0.0004;
                        if (app_match || out_match || mature_collapse_recall)
                            && self.ctx_out_err[c] < best_err
                        {
                            best_err = self.ctx_out_err[c];
                            recall_target = Some(c);
                        }
                    }
                }

                // Path B: Spawn a fresh context slot ONLY if a consolidated nourishing context
                // suffers sustained homeostatic collapse (< -0.0125, zero stationary false positives).
                let spawn_collapse = self.ctx_has_cons[act_c] != 0
                    && self.ctx_r_peak[act_c] > 0.002
                    && self.r_med200 < -0.0125
                    && self.ctx_r_peak[act_c] - self.r_med200 > 0.016;

                let next_ctx = if let Some(c) = recall_target {
                    Some(c)
                } else if spawn_collapse {
                    if self.num_ctx < MAX_CONTEXTS {
                        let nc = self.num_ctx;
                        self.num_ctx += 1;
                        Some(nc)
                    } else {
                        (0..self.num_ctx)
                            .filter(|&c| c != act_c)
                            .min_by_key(|&c| self.ctx_ticks[c])
                    }
                } else {
                    None
                };

                if let Some(target_ctx) = next_ctx {
                    if target_ctx != act_c {
                        if self.ctx_has_cons[act_c] != 0 {
                            self.restore_consolidated_context(act_c);
                        }
                        if self.ctx_has_cons[target_ctx] != 0 {
                            self.restore_consolidated_context(target_ctx);
                        }
                        self.active_ctx = target_ctx;
                        self.ctx_cooldown = 180;
                        self.ctx_out_events = 0;
                        self.ctx_out_err = [0.0; MAX_CONTEXTS];
                        self.r_fast_ema = 0.0;
                        self.r_med200 = 0.0;
                        self.valence_flip_ema = 0.0;
                        self.elig_actor.fill(0.0);
                        self.elig_critic.fill(0.0);
                        self.context_switches = self.context_switches.saturating_add(1);
                        switched_context = true;
                    }
                }
            }
        }

        let act_c = self.active_ctx;
        let v_now = self.critic_value_on(act_c, &self.feat);

        // 2. Credit r_t (caused by state-action at t-1) to active context ONLY if we did not switch.
        if is_plastic && self.has_prev && !switched_context {
            let delta = if self.ablation.no_td_critic {
                reward - self.r_bar_ctx[act_c]
            } else {
                reward + self.params.gamma * v_now - self.v_prev
            };
            let delta = sanitize(delta).clamp(-TD_CLAMP, TD_CLAMP);
            self.r_bar_ctx[act_c] = 0.99 * self.r_bar_ctx[act_c] + 0.01 * reward;
            self.td_abs_ema = 0.99 * self.td_abs_ema + 0.01 * delta.abs();
            if !self.ablation.no_td_critic {
                let v_off = act_c * n_feat;
                for (vi, ei) in self.v_ctx[v_off..v_off + n_feat]
                    .iter_mut()
                    .zip(self.elig_critic.iter())
                {
                    *vi = sanitize(*vi + self.params.eta_critic * delta * ei).clamp(-W_CLAMP, W_CLAMP);
                }
            }
            let rhat_q = self.outcome_pred_live_on(act_c, &self.prev_critic_feat);
            let r_err = sanitize(reward - rhat_q).clamp(-TD_CLAMP, TD_CLAMP);
            let eta_q = if reward.abs() > 1e-4 { 0.14 } else { 0.005 };
            let q_off = act_c * n_feat;
            let n_cue = 2 * k;
            for (qi, fi) in self.q_ctx[q_off..q_off + n_cue]
                .iter_mut()
                .zip(self.prev_critic_feat[..n_cue].iter())
            {
                *qi = sanitize(0.9995 * *qi + eta_q * r_err * fi).clamp(-W_CLAMP, W_CLAMP);
            }
            let eta_a = self.params.eta_actor;
            let w_off = act_c * w_len;
            for (wi, ei) in self.w_ctx[w_off..w_off + w_len]
                .iter_mut()
                .zip(self.elig_actor.iter())
            {
                *wi = sanitize(*wi + eta_a * delta * ei).clamp(-W_CLAMP, W_CLAMP);
            }
        }
        self.prev_critic_feat.copy_from_slice(&self.feat);

        // 2b. Periodic O(1) sensory pore morphogenesis (every 64 ticks, after credit assignment).
        if is_plastic && self.tick > 0 && (self.tick & 63) == 0 {
            self.morph_pore_frontier();
        }

        // 3. Choose a_t from active context `act_c`, then extend traces and motor efference ring.
        let decay = self.params.gamma * self.params.lambda;
        let rho = self.params.noise_rho;
        let innov = self.params.noise_sigma * (1.0 - rho * rho).sqrt();
        let w_off = act_c * w_len;
        let mut out = vec![0.0_f32; view.num_actuators];
        for m in 0..n_act {
            let g = self.gaussian();
            self.noise[m] = rho * self.noise[m] + innov * g;
            let row = &self.w_ctx[w_off + m * n_feat..w_off + m * n_feat + 2 * k];
            let pre: f32 = row.iter().zip(self.feat.iter()).map(|(a, b)| a * b).sum();
            let mean_a = pre.tanh();
            let a = (pre + self.noise[m]).tanh();
            let a_clean = sanitize(a).clamp(-1.0, 1.0);
            out[m] = a_clean;
            self.prev_act[m] = a_clean;
            let pert = a - mean_a;
            if is_plastic {
                let erow = &mut self.elig_actor[m * n_feat..m * n_feat + 2 * k];
                for (e, f) in erow.iter_mut().zip(self.feat.iter()) {
                    *e = decay * *e + pert * f;
                }
            }
        }
        if is_plastic {
            for (e, f) in self.elig_critic.iter_mut().zip(self.feat.iter()) {
                *e = decay * *e + f;
            }
        }
        if n_act > 0 {
            for tap in (1..MOTOR_TAPS).rev() {
                for m in 0..n_act {
                    self.da_ring[tap * n_act + m] = self.da_ring[(tap - 1) * n_act + m];
                }
            }
            for m in 0..n_act {
                self.da_ring[m] = self.prev_act[m] - self.act_mean[m];
            }
        }

        self.v_prev = v_now;
        self.has_prev = true;
        self.tick = self.tick.saturating_add(1);
        Ok(FieldPulse { actuator_out: out })
    }

    fn set_control_mode(&mut self, mode: ControlMode) {
        self.mode = mode;
    }

    fn control_mode(&self) -> ControlMode {
        self.mode
    }

    fn zero_learned_conductances(&mut self) {
        let w_len = self.n_act * self.n_feat;
        for c in 0..MAX_CONTEXTS {
            self.w_ctx[c * w_len..(c + 1) * w_len].copy_from_slice(&self.w_birth);
            self.w_cons[c * w_len..(c + 1) * w_len].copy_from_slice(&self.w_birth);
        }
        self.v_ctx.fill(0.0);
        self.v_cons.fill(0.0);
        self.q_ctx.fill(0.0);
        self.q_cons.fill(0.0);
        self.fwd_j.fill(0.0);
        self.num_ctx = 1;
        self.active_ctx = 0;
        self.ctx_cooldown = 0;
        self.ctx_out_events = 0;
        self.context_switches = 0;
        self.ctx_ticks = [0; MAX_CONTEXTS];
        self.ctx_has_cons = [0; MAX_CONTEXTS];
        self.ctx_r_peak = [0.0; MAX_CONTEXTS];
        self.ctx_out_err = [0.0; MAX_CONTEXTS];
        self.r_bar_ctx = [0.0; MAX_CONTEXTS];
        self.r_fast_ema = 0.0;
        self.r_med200 = 0.0;
        self.valence_flip_ema = 0.0;
        self.elig_actor.fill(0.0);
        self.elig_critic.fill(0.0);
        self.act_mean.fill(0.0);
        self.fwd_b.fill(0.0);
        self.fwd_alpha.fill(0.0);
        self.fwd_var_null.fill(1e-4);
        self.fwd_var_act.fill(1e-4);
        for p in 0..self.k_pores {
            self.mon_ch[p] = self.pore_ch_birth[p];
        }
        self.mon_age.fill(0);
        self.mon_mean_x.fill(0.0);
        self.mon_var_raw.fill(1e-4);
        self.mon_var_exaff.fill(1e-4);
        self.mon_prev_xt.fill(0.0);
        self.mon_prev_dx.fill(0.0);
        self.mon_var_dx.fill(1e-4);
        self.mon_cov_dx.fill(0.0);
        self.motor_canc_g.fill(0.0);
        self.da_ring.fill(0.0);
    }

    fn learned_parameter_hash(&self) -> u64 {
        let mut h = ParamHasher::new(self.seed);
        h.write_usize(self.n_feat);
        h.write_usize(self.num_ctx);
        h.write_usize(self.active_ctx);
        for &c in &self.mon_ch[..self.k_pores] {
            h.write_usize(c as usize);
        }
        for &x in &self.w_ctx {
            h.write_f32(x);
        }
        for &x in &self.v_ctx {
            h.write_f32(x);
        }
        for &x in &self.q_ctx {
            h.write_f32(x);
        }
        for &x in &self.w_cons {
            h.write_f32(x);
        }
        for &x in &self.v_cons {
            h.write_f32(x);
        }
        for &x in &self.q_cons {
            h.write_f32(x);
        }
        for &x in &self.fwd_j {
            h.write_f32(x);
        }
        for &x in &self.fwd_b {
            h.write_f32(x);
        }
        for &x in &self.fwd_alpha {
            h.write_f32(x);
        }
        for &x in &self.act_mean {
            h.write_f32(x);
        }
        for &x in &self.mon_mean_x[..self.k_pores] {
            h.write_f32(x);
        }
        let active_g = self.k_pores * MOTOR_TAPS * self.n_act;
        for &x in &self.motor_canc_g[..active_g] {
            h.write_f32(x);
        }
        for &rb in &self.r_bar_ctx {
            h.write_f32(rb);
        }
        h.finish()
    }

    fn snapshot(&self) -> OrganismSnapshot {
        let agency = if self.tick < 120 || self.ablation.no_forward_agency || self.n_fwd == 0 {
            0.0
        } else {
            let dof_pen = (self.n_act as f32) / 500.0;
            let mut sum_r2 = 0.0_f32;
            let mut active_cnt = 0_usize;
            for j in 0..self.n_fwd {
                let vn = self.fwd_var_null[j];
                if vn > 1e-7 {
                    let r2 = ((vn - self.fwd_var_act[j]) / vn - dof_pen).clamp(0.0, 1.0);
                    sum_r2 += r2;
                    active_cnt += 1;
                }
            }
            if active_cnt > 0 {
                (sum_r2 / (active_cnt as f32)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let k = self.k_pores;
        let w_off = self.active_ctx * self.n_act * self.n_feat;
        OrganismSnapshot {
            tick: self.tick,
            atp: self.atp,
            cell_count: self.n_feat * self.num_ctx,
            tube_count: self.n_act * self.n_feat * self.num_ctx,
            sensor_pore_count: k,
            motor_pore_count: self.n_act,
            sensor_pore_channels: self.mon_ch[..k].to_vec(),
            sensor_pore_conductances: (0..k)
                .map(|p| {
                    let w_sum: f32 = (0..self.n_act)
                        .map(|m| {
                            self.w_ctx[w_off + m * self.n_feat + p].abs()
                                + self.w_ctx[w_off + m * self.n_feat + k + p].abs()
                        })
                        .sum();
                    w_sum.max(1e-4)
                })
                .collect(),
            sensor_pore_delays: vec![0; k],
            motor_pore_channels: (0..self.n_act as u32).collect(),
            motor_pore_conductances: vec![1.0; self.n_act],
            mean_prediction_error: self.td_abs_ema,
            mean_interventional_agency: agency,
            contradiction_splits: self.context_switches,
            committed_motor_pulses: self.tick,
            used_state_bytes: self.used_state_bytes(),
            ..OrganismSnapshot::default()
        }
    }

    fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError> {
        if self.poisoned {
            return Err(PersistError::PoisonedAfterCorruption);
        }
        let mut p = Vec::with_capacity(self.used_state_bytes() + 256);
        put_u64(&mut p, self.seed);
        put_u64(&mut p, self.tick);
        p.push(u8::from(self.mode == ControlMode::FrozenPlasticity));
        p.push(u8::from(self.ablation.no_center_surround));
        p.push(u8::from(self.ablation.no_td_critic));
        p.push(u8::from(self.ablation.no_forward_agency));
        p.push(u8::from(self.ablation.no_pore_frontier));
        p.push(u8::from(self.ablation.no_latent_context));
        p.push(u8::from(self.has_prev));
        p.extend_from_slice(&self.ctx_has_cons);
        for x in params_to_array(&self.params) {
            put_f32(&mut p, x);
        }
        put_u32(&mut p, self.n_in as u32);
        put_u32(&mut p, self.n_act as u32);
        put_u32(&mut p, self.scout_cursor as u32);
        put_u32(&mut p, self.num_ctx as u32);
        put_u32(&mut p, self.active_ctx as u32);
        put_u32(&mut p, self.ctx_cooldown);
        put_u32(&mut p, self.ctx_out_events);
        put_u64(&mut p, self.context_switches);
        for &ct in &self.ctx_ticks {
            put_u32(&mut p, ct);
        }
        for &cp in &self.ctx_r_peak {
            put_f32(&mut p, cp);
        }
        for &ce in &self.ctx_out_err {
            put_f32(&mut p, ce);
        }
        for &rb in &self.r_bar_ctx {
            put_f32(&mut p, rb);
        }
        put_f32(&mut p, self.r_fast_ema);
        put_f32(&mut p, self.r_med200);
        put_f32(&mut p, self.valence_flip_ema);
        put_f32(&mut p, self.atp);
        put_f32(&mut p, self.v_prev);
        put_f32(&mut p, self.td_abs_ema);
        put_u64(&mut p, self.rng.s0);
        put_u64(&mut p, self.rng.s1);
        for uslab in [&self.pore_ch_birth, &self.mon_ch, &self.mon_age] {
            put_u32(&mut p, uslab.len() as u32);
            for &u in uslab {
                put_u32(&mut p, u);
            }
        }
        for slab in [
            &self.mon_mean_x,
            &self.mon_var_raw,
            &self.mon_var_exaff,
            &self.mon_prev_xt,
            &self.mon_prev_dx,
            &self.mon_var_dx,
            &self.mon_cov_dx,
            &self.motor_canc_g,
            &self.da_ring,
            &self.pore_signal,
            &self.w_birth,
            &self.w_ctx,
            &self.v_ctx,
            &self.q_ctx,
            &self.w_cons,
            &self.v_cons,
            &self.q_cons,
            &self.elig_actor,
            &self.elig_critic,
            &self.feat,
            &self.prev_critic_feat,
            &self.prev_feat,
            &self.prev_df,
            &self.prev_act,
            &self.act_mean,
            &self.fwd_b,
            &self.fwd_alpha,
            &self.fwd_j,
            &self.fwd_var_null,
            &self.fwd_var_act,
            &self.noise,
        ] {
            put_u32(&mut p, slab.len() as u32);
            for &x in slab {
                put_f32(&mut p, x);
            }
        }
        let mut out = Vec::with_capacity(p.len() + 24);
        put_u64(&mut out, CKPT_MAGIC);
        put_u32(&mut out, CKPT_VERSION);
        put_u64(&mut out, p.len() as u64);
        out.extend_from_slice(&p);
        let crc = crc32c(&out);
        put_u32(&mut out, crc);
        Ok(out)
    }

    fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError> {
        if self.poisoned {
            return Err(PersistError::PoisonedAfterCorruption);
        }
        match decode_causal(bytes) {
            Ok(decoded) => {
                *self = decoded;
                Ok(())
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }

    fn verify_checkpoint(&self, bytes: &[u8]) -> Result<(), PersistError> {
        decode_causal(bytes).map(|_| ())
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

impl CausalCore {
    #[must_use]
    pub fn used_state_bytes(&self) -> usize {
        4 * (self.pore_ch_birth.len()
            + self.mon_ch.len()
            + self.mon_age.len()
            + self.mon_mean_x.len()
            + self.mon_var_raw.len()
            + self.mon_var_exaff.len()
            + self.mon_prev_xt.len()
            + self.mon_prev_dx.len()
            + self.mon_var_dx.len()
            + self.mon_cov_dx.len()
            + self.motor_canc_g.len()
            + self.da_ring.len()
            + self.pore_signal.len()
            + self.w_birth.len()
            + self.w_ctx.len()
            + self.v_ctx.len()
            + self.q_ctx.len()
            + self.w_cons.len()
            + self.v_cons.len()
            + self.q_cons.len()
            + self.elig_actor.len()
            + self.elig_critic.len()
            + self.feat.len()
            + self.prev_critic_feat.len()
            + self.prev_feat.len()
            + self.prev_df.len()
            + self.prev_act.len()
            + self.act_mean.len()
            + self.fwd_b.len()
            + self.fwd_alpha.len()
            + self.fwd_j.len()
            + self.fwd_var_null.len()
            + self.fwd_var_act.len()
            + self.noise.len())
            + 280
    }
}

fn params_to_array(p: &CausalParams) -> [f32; 8] {
    [
        p.eta_actor,
        p.eta_critic,
        p.gamma,
        p.lambda,
        p.noise_sigma,
        p.noise_rho,
        p.norm_c,
        p.birth_w_scale,
    ]
}

fn params_from_array(a: [f32; 8]) -> CausalParams {
    CausalParams {
        eta_actor: a[0],
        eta_critic: a[1],
        gamma: a[2],
        lambda: a[3],
        noise_sigma: a[4],
        noise_rho: a[5],
        norm_c: a[6],
        birth_w_scale: a[7],
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Reader<'a> {
    d: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], PersistError> {
        let end = self.pos.checked_add(n).ok_or(PersistError::TruncatedInput)?;
        if end > self.d.len() {
            return Err(PersistError::TruncatedInput);
        }
        let s = &self.d[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, PersistError> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, PersistError> {
        let mut b = [0u8; 8];
        b.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(b))
    }
    fn f32(&mut self) -> Result<f32, PersistError> {
        let v = f32::from_bits(self.u32()?);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(PersistError::NonFiniteState)
        }
    }
    fn uslab(&mut self, expect: usize) -> Result<Vec<u32>, PersistError> {
        let n = self.u32()? as usize;
        if n != expect {
            return Err(PersistError::InvalidTopology);
        }
        (0..n).map(|_| self.u32()).collect()
    }
    fn slab(&mut self, expect: usize) -> Result<Vec<f32>, PersistError> {
        let n = self.u32()? as usize;
        if n != expect {
            return Err(PersistError::InvalidTopology);
        }
        (0..n).map(|_| self.f32()).collect()
    }
}

fn decode_causal(bytes: &[u8]) -> Result<CausalCore, PersistError> {
    let max_bytes =
        4 * (MAX_CONTEXTS * 6 + 14) * (CAUSAL_MAX_ACTUATORS + 6) * (2 * CAUSAL_MAX_CHANNELS + 1)
            + 262_144;
    if bytes.len() > max_bytes {
        return Err(PersistError::OversizedInput);
    }
    if bytes.len() < 24 {
        return Err(PersistError::TruncatedInput);
    }
    let body_end = bytes.len() - 4;
    let mut crc_b = [0u8; 4];
    crc_b.copy_from_slice(&bytes[body_end..]);
    if crc32c(&bytes[..body_end]) != u32::from_le_bytes(crc_b) {
        return Err(PersistError::ChecksumMismatch);
    }
    let mut r = Reader {
        d: &bytes[..body_end],
        pos: 0,
    };
    if r.u64()? != CKPT_MAGIC || r.u32()? != CKPT_VERSION {
        return Err(PersistError::BadMagicOrVersion);
    }
    let payload_len = r.u64()? as usize;
    if payload_len != body_end - r.pos {
        return Err(PersistError::TruncatedInput);
    }
    let seed = r.u64()?;
    let tick = r.u64()?;
    let frozen = r.u8()? != 0;
    let ablation = CausalAblation {
        no_center_surround: r.u8()? != 0,
        no_td_critic: r.u8()? != 0,
        no_forward_agency: r.u8()? != 0,
        no_pore_frontier: r.u8()? != 0,
        no_latent_context: r.u8()? != 0,
    };
    let has_prev = r.u8()? != 0;
    let mut ctx_has_cons = [0_u8; MAX_CONTEXTS];
    for x in &mut ctx_has_cons {
        *x = r.u8()?;
    }
    let mut pa = [0.0_f32; 8];
    for x in &mut pa {
        *x = r.f32()?;
    }
    let params = params_from_array(pa);
    let n_in = r.u32()? as usize;
    let n_act = r.u32()? as usize;
    let scout_cursor = r.u32()? as usize;
    let num_ctx = r.u32()? as usize;
    let active_ctx = r.u32()? as usize;
    let ctx_cooldown = r.u32()?;
    let ctx_out_events = r.u32()?;
    let context_switches = r.u64()?;
    if n_in > CAUSAL_MAX_CHANNELS
        || n_act > CAUSAL_MAX_ACTUATORS
        || num_ctx == 0
        || num_ctx > MAX_CONTEXTS
        || active_ctx >= num_ctx
    {
        return Err(PersistError::InvalidTopology);
    }
    let mut ctx_ticks = [0_u32; MAX_CONTEXTS];
    for x in &mut ctx_ticks {
        *x = r.u32()?;
    }
    let mut ctx_r_peak = [0.0_f32; MAX_CONTEXTS];
    for x in &mut ctx_r_peak {
        *x = r.f32()?;
    }
    let mut ctx_out_err = [0.0_f32; MAX_CONTEXTS];
    for x in &mut ctx_out_err {
        *x = r.f32()?;
    }
    let mut r_bar_ctx = [0.0_f32; MAX_CONTEXTS];
    for x in &mut r_bar_ctx {
        *x = r.f32()?;
    }
    let r_fast_ema = r.f32()?;
    let r_med200 = r.f32()?;
    let valence_flip_ema = r.f32()?;
    let (k_pores, n_scout, n_mon, n_feat, n_fwd) = if n_in == 0 && n_act == 0 {
        (0, 0, 0, 0, 0)
    } else {
        let kp = if ablation.no_pore_frontier {
            n_in
        } else {
            n_in.min(MAX_ACTIVE_PORES)
        };
        let ns = if ablation.no_pore_frontier {
            0
        } else {
            n_in.saturating_sub(kp).min(SCOUT_SLOTS)
        };
        (kp, ns, kp + ns, 2 * kp + 1, (2 * kp).min(FWD_MAX_FEATURES))
    };
    let atp = r.f32()?;
    let v_prev = r.f32()?;
    let td_abs_ema = r.f32()?;
    let s0 = r.u64()?;
    let s1 = r.u64()?;
    let pore_ch_birth = r.uslab(k_pores)?;
    let mon_ch = r.uslab(n_mon)?;
    let mon_age = r.uslab(n_mon)?;
    if n_in > 0 && mon_ch.iter().any(|&c| (c as usize) >= n_in) {
        return Err(PersistError::InvalidTopology);
    }
    let mon_mean_x = r.slab(n_mon)?;
    let mon_var_raw = r.slab(n_mon)?;
    let mon_var_exaff = r.slab(n_mon)?;
    let mon_prev_xt = r.slab(n_mon)?;
    let mon_prev_dx = r.slab(n_mon)?;
    let mon_var_dx = r.slab(n_mon)?;
    let mon_cov_dx = r.slab(n_mon)?;
    let motor_canc_g = r.slab(n_mon * MOTOR_TAPS * n_act)?;
    let da_ring = r.slab(MOTOR_TAPS * n_act)?;
    let pore_signal = r.slab(k_pores)?;
    let w_birth = r.slab(n_act * n_feat)?;
    let w_ctx = r.slab(MAX_CONTEXTS * n_act * n_feat)?;
    let v_ctx = r.slab(MAX_CONTEXTS * n_feat)?;
    let q_ctx = r.slab(MAX_CONTEXTS * n_feat)?;
    let w_cons = r.slab(MAX_CONTEXTS * n_act * n_feat)?;
    let v_cons = r.slab(MAX_CONTEXTS * n_feat)?;
    let q_cons = r.slab(MAX_CONTEXTS * n_feat)?;
    let elig_actor = r.slab(n_act * n_feat)?;
    let elig_critic = r.slab(n_feat)?;
    let feat = r.slab(n_feat)?;
    let prev_critic_feat = r.slab(n_feat)?;
    let prev_feat = r.slab(n_fwd)?;
    let prev_df = r.slab(n_fwd)?;
    let prev_act = r.slab(n_act)?;
    let act_mean = r.slab(n_act)?;
    let fwd_b = r.slab(n_fwd)?;
    let fwd_alpha = r.slab(n_fwd)?;
    let fwd_j = r.slab(n_fwd * n_act)?;
    let fwd_var_null = r.slab(n_fwd)?;
    let fwd_var_act = r.slab(n_fwd)?;
    let noise = r.slab(n_act)?;
    if r.pos != r.d.len() {
        return Err(PersistError::InvalidTopology);
    }
    Ok(CausalCore {
        seed,
        mode: if frozen {
            ControlMode::FrozenPlasticity
        } else {
            ControlMode::Natural
        },
        poisoned: false,
        ablation,
        params,
        tick,
        n_in,
        k_pores,
        n_scout,
        n_mon,
        n_act,
        n_feat,
        n_fwd,
        scout_cursor: if n_in > 0 { scout_cursor % n_in } else { 0 },
        num_ctx,
        active_ctx,
        ctx_cooldown,
        ctx_out_events,
        context_switches,
        ctx_ticks,
        ctx_has_cons,
        ctx_r_peak,
        ctx_out_err,
        r_bar_ctx,
        r_fast_ema,
        r_med200,
        valence_flip_ema,
        pore_ch_birth,
        mon_ch,
        mon_age,
        mon_mean_x,
        mon_var_raw,
        mon_var_exaff,
        mon_prev_xt,
        mon_prev_dx,
        mon_var_dx,
        mon_cov_dx,
        motor_canc_g,
        da_ring,
        pore_signal,
        w_birth,
        w_ctx,
        v_ctx,
        q_ctx,
        w_cons,
        v_cons,
        q_cons,
        elig_actor,
        elig_critic,
        feat,
        prev_critic_feat,
        prev_feat,
        prev_df,
        prev_act,
        act_mean,
        fwd_b,
        fwd_alpha,
        fwd_j,
        fwd_var_null,
        fwd_var_act,
        atp,
        v_prev,
        has_prev,
        noise,
        td_abs_ema,
        rng: Rng64::from_raw(s0, s1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(x: &'a [f32], influx: f32, stress: f32) -> FieldView<'a> {
        FieldView {
            ambient_in: x,
            metabolic_influx: influx,
            tissue_stress: stress,
            num_actuators: 4,
        }
    }

    #[test]
    fn frozen_hash_is_bit_invariant_and_checkpoint_roundtrips_and_rejects_corruption() {
        let x: Vec<f32> = (0..96).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut org = CausalCore::new(7);
        for t in 0..80 {
            let r = if t % 5 == 0 { 0.2 } else { 0.0 };
            org.step(view(&x, r, 0.0)).unwrap();
        }
        let bytes = org.encode_checkpoint().unwrap();
        assert!(org.verify_checkpoint(&bytes).is_ok());
        let mut twin = CausalCore::new(99);
        twin.restore_checkpoint(&bytes).unwrap();
        for _ in 0..20 {
            let a = org.step(view(&x, 0.1, 0.0)).unwrap();
            let b = twin.step(view(&x, 0.1, 0.0)).unwrap();
            assert_eq!(a, b);
        }
        org.set_control_mode(ControlMode::FrozenPlasticity);
        let h = org.learned_parameter_hash();
        for _ in 0..200 {
            org.step(view(&x, 0.3, 0.2)).unwrap();
        }
        assert_eq!(h, org.learned_parameter_hash());

        let mut bad = bytes.clone();
        bad[40] ^= 0x10;
        assert!(org.verify_checkpoint(&bad).is_err());
        let mut victim = CausalCore::new(1);
        assert!(victim.restore_checkpoint(&bad).is_err());
        assert!(victim.is_poisoned());
        assert!(victim.step(view(&x, 0.0, 0.0)).is_err());
    }

    #[test]
    fn action_at_t_is_never_credited_with_reward_at_t() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let mut org = CausalCore::new(3);
        org.step(view(&x, 0.0, 0.0)).unwrap();
        let w_after_first = org.w_ctx.clone();
        let mut fresh = CausalCore::new(3);
        fresh.step(view(&x, 5.0, 0.0)).unwrap();
        assert_eq!(w_after_first, fresh.w_ctx);
    }

    #[test]
    fn birth_reset_lesion_restores_exact_newborn_parameters() {
        let x: Vec<f32> = (0..96).map(|i| (i as f32).cos()).collect();
        let mut org = CausalCore::new(11);
        let mut newborn = CausalCore::new(11);
        newborn.step(view(&x, 0.0, 0.0)).unwrap();
        newborn.zero_learned_conductances();
        for t in 0..300 {
            org.step(view(&x, if t % 3 == 0 { 0.4 } else { 0.0 }, 0.0)).unwrap();
        }
        assert_ne!(org.w_ctx, newborn.w_ctx);
        org.zero_learned_conductances();
        assert_eq!(org.w_ctx, newborn.w_ctx);
        assert_eq!(org.mon_ch[..org.k_pores], newborn.mon_ch[..newborn.k_pores]);
        assert!(org.v_ctx.iter().all(|&v| v == 0.0));
    }
}
