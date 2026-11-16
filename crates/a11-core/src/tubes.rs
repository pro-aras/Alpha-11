//! Directed Benna-Fusi cascade tubes, 16-tick efference ring, agency, and rollout clutch.

use a11_medium::{ControlMode, FieldPulse, PersistError};

use crate::body::BodyCore;
use crate::cells::{Cell, CellArena};
use crate::crc32c::crc32c;
use crate::pores::{ChannelScoutStat, PoreMembrane, SensorPore};
use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp, smooth_saturate};

pub const RING_LEN: usize = 16;
/// Minimum per-actuator storage (legacy v3..v5 layout); grows without bound with the body.
pub const MIN_ACTUATOR_DIM: usize = 32;
/// Bounded active sensor-pore slots (k <= 64 per tick).
pub const PRED_BUCKETS: usize = 64;
/// Minimum per-cell predictor storage (legacy v3..v5 width).
pub const MIN_CELL_DIM: usize = 64;
pub const MAX_OUTGOING_PER_CELL: usize = 72;

pub const STRIATAL_PORES: usize = 64;
pub const STRIATAL_STRATA: usize = 4;
pub const STRIATAL_CUES: usize = STRIATAL_STRATA * STRIATAL_PORES;
pub const STRIATAL_FEATS: usize = STRIATAL_CUES + 1;
/// Minimum striatal actuator columns (legacy v5 layout); grows with the body.
pub const MIN_STRIATAL_ACT: usize = 8;
pub const OFC_CONTEXTS: usize = 4;
pub const STRIATAL_TAPS: usize = 4;

const STRIATAL_W_CLAMP: f32 = 4.0;
const STRIATAL_ROW_NORM_CAP: f32 = 3.2;
const STRIATAL_TD_CLAMP: f32 = 2.0;
const STRIATAL_BIRTH_SALT: u64 = 0x5781_A7A1_0FC0_5EED;

/// Binary magic header on every paged cortical column `.slab` file (`A11CSLAB` LE).
pub const COL_SLAB_MAGIC: u64 = 0x4131_3143_534C_4142;
pub const CORTICAL_SLAB_VERSION: u32 = 7;

#[derive(Clone, Debug, PartialEq)]
pub struct PagedColumnEntry {
    pub col_id: u32,
    pub r_peak: f32,
    pub dend_ch: [u32; STRIATAL_PORES],
    pub dend_env: [u8; STRIATAL_PORES],
    pub dend_key: [f32; STRIATAL_PORES],
    pub slab_crc32c: u32,
    pub slab_bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedColumnSlab {
    pub col_id: u32,
    pub na: usize,
    pub ticks: u32,
    pub r_peak: f32,
    pub r_bar: f32,
    pub dend_ch: [u32; STRIATAL_PORES],
    pub dend_env: [u8; STRIATAL_PORES],
    pub dend_key: [f32; STRIATAL_PORES],
    pub w_cons: Vec<f32>,
    pub v_cons: [f32; STRIATAL_FEATS],
    pub q_cons: [f32; STRIATAL_CUES],
    pub fwd_j_cons: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn encode_column_slab(
    col_id: u32,
    na: usize,
    ticks: u32,
    r_peak: f32,
    r_bar: f32,
    dend_ch: &[u32; STRIATAL_PORES],
    dend_env: &[u8; STRIATAL_PORES],
    dend_key: &[f32; STRIATAL_PORES],
    w_cons: &[f32],
    v_cons: &[f32],
    q_cons: &[f32],
    fwd_j_cons: &[f32],
) -> Result<(Vec<u8>, u32), PersistError> {
    let w_len = na * STRIATAL_CUES;
    let fwd_len = STRIATAL_CUES * na;
    if w_cons.len() != w_len
        || v_cons.len() != STRIATAL_FEATS
        || q_cons.len() != STRIATAL_CUES
        || fwd_j_cons.len() != fwd_len
    {
        return Err(PersistError::InvalidTopology);
    }
    let payload_len = 16
        + STRIATAL_PORES * 4
        + STRIATAL_PORES
        + STRIATAL_PORES * 4
        + w_len * 4
        + STRIATAL_FEATS * 4
        + STRIATAL_CUES * 4
        + fwd_len * 4;
    let mut buf = Vec::with_capacity(24 + payload_len);
    buf.extend_from_slice(&COL_SLAB_MAGIC.to_le_bytes());
    buf.extend_from_slice(&CORTICAL_SLAB_VERSION.to_le_bytes());
    buf.extend_from_slice(&col_id.to_le_bytes());
    buf.extend_from_slice(&(payload_len as u32).to_le_bytes());

    let push_f32 = |dst: &mut Vec<u8>, v: f32| -> Result<(), PersistError> {
        if !v.is_finite() {
            return Err(PersistError::NonFiniteState);
        }
        dst.extend_from_slice(&sanitize(v).to_bits().to_le_bytes());
        Ok(())
    };

    buf.extend_from_slice(&(na as u32).to_le_bytes());
    buf.extend_from_slice(&ticks.to_le_bytes());
    push_f32(&mut buf, r_peak)?;
    push_f32(&mut buf, r_bar)?;
    for &ch in dend_ch {
        buf.extend_from_slice(&ch.to_le_bytes());
    }
    buf.extend_from_slice(dend_env);
    for &kv in dend_key {
        push_f32(&mut buf, kv)?;
    }
    for &wv in w_cons {
        push_f32(&mut buf, wv)?;
    }
    for &vv in v_cons {
        push_f32(&mut buf, vv)?;
    }
    for &qv in q_cons {
        push_f32(&mut buf, qv)?;
    }
    for &jv in fwd_j_cons {
        push_f32(&mut buf, jv)?;
    }
    let crc = crc32c(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok((buf, crc))
}

pub fn decode_column_slab(bytes: &[u8]) -> Result<DecodedColumnSlab, PersistError> {
    if bytes.len() < 24 {
        return Err(PersistError::TruncatedInput);
    }
    let magic = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let ver = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if magic != COL_SLAB_MAGIC || ver != CORTICAL_SLAB_VERSION {
        return Err(PersistError::BadMagicOrVersion);
    }
    let col_id = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let payload_len = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let total_expected = 24usize
        .checked_add(payload_len)
        .ok_or(PersistError::OversizedInput)?;
    if bytes.len() < total_expected {
        return Err(PersistError::TruncatedInput);
    }
    if bytes.len() > total_expected {
        return Err(PersistError::InvalidTopology);
    }
    let crc_pos = total_expected - 4;
    let stored_crc = u32::from_le_bytes([
        bytes[crc_pos],
        bytes[crc_pos + 1],
        bytes[crc_pos + 2],
        bytes[crc_pos + 3],
    ]);
    let actual_crc = crc32c(&bytes[..crc_pos]);
    if actual_crc != stored_crc {
        return Err(PersistError::ChecksumMismatch);
    }

    let mut pos = 20usize;
    let rd_u32 = |b: &[u8], at: &mut usize| -> Result<u32, PersistError> {
        let end = (*at).checked_add(4).ok_or(PersistError::TruncatedInput)?;
        if end > crc_pos {
            return Err(PersistError::TruncatedInput);
        }
        let v = u32::from_le_bytes([b[*at], b[*at + 1], b[*at + 2], b[*at + 3]]);
        *at = end;
        Ok(v)
    };
    let rd_f32 = |b: &[u8], at: &mut usize| -> Result<f32, PersistError> {
        let bits = rd_u32(b, at)?;
        let v = f32::from_bits(bits);
        if !v.is_finite() {
            return Err(PersistError::NonFiniteState);
        }
        Ok(sanitize(v))
    };

    let na = rd_u32(bytes, &mut pos)? as usize;
    if !(MIN_STRIATAL_ACT..=4_096).contains(&na) {
        return Err(PersistError::InvalidTopology);
    }
    let w_len = na * STRIATAL_CUES;
    let fwd_len = STRIATAL_CUES * na;
    let exact_payload = 16
        + STRIATAL_PORES * 4
        + STRIATAL_PORES
        + STRIATAL_PORES * 4
        + w_len * 4
        + STRIATAL_FEATS * 4
        + STRIATAL_CUES * 4
        + fwd_len * 4;
    if payload_len != exact_payload {
        return Err(PersistError::InvalidTopology);
    }

    let ticks = rd_u32(bytes, &mut pos)?;
    let r_peak = rd_f32(bytes, &mut pos)?;
    let r_bar = rd_f32(bytes, &mut pos)?;
    let mut dend_ch = [u32::MAX; STRIATAL_PORES];
    for dst in &mut dend_ch {
        *dst = rd_u32(bytes, &mut pos)?;
    }
    let mut dend_env = [0u8; STRIATAL_PORES];
    for dst in &mut dend_env {
        let b = bytes[pos];
        pos += 1;
        if b > 3 {
            return Err(PersistError::InvalidTopology);
        }
        *dst = b;
    }
    let mut dend_key = [0.0_f32; STRIATAL_PORES];
    for dst in &mut dend_key {
        *dst = rd_f32(bytes, &mut pos)?;
    }
    let mut w_cons = vec![0.0_f32; w_len];
    for dst in &mut w_cons {
        *dst = rd_f32(bytes, &mut pos)?;
    }
    let mut v_cons = [0.0_f32; STRIATAL_FEATS];
    for dst in &mut v_cons {
        *dst = rd_f32(bytes, &mut pos)?;
    }
    let mut q_cons = [0.0_f32; STRIATAL_CUES];
    for dst in &mut q_cons {
        *dst = rd_f32(bytes, &mut pos)?;
    }
    let mut fwd_j_cons = vec![0.0_f32; fwd_len];
    for dst in &mut fwd_j_cons {
        *dst = rd_f32(bytes, &mut pos)?;
    }
    if pos != crc_pos {
        return Err(PersistError::InvalidTopology);
    }

    Ok(DecodedColumnSlab {
        col_id,
        na,
        ticks,
        r_peak,
        r_bar,
        dend_ch,
        dend_env,
        dend_key,
        w_cons,
        v_cons,
        q_cons,
        fwd_j_cons,
    })
}

#[inline]
fn try_prony_stride(
    history: &[f32; 16],
    h: usize,
    stride: usize,
    n_win: usize,
    max_c: f32,
) -> Option<f32> {
    if n_win < 2 {
        return None;
    }
    let mut num_c = 0.0_f32;
    let mut den_c = 0.0_f32;
    for k in 0..n_win {
        let y0 = history[(h + 15 - k) & 15];
        let y1 = history[(h + 15 - k - stride) & 15];
        let y2 = history[(h + 15 - k - 2 * stride) & 15];
        let y3 = history[(h + 15 - k - 3 * stride) & 15];
        let d0 = y0 - y1;
        let d1 = y1 - y2;
        let d2 = y2 - y3;
        num_c += (d0 + d2) * d1;
        den_c += 2.0 * d1 * d1;
    }
    if den_c <= 1.0e-7 {
        return None;
    }
    let c = num_c / den_c;
    if !(0.25..=max_c).contains(&c) {
        return None;
    }
    let mut fit_err_sq = 0.0_f32;
    let mut sum_mu = 0.0_f32;
    let mut mus = [0.0_f32; 16];
    let n_cap = n_win.min(16);
    let denom_mu = 2.0 * (1.0 - c);
    for (k, mu_slot) in mus.iter_mut().enumerate().take(n_cap) {
        let y0 = history[(h + 15 - k) & 15];
        let y1 = history[(h + 15 - k - stride) & 15];
        let y2 = history[(h + 15 - k - 2 * stride) & 15];
        let y3 = history[(h + 15 - k - 3 * stride) & 15];
        let d0 = y0 - y1;
        let d1 = y1 - y2;
        let d2 = y2 - y3;
        let r = d0 + d2 - 2.0 * c * d1;
        fit_err_sq += r * r;
        let mk = (y0 + y2 - 2.0 * c * y1) / denom_mu;
        *mu_slot = mk;
        sum_mu += mk;
    }
    if fit_err_sq > 0.0015 * den_c {
        return None;
    }
    let mu = sum_mu / (n_cap as f32);
    for &mk in &mus[..n_cap] {
        if (mk - mu).abs() > 0.012 {
            return None;
        }
    }
    if (-1.05..=1.05).contains(&mu) {
        Some(sanitize_clamp(mu, -1.0, 1.0))
    } else {
        None
    }
}

#[inline]
fn try_post_jump_prony(history: &[f32; 16], h: usize, m_jump: usize) -> Option<f32> {
    let post_len = m_jump + 1;
    if post_len < 6 {
        return None;
    }
    let mut sub_sum = 0.0_f32;
    let mut sub_max = 0.0_f32;
    for m in 0..m_jump {
        let dy = history[(h + 15 - m) & 15] - history[(h + 14 - m) & 15];
        let sq = dy * dy;
        sub_sum += sq;
        if sq > sub_max {
            sub_max = sq;
        }
    }
    if sub_sum <= 1.0e-8 || sub_max > 0.55 * sub_sum {
        return None;
    }
    if post_len >= 8
        && let Some(mu) = try_prony_stride(history, h, 2, (post_len - 6).min(3), 0.9985)
    {
        return Some(mu);
    }
    try_prony_stride(history, h, 1, (post_len - 3).min(3), 0.9990)
}

#[inline]
#[must_use]
pub fn harmonic_dc_from_history(history: &[f32; 16], hist_head: u8, age_ticks: u32) -> (f32, bool) {
    let h = (hist_head as usize) & 15;
    let x_now = sanitize_clamp(history[(h + 15) & 15], -1.0, 1.0);
    if age_ticks < 8 {
        return (x_now, false);
    }
    let n_steps = ((age_ticks as usize) - 1).min(15);
    let mut dy = [0.0_f32; 15];
    let mut sum_step_sq = 0.0_f32;
    let mut max_step_sq = 0.0_f32;
    let mut m_jump = 0usize;
    for (m, dy_slot) in dy.iter_mut().enumerate().take(n_steps) {
        let d = history[(h + 15 - m) & 15] - history[(h + 14 - m) & 15];
        *dy_slot = d;
        let sq = d * d;
        sum_step_sq += sq;
        if sq > max_step_sq {
            max_step_sq = sq;
            m_jump = m;
        }
    }
    if sum_step_sq <= 1.0e-8 {
        return (x_now, false);
    }
    let cap_frac = if n_steps >= 12 { 0.25 } else { 0.50 };
    let mut has_jump = max_step_sq > cap_frac * sum_step_sq;
    if n_steps >= 12 {
        let n_d2 = n_steps - 1;
        let mut s2 = 0.0_f32;
        let mut max_d2_sq = 0.0_f32;
        let mut max_d2_m = 0usize;
        for m in 0..n_d2 {
            let d2 = dy[m] - dy[m + 1];
            let sq = d2 * d2;
            s2 += sq;
            if sq > max_d2_sq {
                max_d2_sq = sq;
                max_d2_m = m;
            }
        }
        let n_d3 = n_steps - 2;
        let mut s3 = 0.0_f32;
        let mut max_d3_sq = 0.0_f32;
        let mut max_d3_m = 0usize;
        for m in 0..n_d3 {
            let d3 = dy[m] - 2.0 * dy[m + 1] + dy[m + 2];
            let sq = d3 * d3;
            s3 += sq;
            if sq > max_d3_sq {
                max_d3_sq = sq;
                max_d3_m = m;
            }
        }
        if s3 > 1.0e-11 && max_d3_sq > 0.32 * s3 {
            has_jump = true;
            m_jump = max_d3_m + 1;
        } else if s2 > 1.0e-10 && max_d2_sq > 0.28 * s2 {
            has_jump = true;
            if ! (max_step_sq > cap_frac * sum_step_sq) {
                m_jump = max_d2_m;
            }
        }
    }
    if has_jump {
        if let Some(mu) = try_post_jump_prony(history, h, m_jump) {
            return (mu, true);
        }
        return (x_now, false);
    }
    let mu2 = try_prony_stride(history, h, 2, 2, 0.9985);
    if age_ticks >= 16
        && let Some(mu4) = try_prony_stride(history, h, 4, 4, 0.994)
    {
        if let Some(m2) = mu2
            && (mu4 - m2).abs() > 0.015
        {
            return (m2, true);
        }
        return (mu4, true);
    }
    if let Some(m2) = mu2 {
        return (m2, true);
    }
    if let Some(mu1) = try_prony_stride(history, h, 1, 3, 0.9990) {
        return (mu1, true);
    }
    if m_jump >= 5
        && let Some(mu_pj) = try_post_jump_prony(history, h, m_jump)
    {
        return (mu_pj, true);
    }
    (x_now, false)
}

#[derive(Clone, Debug, PartialEq)]
pub struct StriatalOfc {
    /// Actuator columns currently allocated in every actuator-strided table.
    pub n_act: usize,
    pub num_ctx: u8,
    pub active_ctx: u8,
    pub impact_refrac: u8,
    pub ctx_has_cons: [u8; OFC_CONTEXTS],
    pub ctx_ticks: [u32; OFC_CONTEXTS],
    pub ctx_r_peak: [f32; OFC_CONTEXTS],
    pub r_bar_ctx: [f32; OFC_CONTEXTS],
    pub total_cols: u32,
    pub dend_mismatch_streak: u8,
    pub hot_col_id: [u32; OFC_CONTEXTS],
    pub hot_last_use: [u64; OFC_CONTEXTS],
    pub hot_dend_ch: [[u32; STRIATAL_PORES]; OFC_CONTEXTS],
    pub hot_dend_env: [[u8; STRIATAL_PORES]; OFC_CONTEXTS],
    pub hot_dend_key: [[f32; STRIATAL_PORES]; OFC_CONTEXTS],
    pub paged_cols: Vec<PagedColumnEntry>,
    pub pore_ch: [u32; STRIATAL_PORES],
    pub pore_age: [u32; STRIATAL_PORES],
    pub pore_mean: [f32; STRIATAL_PORES],
    pub pore_var_raw: [f32; STRIATAL_PORES],
    pub pore_var_exaff: [f32; STRIATAL_PORES],
    pub pore_carr_mean: [f32; STRIATAL_PORES],
    pub pore_carr_var: [f32; STRIATAL_PORES],
    pub pore_prev_x: [f32; STRIATAL_PORES],
    pub pore_var_dx: [f32; STRIATAL_PORES],
    pub pore_orb_err: [f32; STRIATAL_PORES],
    pub canc_g: Vec<f32>,
    pub w_birth: Vec<f32>,
    pub w_ctx: Vec<f32>,
    pub w_cons: Vec<f32>,
    pub v_ctx: Vec<f32>,
    pub v_cons: Vec<f32>,
    pub q_ctx: Vec<f32>,
    pub q_cons: Vec<f32>,
    pub fwd_b: [f32; STRIATAL_CUES],
    pub fwd_alpha: [f32; STRIATAL_CUES],
    pub fwd_j: Vec<f32>,
    pub fwd_j_ctx: Vec<f32>,
    pub fwd_j_cons: Vec<f32>,
    pub fwd_var_null: [f32; STRIATAL_CUES],
    pub fwd_var_act: [f32; STRIATAL_CUES],
    pub fwd_num: [f32; OFC_CONTEXTS],
    pub fwd_den: [f32; OFC_CONTEXTS],
    pub act_mean: Vec<f32>,
    pub ctx_cooldown: u32,
    pub ctx_out_events: u32,
    pub ctx_out_err: [f32; OFC_CONTEXTS],
    pub r_fast_ema: f32,
    pub r_med200: f32,
    pub valence_flip_ema: f32,
    pub da_ring: Vec<f32>,
    pub elig_actor: Vec<f32>,
    pub elig_critic: [f32; STRIATAL_FEATS],
    pub feat: [f32; STRIATAL_FEATS],
    pub prev_critic_feat: [f32; STRIATAL_FEATS],
    pub prev_feat: [f32; STRIATAL_CUES],
    pub prev_df: [f32; STRIATAL_CUES],
    pub prev_act: Vec<f32>,
    pub noise: Vec<f32>,
    pub v_prev: f32,
    pub has_prev: bool,
    pub tick: u64,
    pub rng: Rng64,
}

impl StriatalOfc {
    /// `(actuator columns, tap ring width, actor row-block length, forward-model length)`.
    #[inline]
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        let na = self.n_act;
        (
            na,
            STRIATAL_TAPS * na,
            na * STRIATAL_CUES,
            STRIATAL_CUES * na,
        )
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let (na, tap_dim, w_len, fwd_len) = self.dims();
        let paged_meta = self.paged_cols.len() * (16 + 9 * STRIATAL_PORES);
        3 + OFC_CONTEXTS
            + 6 * 4 * OFC_CONTEXTS
            + 12 * OFC_CONTEXTS
            + 9 * STRIATAL_PORES * OFC_CONTEXTS
            + 10 * 4 * STRIATAL_PORES
            + 4 * STRIATAL_PORES * tap_dim
            + 4 * w_len
            + 2 * 4 * OFC_CONTEXTS * w_len
            + 2 * 4 * OFC_CONTEXTS * STRIATAL_FEATS
            + 2 * 4 * OFC_CONTEXTS * STRIATAL_CUES
            + 4 * 4 * STRIATAL_CUES
            + (1 + 2 * OFC_CONTEXTS) * 4 * fwd_len
            + 3 * 4 * na
            + 4 * tap_dim
            + 4 * w_len
            + 3 * 4 * STRIATAL_FEATS
            + 2 * 4 * STRIATAL_CUES
            + paged_meta
            + 80
    }

    /// Grows every actuator-strided table to `num_actuators` columns. Existing columns keep
    /// their learned values exactly; new actor rows are drawn from the circuit's own birth
    /// distribution (continuing its RNG stream), new forward/cancellation columns start at zero.
    pub fn ensure_actuators(&mut self, num_actuators: usize) {
        let (old_na, _, old_w_len, old_fwd_len) = self.dims();
        let new_na = num_actuators.max(old_na);
        if new_na == old_na {
            return;
        }
        let scale = 0.10 / (STRIATAL_FEATS as f32).sqrt();
        let extra_rows = (new_na - old_na) * STRIATAL_CUES;
        let mut new_birth = Vec::with_capacity(extra_rows);
        for _ in 0..extra_rows {
            new_birth.push(sanitize(scale * self.rng.next_f32_signed()));
        }
        self.w_birth.extend_from_slice(&new_birth);
        let new_w_len = new_na * STRIATAL_CUES;
        let mut grown_ctx = Vec::with_capacity(OFC_CONTEXTS * new_w_len);
        let mut grown_cons = Vec::with_capacity(OFC_CONTEXTS * new_w_len);
        let zeros_extra = vec![0.0_f32; extra_rows];
        for c in 0..OFC_CONTEXTS {
            grown_ctx.extend_from_slice(&self.w_ctx[c * old_w_len..(c + 1) * old_w_len]);
            grown_ctx.extend_from_slice(&new_birth);
            grown_cons.extend_from_slice(&self.w_cons[c * old_w_len..(c + 1) * old_w_len]);
            if self.ctx_has_cons[c] != 0 {
                grown_cons.extend_from_slice(&new_birth);
            } else {
                grown_cons.extend_from_slice(&zeros_extra);
            }
        }
        self.w_ctx = grown_ctx;
        self.w_cons = grown_cons;
        self.elig_actor.resize(new_w_len, 0.0);
        let restride = |src: &[f32], rows: usize, old_c: usize, new_c: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; rows * new_c];
            for r in 0..rows {
                out[r * new_c..r * new_c + old_c].copy_from_slice(&src[r * old_c..(r + 1) * old_c]);
            }
            out
        };
        self.fwd_j = restride(&self.fwd_j, STRIATAL_CUES, old_na, new_na);
        self.fwd_j_ctx = restride(
            &self.fwd_j_ctx,
            OFC_CONTEXTS * STRIATAL_CUES,
            old_na,
            new_na,
        );
        self.fwd_j_cons = restride(
            &self.fwd_j_cons,
            OFC_CONTEXTS * STRIATAL_CUES,
            old_na,
            new_na,
        );
        self.canc_g = restride(&self.canc_g, STRIATAL_PORES * STRIATAL_TAPS, old_na, new_na);
        self.da_ring = restride(&self.da_ring, STRIATAL_TAPS, old_na, new_na);
        self.act_mean.resize(new_na, 0.0);
        self.prev_act.resize(new_na, 0.0);
        self.noise.resize(new_na, 0.0);
        debug_assert_eq!(old_fwd_len, STRIATAL_CUES * old_na);
        self.n_act = new_na;
    }

    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let na = MIN_STRIATAL_ACT;
        let (tap_dim, w_len, fwd_len) =
            (STRIATAL_TAPS * na, na * STRIATAL_CUES, STRIATAL_CUES * na);
        let mut birth_rng = Rng64::from_seed(seed ^ STRIATAL_BIRTH_SALT);
        let scale = 0.10 / (STRIATAL_FEATS as f32).sqrt();
        let w_birth: Vec<f32> = (0..w_len)
            .map(|_| sanitize(scale * birth_rng.next_f32_signed()))
            .collect();
        let mut w_ctx = Vec::with_capacity(OFC_CONTEXTS * w_len);
        for _ in 0..OFC_CONTEXTS {
            w_ctx.extend_from_slice(&w_birth);
        }
        let w_cons = vec![0.0; OFC_CONTEXTS * w_len];
        Self {
            n_act: na,
            num_ctx: 1,
            active_ctx: 0,
            impact_refrac: 0,
            ctx_has_cons: [0; OFC_CONTEXTS],
            ctx_ticks: [0; OFC_CONTEXTS],
            ctx_r_peak: [0.0; OFC_CONTEXTS],
            r_bar_ctx: [0.0; OFC_CONTEXTS],
            total_cols: 1,
            dend_mismatch_streak: 0,
            hot_col_id: [0, 1, 2, 3],
            hot_last_use: [0; OFC_CONTEXTS],
            hot_dend_ch: [[u32::MAX; STRIATAL_PORES]; OFC_CONTEXTS],
            hot_dend_env: [[0; STRIATAL_PORES]; OFC_CONTEXTS],
            hot_dend_key: [[0.0; STRIATAL_PORES]; OFC_CONTEXTS],
            paged_cols: Vec::new(),
            pore_ch: [u32::MAX; STRIATAL_PORES],
            pore_age: [0; STRIATAL_PORES],
            pore_mean: [0.0; STRIATAL_PORES],
            pore_var_raw: [1.0e-4; STRIATAL_PORES],
            pore_var_exaff: [1.0e-4; STRIATAL_PORES],
            pore_carr_mean: [0.0; STRIATAL_PORES],
            pore_carr_var: [1.0e-4; STRIATAL_PORES],
            pore_prev_x: [0.0; STRIATAL_PORES],
            pore_var_dx: [1.0e-5; STRIATAL_PORES],
            pore_orb_err: [1.0; STRIATAL_PORES],
            canc_g: vec![0.0; STRIATAL_PORES * tap_dim],
            w_birth,
            w_ctx,
            w_cons,
            v_ctx: vec![0.0; OFC_CONTEXTS * STRIATAL_FEATS],
            v_cons: vec![0.0; OFC_CONTEXTS * STRIATAL_FEATS],
            q_ctx: vec![0.0; OFC_CONTEXTS * STRIATAL_CUES],
            q_cons: vec![0.0; OFC_CONTEXTS * STRIATAL_CUES],
            fwd_b: [0.0; STRIATAL_CUES],
            fwd_alpha: [0.0; STRIATAL_CUES],
            fwd_j: vec![0.0; fwd_len],
            fwd_j_ctx: vec![0.0; OFC_CONTEXTS * fwd_len],
            fwd_j_cons: vec![0.0; OFC_CONTEXTS * fwd_len],
            fwd_var_null: [1.0e-6; STRIATAL_CUES],
            fwd_var_act: [1.0e-6; STRIATAL_CUES],
            fwd_num: [0.0; OFC_CONTEXTS],
            fwd_den: [0.0; OFC_CONTEXTS],
            act_mean: vec![0.0; na],
            ctx_cooldown: 0,
            ctx_out_events: 0,
            ctx_out_err: [0.0; OFC_CONTEXTS],
            r_fast_ema: 0.0,
            r_med200: 0.0,
            valence_flip_ema: 0.0,
            da_ring: vec![0.0; tap_dim],
            elig_actor: vec![0.0; w_len],
            elig_critic: [0.0; STRIATAL_FEATS],
            feat: [0.0; STRIATAL_FEATS],
            prev_critic_feat: [0.0; STRIATAL_FEATS],
            prev_feat: [0.0; STRIATAL_CUES],
            prev_df: [0.0; STRIATAL_CUES],
            prev_act: vec![0.0; na],
            noise: vec![0.0; na],
            v_prev: 0.0,
            has_prev: false,
            tick: 0,
            rng: birth_rng,
        }
    }

    pub fn reset_learned_keep_birth(&mut self) {
        let (na, tap_dim, w_len, _) = self.dims();
        for c in 0..OFC_CONTEXTS {
            let off = c * w_len;
            self.w_ctx[off..off + w_len].copy_from_slice(&self.w_birth);
        }
        self.w_cons.fill(0.0);
        self.v_ctx.fill(0.0);
        self.v_cons.fill(0.0);
        self.q_ctx.fill(0.0);
        self.q_cons.fill(0.0);
        self.fwd_j.fill(0.0);
        self.fwd_j_ctx.fill(0.0);
        self.fwd_j_cons.fill(0.0);
        self.num_ctx = 1;
        self.active_ctx = 0;
        self.impact_refrac = 0;
        self.ctx_has_cons = [0; OFC_CONTEXTS];
        self.ctx_ticks = [0; OFC_CONTEXTS];
        self.ctx_r_peak = [0.0; OFC_CONTEXTS];
        self.r_bar_ctx = [0.0; OFC_CONTEXTS];
        self.total_cols = 1;
        self.dend_mismatch_streak = 0;
        self.hot_col_id = [0, 1, 2, 3];
        self.hot_last_use = [0; OFC_CONTEXTS];
        self.hot_dend_ch = [[u32::MAX; STRIATAL_PORES]; OFC_CONTEXTS];
        self.hot_dend_env = [[0; STRIATAL_PORES]; OFC_CONTEXTS];
        self.hot_dend_key = [[0.0; STRIATAL_PORES]; OFC_CONTEXTS];
        self.paged_cols.clear();
        self.pore_ch = [u32::MAX; STRIATAL_PORES];
        self.pore_age = [0; STRIATAL_PORES];
        self.pore_mean = [0.0; STRIATAL_PORES];
        self.pore_var_raw = [1.0e-4; STRIATAL_PORES];
        self.pore_var_exaff = [1.0e-4; STRIATAL_PORES];
        self.pore_carr_mean = [0.0; STRIATAL_PORES];
        self.pore_carr_var = [1.0e-4; STRIATAL_PORES];
        self.pore_prev_x = [0.0; STRIATAL_PORES];
        self.pore_var_dx = [1.0e-5; STRIATAL_PORES];
        self.pore_orb_err = [1.0; STRIATAL_PORES];
        self.canc_g.fill(0.0);
        self.fwd_b = [0.0; STRIATAL_CUES];
        self.fwd_alpha = [0.0; STRIATAL_CUES];
        self.fwd_var_null = [1.0e-6; STRIATAL_CUES];
        self.fwd_var_act = [1.0e-6; STRIATAL_CUES];
        self.fwd_num = [0.0; OFC_CONTEXTS];
        self.fwd_den = [0.0; OFC_CONTEXTS];
        self.act_mean = vec![0.0; na];
        self.ctx_cooldown = 0;
        self.ctx_out_events = 0;
        self.ctx_out_err = [0.0; OFC_CONTEXTS];
        self.r_fast_ema = 0.0;
        self.r_med200 = 0.0;
        self.valence_flip_ema = 0.0;
        self.da_ring = vec![0.0; tap_dim];
        self.elig_actor.fill(0.0);
        self.elig_critic = [0.0; STRIATAL_FEATS];
        self.feat = [0.0; STRIATAL_FEATS];
        self.prev_critic_feat = [0.0; STRIATAL_FEATS];
        self.prev_feat = [0.0; STRIATAL_CUES];
        self.prev_df = [0.0; STRIATAL_CUES];
        self.prev_act = vec![0.0; na];
        self.noise = vec![0.0; na];
        self.v_prev = 0.0;
        self.has_prev = false;
        self.tick = 0;
    }

    #[inline]
    fn gaussian(&mut self) -> f32 {
        let s = self.rng.next_f32_unit()
            + self.rng.next_f32_unit()
            + self.rng.next_f32_unit()
            + self.rng.next_f32_unit();
        sanitize((s - 2.0) * 1.732_050_8)
    }

    fn reset_pore_slot(&mut self, p: usize, new_ch: u32) {
        let (na, tap_dim, w_len, fwd_len) = self.dims();
        self.pore_ch[p] = new_ch;
        self.pore_age[p] = 0;
        self.pore_mean[p] = 0.0;
        self.pore_var_raw[p] = 1.0e-4;
        self.pore_var_exaff[p] = 1.0e-4;
        self.pore_carr_mean[p] = 0.0;
        self.pore_carr_var[p] = 1.0e-4;
        self.pore_prev_x[p] = 0.0;
        self.pore_var_dx[p] = 1.0e-5;
        self.pore_orb_err[p] = 1.0;
        let g_off = p * tap_dim;
        self.canc_g[g_off..g_off + tap_dim].fill(0.0);
        for st in 0..STRIATAL_STRATA {
            let fi = st * STRIATAL_PORES + p;
            for c in 0..OFC_CONTEXTS {
                let w_off = c * w_len;
                for m in 0..na {
                    let row = m * STRIATAL_CUES;
                    let wb = self.w_birth[row + fi];
                    self.w_ctx[w_off + row + fi] = wb;
                }
                let v_off = c * STRIATAL_FEATS;
                self.v_ctx[v_off + fi] = 0.0;
                let q_off = c * STRIATAL_CUES;
                self.q_ctx[q_off + fi] = 0.0;
                let j_off = c * fwd_len + fi * na;
                self.fwd_j_ctx[j_off..j_off + na].fill(0.0);
            }
            for m in 0..na {
                self.elig_actor[m * STRIATAL_CUES + fi] = 0.0;
            }
            self.elig_critic[fi] = 0.0;
            self.prev_feat[fi] = 0.0;
            self.prev_df[fi] = 0.0;
            self.fwd_b[fi] = 0.0;
            self.fwd_alpha[fi] = 0.0;
            self.fwd_var_null[fi] = 1.0e-6;
            self.fwd_var_act[fi] = 1.0e-6;
            let j_off = fi * na;
            self.fwd_j[j_off..j_off + na].fill(0.0);
        }
    }

    #[inline]
    #[must_use]
    pub fn is_quad_carrier_slot(&self, p: usize) -> bool {
        if p >= STRIATAL_PORES || self.pore_ch[p] == u32::MAX || self.pore_age[p] < 36 {
            return false;
        }
        let v_c = self.pore_carr_var[p];
        if v_c < 1.2e-3 {
            return false;
        }
        let rough = self.pore_var_dx[p] / v_c.max(1.0e-8);
        rough > 0.001 && rough < 0.025 && self.pore_orb_err[p] < 0.30
    }

    #[must_use]
    pub fn has_dual_population_carrier_bank(&self) -> bool {
        let mut n_carr = 0usize;
        let mut n_non = 0usize;
        for p in 0..STRIATAL_PORES {
            if self.pore_ch[p] == u32::MAX || self.pore_age[p] < 24 {
                continue;
            }
            if self.is_quad_carrier_slot(p) {
                n_carr += 1;
            } else if self.pore_var_raw[p] >= 8.0e-5 && self.pore_orb_err[p] >= 0.34 {
                n_non += 1;
            }
        }
        n_carr >= 8 && n_non >= 10
    }

    #[must_use]
    pub fn consolidated_role_for_channel(&self, ch: u32) -> u8 {
        if ch == u32::MAX {
            return 0;
        }
        let num_c = (self.num_ctx as usize).min(OFC_CONTEXTS);
        let mut role = 0u8;
        for c in 0..num_c {
            if self.ctx_has_cons[c] == 0 {
                continue;
            }
            for p in 0..STRIATAL_PORES {
                if self.hot_dend_ch[c][p] == ch {
                    role |= self.hot_dend_env[c][p] & 3;
                }
            }
        }
        for entry in &self.paged_cols {
            for p in 0..STRIATAL_PORES {
                if entry.dend_ch[p] == ch {
                    role |= entry.dend_env[p] & 3;
                }
            }
        }
        role
    }

    #[inline]
    fn critic_value(&self, ctx: usize, f: &[f32; STRIATAL_FEATS]) -> f32 {
        let off = ctx * STRIATAL_FEATS;
        let mut sum = 0.0_f32;
        for (a, b) in self.v_ctx[off..off + STRIATAL_FEATS].iter().zip(f.iter()) {
            sum += a * b;
        }
        sanitize(sum)
    }

    #[inline]
    fn outcome_pred_live(&self, ctx: usize, f: &[f32; STRIATAL_FEATS]) -> f32 {
        let off = ctx * STRIATAL_CUES;
        let mut sum = 0.0_f32;
        for (a, b) in self.q_ctx[off..off + STRIATAL_CUES]
            .iter()
            .zip(f[..STRIATAL_CUES].iter())
        {
            sum += a * b;
        }
        sanitize(sum)
    }

    #[inline]
    fn outcome_pred_cons(&self, ctx: usize, f: &[f32; STRIATAL_FEATS]) -> f32 {
        let off = ctx * STRIATAL_CUES;
        let mut sum = 0.0_f32;
        for (a, b) in self.q_cons[off..off + STRIATAL_CUES]
            .iter()
            .zip(f[..STRIATAL_CUES].iter())
        {
            sum += a * b;
        }
        sanitize(sum)
    }

    fn consolidate_active_context_with_dend(
        &mut self,
        tonic_valid: &[bool; STRIATAL_PORES],
        tonic_now: &[f32; STRIATAL_PORES],
        apical_carrier: &[bool; STRIATAL_PORES],
        feat_active: &[bool; STRIATAL_PORES],
    ) {
        let (_, _, w_len, fwd_len) = self.dims();
        let c = self.active_ctx as usize;
        let prev_had = self.ctx_has_cons[c] != 0;
        self.w_cons[c * w_len..(c + 1) * w_len]
            .copy_from_slice(&self.w_ctx[c * w_len..(c + 1) * w_len]);
        self.v_cons[c * STRIATAL_FEATS..(c + 1) * STRIATAL_FEATS]
            .copy_from_slice(&self.v_ctx[c * STRIATAL_FEATS..(c + 1) * STRIATAL_FEATS]);
        self.q_cons[c * STRIATAL_CUES..(c + 1) * STRIATAL_CUES]
            .copy_from_slice(&self.q_ctx[c * STRIATAL_CUES..(c + 1) * STRIATAL_CUES]);
        self.fwd_j_cons[c * fwd_len..(c + 1) * fwd_len]
            .copy_from_slice(&self.fwd_j_ctx[c * fwd_len..(c + 1) * fwd_len]);
        let mut env_pick = [false; STRIATAL_PORES];
        let mut env_cnt = 0usize;
        for p in 0..STRIATAL_PORES {
            if tonic_valid[p]
                && self.pore_ch[p] != u32::MAX
                && (self.consolidated_role_for_channel(self.pore_ch[p]) & 1) != 0
                && env_cnt < 16
            {
                env_pick[p] = true;
                env_cnt += 1;
            }
        }
        for p in 0..STRIATAL_PORES {
            if !env_pick[p]
                && tonic_valid[p]
                && self.pore_ch[p] != u32::MAX
                && self.pore_age[p] >= 36
                && apical_carrier[p]
                && env_cnt < 16
            {
                env_pick[p] = true;
                env_cnt += 1;
            }
        }
        for p in 0..STRIATAL_PORES {
            if !tonic_valid[p] || self.pore_ch[p] == u32::MAX {
                self.hot_dend_ch[c][p] = u32::MAX;
                self.hot_dend_env[c][p] = 0;
                self.hot_dend_key[c][p] = 0.0;
                continue;
            }
            let is_env = env_pick[p];
            let prev_same_ch = prev_had && self.hot_dend_ch[c][p] == self.pore_ch[p];
            let dc_est = sanitize_clamp(tonic_now[p], -1.0, 1.0);
            self.hot_dend_ch[c][p] = self.pore_ch[p];
            self.hot_dend_env[c][p] =
                u8::from(is_env) | (u8::from(!is_env && feat_active[p]) << 1);
            self.hot_dend_key[c][p] = if prev_same_ch {
                sanitize_clamp(0.50 * self.hot_dend_key[c][p] + 0.50 * dc_est, -1.0, 1.0)
            } else {
                dc_est
            };
        }
        self.ctx_has_cons[c] = 1;
        self.hot_last_use[c] = self.tick;
    }

    pub fn encode_hot_slot_slab(&self, slot: usize) -> Result<(Vec<u8>, u32), PersistError> {
        let (na, _, w_len, fwd_len) = self.dims();
        if slot >= OFC_CONTEXTS || self.ctx_has_cons[slot] == 0 {
            return Err(PersistError::InvalidTopology);
        }
        encode_column_slab(
            self.hot_col_id[slot],
            na,
            self.ctx_ticks[slot],
            self.ctx_r_peak[slot],
            self.r_bar_ctx[slot],
            &self.hot_dend_ch[slot],
            &self.hot_dend_env[slot],
            &self.hot_dend_key[slot],
            &self.w_cons[slot * w_len..(slot + 1) * w_len],
            &self.v_cons[slot * STRIATAL_FEATS..(slot + 1) * STRIATAL_FEATS],
            &self.q_cons[slot * STRIATAL_CUES..(slot + 1) * STRIATAL_CUES],
            &self.fwd_j_cons[slot * fwd_len..(slot + 1) * fwd_len],
        )
    }

    fn evict_hot_slot_to_paged(&mut self, slot: usize) -> Result<(), PersistError> {
        if slot >= OFC_CONTEXTS || self.ctx_has_cons[slot] == 0 {
            return Ok(());
        }
        let (slab_bytes, slab_crc32c) = self.encode_hot_slot_slab(slot)?;
        let col_id = self.hot_col_id[slot];
        let entry = PagedColumnEntry {
            col_id,
            r_peak: self.ctx_r_peak[slot],
            dend_ch: self.hot_dend_ch[slot],
            dend_env: self.hot_dend_env[slot],
            dend_key: self.hot_dend_key[slot],
            slab_crc32c,
            slab_bytes,
        };
        if let Some(pos) = self.paged_cols.iter().position(|e| e.col_id == col_id) {
            self.paged_cols[pos] = entry;
        } else {
            let ins = self.paged_cols.partition_point(|e| e.col_id < col_id);
            self.paged_cols.insert(ins, entry);
        }
        Ok(())
    }

    #[must_use]
    pub fn remap_slab_for_channels(
        &self,
        dec: &DecodedColumnSlab,
        cur_ch: &[u32; STRIATAL_PORES],
    ) -> DecodedColumnSlab {
        let (na, _, w_len, fwd_len) = self.dims();
        let copy_na = dec.na.min(na);
        let mut w_out = vec![0.0_f32; w_len];
        let mut v_out = [0.0_f32; STRIATAL_FEATS];
        v_out[STRIATAL_CUES] = dec.v_cons[STRIATAL_CUES];
        let mut q_out = [0.0_f32; STRIATAL_CUES];
        let mut j_out = vec![0.0_f32; fwd_len];
        let mut dend_ch_out = [u32::MAX; STRIATAL_PORES];
        let mut dend_env_out = [0u8; STRIATAL_PORES];
        let mut dend_key_out = [0.0_f32; STRIATAL_PORES];

        for p_dst in 0..STRIATAL_PORES {
            let ch = cur_ch[p_dst];
            if ch == u32::MAX {
                continue;
            }
            let src_opt = if dec.dend_ch[p_dst] == ch {
                Some(p_dst)
            } else {
                (0..STRIATAL_PORES).find(|&q| dec.dend_ch[q] == ch)
            };
            if let Some(p_src) = src_opt {
                dend_ch_out[p_dst] = ch;
                dend_env_out[p_dst] = dec.dend_env[p_src];
                dend_key_out[p_dst] = dec.dend_key[p_src];
                for st in 0..STRIATAL_STRATA {
                    let d_fi = st * STRIATAL_PORES + p_dst;
                    let s_fi = st * STRIATAL_PORES + p_src;
                    for m in 0..copy_na {
                        w_out[m * STRIATAL_CUES + d_fi] = dec.w_cons[m * STRIATAL_CUES + s_fi];
                    }
                    v_out[d_fi] = dec.v_cons[s_fi];
                    q_out[d_fi] = dec.q_cons[s_fi];
                    let d_jf = d_fi * na;
                    let s_jf = s_fi * dec.na;
                    j_out[d_jf..d_jf + copy_na]
                        .copy_from_slice(&dec.fwd_j_cons[s_jf..s_jf + copy_na]);
                }
            }
        }
        DecodedColumnSlab {
            col_id: dec.col_id,
            na,
            ticks: dec.ticks,
            r_peak: dec.r_peak,
            r_bar: dec.r_bar,
            dend_ch: dend_ch_out,
            dend_env: dend_env_out,
            dend_key: dend_key_out,
            w_cons: w_out,
            v_cons: v_out,
            q_cons: q_out,
            fwd_j_cons: j_out,
        }
    }

    pub fn apply_decoded_slab_to_hot_slot(&mut self, slot: usize, raw_dec: &DecodedColumnSlab) {
        let cur_ch = self.pore_ch;
        let dec = self.remap_slab_for_channels(raw_dec, &cur_ch);
        let (_, _, w_len, fwd_len) = self.dims();
        self.hot_col_id[slot] = dec.col_id;
        self.ctx_has_cons[slot] = 1;
        self.ctx_ticks[slot] = dec.ticks;
        self.ctx_r_peak[slot] = dec.r_peak;
        self.r_bar_ctx[slot] = dec.r_bar;
        self.hot_dend_ch[slot] = dec.dend_ch;
        self.hot_dend_env[slot] = dec.dend_env;
        self.hot_dend_key[slot] = dec.dend_key;
        self.hot_last_use[slot] = self.tick;

        let w_dst = slot * w_len;
        self.w_cons[w_dst..w_dst + w_len].copy_from_slice(&dec.w_cons);
        let v_dst = slot * STRIATAL_FEATS;
        self.v_cons[v_dst..v_dst + STRIATAL_FEATS].copy_from_slice(&dec.v_cons);
        let q_dst = slot * STRIATAL_CUES;
        self.q_cons[q_dst..q_dst + STRIATAL_CUES].copy_from_slice(&dec.q_cons);
        let j_dst = slot * fwd_len;
        self.fwd_j_cons[j_dst..j_dst + fwd_len].copy_from_slice(&dec.fwd_j_cons);
        self.restore_consolidated_context(slot);
    }

    fn page_in_cold_column(
        &mut self,
        paged_idx: usize,
        target_slot: usize,
    ) -> Result<(), PersistError> {
        if paged_idx >= self.paged_cols.len() || target_slot >= OFC_CONTEXTS {
            return Err(PersistError::InvalidTopology);
        }
        let dec = decode_column_slab(&self.paged_cols[paged_idx].slab_bytes)?;
        let incoming_col_id = dec.col_id;
        if self.ctx_has_cons[target_slot] != 0 && self.hot_col_id[target_slot] != incoming_col_id {
            self.evict_hot_slot_to_paged(target_slot)?;
        }
        if let Some(pos) = self
            .paged_cols
            .iter()
            .position(|e| e.col_id == incoming_col_id)
        {
            self.paged_cols.remove(pos);
        }
        self.apply_decoded_slab_to_hot_slot(target_slot, &dec);
        Ok(())
    }

    #[must_use]
    pub fn dendritic_match_stats(
        dend_ch: &[u32; STRIATAL_PORES],
        dend_env: &[u8; STRIATAL_PORES],
        dend_key: &[f32; STRIATAL_PORES],
        cur_ch: &[u32; STRIATAL_PORES],
        tonic_valid: &[bool; STRIATAL_PORES],
        tonic_now: &[f32; STRIATAL_PORES],
    ) -> (usize, f32, f32) {
        let mut pairs = [(0.0_f32, 0.0_f32); STRIATAL_PORES];
        let mut cnt = 0usize;
        let mut sum_u = 0.0_f32;
        let mut sum_k = 0.0_f32;
        let mut sum_abs = 0.0_f32;

        for p_col in 0..STRIATAL_PORES {
            if (dend_env[p_col] & 1) == 0 || dend_ch[p_col] == u32::MAX {
                continue;
            }
            let ch = dend_ch[p_col];
            let matched_cur = if cur_ch[p_col] == ch && tonic_valid[p_col] {
                Some(p_col)
            } else {
                (0..STRIATAL_PORES).find(|&q| cur_ch[q] == ch && tonic_valid[q])
            };
            if let Some(q) = matched_cur {
                let u = tonic_now[q];
                let k = dend_key[p_col];
                pairs[cnt] = (u, k);
                cnt += 1;
                sum_u += u;
                sum_k += k;
                sum_abs += (u - k).abs();
            }
        }
        if cnt == 0 {
            return (0, 1.0, 0.0);
        }
        let inv = 1.0 / (cnt as f32);
        let mean_u = sum_u * inv;
        let mean_k = sum_k * inv;
        let mean_abs = sanitize(sum_abs * inv);
        let mut dot = 0.0_f32;
        let mut nu = 0.0_f32;
        let mut nk = 0.0_f32;
        for &(u, k) in &pairs[..cnt] {
            let du = u - mean_u;
            let dk = k - mean_k;
            dot += du * dk;
            nu += du * du;
            nk += dk * dk;
        }
        let cos = if nk > 1.0e-4 && nu > 1.0e-4 {
            sanitize_clamp(dot / ((nu * nk).sqrt() + 1.0e-8), -1.0, 1.0)
        } else {
            0.0
        };
        (cnt, mean_abs, cos)
    }

    pub fn restore_consolidated_context(&mut self, c: usize) {
        let (_, _, w_len, fwd_len) = self.dims();
        if self.ctx_has_cons[c] == 0 {
            return;
        }
        self.w_ctx[c * w_len..(c + 1) * w_len]
            .copy_from_slice(&self.w_cons[c * w_len..(c + 1) * w_len]);
        self.v_ctx[c * STRIATAL_FEATS..(c + 1) * STRIATAL_FEATS]
            .copy_from_slice(&self.v_cons[c * STRIATAL_FEATS..(c + 1) * STRIATAL_FEATS]);
        self.q_ctx[c * STRIATAL_CUES..(c + 1) * STRIATAL_CUES]
            .copy_from_slice(&self.q_cons[c * STRIATAL_CUES..(c + 1) * STRIATAL_CUES]);
        self.fwd_j_ctx[c * fwd_len..(c + 1) * fwd_len]
            .copy_from_slice(&self.fwd_j_cons[c * fwd_len..(c + 1) * fwd_len]);
    }

    #[must_use]
    pub fn forward_agency(&self, n_act: usize) -> f32 {
        let cols = n_act.min(self.n_act);
        if self.tick < 120 || cols == 0 {
            return 0.0;
        }
        let mut col_j = [0.0_f32; 128];
        let cap_m = cols.min(128);
        for j in 0..STRIATAL_CUES {
            let base = j * self.n_act;
            for (m, cj) in col_j.iter_mut().enumerate().take(cap_m) {
                let v = self.fwd_j[base + m];
                *cj += v * v;
            }
        }
        let mut max_cj = 0.0_f32;
        for &cj in &col_j[..cap_m] {
            if cj > max_cj {
                max_cj = cj;
            }
        }
        let mut eff_dof = 0usize;
        if max_cj > 5.0e-4 {
            for &cj in &col_j[..cap_m] {
                if cj >= 0.04 * max_cj {
                    eff_dof += 1;
                }
            }
        }
        let min_dof = (cols / 3).max(2);
        let dof = eff_dof.max(min_dof).clamp(2, 32).min(cols);
        let dof_pen = (dof as f32) / 500.0;
        let mut sum_r2 = 0.0_f32;
        let mut active_cnt = 0usize;
        for j in 0..STRIATAL_CUES {
            let vn = self.fwd_var_null[j];
            if vn > 2.0e-6 {
                let r2 = ((vn - self.fwd_var_act[j]) / vn - dof_pen).clamp(0.0, 1.0);
                sum_r2 += r2;
                active_cnt += 1;
            }
        }
        if active_cnt > 0 {
            sanitize_clamp(sum_r2 / (active_cnt as f32), 0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_u32(self.total_cols);
        hasher.write_u8(self.num_ctx);
        hasher.write_u8(self.active_ctx);
        for &cid in &self.hot_col_id {
            hasher.write_u32(cid);
        }
        for &hc in &self.ctx_has_cons {
            hasher.write_u8(hc);
        }
        for c in 0..OFC_CONTEXTS {
            for p in 0..STRIATAL_PORES {
                hasher.write_u32(self.hot_dend_ch[c][p]);
                hasher.write_u8(self.hot_dend_env[c][p]);
                hasher.write_f32(self.hot_dend_key[c][p]);
            }
        }
        hasher.write_usize(self.paged_cols.len());
        for entry in &self.paged_cols {
            hasher.write_u32(entry.col_id);
            hasher.write_u32(entry.slab_crc32c);
        }
        for &ch in &self.pore_ch {
            hasher.write_u32(ch);
        }
        for &pm in &self.pore_mean {
            hasher.write_f32(pm);
        }
        for &cm in &self.pore_carr_mean {
            hasher.write_f32(cm);
        }
        for &oe in &self.pore_orb_err {
            hasher.write_f32(oe);
        }
        for &cg in &self.canc_g {
            hasher.write_f32(cg);
        }
        for &w in &self.w_ctx {
            hasher.write_f32(w);
        }
        for &w in &self.w_cons {
            hasher.write_f32(w);
        }
        for &v in &self.v_ctx {
            hasher.write_f32(v);
        }
        for &v in &self.v_cons {
            hasher.write_f32(v);
        }
        for &q in &self.q_ctx {
            hasher.write_f32(q);
        }
        for &q in &self.q_cons {
            hasher.write_f32(q);
        }
        for &fb in &self.fwd_b {
            hasher.write_f32(fb);
        }
        for &fa in &self.fwd_alpha {
            hasher.write_f32(fa);
        }
        for &fj in &self.fwd_j {
            hasher.write_f32(fj);
        }
        for &fj in &self.fwd_j_ctx {
            hasher.write_f32(fj);
        }
        for &fj in &self.fwd_j_cons {
            hasher.write_f32(fj);
        }
        for &vn in &self.fwd_var_null {
            hasher.write_f32(vn);
        }
        for &va in &self.fwd_var_act {
            hasher.write_f32(va);
        }
        for &am in &self.act_mean {
            hasher.write_f32(am);
        }
        for &rb in &self.r_bar_ctx {
            hasher.write_f32(rb);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TubeKind {
    CellToCell = 0,
    CellToMotor = 1,
}

impl TubeKind {
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CellToCell),
            1 => Some(Self::CellToMotor),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tube {
    pub alive: bool,
    pub kind: TubeKind,
    pub pre: u32,
    pub post: u32,
    pub delay: u8,
    pub u: [f32; 4],
    pub w_pred: f32,
    pub context_gate_cell: Option<u32>,
    pub gate_invert: bool,
    pub contradiction_accum: f32,
    pub chain_score: f32,
    pub myelin_plasticity_accum: f32,
    pub last_upstream_cell: u32,
    pub usage_ema: f32,
    pub age_ticks: u32,
}

impl Tube {
    pub const LOGICAL_BYTES: usize = 1 + 1 + 4 + 4 + 1 + 4 * 4 + 4 + 5 + 1 + 4 * 4 + 4 + 4;

    #[must_use]
    pub fn newborn(kind: TubeKind, pre: u32, post: u32, delay: u8, init_u0: f32) -> Self {
        let u0 = sanitize_clamp(init_u0, -2.5, 2.5);
        Self {
            alive: true,
            kind,
            pre,
            post,
            delay: delay.clamp(1, 16),
            u: [u0, 0.25 * u0, 0.0, 0.0],
            w_pred: 0.0,
            context_gate_cell: None,
            gate_invert: false,
            contradiction_accum: 0.0,
            chain_score: 0.0,
            myelin_plasticity_accum: 0.0,
            last_upstream_cell: u32::MAX,
            usage_ema: 0.02,
            age_ticks: 0,
        }
    }

    #[inline]
    #[must_use]
    pub fn effective_g(&self, cells: &[Cell]) -> f32 {
        let base = self.u[0];
        let gate = match self.context_gate_cell {
            None => 1.0,
            Some(g_idx) => {
                let gv = cells
                    .get(g_idx as usize)
                    .filter(|c| c.alive)
                    .map(|c| (c.v_real.abs() + 0.85 * c.surprise_ema).clamp(0.0, 1.0))
                    .unwrap_or(0.0);
                if self.gate_invert {
                    (1.0 - 1.45 * gv).clamp(0.0, 1.0)
                } else {
                    (0.25 + 1.15 * gv).clamp(0.0, 1.0)
                }
            }
        };
        sanitize(base * gate)
    }

    #[inline]
    pub fn apply_benna_fusi_impulse(&mut self, du: f32, consolidation_boost: f32) {
        self.u[0] = sanitize_clamp(self.u[0] + du, -2.5, 2.5);
        let boost = sanitize_clamp(consolidation_boost, 1.0, 6.0);
        let g01 = (0.045 * boost).min(0.35);
        let g12 = (0.009 * boost).min(0.12);
        let g23 = (0.002 * boost).min(0.04);

        let f01 = g01 * (self.u[0] - self.u[1]);
        let f12 = g12 * (self.u[1] - self.u[2]);
        let f23 = g23 * (self.u[2] - self.u[3]);

        self.u[0] = sanitize_clamp(self.u[0] - 0.5 * f01, -2.5, 2.5);
        self.u[1] = sanitize_clamp(self.u[1] + 0.65 * f01 - 0.5 * f12, -2.5, 2.5);
        self.u[2] = sanitize_clamp(self.u[2] + 0.65 * f12 - 0.5 * f23, -2.5, 2.5);
        self.u[3] = sanitize_clamp(self.u[3] + 0.65 * f23, -2.5, 2.5);
    }

    #[inline]
    #[must_use]
    pub fn stiff_strength(&self) -> f32 {
        sanitize(self.u[2].abs() + self.u[3].abs())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EfferenceEntry {
    pub valid: bool,
    pub tick: u64,
    pub committed: bool,
    pub motor_norm: f32,
    pub motor_out: Vec<f32>,
    pub active_cells: Vec<(u32, f32, f32, f32)>,
    pub pred_with_action: Vec<(u32, f32)>,
    pub pred_null_action: Vec<(u32, f32)>,
    pub somatic_value_est: f32,
    pub raw_valence_at_commit: f32,
}

impl Default for EfferenceEntry {
    fn default() -> Self {
        Self {
            valid: false,
            tick: 0,
            committed: false,
            motor_norm: 0.0,
            motor_out: Vec::new(),
            active_cells: Vec::new(),
            pred_with_action: Vec::new(),
            pred_null_action: Vec::new(),
            somatic_value_est: 0.0,
            raw_valence_at_commit: 0.0,
        }
    }
}

impl EfferenceEntry {
    /// Committed command of actuator `m` (zero for actuators born after this entry).
    #[inline]
    #[must_use]
    pub fn motor(&self, m: usize) -> f32 {
        self.motor_out.get(m).copied().unwrap_or(0.0)
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        1 + 8
            + 1
            + 4
            + 4 * self.motor_out.len()
            + self.active_cells.len() * 16
            + self.pred_with_action.len() * 8
            + self.pred_null_action.len() * 8
            + 8
    }

    #[inline]
    #[must_use]
    pub fn cell_signal(&self, target_cell: u32) -> f32 {
        for &(ci, vr, dv, _) in &self.active_cells {
            if ci == target_cell {
                return sanitize(vr + 0.25 * dv);
            }
        }
        0.0
    }

    #[inline]
    #[must_use]
    pub fn cell_v_real(&self, target_cell: u32) -> Option<f32> {
        for &(ci, vr, _, _) in &self.active_cells {
            if ci == target_cell {
                return Some(vr);
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EfferenceRing {
    pub entries: [EfferenceEntry; RING_LEN],
    pub head: usize,
}

impl Default for EfferenceRing {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| EfferenceEntry::default()),
            head: 0,
        }
    }
}

impl EfferenceRing {
    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        let sum: usize = self.entries.iter().map(EfferenceEntry::logical_bytes).sum();
        sum + 8
    }

    pub fn push(&mut self, entry: EfferenceEntry) {
        let idx = self.head % RING_LEN;
        self.entries[idx] = entry;
        self.head = (idx + 1) % RING_LEN;
    }

    #[must_use]
    pub fn at_lag(&self, lag: usize) -> Option<&EfferenceEntry> {
        if !(1..=RING_LEN).contains(&lag) {
            return None;
        }
        let idx = (self.head + RING_LEN - lag) % RING_LEN;
        let e = &self.entries[idx];
        if e.valid { Some(e) } else { None }
    }
}

pub struct RolloutOutcome {
    pub pulse: FieldPulse,
    pub committed: bool,
    pub rehearsals: u64,
    pub pred_with: Vec<(u32, f32)>,
    pub pred_null: Vec<(u32, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TubeGraph {
    pub slots: Vec<Tube>,
    pub free_list: Vec<u32>,
    pub alive_count: usize,
    pub outgoing_by_cell: Vec<Vec<u32>>,
    /// Per-actuator, per-cell action-to-sensory Jacobian rows (indexed by cell slot, no aliasing).
    pub action_sensory_w: Vec<Vec<f32>>,
    pub pore_action_w: Vec<[f32; PRED_BUCKETS]>,
    pub tectal_map_w: Vec<[f32; PRED_BUCKETS]>,
    /// Per-cell sensory self-prediction weights (indexed by cell slot, no aliasing).
    pub sensory_self_w: Vec<f32>,
    pub pore_self_w: [f32; PRED_BUCKETS],
    pub pore_mean_v: [f32; PRED_BUCKETS],
    pub pore_scale_v: [f32; PRED_BUCKETS],
    pub pore_agency: [f32; PRED_BUCKETS],
    pub pore_dsq_ema: [f32; PRED_BUCKETS],
    pub pore_var_ema: [f32; PRED_BUCKETS],
    pub pore_bg_ild: [f32; PRED_BUCKETS],
    pub pore_floor_v: [f32; PRED_BUCKETS],
    pub pore_ar1_w: [f32; PRED_BUCKETS],
    pub pore_ar2_w: [f32; PRED_BUCKETS],
    pub pore_q5_lvl: [f32; PRED_BUCKETS],
    pub pore_q5_exaff: [f32; PRED_BUCKETS],
    pub pore_vocal_reaff: [f32; PRED_BUCKETS],
    pub tonotopic_floor_ema: f32,
    pub tonotopic_conf_ema: f32,
    pub acoustic_env_ema: f32,
    pub agency_signed_ema: f32,
    pub vocal_eff_s1: f32,
    pub vocal_eff_s2: f32,
    pub vocal_eff_hist: u32,
    pub vocal_onset_hist: u32,
    pub vocal_last_mg: u8,
    pub vocal_q5_cnt: u32,
    pub vocal_y_ema: f32,
    pub vocal_f_ema: [f32; 4],
    pub vocal_blk_sum: [f32; 4],
    pub vocal_s_acc: [f32; 4],
    pub vocal_v_acc: [f32; 4],
    pub even_tick_cnt: u32,
    pub even_y_ema: f32,
    pub even_f_ema: [f32; 4],
    pub even_blk_sum: [f32; 4],
    pub even_s_acc: [f32; 4],
    pub even_v_acc: [f32; 4],
    pub social_loom_ema: f32,
    pub spatial_prox_ring: [f32; RING_LEN],
    pub spatial_prox_cursor: u8,
    pub echo_emit_w: Vec<f32>,
    pub vocal_artic_w: Vec<f32>,
    pub somatic_limb_w: Vec<f32>,
    pub vocal_glottal_w: Vec<f32>,
    pub peer_coch_fast: f32,
    pub peer_coch_slow: f32,
    pub peer_echo_fast: f32,
    pub peer_echo_slow: f32,
    pub peer_disp_coch: f32,
    pub peer_disp_echo: f32,
    pub peer_vocal_hold: f32,
    pub peer_coch_age: u8,
    pub peer_echo_age: u8,
    pub peer_bearing_ema: Vec<f32>,
    pub act_mean: Vec<f32>,
    pub mean_interventional_agency: f32,
    pub striatal: StriatalOfc,
}

impl TubeGraph {
    #[inline]
    pub(crate) fn detect_tonotopic_bank_floor(ambient_in: &[f32]) -> Option<f32> {
        let m = ambient_in.len().min(320);
        if m < 64 {
            return None;
        }
        let mut hist = [0u16; 256];
        let mut bin_sum = [0.0_f32; 256];
        for &raw_v in &ambient_in[..m] {
            if !raw_v.is_finite() {
                continue;
            }
            let av = raw_v.abs();
            if !(0.025..=0.96).contains(&av) {
                continue;
            }
            let b = (((raw_v + 1.0) * 128.0) as isize).clamp(0, 255) as usize;
            hist[b] = hist[b].saturating_add(1);
            bin_sum[b] += raw_v;
        }
        let mut best_cnt = 0u16;
        let mut best_b = 0usize;
        for b in 0..255 {
            let c = hist[b].saturating_add(hist[b + 1]);
            if c > best_cnt {
                best_cnt = c;
                best_b = b;
            }
        }
        let mirror_b = 254usize.saturating_sub(best_b);
        let mirror_cnt = hist[mirror_b].saturating_add(hist[(mirror_b + 1).min(255)]);
        if usize::from(best_cnt) >= 22
            && usize::from(best_cnt) * 14 >= m
            && best_cnt > mirror_cnt.saturating_mul(2)
        {
            let raw_fl = if hist[best_b] >= 8
                && usize::from(hist[best_b]) * 3 >= usize::from(best_cnt)
            {
                bin_sum[best_b] / f32::from(hist[best_b])
            } else {
                (bin_sum[best_b] + bin_sum[best_b + 1]) / f32::from(best_cnt)
            };
            let fl = sanitize_clamp(raw_fl, -0.96, 0.96);
            if fl.abs() >= 0.08 {
                return Some(fl);
            }
        }
        None
    }

    pub fn observe_ambient_bank(&mut self, ambient_in: &[f32], mode: ControlMode) -> bool {
        let detected = Self::detect_tonotopic_bank_floor(ambient_in);
        if mode == ControlMode::Natural {
            if let Some(fl) = detected {
                if self.tonotopic_conf_ema < 1.0e-4 && self.striatal.tick == 0 {
                    self.tonotopic_floor_ema = fl;
                    self.tonotopic_conf_ema = 1.0;
                } else {
                    let lr = if self.tonotopic_conf_ema < 0.20 {
                        0.25
                    } else if fl <= self.tonotopic_floor_ema {
                        0.16
                    } else {
                        0.004
                    };
                    self.tonotopic_floor_ema = sanitize_clamp(
                        (1.0 - lr) * self.tonotopic_floor_ema + lr * fl,
                        -0.96,
                        0.96,
                    );
                    self.tonotopic_conf_ema =
                        sanitize_clamp(0.96 * self.tonotopic_conf_ema + 0.04, 0.0, 1.0);
                }
            } else {
                self.tonotopic_conf_ema =
                    sanitize_clamp(0.992 * self.tonotopic_conf_ema, 0.0, 1.0);
            }
            self.tonotopic_conf_ema >= 0.20
        } else {
            self.tonotopic_conf_ema >= 0.20 || detected.is_some()
        }
    }

    #[inline]
    #[must_use]
    pub fn has_tonotopic_bank(&self) -> bool {
        self.tonotopic_conf_ema >= 0.20
    }

    #[inline]
    fn inv_acoustic_gain(&self) -> f32 {
        if self.tonotopic_conf_ema >= 0.20 && self.acoustic_env_ema >= 0.15 {
            sanitize_clamp(0.84 / self.acoustic_env_ema.max(0.36), 0.98, 1.48)
        } else {
            1.0
        }
    }

    #[inline]
    fn primary_glottal_actuator(&self, is_world_loco: &[bool], act_cap: usize) -> Option<usize> {
        let cap = act_cap
            .min(self.vocal_glottal_w.len())
            .min(is_world_loco.len());
        let loco_ok = |m: usize| !is_world_loco[m] || self.vocal_glottal_w[m] >= 0.085;
        let mut top1_m: Option<usize> = None;
        let mut top1_w = 0.0_f32;
        let mut top2_w = 0.0_f32;
        for m in 0..cap {
            if !loco_ok(m) {
                continue;
            }
            let w = self.vocal_glottal_w[m];
            if w > top1_w {
                top2_w = top1_w;
                top1_w = w;
                top1_m = Some(m);
            } else if w > top2_w {
                top2_w = w;
            }
        }
        let cur = self.vocal_last_mg as usize;
        if cur < cap && self.vocal_glottal_w[cur] >= 0.030 {
            let cur_w = self.vocal_glottal_w[cur];
            let switch_bar = if cur_w >= 0.20 {
                cur_w + 0.14
            } else {
                (cur_w + 0.06).max(1.35 * cur_w).max(0.14)
            };
            if let Some(bm) = top1_m
                && bm != cur
                && top1_w >= switch_bar
            {
                return Some(bm);
            }
            return Some(cur);
        }
        if top1_w >= 0.068 && top1_w >= top2_w + 0.025 {
            top1_m
        } else {
            None
        }
    }

    #[inline]
    fn is_flat_elevated_plateau(&self, _p: usize, history: &[f32; 16], h: usize) -> bool {
        if self.tonotopic_conf_ema < 0.20 {
            return false;
        }
        let ig = self.inv_acoustic_gain();
        let fl_ref = self.tonotopic_floor_ema;
        let mut flat_elev = 0u8;
        let mut step_latch = false;
        for k in 0..15 {
            let x_a = history[(h + k) & 15];
            let x_b = history[(h + k + 1) & 15];
            let d_ab = (x_b - x_a).abs();
            if (x_a - fl_ref) * ig > 0.010 && d_ab * ig < 5.0e-4 {
                flat_elev += 1;
            }
            if k < 12 {
                let x_c = history[(h + k + 2) & 15];
                let x_d = history[(h + k + 3) & 15];
                let x_e = history[(h + k + 4) & 15];
                if d_ab * ig > 0.008
                    && (x_c - x_b).abs() < 2.0e-5
                    && (x_d - x_c).abs() < 2.0e-5
                    && (x_e - x_d).abs() < 2.0e-5
                    && (x_b - fl_ref) * ig > 0.012
                {
                    step_latch = true;
                }
            }
        }
        flat_elev >= 6 || step_latch
    }

    #[inline]
    fn receptor_ref_floor(&self, p: usize, min_h: f32) -> f32 {
        if self.tonotopic_conf_ema >= 0.20 {
            self.tonotopic_floor_ema
        } else {
            self.pore_floor_v[p].min(min_h)
        }
    }

    #[inline]
    fn on_tonotopic_floor(&self, p: usize) -> bool {
        if self.tonotopic_conf_ema < 0.20 {
            return false;
        }
        let zf = (self.pore_floor_v[p] - self.tonotopic_floor_ema) * self.inv_acoustic_gain();
        (-0.045..=0.065).contains(&zf)
    }

    #[inline]
    fn is_tonotopic_receptor(&self, p: usize) -> bool {
        if self.tonotopic_conf_ema < 0.20 {
            return false;
        }
        let zf = (self.pore_floor_v[p] - self.tonotopic_floor_ema) * self.inv_acoustic_gain();
        (-0.032..=0.052).contains(&zf) && self.pore_ar1_w[p] <= 0.86
    }

    #[inline]
    fn z_from_bank(&self, p: usize, min_h: f32, v: f32) -> f32 {
        (v - self.receptor_ref_floor(p, min_h)) * self.inv_acoustic_gain()
    }

    #[inline]
    fn is_cochlear_floor_pore(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let ig = self.inv_acoustic_gain();
        let fl = self.receptor_ref_floor(p, min_h);
        let zf = (self.pore_floor_v[p] - fl) * ig;
        let zm = (min_h - fl) * ig;
        let zmean = (mean_p - fl) * ig;
        (-0.010..=0.065).contains(&zf)
            && (-0.005..=0.19).contains(&zm)
            && (0.0030..=0.31).contains(&zmean)
    }

    #[inline]
    fn is_pure_coch_floor(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let ig = self.inv_acoustic_gain();
        let fl = self.receptor_ref_floor(p, min_h);
        let zf = (self.pore_floor_v[p] - fl) * ig;
        let zm = (min_h - fl) * ig;
        let zmean = (mean_p - fl) * ig;
        (-0.010..=0.065).contains(&zf)
            && (-0.005..=0.19).contains(&zm)
            && (0.0010..=0.30).contains(&zmean)
    }

    #[inline]
    fn is_quiet_coch_floor(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let ig = self.inv_acoustic_gain();
        let fl = self.receptor_ref_floor(p, min_h);
        let zf = (self.pore_floor_v[p] - fl) * ig;
        let zm = (min_h - fl) * ig;
        let zmean = (mean_p - fl) * ig;
        (-0.010..=0.065).contains(&zf)
            && (-0.005..=0.19).contains(&zm)
            && (-0.003..=0.45).contains(&zmean)
    }

    #[inline]
    fn is_glottal_driven_coch(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let ig = self.inv_acoustic_gain();
        let fl = self.receptor_ref_floor(p, min_h);
        (self.pore_floor_v[p] - fl) * ig >= -0.012 && (mean_p - fl) * ig >= 0.006
    }

    #[inline]
    fn is_coch_trough(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let zmean = self.z_from_bank(p, min_h, mean_p);
        (-0.020..=0.025).contains(&zmean)
    }

    #[inline]
    fn is_lso_floor_pore(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        let ig = self.inv_acoustic_gain();
        let fl = self.receptor_ref_floor(p, min_h);
        let zf = (self.pore_floor_v[p] - fl) * ig;
        let zmin = (min_h - fl) * ig;
        let zmean = (mean_p - fl) * ig;
        (-0.14..=0.055).contains(&zf) && zmin <= 0.15 && (-0.16..0.17).contains(&zmean)
    }

    #[inline]
    fn is_somatic_non_lso(&self, p: usize, min_h: f32, mean_p: f32) -> bool {
        if self.tonotopic_conf_ema >= 0.20 {
            !self.is_tonotopic_receptor(p)
        } else {
            mean_p > self.pore_floor_v[p] + 0.015 || min_h > -0.15
        }
    }

    #[inline]
    fn deconv_pore_exc(&self, p: usize, history: &[f32; 16], h: usize, lag: usize) -> f32 {
        let fl = if self.tonotopic_conf_ema >= 0.20 {
            self.tonotopic_floor_ema
        } else {
            self.pore_floor_v[p]
        };
        let ig = self.inv_acoustic_gain();
        let y0 = (history[(h + 15 - lag) & 15] - fl) * ig;
        let y1 = (history[(h + 14 - lag) & 15] - fl) * ig;
        let y2 = (history[(h + 13 - lag) & 15] - fl) * ig;
        let w1 = self.pore_ar1_w[p];
        let w2 = self.pore_ar2_w[p];
        let dc_inv = 1.0 / (1.0 - w1 - w2).clamp(0.28, 1.0);
        sanitize_clamp((y0 - w1 * y1 - w2 * y2) * dc_inv, -1.65, 1.65)
    }

    #[inline]
    fn deconv_pore_sym(&self, p: usize, history: &[f32; 16], h: usize, lag: usize) -> f32 {
        let u_rel = self.deconv_pore_exc(p, history, h, lag);
        let sym = if u_rel > 0.60 {
            0.60 + 0.0667 * (u_rel - 0.60)
        } else if u_rel < -0.60 {
            -0.60 + 0.0667 * (u_rel + 0.60)
        } else {
            u_rel
        };
        sanitize_clamp(sym, -0.66, 0.66)
    }

    #[inline]
    fn motor_efference_active_lags(ring: &EfferenceRing, mg: usize) -> [bool; 18] {
        let mut active = [false; 18];
        let u_init = ring.at_lag(RING_LEN).map_or(0.0_f32, |e| {
            if e.committed {
                sanitize_clamp(e.motor(mg), -0.36, 1.0)
            } else {
                0.0
            }
        });
        let mut s1 = u_init;
        let mut s2 = u_init;
        active[RING_LEN + 1] = s2 >= 0.095;
        for lg in (1..=RING_LEN).rev() {
            let u = ring.at_lag(lg).map_or(0.0_f32, |e| {
                if e.committed {
                    sanitize_clamp(e.motor(mg), -0.36, 1.0)
                } else {
                    0.0
                }
            });
            s1 = sanitize_clamp(0.36 * s1 + 0.64 * u, -0.36, 1.0);
            s2 = sanitize_clamp(0.36 * s2 + 0.64 * s1, -0.36, 1.0);
            active[lg] = s2 >= 0.095;
        }
        active
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    fn motor_contingency_stats(
        &self,
        membrane: &PoreMembrane,
        act_ch: usize,
    ) -> (f32, f32, f32, f32, usize, usize, [f32; 4], [f32; 4]) {
        let mut es = [0.0_f32; 4];
        let mut ec = [0usize; 4];
        let mut os = [0.0_f32; 4];
        let mut oc = [0usize; 4];
        for tube in &self.slots {
            if tube.alive
                && tube.kind == TubeKind::CellToMotor
                && membrane
                    .motor_pores
                    .get(tube.post as usize)
                    .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == act_ch)
            {
                let b = ((tube.pre >> 1) & 3) as usize;
                if (tube.pre & 1) == 0 {
                    es[b] += tube.myelin_plasticity_accum;
                    ec[b] += 1;
                } else {
                    os[b] += tube.myelin_plasticity_accum;
                    oc[b] += 1;
                }
            }
        }
        let e_tot = ec[0] + ec[1] + ec[2] + ec[3];
        let o_tot = oc[0] + oc[1] + oc[2] + oc[3];
        let e_mean = if e_tot > 0 {
            (es[0] + es[1] + es[2] + es[3]) / (e_tot as f32)
        } else {
            0.0
        };
        let o_mean = if o_tot > 0 {
            (os[0] + os[1] + os[2] + os[3]) / (o_tot as f32)
        } else {
            0.0
        };
        let mut em_band = [0.0_f32; 4];
        let mut om_band = [0.0_f32; 4];
        let mut e_peak = f32::NEG_INFINITY;
        let mut o_peak = f32::NEG_INFINITY;
        for b in 0..4 {
            if ec[b] > 0 {
                let m = es[b] / (ec[b] as f32);
                em_band[b] = m;
                if m > e_peak {
                    e_peak = m;
                }
            }
            if oc[b] > 0 {
                let m = os[b] / (oc[b] as f32);
                om_band[b] = m;
                if m > o_peak {
                    o_peak = m;
                }
            }
        }
        let ep = if e_peak.is_finite() { e_peak } else { 0.0 };
        let op = if o_peak.is_finite() { o_peak } else { 0.0 };
        (ep, op, e_mean, o_mean, e_tot, o_tot, em_band, om_band)
    }

    #[inline]
    fn insert_top4(top4: &mut [f32; 4], val: f32) {
        if val > top4[0] {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = top4[0];
            top4[0] = val;
        } else if val > top4[1] {
            top4[3] = top4[2];
            top4[2] = top4[1];
            top4[1] = val;
        } else if val > top4[2] {
            top4[3] = top4[2];
            top4[2] = val;
        } else if val > top4[3] {
            top4[3] = val;
        }
    }

    #[inline]
    fn is_retinal_field_pore(
        &self,
        p: usize,
        sp: &SensorPore,
        scout_stats: &[ChannelScoutStat],
    ) -> bool {
        let ch_u = sp.channel as usize;
        ch_u < scout_stats.len()
            && !self.is_tonotopic_receptor(p)
            && scout_stats[ch_u].is_smooth_spatial_field()
            && self.pore_self_w[p] >= 0.35
            && (0.010..=0.62).contains(&self.pore_agency[p])
            && self.pore_mean_v[p] > -0.02
            && sp.history.iter().all(|&v| v >= -0.10)
    }

    #[must_use]
    pub fn new(
        initial_cells: usize,
        initial_motors: usize,
        max_tubes: usize,
        rng: &mut Rng64,
    ) -> Self {
        let n_cells = initial_cells.max(4);
        let n_motors = initial_motors.max(1);
        let mut graph = Self {
            slots: Vec::with_capacity((n_cells * (n_motors + 3)).min(max_tubes.max(16))),
            free_list: Vec::new(),
            alive_count: 0,
            outgoing_by_cell: vec![Vec::new(); n_cells],
            action_sensory_w: vec![vec![0.0; n_cells.max(MIN_CELL_DIM)]; MIN_ACTUATOR_DIM],
            pore_action_w: vec![[0.0; PRED_BUCKETS]; MIN_ACTUATOR_DIM],
            tectal_map_w: vec![[0.0; PRED_BUCKETS]; MIN_ACTUATOR_DIM],
            sensory_self_w: vec![0.0; n_cells.max(MIN_CELL_DIM)],
            pore_self_w: [0.0; PRED_BUCKETS],
            pore_mean_v: [0.0; PRED_BUCKETS],
            pore_scale_v: [0.05; PRED_BUCKETS],
            pore_agency: [0.0; PRED_BUCKETS],
            pore_dsq_ema: [1.0e-4; PRED_BUCKETS],
            pore_var_ema: [5.0e-3; PRED_BUCKETS],
            pore_bg_ild: [0.0; PRED_BUCKETS],
            pore_floor_v: [0.0; PRED_BUCKETS],
            pore_ar1_w: [0.59; PRED_BUCKETS],
            pore_ar2_w: [0.0; PRED_BUCKETS],
            pore_q5_lvl: [1.0; PRED_BUCKETS],
            pore_q5_exaff: [0.0; PRED_BUCKETS],
            pore_vocal_reaff: [0.0; PRED_BUCKETS],
            tonotopic_floor_ema: 0.0,
            tonotopic_conf_ema: 0.0,
            acoustic_env_ema: 0.0,
            agency_signed_ema: 0.0,
            vocal_eff_s1: 0.0,
            vocal_eff_s2: 0.0,
            vocal_eff_hist: 0,
            vocal_onset_hist: 0,
            vocal_last_mg: u8::MAX,
            vocal_q5_cnt: 0,
            vocal_y_ema: 0.0,
            vocal_f_ema: [0.0; 4],
            vocal_blk_sum: [0.0; 4],
            vocal_s_acc: [0.0; 4],
            vocal_v_acc: [0.0; 4],
            even_tick_cnt: 0,
            even_y_ema: 0.0,
            even_f_ema: [0.0; 4],
            even_blk_sum: [0.0; 4],
            even_s_acc: [0.0; 4],
            even_v_acc: [0.0; 4],
            social_loom_ema: 0.0,
            spatial_prox_ring: [0.0; RING_LEN],
            spatial_prox_cursor: 0,
            echo_emit_w: vec![0.0; MIN_ACTUATOR_DIM],
            vocal_artic_w: vec![0.0; MIN_ACTUATOR_DIM],
            somatic_limb_w: vec![0.0; MIN_ACTUATOR_DIM],
            vocal_glottal_w: vec![0.0; MIN_ACTUATOR_DIM],
            peer_coch_fast: 0.0,
            peer_coch_slow: 0.0,
            peer_echo_fast: 0.0,
            peer_echo_slow: 0.0,
            peer_disp_coch: 0.0,
            peer_disp_echo: 0.0,
            peer_vocal_hold: 0.0,
            peer_coch_age: 255,
            peer_echo_age: 255,
            peer_bearing_ema: vec![0.0; MIN_ACTUATOR_DIM],
            act_mean: vec![0.0; MIN_ACTUATOR_DIM],
            mean_interventional_agency: 0.0,
            striatal: StriatalOfc::from_seed(rng.s0 ^ rng.s1.rotate_left(17)),
        };

        for c in 0..n_cells {
            for hop in 0..2 {
                let dst =
                    (c + 1 + hop * 5 + rng.next_index(n_cells.saturating_sub(1).max(1))) % n_cells;
                if dst != c && graph.alive_count < max_tubes {
                    let delay = 1 + (rng.next_u32() % 3) as u8;
                    let init_w = 0.03 * rng.next_f32_signed();
                    let _ = graph.add_tube(
                        Tube::newborn(TubeKind::CellToCell, c as u32, dst as u32, delay, init_w),
                        max_tubes,
                    );
                }
            }
            for m in 0..n_motors {
                if graph.alive_count < max_tubes {
                    let init_w = 0.012 * rng.next_f32_signed();
                    let _ = graph.add_tube(
                        Tube::newborn(TubeKind::CellToMotor, c as u32, m as u32, 1, init_w),
                        max_tubes,
                    );
                }
            }
        }

        graph
    }

    /// Current per-actuator storage width (never below `MIN_ACTUATOR_DIM`).
    #[inline]
    #[must_use]
    pub fn act_dim(&self) -> usize {
        self.act_mean.len()
    }

    /// Grows every per-actuator learner to cover `num_actuators` (new actuators start naive).
    pub fn ensure_actuator_dim(&mut self, num_actuators: usize) {
        let want = num_actuators.max(MIN_ACTUATOR_DIM);
        if want <= self.act_dim() {
            return;
        }
        let cd = self.cell_dim();
        self.action_sensory_w.resize(want, vec![0.0; cd]);
        self.pore_action_w.resize(want, [0.0; PRED_BUCKETS]);
        self.tectal_map_w.resize(want, [0.0; PRED_BUCKETS]);
        self.echo_emit_w.resize(want, 0.0);
        self.vocal_artic_w.resize(want, 0.0);
        self.somatic_limb_w.resize(want, 0.0);
        self.vocal_glottal_w.resize(want, 0.0);
        self.peer_bearing_ema.resize(want, 0.0);
        self.act_mean.resize(want, 0.0);
    }

    /// Current per-cell predictor width (never below `MIN_CELL_DIM`).
    #[inline]
    #[must_use]
    pub fn cell_dim(&self) -> usize {
        self.sensory_self_w.len()
    }

    /// Grows per-cell predictors to cover every cell slot (new slots start naive).
    pub fn ensure_cell_dim(&mut self, n_cell_slots: usize) {
        let want = n_cell_slots.max(MIN_CELL_DIM);
        if want <= self.cell_dim() {
            return;
        }
        self.sensory_self_w.resize(want, 0.0);
        for row in &mut self.action_sensory_w {
            row.resize(want, 0.0);
        }
    }

    /// Clears the predictors of a freshly (re)born cell slot so it never inherits a dead cell's model.
    pub fn reset_cell_predictor(&mut self, c_idx: u32) {
        let c = c_idx as usize;
        if let Some(w) = self.sensory_self_w.get_mut(c) {
            *w = 0.0;
        }
        for row in &mut self.action_sensory_w {
            if let Some(w) = row.get_mut(c) {
                *w = 0.0;
            }
        }
    }

    #[must_use]
    pub fn logical_bytes(&self) -> usize {
        self.slots.len() * Tube::LOGICAL_BYTES
            + self.free_list.len() * 4
            + 4 * self.cell_dim() * (self.act_dim() + 1)
            + 2 * 4 * PRED_BUCKETS * self.act_dim()
            + 16 * 4 * PRED_BUCKETS
            + 4 * RING_LEN
            + 6 * 4 * self.act_dim()
            + self.striatal.logical_bytes()
            + 263
    }

    pub fn ensure_cell_capacity(&mut self, cell_slots_len: usize) {
        if self.outgoing_by_cell.len() < cell_slots_len {
            self.outgoing_by_cell.resize_with(cell_slots_len, Vec::new);
        }
    }

    pub fn rebuild_adjacency(&mut self, cell_slots_len: usize) {
        self.outgoing_by_cell.clear();
        self.outgoing_by_cell.resize_with(cell_slots_len, Vec::new);
        let mut alive = 0usize;
        for (idx, t) in self.slots.iter().enumerate() {
            if !t.alive {
                continue;
            }
            alive += 1;
            if let Some(list) = self.outgoing_by_cell.get_mut(t.pre as usize)
                && list.len() < MAX_OUTGOING_PER_CELL
            {
                list.push(idx as u32);
            }
        }
        self.alive_count = alive;
    }

    pub fn add_tube(&mut self, tube: Tube, max_tubes: usize) -> Option<u32> {
        if self.alive_count >= max_tubes {
            return None;
        }
        let pre = tube.pre as usize;
        self.ensure_cell_capacity(pre + 1);
        if self.outgoing_by_cell[pre].len() >= MAX_OUTGOING_PER_CELL {
            return None;
        }
        let idx = if let Some(slot) = self.free_list.pop() {
            self.slots[slot as usize] = tube;
            slot
        } else {
            if self.slots.len() >= max_tubes {
                return None;
            }
            let s = self.slots.len() as u32;
            self.slots.push(tube);
            s
        };
        let ins_pos = self.outgoing_by_cell[pre].partition_point(|&x| x < idx);
        self.outgoing_by_cell[pre].insert(ins_pos, idx);
        self.alive_count += 1;
        Some(idx)
    }

    #[must_use]
    pub fn has_direct_tube(&self, kind: TubeKind, pre: u32, post: u32) -> bool {
        let Some(list) = self.outgoing_by_cell.get(pre as usize) else {
            return false;
        };
        for &t_idx in list {
            if let Some(t) = self.slots.get(t_idx as usize)
                && t.alive
                && t.kind == kind
                && t.post == post
            {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn has_cell_tube_at_delay(&self, pre: u32, post: u32, delay: u8) -> bool {
        let Some(list) = self.outgoing_by_cell.get(pre as usize) else {
            return false;
        };
        for &t_idx in list {
            if let Some(t) = self.slots.get(t_idx as usize)
                && t.alive
                && t.kind == TubeKind::CellToCell
                && t.post == post
                && t.delay == delay
            {
                return true;
            }
        }
        false
    }

    pub fn accumulate_recurrent_and_predictive(
        &self,
        ring: &EfferenceRing,
        cells: &[Cell],
        ext_drive: &mut [f32],
        pred_drive: &mut [f32],
        sensory_trophic: &mut [f32],
        mode: ControlMode,
    ) {
        for lag in 1..=RING_LEN {
            let Some(entry) = ring.at_lag(lag) else {
                continue;
            };
            let lag_u8 = lag as u8;
            for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                    continue;
                };
                let src_trophic = cells
                    .get(c_idx as usize)
                    .filter(|c| c.alive)
                    .map_or(0.0, |c| c.sensory_trophic_ema);
                let signal = sanitize(v_past + 0.45 * dv_past);
                for &t_idx in out_list {
                    let Some(tube) = self.slots.get(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive || tube.kind != TubeKind::CellToCell || tube.delay != lag_u8 {
                        continue;
                    }
                    let dst = tube.post as usize;
                    let g = tube.effective_g(cells);
                    if let Some(d_ref) = ext_drive.get_mut(dst) {
                        *d_ref = sanitize(*d_ref + g * signal);
                    }
                    if let Some(p_ref) = pred_drive.get_mut(dst) {
                        *p_ref = sanitize(*p_ref + tube.w_pred * signal);
                    }
                    if mode == ControlMode::Natural && src_trophic > 0.008 {
                        let syn_strength = (g.abs() + 0.5 * tube.w_pred.abs()).min(1.0);
                        if syn_strength > 0.05
                            && let Some(t_ref) = sensory_trophic.get_mut(dst)
                        {
                            *t_ref = sanitize(
                                *t_ref + 0.65 * syn_strength * src_trophic * (0.25 + signal.abs()),
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn evaluate_interventional_agency_and_update_predictors(
        &mut self,
        ring: &EfferenceRing,
        cells: &CellArena,
        membrane: &mut PoreMembrane,
        emb_eco: bool,
        mode: ControlMode,
    ) {
        let ad = self.act_dim();
        let Some(prev) = ring.at_lag(1) else {
            return;
        };

        let n_act_ag = membrane.alive_motor_count.clamp(1, ad);
        if mode != ControlMode::Natural {
            return;
        }
        if !prev.committed {
            if !emb_eco {
                self.agency_signed_ema = sanitize_clamp(0.9995 * self.agency_signed_ema, -0.08, 1.0);
                if self.striatal.tick < 120 {
                    self.mean_interventional_agency =
                        sanitize_clamp(self.agency_signed_ema, 0.0, 1.0);
                }
            } else {
                self.agency_signed_ema =
                    sanitize_clamp(0.9996 * self.agency_signed_ema.max(0.0), 0.0, 1.0);
                self.mean_interventional_agency = self.agency_signed_ema;
            }
        }

        let mut centered_act = vec![0.0_f32; ad];
        let mut da_norm_sq = 0.0_f32;
        if prev.committed {
            for (m, &a_val) in prev.motor_out.iter().enumerate() {
                let da_m = sanitize_clamp(a_val - self.act_mean[m], -2.0, 2.0);
                centered_act[m] = da_m;
                da_norm_sq += da_m * da_m;
                self.act_mean[m] =
                    sanitize_clamp(0.985 * self.act_mean[m] + 0.015 * a_val, -1.0, 1.0);
            }
        }
        let nlms_scale = 1.0 / (1.0 + 0.18 * da_norm_sq);

        let mut muscle_err_gain = vec![0.0_f32; ad];
        let mut muscle_null_sq = vec![0.0_f32; ad];
        let mut muscle_sum_jac = vec![0.0_f32; ad];
        let mut motor_myelin = vec![0.0_f32; ad];

        for &(c_idx, _, _, _) in &prev.active_cells {
            let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                continue;
            };
            for &t_idx in out_list {
                let Some(tube) = self.slots.get(t_idx as usize) else {
                    continue;
                };
                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                    continue;
                }
                let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                    continue;
                };
                if mp.alive {
                    let act_ch = mp.actuator_channel as usize;
                    if act_ch < ad {
                        motor_myelin[act_ch] =
                            sanitize_clamp(motor_myelin[act_ch] + tube.stiff_strength(), 0.0, 2.0);
                    }
                }
            }
        }

        for &(c_idx, v_past, dv_past, _) in &prev.active_cells {
            let Some(cell_now) = cells.slots.get(c_idx as usize) else {
                continue;
            };
            if !cell_now.alive || cell_now.sensory_trophic_ema < 0.005 {
                continue;
            }
            let b = c_idx as usize;
            let actual_dv = sanitize_clamp(cell_now.v_real - v_past, -2.0, 2.0);
            let self_pred = sanitize(self.sensory_self_w[b] * dv_past);
            let mut act_pred = 0.0_f32;
            if prev.committed {
                for (m, &da_m) in centered_act.iter().enumerate() {
                    let w_mb = self.action_sensory_w[m][b];
                    if w_mb.abs() >= 0.018 {
                        act_pred += w_mb * da_m;
                    }
                    if w_mb.abs() > 0.012 {
                        muscle_sum_jac[m] += w_mb.abs();
                    }
                }
            }
            let total_err = sanitize_clamp(actual_dv - (self_pred + act_pred), -2.0, 2.0);
            self.sensory_self_w[b] = sanitize_clamp(
                0.998 * self.sensory_self_w[b] + 0.035 * total_err * dv_past,
                -1.2,
                1.2,
            );
            if prev.committed {
                let clean_err = total_err.clamp(-0.45, 0.45);
                for (m, &da_m) in centered_act.iter().enumerate() {
                    let w_old = self.action_sensory_w[m][b];
                    let contrib_m = w_old * da_m;
                    let w_couple = w_old.abs();
                    if da_m.abs() > 0.03 && w_couple >= 0.012 {
                        let err_without_m = sanitize(total_err + contrib_m);
                        muscle_err_gain[m] +=
                            w_couple * (err_without_m * err_without_m - total_err * total_err);
                        muscle_null_sq[m] += w_couple * (err_without_m * err_without_m) + 1.0e-5;
                    }
                    let sign_flip = da_m.abs() > 0.12 && actual_dv * contrib_m < -1.0e-3;
                    let base_decay = if sign_flip { 0.90 } else { 0.993 };
                    let eff_decay =
                        1.0 - (1.0 - base_decay) * nlms_scale * da_m.abs().clamp(0.03, 1.0);
                    let lr = (if sign_flip { 0.085 } else { 0.045 }) * nlms_scale;
                    self.action_sensory_w[m][b] =
                        sanitize_clamp(eff_decay * w_old + lr * clean_err * da_m, -1.5, 1.5);
                }
            }
        }

        let pore_cap = PRED_BUCKETS.min(membrane.sensor_pores.len());
        let n_scout = membrane.scout_stats.len();
        let quad_bank_ag = self.striatal.has_dual_population_carrier_bank();
        let mut shuf_act = vec![0.0_f32; ad];
        if prev.committed
            && let Some(lag9) = ring.at_lag(9)
            && lag9.committed
            && lag9.tick > 0
        {
            for m in 0..n_act_ag {
                let src_m = (m + 1) % n_act_ag;
                shuf_act[m] = sanitize_clamp(lag9.motor(src_m) - self.act_mean[src_m], -2.0, 2.0);
            }
        }
        let mut ag_null_sq = 0.0_f32;
        let mut ag_act_sq = 0.0_f32;
        let mut ag_shuf_sq = 0.0_f32;
        let mut ag_contrast_sq = 0.0_f32;
        let mut ag_eval = 0usize;
        let mut spatial_prox_sum = 0.0_f32;
        let mut spatial_prox_cnt = 0usize;
        let mut loco_auth = vec![0.0_f32; ad];
        let mut pore_err1 = [0.0_f32; PRED_BUCKETS];
        let adult_pruned_coch = if emb_eco
            && cells.slots.len() >= 255
            && (8 * n_act_ag..150).contains(&cells.alive_count)
        {
            let young_cortex = cells.slots.iter().all(|c| !c.alive || c.age_ticks < 60_000);
            young_cortex
                || self.slots.iter().any(|t| {
                    t.alive
                        && t.kind == TubeKind::CellToMotor
                        && t.myelin_plasticity_accum.abs() > 1.0e-4
                })
        } else {
            false
        };

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive {
                continue;
            }
            let h = sp.hist_head as usize;
            let x_now = sp.history[(h + 15) & 15];
            if sp.age_ticks <= 1 {
                for m in 0..ad {
                    self.pore_action_w[m][p] = 0.0;
                    self.tectal_map_w[m][p] = 0.0;
                }
                self.pore_self_w[p] = 0.0;
                self.pore_mean_v[p] = x_now;
                self.pore_scale_v[p] = 0.05;
                self.pore_agency[p] = 0.0;
                self.pore_dsq_ema[p] = 1.0e-4;
                self.pore_var_ema[p] = 5.0e-3;
                self.pore_bg_ild[p] = 0.0;
                self.pore_floor_v[p] = x_now;
                self.pore_ar1_w[p] = 0.59;
                self.pore_ar2_w[p] = 0.0;
                self.pore_q5_lvl[p] = 1.0;
                self.pore_q5_exaff[p] = 0.0;
                self.pore_vocal_reaff[p] = 0.0;
                continue;
            }
            let x_1 = sp.history[(h + 14) & 15];
            let x_2 = sp.history[(h + 13) & 15];
            let dx1 = sanitize_clamp(x_now - x_1, -2.0, 2.0);
            let dx2 = sanitize_clamp(x_1 - x_2, -2.0, 2.0);
            let w_flat = 1.0 / (1.0 + 400.0 * (dx1.abs() + dx2.abs()));
            let lr_fl = (if x_now <= self.pore_floor_v[p] {
                0.20
            } else {
                0.0008
            }) * w_flat
                * w_flat;
            self.pore_floor_v[p] = sanitize_clamp(
                (1.0 - lr_fl) * self.pore_floor_v[p] + lr_fl * x_now,
                -1.0,
                1.0,
            );
            if sp.age_ticks >= 4 && self.is_tonotopic_receptor(p) {
                let mut drive_p = 0.0_f32;
                if prev.committed {
                    for m in 0..ad {
                        let w_mp = self.pore_action_w[m][p];
                        if w_mp > 0.012 {
                            drive_p += w_mp * prev.motor(m).max(0.0);
                        }
                    }
                }
                if drive_p <= 1.0e-3 {
                    let fl_ar = self.tonotopic_floor_ema;
                    let ig_ar = self.inv_acoustic_gain();
                    let y0 = (x_now - fl_ar) * ig_ar;
                    let y1 = (x_1 - fl_ar) * ig_ar;
                    let y2 = (x_2 - fl_ar) * ig_ar;
                    let ay2 = y2.abs();
                    let ay1 = y1.abs();
                    let ay0 = y0.abs();
                    if y2 * y1 > 0.0
                        && y1 * y0 > 0.0
                        && ay2 > 0.14
                        && (0.35 * ay2..0.70 * ay2).contains(&ay1)
                        && (0.35 * ay1..0.70 * ay1).contains(&ay0)
                    {
                        let e_ar = y0 - (self.pore_ar1_w[p] * y1 + self.pore_ar2_w[p] * y2);
                        let norm_ar = y1 * y1 + y2 * y2 + 1.0e-4;
                        let mut w1_n = sanitize_clamp(
                            self.pore_ar1_w[p] + 0.03 * e_ar * y1 / norm_ar,
                            0.52,
                            0.68,
                        );
                        let mut w2_n = sanitize_clamp(
                            self.pore_ar2_w[p] + 0.02 * e_ar * y2 / norm_ar,
                            -0.08,
                            0.08,
                        );
                        let sum_w = w1_n + w2_n;
                        if sum_w > 0.64 {
                            w2_n = sanitize_clamp(0.64 - w1_n, -0.08, 0.08);
                            w1_n = sanitize_clamp(0.64 - w2_n, 0.52, 0.68);
                        } else if sum_w < 0.56 {
                            w1_n = sanitize_clamp(0.56 - w2_n, 0.52, 0.68);
                            w2_n = sanitize_clamp(0.56 - w1_n, -0.08, 0.08);
                        }
                        self.pore_ar1_w[p] = w1_n;
                        self.pore_ar2_w[p] = w2_n;
                    }
                }
            }
            let gain_p = if self.is_tonotopic_receptor(p) {
                self.inv_acoustic_gain()
            } else {
                1.0
            };
            let dx1_s = sanitize_clamp(dx1 * gain_p, -2.0, 2.0);
            let dx2_s = sanitize_clamp(dx2 * gain_p, -2.0, 2.0);
            let pre_mean_p = self.pore_mean_v[p];
            let pre_scale_p = self.pore_scale_v[p].max(0.015);
            self.pore_mean_v[p] =
                sanitize_clamp(0.9985 * self.pore_mean_v[p] + 0.0015 * x_now, -2.0, 2.0);
            if x_now.abs() > 1.0e-4 || x_1.abs() > 1.0e-4 {
                let dm_p = sanitize_clamp(x_now - self.pore_mean_v[p], -2.0, 2.0);
                self.pore_dsq_ema[p] = sanitize_clamp(
                    0.9980 * self.pore_dsq_ema[p] + 0.0020 * (dx1 * dx1),
                    1.0e-6,
                    2.0,
                );
                self.pore_var_ema[p] = sanitize_clamp(
                    0.9980 * self.pore_var_ema[p] + 0.0020 * (dm_p * dm_p),
                    1.0e-5,
                    2.0,
                );
            }
            self.pore_scale_v[p] = sanitize_clamp(
                0.9960 * self.pore_scale_v[p] + 0.0040 * (2.8 * dx1_s.abs().min(0.18)).max(0.015),
                0.015,
                0.65,
            );
            let scale_p = self.pore_scale_v[p].max(0.015);
            if sp.age_ticks >= 3 && sp.trophic_ema >= 0.025 {
                let ag_pore_cap = if !emb_eco { 0.75 } else { 0.65 };
                let ag_da_ok = !emb_eco || da_norm_sq >= 0.015;
                let ag_delta_min = if !emb_eco { 0.012 } else { 0.020 };
                let quad_carr_ag = quad_bank_ag
                    && p < STRIATAL_PORES
                    && self.striatal.pore_ch[p] == sp.channel
                    && self.striatal.is_quad_carrier_slot(p);
                let striatal_reaff = self.striatal.tick >= 48
                    && p < STRIATAL_PORES
                    && self.striatal.pore_ch[p] == sp.channel
                    && !quad_carr_ag
                    && self.striatal.pore_var_raw[p] > 8.0e-4
                    && self.striatal.pore_var_exaff[p] < 0.28 * self.striatal.pore_var_raw[p];
                if prev.committed
                    && ag_da_ok
                    && sp.age_ticks >= 16
                    && self.pore_agency[p] <= ag_pore_cap
                    && !quad_carr_ag
                    && !striatal_reaff
                    && !self.on_tonotopic_floor(p)
                    && dx2_s.abs() > 1.0e-3
                {
                    let mut act_delta_o = 0.0_f32;
                    let mut shuf_delta_o = 0.0_f32;
                    for m in 0..n_act_ag {
                        let w_mp = self.pore_action_w[m][p];
                        act_delta_o += w_mp * centered_act[m];
                        shuf_delta_o += w_mp * shuf_act[m];
                    }
                    if act_delta_o.abs() > ag_delta_min {
                        let self_w_p = self.pore_self_w[p];
                        let ch_u = sp.channel as usize;
                        let is_pure_optical = !emb_eco
                            && ch_u < n_scout
                            && membrane.scout_stats[ch_u].is_smooth_spatial_field()
                            && self_w_p >= 0.35
                            && pre_mean_p > -0.02
                            && sp.history.iter().all(|&v| v >= -0.05);
                        let dz_now_o = sanitize_clamp(dx1_s / pre_scale_p, -3.0, 3.0);
                        let dz_prev_o = sanitize_clamp(dx2_s / pre_scale_p, -3.0, 3.0);
                        let p_null_o = if is_pure_optical {
                            sanitize(self_w_p * dz_prev_o)
                        } else {
                            let mut num_lvl = 0.0_f32;
                            let mut den_lvl = 0.0_f32;
                            for j in 1..=12usize {
                                let x_j0 = sp.history[(h + 15 - j) & 15];
                                let x_j1 = sp.history[(h + 14 - j) & 15];
                                let x_j2 = sp.history[(h + 13 - j) & 15];
                                let dz_j = sanitize_clamp(
                                    ((x_j0 - x_j1) * gain_p) / pre_scale_p,
                                    -3.0,
                                    3.0,
                                );
                                let dz_j_prev = sanitize_clamp(
                                    ((x_j1 - x_j2) * gain_p) / pre_scale_p,
                                    -3.0,
                                    3.0,
                                );
                                let zlvl_j = sanitize_clamp(
                                    ((x_j1 - pre_mean_p) * gain_p) / pre_scale_p,
                                    -3.0,
                                    3.0,
                                );
                                let res_j = dz_j - self_w_p * dz_j_prev;
                                num_lvl += res_j * zlvl_j;
                                den_lvl += zlvl_j * zlvl_j;
                            }
                            let beta_hi = if !emb_eco { 0.0 } else { 0.95 };
                            let beta_lvl =
                                sanitize_clamp(num_lvl / (den_lvl + 0.50), -0.95, beta_hi);
                            let zlvl_0 = sanitize_clamp(
                                ((x_1 - pre_mean_p) * gain_p) / pre_scale_p,
                                -3.0,
                                3.0,
                            );
                            sanitize_clamp(self_w_p * dz_prev_o + beta_lvl * zlvl_0, -3.0, 3.0)
                        };
                        let e_null = sanitize_clamp(dz_now_o - p_null_o, -3.0, 3.0);
                        let e_act = sanitize_clamp(dz_now_o - (p_null_o + act_delta_o), -3.0, 3.0);
                        let e_shuf =
                            sanitize_clamp(dz_now_o - (p_null_o + shuf_delta_o), -3.0, 3.0);
                        ag_null_sq += e_null * e_null;
                        ag_act_sq += e_act * e_act;
                        ag_shuf_sq += e_shuf * e_shuf;
                        ag_contrast_sq += act_delta_o * act_delta_o;
                        ag_eval += 1;
                    }
                }
                let dz_now = sanitize_clamp(dx1_s / scale_p, -3.0, 3.0);
                let dz_prev = sanitize_clamp(dx2_s / scale_p, -3.0, 3.0);
                let pred_null = sanitize(self.pore_self_w[p] * dz_prev);
                let mut pred_act = pred_null;
                if prev.committed {
                    for (m, &da_m) in centered_act.iter().enumerate() {
                        pred_act += self.pore_action_w[m][p] * da_m;
                    }
                }
                let err_null = sanitize_clamp(dz_now - pred_null, -3.0, 3.0);
                let err_act = sanitize_clamp(dz_now - pred_act, -3.0, 3.0);
                pore_err1[p] = err_act;
                if prev.committed && da_norm_sq > 1.5e-3 {
                    let inst_gain = sanitize_clamp(
                        (err_null * err_null - err_act * err_act) / (err_null * err_null + 0.04),
                        -0.40,
                        1.0,
                    );
                    self.pore_agency[p] =
                        sanitize_clamp(0.990 * self.pore_agency[p] + 0.010 * inst_gain, -0.25, 1.0);
                }
                if !quad_carr_ag {
                    self.pore_self_w[p] = sanitize_clamp(
                        0.998 * self.pore_self_w[p] + 0.024 * err_act.clamp(-1.5, 1.5) * dz_prev,
                        -0.95,
                        0.95,
                    );
                    if prev.committed {
                        let lr_p = 0.032 * nlms_scale;
                        let clean_e = err_act.clamp(-1.2, 1.2);
                        for (m, &da_m) in centered_act.iter().enumerate() {
                            let w_old = self.pore_action_w[m][p];
                            let sign_flip =
                                da_m.abs() > 0.14 && dz_now * (w_old * da_m) < -5.0e-3;
                            let dec = if sign_flip {
                                0.94
                            } else if adult_pruned_coch
                                && da_m.abs() < 0.03
                                && w_old >= 0.012
                                && self.on_tonotopic_floor(p)
                            {
                                0.99995
                            } else {
                                0.9990
                            };
                            self.pore_action_w[m][p] =
                                sanitize_clamp(dec * w_old + lr_p * clean_e * da_m, -1.8, 1.8);
                        }
                    }
                }
            }
        }

        if prev.committed {
            let ag_da_ok = !emb_eco || da_norm_sq >= 0.015;
            if ag_da_ok && ag_eval > 0 {
                let base_sq = ag_null_sq.min(ag_shuf_sq);
                if !emb_eco {
                    let imp = base_sq - ag_act_sq - 0.012 * ag_null_sq - 0.04 * ag_contrast_sq;
                    let raw_gain =
                        sanitize_clamp(imp / (ag_null_sq + ag_contrast_sq + 0.02), -0.45, 0.45);
                    self.agency_signed_ema = sanitize_clamp(
                        0.995 * self.agency_signed_ema + 0.005 * raw_gain,
                        -0.08,
                        1.0,
                    );
                    if self.striatal.tick < 120 {
                        self.mean_interventional_agency =
                            sanitize_clamp(self.agency_signed_ema, 0.0, 1.0);
                    }
                } else {
                    let imp = base_sq - ag_act_sq - 0.012 * ag_null_sq;
                    let raw_gain =
                        sanitize_clamp(imp / (ag_null_sq + ag_contrast_sq + 0.015), -1.0, 1.0);
                    self.agency_signed_ema = sanitize_clamp(
                        0.992 * self.agency_signed_ema.max(0.0) + 0.008 * raw_gain,
                        0.0,
                        1.0,
                    );
                    self.mean_interventional_agency = self.agency_signed_ema;
                }
            } else if !emb_eco {
                self.agency_signed_ema =
                    sanitize_clamp(0.99985 * self.agency_signed_ema, -0.08, 1.0);
                if self.striatal.tick < 120 {
                    self.mean_interventional_agency =
                        sanitize_clamp(self.agency_signed_ema, 0.0, 1.0);
                }
            } else {
                self.agency_signed_ema =
                    sanitize_clamp(0.9996 * self.agency_signed_ema.max(0.0), 0.0, 1.0);
                self.mean_interventional_agency = self.agency_signed_ema;
            }
        }

        let mut field_breadth_cal = vec![0u16; ad];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if sp.alive
                && sp.age_ticks >= 16
                && sp.trophic_ema >= 0.025
                && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
            {
                let mut max_ret_p = 0.0_f32;
                for m in 0..ad {
                    let wa = self.pore_action_w[m][p].abs();
                    if wa > max_ret_p {
                        max_ret_p = wa;
                    }
                }
                for (m, fb) in field_breadth_cal.iter_mut().enumerate() {
                    let wa = self.pore_action_w[m][p].abs();
                    if wa >= 0.025 && wa >= 0.50 * max_ret_p {
                        *fb = fb.saturating_add(1);
                    }
                }
            }
        }
        let mut max_breadth_cal = 0u16;
        for &fb in &field_breadth_cal {
            if fb > max_breadth_cal {
                max_breadth_cal = fb;
            }
        }
        let mut is_world_loco_cal = vec![false; ad];
        if max_breadth_cal >= 4 {
            for m in 0..ad {
                if field_breadth_cal[m].saturating_mul(2) >= max_breadth_cal {
                    is_world_loco_cal[m] = true;
                }
            }
        }
        let mut is_world_ret_cal = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                continue;
            }
            if max_breadth_cal < 4 {
                is_world_ret_cal[p] = true;
            } else {
                let mut loco_w_p = 0.0_f32;
                let mut nonloco_w_p = 0.0_f32;
                for m in 0..ad {
                    let wa = self.pore_action_w[m][p].abs();
                    if is_world_loco_cal[m] {
                        if wa > loco_w_p {
                            loco_w_p = wa;
                        }
                    } else if wa > nonloco_w_p {
                        nonloco_w_p = wa;
                    }
                }
                if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                    is_world_ret_cal[p] = true;
                }
            }
            if is_world_ret_cal[p] && self.pore_agency[p] >= 0.020 {
                let h = sp.hist_head as usize;
                let x_now = sp.history[(h + 15) & 15];
                spatial_prox_sum += (x_now - self.pore_mean_v[p]).max(0.0);
                spatial_prox_cnt += 1;
                for (m, la) in loco_auth.iter_mut().enumerate() {
                    *la += self.pore_action_w[m][p].abs();
                }
            }
        }

        let spatial_prox_now = if spatial_prox_cnt > 0 {
            sanitize_clamp(spatial_prox_sum / (spatial_prox_cnt as f32), 0.0, 2.0)
        } else {
            0.0
        };
        let prox_slot = (self.spatial_prox_cursor as usize) & (RING_LEN - 1);
        self.spatial_prox_ring[prox_slot] = spatial_prox_now;
        self.spatial_prox_cursor = ((prox_slot + 1) & (RING_LEN - 1)) as u8;

        let mut max_loco_auth = 0.08_f32;
        for &la in &loco_auth {
            if la > max_loco_auth {
                max_loco_auth = la;
            }
        }
        let mut max_2_6 = vec![0.0_f32; ad];
        let mut sum_abs15 = vec![0.0_f32; ad];
        let mut committed_15 = 0usize;
        for lag in 1..=15 {
            if let Some(ent) = ring.at_lag(lag)
                && ent.committed
            {
                committed_15 += 1;
                for m in 0..ad {
                    let am = ent.motor(m);
                    let a_abs = am.abs();
                    sum_abs15[m] += a_abs;
                    if (2..=6).contains(&lag) && a_abs > max_2_6[m] {
                        max_2_6[m] = a_abs;
                    }
                }
            }
        }

        let mut has_echo_pore = false;
        let mut max_echo_exc = 0.0_f32;
        let mut top_echo_cal_0 = 0.0_f32;
        let mut top_echo_cal_1 = 0.0_f32;
        let mut min_h_cal = [1.0_f32; PRED_BUCKETS];
        let mut pos_coch_w = vec![0.0_f32; ad];
        let mut pos_coch_cnt = vec![0u16; ad];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                continue;
            }
            let h = sp.hist_head as usize;
            let x_now = sp.history[(h + 15) & 15];
            let mean_p = self.pore_mean_v[p];
            if membrane.scout_stats[ch_u].is_sparse_impulse() {
                let abs_now = (x_now - mean_p).abs();
                if self.pore_agency[p] < 0.035 {
                    let pos_now = (x_now - mean_p).max(0.0);
                    if pos_now > top_echo_cal_0 {
                        top_echo_cal_1 = top_echo_cal_0;
                        top_echo_cal_0 = pos_now;
                    } else if pos_now > top_echo_cal_1 {
                        top_echo_cal_1 = pos_now;
                    }
                }
                let mut sum_abs16 = 0.0_f32;
                for &hv in &sp.history {
                    sum_abs16 += (hv - mean_p).abs();
                }
                let exc = abs_now - sum_abs16 * (1.0 / 16.0);
                if !has_echo_pore || exc > max_echo_exc {
                    max_echo_exc = exc;
                    has_echo_pore = true;
                }
                continue;
            }
            let mut min_h = 1.0_f32;
            for &hv in &sp.history {
                if hv < min_h {
                    min_h = hv;
                }
            }
            min_h_cal[p] = min_h;
            if is_world_ret_cal[p] || self.pore_agency[p] < 0.006 {
                continue;
            }
            if self.is_cochlear_floor_pore(p, min_h, mean_p) {
                for m in 0..ad {
                    let pw = self.pore_action_w[m][p];
                    if pw >= 0.015 {
                        pos_coch_w[m] += pw;
                        pos_coch_cnt[m] = pos_coch_cnt[m].saturating_add(1);
                    }
                }
            }
        }

        let ig_cal = self.inv_acoustic_gain();
        let mut ac_ch = [u32::MAX; PRED_BUCKETS];
        let mut ac_u = [[0.0_f32; 6]; PRED_BUCKETS];
        let mut ac_pk = [0.0_f32; PRED_BUCKETS];
        let mut ac_mon_n = 0usize;
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            let h = sp.hist_head as usize;
            if !sp.alive
                || sp.age_ticks < 24
                || ch_u >= n_scout
                || is_world_ret_cal[p]
                || membrane.scout_stats[ch_u].is_sparse_impulse()
                || !self.is_tonotopic_receptor(p)
                || self.is_flat_elevated_plateau(p, &sp.history, h)
            {
                continue;
            }
            let q5_ok =
                self.pore_q5_lvl[p] > 0.90 || (-0.065..=0.085).contains(&self.pore_q5_lvl[p]);
            let zmean = (self.pore_mean_v[p] - self.tonotopic_floor_ema) * ig_cal;
            if !q5_ok || !(-0.020..=0.22).contains(&zmean) {
                continue;
            }
            let mut u_lags = [0.0_f32; 6];
            let mut pk_u = 0.0_f32;
            for (lg, dst) in u_lags.iter_mut().enumerate() {
                let uv = self.deconv_pore_exc(p, &sp.history, h, lg).max(0.0);
                *dst = uv;
                if uv > pk_u {
                    pk_u = uv;
                }
            }
            if let Some(idx) = ac_ch[..ac_mon_n].iter().position(|&c| c == sp.channel) {
                if pk_u > ac_pk[idx] {
                    ac_u[idx] = u_lags;
                    ac_pk[idx] = pk_u;
                }
            } else {
                ac_ch[ac_mon_n] = sp.channel;
                ac_u[ac_mon_n] = u_lags;
                ac_pk[ac_mon_n] = pk_u;
                ac_mon_n += 1;
            }
        }
        if self.has_tonotopic_bank()
            && ac_mon_n >= 2
            && ring.at_lag(12).is_some_and(|e| e.tick > 0)
        {
            let mut b_env = [0.0_f32; 6];
            for (lg, b_dst) in b_env.iter_mut().enumerate() {
                let mut t0 = 0.0_f32;
                let mut t1 = 0.0_f32;
                let mut t2 = 0.0_f32;
                for row in &ac_u[..ac_mon_n] {
                    let v = row[lg];
                    if v > t0 {
                        t2 = t1;
                        t1 = t0;
                        t0 = v;
                    } else if v > t1 {
                        t2 = t1;
                        t1 = v;
                    } else if v > t2 {
                        t2 = v;
                    }
                }
                *b_dst = 0.70 * t1 + 0.30 * t2;
            }
            let b_on = (b_env[0] + b_env[1] + b_env[2]) * (1.0 / 3.0);
            let b_pre = (b_env[3] + b_env[4] + b_env[5]) * (1.0 / 3.0);
            if b_pre < 0.045 {
                let db3 = sanitize_clamp(b_on - b_pre, 0.0, 0.80);
                let mut de_vec = vec![0.0_f32; ad];
                let mut rested_strong = vec![false; ad];
                let mut pred_db = 0.0_f32;
                let mut norm_de = 0.035_f32;
                for m in 0..ad {
                    let u_init = ring.at_lag(13).map_or(0.0_f32, |ent| {
                        if ent.committed && ent.tick > 0 {
                            sanitize_clamp(ent.motor(m), -0.36, 1.0)
                        } else {
                            0.0
                        }
                    });
                    let mut s1 = u_init;
                    let mut s2 = u_init;
                    let mut e_on_sum = 0.0_f32;
                    let mut e_pre_sum = 0.0_f32;
                    let mut e_rest_max = 0.0_f32;
                    for lg in (1..=12).rev() {
                        let u_m = ring.at_lag(lg).map_or(0.0_f32, |ent| {
                            if ent.committed && ent.tick > 0 {
                                sanitize_clamp(ent.motor(m), -0.36, 1.0)
                            } else {
                                0.0
                            }
                        });
                        s1 = 0.36 * s1 + 0.64 * u_m;
                        s2 = 0.36 * s2 + 0.64 * s1;
                        let e_m = (s2 - 0.08).max(0.0);
                        if (4..=12).contains(&lg) && e_m > e_rest_max {
                            e_rest_max = e_m;
                        }
                        if (4..=6).contains(&lg) {
                            e_pre_sum += e_m;
                        } else if (1..=3).contains(&lg) {
                            e_on_sum += e_m;
                        }
                    }
                    let de_m = if e_pre_sum < 0.12 {
                        ((e_on_sum - e_pre_sum) * (1.0 / 3.0)).max(0.0)
                    } else {
                        0.0
                    };
                    de_vec[m] = de_m;
                    rested_strong[m] = e_rest_max < 0.015 && e_on_sum >= 0.42;
                    pred_db += self.vocal_glottal_w[m] * de_m;
                    norm_de += de_m * de_m;
                }
                let err_db = sanitize_clamp(db3 - pred_db, -0.80, 0.80);
                for m in 0..ad {
                    let de_m = de_vec[m];
                    if de_m <= 5.0e-3 {
                        continue;
                    }
                    if err_db >= 0.0 {
                        let du = ((0.048 / norm_de) * err_db * de_m).min(0.022);
                        self.vocal_glottal_w[m] = sanitize_clamp(
                            0.9992 * self.vocal_glottal_w[m] + du,
                            0.0,
                            1.2,
                        );
                    } else if rested_strong[m] {
                        let du_neg = ((0.012 / norm_de) * err_db * de_m).max(-0.008);
                        self.vocal_glottal_w[m] = sanitize_clamp(
                            0.9995 * self.vocal_glottal_w[m] + du_neg,
                            0.0,
                            1.2,
                        );
                    }
                }
            }
        }

        let mut mg_opt = self.primary_glottal_actuator(&is_world_loco_cal, ad);
        if mg_opt.is_none() {
            let mut best_mg_w = 0.05_f32;
            for m in 0..ad {
                if loco_auth[m] < 0.22 * max_loco_auth
                    && pos_coch_cnt[m] >= 2
                    && pos_coch_w[m] > best_mg_w
                {
                    best_mg_w = pos_coch_w[m];
                    mg_opt = Some(m);
                }
            }
        }

        let mut coch_dx1_sum = 0.0_f32;
        let mut coch_dx_prev_sum = 0.0_f32;
        let mut pk_coch_exc = 0.0_f32;
        let mut coch_cnt = 0usize;
        let mut canvas_err_sum = 0.0_f32;
        let mut canvas_cnt = 0usize;
        let mut som_inst = vec![0.0_f32; ad];
        let mut coch_inst = vec![0.0_f32; ad];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
                || is_world_ret_cal[p]
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            let h = sp.hist_head as usize;
            let reaff_mon = self.is_tonotopic_receptor(p)
                && !self.is_flat_elevated_plateau(p, &sp.history, h)
                && self.pore_vocal_reaff[p] >= 0.018
                && (-0.055..=0.075).contains(&self.pore_q5_lvl[p]);
            let is_monaural = reaff_mon
                || if let Some(mg) = mg_opt {
                    self.pore_action_w[mg][p] >= 0.016
                        && self.is_glottal_driven_coch(p, min_h_cal[p], mean_p)
                } else {
                    self.pore_agency[p] >= 0.010
                        && self.is_cochlear_floor_pore(p, min_h_cal[p], mean_p)
                };
            let x_now = sp.history[(h + 15) & 15];
            let x_1 = sp.history[(h + 14) & 15];
            let x_2 = sp.history[(h + 13) & 15];
            if is_monaural {
                coch_dx1_sum += (x_now - x_1).abs() * ig_cal;
                coch_dx_prev_sum += (x_1 - x_2).abs() * ig_cal;
                let exc_now = (x_now - self.receptor_ref_floor(p, min_h_cal[p])).max(0.0);
                if exc_now > pk_coch_exc {
                    pk_coch_exc = exc_now;
                }
                coch_cnt += 1;
                for m in 0..ad {
                    let pwa = self.pore_action_w[m][p].abs();
                    if pwa >= 0.0045 {
                        coch_inst[m] += pwa;
                    }
                }
            } else if self.is_somatic_non_lso(p, min_h_cal[p], mean_p)
                && self.pore_agency[p] >= 0.010
            {
                let loco_coupled = max_breadth_cal >= 4
                    && is_world_loco_cal
                        .iter()
                        .enumerate()
                        .any(|(ml, &il)| il && self.pore_action_w[ml][p].abs() >= 0.013);
                if !loco_coupled {
                    if mean_p >= 0.02 {
                        canvas_err_sum += pore_err1[p];
                        canvas_cnt += 1;
                    }
                    for m in 0..ad {
                        let pwa = self.pore_action_w[m][p].abs();
                        if pwa >= 0.018 {
                            som_inst[m] += pwa;
                        }
                    }
                }
            }
        }

        if has_echo_pore && committed_15 >= 8 {
            let inv_c15 = 1.0 / (committed_15 as f32);
            let echo_surp = sanitize_clamp(max_echo_exc, -0.15, 0.45);
            for m in 0..ad {
                if loco_auth[m] >= 0.22 * max_loco_auth || Some(m) == mg_opt {
                    self.echo_emit_w[m] = sanitize(0.992 * self.echo_emit_w[m]);
                } else {
                    let mean_15 = sum_abs15[m] * inv_c15;
                    let lead_p = sanitize_clamp(max_2_6[m] - mean_15, -0.5, 0.8);
                    let dec = if max_2_6[m] > 0.08 || mean_15 > 0.08 {
                        0.9980
                    } else {
                        0.9996
                    };
                    self.echo_emit_w[m] = sanitize_clamp(
                        dec * self.echo_emit_w[m] + 0.048 * echo_surp * lead_p,
                        -0.4,
                        1.0,
                    );
                }
            }
        }

        if let Some(mg) = mg_opt
            && coch_cnt >= 3
            && committed_15 >= 10
            && let (Some(e1), Some(e2), Some(e3)) = (ring.at_lag(1), ring.at_lag(2), ring.at_lag(3))
            && e1.committed
            && e2.committed
            && e3.committed
            && e1.motor(mg) > 0.12
            && e2.motor(mg) > 0.12
            && e3.motor(mg) > 0.12
        {
            if self.tonotopic_conf_ema >= 0.20
                && e1.motor(mg) >= 0.58
                && e2.motor(mg) >= 0.58
                && e3.motor(mg) >= 0.58
                && pk_coch_exc > 0.18
            {
                let cur_env = if self.acoustic_env_ema < 0.15 {
                    0.84
                } else {
                    self.acoustic_env_ema
                };
                let lr_env = if pk_coch_exc >= cur_env { 0.28 } else { 0.025 };
                self.acoustic_env_ema = sanitize_clamp(
                    (1.0 - lr_env) * cur_env + lr_env * pk_coch_exc,
                    0.36,
                    0.92,
                );
            }
            if (e1.motor(mg) - e2.motor(mg)).abs() < 0.10
                && (e2.motor(mg) - e3.motor(mg)).abs() < 0.10
            {
                let c_flux_surp = sanitize_clamp(
                    (coch_dx1_sum - coch_dx_prev_sum) / (coch_cnt as f32),
                    -0.25,
                    0.35,
                );
                for m in 0..ad {
                    if loco_auth[m] >= 0.22 * max_loco_auth || m == mg {
                        self.vocal_artic_w[m] = sanitize(0.992 * self.vocal_artic_w[m]);
                    } else {
                        let step_now = (e1.motor(m) - e2.motor(m)).abs();
                        let step_prev = (e2.motor(m) - e3.motor(m)).abs();
                        let step_surp = sanitize_clamp(step_now - step_prev, -0.6, 0.8);
                        self.vocal_artic_w[m] = sanitize_clamp(
                            0.9985 * self.vocal_artic_w[m] + 0.085 * c_flux_surp * step_surp,
                            -0.4,
                            1.0,
                        );
                    }
                }
            }
        }

        let canvas_err_mean = if canvas_cnt >= 2 {
            sanitize_clamp(canvas_err_sum / (canvas_cnt as f32), -0.60, 0.60)
        } else {
            0.0
        };
        let lag2_ent = ring.at_lag(2).filter(|e| e.committed);
        for (m, &is_loco_c) in is_world_loco_cal.iter().enumerate() {
            if is_loco_c
                || loco_auth[m] >= 0.06 * max_loco_auth
                || Some(m) == mg_opt
                || (self.vocal_artic_w[m] >= 0.008
                    && coch_inst[m] > som_inst[m]
                    && self.somatic_limb_w[m] < 0.08)
                || self.echo_emit_w[m] >= 0.010
            {
                self.somatic_limb_w[m] = sanitize(0.980 * self.somatic_limb_w[m]);
            } else {
                let inst_drive = if som_inst[m] >= 0.022 && som_inst[m] >= coch_inst[m] {
                    som_inst[m].min(0.60)
                } else {
                    0.0
                };
                let mut w_som = 0.99982 * self.somatic_limb_w[m] + 0.0014 * inst_drive;
                if canvas_cnt >= 2
                    && committed_15 >= 8
                    && let Some(e2) = lag2_ent
                {
                    let da2_m = sanitize_clamp(e2.motor(m) - self.act_mean[m], -1.0, 1.0);
                    w_som += 0.0020 * canvas_err_mean * da2_m;
                }
                self.somatic_limb_w[m] = sanitize_clamp(w_som, -0.20, 1.0);
            }
        }

        let mut emit_opt: Option<usize> = None;
        let mut best_ew = 0.010_f32;
        for (m, &ew) in self.echo_emit_w.iter().enumerate() {
            if loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt && ew > best_ew {
                best_ew = ew;
                emit_opt = Some(m);
            }
        }
        let mut quiet_echo_before_6 = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut last_emit_pulse = 0.0_f32;
        if quiet_echo_before_6 {
            for lag in 8..=16 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..ad {
                        let is_emit_cand = if let Some(em) = emit_opt {
                            m == em
                        } else {
                            loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt
                        };
                        let am = ent.motor(m);
                        if is_emit_cand && am.abs() > 0.12 {
                            quiet_echo_before_6 = false;
                            if (8..=13).contains(&lag) && am.abs() > last_emit_pulse.abs() {
                                last_emit_pulse = am;
                            }
                        }
                    }
                }
            }
        }

        let max_echo_raw_now = top_echo_cal_0.hypot(top_echo_cal_1);
        let mut quiet_echo_cal_14 = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        if quiet_echo_cal_14 {
            for lag in 1..=14 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..ad {
                        let is_emit_cand = if let Some(em) = emit_opt {
                            m == em
                        } else {
                            loco_auth[m] < 0.22 * max_loco_auth && Some(m) != mg_opt
                        };
                        if is_emit_cand && ent.motor(m).abs() > 0.12 {
                            quiet_echo_cal_14 = false;
                        }
                    }
                }
            }
        }
        let span_e = (self.peer_echo_age as usize).saturating_add(1);
        let d_echo_cal = if quiet_echo_cal_14
            && max_echo_raw_now > 0.018
            && self.peer_echo_fast > 0.016
            && (3..=34).contains(&span_e)
        {
            (max_echo_raw_now - self.peer_echo_fast).clamp(-0.22, 0.22)
        } else {
            0.0
        };
        let ring_warm_cal = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut q_cal_lag = [ring_warm_cal; 17];
        if let Some(mg) = mg_opt
            && ring_warm_cal
        {
            for (d, ql) in q_cal_lag.iter_mut().enumerate().skip(1) {
                *ql = ring
                    .at_lag(d)
                    .is_some_and(|e| !e.committed || e.motor(mg) <= -0.05);
            }
        }
        let quiet_self_cal_7 = ring_warm_cal
            && q_cal_lag[1]
            && q_cal_lag[2]
            && q_cal_lag[3]
            && q_cal_lag[4]
            && q_cal_lag[5]
            && q_cal_lag[6]
            && q_cal_lag[7];

        let peer_call_at_lag = |stim_lag: usize| -> (Option<f32>, f32) {
            let mut top4_cal = [0.0_f32; 4];
            let mut seen_coch_ch = [u32::MAX; PRED_BUCKETS];
            let mut seen_coch_n = 0usize;
            let mut coch_strong_cnt = 0usize;
            let mut coch_weak_cnt = 0usize;
            let mut cnt_c = 0usize;

            let mut top4_lso_pos = [0.0_f32; 4];
            let mut top4_lso_neg = [0.0_f32; 4];
            let mut top4_clean_pos = [0.0_f32; 4];
            let mut top4_clean_neg = [0.0_f32; 4];
            let mut lso_pos_cnt = 0usize;
            let mut lso_neg_cnt = 0usize;
            let mut clean_pos_cnt = 0usize;
            let mut clean_neg_cnt = 0usize;
            let mut lso_signed_sum = 0.0_f32;
            let mut seen_lso_ch = [u32::MAX; PRED_BUCKETS];
            let mut seen_lso_n = 0usize;

            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                    || is_world_ret_cal[p]
                {
                    continue;
                }
                let mean_p = self.pore_mean_v[p];
                let is_monaural_coch = if let Some(mg) = mg_opt {
                    self.pore_action_w[mg][p] >= 0.016
                        && self.is_glottal_driven_coch(p, min_h_cal[p], mean_p)
                } else {
                    self.pore_agency[p] >= 0.010
                        && self.is_cochlear_floor_pore(p, min_h_cal[p], mean_p)
                };
                let h = sp.hist_head as usize;
                if is_monaural_coch {
                    cnt_c += 1;
                    if seen_coch_ch[..seen_coch_n].contains(&sp.channel) {
                        continue;
                    }
                    seen_coch_ch[seen_coch_n] = sp.channel;
                    seen_coch_n += 1;
                    let u0 = self.deconv_pore_exc(p, &sp.history, h, stim_lag).max(0.0);
                    let u1 = self
                        .deconv_pore_exc(p, &sp.history, h, stim_lag + 1)
                        .max(0.0);
                    let u2 = self
                        .deconv_pore_exc(p, &sp.history, h, stim_lag + 2)
                        .max(0.0);
                    let u_sus3 = if u0 > 0.0 && u1 > 0.0 && u2 > 0.0 {
                        u0.min(1.30 * u1 + 0.004).min(1.65 * u2 + 0.006)
                    } else {
                        0.0
                    };
                    if u_sus3 > 0.038 {
                        coch_strong_cnt += 1;
                    }
                    if u_sus3 > 0.016 {
                        coch_weak_cnt += 1;
                    }
                    Self::insert_top4(&mut top4_cal, u_sus3);
                    continue;
                }
                if sp.age_ticks < 40 {
                    continue;
                }
                let mut somatic_other_w = 0.0_f32;
                for m in 0..ad {
                    if loco_auth[m] < 0.22 * max_loco_auth
                        && Some(m) != mg_opt
                        && self.vocal_artic_w[m] < 0.005
                    {
                        somatic_other_w += self.pore_action_w[m][p].abs();
                    }
                }
                let is_coch_trough = self.is_coch_trough(p, min_h_cal[p], mean_p)
                    && mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                let is_glottal_pos = mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
                if is_coch_trough
                    || is_glottal_pos
                    || !self.is_lso_floor_pore(p, min_h_cal[p], mean_p)
                    || somatic_other_w >= 0.038
                    || seen_lso_ch[..seen_lso_n].contains(&sp.channel)
                {
                    continue;
                }
                seen_lso_ch[seen_lso_n] = sp.channel;
                seen_lso_n += 1;
                let bg = self.pore_bg_ild[p];
                let d0 = self.deconv_pore_sym(p, &sp.history, h, stim_lag) - bg;
                let d1 = self.deconv_pore_sym(p, &sp.history, h, stim_lag + 1) - bg;
                let d2 = self.deconv_pore_sym(p, &sp.history, h, stim_lag + 2) - bg;
                let d3 = self.deconv_pore_sym(p, &sp.history, h, stim_lag + 3) - bg;
                if d0.abs() > 0.015 {
                    lso_signed_sum += d0;
                }
                if d0 * d1 > 0.0 && d1 * d2 > 0.0 && d2 * d3 > 0.0 {
                    let sus_d = d0.abs().min(d1.abs()).min(d2.abs()).min(d3.abs());
                    if sus_d > 0.018 {
                        if d0 > 0.0 {
                            lso_pos_cnt += 1;
                            Self::insert_top4(&mut top4_lso_pos, sus_d);
                        } else {
                            lso_neg_cnt += 1;
                            Self::insert_top4(&mut top4_lso_neg, sus_d);
                        }
                    }
                    if sp.age_ticks >= 48 && bg.abs() < 0.018 && sus_d > 0.022 {
                        if d0 > 0.0 {
                            clean_pos_cnt += 1;
                            Self::insert_top4(&mut top4_clean_pos, sus_d);
                        } else {
                            clean_neg_cnt += 1;
                            Self::insert_top4(&mut top4_clean_neg, sus_d);
                        }
                    }
                }
            }
            let (dom_sign, lso_coh_cnt, top4_lso_dom, lso_clean_cnt, top4_lso_clean) =
                if lso_pos_cnt > lso_neg_cnt {
                    (
                        1.0_f32,
                        lso_pos_cnt,
                        top4_lso_pos,
                        clean_pos_cnt,
                        top4_clean_pos,
                    )
                } else if lso_neg_cnt > lso_pos_cnt {
                    (
                        -1.0_f32,
                        lso_neg_cnt,
                        top4_lso_neg,
                        clean_neg_cnt,
                        top4_clean_neg,
                    )
                } else if lso_signed_sum >= 0.0 {
                    (
                        1.0_f32,
                        lso_pos_cnt,
                        top4_lso_pos,
                        clean_pos_cnt,
                        top4_clean_pos,
                    )
                } else {
                    (
                        -1.0_f32,
                        lso_neg_cnt,
                        top4_lso_neg,
                        clean_neg_cnt,
                        top4_clean_neg,
                    )
                };
            let harm_opt = if cnt_c >= 2 {
                let lvl = 0.5 * (top4_cal[0] + top4_cal[1]);
                let ok = (coch_strong_cnt >= 2
                    && lvl > 0.040
                    && lso_coh_cnt >= 3
                    && top4_lso_dom[2] > 0.018)
                    || (coch_strong_cnt >= 1
                        && top4_cal[0] > 0.058
                        && lso_coh_cnt >= 4
                        && top4_lso_dom[3] > 0.020)
                    || (coch_weak_cnt >= 2
                        && lvl > 0.018
                        && lso_clean_cnt >= 5
                        && top4_lso_clean[3] > 0.022);
                if ok { Some(lvl) } else { None }
            } else {
                None
            };
            (harm_opt, dom_sign)
        };

        let span_c = (self.peer_coch_age as usize).saturating_add(1);
        let (peer_lvl_cal_opt, dom_sign_cal) =
            if quiet_self_cal_7 && self.peer_coch_fast > 0.016 && (3..=15).contains(&span_c) {
                peer_call_at_lag(0)
            } else {
                (None, 0.0)
            };
        let d_coch_cal =
            peer_lvl_cal_opt.map_or(0.0, |lvl| (lvl - self.peer_coch_fast).clamp(-0.22, 0.22));

        let mda_over_span = |span: usize| -> (Vec<f32>, usize, f32) {
            let mut mda = vec![0.0_f32; ad];
            let mut cnt = 0usize;
            let s_clamped = span.clamp(1, 15);
            let inv_s = 1.0 / (s_clamped as f32);
            for lag in 1..=s_clamped {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    cnt += 1;
                    for m in 0..ad {
                        mda[m] = sanitize(mda[m] + (ent.motor(m) - self.act_mean[m]) * inv_s);
                    }
                }
            }
            let dp = if spatial_prox_cnt > 0 {
                let past_p =
                    self.spatial_prox_ring[(prox_slot + RING_LEN - s_clamped) & (RING_LEN - 1)];
                spatial_prox_now - past_p
            } else {
                0.0
            };
            (mda, cnt, dp)
        };

        if d_coch_cal.abs() > 0.0015 {
            let (cal_mda_c, cnt_mc, d_prox_c) = mda_over_span(span_c);
            let app_sig_c = sanitize_clamp(
                6.2 * d_coch_cal
                    + if d_coch_cal * d_prox_c > 0.0 {
                        2.0 * d_prox_c
                    } else {
                        0.0
                    },
                -1.2,
                1.2,
            );
            if cnt_mc >= 3 && app_sig_c.abs() >= 0.008 {
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 40
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || is_world_ret_cal[p]
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                    {
                        continue;
                    }
                    let mean_p = self.pore_mean_v[p];
                    let is_monaural_coch = mg_opt.is_some_and(|mg| {
                        self.pore_action_w[mg][p] >= 0.016
                            && self.is_glottal_driven_coch(p, min_h_cal[p], mean_p)
                    });
                    let is_coch_trough = self.is_coch_trough(p, min_h_cal[p], mean_p)
                        && mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                    let mut somatic_other_w = 0.0_f32;
                    for m in 0..ad {
                        if loco_auth[m] < 0.22 * max_loco_auth
                            && Some(m) != mg_opt
                            && self.vocal_artic_w[m] < 0.005
                        {
                            somatic_other_w += self.pore_action_w[m][p].abs();
                        }
                    }
                    if is_monaural_coch
                        || is_coch_trough
                        || mg_opt.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008)
                        || !self.is_lso_floor_pore(p, min_h_cal[p], mean_p)
                        || somatic_other_w >= 0.038
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let bg = self.pore_bg_ild[p];
                    let d_now = self.deconv_pore_sym(p, &sp.history, h, 0) - bg;
                    let d_past = if span_c <= 13 {
                        self.deconv_pore_sym(p, &sp.history, h, span_c) - bg
                    } else if q_cal_lag[span_c.min(16)] {
                        self.z_from_bank(p, min_h_cal[p], sp.history[(h + 15 - span_c.min(15)) & 15])
                            - bg
                    } else {
                        0.0
                    };
                    if d_now * dom_sign_cal <= 0.018
                        || (d_past.abs() > 0.012 && d_now * d_past <= 0.0)
                    {
                        continue;
                    }
                    let d_mid = if d_past.abs() > 0.012 {
                        0.5 * (d_now + d_past)
                    } else {
                        d_now
                    };
                    let s_lso = sanitize_clamp(6.5 * d_mid, -2.5, 2.5);
                    if s_lso.abs() < 0.14 {
                        continue;
                    }
                    for (m, &mda) in cal_mda_c.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9994 * self.tectal_map_w[m][p] + 0.022 * app_sig_c * s_lso * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
                }
            }
        }

        if d_echo_cal.abs() > 0.0015 {
            let (cal_mda_e, cnt_me, d_prox_e) = mda_over_span(span_e);
            let app_sig_e = sanitize_clamp(
                5.4 * d_echo_cal
                    + if d_echo_cal * d_prox_e > 0.0 {
                        2.0 * d_prox_e
                    } else {
                        0.0
                    },
                -1.2,
                1.2,
            );
            if cnt_me >= 3 && app_sig_e.abs() >= 0.008 {
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || !membrane.scout_stats[ch_u].is_sparse_impulse()
                        || self.pore_agency[p] >= 0.035
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let mean_p = self.pore_mean_v[p];
                    let scale_p = self.pore_scale_v[p].max(0.015);
                    let dev_now = (sp.history[(h + 15) & 15] - mean_p).max(0.0);
                    let dev_lag1 = (sp.history[(h + 14) & 15] - mean_p).max(0.0);
                    let dev_use = dev_now.max(0.75 * dev_lag1);
                    if dev_use <= 0.018 {
                        continue;
                    }
                    let s_echo = sanitize_clamp(dev_use / scale_p, 0.0, 2.5);
                    if s_echo < 0.14 {
                        continue;
                    }
                    for (m, &mda) in cal_mda_e.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9994 * self.tectal_map_w[m][p] + 0.026 * app_sig_e * s_echo * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
                }
            }
        } else if !quiet_echo_before_6 && last_emit_pulse > 0.12 {
            let (cal_mda_s, cnt_ms, d_prox_s) = mda_over_span(10);
            if cnt_ms >= 4 && d_prox_s > 0.008 {
                let app_sig_s = sanitize_clamp(3.2 * d_prox_s, 0.0, 0.85);
                for p in 0..pore_cap {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout
                        || !membrane.scout_stats[ch_u].is_sparse_impulse()
                        || self.pore_agency[p] >= 0.035
                    {
                        continue;
                    }
                    let h = sp.hist_head as usize;
                    let mean_p = self.pore_mean_v[p];
                    let scale_p = self.pore_scale_v[p].max(0.015);
                    let mut peak_dev = 0.0_f32;
                    for stim_lag in 6..=11 {
                        let dv = (sp.history[(h + 15 - stim_lag) & 15] - mean_p).max(0.0);
                        if dv > peak_dev {
                            peak_dev = dv;
                        }
                    }
                    if peak_dev <= 0.022 {
                        continue;
                    }
                    let s_echo = sanitize_clamp(peak_dev / scale_p, 0.0, 2.5);
                    for (m, &mda) in cal_mda_s.iter().enumerate() {
                        if loco_auth[m] >= 0.22 * max_loco_auth && mda.abs() > 0.018 {
                            self.tectal_map_w[m][p] = sanitize_clamp(
                                0.9995 * self.tectal_map_w[m][p] + 0.014 * app_sig_s * s_echo * mda,
                                -1.8,
                                1.8,
                            );
                        }
                    }
                }
            }
        }

        if prev.committed {
            for mp in &mut membrane.motor_pores {
                if !mp.alive {
                    continue;
                }
                let ch = mp.actuator_channel as usize;
                if ch >= ad {
                    continue;
                }
                let myelin_m = (motor_myelin[ch] * 1.8).clamp(0.0, 1.0);
                if centered_act[ch].abs() > 0.03 {
                    let r2_m = if muscle_null_sq[ch] > 1.0e-5 {
                        (muscle_err_gain[ch] / muscle_null_sq[ch]).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let jac_m = (muscle_sum_jac[ch] * 2.2).clamp(0.0, 1.0);
                    let target_m =
                        sanitize_clamp(0.35 * r2_m + 0.35 * jac_m + 0.30 * myelin_m, 0.0, 1.0);
                    let alpha = if target_m >= mp.motor_mastery {
                        0.0012
                    } else {
                        0.00025
                    };
                    mp.motor_mastery = sanitize_clamp(
                        (1.0 - alpha) * mp.motor_mastery + alpha * target_m,
                        0.0,
                        1.0,
                    );
                }
            }
        }
    }

    pub fn apply_delayed_credit_and_morphogenesis(
        &mut self,
        ring: &EfferenceRing,
        cells: &mut CellArena,
        membrane: &mut PoreMembrane,
        body: &BodyCore,
        max_tubes: usize,
        rng: &mut Rng64,
    ) -> (u64, u64, u64) {
        let ad = self.act_dim();
        let mut splits = 0u64;
        let mut shortcuts = 0u64;
        let mut atrophied = 0u64;
        let mut polychronous_sprouted = false;

        let now_seq_target = cells.active_indices.iter().copied().find_map(|ci| {
            let c = cells.slots.get(ci as usize)?;
            if c.alive && c.sensory_trophic_ema > 0.008 && c.v_real.abs() > 0.16 {
                Some((ci, c.v_real))
            } else {
                None
            }
        });

        let scale = body.valence_scale_ema.max(1.5e-4);
        let raw_somatic_valence = body.fast_valence;
        let ctx_span = body.context_valence_span();

        let mut embodied_m_glottal: Option<usize> = None;
        let mut is_monaural_cr = [false; PRED_BUCKETS];
        let mut is_lso_cr = [false; PRED_BUCKETS];
        let mut is_world_retina_cr = [false; PRED_BUCKETS];
        let mut embodied_pga: Option<usize> = None;
        if self.has_tonotopic_bank() && ring.at_lag(1).is_some_and(|e| e.tick >= 64) {
            let pore_cap_cr = PRED_BUCKETS.min(membrane.sensor_pores.len());
            let n_scout_cr = membrane.scout_stats.len();
            let act_cap_cr = membrane.motor_pores.len().min(ad);
            let mut field_breadth_cr = vec![0u16; ad];
            for p in 0..pore_cap_cr {
                let sp = &membrane.sensor_pores[p];
                if sp.alive
                    && sp.age_ticks >= 16
                    && sp.trophic_ema >= 0.025
                    && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
                {
                    let mut max_ret_p = 0.0_f32;
                    for m in 0..act_cap_cr {
                        let wa = self.pore_action_w[m][p].abs();
                        if wa > max_ret_p {
                            max_ret_p = wa;
                        }
                    }
                    for (m, fb) in field_breadth_cr.iter_mut().enumerate().take(act_cap_cr) {
                        let wa = self.pore_action_w[m][p].abs();
                        if wa >= 0.025 && wa >= 0.50 * max_ret_p {
                            *fb = fb.saturating_add(1);
                        }
                    }
                }
            }
            let mut max_breadth_cr = 0u16;
            for &fb in &field_breadth_cr[..act_cap_cr] {
                if fb > max_breadth_cr {
                    max_breadth_cr = fb;
                }
            }
            let mut is_world_loco_cr = vec![false; ad];
            if max_breadth_cr >= 4 {
                for m in 0..act_cap_cr {
                    if field_breadth_cr[m].saturating_mul(2) >= max_breadth_cr {
                        is_world_loco_cr[m] = true;
                    }
                }
            }
            for (p, ret_flag) in is_world_retina_cr.iter_mut().enumerate().take(pore_cap_cr) {
                let sp = &membrane.sensor_pores[p];
                if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                    continue;
                }
                if max_breadth_cr < 4 {
                    *ret_flag = true;
                } else {
                    let mut loco_w_p = 0.0_f32;
                    let mut nonloco_w_p = 0.0_f32;
                    for m in 0..act_cap_cr {
                        let wa = self.pore_action_w[m][p].abs();
                        if is_world_loco_cr[m] {
                            if wa > loco_w_p {
                                loco_w_p = wa;
                            }
                        } else if wa > nonloco_w_p {
                            nonloco_w_p = wa;
                        }
                    }
                    if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                        *ret_flag = true;
                    }
                }
            }
            let mut is_glottal_cand_cr = [false; PRED_BUCKETS];
            for (p, gc_flag) in is_glottal_cand_cr.iter_mut().enumerate().take(pore_cap_cr) {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout_cr
                    || is_world_retina_cr[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                let mut min_h = 1.0_f32;
                for &hv in &sp.history {
                    if hv < min_h {
                        min_h = hv;
                    }
                }
                if self.pore_agency[p] >= 0.006
                    && self.is_cochlear_floor_pore(p, min_h, self.pore_mean_v[p])
                {
                    *gc_flag = true;
                }
            }
            let mut max_asw_cr = vec![0.0_f32; ad];
            let mut tot_reaff_cr = vec![0.0_f32; ad];
            let mut pos_coch_w_cr = vec![0.0_f32; ad];
            let mut pos_coch_cnt_cr = vec![0u16; ad];
            let mut soft_coch_w_cr = vec![0.0_f32; ad];
            let mut soft_coch_cnt_cr = vec![0u16; ad];
            let mut is_pure_coch_cr = [false; PRED_BUCKETS];
            let mut is_quiet_coch_cr = [false; PRED_BUCKETS];
            let mut min_h_cr = [1.0_f32; PRED_BUCKETS];
            for p in 0..pore_cap_cr {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout_cr
                    || is_world_retina_cr[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                let mut min_h = 1.0_f32;
                for &hv in &sp.history {
                    if hv < min_h {
                        min_h = hv;
                    }
                }
                min_h_cr[p] = min_h;
                let mean_p = self.pore_mean_v[p];
                if self.is_pure_coch_floor(p, min_h, mean_p) {
                    is_pure_coch_cr[p] = true;
                }
                if self.is_quiet_coch_floor(p, min_h, mean_p) {
                    is_quiet_coch_cr[p] = true;
                }
            }
            for m in 0..act_cap_cr {
                for &w in &self.action_sensory_w[m] {
                    let wa = w.abs();
                    if wa > max_asw_cr[m] {
                        max_asw_cr[m] = wa;
                    }
                }
                for p in 0..pore_cap_cr {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 16
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout_cr
                    {
                        continue;
                    }
                    let pw = self.pore_action_w[m][p];
                    let pwa = pw.abs();
                    if self.pore_agency[p] >= 0.010 && pwa >= 0.015 {
                        tot_reaff_cr[m] += pwa;
                    }
                    if is_glottal_cand_cr[p] && pw >= 0.015 {
                        pos_coch_w_cr[m] += pw;
                        pos_coch_cnt_cr[m] = pos_coch_cnt_cr[m].saturating_add(1);
                    }
                    if (is_pure_coch_cr[p] && pw >= 0.0045) || (is_quiet_coch_cr[p] && pw >= 0.0120)
                    {
                        soft_coch_w_cr[m] += pw;
                        soft_coch_cnt_cr[m] = soft_coch_cnt_cr[m].saturating_add(1);
                    }
                }
            }
            embodied_pga = self.primary_glottal_actuator(&is_world_loco_cr, act_cap_cr);
            embodied_m_glottal = embodied_pga;
            if embodied_m_glottal.is_none() {
                let mut best_gw = 0.05_f32;
                for m in 0..act_cap_cr {
                    if !is_world_loco_cr[m]
                        && (tot_reaff_cr[m] >= 0.045 || max_asw_cr[m] >= 0.065)
                        && pos_coch_cnt_cr[m] >= 2
                        && pos_coch_w_cr[m] > best_gw
                    {
                        best_gw = pos_coch_w_cr[m];
                        embodied_m_glottal = Some(m);
                    }
                }
            }
            if embodied_m_glottal.is_none() {
                let mut best_myelin = 0.14_f32;
                for m in 0..act_cap_cr {
                    if is_world_loco_cr[m] {
                        continue;
                    }
                    let (ep, op, _, _, ec_tot, oc_tot, _, om_b) =
                        self.motor_contingency_stats(membrane, m);
                    let pos_ok = soft_coch_w_cr[m] >= 0.008;
                    let em = if pos_ok && ec_tot >= 3 { ep } else { 0.0 };
                    let om_pos = if pos_ok && oc_tot >= 3 { op } else { 0.0 };
                    let om_min = om_b[0].min(om_b[1]).min(om_b[2]).min(om_b[3]);
                    let om_neg = if oc_tot >= 3
                        && ep < 0.28
                        && om_min <= -0.20
                        && (soft_coch_cnt_cr[m] >= 1
                            || tot_reaff_cr[m] >= 0.012
                            || max_asw_cr[m] >= 0.038)
                        && (soft_coch_w_cr[m] >= 0.004
                            || tot_reaff_cr[m] >= 0.012
                            || max_asw_cr[m] >= 0.038)
                    {
                        -0.85 * om_min
                    } else {
                        0.0
                    };
                    let peak_m = em.max(om_pos).max(om_neg);
                    if peak_m > best_myelin {
                        best_myelin = peak_m;
                        embodied_m_glottal = Some(m);
                    }
                }
            }
            if embodied_m_glottal.is_none() {
                let mut best_sw = 0.018_f32;
                for m in 0..act_cap_cr {
                    if !is_world_loco_cr[m]
                        && (tot_reaff_cr[m] >= 0.020
                            || max_asw_cr[m] >= 0.038
                            || (soft_coch_cnt_cr[m] >= 4 && soft_coch_w_cr[m] >= 0.048))
                        && soft_coch_cnt_cr[m] >= 2
                        && soft_coch_w_cr[m] > best_sw
                    {
                        best_sw = soft_coch_w_cr[m];
                        embodied_m_glottal = Some(m);
                    }
                }
            }
            if let Some(mg) = embodied_m_glottal {
                for p in 0..pore_cap_cr {
                    let sp = &membrane.sensor_pores[p];
                    let reaff_mon = self.is_tonotopic_receptor(p)
                        && !self.is_flat_elevated_plateau(p, &sp.history, sp.hist_head as usize)
                        && self.pore_vocal_reaff[p] >= 0.018
                        && (-0.055..=0.075).contains(&self.pore_q5_lvl[p]);
                    is_monaural_cr[p] = reaff_mon
                        || (is_pure_coch_cr[p] && self.pore_action_w[mg][p] >= 0.0050)
                        || (is_quiet_coch_cr[p] && self.pore_action_w[mg][p] >= 0.0120);
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 40
                        || sp.trophic_ema < 0.025
                        || ch_u >= n_scout_cr
                        || is_world_retina_cr[p]
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                        || is_monaural_cr[p]
                    {
                        continue;
                    }
                    let mean_p = self.pore_mean_v[p];
                    let is_coch_trough = self.is_coch_trough(p, min_h_cr[p], mean_p)
                        && self.pore_action_w[mg][p] < -0.016;
                    let is_glottal_pos = self.pore_action_w[mg][p] >= 0.008;
                    if is_coch_trough
                        || is_glottal_pos
                        || !self.is_lso_floor_pore(p, min_h_cr[p], mean_p)
                    {
                        continue;
                    }
                    let mut somatic_other_w = 0.0_f32;
                    for m in 0..act_cap_cr {
                        if !is_world_loco_cr[m] && m != mg && self.vocal_artic_w[m] < 0.005 {
                            somatic_other_w += self.pore_action_w[m][p].abs();
                        }
                    }
                    if somatic_other_w < 0.038 {
                        is_lso_cr[p] = true;
                    }
                }
            }
        }
        let mg_contingent_active = embodied_m_glottal.is_some_and(|mg| {
            let (ep, op, _, _, ec_t, oc_t, _, _) = self.motor_contingency_stats(membrane, mg);
            (ec_t >= 3 && ep >= 0.18) || (oc_t >= 3 && op >= 0.24)
        });

        for lag in 1..=RING_LEN {
            let Some(entry) = ring.at_lag(lag) else {
                continue;
            };
            if entry.tick == 0 {
                continue;
            }
            let kernel = (-((lag.saturating_sub(1)) as f32) / 2.2).exp();
            let raw_diff = sanitize(raw_somatic_valence - entry.raw_valence_at_commit);
            let post_harvest_refractory = (raw_somatic_valence >= 0.0
                && entry.raw_valence_at_commit > 0.12
                && raw_diff < -0.08)
                || entry.tick == 0;
            let interval_delta = if post_harvest_refractory {
                0.0
            } else {
                raw_diff
            };
            let local_level =
                0.25 * (raw_somatic_valence.abs() + entry.raw_valence_at_commit.abs()) + 1.0e-4;
            let base_level = if interval_delta > 0.0 && entry.raw_valence_at_commit < 0.0 {
                (0.22 * ctx_span).max(0.15 * scale)
            } else {
                local_level.max(0.18 * ctx_span).max(0.15 * scale)
            };
            let denom = interval_delta.abs() + base_level;
            let weber_diff = interval_delta / denom;
            let pain_term = if raw_somatic_valence < 0.0 && raw_diff < 0.0 {
                2.0 * raw_diff
            } else {
                0.0
            };
            let td_adv = sanitize_clamp(1.75 * weber_diff + pain_term, -2.0, 2.0);

            if raw_somatic_valence.abs() > 1.0e-4 || td_adv.abs() > 0.02 {
                for &(c_idx, v_past, dv_past, pre_attn) in &entry.active_cells {
                    let Some(cell) = cells.slots.get_mut(c_idx as usize) else {
                        continue;
                    };
                    if !cell.alive || cell.sensory_trophic_ema < 0.005 {
                        continue;
                    }
                    let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                    let lr_v = 0.04 * kernel * attn;
                    let target_vh = td_adv * v_past;
                    let decay_vh = if cell.v_homeo * target_vh < -1.0e-4 {
                        0.12
                    } else {
                        0.015 * (td_adv.abs() + 0.02)
                    };
                    cell.v_homeo = sanitize_clamp(
                        (1.0 - decay_vh) * cell.v_homeo + lr_v * target_vh,
                        -2.0,
                        2.0,
                    );
                    let target_vdh = td_adv * dv_past;
                    let decay_vdh = if cell.v_delta_homeo * target_vdh < -1.0e-4 {
                        0.12
                    } else {
                        0.015 * (td_adv.abs() + 0.02)
                    };
                    cell.v_delta_homeo = sanitize_clamp(
                        (1.0 - decay_vdh) * cell.v_delta_homeo + 1.4 * lr_v * target_vdh,
                        -2.0,
                        2.0,
                    );
                    if raw_somatic_valence > 0.0 {
                        cell.atp_credit_ema = sanitize_clamp(
                            cell.atp_credit_ema
                                + 0.05 * kernel * attn * raw_somatic_valence * v_past.abs(),
                            0.0,
                            4.0,
                        );
                    }
                }
            }

            let lag_u8 = lag as u8;
            let seq_kernel = (-((lag.saturating_sub(1)) as f32) / 5.5).exp();
            let check_myelin = (entry.tick & 7) == 0;
            let faster_entry = (check_myelin && lag > 1)
                .then(|| ring.at_lag(lag - 1))
                .flatten();
            let slower_entry = (check_myelin && lag < RING_LEN)
                .then(|| ring.at_lag(lag + 1))
                .flatten();
            let mut myelin_shift_requests: Vec<(u32, u8)> = Vec::new();

            // Multi-syllable / long-horizon polychronous CellToCell sequence learning across 100..320 ms (lag 5..=16),
            // leaving fast sensorimotor lags 1..=4 to the committed cortical-motor loop below.
            if lag >= 5 {
                for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
                    let pre_sig = sanitize(v_past + 0.25 * dv_past);
                    if pre_sig.abs() < 1.0e-3 {
                        continue;
                    }
                    let v_faster = faster_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                    let v_slower = slower_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                    let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize).cloned() else {
                        continue;
                    };
                    for t_idx in out_list {
                        let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                            continue;
                        };
                        if !tube.alive || tube.kind != TubeKind::CellToCell || tube.delay != lag_u8
                        {
                            continue;
                        }
                        tube.age_ticks = tube.age_ticks.saturating_add(1);
                        let Some(dst_cell) = cells.slots.get(tube.post as usize) else {
                            continue;
                        };
                        if !dst_cell.alive {
                            continue;
                        }
                        let post_now = dst_cell.v_real;
                        let pred_err = sanitize_clamp(dst_cell.pred_err, -1.5, 1.5);
                        tube.w_pred = sanitize_clamp(
                            0.9995 * tube.w_pred + 0.015 * pred_err * pre_sig,
                            -0.30,
                            0.30,
                        );
                        if entry.committed || raw_somatic_valence.abs() > 0.01 {
                            let du = 0.010 * seq_kernel * td_adv * pre_sig * post_now
                                - 0.002 * (td_adv.abs() + 0.02) * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                        }
                        tube.usage_ema = sanitize_clamp(
                            0.985 * tube.usage_ema + 0.015 * (pre_sig * post_now).abs(),
                            0.0,
                            2.0,
                        );
                        if check_myelin
                            && tube.usage_ema > 0.025
                            && (tube.u[0].abs() + tube.w_pred.abs()) > 0.055
                        {
                            let pol = if tube.u[0].abs() >= 0.02 {
                                tube.u[0].signum()
                            } else {
                                tube.w_pred.signum()
                            };
                            let c_curr = pol * pre_sig * post_now;
                            let c_fast = pol * v_faster * post_now;
                            let c_slow = pol * v_slower * post_now;
                            if c_fast > c_curr + 0.015 || c_slow > c_curr + 0.015 {
                                let d_mu = 0.22 * tube.usage_ema.min(1.0) * (c_fast - c_slow);
                                tube.myelin_plasticity_accum = sanitize_clamp(
                                    0.95 * tube.myelin_plasticity_accum + d_mu,
                                    -1.5,
                                    1.5,
                                );
                                if tube.myelin_plasticity_accum >= 1.0 && tube.delay > 1 {
                                    myelin_shift_requests.push((t_idx, tube.delay - 1));
                                    tube.myelin_plasticity_accum = 0.0;
                                } else if tube.myelin_plasticity_accum <= -1.0
                                    && (tube.delay as usize) < RING_LEN
                                {
                                    myelin_shift_requests.push((t_idx, tube.delay + 1));
                                    tube.myelin_plasticity_accum = 0.0;
                                }
                            } else {
                                tube.myelin_plasticity_accum =
                                    sanitize(0.98 * tube.myelin_plasticity_accum);
                            }
                        }
                    }
                }
                for (t_idx, new_d) in myelin_shift_requests.drain(..) {
                    let (pre, post, old_d) = {
                        let t = &self.slots[t_idx as usize];
                        (t.pre, t.post, t.delay)
                    };
                    if new_d != old_d && !self.has_cell_tube_at_delay(pre, post, new_d) {
                        self.slots[t_idx as usize].delay = new_d;
                        if new_d < old_d {
                            shortcuts += 1;
                        }
                    }
                }

                if !polychronous_sprouted
                    && (lag == 6 || lag == 10 || lag == 14)
                    && raw_somatic_valence > 0.06
                    && td_adv > 0.05
                    && ((entry.tick.wrapping_add(lag as u64)) & 15) == 0
                    && self.alive_count + 16 <= max_tubes
                    && !entry.active_cells.is_empty()
                    && let Some((dst_cell, dst_v)) = now_seq_target
                {
                    let n_act = entry.active_cells.len();
                    let rot = (entry.tick as usize) % n_act;
                    for off in 0..n_act {
                        let (up_cell, up_v, _, _) = entry.active_cells[(rot + off) % n_act];
                        if up_cell == dst_cell || up_v.abs() <= 0.18 {
                            continue;
                        }
                        if self
                            .outgoing_by_cell
                            .get(up_cell as usize)
                            .map_or(0, Vec::len)
                            >= 36
                        {
                            continue;
                        }
                        if !self.has_cell_tube_at_delay(up_cell, dst_cell, lag_u8) {
                            let sign = up_v.signum() * dst_v.signum();
                            let mut seq_tube = Tube::newborn(
                                TubeKind::CellToCell,
                                up_cell,
                                dst_cell,
                                lag_u8,
                                0.022 * sign,
                            );
                            seq_tube.w_pred = 0.08 * sign;
                            if self.add_tube(seq_tube, max_tubes).is_some() {
                                shortcuts += 1;
                                polychronous_sprouted = true;
                            }
                            break;
                        }
                    }
                }
            }

            if !entry.committed || lag > 4 {
                continue;
            }

            if lag == 1
                && let Some(prev_entry) = ring.at_lag(2)
                && prev_entry.committed
            {
                let mut dot = 0.0_f32;
                for m in 0..ad {
                    dot += entry.motor(m) * prev_entry.motor(m);
                }
                if dot.abs() > 0.02 && td_adv.abs() > 0.02 {
                    for mp in &mut membrane.motor_pores {
                        if mp.alive {
                            mp.klinokinesis_gain = sanitize_clamp(
                                0.985 * mp.klinokinesis_gain + 0.038 * td_adv * dot.signum(),
                                -0.75,
                                0.75,
                            );
                        }
                    }
                }
            }

            let base_k_td = (cells.alive_count / 2).max(cells.alive_count.min(24));
            let uncap_k_td = base_k_td.clamp(4, 96).min(cells.alive_count.max(1));
            let cap_k_td = cells.k_winners_target().max(1);
            let ap_scale_td = if cells.alive_count > 128 && uncap_k_td > cap_k_td {
                (uncap_k_td as f32) / (cap_k_td as f32)
            } else {
                1.0
            };
            let mut td_past = vec![0.0_f32; ad];
            let mut max_reward_jac = vec![0.0_f32; ad];
            for &(c_idx, v_past, dv_past, _) in &entry.active_cells {
                let mut cell_is_apical = false;
                if let Some(cell) = cells.slots.get(c_idx as usize)
                    && cell.alive
                {
                    cell_is_apical = cell.tau >= 16.0;
                    if (cell.v_homeo.abs() + cell.v_delta_homeo.abs()) > 0.12 {
                        let b = c_idx as usize;
                        for (m, dst) in max_reward_jac.iter_mut().enumerate() {
                            let j = self.action_sensory_w[m][b].abs();
                            if j > *dst {
                                *dst = j;
                            }
                        }
                    }
                }
                let pre_sig = sanitize(v_past + 0.25 * dv_past);
                if pre_sig.abs() < 1.0e-3 {
                    continue;
                }
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                    continue;
                };
                for &t_idx in out_list {
                    let Some(tube) = self.slots.get(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive || tube.kind != TubeKind::CellToMotor || tube.delay != lag_u8 {
                        continue;
                    }
                    let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                        continue;
                    };
                    if !mp.alive {
                        continue;
                    }
                    let act_ch = mp.actuator_channel as usize;
                    if act_ch < ad {
                        let g = tube.effective_g(&cells.slots);
                        let term = if cell_is_apical && ap_scale_td > 1.0 {
                            g * pre_sig * mp.conductance * ap_scale_td
                        } else {
                            g * pre_sig * mp.conductance
                        };
                        td_past[act_ch] = sanitize(td_past[act_ch] + term);
                    }
                }
            }

            let mut split_requests: Vec<(u32, f32)> = Vec::new();
            let mut shortcut_requests: Vec<(u32, u32, f32)> = Vec::new();

            for &(c_idx, v_past, dv_past, pre_attn) in &entry.active_cells {
                let pre_sig = sanitize(v_past + 0.25 * dv_past);
                if pre_sig.abs() < 1.0e-3 {
                    continue;
                }
                let v_faster = faster_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                let v_slower = slower_entry.map_or(0.0, |e| e.cell_signal(c_idx));
                let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize).cloned() else {
                    continue;
                };

                for t_idx in out_list {
                    let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                        continue;
                    };
                    if !tube.alive {
                        continue;
                    }
                    tube.age_ticks = tube.age_ticks.saturating_add(1);

                    match tube.kind {
                        TubeKind::CellToCell if tube.delay == lag_u8 => {
                            let Some(dst_cell) = cells.slots.get(tube.post as usize) else {
                                continue;
                            };
                            if !dst_cell.alive {
                                continue;
                            }
                            let post_now = dst_cell.v_real;
                            let pred_err =
                                sanitize_clamp(post_now - tube.w_pred * pre_sig, -2.0, 2.0);
                            tube.w_pred =
                                sanitize_clamp(tube.w_pred + 0.035 * pred_err * pre_sig, -2.0, 2.0);
                            let du = 0.02 * kernel * attn * td_adv * pre_sig * post_now
                                - 0.004 * (td_adv.abs() + 0.02) * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                            tube.usage_ema = sanitize_clamp(
                                0.98 * tube.usage_ema + 0.02 * (pre_sig * post_now).abs(),
                                0.0,
                                2.0,
                            );
                            if check_myelin
                                && tube.usage_ema > 0.025
                                && (tube.u[0].abs() + tube.w_pred.abs()) > 0.055
                            {
                                let pol = if tube.u[0].abs() >= 0.02 {
                                    tube.u[0].signum()
                                } else {
                                    tube.w_pred.signum()
                                };
                                let c_curr = pol * pre_sig * post_now;
                                let c_fast = pol * v_faster * post_now;
                                let c_slow = pol * v_slower * post_now;
                                if c_fast > c_curr + 0.015 || c_slow > c_curr + 0.015 {
                                    let d_mu = 0.22 * tube.usage_ema.min(1.0) * (c_fast - c_slow);
                                    tube.myelin_plasticity_accum = sanitize_clamp(
                                        0.95 * tube.myelin_plasticity_accum + d_mu,
                                        -1.5,
                                        1.5,
                                    );
                                    if tube.myelin_plasticity_accum >= 1.0 && tube.delay > 1 {
                                        myelin_shift_requests.push((t_idx, tube.delay - 1));
                                        tube.myelin_plasticity_accum = 0.0;
                                    } else if tube.myelin_plasticity_accum <= -1.0
                                        && (tube.delay as usize) < RING_LEN
                                    {
                                        myelin_shift_requests.push((t_idx, tube.delay + 1));
                                        tube.myelin_plasticity_accum = 0.0;
                                    }
                                } else {
                                    tube.myelin_plasticity_accum =
                                        sanitize(0.98 * tube.myelin_plasticity_accum);
                                }
                            }
                        }
                        TubeKind::CellToMotor if lag == tube.delay as usize => {
                            let m_slot = tube.post as usize;
                            let (act_ch, mp_alive) = match membrane.motor_pores.get(m_slot) {
                                Some(mp) => (mp.actuator_channel as usize, mp.alive),
                                None => continue,
                            };
                            if !mp_alive || act_ch >= ad {
                                continue;
                            }
                            let a_emitted = entry.motor(act_ch);
                            if a_emitted.abs() < 1.0e-3 {
                                continue;
                            }
                            let a_det = smooth_saturate(2.35 * td_past[act_ch]);
                            let a_innov = a_emitted - a_det;
                            if td_adv > 0.03
                                && a_innov.abs() > 0.02
                                && let Some(mp_mut) = membrane.motor_pores.get_mut(m_slot)
                            {
                                mp_mut.conductance = sanitize_clamp(
                                    0.994 * mp_mut.conductance + 0.006 * 0.95,
                                    0.1,
                                    1.5,
                                );
                            }

                            let elig = pre_sig * a_emitted;
                            let eff_td_adv = if Some(act_ch) == embodied_m_glottal
                                && mg_contingent_active
                            {
                                if (a_emitted < 0.0 && td_adv > 0.0)
                                    || (a_emitted > 0.05
                                        && (entry.raw_valence_at_commit > 0.012
                                            || raw_somatic_valence >= -0.010)
                                        && raw_diff < 0.0)
                                {
                                    0.0
                                } else {
                                    td_adv
                                }
                            } else if Some(act_ch) == embodied_m_glottal
                                && a_emitted > 0.05
                                && tube.myelin_plasticity_accum >= 0.18
                                && (entry.raw_valence_at_commit - body.baseline_valence_ema) > 0.015
                                && (raw_somatic_valence - body.baseline_valence_ema) >= -0.015
                                && raw_diff < 0.0
                            {
                                0.0
                            } else {
                                td_adv
                            };
                            let raw_target = elig * eff_td_adv;
                            let reward_coupled = max_reward_jac[act_ch] > 0.009;
                            let boot_thresh = if reward_coupled { 0.028 } else { 0.12 };
                            let bootstrap =
                                ((tube.u[0].abs() - boot_thresh) / 0.07).clamp(0.0, 0.65);
                            let eff_act = if eff_td_adv >= 0.0 {
                                bootstrap * a_emitted + (1.0 - bootstrap) * a_innov
                            } else {
                                a_emitted
                            };
                            let credit_target = attn * pre_sig * eff_act * eff_td_adv;
                            let g_eff = tube.effective_g(&cells.slots);
                            let stiff = tube.u[2] + tube.u[3];
                            let active_branch =
                                g_eff.abs() >= 0.02 || tube.context_gate_cell.is_none();
                            if stiff * raw_target < -0.025 {
                                if active_branch {
                                    let lability = if eff_td_adv < -0.10 { 0.86 } else { 0.93 };
                                    tube.u[1] = sanitize(tube.u[1] * lability);
                                    tube.u[2] = sanitize(tube.u[2] * lability);
                                    tube.u[3] = sanitize(tube.u[3] * (0.5 + 0.5 * lability));
                                }
                                if active_branch && tube.stiff_strength() > 0.045 {
                                    tube.contradiction_accum =
                                        sanitize_clamp(tube.contradiction_accum + 0.28, 0.0, 2.0);
                                    if tube.contradiction_accum >= 1.0 {
                                        split_requests.push((t_idx, raw_target.signum() * 0.58));
                                        tube.contradiction_accum = 0.0;
                                    }
                                }
                                if let Some(ctx_id) = tube.context_gate_cell
                                    && tube.gate_invert
                                    && let Some(ctx_c) = cells.slots.get_mut(ctx_id as usize)
                                    && ctx_c.alive
                                {
                                    ctx_c.v_real = smooth_saturate(ctx_c.v_real + 0.30);
                                }
                            } else if stiff * raw_target > 0.01 {
                                tube.contradiction_accum =
                                    sanitize_clamp(tube.contradiction_accum - 0.32, 0.0, 2.0);
                            } else {
                                tube.contradiction_accum =
                                    sanitize(tube.contradiction_accum * 0.92);
                            }

                            let gate_scale = if tube.context_gate_cell.is_some() {
                                if g_eff.abs() < 1.0e-4 && tube.u[0].abs() > 1.0e-3 {
                                    0.0
                                } else {
                                    1.35
                                }
                            } else {
                                1.0
                            };
                            if gate_scale > 0.0 {
                                let oja_decay = 0.065 * (eff_td_adv.abs() + 0.025) * tube.u[0];
                                let lr_tube = if reward_coupled { 0.34 } else { 0.18 };
                                let du = gate_scale * lr_tube * kernel * credit_target - oja_decay;
                                tube.apply_benna_fusi_impulse(du, 1.0);
                                tube.w_pred = sanitize_clamp(
                                    (1.0 - 0.06 * (eff_td_adv.abs() + 0.025)) * tube.w_pred
                                        + gate_scale * 0.24 * kernel * credit_target,
                                    -1.5,
                                    1.5,
                                );
                            }
                            tube.usage_ema = sanitize_clamp(
                                0.97 * tube.usage_ema + 0.03 * (pre_sig * a_emitted).abs(),
                                0.0,
                                2.0,
                            );

                            if eff_td_adv > 0.02
                                && let Some(prev_e) = ring.at_lag(lag + 1)
                                && let Some(&(up_cell, up_v, _, _)) = prev_e.active_cells.first()
                                && up_cell != c_idx
                                && up_v.abs() > 0.08
                                && cells
                                    .slots
                                    .get(c_idx as usize)
                                    .is_some_and(|cc| cc.surprise_ema < 0.25)
                            {
                                if tube.last_upstream_cell == up_cell {
                                    tube.chain_score =
                                        sanitize_clamp(tube.chain_score + 0.28, 0.0, 2.0);
                                    if tube.chain_score >= 1.0 {
                                        shortcut_requests.push((
                                            up_cell,
                                            tube.post,
                                            0.18 * up_v.signum() * a_emitted.signum(),
                                        ));
                                        tube.chain_score = 0.0;
                                    }
                                } else {
                                    tube.last_upstream_cell = up_cell;
                                    tube.chain_score = 0.25;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            for (t_idx, new_d) in myelin_shift_requests {
                let (pre, post, old_d) = {
                    let t = &self.slots[t_idx as usize];
                    (t.pre, t.post, t.delay)
                };
                if new_d != old_d && !self.has_cell_tube_at_delay(pre, post, new_d) {
                    self.slots[t_idx as usize].delay = new_d;
                    if new_d < old_d {
                        shortcuts += 1;
                    }
                }
            }

            if !split_requests.is_empty() {
                let ctx_cell = cells.find_slow_context_cell().unwrap_or(0);
                if let Some(ctx_c) = cells.slots.get_mut(ctx_cell as usize)
                    && ctx_c.alive
                {
                    ctx_c.v_real = smooth_saturate(ctx_c.v_real + 0.75);
                }
                for (old_t_idx, new_u0) in split_requests {
                    let (kind, pre, post, delay, target_sib_invert) = {
                        let old = &mut self.slots[old_t_idx as usize];
                        let sib_inv = if let Some(existing_ctx) = old.context_gate_cell {
                            if existing_ctx == ctx_cell {
                                !old.gate_invert
                            } else {
                                old.context_gate_cell = Some(ctx_cell);
                                old.gate_invert = true;
                                false
                            }
                        } else {
                            old.context_gate_cell = Some(ctx_cell);
                            old.gate_invert = true;
                            false
                        };
                        old.u[0] = sanitize(old.u[0] * 0.35);
                        (old.kind, old.pre, old.post, old.delay, sib_inv)
                    };
                    let mut reused = false;
                    if let Some(out_list) = self.outgoing_by_cell.get(pre as usize).cloned() {
                        for tid in out_list {
                            if tid != old_t_idx
                                && let Some(sib) = self.slots.get_mut(tid as usize)
                                && sib.alive
                                && sib.kind == kind
                                && sib.post == post
                                && sib.delay == delay
                                && sib.gate_invert == target_sib_invert
                            {
                                sib.context_gate_cell = Some(ctx_cell);
                                sib.u[0] = new_u0;
                                sib.w_pred = 0.8 * new_u0;
                                splits += 1;
                                reused = true;
                                break;
                            }
                        }
                    }
                    if !reused && self.alive_count + 2 <= max_tubes {
                        let mut sibling = Tube::newborn(kind, pre, post, delay, new_u0);
                        sibling.w_pred = 0.8 * new_u0;
                        sibling.context_gate_cell = Some(ctx_cell);
                        sibling.gate_invert = target_sib_invert;
                        if self.add_tube(sibling, max_tubes).is_some() {
                            splits += 1;
                        }
                    }
                    if pre != ctx_cell && !self.has_direct_tube(TubeKind::CellToCell, pre, ctx_cell)
                    {
                        let ctx_drive = Tube::newborn(TubeKind::CellToCell, pre, ctx_cell, 1, 0.42);
                        let _ = self.add_tube(ctx_drive, max_tubes);
                    }
                    if kind == TubeKind::CellToMotor
                        && !self.has_direct_tube(TubeKind::CellToMotor, ctx_cell, post)
                    {
                        let brake = Tube::newborn(TubeKind::CellToMotor, ctx_cell, post, 1, -0.16);
                        let _ = self.add_tube(brake, max_tubes);
                    }
                }
            }

            for (up_cell, motor_post, init_u0) in shortcut_requests {
                if !self.has_direct_tube(TubeKind::CellToMotor, up_cell, motor_post) {
                    let shortcut =
                        Tube::newborn(TubeKind::CellToMotor, up_cell, motor_post, 1, init_u0);
                    if self.add_tube(shortcut, max_tubes).is_some() {
                        shortcuts += 1;
                    }
                }
            }
        }

        let locked_mg_cr = if (self.vocal_last_mg as usize) < ad {
            Some(self.vocal_last_mg as usize)
        } else {
            None
        };
        let mut eff_now_gl = false;
        if ring.at_lag(1).is_some_and(|e| e.tick > 0) {
            if let Some(mg) = embodied_m_glottal {
                let mg_tag = (mg.min(254)) as u8;
                let primary_locked = embodied_pga == Some(mg);
                if primary_locked && self.vocal_last_mg != mg_tag {
                    self.vocal_last_mg = mg_tag;
                    self.vocal_q5_cnt = 0;
                    self.vocal_y_ema = 0.0;
                    self.vocal_f_ema = [0.0; 4];
                    self.vocal_blk_sum = [0.0; 4];
                    self.vocal_s_acc = [0.0; 4];
                    self.vocal_v_acc = [0.0; 4];
                    self.even_tick_cnt = 0;
                    self.even_y_ema = 0.0;
                    self.even_f_ema = [0.0; 4];
                    self.even_blk_sum = [0.0; 4];
                    self.even_s_acc = [0.0; 4];
                    self.even_v_acc = [0.0; 4];
                    self.pore_vocal_reaff = [0.0; PRED_BUCKETS];
                    self.vocal_eff_s1 = 0.0;
                    self.vocal_eff_s2 = 0.0;
                    self.vocal_eff_hist = 0;
                    self.vocal_onset_hist = 0;
                    for lg in (2..=RING_LEN).rev() {
                        let u_hist = ring.at_lag(lg).map_or(0.0_f32, |e| {
                            if e.committed && e.tick > 0 {
                                sanitize_clamp(e.motor(mg), -0.36, 1.0)
                            } else {
                                0.0
                            }
                        });
                        let p_eff = (self.vocal_eff_hist & 1) != 0;
                        self.vocal_eff_s1 =
                            sanitize_clamp(0.36 * self.vocal_eff_s1 + 0.64 * u_hist, -0.36, 1.0);
                        self.vocal_eff_s2 = sanitize_clamp(
                            0.36 * self.vocal_eff_s2 + 0.64 * self.vocal_eff_s1,
                            -0.36,
                            1.0,
                        );
                        let e_now = self.vocal_eff_s2 >= 0.095;
                        let o_now = e_now && !p_eff;
                        self.vocal_eff_hist = (self.vocal_eff_hist << 1) | u32::from(e_now);
                        self.vocal_onset_hist =
                            ((self.vocal_onset_hist << 1) | u32::from(o_now)) & 0x03FF_FFFF;
                    }
                    for tube in &mut self.slots {
                        if tube.alive && tube.kind == TubeKind::CellToMotor && (tube.pre & 1) != 0 {
                            tube.myelin_plasticity_accum = 0.0;
                        }
                    }
                }
            }
            let active_lmg = if (self.vocal_last_mg as usize) < ad {
                Some(self.vocal_last_mg as usize)
            } else {
                None
            };
            let u_gl = if let Some(mg) = active_lmg {
                ring.at_lag(1).map_or(0.0_f32, |e| {
                    if e.committed {
                        sanitize_clamp(e.motor(mg), -0.36, 1.0)
                    } else {
                        0.0
                    }
                })
            } else {
                0.0_f32
            };
            let prev_eff = (self.vocal_eff_hist & 1) != 0;
            self.vocal_eff_s1 = sanitize_clamp(0.36 * self.vocal_eff_s1 + 0.64 * u_gl, -0.36, 1.0);
            self.vocal_eff_s2 = sanitize_clamp(
                0.36 * self.vocal_eff_s2 + 0.64 * self.vocal_eff_s1,
                -0.36,
                1.0,
            );
            let eff_now = self.vocal_eff_s2 >= 0.095 && active_lmg.is_some();
            eff_now_gl = eff_now;
            let onset_now = eff_now_gl && !prev_eff;
            let next_refrac = (self.vocal_onset_hist >> 26).saturating_sub(1);
            let hist_26 = ((self.vocal_onset_hist << 1) | u32::from(onset_now)) & 0x03FF_FFFF;
            self.vocal_eff_hist = (self.vocal_eff_hist << 1) | u32::from(eff_now);
            self.vocal_onset_hist = (next_refrac << 26) | hist_26;
        }

        if self.has_tonotopic_bank() && ring.at_lag(6).is_some_and(|e| e.tick >= 64) {
            let active_lmg = if (self.vocal_last_mg as usize) < ad {
                Some(self.vocal_last_mg as usize)
            } else {
                locked_mg_cr
            };
            let q5_id = if let Some(mg_id) = active_lmg {
                let eff_v_id = Self::motor_efference_active_lags(ring, mg_id);
                (self.vocal_eff_hist & 0x1F) == 0 && !(1..=5).any(|d| eff_v_id[d])
            } else {
                ring.at_lag(1).is_some_and(|e| !e.committed)
            };
            let pore_cap_v = PRED_BUCKETS.min(membrane.sensor_pores.len());
            let n_scout_v = membrane.scout_stats.len();
            let ig_v = self.inv_acoustic_gain();
            let fl_bank = self.tonotopic_floor_ema;

            if eff_now_gl {
                for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_v) {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    let h = sp.hist_head as usize;
                    if !sp.alive {
                        self.pore_vocal_reaff[p] = 0.0;
                        continue;
                    }
                    let q5_ready = (-0.055..=0.075).contains(&self.pore_q5_lvl[p]);
                    let q5_bad = self.pore_q5_lvl[p] <= 0.90 && !q5_ready;
                    if ret_v || !self.is_tonotopic_receptor(p) || q5_bad {
                        if self.pore_vocal_reaff[p].abs() > 1.0e-4 {
                            self.pore_vocal_reaff[p] = sanitize(0.80 * self.pore_vocal_reaff[p]);
                        }
                        continue;
                    }
                    if sp.age_ticks < 32
                        || ch_u >= n_scout_v
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                        || !q5_ready
                    {
                        continue;
                    }
                    if self.is_flat_elevated_plateau(p, &sp.history, h) {
                        self.pore_vocal_reaff[p] = sanitize(0.85 * self.pore_vocal_reaff[p]);
                        continue;
                    }
                    let zmean = (self.pore_mean_v[p] - fl_bank) * ig_v;
                    if !(-0.020..=0.22).contains(&zmean) {
                        continue;
                    }
                    let u0 = self.deconv_pore_exc(p, &sp.history, h, 0).max(0.0);
                    let base_r = self.pore_q5_lvl[p].max(0.0).max(0.75 * self.pore_q5_exaff[p]);
                    self.pore_vocal_reaff[p] = sanitize_clamp(
                        0.96 * self.pore_vocal_reaff[p] + 0.04 * (u0 - base_r),
                        -0.50,
                        1.65,
                    );
                }
            }

            for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_v) {
                if !ret_v
                    && self.is_tonotopic_receptor(p)
                    && self.pore_vocal_reaff[p] >= 0.018
                    && (-0.055..=0.075).contains(&self.pore_q5_lvl[p])
                {
                    let sp_keep = &mut membrane.sensor_pores[p];
                    if sp_keep.alive
                        && !self.is_flat_elevated_plateau(
                            p,
                            &sp_keep.history,
                            sp_keep.hist_head as usize,
                        )
                    {
                        if sp_keep.trophic_ema < 0.055 {
                            sp_keep.trophic_ema = 0.055;
                        }
                        if sp_keep.conductance < 0.075 {
                            sp_keep.conductance = 0.075;
                        }
                    }
                }
            }

            if q5_id {
                for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_v) {
                    let is_ton_p = self.is_tonotopic_receptor(p);
                    let sp_mut = &mut membrane.sensor_pores[p];
                    let ch_u = sp_mut.channel as usize;
                    if !sp_mut.alive
                        || sp_mut.age_ticks < 16
                        || ch_u >= n_scout_v
                        || ret_v
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                        || (sp_mut.trophic_ema < 0.025 && !is_ton_p)
                    {
                        continue;
                    }
                    let h = sp_mut.hist_head as usize;
                    let hist_p = sp_mut.history;
                    let x_now = hist_p[(h + 15) & 15];
                    let x_1 = hist_p[(h + 14) & 15];
                    let z_now = (x_now - fl_bank) * ig_v;
                    let z_1 = (x_1 - fl_bank) * ig_v;
                    let z_cur = z_now.min(z_1);
                    let u0 = self.deconv_pore_exc(p, &hist_p, h, 0).max(0.0);
                    if self.pore_q5_lvl[p] > 0.90 {
                        self.pore_q5_lvl[p] = sanitize_clamp(z_cur, -0.80, 0.85);
                        self.pore_q5_exaff[p] = sanitize_clamp(u0, 0.0, 1.65);
                    } else {
                        let w_q5 = (1.0 / (1.0 + 80.0 * (z_now - z_1).abs())).max(0.15);
                        let lr_ql = if z_cur <= self.pore_q5_lvl[p] {
                            0.12 * w_q5
                        } else {
                            0.0025
                        };
                        self.pore_q5_lvl[p] = sanitize_clamp(
                            (1.0 - lr_ql) * self.pore_q5_lvl[p] + lr_ql * z_cur,
                            -0.80,
                            0.85,
                        );
                        self.pore_q5_exaff[p] = sanitize_clamp(
                            0.985 * self.pore_q5_exaff[p] + 0.015 * u0,
                            0.0,
                            1.65,
                        );
                    }
                }
            }
        }

        let mut even_sensitive_open = body.rest_consolidation_ticks < 9_000;
        if !even_sensitive_open && self.has_tonotopic_bank() {
            if let Some(mg) = embodied_m_glottal {
                let (e_peak, _, _, _, _, _, _, _) = self.motor_contingency_stats(membrane, mg);
                if e_peak >= 0.72 {
                    even_sensitive_open = true;
                }
            }
            if !even_sensitive_open {
                for tube in &mut self.slots {
                    if tube.alive
                        && tube.kind == TubeKind::CellToMotor
                        && (tube.pre & 1) == 0
                        && tube.myelin_plasticity_accum != 0.0
                    {
                        let decayed = sanitize(0.99 * tube.myelin_plasticity_accum);
                        tube.myelin_plasticity_accum =
                            if decayed.abs() < 1.0e-3 { 0.0 } else { decayed };
                    }
                }
            }
        }

        if let Some(mg) = embodied_m_glottal.or(locked_mg_cr)
            && ring.at_lag(16).is_some_and(|e| e.tick > 0)
        {
            const EVEN_LO: [usize; 4] = [2, 5, 8, 10];
            const EVEN_HI: [usize; 4] = [4, 7, 10, 12];
            let eff_v = Self::motor_efference_active_lags(ring, mg);
            let fwd_on = |lg: usize| -> bool { eff_v[lg] && !eff_v[lg + 1] };

            let val_prev1 = ring
                .at_lag(1)
                .map_or(raw_somatic_valence, |e| e.raw_valence_at_commit);
            let val_prev2 = ring
                .at_lag(2)
                .map_or(val_prev1, |e| e.raw_valence_at_commit);
            let d_val_1 = raw_somatic_valence - val_prev1;
            let level_jump = raw_somatic_valence - body.baseline_valence_ema;
            let rim_reentry = raw_somatic_valence < 0.24
                && (3..=15).any(|d| {
                    ring.at_lag(d)
                        .is_some_and(|e| e.raw_valence_at_commit > 0.16)
                });
            let raw_trophic_burst = if !rim_reentry
                && ((level_jump > 0.032 && d_val_1 > 0.042)
                    || (raw_somatic_valence > 0.18 && d_val_1 > 0.068))
                && (val_prev1 >= -0.025 || (raw_somatic_valence > 0.055 && d_val_1 > 0.085))
                && val_prev2 < raw_somatic_valence - 0.042
            {
                sanitize_clamp(9.2 * d_val_1, 0.0, 1.0)
            } else {
                0.0
            };
            let trophic_burst = if even_sensitive_open {
                raw_trophic_burst
            } else {
                0.0
            };

            let mut step_even_ext = false;
            if !body.is_resting() && ring.at_lag(16).is_some_and(|e| e.tick >= 64) {
                let y_even = if raw_trophic_burst > 0.05 {
                    1.0_f32
                } else {
                    0.0_f32
                };
                if self.even_tick_cnt >= 64 {
                    let y_tilde = y_even - self.even_y_ema;
                    for b in 0..4 {
                        let x_b = if (EVEN_LO[b]..=EVEN_HI[b]).any(&fwd_on) {
                            1.0_f32
                        } else {
                            0.0_f32
                        };
                        let x_tilde = x_b - self.even_f_ema[b];
                        self.even_blk_sum[b] = sanitize_clamp(
                            self.even_blk_sum[b] + y_tilde * x_tilde,
                            -64.0,
                            64.0,
                        );
                        self.even_f_ema[b] = sanitize_clamp(
                            0.985 * self.even_f_ema[b] + 0.015 * x_b,
                            0.0,
                            1.0,
                        );
                    }
                    self.even_y_ema =
                        sanitize_clamp(0.985 * self.even_y_ema + 0.015 * y_even, 0.0, 1.0);
                    if (self.even_tick_cnt & 63) == 63 {
                        for b in 0..4 {
                            let bs = self.even_blk_sum[b];
                            self.even_blk_sum[b] = 0.0;
                            self.even_s_acc[b] = sanitize_clamp(
                                0.9994 * self.even_s_acc[b] + bs,
                                -5_000.0,
                                5_000.0,
                            );
                            self.even_v_acc[b] = sanitize_clamp(
                                0.9994 * self.even_v_acc[b] + bs * bs,
                                0.0,
                                50_000.0,
                            );
                        }
                    }
                } else {
                    self.even_y_ema =
                        sanitize_clamp(0.92 * self.even_y_ema + 0.08 * y_even, 0.0, 1.0);
                    for b in 0..4 {
                        let x_b = if (EVEN_LO[b]..=EVEN_HI[b]).any(&fwd_on) {
                            1.0_f32
                        } else {
                            0.0_f32
                        };
                        self.even_f_ema[b] = sanitize_clamp(
                            0.92 * self.even_f_ema[b] + 0.08 * x_b,
                            0.0,
                            1.0,
                        );
                    }
                }
                step_even_ext = (self.even_tick_cnt & 3) == 3;
                self.even_tick_cnt = self.even_tick_cnt.saturating_add(1);
            }

            let mut even_falsified = [false; 4];
            if self.even_tick_cnt >= 9_000 {
                for b in 0..4 {
                    let zb = self.even_s_acc[b] / (self.even_v_acc[b] + 0.16).sqrt();
                    if self.even_v_acc[b] >= 6.0 && zb < 2.85 {
                        even_falsified[b] = true;
                    }
                }
            }

            if step_even_ext
                && (even_falsified[0]
                    || even_falsified[1]
                    || even_falsified[2]
                    || even_falsified[3])
            {
                for tube in &mut self.slots {
                    if !tube.alive
                        || tube.kind != TubeKind::CellToMotor
                        || (tube.pre & 1) != 0
                        || tube.myelin_plasticity_accum <= 0.0
                    {
                        continue;
                    }
                    let b = ((tube.pre >> 1) & 3) as usize;
                    if !even_falsified[b] {
                        continue;
                    }
                    if membrane
                        .motor_pores
                        .get(tube.post as usize)
                        .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                    {
                        let decayed = sanitize(0.92 * tube.myelin_plasticity_accum);
                        tube.myelin_plasticity_accum =
                            if decayed < 1.0e-3 { 0.0 } else { decayed };
                        for uk in &mut tube.u {
                            if *uk > 0.0 {
                                *uk = sanitize(0.90 * *uk);
                            }
                        }
                        if tube.w_pred > 0.0 {
                            tube.w_pred = sanitize(0.90 * tube.w_pred);
                        }
                    }
                }
            }

            if trophic_burst > 0.05 {
                let mut even_delta = [0.0_f32; 4];
                let mut even_hit_lag: [Option<usize>; 4] = [None; 4];
                let mut any_even_upd = false;
                for b in 0..4 {
                    if let Some(lag_on) = (EVEN_LO[b]..=EVEN_HI[b]).find(|&lg| fwd_on(lg)) {
                        if !even_falsified[b] {
                            even_delta[b] = 0.150 * trophic_burst;
                            even_hit_lag[b] = Some(lag_on);
                            any_even_upd = true;
                        }
                    } else if !((EVEN_LO[b] - 1)..=(EVEN_HI[b] + 1)).any(&fwd_on) {
                        even_delta[b] = -0.072 * trophic_burst;
                        any_even_upd = true;
                    }
                }
                if any_even_upd {
                    let mut es_b = [0.0_f32; 4];
                    let mut ec_b = [0usize; 4];
                    for tube in &mut self.slots {
                        if !tube.alive || tube.kind != TubeKind::CellToMotor || (tube.pre & 1) != 0
                        {
                            continue;
                        }
                        if membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                        {
                            let b = ((tube.pre >> 1) & 3) as usize;
                            if even_delta[b] != 0.0 {
                                tube.myelin_plasticity_accum = sanitize_clamp(
                                    tube.myelin_plasticity_accum + even_delta[b],
                                    -0.30,
                                    1.0,
                                );
                            }
                            es_b[b] += tube.myelin_plasticity_accum;
                            ec_b[b] += 1;
                        }
                    }
                    let mut best_even_hit: Option<usize> = None;
                    let mut best_em_b = 0.14_f32;
                    for b in 0..4 {
                        let em_b = if ec_b[b] > 0 {
                            es_b[b] / (ec_b[b] as f32)
                        } else {
                            0.0
                        };
                        if em_b > best_em_b
                            && let Some(lag_on) = even_hit_lag[b]
                        {
                            best_em_b = em_b;
                            best_even_hit = Some(lag_on);
                        }
                    }
                    if let Some(lag_on) = best_even_hit
                        && let Some(on_entry) = ring.at_lag(lag_on)
                    {
                        for &(c_idx, v_past, dv_past, pre_attn) in &on_entry.active_cells {
                            if (c_idx & 1) != 0 {
                                continue;
                            }
                            let pre_sig = sanitize(v_past + 0.25 * dv_past);
                            if pre_sig <= 0.02 {
                                continue;
                            }
                            let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                            let out_len = self
                                .outgoing_by_cell
                                .get(c_idx as usize)
                                .map_or(0, Vec::len);
                            for k in 0..out_len {
                                let t_idx = self.outgoing_by_cell[c_idx as usize][k];
                                let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                                    continue;
                                };
                                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                                    continue;
                                }
                                let is_mg =
                                    membrane.motor_pores.get(tube.post as usize).is_some_and(
                                        |mp| mp.alive && mp.actuator_channel as usize == mg,
                                    );
                                if !is_mg {
                                    continue;
                                }
                                let du = 0.055 * attn * trophic_burst * pre_sig - 0.070 * tube.u[0];
                                tube.apply_benna_fusi_impulse(du, 1.0);
                                tube.w_pred = sanitize_clamp(
                                    0.95 * tube.w_pred + 0.065 * attn * trophic_burst * pre_sig,
                                    -1.5,
                                    1.5,
                                );
                            }
                        }
                    }
                }
            }

            const ODD_LO: [usize; 4] = [5, 9, 13, 17];
            const ODD_HI: [usize; 4] = [10, 15, 20, 26];
            let onset_bits = self.vocal_onset_hist;
            let eff_bits = self.vocal_eff_hist;
            let on_bit = |lg: usize| -> bool { ((onset_bits >> (lg - 1)) & 1) != 0 };
            let q5 = (self.vocal_last_mg as usize) == mg
                && (eff_bits & 0x1F) == 0
                && !(1..=5).any(|d| eff_v[d]);
            let unprompted_bg = (eff_bits & 0xFF) == 0;
            let mature_vocal_gate = body.rest_consolidation_ticks >= 480
                || ring.at_lag(1).is_some_and(|e| e.tick >= 3_200);

            if q5 && self.has_tonotopic_bank() && ring.at_lag(1).is_some_and(|e| e.tick >= 480) {
                let pore_cap_cr = PRED_BUCKETS.min(membrane.sensor_pores.len());
                let n_scout_cr = membrane.scout_stats.len();
                let ig_v = self.inv_acoustic_gain();
                let fl_bank = self.tonotopic_floor_ema;

                let mut cand_ch = [u32::MAX; PRED_BUCKETS];
                let mut cand_p = [0usize; PRED_BUCKETS];
                let mut cand_w = [0.0_f32; PRED_BUCKETS];
                let mut cand_n = 0usize;
                for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_cr) {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    if !sp.alive
                        || sp.age_ticks < 32
                        || ch_u >= n_scout_cr
                        || ret_v
                        || membrane.scout_stats[ch_u].is_sparse_impulse()
                        || !(-0.055..=0.075).contains(&self.pore_q5_lvl[p])
                        || self.pore_vocal_reaff[p] < 0.018
                        || !self.is_tonotopic_receptor(p)
                        || self.is_flat_elevated_plateau(p, &sp.history, sp.hist_head as usize)
                    {
                        continue;
                    }
                    let zmean = (self.pore_mean_v[p] - fl_bank) * ig_v;
                    if !(-0.020..=0.22).contains(&zmean) {
                        continue;
                    }
                    let w_score = self.pore_vocal_reaff[p];
                    if let Some(existing) =
                        cand_ch[..cand_n].iter().position(|&c| c == sp.channel)
                    {
                        if w_score > cand_w[existing] {
                            cand_w[existing] = w_score;
                            cand_p[existing] = p;
                        }
                    } else {
                        cand_ch[cand_n] = sp.channel;
                        cand_p[cand_n] = p;
                        cand_w[cand_n] = w_score;
                        cand_n += 1;
                    }
                }

                let top_k = cand_n.min(6);
                for i in 0..top_k {
                    let mut best_j = i;
                    for j in (i + 1)..cand_n {
                        if cand_w[j] > cand_w[best_j] {
                            best_j = j;
                        }
                    }
                    if best_j != i {
                        cand_w.swap(i, best_j);
                        cand_p.swap(i, best_j);
                        cand_ch.swap(i, best_j);
                    }
                }

                if top_k > 0 {
                    let mut strong = 0usize;
                    let mut med = 0usize;
                    let mut peak_ex = 0.0_f32;
                    let mut anchor_hit = false;
                    for idx in 0..top_k {
                        let p = cand_p[idx];
                        let sp = &membrane.sensor_pores[p];
                        let h = sp.hist_head as usize;
                        let u0 = self.deconv_pore_exc(p, &sp.history, h, 0).max(0.0);
                        let u1 = self.deconv_pore_exc(p, &sp.history, h, 1).max(0.0);
                        let u2 = self.deconv_pore_exc(p, &sp.history, h, 2).max(0.0);
                        let base = self.pore_q5_lvl[p].max(0.0).max(0.75 * self.pore_q5_exaff[p]);
                        let e0 = (u0 - base).max(0.0);
                        let e1 = (u1 - base).max(0.0);
                        let e2 = (u2 - base).max(0.0);
                        let sus = e0.min(1.25 * e1 + 0.004).min(1.45 * e2 + 0.006);
                        if sus > 0.032 {
                            strong += 1;
                        }
                        if sus > 0.020 {
                            med += 1;
                        }
                        if cand_w[idx] >= 0.060 && sus > 0.042 {
                            anchor_hit = true;
                        }
                        if sus > peak_ex {
                            peak_ex = sus;
                        }
                    }

                    let y_now = if anchor_hit
                        || (strong >= 2 && peak_ex > 0.035)
                        || (med >= 2 && peak_ex > 0.044)
                        || peak_ex > 0.050
                    {
                        1.0_f32
                    } else {
                        0.0_f32
                    };

                    let mut odd_hit_lag: [Option<usize>; 4] = [None; 4];
                    if self.vocal_q5_cnt >= 32 {
                        let y_tilde = y_now - self.vocal_y_ema;
                        for b in 0..4 {
                            let hit_lg = (ODD_LO[b]..=ODD_HI[b]).find(|&lg| on_bit(lg));
                            if y_now > 0.5 {
                                odd_hit_lag[b] = hit_lg;
                            }
                            let x_b = if hit_lg.is_some() { 1.0_f32 } else { 0.0_f32 };
                            let x_tilde = x_b - self.vocal_f_ema[b];
                            self.vocal_blk_sum[b] = sanitize_clamp(
                                self.vocal_blk_sum[b] + y_tilde * x_tilde,
                                -64.0,
                                64.0,
                            );
                            self.vocal_f_ema[b] = sanitize_clamp(
                                0.985 * self.vocal_f_ema[b] + 0.015 * x_b,
                                0.0,
                                1.0,
                            );
                        }
                        self.vocal_y_ema =
                            sanitize_clamp(0.985 * self.vocal_y_ema + 0.015 * y_now, 0.0, 1.0);

                        if (self.vocal_q5_cnt & 63) == 63 {
                            for b in 0..4 {
                                let bs = self.vocal_blk_sum[b];
                                self.vocal_blk_sum[b] = 0.0;
                                self.vocal_s_acc[b] = sanitize_clamp(
                                    0.9994 * self.vocal_s_acc[b] + bs,
                                    -5_000.0,
                                    5_000.0,
                                );
                                self.vocal_v_acc[b] = sanitize_clamp(
                                    0.9994 * self.vocal_v_acc[b] + bs * bs,
                                    0.0,
                                    50_000.0,
                                );
                            }
                        }
                    } else {
                        self.vocal_y_ema =
                            sanitize_clamp(0.92 * self.vocal_y_ema + 0.08 * y_now, 0.0, 1.0);
                        for b in 0..4 {
                            let x_b = if (ODD_LO[b]..=ODD_HI[b]).any(&on_bit) {
                                1.0_f32
                            } else {
                                0.0_f32
                            };
                            self.vocal_f_ema[b] = sanitize_clamp(
                                0.92 * self.vocal_f_ema[b] + 0.08 * x_b,
                                0.0,
                                1.0,
                            );
                        }
                    }
                    self.vocal_q5_cnt = self.vocal_q5_cnt.saturating_add(1);

                    if mature_vocal_gate && self.vocal_q5_cnt >= 384 {
                        let mut z_band = [0.0_f32; 4];
                        let mut band_contingent = [false; 4];
                        let mut any_contingent = false;
                        for b in 0..4 {
                            let zb = self.vocal_s_acc[b] / (self.vocal_v_acc[b] + 0.16).sqrt();
                            z_band[b] = zb;
                            if self.vocal_v_acc[b] >= 1.0 && zb > 3.30 {
                                band_contingent[b] = true;
                                any_contingent = true;
                            }
                        }
                        let max_zb = z_band[0].max(z_band[1]).max(z_band[2]).max(z_band[3]);
                        let max_f = self.vocal_f_ema[0]
                            .max(self.vocal_f_ema[1])
                            .max(self.vocal_f_ema[2])
                            .max(self.vocal_f_ema[3]);
                        let habituate_unanswered = !any_contingent
                            && self.vocal_q5_cnt >= 2_000
                            && max_zb < 2.75
                            && max_f >= 0.008;

                        let mut os_b = [0.0_f32; 4];
                        let mut oc_b = [0usize; 4];
                        for tube in &mut self.slots {
                            if !tube.alive || tube.kind != TubeKind::CellToMotor || (tube.pre & 1) == 0
                            {
                                continue;
                            }
                            if membrane
                                .motor_pores
                                .get(tube.post as usize)
                                .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg)
                            {
                                let b = ((tube.pre >> 1) & 3) as usize;
                                let cur = tube.myelin_plasticity_accum;
                                let next_m = if band_contingent[b] {
                                    let target_om =
                                        (0.74 + 0.14 * (z_band[b] - 3.30)).clamp(0.0, 1.0);
                                    0.985 * cur + 0.015 * target_om
                                } else if habituate_unanswered {
                                    0.996 * cur + 0.004 * (-0.28)
                                } else if cur < 0.0 && max_zb >= 2.95 {
                                    0.96 * cur
                                } else {
                                    let decayed = 0.9985 * cur;
                                    if decayed.abs() < 1.0e-3 { 0.0 } else { decayed }
                                };
                                tube.myelin_plasticity_accum = sanitize_clamp(next_m, -0.30, 1.0);
                                os_b[b] += tube.myelin_plasticity_accum;
                                oc_b[b] += 1;
                            }
                        }

                        if y_now > 0.5 && any_contingent {
                            let mut best_odd_hit: Option<usize> = None;
                            let mut best_om_b = 0.14_f32;
                            for b in 0..4 {
                                let om_b = if oc_b[b] > 0 {
                                    os_b[b] / (oc_b[b] as f32)
                                } else {
                                    0.0
                                };
                                if om_b > best_om_b
                                    && let Some(lag_on) = odd_hit_lag[b]
                                    && lag_on <= RING_LEN
                                {
                                    best_om_b = om_b;
                                    best_odd_hit = Some(lag_on);
                                }
                            }
                            if let Some(lag_on) = best_odd_hit
                                && let Some(on_entry) = ring.at_lag(lag_on)
                            {
                                for &(c_idx, v_past, dv_past, pre_attn) in &on_entry.active_cells {
                                    if (c_idx & 1) == 0 {
                                        continue;
                                    }
                                    let pre_sig = sanitize(v_past + 0.25 * dv_past);
                                    if pre_sig <= 0.02 {
                                        continue;
                                    }
                                    let attn = sanitize_clamp(pre_attn, 0.35, 1.85);
                                    let out_len = self
                                        .outgoing_by_cell
                                        .get(c_idx as usize)
                                        .map_or(0, Vec::len);
                                    for k in 0..out_len {
                                        let t_idx = self.outgoing_by_cell[c_idx as usize][k];
                                        let Some(tube) = self.slots.get_mut(t_idx as usize) else {
                                            continue;
                                        };
                                        if !tube.alive || tube.kind != TubeKind::CellToMotor {
                                            continue;
                                        }
                                        let is_mg = membrane
                                            .motor_pores
                                            .get(tube.post as usize)
                                            .is_some_and(|mp| {
                                                mp.alive && mp.actuator_channel as usize == mg
                                            });
                                        if !is_mg {
                                            continue;
                                        }
                                        let du = 0.048 * attn * pre_sig - 0.070 * tube.u[0];
                                        tube.apply_benna_fusi_impulse(du, 1.0);
                                        if tube.w_pred < 0.0 {
                                            tube.w_pred =
                                                sanitize_clamp(0.96 * tube.w_pred, -1.5, 0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if unprompted_bg {
                let (e_pk, o_pk, _, _, _, _, _, _) = self.motor_contingency_stats(membrane, mg);
                if e_pk > 0.28 || o_pk > 0.28 {
                    for tube in &mut self.slots {
                        if !tube.alive || tube.kind != TubeKind::CellToMotor {
                            continue;
                        }
                        let is_mg = membrane
                            .motor_pores
                            .get(tube.post as usize)
                            .is_some_and(|mp| mp.alive && mp.actuator_channel as usize == mg);
                        if !is_mg {
                            continue;
                        }
                        if (tube.pre & 1) == 0 && e_pk > 0.28 {
                            let du = 0.008 * e_pk - 0.055 * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                            tube.w_pred = sanitize_clamp(
                                0.992 * tube.w_pred + 0.008 * 0.16 * e_pk,
                                -1.5,
                                1.5,
                            );
                        } else if (tube.pre & 1) != 0 && o_pk > 0.28 {
                            let du = 0.008 * o_pk - 0.055 * tube.u[0];
                            tube.apply_benna_fusi_impulse(du, 1.0);
                            if tube.w_pred < 0.0 {
                                tube.w_pred = sanitize_clamp(0.992 * tube.w_pred, -1.5, 0.0);
                            }
                        }
                    }
                }
            }
        }

        if self.has_tonotopic_bank()
            && ring
                .at_lag(1)
                .is_some_and(|e| e.tick >= 160 && (e.tick & 7) == 0)
            && membrane.alive_sensor_count >= 52
            && !cells.active_indices.is_empty()
        {
            let pore_cap_tr = PRED_BUCKETS.min(membrane.sensor_pores.len());
            let n_scout_tr = membrane.scout_stats.len();
            let ig_tr = self.inv_acoustic_gain();
            let mut conf_ch = [u32::MAX; PRED_BUCKETS];
            let mut confirmed_coch = 0usize;
            let mut trial_young = 0usize;
            let mut retina_cnt = 0usize;
            let active_lmg_tr = (self.vocal_last_mg as usize) < ad;
            for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_tr) {
                let sp = &membrane.sensor_pores[p];
                if !sp.alive {
                    continue;
                }
                if ret_v {
                    retina_cnt += 1;
                } else if sp.age_ticks <= 180 {
                    trial_young += 1;
                }
                if self.is_tonotopic_receptor(p)
                    && !self.is_flat_elevated_plateau(p, &sp.history, sp.hist_head as usize)
                    && self.pore_vocal_reaff[p] >= 0.018
                    && (-0.065..=0.085).contains(&self.pore_q5_lvl[p])
                    && !conf_ch[..confirmed_coch].contains(&sp.channel)
                {
                    conf_ch[confirmed_coch] = sp.channel;
                    confirmed_coch += 1;
                }
            }

            if confirmed_coch < 6 && trial_young < 4 && n_scout_tr > 0 {
                let mut best_vp: Option<usize> = None;
                let mut best_tier = 2u8;
                let mut best_vscore = f32::MAX;
                for (p, &ret_v) in is_world_retina_cr.iter().enumerate().take(pore_cap_tr) {
                    let sp = &membrane.sensor_pores[p];
                    let ch_u = sp.channel as usize;
                    let is_dup_ch = membrane.sensor_pores[..p]
                        .iter()
                        .any(|s| s.alive && s.channel == sp.channel);
                    let agency_cap = if is_dup_ch { 0.025 } else { 0.010 };
                    if !sp.alive
                        || sp.age_ticks <= 180
                        || (ch_u < n_scout_tr && membrane.scout_stats[ch_u].is_sparse_impulse())
                        || self.pore_vocal_reaff[p] >= 0.008
                        || (ret_v && !is_dup_ch && retina_cnt <= 28)
                        || (!ret_v && self.pore_agency[p] >= agency_cap)
                    {
                        continue;
                    }
                    let non_ton = !self.is_tonotopic_receptor(p)
                        || self.is_flat_elevated_plateau(p, &sp.history, sp.hist_head as usize)
                        || (self.pore_q5_lvl[p] <= 0.90
                            && !(-0.065..=0.085).contains(&self.pore_q5_lvl[p]));
                    let dead_ton_band = active_lmg_tr
                        || self.pore_var_ema[p] * ig_tr * ig_tr < 1.2e-4;
                    let tier = if non_ton {
                        0u8
                    } else if sp.age_ticks > 320
                        && dead_ton_band
                        && self.pore_vocal_reaff[p] < 0.004
                    {
                        1u8
                    } else {
                        continue;
                    };
                    let vscore = self.pore_agency[p] + 0.25 * sp.trophic_ema;
                    if tier < best_tier || (tier == best_tier && vscore < best_vscore) {
                        best_tier = tier;
                        best_vscore = vscore;
                        best_vp = Some(p);
                    }
                }

                if let Some(vp) = best_vp {
                    let start = rng.next_index(n_scout_tr);
                    let mut best_c: Option<usize> = None;
                    let mut best_sc = 0.0_f32;
                    let mut sampled = 0usize;
                    for k in 0..n_scout_tr {
                        let c = (start + k) % n_scout_tr;
                        let st = &membrane.scout_stats[c];
                        let pq = st.passive_quality();
                        if pq <= 0.018
                            || st.is_sparse_impulse()
                            || st.diff_sq_ema < 1.5e-5
                            || st.diff_sq_ema < 0.012 * st.var_ema
                        {
                            continue;
                        }
                        let zm = (st.mean - self.tonotopic_floor_ema) * ig_tr;
                        if !(-0.018..=0.18).contains(&zm) {
                            continue;
                        }
                        let c_u32 = c as u32;
                        if membrane
                            .sensor_pores
                            .iter()
                            .any(|s| s.alive && s.channel == c_u32)
                        {
                            continue;
                        }
                        let sc = pq + 0.25;
                        if sc > best_sc {
                            best_sc = sc;
                            best_c = Some(c);
                        }
                        sampled += 1;
                        if sampled >= 6 {
                            break;
                        }
                    }
                    if let Some(c_pick) = best_c {
                        let n_act_cells = cells.active_indices.len();
                        let tgt = cells.active_indices[rng.next_index(n_act_cells)];
                        let dtgt = cells.active_indices[rng.next_index(n_act_cells)];
                        let st_pick = membrane.scout_stats[c_pick];
                        let mut nb = SensorPore::newborn(c_pick as u32, 1, tgt, dtgt, 0.16);
                        nb.history = [st_pick.prev_val; 16];
                        nb.last_val = st_pick.prev_val;
                        membrane.sensor_pores[vp] = nb;
                        for m in 0..ad {
                            self.pore_action_w[m][vp] = 0.0;
                            self.tectal_map_w[m][vp] = 0.0;
                        }
                        self.pore_self_w[vp] = 0.0;
                        self.pore_mean_v[vp] = st_pick.mean;
                        self.pore_scale_v[vp] = 0.05;
                        self.pore_agency[vp] = 0.0;
                        self.pore_dsq_ema[vp] = 1.0e-4;
                        self.pore_var_ema[vp] = 5.0e-3;
                        self.pore_bg_ild[vp] = 0.0;
                        self.pore_floor_v[vp] = st_pick.prev_val;
                        self.pore_ar1_w[vp] = 0.59;
                        self.pore_ar2_w[vp] = 0.0;
                        self.pore_q5_lvl[vp] = 1.0;
                        self.pore_q5_exaff[vp] = 0.0;
                        self.pore_vocal_reaff[vp] = 0.0;
                    }
                }
            }
        }

        if (rng.next_u32() & 15) == 0 && self.alive_count > 32 {
            for idx in 0..self.slots.len() {
                if !self.slots[idx].alive {
                    continue;
                }
                let t = &self.slots[idx];
                if t.age_ticks > 600
                    && t.u[0].abs() < 0.004
                    && t.stiff_strength() < 0.005
                    && t.usage_ema < 0.002
                    && !(t.kind == TubeKind::CellToMotor && t.myelin_plasticity_accum.abs() > 0.14)
                {
                    let pre = t.pre as usize;
                    self.slots[idx].alive = false;
                    self.free_list.push(idx as u32);
                    self.alive_count = self.alive_count.saturating_sub(1);
                    if let Some(list) = self.outgoing_by_cell.get_mut(pre) {
                        list.retain(|&x| x != idx as u32);
                    }
                    atrophied += 1;
                    break;
                }
            }
        }

        (splits, shortcuts, atrophied)
    }

    pub fn consolidate_rest_step(&mut self) {
        for tube in &mut self.slots {
            if !tube.alive {
                continue;
            }
            if tube.u[0].abs() >= 0.045
                || (tube.kind == TubeKind::CellToMotor
                    && tube.myelin_plasticity_accum > 0.24
                    && tube.u[0].abs() >= 0.018)
            {
                tube.apply_benna_fusi_impulse(0.0, 4.0);
            } else if tube.stiff_strength() < 0.025 {
                tube.u[0] = sanitize(tube.u[0] * 0.995);
                tube.u[1] = sanitize(tube.u[1] * 0.995);
            }
        }
    }

    pub fn prune_tubes_referencing_cells(&mut self, dead_cells: &[u32]) -> u64 {
        if dead_cells.is_empty() {
            return 0;
        }
        let mut removed = 0u64;
        for idx in 0..self.slots.len() {
            if !self.slots[idx].alive {
                continue;
            }
            let pre = self.slots[idx].pre;
            let post = self.slots[idx].post;
            let kind = self.slots[idx].kind;
            let dead_pre = dead_cells.contains(&pre);
            let dead_post = kind == TubeKind::CellToCell && dead_cells.contains(&post);
            if dead_pre || dead_post {
                self.slots[idx].alive = false;
                self.free_list.push(idx as u32);
                self.alive_count = self.alive_count.saturating_sub(1);
                if let Some(list) = self.outgoing_by_cell.get_mut(pre as usize) {
                    list.retain(|&x| x != idx as u32);
                }
                removed += 1;
            } else if let Some(g) = self.slots[idx].context_gate_cell
                && dead_cells.contains(&g)
            {
                self.slots[idx].context_gate_cell = None;
            }
        }
        removed
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rollout_and_select_pulse(
        &mut self,
        num_actuators: usize,
        emb_eco: bool,
        cells: &CellArena,
        membrane: &PoreMembrane,
        body: &mut BodyCore,
        ring: &EfferenceRing,
        mode: ControlMode,
        rng: &mut Rng64,
    ) -> RolloutOutcome {
        let ad = self.act_dim();
        if num_actuators == 0 {
            return RolloutOutcome {
                pulse: FieldPulse::silent(0),
                committed: false,
                rehearsals: 0,
                pred_with: Vec::new(),
                pred_null: Vec::new(),
            };
        }

        let mut basal_drive = vec![0.0_f32; num_actuators];
        let mut apical_drive = vec![0.0_f32; num_actuators];
        let mut tube_drive = vec![0.0_f32; num_actuators];
        let mut tube_pred_action_val = vec![0.0_f32; num_actuators];
        let mut slow_inhib_brake = vec![0.0_f32; num_actuators];
        let mut stiff_motor_conf = vec![0.0_f32; num_actuators];
        let mut mastery_by_act = vec![0.0_f32; num_actuators];
        let mut mp_cond_by_act = vec![0.0_f32; num_actuators];
        let mut pyr_env_sum = vec![0.0_f32; ad];
        let mut pyr_env_cnt = vec![0u16; ad];

        for mp in &membrane.motor_pores {
            if mp.alive {
                let ch = mp.actuator_channel as usize;
                if ch < num_actuators {
                    mastery_by_act[ch] = mp.motor_mastery.clamp(0.0, 1.0);
                    mp_cond_by_act[ch] = mp.conductance.clamp(0.0, 2.0);
                }
            }
        }

        let base_k_ro = (cells.alive_count / 2).max(cells.alive_count.min(24));
        let uncap_k_ro = base_k_ro.clamp(4, 96).min(cells.alive_count.max(1));
        let cap_k_ro = cells.k_winners_target().max(1);
        let ap_scale_ro = if cells.alive_count > 128 && uncap_k_ro > cap_k_ro {
            (uncap_k_ro as f32) / (cap_k_ro as f32)
        } else {
            1.0
        };
        for &c_idx in &cells.active_indices {
            let Some(cell) = cells.slots.get(c_idx as usize) else {
                continue;
            };
            if !cell.alive {
                continue;
            }
            let sig = sanitize(cell.v_real + 0.25 * cell.delta_v);
            let is_slow = cell.tau >= 5.0;
            let is_apical = cell.tau >= 16.0;
            let sig_m = if is_apical && ap_scale_ro > 1.0 {
                sanitize(sig * ap_scale_ro)
            } else {
                sig
            };
            let Some(out_list) = self.outgoing_by_cell.get(c_idx as usize) else {
                continue;
            };
            for &t_idx in out_list {
                let Some(tube) = self.slots.get(t_idx as usize) else {
                    continue;
                };
                if !tube.alive || tube.kind != TubeKind::CellToMotor {
                    continue;
                };
                let Some(mp) = membrane.motor_pores.get(tube.post as usize) else {
                    continue;
                };
                if !mp.alive {
                    continue;
                }
                let act_ch = mp.actuator_channel as usize;
                if act_ch >= num_actuators {
                    continue;
                }
                if act_ch < ad {
                    pyr_env_sum[act_ch] =
                        sanitize(pyr_env_sum[act_ch] + sig.abs() * mp.conductance);
                    pyr_env_cnt[act_ch] = pyr_env_cnt[act_ch].saturating_add(1);
                }
                let g = tube.effective_g(&cells.slots);
                let contrib = sanitize(g * sig_m * mp.conductance);
                if is_apical {
                    apical_drive[act_ch] = sanitize(apical_drive[act_ch] + contrib);
                } else {
                    basal_drive[act_ch] = sanitize(basal_drive[act_ch] + contrib);
                }
                let pred_contrib =
                    sanitize((0.65 * g + 0.85 * tube.w_pred) * sig_m * mp.conductance);
                tube_pred_action_val[act_ch] =
                    sanitize(tube_pred_action_val[act_ch] + pred_contrib);
                stiff_motor_conf[act_ch] =
                    sanitize_clamp(stiff_motor_conf[act_ch] + tube.stiff_strength(), 0.0, 2.0);
                if is_slow && g < -0.01 {
                    slow_inhib_brake[act_ch] =
                        sanitize(slow_inhib_brake[act_ch] + (-g) * sig.abs());
                }
            }
        }
        for m in 0..num_actuators {
            tube_drive[m] = sanitize(
                1.28 * smooth_saturate(0.78 * basal_drive[m])
                    + 0.96 * smooth_saturate(0.78 * apical_drive[m]),
            );
        }

        let mut explore_wave = vec![0.0_f32; num_actuators];
        body.next_exploratory_wave(rng, &mut explore_wave);
        let mut explore_alt = vec![0.0_f32; num_actuators];
        for dst in &mut explore_alt {
            *dst = rng.next_f32_signed();
        }

        let mut prev_act = vec![0.0_f32; num_actuators];
        let mut recent_val_delta = 0.0_f32;
        if let Some(p1) = ring.at_lag(1)
            && p1.committed
        {
            for (m, dst) in prev_act.iter_mut().enumerate().take(ad) {
                *dst = p1.motor(m);
            }
            recent_val_delta = sanitize_clamp(
                body.fast_valence + cells.somatic_value_estimate() - p1.somatic_value_est,
                -2.0,
                2.0,
            );
        }

        let mut klino_gain_mean = 0.0_f32;
        let mut klino_cnt = 0usize;
        for mp in &membrane.motor_pores {
            if mp.alive {
                klino_gain_mean += mp.klinokinesis_gain;
                klino_cnt += 1;
            }
        }
        if klino_cnt > 0 {
            klino_gain_mean /= klino_cnt as f32;
        }

        let mut sensory_cells: Vec<(u32, f32, f32)> = Vec::with_capacity(32);
        for &c_idx in &cells.active_indices {
            if sensory_cells.len() >= 32 {
                break;
            }
            if let Some(c) = cells.slots.get(c_idx as usize)
                && c.alive
                && c.sensory_trophic_ema > 0.005
            {
                sensory_cells.push((c_idx, c.v_real, c.delta_v));
            }
        }

        let thermal_need = (-body.fast_valence * 6.0).clamp(0.0, 1.0);
        let acute_distress = (-body.fast_valence * 22.0).clamp(0.0, 1.0);
        let atp_surplus = ((body.atp - 6.0) / 10.0).clamp(0.0, 1.5) * (1.0 - thermal_need);
        let last_tick = ring.at_lag(1).map_or(0, |e| e.tick);
        let probe_window = atp_surplus > 0.20 && (last_tick & 15) < 3;
        let pore_cap = PRED_BUCKETS.min(membrane.sensor_pores.len());
        let n_scout = membrane.scout_stats.len();
        let mut field_breadth = vec![0u16; ad];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if sp.alive
                && sp.age_ticks >= 16
                && sp.trophic_ema >= 0.025
                && self.is_retinal_field_pore(p, sp, &membrane.scout_stats)
            {
                for m in 0..num_actuators.min(ad) {
                    if self.pore_action_w[m][p].abs() >= 0.022 {
                        field_breadth[m] = field_breadth[m].saturating_add(1);
                    }
                }
            }
        }
        let mut max_breadth = 0u16;
        for &fb in &field_breadth[..num_actuators.min(ad)] {
            if fb > max_breadth {
                max_breadth = fb;
            }
        }
        let mut is_world_loco = vec![false; ad];
        if max_breadth >= 4 {
            for m in 0..num_actuators.min(ad) {
                if field_breadth[m].saturating_mul(2) >= max_breadth {
                    is_world_loco[m] = true;
                }
            }
        }

        let act_cap = num_actuators.min(ad);
        let mut is_world_retina = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || !self.is_retinal_field_pore(p, sp, &membrane.scout_stats) {
                continue;
            }
            if max_breadth < 4 {
                is_world_retina[p] = true;
            } else {
                let mut loco_w_p = 0.0_f32;
                let mut nonloco_w_p = 0.0_f32;
                for m in 0..act_cap {
                    let wa = self.pore_action_w[m][p].abs();
                    if is_world_loco[m] {
                        if wa > loco_w_p {
                            loco_w_p = wa;
                        }
                    } else if wa > nonloco_w_p {
                        nonloco_w_p = wa;
                    }
                }
                if loco_w_p >= 0.018 && loco_w_p >= 0.80 * nonloco_w_p {
                    is_world_retina[p] = true;
                }
            }
        }

        let mut min_h_ro = [1.0_f32; PRED_BUCKETS];
        let mut is_glottal_cand_pore = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
            {
                continue;
            }
            let mut min_h = 1.0_f32;
            for &hv in &sp.history {
                if hv < min_h {
                    min_h = hv;
                }
            }
            min_h_ro[p] = min_h;
            if is_world_retina[p] {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            if self.pore_agency[p] >= 0.006 && self.is_cochlear_floor_pore(p, min_h, mean_p) {
                is_glottal_cand_pore[p] = true;
            }
        }

        let mut max_asw_jac = vec![0.0_f32; ad];
        let mut tot_pore_reaff = vec![0.0_f32; ad];
        let mut pos_cochlea_w = vec![0.0_f32; ad];
        let mut pos_cochlea_cnt = vec![0u16; ad];
        for m in 0..act_cap {
            for &w in &self.action_sensory_w[m] {
                let wa = w.abs();
                if wa > max_asw_jac[m] {
                    max_asw_jac[m] = wa;
                }
            }
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                    continue;
                }
                let pw = self.pore_action_w[m][p];
                let pwa = pw.abs();
                if self.pore_agency[p] >= 0.010 && pwa >= 0.015 {
                    tot_pore_reaff[m] += pwa;
                }
                if is_glottal_cand_pore[p] && pw >= 0.015 {
                    pos_cochlea_w[m] += pw;
                    pos_cochlea_cnt[m] = pos_cochlea_cnt[m].saturating_add(1);
                }
            }
        }

        let mut m_glottal = self.primary_glottal_actuator(&is_world_loco, act_cap);
        if m_glottal.is_none() {
            let mut best_glottal_w = 0.05_f32;
            for m in 0..act_cap {
                if !is_world_loco[m]
                    && (tot_pore_reaff[m] >= 0.045 || max_asw_jac[m] >= 0.065)
                    && pos_cochlea_cnt[m] >= 2
                    && pos_cochlea_w[m] > best_glottal_w
                {
                    best_glottal_w = pos_cochlea_w[m];
                    m_glottal = Some(m);
                }
            }
        }

        let mut is_monaural_coch = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || is_world_retina[p]
                || membrane.scout_stats[ch_u].is_sparse_impulse()
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            let reaff_mon = self.is_tonotopic_receptor(p)
                && !self.is_flat_elevated_plateau(p, &sp.history, sp.hist_head as usize)
                && self.pore_vocal_reaff[p] >= 0.018
                && (-0.055..=0.075).contains(&self.pore_q5_lvl[p]);
            is_monaural_coch[p] = reaff_mon
                || if let Some(mg) = m_glottal {
                    self.pore_action_w[mg][p] >= 0.016
                        && self.is_glottal_driven_coch(p, min_h_ro[p], mean_p)
                } else {
                    self.pore_agency[p] >= 0.010
                        && self.is_cochlear_floor_pore(p, min_h_ro[p], mean_p)
                };
        }

        let mut somatic_prop_reaff = vec![0.0_f32; ad];
        let mut abs_cochlea_reaff = vec![0.0_f32; ad];
        let mut coch_artic_cnt = vec![0u16; ad];
        for m in 0..act_cap {
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 || ch_u >= n_scout {
                    continue;
                }
                let pwa = self.pore_action_w[m][p].abs();
                let is_echo = membrane.scout_stats[ch_u].is_sparse_impulse();
                if is_monaural_coch[p] {
                    if pwa >= 0.0045 {
                        abs_cochlea_reaff[m] += pwa;
                        coch_artic_cnt[m] = coch_artic_cnt[m].saturating_add(1);
                    }
                } else if !is_world_retina[p]
                    && !is_echo
                    && self.is_somatic_non_lso(p, min_h_ro[p], self.pore_mean_v[p])
                    && self.pore_agency[p] >= 0.012
                    && pwa >= 0.020
                    && !(max_breadth >= 4
                        && is_world_loco
                            .iter()
                            .enumerate()
                            .any(|(ml, &il)| il && self.pore_action_w[ml][p].abs() >= 0.013))
                {
                    somatic_prop_reaff[m] += pwa;
                }
            }
        }

        let mut quiet_self_7 = false;
        let mut has_recent_breath_13 = true;
        let mut has_recent_breath_14 = true;
        let mut q_lag = [false; 17];
        if let Some(mg) = m_glottal
            && ring.at_lag(16).is_some_and(|e| e.tick > 0)
        {
            q_lag[0] = true;
            for (d, ql) in q_lag.iter_mut().enumerate().skip(1) {
                *ql = ring
                    .at_lag(d)
                    .is_some_and(|e| !e.committed || e.motor(mg) <= -0.05);
            }
            quiet_self_7 =
                q_lag[1] && q_lag[2] && q_lag[3] && q_lag[4] && q_lag[5] && q_lag[6] && q_lag[7];
            has_recent_breath_13 = false;
            has_recent_breath_14 = false;
            for k in 1..=6 {
                if q_lag[k]
                    && q_lag[k + 1]
                    && q_lag[k + 2]
                    && q_lag[k + 3]
                    && q_lag[k + 4]
                    && q_lag[k + 5]
                    && q_lag[k + 6]
                    && q_lag[k + 7]
                {
                    has_recent_breath_14 = true;
                    if k <= 5 {
                        has_recent_breath_13 = true;
                    }
                    break;
                }
            }
        }

        let mut loco_nov_sum = 0.0_f32;
        let mut loco_mast_sum = 0.0_f32;
        let mut loco_nov_cnt = 0usize;
        for m in 0..act_cap {
            if is_world_loco[m] {
                let nov = (1.0 - 10.0 * (mastery_by_act[m] - 0.30).max(0.0)).clamp(0.0, 1.0);
                loco_nov_sum += nov;
                loco_mast_sum += mastery_by_act[m];
                loco_nov_cnt += 1;
            }
        }
        let loco_ep_nov = if loco_nov_cnt > 0 {
            loco_nov_sum / (loco_nov_cnt as f32)
        } else {
            1.0
        };
        let loco_mat_emb = if loco_nov_cnt > 0 {
            let mean_lm = loco_mast_sum / (loco_nov_cnt as f32);
            ((mean_lm - 0.12) * 5.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let active_embodied = if !emb_eco {
            max_breadth >= 4 && m_glottal.is_some()
        } else {
            max_breadth >= 4 && m_glottal.is_some() && loco_mat_emb >= 0.85 && body.atp > 6.2
        };
        let loco_mat = if active_embodied {
            loco_mat_emb
        } else if !emb_eco {
            (1.0 - loco_ep_nov).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let emb_ep_nov = (1.0 - loco_mat_emb).clamp(0.0, 1.0);
        let eff_ep_surplus = atp_surplus * (1.0 - acute_distress);
        let elder_forager = mode == ControlMode::Natural
            && emb_eco
            && active_embodied
            && cells
                .slots
                .first()
                .is_some_and(|c| c.alive && c.age_ticks > 150_000);
        let mvt_val_bar = if elder_forager { 0.024 } else { 0.015 };
        let mvt_patch_depleted = emb_eco
            && active_embodied
            && body.atp > 22.0
            && body.fast_valence < mvt_val_bar
            && acute_distress < 0.05;
        let mvt_wave = if mvt_patch_depleted {
            if elder_forager { 0.34 } else { 0.24 }
        } else {
            0.0
        };
        let mvt_probe = if mvt_patch_depleted {
            if elder_forager { 0.42 } else { 0.32 }
        } else {
            0.0
        };
        let wave_ep_surplus = if active_embodied {
            if !emb_eco {
                eff_ep_surplus.min(0.06 + 0.44 * emb_ep_nov)
            } else {
                eff_ep_surplus.min((0.55 * emb_ep_nov).max(mvt_wave))
            }
        } else if !emb_eco {
            atp_surplus.min(0.28 + 1.23 * loco_ep_nov)
        } else {
            atp_surplus
        };
        let noise_ep_surplus = if active_embodied {
            if !emb_eco {
                eff_ep_surplus.min(0.02 + 0.38 * emb_ep_nov)
            } else {
                eff_ep_surplus.min(0.45 * emb_ep_nov)
            }
        } else if !emb_eco {
            atp_surplus.min(0.10 + 1.41 * loco_ep_nov)
        } else {
            atp_surplus
        };
        let probe_surplus = if active_embodied {
            if !emb_eco {
                eff_ep_surplus.min(0.04 + 0.40 * emb_ep_nov)
            } else {
                eff_ep_surplus.min((0.50 * emb_ep_nov).max(mvt_probe))
            }
        } else if !emb_eco {
            atp_surplus.min(0.45 + 1.06 * loco_ep_nov)
        } else {
            atp_surplus
        };

        let mut homeo_grad = vec![0.0_f32; num_actuators];
        let mut model_grad = vec![0.0_f32; num_actuators];
        let mut v_align = vec![0.0_f32; num_actuators];
        let mut epistemic_drive = vec![0.0_f32; num_actuators];
        for &(c_idx, _, _) in &sensory_cells {
            let b = c_idx as usize;
            let cell = &cells.slots[c_idx as usize];
            let homeo_desire = sanitize(0.45 * cell.v_homeo + 0.85 * cell.v_delta_homeo);
            let valence_mag = cell.v_homeo.abs() + cell.v_delta_homeo.abs();
            let predictable_neutral = cell.surprise_ema < 0.45 && valence_mag < 0.18;
            for m in 0..act_cap {
                let jac = self.action_sensory_w[m][b];
                if homeo_desire.abs() > 1.0e-4 {
                    homeo_grad[m] = sanitize(homeo_grad[m] + homeo_desire * jac);
                }
                if predictable_neutral
                    && wave_ep_surplus > 0.01
                    && jac.abs() > 0.026
                    && (!active_embodied || is_world_loco[m])
                {
                    let wave_dir = if explore_wave[m] >= 0.0 { 1.0 } else { -1.0 };
                    let next_ep = sanitize(
                        epistemic_drive[m] + 2.10 * wave_ep_surplus * jac.abs() * wave_dir,
                    );
                    epistemic_drive[m] = if active_embodied {
                        next_ep.clamp(-0.26, 0.26)
                    } else {
                        next_ep
                    };
                }
            }
        }

        let mut m_echo_emitter: Option<usize> = None;
        let mut best_emit_w = 0.010_f32;
        for m in 0..act_cap {
            if !is_world_loco[m]
                && Some(m) != m_glottal
                && somatic_prop_reaff[m] < 0.030
                && self.vocal_artic_w[m] < 0.005
                && self.echo_emit_w[m] > best_emit_w
            {
                best_emit_w = self.echo_emit_w[m];
                m_echo_emitter = Some(m);
            }
        }
        let ring_warm = ring.at_lag(16).is_some_and(|e| e.tick > 0);
        let mut quiet_echo_self = ring_warm;
        if quiet_echo_self {
            for lag in 1..=14 {
                if let Some(ent) = ring.at_lag(lag)
                    && ent.committed
                {
                    for m in 0..act_cap {
                        let is_emit_cand = if let Some(em) = m_echo_emitter {
                            m == em
                        } else {
                            !is_world_loco[m]
                                && somatic_prop_reaff[m] < 0.030
                                && self.vocal_artic_w[m] < 0.005
                                && Some(m) != m_glottal
                        };
                        if is_emit_cand && ent.motor(m).abs() > 0.12 {
                            quiet_echo_self = false;
                        }
                    }
                }
            }
        }

        let mut vis_prox_sum = 0.0_f32;
        let mut vis_prox_cnt = 0usize;
        let mut ret_jump_sum = 0.0_f32;
        let mut ret_jump_cnt = 0usize;
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            if is_world_retina[p] {
                let h = sp.hist_head as usize;
                let x_now = sp.history[(h + 15) & 15];
                let x_prev1 = sp.history[(h + 14) & 15];
                ret_jump_sum += (x_now - x_prev1).abs();
                ret_jump_cnt += 1;
                if self.pore_agency[p] >= 0.020 {
                    vis_prox_sum += (x_now - self.pore_mean_v[p]).max(0.0);
                    vis_prox_cnt += 1;
                }
            }
        }
        let ret_jump_mean = if ret_jump_cnt >= 4 {
            ret_jump_sum / (ret_jump_cnt as f32)
        } else {
            0.0
        };
        if mode == ControlMode::Natural && (ret_jump_mean > 0.085 || acute_distress > 0.48) {
            self.peer_vocal_hold = 0.0;
            self.social_loom_ema = 0.0;
            self.peer_coch_age = 255;
            self.peer_echo_age = 255;
            self.peer_disp_coch = 0.0;
            self.peer_disp_echo = 0.0;
            self.peer_bearing_ema.fill(0.0);
        }
        let vis_prox_now = if vis_prox_cnt > 0 {
            sanitize_clamp(vis_prox_sum / (vis_prox_cnt as f32), 0.0, 2.0)
        } else {
            0.0
        };

        let mut acoustic_taxis = vec![0.0_f32; ad];
        let mut top4_coch = [0.0_f32; 4];
        let mut seen_coch_ro = [u32::MAX; PRED_BUCKETS];
        let mut seen_coch_ro_n = 0usize;
        let mut coch_strong_cnt = 0usize;
        let mut coch_weak_cnt = 0usize;
        let mut sym_coch_cnt = 0usize;
        let mut top_echo_0 = 0.0_f32;
        let mut top_echo_1 = 0.0_f32;
        let mut echo_pore_cnt = 0usize;

        let mut top4_lso_pos = [0.0_f32; 4];
        let mut top4_lso_neg = [0.0_f32; 4];
        let mut top4_clean_pos = [0.0_f32; 4];
        let mut top4_clean_neg = [0.0_f32; 4];
        let mut lso_pos_cnt = 0usize;
        let mut lso_neg_cnt = 0usize;
        let mut clean_pos_cnt = 0usize;
        let mut clean_neg_cnt = 0usize;
        let mut lso_signed_sum = 0.0_f32;
        let mut seen_lso_ro = [u32::MAX; PRED_BUCKETS];
        let mut seen_lso_ro_n = 0usize;

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            if is_monaural_coch[p] {
                sym_coch_cnt += 1;
                if mode == ControlMode::Natural && self.pore_bg_ild[p] != 0.0 {
                    self.pore_bg_ild[p] = 0.0;
                }
                if !seen_coch_ro[..seen_coch_ro_n].contains(&sp.channel) {
                    seen_coch_ro[seen_coch_ro_n] = sp.channel;
                    seen_coch_ro_n += 1;
                    let h = sp.hist_head as usize;
                    let u0 = self.deconv_pore_exc(p, &sp.history, h, 0).max(0.0);
                    let u1 = self.deconv_pore_exc(p, &sp.history, h, 1).max(0.0);
                    let u2 = self.deconv_pore_exc(p, &sp.history, h, 2).max(0.0);
                    let u_sus3 = if u0 > 0.0 && u1 > 0.0 && u2 > 0.0 {
                        u0.min(1.30 * u1 + 0.004).min(1.65 * u2 + 0.006)
                    } else {
                        0.0
                    };
                    if u_sus3 > 0.038 {
                        coch_strong_cnt += 1;
                    }
                    if u_sus3 > 0.016 {
                        coch_weak_cnt += 1;
                    }
                    Self::insert_top4(&mut top4_coch, u_sus3);
                }
            }
            if ch_u < n_scout
                && membrane.scout_stats[ch_u].is_sparse_impulse()
                && !is_world_retina[p]
                && self.pore_agency[p] < 0.035
            {
                echo_pore_cnt += 1;
                let h = sp.hist_head as usize;
                let dev_now = (sp.history[(h + 15) & 15] - self.pore_mean_v[p]).max(0.0);
                if dev_now > top_echo_0 {
                    top_echo_1 = top_echo_0;
                    top_echo_0 = dev_now;
                } else if dev_now > top_echo_1 {
                    top_echo_1 = dev_now;
                }
            } else if quiet_self_7
                && sp.age_ticks >= 40
                && ch_u < n_scout
                && !membrane.scout_stats[ch_u].is_sparse_impulse()
                && !is_monaural_coch[p]
                && !is_world_retina[p]
                && !seen_lso_ro[..seen_lso_ro_n].contains(&sp.channel)
            {
                let mean_p = self.pore_mean_v[p];
                let mut somatic_other_w = 0.0_f32;
                for m in 0..act_cap {
                    if !is_world_loco[m] && Some(m) != m_glottal && self.vocal_artic_w[m] < 0.005 {
                        somatic_other_w += self.pore_action_w[m][p].abs();
                    }
                }
                let is_coch_trough = self.is_coch_trough(p, min_h_ro[p], mean_p)
                    && m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
                let is_glottal_pos = m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
                if !is_coch_trough
                    && !is_glottal_pos
                    && self.is_lso_floor_pore(p, min_h_ro[p], mean_p)
                    && somatic_other_w < 0.038
                {
                    seen_lso_ro[seen_lso_ro_n] = sp.channel;
                    seen_lso_ro_n += 1;
                    let h = sp.hist_head as usize;
                    let bg = self.pore_bg_ild[p];
                    let d0 = self.deconv_pore_sym(p, &sp.history, h, 0) - bg;
                    let d1 = self.deconv_pore_sym(p, &sp.history, h, 1) - bg;
                    let d2 = self.deconv_pore_sym(p, &sp.history, h, 2) - bg;
                    let d3 = self.deconv_pore_sym(p, &sp.history, h, 3) - bg;
                    if d0.abs() > 0.015 {
                        lso_signed_sum += d0;
                    }
                    if d0 * d1 > 0.0 && d1 * d2 > 0.0 && d2 * d3 > 0.0 {
                        let sus_d = d0.abs().min(d1.abs()).min(d2.abs()).min(d3.abs());
                        if sus_d > 0.018 {
                            if d0 > 0.0 {
                                lso_pos_cnt += 1;
                                Self::insert_top4(&mut top4_lso_pos, sus_d);
                            } else {
                                lso_neg_cnt += 1;
                                Self::insert_top4(&mut top4_lso_neg, sus_d);
                            }
                        }
                        if sp.age_ticks >= 48 && bg.abs() < 0.018 && sus_d > 0.022 {
                            if d0 > 0.0 {
                                clean_pos_cnt += 1;
                                Self::insert_top4(&mut top4_clean_pos, sus_d);
                            } else {
                                clean_neg_cnt += 1;
                                Self::insert_top4(&mut top4_clean_neg, sus_d);
                            }
                        }
                    }
                }
            }
        }
        let max_echo_peer_now = top_echo_0.hypot(top_echo_1);
        let (dom_sign_ro, lso_coh_cnt, top4_lso_dom, lso_clean_cnt, top4_lso_clean) =
            if lso_pos_cnt > lso_neg_cnt {
                (
                    1.0_f32,
                    lso_pos_cnt,
                    top4_lso_pos,
                    clean_pos_cnt,
                    top4_clean_pos,
                )
            } else if lso_neg_cnt > lso_pos_cnt {
                (
                    -1.0_f32,
                    lso_neg_cnt,
                    top4_lso_neg,
                    clean_neg_cnt,
                    top4_clean_neg,
                )
            } else if lso_signed_sum >= 0.0 {
                (
                    1.0_f32,
                    lso_pos_cnt,
                    top4_lso_pos,
                    clean_pos_cnt,
                    top4_clean_pos,
                )
            } else {
                (
                    -1.0_f32,
                    lso_neg_cnt,
                    top4_lso_neg,
                    clean_neg_cnt,
                    top4_clean_neg,
                )
            };
        let mut coch_lvl_now = 0.0_f32;
        let peer_harmonic_now = if quiet_self_7 && sym_coch_cnt >= 2 {
            coch_lvl_now = sanitize_clamp(0.5 * (top4_coch[0] + top4_coch[1]), 0.0, 2.0);
            (coch_strong_cnt >= 2
                && coch_lvl_now > 0.040
                && lso_coh_cnt >= 3
                && top4_lso_dom[2] > 0.018)
                || (coch_strong_cnt >= 1
                    && top4_coch[0] > 0.058
                    && lso_coh_cnt >= 4
                    && top4_lso_dom[3] > 0.020)
                || (coch_weak_cnt >= 2
                    && coch_lvl_now > 0.018
                    && lso_clean_cnt >= 5
                    && top4_lso_clean[3] > 0.022)
        } else {
            false
        };
        let peer_echo_now = echo_pore_cnt >= 1 && quiet_echo_self && top_echo_0 > 0.018;

        if mode == ControlMode::Natural {
            self.peer_coch_age = self.peer_coch_age.saturating_add(1);
            self.peer_echo_age = self.peer_echo_age.saturating_add(1);
            if peer_harmonic_now || peer_echo_now {
                self.peer_vocal_hold = 1.0;
            } else {
                self.peer_vocal_hold = sanitize(0.970 * self.peer_vocal_hold);
            }
        }
        let active_peer_hold = if mode == ControlMode::Natural && self.peer_vocal_hold > 0.22 {
            self.peer_vocal_hold
        } else {
            0.0
        };

        let mut lso_norm_now = [0.0_f32; PRED_BUCKETS];
        let mut lso_elder_ok = [false; PRED_BUCKETS];
        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            let ch_u = sp.channel as usize;
            if !sp.alive
                || sp.age_ticks < 16
                || sp.trophic_ema < 0.025
                || ch_u >= n_scout
                || membrane.scout_stats[ch_u].is_sparse_impulse()
                || is_monaural_coch[p]
                || is_world_retina[p]
            {
                continue;
            }
            let mean_p = self.pore_mean_v[p];
            let mut somatic_other_w = 0.0_f32;
            for m in 0..act_cap {
                if !is_world_loco[m] && Some(m) != m_glottal && self.vocal_artic_w[m] < 0.005 {
                    somatic_other_w += self.pore_action_w[m][p].abs();
                }
            }
            let is_coch_trough = self.is_coch_trough(p, min_h_ro[p], mean_p)
                && m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] < -0.016);
            let is_glottal_pos = m_glottal.is_some_and(|mg| self.pore_action_w[mg][p] >= 0.008);
            if is_coch_trough
                || is_glottal_pos
                || !self.is_lso_floor_pore(p, min_h_ro[p], mean_p)
                || somatic_other_w >= 0.038
            {
                continue;
            }
            let h = sp.hist_head as usize;
            if quiet_self_7 {
                let u_0 = self.deconv_pore_sym(p, &sp.history, h, 0);
                let d0 = u_0 - self.pore_bg_ild[p];
                if peer_harmonic_now
                    && sp.age_ticks >= 40
                    && lso_coh_cnt >= 3
                    && d0 * dom_sign_ro > 0.020
                {
                    lso_norm_now[p] = d0;
                    lso_elder_ok[p] = true;
                }
                if mode == ControlMode::Natural {
                    if sp.age_ticks < 40 {
                        self.pore_bg_ild[p] = if self.pore_bg_ild[p] == 0.0 {
                            sanitize_clamp(u_0, -0.70, 0.70)
                        } else {
                            sanitize_clamp(0.85 * self.pore_bg_ild[p] + 0.15 * u_0, -0.70, 0.70)
                        };
                    } else {
                        let bg_err = (u_0 - self.pore_bg_ild[p]).clamp(-0.045, 0.045);
                        let lr_bg = if !peer_harmonic_now && !peer_echo_now {
                            0.14
                        } else {
                            0.035
                        };
                        self.pore_bg_ild[p] =
                            sanitize_clamp(self.pore_bg_ild[p] + lr_bg * bg_err, -0.70, 0.70);
                    }
                }
            }
        }

        for p in 0..pore_cap {
            let sp = &membrane.sensor_pores[p];
            if !sp.alive || sp.age_ticks < 16 || sp.trophic_ema < 0.025 {
                continue;
            }
            let h = sp.hist_head as usize;
            let scale_p = self.pore_scale_v[p].max(0.015);
            let mean_p = self.pore_mean_v[p];
            let ch_u = sp.channel as usize;

            if !is_world_retina[p] && ch_u < n_scout {
                let is_echo = membrane.scout_stats[ch_u].is_sparse_impulse();
                if is_echo && self.pore_agency[p] < 0.035 {
                    let mut peak_echo_dev = 0.0_f32;
                    for lag in 0..=2 {
                        let dev_l = (sp.history[(h + 15 - lag) & 15] - mean_p).max(0.0);
                        if dev_l > peak_echo_dev {
                            peak_echo_dev = dev_l;
                        }
                    }
                    if quiet_echo_self && peak_echo_dev > 0.018 {
                        let s_call = sanitize_clamp(peak_echo_dev / scale_p, 0.0, 2.5);
                        for m in 0..act_cap {
                            let tw = self.tectal_map_w[m][p];
                            if tw.abs() >= 0.010 {
                                acoustic_taxis[m] =
                                    sanitize(acoustic_taxis[m] + 2.95 * tw * s_call);
                            }
                        }
                    }
                } else if !is_echo && lso_elder_ok[p] {
                    let d0 = lso_norm_now[p];
                    let s_call = sanitize_clamp(6.5 * d0, -2.5, 2.5);
                    for m in 0..act_cap {
                        let tw = self.tectal_map_w[m][p];
                        if tw.abs() >= 0.010 {
                            acoustic_taxis[m] = sanitize(acoustic_taxis[m] + 3.25 * tw * s_call);
                        }
                    }
                }
            }
        }

        let mut taxis_sq = 0.0_f32;
        for m in 0..act_cap {
            if max_breadth < 4 || is_world_loco[m] {
                taxis_sq += acoustic_taxis[m] * acoustic_taxis[m];
            }
        }
        let raw_taxis_norm = taxis_sq.sqrt();
        let mut cpa_flipped = false;
        if mode == ControlMode::Natural {
            if raw_taxis_norm > 0.030 {
                let mut old_sq = 0.0_f32;
                let mut dot_bear = 0.0_f32;
                for m in 0..act_cap {
                    if max_breadth < 4 || is_world_loco[m] {
                        let b_m = self.peer_bearing_ema[m];
                        let unit_m = acoustic_taxis[m] / raw_taxis_norm;
                        old_sq += b_m * b_m;
                        dot_bear += b_m * unit_m;
                    }
                }
                let old_norm = old_sq.sqrt();
                if old_norm > 0.14 && dot_bear < 0.0 {
                    cpa_flipped = true;
                    for m in 0..act_cap {
                        let unit_m = if max_breadth < 4 || is_world_loco[m] {
                            acoustic_taxis[m] / raw_taxis_norm
                        } else {
                            0.0
                        };
                        self.peer_bearing_ema[m] = sanitize_clamp(0.80 * unit_m, -1.0, 1.0);
                    }
                    if peer_harmonic_now {
                        self.peer_coch_fast = sanitize_clamp(coch_lvl_now, 0.0, 2.0);
                        self.peer_coch_age = 0;
                        self.peer_disp_coch = 0.0;
                    } else {
                        self.peer_coch_age = 255;
                        self.peer_disp_coch = 0.0;
                    }
                    if peer_echo_now {
                        self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                        self.peer_echo_age = 0;
                        self.peer_disp_echo = 0.0;
                    } else {
                        self.peer_echo_age = 255;
                        self.peer_disp_echo = 0.0;
                    }
                    self.social_loom_ema = sanitize(0.25 * self.social_loom_ema);
                } else {
                    for m in 0..act_cap {
                        let unit_m = if max_breadth < 4 || is_world_loco[m] {
                            acoustic_taxis[m] / raw_taxis_norm
                        } else {
                            0.0
                        };
                        self.peer_bearing_ema[m] = sanitize_clamp(
                            0.58 * self.peer_bearing_ema[m] + 0.42 * unit_m,
                            -1.0,
                            1.0,
                        );
                    }
                }
            } else {
                for m in 0..act_cap {
                    self.peer_bearing_ema[m] = sanitize(0.982 * self.peer_bearing_ema[m]);
                }
            }
        }
        let mut bear_sq = 0.0_f32;
        for m in 0..act_cap {
            bear_sq += self.peer_bearing_ema[m] * self.peer_bearing_ema[m];
        }
        let bear_norm = bear_sq.sqrt();
        if raw_taxis_norm < 0.030 && active_peer_hold > 0.25 && bear_norm > 0.16 {
            for m in 0..act_cap {
                if max_breadth < 4 || is_world_loco[m] {
                    acoustic_taxis[m] =
                        sanitize(0.88 * active_peer_hold * (self.peer_bearing_ema[m] / bear_norm));
                }
            }
        }
        let mut v_inst = vec![0.0_f32; ad];
        if let Some(ent15) = ring.at_lag(15)
            && ent15.committed
        {
            for m in 0..act_cap {
                v_inst[m] = sanitize(0.84 * ent15.motor(m));
            }
        }
        let mut v_by_lag = vec![vec![0.0_f32; ad]; 16];
        for lag in (1..=15).rev() {
            if let Some(ent) = ring.at_lag(lag)
                && ent.committed
            {
                for m in 0..act_cap {
                    v_inst[m] = sanitize((v_inst[m] + 0.185 * ent.motor(m)) * 0.82);
                }
            } else {
                for m in 0..act_cap {
                    v_inst[m] = sanitize(v_inst[m] * 0.82);
                }
            }
            v_by_lag[lag].clone_from(&v_inst);
        }
        let v_ego_now = if bear_norm > 0.12 {
            let mut proj = 0.0_f32;
            for (m, &vm) in v_by_lag[1].iter().enumerate().take(act_cap) {
                if max_breadth < 4 || is_world_loco[m] {
                    proj += (self.peer_bearing_ema[m] / bear_norm) * vm;
                }
            }
            let bear_conf = 0.62 * (bear_norm / 0.36).clamp(0.0, 1.0);
            sanitize_clamp(bear_conf * proj, -0.55, 0.55)
        } else {
            0.0
        };

        let mut loom_num = 0.0_f32;
        let mut loom_den = 0.0_f32;
        if mode == ControlMode::Natural && !cpa_flipped {
            self.peer_disp_coch = sanitize_clamp(self.peer_disp_coch + v_ego_now, -7.5, 7.5);
            self.peer_disp_echo = sanitize_clamp(self.peer_disp_echo + v_ego_now, -7.5, 7.5);

            if peer_harmonic_now {
                let dt_c = self.peer_coch_age;
                let drifted_c = dt_c > 16;
                if drifted_c || self.peer_coch_slow <= 0.014 {
                    self.peer_coch_fast = coch_lvl_now;
                    self.peer_coch_slow = coch_lvl_now;
                    self.peer_coch_age = 0;
                    self.peer_disp_coch = 0.0;
                } else if dt_c >= 3 {
                    let prev_fast = self.peer_coch_fast;
                    let u_mid = 0.5 * (prev_fast + coch_lvl_now);
                    let sens_c = ((0.10 + 0.48 * u_mid) * u_mid).clamp(0.0038, 0.045);
                    let disp_close_c = ((coch_lvl_now - prev_fast) / sens_c).clamp(-7.5, 7.5);
                    let v_peer_c = sanitize_clamp(
                        (disp_close_c - self.peer_disp_coch) / (dt_c as f32),
                        -0.75,
                        0.75,
                    );
                    self.peer_coch_fast = sanitize_clamp(coch_lvl_now, 0.0, 2.0);
                    self.peer_coch_slow =
                        sanitize_clamp(0.82 * self.peer_coch_slow + 0.18 * coch_lvl_now, 0.0, 2.0);
                    self.peer_coch_age = 0;
                    self.peer_disp_coch = 0.0;
                    loom_num = sanitize(loom_num + 1.35 * (v_peer_c / 0.085));
                    loom_den = sanitize(loom_den + 1.35);
                }
            }

            if peer_echo_now {
                let dt_e = self.peer_echo_age;
                let drifted_e = dt_e > 28 || (dt_e > 14 && self.peer_disp_echo.abs() > 2.4);
                if drifted_e || self.peer_echo_slow <= 0.014 {
                    self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_slow = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_age = 0;
                    self.peer_disp_echo = 0.0;
                } else if dt_e >= 3 {
                    let prev_fast = self.peer_echo_fast;
                    let a_mid = 0.5 * (prev_fast + max_echo_peer_now);
                    let sens_e = ((0.020 + 0.22 * a_mid) * a_mid).clamp(0.0035, 0.075);
                    let disp_close_e = ((max_echo_peer_now - prev_fast) / sens_e).clamp(-7.5, 7.5);
                    let v_peer_e = sanitize_clamp(
                        (disp_close_e - self.peer_disp_echo) / (dt_e as f32),
                        -0.75,
                        0.75,
                    );
                    self.peer_echo_fast = sanitize_clamp(max_echo_peer_now, 0.0, 2.0);
                    self.peer_echo_slow = sanitize_clamp(
                        0.82 * self.peer_echo_slow + 0.18 * max_echo_peer_now,
                        0.0,
                        2.0,
                    );
                    self.peer_echo_age = 0;
                    self.peer_disp_echo = 0.0;
                    loom_num = sanitize(loom_num + 1.80 * (v_peer_e / 0.085));
                    loom_den = sanitize(loom_den + 1.80);
                }
            }
        }

        let min_sample_age = self.peer_coch_age.min(self.peer_echo_age);
        if loom_den > 0.024 {
            let inst_loom = smooth_saturate(2.8 * loom_num / (loom_den + 0.05));
            if mode == ControlMode::Natural {
                self.social_loom_ema =
                    sanitize_clamp(0.28 * self.social_loom_ema + 0.72 * inst_loom, -1.0, 1.0);
            }
        } else if mode == ControlMode::Natural && !cpa_flipped {
            let decay = if min_sample_age <= 15 && active_peer_hold > 0.25 {
                0.945
            } else {
                0.62
            };
            self.social_loom_ema = sanitize(decay * self.social_loom_ema);
        }
        let peer_present_gate = (active_peer_hold * 1.6).clamp(0.0, 1.0);
        let foveated_loom = if mode == ControlMode::Natural {
            self.social_loom_ema * peer_present_gate
        } else {
            0.0
        };

        let ers = body.ers_scarcity.clamp(0.0, 1.0);
        let ex_scale = (0.12 + 0.68 * body.temperature).clamp(0.08, 0.90);
        let hunger = ((6.0 - body.atp) / 6.0).clamp(0.0, 1.0).max(thermal_need);
        let comfort_atp = (0.65 * body.max_atp).max(6.0);
        let unsated = ((comfort_atp - body.atp) / comfort_atp).clamp(0.0, 1.0);
        let eff_hunger = if !emb_eco {
            hunger.max(loco_mat * (0.28 + 0.48 * unsated))
        } else if active_embodied {
            let cruise_floor = if elder_forager { 0.26 } else { 0.18 };
            let adult_cruise = loco_mat
                * (0.30 * unsated).max(cruise_floor)
                * ((body.atp - 6.0) / 6.0).clamp(0.0, 1.0);
            hunger.max(adult_cruise)
        } else {
            hunger
        };
        let vigor = if active_embodied {
            if !emb_eco {
                1.0 + 0.90 * hunger + 0.35 * (eff_hunger - hunger).max(0.0) + 0.30 * ers
            } else {
                1.0 + 0.90 * eff_hunger + 0.30 * ers
            }
        } else {
            1.0 + 0.90 * hunger + 0.30 * ers
        };
        let starvation_guard = ((body.atp - 2.0) / 1.8).clamp(0.0, 1.0);
        let close_peer_suppress = (1.0
            - 5.0 * (coch_lvl_now - 0.115).max(0.0)
            - 4.0 * (max_echo_peer_now - 0.165).max(0.0))
        .clamp(0.0, 1.0);
        let long_social_forage_guard = if let Some(mg_t) = m_glottal {
            let (ep_t, op_t, _, _, _, _, _, om_t) = self.motor_contingency_stats(membrane, mg_t);
            let ol_pk = om_t[2].max(om_t[3]);
            let os_pk = om_t[0].max(om_t[1]).max(ep_t.max(0.0));
            if elder_forager && ep_t < 0.28 && op_t < 0.28 {
                0.0
            } else if ol_pk >= 0.28 && ol_pk > os_pk {
                ((body.atp - 76.0) / 16.0).clamp(0.0, 1.0)
            } else {
                1.0
            }
        } else if elder_forager {
            0.0
        } else {
            1.0
        };
        let taxis_gate = (starvation_guard
            * long_social_forage_guard
            * close_peer_suppress
            * (1.0 - 1.5 * acute_distress))
            .clamp(0.0, 1.0);
        let express_gate = ((1.0 - 0.14 * hunger) * (1.0 - acute_distress)).clamp(0.0, 1.0);
        let social_loom = express_gate * foveated_loom;
        let engage_sig = smooth_saturate(4.6 * social_loom);
        let approach_engage = engage_sig.clamp(0.0, 0.98);
        let recede_call = (-engage_sig).clamp(0.0, 0.98);
        let limb_loom_scale = (1.0 + 0.94 * approach_engage - 0.96 * recede_call).clamp(0.04, 1.94);
        let mut homeo_sq = 0.0_f32;
        for m in 0..act_cap {
            if max_breadth < 4 || is_world_loco[m] {
                homeo_sq += homeo_grad[m] * homeo_grad[m];
            }
        }
        let homeo_vec_norm = homeo_sq.sqrt();
        let vis_target_suppress = (1.0 - 1.35 * homeo_vec_norm).clamp(0.38, 1.0);
        let vis_occ_gate = if vis_prox_now < 0.180 {
            (1.0 - 2.6 * (vis_prox_now - 0.05).max(0.0)).clamp(0.26, 1.0)
        } else {
            (0.26 * (1.0 - (vis_prox_now - 0.180) / 0.030)).clamp(0.0, 0.26)
        };
        let social_drive_mag = 0.64 * vis_occ_gate;
        for m in 0..num_actuators {
            if m < ad {
                let social_va = if active_peer_hold > 0.25
                    && bear_norm > 0.16
                    && (max_breadth < 4 || is_world_loco[m])
                {
                    let u_peer_m = if raw_taxis_norm > 0.040 {
                        0.60 * (acoustic_taxis[m] / raw_taxis_norm)
                            + 0.40 * (self.peer_bearing_ema[m] / bear_norm)
                    } else {
                        self.peer_bearing_ema[m] / bear_norm
                    };
                    taxis_gate * vis_target_suppress * social_drive_mag * u_peer_m
                } else {
                    0.0
                };
                v_align[m] = sanitize_clamp(social_va, -0.78, 0.78);
            }
            model_grad[m] = sanitize_clamp(homeo_grad[m] + v_align[m], -1.5, 1.5);
        }
        let mut glottal_urge_gain = 0.0_f32;
        let mut glottal_listen_gain = 0.0_f32;
        let mut glottal_syl_exp = false;
        let mut glottal_syl_inhale = false;
        let mut mg_urge_active: Option<usize> = None;
        let mut mg_listen_active: Option<usize> = None;
        let mut sated_listen_mode = false;
        if emb_eco
            && mode == ControlMode::Natural
            && (body.rest_consolidation_ticks >= 480
                || ring.at_lag(1).is_some_and(|e| e.tick >= 3_200))
            && body.atp > 1.2
            && acute_distress < 0.96
            && ring_warm
        {
            let mut soft_coch_w_ro = vec![0.0_f32; ad];
            let mut soft_coch_cnt_ro = vec![0u16; ad];
            let mut sat_coch_cnt_ro = vec![0u16; ad];
            for p in 0..pore_cap {
                let sp = &membrane.sensor_pores[p];
                let ch_u = sp.channel as usize;
                if !sp.alive
                    || sp.age_ticks < 16
                    || sp.trophic_ema < 0.025
                    || ch_u >= n_scout
                    || is_world_retina[p]
                    || membrane.scout_stats[ch_u].is_sparse_impulse()
                {
                    continue;
                }
                let is_pure_ro = self.is_pure_coch_floor(p, min_h_ro[p], self.pore_mean_v[p]);
                let is_quiet_ro = self.is_quiet_coch_floor(p, min_h_ro[p], self.pore_mean_v[p]);
                if is_pure_ro || is_quiet_ro {
                    let pw_bar = if is_pure_ro { 0.0045 } else { 0.0120 };
                    for m in 0..act_cap {
                        let pw = self.pore_action_w[m][p];
                        if pw >= pw_bar {
                            soft_coch_w_ro[m] += pw;
                            soft_coch_cnt_ro[m] = soft_coch_cnt_ro[m].saturating_add(1);
                        }
                    }
                } else if (self.z_from_bank(p, min_h_ro[p], min_h_ro[p]) > 0.19
                    || self.z_from_bank(p, min_h_ro[p], self.pore_mean_v[p]) > 0.45)
                    && (self.pore_floor_v[p] - self.receptor_ref_floor(p, min_h_ro[p]))
                        * self.inv_acoustic_gain()
                        >= -0.012
                {
                    for m in 0..act_cap {
                        if self.pore_action_w[m][p] >= 0.006 {
                            sat_coch_cnt_ro[m] = sat_coch_cnt_ro[m].saturating_add(1);
                        }
                    }
                }
            }
            let mut mg_urge_opt = m_glottal;
            if mg_urge_opt.is_none() {
                let mut best_my = 0.14_f32;
                for m in 0..act_cap {
                    if is_world_loco[m] {
                        continue;
                    }
                    let sat_apnea = cells.alive_count < 150
                        && mastery_by_act[m] >= 0.52
                        && soft_coch_w_ro[m] < 0.008
                        && sat_coch_cnt_ro[m] >= 2
                        && (1..=6).all(|d| {
                            ring.at_lag(d)
                                .is_some_and(|e| e.committed && e.motor(m) > 0.05)
                        });
                    let (ep, op, _, _, ec_tot, oc_tot, _, om_b) =
                        self.motor_contingency_stats(membrane, m);
                    let pos_ok = soft_coch_w_ro[m] >= 0.008;
                    let em = if (pos_ok || sat_apnea) && ec_tot >= 3 {
                        ep
                    } else {
                        0.0
                    };
                    let om_min = om_b[0].min(om_b[1]).min(om_b[2]).min(om_b[3]);
                    let om_pos = if (pos_ok || sat_apnea) && oc_tot >= 3 {
                        op
                    } else {
                        0.0
                    };
                    let om_neg = if oc_tot >= 3
                        && ep < 0.28
                        && om_min <= -0.20
                        && (soft_coch_cnt_ro[m] >= 1
                            || tot_pore_reaff[m] >= 0.012
                            || max_asw_jac[m] >= 0.038)
                        && (soft_coch_w_ro[m] >= 0.004
                            || tot_pore_reaff[m] >= 0.012
                            || max_asw_jac[m] >= 0.038)
                    {
                        -0.85 * om_min
                    } else {
                        0.0
                    };
                    let pk = if sat_apnea {
                        if em.max(om_pos) >= 0.65 {
                            em.max(om_pos)
                        } else {
                            0.0
                        }
                    } else {
                        em.max(om_pos).max(om_neg)
                    };
                    if pk > best_my {
                        best_my = pk;
                        mg_urge_opt = Some(m);
                    }
                }
            }
            if mg_urge_opt.is_none() {
                let mut best_sw = 0.018_f32;
                for m in 0..act_cap {
                    if !is_world_loco[m]
                        && (tot_pore_reaff[m] >= 0.020
                            || max_asw_jac[m] >= 0.038
                            || (soft_coch_cnt_ro[m] >= 4 && soft_coch_w_ro[m] >= 0.048))
                        && soft_coch_cnt_ro[m] >= 2
                        && soft_coch_w_ro[m] > best_sw
                    {
                        best_sw = soft_coch_w_ro[m];
                        mg_urge_opt = Some(m);
                    }
                }
            }
            if let Some(mg_act) = mg_urge_opt {
                let (e_peak, o_peak, e_mean, o_mean, _, _, _, om_band) =
                    self.motor_contingency_stats(membrane, mg_act);
                let u_even = e_peak.max(0.0);
                let u_odd = o_peak.max(0.0);
                let vocal_sensitive_ro =
                    body.rest_consolidation_ticks < 9_000 || u_even >= 0.72 || u_odd >= 0.55;
                let contingent_urge = if vocal_sensitive_ro {
                    u_even.max(u_odd).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let social_turn = u_odd > u_even;
                glottal_urge_gain = ((contingent_urge - 0.28) / 0.28).clamp(0.0, 1.0);
                if glottal_urge_gain >= 0.25 {
                    mg_urge_active = Some(mg_act);
                    let mut q_urge = [false; 17];
                    q_urge[0] = true;
                    for (d, ql) in q_urge.iter_mut().enumerate().skip(1) {
                        *ql = ring
                            .at_lag(d)
                            .is_some_and(|e| !e.committed || e.motor(mg_act) <= -0.05);
                    }
                    let jit = ring.at_lag(1).map_or(0, |e| (e.tick & 1) as usize);
                    glottal_syl_exp = if social_turn {
                        let l_eff = self.vocal_eff_hist.trailing_zeros() as usize;
                        let tick_now = ring.at_lag(1).map_or(0, |e| e.tick.saturating_add(1));
                        let k_eff = if l_eff >= 28 {
                            7
                        } else {
                            let t_anchor = tick_now.wrapping_sub(l_eff as u64);
                            const SOCIAL_KEFF: [usize; 4] = [7, 8, 9, 22];
                            SOCIAL_KEFF
                                [((t_anchor.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize]
                        };
                        let q_s0 = (1..=7).all(|d| q_urge[d]) && l_eff >= k_eff;
                        let q_s1 = !q_urge[1] && (2..=8).all(|d| q_urge[d]);
                        let q_s2 = !q_urge[1] && !q_urge[2] && (3..=9).all(|d| q_urge[d]);
                        let q_s3 =
                            !q_urge[1] && !q_urge[2] && !q_urge[3] && (4..=10).all(|d| q_urge[d]);
                        q_s0 || q_s1 || q_s2 || q_s3
                    } else {
                        let k_s = 8 + jit;
                        let q_8_now = (1..=k_s).all(|d| q_urge[d]);
                        let q_8_s1 = !q_urge[1] && (2..=(k_s + 1)).all(|d| q_urge[d]);
                        let q_8_s2 =
                            !q_urge[1] && !q_urge[2] && (3..=(k_s + 2)).all(|d| q_urge[d]);
                        let q_8_s3 = !q_urge[1]
                            && !q_urge[2]
                            && !q_urge[3]
                            && (4..=(k_s + 3)).all(|d| q_urge[d]);
                        q_8_now || q_8_s1 || q_8_s2 || q_8_s3
                    };
                    glottal_syl_inhale = !glottal_syl_exp;
                    if glottal_syl_exp {
                        let pyr_mean = if pyr_env_cnt[mg_act] > 0 {
                            pyr_env_sum[mg_act] / (pyr_env_cnt[mg_act] as f32)
                        } else {
                            0.25
                        };
                        let corticobulbar_drive =
                            glottal_urge_gain * (0.58 + 1.05 * pyr_mean.clamp(0.12, 0.75));
                        tube_drive[mg_act] = sanitize(
                            (tube_drive[mg_act] + corticobulbar_drive)
                                .max(0.78 * glottal_urge_gain),
                        );
                    }
                } else if e_peak < 0.32 && o_peak < 0.18 {
                    let o_short_mean = 0.5 * (om_band[0] + om_band[1]);
                    let o_long_mean = 0.5 * (om_band[2] + om_band[3]);
                    let o_long_min = om_band[2].min(om_band[3]);
                    let o_all_min = om_band[0].min(om_band[1]).min(o_long_min);
                    let adult_pruned = (8 * num_actuators..150).contains(&cells.alive_count);
                    let mature_social_age = ring.at_lag(1).is_some_and(|e| e.tick >= 9_000)
                        || cells
                            .slots
                            .first()
                            .is_some_and(|c| c.alive && c.age_ticks >= 9_000);
                    let adult_pruned_listen = adult_pruned
                        && body.atp > 24.0
                        && o_short_mean <= -0.25
                        && o_peak <= 0.04
                        && e_mean <= -0.22
                        && e_peak < 0.08
                        && soft_coch_cnt_ro[mg_act] >= 4;
                    let yoked_social_listen = mature_social_age
                        && body.atp > 6.0
                        && e_peak < 0.24
                        && e_mean < 0.05
                        && o_peak <= 0.075
                        && o_all_min <= -0.22
                        && (soft_coch_cnt_ro[mg_act] >= 1
                            || tot_pore_reaff[mg_act] >= 0.012
                            || max_asw_jac[mg_act] >= 0.038);
                    let sated_adult_listen = adult_pruned_listen || yoked_social_listen;
                    let dev_listen = !sated_adult_listen
                        && cells.alive_count >= 150
                        && soft_coch_cnt_ro[mg_act] >= 4
                        && (o_mean <= -0.20 || o_all_min <= -0.22)
                        && o_peak <= 0.05
                        && e_peak < 0.08
                        && e_mean < 0.02;
                    if (sated_adult_listen || dev_listen) && body.atp > 2.0 && acute_distress < 0.85
                    {
                        mg_listen_active = Some(mg_act);
                        sated_listen_mode = sated_adult_listen;
                        let min_o = o_short_mean.min(o_long_mean).min(o_all_min).min(o_mean);
                        glottal_listen_gain = if sated_adult_listen {
                            ((-min_o - 0.18) / 0.08).clamp(0.0, 1.0)
                        } else {
                            ((-min_o - 0.18) / 0.10).clamp(0.0, 1.0)
                        };
                        let listen_pull = if adult_pruned_listen {
                            0.42
                        } else if yoked_social_listen {
                            0.32
                        } else {
                            0.14
                        };
                        tube_drive[mg_act] =
                            sanitize(tube_drive[mg_act] - listen_pull * glottal_listen_gain);
                    }
                }
            }
        }

        let idle_ers_damp = 1.0 - 0.45 * ers;
        let mut has_survival_vec = vec![false; num_actuators];
        let mut is_glottal_vec = vec![false; num_actuators];
        let mut limb_share_vec = vec![0.0_f32; num_actuators];
        let mut emit_share_vec = vec![0.0_f32; num_actuators];
        let mut candidates: [Vec<f32>; 5] = std::array::from_fn(|_| vec![0.0_f32; num_actuators]);
        for m in 0..num_actuators {
            let is_loco_m = m < ad && is_world_loco[m];
            let is_glottal_act =
                Some(m) == m_glottal || Some(m) == mg_urge_active || Some(m) == mg_listen_active;
            let is_urge_glottal = Some(m) == mg_urge_active;
            let is_urge_vocal = is_urge_glottal;
            is_glottal_vec[m] = is_glottal_act;
            let is_echo_emitter = Some(m) != mg_urge_active
                && Some(m) != mg_listen_active
                && Some(m) == m_echo_emitter;
            let som_dom = if m < ad {
                somatic_prop_reaff[m].max(0.5 * self.somatic_limb_w[m])
            } else {
                0.0
            };
            let is_vocal_artic = m < ad
                && !is_loco_m
                && !is_glottal_act
                && !is_echo_emitter
                && ((self.vocal_artic_w[m] >= 0.005 && abs_cochlea_reaff[m] > som_dom)
                    || (coch_artic_cnt[m] >= 2
                        && abs_cochlea_reaff[m] >= 0.011
                        && abs_cochlea_reaff[m] > 1.2 * som_dom));
            let is_somatic_limb = m < ad
                && !is_loco_m
                && !is_glottal_act
                && !is_echo_emitter
                && !is_vocal_artic
                && ((somatic_prop_reaff[m] >= 0.022
                    && somatic_prop_reaff[m] >= abs_cochlea_reaff[m])
                    || self.somatic_limb_w[m] >= 0.022);
            let homeo_m_abs = (homeo_grad[m] * vigor).abs();
            let raw_td = tube_drive[m] * vigor;
            let surv_thresh = if active_embodied && is_loco_m {
                0.065
            } else {
                0.14
            };
            let has_survival_drive = if elder_forager && mvt_patch_depleted && is_loco_m {
                homeo_m_abs >= 0.065
            } else {
                raw_td.abs() + homeo_m_abs >= surv_thresh
                    || (active_embodied && is_loco_m && v_align[m].abs() >= 0.06)
            };
            has_survival_vec[m] = has_survival_drive;
            let nmj_gate = ((mp_cond_by_act[m] - 0.16) * 3.5).clamp(0.0, 1.0);
            let limb_share = if is_somatic_limb && body.atp > 1.3 {
                (1.0 - 2.0 * homeo_grad[m].abs() - 2.0 * acute_distress).clamp(0.0, 1.0) * nmj_gate
            } else {
                0.0
            };
            let emit_share = if is_echo_emitter {
                (1.0 - 2.0 * acute_distress).clamp(0.0, 1.0) * nmj_gate
            } else {
                0.0
            };
            limb_share_vec[m] = limb_share;
            emit_share_vec[m] = emit_share;
            let mod_scale = 1.0 + limb_share * (limb_loom_scale - 1.0);
            let td = raw_td * mod_scale;
            let mg = model_grad[m] * vigor;
            let mastery = mastery_by_act[m];
            let stiff_c = stiff_motor_conf[m].min(1.0);
            let urge_babble_atten = if is_urge_vocal && glottal_syl_exp {
                1.0 - 0.75 * glottal_urge_gain
            } else if is_vocal_artic && mg_urge_active.is_some() {
                1.0 - 0.68 * glottal_urge_gain
            } else if Some(m) == mg_listen_active {
                1.0 - (if sated_listen_mode { 0.88 } else { 0.52 }) * glottal_listen_gain
            } else {
                1.0
            };
            let babble_maturity =
                (1.0 - 0.35 * mastery) * (1.0 - 0.20 * stiff_c) * urge_babble_atten;
            let ep_listen_atten = if Some(m) == mg_listen_active && sated_listen_mode {
                1.0 - 0.85 * glottal_listen_gain
            } else {
                1.0
            };
            let ep = if !has_survival_drive {
                epistemic_drive[m] * idle_ers_damp * mod_scale * ep_listen_atten
            } else {
                0.0
            };
            let newborn_spontaneity = 0.28 * (1.0 - mastery);
            let idle_ex_gate = (if ep.abs() > 0.05 {
                0.38 * babble_maturity + 0.12
            } else {
                (0.08 + newborn_spontaneity + 0.20 * noise_ep_surplus.min(1.0)) * babble_maturity
            }) * idle_ers_damp
                * mod_scale;
            let surv_mod = if homeo_m_abs < 0.04 { mod_scale } else { 1.0 };
            let local_ex_scale = if has_survival_drive {
                ex_scale * (0.45 + 0.55 * babble_maturity) * surv_mod
            } else {
                ex_scale * idle_ex_gate
            };
            let ex = explore_wave[m];
            let ex2 = explore_alt[m];
            let pa = prev_act[m];

            candidates[0][m] =
                smooth_saturate(2.35 * td + 1.45 * mg + ep + 0.14 * local_ex_scale * ex);
            candidates[1][m] = smooth_saturate(
                1.75 * td
                    + 1.15 * mg
                    + ep
                    + (0.65 * klino_gain_mean.abs().max(0.15) * pa) * mod_scale.min(1.15)
                    + 0.14 * local_ex_scale * ex,
            );
            let td_c2 = if is_urge_vocal && glottal_syl_exp {
                1.60 * td
            } else {
                0.95 * td
            };
            candidates[2][m] =
                smooth_saturate(2.25 * mg + td_c2 + ep + 0.28 * local_ex_scale * ex2);
            let td_c3 = if is_urge_vocal && glottal_syl_exp {
                1.55 * td
            } else {
                0.80 * td
            };
            let td_c4 = if is_urge_vocal && glottal_syl_exp {
                1.50 * td
            } else {
                0.55 * td
            };
            candidates[3][m] = smooth_saturate(
                td_c3
                    + 0.80 * mg
                    + ep
                    + (0.42 + 0.48 * ex_scale) * babble_maturity * idle_ers_damp * mod_scale * ex,
            );
            candidates[4][m] = smooth_saturate(
                td_c4
                    + 0.55 * mg
                    + 0.82 * ex_scale * babble_maturity * idle_ers_damp * mod_scale * ex2,
            );
        }

        let pain_bar = 0.045;
        let mut policy_conf = 0.0_f32;
        let mut max_act_conf = 0.0_f32;
        for (m, &pv) in tube_pred_action_val.iter().enumerate() {
            let ac = pv.abs() + 0.8 * homeo_grad[m].abs();
            policy_conf += ac;
            if ac > max_act_conf {
                max_act_conf = ac;
            }
        }
        let mean_policy_conf = policy_conf / (num_actuators.max(1) as f32);
        let eff_policy_conf = mean_policy_conf.max(0.42 * max_act_conf);
        let eff_probe = probe_window && (!active_embodied || probe_surplus > 0.08);
        let forage_thresh = 0.0067 + 0.0065 * eff_hunger;
        let forage_search_w = if eff_policy_conf < forage_thresh {
            0.035 * body.temperature.max(0.28 * loco_mat)
        } else if eff_probe {
            0.12 * probe_surplus
        } else {
            0.0
        };
        let effort_w = if eff_policy_conf < forage_thresh || eff_probe {
            0.0
        } else {
            0.065 * (1.35 - 0.65 * atp_surplus.min(1.5)) * (1.0 - 0.70 * eff_hunger)
        };
        let epistemic_w = 1.85 * noise_ep_surplus;
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut rehearsals = 0u64;

        for (k_idx, cand) in candidates.iter().enumerate() {
            let mut pred_val = 0.0_f32;
            let mut epistemic_val = 0.0_f32;
            let mut act_energy = 0.0_f32;
            let mut brake_pen = 0.0_f32;
            let mut cont_dot = 0.0_f32;

            for m in 0..num_actuators {
                act_energy += cand[m] * cand[m];
                brake_pen += slow_inhib_brake[m] * cand[m].abs();
                cont_dot += cand[m] * prev_act[m];
                let urge_exp_m = if is_glottal_vec[m] && glottal_syl_exp {
                    0.085 * glottal_urge_gain
                } else {
                    0.0
                };
                pred_val += (tube_pred_action_val[m] + 0.55 * v_align[m] + urge_exp_m) * cand[m];
            }

            for &(c_idx, v_now, dv_now) in &sensory_cells {
                let b = c_idx as usize;
                let self_dv = self.sensory_self_w[b] * dv_now;
                let mut act_dv = 0.0_f32;
                let mut ep_act_dv = 0.0_f32;
                for (m, &am) in cand.iter().enumerate().take(ad) {
                    let jac = self.action_sensory_w[m][b];
                    let da_m = am - self.act_mean[m];
                    act_dv += jac * da_m;
                    if jac.abs() > 0.026
                        && !has_survival_vec[m]
                        && (!active_embodied || is_world_loco[m])
                    {
                        ep_act_dv += (jac * da_m).abs();
                    }
                }
                let p_null = smooth_saturate(v_now + self_dv);
                let v_step1 = smooth_saturate(v_now + self_dv + act_dv);
                let delta_from_act = v_step1 - p_null;
                let cell = &cells.slots[c_idx as usize];
                pred_val +=
                    0.50 * cell.v_homeo * delta_from_act + 0.75 * cell.v_delta_homeo * act_dv;
                if cell.surprise_ema < 0.45
                    && (cell.v_homeo.abs() + cell.v_delta_homeo.abs()) < 0.18
                {
                    epistemic_val += ep_act_dv;
                }
            }

            pred_val += klino_gain_mean * recent_val_delta * cont_dot;
            pred_val -= 0.75 * brake_pen;
            pred_val = sanitize(pred_val);

            let vetoed = pred_val < -pain_bar;
            if vetoed {
                rehearsals += 1;
            }
            let mean_sq_energy = act_energy / (num_actuators as f32);
            let rms = mean_sq_energy.sqrt();
            let mut total_score = pred_val + epistemic_w * epistemic_val + forage_search_w * rms
                - effort_w * mean_sq_energy;
            if vetoed {
                total_score -= 1.5;
            }

            if total_score > best_score {
                best_score = total_score;
                best_idx = k_idx;
            }
        }

        let chosen = candidates[best_idx].clone();

        let mut max_abs = 0.0_f32;
        for &v in &chosen {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }

        let committed = max_abs >= body.motor_clutch_threshold;
        let final_out = if committed {
            let mut smoothed = chosen;
            for m in 0..num_actuators {
                let maturity = (0.65 * mastery_by_act[m] + 0.35 * stiff_motor_conf[m].min(1.0))
                    .clamp(0.0, 1.0);
                let is_loco_m = m < ad && is_world_loco[m];
                let ls = limb_share_vec[m];
                let emit_s = emit_share_vec[m];
                let social_active = mode == ControlMode::Natural && active_peer_hold > 0.25;
                let recent_breath_ok = if explore_wave[m] + 0.65 * explore_alt[m] < -0.15 {
                    has_recent_breath_13
                } else {
                    has_recent_breath_14
                };
                let urge_inhale_m = Some(m) == mg_urge_active
                    && mode == ControlMode::Natural
                    && glottal_syl_inhale
                    && acute_distress < 0.96;
                let unconfirmed_breath_m = Some(m) == m_glottal
                    && mg_urge_active.is_none()
                    && mode == ControlMode::Natural
                    && acute_distress < 0.75
                    && ring.at_lag(1).is_some_and(|e| e.tick >= 1_200)
                    && (self.vocal_eff_hist & 0x01) == 0
                    && (self.vocal_eff_hist & 0x0001_FFFE) != 0;
                let soft_inhale_m = urge_inhale_m || unconfirmed_breath_m;
                let hb_inhale = is_glottal_vec[m]
                    && mode == ControlMode::Natural
                    && ((!recent_breath_ok && acute_distress < 0.40) || soft_inhale_m);
                let spindle_alpha = if emit_s > 0.05 || hb_inhale {
                    if soft_inhale_m && recent_breath_ok {
                        0.94
                    } else {
                        0.92
                    }
                } else if ls > 0.05 && social_active {
                    0.93
                } else if is_glottal_vec[m] {
                    if glottal_syl_exp && glottal_urge_gain >= 0.25 {
                        0.88
                    } else {
                        (0.86 - 0.18 * maturity).clamp(0.68, 0.86)
                    }
                } else if has_survival_vec[m] || is_loco_m {
                    (0.92 - 0.30 * maturity).clamp(0.58, 0.92)
                } else {
                    (0.95 - 0.80 * maturity).clamp(0.38, 0.95)
                };
                let urge_open_artic = mg_urge_active.is_some()
                    && m < ad
                    && !is_loco_m
                    && !is_glottal_vec[m]
                    && (self.vocal_artic_w[m] >= 0.004 || abs_cochlea_reaff[m] >= 0.008);
                let raw_t = if urge_open_artic {
                    smoothed[m] * (1.0 - 0.64 * glottal_urge_gain)
                } else if emit_s > 0.05 {
                    let ping_bout = mode == ControlMode::Natural
                        && body.atp > 1.2
                        && quiet_echo_self
                        && explore_alt[m] > 0.78;
                    if ping_bout {
                        (0.86 * emit_s).clamp(0.0, 0.90)
                    } else {
                        0.0
                    }
                } else if hb_inhale {
                    if soft_inhale_m && recent_breath_ok {
                        -0.28
                    } else {
                        -0.38
                    }
                } else if Some(m) == mg_urge_active && glottal_syl_exp {
                    smoothed[m].max(0.74 * glottal_urge_gain)
                } else if Some(m) == mg_listen_active {
                    if sated_listen_mode {
                        smoothed[m] * (1.0 - 0.72 * glottal_listen_gain)
                            - 0.14 * glottal_listen_gain
                    } else {
                        smoothed[m] - 0.12 * glottal_listen_gain
                    }
                } else if ls > 0.05 && social_active {
                    let base_s = smoothed[m] * (1.0 - 0.28 * ls * peer_present_gate);
                    let shunted = base_s * (1.0 - 0.96 * ls * recede_call);
                    let unit_dir = if smoothed[m].abs() >= 0.04 {
                        smoothed[m].signum()
                    } else if prev_act[m].abs() >= 0.04 {
                        prev_act[m].signum()
                    } else if explore_wave[m] >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let headroom = (0.96 - shunted.abs()).max(0.0);
                    shunted + ls * approach_engage * headroom * unit_dir
                } else if is_loco_m
                    && active_peer_hold > 0.25
                    && bear_norm > 0.16
                    && acute_distress < 0.35
                {
                    let brake_safe = (1.0 - 2.0 * slow_inhib_brake[m]).clamp(0.0, 1.0);
                    if vis_prox_now < 0.180 && body.atp > 2.0 {
                        smoothed[m] + 0.62 * brake_safe * v_align[m]
                    } else {
                        smoothed[m]
                    }
                } else {
                    smoothed[m]
                };
                let target_act = sanitize_clamp(raw_t, -1.0, 1.0);
                smoothed[m] = sanitize_clamp(
                    (1.0 - spindle_alpha) * prev_act[m] + spindle_alpha * target_act,
                    -1.0,
                    1.0,
                );
            }
            smoothed
        } else {
            rehearsals += 1;
            vec![0.0_f32; num_actuators]
        };

        let mut pred_with = Vec::with_capacity(sensory_cells.len());
        let mut pred_null = Vec::with_capacity(sensory_cells.len());
        for &(c_idx, v_now, dv_now) in &sensory_cells {
            let b = c_idx as usize;
            let base_dv = self.sensory_self_w[b] * dv_now;
            let p_null = smooth_saturate(v_now + base_dv);
            let mut act_dv = 0.0_f32;
            if committed {
                for (m, &am) in final_out.iter().enumerate().take(ad) {
                    let w_mb = self.action_sensory_w[m][b];
                    if w_mb.abs() >= 0.022 {
                        act_dv += w_mb * (am - self.act_mean[m]);
                    }
                }
            }
            let p_act = smooth_saturate(v_now + base_dv + act_dv);
            pred_with.push((c_idx, p_act));
            pred_null.push((c_idx, p_null));
        }

        RolloutOutcome {
            pulse: FieldPulse {
                actuator_out: final_out,
            },
            committed,
            rehearsals,
            pred_with,
            pred_null,
        }
    }

    #[must_use]
    pub fn topology_counts(&self, membrane: &PoreMembrane) -> (usize, usize, usize) {
        let mut stiff = 0usize;
        let mut inhib = 0usize;
        let mut delay_tubes = 0usize;

        for t in &self.slots {
            if !t.alive {
                continue;
            }
            if t.stiff_strength() > 0.06 {
                stiff += 1;
            }
            if t.u[0] < -0.015 {
                inhib += 1;
            }
            if t.delay >= 2 && t.u[0].abs() > 0.02 {
                delay_tubes += 1;
            }
        }
        for sp in &membrane.sensor_pores {
            if sp.alive && sp.delay >= 2 && sp.conductance > 0.03 {
                delay_tubes += 1;
            }
        }
        (stiff, inhib, delay_tubes)
    }

    #[must_use]
    pub fn predict_sensory_cells_for_pulse(
        &self,
        cells: &CellArena,
        final_out: &[f32],
        committed: bool,
    ) -> (Vec<(u32, f32)>, Vec<(u32, f32)>) {
        let ad = self.act_dim();
        let mut sensory_cells: Vec<(u32, f32, f32)> = Vec::with_capacity(32);
        for &c_idx in &cells.active_indices {
            if sensory_cells.len() >= 32 {
                break;
            }
            if let Some(c) = cells.slots.get(c_idx as usize)
                && c.alive
                && c.sensory_trophic_ema > 0.005
            {
                sensory_cells.push((c_idx, c.v_real, c.delta_v));
            }
        }
        let mut pred_with = Vec::with_capacity(sensory_cells.len());
        let mut pred_null = Vec::with_capacity(sensory_cells.len());
        for &(c_idx, v_now, dv_now) in &sensory_cells {
            let b = c_idx as usize;
            let base_dv = self.sensory_self_w[b] * dv_now;
            let p_null = smooth_saturate(v_now + base_dv);
            let mut act_dv = 0.0_f32;
            if committed {
                for (m, &am) in final_out.iter().enumerate().take(ad) {
                    let w_mb = self.action_sensory_w[m][b];
                    if w_mb.abs() >= 0.022 {
                        act_dv += w_mb * (am - self.act_mean[m]);
                    }
                }
            }
            let p_act = smooth_saturate(v_now + base_dv + act_dv);
            pred_with.push((c_idx, p_act));
            pred_null.push((c_idx, p_null));
        }
        (pred_with, pred_null)
    }

    pub fn step_striatal_ofc(
        &mut self,
        ambient_in: &[f32],
        metabolic_influx: f32,
        tissue_stress: f32,
        num_actuators: usize,
        emb_eco: bool,
        membrane: &mut PoreMembrane,
        mode: ControlMode,
    ) -> Result<(Option<Vec<f32>>, u64), PersistError> {
        if num_actuators == 0 || emb_eco {
            return Ok((None, 0));
        }
        let is_plastic = mode == ControlMode::Natural;
        if is_plastic {
            self.striatal.ensure_actuators(num_actuators);
        }
        let (na, tap_dim, w_len, fwd_len) = self.striatal.dims();
        let act_s = num_actuators.min(na);
        let act_scale = 6.0 / (act_s.max(6) as f32);
        let canc_decay = 1.0 - 0.001 * act_scale;
        let prune_warmup = 44u32.max((3 * act_s) as u32);
        let j_decay = 1.0 - 0.0005 * act_scale;
        let reward = sanitize(metabolic_influx).max(0.0) - sanitize(tissue_stress).max(0.0);

        let mut ring_norm_sq = 0.0_f32;
        for tap in 0..STRIATAL_TAPS {
            let r_off = tap * na;
            for m in 0..act_s {
                let d = self.striatal.da_ring[r_off + m];
                ring_norm_sq += d * d;
            }
        }
        let eta_g = 0.08 / (ring_norm_sq + 0.15);

        let pore_cap = membrane.sensor_pores.len().min(STRIATAL_PORES);
        let n_scout = membrane.scout_stats.len();
        let num_c_init = (self.striatal.num_ctx as usize).min(OFC_CONTEXTS);
        let has_any_env = (0..num_c_init).any(|c| {
            self.striatal.ctx_has_cons[c] != 0
                && self.striatal.hot_dend_env[c]
                    .iter()
                    .any(|&ev| (ev & 1) != 0)
        }) || self
            .striatal
            .paged_cols
            .iter()
            .any(|e| e.dend_env.iter().any(|&ev| (ev & 1) != 0));

        let mut live_pore_ch = [u32::MAX; STRIATAL_PORES];
        let mut raw_x_arr = [0.0_f32; STRIATAL_PORES];
        let mut is_white_arr = [false; STRIATAL_PORES];
        let mut is_dead_flat_arr = [false; STRIATAL_PORES];
        let mut quad_cand = [false; STRIATAL_PORES];
        let mut tonic_valid = [false; STRIATAL_PORES];
        let mut tonic_now = [0.0_f32; STRIATAL_PORES];
        let mut prony_valid_arr = [false; STRIATAL_PORES];
        let mut apical_carrier = [false; STRIATAL_PORES];
        let mut feat_active = [false; STRIATAL_PORES];
        let mut pore_sig = [0.0_f32; STRIATAL_PORES];
        let mut c1_wave_pores = 0usize;
        let mut max_smooth_raw = 0.0_f32;
        let mut n_quad_cand = 0usize;
        let mut n_non_carr = 0usize;

        for p in 0..STRIATAL_PORES {
            if p >= pore_cap || !membrane.sensor_pores[p].alive {
                if is_plastic && self.striatal.pore_ch[p] != u32::MAX {
                    self.striatal.reset_pore_slot(p, u32::MAX);
                }
                continue;
            }
            let ch = membrane.sensor_pores[p].channel;
            live_pore_ch[p] = ch;
            if is_plastic && self.striatal.pore_ch[p] != ch {
                self.striatal.reset_pore_slot(p, ch);
            }
            let ch_u = ch as usize;
            let raw_x = sanitize_clamp(ambient_in.get(ch_u).copied().unwrap_or(0.0), -1.0, 1.0);
            raw_x_arr[p] = raw_x;

            if is_plastic {
                if self.striatal.pore_age[p] == 0 {
                    self.striatal.pore_carr_mean[p] = raw_x;
                    self.striatal.pore_prev_x[p] = raw_x;
                    self.striatal.pore_carr_var[p] = 1.0e-4;
                    self.striatal.pore_var_dx[p] = 1.0e-5;
                    self.striatal.pore_orb_err[p] = 1.0;
                } else {
                    let dx = sanitize(raw_x - self.striatal.pore_prev_x[p]);
                    let mid = sanitize(
                        0.5 * (raw_x + self.striatal.pore_prev_x[p])
                            - self.striatal.pore_carr_mean[p],
                    );
                    self.striatal.pore_carr_mean[p] = sanitize_clamp(
                        self.striatal.pore_carr_mean[p]
                            + 0.03 * (raw_x - self.striatal.pore_carr_mean[p]),
                        -1.0,
                        1.0,
                    );
                    self.striatal.pore_carr_var[p] =
                        sanitize(0.96 * self.striatal.pore_carr_var[p] + 0.04 * mid * mid);
                    self.striatal.pore_var_dx[p] =
                        sanitize(0.96 * self.striatal.pore_var_dx[p] + 0.04 * dx * dx);
                    let r2 = (mid * mid) / (2.0 * self.striatal.pore_carr_var[p].max(1.0e-7))
                        + (dx * dx) / (2.0 * self.striatal.pore_var_dx[p].max(1.0e-8));
                    let r_dev = (r2 - 1.0).abs().min(4.0);
                    self.striatal.pore_orb_err[p] =
                        sanitize(0.96 * self.striatal.pore_orb_err[p] + 0.04 * r_dev);
                    self.striatal.pore_prev_x[p] = raw_x;
                }
            }

            if membrane.sensor_pores[p].age_ticks >= 1 {
                let (dc_v, pr_ok) = harmonic_dc_from_history(
                    &membrane.sensor_pores[p].history,
                    membrane.sensor_pores[p].hist_head,
                    membrane.sensor_pores[p].age_ticks,
                );
                tonic_now[p] = dc_v;
                tonic_valid[p] = true;
                prony_valid_arr[p] = pr_ok;
            }

            let is_white = ch_u < n_scout
                && membrane.scout_stats[ch_u].initialized
                && membrane.scout_stats[ch_u].var_ema > 1.0e-3
                && membrane.scout_stats[ch_u].diff_sq_ema
                    > 1.45 * membrane.scout_stats[ch_u].var_ema;
            let mut min_h_s = f32::MAX;
            let mut max_h_s = f32::MIN;
            for &hv in &membrane.sensor_pores[p].history {
                min_h_s = min_h_s.min(hv);
                max_h_s = max_h_s.max(hv);
            }
            let max_abs_h = min_h_s.abs().max(max_h_s.abs());
            let is_dead_flat = (!has_any_env || is_plastic)
                && membrane.sensor_pores[p].age_ticks >= 16
                && (max_h_s - min_h_s) < 1.0e-5
                && (max_abs_h < 1.0e-5 || max_h_s < -0.05);
            is_white_arr[p] = is_white;
            is_dead_flat_arr[p] = is_dead_flat;

            if self.striatal.is_quad_carrier_slot(p) {
                quad_cand[p] = true;
                n_quad_cand += 1;
            } else if !is_white
                && !is_dead_flat
                && self.striatal.pore_age[p] >= 24
                && self.striatal.pore_orb_err[p] >= 0.34
            {
                n_non_carr += 1;
            }
        }

        let carrier_bank = n_quad_cand >= 8 && n_non_carr >= 10;
        if carrier_bank {
            for p in 0..STRIATAL_PORES {
                apical_carrier[p] = quad_cand[p]
                    || (self.striatal.consolidated_role_for_channel(live_pore_ch[p]) & 1) != 0;
            }
            if is_plastic && n_quad_cand > 16 && (self.striatal.tick & 3) == 0 {
                let mut victim_p: Option<usize> = None;
                for p in (0..STRIATAL_PORES).rev() {
                    if !quad_cand[p] {
                        continue;
                    }
                    let role = self.striatal.consolidated_role_for_channel(live_pore_ch[p]);
                    if role == 0 {
                        victim_p = Some(p);
                        break;
                    }
                }
                if let Some(vp) = victim_p {
                    membrane.sensor_pores[vp].trophic_ema = 0.0;
                    membrane.sensor_pores[vp].conductance = 0.01;
                    apical_carrier[vp] = false;
                }
            }
        }

        let mut n_prony_env = 0usize;
        let mut basal_coherent = [false; STRIATAL_PORES];
        for p in 0..STRIATAL_PORES {
            if live_pore_ch[p] == u32::MAX {
                continue;
            }
            if apical_carrier[p] && prony_valid_arr[p] {
                n_prony_env += 1;
            }
            let ch = live_pore_ch[p];
            let ch_u = ch as usize;
            let raw_x = raw_x_arr[p];
            let matched = self.striatal.pore_ch[p] == ch;
            let mean_x = if matched {
                self.striatal.pore_mean[p]
            } else {
                0.0
            };
            let xc = raw_x - mean_x;
            let g_off = p * tap_dim;
            let mut reaff = 0.0_f32;
            if matched && (is_plastic || !has_any_env) && !apical_carrier[p] {
                for tap in 0..STRIATAL_TAPS {
                    let r_off = tap * na;
                    for m in 0..act_s {
                        reaff += self.striatal.canc_g[g_off + r_off + m]
                            * self.striatal.da_ring[r_off + m];
                    }
                }
            }
            let x_tilde = xc - reaff;
            let vel_c = if ch_u < n_scout {
                membrane.scout_stats[ch_u].cov_dx_ema
                    / membrane.scout_stats[ch_u].diff_sq_ema.max(1.0e-8)
            } else {
                0.0
            };
            let cons_role = if has_any_env {
                self.striatal.consolidated_role_for_channel(ch)
            } else {
                0
            };
            if is_plastic {
                let age = self.striatal.pore_age[p];
                let beta_m = if age < 48 { 0.06 } else { 0.015 };
                self.striatal.pore_mean[p] =
                    sanitize_clamp(self.striatal.pore_mean[p] + beta_m * xc, -1.0, 1.0);
                self.striatal.pore_var_raw[p] =
                    sanitize((1.0 - beta_m) * self.striatal.pore_var_raw[p] + beta_m * xc * xc);
                self.striatal.pore_var_exaff[p] = sanitize(
                    (1.0 - beta_m) * self.striatal.pore_var_exaff[p] + beta_m * x_tilde * x_tilde,
                );
                if !apical_carrier[p] {
                    for tap in 0..STRIATAL_TAPS {
                        let r_off = tap * na;
                        for m in 0..act_s {
                            let idx = g_off + r_off + m;
                            self.striatal.canc_g[idx] = sanitize_clamp(
                                canc_decay * self.striatal.canc_g[idx]
                                    + eta_g * x_tilde * self.striatal.da_ring[r_off + m],
                                -2.0,
                                2.0,
                            );
                        }
                    }
                }
                self.striatal.pore_age[p] = age.saturating_add(1);

                let anchored_spine = cons_role != 0;
                if anchored_spine {
                    if membrane.sensor_pores[p].trophic_ema < 0.12 {
                        membrane.sensor_pores[p].trophic_ema = 0.12;
                    }
                    if membrane.sensor_pores[p].conductance < 0.18 {
                        membrane.sensor_pores[p].conductance = 0.18;
                    }
                }

                let is_newtonian_field = carrier_bank
                    && ch_u < n_scout
                    && membrane.scout_stats[ch_u].is_smooth_spatial_field();
                let is_brownian_ou = carrier_bank
                    && age >= 48
                    && vel_c < 0.08
                    && self.striatal.pore_var_raw[p] > 2.0e-3;
                if !apical_carrier[p]
                    && !anchored_spine
                    && !is_newtonian_field
                    && !self.is_tonotopic_receptor(p)
                    && (is_brownian_ou
                        || (age >= prune_warmup
                            && self.striatal.pore_var_raw[p] > 8.0e-4
                            && self.striatal.pore_var_exaff[p]
                                < 0.25 * self.striatal.pore_var_raw[p]))
                {
                    membrane.sensor_pores[p].trophic_ema = 0.0;
                    membrane.sensor_pores[p].conductance = 0.01;
                }
            }
            let age_now = self.striatal.pore_age[p];
            let is_self_copy_like = age_now >= 36
                && self.striatal.pore_var_raw[p] > 8.0e-4
                && self.striatal.pore_var_exaff[p] < 0.25 * self.striatal.pore_var_raw[p];
            let is_ou_like = carrier_bank
                && (cons_role & 2) == 0
                && age_now >= 36
                && vel_c < 0.08
                && self.striatal.pore_var_raw[p] > 2.0e-3;
            feat_active[p] = if carrier_bank && (cons_role & 2) != 0 {
                !apical_carrier[p]
            } else {
                !is_white_arr[p]
                    && !is_dead_flat_arr[p]
                    && !apical_carrier[p]
                    && (!carrier_bank || (!is_self_copy_like && !is_ou_like))
            };
            basal_coherent[p] = if carrier_bank {
                !apical_carrier[p]
                    && ((cons_role & 2) != 0
                        || (!is_white_arr[p]
                            && feat_active[p]
                            && age_now >= 48
                            && vel_c > 0.14
                            && !is_self_copy_like))
            } else {
                feat_active[p]
            };
            let neg_floor = if (!has_any_env || is_plastic) && mean_x < -0.05 {
                mean_x
            } else {
                0.0
            };
            pore_sig[p] = sanitize_clamp(raw_x - neg_floor - reaff, -1.0, 1.0);
            let is_sparse = ch_u < n_scout && membrane.scout_stats[ch_u].is_sparse_impulse();
            if !is_sparse {
                if !is_white_arr[p] {
                    max_smooth_raw = max_smooth_raw.max(raw_x.abs());
                }
                let wave_ok = if is_plastic {
                    self.striatal.pore_var_exaff[p] > 1.5e-4
                        && self.striatal.pore_var_exaff[p] >= 0.28 * self.striatal.pore_var_raw[p]
                } else {
                    raw_x.abs() > 1.0e-5
                };
                if wave_ok {
                    c1_wave_pores += 1;
                }
            }
        }

        let active_continuous_field = if is_plastic {
            self.striatal.tick < 48 || (c1_wave_pores >= 2 && max_smooth_raw > 1.0e-4)
        } else {
            c1_wave_pores >= 1
        };
        if !active_continuous_field {
            if is_plastic {
                self.striatal.has_prev = false;
                self.striatal.tick = self.striatal.tick.saturating_add(1);
            }
            return Ok((None, 0));
        }

        let mut ro_paged_slab: Option<DecodedColumnSlab> = None;
        if !is_plastic && has_any_env {
            let num_c_ro = (self.striatal.num_ctx as usize).min(OFC_CONTEXTS);
            let mut n_ro_prony = 0usize;
            for p in 0..STRIATAL_PORES {
                if prony_valid_arr[p]
                    && (self.striatal.consolidated_role_for_channel(live_pore_ch[p]) & 1) != 0
                {
                    n_ro_prony += 1;
                }
            }
            let (ro_mad_max, ro_cos_min) = if n_ro_prony >= 5 {
                (0.115, 0.72)
            } else {
                (0.28, 0.56)
            };
            let mut best_hot: Option<(usize, f32)> = None;
            for c in 0..num_c_ro {
                if self.striatal.ctx_has_cons[c] == 0 {
                    continue;
                }
                let (cnt, mad, cos) = StriatalOfc::dendritic_match_stats(
                    &self.striatal.hot_dend_ch[c],
                    &self.striatal.hot_dend_env[c],
                    &self.striatal.hot_dend_key[c],
                    &live_pore_ch,
                    &tonic_valid,
                    &tonic_now,
                );
                if cnt >= 5 && mad < ro_mad_max && cos > ro_cos_min {
                    let sc = cos - 2.0 * mad;
                    if best_hot.is_none_or(|(_, b)| sc > b) {
                        best_hot = Some((c, sc));
                    }
                }
            }
            let mut best_cold: Option<(usize, f32)> = None;
            for (idx, entry) in self.striatal.paged_cols.iter().enumerate() {
                let (cnt, mad, cos) = StriatalOfc::dendritic_match_stats(
                    &entry.dend_ch,
                    &entry.dend_env,
                    &entry.dend_key,
                    &live_pore_ch,
                    &tonic_valid,
                    &tonic_now,
                );
                if cnt >= 5 && mad < ro_mad_max && cos > ro_cos_min {
                    let sc = cos - 2.0 * mad;
                    if best_cold.is_none_or(|(_, b)| sc > b) {
                        best_cold = Some((idx, sc));
                    }
                }
            }
            if let Some((p_idx, p_sc)) = best_cold
                && best_hot.is_none_or(|(_, h_sc)| p_sc > h_sc)
            {
                let raw_dec = decode_column_slab(&self.striatal.paged_cols[p_idx].slab_bytes)?;
                ro_paged_slab =
                    Some(self.striatal.remap_slab_for_channels(&raw_dec, &live_pore_ch));
            } else if let Some((h_c, _)) = best_hot {
                let mut v_arr = [0.0_f32; STRIATAL_FEATS];
                v_arr.copy_from_slice(
                    &self.striatal.v_cons[h_c * STRIATAL_FEATS..(h_c + 1) * STRIATAL_FEATS],
                );
                let mut q_arr = [0.0_f32; STRIATAL_CUES];
                q_arr.copy_from_slice(
                    &self.striatal.q_cons[h_c * STRIATAL_CUES..(h_c + 1) * STRIATAL_CUES],
                );
                let raw_hot = DecodedColumnSlab {
                    col_id: self.striatal.hot_col_id[h_c],
                    na,
                    ticks: self.striatal.ctx_ticks[h_c],
                    r_peak: self.striatal.ctx_r_peak[h_c],
                    r_bar: self.striatal.r_bar_ctx[h_c],
                    dend_ch: self.striatal.hot_dend_ch[h_c],
                    dend_env: self.striatal.hot_dend_env[h_c],
                    dend_key: self.striatal.hot_dend_key[h_c],
                    w_cons: self.striatal.w_cons[h_c * w_len..(h_c + 1) * w_len].to_vec(),
                    v_cons: v_arr,
                    q_cons: q_arr,
                    fwd_j_cons: self.striatal.fwd_j_cons[h_c * fwd_len..(h_c + 1) * fwd_len]
                        .to_vec(),
                };
                ro_paged_slab =
                    Some(self.striatal.remap_slab_for_channels(&raw_hot, &live_pore_ch));
            }
            if let Some(ref dec) = ro_paged_slab {
                let mut mask_cnt = 0usize;
                for (p, &ev) in dec.dend_env.iter().enumerate() {
                    if live_pore_ch[p] != u32::MAX && (ev & 2) != 0 {
                        mask_cnt += 1;
                    }
                }
                if mask_cnt >= 4 {
                    for p in 0..STRIATAL_PORES {
                        feat_active[p] = live_pore_ch[p] != u32::MAX && (dec.dend_env[p] & 2) != 0;
                    }
                }
            }
        }

        let mut mean_sig = 0.0_f32;
        let mut alive_cnt = 0usize;
        for p in 0..STRIATAL_PORES {
            if feat_active[p] {
                mean_sig += pore_sig[p];
                alive_cnt += 1;
            }
        }
        if alive_cnt > 0 {
            mean_sig /= alive_cnt as f32;
        }
        let mut energy = 0.0_f32;
        for p in 0..STRIATAL_PORES {
            if feat_active[p] {
                let z = pore_sig[p] - mean_sig;
                let on_lo = z.clamp(0.0, 0.22);
                let on_hi = (z - 0.22).max(0.0);
                let off_lo = (-z).clamp(0.0, 0.22);
                let off_hi = (-z - 0.22).max(0.0);
                self.striatal.feat[p] = on_lo;
                self.striatal.feat[STRIATAL_PORES + p] = on_hi;
                self.striatal.feat[2 * STRIATAL_PORES + p] = off_lo;
                self.striatal.feat[3 * STRIATAL_PORES + p] = off_hi;
                energy += on_lo * on_lo + on_hi * on_hi + off_lo * off_lo + off_hi * off_hi;
            } else {
                for st in 0..STRIATAL_STRATA {
                    self.striatal.feat[st * STRIATAL_PORES + p] = 0.0;
                }
            }
        }
        let inv = 1.0 / (0.05 + energy).sqrt();
        for f in &mut self.striatal.feat[..STRIATAL_CUES] {
            *f = sanitize(*f * inv);
        }
        self.striatal.feat[STRIATAL_CUES] = 1.0;

        if self.striatal.has_prev {
            let act_c = self.striatal.active_ctx as usize;
            let num_c = self.striatal.num_ctx as usize;
            let mut da = vec![0.0_f32; na];
            let mut da_norm_sq = 0.0_f32;
            for m in 0..act_s {
                let d = self.striatal.prev_act[m] - self.striatal.act_mean[m];
                da[m] = d;
                da_norm_sq += d * d;
            }
            let beta_v = 0.004_f32;
            let nlms_m_scale = ((act_s.max(6) as f32) * (1.0 / 6.0)).clamp(1.0, 6.0);
            let eta_j = (0.015 * nlms_m_scale) / (da_norm_sq + 0.25);
            let mut step_num = [0.0_f32; OFC_CONTEXTS];
            let mut step_den = [0.0_f32; OFC_CONTEXTS];
            for j in 0..STRIATAL_CUES {
                let p_slot = j & 63;
                let df = self.striatal.feat[j] - self.striatal.prev_feat[j];
                let pdf = self.striatal.prev_df[j];
                let pred_null = self.striatal.fwd_b[j] + self.striatal.fwd_alpha[j] * pdf;
                let e_null = df - pred_null;
                let j_off = j * na;
                if is_plastic && feat_active[p_slot] && self.striatal.pore_age[p_slot] >= 8 {
                    for c in 0..num_c {
                        let c_base = c * fwd_len + j_off;
                        let mut yh = 0.0_f32;
                        if self.striatal.ctx_has_cons[c] != 0 {
                            for m in 0..act_s {
                                yh += self.striatal.fwd_j_cons[c_base + m] * da[m];
                            }
                        } else {
                            for m in 0..act_s {
                                yh += self.striatal.fwd_j_ctx[c_base + m] * da[m];
                            }
                        }
                        step_num[c] += e_null * yh;
                        step_den[c] += yh * yh;
                    }
                }
                let mut act_pred = 0.0_f32;
                for m in 0..act_s {
                    act_pred += self.striatal.fwd_j[j_off + m] * da[m];
                }
                let e_act = e_null - act_pred;
                if is_plastic && feat_active[p_slot] && self.striatal.pore_age[p_slot] >= 3 {
                    self.striatal.fwd_var_null[j] = sanitize(
                        (1.0 - beta_v) * self.striatal.fwd_var_null[j] + beta_v * e_null * e_null,
                    );
                    self.striatal.fwd_var_act[j] = sanitize(
                        (1.0 - beta_v) * self.striatal.fwd_var_act[j] + beta_v * e_act * e_act,
                    );
                    self.striatal.fwd_b[j] =
                        sanitize_clamp(self.striatal.fwd_b[j] + 0.02 * e_null, -1.0, 1.0);
                    let denom_p = pdf * pdf + 0.01;
                    self.striatal.fwd_alpha[j] = sanitize_clamp(
                        self.striatal.fwd_alpha[j] + 0.04 * e_null * pdf / denom_p,
                        -0.98,
                        0.98,
                    );
                    for m in 0..act_s {
                        let idx = j_off + m;
                        self.striatal.fwd_j[idx] = sanitize_clamp(
                            j_decay * self.striatal.fwd_j[idx] + eta_j * e_act * da[m],
                            -2.0,
                            2.0,
                        );
                    }
                    let live_base = act_c * fwd_len + j_off;
                    let mut ctx_pred = 0.0_f32;
                    for m in 0..act_s {
                        ctx_pred += self.striatal.fwd_j_ctx[live_base + m] * da[m];
                    }
                    let e_ctx = e_null - ctx_pred;
                    for m in 0..act_s {
                        let idx = live_base + m;
                        self.striatal.fwd_j_ctx[idx] = sanitize_clamp(
                            j_decay * self.striatal.fwd_j_ctx[idx] + eta_j * e_ctx * da[m],
                            -2.0,
                            2.0,
                        );
                    }
                }
                self.striatal.prev_df[j] = df;
            }
            if is_plastic {
                if da_norm_sq > 0.02 {
                    for c in 0..num_c {
                        self.striatal.fwd_num[c] =
                            sanitize(0.97 * self.striatal.fwd_num[c] + 0.03 * step_num[c]);
                        self.striatal.fwd_den[c] =
                            sanitize(0.97 * self.striatal.fwd_den[c] + 0.03 * step_den[c]);
                    }
                }
                let mut refrac = self.striatal.impact_refrac >> 5;
                let mut streak = self.striatal.impact_refrac & 31;
                if tissue_stress >= 0.055 {
                    refrac = 6;
                    streak = streak.saturating_sub(2);
                } else if refrac > 0 {
                    refrac -= 1;
                } else if da_norm_sq > 0.03 && step_den[0] > 8.0e-5 {
                    if step_num[0] < -0.25 * step_den[0] {
                        streak = (streak + 1).min(31);
                    } else if step_num[0] > 0.10 * step_den[0] {
                        streak = streak.saturating_sub(2);
                    }
                }
                self.striatal.impact_refrac = (refrac << 5) | (streak & 31);
                for m in 0..act_s {
                    self.striatal.act_mean[m] = sanitize_clamp(
                        0.99 * self.striatal.act_mean[m] + 0.01 * self.striatal.prev_act[m],
                        -1.0,
                        1.0,
                    );
                }
            }
        }
        self.striatal
            .prev_feat
            .copy_from_slice(&self.striatal.feat[..STRIATAL_CUES]);
        if is_plastic && self.striatal.tick >= 120 {
            let inst_ag = self
                .striatal
                .forward_agency(num_actuators)
                .min(0.012)
                .max(self.agency_signed_ema.clamp(0.0, 1.0));
            self.mean_interventional_agency = sanitize_clamp(
                0.9985 * self.mean_interventional_agency + 0.0015 * inst_ag,
                0.0,
                1.0,
            );
        }

        let mut switched_context = false;
        let mut act_mismatch_inst = false;
        let mut splits = 0u64;
        if is_plastic && self.striatal.has_prev {
            let act_c = self.striatal.active_ctx as usize;
            let num_c = self.striatal.num_ctx as usize;
            self.striatal.ctx_ticks[act_c] = self.striatal.ctx_ticks[act_c].saturating_add(1);
            self.striatal.hot_last_use[act_c] = self.striatal.tick;
            if self.striatal.ctx_cooldown > 0 {
                self.striatal.ctx_cooldown -= 1;
            }
            self.striatal.r_fast_ema =
                sanitize(0.97 * self.striatal.r_fast_ema + 0.03 * reward).min(0.022);
            self.striatal.r_med200 =
                sanitize(0.992 * self.striatal.r_med200 + 0.008 * reward).min(0.022);

            let (act_env_cnt, act_mad, act_cos) = if self.striatal.ctx_has_cons[act_c] != 0 {
                let cur_ch = self.striatal.pore_ch;
                StriatalOfc::dendritic_match_stats(
                    &self.striatal.hot_dend_ch[act_c],
                    &self.striatal.hot_dend_env[act_c],
                    &self.striatal.hot_dend_key[act_c],
                    &cur_ch,
                    &tonic_valid,
                    &tonic_now,
                )
            } else {
                (0, 0.0, 1.0)
            };
            let env_mismatch_raw = act_env_cnt >= 5 && (act_mad > 0.115 || act_cos < 0.76);
            act_mismatch_inst = env_mismatch_raw && n_prony_env >= 5;
            if act_mismatch_inst {
                self.striatal.dend_mismatch_streak =
                    self.striatal.dend_mismatch_streak.saturating_add(1);
            } else if !env_mismatch_raw {
                self.striatal.dend_mismatch_streak = 0;
            }
            let dend_mismatch_now = self.striatal.dend_mismatch_streak >= 6;

            let has_fwd_act = self.striatal.fwd_den[act_c] > 2.0e-5;
            let beta_act = if has_fwd_act {
                self.striatal.fwd_num[act_c] / (self.striatal.fwd_den[act_c] + 1.0e-5)
            } else {
                0.0
            };

            let prev_cf = self.striatal.prev_critic_feat;
            let rh_act_live = self.striatal.outcome_pred_live(act_c, &prev_cf);
            let mut app_rec_sig = 0.0_f32;
            if reward > 0.005 {
                for c in 0..num_c {
                    if c != act_c && self.striatal.ctx_has_cons[c] != 0 {
                        let rh_c = self.striatal.outcome_pred_cons(c, &prev_cf);
                        if rh_c > 0.006 && rh_act_live < 0.003 {
                            app_rec_sig = app_rec_sig.max(reward * (rh_c - rh_act_live));
                        }
                    }
                }
            }
            if app_rec_sig > 0.0 {
                self.striatal.valence_flip_ema =
                    sanitize(0.85 * self.striatal.valence_flip_ema + 0.15 * app_rec_sig);
            } else {
                self.striatal.valence_flip_ema = sanitize(0.96 * self.striatal.valence_flip_ema);
            }

            if reward.abs() > 0.005 {
                self.striatal.ctx_out_events = self.striatal.ctx_out_events.saturating_add(1);
                for c in 0..num_c {
                    let rh = if c != act_c && self.striatal.ctx_has_cons[c] != 0 {
                        self.striatal.outcome_pred_cons(c, &prev_cf)
                    } else {
                        self.striatal.outcome_pred_live(c, &prev_cf)
                    };
                    let err_sq = (reward - rh) * (reward - rh);
                    self.striatal.ctx_out_err[c] =
                        sanitize(0.88 * self.striatal.ctx_out_err[c] + 0.12 * err_sq);
                }
            }

            if !env_mismatch_raw && (!carrier_bank || n_prony_env >= 6) {
                if carrier_bank {
                    if self.striatal.ctx_has_cons[act_c] == 0 {
                        if self.striatal.ctx_ticks[act_c] >= 480
                            && ((has_fwd_act && beta_act > 0.10 && self.striatal.r_med200 > -0.025)
                                || self.striatal.ctx_ticks[act_c] >= 680)
                        {
                            self.striatal.ctx_r_peak[act_c] = self.striatal.r_med200;
                            self.striatal.consolidate_active_context_with_dend(
                                &tonic_valid,
                                &tonic_now,
                                &apical_carrier,
                                &basal_coherent,
                            );
                        }
                    } else if self.striatal.ctx_ticks[act_c] < 1_150
                        && self.striatal.ctx_ticks[act_c] >= 520
                        && (self.striatal.r_med200 >= self.striatal.ctx_r_peak[act_c] - 0.004
                            || (self.striatal.r_med200 > 0.0015
                                && (self.striatal.ctx_ticks[act_c] & 31) == 0)
                            || (self.striatal.ctx_ticks[act_c] == 1_148
                                && self.striatal.r_med200
                                    >= self.striatal.ctx_r_peak[act_c] - 0.010))
                        && (!has_fwd_act || beta_act > 0.08)
                    {
                        self.striatal.ctx_r_peak[act_c] =
                            self.striatal.ctx_r_peak[act_c].max(self.striatal.r_med200);
                        self.striatal.consolidate_active_context_with_dend(
                            &tonic_valid,
                            &tonic_now,
                            &apical_carrier,
                            &basal_coherent,
                        );
                    } else if self.striatal.ctx_ticks[act_c] >= 1_150
                        && self.striatal.r_med200 > self.striatal.ctx_r_peak[act_c] + 0.002
                        && (!has_fwd_act || beta_act > 0.15)
                    {
                        self.striatal.ctx_r_peak[act_c] = self.striatal.r_med200;
                        self.striatal.consolidate_active_context_with_dend(
                            &tonic_valid,
                            &tonic_now,
                            &apical_carrier,
                            &basal_coherent,
                        );
                    } else if self.striatal.ctx_ticks[act_c] < 1_150
                        && has_fwd_act
                        && beta_act > 0.50
                        && (self.striatal.ctx_ticks[act_c] & 63) == 0
                    {
                        let f_off = act_c * fwd_len;
                        self.striatal.fwd_j_cons[f_off..f_off + fwd_len]
                            .copy_from_slice(&self.striatal.fwd_j_ctx[f_off..f_off + fwd_len]);
                    }
                } else if self.striatal.ctx_ticks[act_c] >= 520
                    && self.striatal.r_med200 > 0.002
                    && self.striatal.r_med200 >= self.striatal.ctx_r_peak[act_c] - 0.003
                    && (!has_fwd_act || beta_act > 0.15)
                {
                    self.striatal.ctx_r_peak[act_c] =
                        self.striatal.ctx_r_peak[act_c].max(self.striatal.r_med200);
                    self.striatal.consolidate_active_context_with_dend(
                        &tonic_valid,
                        &tonic_now,
                        &apical_carrier,
                        &feat_active,
                    );
                } else if act_c > 0
                    && self.striatal.ctx_has_cons[act_c] == 0
                    && self.striatal.ctx_ticks[act_c] >= 400
                    && has_fwd_act
                    && beta_act > 0.20
                    && (self.striatal.r_med200 > -0.008
                        || (self.striatal.ctx_ticks[act_c] >= 650
                            && self.striatal.r_med200 > -0.010))
                {
                    self.striatal.ctx_r_peak[act_c] = self.striatal.r_med200;
                    self.striatal.consolidate_active_context_with_dend(
                        &tonic_valid,
                        &tonic_now,
                        &apical_carrier,
                        &feat_active,
                    );
                } else if self.striatal.ctx_has_cons[act_c] != 0
                    && has_fwd_act
                    && beta_act > 0.55
                    && (self.striatal.ctx_ticks[act_c] & 63) == 0
                {
                    let f_off = act_c * fwd_len;
                    self.striatal.fwd_j_cons[f_off..f_off + fwd_len]
                        .copy_from_slice(&self.striatal.fwd_j_ctx[f_off..f_off + fwd_len]);
                }
            }

            let can_switch = self.striatal.ctx_cooldown == 0
                && (dend_mismatch_now || self.striatal.ctx_ticks[act_c] >= 220);
            if can_switch {
                if dend_mismatch_now {
                    for p in 0..STRIATAL_PORES {
                        if quad_cand[p] && tonic_valid[p] {
                            self.striatal.pore_carr_mean[p] = tonic_now[p];
                            self.striatal.pore_prev_x[p] = raw_x_arr[p];
                        }
                    }
                    let cur_ch = self.striatal.pore_ch;
                    let mut best_hot: Option<(usize, f32)> = None;
                    for c in 0..num_c {
                        if c != act_c && self.striatal.ctx_has_cons[c] != 0 {
                            let (cnt, mad, cos) = StriatalOfc::dendritic_match_stats(
                                &self.striatal.hot_dend_ch[c],
                                &self.striatal.hot_dend_env[c],
                                &self.striatal.hot_dend_key[c],
                                &cur_ch,
                                &tonic_valid,
                                &tonic_now,
                            );
                            if cnt >= 5 && mad < 0.065 && cos > 0.90 {
                                let sc = cos - 2.0 * mad;
                                if best_hot.is_none_or(|(_, b)| sc > b) {
                                    best_hot = Some((c, sc));
                                }
                            }
                        }
                    }
                    let mut best_cold: Option<(usize, f32)> = None;
                    for (idx, entry) in self.striatal.paged_cols.iter().enumerate() {
                        let (cnt, mad, cos) = StriatalOfc::dendritic_match_stats(
                            &entry.dend_ch,
                            &entry.dend_env,
                            &entry.dend_key,
                            &cur_ch,
                            &tonic_valid,
                            &tonic_now,
                        );
                        if cnt >= 5 && mad < 0.065 && cos > 0.90 {
                            let sc = cos - 2.0 * mad;
                            if best_cold.is_none_or(|(_, b)| sc > b) {
                                best_cold = Some((idx, sc));
                            }
                        }
                    }

                    if self.striatal.ctx_has_cons[act_c] != 0 {
                        self.striatal.restore_consolidated_context(act_c);
                    }

                    if let Some((p_idx, p_sc)) = best_cold
                        && best_hot.is_none_or(|(_, h_sc)| p_sc > h_sc + 0.01)
                    {
                        let target_slot = if num_c < OFC_CONTEXTS {
                            self.striatal.num_ctx += 1;
                            num_c
                        } else {
                            (0..num_c)
                                .filter(|&c| c != act_c)
                                .min_by_key(|&c| self.striatal.hot_last_use[c])
                                .unwrap_or(0)
                        };
                        self.striatal.page_in_cold_column(p_idx, target_slot)?;
                        let fj_dst = target_slot * fwd_len;
                        self.striatal
                            .fwd_j
                            .copy_from_slice(&self.striatal.fwd_j_cons[fj_dst..fj_dst + fwd_len]);
                        self.striatal.active_ctx = target_slot as u8;
                        self.striatal.impact_refrac = 0;
                        self.striatal.ctx_cooldown = 24;
                        self.striatal.dend_mismatch_streak = 0;
                        self.striatal.ctx_out_events = 0;
                        self.striatal.ctx_out_err = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_num = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_den = [0.0; OFC_CONTEXTS];
                        self.striatal.r_fast_ema = 0.0;
                        self.striatal.r_med200 = 0.0;
                        self.striatal.valence_flip_ema = 0.0;
                        self.striatal.elig_actor.fill(0.0);
                        self.striatal.elig_critic = [0.0; STRIATAL_FEATS];
                        switched_context = true;
                        splits = 1;
                    } else if let Some((target_ctx, _)) = best_hot {
                        self.striatal.restore_consolidated_context(target_ctx);
                        let fj_dst = target_ctx * fwd_len;
                        self.striatal
                            .fwd_j
                            .copy_from_slice(&self.striatal.fwd_j_cons[fj_dst..fj_dst + fwd_len]);
                        self.striatal.hot_last_use[target_ctx] = self.striatal.tick;
                        self.striatal.active_ctx = target_ctx as u8;
                        self.striatal.impact_refrac = 0;
                        self.striatal.ctx_cooldown = 24;
                        self.striatal.dend_mismatch_streak = 0;
                        self.striatal.ctx_out_events = 0;
                        self.striatal.ctx_out_err = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_num = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_den = [0.0; OFC_CONTEXTS];
                        self.striatal.r_fast_ema = 0.0;
                        self.striatal.r_med200 = 0.0;
                        self.striatal.valence_flip_ema = 0.0;
                        self.striatal.elig_actor.fill(0.0);
                        self.striatal.elig_critic = [0.0; STRIATAL_FEATS];
                        switched_context = true;
                        splits = 1;
                    } else {
                        let target_ctx = if num_c < OFC_CONTEXTS {
                            self.striatal.num_ctx += 1;
                            num_c
                        } else {
                            (0..num_c)
                                .filter(|&c| c != act_c)
                                .min_by_key(|&c| self.striatal.hot_last_use[c])
                                .unwrap_or(0)
                        };
                        if self.striatal.ctx_has_cons[target_ctx] != 0 {
                            self.striatal.evict_hot_slot_to_paged(target_ctx)?;
                        }
                        self.striatal.hot_col_id[target_ctx] = self.striatal.total_cols;
                        self.striatal.total_cols = self.striatal.total_cols.saturating_add(1);
                        self.striatal.ctx_has_cons[target_ctx] = 0;
                        self.striatal.ctx_ticks[target_ctx] = 0;
                        self.striatal.ctx_r_peak[target_ctx] = 0.0;
                        self.striatal.r_bar_ctx[target_ctx] = 0.0;
                        self.striatal.hot_last_use[target_ctx] = self.striatal.tick;
                        self.striatal.hot_dend_ch[target_ctx] = [u32::MAX; STRIATAL_PORES];
                        self.striatal.hot_dend_env[target_ctx] = [0; STRIATAL_PORES];
                        self.striatal.hot_dend_key[target_ctx] = [0.0; STRIATAL_PORES];

                        let dst_off = target_ctx * w_len;
                        self.striatal.w_cons[dst_off..dst_off + w_len].fill(0.0);
                        for m in 0..na {
                            let row = m * STRIATAL_CUES;
                            for st in 0..STRIATAL_STRATA {
                                let st_base = row + st * STRIATAL_PORES;
                                let mut mean_b = 0.0_f32;
                                for p in 0..STRIATAL_PORES {
                                    mean_b += self.striatal.w_birth[st_base + p];
                                }
                                mean_b *= 1.0 / (STRIATAL_PORES as f32);
                                for p in 0..STRIATAL_PORES {
                                    self.striatal.w_ctx[dst_off + st_base + p] = sanitize_clamp(
                                        self.striatal.w_birth[st_base + p] - mean_b,
                                        -STRIATAL_W_CLAMP,
                                        STRIATAL_W_CLAMP,
                                    );
                                }
                            }
                        }
                        let v_dst = target_ctx * STRIATAL_FEATS;
                        self.striatal.v_ctx[v_dst..v_dst + STRIATAL_FEATS].fill(0.0);
                        self.striatal.v_cons[v_dst..v_dst + STRIATAL_FEATS].fill(0.0);
                        let q_dst = target_ctx * STRIATAL_CUES;
                        self.striatal.q_ctx[q_dst..q_dst + STRIATAL_CUES].fill(0.0);
                        self.striatal.q_cons[q_dst..q_dst + STRIATAL_CUES].fill(0.0);
                        let fj_dst = target_ctx * fwd_len;
                        for k in 0..fwd_len {
                            self.striatal.fwd_j_ctx[fj_dst + k] =
                                sanitize(0.5 * self.striatal.fwd_j[k]);
                        }
                        self.striatal.fwd_j_cons[fj_dst..fj_dst + fwd_len].fill(0.0);

                        self.striatal.active_ctx = target_ctx as u8;
                        self.striatal.impact_refrac = 0;
                        self.striatal.ctx_cooldown = 24;
                        self.striatal.dend_mismatch_streak = 0;
                        self.striatal.ctx_out_events = 0;
                        self.striatal.ctx_out_err = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_num = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_den = [0.0; OFC_CONTEXTS];
                        self.striatal.r_fast_ema = 0.0;
                        self.striatal.r_med200 = 0.0;
                        self.striatal.valence_flip_ema = 0.0;
                        self.striatal.elig_actor.fill(0.0);
                        self.striatal.elig_critic = [0.0; STRIATAL_FEATS];
                        switched_context = true;
                        splits = 1;
                    }
                } else if !has_any_env && self.striatal.ctx_ticks[act_c] >= 220 {
                    let mut recall_target = None;
                    let mut best_score = f32::MIN;
                    for c in 0..num_c {
                        if c != act_c && self.striatal.ctx_has_cons[c] != 0 {
                            let has_fwd_c = self.striatal.fwd_den[c] > 2.0e-5;
                            let beta_c = if has_fwd_c {
                                self.striatal.fwd_num[c] / (self.striatal.fwd_den[c] + 1.0e-5)
                            } else {
                                0.0
                            };
                            let mosaic_match =
                                has_fwd_c && has_fwd_act && beta_c > 0.18 && beta_act < -0.18;
                            let val_match = has_fwd_c
                                && beta_c > 0.12
                                && (self.striatal.valence_flip_ema > 0.00011
                                    || (self.striatal.ctx_out_events >= 5
                                        && self.striatal.ctx_out_err[act_c]
                                            > self.striatal.ctx_out_err[c] + 0.0008
                                        && self.striatal.ctx_out_err[c] < 0.0025));
                            if mosaic_match || val_match {
                                let score = beta_c - 40.0 * self.striatal.ctx_out_err[c];
                                if score > best_score {
                                    best_score = score;
                                    recall_target = Some(c);
                                }
                            }
                        }
                    }

                    let hedonic_collapse = self.striatal.ctx_r_peak[act_c] > 0.002
                        && ((self.striatal.r_med200 < -0.0115
                            && self.striatal.ctx_r_peak[act_c] - self.striatal.r_med200 > 0.015
                            && (act_c == 0 || !has_fwd_act || beta_act < 0.05))
                            || (num_c == 1
                                && act_c == 0
                                && self.striatal.ctx_ticks[0] >= 700
                                && has_fwd_act
                                && beta_act < -0.35
                                && self.striatal.r_fast_ema < -0.018));
                    let kin_streak = self.striatal.impact_refrac & 31;
                    let kin_rev_collapse = num_c == 1
                        && act_c == 0
                        && self.striatal.ctx_ticks[0] >= 700
                        && self.striatal.ctx_r_peak[0] > 0.002
                        && self.striatal.fwd_den[0] > 5.0e-5
                        && beta_act < -0.45
                        && kin_streak >= 16
                        && reward <= 0.0005
                        && self.striatal.r_fast_ema <= 0.008;
                    let spawn_collapse = self.striatal.ctx_has_cons[act_c] != 0
                        && (hedonic_collapse || kin_rev_collapse);

                    let mut spawning_new = false;
                    let next_ctx = if let Some(c) = recall_target {
                        Some(c)
                    } else if spawn_collapse {
                        spawning_new = true;
                        if num_c < OFC_CONTEXTS {
                            self.striatal.num_ctx += 1;
                            Some(num_c)
                        } else {
                            (0..num_c)
                                .filter(|&c| c != act_c)
                                .min_by_key(|&c| self.striatal.ctx_ticks[c])
                        }
                    } else {
                        None
                    };

                    if let Some(target_ctx) = next_ctx
                        && target_ctx != act_c
                    {
                        if spawning_new {
                            if self.striatal.ctx_has_cons[target_ctx] != 0 {
                                self.striatal.evict_hot_slot_to_paged(target_ctx)?;
                            }
                            self.striatal.hot_col_id[target_ctx] = self.striatal.total_cols;
                            self.striatal.total_cols = self.striatal.total_cols.saturating_add(1);
                            self.striatal.ctx_has_cons[target_ctx] = 0;
                            self.striatal.ctx_ticks[target_ctx] = 0;
                            self.striatal.ctx_r_peak[target_ctx] = 0.0;
                            self.striatal.r_bar_ctx[target_ctx] = 0.0;
                            self.striatal.hot_dend_ch[target_ctx] = [u32::MAX; STRIATAL_PORES];
                            self.striatal.hot_dend_env[target_ctx] = [0; STRIATAL_PORES];
                            self.striatal.hot_dend_key[target_ctx] = [0.0; STRIATAL_PORES];
                            self.striatal.w_cons[target_ctx * w_len..(target_ctx + 1) * w_len]
                                .fill(0.0);
                            self.striatal.v_cons
                                [target_ctx * STRIATAL_FEATS..(target_ctx + 1) * STRIATAL_FEATS]
                                .fill(0.0);
                            self.striatal.q_cons
                                [target_ctx * STRIATAL_CUES..(target_ctx + 1) * STRIATAL_CUES]
                                .fill(0.0);
                            self.striatal.fwd_j_cons
                                [target_ctx * fwd_len..(target_ctx + 1) * fwd_len]
                                .fill(0.0);
                        }
                        if self.striatal.ctx_has_cons[target_ctx] == 0
                            && self.striatal.ctx_has_cons[act_c] != 0
                        {
                            let dst_off = target_ctx * w_len;
                            for m in 0..na {
                                let row = m * STRIATAL_CUES;
                                for st in 0..STRIATAL_STRATA {
                                    let st_base = row + st * STRIATAL_PORES;
                                    let mut mean_b = 0.0_f32;
                                    for p in 0..STRIATAL_PORES {
                                        mean_b += self.striatal.w_birth[st_base + p];
                                    }
                                    mean_b *= 1.0 / (STRIATAL_PORES as f32);
                                    for p in 0..STRIATAL_PORES {
                                        self.striatal.w_ctx[dst_off + st_base + p] = sanitize_clamp(
                                            self.striatal.w_birth[st_base + p] - mean_b,
                                            -STRIATAL_W_CLAMP,
                                            STRIATAL_W_CLAMP,
                                        );
                                    }
                                }
                            }
                            let v_dst = target_ctx * STRIATAL_FEATS;
                            self.striatal.v_ctx[v_dst..v_dst + STRIATAL_FEATS].fill(0.0);
                            let q_dst = target_ctx * STRIATAL_CUES;
                            self.striatal.q_ctx[q_dst..q_dst + STRIATAL_CUES].fill(0.0);
                            let fj_src = act_c * fwd_len;
                            let fj_dst = target_ctx * fwd_len;
                            if has_fwd_act && beta_act < -0.20 {
                                for k in 0..fwd_len {
                                    let inv_j = sanitize(-self.striatal.fwd_j_cons[fj_src + k]);
                                    self.striatal.fwd_j_ctx[fj_dst + k] = inv_j;
                                    self.striatal.fwd_j[k] = inv_j;
                                }
                            } else if has_fwd_act && beta_act > 0.20 {
                                self.striatal.fwd_j_ctx[fj_dst..fj_dst + fwd_len].copy_from_slice(
                                    &self.striatal.fwd_j_cons[fj_src..fj_src + fwd_len],
                                );
                            } else {
                                self.striatal.fwd_j_ctx[fj_dst..fj_dst + fwd_len].fill(0.0);
                            }
                        }
                        if self.striatal.ctx_has_cons[act_c] != 0 {
                            self.striatal.restore_consolidated_context(act_c);
                        }
                        if self.striatal.ctx_has_cons[target_ctx] != 0 {
                            self.striatal.restore_consolidated_context(target_ctx);
                            let fj_dst = target_ctx * fwd_len;
                            self.striatal.fwd_j.copy_from_slice(
                                &self.striatal.fwd_j_cons[fj_dst..fj_dst + fwd_len],
                            );
                        }
                        self.striatal.hot_last_use[target_ctx] = self.striatal.tick;
                        self.striatal.active_ctx = target_ctx as u8;
                        self.striatal.impact_refrac = 0;
                        self.striatal.ctx_cooldown = 180;
                        self.striatal.ctx_out_events = 0;
                        self.striatal.ctx_out_err = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_num = [0.0; OFC_CONTEXTS];
                        self.striatal.fwd_den = [0.0; OFC_CONTEXTS];
                        self.striatal.r_fast_ema = 0.0;
                        self.striatal.r_med200 = 0.0;
                        self.striatal.valence_flip_ema = 0.0;
                        self.striatal.elig_actor.fill(0.0);
                        self.striatal.elig_critic = [0.0; STRIATAL_FEATS];
                        switched_context = true;
                        splits = 1;
                    }
                }
            }
        }

        let act_c = self.striatal.active_ctx as usize;
        let has_cons_model = self.striatal.ctx_has_cons.iter().any(|&x| x != 0)
            || !self.striatal.paged_cols.is_empty();
        let feat_now = self.striatal.feat;
        let v_now = self.striatal.critic_value(act_c, &feat_now);
        let num_c_ro = self.striatal.num_ctx as usize;
        let read_c = if !is_plastic
            && ro_paged_slab.is_none()
            && num_c_ro > 1
            && (self.striatal.ctx_has_cons[act_c] == 0
                || self.striatal.ctx_r_peak[act_c] < 0.0)
        {
            (0..num_c_ro)
                .filter(|&c| self.striatal.ctx_has_cons[c] != 0)
                .max_by(|&a, &b| {
                    self.striatal.ctx_r_peak[a].total_cmp(&self.striatal.ctx_r_peak[b])
                })
                .unwrap_or(act_c)
        } else {
            act_c
        };
        let recalled_mature = is_plastic
            && carrier_bank
            && !act_mismatch_inst
            && self.striatal.ctx_has_cons[act_c] != 0
            && self.striatal.ctx_ticks[act_c] >= 1_150;
        let use_cons_w = ro_paged_slab.is_some()
            || recalled_mature
            || (!is_plastic
                && self.striatal.ctx_has_cons[read_c] != 0
                && (read_c != act_c
                    || self.striatal.r_med200 + 0.003 < self.striatal.ctx_r_peak[read_c]));

        let mut has_kin_loop = vec![false; act_s];
        let mut loop_mask = vec![[false; STRIATAL_PORES]; act_s];
        {
            let fj_ref_off = read_c * fwd_len;
            let j_ref: &[f32] = if let Some(ref dec) = ro_paged_slab {
                &dec.fwd_j_cons
            } else if use_cons_w {
                &self.striatal.fwd_j_cons[fj_ref_off..fj_ref_off + fwd_len]
            } else {
                &self.striatal.fwd_j_ctx[fj_ref_off..fj_ref_off + fwd_len]
            };
            let mut jm_p = vec![[0.0_f32; STRIATAL_PORES]; act_s];
            let mut jm_max = vec![0.0_f32; act_s];
            let mut jm_sum = vec![0.0_f32; act_s];
            let mut jm_sq_norm = vec![0.0_f32; act_s];
            let mut kin_cand = vec![false; act_s];
            for m in 0..act_s {
                for p in 0..STRIATAL_PORES {
                    if p >= pore_cap || !membrane.sensor_pores[p].alive {
                        continue;
                    }
                    if ro_paged_slab.is_none()
                        && self.striatal.pore_ch[p] != membrane.sensor_pores[p].channel
                    {
                        continue;
                    }
                    let mut s_j = 0.0_f32;
                    for st in 0..STRIATAL_STRATA {
                        let idx = (st * STRIATAL_PORES + p) * na + m;
                        let j2 = j_ref[idx];
                        let s_j_term = if ro_paged_slab.is_some() || use_cons_w {
                            j2 * j2
                        } else {
                            let j1 = self.striatal.fwd_j[idx];
                            (j1 * j1).max(j2 * j2)
                        };
                        s_j += s_j_term;
                    }
                    jm_p[m][p] = s_j;
                    jm_max[m] = jm_max[m].max(s_j);
                    jm_sum[m] += s_j;
                    jm_sq_norm[m] += s_j * s_j;
                }
                kin_cand[m] =
                    self.striatal.tick >= 160 && jm_max[m] >= 0.0020 && jm_sum[m] >= 0.007;
            }
            for m in 0..act_s {
                if !kin_cand[m] {
                    continue;
                }
                let mut grp_p = jm_p[m];
                let sq_m = jm_sq_norm[m];
                for m2 in 0..act_s {
                    if m2 == m || !kin_cand[m2] {
                        continue;
                    }
                    let mut dot = 0.0_f32;
                    for p in 0..STRIATAL_PORES {
                        dot += jm_p[m][p] * jm_p[m2][p];
                    }
                    if dot * dot >= 0.015 * sq_m * jm_sq_norm[m2] {
                        for p in 0..STRIATAL_PORES {
                            grp_p[p] = grp_p[p].max(jm_p[m2][p]);
                        }
                    }
                }
                let mut grp_max = 0.0_f32;
                for &gv in &grp_p {
                    grp_max = grp_max.max(gv);
                }
                let thr = 0.025 * grp_max;
                let mut cnt = 0usize;
                for p in 0..STRIATAL_PORES {
                    if feat_active[p] && grp_p[p] >= thr {
                        loop_mask[m][p] = true;
                        cnt += 1;
                    }
                }
                has_kin_loop[m] = cnt >= 3;
            }
        }
        let loop_feat_for =
            |m: usize, has_kl: &[bool], lmask: &[[bool; STRIATAL_PORES]]| -> [f32; STRIATAL_CUES] {
                let mut out_f = [0.0_f32; STRIATAL_CUES];
                if !has_kl[m] {
                    out_f.copy_from_slice(&feat_now[..STRIATAL_CUES]);
                    return out_f;
                }
                let mask_m = &lmask[m];
                let mut mean_m = 0.0_f32;
                let mut cnt_m = 0usize;
                for p in 0..STRIATAL_PORES {
                    if mask_m[p] {
                        mean_m += pore_sig[p];
                        cnt_m += 1;
                    }
                }
                if cnt_m > 0 {
                    mean_m /= cnt_m as f32;
                }
                let mut energy_m = 0.0_f32;
                for p in 0..STRIATAL_PORES {
                    if mask_m[p] {
                        let z = pore_sig[p] - mean_m;
                        let on_lo = z.clamp(0.0, 0.22);
                        let on_hi = (z - 0.22).max(0.0);
                        let off_lo = (-z).clamp(0.0, 0.22);
                        let off_hi = (-z - 0.22).max(0.0);
                        out_f[p] = on_lo;
                        out_f[STRIATAL_PORES + p] = on_hi;
                        out_f[2 * STRIATAL_PORES + p] = off_lo;
                        out_f[3 * STRIATAL_PORES + p] = off_hi;
                        energy_m +=
                            on_lo * on_lo + on_hi * on_hi + off_lo * off_lo + off_hi * off_hi;
                    }
                }
                let inv_m = 1.0 / (0.05 + energy_m).sqrt();
                for fv in &mut out_f {
                    *fv = sanitize(*fv * inv_m);
                }
                out_f
            };

        if is_plastic && self.striatal.has_prev && !switched_context && !act_mismatch_inst {
            let eta_v = 0.05_f32;
            let eta_w = if recalled_mature {
                0.015_f32
            } else if has_cons_model {
                0.068_f32
            } else {
                0.05_f32
            };
            let delta = sanitize_clamp(
                reward + 0.97 * v_now - self.striatal.v_prev,
                -STRIATAL_TD_CLAMP,
                STRIATAL_TD_CLAMP,
            );
            self.striatal.r_bar_ctx[act_c] =
                sanitize(0.99 * self.striatal.r_bar_ctx[act_c] + 0.01 * reward);
            let v_off = act_c * STRIATAL_FEATS;
            for (vi, ei) in self.striatal.v_ctx[v_off..v_off + STRIATAL_FEATS]
                .iter_mut()
                .zip(self.striatal.elig_critic.iter())
            {
                *vi = sanitize_clamp(
                    *vi + eta_v * delta * ei,
                    -STRIATAL_W_CLAMP,
                    STRIATAL_W_CLAMP,
                );
            }
            let prev_cf = self.striatal.prev_critic_feat;
            let rhat_q = self.striatal.outcome_pred_live(act_c, &prev_cf);
            let r_err = sanitize_clamp(reward - rhat_q, -STRIATAL_TD_CLAMP, STRIATAL_TD_CLAMP);
            let eta_q = if reward.abs() > 1.0e-4 { 0.14 } else { 0.005 };
            let q_off = act_c * STRIATAL_CUES;
            for (qi, fi) in self.striatal.q_ctx[q_off..q_off + STRIATAL_CUES]
                .iter_mut()
                .zip(prev_cf[..STRIATAL_CUES].iter())
            {
                *qi = sanitize_clamp(
                    0.9995 * *qi + eta_q * r_err * fi,
                    -STRIATAL_W_CLAMP,
                    STRIATAL_W_CLAMP,
                );
            }
            let w_off = act_c * w_len;
            for m in 0..act_s {
                let row = m * STRIATAL_CUES;
                for p_cue in 0..STRIATAL_CUES {
                    let idx_w = w_off + row + p_cue;
                    let base_w = if recalled_mature {
                        0.996 * self.striatal.w_ctx[idx_w] + 0.004 * self.striatal.w_cons[idx_w]
                    } else {
                        0.9998 * self.striatal.w_ctx[idx_w]
                    };
                    self.striatal.w_ctx[idx_w] = sanitize_clamp(
                        base_w + eta_w * delta * self.striatal.elig_actor[row + p_cue],
                        -STRIATAL_W_CLAMP,
                        STRIATAL_W_CLAMP,
                    );
                }
                let row_slice = &mut self.striatal.w_ctx[w_off + row..w_off + row + STRIATAL_CUES];
                let mut norm_sq = 0.0_f32;
                if has_kin_loop[m] {
                    let mut cnt_l = 0usize;
                    for &in_l in &loop_mask[m] {
                        if in_l {
                            cnt_l += 1;
                        }
                    }
                    let inv_l = if cnt_l > 0 { 1.0 / (cnt_l as f32) } else { 0.0 };
                    for st in 0..STRIATAL_STRATA {
                        let st_base = st * STRIATAL_PORES;
                        let mut sum_st = 0.0_f32;
                        for p in 0..STRIATAL_PORES {
                            if loop_mask[m][p] {
                                sum_st += row_slice[st_base + p];
                            } else {
                                row_slice[st_base + p] = 0.0;
                            }
                        }
                        let mean_st = sum_st * inv_l;
                        for p in 0..STRIATAL_PORES {
                            if loop_mask[m][p] {
                                let centered = sanitize_clamp(
                                    row_slice[st_base + p] - mean_st,
                                    -STRIATAL_W_CLAMP,
                                    STRIATAL_W_CLAMP,
                                );
                                row_slice[st_base + p] = centered;
                                norm_sq += centered * centered;
                            }
                        }
                    }
                } else {
                    for st in 0..STRIATAL_STRATA {
                        let st_base = st * STRIATAL_PORES;
                        let mut sum_st = 0.0_f32;
                        for p in 0..STRIATAL_PORES {
                            sum_st += row_slice[st_base + p];
                        }
                        let mean_st = sum_st * (1.0 / (STRIATAL_PORES as f32));
                        for p in 0..STRIATAL_PORES {
                            let centered = sanitize_clamp(
                                row_slice[st_base + p] - mean_st,
                                -STRIATAL_W_CLAMP,
                                STRIATAL_W_CLAMP,
                            );
                            row_slice[st_base + p] = centered;
                            norm_sq += centered * centered;
                        }
                    }
                }
                if norm_sq > STRIATAL_ROW_NORM_CAP * STRIATAL_ROW_NORM_CAP {
                    let sc = STRIATAL_ROW_NORM_CAP / norm_sq.sqrt();
                    for wi in row_slice.iter_mut() {
                        *wi = sanitize(*wi * sc);
                    }
                }
            }
        }
        self.striatal.prev_critic_feat = feat_now;

        let decay = 0.97_f32 * 0.90_f32;
        let rho = 0.85_f32;
        let innov = 0.60_f32 * (1.0_f32 - rho * rho).sqrt();
        let w_off = read_c * w_len;
        let v_read_off = read_c * STRIATAL_FEATS;
        let q_read_off = read_c * STRIATAL_CUES;
        let fj_read_off = read_c * fwd_len;
        let v_slice: &[f32] = if let Some(ref dec) = ro_paged_slab {
            &dec.v_cons[..STRIATAL_CUES]
        } else if use_cons_w {
            &self.striatal.v_cons[v_read_off..v_read_off + STRIATAL_CUES]
        } else {
            &self.striatal.v_ctx[v_read_off..v_read_off + STRIATAL_CUES]
        };
        let q_slice: &[f32] = if let Some(ref dec) = ro_paged_slab {
            &dec.q_cons
        } else if use_cons_w {
            &self.striatal.q_cons[q_read_off..q_read_off + STRIATAL_CUES]
        } else {
            &self.striatal.q_ctx[q_read_off..q_read_off + STRIATAL_CUES]
        };
        let j_slice: &[f32] = if let Some(ref dec) = ro_paged_slab {
            &dec.fwd_j_cons
        } else if use_cons_w {
            &self.striatal.fwd_j_cons[fj_read_off..fj_read_off + fwd_len]
        } else {
            &self.striatal.fwd_j_ctx[fj_read_off..fj_read_off + fwd_len]
        };

        let mut top_habit = [0.0_f32; 6];
        for m in 0..act_s {
            let row = m * STRIATAL_CUES;
            let ws: &[f32] = if let Some(ref dec) = ro_paged_slab {
                &dec.w_cons[row..row + STRIATAL_CUES]
            } else if use_cons_w {
                &self.striatal.w_cons[w_off + row..w_off + row + STRIATAL_CUES]
            } else {
                &self.striatal.w_ctx[w_off + row..w_off + row + STRIATAL_CUES]
            };
            let mut rsq = 0.0_f32;
            for &wv in ws {
                rsq += wv * wv;
            }
            let mut ins = rsq.sqrt();
            for slot in &mut top_habit {
                if ins > *slot {
                    std::mem::swap(&mut ins, slot);
                }
            }
        }
        let h_cnt = act_s.clamp(1, 6);
        let mut habit_norm_sum = 0.0_f32;
        for &hn in &top_habit[..h_cnt] {
            habit_norm_sum += hn;
        }
        let mean_habit_norm = habit_norm_sum / (h_cnt as f32);
        let mb_scale = 1.0 - (mean_habit_norm / 1.35).clamp(0.0, 0.90);
        let mb_gate = if ro_paged_slab.is_some() || recalled_mature {
            let mut j_sq = 0.0_f32;
            for &jv in j_slice {
                j_sq += jv * jv;
            }
            (j_sq * 20.0).clamp(0.0, 1.0) * mb_scale
        } else if has_cons_model {
            (self.striatal.forward_agency(act_s) * 400.0).clamp(0.0, 1.0) * mb_scale
        } else {
            0.0
        };
        let mut mb_drive = vec![0.0_f32; na];
        if mb_gate > 1.0e-3 {
            let mut pool_st = [0.0_f32; STRIATAL_STRATA];
            let mut v_mean_st = [0.0_f32; STRIATAL_STRATA];
            for st in 0..STRIATAL_STRATA {
                let st_base = st * STRIATAL_PORES;
                let mut sum_q = 0.0_f32;
                let mut mv = 0.0_f32;
                for p in 0..STRIATAL_PORES {
                    sum_q += q_slice[st_base + p];
                    mv += v_slice[st_base + p];
                }
                pool_st[st] = if (st & 1) != 0 {
                    (0.30 * sum_q).clamp(-0.26, 0.26)
                } else {
                    (0.15 * sum_q).clamp(-0.12, 0.0)
                };
                v_mean_st[st] = mv * (1.0 / (STRIATAL_PORES as f32));
            }
            let mut jf_norm_sq = 0.0_f32;
            for m in 0..act_s {
                let f_m = loop_feat_for(m, &has_kin_loop, &loop_mask);
                let mut gm = 0.0_f32;
                for st in 0..STRIATAL_STRATA {
                    let st_base = st * STRIATAL_PORES;
                    let mut mean_j = 0.0_f32;
                    let mut cnt_j = 0usize;
                    for p in 0..STRIATAL_PORES {
                        if !has_kin_loop[m] || loop_mask[m][p] {
                            mean_j += j_slice[(st_base + p) * na + m];
                            cnt_j += 1;
                        }
                    }
                    if cnt_j > 0 {
                        mean_j /= cnt_j as f32;
                    }
                    let app = pool_st[st];
                    let vm = v_mean_st[st];
                    for p in 0..STRIATAL_PORES {
                        if has_kin_loop[m] && !loop_mask[m][p] {
                            continue;
                        }
                        let idx = st_base + p;
                        let f_v = f_m[idx];
                        let j_v = j_slice[idx * na + m] - mean_j;
                        let val_v = 1.6 * q_slice[idx] + 0.25 * (v_slice[idx] - vm) + app;
                        gm += f_v * val_v * j_v;
                        jf_norm_sq += (f_v * j_v) * (f_v * j_v);
                    }
                }
                mb_drive[m] = gm;
            }
            let inv_jf = 9.0 * mb_gate / (jf_norm_sq + 2.5e-4).sqrt();
            let mut mb_sq = 0.0_f32;
            for d in &mut mb_drive[..act_s] {
                *d = sanitize(*d * inv_jf);
                mb_sq += *d * *d;
            }
            let max_mb = 1.25 * mb_scale;
            if mb_sq > max_mb * max_mb && mb_sq > 1.0e-12 {
                let sc = max_mb / mb_sq.sqrt();
                for d in &mut mb_drive[..act_s] {
                    *d = sanitize(*d * sc);
                }
            }
        }

        let det_phase = if let Some(ref dec) = ro_paged_slab {
            let mut osc_a = 0.0_f32;
            let mut osc_b = 0.0_f32;
            for p_c in 0..STRIATAL_PORES {
                if (dec.dend_env[p_c] & 1) != 0 && live_pore_ch[p_c] != u32::MAX {
                    let diff = raw_x_arr[p_c] - dec.dend_key[p_c];
                    if dec.dend_key[p_c] >= 0.30 {
                        osc_a += diff;
                    } else {
                        osc_b += diff;
                    }
                }
            }
            Some(osc_b.atan2(osc_a + 1.0e-6))
        } else {
            None
        };

        let plastic_noise_gain = if recalled_mature {
            if tissue_stress > 0.02 { 1.35 } else { 0.20 }
        } else if is_plastic
            && carrier_bank
            && self.striatal.ctx_has_cons[act_c] != 0
            && self.striatal.r_fast_ema > -0.008
            && tissue_stress <= 0.015
        {
            1.0 - 0.40 * (mean_habit_norm / 1.2).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let pre_gain_pl = if recalled_mature && tissue_stress > 0.02 {
            0.15
        } else {
            1.0
        };

        let mut out = vec![0.0_f32; num_actuators];
        for (m, dst) in out.iter_mut().enumerate() {
            if let Some(ph) = det_phase {
                let m_ang = ph * (1.0 + 0.37 * ((m & 7) as f32)) + (m as f32) * 1.919_862_2;
                let (pre_gain, d_amp) = if tissue_stress > 0.02 {
                    (0.15, 1.45)
                } else {
                    (1.0, 0.14)
                };
                let dither = d_amp * m_ang.sin();
                if m >= act_s {
                    *dst = sanitize_clamp(dither.tanh(), -1.0, 1.0);
                    continue;
                }
                let row = m * STRIATAL_CUES;
                let w_slice: &[f32] = if let Some(ref dec) = ro_paged_slab {
                    &dec.w_cons[row..row + STRIATAL_CUES]
                } else {
                    &self.striatal.w_cons[w_off + row..w_off + row + STRIATAL_CUES]
                };
                let f_m = loop_feat_for(m, &has_kin_loop, &loop_mask);
                let mut pre = mb_drive[m];
                for (w_val, f_val) in w_slice.iter().zip(f_m.iter()) {
                    pre += w_val * f_val;
                }
                let a_clean = sanitize_clamp((pre_gain * pre + dither).tanh(), -1.0, 1.0);
                *dst = a_clean;
                self.striatal.prev_act[m] = a_clean;
                continue;
            }

            let g = self.striatal.gaussian();
            if m >= act_s {
                *dst = sanitize_clamp((innov * g).tanh(), -1.0, 1.0);
                continue;
            }
            self.striatal.noise[m] =
                sanitize_clamp(rho * self.striatal.noise[m] + innov * g, -4.0, 4.0);
            let row = m * STRIATAL_CUES;
            let w_slice = if use_cons_w {
                &self.striatal.w_cons[w_off + row..w_off + row + STRIATAL_CUES]
            } else {
                &self.striatal.w_ctx[w_off + row..w_off + row + STRIATAL_CUES]
            };
            let f_m = loop_feat_for(m, &has_kin_loop, &loop_mask);
            let mut pre = mb_drive[m];
            for (w_val, f_val) in w_slice.iter().zip(f_m.iter()) {
                pre += w_val * f_val;
            }
            let scaled_pre = pre_gain_pl * pre;
            let eff_noise = plastic_noise_gain * self.striatal.noise[m];
            let mean_a = scaled_pre.tanh();
            let a = (scaled_pre + eff_noise).tanh();
            let a_clean = sanitize_clamp(a, -1.0, 1.0);
            *dst = a_clean;
            self.striatal.prev_act[m] = a_clean;
            let pert = a - mean_a;
            if is_plastic {
                if has_kin_loop[m] {
                    for st in 0..STRIATAL_STRATA {
                        let st_base = st * STRIATAL_PORES;
                        for p in 0..STRIATAL_PORES {
                            let e = &mut self.striatal.elig_actor[row + st_base + p];
                            if loop_mask[m][p] {
                                *e = sanitize(decay * *e + pert * f_m[st_base + p]);
                            } else {
                                *e = sanitize(0.5 * *e);
                            }
                        }
                    }
                } else {
                    for (e, f_val) in self.striatal.elig_actor[row..row + STRIATAL_CUES]
                        .iter_mut()
                        .zip(f_m.iter())
                    {
                        *e = sanitize(decay * *e + pert * f_val);
                    }
                }
            }
        }
        if is_plastic {
            for (e, f_val) in self.striatal.elig_critic.iter_mut().zip(feat_now.iter()) {
                *e = sanitize(decay * *e + f_val);
            }
        }
        for tap in (1..STRIATAL_TAPS).rev() {
            let dst_off = tap * na;
            let src_off = (tap - 1) * na;
            for m in 0..act_s {
                self.striatal.da_ring[dst_off + m] = self.striatal.da_ring[src_off + m];
            }
        }
        for m in 0..act_s {
            self.striatal.da_ring[m] =
                sanitize(self.striatal.prev_act[m] - self.striatal.act_mean[m]);
        }

        self.striatal.v_prev = v_now;
        self.striatal.has_prev = true;
        if is_plastic {
            self.striatal.tick = self.striatal.tick.saturating_add(1);
        }
        Ok((Some(out), splits))
    }

    pub fn zero_learned_conductances(&mut self) {
        for t in &mut self.slots {
            t.u = [0.0; 4];
            t.w_pred = 0.0;
            t.contradiction_accum = 0.0;
            t.chain_score = 0.0;
            t.myelin_plasticity_accum = 0.0;
        }
        for row in &mut self.action_sensory_w {
            row.fill(0.0);
        }
        for row in &mut self.pore_action_w {
            *row = [0.0; PRED_BUCKETS];
        }
        for row in &mut self.tectal_map_w {
            *row = [0.0; PRED_BUCKETS];
        }
        self.sensory_self_w.fill(0.0);
        self.pore_self_w = [0.0; PRED_BUCKETS];
        self.pore_mean_v = [0.0; PRED_BUCKETS];
        self.pore_scale_v = [0.05; PRED_BUCKETS];
        self.pore_agency = [0.0; PRED_BUCKETS];
        self.pore_dsq_ema = [1.0e-4; PRED_BUCKETS];
        self.pore_var_ema = [5.0e-3; PRED_BUCKETS];
        self.pore_bg_ild = [0.0; PRED_BUCKETS];
        self.pore_floor_v = [0.0; PRED_BUCKETS];
        self.pore_ar1_w = [0.59; PRED_BUCKETS];
        self.pore_ar2_w = [0.0; PRED_BUCKETS];
        self.pore_q5_lvl = [1.0; PRED_BUCKETS];
        self.pore_q5_exaff = [0.0; PRED_BUCKETS];
        self.pore_vocal_reaff = [0.0; PRED_BUCKETS];
        self.tonotopic_floor_ema = 0.0;
        self.tonotopic_conf_ema = 0.0;
        self.acoustic_env_ema = 0.0;
        self.agency_signed_ema = 0.0;
        self.vocal_eff_s1 = 0.0;
        self.vocal_eff_s2 = 0.0;
        self.vocal_eff_hist = 0;
        self.vocal_onset_hist = 0;
        self.vocal_last_mg = u8::MAX;
        self.vocal_q5_cnt = 0;
        self.vocal_y_ema = 0.0;
        self.vocal_f_ema = [0.0; 4];
        self.vocal_blk_sum = [0.0; 4];
        self.vocal_s_acc = [0.0; 4];
        self.vocal_v_acc = [0.0; 4];
        self.even_tick_cnt = 0;
        self.even_y_ema = 0.0;
        self.even_f_ema = [0.0; 4];
        self.even_blk_sum = [0.0; 4];
        self.even_s_acc = [0.0; 4];
        self.even_v_acc = [0.0; 4];
        self.social_loom_ema = 0.0;
        self.spatial_prox_ring = [0.0; RING_LEN];
        self.spatial_prox_cursor = 0;
        self.echo_emit_w.fill(0.0);
        self.vocal_artic_w.fill(0.0);
        self.somatic_limb_w.fill(0.0);
        self.vocal_glottal_w.fill(0.0);
        self.peer_coch_fast = 0.0;
        self.peer_coch_slow = 0.0;
        self.peer_echo_fast = 0.0;
        self.peer_echo_slow = 0.0;
        self.peer_disp_coch = 0.0;
        self.peer_disp_echo = 0.0;
        self.peer_vocal_hold = 0.0;
        self.peer_coch_age = 255;
        self.peer_echo_age = 255;
        self.peer_bearing_ema.fill(0.0);
        self.act_mean.fill(0.0);
        self.mean_interventional_agency = 0.0;
        self.striatal.reset_learned_keep_birth();
    }

    pub fn hash_learned(&self, hasher: &mut ParamHasher) {
        hasher.write_usize(self.slots.len());
        hasher.write_usize(self.alive_count);
        for t in &self.slots {
            hasher.write_bool(t.alive);
            if t.alive {
                hasher.write_u8(t.kind as u8);
                hasher.write_u32(t.pre);
                hasher.write_u32(t.post);
                hasher.write_u8(t.delay);
                for &uk in &t.u {
                    hasher.write_f32(uk);
                }
                hasher.write_f32(t.w_pred);
                hasher.write_f32(t.myelin_plasticity_accum);
                match t.context_gate_cell {
                    None => hasher.write_u32(u32::MAX),
                    Some(g) => hasher.write_u32(g),
                }
                hasher.write_bool(t.gate_invert);
            }
        }
        for row in &self.action_sensory_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for row in &self.pore_action_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for row in &self.tectal_map_w {
            for &w in row {
                hasher.write_f32(w);
            }
        }
        for &w in &self.sensory_self_w {
            hasher.write_f32(w);
        }
        for &w in &self.pore_self_w {
            hasher.write_f32(w);
        }
        for &mv in &self.pore_mean_v {
            hasher.write_f32(mv);
        }
        for &sv in &self.pore_scale_v {
            hasher.write_f32(sv);
        }
        for &pa in &self.pore_agency {
            hasher.write_f32(pa);
        }
        for &dsq in &self.pore_dsq_ema {
            hasher.write_f32(dsq);
        }
        for &pvr in &self.pore_var_ema {
            hasher.write_f32(pvr);
        }
        for &pbg in &self.pore_bg_ild {
            hasher.write_f32(pbg);
        }
        for &pfv in &self.pore_floor_v {
            hasher.write_f32(pfv);
        }
        for &a1 in &self.pore_ar1_w {
            hasher.write_f32(a1);
        }
        for &a2 in &self.pore_ar2_w {
            hasher.write_f32(a2);
        }
        for &ql in &self.pore_q5_lvl {
            hasher.write_f32(ql);
        }
        for &qex in &self.pore_q5_exaff {
            hasher.write_f32(qex);
        }
        for &pvr in &self.pore_vocal_reaff {
            hasher.write_f32(pvr);
        }
        hasher.write_f32(self.tonotopic_floor_ema);
        hasher.write_f32(self.tonotopic_conf_ema);
        hasher.write_f32(self.acoustic_env_ema);
        hasher.write_f32(self.agency_signed_ema);
        hasher.write_f32(self.vocal_eff_s1);
        hasher.write_f32(self.vocal_eff_s2);
        hasher.write_u32(self.vocal_eff_hist);
        hasher.write_u32(self.vocal_onset_hist);
        hasher.write_u8(self.vocal_last_mg);
        hasher.write_u32(self.vocal_q5_cnt);
        hasher.write_f32(self.vocal_y_ema);
        for &vf in &self.vocal_f_ema {
            hasher.write_f32(vf);
        }
        for &vbs in &self.vocal_blk_sum {
            hasher.write_f32(vbs);
        }
        for &vsa in &self.vocal_s_acc {
            hasher.write_f32(vsa);
        }
        for &vva in &self.vocal_v_acc {
            hasher.write_f32(vva);
        }
        hasher.write_u32(self.even_tick_cnt);
        hasher.write_f32(self.even_y_ema);
        for &ef in &self.even_f_ema {
            hasher.write_f32(ef);
        }
        for &ebs in &self.even_blk_sum {
            hasher.write_f32(ebs);
        }
        for &esa in &self.even_s_acc {
            hasher.write_f32(esa);
        }
        for &eva in &self.even_v_acc {
            hasher.write_f32(eva);
        }
        hasher.write_f32(self.social_loom_ema);
        for &spv in &self.spatial_prox_ring {
            hasher.write_f32(spv);
        }
        hasher.write_u8(self.spatial_prox_cursor);
        for &ew in &self.echo_emit_w {
            hasher.write_f32(ew);
        }
        for &vw in &self.vocal_artic_w {
            hasher.write_f32(vw);
        }
        for &sw in &self.somatic_limb_w {
            hasher.write_f32(sw);
        }
        for &vgw in &self.vocal_glottal_w {
            hasher.write_f32(vgw);
        }
        hasher.write_f32(self.peer_coch_fast);
        hasher.write_f32(self.peer_coch_slow);
        hasher.write_f32(self.peer_echo_fast);
        hasher.write_f32(self.peer_echo_slow);
        hasher.write_f32(self.peer_disp_coch);
        hasher.write_f32(self.peer_disp_echo);
        hasher.write_f32(self.peer_vocal_hold);
        hasher.write_u8(self.peer_coch_age);
        hasher.write_u8(self.peer_echo_age);
        for &pb in &self.peer_bearing_ema {
            hasher.write_f32(pb);
        }
        for &am in &self.act_mean {
            hasher.write_f32(am);
        }
        self.striatal.hash_learned(hasher);
    }
}
