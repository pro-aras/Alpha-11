//! Versioned, CRC32C-verified, shadow-decoded transactional checkpoint persistence.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

use a11_medium::{ControlMode, PersistError};

use crate::Alpha11Organism;
use crate::OrganismConfig;
use crate::body::{BodyCore, EXPLORE_CHANNELS};
use crate::cells::{Cell, CellArena};
use crate::crc32c::{crc32c, crc32c_update};
use crate::pores::{ChannelScoutStat, MAX_SCOUT_CHANNELS, MotorPore, PoreMembrane, SensorPore};
use crate::rng::{Rng64, sanitize};
use crate::tubes::{
    EfferenceEntry, EfferenceRing, OFC_CONTEXTS, PRED_BUCKETS, RING_LEN, STRIATAL_ACTUATORS,
    STRIATAL_CUES, STRIATAL_FEATS, STRIATAL_FWD_LEN, STRIATAL_PORES, STRIATAL_TAP_DIM,
    STRIATAL_W_LEN, StriatalOfc, TRACKED_ACTUATORS, Tube, TubeGraph, TubeKind,
};

pub const FILE_MAGIC: u64 = 0x4131_3143_484B_5031;
pub const FORMAT_VERSION: u32 = 5;
pub const MIN_COMPAT_FORMAT_VERSION: u32 = 3;
pub const SLAB_COUNT: u32 = 6;

pub const STRIATAL_SERIALIZED_BYTES: usize = 3
    + OFC_CONTEXTS
    + 3 * 4 * OFC_CONTEXTS
    + 5 * 4 * STRIATAL_PORES
    + 4 * STRIATAL_PORES * STRIATAL_TAP_DIM
    + 4 * STRIATAL_W_LEN
    + 2 * 4 * OFC_CONTEXTS * STRIATAL_W_LEN
    + 2 * 4 * OFC_CONTEXTS * STRIATAL_FEATS
    + 2 * 4 * OFC_CONTEXTS * STRIATAL_CUES
    + 4 * 4 * STRIATAL_CUES
    + (1 + 2 * OFC_CONTEXTS) * 4 * STRIATAL_FWD_LEN
    + 2 * 4 * OFC_CONTEXTS
    + 4 * STRIATAL_ACTUATORS
    + 8
    + 4 * OFC_CONTEXTS
    + 12
    + 4 * STRIATAL_TAP_DIM
    + 4 * STRIATAL_W_LEN
    + 3 * 4 * STRIATAL_FEATS
    + 2 * 4 * STRIATAL_CUES
    + 2 * 4 * STRIATAL_ACTUATORS
    + 4
    + 1
    + 24;

pub const TAG_BODY: u32 = 0x424F_4459;
pub const TAG_CELL: u32 = 0x4345_4C4C;
pub const TAG_TUBE: u32 = 0x5455_4245;
pub const TAG_PORE: u32 = 0x504F_5245;
pub const TAG_RING: u32 = 0x5249_4E47;
pub const TAG_SCOT: u32 = 0x5343_4F54;

const HEADER_OVERHEAD_MARGIN: usize = 256 * 1024;

struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline]
    fn write_bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    #[inline]
    fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn write_f32(&mut self, v: f32) -> Result<(), PersistError> {
        if !v.is_finite() {
            return Err(PersistError::NonFiniteState);
        }
        self.buf
            .extend_from_slice(&sanitize(v).to_bits().to_le_bytes());
        Ok(())
    }
}

struct ByteReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], PersistError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(PersistError::TruncatedInput)?;
        if end > self.data.len() {
            return Err(PersistError::TruncatedInput);
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    #[inline]
    fn read_u8(&mut self) -> Result<u8, PersistError> {
        let b = self.take(1)?;
        Ok(b[0])
    }

    #[inline]
    fn read_bool(&mut self) -> Result<bool, PersistError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PersistError::InvalidTopology),
        }
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, PersistError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, PersistError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, PersistError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    fn read_f32(&mut self) -> Result<f32, PersistError> {
        let bits = self.read_u32()?;
        let v = f32::from_bits(bits);
        if !v.is_finite() {
            return Err(PersistError::NonFiniteState);
        }
        Ok(sanitize(v))
    }
}

fn append_slab(out: &mut Vec<u8>, tag: u32, payload: &[u8]) -> Result<(), PersistError> {
    let len_u32 = u32::try_from(payload.len()).map_err(|_| PersistError::OversizedInput)?;
    let tag_bytes = tag.to_le_bytes();
    let len_bytes = len_u32.to_le_bytes();

    let mut crc = 0xFFFF_FFFF;
    crc = crc32c_update(crc, &tag_bytes);
    crc = crc32c_update(crc, &len_bytes);
    crc = crc32c_update(crc, payload);
    let slab_crc = !crc;

    out.extend_from_slice(&tag_bytes);
    out.extend_from_slice(&len_bytes);
    out.extend_from_slice(payload);
    out.extend_from_slice(&slab_crc.to_le_bytes());
    Ok(())
}

pub fn encode_organism(org: &Alpha11Organism) -> Result<Vec<u8>, PersistError> {
    if org.poisoned {
        return Err(PersistError::PoisonedAfterCorruption);
    }
    if org.used_state_bytes() > org.config.max_state_bytes {
        return Err(PersistError::OversizedInput);
    }

    let mut body_w = ByteWriter::with_capacity(256);
    body_w.write_u64(org.seed);
    body_w.write_u64(org.config.initial_cells as u64);
    body_w.write_u64(org.config.max_cells as u64);
    body_w.write_u64(org.config.max_tubes as u64);
    body_w.write_u64(org.config.max_pores as u64);
    body_w.write_u64(org.config.max_state_bytes as u64);
    body_w.write_f32(org.config.initial_atp)?;
    body_w.write_f32(org.config.max_atp)?;
    let mode_byte = match org.mode {
        ControlMode::Natural => 0u8,
        ControlMode::FrozenPlasticity => 1u8,
    };
    body_w.write_u8(mode_byte);
    body_w.write_u64(org.rng.s0);
    body_w.write_u64(org.rng.s1);
    body_w.write_u64(org.explore_rng.s0);
    body_w.write_u64(org.explore_rng.s1);
    body_w.write_u64(org.tick);
    body_w.write_u64(org.committed_motor_pulses);
    body_w.write_u64(org.subthreshold_rehearsals);
    body_w.write_u64(org.mitosis_births);
    body_w.write_u64(org.contradiction_splits);
    body_w.write_u64(org.myelinated_shortcuts);
    body_w.write_u64(org.atrophied_structures);
    body_w.write_f32(org.last_motor_energy)?;

    body_w.write_f32(org.body.atp)?;
    body_w.write_f32(org.body.max_atp)?;
    body_w.write_f32(org.body.initial_atp)?;
    body_w.write_f32(org.body.adenosine)?;
    body_w.write_f32(org.body.temperature)?;
    body_w.write_f32(org.body.motor_clutch_threshold)?;
    body_w.write_f32(org.body.baseline_valence_ema)?;
    body_w.write_f32(org.body.prev_raw_valence)?;
    body_w.write_f32(org.body.valence_scale_ema)?;
    body_w.write_f32(org.body.valence_min_ema)?;
    body_w.write_f32(org.body.valence_max_ema)?;
    body_w.write_f32(org.body.ers_scarcity)?;
    body_w.write_f32(org.body.fast_valence)?;
    body_w.write_u16(org.body.rest_ticks_remaining);
    body_w.write_u64(org.body.rest_consolidation_ticks);
    for &v in &org.body.explore_momentum {
        body_w.write_f32(v)?;
    }
    for &v in &org.body.explore_dc_accum {
        body_w.write_f32(v)?;
    }
    for &v in &org.body.explore_target {
        body_w.write_f32(v)?;
    }
    for &b in &org.body.explore_hold_ticks {
        body_w.write_u8(b);
    }

    let mut cell_w = ByteWriter::with_capacity(org.cells.slots.len() * 68 + 64);
    cell_w.write_u32(org.cells.slots.len() as u32);
    cell_w.write_u32(org.cells.alive_count as u32);
    cell_w.write_u32(org.cells.min_cells_floor as u32);
    for c in &org.cells.slots {
        cell_w.write_bool(c.alive);
        cell_w.write_f32(c.v_real)?;
        cell_w.write_f32(c.v_prev)?;
        cell_w.write_f32(c.delta_v)?;
        cell_w.write_f32(c.v_pred)?;
        cell_w.write_f32(c.pred_err)?;
        cell_w.write_f32(c.phi)?;
        cell_w.write_f32(c.tau)?;
        cell_w.write_f32(c.firing_rate_ema)?;
        cell_w.write_f32(c.excitability_bias)?;
        cell_w.write_f32(c.v_homeo)?;
        cell_w.write_f32(c.v_delta_homeo)?;
        cell_w.write_f32(c.atp_credit_ema)?;
        cell_w.write_f32(c.surprise_ema)?;
        cell_w.write_f32(c.sensory_trophic_ema)?;
        cell_w.write_u16(c.mitosis_cooldown);
        cell_w.write_u32(c.age_ticks);
    }
    cell_w.write_u32(org.cells.free_list.len() as u32);
    for &idx in &org.cells.free_list {
        cell_w.write_u32(idx);
    }
    cell_w.write_u32(org.cells.active_indices.len() as u32);
    for &idx in &org.cells.active_indices {
        cell_w.write_u32(idx);
    }
    cell_w.write_f32(org.cells.theta_phase)?;
    cell_w.write_f32(org.cells.error_reversal_gate)?;

    let mut tube_w =
        ByteWriter::with_capacity(org.tubes.slots.len() * 64 + 9600 + STRIATAL_SERIALIZED_BYTES);
    tube_w.write_u32(org.tubes.slots.len() as u32);
    tube_w.write_u32(org.tubes.alive_count as u32);
    for t in &org.tubes.slots {
        tube_w.write_bool(t.alive);
        tube_w.write_u8(t.kind as u8);
        tube_w.write_u32(t.pre);
        tube_w.write_u32(t.post);
        tube_w.write_u8(t.delay);
        for &uk in &t.u {
            tube_w.write_f32(uk)?;
        }
        tube_w.write_f32(t.w_pred)?;
        match t.context_gate_cell {
            None => {
                tube_w.write_bool(false);
                tube_w.write_u32(0);
            }
            Some(g) => {
                tube_w.write_bool(true);
                tube_w.write_u32(g);
            }
        }
        tube_w.write_bool(t.gate_invert);
        tube_w.write_f32(t.contradiction_accum)?;
        tube_w.write_f32(t.chain_score)?;
        tube_w.write_f32(t.myelin_plasticity_accum)?;
        tube_w.write_u32(t.last_upstream_cell);
        tube_w.write_f32(t.usage_ema)?;
        tube_w.write_u32(t.age_ticks);
    }
    tube_w.write_u32(org.tubes.free_list.len() as u32);
    for &idx in &org.tubes.free_list {
        tube_w.write_u32(idx);
    }
    for row in &org.tubes.action_sensory_w {
        for &w in row {
            tube_w.write_f32(w)?;
        }
    }
    for row in &org.tubes.pore_action_w {
        for &w in row {
            tube_w.write_f32(w)?;
        }
    }
    for row in &org.tubes.tectal_map_w {
        for &w in row {
            tube_w.write_f32(w)?;
        }
    }
    for &w in &org.tubes.sensory_self_w {
        tube_w.write_f32(w)?;
    }
    for &w in &org.tubes.pore_self_w {
        tube_w.write_f32(w)?;
    }
    for &mv in &org.tubes.pore_mean_v {
        tube_w.write_f32(mv)?;
    }
    for &sv in &org.tubes.pore_scale_v {
        tube_w.write_f32(sv)?;
    }
    for &pa in &org.tubes.pore_agency {
        tube_w.write_f32(pa)?;
    }
    for &dsq in &org.tubes.pore_dsq_ema {
        tube_w.write_f32(dsq)?;
    }
    for &pvr in &org.tubes.pore_var_ema {
        tube_w.write_f32(pvr)?;
    }
    for &pbg in &org.tubes.pore_bg_ild {
        tube_w.write_f32(pbg)?;
    }
    tube_w.write_f32(org.tubes.social_loom_ema)?;
    for &spv in &org.tubes.spatial_prox_ring {
        tube_w.write_f32(spv)?;
    }
    tube_w.write_u8(org.tubes.spatial_prox_cursor);
    for &ew in &org.tubes.echo_emit_w {
        tube_w.write_f32(ew)?;
    }
    for &vw in &org.tubes.vocal_artic_w {
        tube_w.write_f32(vw)?;
    }
    for &sw in &org.tubes.somatic_limb_w {
        tube_w.write_f32(sw)?;
    }
    tube_w.write_f32(org.tubes.peer_coch_fast)?;
    tube_w.write_f32(org.tubes.peer_coch_slow)?;
    tube_w.write_f32(org.tubes.peer_echo_fast)?;
    tube_w.write_f32(org.tubes.peer_echo_slow)?;
    tube_w.write_f32(org.tubes.peer_disp_coch)?;
    tube_w.write_f32(org.tubes.peer_disp_echo)?;
    tube_w.write_f32(org.tubes.peer_vocal_hold)?;
    tube_w.write_u8(org.tubes.peer_coch_age);
    tube_w.write_u8(org.tubes.peer_echo_age);
    for &pb in &org.tubes.peer_bearing_ema {
        tube_w.write_f32(pb)?;
    }
    for &am in &org.tubes.act_mean {
        tube_w.write_f32(am)?;
    }
    tube_w.write_f32(org.tubes.mean_interventional_agency)?;
    let before_striatal = tube_w.buf.len();
    let st = &org.tubes.striatal;
    tube_w.write_u8(st.num_ctx);
    tube_w.write_u8(st.active_ctx);
    tube_w.write_u8(st.impact_refrac);
    for &hc in &st.ctx_has_cons {
        tube_w.write_u8(hc);
    }
    for &ct in &st.ctx_ticks {
        tube_w.write_u32(ct);
    }
    for &rp in &st.ctx_r_peak {
        tube_w.write_f32(rp)?;
    }
    for &rb in &st.r_bar_ctx {
        tube_w.write_f32(rb)?;
    }
    for &pc in &st.pore_ch {
        tube_w.write_u32(pc);
    }
    for &pa in &st.pore_age {
        tube_w.write_u32(pa);
    }
    for &pm in &st.pore_mean {
        tube_w.write_f32(pm)?;
    }
    for &vr in &st.pore_var_raw {
        tube_w.write_f32(vr)?;
    }
    for &ve in &st.pore_var_exaff {
        tube_w.write_f32(ve)?;
    }
    for &cg in &st.canc_g {
        tube_w.write_f32(cg)?;
    }
    for &wb in &st.w_birth {
        tube_w.write_f32(wb)?;
    }
    for &wc in &st.w_ctx {
        tube_w.write_f32(wc)?;
    }
    for &ws in &st.w_cons {
        tube_w.write_f32(ws)?;
    }
    for &vc in &st.v_ctx {
        tube_w.write_f32(vc)?;
    }
    for &vs in &st.v_cons {
        tube_w.write_f32(vs)?;
    }
    for &qc in &st.q_ctx {
        tube_w.write_f32(qc)?;
    }
    for &qs in &st.q_cons {
        tube_w.write_f32(qs)?;
    }
    for &fb in &st.fwd_b {
        tube_w.write_f32(fb)?;
    }
    for &fa in &st.fwd_alpha {
        tube_w.write_f32(fa)?;
    }
    for &fj in &st.fwd_j {
        tube_w.write_f32(fj)?;
    }
    for &fjc in &st.fwd_j_ctx {
        tube_w.write_f32(fjc)?;
    }
    for &fjs in &st.fwd_j_cons {
        tube_w.write_f32(fjs)?;
    }
    for &vn in &st.fwd_var_null {
        tube_w.write_f32(vn)?;
    }
    for &va in &st.fwd_var_act {
        tube_w.write_f32(va)?;
    }
    for &fnm in &st.fwd_num {
        tube_w.write_f32(fnm)?;
    }
    for &fdn in &st.fwd_den {
        tube_w.write_f32(fdn)?;
    }
    for &sam in &st.act_mean {
        tube_w.write_f32(sam)?;
    }
    tube_w.write_u32(st.ctx_cooldown);
    tube_w.write_u32(st.ctx_out_events);
    for &oe in &st.ctx_out_err {
        tube_w.write_f32(oe)?;
    }
    tube_w.write_f32(st.r_fast_ema)?;
    tube_w.write_f32(st.r_med200)?;
    tube_w.write_f32(st.valence_flip_ema)?;
    for &dr in &st.da_ring {
        tube_w.write_f32(dr)?;
    }
    for &ea in &st.elig_actor {
        tube_w.write_f32(ea)?;
    }
    for &ec in &st.elig_critic {
        tube_w.write_f32(ec)?;
    }
    for &ft in &st.feat {
        tube_w.write_f32(ft)?;
    }
    for &pcf in &st.prev_critic_feat {
        tube_w.write_f32(pcf)?;
    }
    for &pf in &st.prev_feat {
        tube_w.write_f32(pf)?;
    }
    for &pdf in &st.prev_df {
        tube_w.write_f32(pdf)?;
    }
    for &pa in &st.prev_act {
        tube_w.write_f32(pa)?;
    }
    for &nz in &st.noise {
        tube_w.write_f32(nz)?;
    }
    tube_w.write_f32(st.v_prev)?;
    tube_w.write_bool(st.has_prev);
    tube_w.write_u64(st.tick);
    tube_w.write_u64(st.rng.s0);
    tube_w.write_u64(st.rng.s1);
    if tube_w.buf.len().saturating_sub(before_striatal) != STRIATAL_SERIALIZED_BYTES {
        return Err(PersistError::InvalidTopology);
    }

    let mut pore_w = ByteWriter::with_capacity(org.membrane.sensor_pores.len() * 112 + 512);
    pore_w.write_u32(org.membrane.sensor_pores.len() as u32);
    pore_w.write_u32(org.membrane.alive_sensor_count as u32);
    for sp in &org.membrane.sensor_pores {
        pore_w.write_bool(sp.alive);
        pore_w.write_u32(sp.channel);
        pore_w.write_u8(sp.delay);
        pore_w.write_u32(sp.target_cell);
        pore_w.write_u32(sp.delta_target_cell);
        pore_w.write_f32(sp.conductance)?;
        pore_w.write_f32(sp.trophic_ema)?;
        pore_w.write_f32(sp.last_val)?;
        pore_w.write_f32(sp.adapt_mean)?;
        pore_w.write_bool(sp.adapt_init);
        for &hv in &sp.history {
            pore_w.write_f32(hv)?;
        }
        pore_w.write_u8(sp.hist_head);
        pore_w.write_u32(sp.age_ticks);
    }
    pore_w.write_u32(org.membrane.sensor_free_list.len() as u32);
    for &idx in &org.membrane.sensor_free_list {
        pore_w.write_u32(idx);
    }
    pore_w.write_u32(org.membrane.motor_pores.len() as u32);
    pore_w.write_u32(org.membrane.alive_motor_count as u32);
    for mp in &org.membrane.motor_pores {
        pore_w.write_bool(mp.alive);
        pore_w.write_u32(mp.actuator_channel);
        pore_w.write_f32(mp.conductance)?;
        pore_w.write_f32(mp.klinokinesis_gain)?;
        pore_w.write_f32(mp.motor_mastery)?;
        pore_w.write_f32(mp.jerk_ema)?;
        pore_w.write_f32(mp.last_out)?;
    }
    pore_w.write_u32(org.membrane.motor_free_list.len() as u32);
    for &idx in &org.membrane.motor_free_list {
        pore_w.write_u32(idx);
    }

    let mut ring_w = ByteWriter::with_capacity(4096);
    ring_w.write_u32(org.ring.head as u32);
    for e in &org.ring.entries {
        ring_w.write_bool(e.valid);
        ring_w.write_u64(e.tick);
        ring_w.write_bool(e.committed);
        ring_w.write_f32(e.motor_norm)?;
        for &mv in &e.motor_out {
            ring_w.write_f32(mv)?;
        }
        ring_w.write_f32(e.somatic_value_est)?;
        ring_w.write_f32(e.raw_valence_at_commit)?;
        ring_w.write_u32(e.active_cells.len() as u32);
        for &(ci, vr, dv, pa) in &e.active_cells {
            ring_w.write_u32(ci);
            ring_w.write_f32(vr)?;
            ring_w.write_f32(dv)?;
            ring_w.write_f32(pa)?;
        }
        ring_w.write_u32(e.pred_with_action.len() as u32);
        for &(ci, pv) in &e.pred_with_action {
            ring_w.write_u32(ci);
            ring_w.write_f32(pv)?;
        }
        ring_w.write_u32(e.pred_null_action.len() as u32);
        for &(ci, pv) in &e.pred_null_action {
            ring_w.write_u32(ci);
            ring_w.write_f32(pv)?;
        }
    }

    let mut scot_w = ByteWriter::with_capacity(org.membrane.scout_stats.len() * 44 + 48);
    scot_w.write_u32(org.membrane.scout_cohort_idx);
    scot_w.write_f32(org.membrane.passive_health_ema)?;
    scot_w.write_f32(org.membrane.motor_env_fast)?;
    scot_w.write_f32(org.membrane.motor_env_mean)?;
    scot_w.write_f32(org.membrane.motor_env_var)?;
    scot_w.write_u32(org.membrane.scout_stats.len() as u32);
    for st in &org.membrane.scout_stats {
        scot_w.write_bool(st.initialized);
        scot_w.write_f32(st.mean)?;
        scot_w.write_f32(st.var_ema)?;
        scot_w.write_f32(st.diff_sq_ema)?;
        scot_w.write_f32(st.prev_val)?;
        scot_w.write_f32(st.ar1_w)?;
        scot_w.write_f32(st.ar1_err_ema)?;
        scot_w.write_f32(st.prev_dx)?;
        scot_w.write_f32(st.cov_dx_ema)?;
        scot_w.write_f32(st.fourth_ema)?;
        scot_w.write_f32(st.motor_cov_ema)?;
    }

    let mut slabs_buf = Vec::with_capacity(
        body_w.buf.len()
            + cell_w.buf.len()
            + tube_w.buf.len()
            + pore_w.buf.len()
            + ring_w.buf.len()
            + scot_w.buf.len()
            + 72,
    );
    append_slab(&mut slabs_buf, TAG_BODY, &body_w.buf)?;
    append_slab(&mut slabs_buf, TAG_CELL, &cell_w.buf)?;
    append_slab(&mut slabs_buf, TAG_TUBE, &tube_w.buf)?;
    append_slab(&mut slabs_buf, TAG_PORE, &pore_w.buf)?;
    append_slab(&mut slabs_buf, TAG_RING, &ring_w.buf)?;
    append_slab(&mut slabs_buf, TAG_SCOT, &scot_w.buf)?;

    let max_allowed = org
        .config
        .max_state_bytes
        .saturating_add(HEADER_OVERHEAD_MARGIN);
    if slabs_buf.len().saturating_add(28) > max_allowed {
        return Err(PersistError::OversizedInput);
    }

    let mut full = Vec::with_capacity(24 + slabs_buf.len() + 4);
    full.extend_from_slice(&FILE_MAGIC.to_le_bytes());
    full.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    full.extend_from_slice(&SLAB_COUNT.to_le_bytes());
    full.extend_from_slice(&(slabs_buf.len() as u64).to_le_bytes());
    full.extend_from_slice(&slabs_buf);
    let file_crc = crc32c(&full);
    full.extend_from_slice(&file_crc.to_le_bytes());

    Ok(full)
}

pub fn shadow_decode_organism(
    bytes: &[u8],
    caller_max_state_bytes: usize,
) -> Result<Alpha11Organism, PersistError> {
    let max_allowed = caller_max_state_bytes.saturating_add(HEADER_OVERHEAD_MARGIN);
    if bytes.len() > max_allowed {
        return Err(PersistError::OversizedInput);
    }
    if bytes.len() < 28 {
        return Err(PersistError::TruncatedInput);
    }

    let mut hdr = ByteReader::new(&bytes[..24]);
    let magic = hdr.read_u64()?;
    let version = hdr.read_u32()?;
    if magic != FILE_MAGIC || !(MIN_COMPAT_FORMAT_VERSION..=FORMAT_VERSION).contains(&version) {
        return Err(PersistError::BadMagicOrVersion);
    }
    let slab_count = hdr.read_u32()?;
    let total_payload_len_u64 = hdr.read_u64()?;
    let total_payload_len =
        usize::try_from(total_payload_len_u64).map_err(|_| PersistError::OversizedInput)?;
    if total_payload_len > max_allowed {
        return Err(PersistError::OversizedInput);
    }
    let expected_total = 24usize
        .checked_add(total_payload_len)
        .and_then(|v| v.checked_add(4))
        .ok_or(PersistError::OversizedInput)?;

    if bytes.len() < expected_total {
        return Err(PersistError::TruncatedInput);
    }
    if bytes.len() > expected_total {
        return Err(PersistError::InvalidTopology);
    }
    if slab_count != SLAB_COUNT {
        return Err(PersistError::InvalidTopology);
    }

    let expected_file_crc = u32::from_le_bytes([
        bytes[expected_total - 4],
        bytes[expected_total - 3],
        bytes[expected_total - 2],
        bytes[expected_total - 1],
    ]);
    let actual_file_crc = crc32c(&bytes[..expected_total - 4]);
    if actual_file_crc != expected_file_crc {
        return Err(PersistError::ChecksumMismatch);
    }

    let expected_tags = [TAG_BODY, TAG_CELL, TAG_TUBE, TAG_PORE, TAG_RING, TAG_SCOT];
    let mut payloads: [&[u8]; 6] = [&[]; 6];
    let mut pos = 24usize;
    let payload_end = expected_total - 4;

    for (idx, &exp_tag) in expected_tags.iter().enumerate() {
        if pos.saturating_add(12) > payload_end {
            return Err(PersistError::TruncatedInput);
        }
        let tag = u32::from_le_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]);
        let s_len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(s_len)
            .ok_or(PersistError::TruncatedInput)?;
        let crc_end = data_end
            .checked_add(4)
            .ok_or(PersistError::TruncatedInput)?;
        if crc_end > payload_end {
            return Err(PersistError::TruncatedInput);
        }

        let stored_slab_crc = u32::from_le_bytes([
            bytes[data_end],
            bytes[data_end + 1],
            bytes[data_end + 2],
            bytes[data_end + 3],
        ]);
        let computed_slab_crc = crc32c(&bytes[pos..data_end]);
        if computed_slab_crc != stored_slab_crc {
            return Err(PersistError::ChecksumMismatch);
        }
        if tag != exp_tag {
            return Err(PersistError::InvalidTopology);
        }

        payloads[idx] = &bytes[data_start..data_end];
        pos = crc_end;
    }
    if pos != payload_end {
        return Err(PersistError::InvalidTopology);
    }

    let mut br = ByteReader::new(payloads[0]);
    let seed = br.read_u64()?;
    let initial_cells =
        usize::try_from(br.read_u64()?).map_err(|_| PersistError::InvalidTopology)?;
    let max_cells = usize::try_from(br.read_u64()?).map_err(|_| PersistError::InvalidTopology)?;
    let max_tubes = usize::try_from(br.read_u64()?).map_err(|_| PersistError::InvalidTopology)?;
    let max_pores = usize::try_from(br.read_u64()?).map_err(|_| PersistError::InvalidTopology)?;
    let max_state_bytes =
        usize::try_from(br.read_u64()?).map_err(|_| PersistError::InvalidTopology)?;
    let initial_atp = br.read_f32()?;
    let max_atp = br.read_f32()?;
    if max_cells < 4
        || max_tubes < 4
        || max_pores < 2
        || max_state_bytes < 1024
        || max_state_bytes > caller_max_state_bytes
        || max_atp < 1.0
        || initial_atp < 0.0
        || initial_atp > max_atp + 1.0e-3
    {
        return Err(PersistError::InvalidTopology);
    }
    let config = OrganismConfig {
        initial_cells,
        max_cells,
        max_tubes,
        max_pores,
        max_state_bytes,
        initial_atp,
        max_atp,
    };

    let mode = match br.read_u8()? {
        0 => ControlMode::Natural,
        1 => ControlMode::FrozenPlasticity,
        _ => return Err(PersistError::InvalidTopology),
    };
    let s0 = br.read_u64()?;
    let s1 = br.read_u64()?;
    let es0 = br.read_u64()?;
    let es1 = br.read_u64()?;
    if (s0 == 0 && s1 == 0) || (es0 == 0 && es1 == 0) {
        return Err(PersistError::InvalidTopology);
    }
    let rng = Rng64::from_raw(s0, s1);
    let explore_rng = Rng64::from_raw(es0, es1);

    let tick = br.read_u64()?;
    let committed_motor_pulses = br.read_u64()?;
    let subthreshold_rehearsals = br.read_u64()?;
    let mitosis_births = br.read_u64()?;
    let contradiction_splits = br.read_u64()?;
    let myelinated_shortcuts = br.read_u64()?;
    let atrophied_structures = br.read_u64()?;
    let last_motor_energy = br.read_f32()?;

    let atp = br.read_f32()?;
    let body_max_atp = br.read_f32()?;
    let body_initial_atp = br.read_f32()?;
    let adenosine = br.read_f32()?;
    let temperature = br.read_f32()?;
    let motor_clutch_threshold = br.read_f32()?;
    let baseline_valence_ema = br.read_f32()?;
    let prev_raw_valence = br.read_f32()?;
    let valence_scale_ema = br.read_f32()?;
    let (valence_min_ema, valence_max_ema, ers_scarcity) = if version >= 4 {
        let vmin = br.read_f32()?;
        let vmax = br.read_f32()?;
        let ers = br.read_f32()?;
        if !(0.0..=1.0001).contains(&ers) || vmax < vmin {
            return Err(PersistError::InvalidTopology);
        }
        (vmin, vmax, ers.clamp(0.0, 1.0))
    } else {
        (
            (baseline_valence_ema - 0.04).min(-0.02),
            (baseline_valence_ema + 0.04).max(0.02),
            0.0,
        )
    };
    let fast_valence = br.read_f32()?;
    let rest_ticks_remaining = br.read_u16()?;
    let rest_consolidation_ticks = br.read_u64()?;

    if atp < -1.0e-4
        || atp > body_max_atp + 1.0e-3
        || !(0.0..=1.0001).contains(&adenosine)
        || !(0.019..=1.0001).contains(&temperature)
    {
        return Err(PersistError::InvalidTopology);
    }

    let mut explore_momentum = [0.0_f32; EXPLORE_CHANNELS];
    for dst in &mut explore_momentum {
        *dst = br.read_f32()?;
    }
    let mut explore_dc_accum = [0.0_f32; EXPLORE_CHANNELS];
    for dst in &mut explore_dc_accum {
        *dst = br.read_f32()?;
    }
    let mut explore_target = [0.0_f32; EXPLORE_CHANNELS];
    for dst in &mut explore_target {
        *dst = br.read_f32()?;
    }
    let mut explore_hold_ticks = [0u8; EXPLORE_CHANNELS];
    for dst in &mut explore_hold_ticks {
        *dst = br.read_u8()?;
    }
    if br.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }

    let body = BodyCore {
        atp,
        max_atp: body_max_atp,
        initial_atp: body_initial_atp,
        adenosine,
        temperature,
        motor_clutch_threshold,
        baseline_valence_ema,
        prev_raw_valence,
        valence_scale_ema,
        valence_min_ema,
        valence_max_ema,
        ers_scarcity,
        fast_valence,
        rest_ticks_remaining,
        rest_consolidation_ticks,
        explore_momentum,
        explore_dc_accum,
        explore_target,
        explore_hold_ticks,
    };

    let mut cr = ByteReader::new(payloads[1]);
    let cell_slots_len = cr.read_u32()? as usize;
    let cell_alive_count = cr.read_u32()? as usize;
    let min_cells_floor = cr.read_u32()? as usize;
    if cell_slots_len == 0 || cell_slots_len > config.max_cells || cell_alive_count > cell_slots_len
    {
        return Err(PersistError::InvalidTopology);
    }
    let mut cell_slots = Vec::with_capacity(cell_slots_len);
    let mut counted_alive_cells = 0usize;
    for _ in 0..cell_slots_len {
        let alive = cr.read_bool()?;
        let v_real = cr.read_f32()?;
        let v_prev = cr.read_f32()?;
        let delta_v = cr.read_f32()?;
        let v_pred = cr.read_f32()?;
        let pred_err = cr.read_f32()?;
        let phi = cr.read_f32()?;
        let tau = cr.read_f32()?;
        let firing_rate_ema = cr.read_f32()?;
        let excitability_bias = cr.read_f32()?;
        let v_homeo = cr.read_f32()?;
        let v_delta_homeo = cr.read_f32()?;
        let atp_credit_ema = cr.read_f32()?;
        let surprise_ema = cr.read_f32()?;
        let sensory_trophic_ema = cr.read_f32()?;
        let mitosis_cooldown = cr.read_u16()?;
        let age_ticks = cr.read_u32()?;

        if alive {
            counted_alive_cells += 1;
            if !(1.0..=2048.0).contains(&tau) {
                return Err(PersistError::InvalidTopology);
            }
        }
        cell_slots.push(Cell {
            alive,
            v_real,
            v_prev,
            delta_v,
            v_pred,
            pred_err,
            phi,
            tau,
            firing_rate_ema,
            excitability_bias,
            v_homeo,
            v_delta_homeo,
            atp_credit_ema,
            surprise_ema,
            sensory_trophic_ema,
            mitosis_cooldown,
            age_ticks,
        });
    }
    if counted_alive_cells != cell_alive_count {
        return Err(PersistError::InvalidTopology);
    }

    let cell_free_len = cr.read_u32()? as usize;
    if cell_free_len > cell_slots_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut cell_free_list = Vec::with_capacity(cell_free_len);
    let mut seen_free_cell = vec![false; cell_slots_len];
    for _ in 0..cell_free_len {
        let idx = cr.read_u32()?;
        let u = idx as usize;
        if u >= cell_slots_len || cell_slots[u].alive || seen_free_cell[u] {
            return Err(PersistError::InvalidTopology);
        }
        seen_free_cell[u] = true;
        cell_free_list.push(idx);
    }

    let active_len = cr.read_u32()? as usize;
    if active_len > cell_slots_len || active_len > 96 {
        return Err(PersistError::InvalidTopology);
    }
    let mut active_indices = Vec::with_capacity(active_len);
    for _ in 0..active_len {
        let idx = cr.read_u32()?;
        let u = idx as usize;
        if u >= cell_slots_len || !cell_slots[u].alive {
            return Err(PersistError::InvalidTopology);
        }
        active_indices.push(idx);
    }
    let (theta_phase, error_reversal_gate) = if version >= 4 {
        let tp = cr.read_f32()?;
        let erg = cr.read_f32()?;
        if !(0.0..=std::f32::consts::TAU + 1.0e-3).contains(&tp) || !(0.0..=1.0001).contains(&erg) {
            return Err(PersistError::InvalidTopology);
        }
        (
            tp.clamp(0.0, std::f32::consts::TAU - 1.0e-5),
            erg.clamp(0.0, 1.0),
        )
    } else {
        (0.0, 0.0)
    };
    if cr.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }

    let cells = CellArena {
        slots: cell_slots,
        free_list: cell_free_list,
        alive_count: cell_alive_count,
        active_indices,
        min_cells_floor,
        theta_phase,
        error_reversal_gate,
    };

    let mut pr = ByteReader::new(payloads[3]);
    let sensor_len = pr.read_u32()? as usize;
    let alive_sensor_count = pr.read_u32()? as usize;
    if sensor_len > config.max_pores || alive_sensor_count > sensor_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut sensor_pores = Vec::with_capacity(sensor_len);
    let mut counted_sensors = 0usize;
    for _ in 0..sensor_len {
        let alive = pr.read_bool()?;
        let channel = pr.read_u32()?;
        let delay = pr.read_u8()?;
        let target_cell = pr.read_u32()?;
        let delta_target_cell = pr.read_u32()?;
        let conductance = pr.read_f32()?;
        let trophic_ema = pr.read_f32()?;
        let last_val = pr.read_f32()?;
        let adapt_mean = pr.read_f32()?;
        let adapt_init = pr.read_bool()?;
        let mut history = [0.0_f32; 16];
        for dst in &mut history {
            *dst = pr.read_f32()?;
        }
        let hist_head = pr.read_u8()?;
        let age_ticks = pr.read_u32()?;

        if hist_head >= 16 {
            return Err(PersistError::InvalidTopology);
        }
        if alive {
            counted_sensors += 1;
            if !(1..=16).contains(&delay)
                || (target_cell as usize) >= cells.slots.len()
                || (delta_target_cell as usize) >= cells.slots.len()
                || conductance < 0.0
            {
                return Err(PersistError::InvalidTopology);
            }
        }
        sensor_pores.push(SensorPore {
            alive,
            channel,
            delay,
            target_cell,
            delta_target_cell,
            conductance,
            trophic_ema,
            last_val,
            adapt_mean,
            adapt_init,
            history,
            hist_head,
            age_ticks,
        });
    }
    if counted_sensors != alive_sensor_count {
        return Err(PersistError::InvalidTopology);
    }

    let sfree_len = pr.read_u32()? as usize;
    if sfree_len > sensor_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut sensor_free_list = Vec::with_capacity(sfree_len);
    for _ in 0..sfree_len {
        let idx = pr.read_u32()?;
        if (idx as usize) >= sensor_len || sensor_pores[idx as usize].alive {
            return Err(PersistError::InvalidTopology);
        }
        sensor_free_list.push(idx);
    }

    let motor_len = pr.read_u32()? as usize;
    let alive_motor_count = pr.read_u32()? as usize;
    if motor_len > config.max_pores
        || alive_motor_count > motor_len
        || alive_sensor_count.saturating_add(alive_motor_count) > config.max_pores
    {
        return Err(PersistError::InvalidTopology);
    }
    let mut motor_pores = Vec::with_capacity(motor_len);
    let mut counted_motors = 0usize;
    for _ in 0..motor_len {
        let alive = pr.read_bool()?;
        let actuator_channel = pr.read_u32()?;
        let conductance = pr.read_f32()?;
        let klinokinesis_gain = pr.read_f32()?;
        let motor_mastery = pr.read_f32()?;
        let jerk_ema = pr.read_f32()?;
        let last_out = pr.read_f32()?;
        if alive {
            counted_motors += 1;
            if conductance < 0.0 || !(0.0..=1.0001).contains(&motor_mastery) || jerk_ema < 0.0 {
                return Err(PersistError::InvalidTopology);
            }
        }
        motor_pores.push(MotorPore {
            alive,
            actuator_channel,
            conductance,
            klinokinesis_gain,
            motor_mastery,
            jerk_ema,
            last_out,
        });
    }
    if counted_motors != alive_motor_count {
        return Err(PersistError::InvalidTopology);
    }

    let mfree_len = pr.read_u32()? as usize;
    if mfree_len > motor_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut motor_free_list = Vec::with_capacity(mfree_len);
    for _ in 0..mfree_len {
        let idx = pr.read_u32()?;
        if (idx as usize) >= motor_len || motor_pores[idx as usize].alive {
            return Err(PersistError::InvalidTopology);
        }
        motor_free_list.push(idx);
    }
    if pr.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }

    let mut tr = ByteReader::new(payloads[2]);
    let tube_slots_len = tr.read_u32()? as usize;
    let tube_alive_count = tr.read_u32()? as usize;
    if tube_slots_len > config.max_tubes || tube_alive_count > tube_slots_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut tube_slots = Vec::with_capacity(tube_slots_len);
    let mut counted_tubes = 0usize;
    for _ in 0..tube_slots_len {
        let alive = tr.read_bool()?;
        let kind = TubeKind::from_u8(tr.read_u8()?).ok_or(PersistError::InvalidTopology)?;
        let pre = tr.read_u32()?;
        let post = tr.read_u32()?;
        let delay = tr.read_u8()?;
        let mut u = [0.0_f32; 4];
        for dst in &mut u {
            *dst = tr.read_f32()?;
        }
        let w_pred = tr.read_f32()?;
        let has_gate = tr.read_bool()?;
        let gate_raw = tr.read_u32()?;
        let context_gate_cell = if has_gate { Some(gate_raw) } else { None };
        let gate_invert = tr.read_bool()?;
        let contradiction_accum = tr.read_f32()?;
        let chain_score = tr.read_f32()?;
        let myelin_plasticity_accum = if version >= 4 { tr.read_f32()? } else { 0.0 };
        let last_upstream_cell = tr.read_u32()?;
        let usage_ema = tr.read_f32()?;
        let age_ticks = tr.read_u32()?;

        if alive {
            counted_tubes += 1;
            if !(1..=16).contains(&delay) {
                return Err(PersistError::InvalidTopology);
            }
            if (pre as usize) >= cells.slots.len() || !cells.slots[pre as usize].alive {
                return Err(PersistError::InvalidTopology);
            }
            match kind {
                TubeKind::CellToCell => {
                    if (post as usize) >= cells.slots.len() || !cells.slots[post as usize].alive {
                        return Err(PersistError::InvalidTopology);
                    }
                }
                TubeKind::CellToMotor => {
                    if (post as usize) >= motor_pores.len() || !motor_pores[post as usize].alive {
                        return Err(PersistError::InvalidTopology);
                    }
                }
            }
            if let Some(g) = context_gate_cell
                && (g as usize) >= cells.slots.len()
            {
                return Err(PersistError::InvalidTopology);
            }
        }
        tube_slots.push(Tube {
            alive,
            kind,
            pre,
            post,
            delay,
            u,
            w_pred,
            context_gate_cell,
            gate_invert,
            contradiction_accum,
            chain_score,
            myelin_plasticity_accum,
            last_upstream_cell,
            usage_ema,
            age_ticks,
        });
    }
    if counted_tubes != tube_alive_count {
        return Err(PersistError::InvalidTopology);
    }

    let tfree_len = tr.read_u32()? as usize;
    if tfree_len > tube_slots_len {
        return Err(PersistError::InvalidTopology);
    }
    let mut tube_free_list = Vec::with_capacity(tfree_len);
    let mut seen_free_tube = vec![false; tube_slots_len];
    for _ in 0..tfree_len {
        let idx = tr.read_u32()?;
        let u = idx as usize;
        if u >= tube_slots_len || tube_slots[u].alive || seen_free_tube[u] {
            return Err(PersistError::InvalidTopology);
        }
        seen_free_tube[u] = true;
        tube_free_list.push(idx);
    }

    let mut action_sensory_w = [[0.0_f32; PRED_BUCKETS]; TRACKED_ACTUATORS];
    for row in &mut action_sensory_w {
        for dst in row {
            *dst = tr.read_f32()?;
        }
    }
    let mut pore_action_w = [[0.0_f32; PRED_BUCKETS]; TRACKED_ACTUATORS];
    let mut tectal_map_w = [[0.0_f32; PRED_BUCKETS]; TRACKED_ACTUATORS];
    if version >= 4 {
        for row in &mut pore_action_w {
            for dst in row {
                *dst = tr.read_f32()?;
            }
        }
        for row in &mut tectal_map_w {
            for dst in row {
                *dst = tr.read_f32()?;
            }
        }
    }
    let mut sensory_self_w = [0.0_f32; PRED_BUCKETS];
    for dst in &mut sensory_self_w {
        *dst = tr.read_f32()?;
    }
    let mut pore_self_w = [0.0_f32; PRED_BUCKETS];
    let mut pore_mean_v = [0.0_f32; PRED_BUCKETS];
    let mut pore_scale_v = [0.05_f32; PRED_BUCKETS];
    let mut pore_agency = [0.0_f32; PRED_BUCKETS];
    let mut pore_dsq_ema = [1.0e-4_f32; PRED_BUCKETS];
    let mut pore_var_ema = [5.0e-3_f32; PRED_BUCKETS];
    let mut pore_bg_ild = [0.0_f32; PRED_BUCKETS];
    let mut social_loom_ema = 0.0_f32;
    let mut spatial_prox_ring = [0.0_f32; RING_LEN];
    let mut spatial_prox_cursor = 0u8;
    let mut echo_emit_w = [0.0_f32; TRACKED_ACTUATORS];
    let mut vocal_artic_w = [0.0_f32; TRACKED_ACTUATORS];
    let mut somatic_limb_w = [0.0_f32; TRACKED_ACTUATORS];
    let mut peer_coch_fast = 0.0_f32;
    let mut peer_coch_slow = 0.0_f32;
    let mut peer_echo_fast = 0.0_f32;
    let mut peer_echo_slow = 0.0_f32;
    let mut peer_disp_coch = 0.0_f32;
    let mut peer_disp_echo = 0.0_f32;
    let mut peer_vocal_hold = 0.0_f32;
    let mut peer_coch_age = 255u8;
    let mut peer_echo_age = 255u8;
    let mut peer_bearing_ema = [0.0_f32; TRACKED_ACTUATORS];
    if version >= 4 {
        for dst in &mut pore_self_w {
            *dst = tr.read_f32()?.clamp(-2.0, 2.0);
        }
        for dst in &mut pore_mean_v {
            *dst = tr.read_f32()?;
        }
        for dst in &mut pore_scale_v {
            *dst = tr.read_f32()?.clamp(0.005, 4.0);
        }
        for dst in &mut pore_agency {
            *dst = tr.read_f32()?.clamp(-1.0, 1.0);
        }
        for dst in &mut pore_dsq_ema {
            *dst = tr.read_f32()?.clamp(1.0e-6, 4.0);
        }
        for dst in &mut pore_var_ema {
            *dst = tr.read_f32()?.clamp(1.0e-5, 4.0);
        }
        for dst in &mut pore_bg_ild {
            *dst = tr.read_f32()?.clamp(-1.0, 1.0);
        }
        social_loom_ema = tr.read_f32()?.clamp(-1.0, 1.0);
        for dst in &mut spatial_prox_ring {
            *dst = tr.read_f32()?.clamp(0.0, 4.0);
        }
        spatial_prox_cursor = tr.read_u8()?;
        if (spatial_prox_cursor as usize) >= RING_LEN {
            return Err(PersistError::InvalidTopology);
        }
        for dst in &mut echo_emit_w {
            *dst = tr.read_f32()?.clamp(-1.0, 2.0);
        }
        for dst in &mut vocal_artic_w {
            *dst = tr.read_f32()?.clamp(-1.0, 2.0);
        }
        for dst in &mut somatic_limb_w {
            *dst = tr.read_f32()?.clamp(-1.0, 2.0);
        }
        peer_coch_fast = tr.read_f32()?.clamp(-1.0, 2.0);
        peer_coch_slow = tr.read_f32()?.clamp(-1.0, 2.0);
        peer_echo_fast = tr.read_f32()?.clamp(0.0, 2.0);
        peer_echo_slow = tr.read_f32()?.clamp(0.0, 2.0);
        peer_disp_coch = tr.read_f32()?.clamp(-16.0, 16.0);
        peer_disp_echo = tr.read_f32()?.clamp(-16.0, 16.0);
        peer_vocal_hold = tr.read_f32()?.clamp(0.0, 1.0);
        peer_coch_age = tr.read_u8()?;
        peer_echo_age = tr.read_u8()?;
        for dst in &mut peer_bearing_ema {
            *dst = tr.read_f32()?.clamp(-1.0, 1.0);
        }
    }
    let mut act_mean = [0.0_f32; TRACKED_ACTUATORS];
    for dst in &mut act_mean {
        *dst = tr.read_f32()?;
    }
    let mean_interventional_agency = tr.read_f32()?;
    let striatal = if version >= 5 {
        let num_ctx = tr.read_u8()?;
        let active_ctx = tr.read_u8()?;
        let impact_refrac = tr.read_u8()?;
        if num_ctx == 0 || (num_ctx as usize) > OFC_CONTEXTS || active_ctx >= num_ctx {
            return Err(PersistError::InvalidTopology);
        }
        let mut ctx_has_cons = [0u8; OFC_CONTEXTS];
        for dst in &mut ctx_has_cons {
            let v = tr.read_u8()?;
            if v > 1 {
                return Err(PersistError::InvalidTopology);
            }
            *dst = v;
        }
        let mut ctx_ticks = [0u32; OFC_CONTEXTS];
        for dst in &mut ctx_ticks {
            *dst = tr.read_u32()?;
        }
        let mut ctx_r_peak = [0.0_f32; OFC_CONTEXTS];
        for dst in &mut ctx_r_peak {
            *dst = tr.read_f32()?;
        }
        let mut r_bar_ctx = [0.0_f32; OFC_CONTEXTS];
        for dst in &mut r_bar_ctx {
            *dst = tr.read_f32()?;
        }
        let mut pore_ch = [u32::MAX; STRIATAL_PORES];
        for dst in &mut pore_ch {
            *dst = tr.read_u32()?;
        }
        let mut pore_age = [0u32; STRIATAL_PORES];
        for dst in &mut pore_age {
            *dst = tr.read_u32()?;
        }
        let mut pore_mean = [0.0_f32; STRIATAL_PORES];
        for dst in &mut pore_mean {
            *dst = tr.read_f32()?;
        }
        let mut pore_var_raw = [1.0e-4_f32; STRIATAL_PORES];
        for dst in &mut pore_var_raw {
            *dst = tr.read_f32()?;
        }
        let mut pore_var_exaff = [1.0e-4_f32; STRIATAL_PORES];
        for dst in &mut pore_var_exaff {
            *dst = tr.read_f32()?;
        }
        let mut canc_g = vec![0.0_f32; STRIATAL_PORES * STRIATAL_TAP_DIM];
        for dst in &mut canc_g {
            *dst = tr.read_f32()?;
        }
        let mut w_birth = vec![0.0_f32; STRIATAL_W_LEN];
        for dst in &mut w_birth {
            *dst = tr.read_f32()?;
        }
        let mut w_ctx = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_W_LEN];
        for dst in &mut w_ctx {
            *dst = tr.read_f32()?;
        }
        let mut w_cons = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_W_LEN];
        for dst in &mut w_cons {
            *dst = tr.read_f32()?;
        }
        let mut v_ctx = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_FEATS];
        for dst in &mut v_ctx {
            *dst = tr.read_f32()?;
        }
        let mut v_cons = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_FEATS];
        for dst in &mut v_cons {
            *dst = tr.read_f32()?;
        }
        let mut q_ctx = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_CUES];
        for dst in &mut q_ctx {
            *dst = tr.read_f32()?;
        }
        let mut q_cons = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_CUES];
        for dst in &mut q_cons {
            *dst = tr.read_f32()?;
        }
        let mut fwd_b = [0.0_f32; STRIATAL_CUES];
        for dst in &mut fwd_b {
            *dst = tr.read_f32()?;
        }
        let mut fwd_alpha = [0.0_f32; STRIATAL_CUES];
        for dst in &mut fwd_alpha {
            *dst = tr.read_f32()?;
        }
        let mut fwd_j = vec![0.0_f32; STRIATAL_FWD_LEN];
        for dst in &mut fwd_j {
            *dst = tr.read_f32()?;
        }
        let mut fwd_j_ctx = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_FWD_LEN];
        for dst in &mut fwd_j_ctx {
            *dst = tr.read_f32()?;
        }
        let mut fwd_j_cons = vec![0.0_f32; OFC_CONTEXTS * STRIATAL_FWD_LEN];
        for dst in &mut fwd_j_cons {
            *dst = tr.read_f32()?;
        }
        let mut fwd_var_null = [1.0e-4_f32; STRIATAL_CUES];
        for dst in &mut fwd_var_null {
            *dst = tr.read_f32()?;
        }
        let mut fwd_var_act = [1.0e-4_f32; STRIATAL_CUES];
        for dst in &mut fwd_var_act {
            *dst = tr.read_f32()?;
        }
        let mut fwd_num = [0.0_f32; OFC_CONTEXTS];
        for dst in &mut fwd_num {
            *dst = tr.read_f32()?;
        }
        let mut fwd_den = [0.0_f32; OFC_CONTEXTS];
        for dst in &mut fwd_den {
            *dst = tr.read_f32()?;
        }
        let mut s_act_mean = [0.0_f32; STRIATAL_ACTUATORS];
        for dst in &mut s_act_mean {
            *dst = tr.read_f32()?;
        }
        let ctx_cooldown = tr.read_u32()?;
        let ctx_out_events = tr.read_u32()?;
        let mut ctx_out_err = [0.0_f32; OFC_CONTEXTS];
        for dst in &mut ctx_out_err {
            *dst = tr.read_f32()?;
        }
        let r_fast_ema = tr.read_f32()?;
        let r_med200 = tr.read_f32()?;
        let valence_flip_ema = tr.read_f32()?;
        let mut da_ring = [0.0_f32; STRIATAL_TAP_DIM];
        for dst in &mut da_ring {
            *dst = tr.read_f32()?;
        }
        let mut elig_actor = vec![0.0_f32; STRIATAL_W_LEN];
        for dst in &mut elig_actor {
            *dst = tr.read_f32()?;
        }
        let mut elig_critic = [0.0_f32; STRIATAL_FEATS];
        for dst in &mut elig_critic {
            *dst = tr.read_f32()?;
        }
        let mut feat = [0.0_f32; STRIATAL_FEATS];
        for dst in &mut feat {
            *dst = tr.read_f32()?;
        }
        let mut prev_critic_feat = [0.0_f32; STRIATAL_FEATS];
        for dst in &mut prev_critic_feat {
            *dst = tr.read_f32()?;
        }
        let mut prev_feat = [0.0_f32; STRIATAL_CUES];
        for dst in &mut prev_feat {
            *dst = tr.read_f32()?;
        }
        let mut prev_df = [0.0_f32; STRIATAL_CUES];
        for dst in &mut prev_df {
            *dst = tr.read_f32()?;
        }
        let mut prev_act = [0.0_f32; STRIATAL_ACTUATORS];
        for dst in &mut prev_act {
            *dst = tr.read_f32()?;
        }
        let mut noise = [0.0_f32; STRIATAL_ACTUATORS];
        for dst in &mut noise {
            *dst = tr.read_f32()?;
        }
        let v_prev = tr.read_f32()?;
        let has_prev = tr.read_bool()?;
        let s_tick = tr.read_u64()?;
        let rs0 = tr.read_u64()?;
        let rs1 = tr.read_u64()?;
        if (rs0 | rs1) == 0 {
            return Err(PersistError::InvalidTopology);
        }
        StriatalOfc {
            num_ctx,
            active_ctx,
            impact_refrac,
            ctx_has_cons,
            ctx_ticks,
            ctx_r_peak,
            r_bar_ctx,
            pore_ch,
            pore_age,
            pore_mean,
            pore_var_raw,
            pore_var_exaff,
            canc_g,
            w_birth,
            w_ctx,
            w_cons,
            v_ctx,
            v_cons,
            q_ctx,
            q_cons,
            fwd_b,
            fwd_alpha,
            fwd_j,
            fwd_j_ctx,
            fwd_j_cons,
            fwd_var_null,
            fwd_var_act,
            fwd_num,
            fwd_den,
            act_mean: s_act_mean,
            ctx_cooldown,
            ctx_out_events,
            ctx_out_err,
            r_fast_ema,
            r_med200,
            valence_flip_ema,
            da_ring,
            elig_actor,
            elig_critic,
            feat,
            prev_critic_feat,
            prev_feat,
            prev_df,
            prev_act,
            noise,
            v_prev,
            has_prev,
            tick: s_tick,
            rng: Rng64 { s0: rs0, s1: rs1 },
        }
    } else {
        StriatalOfc::from_seed(seed)
    };
    if tr.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }

    let mut tubes = TubeGraph {
        slots: tube_slots,
        free_list: tube_free_list,
        alive_count: tube_alive_count,
        outgoing_by_cell: Vec::new(),
        action_sensory_w,
        pore_action_w,
        tectal_map_w,
        sensory_self_w,
        pore_self_w,
        pore_mean_v,
        pore_scale_v,
        pore_agency,
        pore_dsq_ema,
        pore_var_ema,
        pore_bg_ild,
        social_loom_ema,
        spatial_prox_ring,
        spatial_prox_cursor,
        echo_emit_w,
        vocal_artic_w,
        somatic_limb_w,
        peer_coch_fast,
        peer_coch_slow,
        peer_echo_fast,
        peer_echo_slow,
        peer_disp_coch,
        peer_disp_echo,
        peer_vocal_hold,
        peer_coch_age,
        peer_echo_age,
        peer_bearing_ema,
        act_mean,
        mean_interventional_agency,
        striatal,
    };
    tubes.rebuild_adjacency(cells.slots.len());

    let mut rr = ByteReader::new(payloads[4]);
    let ring_head = rr.read_u32()? as usize;
    if ring_head >= RING_LEN {
        return Err(PersistError::InvalidTopology);
    }
    let mut entries: [EfferenceEntry; RING_LEN] =
        std::array::from_fn(|_| EfferenceEntry::default());
    for entry in &mut entries {
        let valid = rr.read_bool()?;
        let e_tick = rr.read_u64()?;
        let committed = rr.read_bool()?;
        let motor_norm = rr.read_f32()?;
        let mut motor_out = [0.0_f32; TRACKED_ACTUATORS];
        for dst in &mut motor_out {
            *dst = rr.read_f32()?;
        }
        let somatic_value_est = rr.read_f32()?;
        let raw_valence_at_commit = rr.read_f32()?;

        let act_cells_len = rr.read_u32()? as usize;
        if act_cells_len > 96 {
            return Err(PersistError::InvalidTopology);
        }
        let mut active_cells = Vec::with_capacity(act_cells_len);
        for _ in 0..act_cells_len {
            let ci = rr.read_u32()?;
            let vr = rr.read_f32()?;
            let dv = rr.read_f32()?;
            let pa = if version >= 4 { rr.read_f32()? } else { 1.0 };
            if valid && (ci as usize) >= cells.slots.len() {
                return Err(PersistError::InvalidTopology);
            }
            active_cells.push((ci, vr, dv, pa));
        }

        let pw_len = rr.read_u32()? as usize;
        if pw_len > 64 {
            return Err(PersistError::InvalidTopology);
        }
        let mut pred_with_action = Vec::with_capacity(pw_len);
        for _ in 0..pw_len {
            let ci = rr.read_u32()?;
            let pv = rr.read_f32()?;
            if valid && (ci as usize) >= cells.slots.len() {
                return Err(PersistError::InvalidTopology);
            }
            pred_with_action.push((ci, pv));
        }

        let pn_len = rr.read_u32()? as usize;
        if pn_len != pw_len {
            return Err(PersistError::InvalidTopology);
        }
        let mut pred_null_action = Vec::with_capacity(pn_len);
        for _ in 0..pn_len {
            let ci = rr.read_u32()?;
            let pv = rr.read_f32()?;
            if valid && (ci as usize) >= cells.slots.len() {
                return Err(PersistError::InvalidTopology);
            }
            pred_null_action.push((ci, pv));
        }

        *entry = EfferenceEntry {
            valid,
            tick: e_tick,
            committed,
            motor_norm,
            motor_out,
            active_cells,
            pred_with_action,
            pred_null_action,
            somatic_value_est,
            raw_valence_at_commit,
        };
    }
    if rr.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }
    let ring = EfferenceRing {
        entries,
        head: ring_head,
    };

    let mut sr = ByteReader::new(payloads[5]);
    let scout_cohort_idx = sr.read_u32()?;
    let passive_health_ema = sr.read_f32()?;
    let motor_env_fast = sr.read_f32()?;
    let motor_env_mean = sr.read_f32()?;
    let motor_env_var = sr.read_f32()?;
    let stats_len = sr.read_u32()?;
    let stats_len = stats_len as usize;
    if stats_len > MAX_SCOUT_CHANNELS {
        return Err(PersistError::OversizedInput);
    }
    let mut scout_stats = Vec::with_capacity(stats_len);
    for _ in 0..stats_len {
        let initialized = sr.read_bool()?;
        let mean = sr.read_f32()?;
        let var_ema = sr.read_f32()?;
        let diff_sq_ema = sr.read_f32()?;
        let prev_val = sr.read_f32()?;
        let ar1_w = sr.read_f32()?;
        let ar1_err_ema = sr.read_f32()?;
        let prev_dx = sr.read_f32()?;
        let cov_dx_ema = sr.read_f32()?;
        let fourth_ema = sr.read_f32()?;
        let motor_cov_ema = sr.read_f32()?;
        scout_stats.push(ChannelScoutStat {
            initialized,
            mean,
            var_ema,
            diff_sq_ema,
            prev_val,
            ar1_w,
            ar1_err_ema,
            prev_dx,
            cov_dx_ema,
            fourth_ema,
            motor_cov_ema,
        });
    }
    if sr.remaining() != 0 {
        return Err(PersistError::InvalidTopology);
    }

    let membrane = PoreMembrane {
        sensor_pores,
        sensor_free_list,
        alive_sensor_count,
        motor_pores,
        motor_free_list,
        alive_motor_count,
        scout_stats,
        scout_cohort_idx,
        passive_health_ema,
        motor_env_fast,
        motor_env_mean,
        motor_env_var,
    };

    let decoded = Alpha11Organism {
        seed,
        config,
        mode,
        poisoned: false,
        rng,
        explore_rng,
        tick,
        body,
        cells,
        tubes,
        membrane,
        ring,
        committed_motor_pulses,
        subthreshold_rehearsals,
        mitosis_births,
        contradiction_splits,
        myelinated_shortcuts,
        atrophied_structures,
        last_motor_energy,
    };

    if decoded.used_state_bytes() > decoded.config.max_state_bytes
        || decoded.used_state_bytes() > caller_max_state_bytes
    {
        return Err(PersistError::OversizedInput);
    }

    Ok(decoded)
}

fn read_bounded_file(path: &Path, max_bytes: usize) -> Result<Vec<u8>, PersistError> {
    let meta = fs::metadata(path).map_err(|_| PersistError::IoFailure)?;
    let file_len = usize::try_from(meta.len()).map_err(|_| PersistError::OversizedInput)?;
    if file_len > max_bytes {
        return Err(PersistError::OversizedInput);
    }
    let f = File::open(path).map_err(|_| PersistError::IoFailure)?;
    let mut buf = Vec::with_capacity(file_len);
    f.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|_| PersistError::IoFailure)?;
    if buf.len() > max_bytes {
        return Err(PersistError::OversizedInput);
    }
    Ok(buf)
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<(), PersistError> {
    let mut f = File::create(path).map_err(|_| PersistError::IoFailure)?;
    f.write_all(bytes).map_err(|_| PersistError::IoFailure)?;
    f.sync_all().map_err(|_| PersistError::IoFailure)?;
    Ok(())
}

pub fn save_transactional_files(
    org: &Alpha11Organism,
    primary: &Path,
    prev: &Path,
    tmp: &Path,
) -> Result<(), PersistError> {
    if org.poisoned {
        return Err(PersistError::PoisonedAfterCorruption);
    }
    let bytes = encode_organism(org)?;
    // Fail closed if the freshly encoded generation does not shadow-decode cleanly.
    shadow_decode_organism(&bytes, org.config.max_state_bytes)?;

    for p in [tmp, primary, prev] {
        if let Some(parent) = p.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|_| PersistError::IoFailure)?;
        }
    }

    // Rotate `primary -> prev` ONLY after `primary` passes a bounded read + full CRC32C shadow decode.
    // A corrupted `primary` is never allowed to overwrite the last known-good `prev` generation.
    if primary.exists() {
        let cap = org
            .config
            .max_state_bytes
            .saturating_add(HEADER_OVERHEAD_MARGIN);
        if let Ok(existing) = read_bounded_file(primary, cap)
            && shadow_decode_organism(&existing, org.config.max_state_bytes).is_ok()
        {
            write_synced_file(tmp, &existing)?;
            fs::rename(tmp, prev).map_err(|_| PersistError::IoFailure)?;
        }
    }

    write_synced_file(tmp, &bytes)?;
    fs::rename(tmp, primary).map_err(|_| PersistError::IoFailure)?;
    if let Some(parent) = primary.parent()
        && !parent.as_os_str().is_empty()
        && let Ok(dir_file) = File::open(parent)
    {
        let _ = dir_file.sync_all();
    }
    Ok(())
}

pub fn load_transactional_files(
    org: &mut Alpha11Organism,
    primary: &Path,
    prev: &Path,
) -> Result<bool, PersistError> {
    if org.poisoned {
        return Err(PersistError::PoisonedAfterCorruption);
    }
    let cap = org
        .config
        .max_state_bytes
        .saturating_add(HEADER_OVERHEAD_MARGIN);

    let primary_err = match read_bounded_file(primary, cap) {
        Ok(bytes) => match shadow_decode_organism(&bytes, org.config.max_state_bytes) {
            Ok(decoded) => {
                *org = decoded;
                org.poisoned = false;
                return Ok(false);
            }
            Err(e) => e,
        },
        Err(e) => e,
    };

    match read_bounded_file(prev, cap) {
        Ok(prev_bytes) => match shadow_decode_organism(&prev_bytes, org.config.max_state_bytes) {
            Ok(decoded) => {
                *org = decoded;
                org.poisoned = false;
                Ok(true)
            }
            Err(_) => {
                org.poisoned = true;
                Err(primary_err)
            }
        },
        Err(_) => {
            org.poisoned = true;
            Err(primary_err)
        }
    }
}
