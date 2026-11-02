//! Alpha 11 core biophysical organism substrate, morphogenesis laws, and fail-closed persistence.

pub mod body;
pub mod causal_core;
pub mod cells;
pub mod crc32c;
pub mod persistence;
pub mod pores;
pub mod rng;
pub mod tubes;

use std::path::Path;

use a11_medium::{
    ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError,
};

use crate::body::BodyCore;
use crate::cells::CellArena;
use crate::persistence::{
    encode_organism, load_transactional_files, save_transactional_files, shadow_decode_organism,
};
use crate::pores::PoreMembrane;
use crate::rng::{ParamHasher, Rng64, sanitize, sanitize_clamp};
use crate::tubes::{
    EfferenceEntry, EfferenceRing, TRACKED_ACTUATORS, Tube, TubeGraph, TubeKind,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrganismConfig {
    pub initial_cells: usize,
    pub max_cells: usize,
    pub max_tubes: usize,
    pub max_pores: usize,
    pub max_state_bytes: usize,
    pub initial_atp: f32,
    pub max_atp: f32,
}

impl Default for OrganismConfig {
    fn default() -> Self {
        Self {
            initial_cells: 32,
            max_cells: 65_536,
            max_tubes: 524_288,
            max_pores: 512,
            max_state_bytes: 256 * 1024 * 1024,
            initial_atp: 12.0,
            max_atp: 100.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Alpha11Organism {
    pub(crate) seed: u64,
    pub(crate) config: OrganismConfig,
    pub(crate) mode: ControlMode,
    pub(crate) poisoned: bool,
    pub(crate) rng: Rng64,
    pub(crate) explore_rng: Rng64,
    pub(crate) tick: u64,
    pub(crate) body: BodyCore,
    pub(crate) cells: CellArena,
    pub(crate) tubes: TubeGraph,
    pub(crate) membrane: PoreMembrane,
    pub(crate) ring: EfferenceRing,
    pub(crate) committed_motor_pulses: u64,
    pub(crate) subthreshold_rehearsals: u64,
    pub(crate) mitosis_births: u64,
    pub(crate) contradiction_splits: u64,
    pub(crate) myelinated_shortcuts: u64,
    pub(crate) atrophied_structures: u64,
    pub(crate) last_motor_energy: f32,
}

impl Alpha11Organism {
    #[must_use]
    pub fn new(seed: u64, config: OrganismConfig) -> Self {
        let mut rng = Rng64::from_seed(seed);
        let explore_rng = Rng64::from_seed(seed ^ 0xE791_0AE5_5EED_CAFE);
        let body = BodyCore::new(config.initial_atp, config.max_atp);
        let cells = CellArena::new(config.initial_cells, config.max_cells, &mut rng);
        let membrane = PoreMembrane::new(cells.alive_count, config.max_pores, &mut rng);
        let tubes = TubeGraph::new(
            cells.alive_count,
            membrane.alive_motor_count,
            config.max_tubes,
            &mut rng,
        );
        Self {
            seed,
            config,
            mode: ControlMode::Natural,
            poisoned: false,
            rng,
            explore_rng,
            tick: 0,
            body,
            cells,
            tubes,
            membrane,
            ring: EfferenceRing::default(),
            committed_motor_pulses: 0,
            subthreshold_rehearsals: 0,
            mitosis_births: 0,
            contradiction_splits: 0,
            myelinated_shortcuts: 0,
            atrophied_structures: 0,
            last_motor_energy: 0.0,
        }
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    #[must_use]
    pub fn config(&self) -> OrganismConfig {
        self.config
    }

    #[must_use]
    pub fn used_state_bytes(&self) -> usize {
        BodyCore::LOGICAL_BYTES
            + self.cells.logical_bytes()
            + self.tubes.logical_bytes()
            + self.membrane.logical_bytes()
            + self.ring.logical_bytes()
            + 96
    }

    #[must_use]
    pub fn motor_maturation_by_channel(&self) -> Vec<(u32, f32, f32)> {
        let mut out = Vec::with_capacity(self.membrane.alive_motor_count);
        for mp in &self.membrane.motor_pores {
            if mp.alive {
                out.push((mp.actuator_channel, mp.motor_mastery, mp.jerk_ema));
            }
        }
        out
    }

    #[must_use]
    pub fn mean_motor_mastery(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut cnt = 0usize;
        for mp in &self.membrane.motor_pores {
            if mp.alive {
                sum += mp.motor_mastery;
                cnt += 1;
            }
        }
        if cnt > 0 {
            sanitize(sum / (cnt as f32))
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn mean_motor_jerk(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut cnt = 0usize;
        for mp in &self.membrane.motor_pores {
            if mp.alive {
                sum += mp.jerk_ema;
                cnt += 1;
            }
        }
        if cnt > 0 {
            sanitize(sum / (cnt as f32))
        } else {
            0.0
        }
    }

    pub fn save_transactional(
        &self,
        primary: &Path,
        prev: &Path,
        tmp: &Path,
    ) -> Result<(), PersistError> {
        save_transactional_files(self, primary, prev, tmp)
    }

    pub fn load_transactional(
        &mut self,
        primary: &Path,
        prev: &Path,
    ) -> Result<bool, PersistError> {
        load_transactional_files(self, primary, prev)
    }

    fn wire_new_motor_pores(&mut self, prev_motor_slots: usize) {
        let new_motor_slots = self.membrane.motor_pores.len();
        if new_motor_slots <= prev_motor_slots {
            return;
        }
        for m_slot in prev_motor_slots..new_motor_slots {
            if !self.membrane.motor_pores[m_slot].alive {
                continue;
            }
            for c_idx in 0..self.cells.slots.len().min(64) {
                if !self.cells.slots[c_idx].alive {
                    continue;
                }
                let init_u = 0.012 * self.rng.next_f32_signed();
                let _ = self.tubes.add_tube(
                    Tube::newborn(TubeKind::CellToMotor, c_idx as u32, m_slot as u32, 1, init_u),
                    self.config.max_tubes,
                );
            }
        }
    }

    fn wire_mitosis_family(&mut self, child_idx: u32, parent_idx: u32) {
        self.tubes.ensure_cell_capacity(self.cells.slots.len());
        let mut up_tube = Tube::newborn(TubeKind::CellToCell, child_idx, parent_idx, 1, 0.30);
        up_tube.w_pred = 0.20;
        let _ = self.tubes.add_tube(up_tube, self.config.max_tubes);

        let conv_delays = [1_u8, 3, 6];
        let mut co_wired = 0usize;
        for idx_pos in 0..self.cells.active_indices.len() {
            let co_idx = self.cells.active_indices[idx_pos];
            if co_idx == child_idx || co_idx == parent_idx {
                continue;
            }
            if self
                .tubes
                .outgoing_by_cell
                .get(co_idx as usize)
                .map_or(0, Vec::len)
                >= 48
            {
                continue;
            }
            let delay = conv_delays[co_wired % conv_delays.len()];
            let mut conv_tube =
                Tube::newborn(TubeKind::CellToCell, co_idx, parent_idx, delay, 0.22);
            conv_tube.w_pred = 0.14;
            let _ = self.tubes.add_tube(conv_tube, self.config.max_tubes);
            co_wired += 1;
            if co_wired >= 2 {
                break;
            }
        }

        let mut down_tube = Tube::newborn(TubeKind::CellToCell, parent_idx, child_idx, 1, -0.14);
        down_tube.w_pred = 0.32;
        let _ = self.tubes.add_tube(down_tube, self.config.max_tubes);

        for m_slot in 0..self.membrane.motor_pores.len().min(TRACKED_ACTUATORS) {
            if !self.membrane.motor_pores[m_slot].alive {
                continue;
            }
            let init_u = 0.006 * self.rng.next_f32_signed();
            let _ = self.tubes.add_tube(
                Tube::newborn(TubeKind::CellToMotor, parent_idx, m_slot as u32, 1, init_u),
                self.config.max_tubes,
            );
        }
    }
}

impl OrganismEngine for Alpha11Organism {
    fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError> {
        if self.poisoned {
            return Err(PersistError::PoisonedAfterCorruption);
        }

        if self.mode == ControlMode::Natural {
            let prev_motors = self.membrane.motor_pores.len();
            self.membrane.ensure_actuators_bound(
                view.num_actuators,
                self.config.max_pores,
                self.mode,
            );
            self.wire_new_motor_pores(prev_motors);
        }

        self.membrane.prepare_scout_cycle(
            view.ambient_in.len(),
            view.num_actuators,
            self.last_motor_energy,
            self.mode,
            &mut self.rng,
        );

        let n_cells = self.cells.slots.len();
        let mut ext_drive = vec![0.0_f32; n_cells];
        let mut pred_drive = vec![0.0_f32; n_cells];
        let mut sensory_trophic = vec![0.0_f32; n_cells];

        self.membrane.absorb_and_scout(
            view.ambient_in,
            &mut ext_drive,
            &mut sensory_trophic,
            self.mode,
        );
        self.tubes.accumulate_recurrent_and_predictive(
            &self.ring,
            &self.cells.slots,
            &mut ext_drive,
            &mut pred_drive,
            &mut sensory_trophic,
            self.mode,
        );

        if self.body.is_resting()
            && self.mode == ControlMode::Natural
            && let Some(replay_entry) = self.ring.at_lag(((self.tick as usize) % 8) + 1)
        {
            for &(c_idx, v_past, _) in &replay_entry.active_cells {
                if let Some(d) = ext_drive.get_mut(c_idx as usize) {
                    *d = sanitize(*d + 0.16 * v_past);
                }
            }
        }

        let raw_valence_now = sanitize(
            2.6 * sanitize(view.metabolic_influx).max(0.0)
                - 3.2 * sanitize(view.tissue_stress).max(0.0),
        );
        let somatic_drop = (self.body.baseline_valence_ema - raw_valence_now - 0.02).max(0.0)
            + (self.body.prev_raw_valence - raw_valence_now - 0.03).max(0.0);
        let ne_network_reset = sanitize_clamp(4.0 * somatic_drop, 0.0, 0.88);
        self.cells.integrate_step(
            &ext_drive,
            &pred_drive,
            &sensory_trophic,
            ne_network_reset,
            self.mode,
        );

        self.tubes
            .evaluate_interventional_agency_and_update_predictors(
                &self.ring,
                &self.cells,
                &mut self.membrane,
                self.mode,
            );

        let mean_surprise = self.cells.mean_surprise();
        let _somatic_advantage = self.body.update_metabolism(
            view.metabolic_influx,
            view.tissue_stress,
            self.cells.alive_count,
            self.tubes.alive_count,
            self.membrane.total_alive_pores(),
            self.last_motor_energy,
            mean_surprise,
            self.mode,
        );

        if self.mode == ControlMode::Natural {
            let (splits, shortcuts, pruned_tubes) =
                self.tubes.apply_delayed_credit_and_morphogenesis(
                    &self.ring,
                    &mut self.cells,
                    &mut self.membrane,
                    self.body.valence_scale_ema,
                    self.body.fast_valence,
                    self.config.max_tubes,
                    &mut self.rng,
                );
            self.contradiction_splits = self.contradiction_splits.saturating_add(splits);
            self.myelinated_shortcuts = self.myelinated_shortcuts.saturating_add(shortcuts);
            self.atrophied_structures = self.atrophied_structures.saturating_add(pruned_tubes);

            if self.body.is_resting() {
                self.tubes.consolidate_rest_step();
            }

            let state_budget_ok =
                self.used_state_bytes() < self.config.max_state_bytes.saturating_sub(16_384);

            if (self.tick & 3) == 0 && state_budget_ok {
                let primary_cap = self.cells.min_cells_floor.max(8);
                let alive_cells: Vec<u32> = self
                    .cells
                    .slots
                    .iter()
                    .enumerate()
                    .take(primary_cap)
                    .filter(|(_, c)| c.alive)
                    .map(|(i, _)| i as u32)
                    .collect();
                let atrophied_pores = self.membrane.morph_pores(
                    self.config.max_pores,
                    &alive_cells,
                    &mut self.rng,
                );
                self.atrophied_structures =
                    self.atrophied_structures.saturating_add(atrophied_pores);
            }

            if self.tick.rem_euclid(24) == 0
                && state_budget_ok
                && self.body.atp >= 3.2
                && let Some(child_idx) = self.cells.find_mitosis_candidate(self.config.max_cells)
                && let Some(parent_idx) = self.cells.spawn_mitosis_parent(
                    child_idx,
                    self.config.max_cells,
                    &mut self.rng,
                )
            {
                self.body.atp = sanitize((self.body.atp - 0.40).max(0.0));
                self.mitosis_births = self.mitosis_births.saturating_add(1);
                self.wire_mitosis_family(child_idx, parent_idx);
            }

            if (self.tick & 31) == 0 {
                let dead_cells = self.cells.prune_starved_cells();
                if !dead_cells.is_empty() {
                    let dead_tubes = self.tubes.prune_tubes_referencing_cells(&dead_cells);
                    self.atrophied_structures = self
                        .atrophied_structures
                        .saturating_add(dead_cells.len() as u64 + dead_tubes);
                }
            }
        }

        let outcome = self.tubes.rollout_and_select_pulse(
            view.num_actuators,
            &self.cells,
            &self.membrane,
            &mut self.body,
            &self.ring,
            self.mode,
            &mut self.explore_rng,
        );
        let pulse = outcome.pulse;
        let committed = outcome.committed;
        let rehearsals = outcome.rehearsals;
        let pred_with = outcome.pred_with;
        let pred_null = outcome.pred_null;
        self.subthreshold_rehearsals = self.subthreshold_rehearsals.saturating_add(rehearsals);
        if committed {
            self.committed_motor_pulses = self.committed_motor_pulses.saturating_add(1);
        }

        let mut energy_sum = 0.0_f32;
        for &v in &pulse.actuator_out {
            energy_sum += v * v;
        }
        self.last_motor_energy = if !pulse.actuator_out.is_empty() {
            sanitize_clamp(energy_sum / (pulse.actuator_out.len() as f32), 0.0, 4.0)
        } else {
            0.0
        };

        if committed {
            for mp in &mut self.membrane.motor_pores {
                if mp.alive {
                    let ch = mp.actuator_channel as usize;
                    let next_out = pulse.actuator_out.get(ch).copied().unwrap_or(0.0);
                    let amp = next_out.abs().max(mp.last_out.abs());
                    if amp > 0.03 {
                        let norm_dj =
                            ((next_out - mp.last_out).abs() / (0.15 + amp)).clamp(0.0, 2.0);
                        mp.jerk_ema =
                            sanitize_clamp(0.992 * mp.jerk_ema + 0.008 * norm_dj, 0.0, 2.0);
                    }
                    mp.last_out = next_out;
                }
            }
        }

        let mut motor_arr = [0.0_f32; TRACKED_ACTUATORS];
        let mut norm_sq = 0.0_f32;
        for (i, dst) in motor_arr.iter_mut().enumerate() {
            let v = pulse.actuator_out.get(i).copied().unwrap_or(0.0);
            *dst = v;
            norm_sq += v * v;
        }

        let mut active_snap = Vec::with_capacity(self.cells.active_indices.len());
        for &c_idx in &self.cells.active_indices {
            if let Some(c) = self.cells.slots.get(c_idx as usize)
                && c.alive
            {
                active_snap.push((c_idx, c.v_real, c.delta_v));
            }
        }

        self.ring.push(EfferenceEntry {
            valid: true,
            tick: self.tick,
            committed,
            motor_norm: sanitize(norm_sq.sqrt()),
            motor_out: motor_arr,
            active_cells: active_snap,
            pred_with_action: pred_with,
            pred_null_action: pred_null,
            somatic_value_est: self.cells.somatic_value_estimate(),
            raw_valence_at_commit: self.body.fast_valence,
        });

        self.tick = self.tick.saturating_add(1);
        Ok(pulse)
    }

    fn set_control_mode(&mut self, mode: ControlMode) {
        self.mode = mode;
    }

    fn control_mode(&self) -> ControlMode {
        self.mode
    }

    fn zero_learned_conductances(&mut self) {
        let prev_motors = self.membrane.motor_pores.len();
        let mut birth_rng = Rng64::from_seed(self.seed);
        self.cells =
            CellArena::new(self.config.initial_cells, self.config.max_cells, &mut birth_rng);
        self.membrane =
            PoreMembrane::new(self.cells.alive_count, self.config.max_pores, &mut birth_rng);
        self.tubes = TubeGraph::new(
            self.cells.alive_count,
            self.membrane.alive_motor_count,
            self.config.max_tubes,
            &mut birth_rng,
        );
        if prev_motors > 0 {
            let before = self.membrane.motor_pores.len();
            self.membrane.ensure_actuators_bound(
                prev_motors,
                self.config.max_pores,
                ControlMode::Natural,
            );
            self.wire_new_motor_pores(before);
        }
        self.rng = birth_rng;
        self.explore_rng = Rng64::from_seed(self.seed ^ 0xE791_0AE5_5EED_CAFE);
        self.ring = EfferenceRing::default();
        self.committed_motor_pulses = 0;
        self.subthreshold_rehearsals = 0;
        self.mitosis_births = 0;
        self.contradiction_splits = 0;
        self.myelinated_shortcuts = 0;
        self.atrophied_structures = 0;
        self.last_motor_energy = 0.0;
        self.body.zero_learned_state();
        self.cells.zero_learned_conductances();
        self.tubes.zero_learned_conductances();
        self.membrane.zero_learned_conductances();
    }

    fn learned_parameter_hash(&self) -> u64 {
        let mut hasher = ParamHasher::new(self.seed);
        self.body.hash_learned(&mut hasher);
        self.cells.hash_learned(&mut hasher);
        self.tubes.hash_learned(&mut hasher);
        self.membrane.hash_learned(&mut hasher);
        hasher.finish()
    }

    fn snapshot(&self) -> OrganismSnapshot {
        let (mean_tau, max_tau, tau_octave_count) = self.cells.tau_stats();
        let (stiff_highway_tubes, inhibitory_tubes, active_echo_delay_tubes) =
            self.tubes.topology_counts(&self.membrane);

        let mut sensor_pore_channels = Vec::with_capacity(self.membrane.alive_sensor_count);
        let mut sensor_pore_conductances = Vec::with_capacity(self.membrane.alive_sensor_count);
        let mut sensor_pore_delays = Vec::with_capacity(self.membrane.alive_sensor_count);
        for sp in &self.membrane.sensor_pores {
            if sp.alive {
                sensor_pore_channels.push(sp.channel);
                sensor_pore_conductances.push(sp.conductance);
                sensor_pore_delays.push(sp.delay);
            }
        }

        let mut motor_pore_channels = Vec::with_capacity(self.membrane.alive_motor_count);
        let mut motor_pore_conductances = Vec::with_capacity(self.membrane.alive_motor_count);
        for mp in &self.membrane.motor_pores {
            if mp.alive {
                motor_pore_channels.push(mp.actuator_channel);
                motor_pore_conductances.push(mp.conductance);
            }
        }

        OrganismSnapshot {
            tick: self.tick,
            atp: self.body.atp,
            adenosine: self.body.adenosine,
            temperature: self.body.temperature,
            motor_clutch_threshold: self.body.motor_clutch_threshold,
            cell_count: self.cells.alive_count,
            tube_count: self.tubes.alive_count,
            sensor_pore_count: self.membrane.alive_sensor_count,
            motor_pore_count: self.membrane.alive_motor_count,
            active_cells_last_tick: self.cells.active_indices.len(),
            stiff_highway_tubes,
            inhibitory_tubes,
            active_echo_delay_tubes,
            mean_tau,
            max_tau,
            tau_octave_count,
            mean_interventional_agency: self.tubes.mean_interventional_agency,
            mean_prediction_error: self.cells.mean_prediction_error(),
            committed_motor_pulses: self.committed_motor_pulses,
            subthreshold_rehearsals: self.subthreshold_rehearsals,
            rest_consolidation_ticks: self.body.rest_consolidation_ticks,
            mitosis_births: self.mitosis_births,
            contradiction_splits: self.contradiction_splits,
            myelinated_shortcuts: self.myelinated_shortcuts,
            atrophied_structures: self.atrophied_structures,
            sensor_pore_channels,
            sensor_pore_conductances,
            sensor_pore_delays,
            motor_pore_channels,
            motor_pore_conductances,
            used_state_bytes: self.used_state_bytes(),
        }
    }

    fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError> {
        encode_organism(self)
    }

    fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError> {
        if self.poisoned {
            return Err(PersistError::PoisonedAfterCorruption);
        }
        match shadow_decode_organism(bytes, self.config.max_state_bytes) {
            Ok(decoded) => {
                *self = decoded;
                Ok(())
            }
            Err(err) => {
                self.poisoned = true;
                Err(err)
            }
        }
    }

    fn verify_checkpoint(&self, bytes: &[u8]) -> Result<(), PersistError> {
        shadow_decode_organism(bytes, self.config.max_state_bytes).map(|_| ())
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::crc32c::crc32c;

    fn rechecksum_checkpoint(bytes: &mut [u8]) {
        let payload_end = bytes.len() - 4;
        let mut pos = 24usize;
        while pos < payload_end {
            let s_len = u32::from_le_bytes([
                bytes[pos + 4],
                bytes[pos + 5],
                bytes[pos + 6],
                bytes[pos + 7],
            ]) as usize;
            let data_end = pos + 8 + s_len;
            let scrc = crc32c(&bytes[pos..data_end]);
            bytes[data_end..data_end + 4].copy_from_slice(&scrc.to_le_bytes());
            pos = data_end + 4;
        }
        let fcrc = crc32c(&bytes[..payload_end]);
        bytes[payload_end..].copy_from_slice(&fcrc.to_le_bytes());
    }

    #[test]
    fn zero_double_quote_literals_in_all_non_test_source_lines() {
        let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let entries = fs::read_dir(&src_dir).expect("read src dir");
        let mut checked_files = 0usize;
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let raw = fs::read_to_string(&path).expect("read rs file");
            let non_test_part = raw.split("#[cfg(test)]").next().unwrap_or("");
            for (line_no, line) in non_test_part.lines().enumerate() {
                assert!(
                    !line.contains('"'),
                    "Found forbidden double-quote in {:?} at line {}: {}",
                    path,
                    line_no + 1,
                    line
                );
            }
            checked_files += 1;
        }
        assert!(checked_files >= 7);
    }

    #[test]
    fn crc32c_matches_rfc3720_standard_vector() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn frozen_plasticity_hash_is_strictly_invariant_over_500_steps() {
        let mut rng = Rng64::from_seed(42);
        let cfg = OrganismConfig::default();
        let mut frozen = Alpha11Organism::new(777, cfg);
        frozen.set_control_mode(ControlMode::FrozenPlasticity);
        let mut natural = Alpha11Organism::new(777, cfg);
        natural.set_control_mode(ControlMode::Natural);

        let h_frozen_before = frozen.learned_parameter_hash();
        let h_natural_before = natural.learned_parameter_hash();
        assert_eq!(h_frozen_before, h_natural_before);

        let mut ambient = vec![0.0_f32; 128];
        for t in 0..500u64 {
            let phase = (t as f32) * 0.11;
            for (c, slot) in ambient.iter_mut().enumerate() {
                *slot = if c < 32 {
                    (phase + (c as f32) * 0.08).sin() * 0.8
                } else if c < 64 {
                    0.0
                } else {
                    rng.next_f32_signed()
                };
            }
            let view = FieldView {
                ambient_in: &ambient,
                metabolic_influx: if t % 17 == 0 { 0.08 } else { 0.0 },
                tissue_stress: if t % 53 == 0 { 0.06 } else { 0.0 },
                num_actuators: 8,
            };
            let p_f = frozen.step(view).expect("frozen step");
            let p_n = natural.step(view).expect("natural step");
            assert_eq!(p_f.actuator_out.len(), 8);
            assert_eq!(p_n.actuator_out.len(), 8);
        }

        let h_frozen_after = frozen.learned_parameter_hash();
        let h_natural_after = natural.learned_parameter_hash();

        assert_eq!(
            h_frozen_before, h_frozen_after,
            "FrozenPlasticity must not mutate a single learned parameter bit"
        );
        assert_ne!(
            h_natural_before, h_natural_after,
            "Natural plasticity must adapt learned parameters"
        );
    }

    #[test]
    fn persistence_exact_200_step_determinism_corruption_rejection_and_prev_fallback() {
        let cfg = OrganismConfig::default();
        let mut uninterrupted = Alpha11Organism::new(2026, cfg);
        let mut split_run = Alpha11Organism::new(2026, cfg);

        let mut ambient = vec![0.0_f32; 96];
        let mut world_rng = Rng64::from_seed(99);
        let mut pulses_uninterrupted = Vec::with_capacity(100);
        let mut pulses_restored = Vec::with_capacity(100);
        let mut saved_at_100 = Vec::new();
        let mut restored = Alpha11Organism::new(999_999, cfg);

        for t in 0..200u64 {
            for (c, v) in ambient.iter_mut().enumerate() {
                *v = if c < 24 {
                    ((t as f32) * 0.09 + (c as f32) * 0.1).cos() * 0.7
                } else {
                    world_rng.next_f32_signed()
                };
            }
            let view = FieldView {
                ambient_in: &ambient,
                metabolic_influx: if t % 11 == 0 { 0.05 } else { 0.0 },
                tissue_stress: if t % 37 == 0 { 0.04 } else { 0.0 },
                num_actuators: 6,
            };

            let pu = uninterrupted.step(view).expect("uninterrupted step");
            if t < 100 {
                let _ = split_run.step(view).expect("split step");
                if t == 99 {
                    saved_at_100 = split_run.encode_checkpoint().expect("encode@100");
                    restored
                        .restore_checkpoint(&saved_at_100)
                        .expect("restore@100");
                }
            } else {
                pulses_uninterrupted.push(pu);
                let pr = restored.step(view).expect("restored step");
                pulses_restored.push(pr);
            }
        }

        assert_eq!(pulses_uninterrupted, pulses_restored);
        assert_eq!(uninterrupted.snapshot(), restored.snapshot());
        assert_eq!(
            uninterrupted.learned_parameter_hash(),
            restored.learned_parameter_hash()
        );
        assert_eq!(
            uninterrupted.encode_checkpoint().expect("final u"),
            restored.encode_checkpoint().expect("final r")
        );

        let mut pos = 24usize;
        let payload_end = saved_at_100.len() - 4;
        let mut slab_count_checked = 0usize;
        while pos < payload_end {
            let s_len = u32::from_le_bytes([
                saved_at_100[pos + 4],
                saved_at_100[pos + 5],
                saved_at_100[pos + 6],
                saved_at_100[pos + 7],
            ]) as usize;
            let flip_offset = pos + 8 + (s_len / 2);
            let mut corrupted = saved_at_100.clone();
            corrupted[flip_offset] ^= 0x04;

            let mut victim = Alpha11Organism::new(11, cfg);
            let snap_before = victim.snapshot();
            let err = victim
                .restore_checkpoint(&corrupted)
                .expect_err("bit flip must fail");
            assert_eq!(err, PersistError::ChecksumMismatch);
            assert!(victim.is_poisoned());
            assert_eq!(victim.snapshot(), snap_before);
            assert_eq!(
                victim.step(FieldView {
                    ambient_in: &[0.0; 8],
                    metabolic_influx: 0.0,
                    tissue_stress: 0.0,
                    num_actuators: 2,
                }),
                Err(PersistError::PoisonedAfterCorruption)
            );
            assert_eq!(
                victim.encode_checkpoint(),
                Err(PersistError::PoisonedAfterCorruption)
            );

            pos = pos + 8 + s_len + 4;
            slab_count_checked += 1;
        }
        assert_eq!(slab_count_checked, 6);

        let mut victim_trunc = Alpha11Organism::new(12, cfg);
        assert_eq!(
            victim_trunc.restore_checkpoint(&saved_at_100[..saved_at_100.len() - 9]),
            Err(PersistError::TruncatedInput)
        );
        assert!(victim_trunc.is_poisoned());

        let small_cfg = OrganismConfig {
            max_state_bytes: 2048,
            ..cfg
        };
        let mut victim_over = Alpha11Organism::new(13, small_cfg);
        let huge = vec![0u8; 300_000];
        assert_eq!(
            victim_over.restore_checkpoint(&huge),
            Err(PersistError::OversizedInput)
        );
        assert!(victim_over.is_poisoned());

        let mut bad_magic = saved_at_100.clone();
        bad_magic[0] ^= 0xFF;
        let mut victim_magic = Alpha11Organism::new(14, cfg);
        assert_eq!(
            victim_magic.restore_checkpoint(&bad_magic),
            Err(PersistError::BadMagicOrVersion)
        );
        assert!(victim_magic.is_poisoned());

        let mut nan_injected = saved_at_100.clone();
        let body_atp_offset = 24 + 8 + 8 * 6;
        nan_injected[body_atp_offset..body_atp_offset + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        rechecksum_checkpoint(&mut nan_injected);
        let mut victim_nan = Alpha11Organism::new(15, cfg);
        assert_eq!(
            victim_nan.restore_checkpoint(&nan_injected),
            Err(PersistError::NonFiniteState)
        );
        assert!(victim_nan.is_poisoned());
        assert_eq!(
            victim_nan.step(FieldView {
                ambient_in: &[0.0; 8],
                metabolic_influx: 0.0,
                tissue_stress: 0.0,
                num_actuators: 2,
            }),
            Err(PersistError::PoisonedAfterCorruption)
        );

        let tmp_dir = std::env::temp_dir().join(format!("a11_core_tx_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).expect("mkdir");
        let primary = tmp_dir.join("ckpt.bin");
        let prev = tmp_dir.join("ckpt.prev");
        let tmp = tmp_dir.join("ckpt.tmp");

        split_run
            .save_transactional(&primary, &prev, &tmp)
            .expect("save gen1");
        uninterrupted
            .save_transactional(&primary, &prev, &tmp)
            .expect("save gen2 rotates gen1 to prev");

        let mut bad_disk = fs::read(&primary).expect("read primary");
        bad_disk[40] ^= 0x80;
        fs::write(&primary, &bad_disk).expect("corrupt primary");

        let mut recovered = Alpha11Organism::new(1, cfg);
        let used_prev = recovered
            .load_transactional(&primary, &prev)
            .expect("fallback to prev");
        assert!(used_prev);
        assert!(!recovered.is_poisoned());
        assert_eq!(recovered.snapshot().tick, 100);

        // Saving again while `primary` is corrupted must NOT rotate the corrupt bytes over `.prev`.
        let good_prev = fs::read(&prev).expect("read prev");
        uninterrupted
            .save_transactional(&primary, &prev, &tmp)
            .expect("save gen3 with corrupted primary on disk");
        assert_eq!(fs::read(&prev).expect("read prev again"), good_prev);
        let mut after = Alpha11Organism::new(3, cfg);
        assert!(!after.load_transactional(&primary, &prev).expect("primary valid again"));

        fs::write(&prev, &bad_disk).expect("corrupt prev too");
        fs::write(&primary, &bad_disk).expect("corrupt primary again");
        let mut both_bad = Alpha11Organism::new(2, cfg);
        assert!(both_bad.load_transactional(&primary, &prev).is_err());
        assert!(both_bad.is_poisoned());
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn zero_mitosis_when_atp_is_depleted() {
        let cfg = OrganismConfig {
            initial_atp: 0.0,
            ..OrganismConfig::default()
        };
        let mut org = Alpha11Organism::new(55, cfg);
        let mut rng = Rng64::from_seed(9);
        let mut ambient = vec![0.0_f32; 64];
        for _ in 0..600 {
            for slot in &mut ambient {
                *slot = rng.next_f32_signed();
            }
            org.step(FieldView {
                ambient_in: &ambient,
                metabolic_influx: 0.0,
                tissue_stress: 0.05,
                num_actuators: 4,
            })
            .expect("starved step");
        }
        assert_eq!(org.snapshot().mitosis_births, 0);
    }
}
