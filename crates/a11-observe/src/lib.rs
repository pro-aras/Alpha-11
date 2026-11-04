//! Non-invasive observational microscope, forensic hygiene & persistence auditor,
//! and multi-seed 6-control scientific twin engine for Alpha 11.

pub mod campaign;
pub mod campaign_social_portals;
pub mod forensics;
pub mod gates;
pub mod open_habitat;
pub mod report;
pub mod stats;
pub mod twins;

pub use forensics::{
    CorruptionCaseOutcome, FORBIDDEN_IDENTIFIERS, HygieneFinding, HygieneViolationKind,
    LatencyMemoryProfile, PersistenceAuditConfig, PersistenceAuditReport,
    PlasticityLesionAuditReport, ProcMemoryStats, SourceHygieneReport, audit_persistence,
    audit_persistence_with_config, audit_plasticity_and_lesion, parse_proc_status_memory,
    profile_latency, pulses_bit_identical, read_linux_proc_memory, scan_crate_source,
    scan_source_str, snapshots_bit_identical,
};
pub use report::{
    format_forensics_json, format_forensics_markdown, format_identical_twins_json,
    format_identical_twins_markdown, format_twin_study_json, format_twin_study_markdown,
    write_forensics_files, write_twin_study_files,
};
pub use stats::{
    DEFAULT_PERMUTATION_SEED, DEFAULT_PERMUTATIONS, DeterministicRng, EmpiricalDistribution,
    IndividualityDivergence, MannWhitneyComparison, compute_individuality_divergence,
    cosine_distance, erfc_approx, jensen_shannon_divergence, mann_whitney_u,
    mann_whitney_u_with_permutations, mean_pairwise_cosine_distance, mean_pairwise_jsd,
    quantile_sorted, shannon_entropy_bits,
};
pub use twins::{
    ArmCohortSummary, BandAllocation, BandCohortSummary, BandRole, BandSpec, DEFAULT_TWELVE_SEEDS,
    DevelopmentalCohortPoint, DevelopmentalSample, IdenticalTwinConfig, IdenticalTwinReport,
    PairwiseMetricComparison, PulseTemporalChannelShuffler, SeedArmRun, TwinArm, TwinStudyConfig,
    TwinStudyReport, compute_band_allocations, decode_snapshot_channels,
    developmental_checkpoint_ticks, observe_identical_twin_divergence, run_twin_study,
    run_twin_study_instrumented, run_twin_study_with_perturbation,
};

/// Confirms the `a11-observe` crate is linked and ready.
#[must_use]
pub const fn workspace_ready() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use a11_medium::{
        ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError,
        WorldMedium,
    };

    const NUM_CHANNELS: usize = 12;
    const NUM_ACTUATORS: usize = 4;
    const ECHO_DELAY: usize = 4;
    const MOCK_MAGIC: [u8; 4] = *b"A11M";
    const MOCK_VERSION: u32 = 1;
    const MOCK_MAX_BYTES: usize = 1024 * 1024;

    /// Synthetic 12-channel medium with Photonic (`0..4`), Echo (`4..8`, `\Delta = 4`),
    /// and Noisy-TV Static (`8..12`) bands for self-verifying the observer microscope.
    struct SyntheticMultiBandWorld {
        tick: u64,
        phase_offset: f32,
        darkness: bool,
        polarity: f32,
        rng: DeterministicRng,
        ambient: [f32; NUM_CHANNELS],
        target_dir: [f32; NUM_ACTUATORS],
        echo_ring: [[f32; NUM_ACTUATORS]; ECHO_DELAY],
        echo_cursor: usize,
        last_influx: f32,
        last_stress: f32,
    }

    impl SyntheticMultiBandWorld {
        fn new(seed: u64) -> Self {
            let mut rng = DeterministicRng::new(seed ^ 0xABCD_EF01_2345_6789);
            let phase_offset = (rng.next_f64() * std::f64::consts::TAU) as f32;
            let mut w = Self {
                tick: 0,
                phase_offset,
                darkness: false,
                polarity: 1.0,
                rng,
                ambient: [0.0; NUM_CHANNELS],
                target_dir: [0.0; NUM_ACTUATORS],
                echo_ring: [[0.0; NUM_ACTUATORS]; ECHO_DELAY],
                echo_cursor: 0,
                last_influx: 0.0,
                last_stress: 0.0,
            };
            w.refresh_ambient([0.0; NUM_ACTUATORS]);
            w
        }

        fn set_darkness(&mut self, dark: bool) {
            self.darkness = dark;
        }

        fn refresh_ambient(&mut self, delayed_pulse: [f32; NUM_ACTUATORS]) {
            let phase = self.phase_offset + (self.tick as f32) * 0.17;
            for (i, &delayed_val) in delayed_pulse.iter().enumerate() {
                let offset = (i as f32) * 1.570_796_4;
                let wave = (phase + offset).sin();
                self.target_dir[i] = wave * self.polarity;
                // Channels 0..4: Photonic 2D band (vanishes in darkness).
                self.ambient[i] = if self.darkness { 0.0 } else { wave };
                // Channels 4..8: Active time-of-flight echo return from `t - ECHO_DELAY`.
                let echo_carrier = if delayed_val.abs() > 0.02 {
                    (delayed_val * 0.5 + 0.5 * wave).clamp(-1.0, 1.0)
                } else {
                    0.15 * wave
                };
                self.ambient[4 + i] = echo_carrier;
                // Channels 8..12: High-amplitude un-predictable Noisy-TV static.
                self.ambient[8 + i] = ((self.rng.next_f64() * 2.0) - 1.0) as f32;
            }
        }
    }

    impl WorldMedium for SyntheticMultiBandWorld {
        fn num_ambient_channels(&self) -> usize {
            NUM_CHANNELS
        }

        fn num_actuators(&self) -> usize {
            NUM_ACTUATORS
        }

        fn current_view(&self) -> FieldView<'_> {
            FieldView {
                ambient_in: &self.ambient,
                metabolic_influx: self.last_influx,
                tissue_stress: self.last_stress,
                num_actuators: NUM_ACTUATORS,
            }
        }

        fn apply_pulse(&mut self, pulse: &FieldPulse) {
            let mut act = [0.0_f32; NUM_ACTUATORS];
            for (i, slot) in act.iter_mut().enumerate() {
                *slot = pulse.actuator_out.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
            }

            // Alignment between organism's pulse at tick `t` and current physical target.
            let mut dot = 0.0_f32;
            let mut energy = 0.0_f32;
            for (&act_val, &target_val) in act.iter().zip(self.target_dir.iter()) {
                dot += act_val * target_val;
                energy += act_val * act_val;
            }
            self.last_influx = (dot * 0.5).max(0.0);
            self.last_stress = ((-dot) * 0.25).max(0.0) + 0.02 * energy;

            let delayed = self.echo_ring[self.echo_cursor];
            self.echo_ring[self.echo_cursor] = act;
            self.echo_cursor = (self.echo_cursor + 1) % ECHO_DELAY;
            self.tick += 1;
            self.refresh_ambient(delayed);
        }
    }

    /// Synthetic `t - 1` efference-copy organism implementing `OrganismEngine` with CRC32C
    /// shadow-decoded persistence and strict `FrozenPlasticity` / flagless lesion semantics.
    #[derive(Clone, Debug)]
    struct SyntheticCausalOrganism {
        mode: ControlMode,
        poisoned: bool,
        tick: u64,
        atp: f32,
        temperature: f32,
        cells: u32,
        mean_tau: f32,
        max_tau: f32,
        agency_ema: f32,
        pred_err_ema: f32,
        mitosis_births: u64,
        committed_pulses: u64,
        rng: DeterministicRng,
        sensor_conductances: [f32; NUM_CHANNELS],
        sensor_delays: [u8; NUM_CHANNELS],
        pred_weights: [f32; NUM_CHANNELS],
        motor_weights: [[f32; NUM_CHANNELS]; NUM_ACTUATORS],
        prev_ambient: [f32; NUM_CHANNELS],
        prev_pulse: [f32; NUM_ACTUATORS],
        pulse_ring: [[f32; NUM_ACTUATORS]; ECHO_DELAY],
        ring_idx: usize,
        has_prev: bool,
    }

    impl SyntheticCausalOrganism {
        fn new(seed: u64) -> Self {
            let mut rng = DeterministicRng::new(seed ^ 0x1122_3344_5566_7788);
            let mut motor_weights = [[0.0_f32; NUM_CHANNELS]; NUM_ACTUATORS];
            for row in &mut motor_weights {
                for w in row.iter_mut() {
                    *w = (((rng.next_f64() * 2.0) - 1.0) * 0.03) as f32;
                }
            }
            Self {
                mode: ControlMode::Natural,
                poisoned: false,
                tick: 0,
                atp: 1.0,
                temperature: 0.25,
                cells: 32,
                mean_tau: 1.0,
                max_tau: 1.0,
                agency_ema: 0.0,
                pred_err_ema: 0.5,
                mitosis_births: 0,
                committed_pulses: 0,
                rng,
                sensor_conductances: [0.10; NUM_CHANNELS],
                sensor_delays: [1; NUM_CHANNELS],
                pred_weights: [0.0; NUM_CHANNELS],
                motor_weights,
                prev_ambient: [0.0; NUM_CHANNELS],
                prev_pulse: [0.0; NUM_ACTUATORS],
                pulse_ring: [[0.0; NUM_ACTUATORS]; ECHO_DELAY],
                ring_idx: 0,
                has_prev: false,
            }
        }
    }

    impl OrganismEngine for SyntheticCausalOrganism {
        fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }

            let mut cur_amb = [0.0_f32; NUM_CHANNELS];
            for (i, slot) in cur_amb.iter_mut().enumerate() {
                *slot = view.ambient_in.get(i).copied().unwrap_or(0.0);
            }

            let delayed_pulse = self.pulse_ring[self.ring_idx];

            // Out-of-sample `t - 1` prediction & counterfactual agency measurement BEFORE learning.
            if self.has_prev {
                let mut err_sum = 0.0_f32;
                let mut agency_gain = 0.0_f32;
                for (ch, &cur_val) in cur_amb.iter().enumerate() {
                    let echo_act = delayed_pulse[ch % NUM_ACTUATORS];
                    let pred_with_act =
                        self.pred_weights[ch] * self.prev_ambient[ch] + 0.35 * echo_act;
                    let pred_no_act = self.pred_weights[ch] * self.prev_ambient[ch];
                    let err_act = (cur_val - pred_with_act).abs();
                    let err_null = (cur_val - pred_no_act).abs();
                    err_sum += err_act;
                    agency_gain += (err_null - err_act).max(0.0);
                }
                let mean_err = err_sum / (NUM_CHANNELS as f32);
                self.pred_err_ema = 0.92 * self.pred_err_ema + 0.08 * mean_err;
                self.agency_ema =
                    0.92 * self.agency_ema + 0.08 * (agency_gain / (NUM_CHANNELS as f32));

                if self.mode == ControlMode::Natural {
                    let reward_signal = view.metabolic_influx - 0.8 * view.tissue_stress;
                    for (ch, &cur_val) in cur_amb.iter().enumerate() {
                        let pred = self.pred_weights[ch] * self.prev_ambient[ch];
                        let resid = cur_val - pred;
                        // Update temporal auto-predictor.
                        self.pred_weights[ch] =
                            (self.pred_weights[ch] + 0.08 * resid * self.prev_ambient[ch])
                                .clamp(-1.5, 1.5);

                        let signal_mag = cur_val.abs();
                        // Out-of-sample predictive error reduction over zero-prediction baseline.
                        let pred_improvement = signal_mag - resid.abs();
                        let echo_resid = (cur_val - 0.5 * delayed_pulse[ch % NUM_ACTUATORS]).abs();
                        let echo_improvement = signal_mag - echo_resid;

                        // Trophic pore law: predictable or echo-aligned active channels grow;
                        // silent channels (darkness) or un-predictable static (Noisy TV) pay ATP rent and atrophy.
                        let predict_quality = pred_improvement.max(echo_improvement);
                        let delta_cond = if signal_mag < 1e-4 {
                            -0.03 * self.sensor_conductances[ch] - 0.003
                        } else {
                            0.05 * predict_quality - 0.006
                        };
                        self.sensor_conductances[ch] =
                            (self.sensor_conductances[ch] + delta_cond).clamp(0.001, 2.5);

                        if echo_improvement > 0.02 && echo_improvement > pred_improvement {
                            self.sensor_delays[ch] = ECHO_DELAY as u8;
                        }

                        // `t - 1` efference-copy motor credit using `self.prev_pulse` and `self.prev_ambient`.
                        for a in 0..NUM_ACTUATORS {
                            let elig = self.prev_pulse[a] * self.prev_ambient[ch];
                            self.motor_weights[a][ch] = (self.motor_weights[a][ch]
                                + 0.14 * reward_signal * elig * self.sensor_conductances[ch])
                                .clamp(-2.5, 2.5);
                        }
                    }

                    if view.metabolic_influx > 0.15 && self.cells < 96 && self.tick % 25 == 0 {
                        self.cells += 2;
                        self.mitosis_births += 1;
                        self.mean_tau = (self.mean_tau + 0.35).min(12.0);
                        self.max_tau = (self.max_tau + 1.1).min(32.0);
                    }
                }
            }

            self.atp = (self.atp + 0.15 * view.metabolic_influx - 0.10 * view.tissue_stress - 0.01)
                .clamp(0.05, 5.0);
            self.temperature = (0.35 - 0.05 * self.atp).clamp(0.02, 0.5);

            let mut out = vec![0.0_f32; view.num_actuators];
            for (a, slot) in out.iter_mut().enumerate().take(NUM_ACTUATORS) {
                let mut drive = 0.0_f32;
                for (ch, &cur_val) in cur_amb.iter().enumerate() {
                    drive += self.motor_weights[a][ch] * self.sensor_conductances[ch] * cur_val;
                }
                let explore = if self.mode == ControlMode::Natural {
                    (((self.rng.next_f64() * 2.0) - 1.0) as f32) * self.temperature
                } else {
                    0.0
                };
                *slot = (drive + explore).tanh();
            }

            let mut committed = [0.0_f32; NUM_ACTUATORS];
            for (i, slot) in committed.iter_mut().enumerate() {
                *slot = out.get(i).copied().unwrap_or(0.0);
            }
            if committed.iter().any(|&v| v.abs() > 0.05) {
                self.committed_pulses += 1;
            }
            self.pulse_ring[self.ring_idx] = committed;
            self.ring_idx = (self.ring_idx + 1) % ECHO_DELAY;
            self.prev_pulse = committed;
            self.prev_ambient = cur_amb;
            self.has_prev = true;
            self.tick += 1;

            Ok(FieldPulse { actuator_out: out })
        }

        fn set_control_mode(&mut self, mode: ControlMode) {
            self.mode = mode;
        }

        fn control_mode(&self) -> ControlMode {
            self.mode
        }

        fn zero_learned_conductances(&mut self) {
            self.sensor_conductances = [0.0; NUM_CHANNELS];
            self.pred_weights = [0.0; NUM_CHANNELS];
            self.motor_weights = [[0.0; NUM_CHANNELS]; NUM_ACTUATORS];
        }

        fn learned_parameter_hash(&self) -> u64 {
            let mut h = 0xCBF2_9CE4_8422_2325_u64;
            let mut mix_u32 = |v: u32| {
                for b in v.to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0100_0000_01B3);
                }
            };
            mix_u32(self.cells);
            for &c in &self.sensor_conductances {
                mix_u32(c.to_bits());
            }
            for &d in &self.sensor_delays {
                mix_u32(u32::from(d));
            }
            for &pw in &self.pred_weights {
                mix_u32(pw.to_bits());
            }
            for row in &self.motor_weights {
                for &mw in row {
                    mix_u32(mw.to_bits());
                }
            }
            h
        }

        fn snapshot(&self) -> OrganismSnapshot {
            let octaves = (self.max_tau.max(1.0).log2().ceil() as usize).max(1);
            OrganismSnapshot {
                tick: self.tick,
                atp: self.atp,
                adenosine: (1.0 - 0.15 * self.atp).clamp(0.0, 1.0),
                temperature: self.temperature,
                motor_clutch_threshold: 0.05,
                cell_count: self.cells as usize,
                tube_count: (self.cells as usize) * 3,
                sensor_pore_count: NUM_CHANNELS,
                motor_pore_count: NUM_ACTUATORS,
                active_cells_last_tick: (self.cells as usize) / 2,
                stiff_highway_tubes: (self.mitosis_births as usize) * 2,
                inhibitory_tubes: 4,
                active_echo_delay_tubes: self
                    .sensor_delays
                    .iter()
                    .filter(|&&d| usize::from(d) == ECHO_DELAY)
                    .count(),
                mean_tau: self.mean_tau,
                max_tau: self.max_tau,
                tau_octave_count: octaves,
                mean_interventional_agency: self.agency_ema,
                mean_prediction_error: self.pred_err_ema,
                committed_motor_pulses: self.committed_pulses,
                subthreshold_rehearsals: self.tick.saturating_sub(self.committed_pulses),
                rest_consolidation_ticks: 0,
                mitosis_births: self.mitosis_births,
                contradiction_splits: 0,
                myelinated_shortcuts: self.mitosis_births / 2,
                atrophied_structures: self
                    .sensor_conductances
                    .iter()
                    .filter(|&&c| c < 0.02)
                    .count() as u64,
                sensor_pore_channels: (0..(NUM_CHANNELS as u32)).collect(),
                sensor_pore_conductances: self.sensor_conductances.to_vec(),
                sensor_pore_delays: self.sensor_delays.to_vec(),
                motor_pore_channels: (0..(NUM_ACTUATORS as u32)).collect(),
                motor_pore_conductances: self
                    .motor_weights
                    .iter()
                    .map(|row| row.iter().map(|w| w.abs()).sum::<f32>())
                    .collect(),
                used_state_bytes: std::mem::size_of::<Self>() + (self.cells as usize) * 32,
            }
        }

        fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }
            let mut payload = Vec::with_capacity(512);
            payload.push(match self.mode {
                ControlMode::Natural => 0,
                ControlMode::FrozenPlasticity => 1,
            });
            payload.push(u8::from(self.has_prev));
            payload.push(self.ring_idx as u8);
            payload.extend_from_slice(&self.tick.to_le_bytes());
            payload.extend_from_slice(&self.atp.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.temperature.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.cells.to_le_bytes());
            payload.extend_from_slice(&self.mean_tau.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.max_tau.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.agency_ema.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.pred_err_ema.to_bits().to_le_bytes());
            payload.extend_from_slice(&self.mitosis_births.to_le_bytes());
            payload.extend_from_slice(&self.committed_pulses.to_le_bytes());
            payload.extend_from_slice(&self.rng.raw_state().to_le_bytes());
            for &v in &self.sensor_conductances {
                payload.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            payload.extend_from_slice(&self.sensor_delays);
            for &v in &self.pred_weights {
                payload.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for row in &self.motor_weights {
                for &v in row {
                    payload.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }
            for &v in &self.prev_ambient {
                payload.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for &v in &self.prev_pulse {
                payload.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            for row in &self.pulse_ring {
                for &v in row {
                    payload.extend_from_slice(&v.to_bits().to_le_bytes());
                }
            }

            let mut framed = Vec::with_capacity(16 + payload.len());
            framed.extend_from_slice(&MOCK_MAGIC);
            framed.extend_from_slice(&MOCK_VERSION.to_le_bytes());
            framed.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            let crc = crc32c(&payload);
            framed.extend_from_slice(&crc.to_le_bytes());
            framed.extend_from_slice(&payload);
            Ok(framed)
        }

        fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError> {
            let decoded = match shadow_decode_mock(bytes) {
                Ok(d) => d,
                Err(err) => {
                    self.poisoned = true;
                    return Err(err);
                }
            };
            *self = decoded;
            Ok(())
        }

        fn verify_checkpoint(&self, bytes: &[u8]) -> Result<(), PersistError> {
            let mut shadow = self.clone();
            shadow.poisoned = false;
            shadow.restore_checkpoint(bytes)
        }

        fn is_poisoned(&self) -> bool {
            self.poisoned
        }
    }

    fn shadow_decode_mock(bytes: &[u8]) -> Result<SyntheticCausalOrganism, PersistError> {
        if bytes.len() > MOCK_MAX_BYTES {
            return Err(PersistError::OversizedInput);
        }
        if bytes.len() < 16 {
            return Err(PersistError::TruncatedInput);
        }
        if bytes[0..4] != MOCK_MAGIC {
            return Err(PersistError::BadMagicOrVersion);
        }
        let ver = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if ver != MOCK_VERSION {
            return Err(PersistError::BadMagicOrVersion);
        }
        let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let expected_crc = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if bytes.len() != 16 + payload_len {
            return Err(if bytes.len() < 16 + payload_len {
                PersistError::TruncatedInput
            } else {
                PersistError::OversizedInput
            });
        }
        let payload = &bytes[16..];
        if crc32c(payload) != expected_crc {
            return Err(PersistError::ChecksumMismatch);
        }

        let mut cur = 0usize;
        let read_u8 = |c: &mut usize| -> Result<u8, PersistError> {
            let v = *payload.get(*c).ok_or(PersistError::TruncatedInput)?;
            *c += 1;
            Ok(v)
        };
        let read_u32 = |c: &mut usize| -> Result<u32, PersistError> {
            let s = payload.get(*c..(*c + 4)).ok_or(PersistError::TruncatedInput)?;
            *c += 4;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        let read_u64 = |c: &mut usize| -> Result<u64, PersistError> {
            let s = payload.get(*c..(*c + 8)).ok_or(PersistError::TruncatedInput)?;
            *c += 8;
            Ok(u64::from_le_bytes([
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            ]))
        };
        let read_f32 = |c: &mut usize| -> Result<f32, PersistError> {
            let bits = read_u32(c)?;
            let v = f32::from_bits(bits);
            if !v.is_finite() {
                return Err(PersistError::NonFiniteState);
            }
            Ok(v)
        };

        let mode = match read_u8(&mut cur)? {
            0 => ControlMode::Natural,
            1 => ControlMode::FrozenPlasticity,
            _ => return Err(PersistError::InvalidTopology),
        };
        let has_prev = match read_u8(&mut cur)? {
            0 => false,
            1 => true,
            _ => return Err(PersistError::InvalidTopology),
        };
        let ring_idx = read_u8(&mut cur)? as usize;
        if ring_idx >= ECHO_DELAY {
            return Err(PersistError::InvalidTopology);
        }
        let tick = read_u64(&mut cur)?;
        let atp = read_f32(&mut cur)?;
        let temperature = read_f32(&mut cur)?;
        let cells = read_u32(&mut cur)?;
        let mean_tau = read_f32(&mut cur)?;
        let max_tau = read_f32(&mut cur)?;
        let agency_ema = read_f32(&mut cur)?;
        let pred_err_ema = read_f32(&mut cur)?;
        let mitosis_births = read_u64(&mut cur)?;
        let committed_pulses = read_u64(&mut cur)?;
        let rng_state = read_u64(&mut cur)?;

        let mut sensor_conductances = [0.0_f32; NUM_CHANNELS];
        for slot in &mut sensor_conductances {
            *slot = read_f32(&mut cur)?;
        }
        let mut sensor_delays = [0_u8; NUM_CHANNELS];
        for slot in &mut sensor_delays {
            *slot = read_u8(&mut cur)?;
        }
        let mut pred_weights = [0.0_f32; NUM_CHANNELS];
        for slot in &mut pred_weights {
            *slot = read_f32(&mut cur)?;
        }
        let mut motor_weights = [[0.0_f32; NUM_CHANNELS]; NUM_ACTUATORS];
        for row in &mut motor_weights {
            for slot in row {
                *slot = read_f32(&mut cur)?;
            }
        }
        let mut prev_ambient = [0.0_f32; NUM_CHANNELS];
        for slot in &mut prev_ambient {
            *slot = read_f32(&mut cur)?;
        }
        let mut prev_pulse = [0.0_f32; NUM_ACTUATORS];
        for slot in &mut prev_pulse {
            *slot = read_f32(&mut cur)?;
        }
        let mut pulse_ring = [[0.0_f32; NUM_ACTUATORS]; ECHO_DELAY];
        for row in &mut pulse_ring {
            for slot in row {
                *slot = read_f32(&mut cur)?;
            }
        }
        if cur != payload.len() {
            return Err(PersistError::InvalidTopology);
        }

        Ok(SyntheticCausalOrganism {
            mode,
            poisoned: false,
            tick,
            atp,
            temperature,
            cells,
            mean_tau,
            max_tau,
            agency_ema,
            pred_err_ema,
            mitosis_births,
            committed_pulses,
            rng: DeterministicRng::from_raw_state(rng_state),
            sensor_conductances,
            sensor_delays,
            pred_weights,
            motor_weights,
            prev_ambient,
            prev_pulse,
            pulse_ring,
            ring_idx,
            has_prev,
        })
    }

    fn crc32c(bytes: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
            }
        }
        !crc
    }

    fn test_bands() -> Vec<BandSpec> {
        vec![
            BandSpec::new("Photonic 2D", 0..4, BandRole::StructuredPhysical),
            BandSpec::new("Wave-Echo Sonar", 4..8, BandRole::ActiveEcho),
            BandSpec::new("Noisy TV Static", 8..12, BandRole::NoisyTvDistractor),
        ]
    }

    #[test]
    fn persistence_plasticity_and_latency_auditors_verify_all_invariants() {
        let persist_cfg = PersistenceAuditConfig {
            seed: 42,
            warmup_steps: 100,
            continuation_steps: 100,
            oversized_bytes_len: MOCK_MAX_BYTES + 64,
        };
        let persist_report = audit_persistence_with_config(
            persist_cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("persistence audit must succeed on valid mock");
        assert!(
            persist_report.all_passed(),
            "persistence audit failed: {persist_report:?}"
        );

        let plasticity_report = audit_plasticity_and_lesion(
            42,
            500,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("plasticity & lesion audit must succeed");
        assert!(
            plasticity_report.all_passed(),
            "plasticity/lesion audit failed: {plasticity_report:?}"
        );

        let mut org = SyntheticCausalOrganism::new(77);
        let mut world = SyntheticMultiBandWorld::new(77);
        let lat_profile =
            profile_latency(&mut org, &mut world, 120).expect("latency profiling must succeed");
        assert_eq!(lat_profile.steps_profiled, 120);
        assert!(lat_profile.step_latency_us.p99 >= lat_profile.step_latency_us.p50);
        assert!(lat_profile.checkpoint_bytes > 16);
        assert!(lat_profile.peak_used_state_bytes > 0);
    }

    #[test]
    fn multi_seed_six_control_twin_engine_and_reports_are_deterministic_and_accurate() {
        let mut cfg = TwinStudyConfig::new("Synthetic_6Arm_Verification", 240, test_bands());
        cfg.post_life_eval_ticks = 120;

        let report_a = run_twin_study(
            &cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("twin study run A");
        let report_b = run_twin_study(
            &cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("twin study run B");

        // Exact determinism across repeated multi-seed executions.
        assert_eq!(report_a, report_b);
        assert!(report_a.lesion_preserved_natural_mode_all_seeds);
        assert_eq!(report_a.arm_summaries.len(), 6);

        let nat = report_a
            .arm_summary(TwinArm::Natural)
            .expect("Natural summary");
        let froz = report_a
            .arm_summary(TwinArm::FrozenPlasticity)
            .expect("Frozen summary");
        let yoked = report_a
            .arm_summary(TwinArm::YokedMetabolism)
            .expect("Yoked summary");
        let shuf = report_a
            .arm_summary(TwinArm::ActionShuffled)
            .expect("ActionShuffled summary");
        let unlesioned = report_a
            .arm_summary(TwinArm::UnlesionedPostLife)
            .expect("UnlesionedPostLife summary");
        let lesioned = report_a
            .arm_summary(TwinArm::FlaglessPostLifeLesion)
            .expect("FlaglessPostLifeLesion summary");

        assert!(nat.achieved_influx_per_tick.p50 > froz.achieved_influx_per_tick.p50);
        assert!(nat.achieved_influx_per_tick.p50 > yoked.achieved_influx_per_tick.p50);
        assert!(nat.achieved_influx_per_tick.p50 > shuf.achieved_influx_per_tick.p50);
        assert!(unlesioned.achieved_influx_per_tick.p50 > lesioned.achieved_influx_per_tick.p50);

        // Noisy TV distractor band (uniform random = 33.3%) is starved below 10% in Natural.
        assert!(
            nat.noisy_tv_conductance_share.p50 < 0.10,
            "expected Noisy TV share < 0.10, got {}",
            nat.noisy_tv_conductance_share.p50
        );

        let md = format_twin_study_markdown(&report_a);
        let json = format_twin_study_json(&report_a);
        assert!(md.contains("Observational Twin Study: Synthetic_6Arm_Verification"));
        assert!(md.contains("Wave-Echo Sonar"));
        assert!(md.contains("Noisy TV Static"));
        assert!(json.starts_with("{\n"));
        assert!(json.ends_with("}\n"));
        assert!(json.contains("\"rank_biserial\":"));
    }

    #[test]
    fn day_to_darkness_perturbation_and_identical_twins_capture_organ_shift_and_divergence() {
        let cfg = TwinStudyConfig {
            study_name: "Cavefish_Transition_Check".to_string(),
            seeds: DEFAULT_TWELVE_SEEDS[..10].to_vec(),
            horizon_ticks: 300,
            post_life_eval_ticks: 80,
            bands: test_bands(),
        };

        let report = run_twin_study_with_perturbation(
            &cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
            |tick, world| {
                if tick == 151 {
                    world.set_darkness(true);
                }
            },
        )
        .expect("cavefish perturbation run");

        let nat = report
            .arm_summary(TwinArm::Natural)
            .expect("Natural summary");
        let mid_pt = nat
            .developmental_curve
            .iter()
            .find(|p| p.tick == 150)
            .expect("50% checkpoint at tick 150");
        let end_pt = nat
            .developmental_curve
            .last()
            .expect("100% checkpoint at tick 300");

        let photonic_mid = mid_pt.band_conductance_shares[0].1.p50;
        let photonic_end = end_pt.band_conductance_shares[0].1.p50;
        let sonar_mid = mid_pt.band_conductance_shares[1].1.p50;
        let sonar_end = end_pt.band_conductance_shares[1].1.p50;

        assert!(
            photonic_end < photonic_mid,
            "expected photonic share to drop in darkness ({photonic_mid} -> {photonic_end})"
        );
        assert!(
            sonar_end > sonar_mid,
            "expected sonar share to rise in darkness ({sonar_mid} -> {sonar_end})"
        );

        let id_cfg = IdenticalTwinConfig::default_twelve(999, 150, test_bands());
        let id_report = observe_identical_twin_divergence(
            &id_cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("identical twin divergence run");
        assert_eq!(id_report.cohort_summary.seed_count, 12);
        assert!(id_report.cohort_summary.individuality.composite_individuality_index > 0.0);
        let id_md = format_identical_twins_markdown(&id_report);
        let id_json = format_identical_twins_json(&id_report);
        assert!(id_md.contains("Identical-Genome Constrained-Randomness Divergence"));
        assert!(id_json.contains("\"composite_individuality_index\":"));
        assert_balanced_valid_json_structure(&id_json);
    }

    #[test]
    fn self_hygiene_and_json_forensics_formatting_are_clean_and_well_formed() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let observe_src = manifest_dir.join("src");
        let observe_hygiene =
            scan_crate_source(&observe_src).expect("scan a11-observe/src directory");
        assert!(observe_hygiene.files_scanned >= 8);
        assert_eq!(observe_hygiene.unsafe_block_count, 0);
        assert_eq!(observe_hygiene.unwrap_call_count, 0);

        let medium_src = manifest_dir.join("../a11-medium/src");
        let medium_hygiene = scan_crate_source(&medium_src).expect("scan a11-medium/src directory");
        assert!(
            medium_hygiene.is_clean(),
            "expected a11-medium/src to be 100% clean, got {medium_hygiene:?}"
        );

        let persist_cfg = PersistenceAuditConfig {
            seed: 9,
            warmup_steps: 100,
            continuation_steps: 100,
            oversized_bytes_len: MOCK_MAX_BYTES + 32,
        };
        let p_rep = audit_persistence_with_config(
            persist_cfg,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("persistence audit");
        let pl_rep = audit_plasticity_and_lesion(
            9,
            500,
            SyntheticCausalOrganism::new,
            SyntheticMultiBandWorld::new,
        )
        .expect("plasticity audit");
        let mut org = SyntheticCausalOrganism::new(9);
        let mut world = SyntheticMultiBandWorld::new(9);
        let lat_rep = profile_latency(&mut org, &mut world, 50).expect("latency profile");

        let f_md =
            format_forensics_markdown(Some(&medium_hygiene), Some(&p_rep), Some(&pl_rep), Some(&lat_rep));
        let f_json =
            format_forensics_json(Some(&medium_hygiene), Some(&p_rep), Some(&pl_rep), Some(&lat_rep));
        assert!(f_md.contains("Transactional CRC32C Persistence"));
        assert_balanced_valid_json_structure(&f_json);
    }

    fn assert_balanced_valid_json_structure(json: &str) {
        assert!(!json.contains("NaN"), "JSON must not contain NaN");
        assert!(!json.contains("Infinity"), "JSON must not contain Infinity");
        assert!(!json.contains(": inf"), "JSON must not contain bare inf");
        let mut stack = Vec::new();
        let mut in_str = false;
        let mut escaped = false;
        for ch in json.chars() {
            if in_str {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_str = false;
                }
                continue;
            }
            match ch {
                '"' => in_str = true,
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' | ']' => {
                    assert_eq!(stack.pop(), Some(ch), "mismatched JSON delimiter `{ch}`");
                }
                _ => {}
            }
        }
        assert!(!in_str, "unclosed JSON string");
        assert!(stack.is_empty(), "unclosed JSON delimiters: {stack:?}");
    }
}
