//! Pure numeric, coordinate-free, modality-free boundary contract for Alpha 11.
//!
//! No environment geometry, pixel widths, modality tags, string labels, or
//! anthropomorphic counters cross this boundary. The organism perceives only an
//! unlabelled slice of wave fluxes plus somatic energy/stress scalars, and emits
//! continuous actuator pulses.

/// Read-only view of the local ambient field presented to an organism on a single tick.
#[derive(Clone, Copy, Debug)]
pub struct FieldView<'a> {
    /// Raw unlabelled ambient wave channels (typically bounded in `[-1.0, 1.0]`).
    /// Channel order carries no predetermined meaning; the organism must discover
    /// spatiotemporal predictability, cross-channel covariance, and echo delays on its own.
    pub ambient_in: &'a [f32],
    /// Raw somatic nourishment absorbed across the membrane since the previous tick (`>= 0.0`).
    pub metabolic_influx: f32,
    /// Raw somatic tissue stress / noxious load experienced since the previous tick (`>= 0.0`).
    pub tissue_stress: f32,
    /// Number of continuous actuator channels exposed by the medium on this tick.
    pub num_actuators: usize,
}

/// Continuous force / wave emission output produced by an organism on a single tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldPulse {
    /// Continuous actuator outputs in `[-1.0, 1.0]`, length `== FieldView::num_actuators`.
    pub actuator_out: Vec<f32>,
}

impl FieldPulse {
    /// Creates a zero-force pulse of length `num_actuators`.
    #[must_use]
    pub fn silent(num_actuators: usize) -> Self {
        Self {
            actuator_out: vec![0.0; num_actuators],
        }
    }
}

/// Experimental plasticity control mode for read-only scientific twin comparisons.
///
/// Post-life lesions are NEVER implemented as a `ControlMode` branch; instead,
/// learned conductances are physically zeroed in memory while running `ControlMode::Natural`
/// or `ControlMode::FrozenPlasticity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlMode {
    /// Normal open-ended growth, plasticity, and metabolic homeostasis.
    Natural,
    /// Strictly blocks every learned parameter write, conductance update, pore birth/retraction,
    /// cell mitosis, contradiction split, and homeostatic weight drift across the entire organism.
    FrozenPlasticity,
}

/// Pure biophysical and topological telemetry snapshot of an organism.
///
/// Strictly excludes anthropomorphic labels ("words", "concepts", "thoughts", "eyes", "ears").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrganismSnapshot {
    pub tick: u64,
    pub atp: f32,
    pub adenosine: f32,
    pub temperature: f32,
    pub motor_clutch_threshold: f32,

    pub cell_count: usize,
    pub tube_count: usize,
    pub sensor_pore_count: usize,
    pub motor_pore_count: usize,

    pub active_cells_last_tick: usize,
    pub stiff_highway_tubes: usize,
    pub inhibitory_tubes: usize,
    pub active_echo_delay_tubes: usize,

    pub mean_tau: f32,
    pub max_tau: f32,
    pub tau_octave_count: usize,

    pub mean_interventional_agency: f32,
    pub mean_prediction_error: f32,

    pub committed_motor_pulses: u64,
    pub subthreshold_rehearsals: u64,
    pub rest_consolidation_ticks: u64,

    pub mitosis_births: u64,
    pub contradiction_splits: u64,
    pub myelinated_shortcuts: u64,
    pub atrophied_structures: u64,

    /// External `ambient_in` channel indices currently bound by live `SensorPore`s.
    pub sensor_pore_channels: Vec<u32>,
    /// Conductance strengths matching `sensor_pore_channels`.
    pub sensor_pore_conductances: Vec<f32>,
    /// Tuned temporal delay ticks (`1..=16`) matching `sensor_pore_channels`.
    pub sensor_pore_delays: Vec<u8>,

    /// External actuator channel indices currently bound by live `MotorPore`s.
    pub motor_pore_channels: Vec<u32>,
    /// Conductance strengths matching `motor_pore_channels`.
    pub motor_pore_conductances: Vec<f32>,

    /// Deterministic logical state bytes occupied by the organism's slabs (distinct from OS RSS).
    pub used_state_bytes: usize,
}

/// Fail-closed persistence error codes (numeric/enum only, zero string formatting needed in core).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistError {
    IoFailure,
    OversizedInput,
    TruncatedInput,
    BadMagicOrVersion,
    ChecksumMismatch,
    InvalidTopology,
    NonFiniteState,
    PoisonedAfterCorruption,
}

/// Universal interface implemented by the Alpha 11 organism substrate (`a11-core`).
pub trait OrganismEngine {
    /// Advances the organism by one causal tick given `view` and returns its emitted `FieldPulse`.
    /// Returns `Err(PersistError::PoisonedAfterCorruption)` if a prior failed restore poisoned this instance.
    fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError>;

    /// Sets or inspects the experimental plasticity control mode.
    fn set_control_mode(&mut self, mode: ControlMode);
    fn control_mode(&self) -> ControlMode;

    /// Flagless weight/conductance lesion: physically zeroes all learned synaptic/tube/pore
    /// conductances and homeostatic value weights in RAM without changing `ControlMode` or setting any flag.
    fn zero_learned_conductances(&mut self);

    ///Deterministic fingerprint of all learned parameters (for verifying strict `FrozenPlasticity` invariance).
    fn learned_parameter_hash(&self) -> u64;

    /// Read-only biophysical and topological snapshot.
    fn snapshot(&self) -> OrganismSnapshot;

    /// Serializes the complete organism state into versioned, CRC32C-protected slabs.
    fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError>;

    /// Shadow-decodes `bytes` completely in temporary memory, validates CRC32C, bounds, finiteness,
    /// and topology, and atomically swaps into `self`. On any error, `self` is NOT overwritten with
    /// partial state and becomes fail-closed (`PoisonedAfterCorruption`) against further stepping/saving.
    fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError>;

    /// Read-only verification: fully shadow-decodes `bytes` (bounds, CRC32C, finiteness, topology)
    /// WITHOUT touching `self`. Hosts must call this on an existing primary checkpoint before
    /// rotating it into the `.prev` generation, so corruption never overwrites a known-good backup.
    fn verify_checkpoint(&self, bytes: &[u8]) -> Result<(), PersistError>;

    /// Returns true if a rejected restore has fail-closed this engine instance.
    fn is_poisoned(&self) -> bool;
}

/// Universal interface for any single-organism physical or human-activity medium.
pub trait WorldMedium {
    /// Number of ambient input wave channels exposed by this medium.
    fn num_ambient_channels(&self) -> usize;

    /// Number of continuous actuator channels accepted by this medium.
    fn num_actuators(&self) -> usize;

    /// Current `FieldView` presented to the organism before its next `step()`.
    fn current_view(&self) -> FieldView<'_>;

    /// Advances world physics by one tick using the organism's emitted `FieldPulse`.
    fn apply_pulse(&mut self, pulse: &FieldPulse);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silent_pulse_matches_actuator_len() {
        let pulse = FieldPulse::silent(12);
        assert_eq!(pulse.actuator_out.len(), 12);
        assert!(pulse.actuator_out.iter().all(|&v| v == 0.0));
    }
}
