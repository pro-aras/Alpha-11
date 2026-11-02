//! In-world Layer-2 human-activity portal multiplexer (`CompositePortalMedium`).
//!
//! Wraps any primary survival habitat `W: WorldMedium` and attaches `1..=4` unforced
//! human-activity portals (`DrawingScratchpad`, `PatternMemoryBoard`, `RhythmEchoPad`,
//! `BalanceCatchToy`) into a unified `WorldMedium` channel and actuator space.
//!
//! Each attached portal is gated by a smooth continuous `portal_focus` engagement actuator
//! so an organism living in `W` can voluntarily engage a portal when `ATP`-replete and
//! disengage back to physical foraging whenever hungry. Attached portals NEVER add extra
//! `metabolic_influx` on top of the survival habitat `W`.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::{
    actuator_at,
    balance_toy::{BalanceCatchToy, BalanceToyObservation},
    pattern_board::{PatternBoardObservation, PatternMemoryBoard},
    rhythm_pad::{
        BinauralVocalOrgan, RhythmEchoPad, RhythmPadObservation, VOCAL_ACTUATORS,
        VOCAL_FRAME_SAMPLES, VOCAL_ORGAN_CHANNELS, VocalOrganObservation,
    },
    sanitize_nonneg, sanitize_unit_signed,
    scratchpad::{CANVAS_SIDE, DrawingScratchpad, ScratchpadObservation},
};

/// Peripheral sensory gain of a disengaged portal (`focus == 0.0`).
///
/// Set to `0.0` (audit H41): unfocused portals previously leaked 15% of their signal into the host's
/// channels, distracting cave senses. A portal is now only perceivable once the organism's own focus
/// actuator opens it, so any interest in it must come from action contingency, not passive leakage.
pub const PERIPHERAL_PORTAL_BEACON_GAIN: f32 = 0.0;

/// Typed Layer-2 human-activity portal attachment.
#[derive(Clone, Debug)]
pub enum ActivityPortal {
    Scratchpad(Box<DrawingScratchpad>),
    PatternBoard(Box<PatternMemoryBoard>),
    RhythmPad(Box<RhythmEchoPad>),
    BalanceToy(Box<BalanceCatchToy>),
}

impl From<DrawingScratchpad> for ActivityPortal {
    fn from(pad: DrawingScratchpad) -> Self {
        Self::Scratchpad(Box::new(pad))
    }
}

impl From<PatternMemoryBoard> for ActivityPortal {
    fn from(board: PatternMemoryBoard) -> Self {
        Self::PatternBoard(Box::new(board))
    }
}

impl From<RhythmEchoPad> for ActivityPortal {
    fn from(pad: RhythmEchoPad) -> Self {
        Self::RhythmPad(Box::new(pad))
    }
}

impl From<BalanceCatchToy> for ActivityPortal {
    fn from(toy: BalanceCatchToy) -> Self {
        Self::BalanceToy(Box::new(toy))
    }
}

impl ActivityPortal {
    /// Borrows the underlying [`DrawingScratchpad`] if this portal is `Scratchpad`.
    #[must_use]
    pub fn as_scratchpad(&self) -> Option<&DrawingScratchpad> {
        match self {
            Self::Scratchpad(p) => Some(p),
            _ => None,
        }
    }

    /// Borrows the underlying [`PatternMemoryBoard`] if this portal is `PatternBoard`.
    #[must_use]
    pub fn as_pattern_board(&self) -> Option<&PatternMemoryBoard> {
        match self {
            Self::PatternBoard(p) => Some(p),
            _ => None,
        }
    }

    /// Borrows the underlying [`RhythmEchoPad`] if this portal is `RhythmPad`.
    #[must_use]
    pub fn as_rhythm_pad(&self) -> Option<&RhythmEchoPad> {
        match self {
            Self::RhythmPad(p) => Some(p),
            _ => None,
        }
    }

    /// Borrows the underlying [`BalanceCatchToy`] if this portal is `BalanceToy`.
    #[must_use]
    pub fn as_balance_toy(&self) -> Option<&BalanceCatchToy> {
        match self {
            Self::BalanceToy(p) => Some(p),
            _ => None,
        }
    }
}

impl WorldMedium for ActivityPortal {
    fn num_ambient_channels(&self) -> usize {
        match self {
            Self::Scratchpad(p) => p.num_ambient_channels(),
            Self::PatternBoard(p) => p.num_ambient_channels(),
            Self::RhythmPad(p) => p.num_ambient_channels(),
            Self::BalanceToy(p) => p.num_ambient_channels(),
        }
    }

    fn num_actuators(&self) -> usize {
        match self {
            Self::Scratchpad(p) => p.num_actuators(),
            Self::PatternBoard(p) => p.num_actuators(),
            Self::RhythmPad(p) => p.num_actuators(),
            Self::BalanceToy(p) => p.num_actuators(),
        }
    }

    fn current_view(&self) -> FieldView<'_> {
        match self {
            Self::Scratchpad(p) => p.current_view(),
            Self::PatternBoard(p) => p.current_view(),
            Self::RhythmPad(p) => p.current_view(),
            Self::BalanceToy(p) => p.current_view(),
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        match self {
            Self::Scratchpad(p) => p.apply_pulse(pulse),
            Self::PatternBoard(p) => p.apply_pulse(pulse),
            Self::RhythmPad(p) => p.apply_pulse(pulse),
            Self::BalanceToy(p) => p.apply_pulse(pulse),
        }
    }
}

/// Read-only behavioral engagement summary across a [`CompositePortalMedium`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompositePortalObservation {
    pub total_ticks: u64,
    /// Ticks where each attached portal had `focus_level >= 0.25`.
    pub portal_engaged_ticks: Vec<u64>,
    /// Mean `focus_level` in `[0.0, 1.0]` for each attached portal.
    pub mean_portal_focus: Vec<f32>,
    pub scratchpad: Option<ScratchpadObservation>,
    pub pattern_board: Option<PatternBoardObservation>,
    pub rhythm_pad: Option<RhythmPadObservation>,
    pub balance_toy: Option<BalanceToyObservation>,
}

/// Multiplexes a primary habitat `W` with `1..=4` human-activity portals gated by `portal_focus` actuators.
///
/// ### Actuator Layout
/// - `0 .. host.num_actuators()`: Primary habitat `W` actuators.
/// - For each attached portal `k`:
///   - `1` `portal_focus` engagement actuator (positive values smoothly engage portal `k`; `<= 0.0` disengages).
///   - `portal_k.num_actuators()` portal actuators (scaled smoothly by `focus_level[k]`).
#[derive(Clone, Debug)]
pub struct CompositePortalMedium<W: WorldMedium> {
    host: W,
    portals: Vec<ActivityPortal>,
    focus_levels: Vec<f32>,

    total_ticks: u64,
    engaged_ticks: Vec<u64>,
    focus_sums: Vec<f64>,

    ambient_buf: Vec<f32>,
    cached_metabolic_influx: f32,
    cached_tissue_stress: f32,
    total_actuators: usize,
}

impl<W: WorldMedium> CompositePortalMedium<W> {
    /// Attaches up to `4` `ActivityPortal`s onto primary survival medium `host`.
    #[must_use]
    pub fn new(host: W, mut portals: Vec<ActivityPortal>) -> Self {
        if portals.len() > 4 {
            portals.truncate(4);
        }
        let n_portals = portals.len();
        let host_ambient = host.num_ambient_channels();
        let portals_ambient: usize = portals.iter().map(WorldMedium::num_ambient_channels).sum();

        let host_act = host.num_actuators();
        let portals_act: usize = portals
            .iter()
            .map(|p| 1 + p.num_actuators())
            .sum();

        let mut medium = Self {
            host,
            portals,
            focus_levels: vec![0.0; n_portals],
            total_ticks: 0,
            engaged_ticks: vec![0; n_portals],
            focus_sums: vec![0.0; n_portals],
            ambient_buf: vec![0.0; host_ambient + portals_ambient],
            cached_metabolic_influx: 0.0,
            cached_tissue_stress: 0.0,
            total_actuators: host_act + portals_act,
        };
        medium.refresh_cached_view();
        medium
    }

    /// Convenience builder attaching all 4 standard human-activity portals with deterministic sub-seeds.
    #[must_use]
    pub fn with_all_portals(host: W, seed: u64) -> Self {
        let portals = vec![
            ActivityPortal::from(DrawingScratchpad::with_basal_influx(
                seed ^ 0x1111_0001,
                0.0,
            )),
            ActivityPortal::from(PatternMemoryBoard::with_basal_influx(
                seed ^ 0x2222_0002,
                0.0,
            )),
            ActivityPortal::from(RhythmEchoPad::with_basal_influx(
                seed ^ 0x3333_0003,
                0.0,
            )),
            ActivityPortal::from(BalanceCatchToy::with_crash_jolt(
                seed ^ 0x4444_0004,
                0.0,
                0.0,
            )),
        ];
        Self::new(host, portals)
    }

    /// Shared reference to the primary habitat `W`.
    #[must_use]
    pub fn host(&self) -> &W {
        &self.host
    }

    /// Mutable reference to the primary habitat `W`. After mutating `host` externally,
    /// call [`Self::sync_view`] if `current_view()` will be read before the next `apply_pulse()`.
    pub fn host_mut(&mut self) -> &mut W {
        &mut self.host
    }

    /// Re-synchronizes the composite ambient/somatic buffer from `host` and attached `portals`.
    pub fn sync_view(&mut self) {
        self.refresh_cached_view();
    }

    /// Attached human-activity portals (`0..=4`).
    #[must_use]
    pub fn portals(&self) -> &[ActivityPortal] {
        &self.portals
    }

    /// Current smoothed engagement focus levels (`[0.0, 1.0]`, sum `<= 1.0`) for each attached portal.
    #[must_use]
    pub fn portal_focus_levels(&self) -> &[f32] {
        &self.focus_levels
    }

    /// Actuator index of the `portal_focus` channel for attached portal `portal_index`.
    #[must_use]
    pub fn portal_focus_actuator_index(&self, portal_index: usize) -> Option<usize> {
        if portal_index >= self.portals.len() {
            return None;
        }
        let mut idx = self.host.num_actuators();
        for k in 0..portal_index {
            idx += 1 + self.portals[k].num_actuators();
        }
        Some(idx)
    }

    /// Computes read-only composite engagement and per-portal observation metrics.
    #[must_use]
    pub fn observation(&self) -> CompositePortalObservation {
        let denom = self.total_ticks.max(1) as f64;
        let mean_portal_focus = self
            .focus_sums
            .iter()
            .map(|&s| (s / denom).clamp(0.0, 1.0) as f32)
            .collect();

        let mut scratchpad = None;
        let mut pattern_board = None;
        let mut rhythm_pad = None;
        let mut balance_toy = None;

        for p in &self.portals {
            match p {
                ActivityPortal::Scratchpad(s) => scratchpad = Some(s.observation()),
                ActivityPortal::PatternBoard(b) => pattern_board = Some(b.observation()),
                ActivityPortal::RhythmPad(r) => rhythm_pad = Some(r.observation()),
                ActivityPortal::BalanceToy(t) => balance_toy = Some(t.observation()),
            }
        }

        CompositePortalObservation {
            total_ticks: self.total_ticks,
            portal_engaged_ticks: self.engaged_ticks.clone(),
            mean_portal_focus,
            scratchpad,
            pattern_board,
            rhythm_pad,
            balance_toy,
        }
    }

    fn refresh_cached_view(&mut self) {
        let host_view = self.host.current_view();
        // Metabolic influx comes strictly from the survival habitat W (never from portals).
        self.cached_metabolic_influx = sanitize_nonneg(host_view.metabolic_influx);
        let mut stress = sanitize_nonneg(host_view.tissue_stress);

        let expected_len = self.host.num_ambient_channels()
            + self
                .portals
                .iter()
                .map(WorldMedium::num_ambient_channels)
                .sum::<usize>();
        if self.ambient_buf.len() != expected_len {
            self.ambient_buf.resize(expected_len, 0.0);
        }

        let mut offset = 0;
        let host_n = self.host.num_ambient_channels();
        for i in 0..host_n {
            let v = host_view.ambient_in.get(i).copied().unwrap_or(0.0);
            self.ambient_buf[offset + i] = sanitize_unit_signed(v);
        }
        offset += host_n;

        for (k, portal) in self.portals.iter().enumerate() {
            let focus = self.focus_levels[k].clamp(0.0, 1.0);
            let gain = PERIPHERAL_PORTAL_BEACON_GAIN + (1.0 - PERIPHERAL_PORTAL_BEACON_GAIN) * focus;
            let p_view = portal.current_view();
            stress += focus * sanitize_nonneg(p_view.tissue_stress);

            let p_n = portal.num_ambient_channels();
            for i in 0..p_n {
                let v = p_view.ambient_in.get(i).copied().unwrap_or(0.0);
                self.ambient_buf[offset + i] = sanitize_unit_signed(v * gain);
            }
            offset += p_n;
        }

        self.cached_tissue_stress = sanitize_nonneg(stress);
        self.total_actuators = self.host.num_actuators()
            + self
                .portals
                .iter()
                .map(|p| 1 + p.num_actuators())
                .sum::<usize>();
    }
}

impl<W: WorldMedium> WorldMedium for CompositePortalMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.ambient_buf.len()
    }

    fn num_actuators(&self) -> usize {
        self.total_actuators
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.cached_metabolic_influx,
            tissue_stress: self.cached_tissue_stress,
            num_actuators: self.total_actuators,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.total_ticks = self.total_ticks.saturating_add(1);
        let host_act_count = self.host.num_actuators();

        // 1. Update smoothed engagement focus levels from each portal's portal_focus actuator.
        let mut act_cursor = host_act_count;
        for (k, portal) in self.portals.iter().enumerate() {
            let raw_focus = actuator_at(pulse, act_cursor).max(0.0);
            self.focus_levels[k] = (0.70 * self.focus_levels[k] + 0.30 * raw_focus).clamp(0.0, 1.0);
            act_cursor += 1 + portal.num_actuators();
        }

        // Soft-normalize attention if sum of portal focus exceeds 1.0.
        let focus_sum: f32 = self.focus_levels.iter().sum();
        if focus_sum > 1.0 {
            let inv = 1.0 / focus_sum;
            for f in &mut self.focus_levels {
                *f *= inv;
            }
        }
        let total_focus = focus_sum.min(1.0);

        for (k, &f) in self.focus_levels.iter().enumerate() {
            self.focus_sums[k] += f64::from(f);
            if f >= 0.25 {
                self.engaged_ticks[k] = self.engaged_ticks[k].saturating_add(1);
            }
        }

        // 2. Step primary habitat W: fully unattenuated when disengaged (total_focus == 0),
        //    gently damped while deeply absorbed in a portal.
        let host_gain = (1.0 - 0.65 * total_focus).clamp(0.35, 1.0);
        let mut host_pulse = FieldPulse {
            actuator_out: Vec::with_capacity(host_act_count),
        };
        for i in 0..host_act_count {
            host_pulse
                .actuator_out
                .push(sanitize_unit_signed(actuator_at(pulse, i) * host_gain));
        }
        self.host.apply_pulse(&host_pulse);

        // 3. Step each attached human-activity portal with focus-gated actuators.
        let mut p_cursor = host_act_count;
        for (k, portal) in self.portals.iter_mut().enumerate() {
            let focus = self.focus_levels[k];
            let n_act = portal.num_actuators();
            let mut portal_pulse = FieldPulse {
                actuator_out: Vec::with_capacity(n_act),
            };
            for j in 0..n_act {
                let raw = actuator_at(pulse, p_cursor + 1 + j);
                portal_pulse
                    .actuator_out
                    .push(sanitize_unit_signed(raw * focus));
            }
            portal.apply_pulse(&portal_pulse);
            p_cursor += 1 + n_act;
        }

        self.refresh_cached_view();
    }
}

// =============================================================================
// Always-Coupled Whole-Body Embodied Organs (Hands + Vestibular + 16 kHz Voice)
// =============================================================================

/// Hand/stylus sensory channels (`36` `6x6` pooled canvas cells + `16` tactile/velocity/`3x3` foveal).
pub const HAND_ORGAN_CHANNELS: usize = 52;
/// Vestibular & postural balance channels (`16` pooled Newtonian cradle wave channels).
pub const VESTIBULAR_ORGAN_CHANNELS: usize = 16;
/// Total body-organ sensory channels (`52` hand + `16` vestibular + `80` `16 kHz` binaural cochlear/LSO/OU).
pub const EMBODIED_ORGAN_CHANNELS: usize =
    HAND_ORGAN_CHANNELS + VESTIBULAR_ORGAN_CHANNELS + VOCAL_ORGAN_CHANNELS;

/// Hand/stylus muscles (`dx`, `dy`, `pressure`, `pigment_tone`).
pub const HAND_ORGAN_ACTUATORS: usize = 4;
/// Vestibular postural muscles (`torque`, `damping_grip`).
pub const VESTIBULAR_ORGAN_ACTUATORS: usize = 2;
/// Total body-organ muscles (`4` hand + `2` vestibular + `5` vocal articulators = `11`).
pub const EMBODIED_ORGAN_ACTUATORS: usize =
    HAND_ORGAN_ACTUATORS + VESTIBULAR_ORGAN_ACTUATORS + VOCAL_ACTUATORS;

/// Read-only developmental observation across all coupled body organs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmbodiedBodyObservation {
    pub scratchpad: ScratchpadObservation,
    pub balance: BalanceToyObservation,
    pub vocal: VocalOrganObservation,
}

/// Always-coupled physical body organs (`DrawingScratchpad` hands + `BalanceCatchToy` vestibular
/// balance + `BinauralVocalOrgan` `16 kHz` vocal tract & binaural cochlea).
///
/// Unlike `CompositePortalMedium`, there is NO `focus == 0` blindfold: the organism's hands,
/// vestibular balance, vocal cords, and ears are physically attached to its body from birth.
/// Zero artificial `metabolic_influx` is ever injected by body organs.
#[derive(Clone, Debug)]
pub struct EmbodiedBodyOrgans {
    scratchpad: DrawingScratchpad,
    balance: BalanceCatchToy,
    vocal: BinauralVocalOrgan,
    organ_ambient: [f32; EMBODIED_ORGAN_CHANNELS],
    last_organ_stress: f32,
}

impl EmbodiedBodyOrgans {
    /// Creates infant-scale (`8.8 cm` vocal tract) whole-body organs.
    #[must_use]
    pub fn new_infant(seed: u64) -> Self {
        Self::with_vocal(seed, BinauralVocalOrgan::new_infant(seed ^ 0x5555_0005))
    }

    /// Creates adult-scale (`14.5 cm` vocal tract) whole-body organs.
    #[must_use]
    pub fn new_adult(seed: u64) -> Self {
        Self::with_vocal(seed, BinauralVocalOrgan::new_adult(seed ^ 0x6666_0006))
    }

    #[must_use]
    fn with_vocal(seed: u64, vocal: BinauralVocalOrgan) -> Self {
        let scratchpad = DrawingScratchpad::with_basal_influx(seed ^ 0x1111_0001, 0.0);
        let balance = BalanceCatchToy::with_crash_jolt(seed ^ 0x4444_0004, 0.0, 0.0);
        let mut organs = Self {
            scratchpad,
            balance,
            vocal,
            organ_ambient: [0.0; EMBODIED_ORGAN_CHANNELS],
            last_organ_stress: 0.0,
        };
        organs.refresh_organ_ambient();
        organs
    }

    /// Enables live `16 kHz` rolling `10 s` mono/stereo + newborn-first-`10 s` PCM capture.
    pub fn enable_pcm_recording(&mut self) {
        self.vocal.enable_pcm_recording();
    }

    /// Steps hand stylus + vestibular balance and radiates this organism's `320`-sample `16 kHz` voice
    /// before binaural air mixing, allowing cohabiting organisms to hear each other on the same tick.
    pub fn step_limbs_and_radiate_voice(
        &mut self,
        body_actuators: &[f32],
    ) -> [f32; VOCAL_FRAME_SAMPLES] {
        let get_a = |i: usize| -> f32 {
            body_actuators
                .get(i)
                .copied()
                .map_or(0.0, sanitize_unit_signed)
        };

        let hand_pulse = FieldPulse {
            actuator_out: vec![get_a(0), get_a(1), get_a(2), get_a(3), 0.0],
        };
        self.scratchpad.apply_pulse(&hand_pulse);

        let bal_pulse = FieldPulse {
            actuator_out: vec![get_a(4), get_a(5), 0.0],
        };
        self.balance.apply_pulse(&bal_pulse);

        let vocal_cmd = [get_a(6), get_a(7), get_a(8), get_a(9), get_a(10)];
        self.vocal.radiate_voice(&vocal_cmd)
    }

    /// Completes binaural `16 kHz` cochlear transduction with an optional peer wave arriving from
    /// relative 2D world offset `(dx, dy)` and refreshes all `148` body sensory channels.
    pub fn finish_binaural_transduction(
        &mut self,
        peer_source: Option<(&[f32; VOCAL_FRAME_SAMPLES], (f32, f32))>,
    ) {
        self.vocal.transduce_binaural_air(peer_source, true);
        self.refresh_organ_ambient();
    }

    /// Completes binaural `16 kHz` cochlear transduction including spatialized Earth soundscapes
    /// (`env_ears`) and stone-shelter wall reverberation (`shelter_enclosure`), then refreshes all
    /// `148` body sensory channels.
    pub fn finish_binaural_ecology_transduction(
        &mut self,
        peer_source: Option<(&[f32; VOCAL_FRAME_SAMPLES], (f32, f32))>,
        env_ears: Option<(&[f32; VOCAL_FRAME_SAMPLES], &[f32; VOCAL_FRAME_SAMPLES])>,
        shelter_enclosure: f32,
    ) {
        self.vocal
            .transduce_binaural_ecology(peer_source, env_ears, shelter_enclosure, true);
        self.refresh_organ_ambient();
    }

    /// Instantaneous postural stability (`[0.25, 1.0]`) and mechanical hand work (`[0.0, 1.5]`)
    /// exerted on external world objects this tick.
    #[must_use]
    pub fn embodied_kinematics(&self) -> (f32, f32) {
        (
            self.balance.upright_stability(),
            self.scratchpad.mechanical_hand_work(),
        )
    }

    /// Couples terrain acceleration/slope sway into the vestibular limb and refreshes receptors.
    pub fn apply_terrain_sway(&mut self, lateral_accel: f32) {
        self.balance.apply_terrain_sway(lateral_accel);
        self.refresh_organ_ambient();
    }

    /// Couples two cohabiting organisms' hand surfaces onto a shared physical stone hearth slab
    /// when within tactile reach (`coupling in [0.0, 0.35]`).
    pub fn couple_shared_hands(a: &mut Self, b: &mut Self, coupling: f32) {
        a.scratchpad.couple_shared_surface(&mut b.scratchpad, coupling);
        a.refresh_organ_ambient();
        b.refresh_organ_ambient();
    }

    fn refresh_organ_ambient(&mut self) {
        let sp_view = self.scratchpad.current_view();
        // 0..36: 2x2 spatial pooling of the 12x12 pigment canvas (`6 x 6 = 36` channels).
        for gy in 0..6 {
            for gx in 0..6 {
                let r0 = gy * 2;
                let c0 = gx * 2;
                let mut sum = 0.0_f32;
                for dr in 0..2 {
                    for dc in 0..2 {
                        let idx = (r0 + dr) * CANVAS_SIDE + (c0 + dc);
                        sum += sp_view.ambient_in.get(idx).copied().unwrap_or(0.0);
                    }
                }
                self.organ_ambient[gy * 6 + gx] = sanitize_unit_signed(0.25 * sum);
            }
        }
        // 36..52: 16 stylus tactile, velocity, and local 3x3 foveal feedback channels.
        for i in 0..16 {
            let v = sp_view.ambient_in.get(144 + i).copied().unwrap_or(0.0);
            self.organ_ambient[36 + i] = sanitize_unit_signed(v);
        }

        // 52..68: 16 pooled Newtonian vestibular/tilt wave channels (`96 / 6 = 16`).
        let bal_view = self.balance.current_view();
        self.last_organ_stress = sanitize_nonneg(bal_view.tissue_stress);
        for g in 0..VESTIBULAR_ORGAN_CHANNELS {
            let mut sum = 0.0_f32;
            for k in 0..6 {
                sum += bal_view.ambient_in.get(g * 6 + k).copied().unwrap_or(0.0);
            }
            self.organ_ambient[HAND_ORGAN_CHANNELS + g] = sanitize_unit_signed(sum / 6.0);
        }

        // 68..148: 80 binaural 16 kHz cochlear + brainstem LSO + OU(0.97) channels.
        let voc_ch = self.vocal.sensory_channels();
        self.organ_ambient[(HAND_ORGAN_CHANNELS + VESTIBULAR_ORGAN_CHANNELS)..]
            .copy_from_slice(voc_ch);
    }

    /// Latest `148` unlabeled body-organ sensory channels in `[-1.0, 1.0]`.
    #[must_use]
    pub fn sensory_channels(&self) -> &[f32; EMBODIED_ORGAN_CHANNELS] {
        &self.organ_ambient
    }

    /// Physical somatic jolt stress from losing postural balance on the latest tick (`>= 0.0`).
    #[must_use]
    pub fn last_organ_stress(&self) -> f32 {
        self.last_organ_stress
    }

    /// Shared reference to the `16 kHz` vocal & binaural cochlear organ.
    #[must_use]
    pub fn vocal(&self) -> &BinauralVocalOrgan {
        &self.vocal
    }

    /// Mutable reference to the `16 kHz` vocal & binaural cochlear organ.
    pub fn vocal_mut(&mut self) -> &mut BinauralVocalOrgan {
        &mut self.vocal
    }

    /// Monotonically advances anatomical vocal-tract length & laryngeal descent (`8.8 cm -> 14.5 cm`)
    /// from physiological age (`age_ticks`), endogenous cellular growth (`mitosis_births`), and vocal mastery.
    pub fn advance_vocal_ontogeny(
        &mut self,
        age_ticks: u64,
        mitosis_births: u64,
        vocal_mastery: f32,
    ) {
        self.vocal
            .advance_ontogeny(age_ticks, mitosis_births, vocal_mastery);
    }

    /// Shared reference to the hand pigment canvas.
    #[must_use]
    pub fn scratchpad(&self) -> &DrawingScratchpad {
        &self.scratchpad
    }

    /// Shared reference to the vestibular balance limb.
    #[must_use]
    pub fn balance(&self) -> &BalanceCatchToy {
        &self.balance
    }

    /// Read-only developmental observation across hands, balance, and `16 kHz` voice.
    #[must_use]
    pub fn observation(&self) -> EmbodiedBodyObservation {
        EmbodiedBodyObservation {
            scratchpad: self.scratchpad.observation(),
            balance: self.balance.observation(),
            vocal: self.vocal.observation(),
        }
    }
}

/// Couples any primary habitat `W: WorldMedium` directly with [`EmbodiedBodyOrgans`] (`+148` sensory
/// channels, `+11` muscles) with zero focus blindfold and zero artificial reward injection.
#[derive(Clone, Debug)]
pub struct WholeBodyEmbodiedMedium<W: WorldMedium> {
    host: W,
    organs: EmbodiedBodyOrgans,
    ambient_buf: Vec<f32>,
    cached_metabolic_influx: f32,
    cached_tissue_stress: f32,
    total_actuators: usize,
}

impl<W: WorldMedium> WholeBodyEmbodiedMedium<W> {
    /// Attaches infant-scale whole-body organs (`8.8 cm` vocal tract) onto survival habitat `host`.
    #[must_use]
    pub fn new_infant(host: W, seed: u64) -> Self {
        Self::with_organs(host, EmbodiedBodyOrgans::new_infant(seed))
    }

    /// Attaches adult-scale whole-body organs (`14.5 cm` vocal tract) onto survival habitat `host`.
    #[must_use]
    pub fn new_adult(host: W, seed: u64) -> Self {
        Self::with_organs(host, EmbodiedBodyOrgans::new_adult(seed))
    }

    #[must_use]
    fn with_organs(host: W, organs: EmbodiedBodyOrgans) -> Self {
        let host_n = host.num_ambient_channels();
        let host_act = host.num_actuators();
        let mut medium = Self {
            host,
            organs,
            ambient_buf: vec![0.0; host_n + EMBODIED_ORGAN_CHANNELS],
            cached_metabolic_influx: 0.0,
            cached_tissue_stress: 0.0,
            total_actuators: host_act + EMBODIED_ORGAN_ACTUATORS,
        };
        medium.refresh_cached_view();
        medium
    }

    /// Shared reference to the survival habitat `W`.
    #[must_use]
    pub fn host(&self) -> &W {
        &self.host
    }

    /// Mutable reference to the survival habitat `W`.
    pub fn host_mut(&mut self) -> &mut W {
        &mut self.host
    }

    /// Shared reference to the coupled body organs.
    #[must_use]
    pub fn organs(&self) -> &EmbodiedBodyOrgans {
        &self.organs
    }

    /// Mutable reference to the coupled body organs.
    pub fn organs_mut(&mut self) -> &mut EmbodiedBodyOrgans {
        &mut self.organs
    }

    /// Re-synchronizes the unified ambient/somatic buffer after external host mutation.
    pub fn sync_view(&mut self) {
        self.refresh_cached_view();
    }

    fn refresh_cached_view(&mut self) {
        let host_view = self.host.current_view();
        self.cached_metabolic_influx = sanitize_nonneg(host_view.metabolic_influx);
        self.cached_tissue_stress =
            sanitize_nonneg(host_view.tissue_stress + self.organs.last_organ_stress());

        let host_n = self.host.num_ambient_channels();
        let expected_len = host_n + EMBODIED_ORGAN_CHANNELS;
        if self.ambient_buf.len() != expected_len {
            self.ambient_buf.resize(expected_len, 0.0);
        }
        for i in 0..host_n {
            let v = host_view.ambient_in.get(i).copied().unwrap_or(0.0);
            self.ambient_buf[i] = sanitize_unit_signed(v);
        }
        self.ambient_buf[host_n..].copy_from_slice(self.organs.sensory_channels());
        self.total_actuators = self.host.num_actuators() + EMBODIED_ORGAN_ACTUATORS;
    }
}

impl<W: WorldMedium> WorldMedium for WholeBodyEmbodiedMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.ambient_buf.len()
    }

    fn num_actuators(&self) -> usize {
        self.total_actuators
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.cached_metabolic_influx,
            tissue_stress: self.cached_tissue_stress,
            num_actuators: self.total_actuators,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let host_act = self.host.num_actuators();
        let mut host_pulse = FieldPulse {
            actuator_out: Vec::with_capacity(host_act),
        };
        for i in 0..host_act {
            host_pulse.actuator_out.push(actuator_at(pulse, i));
        }
        self.host.apply_pulse(&host_pulse);

        let mut body_act = [0.0_f32; EMBODIED_ORGAN_ACTUATORS];
        for (j, slot) in body_act.iter_mut().enumerate() {
            *slot = actuator_at(pulse, host_act + j);
        }
        let _ = self.organs.step_limbs_and_radiate_voice(&body_act);
        self.organs.finish_binaural_transduction(None);
        self.refresh_cached_view();
    }
}
