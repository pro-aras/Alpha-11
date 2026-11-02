//! Unforced human-activity surfaces, Layer-2 portal multiplexer, and crash-safe live host for Alpha 11.
//!
//! All mediums in this crate strictly obey the Alpha 11 causal & thermodynamic boundary contract:
//! - Coordinate-free, modality-free `FieldView` slices bounded in `[-1.0, 1.0]`.
//! - Zero staged game-score rewards (`metabolic_influx` is strictly invariant to what the organism
//!   draws, matches, plays, or balances) and zero screen-flicker bonuses.
//! - Built-in exogenous `Noisy TV` static channels on every portal for agency vs distractor analysis.
//! - Read-only behavioral observation metrics that inspect emergent trajectories without mutating physics.

pub mod balance_toy;
pub mod composite;
pub mod live_host;
pub mod pattern_board;
pub mod rhythm_pad;
pub mod scratchpad;

pub use balance_toy::{
    BALANCE_TOY_ACTUATORS, BALANCE_TOY_AMBIENT_CHANNELS, BalanceCatchToy, BalanceToyObservation,
};
pub use composite::{
    ActivityPortal, CompositePortalMedium, CompositePortalObservation, EMBODIED_ORGAN_ACTUATORS,
    EMBODIED_ORGAN_CHANNELS, EmbodiedBodyObservation, EmbodiedBodyOrgans, HAND_ORGAN_ACTUATORS,
    HAND_ORGAN_CHANNELS, VESTIBULAR_ORGAN_ACTUATORS, VESTIBULAR_ORGAN_CHANNELS,
    WholeBodyEmbodiedMedium,
};
pub use live_host::{
    LiveHostConfig, LiveHostError, LiveHostRunner, LiveHostStats, RecoveryOutcome,
    format_telemetry_jsonl, read_bounded_file, read_linux_rss_bytes,
};
pub use pattern_board::{
    PATTERN_BOARD_ACTUATORS, PATTERN_BOARD_AMBIENT_CHANNELS, PatternBoardObservation,
    PatternMemoryBoard,
};
pub use rhythm_pad::{
    BinauralVocalOrgan, ErbCochlea32, PhysicalVocalTract, RHYTHM_PAD_ACTUATORS,
    RHYTHM_PAD_AMBIENT_CHANNELS, RhythmEchoPad, RhythmPadObservation, VOCAL_ACTUATORS,
    VOCAL_ERB_BANDS, VOCAL_FFT_SIZE, VOCAL_FRAME_SAMPLES, VOCAL_ORGAN_CHANNELS, VOCAL_RING_SAMPLES,
    VOCAL_SR_HZ, VocalOrganObservation, encode_wav_16khz_mono, encode_wav_16khz_stereo,
    fft_radix2_512, measure_frame_acoustics,
};
pub use scratchpad::{
     DrawingScratchpad, SCRATCHPAD_ACTUATORS, SCRATCHPAD_AMBIENT_CHANNELS, ScratchpadObservation,
};

use a11_medium::FieldPulse;

/// Deterministic pure-Rust `SplitMix64` PRNG for reproducible portal physics and Noisy-TV channels.
#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0.0, 1.0)`.
    #[inline]
    pub(crate) fn next_f32_unit(&mut self) -> f32 {
        let upper = (self.next_u64() >> 40) as u32;
        (upper as f32) * (1.0 / ((1_u32 << 24) as f32))
    }

    /// Uniform `f32` in `[-1.0, 1.0)`.
    #[inline]
    pub(crate) fn next_f32_signed(&mut self) -> f32 {
        self.next_f32_unit() * 2.0 - 1.0
    }

    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[inline]
pub(crate) fn sanitize_unit_signed(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn sanitize_nonneg(v: f32) -> f32 {
    if v.is_finite() && v > 0.0 {
        v
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn actuator_at(pulse: &FieldPulse, idx: usize) -> f32 {
    pulse
        .actuator_out
        .get(idx)
        .copied()
        .map_or(0.0, sanitize_unit_signed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use a11_medium::{ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError, WorldMedium};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn assert_valid_view(view: FieldView<'_>, expected_channels: usize, expected_actuators: usize) {
        assert_eq!(view.ambient_in.len(), expected_channels);
        assert_eq!(view.num_actuators, expected_actuators);
        assert!(view.metabolic_influx.is_finite() && view.metabolic_influx >= 0.0);
        assert!(view.tissue_stress.is_finite() && view.tissue_stress >= 0.0);
        for (idx, &ch) in view.ambient_in.iter().enumerate() {
            assert!(
                ch.is_finite() && (-1.0..=1.0).contains(&ch),
                "channel {idx} out of bounds: {ch}"
            );
        }
    }

    // =========================================================================
    // 1. DrawingScratchpad Verification
    // =========================================================================

    #[test]
    fn scratchpad_channels_settling_delay_and_metabolic_invariance() {
        let mut active = DrawingScratchpad::new(42);
        let mut silent = DrawingScratchpad::new(42);
        let mut random = DrawingScratchpad::new(42);
        let mut rng = SplitMix64::new(999);

        assert_valid_view(
            active.current_view(),
            SCRATCHPAD_AMBIENT_CHANNELS,
            SCRATCHPAD_ACTUATORS,
        );

        // Verify exact 2-tick stroke-to-pigment settling dynamics:
        // Tick 0 -> apply stroke pulse -> after 1 tick settled pigment is still 0.0 -> after 2nd tick settled > 0.0.
        let stroke = FieldPulse {
            actuator_out: vec![0.6, 0.3, 0.9, 0.4, 0.0],
        };
        active.apply_pulse(&stroke);
        assert!(
            active.settled_pigment().iter().all(|&p| p == 0.0),
            "Pigment must still be wet at t+1 before 2-tick settling completes"
        );

        active.apply_pulse(&FieldPulse::silent(SCRATCHPAD_ACTUATORS));
        let max_settled: f32 = active
            .settled_pigment()
            .iter()
            .copied()
            .fold(0.0, f32::max);
        assert!(
            max_settled > 0.05,
            "Pigment must settle onto durable canvas at t+2, got {max_settled}"
        );

        // Reset and compare 120 ticks of active drawing vs silent vs random + adversarial NaN/Inf pulses.
        let mut active = DrawingScratchpad::new(77);
        for t in 0..120 {
            let angle = t as f32 * 0.08;
            let p_active = FieldPulse {
                actuator_out: vec![angle.cos() * 0.7, angle.sin() * 0.7, 0.95, 0.5, 0.0],
            };
            let p_silent = FieldPulse::silent(SCRATCHPAD_ACTUATORS);
            let p_rand = if t == 50 {
                FieldPulse {
                    actuator_out: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 99.0, -99.0],
                }
            } else {
                FieldPulse {
                    actuator_out: (0..SCRATCHPAD_ACTUATORS)
                        .map(|_| rng.next_f32_signed())
                        .collect(),
                }
            };

            active.apply_pulse(&p_active);
            silent.apply_pulse(&p_silent);
            random.apply_pulse(&p_rand);

            let va = active.current_view();
            let vs = silent.current_view();
            let vr = random.current_view();

            assert_valid_view(va, SCRATCHPAD_AMBIENT_CHANNELS, SCRATCHPAD_ACTUATORS);
            assert_valid_view(vs, SCRATCHPAD_AMBIENT_CHANNELS, SCRATCHPAD_ACTUATORS);
            assert_valid_view(vr, SCRATCHPAD_AMBIENT_CHANNELS, SCRATCHPAD_ACTUATORS);

            // Strict reward purity: identical metabolic_influx across optimal, silent, and random actions!
            assert_eq!(va.metabolic_influx, vs.metabolic_influx);
            assert_eq!(va.metabolic_influx, vr.metabolic_influx);
            assert_eq!(va.tissue_stress, 0.0);
        }
    }

    #[test]
    fn scratchpad_structured_line_beats_teleporting_dots_and_detects_revisits() {
        // Arm A: Continuous structured sweeping line across canvas + return pass to revisit marks.
        let mut line_pad = DrawingScratchpad::new(101);
        // Move to left side first with lifted stylus
        for _ in 0..4 {
            line_pad.apply_pulse(&FieldPulse {
                actuator_out: vec![-1.0, -0.5, 0.0, 0.0, 0.0],
            });
        }
        // Draw continuous diagonal line across the canvas
        for _ in 0..24 {
            line_pad.apply_pulse(&FieldPulse {
                actuator_out: vec![0.65, 0.35, 0.95, 0.3, 0.0],
            });
        }
        let obs_one_pass = line_pad.observation();

        // Now reverse direction to revisit and inspect/extend its own prior line
        for _ in 0..24 {
            line_pad.apply_pulse(&FieldPulse {
                actuator_out: vec![-0.65, -0.35, 0.85, -0.2, 0.0],
            });
        }
        // Let final stroke settle
        line_pad.apply_pulse(&FieldPulse::silent(SCRATCHPAD_ACTUATORS));
        let obs_line = line_pad.observation();

        // Arm B: Random teleporting dots (glide across canvas with pressure=0 to random positions, tap 1 tick).
        let mut dots_pad = DrawingScratchpad::new(101);
        let mut rng = SplitMix64::new(2026);
        for _ in 0..10 {
            let target_x = rng.next_f32_unit() * 11.0;
            let target_y = rng.next_f32_unit() * 11.0;
            for _ in 0..4 {
                let (sx, sy) = dots_pad.stylus_position();
                let ux = ((target_x - sx) * 0.35).clamp(-1.0, 1.0);
                let uy = ((target_y - sy) * 0.35).clamp(-1.0, 1.0);
                dots_pad.apply_pulse(&FieldPulse {
                    actuator_out: vec![ux, uy, 0.0, 0.0, 0.0],
                });
            }
            dots_pad.apply_pulse(&FieldPulse {
                actuator_out: vec![0.0, 0.0, 0.95, rng.next_f32_signed(), 0.0],
            });
        }
        dots_pad.apply_pulse(&FieldPulse::silent(SCRATCHPAD_ACTUATORS));
        let obs_dots = dots_pad.observation();

        assert!(
            obs_line.active_stroke_ticks > 20,
            "Expected active stroke ticks > 20, got {}",
            obs_line.active_stroke_ticks
        );
        assert!(
            obs_line.canvas_coverage_fraction > 0.08,
            "Expected meaningful canvas coverage, got {}",
            obs_line.canvas_coverage_fraction
        );
        assert!(
            obs_line.spatial_autocorrelation > obs_dots.spatial_autocorrelation + 0.15,
            "Structured continuous lines ({:.3}) must exceed random teleporting dots ({:.3}) in Moran's I",
            obs_line.spatial_autocorrelation,
            obs_dots.spatial_autocorrelation
        );
        assert!(
            obs_line.revisit_ratio > obs_one_pass.revisit_ratio,
            "Returning over prior marks ({:.3}) must increase revisit_ratio over single forward pass ({:.3})",
            obs_line.revisit_ratio,
            obs_one_pass.revisit_ratio
        );
    }

    // =========================================================================
    // 2. PatternMemoryBoard Verification
    // =========================================================================

    fn drive_cursor_to_tile(board: &mut PatternMemoryBoard, target_tile: usize) {
        let target_x = (target_tile % 4) as f32;
        let target_y = (target_tile / 4) as f32;
        for _ in 0..8 {
            let (cx, cy) = board.cursor_position();
            let ux = ((target_x - cx) * 1.4).clamp(-1.0, 1.0);
            let uy = ((target_y - cy) * 1.4).clamp(-1.0, 1.0);
            board.apply_pulse(&FieldPulse {
                actuator_out: vec![ux, uy, 0.0, 0.0],
            });
        }
    }

    #[test]
    fn pattern_board_pair_resonance_lock_and_metabolic_invariance() {
        let mut solver = PatternMemoryBoard::new(31415);
        let mut silent = PatternMemoryBoard::new(31415);
        let mut random = PatternMemoryBoard::new(31415);
        let mut rng = SplitMix64::new(777);

        assert_valid_view(
            solver.current_view(),
            PATTERN_BOARD_AMBIENT_CHANNELS,
            PATTERN_BOARD_ACTUATORS,
        );

        // Locate a matching pair from external test helper (never exposed in FieldView).
        let base_pair = solver.tile_pair_id(0).unwrap();
        let partner = (1..16)
            .find(|&idx| solver.tile_pair_id(idx) == Some(base_pair))
            .expect("Every tile must have a matching pair");

        // Move to tile 0 and flip it
        drive_cursor_to_tile(&mut solver, 0);
        solver.apply_pulse(&FieldPulse {
            actuator_out: vec![0.0, 0.0, 0.9, 0.5],
        });
        assert_eq!(solver.observation().tiles_inspected, 1);

        // Move to partner within the 12-tick decay window and flip it
        drive_cursor_to_tile(&mut solver, partner);
        solver.apply_pulse(&FieldPulse {
            actuator_out: vec![0.0, 0.0, 0.9, 0.5],
        });

        let obs = solver.observation();
        assert_eq!(
            obs.matched_resonances_locked, 1,
            "Touching matching tiles within 12 ticks must lock harmonic resonance"
        );
        assert!(obs.unique_tiles_visited >= 2);

        // Step silent and random boards for the same tick count and verify strict metabolic invariance.
        let total_steps = solver.tick();
        for _ in 0..total_steps {
            silent.apply_pulse(&FieldPulse::silent(PATTERN_BOARD_ACTUATORS));
            let p_rand = FieldPulse {
                actuator_out: (0..PATTERN_BOARD_ACTUATORS)
                    .map(|_| rng.next_f32_signed())
                    .collect(),
            };
            random.apply_pulse(&p_rand);

            assert_valid_view(
                silent.current_view(),
                PATTERN_BOARD_AMBIENT_CHANNELS,
                PATTERN_BOARD_ACTUATORS,
            );
            assert_valid_view(
                random.current_view(),
                PATTERN_BOARD_AMBIENT_CHANNELS,
                PATTERN_BOARD_ACTUATORS,
            );
        }

        assert_valid_view(
            solver.current_view(),
            PATTERN_BOARD_AMBIENT_CHANNELS,
            PATTERN_BOARD_ACTUATORS,
        );
        assert_eq!(
            solver.current_view().metabolic_influx,
            silent.current_view().metabolic_influx
        );
        assert_eq!(
            solver.current_view().metabolic_influx,
            random.current_view().metabolic_influx
        );
        assert_eq!(solver.current_view().tissue_stress, 0.0);
    }

    // =========================================================================
    // 3. RhythmEchoPad Verification
    // =========================================================================

    #[test]
    fn rhythm_pad_call_response_echo_and_metabolic_invariance() {
        let mut responder = RhythmEchoPad::new(808);
        let mut masher = RhythmEchoPad::new(808);
        let mut silent = RhythmEchoPad::new(808);

        assert_valid_view(
            responder.current_view(),
            RHYTHM_PAD_AMBIENT_CHANNELS,
            RHYTHM_PAD_ACTUATORS,
        );

        // Run 3 full 36-tick bars (108 ticks):
        // - `responder` listens quietly during call window (0..18) and plays exact motif on beats 18, 24, 30.
        // - `masher` mashes all 4 pads constantly every tick.
        // - `silent` emits zero pulses.
        for _ in 0..(rhythm_pad::CYCLE_TICKS * 3) {
            let phase = responder.tick() % rhythm_pad::CYCLE_TICKS;
            let motif = responder.active_motif();

            let mut resp_act = vec![0.0_f32; RHYTHM_PAD_ACTUATORS];
            if phase == 18 {
                resp_act[motif[0]] = 0.95;
            } else if phase == 24 {
                resp_act[motif[1]] = 0.95;
            } else if phase == 30 {
                resp_act[motif[2]] = 0.95;
            }

            responder.apply_pulse(&FieldPulse {
                actuator_out: resp_act,
            });
            masher.apply_pulse(&FieldPulse {
                actuator_out: vec![0.95, 0.95, 0.95, 0.95, 0.0],
            });
            silent.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));

            let vr = responder.current_view();
            let vm = masher.current_view();
            let vs = silent.current_view();
            assert_valid_view(vr, RHYTHM_PAD_AMBIENT_CHANNELS, RHYTHM_PAD_ACTUATORS);
            assert_valid_view(vm, RHYTHM_PAD_AMBIENT_CHANNELS, RHYTHM_PAD_ACTUATORS);
            assert_valid_view(vs, RHYTHM_PAD_AMBIENT_CHANNELS, RHYTHM_PAD_ACTUATORS);

            assert_eq!(vr.metabolic_influx, vs.metabolic_influx);
            assert_eq!(vm.metabolic_influx, vs.metabolic_influx);
            assert_eq!(vr.tissue_stress, 0.0);
        }

        let obs_resp = responder.observation();
        let obs_mash = masher.observation();
        let obs_silent = silent.observation();

        assert_eq!(obs_resp.total_pad_strikes, 9);
        assert_eq!(obs_silent.total_pad_strikes, 0);
        assert!(
            obs_resp.listening_vs_playing_ratio > 0.95,
            "Turn-taking responder should achieve ~1.0 listening_vs_playing_ratio, got {}",
            obs_resp.listening_vs_playing_ratio
        );
        assert!(
            obs_mash.listening_vs_playing_ratio < 0.10,
            "Constant masher should collapse listening_vs_playing_ratio, got {}",
            obs_mash.listening_vs_playing_ratio
        );
        assert!(
            obs_resp.phase_locking_value > 0.95,
            "Beat-locked responder PLV should exceed 0.95, got {}",
            obs_resp.phase_locking_value
        );
        assert!(
            obs_mash.phase_locking_value < 0.05,
            "Constant DC mashing PLV should cancel near 0, got {}",
            obs_mash.phase_locking_value
        );
        assert!(
            obs_resp.motif_continuation_similarity > 0.95,
            "Exact A->B->C motif continuation should score > 0.95, got {}",
            obs_resp.motif_continuation_similarity
        );
    }

    // =========================================================================
    // 4. BalanceCatchToy Verification
    // =========================================================================

    #[test]
    fn balance_toy_predictive_counter_torque_outlasts_zero_and_random_torque() {
        let mut predictive = BalanceCatchToy::new(2025);
        let mut passive = BalanceCatchToy::new(2025);
        let mut random = BalanceCatchToy::new(2025);
        let mut rng = SplitMix64::new(555);

        assert_valid_view(
            predictive.current_view(),
            BALANCE_TOY_AMBIENT_CHANNELS,
            BALANCE_TOY_ACTUATORS,
        );

        for _ in 0..300 {
            let (_angle, _avel, ball_pos, ball_vel) = predictive.physical_state();
            let counter_torque =
                (-(ball_pos + balance_toy::PHASE_LEAD_TAU_TICKS * ball_vel) * 1.1).clamp(-1.0, 1.0);

            predictive.apply_pulse(&FieldPulse {
                actuator_out: vec![counter_torque, 0.3, 0.0],
            });
            passive.apply_pulse(&FieldPulse::silent(BALANCE_TOY_ACTUATORS));
            random.apply_pulse(&FieldPulse {
                actuator_out: vec![
                    rng.next_f32_signed(),
                    rng.next_f32_signed(),
                    rng.next_f32_signed(),
                ],
            });

            let vp = predictive.current_view();
            let vz = passive.current_view();
            let vr = random.current_view();
            assert_valid_view(vp, BALANCE_TOY_AMBIENT_CHANNELS, BALANCE_TOY_ACTUATORS);
            assert_valid_view(vz, BALANCE_TOY_AMBIENT_CHANNELS, BALANCE_TOY_ACTUATORS);
            assert_valid_view(vr, BALANCE_TOY_AMBIENT_CHANNELS, BALANCE_TOY_ACTUATORS);

            assert_eq!(vp.metabolic_influx, vz.metabolic_influx);
            assert_eq!(vp.metabolic_influx, vr.metabolic_influx);
            assert_eq!(vp.tissue_stress, 0.0);
        }

        let obs_pred = predictive.observation();
        let obs_zero = passive.observation();
        let obs_rand = random.observation();

        assert_eq!(obs_pred.sphere_falls, 0);
        assert!(
            obs_zero.sphere_falls >= 4,
            "Uncontrolled cradle must drop the rolling sphere repeatedly, falls={}",
            obs_zero.sphere_falls
        );
        assert!(
            obs_pred.mean_upright_duration_ticks > obs_zero.mean_upright_duration_ticks * 3.0,
            "Predictive counter-torque ({:.1}) must yield much longer upright duration than zero torque ({:.1})",
            obs_pred.mean_upright_duration_ticks,
            obs_zero.mean_upright_duration_ticks
        );
        assert!(
            obs_pred.mean_upright_duration_ticks > obs_rand.mean_upright_duration_ticks * 2.0,
            "Predictive counter-torque ({:.1}) must beat random torque ({:.1})",
            obs_pred.mean_upright_duration_ticks,
            obs_rand.mean_upright_duration_ticks
        );
        assert!(
            obs_pred.torque_phase_lead_correlation > 0.85,
            "Predictive controller must register strong positive phase-lead correlation, got {}",
            obs_pred.torque_phase_lead_correlation
        );
        assert_eq!(obs_zero.torque_phase_lead_correlation, 0.0);
    }

    // =========================================================================
    // 5. CompositePortalMedium Verification
    // =========================================================================

    struct MockHabitat {
        ambient: Vec<f32>,
        influx: f32,
        stress: f32,
    }

    impl MockHabitat {
        fn new() -> Self {
            Self {
                ambient: vec![0.25; 32],
                influx: 0.021,
                stress: 0.003,
            }
        }
    }

    impl WorldMedium for MockHabitat {
        fn num_ambient_channels(&self) -> usize {
            self.ambient.len()
        }

        fn num_actuators(&self) -> usize {
            3
        }

        fn current_view(&self) -> FieldView<'_> {
            FieldView {
                ambient_in: &self.ambient,
                metabolic_influx: self.influx,
                tissue_stress: self.stress,
                num_actuators: 3,
            }
        }

        fn apply_pulse(&mut self, pulse: &FieldPulse) {
            let f0 = actuator_at(pulse, 0);
            for (i, ch) in self.ambient.iter_mut().enumerate() {
                *ch = ((f0 + i as f32 * 0.1).sin() * 0.8).clamp(-1.0, 1.0);
            }
        }
    }

    #[test]
    fn composite_portal_multiplexes_cleanly_without_nans_or_extra_influx() {
        let host = MockHabitat::new();
        let mut comp = CompositePortalMedium::with_all_portals(host, 9001);

        let expected_ambient = 32
            + SCRATCHPAD_AMBIENT_CHANNELS
            + PATTERN_BOARD_AMBIENT_CHANNELS
            + RHYTHM_PAD_AMBIENT_CHANNELS
            + BALANCE_TOY_AMBIENT_CHANNELS;
        let expected_actuators = 3
            + (1 + SCRATCHPAD_ACTUATORS)
            + (1 + PATTERN_BOARD_ACTUATORS)
            + (1 + RHYTHM_PAD_ACTUATORS)
            + (1 + BALANCE_TOY_ACTUATORS);

        assert_valid_view(comp.current_view(), expected_ambient, expected_actuators);
        assert_eq!(comp.current_view().metabolic_influx, 0.021);

        // Engage Scratchpad (portal 0) strongly for 20 ticks, then disengage back to foraging for 20 ticks.
        let sp_focus_idx = comp.portal_focus_actuator_index(0).unwrap();
        for _ in 0..20 {
            let mut out = vec![0.0; expected_actuators];
            out[sp_focus_idx] = 1.0; // portal_focus for Scratchpad
            out[sp_focus_idx + 1] = 0.7; // dx
            out[sp_focus_idx + 2] = 0.4; // dy
            out[sp_focus_idx + 3] = 0.95; // pressure
            comp.apply_pulse(&FieldPulse { actuator_out: out });

            let v = comp.current_view();
            assert_valid_view(v, expected_ambient, expected_actuators);
            // Portals NEVER inject extra metabolic_influx on top of MockHabitat!
            assert_eq!(v.metabolic_influx, 0.021);
        }

        assert!(comp.portal_focus_levels()[0] > 0.85);
        assert!(
            comp.observation()
                .scratchpad
                .unwrap()
                .active_stroke_ticks
                > 10
        );

        // Disengage back to foraging (all portal_focus = 0.0)
        for _ in 0..25 {
            let mut out = vec![0.0; expected_actuators];
            out[0] = 0.8; // forage in host habitat
            comp.apply_pulse(&FieldPulse { actuator_out: out });
            assert_valid_view(comp.current_view(), expected_ambient, expected_actuators);
        }
        assert!(
            comp.portal_focus_levels()[0] < 0.01,
            "Portal focus must decay smoothly near 0 upon disengagement"
        );
    }

    // =========================================================================
    // 6. LiveHostRunner Telemetry, Rotation & .prev Shadow-Recovery Verification
    // =========================================================================

    const MOCK_MAGIC: [u8; 4] = *b"A11M";
    const MOCK_VERSION: u32 = 1;

    /// Pure-Rust Castagnoli CRC32C for verifying realistic fail-closed checkpoint corruption handling.
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

    #[derive(Clone, Default)]
    struct MockOrganism {
        tick: u64,
        atp: f32,
        mode: ControlModeCompat,
        conductance_sum: f32,
        poisoned: bool,
    }

    #[derive(Default, Clone, Copy)]
    struct ControlModeCompat(Option<ControlMode>);

    impl MockOrganism {
        fn new() -> Self {
            Self {
                tick: 0,
                atp: 1.0,
                mode: ControlModeCompat(Some(ControlMode::Natural)),
                conductance_sum: 0.5,
                poisoned: false,
            }
        }
    }

    impl OrganismEngine for MockOrganism {
        fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }
            self.tick += 1;
            self.atp = (self.atp + view.metabolic_influx - 0.005).clamp(0.1, 2.0);
            if self.control_mode() == ControlMode::Natural {
                self.conductance_sum += 0.001;
            }
            let mut out = vec![0.0; view.num_actuators];
            if !out.is_empty() {
                out[0] = ((self.tick as f32) * 0.15).sin() * 0.5;
            }
            Ok(FieldPulse { actuator_out: out })
        }

        fn set_control_mode(&mut self, mode: ControlMode) {
            self.mode = ControlModeCompat(Some(mode));
        }

        fn control_mode(&self) -> ControlMode {
            self.mode.0.unwrap_or(ControlMode::Natural)
        }

        fn zero_learned_conductances(&mut self) {
            self.conductance_sum = 0.0;
        }

        fn learned_parameter_hash(&self) -> u64 {
            u64::from(self.conductance_sum.to_bits())
        }

        fn snapshot(&self) -> OrganismSnapshot {
            OrganismSnapshot {
                tick: self.tick,
                atp: self.atp,
                adenosine: 0.12,
                temperature: 0.05,
                motor_clutch_threshold: 0.30,
                cell_count: 16,
                tube_count: 48,
                sensor_pore_count: 12,
                motor_pore_count: 5,
                mean_tau: 4.5,
                max_tau: 16.0,
                mean_interventional_agency: 0.24,
                subthreshold_rehearsals: self.tick / 2,
                mitosis_births: 2,
                contradiction_splits: 1,
                used_state_bytes: 4096,
                ..OrganismSnapshot::default()
            }
        }

        fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }
            let mut payload = Vec::with_capacity(28);
            payload.extend_from_slice(&MOCK_MAGIC);
            payload.extend_from_slice(&MOCK_VERSION.to_le_bytes());
            payload.extend_from_slice(&self.tick.to_le_bytes());
            payload.extend_from_slice(&self.atp.to_le_bytes());
            payload.extend_from_slice(&self.conductance_sum.to_le_bytes());
            let checksum = crc32c(&payload);
            payload.extend_from_slice(&checksum.to_le_bytes());
            Ok(payload)
        }

        fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }
            if bytes.len() < 28 {
                self.poisoned = true;
                return Err(PersistError::TruncatedInput);
            }
            if bytes[0..4] != MOCK_MAGIC {
                self.poisoned = true;
                return Err(PersistError::BadMagicOrVersion);
            }
            let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
            if ver != MOCK_VERSION {
                self.poisoned = true;
                return Err(PersistError::BadMagicOrVersion);
            }
            let expected_crc = crc32c(&bytes[0..24]);
            let actual_crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
            if expected_crc != actual_crc {
                self.poisoned = true;
                return Err(PersistError::ChecksumMismatch);
            }
            let tick = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
            let atp = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
            let cond = f32::from_le_bytes(bytes[20..24].try_into().unwrap());
            if !atp.is_finite() || !cond.is_finite() {
                self.poisoned = true;
                return Err(PersistError::NonFiniteState);
            }
            self.tick = tick;
            self.atp = atp;
            self.conductance_sum = cond;
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

    fn temp_test_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("a11_portals_{tag}_{}_{nanos}", std::process::id()))
    }

    #[test]
    fn live_host_writes_bounded_jsonl_and_recovers_from_prev_on_primary_corruption() {
        let dir = temp_test_dir("live_host_recovery");
        let mut cfg = LiveHostConfig::in_directory(&dir);
        cfg.telemetry_interval_ticks = 1;
        cfg.checkpoint_interval_ticks = 20;
        cfg.max_log_bytes = 900; // Small cap to trigger bounded rotation during test

        // 1. Genesis startup (neither primary nor .prev exists)
        let (mut runner, outcome) = LiveHostRunner::open_or_recover(
            cfg.clone(),
            DrawingScratchpad::new(11),
            MockOrganism::new,
        )
        .expect("Genesis startup must succeed");
        assert_eq!(outcome, RecoveryOutcome::Genesis);

        // Run 40 ticks -> saves checkpoint at tick 20 (becomes .prev at tick 40) and tick 40 (primary).
        runner.run_ticks(40).expect("Closed-loop run must succeed");
        let stats = runner.stats();
        assert_eq!(stats.steps_executed, 40);
        assert_eq!(stats.checkpoints_saved, 2);
        assert!(
            stats.log_rotations >= 1,
            "Expected bounded JSONL log rotation under 900-byte cap"
        );
        assert_eq!(stats.last_used_state_bytes, 4096);

        // Verify JSONL file size bound and required keys on every line
        let active_log = fs::read_to_string(&cfg.telemetry_path).expect("Active JSONL must exist");
        assert!(active_log.len() <= cfg.max_log_bytes);
        assert!(cfg.rotated_telemetry_path().exists());

        let required_keys = [
            "\"tick\":",
            "\"atp\":",
            "\"adenosine\":",
            "\"temperature\":",
            "\"cell_count\":",
            "\"tube_count\":",
            "\"sensor_pore_count\":",
            "\"mean_tau\":",
            "\"max_tau\":",
            "\"mean_interventional_agency\":",
            "\"subthreshold_rehearsals\":",
            "\"mitosis_births\":",
            "\"contradiction_splits\":",
            "\"used_state_bytes\":",
            "\"step_us\":",
        ];
        for line in active_log.lines() {
            assert!(line.starts_with('{') && line.ends_with('}'));
            for key in required_keys {
                assert!(line.contains(key), "Missing JSONL key {key} in line: {line}");
            }
        }

        // 2. Normal restart from intact primary (tick == 40)
        let (restored, outcome2) = LiveHostRunner::open_or_recover(
            cfg.clone(),
            DrawingScratchpad::new(11),
            MockOrganism::new,
        )
        .expect("Primary restore must succeed");
        assert!(matches!(outcome2, RecoveryOutcome::RestoredPrimary { .. }));
        assert_eq!(restored.engine().snapshot().tick, 40);

        // 3. Corrupt primary checkpoint with a bit-flip and verify clean shadow-recovery from .prev (tick == 20)!
        //    Note: MockOrganism permanently poisons any candidate that sees the corrupted primary,
        //    proving LiveHostRunner discards poisoned candidates and restores cleanly on a fresh shadow instance.
        let mut raw_primary = fs::read(&cfg.checkpoint_path).unwrap();
        raw_primary[10] ^= 0xFF;
        fs::write(&cfg.checkpoint_path, &raw_primary).unwrap();

        let (mut recovered, outcome3) = LiveHostRunner::open_or_recover(
            cfg.clone(),
            DrawingScratchpad::new(11),
            MockOrganism::new,
        )
        .expect("Must recover cleanly from .prev when primary is corrupted");
        assert_eq!(
            outcome3,
            RecoveryOutcome::RecoveredFromPrev {
                primary_error: PersistError::ChecksumMismatch,
                bytes_len: 28,
            }
        );
        assert_eq!(recovered.engine().snapshot().tick, 20);
        assert!(!recovered.engine().is_poisoned());

        // Recovered runner continues stepping cleanly from tick 20 -> 25
        recovered.run_ticks(5).expect("Stepping after .prev recovery must succeed");
        assert_eq!(recovered.engine().snapshot().tick, 25);

        // 4. Corrupt BOTH primary and .prev -> MUST fail closed (never reset to tick 0)!
        fs::write(&cfg.checkpoint_path, b"corrupt").unwrap();
        fs::write(cfg.prev_checkpoint_path(), b"also_corrupt").unwrap();
        let res = LiveHostRunner::open_or_recover(
            cfg,
            DrawingScratchpad::new(11),
            MockOrganism::new,
        );
        assert!(
            res.is_err(),
            "Must fail closed when both primary and .prev checkpoints are corrupted"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pattern_board_reveal_decays_after_12_ticks_and_tracks_revisit_intervals() {
        let mut board = PatternMemoryBoard::new(999);
        drive_cursor_to_tile(&mut board, 0);
        let concealed_sig = board.current_view().ambient_in[0..8].to_vec();

        // Touch tile 0 to reveal for 12 ticks
        board.apply_pulse(&FieldPulse {
            actuator_out: vec![0.0, 0.0, 0.9, 0.0],
        });
        let revealed_sig = board.current_view().ambient_in[0..8].to_vec();
        let diff_open: f32 = concealed_sig
            .iter()
            .zip(&revealed_sig)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_open > 0.5,
            "Touching concealed tile must reveal distinct harmonic signature"
        );

        // Step 12 silent ticks so reveal window expires cleanly back to concealed grain
        for _ in 0..12 {
            board.apply_pulse(&FieldPulse::silent(PATTERN_BOARD_ACTUATORS));
        }
        let after_decay = board.current_view().ambient_in[0..8].to_vec();
        let diff_closed: f32 = concealed_sig
            .iter()
            .zip(&after_decay)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_closed < 1e-4,
            "Single unmatched tile must decay back to concealed state after 12 ticks"
        );

        // Move away to tile 3 and return to tile 0 to verify repeat_visit_interval_mean > 0
        drive_cursor_to_tile(&mut board, 3);
        drive_cursor_to_tile(&mut board, 0);
        let obs = board.observation();
        assert!(obs.repeat_visit_interval_mean > 0.0);
        assert!((0.0..=1.0).contains(&obs.cursor_dwell_stability));
    }

    #[test]
    fn rhythm_pad_room_reverb_arrives_at_3_to_7_tick_delays() {
        let mut pad = RhythmEchoPad::new(1234);
        // Advance into the quiet window (tick 22) and let external call decay first
        while pad.tick() < 22 {
            pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));
        }
        let mut baseline_pad = pad.clone();

        // Strike pad 2 sharply on `pad`, keep `baseline_pad` silent
        pad.apply_pulse(&FieldPulse {
            actuator_out: vec![0.0, 0.0, 1.0, 0.0, 0.0],
        });
        baseline_pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));

        // 1 and 2 ticks after strike: 3..7 tick room reverb channels (64..96) must not yet differ!
        for d in 1..=2 {
            pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));
            baseline_pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));
            let reverb_diff: f32 = pad.current_view().ambient_in[64..96]
                .iter()
                .zip(&baseline_pad.current_view().ambient_in[64..96])
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                reverb_diff < 1e-6,
                "Room echo must not arrive early at delay {d}, got diff {reverb_diff}"
            );
        }

        // At 3..=7 ticks after strike: delayed room reverberation must arrive on channels 64..96!
        for d in 3..=7 {
            pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));
            baseline_pad.apply_pulse(&FieldPulse::silent(RHYTHM_PAD_ACTUATORS));
            let reverb_diff: f32 = pad.current_view().ambient_in[64..96]
                .iter()
                .zip(&baseline_pad.current_view().ambient_in[64..96])
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                reverb_diff > 0.02,
                "Expected delayed room reverberation arrival at delay {d}, got {reverb_diff}"
            );
        }
    }

    #[test]
    fn live_host_deterministic_200_tick_continuation_and_oversized_rejection() {
        let dir_a = temp_test_dir("det_uninterrupted");
        let dir_b = temp_test_dir("det_split");

        let mut cfg_a = LiveHostConfig::in_directory(&dir_a);
        cfg_a.checkpoint_interval_ticks = 0;
        let (mut runner_a, _) = LiveHostRunner::open_or_recover(
            cfg_a,
            BalanceCatchToy::new(77),
            MockOrganism::new,
        )
        .unwrap();
        runner_a.run_ticks(200).unwrap();

        let mut cfg_b = LiveHostConfig::in_directory(&dir_b);
        cfg_b.checkpoint_interval_ticks = 100;
        let (mut runner_b1, _) = LiveHostRunner::open_or_recover(
            cfg_b.clone(),
            BalanceCatchToy::new(77),
            MockOrganism::new,
        )
        .unwrap();
        runner_b1.run_ticks(100).unwrap();
        let saved_world = runner_b1.world().clone();
        drop(runner_b1);

        let (mut runner_b2, outcome) = LiveHostRunner::open_or_recover(
            cfg_b.clone(),
            saved_world,
            MockOrganism::new,
        )
        .unwrap();
        assert!(matches!(outcome, RecoveryOutcome::RestoredPrimary { .. }));
        runner_b2.run_ticks(100).unwrap();

        assert_eq!(runner_a.engine().snapshot(), runner_b2.engine().snapshot());
        assert_eq!(
            runner_a.engine().learned_parameter_hash(),
            runner_b2.engine().learned_parameter_hash()
        );

        // Verify bounded read rejects oversized checkpoint without unbounded allocation
        let mut tiny_cfg = cfg_b;
        tiny_cfg.max_checkpoint_bytes = 10;
        fs::remove_file(tiny_cfg.prev_checkpoint_path()).ok();
        let err = LiveHostRunner::open_or_recover(
            tiny_cfg,
            BalanceCatchToy::new(77),
            MockOrganism::new,
        );
        assert_eq!(
            err.err(),
            Some(LiveHostError::Persist(PersistError::OversizedInput))
        );

        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    // =========================================================================
    // 7. Pure Math & DSP Physics Verification: FFT, ERB Cochlea, Vocal Tract & WAV
    // =========================================================================

    #[test]
    fn vocal_dsp_fft_parseval_zero_lung_silence_and_wav_header() {
        // (a) Radix-2 512-pt FFT Parseval energy theorem & exact bin peak at 1,000 Hz (bin 32).
        let mut re = [0.0_f32; VOCAL_FFT_SIZE];
        let mut im = [0.0_f32; VOCAL_FFT_SIZE];
        let mut time_energy = 0.0_f64;
        for (n, slot) in re.iter_mut().enumerate() {
            let phase = 2.0 * std::f32::consts::PI * 1000.0 * (n as f32) / (VOCAL_SR_HZ as f32);
            *slot = phase.sin();
            time_energy += f64::from(*slot) * f64::from(*slot);
        }
        fft_radix2_512(&mut re, &mut im);
        let mut freq_energy = 0.0_f64;
        let mut peak_bin = 0usize;
        let mut peak_mag = -1.0_f32;
        for k in 0..VOCAL_FFT_SIZE {
            let p = re[k] * re[k] + im[k] * im[k];
            freq_energy += f64::from(p) / (VOCAL_FFT_SIZE as f64);
            if k <= VOCAL_FFT_SIZE / 2 && p > peak_mag {
                peak_mag = p;
                peak_bin = k;
            }
        }
        assert_eq!(peak_bin, 32, "1000 Hz @ 16 kHz / 512-pt FFT must peak at bin 32");
        assert!(
            (time_energy - freq_energy).abs() < 1e-2,
            "Parseval energy mismatch: time={time_energy} vs freq={freq_energy}"
        );

        // (b) Zero/negative subglottal lung command emits exact 0.0 Pa silence.
        let mut tract = PhysicalVocalTract::new_infant(42);
        let mut pcm = [1.0_f32; VOCAL_FRAME_SAMPLES];
        tract.synthesize_frame(&[0.0, 0.4, 0.3, -0.5, 0.2], &mut pcm);
        assert!(
            pcm.iter().all(|&x| x == 0.0),
            "Subthreshold lung drive must produce exact 0.0 acoustic pressure"
        );

        // (c) Voiced front vs back tongue positions produce distinct 16 kHz F2 formant peaks & bounded ERB channels.
        let mut cochlea = ErbCochlea32::new();
        let mut f2_front = 0.0_f32;
        for _ in 0..8 {
            tract.synthesize_frame(&[0.90, 0.1, -0.35, 0.90, 0.30], &mut pcm);
            let (fast, echo, flux) = cochlea.transduce_frame(&pcm);
            assert!(fast.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
            assert!(echo.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
            assert!(flux.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
            let (rms, _f0, _f1, f2) = measure_frame_acoustics(&pcm);
            assert!(rms > 0.03);
            f2_front = f2;
        }
        let mut tract_back = PhysicalVocalTract::new_infant(42);
        let mut f2_back = 0.0_f32;
        for _ in 0..8 {
            tract_back.synthesize_frame(&[0.90, 0.1, -0.35, -0.90, -0.30], &mut pcm);
            let (rms, _f0, _f1, f2) = measure_frame_acoustics(&pcm);
            assert!(rms > 0.03);
            f2_back = f2;
        }
        assert!(
            f2_front > f2_back + 700.0,
            "Front vowel F2 ({f2_front:.1} Hz) must exceed back vowel F2 ({f2_back:.1} Hz) by > 700 Hz"
        );

        // (d) 16 kHz mono and stereo PCM16 WAV headers.
        let mono_wav = encode_wav_16khz_mono(&pcm);
        assert_eq!(mono_wav.len(), 44 + VOCAL_FRAME_SAMPLES * 2);
        assert_eq!(&mono_wav[0..4], b"RIFF");
        assert_eq!(&mono_wav[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([mono_wav[22], mono_wav[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([mono_wav[24], mono_wav[25], mono_wav[26], mono_wav[27]]),
            VOCAL_SR_HZ
        );

        let stereo_wav = encode_wav_16khz_stereo(&[0.25, -0.25, 0.5, -0.5]);
        assert_eq!(stereo_wav.len(), 44 + 8);
        assert_eq!(u16::from_le_bytes([stereo_wav[22], stereo_wav[23]]), 2);

        // (e) BinauralVocalOrgan & WholeBodyEmbodiedMedium physical ITD/ILD & zero free ATP.
        let mut adult = BinauralVocalOrgan::new_adult(7);
        let mut infant = BinauralVocalOrgan::new_infant(8);
        let mut peer_wave = [0.0_f32; VOCAL_FRAME_SAMPLES];
        for _ in 0..4 {
            peer_wave = adult.radiate_voice(&[0.85, 0.2, 0.3, -0.4, 0.1]);
        }
        let mut infant_left = infant.clone();
        for _ in 0..4 {
            infant.step_organ(&[0.0; VOCAL_ACTUATORS], Some((&peer_wave, (4.0, 0.0))));
            infant_left.step_organ(&[0.0; VOCAL_ACTUATORS], Some((&peer_wave, (-4.0, 0.0))));
        }
        let lso_right_mean: f32 = infant.sensory_channels()[32..64].iter().sum::<f32>() / 32.0;
        let lso_left_mean: f32 = infant_left.sensory_channels()[32..64].iter().sum::<f32>() / 32.0;
        assert!(
            lso_right_mean > lso_left_mean + 0.08,
            "Right-side peer sound must drive higher LSO azimuth ({lso_right_mean}) than left-side ({lso_left_mean})"
        );

        let mut wb = WholeBodyEmbodiedMedium::new_infant(
            PatternMemoryBoard::with_basal_influx(99, 0.0),
            99,
        );
        assert_valid_view(
            wb.current_view(),
            PATTERN_BOARD_AMBIENT_CHANNELS + EMBODIED_ORGAN_CHANNELS,
            PATTERN_BOARD_ACTUATORS + EMBODIED_ORGAN_ACTUATORS,
        );
        wb.apply_pulse(&FieldPulse {
            actuator_out: vec![0.5; PATTERN_BOARD_ACTUATORS + EMBODIED_ORGAN_ACTUATORS],
        });
        assert_eq!(
            wb.current_view().metabolic_influx,
            0.0,
            "Embodied body organs must never inject artificial metabolic_influx"
        );

        // (f) Ontogenetic laryngeal descent grows vocal tract monotonically 8.8 cm -> 14.5 cm and deepens F0.
        let mut growing_tract = PhysicalVocalTract::new_infant(77);
        assert!((growing_tract.tract_length_cm() - 8.8).abs() < 1e-4);
        let (f0_newborn, f1_newborn, _, _) = growing_tract.current_formants_hz();
        growing_tract.advance_ontogeny(32_000, 120, 0.42);
        assert!(growing_tract.tract_length_cm() > 14.3);
        let mut adult_pcm = [0.0_f32; VOCAL_FRAME_SAMPLES];
        for _ in 0..4 {
            growing_tract.synthesize_frame(&[0.85, 0.0, 0.0, 0.0, 0.0], &mut adult_pcm);
        }
        let (f0_matured, f1_matured, _, _) = growing_tract.current_formants_hz();
        let (_, measured_adult_f0, _, _) = measure_frame_acoustics(&adult_pcm);
        assert!(
            f0_matured + 40.0 < f0_newborn && f1_matured < f1_newborn,
            "Laryngeal descent must deepen F0 ({f0_newborn:.1} -> {f0_matured:.1}) and F1 ({f1_newborn:.1} -> {f1_matured:.1})"
        );
        assert!(
            (measured_adult_f0 - f0_matured).abs() < 25.0,
            "YIN F0 estimator must track adult F0 without octave doubling (true={f0_matured:.1}, measured={measured_adult_f0:.1})"
        );

        // (g) Multi-syllable formant transition counter stays 0 on steady hum and increments on vowel glides.
        let mut mono_organ = BinauralVocalOrgan::new_adult(55);
        // Settle initial articulator spring transient for 6 frames, then reset window and hum steadily for 16 frames.
        for _ in 0..6 {
            mono_organ.step_organ(&[0.85, 0.0, 0.1, 0.0, 0.0], None);
        }
        mono_organ.reset_window_observations();
        for _ in 0..16 {
            mono_organ.step_organ(&[0.85, 0.0, 0.1, 0.0, 0.0], None);
        }
        assert_eq!(
            mono_organ.observation().syllable_transitions,
            0,
            "Steady monotone hum must register 0 multi-syllable formant transitions"
        );

        let mut contour_organ = BinauralVocalOrgan::new_adult(55);
        for _ in 0..8 {
            contour_organ.step_organ(&[0.85, 0.0, 0.75, -0.85, -0.35], None);
        }
        for _ in 0..8 {
            contour_organ.step_organ(&[0.85, 0.0, -0.65, 0.85, 0.35], None);
        }
        assert!(
            contour_organ.observation().syllable_transitions >= 1,
            "Back-to-front two-vowel articulatory contour must register >= 1 syllable_transitions"
        );
    }
}
