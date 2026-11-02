//! Continuous physical, wave-echo, harmonic, and multi-generational environments for Alpha 11.
//!
//! All environments obey strict biophysical & causal-integrity invariants:
//! - 100% pure safe Rust (`std` only, `unsafe_code = "forbid"`).
//! - Zero discrete movement grids, zero GPS coordinate channels, zero sprite-size or modality tags,
//!   and zero exogenous `visual_delta` rewards.
//! - Exact finite-speed time-of-flight wave-echo physics (`\Delta \in [2, 14]` ticks) with zero
//!   `t+0`/`t+1` motor self-reafference in empty space.

pub mod harmonic;
pub mod radiant_cave;
pub mod rng;
pub mod terrarium;
pub mod wavefield;
pub mod wrappers;

pub use harmonic::{HARMONIC_ACTUATORS, HARMONIC_CHANNELS, HarmonicMetrics, HarmonicWaveStream};
pub use radiant_cave::{
    AdaptiveCaveWorld, BrightRadiantWorld, CAVE_ACTUATORS, CaveMetrics, DEFAULT_ARENA_RADIUS,
    DEFAULT_CAVE_CHANNELS, DenseFieldWorld, MAX_DENSE_CAVE_CHANNELS, PitchBlackEchoCave,
    ResonantValve, SolidObstacle, ThermalVent,
};
pub use rng::DeterministicRng;
pub use terrarium::{
    BasaltRidge, DEFAULT_DAY_NIGHT_PERIOD, EARTH_AUDIO_FRAME_SAMPLES, EARTH_SEASON_PERIOD_TICKS,
    EarthEcologyMetrics, ForagePod, MARK_DECAY_PER_TICK, MARK_GRID_DIM, MAX_TERRARIUM_ORGANISMS,
    MultiGenTerrarium, PushableMatter, TERRARIUM_ACTUATORS, TERRARIUM_CHANNELS, TerrariumMetrics,
    TerrariumSpring, ThermalShelter, compute_day_night_lux,
};
pub use wavefield::{
    DEFAULT_MAX_ECHO_RANGE, DEFAULT_WAVE_SPEED, DISTANCE_ATTENUATION_COEFF, EMIT_PULSE_THRESHOLD,
    MAX_ECHO_DELAY_TICKS, MIN_ECHO_DELAY_TICKS, Reflector, ReflectorKind, WaveEchoRing,
    compute_distance_attenuation, compute_echo_delay_ticks, compute_one_way_delay_ticks,
};
pub use wrappers::{ActionShuffledMedium, PermutedMedium, YokedMedium};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use a11_medium::{
        ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError,
        WorldMedium,
    };

    use super::*;

    #[test]
    fn pitch_black_echo_cave_time_of_flight_and_attenuation() {
        let (photonic_band, echo_band, _noisy_band) = CaveMetrics::channel_band_ranges();

        // 1. Default PitchBlackEchoCave: Band A strictly 0.0, silent pulses keep Band B strictly 0.0.
        let mut default_cave = PitchBlackEchoCave::new(42);
        assert!(
            default_cave.current_view().ambient_in[photonic_band.clone()]
                .iter()
                .all(|&v| v == 0.0)
        );
        let silent = FieldPulse::silent(default_cave.num_actuators());
        for _ in 0..25 {
            default_cave.apply_pulse(&silent);
            let view = default_cave.current_view();
            assert!(
                view.ambient_in[photonic_band.clone()]
                    .iter()
                    .all(|&v| v == 0.0),
                "Photonic Band A (0..64) must remain strictly 0.0 in PitchBlackEchoCave"
            );
            assert!(
                view.ambient_in[echo_band.clone()]
                    .iter()
                    .all(|&v| v == 0.0),
                "Silent FieldPulse must yield strictly 0.0 in Echo Band B (64..96)"
            );
        }

        // 2. Isolated single-reflector sweep verifying exact delay \Delta in [2, 14],
        //    zero signal at t+1, directional tuning, and closer -> earlier & louder.
        let d_near = 3.3_f32; // round(2 * 3.3 / 2.2) = 3 ticks
        let d_far = 9.9_f32; // round(2 * 9.9 / 2.2) = 9 ticks
        let expected_near_delay = compute_echo_delay_ticks(d_near, DEFAULT_WAVE_SPEED);
        let expected_far_delay = compute_echo_delay_ticks(d_far, DEFAULT_WAVE_SPEED);
        assert_eq!(expected_near_delay, 3);
        assert_eq!(expected_far_delay, 9);

        let run_isolated_ping = |dist: f32, angle: f32, kind: ReflectorKind| -> (u64, f32, usize) {
            let rx = dist * angle.cos();
            let ry = dist * angle.sin();
            // Arena radius 40.0 > max_range 16.0 so outer wall does not reflect at origin.
            let mut cave = PitchBlackEchoCave::with_isolated_reflectors(
                777,
                40.0,
                &[(rx, ry, kind)],
            );
            let mut ping = FieldPulse::silent(cave.num_actuators());
            ping.actuator_out[4] = 0.90;
            ping.actuator_out[5] = 0.25;

            // Emit at t=0 -> after 1st apply_pulse, world is at t+1.
            cave.apply_pulse(&ping);
            let view_t1 = cave.current_view();
            assert!(
                view_t1.ambient_in[photonic_band.clone()]
                    .iter()
                    .all(|&v| v == 0.0)
            );
            assert!(
                view_t1.ambient_in[echo_band.clone()]
                    .iter()
                    .all(|&v| v == 0.0),
                "t+1 must be strictly 0.0 (zero immediate motor self-copy)"
            );

            let mut first_arrival_delay = 0_u64;
            let mut peak_abs = 0.0_f32;
            let mut peak_ch = 0_usize;

            for step_after in 2..=16_u64 {
                cave.apply_pulse(&silent);
                let view = cave.current_view();
                assert!(
                    view.ambient_in[photonic_band.clone()]
                        .iter()
                        .all(|&v| v == 0.0)
                );
                let echo_slice = &view.ambient_in[echo_band.clone()];
                let (ch, max_mag) = echo_slice
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v.abs()))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((0, 0.0));
                if max_mag > 1e-6 {
                    assert_eq!(
                        first_arrival_delay, 0,
                        "isolated single reflector should produce one clean echo arrival"
                    );
                    first_arrival_delay = step_after;
                    peak_abs = max_mag;
                    peak_ch = ch;
                }
            }
            (first_arrival_delay, peak_abs, peak_ch)
        };

        let (delay_near, amp_near, ch_east) =
            run_isolated_ping(d_near, 0.0, ReflectorKind::NourishingMatter);
        let (delay_far, amp_far, ch_north) = run_isolated_ping(
            d_far,
            std::f32::consts::FRAC_PI_2,
            ReflectorKind::NourishingMatter,
        );

        assert_eq!(delay_near, expected_near_delay);
        assert_eq!(delay_far, expected_far_delay);
        assert!(
            delay_near < delay_far,
            "closer object must arrive strictly earlier ({delay_near} vs {delay_far})"
        );
        assert!(
            amp_near > amp_far,
            "closer object must return louder echo ({amp_near} vs {amp_far})"
        );
        assert_eq!(ch_east, 0, "east (+x) reflector peaks at receiver channel 0");
        assert_eq!(
            ch_north, 8,
            "north (+y) reflector peaks at receiver channel 8 (90 deg of 32)"
        );

        // Verify distinct reflection phase between soft nourishing matter (+) and hard rock (-).
        let mut soft_cave = PitchBlackEchoCave::with_isolated_reflectors(
            19,
            40.0,
            &[(4.4, 0.0, ReflectorKind::NourishingMatter)],
        );
        let mut hard_cave = PitchBlackEchoCave::with_isolated_reflectors(
            19,
            40.0,
            &[(4.4, 0.0, ReflectorKind::HardObstacle)],
        );
        let mut ping = FieldPulse::silent(6);
        ping.actuator_out[4] = 0.85;
        soft_cave.apply_pulse(&ping);
        hard_cave.apply_pulse(&ping);
        for _ in 2..=4 {
            soft_cave.apply_pulse(&silent);
            hard_cave.apply_pulse(&silent);
        }
        let soft_ch0 = soft_cave.current_view().ambient_in[64];
        let hard_ch0 = hard_cave.current_view().ambient_in[64];
        assert!(soft_ch0 > 0.1, "nourishing matter returns positive phase");
        assert!(hard_ch0 < -0.1, "hard obstacle returns inverted negative phase");
    }

    #[test]
    fn noisy_tv_band_has_high_variance_zero_predictability_and_pulse_independence() {
        let (_photonic, _echo, noisy_band) = CaveMetrics::channel_band_ranges();
        let mut world_silent = BrightRadiantWorld::new(2026);
        let mut world_active = BrightRadiantWorld::new(2026);

        let silent = FieldPulse::silent(6);
        let steps = 600_usize;
        let num_noisy = noisy_band.len();
        let mut series = vec![Vec::with_capacity(steps); num_noisy];

        for t in 0..steps {
            let v_s = world_silent.current_view();
            let v_a = world_active.current_view();

            // Band C must be bit-for-bit identical regardless of FieldPulse history!
            assert_eq!(
                &v_s.ambient_in[noisy_band.clone()],
                &v_a.ambient_in[noisy_band.clone()],
                "Noisy TV Band C (96..128) must be completely independent of FieldPulse"
            );

            for (k, &val) in v_s.ambient_in[noisy_band.clone()].iter().enumerate() {
                series[k].push(val);
            }

            world_silent.apply_pulse(&silent);
            let active_pulse = FieldPulse {
                actuator_out: vec![
                    ((t as f32) * 0.17).sin(),
                    ((t as f32) * 0.11).cos(),
                    0.8,
                    -0.6,
                    if t % 3 == 0 { 0.9 } else { 0.0 },
                    -0.4,
                ],
            };
            world_active.apply_pulse(&active_pulse);
        }

        let mut mean_var = 0.0_f32;
        let mut mean_abs_lag1 = 0.0_f32;
        for ch_series in &series {
            let n = ch_series.len() as f32;
            let mean = ch_series.iter().sum::<f32>() / n;
            let var = ch_series.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
            assert!(
                var > 0.15,
                "each Noisy TV channel must have high variance (> 0.15), got {var}"
            );

            let mut cov1 = 0.0_f32;
            for w in ch_series.windows(2) {
                cov1 += (w[0] - mean) * (w[1] - mean);
            }
            cov1 /= (ch_series.len() - 1) as f32;
            let lag1_corr = cov1 / var.max(1e-6);
            assert!(
                lag1_corr.abs() < 0.12,
                "Noisy TV channel must have near-zero lag-1 autocorrelation, got {lag1_corr}"
            );

            mean_var += var;
            mean_abs_lag1 += lag1_corr.abs();
        }
        mean_var /= num_noisy as f32;
        mean_abs_lag1 /= num_noisy as f32;
        assert!(mean_var > 0.20);
        assert!(mean_abs_lag1 < 0.06);
    }

    #[test]
    fn dense_field_world_4096_steps_cleanly_without_nans() {
        let mut dense = DenseFieldWorld::new(99);
        assert_eq!(dense.num_ambient_channels(), 4096);
        assert_eq!(dense.num_actuators(), 6);

        for t in 0..48 {
            let view = dense.current_view();
            assert_eq!(view.ambient_in.len(), 4096);
            assert!(view.metabolic_influx.is_finite() && view.metabolic_influx >= 0.0);
            assert!(view.tissue_stress.is_finite() && view.tissue_stress >= 0.0);
            assert!(
                view.ambient_in.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
                "all 4,096 channels must remain finite and bounded in [-1, 1] at tick {t}"
            );

            if t == 16 {
                dense.set_illumination(0.0);
                let (photonic_range, _, _) = dense.band_ranges();
                assert!(
                    dense.current_view().ambient_in[photonic_range]
                        .iter()
                        .all(|&v| v == 0.0)
                );
            } else if t == 28 {
                dense.set_illumination(1.0);
                dense.trigger_causal_reversal();
            }

            let pulse = if t == 35 {
                // Inject adversarial NaN / Inf to verify fail-safe actuator sanitization!
                FieldPulse {
                    actuator_out: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 5.0, -9.0, 0.5],
                }
            } else {
                FieldPulse {
                    actuator_out: vec![
                        ((t as f32) * 0.2).cos(),
                        ((t as f32) * 0.2).sin(),
                        0.6,
                        0.4,
                        if t % 5 == 0 { 0.85 } else { 0.0 },
                        0.3,
                    ],
                }
            };
            dense.apply_pulse(&pulse);
        }
    }

    #[test]
    fn permuted_action_shuffled_and_yoked_wrappers_preserve_invariants() {
        let base = BrightRadiantWorld::new(31415);
        let mut permuted = PermutedMedium::new(base.clone(), 271828);
        let mut raw_world = base;

        assert_eq!(permuted.num_ambient_channels(), 128);
        assert_eq!(permuted.num_actuators(), 6);

        // Verify exact bijection over all 128 sensor channels and recovery of true bands.
        let mut recovered_photonic = BTreeSet::new();
        let mut recovered_echo = BTreeSet::new();
        let mut recovered_noisy = BTreeSet::new();
        for s_idx in 0..128 {
            let orig = permuted.invert_sensor_channel(s_idx);
            assert_eq!(permuted.scramble_sensor_channel(orig), s_idx);
            match orig {
                0..64 => {
                    recovered_photonic.insert(orig);
                }
                64..96 => {
                    recovered_echo.insert(orig);
                }
                96..128 => {
                    recovered_noisy.insert(orig);
                }
                _ => panic!("unexpected channel index {orig}"),
            }
        }
        assert_eq!(recovered_photonic.len(), 64);
        assert_eq!(recovered_echo.len(), 32);
        assert_eq!(recovered_noisy.len(), 32);

        // Verify motor + sensor permutation commutativity with true world physics!
        for step in 0..20 {
            let p_view = permuted.current_view();
            let r_view = raw_world.current_view();
            for s_idx in 0..128 {
                let true_idx = permuted.invert_sensor_channel(s_idx);
                assert_eq!(
                    p_view.ambient_in[s_idx], r_view.ambient_in[true_idx],
                    "scrambled channel must equal true underlying channel at step {step}"
                );
            }

            let raw_pulse = FieldPulse {
                actuator_out: vec![0.7, -0.3, 0.4, 0.2, if step == 2 { 0.9 } else { 0.0 }, -0.5],
            };
            let mut scrambled_pulse = FieldPulse::silent(6);
            for m_scrambled in 0..6 {
                let m_true = permuted.invert_motor_channel(m_scrambled);
                scrambled_pulse.actuator_out[m_scrambled] = raw_pulse.actuator_out[m_true];
            }
            raw_world.apply_pulse(&raw_pulse);
            permuted.apply_pulse(&scrambled_pulse);
        }

        // Verify ActionShuffledMedium preserves L2 norm on initial permute & stays bounded.
        let mut shuffled = ActionShuffledMedium::new(BrightRadiantWorld::new(7), 9001);
        let test_pulse = FieldPulse {
            actuator_out: vec![0.9, -0.4, 0.2, 0.0, -0.7, 0.1],
        };
        let expected_norm_sq: f32 = test_pulse.actuator_out.iter().map(|v| v * v).sum();
        shuffled.apply_pulse(&test_pulse);
        let applied_norm_sq: f32 = shuffled.last_applied_pulse().iter().map(|v| v * v).sum();
        assert!((expected_norm_sq - applied_norm_sq).abs() < 1e-5);

        // Verify YokedMedium overrides metabolic_influx & tissue_stress from donor stream.
        let donor = vec![(0.12, 0.01), (0.34, 0.05), (0.0, 0.22)];
        let mut yoked = YokedMedium::new(PitchBlackEchoCave::new(11), donor.clone());
        for &(exp_in, exp_st) in &donor {
            let v = yoked.current_view();
            assert!((v.metabolic_influx - exp_in).abs() < 1e-6);
            assert!((v.tissue_stress - exp_st).abs() < 1e-6);
            yoked.apply_pulse(&FieldPulse::silent(6));
        }
        // Past end of donor stream defaults cleanly to (0.0, 0.0).
        assert_eq!(yoked.current_view().metabolic_influx, 0.0);
        assert_eq!(yoked.current_view().tissue_stress, 0.0);
    }

    #[derive(Clone)]
    struct MockForagerAgent {
        id: usize,
        mode: ControlMode,
        poisoned: bool,
        tick: u64,
    }

    impl MockForagerAgent {
        fn new(id: usize) -> Self {
            Self {
                id,
                mode: ControlMode::Natural,
                poisoned: false,
                tick: 0,
            }
        }
    }

    impl OrganismEngine for MockForagerAgent {
        fn step(&mut self, view: FieldView<'_>) -> Result<FieldPulse, PersistError> {
            if self.poisoned {
                return Err(PersistError::PoisonedAfterCorruption);
            }
            assert_eq!(view.ambient_in.len(), TERRARIUM_CHANNELS);
            assert!(view.ambient_in.iter().all(|v| v.is_finite()));
            let t = self.tick as f32;
            self.tick += 1;

            // Agents commute between their home shelter (+-4.5, 0) and same-side spring (+-10.5, 6.5).
            let side_x = if self.id % 2 == 1 { 1.0 } else { -1.0 };
            let phase = (t * 0.06 + (self.id as f32) * 0.4).sin();
            let fx = side_x * 0.55 * phase;
            let fy = 0.45 * phase;
            let mark_x = 0.75;
            let mark_y = 0.45;
            let ping = if self.tick % 6 == (self.id as u64 % 6) {
                0.85
            } else {
                0.0
            };
            Ok(FieldPulse {
                actuator_out: vec![fx, fy, mark_x, mark_y, ping, 0.3],
            })
        }

        fn set_control_mode(&mut self, mode: ControlMode) {
            self.mode = mode;
        }

        fn control_mode(&self) -> ControlMode {
            self.mode
        }

        fn zero_learned_conductances(&mut self) {}

        fn learned_parameter_hash(&self) -> u64 {
            self.id as u64
        }

        fn snapshot(&self) -> OrganismSnapshot {
            OrganismSnapshot {
                tick: self.tick,
                ..OrganismSnapshot::default()
            }
        }

        fn encode_checkpoint(&self) -> Result<Vec<u8>, PersistError> {
            Ok(vec![1, 2, 3, 4])
        }

        fn restore_checkpoint(&mut self, bytes: &[u8]) -> Result<(), PersistError> {
            if bytes.is_empty() {
                self.poisoned = true;
                return Err(PersistError::TruncatedInput);
            }
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

    #[test]
    fn multigen_terrarium_day_night_cycles_and_persistent_cultural_marks() {
        let period = 80_u64;
        let mut terrarium = MultiGenTerrarium::with_day_night_period(404, 3, period);
        let mut agents = vec![
            MockForagerAgent::new(0),
            MockForagerAgent::new(1),
            MockForagerAgent::new(2),
        ];

        let mut saw_full_day = false;
        let mut saw_pitch_black_night = false;

        for step in 0..180 {
            if step == 90 {
                // Spawn a newborn into slot 2 mid-run while elders & trails persist!
                terrarium.respawn_newborn_in_slot(2);
                agents[2] = MockForagerAgent::new(2);
            }

            let pulses = terrarium
                .step_population(&mut agents)
                .expect("healthy mock organisms should step cleanly");
            assert_eq!(pulses.len(), 3);

            let lux = terrarium.illumination();
            if (lux - 1.0).abs() < 1e-5 {
                saw_full_day = true;
            }
            if lux == 0.0 {
                saw_pitch_black_night = true;
                // During pitch-black night, Photonic Band A (0..64) must be strictly 0.0 for all agents!
                for idx in 0..3 {
                    let v = terrarium.agent_view(idx);
                    assert!(
                        v.ambient_in[0..64].iter().all(|&x| x == 0.0),
                        "agent {idx} photonic band must be 0.0 during pitch-black night"
                    );
                }
            }
        }

        assert!(saw_full_day, "terrarium must reach bright day (lux = 1.0)");
        assert!(
            saw_pitch_black_night,
            "terrarium must reach pitch-black night (lux = 0.0)"
        );

        let m = terrarium.metrics();
        assert!(m.night_transitions >= 2);
        assert!(m.day_transitions >= 2);
        assert!(
            m.total_mark_mass > 0.4 && m.persistent_mark_coverage > 0.015,
            "elders commuting between shelter and spring must carve durable physical world marks, got mass={}, cov={}",
            m.total_mark_mass,
            m.persistent_mark_coverage
        );
        assert!(
            m.persistent_mark_coverage < 0.42,
            "trail marks must stay focused on active corridors without coating the floor, got {}",
            m.persistent_mark_coverage
        );
        assert!(
            m.mark_corridor_alignment > 0.35,
            "marks carved between shelter and spring should register strong corridor alignment, got {}",
            m.mark_corridor_alignment
        );
        assert!(m.agent_ages[0] > m.agent_ages[2], "elder must be older than respawned newborn");

        // Verify mark durability under MARK_DECAY_PER_TICK = 0.9982 (0.9982^200 ~= 0.6974).
        let silent_pop = vec![FieldPulse::silent(TERRARIUM_ACTUATORS); 3];
        // Settle residual agent inertia first so coasting does not deposit additional marks.
        for _ in 0..10 {
            terrarium.apply_population_pulses(&silent_pop);
        }
        let mass_before = terrarium.metrics().total_mark_mass;
        for _ in 0..200 {
            terrarium.apply_population_pulses(&silent_pop);
        }
        let mass_after = terrarium.metrics().total_mark_mass;
        let ratio = mass_after / mass_before;
        assert!(
            (0.68..=0.72).contains(&ratio),
            "0.9982/tick decay over 200 stationary silent ticks must retain ~69.7% mark mass, got {ratio}"
        );
    }

    #[test]
    fn harmonic_wave_stream_phase_locking_and_delayed_resonance() {
        let mut in_phase_world = HarmonicWaveStream::new(88);
        let mut anti_phase_world = HarmonicWaveStream::new(88);
        let (_, echo_band, _) = HarmonicMetrics::channel_band_ranges();

        // Check strict 3..=8 tick echo delay in Band B (48..64): t+1 and t+2 must be 0.0!
        let mut delay_check = HarmonicWaveStream::new(88);
        let silent = FieldPulse::silent(HARMONIC_ACTUATORS);
        let impulse = FieldPulse {
            actuator_out: vec![0.9, 0.0, 0.0, 0.0],
        };
        delay_check.apply_pulse(&impulse); // t+1
        assert!(
            delay_check.current_view().ambient_in[echo_band.clone()]
                .iter()
                .all(|&v| v == 0.0),
            "t+1 must be 0.0 in HarmonicWaveStream echo band"
        );
        delay_check.apply_pulse(&silent); // t+2
        assert!(
            delay_check.current_view().ambient_in[echo_band.clone()]
                .iter()
                .all(|&v| v == 0.0),
            "t+2 must be 0.0 in HarmonicWaveStream echo band"
        );
        delay_check.apply_pulse(&silent); // t+3
        assert!(
            delay_check.current_view().ambient_in[echo_band]
                .iter()
                .any(|&v| v.abs() > 0.1),
            "t+3 must receive resonant phase echo"
        );

        // Compare phase-locked harvesting vs anti-phase collision over 120 ticks.
        for _ in 0..120 {
            let (c1, s1, e1) = in_phase_world.current_feeding_crest();
            let norm1 = c1.hypot(s1).max(1e-5);
            in_phase_world.apply_pulse(&FieldPulse {
                actuator_out: vec![c1 / norm1, s1 / norm1, e1, 0.5],
            });

            let (c2, s2, e2) = anti_phase_world.current_feeding_crest();
            let norm2 = c2.hypot(s2).max(1e-5);
            anti_phase_world.apply_pulse(&FieldPulse {
                actuator_out: vec![-c2 / norm2, -s2 / norm2, -e2, 0.0],
            });
        }

        assert!(in_phase_world.metrics().total_metabolic_influx > 15.0);
        assert_eq!(in_phase_world.metrics().total_tissue_stress, 0.0);
        assert_eq!(anti_phase_world.metrics().total_metabolic_influx, 0.0);
        assert!(anti_phase_world.metrics().total_tissue_stress > 8.0);
    }

    #[test]
    fn bright_radiant_world_springs_valve_boost_and_wall_physics() {
        let mut world = BrightRadiantWorld::new(1234);
        assert!(
            world.thermal_vents().iter().any(|v| world.is_vent_nourishing(v)),
            "world must contain nourishing vents initially"
        );

        // Drive toward the resonant valve and trigger manipulation actuators 2..4.
        let valve_pos = world.valve().expect("valve present").pos;
        for _ in 0..45 {
            let pos = world.metrics().position;
            let dx = valve_pos.0 - pos.0;
            let dy = valve_pos.1 - pos.1;
            let d = dx.hypot(dy).max(1e-4);
            world.apply_pulse(&FieldPulse {
                actuator_out: vec![dx / d, dy / d, 0.9, 0.9, 0.0, 0.0],
            });
        }
        assert!(
            world.metrics().valve_activations >= 1,
            "manipulating near resonant valve must activate 80-tick boost"
        );
        assert!(world.metrics().valve_boost_ticks_remaining > 0);

        // Drive hard into the +x perimeter wall to verify continuous wall containment & impact stress.
        for _ in 0..140 {
            world.apply_pulse(&FieldPulse {
                actuator_out: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            });
        }
        assert!(world.metrics().wall_impacts >= 1);
        assert!(world.metrics().total_tissue_stress > 0.05);
        let final_r = world.metrics().position.0.hypot(world.metrics().position.1);
        assert!(
            final_r <= DEFAULT_ARENA_RADIUS,
            "organism must remain bounded inside circular arena radius"
        );
    }

    #[test]
    fn motionless_zero_motor_depletes_starter_halo_and_starves() {
        for seed in 0..12_u64 {
            let mut world = BrightRadiantWorld::new(1000 + seed);
            let silent = FieldPulse::silent(world.num_actuators());

            // Tick 1 provides gentle newborn ignition gradient from starter halo.
            world.apply_pulse(&silent);
            assert!(
                world.current_view().metabolic_influx > 0.001,
                "seed {seed}: newborn at origin must sense initial starter thermal halo"
            );

            // Step motionless through tick 98: starter halo must deplete (< 100 ticks) and relocate away!
            for _ in 1..98 {
                world.apply_pulse(&silent);
            }
            assert!(
                world.metrics().springs_depleted >= 1,
                "seed {seed}: motionless camping in starter halo must deplete and relocate starter spring"
            );

            // From tick 99 through tick 500, motionless organism must receive strictly 0.0 influx!
            for tick in 99..=500 {
                world.apply_pulse(&silent);
                assert_eq!(
                    world.current_view().metabolic_influx,
                    0.0,
                    "seed {seed} tick {tick}: motionless organism must receive strictly 0.0 influx after starter depletion"
                );
            }
        }
    }

    #[test]
    fn terrarium_somatic_gated_marks_resist_wallpaper_saturation_and_insulate_chill() {
        // 1. Verify barren-zone twitching without somatic warmth/nutrients deposits 0.0 wallpaper.
        let mut barren = MultiGenTerrarium::with_day_night_period(77, 1, 80);
        // Drive agent out to barren north wall (0, 18.5), far from shelters (+-4.5, 0) and springs (+-10, +-6.5).
        for _ in 0..90 {
            barren.apply_population_pulses(&[FieldPulse {
                actuator_out: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            }]);
        }
        // Let any residual carried trace decay completely while stationary at north wall.
        for _ in 0..80 {
            barren.apply_population_pulses(&[FieldPulse::silent(TERRARIUM_ACTUATORS)]);
        }
        let pos_north = barren.metrics().agent_positions[0];
        let mark_before_twitch = barren.sample_mark_at(pos_north);
        for _ in 0..40 {
            barren.apply_population_pulses(&[FieldPulse {
                actuator_out: vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            }]);
        }
        let mark_after_twitch = barren.sample_mark_at(pos_north);
        assert!(
            mark_after_twitch <= mark_before_twitch + 1e-6,
            "twitching in barren zone without carried warmth/nutrients must not deposit marks"
        );

        // 2. Verify 2,000-tick 3-agent foraging run maintains crisp corridor coverage (< 40%) & high alignment.
        let mut long_run = MultiGenTerrarium::with_day_night_period(404, 3, 160);
        let mut agents = vec![
            MockForagerAgent::new(0),
            MockForagerAgent::new(1),
            MockForagerAgent::new(2),
        ];
        for _ in 0..2000 {
            long_run
                .step_population(&mut agents)
                .expect("2000-tick multi-agent step succeeds");
        }
        let lm = long_run.metrics();
        assert!(
            lm.persistent_mark_coverage > 0.04 && lm.persistent_mark_coverage < 0.40,
            "2000-tick 3-agent run must keep crisp corridor coverage (< 40%), got {}",
            lm.persistent_mark_coverage
        );
        assert!(
            lm.mark_corridor_alignment > 0.45,
            "2000-tick 3-agent corridor alignment must remain focused (> 0.45), got {}",
            lm.mark_corridor_alignment
        );

        // 3. Verify standing on an inscribed corridor at night provides conductive chill insulation
        //    (+0.15 * mark_strength) with strictly 0.0 free metabolic_influx!
        let mut unmarked_night = MultiGenTerrarium::with_day_night_period(909, 1, 40);
        let mut marked_night = MultiGenTerrarium::with_day_night_period(909, 1, 40);
        // Move both agents out to (0.0, 7.5) — outside shelters & spring halos.
        for _ in 0..35 {
            let move_north = [FieldPulse {
                actuator_out: vec![0.42, 0.85, 0.0, 0.0, 0.0, 0.0],
            }];
            unmarked_night.apply_population_pulses(&move_north);
            let mark_move = [FieldPulse {
                actuator_out: vec![0.42, 0.85, 1.0, 1.0, 0.0, 0.0],
            }];
            marked_night.apply_population_pulses(&mark_move);
        }
        // Step through a night window while stationary.
        let mut stress_unmarked = 0.0_f32;
        let mut stress_marked = 0.0_f32;
        for _ in 0..40 {
            let s = [FieldPulse::silent(TERRARIUM_ACTUATORS)];
            unmarked_night.apply_population_pulses(&s);
            marked_night.apply_population_pulses(&s);
            if marked_night.illumination() == 0.0 {
                stress_unmarked += unmarked_night.agent_view(0).tissue_stress;
                stress_marked += marked_night.agent_view(0).tissue_stress;
                assert_eq!(
                    marked_night.agent_view(0).metabolic_influx,
                    0.0,
                    "inscribed floor marks must never grant free metabolic_influx"
                );
            }
        }
        assert!(
            stress_marked < stress_unmarked,
            "warm inscribed trail must impart conductive night-chill insulation ({stress_marked} < {stress_unmarked})"
        );
    }

    #[test]
    fn earth_ecology_basalt_ridge_occlusion_and_obligate_cooperative_grove_pods() {
        // 1. Verify opaque basalt ridge blocks 100% of peer visual silhouette while diffracting sound at 0.68.
        let mut solo_world = MultiGenTerrarium::new_earth_ecology(2026, 1);
        let mut occluded_peer_world = MultiGenTerrarium::new_earth_ecology(2026, 2);
        let mut clear_peer_world = MultiGenTerrarium::new_earth_ecology(2026, 2);

        let valley_pos = (-4.2, 0.0);
        let hidden_grove_pos = (-14.2, 0.5); // Across West Ridge (-11.6, -2.6)..(-11.6, 2.6)
        let clear_pos = (-1.5, 1.8);

        solo_world.respawn_at_position(0, valley_pos);
        occluded_peer_world.respawn_at_position(0, valley_pos);
        occluded_peer_world.respawn_at_position(1, hidden_grove_pos);
        clear_peer_world.respawn_at_position(0, valley_pos);
        clear_peer_world.respawn_at_position(1, clear_pos);

        assert!(!occluded_peer_world.has_line_of_sight(valley_pos, hidden_grove_pos));
        assert!(clear_peer_world.has_line_of_sight(valley_pos, clear_pos));
        assert!((occluded_peer_world.acoustic_transmittance(valley_pos, hidden_grove_pos) - 0.68).abs() < 1e-6);
        assert!((clear_peer_world.acoustic_transmittance(valley_pos, clear_pos) - 1.0).abs() < 1e-6);

        // Agent 0's photonic band (0..64) must be bit-for-bit identical whether a peer is behind the basalt ridge or absent!
        assert_eq!(
            &solo_world.agent_view(0).ambient_in[0..64],
            &occluded_peer_world.agent_view(0).ambient_in[0..64],
            "opaque basalt ridge must block 100% of occluded peer photons"
        );
        assert_ne!(
            &solo_world.agent_view(0).ambient_in[0..64],
            &clear_peer_world.agent_view(0).ambient_in[0..64],
            "unoccluded peer in clear sight must cast a visible silhouette"
        );

        // 2. Verify obligate 2-organism cooperative cracking of heavy_shell grove pods.
        let mut coop_world = MultiGenTerrarium::new_earth_ecology(2026, 2);
        let heavy_pod_idx = coop_world
            .forage_pods()
            .iter()
            .position(|p| p.heavy_shell)
            .expect("earth ecology spawns heavy_shell grove pods");
        let pod_pos = coop_world.forage_pods()[heavy_pod_idx].pos;
        coop_world.respawn_at_position(0, (pod_pos.0 + 0.7, pod_pos.1));
        coop_world.respawn_at_position(1, (8.0, 0.0)); // Far away initially

        let work_pulse = FieldPulse {
            actuator_out: vec![0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
        };
        let idle_pulse = FieldPulse::silent(TERRARIUM_ACTUATORS);
        for _ in 0..45 {
            coop_world.set_agent_embodied_kinematics(0, 0.95, 1.0);
            coop_world.set_agent_embodied_kinematics(1, 0.95, 0.0);
            coop_world.apply_population_pulses(&[work_pulse.clone(), idle_pulse.clone()]);
        }
        let solo_pod = coop_world.forage_pods()[heavy_pod_idx];
        assert!(
            solo_pod.shell_integrity >= 0.22 - 1e-5 && solo_pod.open_nutrients == 0.0,
            "1 organism working solo must never crack a heavy_shell grove pod, got {:?}",
            solo_pod
        );
        assert!(
            coop_world.ecology_metrics().acoustic_impact_events >= 1,
            "solo prying on heavy_shell pod must radiate woody 16 kHz tap sounds"
        );

        // Now bring peer organism 1 beside the same heavy_shell pod so both pry simultaneously.
        coop_world.respawn_at_position(1, (pod_pos.0 - 0.7, pod_pos.1));
        for _ in 0..12 {
            coop_world.set_agent_embodied_kinematics(0, 0.95, 1.0);
            coop_world.set_agent_embodied_kinematics(1, 0.95, 1.0);
            coop_world.apply_population_pulses(&[work_pulse.clone(), work_pulse.clone()]);
        }
        let cracked_pod = coop_world.forage_pods()[heavy_pod_idx];
        assert_eq!(cracked_pod.shell_integrity, 0.0);
        assert!(coop_world.ecology_metrics().coop_grove_pods_cracked >= 1);
        assert!(coop_world.agent_view(0).metabolic_influx > 0.05);
        assert!(coop_world.agent_view(1).metabolic_influx > 0.05);
    }
}
