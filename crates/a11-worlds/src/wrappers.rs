//! Scientific Null-Control & Coordinate-Scrambling Wrappers (`PermutedMedium`,
//! `ActionShuffledMedium`, `YokedMedium`).
//!
//! Used by external scientific evaluation harnesses to verify coordinate-free organogenesis,
//! interventional causal agency, and homeostasis under yoked reward streams.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::rng::{DeterministicRng, sanitize_actuator};

/// Coordinate-scrambling wrapper applying secret seeded random bijections `perm_in`
/// to `ambient_in` and `perm_out` to `actuator_out` on every tick.
///
/// Proves 100% coordinate-free, label-free organogenesis: the organism never sees channel
/// indices or band boundaries, while external observers can decode true physical bands via
/// [`PermutedMedium::invert_sensor_channel`].
#[derive(Clone, Debug)]
pub struct PermutedMedium<W: WorldMedium> {
    inner: W,
    /// Maps `scrambled_sensor_idx -> true_physical_sensor_idx`.
    perm_in: Vec<usize>,
    /// Inverse map: `true_physical_sensor_idx -> scrambled_sensor_idx`.
    inv_perm_in: Vec<usize>,
    /// Maps `scrambled_motor_idx -> true_physical_motor_idx`.
    perm_out: Vec<usize>,
    /// Inverse map: `true_physical_motor_idx -> scrambled_motor_idx`.
    inv_perm_out: Vec<usize>,
    scrambled_ambient: Vec<f32>,
    unscrambled_pulse_buf: FieldPulse,
}

impl<W: WorldMedium> PermutedMedium<W> {
    /// Wraps `inner` with deterministic random channel bijections derived from `seed`.
    #[must_use]
    pub fn new(inner: W, seed: u64) -> Self {
        let mut rng = DeterministicRng::new(seed ^ 0xA11_B1EC_7104);
        let n_in = inner.num_ambient_channels();
        let n_out = inner.num_actuators();

        let mut perm_in: Vec<usize> = (0..n_in).collect();
        rng.shuffle(&mut perm_in);
        let mut inv_perm_in = vec![0usize; n_in];
        for (scrambled, &orig) in perm_in.iter().enumerate() {
            inv_perm_in[orig] = scrambled;
        }

        let mut perm_out: Vec<usize> = (0..n_out).collect();
        rng.shuffle(&mut perm_out);
        let mut inv_perm_out = vec![0usize; n_out];
        for (scrambled, &orig) in perm_out.iter().enumerate() {
            inv_perm_out[orig] = scrambled;
        }

        let mut wrapper = Self {
            inner,
            perm_in,
            inv_perm_in,
            perm_out,
            inv_perm_out,
            scrambled_ambient: vec![0.0; n_in],
            unscrambled_pulse_buf: FieldPulse::silent(n_out),
        };
        wrapper.sync_ambient();
        wrapper
    }

    /// Decodes a scrambled sensor channel index (observed on an organism's `SensorPore`)
    /// back into the true physical channel index of the underlying world `W`.
    #[inline]
    #[must_use]
    pub fn invert_sensor_channel(&self, scrambled_idx: usize) -> usize {
        self.perm_in
            .get(scrambled_idx)
            .copied()
            .unwrap_or(scrambled_idx)
    }

    /// Encodes a true physical sensor channel index of `W` into its scrambled index.
    #[inline]
    #[must_use]
    pub fn scramble_sensor_channel(&self, true_idx: usize) -> usize {
        self.inv_perm_in.get(true_idx).copied().unwrap_or(true_idx)
    }

    /// Decodes a scrambled motor channel index (observed on an organism's `MotorPore`)
    /// back into the true physical actuator index of the underlying world `W`.
    #[inline]
    #[must_use]
    pub fn invert_motor_channel(&self, scrambled_idx: usize) -> usize {
        self.perm_out
            .get(scrambled_idx)
            .copied()
            .unwrap_or(scrambled_idx)
    }

    /// Encodes a true physical actuator index of `W` into its scrambled index.
    #[inline]
    #[must_use]
    pub fn scramble_motor_channel(&self, true_idx: usize) -> usize {
        self.inv_perm_out.get(true_idx).copied().unwrap_or(true_idx)
    }

    /// Read-only reference to the wrapped world medium.
    #[must_use]
    pub fn inner(&self) -> &W {
        &self.inner
    }

    /// Applies a mutation to the inner world (such as `set_illumination` or
    /// `trigger_causal_reversal`) and immediately resynchronizes the scrambled ambient buffer.
    pub fn with_inner_mut<R>(&mut self, f: impl FnOnce(&mut W) -> R) -> R {
        let out = f(&mut self.inner);
        self.sync_ambient();
        out
    }

    /// Unwraps and returns the underlying world `W`.
    #[must_use]
    pub fn into_inner(self) -> W {
        self.inner
    }

    /// Re-populates `scrambled_ambient` from `inner.current_view().ambient_in`.
    pub fn sync_ambient(&mut self) {
        let raw = self.inner.current_view().ambient_in;
        let n = self.perm_in.len().min(raw.len());
        if self.scrambled_ambient.len() != self.perm_in.len() {
            self.scrambled_ambient.resize(self.perm_in.len(), 0.0);
        }
        for s_idx in 0..n {
            let orig_idx = self.perm_in[s_idx];
            self.scrambled_ambient[s_idx] = raw.get(orig_idx).copied().unwrap_or(0.0);
        }
    }
}

impl<W: WorldMedium> WorldMedium for PermutedMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.perm_in.len()
    }

    fn num_actuators(&self) -> usize {
        self.perm_out.len()
    }

    fn current_view(&self) -> FieldView<'_> {
        let inner_view = self.inner.current_view();
        FieldView {
            ambient_in: &self.scrambled_ambient,
            metabolic_influx: inner_view.metabolic_influx,
            tissue_stress: inner_view.tissue_stress,
            num_actuators: self.perm_out.len(),
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let n_out = self.perm_out.len();
        if self.unscrambled_pulse_buf.actuator_out.len() != n_out {
            self.unscrambled_pulse_buf.actuator_out.resize(n_out, 0.0);
        }
        self.unscrambled_pulse_buf.actuator_out.fill(0.0);
        for (scrambled_m, &true_m) in self.perm_out.iter().enumerate() {
            let val = sanitize_actuator(pulse.actuator_out.get(scrambled_m).copied());
            if let Some(slot) = self.unscrambled_pulse_buf.actuator_out.get_mut(true_m) {
                *slot = val;
            }
        }
        self.inner.apply_pulse(&self.unscrambled_pulse_buf);
        self.sync_ambient();
    }
}

/// Action-shuffled null-control wrapper decoupling action-outcome contingency (`P(s' | s, a)`)
/// while preserving the organism's emitted action magnitude distribution.
///
/// Each incoming [`FieldPulse`] has its actuator channels randomly permuted via an independent
/// seeded PRNG and swapped through a temporal reservoir ring so neither channel identity nor
/// action timing correlates with the organism's internal state.
#[derive(Clone, Debug)]
pub struct ActionShuffledMedium<W: WorldMedium> {
    inner: W,
    rng: DeterministicRng,
    pool_capacity: usize,
    reservoir: Vec<Vec<f32>>,
    applied_buf: FieldPulse,
}

impl<W: WorldMedium> ActionShuffledMedium<W> {
    /// Creates an action-shuffled control medium with a default `16`-step shuffle reservoir.
    #[must_use]
    pub fn new(inner: W, seed: u64) -> Self {
        Self::with_pool_capacity(inner, seed, 16)
    }

    /// Creates an action-shuffled control medium with explicit `pool_capacity` (`>= 2`).
    #[must_use]
    pub fn with_pool_capacity(inner: W, seed: u64, pool_capacity: usize) -> Self {
        let n_out = inner.num_actuators();
        let cap = pool_capacity.max(2);
        Self {
            inner,
            rng: DeterministicRng::new(seed ^ 0xA11_540F_F1ED),
            pool_capacity: cap,
            reservoir: Vec::with_capacity(cap),
            applied_buf: FieldPulse::silent(n_out),
        }
    }

    /// Read-only reference to the wrapped world medium.
    #[must_use]
    pub fn inner(&self) -> &W {
        &self.inner
    }

    /// Mutable reference to the wrapped world medium.
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Unwraps and returns the underlying world `W`.
    #[must_use]
    pub fn into_inner(self) -> W {
        self.inner
    }

    /// Returns the actual shuffled actuator slice delivered on the most recent `apply_pulse`.
    #[must_use]
    pub fn last_applied_pulse(&self) -> &[f32] {
        &self.applied_buf.actuator_out
    }
}

impl<W: WorldMedium> WorldMedium for ActionShuffledMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.inner.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.inner.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        self.inner.current_view()
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        let n_out = self.inner.num_actuators();
        let mut incoming = Vec::with_capacity(n_out);
        for idx in 0..n_out {
            incoming.push(sanitize_actuator(pulse.actuator_out.get(idx).copied()));
        }

        // Permute actuator channels to sever spatial motor directionality.
        self.rng.shuffle(&mut incoming);

        // Stream-swap through temporal reservoir to sever causal action timing while
        // preserving the exact multiset of emitted action magnitudes across the run.
        let chosen = if self.reservoir.len() < self.pool_capacity {
            self.reservoir.push(incoming.clone());
            let pick = self.rng.next_usize(self.reservoir.len());
            let out = self.reservoir[pick].clone();
            self.reservoir[pick] = incoming;
            out
        } else {
            let pick = self.rng.next_usize(self.pool_capacity);
            std::mem::replace(&mut self.reservoir[pick], incoming)
        };

        self.applied_buf.actuator_out = chosen;
        self.inner.apply_pulse(&self.applied_buf);
    }
}

/// Yoked homeostasis null-control wrapper running `W` normally for sensory physics while
/// overriding `metabolic_influx` and `tissue_stress` from a recorded donor stream.
#[derive(Clone, Debug)]
pub struct YokedMedium<W: WorldMedium> {
    inner: W,
    donor_stream: Vec<(f32, f32)>,
    cursor: usize,
}

impl<W: WorldMedium> YokedMedium<W> {
    /// Wraps `inner` and overrides `(metabolic_influx, tissue_stress)` on tick `t` with
    /// `donor_stream[t]` (defaulting safely to `(0.0, 0.0)` once `t >= donor_stream.len()`).
    #[must_use]
    pub fn new(inner: W, donor_stream: Vec<(f32, f32)>) -> Self {
        Self {
            inner,
            donor_stream,
            cursor: 0,
        }
    }

    /// Current replay cursor into `donor_stream`.
    #[must_use]
    pub fn donor_cursor(&self) -> usize {
        self.cursor
    }

    /// Read-only view of the recorded donor `(metabolic_influx, tissue_stress)` stream.
    #[must_use]
    pub fn donor_stream(&self) -> &[(f32, f32)] {
        &self.donor_stream
    }

    /// Read-only reference to the wrapped world medium.
    #[must_use]
    pub fn inner(&self) -> &W {
        &self.inner
    }

    /// Mutable reference to the wrapped world medium.
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Unwraps and returns the underlying world `W`.
    #[must_use]
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: WorldMedium> WorldMedium for YokedMedium<W> {
    fn num_ambient_channels(&self) -> usize {
        self.inner.num_ambient_channels()
    }

    fn num_actuators(&self) -> usize {
        self.inner.num_actuators()
    }

    fn current_view(&self) -> FieldView<'_> {
        let inner_view = self.inner.current_view();
        let (raw_influx, raw_stress) = self
            .donor_stream
            .get(self.cursor)
            .copied()
            .unwrap_or((0.0, 0.0));
        let influx = if raw_influx.is_finite() {
            raw_influx.max(0.0)
        } else {
            0.0
        };
        let stress = if raw_stress.is_finite() {
            raw_stress.max(0.0)
        } else {
            0.0
        };

        FieldView {
            ambient_in: inner_view.ambient_in,
            metabolic_influx: influx,
            tissue_stress: stress,
            num_actuators: inner_view.num_actuators,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.inner.apply_pulse(pulse);
        self.cursor = self.cursor.saturating_add(1);
    }
}
