//! Pure-Rust deterministic pseudo-random number generator (`SplitMix64` + `XoShiRo256**`).
//!
//! Provides reproducible stochasticity, unbiased bounded integer sampling, and
//! Fisher-Yates shuffling without external crate dependencies.

/// Fast, reproducible `XoShiRo256**` pseudo-random generator seeded via `SplitMix64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicRng {
    state: [u64; 4],
}

impl DeterministicRng {
    /// Initializes a new deterministic generator from any 64-bit seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut sm = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut state = [0u64; 4];
        for slot in &mut state {
            *slot = splitmix64_next(&mut sm);
        }
        if state == [0, 0, 0, 0] {
            state[0] = 0xD1B5_4A32_D192_ED03;
        }
        Self { state }
    }

    /// Forks an independent child generator mixed with `salt`.
    #[must_use]
    pub fn fork(&mut self, salt: u64) -> Self {
        Self::new(self.next_u64() ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9))
    }

    /// Generates the next 64-bit unsigned value.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    /// Generates a 32-bit unsigned value.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generates a uniform `f32` in `[0.0, 1.0)`.
    pub fn next_f32_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Generates a uniform bipolar `f32` in `[-1.0, 1.0]`.
    pub fn next_f32_bipolar(&mut self) -> f32 {
        self.next_f32_unit() * 2.0 - 1.0
    }

    /// Generates a uniform `f32` in `[min, max)`.
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32_unit()
    }

    /// Unbiased uniform index in `0..bound` (returns `0` when `bound <= 1`).
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        let b = bound as u64;
        let threshold = b.wrapping_neg() % b;
        loop {
            let candidate = self.next_u64();
            if candidate >= threshold {
                return (candidate % b) as usize;
            }
        }
    }

    /// Unbiased in-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Safely reads and sanitizes actuator channel `idx` from `pulse`, clamping to `[-1.0, 1.0]`
/// and mapping non-finite values (`NaN`, `+-Inf`) to `0.0`.
#[inline]
#[must_use]
pub fn sanitize_actuator(value: Option<f32>) -> f32 {
    match value {
        Some(v) if v.is_finite() => v.clamp(-1.0, 1.0),
        _ => 0.0,
    }
}
