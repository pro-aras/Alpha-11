//! Deterministic pseudo-random generator, numeric hygiene, and parameter hashing.

#[inline]
#[must_use]
pub fn sanitize(x: f32) -> f32 {
    if !x.is_finite() || x.abs() < 1.0e-20 {
        0.0
    } else {
        x
    }
}

#[inline]
#[must_use]
pub fn sanitize_clamp(x: f32, lo: f32, hi: f32) -> f32 {
    let s = sanitize(x);
    s.clamp(lo, hi)
}

#[inline]
#[must_use]
pub fn smooth_saturate(x: f32) -> f32 {
    let s = sanitize(x);
    if s >= 3.0 {
        1.0
    } else if s <= -3.0 {
        -1.0
    } else {
        let s2 = s * s;
        sanitize_clamp(s * (27.0 + s2) / (27.0 + 9.0 * s2), -1.0, 1.0)
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng64 {
    pub s0: u64,
    pub s1: u64,
}

impl Rng64 {
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed ^ 0xA110_C0DE_5EED_0001;
        let mut s0 = splitmix64(&mut sm);
        let mut s1 = splitmix64(&mut sm);
        if s0 == 0 && s1 == 0 {
            s0 = 0x9E37_79B9_7F4A_7C15;
            s1 = 0xBF58_476D_1CE4_E5B9;
        }
        Self { s0, s1 }
    }

    #[must_use]
    pub fn from_raw(s0: u64, s1: u64) -> Self {
        if s0 == 0 && s1 == 0 {
            Self::from_seed(0)
        } else {
            Self { s0, s1 }
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);
        s1 ^= s0;
        self.s0 = s0.rotate_left(49) ^ s1 ^ (s1 << 21);
        self.s1 = s1.rotate_left(28);
        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    pub fn next_f32_unit(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        (bits as f32) * (1.0 / 16_777_216.0)
    }

    #[inline]
    pub fn next_f32_signed(&mut self) -> f32 {
        self.next_f32_unit() * 2.0 - 1.0
    }

    #[inline]
    pub fn next_index(&mut self, len: usize) -> usize {
        if len <= 1 {
            0
        } else {
            (self.next_u64() % (len as u64)) as usize
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParamHasher {
    state: u64,
}

impl ParamHasher {
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut h = Self {
            state: 0xCBF2_9CE4_8422_2325,
        };
        h.write_u64(seed);
        h
    }

    #[inline]
    pub fn write_u8(&mut self, v: u8) {
        self.state ^= u64::from(v);
        self.state = self.state.wrapping_mul(0x0100_0000_01B3);
    }

    #[inline]
    pub fn write_bool(&mut self, v: bool) {
        self.write_u8(u8::from(v));
    }

    #[inline]
    pub fn write_u16(&mut self, v: u16) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }

    #[inline]
    pub fn write_u32(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }

    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.write_u8(b);
        }
    }

    #[inline]
    pub fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    #[inline]
    pub fn write_f32(&mut self, v: f32) {
        let clean = sanitize(v);
        let bits = if clean == 0.0 { 0u32 } else { clean.to_bits() };
        self.write_u32(bits);
    }

    #[inline]
    #[must_use]
    pub fn finish(self) -> u64 {
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
