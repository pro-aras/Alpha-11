//! Pure-Rust Castagnoli CRC32C checksum implementation.

const CASTAGNOLI_POLY: u32 = 0x82F6_3B78;

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ CASTAGNOLI_POLY;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

const CRC32C_TABLE: [u32; 256] = build_crc32c_table();

#[inline]
#[must_use]
pub fn crc32c_update(mut state: u32, bytes: &[u8]) -> u32 {
    for &b in bytes {
        let idx = ((state ^ u32::from(b)) & 0xFF) as usize;
        state = (state >> 8) ^ CRC32C_TABLE[idx];
    }
    state
}

#[inline]
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    !crc32c_update(0xFFFF_FFFF, bytes)
}
