//! Hidden spatial-wave pair surface (`PatternMemoryBoard`) for unforced memory-puzzle observation.
//!
//! Exposes `176` unlabelled ambient channels (`128` tile harmonic wave channels across a `4x4`
//! surface + `16` cursor/touch/sonar/foveal resonance channels + `32` exogenous Noisy-TV channels)
//! and `4` continuous actuators (`cx`, `cy`, `flip_press`, `listen_ping`).
//!
//! Touching a concealed tile reveals its harmonic signature for `12` ticks; touching two matching
//! tiles within the `12`-tick decay window locks both into a gentle standing harmonic resonance,
//! increasing field predictability with strictly zero game-score `metabolic_influx` bonus.

use a11_medium::{FieldPulse, FieldView, WorldMedium};

use crate::{SplitMix64, actuator_at, sanitize_nonneg, sanitize_unit_signed};

/// Side length of the `4 x 4` spatial-wave pair surface.
pub const BOARD_SIDE: usize = 4;
/// Number of tiles on the surface (`16` tiles = `8` harmonic pairs).
pub const NUM_TILES: usize = BOARD_SIDE * BOARD_SIDE;
/// Number of wave feature channels per tile.
pub const FEATURES_PER_TILE: usize = 8;
/// Number of tile wave channels (`16 * 8 = 128`).
pub const TILE_WAVE_CHANNELS: usize = NUM_TILES * FEATURES_PER_TILE;
/// Number of cursor, alignment, sonar ping, and foveal tile channels.
pub const CURSOR_CHANNELS: usize = 16;
/// Number of exogenous distractor channels (`Noisy TV`).
pub const PATTERN_NOISY_TV_CHANNELS: usize = 32;
/// Total ambient channels exposed by [`PatternMemoryBoard`] (`128 + 16 + 32 = 176`).
pub const PATTERN_BOARD_AMBIENT_CHANNELS: usize =
    TILE_WAVE_CHANNELS + CURSOR_CHANNELS + PATTERN_NOISY_TV_CHANNELS;
/// Number of continuous actuators accepted by [`PatternMemoryBoard`].
pub const PATTERN_BOARD_ACTUATORS: usize = 4;

/// Reveal window duration in ticks after touching a concealed tile.
pub const REVEAL_WINDOW_TICKS: u8 = 12;
/// Default flat basal metabolic trickle in standalone portal mode (`>= 0.0`).
pub const DEFAULT_PATTERN_BASAL_INFLUX: f32 = 0.008;

const MAX_BOARD_COORD: f32 = (BOARD_SIDE - 1) as f32;

/// Read-only behavioral observation metrics for [`PatternMemoryBoard`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PatternBoardObservation {
    /// Total concealed-tile reveal touches triggered.
    pub tiles_inspected: u64,
    /// Number of distinct tiles (`0..=16`) visited or inspected by the organism.
    pub unique_tiles_visited: usize,
    /// Number of matching tile pairs (`0..=8`) locked into steady harmonic resonance.
    pub matched_resonances_locked: usize,
    /// Mean tick interval between repeat visits/inspections of the same tile (`0.0` if none).
    pub repeat_visit_interval_mean: f32,
    /// Cursor dwell stability on tile centers (`[0.0, 1.0]`, high for steady dwell, low for thrashing).
    pub cursor_dwell_stability: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TileState {
    pair_id: u8,
    reveal_ticks_remaining: u8,
    opened_at_tick: u64,
    locked: bool,
}

/// `4x4` concealed spatial-wave pair board with `12`-tick reveal decay and zero staged rewards.
#[derive(Clone, Debug)]
pub struct PatternMemoryBoard {
    tick: u64,
    basal_influx: f32,
    rng: SplitMix64,

    tiles: [TileState; NUM_TILES],
    pair_signatures: [[f32; FEATURES_PER_TILE]; NUM_TILES / 2],
    pair_omegas: [f32; NUM_TILES / 2],

    cursor_x: f32,
    cursor_y: f32,
    cursor_vx: f32,
    cursor_vy: f32,
    current_tile_idx: usize,
    prev_flip_pressed: bool,
    ping_echo_energy: f32,

    // Read-only behavioral observation state
    tiles_inspected: u64,
    visited_mask: u16,
    matched_resonances_locked: usize,
    last_depart_or_inspect_tick: [Option<u64>; NUM_TILES],
    repeat_interval_sum: f64,
    repeat_interval_count: u64,
    tile_switch_count: u64,
    alignment_sum: f64,

    ambient_buf: Vec<f32>,
}

impl Default for PatternMemoryBoard {
    fn default() -> Self {
        Self::new(0x4E4D_B0A8_D011_A110)
    }
}

impl PatternMemoryBoard {
    /// Creates a new `PatternMemoryBoard` with deterministic pair layout from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self::with_basal_influx(seed, DEFAULT_PATTERN_BASAL_INFLUX)
    }

    /// Creates a new `PatternMemoryBoard` with explicit constant `basal_influx` (`>= 0.0`).
    #[must_use]
    pub fn with_basal_influx(seed: u64, basal_influx: f32) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut assignment = [0_u8; NUM_TILES];
        for (i, slot) in assignment.iter_mut().enumerate() {
            *slot = (i / 2) as u8;
        }
        rng.shuffle(&mut assignment);

        let mut tiles = [TileState::default(); NUM_TILES];
        for i in 0..NUM_TILES {
            tiles[i].pair_id = assignment[i];
        }

        // Distinct spatial-harmonic orthogonal basis signatures for the 8 pairs.
        let mut pair_signatures = [[0.0_f32; FEATURES_PER_TILE]; NUM_TILES / 2];
        let mut pair_omegas = [0.0_f32; NUM_TILES / 2];
        for (p, sig_row) in pair_signatures.iter_mut().enumerate() {
            pair_omegas[p] = 0.22 + 0.09 * p as f32;
            for (f, sig_val) in sig_row.iter_mut().enumerate() {
                let phase = (p + 1) as f32 * (f + 1) as f32 * std::f32::consts::FRAC_PI_4;
                let walsh = if ((p & f).count_ones() & 1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                *sig_val = (0.55 * walsh + 0.35 * phase.cos()).clamp(-0.90, 0.90);
            }
        }

        let start_x = 1.5_f32;
        let start_y = 1.5_f32;
        let start_tile = coord_to_tile(start_x, start_y);

        let mut board = Self {
            tick: 0,
            basal_influx: sanitize_nonneg(basal_influx),
            rng,
            tiles,
            pair_signatures,
            pair_omegas,
            cursor_x: start_x,
            cursor_y: start_y,
            cursor_vx: 0.0,
            cursor_vy: 0.0,
            current_tile_idx: start_tile,
            prev_flip_pressed: false,
            ping_echo_energy: 0.0,
            tiles_inspected: 0,
            visited_mask: 0,
            matched_resonances_locked: 0,
            last_depart_or_inspect_tick: [None; NUM_TILES],
            repeat_interval_sum: 0.0,
            repeat_interval_count: 0,
            tile_switch_count: 0,
            alignment_sum: 0.0,
            ambient_buf: vec![0.0; PATTERN_BOARD_AMBIENT_CHANNELS],
        };
        board.refresh_ambient();
        board
    }

    /// Current simulation tick of this board.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Current cursor coordinates `(x, y)` in `[0.0, 3.0]`.
    #[must_use]
    pub fn cursor_position(&self) -> (f32, f32) {
        (self.cursor_x, self.cursor_y)
    }

    /// Returns the underlying pair ID (`0..8`) at `tile_idx` (`0..16`) for external test verification.
    /// Never exposed through `FieldView`.
    #[must_use]
    pub fn tile_pair_id(&self, tile_idx: usize) -> Option<u8> {
        self.tiles.get(tile_idx).map(|t| t.pair_id)
    }

    /// Computes read-only behavioral observation metrics without mutating world state.
    #[must_use]
    pub fn observation(&self) -> PatternBoardObservation {
        let repeat_visit_interval_mean = if self.repeat_interval_count == 0 {
            0.0
        } else {
            (self.repeat_interval_sum / self.repeat_interval_count as f64) as f32
        };

        let cursor_dwell_stability = if self.tick == 0 {
            1.0
        } else {
            let switch_rate = (self.tile_switch_count as f32 / self.tick as f32).clamp(0.0, 1.0);
            let mean_align = (self.alignment_sum / self.tick as f64) as f32;
            (0.65 * (1.0 - switch_rate) + 0.35 * mean_align).clamp(0.0, 1.0)
        };

        PatternBoardObservation {
            tiles_inspected: self.tiles_inspected,
            unique_tiles_visited: self.visited_mask.count_ones() as usize,
            matched_resonances_locked: self.matched_resonances_locked,
            repeat_visit_interval_mean,
            cursor_dwell_stability,
        }
    }

    fn compute_tile_feature(&self, tile_idx: usize, feat_idx: usize) -> f32 {
        let tile = self.tiles[tile_idx];
        let p = tile.pair_id as usize;
        let base_sig = self.pair_signatures[p][feat_idx];
        let omega = self.pair_omegas[p];
        let t = self.tick as f32;

        if tile.locked {
            // Steady gentle harmonic resonance once matched: high predictability, zero reward bonus.
            let carrier = 0.72 + 0.28 * (omega * 0.5 * t + feat_idx as f32 * 0.4).cos();
            0.70 * base_sig * carrier
        } else if tile.reveal_ticks_remaining > 0 {
            let env = tile.reveal_ticks_remaining as f32 / REVEAL_WINDOW_TICKS as f32;
            let smooth_env = env.sqrt();
            let carrier = 0.75 + 0.25 * (omega * t + feat_idx as f32 * 0.5).cos();
            smooth_env * base_sig * carrier
        } else {
            // Concealed tile: neutral subtle physical border grain, zero pair identity leakage.
            let r = (tile_idx / BOARD_SIDE) as f32;
            let c = (tile_idx % BOARD_SIDE) as f32;
            0.04 * ((r * 1.3 + c * 0.9 + feat_idx as f32).sin())
        }
    }

    fn refresh_ambient(&mut self) {
        // 0..128: 4x4 tile wave channels (8 features per tile)
        for t_idx in 0..NUM_TILES {
            let base = t_idx * FEATURES_PER_TILE;
            for f in 0..FEATURES_PER_TILE {
                self.ambient_buf[base + f] =
                    sanitize_unit_signed(self.compute_tile_feature(t_idx, f));
            }
        }

        // 128..144: Cursor state, tile alignment, sonar ping echo, interference, and foveal 8-feature read
        let t_idx = self.current_tile_idx;
        let tc = (t_idx % BOARD_SIDE) as f32;
        let tr = (t_idx / BOARD_SIDE) as f32;
        let dist_sq = (self.cursor_x - tc).powi(2) + (self.cursor_y - tr).powi(2);
        let align = (-dist_sq / 0.20).exp().clamp(0.0, 1.0);

        let tile = self.tiles[t_idx];
        let touch_state = if tile.locked {
            1.0
        } else if tile.reveal_ticks_remaining > 0 {
            0.5
        } else {
            -0.5
        };

        let half = MAX_BOARD_COORD * 0.5;
        let c_base = TILE_WAVE_CHANNELS;
        self.ambient_buf[c_base] = sanitize_unit_signed((self.cursor_x - half) / half);
        self.ambient_buf[c_base + 1] = sanitize_unit_signed((self.cursor_y - half) / half);
        self.ambient_buf[c_base + 2] = sanitize_unit_signed(self.cursor_vx / 0.85);
        self.ambient_buf[c_base + 3] = sanitize_unit_signed(self.cursor_vy / 0.85);
        self.ambient_buf[c_base + 4] = sanitize_unit_signed(align);
        self.ambient_buf[c_base + 5] = sanitize_unit_signed(touch_state);
        self.ambient_buf[c_base + 6] = sanitize_unit_signed(self.ping_echo_energy);
        // Channel `c_base + 7` previously broadcast a match/mismatch teacher cue (+0.95 / -0.45).
        // Removed (audit P26): the organism must infer pair identity from the tile wave features alone.
        self.ambient_buf[c_base + 7] = 0.0;

        for f in 0..FEATURES_PER_TILE {
            self.ambient_buf[c_base + 8 + f] =
                sanitize_unit_signed(self.compute_tile_feature(t_idx, f));
        }

        // 144..176: Exogenous Noisy-TV distractor channels
        let tv_start = TILE_WAVE_CHANNELS + CURSOR_CHANNELS;
        for ch in &mut self.ambient_buf[tv_start..PATTERN_BOARD_AMBIENT_CHANNELS] {
            *ch = sanitize_unit_signed(self.rng.next_f32_signed() * 0.85);
        }
    }
}

impl WorldMedium for PatternMemoryBoard {
    fn num_ambient_channels(&self) -> usize {
        PATTERN_BOARD_AMBIENT_CHANNELS
    }

    fn num_actuators(&self) -> usize {
        PATTERN_BOARD_ACTUATORS
    }

    fn current_view(&self) -> FieldView<'_> {
        FieldView {
            ambient_in: &self.ambient_buf,
            metabolic_influx: self.basal_influx,
            tissue_stress: 0.0,
            num_actuators: PATTERN_BOARD_ACTUATORS,
        }
    }

    fn apply_pulse(&mut self, pulse: &FieldPulse) {
        self.tick = self.tick.saturating_add(1);

        let cx = actuator_at(pulse, 0);
        let cy = actuator_at(pulse, 1);
        let flip_press = actuator_at(pulse, 2);
        let listen_ping = actuator_at(pulse, 3);

        // 1. Decay existing open unlocked tiles by 1 tick.
        for tile in &mut self.tiles {
            if !tile.locked && tile.reveal_ticks_remaining > 0 {
                tile.reveal_ticks_remaining -= 1;
            }
        }

        // 2. Smooth damped cursor kinematics over [0.0, 3.0] x [0.0, 3.0].
        let prev_tile = self.current_tile_idx;
        self.cursor_vx = self.cursor_vx * 0.40 + 0.55 * cx;
        self.cursor_vy = self.cursor_vy * 0.40 + 0.55 * cy;

        let next_x = (self.cursor_x + self.cursor_vx).clamp(0.0, MAX_BOARD_COORD);
        let next_y = (self.cursor_y + self.cursor_vy).clamp(0.0, MAX_BOARD_COORD);
        let speed = ((next_x - self.cursor_x).powi(2) + (next_y - self.cursor_y).powi(2)).sqrt();
        self.cursor_x = next_x;
        self.cursor_y = next_y;

        let new_tile = coord_to_tile(self.cursor_x, self.cursor_y);
        self.current_tile_idx = new_tile;

        let tc = (new_tile % BOARD_SIDE) as f32;
        let tr = (new_tile / BOARD_SIDE) as f32;
        let dist_sq = (self.cursor_x - tc).powi(2) + (self.cursor_y - tr).powi(2);
        let align = (-dist_sq / 0.20).exp().clamp(0.0, 1.0);
        self.alignment_sum += f64::from(align);

        if new_tile != prev_tile {
            self.tile_switch_count = self.tile_switch_count.saturating_add(1);
            self.last_depart_or_inspect_tick[prev_tile] = Some(self.tick);
            if let Some(prev_visit_t) = self.last_depart_or_inspect_tick[new_tile] {
                let dt = self.tick.saturating_sub(prev_visit_t);
                if dt > 0 {
                    self.repeat_interval_sum += dt as f64;
                    self.repeat_interval_count = self.repeat_interval_count.saturating_add(1);
                }
            }
            self.visited_mask |= 1_u16 << new_tile;
        } else if speed > 0.01 {
            self.visited_mask |= 1_u16 << new_tile;
        }

        // 3. Touch / flip physics when flip_press > 0.25 on a concealed tile.
        let pressed_now = flip_press > 0.25;
        let can_trigger = pressed_now && (!self.prev_flip_pressed || new_tile != prev_tile);
        self.prev_flip_pressed = pressed_now;

        if can_trigger
            && !self.tiles[new_tile].locked
            && self.tiles[new_tile].reveal_ticks_remaining == 0
        {
            self.tiles_inspected = self.tiles_inspected.saturating_add(1);
            self.visited_mask |= 1_u16 << new_tile;

            if new_tile == prev_tile {
                if let Some(prev_t) = self.last_depart_or_inspect_tick[new_tile] {
                    let dt = self.tick.saturating_sub(prev_t);
                    if dt > 0 {
                        self.repeat_interval_sum += dt as f64;
                        self.repeat_interval_count = self.repeat_interval_count.saturating_add(1);
                    }
                }
            }
            self.last_depart_or_inspect_tick[new_tile] = Some(self.tick);

            let target_pair = self.tiles[new_tile].pair_id;
            let matching_open = (0..NUM_TILES).find(|&idx| {
                idx != new_tile
                    && !self.tiles[idx].locked
                    && self.tiles[idx].reveal_ticks_remaining > 0
                    && self.tiles[idx].pair_id == target_pair
            });

            if let Some(partner_idx) = matching_open {
                // Constructive harmonic lock between matching pair!
                self.tiles[new_tile].locked = true;
                self.tiles[new_tile].reveal_ticks_remaining = 0;
                self.tiles[partner_idx].locked = true;
                self.tiles[partner_idx].reveal_ticks_remaining = 0;
                self.matched_resonances_locked = self.matched_resonances_locked.saturating_add(1);
            } else {
                // Cap concurrent unlocked revealed tiles to 2 (if 2 were already open, close oldest).
                let open_unlocked_count = self
                    .tiles
                    .iter()
                    .filter(|t| !t.locked && t.reveal_ticks_remaining > 0)
                    .count();
                if open_unlocked_count >= 2 {
                    if let Some((oldest_idx, _)) = self
                        .tiles
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| !t.locked && t.reveal_ticks_remaining > 0)
                        .min_by_key(|(_, t)| t.opened_at_tick)
                    {
                        self.tiles[oldest_idx].reveal_ticks_remaining = 0;
                    }
                }
                self.tiles[new_tile].reveal_ticks_remaining = REVEAL_WINDOW_TICKS;
                self.tiles[new_tile].opened_at_tick = self.tick;
            }
        }

        // 4. Acoustic sonar ping resonance across active/locked wave tiles.
        let active_fraction = self
            .tiles
            .iter()
            .filter(|t| t.locked || t.reveal_ticks_remaining > 0)
            .count() as f32
            / NUM_TILES as f32;
        if listen_ping > 0.2 {
            let ping_amp = (listen_ping - 0.2) / 0.8;
            self.ping_echo_energy =
                (0.50 * self.ping_echo_energy + 0.50 * ping_amp * (0.35 + 0.65 * active_fraction))
                    .clamp(-1.0, 1.0);
        } else {
            self.ping_echo_energy *= 0.65;
        }

        self.refresh_ambient();
    }
}

fn coord_to_tile(x: f32, y: f32) -> usize {
    let col = (x.round() as usize).min(BOARD_SIDE - 1);
    let row = (y.round() as usize).min(BOARD_SIDE - 1);
    row * BOARD_SIDE + col
}
