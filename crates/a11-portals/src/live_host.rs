//! Crash-safe live observation loop & bounded JSONL behavioral telemetry recorder (`LiveHostRunner`).
//!
//! Generic over any `O: OrganismEngine` and `W: WorldMedium` using 100% pure Rust (`std` only):
//! - Executes unforced closed-loop `current_view() -> O::step() -> W::apply_pulse()` ticks.
//! - Writes pure-JSONL snapshots (`tick`, `atp`, `adenosine`, `temperature`, `cell_count`,
//!   `tube_count`, `sensor_pore_count`, `mean_tau`, `max_tau`, `mean_interventional_agency`,
//!   `subthreshold_rehearsals`, `mitosis_births`, `contradiction_splits`, `used_state_bytes`,
//!   `step_us`) with bounded rotating file size (`<telemetry>.jsonl` -> `<telemetry>.jsonl.1`).
//! - Performs crash-safe `.tmp -> primary` + `.prev` atomic checkpointing and shadow-restore
//!   startup recovery (falling back to `.prev` if `primary` is corrupted, and failing closed
//!   if all present checkpoints are corrupted).

use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use a11_medium::{OrganismEngine, OrganismSnapshot, PersistError, WorldMedium};

use crate::sanitize_unit_signed;

/// Default maximum size of the active `.jsonl` telemetry file before rotating to `.1` (8 MiB).
pub const DEFAULT_MAX_LOG_BYTES: usize = 8 * 1024 * 1024;
/// Default maximum allowed checkpoint size on disk for bounded reads (64 MiB).
pub const DEFAULT_MAX_CHECKPOINT_BYTES: usize = 64 * 1024 * 1024;

const LATENCY_WINDOW_CAP: usize = 1024;

/// Configuration for [`LiveHostRunner`].
#[derive(Clone, Debug)]
pub struct LiveHostConfig {
    /// Path to the primary organism checkpoint file (e.g. `live/organism.ckpt`).
    pub checkpoint_path: PathBuf,
    /// Path to the rotating pure-JSONL telemetry log (e.g. `live/telemetry.jsonl`).
    pub telemetry_path: PathBuf,
    /// Emit a JSONL telemetry snapshot every `N` steps (`>= 1`).
    pub telemetry_interval_ticks: u64,
    /// Save an atomic `.tmp -> primary` (+ `.prev`) checkpoint every `N` steps (`0` disables auto-save).
    pub checkpoint_interval_ticks: u64,
    /// Maximum bytes in `telemetry_path` before atomic rotation to `<telemetry_path>.1`.
    pub max_log_bytes: usize,
    /// Strict upper bound on checkpoint bytes accepted during bounded file reads and writes.
    pub max_checkpoint_bytes: usize,
}

impl LiveHostConfig {
    /// Creates a standard configuration rooted inside `dir`.
    #[must_use]
    pub fn in_directory(dir: impl AsRef<Path>) -> Self {
        let base = dir.as_ref();
        Self {
            checkpoint_path: base.join("organism.ckpt"),
            telemetry_path: base.join("telemetry.jsonl"),
            telemetry_interval_ticks: 10,
            checkpoint_interval_ticks: 250,
            max_log_bytes: DEFAULT_MAX_LOG_BYTES,
            max_checkpoint_bytes: DEFAULT_MAX_CHECKPOINT_BYTES,
        }
    }

    /// Temporary staging file used during atomic checkpoint saves (`<checkpoint_path>.tmp`).
    #[must_use]
    pub fn tmp_checkpoint_path(&self) -> PathBuf {
        append_ext(&self.checkpoint_path, "tmp")
    }

    /// Previous generation backup checkpoint file (`<checkpoint_path>.prev`).
    #[must_use]
    pub fn prev_checkpoint_path(&self) -> PathBuf {
        append_ext(&self.checkpoint_path, "prev")
    }

    /// Temporary staging file used when rotating `primary` into `.prev` (`<checkpoint_path>.prev.tmp`).
    #[must_use]
    pub fn prev_tmp_checkpoint_path(&self) -> PathBuf {
        append_ext(&self.checkpoint_path, "prev.tmp")
    }

    /// Single bounded archive file for rotated JSONL telemetry (`<telemetry_path>.1`).
    #[must_use]
    pub fn rotated_telemetry_path(&self) -> PathBuf {
        append_ext(&self.telemetry_path, "1")
    }
}

/// Startup recovery outcome reported by [`LiveHostRunner::open_or_recover`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// Neither `primary` nor `.prev` existed on disk; initialized fresh genesis engine.
    Genesis,
    /// Restored cleanly from `primary` checkpoint.
    RestoredPrimary { bytes_len: usize },
    /// `primary` was missing or rejected (`primary_error`); restored cleanly from `.prev` backup
    /// and atomically healed `primary`.
    RecoveredFromPrev {
        primary_error: PersistError,
        bytes_len: usize,
    },
}

/// Fail-closed error returned by [`LiveHostRunner`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiveHostError {
    Persist(PersistError),
    IoFailure,
}

impl From<PersistError> for LiveHostError {
    fn from(err: PersistError) -> Self {
        Self::Persist(err)
    }
}

impl From<io::Error> for LiveHostError {
    fn from(_: io::Error) -> Self {
        Self::IoFailure
    }
}

/// Runtime latency, state-size, and rotation summary for [`LiveHostRunner`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LiveHostStats {
    pub steps_executed: u64,
    pub telemetry_lines_written: u64,
    pub log_rotations: u64,
    pub checkpoints_saved: u64,
    pub p50_step_us: f64,
    pub p95_step_us: f64,
    pub p99_step_us: f64,
    /// Deterministic logical state bytes reported by the organism (distinct from OS RSS).
    pub last_used_state_bytes: usize,
    /// Optional OS process RSS bytes read from `/proc/self/statm` on Linux (`None` if unavailable).
    pub os_rss_bytes: Option<usize>,
}

/// Crash-safe closed-loop live runner and JSONL telemetry recorder.
pub struct LiveHostRunner<O: OrganismEngine, W: WorldMedium> {
    config: LiveHostConfig,
    engine: O,
    world: W,

    steps_executed: u64,
    telemetry_lines_written: u64,
    log_rotations: u64,
    checkpoints_saved: u64,
    current_log_bytes: usize,
    last_used_state_bytes: usize,

    latency_ring_us: Vec<f64>,
    latency_cursor: usize,
}

impl<O: OrganismEngine + Default, W: WorldMedium> LiveHostRunner<O, W> {
    /// Opens or recovers an organism using `O::default` as the shadow-decode factory.
    pub fn open_or_init(
        config: LiveHostConfig,
        world: W,
    ) -> Result<(Self, RecoveryOutcome), LiveHostError> {
        Self::open_or_recover(config, world, O::default)
    }
}

impl<O: OrganismEngine, W: WorldMedium> LiveHostRunner<O, W> {
    /// Creates a `LiveHostRunner` around an already-instantiated `engine` and `world` without
    /// reading existing checkpoints from disk. Rejects immediately if `engine.is_poisoned()`.
    pub fn new(config: LiveHostConfig, engine: O, world: W) -> Result<Self, LiveHostError> {
        if engine.is_poisoned() {
            return Err(LiveHostError::Persist(PersistError::PoisonedAfterCorruption));
        }
        ensure_parent_dirs(&config)?;
        let current_log_bytes = existing_file_len(&config.telemetry_path);
        let last_used_state_bytes = engine.snapshot().used_state_bytes;

        Ok(Self {
            config,
            engine,
            world,
            steps_executed: 0,
            telemetry_lines_written: 0,
            log_rotations: 0,
            checkpoints_saved: 0,
            current_log_bytes,
            last_used_state_bytes,
            latency_ring_us: Vec::with_capacity(LATENCY_WINDOW_CAP),
            latency_cursor: 0,
        })
    }

    /// Startup shadow-restore recovery protocol:
    /// 1. If neither `primary` nor `.prev` exists on disk, initializes `Genesis` via `make_engine()`.
    /// 2. If `primary` exists, performs a bounded read and attempts shadow `restore_checkpoint` on a
    ///    fresh `make_engine()` instance.
    /// 3. If `primary` fails (discarding the poisoned candidate) and `.prev` exists, performs a
    ///    bounded read of `.prev`, shadow-restores on a fresh `make_engine()` instance, and heals
    ///    `primary` atomically via `.tmp -> primary`.
    /// 4. If present checkpoints are all corrupted, fails closed (`Err`) without ever resetting to blank state.
    pub fn open_or_recover<F>(
        config: LiveHostConfig,
        world: W,
        mut make_engine: F,
    ) -> Result<(Self, RecoveryOutcome), LiveHostError>
    where
        F: FnMut() -> O,
    {
        ensure_parent_dirs(&config)?;
        let prev_path = config.prev_checkpoint_path();
        let has_primary = config.checkpoint_path.exists();
        let has_prev = prev_path.exists();

        if !has_primary && !has_prev {
            let genesis = make_engine();
            let runner = Self::new(config, genesis, world)?;
            return Ok((runner, RecoveryOutcome::Genesis));
        }

        let primary_err = if has_primary {
            match read_bounded_file(&config.checkpoint_path, config.max_checkpoint_bytes) {
                Ok(bytes) => {
                    let mut candidate = make_engine();
                    match candidate.restore_checkpoint(&bytes) {
                        Ok(()) if !candidate.is_poisoned() => {
                            let len = bytes.len();
                            let runner = Self::new(config, candidate, world)?;
                            return Ok((runner, RecoveryOutcome::RestoredPrimary { bytes_len: len }));
                        }
                        Ok(()) => PersistError::PoisonedAfterCorruption,
                        Err(e) => e,
                    }
                }
                Err(e) => e,
            }
        } else {
            PersistError::IoFailure
        };

        // Primary failed or was absent while .prev exists: attempt fallback from .prev on a fresh candidate.
        if !has_prev {
            return Err(LiveHostError::Persist(primary_err));
        }

        let prev_bytes = read_bounded_file(&prev_path, config.max_checkpoint_bytes)?;
        let mut fallback = make_engine();
        fallback.restore_checkpoint(&prev_bytes)?;
        if fallback.is_poisoned() {
            return Err(LiveHostError::Persist(PersistError::PoisonedAfterCorruption));
        }

        // Heal primary atomically from validated .prev bytes without disturbing .prev.
        write_atomic_synced(
            &config.tmp_checkpoint_path(),
            &config.checkpoint_path,
            &prev_bytes,
        )?;

        let len = prev_bytes.len();
        let runner = Self::new(config, fallback, world)?;
        Ok((
            runner,
            RecoveryOutcome::RecoveredFromPrev {
                primary_error: primary_err,
                bytes_len: len,
            },
        ))
    }

    /// Shared reference to the live organism engine.
    #[must_use]
    pub fn engine(&self) -> &O {
        &self.engine
    }

    /// Mutable reference to the live organism engine.
    pub fn engine_mut(&mut self) -> &mut O {
        &mut self.engine
    }

    /// Shared reference to the active world medium.
    #[must_use]
    pub fn world(&self) -> &W {
        &self.world
    }

    /// Mutable reference to the active world medium.
    pub fn world_mut(&mut self) -> &mut W {
        &mut self.world
    }

    /// Configuration of this host runner.
    #[must_use]
    pub fn config(&self) -> &LiveHostConfig {
        &self.config
    }

    /// Computes step latency percentiles (`p50/p95/p99`) and reports deterministic `used_state_bytes`
    /// strictly separated from OS process RSS.
    #[must_use]
    pub fn stats(&self) -> LiveHostStats {
        let (p50, p95, p99) = compute_percentiles(&self.latency_ring_us);
        LiveHostStats {
            steps_executed: self.steps_executed,
            telemetry_lines_written: self.telemetry_lines_written,
            log_rotations: self.log_rotations,
            checkpoints_saved: self.checkpoints_saved,
            p50_step_us: p50,
            p95_step_us: p95,
            p99_step_us: p99,
            last_used_state_bytes: self.last_used_state_bytes,
            os_rss_bytes: read_linux_rss_bytes(),
        }
    }

    /// Executes a single unforced `current_view -> step -> apply_pulse` tick, recording periodic
    /// JSONL telemetry and atomic checkpoints according to `LiveHostConfig`.
    pub fn step_once(&mut self) -> Result<(), LiveHostError> {
        if self.engine.is_poisoned() {
            return Err(LiveHostError::Persist(PersistError::PoisonedAfterCorruption));
        }

        let t0 = Instant::now();
        let view = self.world.current_view();
        let expected_act = view.num_actuators;
        let mut pulse = self.engine.step(view)?;

        if pulse.actuator_out.len() != expected_act {
            pulse.actuator_out.resize(expected_act, 0.0);
        }
        for v in &mut pulse.actuator_out {
            *v = sanitize_unit_signed(*v);
        }

        self.world.apply_pulse(&pulse);
        let step_us = t0.elapsed().as_nanos() as f64 / 1_000.0;

        self.record_latency(step_us);
        self.steps_executed = self.steps_executed.saturating_add(1);

        let telem_every = self.config.telemetry_interval_ticks.max(1);
        if self.steps_executed % telem_every == 0 {
            let snap = self.engine.snapshot();
            self.last_used_state_bytes = snap.used_state_bytes;
            let line = format_telemetry_jsonl(&snap, step_us);
            self.append_telemetry_line(&line)?;
        }

        if self.config.checkpoint_interval_ticks > 0
            && self.steps_executed % self.config.checkpoint_interval_ticks == 0
        {
            self.save_checkpoint()?;
        }

        Ok(())
    }

    /// Executes `num_ticks` consecutive closed-loop ticks, stopping immediately on any fail-closed error.
    pub fn run_ticks(&mut self, num_ticks: u64) -> Result<(), LiveHostError> {
        for _ in 0..num_ticks {
            self.step_once()?;
        }
        Ok(())
    }

    /// Atomically encodes and persists the organism checkpoint:
    /// 1. Refuses to save if `self.engine.is_poisoned()`.
    /// 2. Encodes checkpoint bytes via `O::encode_checkpoint()` and checks size bounds.
    /// 3. Writes new bytes to `<checkpoint>.tmp` and `sync_all()`.
    /// 4. Copies existing valid `<checkpoint>` to `<checkpoint>.prev.tmp`, `sync_all()`, and
    ///    atomically renames to `<checkpoint>.prev` (preserving both files at every instant).
    /// 5. Atomically renames `<checkpoint>.tmp` -> `<checkpoint>` and syncs parent directory.
    pub fn save_checkpoint(&mut self) -> Result<usize, LiveHostError> {
        if self.engine.is_poisoned() {
            return Err(LiveHostError::Persist(PersistError::PoisonedAfterCorruption));
        }

        let bytes = self.engine.encode_checkpoint()?;
        if bytes.is_empty() {
            return Err(LiveHostError::Persist(PersistError::TruncatedInput));
        }
        if bytes.len() > self.config.max_checkpoint_bytes {
            return Err(LiveHostError::Persist(PersistError::OversizedInput));
        }
        // Fail closed if the new generation does not shadow-decode.
        self.engine.verify_checkpoint(&bytes)?;

        ensure_parent_dirs(&self.config)?;

        let tmp_path = self.config.tmp_checkpoint_path();
        write_synced_file(&tmp_path, &bytes)?;

        // Rotate `primary -> prev` ONLY if `primary` passes a full shadow decode; a corrupted
        // primary must never overwrite the last known-good `.prev` generation.
        if self.config.checkpoint_path.exists()
            && let Ok(existing_primary) =
                read_bounded_file(&self.config.checkpoint_path, self.config.max_checkpoint_bytes)
            && self.engine.verify_checkpoint(&existing_primary).is_ok()
        {
            let prev_tmp = self.config.prev_tmp_checkpoint_path();
            let prev_path = self.config.prev_checkpoint_path();
            write_atomic_synced(&prev_tmp, &prev_path, &existing_primary)?;
        }

        fs::rename(&tmp_path, &self.config.checkpoint_path)?;
        sync_parent_dir(&self.config.checkpoint_path);

        self.checkpoints_saved = self.checkpoints_saved.saturating_add(1);
        Ok(bytes.len())
    }

    fn record_latency(&mut self, step_us: f64) {
        let clean = if step_us.is_finite() && step_us >= 0.0 {
            step_us
        } else {
            0.0
        };
        if self.latency_ring_us.len() < LATENCY_WINDOW_CAP {
            self.latency_ring_us.push(clean);
        } else {
            self.latency_ring_us[self.latency_cursor] = clean;
            self.latency_cursor = (self.latency_cursor + 1) % LATENCY_WINDOW_CAP;
        }
    }

    fn append_telemetry_line(&mut self, line: &str) -> Result<(), LiveHostError> {
        let max_bytes = self.config.max_log_bytes.max(512);
        let line_bytes = line.as_bytes();

        if self.current_log_bytes > 0 && self.current_log_bytes + line_bytes.len() > max_bytes {
            let rotated = self.config.rotated_telemetry_path();
            if self.config.telemetry_path.exists() {
                fs::rename(&self.config.telemetry_path, &rotated)?;
            }
            self.current_log_bytes = 0;
            self.log_rotations = self.log_rotations.saturating_add(1);
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.telemetry_path)?;
        file.write_all(line_bytes)?;
        file.flush()?;

        self.current_log_bytes += line_bytes.len();
        self.telemetry_lines_written = self.telemetry_lines_written.saturating_add(1);
        Ok(())
    }
}

/// Formats a single pure-JSONL snapshot line without external serialization crates.
#[must_use]
pub fn format_telemetry_jsonl(snap: &OrganismSnapshot, step_us: f64) -> String {
    let mut s = String::with_capacity(384);
    let _ = writeln!(
        s,
        "{{\"tick\":{},\"atp\":{},\"adenosine\":{},\"temperature\":{},\"cell_count\":{},\"tube_count\":{},\"sensor_pore_count\":{},\"mean_tau\":{},\"max_tau\":{},\"mean_interventional_agency\":{},\"subthreshold_rehearsals\":{},\"mitosis_births\":{},\"contradiction_splits\":{},\"used_state_bytes\":{},\"step_us\":{}}}",
        snap.tick,
        fmt_f32(snap.atp),
        fmt_f32(snap.adenosine),
        fmt_f32(snap.temperature),
        snap.cell_count,
        snap.tube_count,
        snap.sensor_pore_count,
        fmt_f32(snap.mean_tau),
        fmt_f32(snap.max_tau),
        fmt_f32(snap.mean_interventional_agency),
        snap.subthreshold_rehearsals,
        snap.mitosis_births,
        snap.contradiction_splits,
        snap.used_state_bytes,
        fmt_f64(step_us),
    );
    s
}

fn fmt_f32(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.6}")
    } else {
        "0.000000".to_string()
    }
}

fn fmt_f64(v: f64) -> String {
    if v.is_finite() && v >= 0.0 {
        format!("{v:.3}")
    } else {
        "0.000".to_string()
    }
}

fn append_ext(base: &Path, suffix: &str) -> PathBuf {
    let mut os = base.as_os_str().to_os_string();
    os.push(".");
    os.push(suffix);
    PathBuf::from(os)
}

fn ensure_parent_dirs(config: &LiveHostConfig) -> Result<(), LiveHostError> {
    if let Some(parent) = config.checkpoint_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    if let Some(parent) = config.telemetry_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn existing_file_len(path: &Path) -> usize {
    fs::metadata(path).map_or(0, |m| m.len() as usize)
}

/// Reads at most `max_bytes` from `path`, failing closed with `OversizedInput` if larger
/// or `TruncatedInput` if `0` bytes.
pub fn read_bounded_file(path: &Path, max_bytes: usize) -> Result<Vec<u8>, PersistError> {
    let file = File::open(path).map_err(|_| PersistError::IoFailure)?;
    let meta = file.metadata().map_err(|_| PersistError::IoFailure)?;
    if meta.len() == 0 {
        return Err(PersistError::TruncatedInput);
    }
    if meta.len() > max_bytes as u64 {
        return Err(PersistError::OversizedInput);
    }

    let limit = (max_bytes as u64).saturating_add(1);
    let mut buf = Vec::with_capacity(meta.len() as usize);
    file.take(limit)
        .read_to_end(&mut buf)
        .map_err(|_| PersistError::IoFailure)?;

    if buf.is_empty() {
        return Err(PersistError::TruncatedInput);
    }
    if buf.len() > max_bytes {
        return Err(PersistError::OversizedInput);
    }
    Ok(buf)
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> Result<(), LiveHostError> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_atomic_synced(tmp_path: &Path, dst_path: &Path, bytes: &[u8]) -> Result<(), LiveHostError> {
    write_synced_file(tmp_path, bytes)?;
    fs::rename(tmp_path, dst_path)?;
    sync_parent_dir(dst_path);
    Ok(())
}

fn sync_parent_dir(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

fn compute_percentiles(samples: &[f64]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let idx = |q: f64| -> usize { ((q * (n - 1) as f64).round() as usize).min(n - 1) };
    (sorted[idx(0.50)], sorted[idx(0.95)], sorted[idx(0.99)])
}

/// Bounded read of Linux `/proc/self/statm` resident set size in bytes (strictly separate from logical slab bytes).
#[must_use]
pub fn read_linux_rss_bytes() -> Option<usize> {
    let file = File::open("/proc/self/statm").ok()?;
    let mut s = String::with_capacity(128);
    file.take(128).read_to_string(&mut s).ok()?;
    let mut parts = s.split_whitespace();
    let _total_pages = parts.next()?;
    let rss_pages: usize = parts.next()?.parse().ok()?;
    // Standard 4096-byte page size on Linux x86_64/aarch64 without unsafe libc::sysconf.
    Some(rss_pages.saturating_mul(4096))
}
