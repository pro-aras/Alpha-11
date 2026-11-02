//! Source-code hygiene scanner, transactional persistence fail-closed auditor,
//! `FrozenPlasticity` / flagless lesion verifier, and step-latency + memory profiler.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use a11_medium::{
    ControlMode, FieldPulse, FieldView, OrganismEngine, OrganismSnapshot, PersistError, WorldMedium,
};

use crate::stats::EmpiricalDistribution;

/// Forbidden anthropomorphic or split-brain identifiers banned from `a11-core` production code.
pub const FORBIDDEN_IDENTIFIERS: &[&str] = &[
    "words",
    "meanings",
    "concepts",
    "thoughts",
    "nursery_mode",
    "word_count",
    "meaning_count",
    "concept_count",
    "thought_count",
];

/// Category of source-hygiene finding detected in production (non-`#[cfg(test)]`) Rust code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HygieneViolationKind {
    StringLiteral,
    ForbiddenIdentifier,
    UnsafeKeyword,
    UnwrapCall,
}

/// Single localized hygiene finding in a Rust source file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HygieneFinding {
    pub file: String,
    pub line: usize,
    pub kind: HygieneViolationKind,
    pub token: String,
}

/// Aggregated production-code hygiene report across one file or an entire crate directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceHygieneReport {
    pub files_scanned: usize,
    pub string_literal_count: usize,
    pub forbidden_identifier_count: usize,
    pub unsafe_block_count: usize,
    pub unwrap_call_count: usize,
    pub findings: Vec<HygieneFinding>,
}

impl SourceHygieneReport {
    /// Returns `true` iff zero string literals, zero forbidden identifiers, zero `unsafe` keywords,
    /// and zero `.unwrap(` calls appear in production paths.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.string_literal_count == 0
            && self.forbidden_identifier_count == 0
            && self.unsafe_block_count == 0
            && self.unwrap_call_count == 0
    }

    fn merge(&mut self, other: Self) {
        self.files_scanned += other.files_scanned;
        self.string_literal_count += other.string_literal_count;
        self.forbidden_identifier_count += other.forbidden_identifier_count;
        self.unsafe_block_count += other.unsafe_block_count;
        self.unwrap_call_count += other.unwrap_call_count;
        self.findings.extend(other.findings);
    }
}

/// Recursively scans all `.rs` files under `dir` (excluding `#[cfg(test)]` items and comments).
pub fn scan_crate_source(dir: &Path) -> io::Result<SourceHygieneReport> {
    let mut rs_files = Vec::new();
    collect_rs_files(dir, &mut rs_files)?;
    rs_files.sort();

    let mut total = SourceHygieneReport::default();
    for path in rs_files {
        let src = fs::read_to_string(&path)?;
        let rel = path.strip_prefix(dir).unwrap_or(&path);
        let label = rel.to_string_lossy().into_owned();
        let file_report = scan_source_str(&label, &src);
        total.merge(file_report);
    }
    Ok(total)
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    if dir.is_file() {
        if dir.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(dir.to_path_buf());
        }
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Scans a single Rust source buffer `src`, stripping comments and `#[cfg(test)]` items.
#[must_use]
pub fn scan_source_str(file_label: &str, src: &str) -> SourceHygieneReport {
    let mut report = SourceHygieneReport {
        files_scanned: 1,
        ..SourceHygieneReport::default()
    };
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    let mut line = 1usize;

    while pos < len {
        let (next_pos, next_line) = skip_ws_and_comments(bytes, pos, line);
        pos = next_pos;
        line = next_line;
        if pos >= len {
            break;
        }

        // Check for outer `#[cfg(test)]` attribute and skip the annotated item completely.
        if bytes[pos] == b'#'
            && let Some((after_attr_pos, after_attr_line, is_cfg_test)) =
                try_parse_outer_attribute(bytes, pos, line)
            && is_cfg_test
        {
            let (after_item_pos, after_item_line) =
                skip_cfg_test_item(bytes, after_attr_pos, after_attr_line);
            pos = after_item_pos;
            line = after_item_line;
            continue;
        }

        // Raw string / raw byte-string / raw c-string (`r"..."`, `r#"..."#`, `br#"..."#`, `cr#"..."#`)
        if let Some((end_pos, end_line, snippet)) = try_consume_raw_string(bytes, pos, line) {
            report.string_literal_count += 1;
            report.findings.push(HygieneFinding {
                file: file_label.to_string(),
                line,
                kind: HygieneViolationKind::StringLiteral,
                token: snippet,
            });
            pos = end_pos;
            line = end_line;
            continue;
        }

        // Byte or C quoted string (`b"..."`, `c"..."`)
        if (bytes[pos] == b'b' || bytes[pos] == b'c') && bytes.get(pos + 1) == Some(&b'"') {
            let start_line = line;
            let (end_pos, end_line, snippet) = consume_quoted_string(bytes, pos + 1, line);
            report.string_literal_count += 1;
            report.findings.push(HygieneFinding {
                file: file_label.to_string(),
                line: start_line,
                kind: HygieneViolationKind::StringLiteral,
                token: snippet,
            });
            pos = end_pos;
            line = end_line;
            continue;
        }

        // Byte character literal (`b'x'`, `b'\"'`)
        if bytes[pos] == b'b' && bytes.get(pos + 1) == Some(&b'\'') {
            let (end_pos, end_line) = consume_char_or_lifetime(bytes, pos + 1, line);
            pos = end_pos;
            line = end_line;
            continue;
        }

        // Standard string literal (`"..."`)
        if bytes[pos] == b'"' {
            let start_line = line;
            let (end_pos, end_line, snippet) = consume_quoted_string(bytes, pos, line);
            report.string_literal_count += 1;
            report.findings.push(HygieneFinding {
                file: file_label.to_string(),
                line: start_line,
                kind: HygieneViolationKind::StringLiteral,
                token: snippet,
            });
            pos = end_pos;
            line = end_line;
            continue;
        }

        // Character literal (`'a'`, `'"'`, `'\\'`) or lifetime (`'a`)
        if bytes[pos] == b'\'' {
            let (end_pos, end_line) = consume_char_or_lifetime(bytes, pos, line);
            pos = end_pos;
            line = end_line;
            continue;
        }

        // Raw identifier prefix `r#ident`
        if bytes[pos] == b'r'
            && bytes.get(pos + 1) == Some(&b'#')
            && bytes.get(pos + 2).is_some_and(|&b| is_ident_start(b))
        {
            pos += 2;
        }

        // Identifier / keyword token
        if is_ident_start(bytes[pos]) {
            let start = pos;
            pos += 1;
            while pos < len && is_ident_continue(bytes[pos]) {
                pos += 1;
            }
            let ident = &src[start..pos];
            if ident == "unsafe" {
                report.unsafe_block_count += 1;
                report.findings.push(HygieneFinding {
                    file: file_label.to_string(),
                    line,
                    kind: HygieneViolationKind::UnsafeKeyword,
                    token: ident.to_string(),
                });
            } else if ident == "unwrap" {
                let (peek_pos, _) = skip_ws_and_comments(bytes, pos, line);
                if bytes.get(peek_pos) == Some(&b'(') {
                    report.unwrap_call_count += 1;
                    report.findings.push(HygieneFinding {
                        file: file_label.to_string(),
                        line,
                        kind: HygieneViolationKind::UnwrapCall,
                        token: "unwrap(".to_string(),
                    });
                }
            }
            if is_forbidden_identifier(ident) {
                report.forbidden_identifier_count += 1;
                report.findings.push(HygieneFinding {
                    file: file_label.to_string(),
                    line,
                    kind: HygieneViolationKind::ForbiddenIdentifier,
                    token: ident.to_string(),
                });
            }
            continue;
        }

        if bytes[pos] == b'\n' {
            line += 1;
        }
        pos += 1;
    }

    report
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_forbidden_identifier(ident: &str) -> bool {
    let lower = ident.to_ascii_lowercase();
    if lower.contains("nursery_mode") {
        return true;
    }
    FORBIDDEN_IDENTIFIERS.iter().any(|&banned| lower == banned)
}

fn skip_ws_and_comments(bytes: &[u8], mut pos: usize, mut line: usize) -> (usize, usize) {
    let len = bytes.len();
    loop {
        while pos < len && bytes[pos].is_ascii_whitespace() {
            if bytes[pos] == b'\n' {
                line += 1;
            }
            pos += 1;
        }
        if pos + 1 < len && bytes[pos] == b'/' && bytes[pos + 1] == b'/' {
            pos += 2;
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        if pos + 1 < len && bytes[pos] == b'/' && bytes[pos + 1] == b'*' {
            pos += 2;
            let mut depth = 1usize;
            while pos < len && depth > 0 {
                if pos + 1 < len && bytes[pos] == b'/' && bytes[pos + 1] == b'*' {
                    depth += 1;
                    pos += 2;
                } else if pos + 1 < len && bytes[pos] == b'*' && bytes[pos + 1] == b'/' {
                    depth -= 1;
                    pos += 2;
                } else {
                    if bytes[pos] == b'\n' {
                        line += 1;
                    }
                    pos += 1;
                }
            }
            continue;
        }
        break;
    }
    (pos, line)
}

fn try_consume_raw_string(
    bytes: &[u8],
    pos: usize,
    mut line: usize,
) -> Option<(usize, usize, String)> {
    let len = bytes.len();
    let r_pos = if bytes[pos] == b'r' {
        pos
    } else if (bytes[pos] == b'b' || bytes[pos] == b'c') && bytes.get(pos + 1) == Some(&b'r') {
        pos + 1
    } else {
        return None;
    };

    let mut cur = r_pos + 1;
    let mut hashes = 0usize;
    while cur < len && bytes[cur] == b'#' {
        hashes += 1;
        cur += 1;
    }
    if cur >= len || bytes[cur] != b'"' {
        return None;
    }
    cur += 1;
    let content_start = cur;
    let mut content_end = len;
    while cur < len {
        if bytes[cur] == b'\n' {
            line += 1;
        }
        if bytes[cur] == b'"' {
            let mut matched = 0usize;
            while matched < hashes && bytes.get(cur + 1 + matched) == Some(&b'#') {
                matched += 1;
            }
            if matched == hashes {
                content_end = cur;
                cur += 1 + hashes;
                break;
            }
        }
        cur += 1;
    }
    let preview_end = content_end.min(content_start + 32);
    let snippet = String::from_utf8_lossy(&bytes[content_start..preview_end]).into_owned();
    Some((cur, line, format!("r\"{}\"", snippet)))
}

fn consume_quoted_string(bytes: &[u8], pos: usize, mut line: usize) -> (usize, usize, String) {
    let len = bytes.len();
    let mut cur = pos + 1;
    let content_start = cur;
    let mut content_end = len;
    while cur < len {
        let b = bytes[cur];
        if b == b'\\' {
            cur += 1;
            if cur < len {
                if bytes[cur] == b'\n' {
                    line += 1;
                }
                cur += 1;
            }
            continue;
        }
        if b == b'\n' {
            line += 1;
        }
        if b == b'"' {
            content_end = cur;
            cur += 1;
            break;
        }
        cur += 1;
    }
    let preview_end = content_end.min(content_start + 32);
    let snippet = String::from_utf8_lossy(&bytes[content_start..preview_end]).into_owned();
    (cur, line, format!("\"{}\"", snippet))
}

fn consume_char_or_lifetime(bytes: &[u8], pos: usize, line: usize) -> (usize, usize) {
    let len = bytes.len();
    if pos + 1 >= len {
        return (pos + 1, line);
    }
    if bytes[pos + 1] == b'\\' {
        let mut cur = pos + 2;
        while cur < len && bytes[cur] != b'\n' {
            if bytes[cur] == b'\'' {
                return (cur + 1, line);
            }
            cur += 1;
        }
        return (pos + 1, line);
    }
    if let Ok(rest) = std::str::from_utf8(&bytes[(pos + 1)..])
        && let Some(ch) = rest.chars().next()
        && ch != '\''
        && ch != '\n'
    {
        let close_idx = pos + 1 + ch.len_utf8();
        if bytes.get(close_idx) == Some(&b'\'') {
            return (close_idx + 1, line);
        }
    }
    (pos + 1, line)
}

fn try_parse_outer_attribute(
    bytes: &[u8],
    pos: usize,
    line: usize,
) -> Option<(usize, usize, bool)> {
    let (after_hash_pos, after_hash_line) = skip_ws_and_comments(bytes, pos + 1, line);
    if bytes.get(after_hash_pos) != Some(&b'[') {
        return None;
    }
    let len = bytes.len();
    let mut cur = after_hash_pos + 1;
    let mut cur_line = after_hash_line;
    let mut bracket_depth = 1usize;
    let mut normalized = String::new();

    while cur < len && bracket_depth > 0 {
        let (p, l) = skip_ws_and_comments(bytes, cur, cur_line);
        cur = p;
        cur_line = l;
        if cur >= len {
            break;
        }
        if let Some((end_p, end_l, _)) = try_consume_raw_string(bytes, cur, cur_line) {
            cur = end_p;
            cur_line = end_l;
            continue;
        }
        if bytes[cur] == b'"' {
            let (end_p, end_l, _) = consume_quoted_string(bytes, cur, cur_line);
            cur = end_p;
            cur_line = end_l;
            continue;
        }
        if bytes[cur] == b'\'' {
            let (end_p, end_l) = consume_char_or_lifetime(bytes, cur, cur_line);
            cur = end_p;
            cur_line = end_l;
            continue;
        }
        let b = bytes[cur];
        if b == b'[' {
            bracket_depth += 1;
            normalized.push('[');
        } else if b == b']' {
            bracket_depth -= 1;
            if bracket_depth > 0 {
                normalized.push(']');
            }
        } else if !b.is_ascii_whitespace() && b.is_ascii() {
            normalized.push(b as char);
        }
        if b == b'\n' {
            cur_line += 1;
        }
        cur += 1;
    }

    let is_cfg_test = normalized == "cfg(test)";
    Some((cur, cur_line, is_cfg_test))
}

fn skip_cfg_test_item(bytes: &[u8], mut pos: usize, mut line: usize) -> (usize, usize) {
    let len = bytes.len();
    // Skip any additional outer attributes attached to the same test item.
    loop {
        let (p, l) = skip_ws_and_comments(bytes, pos, line);
        pos = p;
        line = l;
        if pos < len
            && bytes[pos] == b'#'
            && let Some((after_p, after_l, _)) = try_parse_outer_attribute(bytes, pos, line)
        {
            pos = after_p;
            line = after_l;
            continue;
        }
        break;
    }

    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut brace_depth = 0usize;
    let mut entered_body_brace = false;

    while pos < len {
        let (p, l) = skip_ws_and_comments(bytes, pos, line);
        pos = p;
        line = l;
        if pos >= len {
            break;
        }
        if let Some((end_p, end_l, _)) = try_consume_raw_string(bytes, pos, line) {
            pos = end_p;
            line = end_l;
            continue;
        }
        if (bytes[pos] == b'b' || bytes[pos] == b'c') && bytes.get(pos + 1) == Some(&b'"') {
            let (end_p, end_l, _) = consume_quoted_string(bytes, pos + 1, line);
            pos = end_p;
            line = end_l;
            continue;
        }
        if bytes[pos] == b'b' && bytes.get(pos + 1) == Some(&b'\'') {
            let (end_p, end_l) = consume_char_or_lifetime(bytes, pos + 1, line);
            pos = end_p;
            line = end_l;
            continue;
        }
        if bytes[pos] == b'"' {
            let (end_p, end_l, _) = consume_quoted_string(bytes, pos, line);
            pos = end_p;
            line = end_l;
            continue;
        }
        if bytes[pos] == b'\'' {
            let (end_p, end_l) = consume_char_or_lifetime(bytes, pos, line);
            pos = end_p;
            line = end_l;
            continue;
        }

        match bytes[pos] {
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            b'{' => {
                if paren_depth == 0 && bracket_depth == 0 {
                    entered_body_brace = true;
                }
                brace_depth += 1;
            }
            b'}' => {
                brace_depth = brace_depth.saturating_sub(1);
                if entered_body_brace && brace_depth == 0 && paren_depth == 0 && bracket_depth == 0
                {
                    return (pos + 1, line);
                }
            }
            b';' if !entered_body_brace
                && paren_depth == 0
                && bracket_depth == 0
                && brace_depth == 0 =>
            {
                return (pos + 1, line);
            }
            b'\n' => line += 1,
            _ => {}
        }
        pos += 1;
    }
    (pos, line)
}

/// Configuration for `audit_persistence_with_config`.
#[derive(Clone, Copy, Debug)]
pub struct PersistenceAuditConfig {
    pub seed: u64,
    pub warmup_steps: usize,
    pub continuation_steps: usize,
    /// Size of synthetic oversized buffer injected to verify bounded checkpoint decode.
    pub oversized_bytes_len: usize,
}

impl PersistenceAuditConfig {
    /// Standard persistence audit config (`100` warmup + `100` continuation steps, `128 MiB + 1` oversized test).
    #[must_use]
    pub const fn default_with_seed(seed: u64) -> Self {
        Self {
            seed,
            warmup_steps: 100,
            continuation_steps: 100,
            oversized_bytes_len: 128 * 1024 * 1024 + 1,
        }
    }
}

/// Detailed outcome of a single corruption-injection fail-closed check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorruptionCaseOutcome {
    pub label: String,
    pub returned_error: Option<PersistError>,
    pub poisoned_after_reject: bool,
    pub live_hash_preserved: bool,
    pub step_blocked_with_poison: bool,
    pub encode_blocked_with_poison: bool,
}

impl CorruptionCaseOutcome {
    /// Returns `true` iff the corrupted input returned `Err(...)`, preserved pre-existing RAM hash,
    /// marked the organism poisoned, and blocked subsequent `step()` and `encode_checkpoint()`.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.returned_error.is_some()
            && self.poisoned_after_reject
            && self.live_hash_preserved
            && self.step_blocked_with_poison
            && self.encode_blocked_with_poison
    }
}

/// Full report returned by `audit_persistence`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistenceAuditReport {
    pub checkpoint_bytes_at_100: usize,
    pub restore_at_100_snapshot_equal: bool,
    pub restore_at_100_hash_equal: bool,
    pub continuation_200_pulses_bit_identical: bool,
    pub continuation_200_snapshot_bit_identical: bool,
    pub continuation_200_hash_identical: bool,
    pub payload_bitflip_case: CorruptionCaseOutcome,
    pub header_or_crc_bitflip_cases: Vec<CorruptionCaseOutcome>,
    pub truncation_cases: Vec<CorruptionCaseOutcome>,
    pub oversized_case: CorruptionCaseOutcome,
}

impl PersistenceAuditReport {
    /// Returns `true` iff deterministic 200-step continuation AND every corruption/fail-closed test passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checkpoint_bytes_at_100 > 0
            && self.restore_at_100_snapshot_equal
            && self.restore_at_100_hash_equal
            && self.continuation_200_pulses_bit_identical
            && self.continuation_200_snapshot_bit_identical
            && self.continuation_200_hash_identical
            && self.payload_bitflip_case.passed()
            && !self.header_or_crc_bitflip_cases.is_empty()
            && self
                .header_or_crc_bitflip_cases
                .iter()
                .all(CorruptionCaseOutcome::passed)
            && self.truncation_cases.len() == 5
            && self
                .truncation_cases
                .iter()
                .all(CorruptionCaseOutcome::passed)
            && self.oversized_case.passed()
    }
}

/// Executes the full transactional persistence & fail-closed audit (`100 + 100` continuation + corruptions).
pub fn audit_persistence<O, W, FO, FW>(
    seed: u64,
    make_organism: FO,
    make_world: FW,
) -> Result<PersistenceAuditReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    audit_persistence_with_config(
        PersistenceAuditConfig::default_with_seed(seed),
        make_organism,
        make_world,
    )
}

/// Executes the transactional persistence & fail-closed audit with explicit configuration.
pub fn audit_persistence_with_config<O, W, FO, FW>(
    config: PersistenceAuditConfig,
    make_organism: FO,
    make_world: FW,
) -> Result<PersistenceAuditReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    let mut org_uninterrupted = make_organism(config.seed);
    let mut world_a = make_world(config.seed);
    let mut warmup_pulses = Vec::with_capacity(config.warmup_steps);

    for _ in 0..config.warmup_steps {
        let view = world_a.current_view();
        let pulse = org_uninterrupted.step(view)?;
        world_a.apply_pulse(&pulse);
        warmup_pulses.push(pulse);
    }

    let checkpoint_100 = org_uninterrupted.encode_checkpoint()?;
    let snap_100 = org_uninterrupted.snapshot();
    let hash_100 = org_uninterrupted.learned_parameter_hash();

    // Advance a second identical world to tick 100 using the exact recorded pulses,
    // and restore checkpoint_100 into a fresh organism created with a DIFFERENT seed.
    let mut world_b = make_world(config.seed);
    for pulse in &warmup_pulses {
        world_b.apply_pulse(pulse);
    }
    let mut org_restored = make_organism(config.seed.wrapping_add(0xDEAD_BEEF_CAFE_1234));
    org_restored.restore_checkpoint(&checkpoint_100)?;

    let restore_at_100_snapshot_equal =
        snapshots_bit_identical(&snap_100, &org_restored.snapshot());
    let restore_at_100_hash_equal = hash_100 == org_restored.learned_parameter_hash();

    let mut continuation_200_pulses_bit_identical = true;
    for _ in 0..config.continuation_steps {
        let view_a = world_a.current_view();
        let pulse_a = org_uninterrupted.step(view_a)?;
        world_a.apply_pulse(&pulse_a);

        let view_b = world_b.current_view();
        let pulse_b = org_restored.step(view_b)?;
        world_b.apply_pulse(&pulse_b);

        if !pulses_bit_identical(&pulse_a, &pulse_b) {
            continuation_200_pulses_bit_identical = false;
        }
    }

    let continuation_200_snapshot_bit_identical =
        snapshots_bit_identical(&org_uninterrupted.snapshot(), &org_restored.snapshot());
    let continuation_200_hash_identical =
        org_uninterrupted.learned_parameter_hash() == org_restored.learned_parameter_hash();

    // (a) Single-bit flip in payload (3/4 into buffer, away from leading header).
    let payload_case = if checkpoint_100.is_empty() {
        run_corruption_case(
            "payload_bitflip_empty",
            config.seed,
            &[],
            &make_organism,
            &make_world,
        )?
    } else {
        let mut mutated = checkpoint_100.clone();
        let idx = (mutated.len() * 3 / 4).min(mutated.len() - 1);
        mutated[idx] ^= 0x01;
        run_corruption_case(
            "payload_single_bitflip",
            config.seed,
            &mutated,
            &make_organism,
            &make_world,
        )?
    };

    // (b) Single-bit flip in header (byte 0) and CRC/header tail (byte min(4, len-1) or len-1).
    let mut header_or_crc_cases = Vec::new();
    if !checkpoint_100.is_empty() {
        let mut flip_header = checkpoint_100.clone();
        flip_header[0] ^= 0x01;
        header_or_crc_cases.push(run_corruption_case(
            "header_byte0_bitflip",
            config.seed,
            &flip_header,
            &make_organism,
            &make_world,
        )?);

        let crc_idx = if checkpoint_100.len() >= 9 {
            8
        } else {
            checkpoint_100.len() - 1
        };
        let mut flip_crc = checkpoint_100.clone();
        flip_crc[crc_idx] ^= 0x80;
        header_or_crc_cases.push(run_corruption_case(
            "header_or_crc_bitflip",
            config.seed,
            &flip_crc,
            &make_organism,
            &make_world,
        )?);
    }

    // (c) Truncated byte slices at 5 distinct offsets.
    let offsets = choose_five_truncation_offsets(checkpoint_100.len());
    let mut truncation_cases = Vec::with_capacity(offsets.len());
    for (k, &cut) in offsets.iter().enumerate() {
        let slice = &checkpoint_100[..cut.min(checkpoint_100.len())];
        let label = format!("truncated_offset_{k}_len_{cut}");
        truncation_cases.push(run_corruption_case(
            &label,
            config.seed,
            slice,
            &make_organism,
            &make_world,
        )?);
    }

    // (d) Oversized input.
    let over_len = config.oversized_bytes_len.max(checkpoint_100.len() + 1);
    let mut oversized = vec![0_u8; over_len];
    let copy_len = checkpoint_100.len().min(over_len);
    oversized[..copy_len].copy_from_slice(&checkpoint_100[..copy_len]);
    let oversized_case = run_corruption_case(
        "oversized_input",
        config.seed,
        &oversized,
        &make_organism,
        &make_world,
    )?;

    Ok(PersistenceAuditReport {
        checkpoint_bytes_at_100: checkpoint_100.len(),
        restore_at_100_snapshot_equal,
        restore_at_100_hash_equal,
        continuation_200_pulses_bit_identical,
        continuation_200_snapshot_bit_identical,
        continuation_200_hash_identical,
        payload_bitflip_case: payload_case,
        header_or_crc_bitflip_cases: header_or_crc_cases,
        truncation_cases,
        oversized_case,
    })
}

fn choose_five_truncation_offsets(total_len: usize) -> [usize; 5] {
    if total_len >= 8 {
        [
            0,
            1,
            total_len / 4,
            total_len / 2,
            total_len.saturating_sub(1),
        ]
    } else if total_len >= 5 {
        [0, 1, 2, 3, 4]
    } else {
        [0, 0, 0, 0, 0]
    }
}

fn run_corruption_case<O, W, FO, FW>(
    label: &str,
    seed: u64,
    corrupted: &[u8],
    make_organism: &FO,
    make_world: &FW,
) -> Result<CorruptionCaseOutcome, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    let mut victim = make_organism(seed);
    let mut world = make_world(seed);
    for _ in 0..5 {
        let view = world.current_view();
        let pulse = victim.step(view)?;
        world.apply_pulse(&pulse);
    }
    let pre_hash = victim.learned_parameter_hash();

    let restore_res = victim.restore_checkpoint(corrupted);
    let returned_error = restore_res.err();
    let poisoned_after_reject = victim.is_poisoned();
    let live_hash_preserved = victim.learned_parameter_hash() == pre_hash;

    let dummy_ambient = [0.0_f32; 8];
    let dummy_view = FieldView {
        ambient_in: &dummy_ambient[..world.num_ambient_channels().min(dummy_ambient.len())],
        metabolic_influx: 0.0,
        tissue_stress: 0.0,
        num_actuators: world.num_actuators(),
    };
    let step_res = if world.num_ambient_channels() <= dummy_ambient.len() {
        victim.step(dummy_view)
    } else {
        let view = world.current_view();
        victim.step(view)
    };
    let step_blocked_with_poison = step_res == Err(PersistError::PoisonedAfterCorruption);
    let encode_blocked_with_poison =
        victim.encode_checkpoint() == Err(PersistError::PoisonedAfterCorruption);

    Ok(CorruptionCaseOutcome {
        label: label.to_string(),
        returned_error,
        poisoned_after_reject,
        live_hash_preserved,
        step_blocked_with_poison,
        encode_blocked_with_poison,
    })
}

/// Bit-for-bit equality check on `FieldPulse` (`f32::to_bits`).
#[must_use]
pub fn pulses_bit_identical(a: &FieldPulse, b: &FieldPulse) -> bool {
    a.actuator_out.len() == b.actuator_out.len()
        && a.actuator_out
            .iter()
            .zip(b.actuator_out.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Bit-for-bit equality check on `OrganismSnapshot` (using `f32::to_bits` on all floating-point fields).
#[must_use]
pub fn snapshots_bit_identical(a: &OrganismSnapshot, b: &OrganismSnapshot) -> bool {
    a.tick == b.tick
        && a.atp.to_bits() == b.atp.to_bits()
        && a.adenosine.to_bits() == b.adenosine.to_bits()
        && a.temperature.to_bits() == b.temperature.to_bits()
        && a.motor_clutch_threshold.to_bits() == b.motor_clutch_threshold.to_bits()
        && a.cell_count == b.cell_count
        && a.tube_count == b.tube_count
        && a.sensor_pore_count == b.sensor_pore_count
        && a.motor_pore_count == b.motor_pore_count
        && a.active_cells_last_tick == b.active_cells_last_tick
        && a.stiff_highway_tubes == b.stiff_highway_tubes
        && a.inhibitory_tubes == b.inhibitory_tubes
        && a.active_echo_delay_tubes == b.active_echo_delay_tubes
        && a.mean_tau.to_bits() == b.mean_tau.to_bits()
        && a.max_tau.to_bits() == b.max_tau.to_bits()
        && a.tau_octave_count == b.tau_octave_count
        && a.mean_interventional_agency.to_bits() == b.mean_interventional_agency.to_bits()
        && a.mean_prediction_error.to_bits() == b.mean_prediction_error.to_bits()
        && a.committed_motor_pulses == b.committed_motor_pulses
        && a.subthreshold_rehearsals == b.subthreshold_rehearsals
        && a.rest_consolidation_ticks == b.rest_consolidation_ticks
        && a.mitosis_births == b.mitosis_births
        && a.contradiction_splits == b.contradiction_splits
        && a.myelinated_shortcuts == b.myelinated_shortcuts
        && a.atrophied_structures == b.atrophied_structures
        && a.sensor_pore_channels == b.sensor_pore_channels
        && f32_slices_bit_identical(&a.sensor_pore_conductances, &b.sensor_pore_conductances)
        && a.sensor_pore_delays == b.sensor_pore_delays
        && a.motor_pore_channels == b.motor_pore_channels
        && f32_slices_bit_identical(&a.motor_pore_conductances, &b.motor_pore_conductances)
        && a.used_state_bytes == b.used_state_bytes
}

fn f32_slices_bit_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Report returned by `audit_plasticity_and_lesion`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlasticityLesionAuditReport {
    pub frozen_steps_tested: usize,
    pub max_frozen_hash_changed_bits: u32,
    pub frozen_topology_invariant: bool,
    pub post_growth_frozen_hash_changed_bits: u32,
    pub lesion_preserved_natural_mode: bool,
    pub lesion_zeroed_all_pore_conductances: bool,
}

impl PlasticityLesionAuditReport {
    /// Returns `true` iff `FrozenPlasticity` mutated `0` bits across 500 steps AND flagless lesion
    /// zeroed conductances while preserving `ControlMode::Natural`.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.frozen_steps_tested >= 500
            && self.max_frozen_hash_changed_bits == 0
            && self.frozen_topology_invariant
            && self.post_growth_frozen_hash_changed_bits == 0
            && self.lesion_preserved_natural_mode
            && self.lesion_zeroed_all_pore_conductances
    }
}

/// Verifies `FrozenPlasticity` 500-step parameter-hash invariance and flagless `zero_learned_conductances()`.
pub fn audit_plasticity_and_lesion<O, W, FO, FW>(
    seed: u64,
    frozen_steps: usize,
    make_organism: FO,
    make_world: FW,
) -> Result<PlasticityLesionAuditReport, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
    FO: Fn(u64) -> O,
    FW: Fn(u64) -> W,
{
    let steps = frozen_steps.max(500);

    // 1. Test 500+ steps of FrozenPlasticity from tick 0.
    let mut frozen_org = make_organism(seed);
    let mut frozen_world = make_world(seed);
    frozen_org.set_control_mode(ControlMode::FrozenPlasticity);
    let init_hash = frozen_org.learned_parameter_hash();
    let init_snap = frozen_org.snapshot();
    let mut max_frozen_hash_changed_bits = 0_u32;
    let mut frozen_topology_invariant = true;

    for _ in 0..steps {
        let view = frozen_world.current_view();
        let pulse = frozen_org.step(view)?;
        frozen_world.apply_pulse(&pulse);

        let cur_hash = frozen_org.learned_parameter_hash();
        let changed = (cur_hash ^ init_hash).count_ones();
        if changed > max_frozen_hash_changed_bits {
            max_frozen_hash_changed_bits = changed;
        }
    }
    let end_snap = frozen_org.snapshot();
    if end_snap.cell_count != init_snap.cell_count
        || end_snap.tube_count != init_snap.tube_count
        || end_snap.sensor_pore_channels != init_snap.sensor_pore_channels
        || !f32_slices_bit_identical(
            &end_snap.sensor_pore_conductances,
            &init_snap.sensor_pore_conductances,
        )
        || !f32_slices_bit_identical(
            &end_snap.motor_pore_conductances,
            &init_snap.motor_pore_conductances,
        )
    {
        frozen_topology_invariant = false;
    }

    // 2. Grow an organism in Natural mode for 100 steps, verify post-growth FrozenPlasticity invariance,
    // and then verify flagless `zero_learned_conductances()` under `ControlMode::Natural`.
    let mut active_org = make_organism(seed);
    let mut active_world = make_world(seed);
    active_org.set_control_mode(ControlMode::Natural);
    for _ in 0..100 {
        let view = active_world.current_view();
        let pulse = active_org.step(view)?;
        active_world.apply_pulse(&pulse);
    }

    active_org.set_control_mode(ControlMode::FrozenPlasticity);
    let grown_hash = active_org.learned_parameter_hash();
    let mut post_growth_frozen_hash_changed_bits = 0_u32;
    for _ in 0..100 {
        let view = active_world.current_view();
        let pulse = active_org.step(view)?;
        active_world.apply_pulse(&pulse);
        let diff = (active_org.learned_parameter_hash() ^ grown_hash).count_ones();
        if diff > post_growth_frozen_hash_changed_bits {
            post_growth_frozen_hash_changed_bits = diff;
        }
    }

    active_org.set_control_mode(ControlMode::Natural);
    active_org.zero_learned_conductances();
    let lesion_preserved_natural_mode = active_org.control_mode() == ControlMode::Natural;
    let lesioned_snap = active_org.snapshot();
    let lesion_zeroed_all_pore_conductances = lesioned_snap
        .sensor_pore_conductances
        .iter()
        .all(|&c| c == 0.0)
        && lesioned_snap
            .motor_pore_conductances
            .iter()
            .all(|&c| c == 0.0);

    Ok(PlasticityLesionAuditReport {
        frozen_steps_tested: steps,
        max_frozen_hash_changed_bits,
        frozen_topology_invariant,
        post_growth_frozen_hash_changed_bits,
        lesion_preserved_natural_mode,
        lesion_zeroed_all_pore_conductances,
    })
}

/// Linux OS process memory telemetry parsed from `/proc/self/status`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcMemoryStats {
    pub vm_rss_bytes: usize,
    pub vm_hwm_bytes: usize,
}

/// Reads current process `VmRSS` and `VmHWM` from `/proc/self/status` on Linux.
#[must_use]
pub fn read_linux_proc_memory() -> Option<ProcMemoryStats> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    parse_proc_status_memory(&text)
}

/// Parses `VmRSS:` and `VmHWM:` lines (in kB) from `/proc/self/status` text.
#[must_use]
pub fn parse_proc_status_memory(status_text: &str) -> Option<ProcMemoryStats> {
    let mut rss_kb: Option<usize> = None;
    let mut hwm_kb: Option<usize> = None;
    for line in status_text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = parse_kb_value(rest);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm_kb = parse_kb_value(rest);
        }
    }
    match (rss_kb, hwm_kb) {
        (Some(rss), Some(hwm)) => Some(ProcMemoryStats {
            vm_rss_bytes: rss.saturating_mul(1024),
            vm_hwm_bytes: hwm.saturating_mul(1024),
        }),
        (Some(rss), None) => Some(ProcMemoryStats {
            vm_rss_bytes: rss.saturating_mul(1024),
            vm_hwm_bytes: rss.saturating_mul(1024),
        }),
        _ => None,
    }
}

fn parse_kb_value(s: &str) -> Option<usize> {
    let token = s.split_whitespace().next()?;
    token.parse::<usize>().ok()
}

/// Worst-case step latency and logical/OS memory profile across `steps` closed-loop ticks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LatencyMemoryProfile {
    pub steps_profiled: usize,
    pub num_ambient_channels: usize,
    pub num_actuators: usize,
    /// Per-step `OrganismEngine::step` duration distribution in microseconds (`us`).
    pub step_latency_us: EmpiricalDistribution,
    /// Full closed-loop tick (`step` + `apply_pulse`) duration distribution in microseconds (`us`).
    pub full_tick_latency_us: EmpiricalDistribution,
    pub peak_used_state_bytes: usize,
    pub final_used_state_bytes: usize,
    pub checkpoint_bytes: usize,
    pub os_memory: Option<ProcMemoryStats>,
}

/// Profiles per-step latency (`ns`/`us` -> `p50`, `p95`, `p99`, `max`), logical `used_state_bytes`,
/// serialized `checkpoint_bytes`, and Linux `/proc/self/status` `VmRSS` / `VmHWM`.
pub fn profile_latency<O, W>(
    organism: &mut O,
    world: &mut W,
    steps: usize,
) -> Result<LatencyMemoryProfile, PersistError>
where
    O: OrganismEngine,
    W: WorldMedium,
{
    let mut step_us = Vec::with_capacity(steps);
    let mut full_us = Vec::with_capacity(steps);
    let mut peak_used_state_bytes = organism.snapshot().used_state_bytes;

    for _ in 0..steps {
        let t_tick_start = Instant::now();
        let pulse = {
            let view = world.current_view();
            let t_step_start = Instant::now();
            let out = organism.step(view)?;
            let step_elapsed = t_step_start.elapsed();
            step_us.push((step_elapsed.as_nanos() as f64) / 1_000.0);
            out
        };
        world.apply_pulse(&pulse);
        let tick_elapsed = t_tick_start.elapsed();
        full_us.push((tick_elapsed.as_nanos() as f64) / 1_000.0);

        // Sample peak logical memory periodically + on final tick without polluting `step_us`.
        let used = organism.snapshot().used_state_bytes;
        if used > peak_used_state_bytes {
            peak_used_state_bytes = used;
        }
    }

    let final_used_state_bytes = organism.snapshot().used_state_bytes;
    if final_used_state_bytes > peak_used_state_bytes {
        peak_used_state_bytes = final_used_state_bytes;
    }
    let checkpoint_bytes = organism.encode_checkpoint()?.len();
    let os_memory = read_linux_proc_memory();

    Ok(LatencyMemoryProfile {
        steps_profiled: steps,
        num_ambient_channels: world.num_ambient_channels(),
        num_actuators: world.num_actuators(),
        step_latency_us: EmpiricalDistribution::from_f64_slice(&step_us),
        full_tick_latency_us: EmpiricalDistribution::from_f64_slice(&full_us),
        peak_used_state_bytes,
        final_used_state_bytes,
        checkpoint_bytes,
        os_memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_scanner_catches_strings_and_forbidden_idents_while_ignoring_tests_and_comments() {
        let sample_rs = r#"
//! Module doc comment mentioning "words", concepts, nursery_mode, and unwrap() — must be ignored!
/* Multi-line /* nested "string" nursery_mode unsafe unwrap() */ comment */

pub struct CleanCell<'a> {
    pub v_real: &'a [f32],
    pub quote_char: char,
}

pub fn dirty_production_fn(flag: bool) -> usize {
    let ch = '"';
    let nursery_mode = flag;
    let thoughts = 42usize;
    let bad_label = "implanted_english_word";
    let opt = Some(1usize);
    let v = opt.unwrap();
    let _ = ch;
    let _ = nursery_mode;
    let _ = bad_label;
    thoughts + v
}

#[cfg(test)]
mod tests {
    #[test]
    fn allowed_in_tests() {
        let words = "allowed inside cfg(test) with } tricky brace";
        let meanings = Some(words).unwrap();
        assert_eq!(meanings.len(), 44);
    }
}

#[cfg(test)]
const EXTRA_TEST_CONST: [&str; 2] = ["ignored_one", "ignored_two"];
"#;

        let report = scan_source_str("synthetic.rs", sample_rs);
        assert!(!report.is_clean());
        assert_eq!(report.string_literal_count, 1);
        // `nursery_mode` appears twice (decl + use) and `thoughts` appears twice (decl + use).
        assert_eq!(report.forbidden_identifier_count, 4);
        assert_eq!(report.unwrap_call_count, 1);
        assert_eq!(report.unsafe_block_count, 0);

        let clean_rs = r#"
//! Pure numeric code with lifetimes `'a` and char literals `'"'`.
pub fn step_wave<'a>(input: &'a [f32]) -> f32 {
    let delimiter = '"';
    let _ = delimiter;
    input.first().copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn check() {
        let words = "fine in tests";
        assert_eq!(words.len(), 13);
    }
}
"#;
        let clean_report = scan_source_str("clean.rs", clean_rs);
        assert!(clean_report.is_clean(), "expected clean report, got {clean_report:?}");
    }

    #[test]
    fn proc_status_parser_extracts_rss_and_hwm() {
        let sample = "\
Name:\ta11-runner\n\
VmPeak:\t   65536 kB\n\
VmSize:\t   61200 kB\n\
VmHWM:\t   42000 kB\n\
VmRSS:\t   38500 kB\n";
        let mem = parse_proc_status_memory(sample);
        assert_eq!(
            mem,
            Some(ProcMemoryStats {
                vm_rss_bytes: 38_500 * 1024,
                vm_hwm_bytes: 42_000 * 1024,
            })
        );
        if Path::new("/proc/self/status").exists() {
            let live = read_linux_proc_memory();
            assert!(live.is_some());
            if let Some(m) = live {
                assert!(m.vm_rss_bytes > 0);
            }
        }
    }
}
