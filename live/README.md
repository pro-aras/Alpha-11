# Alpha 11 Live Host (`a11-live`)

`a11-live` (`crates/a11-live/src/main.rs`) runs continuous, crash-safe, unforced embodied life for `Alpha11Organism` with transactional `CRC32C` shadow-decoded checkpoints, `.prev` generation fallback, live JSON/JSONL telemetry, and unedited `16 kHz` `.wav` acoustic recordings.

## Build

```bash
cargo build --release -p a11-live
```

## 1. Embodied Open Valley Habitat (`--habitat open`, Default)

Runs a growing focal `Alpha11Organism` (`276` sensory channels, `17` continuous muscles) cohabiting continuously with an autonomous persisted Companion `Elder` `Alpha11Organism` inside `OpenSocialHabitat`:
- **Living Earth Ecology (`MultiGenTerrarium`)**: `8x8` photonic retina, `2..14`-tick wave echoes, seasonal springs with `16 kHz` stereo water murmurs (`ITD`/`ILD`), opaque basalt canyon ridges (blocking `100%` direct sight while diffracting `16 kHz` airborne voice, water, and woody shell-taps at `0.68x`), `6` solo valley pods + `4` obligate two-organism heavy grove pods, pushable mineral boulder anvils, thermal shelters, day/night cycles, and somatic-gated terrain trails.
- **Embodied Body Organs (`EmbodiedBodyOrgans`)**: `DrawingScratchpad` tactile/visual hands + shared stone-hearth contact, `BalanceCatchToy` vestibular limb coupled to locomotion sway and manual shell-cracking stability, and `BinauralVocalOrgan` (`16 kHz` articulatory `PhysicalVocalTract` maturing ontogenetically from `8.8 cm -> 14.5 cm` + `32`-band `ErbCochlea32` & `LSO` binaural ears).

```bash
# Continuous real-time 1.0x (50 Hz = 20 ms/tick) background daemon
./target/release/a11-live \
  --state-dir live/state/open \
  --habitat open \
  --steps 0 \
  --pace-ms 20

# Finite unthrottled run (e.g., 5,000 ticks)
./target/release/a11-live \
  --state-dir live/state/open \
  --habitat open \
  --seed 11 \
  --steps 5000
```

### Graceful Stop & Zero-Loss Binary Hot-Swap

Touch `<state-dir>/STOP` to flush both `organism.ckpt` and `elder.ckpt` cleanly on the next 16-tick (`<= 320 ms`) boundary and exit with code `0`:

```bash
touch live/state/open/STOP
# After building an updated binary, remove STOP and resume from the exact tick:
rm -f live/state/open/STOP
./target/release/a11-live --state-dir live/state/open --habitat open --steps 0 --pace-ms 20
```

### State Directory Artifacts (`live/state/open/`)

All runtime state under `live/state/` is gitignored:
- `organism.ckpt` & `organism.ckpt.prev`: Focal organism transactional `CRC32C` checkpoints.
- `elder.ckpt` & `elder.ckpt.prev`: Cohabiting Companion Elder transactional `CRC32C` checkpoints.
- `development_status.json`: Atomic snapshot updated every `250` ticks (`5 s` real time) with brain morphogenesis, sensory pore counts, muscle mastery/jerk, hand/balance metrics, `16 kHz` vocal formant/pitch/multi-syllable contour stats, Earth ecology cooperation/occlusion telemetry, and OS RSS/latency.
- `telemetry.jsonl`: Bounded rotating append-only JSONL telemetry stream.
- `newborn_first_10s.wav`: Unedited first `10 s` (`16 kHz` mono PCM) captured at `Genesis` (`tick = 0`).
- `latest_10s_voice_mono.wav`: Rolling `10 s` (`16 kHz` mono PCM) isolated focal vocal tract waveform.
- `latest_10s_room_stereo.wav`: Rolling `10 s` (`16 kHz` stereo PCM) full binaural valley soundscape (focal + companion voices, cave echoes, flowing springs, shell-taps, and boulder rumbles).

---

## 2. Composite Portal Habitat (`--habitat composite`)

Runs a single `Alpha11Organism` inside `AdaptiveCaveWorld` (with smooth day/night illumination cycles) multiplexed with all 4 unforced human-activity portals (`DrawingScratchpad`, `PatternMemoryBoard`, `RhythmEchoPad`, `BalanceCatchToy`) via `CompositePortalMedium`:

```bash
./target/release/a11-live \
  --state-dir live/state/composite \
  --habitat composite \
  --seed 11 \
  --steps 5000
```

---

## 3. Multi-Generational Cohort Terrarium (`--habitat terrarium`)

Runs a 4-organism cohort (`2` founding elders + `2` younger organisms joining after a `400`-tick elder lead) inside a shared `MultiGenTerrarium` with atomic 4-organism `terrarium_population.ckpt` + `.prev` bundles:

```bash
./target/release/a11-live \
  --state-dir live/state/terrarium \
  --habitat terrarium \
  --seed 11 \
  --steps 3000
```

## Crash-Safe Recovery Guarantees

Every checkpoint loader (`LiveHostRunner`, `elder.ckpt`, and `terrarium_population.ckpt`) enforces bounded file reads, `CRC32C` (`RFC 3720`) verification, complete shadow-decoding into temporary memory before live swap, atomic `.tmp -> primary` + `.prev.tmp -> .prev` `fsync` rotation, automatic healing from `.prev` if `primary` is damaged, and fail-closed termination if all checkpoints are corrupted (never silently resetting a living organism).
