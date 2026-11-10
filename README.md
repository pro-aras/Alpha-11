# Alpha 11 (`v11.1.0`) — Self-Wiring Biophysical Organism & Embodied 16 kHz Earth Ecology

> **Repository:** `https://github.com/pro-aras/Alpha-11`  
> **License:** [MIT](./LICENSE)  
> **Toolchain & Dependencies:** Rust `>= 1.85` (Edition `2024`) · **`0` external crates (`100%` pure Rust `std`)** · `unsafe_code = "forbid"`  
> **Machine Summary:** [`llms.txt`](./llms.txt) · **Scientific Protocol:** [`observations/METHODOLOGY.md`](./observations/METHODOLOGY.md)

Alpha 11 is a self-contained, zero-dependency Rust artificial-life and biocomputation system in which continuously living cellular organisms (`Alpha11Organism`) develop **without pre-named sensory organs, string labels, pre-seeded phoneme tables, external LLM scaffolding, or task-specific reward bonuses**.

At birth, an `Alpha11Organism` sees **only an unlabeled raw `&[f32]` wave slice** plus two physiological scalars (`metabolic_influx` and `tissue_stress`) across the strict numeric boundary in [`crates/a11-medium`](./crates/a11-medium/src/lib.rs), and emits continuous `[-1.0, 1.0]` muscle forces. Every sense organ, motor synergy, temporal memory chain, sleep consolidation cycle, hand action, balance reflex, and `16 kHz` vocalization must self-wire and prove closed-loop causal necessity against frozen, action-shuffled, yoked, distractor ("Noisy TV"), barren-world, muted-voice, and flagless post-maturation brain-lesion controls.

---

## 1. Core Biophysical & Ecological Architecture

### A. Zero-String Self-Organizing Brain (`crates/a11-core`)
- **Mechanical Purity (`[Study 0]` Enforced)**: `crates/a11-core/src/` (`9` files) contains **`0` double-quote string literals**, **`0` forbidden anthropomorphic identifiers** (`words`, `meanings`, `concepts`, `thoughts`, `nursery_mode`), **`0` `unsafe` blocks**, and **`0` `.unwrap(` calls**. `a11-core` depends exclusively on `a11-medium`—never on any world or portal crate.
- **Bounded `O(1)` Self-Wired Sensory & Motor Pore Membrane (`pores.rs`)**: Rather than scanning unbounded `O(N)` world channels per tick, the membrane scouts unlabeled channels, cancels motor self-reafference, prunes uninformative Brownian/Noisy-TV static (`13.3% -> 3.9%`), and maintains a strictly bounded active frontier (`K <= 64` pores). Step latency stays flat across a `32x` channel expansion (`p50 = 74.8 us` on `128` channels vs `p50 = 66.6 us` on `4,096` channels).
- **Multi-Scale Cortical Cells & Phasic Norepinephrine Network Reset (`cells.rs`, `lib.rs`)**: Sparse `k-WTA` predictive cells span `7.7` octaves of intrinsic time-constants (`tau` from fast `1..4`-tick reflexes up to `240+`-tick multi-second contextual memory) via energy-gated mitosis (`mitosis_births = 0` when `ATP = 0`). Phasic somatic drops trigger a Locus Coeruleus norepinephrine-like working-memory reset (`ne_network_reset`), damping stale slow-`tau` attractors when environmental contingencies shift.
- **Polychronous Delay-Line Morphogenesis, Dual-Stream Vocal Contingency & `StriatalOfc` Context Consolidation (`tubes.rs`)**: Newly divided cells wire staggered `[1, 3, 6]`-tick (`20..120 ms`) convergent axonal delays, sprout sparse polychronous links (`lag = 6, 10, 14`), and learn multi-step `100..320 ms` predictive sequences (`lag = 5..=16`). Motor credit is applied strictly via `t-1` / `t-\Delta` efference-copy intervals after causal delays complete—never crediting a newly selected action `a_t` for simultaneous `r_t`. On continuous smooth spatial/wave fields, `StriatalOfc` stratifies centered exafferent pore signals into `4` monotonic inner-plexiform contrast sublaminae (`on_lo`, `on_hi`, `off_lo`, `off_hi`), cancels `4`-tap motor self-reafference, consolidates up to `4` latent orbitofrontal contexts across causal reversals (`A -> B -> A`), and arbitrates developmental prefrontal model-based Jacobian-valence drive against dorsolateral striatal habit norm. On embodied vocal effectors, even/odd `CellToMotor` tubes track trophic and conversational Rescorla contingency (`\Delta P`), pacing `~5 Hz` theta syllable calling vs. turn-taking listening.
- **Homeostatic `ATP` Allostasis, Marginal-Value Patch Leaving & Adenosine Sleep Consolidation (`body.rs`, `lib.rs`, `tubes.rs`)**: Basal rent, cell/tube/pore structure rent, motor effort, and tissue repair burn somatic `ATP`, while mature neurotrophic floors (`trophic_floor`) and Charnov marginal-value patch departure sustain lifelong foraging reserves (`ATP ~ 99.4` over `500,000` ticks, `0.0000` starvation). Accumulated metabolic adenosine triggers `20`-tick sleep bouts (`motor_clutch_threshold = 0.78`) during which labile synaptic traces consolidate into stiff long-term memory (`stiff_consolidated_tubes`)—waking immediately upon external tissue stress or somatic surprise.
- **Transactional Fail-Closed `CRC32C` Persistence (`persistence.rs`, `crc32c.rs`)**: Checkpoints (`FORMAT_VERSION = 5`, backward-compatible upgrade from `v3`/`v4`) serialize deterministic `used_state_bytes` with versioned headers and `RFC 3720` `CRC32C` checksums, enforce bounded reads and full shadow-decoding into temporary memory before swap, rotate `.prev` fallback generations atomically, and fail closed (`PoisonedAfterCorruption`) on any bit-flip or truncation without mutating live RAM.

### B. Unscripted `276`-Channel, `17`-Muscle Embodied Valley (`OpenSocialHabitat`)
In its primary `--habitat open` mode (`a11-live` & `alpha11-observe observe-development`), a newborn `Alpha11Organism` grows up alongside an autonomous, persisted Companion `Elder` `Alpha11Organism` inside a coupled physical body and valley (`zero` artificial organ or vocal rewards):
1. **Living Earth Ecology (`MultiGenTerrarium`, `crates/a11-worlds/src/terrarium.rs`)**:
   - `8x8` opponent photonic retina, `2..14`-tick active wave echoes, smooth day/night illumination cycles, thermal rock shelters, and somatic-warmth-gated `16x16` terrain trails.
   - `6` staggered seasonal watershed springs radiating continuous `16 kHz` stereo water murmurs with physical `1/d` attenuation, Woodworth interaural time difference (`ITD`), and azimuthal level difference (`ILD`).
   - **Opaque Basalt Canyon Ridges & Acoustic Diffraction**: Two basalt canyon walls (`(±11.6, -2.6)..(±11.6, 2.6)`) block `100%` (`0.0`) of direct visual rays across the canyon (`visual_occlusions_active`) while diffracting airborne `16 kHz` voice, water murmurs, and shell-taps around the rock at `0.68x` amplitude (`occluded_vocal_approaches`).
   - **Solo Pods, Obligate Two-Organism Heavy Grove Pods & Mineral Anvils**: `6` solo valley pods crack only through sustained downward hand pressure + lateral stroke work (scaled by upright postural stability), while `4` rich canyon-grove pods (`1.85x` nutrients) behind the basalt ridges have thick shells (`0.22` solo floor, radiating `540/1460 Hz` woody taps) that **cannot be cracked solo** (`0.00` cracked by `SoloOrphan`), requiring `>= 2` organisms within `2.8 m` working simultaneously (`coop_grove_pods_cracked`) or rolling a heavy mineral boulder anvil (`2.67x` cooperative mechanical advantage).
2. **Embodied Hands, Vestibular Balance & Ontogenetic `16 kHz` Voice (`crates/a11-portals`)**:
   - **`DrawingScratchpad` (`scratchpad.rs`)**: Tactile/visual hand manipulation, pod shell cracking, and shared stone-hearth tactile contact (`couple_shared_surface`) when cohabiting organisms stand within `<= 3.6 m`.
   - **`BalanceCatchToy` (`balance_toy.rs`)**: Continuous gravitational/terrain-sway vestibular limb whose upright stability directly scales manual tool/foraging leverage.
   - **`BinauralVocalOrgan` (`rhythm_pad.rs`)**: `5`-muscle physical `16 kHz` source-filter articulatory synthesizer (`PhysicalVocalTract`: lung pressure, laryngeal tension, jaw aperture, tongue front/back, lip rounding) whose vocal tract elongates monotonically with endogenous maturation from **`8.8 cm` (infant, `F0 ~ 353 Hz`) to `14.5 cm` (adult, `F0 ~ 250 Hz`)** (`FrozenTwin = 8.80 cm`), heard through `32`-band `ErbCochlea32` + `LSO` binaural ears alongside `40 ms` rocky-cave echoes, `YIN` fundamental pitch tracking, and intra-bout multi-syllable formant/pitch contour tracking (`syllable_transitions`).

---

## 2. Workspace Layout (`6` Crates, Zero External Dependencies)

```text
Alpha-11/
├── Cargo.toml                  # Workspace manifest (Edition 2024, rustc >= 1.85, unsafe = forbid)
├── Cargo.lock                  # Pure workspace lockfile (0 external crates.io dependencies)
├── LICENSE                     # MIT License
├── AGENTS.md                   # Biological purity, causal credit & privacy workspace laws
├── llms.txt                    # Concise machine-readable architecture & command summary
├── .github/workflows/ci.yml    # Automated build, 48 unit tests & Study 0..6 self-check CI
├── crates/
│   ├── a11-medium/             # Pure numeric FieldView / FieldPulse / OrganismEngine / WorldMedium skin
│   ├── a11-core/               # Zero-string biophysical organism core (9 source files)
│   │   └── src/
│   │       ├── body.rs         # Homeostatic ATP, hypothalamic temperature/vigor & adenosine sleep
│   │       ├── causal_core.rs  # Minimal 12/12 DEV-30 reference core (O(1) pore frontier + lOFC)
│   │       ├── cells.rs        # Multi-scale k-WTA predictive cells, mitosis & phasic NE reset
│   │       ├── crc32c.rs       # Pure safe Rust RFC 3720 CRC32C checksum
│   │       ├── lib.rs          # Unified Alpha11Organism::step(FieldView -> FieldPulse)
│   │       ├── persistence.rs  # Versioned shadow-decoded binary checkpoints & fail-closed poison
│   │       ├── pores.rs        # Self-wired sensory/motor pores, corollary discharge & cerebellar mastery
│   │       ├── rng.rs          # Deterministic SplitMix64/Xoshiro RNG & numerical sanitizers
│   │       └── tubes.rs        # Polychronous [1,3,6] delay tubes, sequence learning & spine splits
│   ├── a11-worlds/             # Continuous wave worlds & 16 kHz binaural MultiGenTerrarium Earth valley
│   ├── a11-portals/            # Embodied hands, vestibular limb, 16 kHz voice/cochlea & LiveHostRunner
│   ├── a11-observe/            # Read-only scientific microscope (alpha11-observe & alpha11-gates)
│   └── a11-live/               # Crash-safe continuous live daemon (a11-live)
├── live/
│   └── README.md               # Live daemon CLI modes, hot-swap protocol & state artifact reference
└── observations/
    ├── METHODOLOGY.md          # Non-parametric twin controls, causal isolation & persistence audits
    ├── 00..06_*.{md,json}      # Deterministically reproducible Studies 0..6 reports & datasets
    └── open_world_development.json # 12-seed 8,000-tick whole-body maturation & 10-gate Holm dataset
```

---

## 3. Requirements & Installation

- **Rust Toolchain**: Stable Rust **`>= 1.85.0`** (`rustup update stable`).
- **External Libraries**: **None.** Alpha 11 uses `100%` safe Rust `std` (`0` crates from `crates.io`, `0` C/C++ FFI libraries, `0` network calls, `0` GPU drivers required).
- **Supported Platforms**: Linux (`x86_64` / `aarch64`, with `/proc/self/status` `VmRSS` telemetry), macOS, and Windows.

```bash
git clone https://github.com/pro-aras/Alpha-11.git
cd Alpha-11

# Build all optimized release libraries and binaries
cargo build --release --workspace --bins

# Run all 48 release unit & invariant tests (hygiene, CRC32C, v4->v5 upgrade, FrozenPlasticity)
cargo test --release --workspace
```

---

## 4. Running Alpha 11

### A. Continuous Live Embodied Valley (`a11-live`)
Launch the focal `Alpha11Organism` and its cohabiting Companion `Elder` continuously at real-time `50 Hz` (`20 ms/tick`, `< 10 MB` OS RSS):

```bash
./target/release/a11-live \
  --state-dir live/state/open \
  --habitat open \
  --steps 0 \
  --pace-ms 20
```

- **Inspect live status (`every 5s`)**: Read `live/state/open/development_status.json` for live cell/tube/stiff-LTM counts, self-wired pore counts by organ (`retina`, `wave_echo`, `hands_canvas`, `vestibular`, `binaural_cochlea` vs `distractors`), neuromuscular mastery/jerk, `16 kHz` vocal tract length (`cm`), `F0`/`F1`/`F2` (`Hz`), `syllable_transitions`, `alternating_responses`, `visual_occlusions_active`, `occluded_vocal_approaches`, and `coop_grove_pods_cracked`.
- **Listen to unedited `16 kHz` audio**: Open `live/state/open/newborn_first_10s.wav` (saved once at `tick = 0`), `live/state/open/latest_10s_voice_mono.wav`, and `live/state/open/latest_10s_room_stereo.wav` (full stereo valley mix).
- **Graceful stop or zero-loss binary hot-swap**:
  ```bash
  touch live/state/open/STOP
  # Flushes transactional organism.ckpt + elder.ckpt within <= 320 ms and exits 0.
  # Remove STOP and re-run the same command to resume from the exact tick.
  ```
- Additional habitat modes (`--habitat composite` and `--habitat terrarium`) are documented in [`live/README.md`](./live/README.md).

### B. 12-Seed Whole-Body Developmental & 10-Gate Holm-Bonferroni Campaign
Run the full `8,000`-tick (`160 s` simulated life across `Newborn -> Infant -> Juvenile -> Mature`) 12-seed developmental study comparing `Natural` against `FrozenPlasticity`, `ActionShuffled`, `Intact vs Flagless Lesioned`, `BarrenTwin`, `SoloOrphan`, and `VocalMuted` (`~45 s` on modern multi-core CPUs):

```bash
./target/release/alpha11-observe observe-development --seeds 12 --ticks 8000
```

Writes the complete machine-readable report to `observations/open_world_development.json`.

### C. Forensic Hygiene, Persistence, Plasticity & Twin Studies `0..6` (`self-check`)
Regenerate all forensic and multi-world twin-control reports (`observations/00_forensics_and_latency.*` through `06_human_activity_portals.*`, `~27 s`):

```bash
./target/release/alpha11-observe self-check
```

### D. Locked `12`-Gate Causal Benchmark (`alpha11-gates`) & Lifelong / Vocal Judges (`alpha11-life`)
Run the `12`-gate causal benchmark (`G1..G9` + invariants `I1..I3`), `500,000`-tick lifelong energy homeostasis judge (`E1..E5`), and `40,000`-tick contingent vocal turn-taking judges (`teach` at `200 ms` delay and `teach2` at `40 ms` delay, `T1..T3`):

```bash
./target/release/alpha11-gates alpha11 dev
./target/release/alpha11-gates alpha11 dev30
./target/release/alpha11-life energy --seeds 10 --ticks 500000
./target/release/alpha11-life teach --seeds 10 --ticks 40000
./target/release/alpha11-life teach2 --seeds 10 --ticks 40000
```

---

## 5. Verified Empirical Results (Holm-Bonferroni Adjusted)

### A. Locked `12`-Gate Causal, Reversal, Recall & Torture Suite on `Alpha11Organism` — `12/12` PASS Across All Splits
| Gate / Invariant | `dev` (`n = 10`) | `dev30` (`n = 30`) | Single-Use `heldout` (`n = 12`, `seeds 200..211`) | Verdict |
| :--- | ---: | ---: | ---: | :---: |
| **`G1` Natural > Frozen** | `+99.83 vs -28.19` (`t=+12.37, 10/10`) | `+93.05 vs -22.12` (`t=+17.22, 30/30`) | `+78.62 vs -29.16` (`t=+14.79, p_holm=0.00000, 12/12`) | **PASS** |
| **`G2` Natural > Yoked** | `+99.83 vs -65.58` (`t=+9.49, 10/10`) | `+93.05 vs -66.38` (`t=+12.43, 30/30`) | `+78.62 vs -80.14` (`t=+12.02, p_holm=0.00000, 12/12`) | **PASS** |
| **`G3` Natural > ActionShuffled** | `+99.83 vs +13.28` (`t=+7.58, 10/10`) | `+93.05 vs +14.04` (`t=+11.56, 30/30`) | `+78.62 vs +14.57` (`t=+10.80, p_holm=0.00000, 12/12`) | **PASS** |
| **`G4` Transfer: Trained-Frozen > Newborn-Frozen** | `+44.47 vs -9.27` (`t=+10.40, 10/10`) | `+49.08 vs -7.35` (`t=+19.23, 30/30`) | `+41.06 vs -10.65` (`t=+14.28, p_holm=0.00000, 12/12`) | **PASS** |
| **`G5` Transfer: Trained-Frozen > BirthReset-Lesion** | `+44.47 vs -9.27` (`t=+10.40, 10/10`) | `+49.08 vs -7.35` (`t=+19.23, 30/30`) | `+41.06 vs -10.65` (`t=+14.28, p_holm=0.00000, 12/12`) | **PASS** |
| **`G6` Reversal (`A->B`): post Natural > Frozen Clone** | `-13.19 vs -23.58` (`t=+4.91, 9/10`) | `-12.23 vs -24.06` (`t=+8.41, 27/30`) | `-12.68 vs -23.61` (`t=+7.53, p_holm=0.00003, 12/12`) | **PASS** |
| **`G7` Reversal Speed (`A->B`): post gain > life gain** | `+4.08 vs +1.66` (`t=+3.44, p_holm=0.01486, 8/10`) | `+2.94 vs +1.24` (`t=+2.42, p_holm=0.02213, 22/30`) | `+4.51 vs +0.91` (`t=+4.36, p_holm=0.00228, 12/12`) | **PASS** |
| **`G8` Savings (`A->B->A`): return-A > Plastic Newborn** | `+2.63 vs -2.12` (`t=+3.09, p_holm=0.01486, 8/10`) | `+1.85 vs -1.48` (`t=+3.50, p_holm=0.00308, 22/30`) | `+2.07 vs -2.03` (`t=+2.88, p_holm=0.01500, 10/12`) | **PASS** |
| **`G9` Torture (`Perm+OU97+SelfCopy`): Natural > Frozen** | `+39.97 vs -26.40` (`t=+10.45, 10/10`) | `+54.53 vs -25.07` (`t=+16.47, 30/30`) | `+57.21 vs -24.64` (`t=+10.89, p_holm=0.00000, 12/12`) | **PASS** |
| **`I1` Counterfactual Agency Purity** | `nat=0.0027, p_shuf=0.00358, p_exo=0.00187` | `nat=0.0025, p_shuf=0.00000, p_exo=0.00000` | `nat=0.0024, p_shuf=0.00080, p_exo=0.00013` | **PASS** |
| **`I2` Bounded `O(1)` `4096`-Ch & Pore Sieve** | `p99=104.3 us, pores=64, dist=0.041` | `p99=107.8 us, pores=64, dist=0.041` | `p99=105.1 us, pores=64, dist=0.049 (<0.35)` | **PASS** |
| **`I3` Honest `ATP` Homeostasis & Zero Famine Mitosis** | `famine=0, ATP 59.71 vs 32.10 (p=0.00056)` | `famine=0, ATP 54.75 vs 35.09 (p=0.00003)` | `famine=0, ATP 43.32 vs 31.73 (p=0.00923)` | **PASS** |

### B. `500,000`-Tick Lifelong Energy Allostasis (`alpha11-life energy`, `n = 10` Seeds) — `5/5` PASS
- **`E1` Late `net/tick` (`Natural > Frozen`)**: `-0.00159 vs -0.01262` (`t = +15.00, p_holm = 0.00000, wins = 10/10` **PASS**)
- **`E2` Late `net/tick` (`Natural > ActionShuffled`)**: `-0.00159 vs -0.01240` (`t = +23.50, p_holm = 0.00000, wins = 10/10` **PASS**)
- **`E3` Late Mean `ATP` (`Natural > Frozen`)**: **`+99.47 vs +5.60`** (`t = +87.07, p_holm = 0.00000, wins = 10/10` **PASS**)
- **`E4` Foraging Income Retention (`late/early >= 0.80` in `>= 80%` seeds)**: **`8/10` seeds kept** (`mean = 0.9698` **PASS**)
- **`E5` Late Starvation Fraction (`<= 0.10`)**: **`0.0000` (`0.0%` starvation)** (`cells = 136.0, elder_reference_hash = 5684ab80dc45f29c` **PASS**)

### C. Contingent Vocal Turn-Taking Across Both Delay Horizons (`alpha11-life teach` & `teach2`, `40,000` Ticks, `n = 10` Seeds) — `3/3` PASS on Both
1. **Long-Delay (`200 ms`, `delay = 10`, `refract = 200` ticks = `4.0 s`) Conversational Turn-Taking (`alpha11-life teach`) — `3/3` PASS**:
   - **`T1` Contingent vs. Yoked Food Calling (`Nat FEED_C > Nat FEED_Y`)**: `+77.42 vs +47.34` onsets/k (`t = +6.09, p_holm = 0.00036, wins = 10/10` **PASS**)
   - **`T2` Contingent vs. Yoked Vocal Turn-Taking (`Nat VOICE_C > Nat VOICE_Y`)**: `+44.50 vs +32.39` onsets/k (`t = +2.57, p_holm = 0.03021, wins = 8/10` **PASS**)
   - **`T3` Plasticity Interaction (`(NatC - NatY) > (FrzC - FrzY)`)**: `+30.07 vs -5.45` (`t = +6.47, p_holm = 0.00035, wins = 10/10` **PASS**)
   - **Flagless Paired Ablations (`n = 10` seeds)**: `M29 vs M28_base` (`3/3 PASS`: `T1 t = +6.36, p_holm = 0.00039`; `T2 t = +3.05, p_holm = 0.02486`; `T3 t = +3.11, p_holm = 0.02486`) and `M29 vs M29_no_b23` (`3/3 PASS`: `T1 t = +5.56, p_holm = 0.00106`; `T2 t = +2.65, p_holm = 0.02666`; `T3 t = +3.45, p_holm = 0.01444`).
2. **Short-Delay (`40 ms`, `delay = 2`, `refract = 20` ticks = `0.4 s`) Vocal Turn-Taking (`alpha11-life teach2`) — `3/3` PASS**:
   - **`T1` Contingent vs. Yoked Food Calling (`Nat FEED_C > Nat FEED_Y`)**: `+84.42 vs +35.09` onsets/k (`t = +6.57, p_holm = 0.00031, wins = 10/10` **PASS**)
   - **`T2` Contingent vs. Yoked Vocal Turn-Taking (`Nat VOICE_C > Nat VOICE_Y`)**: `+44.49 vs +25.49` onsets/k (`t = +2.82, p_holm = 0.02005, wins = 7/10` **PASS**)
   - **`T3` Plasticity Interaction (`(NatC - NatY) > (FrzC - FrzY)`)**: `+49.33 vs -1.60` (`t = +6.11, p_holm = 0.00036, wins = 9/10` **PASS**)

### D. Pre-Registered 10-Gate Whole-Body Longitudinal Campaign (`observe-development`, `n = 12` Seeds) — `8/10` PASS
- Mature `Alpha11Organism` (`8,000` ticks): **`222.6` cells** (`>= 150`), **`12.87 cm` vocal tract** (`>= 12.0 cm`), passing `Survival` (`t = +7.03`), `Motor Control` (`t = +6.93`), `Foraging Necessity` (`Intact vs Flagless Lesion`, `t = +4.88, p_holm = 0.00196`), `Agency Purity` (`t = +3.05, p_holm = 0.03337`), `Mastery Growth` (`t = +57.95`), `Jerk Reduction` (`t = +6.55`), `Earth Ecology Vitality` (`t = +10.63`), and `Markov Error-Recovery` (`t = +9.15`); `Occluded Acoustic Coordination` (Gate 9) and `Role-Swap Separation` (Gate 10) remain open.

### E. Forensic Hygiene, Persistence & `O(1)` Latency (`Study 0`)
- **Source Hygiene (`a11-core`, `a11-medium`, `a11-worlds`)**: `0` string literals, `0` forbidden identifiers, `0` `unsafe` blocks, `0` `.unwrap(` calls.
- **Transactional Persistence (`FORMAT_VERSION = 5`)**: `200`-step save/restore continuation is `100%` bit-identical (`f32::to_bits`), seamless `v3/v4 -> v5` checkpoint upgrade is verified, and single-bit flips, truncations, and oversized buffers fail closed (`true/true/true`).
- **Frozen Plasticity Invariance**: `0` bits changed in `learned_parameter_hash()` across `500` active steps (both newborn and post-growth).
- **Perceptual Scaling (`128` vs `4,096` Ambient Channels)**: `128ch` `p50 = 97.96 us` (`p99 = 124.75 us`) vs `4096ch` `p50 = 91.64 us` (`p99 = 104.53 us`, `< 2,000 us`); live `2`-organism `276ch` valley daemon runs at **`~8.7–13.5 MiB` OS RSS**.

---

## 6. Scientific Scope & Honesty Statement

The verified multi-seed evidence (`12/12` across `dev`, `dev30`, and the single-use `heldout` vault; `5/5` `500,000`-tick `energy`; `3/3` `40,000`-tick `teach` at `200 ms` delay and `3/3` `teach2` at `40 ms` delay; `8/10` longitudinal development gates; `Study 0..6`) demonstrates **closed-loop self-wired sensory organogenesis, bounded `O(1)` perceptual scaling, counterfactual motor agency, lifelong metabolic allostasis (`ATP ~ 99.5`), post-training frozen skill transfer, faster adult reversal re-acquisition than initial acquisition, `A -> B -> A` memory recall savings, both short-delay (`40 ms`) and long-delay (`200 ms`) contingent vocal turn-taking, and crash-safe continual persistence**. It does **not** yet show occluded two-organism referential vocal coordination (Gate 9) or spontaneous role separation (Gate 10) — and nothing here is compositional human-level language, consciousness, or general intelligence. Falsified mechanisms are kept out of the code and recorded by the authors.

---

## License

Released under the [MIT License](./LICENSE).
