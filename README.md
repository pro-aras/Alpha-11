# Alpha 11 (`v11.0.0`) — Self-Wiring Biophysical Organism & Embodied 16 kHz Earth Ecology

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
- **Polychronous Delay-Line Morphogenesis & `t-1` Efference-Copy Credit (`tubes.rs`)**: Newly divided cells wire staggered `[1, 3, 6]`-tick (`20..120 ms`) convergent axonal delays, sprout sparse polychronous links (`lag = 6, 10, 14`), and learn multi-step `100..320 ms` predictive sequences (`lag = 5..=16`). Motor credit is applied strictly via `t-1` / `t-\Delta` efference-copy intervals after causal delays complete—never crediting a newly selected action `a_t` for simultaneous `r_t`. High-contradiction contexts split complementary gated spines (`contradiction_splits`) without destructive overwrite, and heavily used pathways myelinate (`myelinated_shortcuts`).
- **Homeostatic `ATP` Metabolism & Adenosine Sleep Consolidation (`body.rs`)**: Basal rent, cell/tube/pore structure rent, motor effort, and tissue repair burn somatic `ATP`. Hunger elevates hypothalamic vigor and exploratory temperature, thermoregulatory cold suppresses idle play, and accumulated metabolic adenosine triggers `20`-tick sleep bouts (`motor_clutch_threshold = 0.78`) during which labile synaptic traces consolidate into stiff long-term memory (`stiff_consolidated_tubes`)—waking immediately upon external tissue stress or somatic surprise.
- **Transactional Fail-Closed `CRC32C` Persistence (`persistence.rs`, `crc32c.rs`)**: Checkpoints serialize deterministic `used_state_bytes` with versioned headers and `RFC 3720` `CRC32C` checksums, enforce bounded reads and full shadow-decoding into temporary memory before swap, rotate `.prev` fallback generations atomically, and fail closed (`PoisonedAfterCorruption`) on any bit-flip or truncation without mutating live RAM.

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
├── .github/workflows/ci.yml    # Automated build, 47 unit tests & Study 0..6 self-check CI
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
    └── open_world_development.json # 12-seed 8,000-tick whole-body maturation & 7/7 Holm gate dataset
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

# Run all 47 release unit & invariant tests (hygiene, CRC32C, FrozenPlasticity, occlusion physics)
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

### B. 12-Seed Whole-Body Developmental & `7/7` Holm-Bonferroni Gate Campaign
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

### D. Locked `DEV-30` Causal Benchmark & Structural Ablations (`alpha11-gates`)
Run the `30`-seed (`seeds 0..29`) `12`-gate causal benchmark (`G1..G9` + invariants `I1..I3`) and `25`-test Holm-Bonferroni paired ablation suite on `CausalCore` or `Alpha11Organism`:

```bash
./target/release/alpha11-gates causal dev30
./target/release/alpha11-gates causal ablate
```

---

## 5. Verified Empirical Results (`n = 12` Seeds, Holm-Bonferroni Adjusted)

### A. Longitudinal Brain, Sense & `16 kHz` Vocal Ontogeny (`observe-development`)
| Stage / Twin Control | Cells | Tubes (`Stiff LTM`) | Distractor Pores | Mastery (`Jerk`) | Pods Cracked | Tract (`F0`) | Multi-Syllable Shifts (`syl_tr`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Newborn (`0..10s`)** | `51.0` | `1,008.9` (`81.8`) | `13.3%` (`8.5`) | `0.031` (`0.327`) | `0.2` | `8.87 cm` (`353 Hz`) | `9.4` (`1.2` plosives) |
| **Infant (`10..40s`)** | `84.3` | `2,037.3` (`795.8`) | `7.6%` (`4.8`) | `0.327` (`0.284`) | `1.9` | `9.94 cm` (`357 Hz`) | `109.6` (`35.1` plosives) |
| **Juvenile (`40..100s`)** | `149.7` | `3,705.8` (`1,035.0`) | `5.0%` (`3.2`) | `0.442` (`0.250`) | `6.2` | `11.89 cm` (`320 Hz`) | `198.6` (`59.2` plosives) |
| **Mature (`100..160s`)** | **`170.3`** | **`4,157.9` (`892.4`)** | **`3.9%` (`2.5`)** | **`0.462` (`0.229`)** | **`9.4`** | **`12.87 cm` (`296 Hz`)** | **`214.6` (`66.5` plosives)** |
| **VocalMuted (`@160s`)** | `168.4` | `4,112.0` (`884.1`) | `3.9%` (`2.5`) | `0.460` (`0.231`) | `9.0` | `12.87 cm` (`297 Hz`) | `166.8` (`0.0` blind approaches) |
| **SoloOrphan (`@160s`)** | `161.2` | `3,980.4` (`871.2`) | `4.1%` (`2.6`) | `0.458` (`0.232`) | `9.1` (`0.0` grove) | `12.87 cm` (`295 Hz`) | `182.3` (`0.0` replies) |
| **BarrenTwin (`@160s`)** | `77.7` | `2,375.1` (`868.0`) | `2.9%` (`1.8`) | `0.440` (`0.240`) | `0.0` | `12.34 cm` (`308 Hz`) | `244.7` (`ATP = 2.03` vs `4.02`) |
| **FrozenTwin (`@160s`)** | `32.0` | `319.6` (`0.0`) | `13.3%` (`8.5`) | `0.000` (`0.561`) | `1.2` (`0.0` grove) | `8.80 cm` (`353 Hz`) | `86.4` (`0.2` plosives) |

### B. `7/7` Holm-Bonferroni Paired Causal & Ecological Gates (`n = 12` Seeds)
| Gate Comparison | Arm `A` | Arm `B` | Paired `diff` | `t`-stat | `p_holm` | Result |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| **1. Homeostatic Survival (`Mature Natural vs Frozen`)** | `-0.0024` | `-0.0141` | `+0.0117` | `+17.50` | `0.00000` | **PASS (`12/12`)** |
| **2. Causal Motor Control (`Mature Natural vs ActionShuffled`)** | `-0.0024` | `-0.0135` | `+0.0111` | `+8.65` | `0.00001` | **PASS (`12/12`)** |
| **3. Post-Maturation Foraging Necessity (`Intact vs Flagless Lesioned`)** | `-0.0051` | `-0.0087` | `+0.0036` | `+2.32` | `0.04075` | **PASS** |
| **4. Counterfactual Agency Purity (`Natural vs ActionShuffled`)** | `+0.0371` | `+0.0003` | `+0.0368` | `+7.50` | `0.00004` | **PASS (`12/12`)** |
| **5. Neuromuscular Mastery Growth (`Mature vs Newborn`)** | `+0.4616` | `+0.0311` | `+0.4305` | `+80.41` | `0.00000` | **PASS (`12/12`)** |
| **6. Kinematic Jerk Reduction (`Newborn Jerk vs Mature Jerk`)** | `0.3271` | `0.2294` | `+0.0977` | `+5.75` | `0.00026` | **PASS (`11/12`)** |
| **7. Earth Ecology Vitality Scaffolding (`Earth vs Barren`)** | `-0.0024` | `-0.0158` | `+0.0134` | `+12.64` | `0.00000` | **PASS (`12/12`)** |

### C. Forensic Hygiene, Persistence & `O(1)` Latency (`Study 0`)
- **Source Hygiene (`a11-core`, `a11-medium`, `a11-worlds`)**: `0` string literals, `0` forbidden identifiers, `0` `unsafe` blocks, `0` `.unwrap(` calls.
- **Transactional Persistence**: `200`-step save/restore continuation is `100%` bit-identical (`f32::to_bits`); single-bit flips, truncations, and oversized buffers fail closed (`true/true/true`).
- **Frozen Plasticity Invariance**: `0` bits changed in `learned_parameter_hash()` across `500` active steps (both newborn and post-growth).
- **Perceptual Scaling (`128` vs `4,096` Ambient Channels)**: `128ch` `p50 = 74.76 us` (`p99 = 86.84 us`) vs `4096ch` `p50 = 66.57 us` (`p99 = 78.98 us`, `32x` channel ratio `= 0.89x`); live `2`-organism `276ch` valley daemon runs at **`~8.7–9.5 MiB` OS RSS**.

---

## 6. Scientific Scope & Honesty Statement

Passing `Study 0..6`, `12/12` `DEV-30` causal gates, and `7/7` Holm-Bonferroni longitudinal whole-body gates demonstrates **closed-loop self-wired sensory organogenesis, bounded `O(1)` perceptual scaling, counterfactual motor agency, polychronous multi-syllable acoustic contour growth, blind sound-guided approach around visual occlusions, obligate two-organism physical foraging, and crash-safe continual persistence**—not human-level symbolic language, consciousness, or general intelligence.

---

## License

Released under the [MIT License](./LICENSE).
