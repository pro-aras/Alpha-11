# Alpha 11 Observational Methodology & Control-Twin Audit Protocol

## 1. Scientific Mandate: Non-Invasive Observation Over Scripted Exams

Across earlier generations (`Alpha 1` through `Alpha 10`), whenever a general neural substrate hit a causal credit or scaling bottleneck, task-specific heuristics risked creeping toward the benchmark. In `Alpha 11`, the `a11-observe` crate ([`crates/a11-observe/src/lib.rs`](../crates/a11-observe/src/lib.rs)) operates strictly as an **external, read-only scientific microscope**:

- **Full Distribution Reporting + Holm-Bonferroni Family-Wise Error Control**: Every experiment reports full non-parametric empirical distributions (`min`, `p10`, `p25`, `p50`/median, `p75`, `p90`, `p95`, `p99`, `max`, `IQR`, `mean`, `std_dev`, `CV`) alongside Mann-Whitney $U$, rank-biserial effect size $r_{\text{rb}} \in [-1, +1]$, $1,000$-permutation two-sided $p$-values, and Holm-Bonferroni corrected paired $t$-tests across `>= 12` deterministic seeds.
- **Observer-Only Physical Band Labels**: `BandSpec` descriptors (`Photonic 2D`, `Wave-Echo Sonar`, `Noisy TV Static`) live exclusively inside `a11-observe` ([`twins.rs`](../crates/a11-observe/src/twins.rs)). The organism only ever receives an unlabelled `&[f32]` wave slice via [`FieldView`](../crates/a11-medium/src/lib.rs) and emits continuous forces via [`FieldPulse`](../crates/a11-medium/src/lib.rs).

---

## 2. How the Boundary & Forensic Microscope Eliminate Domain Shortcuts

`a11-medium` and `a11-observe` make hardcoded task shortcuts structurally impossible or immediately visible:

| # | Eliminated Shortcut | How `Alpha 11` Prevents It Structurally & Forensically |
| ---: | :--- | :--- |
| **1** | **Hardcoded Grid Quantizers** (`coords_to_cell` quantizing space into fixed tiles) | [`FieldView`](../crates/a11-medium/src/lib.rs) carries zero grid dimensions, tile indices, or room bounds. Spatial structure must be inferred from continuous cross-channel wave covariance and sensorimotor transformations. |
| **2** | **GPS Coordinate Cheats** (reading `(x, y)` directly from proprioception) | No GPS or coordinate tuple exists in `FieldView`. Channel order can be arbitrarily permuted in multi-medium tests (`Study 2`) without changing organism code. |
| **3** | **Sprite Pixel-Size Sieves** (hardcoded object bounding-box thresholds) | Zero segmented object blobs or bounding-box areas cross `a11-medium`; raw wave fluxes enter unlabelled pores directly. |
| **4** | **Hand-Coded Dialogue Scripts** | Acoustic/vocal muscle channels are ordinary continuous `actuator_out` floats with zero room-type conditionals. [`scan_crate_source`](../crates/a11-observe/src/forensics.rs) audits `a11-core` for `0` production strings and `0` forbidden identifiers (`words`, `meanings`, `concepts`, `thoughts`, `nursery_mode`). |
| **5** | **Pre-Split Feature-to-Syllable Slots** | Motor pores bind generically to unlabelled actuator indices (`motor_pore_channels`); no acoustic slot is pre-wired to any visual feature dimension. |
| **6** | **Fixed Rule Templates** | Temporal depth emerges from continuous cell time-constants (`mean_tau`, `max_tau`, `tau_octave_count`), polychronous `[1, 3, 6]` delay chains, vertical mitosis (`mitosis_births`), contradiction splits (`contradiction_splits`), and myelinated shortcuts (`myelinated_shortcuts`) in [`OrganismSnapshot`](../crates/a11-medium/src/lib.rs). |
| **7** | **Fixed Discrete Button Compasses** | `FieldView::num_actuators` and `FieldPulse::actuator_out` expose arbitrary-dimensional continuous `[-1.0, 1.0]` muscle channels across physical terraria and embodied organs alike. |

In addition, [`audit_plasticity_and_lesion`](../crates/a11-observe/src/forensics.rs) blocks boolean lesion fakes by verifying that [`OrganismEngine::zero_learned_conductances`](../crates/a11-medium/src/lib.rs) preserves `control_mode() == ControlMode::Natural` while physically resetting grown connectome/conductance state to newborn baseline, and verifies `0`-bit drift in `learned_parameter_hash()` across `500` active steps under `ControlMode::FrozenPlasticity`.

---

## 3. Causal Isolation Across Parallel Twin Controls

Every multi-seed study in [`run_twin_study`](../crates/a11-observe/src/twins.rs) (`N = 12` deterministic seeds default) and `observe-development` places identical initial organisms and worlds through parallel causal controls:

1. **`Natural` (`1..=T`)**: Open-ended metabolic homeostasis, pore budding/atrophy, `t - 1` / `t - \Delta` polychronous tube plasticity, vertical mitosis, contradiction splitting, Locus Coeruleus somatic-drop reset, and adenosine sleep consolidation.
2. **`FrozenPlasticity` (`1..=T`)**: Identical seed and world running `ControlMode::FrozenPlasticity`, locking every learned weight, conductance, cell birth, and structural change (`0`-bit parameter hash delta).
3. **`YokedMetabolism` (`1..=T`)**: Identical `Natural` plasticity rules on `(O(seed_i), W(seed_i))`, except its `FieldView::{metabolic_influx, tissue_stress}` is overwritten with the per-tick physiological stream of cohort-deranged donor `Natural[(i + 1) mod N]`, breaking `t - 1` action-outcome contingency while preserving reward distribution statistics.
4. **`ActionShuffled` (`1..=T`)**: Identical `Natural` organism whose emitted `FieldPulse` undergoes random actuator-channel permutation plus an 8-slot temporal reservoir swap before reaching `W`, preserving pulse energy while breaking sensorimotor timing.
5. **`NoisyTV` Distractor Band Analysis**: Measures the fraction of sensory pores bound to high-variance, zero-predictability static channels (`NoisyTV` + `OU(0.97)` Brownian distractors) relative to uniform random baseline.
6. **`FlaglessPostLifeLesion` vs `Intact`**: Clones the mature `Natural` organism into identical exposed foraging probes—one intact, one after flagless `zero_learned_conductances()`—measuring immediate survival necessity of the self-grown brain.
7. **Ecological & Acoustic Controls (`observe-development`)**: `BarrenTwin` (flat valley without Earth ecology scaffolding), `SoloOrphan` (no cohabiting Elder), and `VocalMuted` (vocal muscles active, airborne `16 kHz` voice muted across canyon ridges).

---

## 4. Transactional Fail-Closed Persistence & `O(1)` Latency Verification ([`forensics.rs`](../crates/a11-observe/src/forensics.rs))

Before trusting any long-horizon run, [`audit_persistence`](../crates/a11-observe/src/forensics.rs) and [`profile_latency`](../crates/a11-observe/src/forensics.rs) verify:
1. **Deterministic Continuation**: `save@100 -> restore into fresh O -> step 100` matches `uninterrupted 200` bit-for-bit (`f32::to_bits`) across `FieldPulse`, `OrganismSnapshot`, and `learned_parameter_hash()`.
2. **Corruption Rejection & Fail-Closed Poisoning**: Single-bit flips in payload, single-bit flips in magic/header/CRC, 5 distinct byte truncations, and oversized buffers all return `Err(...)`, preserve pre-corruption parameter hashes in RAM, set `is_poisoned() == true`, and block subsequent `step()` and `encode_checkpoint()` with `Err(PersistError::PoisonedAfterCorruption)`.
3. **Bounded Perceptual & Memory Scaling**: Reports `p50`, `p95`, `p99`, and `max` step latency (`us`) across a `32x` ambient channel expansion (`128` vs `4,096` channels), logical `used_state_bytes`, serialized `checkpoint_bytes`, and Linux `/proc/self/status` `VmRSS` and `VmHWM` separately.
