# Study 2: Coordinate-Free Bijective Wire Permutation Invariance (`PermutedMedium`)

All `128` ambient sensory channels and all `6` motor actuator channels were bijectively scrambled via `PermutedMedium::new(world, seed ^ 0x5C8A_4B1E_9D2F_7103)` before entering `Alpha11Organism`. Channel identities were inverted **exclusively inside `a11-observe`** (`invert_sensor_channel` / `invert_motor_channel`) after each snapshot.

## Invariance Summary (`Permuted` vs `Unpermuted` `Natural` Cohort, `N = 12` Seeds)

| Habitat | Observable | Permuted `p50` | Unpermuted `p50` | `median(Perm) - median(Unperm)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `BrightRadiantWorld` | `achieved_influx_per_tick` | `0.0034` | `0.0037` | `-0.00031` | `-0.1250` | `0.6274` |
| `BrightRadiantWorld` | `decoded_photonic_conductance_share` | `0.9735` | `0.9803` | `-0.00675` | `-0.0139` | `0.9720` |
| `BrightRadiantWorld` | `decoded_sonar_conductance_share` | `0.0242` | `0.0197` | `+0.00447` | `-0.0208` | `0.9481` |
| `BrightRadiantWorld` | `decoded_noisy_tv_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `-0.0694` | `1.0000` |
| `BrightRadiantWorld` | `active_pings_emitted` | `1030.5000` | `1111.0000` | `-80.50000` | `-0.3750` | `0.1578` |
| `BrightRadiantWorld` | `decoded_emit_pulse_motor_conductance` | `0.9498` | `0.9498` | `-0.00004` | `+0.0486` | `0.8521` |
| `PitchBlackEchoCave` | `achieved_influx_per_tick` | `0.0017` | `0.0018` | `-0.00006` | `-0.2083` | `0.4236` |
| `PitchBlackEchoCave` | `decoded_photonic_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `+0.0000` | `1.0000` |
| `PitchBlackEchoCave` | `decoded_sonar_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `-0.0347` | `1.0000` |
| `PitchBlackEchoCave` | `decoded_noisy_tv_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `-0.0972` | `0.4935` |
| `PitchBlackEchoCave` | `active_pings_emitted` | `1354.0000` | `1366.0000` | `-12.00000` | `-0.1597` | `0.5205` |
| `PitchBlackEchoCave` | `decoded_emit_pulse_motor_conductance` | `0.9500` | `0.9500` | `-0.00004` | `-0.0625` | `0.7842` |

---

# Observational Twin Study: 02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0034 [0.0024]` (`0.0007..0.0068`) | `0.0001 [0.0004]` | `0.0033 [0.0020]` | `0.0045 -> 0.0000` | `0.0000 [0.0025]` | `0.0409 [0.0280]` | `0.226` |
| `FrozenPlasticity` | `2400` | `0.0006 [0.0004]` (`0.0003..0.0015`) | `0.0005 [0.0019]` | `0.0000 [0.0021]` | `0.0010 -> 0.0001` | `0.0000 [0.0000]` | `0.0345 [0.0070]` | `9.768` |
| `YokedMetabolism` | `2400` | `0.0012 [0.0013]` (`0.0002..0.0062`) | `0.0012 [0.0016]` | `0.0001 [0.0017]` | `0.0024 -> 0.0000` | `0.0000 [0.0000]` | `0.0435 [0.0514]` | `0.007` |
| `ActionShuffled` | `2400` | `0.0007 [0.0007]` (`0.0002..0.0026`) | `0.0005 [0.0010]` | `0.0003 [0.0004]` | `0.0012 -> 0.0000` | `0.0000 [0.0000]` | `0.0368 [0.0229]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0039 [0.0074]` (`0.0001..0.0155`) | `0.0012 [0.0037]` | `0.0031 [0.0079]` | `0.0062 -> 0.0000` | `0.0000 [0.0011]` | `0.0365 [0.0107]` | `1.285` |
| `FlaglessPostLifeLesion` | `600` | `0.0023 [0.0020]` (`0.0006..0.0115`) | `0.0013 [0.0022]` | `0.0016 [0.0022]` | `0.0024 -> 0.0005` | `0.0000 [0.0000]` | `0.0114 [0.0007]` | `1.032` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `69.5 [8.0]` (`p95=80.8`) | `763.5 [96.0]` | `166.0` | `64.5` | `39.0` | `5.21 / 15.32 / 4.5` | `37.5 / 0.0 / 2.5 / 157.5` | `2056.5 / 372.0 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2270.5 / 129.5 / 105.0` |
| `YokedMetabolism` | `70.0 [8.0]` (`p95=80.3`) | `766.5 [95.0]` | `150.0` | `41.0` | `39.0` | `5.23 / 15.71 / 4.5` | `38.0 / 0.0 / 2.5 / 178.0` | `2087.5 / 348.5 / 105.0` |
| `ActionShuffled` | `55.0 [2.2]` (`p95=63.5`) | `589.0 [26.5]` | `109.0` | `34.5` | `24.5` | `4.78 / 13.04 / 4.0` | `23.0 / 0.0 / 1.0 / 145.0` | `2177.5 / 304.0 / 105.0` |
| `UnlesionedPostLife` | `69.5 [8.0]` (`p95=80.8`) | `763.5 [96.0]` | `166.0` | `64.5` | `39.0` | `5.21 / 15.32 / 4.5` | `37.5 / 0.0 / 2.5 / 157.5` | `2577.5 / 478.0 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00287` | `6.0` (`138.0`) | `+0.9167` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00329` | `4.0` (`140.0`) | `+0.9444` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00008` | `39.5` (`39.5`) | `-0.4514` | `0.0669` | `0.0552` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `42.0` (`102.0`) | `+0.4167` | `0.0509` | `0.0165` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `-0.25000` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00223` | `31.0` (`113.0`) | `+0.5694` | `0.0240` | `0.0194` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00321` | `15.0` (`129.0`) | `+0.7917` | `0.0020` | `0.0011` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00000` | `64.5` (`64.5`) | `-0.1042` | `0.6703` | `0.6301` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `52.0` (`92.0`) | `+0.2778` | `0.1728` | `0.1610` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `0.7473` | `0.5807` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00276` | `14.0` (`130.0`) | `+0.8056` | `0.0010` | `0.0009` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00298` | `2.0` (`142.0`) | `+0.9722` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `69.5` (`74.5`) | `+0.0347` | `0.8072` | `0.8794` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `45.5` (`98.5`) | `+0.3681` | `0.0509` | `0.0485` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `60.5` (`60.5`) | `-0.1597` | `0.4905` | `0.3281` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00163` | `60.0` (`84.0`) | `+0.1667` | `0.5145` | `0.5067` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00150` | `61.0` (`83.0`) | `+0.1528` | `0.5534` | `0.5444` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00174` | `14.0` (`130.0`) | `+0.8056` | `0.0010` | `0.0009` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `97.4% [3.4%]` (`p95=100.0%`) | `56.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `2.4% [3.8%]` (`p95=5.6%`) | `3.5` | `1.00 [0.25]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.7%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `45.0% [21.3%]` (`p95=62.2%`) | `9.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `22.5% [15.0%]` (`p95=39.5%`) | `4.5` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `25.0% [11.2%]` (`p95=46.2%`) | `5.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `96.2% [2.5%]` (`p95=99.2%`) | `54.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `3.8% [2.0%]` (`p95=9.0%`) | `5.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=1.5%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `99.9% [1.2%]` (`p95=100.0%`) | `53.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.7%]` (`p95=1.2%`) | `0.0` | `0.00 [1.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=1.5%`) | `0.0` | `0.00 [0.25]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `97.4% [3.4%]` (`p95=100.0%`) | `56.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `2.4% [3.8%]` (`p95=5.6%`) | `3.5` | `1.00 [0.25]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.7%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `9.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `4.5` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `5.0` | `1.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `30.0%` | `45.0%` / `1.00` | `30.0%` / `1.00` | `30.0%` / `1.00` |
| `240` | `10%` | `0.0169` | `12.458` | `40.0` | `416.5` | `4.31` | `4.0` | `0.0000` | `0.3%` | `97.1%` / `1.00` | `2.9%` / `1.00` | `0.3%` / `1.00` |
| `600` | `25%` | `0.0002` | `6.507` | `55.0` | `596.5` | `4.88` | `4.5` | `0.0000` | `0.0%` | `96.6%` / `1.00` | `2.9%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `2.306` | `64.0` | `703.0` | `5.11` | `4.5` | `0.0000` | `0.0%` | `98.4%` / `1.00` | `1.6%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `1.482` | `64.0` | `700.0` | `5.11` | `4.5` | `0.0000` | `0.0%` | `99.1%` / `1.00` | `0.9%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `0.226` | `69.5` | `763.5` | `5.21` | `4.5` | `0.0000` | `0.0%` | `97.4%` / `1.00` | `2.4%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.5260` | `0.2559 / 0.4518` | `0.0451 / 0.1214` | `3.6243 / 0.9794` | `0.0274` | `0.2848` |
| `FrozenPlasticity` | `12` | `0.5722` | `0.8470 / 1.0000` | `0.0002 / 0.0007` | `3.6955 / 0.9987` | `0.0005` | `0.4276` |
| `YokedMetabolism` | `12` | `0.9904` | `0.2794 / 0.6136` | `0.0706 / 0.1772` | `3.5942 / 0.9713` | `0.0534` | `0.3352` |
| `ActionShuffled` | `12` | `0.7751` | `0.2502 / 0.4396` | `0.0569 / 0.1266` | `3.6353 / 0.9824` | `0.0524` | `0.3106` |
| `UnlesionedPostLife` | `12` | `0.9079` | `0.2559 / 0.4518` | `0.0908 / 0.3115` | `3.3568 / 0.9071` | `0.2665` | `0.3283` |
| `FlaglessPostLifeLesion` | `12` | `0.9811` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6729 / 0.9926` | `0.0027` | `0.2480` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1030.5 [146.5]` | `336.675 [399.859]` | `0.950 [0.004]` | `2.5 [1.2]` | `3.0` | `231.15` | `0.0` |
| `FrozenPlasticity` | `1422.0 [33.2]` | `510.949 [147.993]` | `0.450 [0.000]` | `2.0 [0.5]` | `6.5` | `180.12` | `0.5` |
| `YokedMetabolism` | `960.0 [131.8]` | `269.674 [91.365]` | `0.950 [0.001]` | `2.0 [1.0]` | `15.0` | `212.13` | `0.5` |
| `ActionShuffled` | `1067.0 [86.2]` | `547.680 [480.690]` | `0.949 [0.004]` | `1.5 [1.0]` | `5.5` | `174.82` | `1.0` |
| `UnlesionedPostLife` | `272.0 [100.5]` | `76.724 [45.626]` | `0.950 [0.004]` | `1.0 [1.0]` | `0.5` | `51.54` | `0.0` |
| `FlaglessPostLifeLesion` | `419.0 [23.5]` | `241.634 [56.752]` | `0.000 [0.000]` | `1.0 [0.0]` | `1.0` | `62.12` | `0.0` |

### Cross-Habitat Modality Comparison (`02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` vs `02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `-323.50000` | `-1.0000` | `0.0010` |
| `echo_energy_received` | `-21.28281` | `+0.0139` | `0.9810` |
| `photonic_band_conductance_share` | `+0.97353` | `+1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `+0.02418` | `+0.0208` | `0.9291` |

---

# Observational Twin Study: 02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0017 [0.0010]` (`0.0006..0.0035`) | `0.0009 [0.0011]` | `0.0008 [0.0011]` | `0.0034 -> 0.0000` | `0.0000 [0.0000]` | `0.0122 [0.0011]` | `8.128` |
| `FrozenPlasticity` | `2400` | `0.0005 [0.0006]` (`0.0002..0.0024`) | `0.0005 [0.0012]` | `-0.0002 [0.0017]` | `0.0010 -> 0.0001` | `0.0000 [0.0000]` | `0.0294 [0.0094]` | `9.743` |
| `YokedMetabolism` | `2400` | `0.0005 [0.0003]` (`0.0002..0.0025`) | `0.0012 [0.0029]` | `-0.0006 [0.0028]` | `0.0007 -> 0.0000` | `0.0000 [0.0000]` | `0.0126 [0.0016]` | `8.575` |
| `ActionShuffled` | `2400` | `0.0005 [0.0005]` (`0.0002..0.0014`) | `0.0006 [0.0014]` | `0.0001 [0.0010]` | `0.0009 -> 0.0000` | `0.0000 [0.0000]` | `0.0125 [0.0014]` | `6.390` |
| `UnlesionedPostLife` | `600` | `0.0031 [0.0025]` (`0.0006..0.0091`) | `0.0007 [0.0009]` | `0.0021 [0.0036]` | `0.0043 -> 0.0000` | `0.0000 [0.0000]` | `0.0116 [0.0009]` | `10.313` |
| `FlaglessPostLifeLesion` | `600` | `0.0014 [0.0010]` (`0.0006..0.0059`) | `0.0000 [0.0002]` | `0.0013 [0.0010]` | `0.0020 -> 0.0000` | `0.0000 [0.0000]` | `0.0114 [0.0007]` | `8.474` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `42.0 [2.2]` (`p95=43.9`) | `390.5 [41.2]` | `40.0` | `10.0` | `10.0` | `4.47 / 9.99 / 4.0` | `10.0 / 0.0 / 1.0 / 132.5` | `2238.5 / 163.5 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2270.0 / 130.0 / 105.0` |
| `YokedMetabolism` | `41.5 [1.5]` (`p95=44.0`) | `388.0 [40.2]` | `38.0` | `10.0` | `9.5` | `4.36 / 8.87 / 4.0` | `9.5 / 0.0 / 1.0 / 130.5` | `2241.5 / 161.0 / 105.0` |
| `ActionShuffled` | `40.5 [1.0]` (`p95=41.5`) | `392.0 [19.2]` | `34.0` | `8.5` | `8.5` | `4.37 / 10.06 / 4.0` | `8.5 / 0.0 / 0.0 / 100.5` | `2235.5 / 164.5 / 105.0` |
| `UnlesionedPostLife` | `42.0 [2.2]` (`p95=43.9`) | `390.5 [41.2]` | `40.0` | `10.0` | `10.0` | `4.47 / 9.99 / 4.0` | `10.0 / 0.0 / 1.0 / 132.5` | `2786.5 / 215.5 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `574.5 / 25.5 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00116` | `23.0` (`121.0`) | `+0.6806` | `0.0050` | `0.0051` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00100` | `45.0` (`99.0`) | `+0.3750` | `0.1578` | `0.1260` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00008` | `41.0` (`41.0`) | `-0.4306` | `0.0699` | `0.0599` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `-0.25000` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00121` | `22.0` (`122.0`) | `+0.6944` | `0.0050` | `0.0043` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00142` | `25.0` (`119.0`) | `+0.6528` | `0.0060` | `0.0073` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00002` | `52.0` (`52.0`) | `-0.2778` | `0.2268` | `0.2087` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `59.5` (`59.5`) | `-0.1736` | `0.3526` | `0.2860` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00124` | `13.0` (`131.0`) | `+0.8194` | `0.0010` | `0.0007` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00064` | `34.0` (`110.0`) | `+0.5278` | `0.0360` | `0.0304` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `65.0` (`65.0`) | `-0.0972` | `0.6853` | `0.6661` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `71.5` (`71.5`) | `-0.0069` | `1.0000` | `1.0000` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00169` | `37.0` (`107.0`) | `+0.4861` | `0.0500` | `0.0464` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00075` | `50.0` (`94.0`) | `+0.3056` | `0.2438` | `0.2145` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00083` | `29.0` (`115.0`) | `+0.5972` | `0.0180` | `0.0141` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=3.3%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `45.0% [21.3%]` (`p95=62.2%`) | `9.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `22.5% [15.0%]` (`p95=39.5%`) | `4.5` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `25.0% [11.2%]` (`p95=46.2%`) | `5.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `50.0% [100.0%]` (`p95=100.0%`) | `1.0` | `0.50 [1.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=45.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=3.3%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `9.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `4.5` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `5.0` | `1.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `30.0%` | `45.0%` / `1.00` | `30.0%` / `1.00` | `30.0%` / `1.00` |
| `240` | `10%` | `0.0120` | `11.604` | `39.5` | `410.0` | `4.25` | `4.0` | `0.0000` | `19.3%` | `0.0%` / `0.00` | `74.2%` / `1.00` | `19.3%` / `1.00` |
| `600` | `25%` | `0.0000` | `10.796` | `41.5` | `418.5` | `4.44` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `9.935` | `42.0` | `404.5` | `4.47` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `0.0%` / `0.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `9.034` | `42.0` | `397.0` | `4.47` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `8.128` | `42.0` | `390.5` | `4.47` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `0.0%` / `0.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.4924` | `0.6808 / 1.0000` | `0.0794 / 0.2178` | `3.6990 / 0.9996` | `0.0012` | `0.3975` |
| `FrozenPlasticity` | `12` | `0.7868` | `0.8470 / 1.0000` | `0.0002 / 0.0007` | `3.6954 / 0.9986` | `0.0005` | `0.4467` |
| `YokedMetabolism` | `12` | `0.9823` | `0.7468 / 1.0000` | `0.0343 / 0.1742` | `3.6994 / 0.9997` | `0.0008` | `0.4442` |
| `ActionShuffled` | `12` | `0.6021` | `0.7509 / 1.0000` | `0.0296 / 0.2016` | `3.6992 / 0.9997` | `0.0009` | `0.4141` |
| `UnlesionedPostLife` | `12` | `0.7762` | `0.6808 / 1.0000` | `0.1064 / 0.2772` | `3.6944 / 0.9984` | `0.0045` | `0.4310` |
| `FlaglessPostLifeLesion` | `12` | `0.8437` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6848 / 0.9958` | `0.0023` | `0.2390` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1354.0 [66.0]` | `357.958 [103.842]` | `0.950 [0.000]` | `2.0 [0.5]` | `19.5` | `178.65` | `0.0` |
| `FrozenPlasticity` | `1428.5 [23.5]` | `466.267 [96.941]` | `0.450 [0.000]` | `2.0 [0.0]` | `5.0` | `179.73` | `1.0` |
| `YokedMetabolism` | `1355.5 [46.0]` | `439.093 [142.534]` | `0.950 [0.000]` | `2.0 [1.0]` | `8.0` | `178.47` | `1.0` |
| `ActionShuffled` | `1409.0 [31.5]` | `508.502 [149.635]` | `0.950 [0.003]` | `2.0 [1.5]` | `13.5` | `168.57` | `1.0` |
| `UnlesionedPostLife` | `350.5 [9.5]` | `129.769 [15.482]` | `0.950 [0.000]` | `1.0 [0.0]` | `1.0` | `44.63` | `0.0` |
| `FlaglessPostLifeLesion` | `365.5 [16.5]` | `143.437 [14.118]` | `0.000 [0.000]` | `1.0 [0.2]` | `0.0` | `46.54` | `0.0` |

### Cross-Habitat Modality Comparison (`02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` vs `02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `+323.50000` | `+1.0000` | `0.0010` |
| `echo_energy_received` | `+21.28281` | `-0.0139` | `0.9810` |
| `photonic_band_conductance_share` | `-0.97353` | `-1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `-0.02418` | `-0.0208` | `0.9291` |
