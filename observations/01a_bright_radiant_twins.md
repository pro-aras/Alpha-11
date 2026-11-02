# Observational Twin Study: 01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0037 [0.0017]` (`0.0013..0.0103`) | `0.0002 [0.0005]` | `0.0037 [0.0019]` | `0.0068 -> 0.0000` | `0.0000 [0.0042]` | `0.0304 [0.0207]` | `0.256` |
| `FrozenPlasticity` | `2400` | `0.0008 [0.0004]` (`0.0003..0.0015`) | `0.0005 [0.0008]` | `0.0002 [0.0011]` | `0.0017 -> 0.0000` | `0.0000 [0.0000]` | `0.0216 [0.0095]` | `10.482` |
| `YokedMetabolism` | `2400` | `0.0021 [0.0022]` (`0.0002..0.0142`) | `0.0016 [0.0034]` | `0.0008 [0.0035]` | `0.0033 -> 0.0000` | `0.0004 [0.0020]` | `0.0469 [0.0338]` | `0.094` |
| `ActionShuffled` | `2400` | `0.0005 [0.0007]` (`0.0002..0.0020`) | `0.0004 [0.0005]` | `0.0004 [0.0006]` | `0.0008 -> 0.0000` | `0.0000 [0.0003]` | `0.0520 [0.0333]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0047 [0.0050]` (`0.0016..0.0197`) | `0.0019 [0.0056]` | `0.0021 [0.0052]` | `0.0094 -> 0.0000` | `0.0000 [0.0000]` | `0.0344 [0.0134]` | `1.615` |
| `FlaglessPostLifeLesion` | `600` | `0.0027 [0.0028]` (`0.0010..0.0080`) | `0.0011 [0.0027]` | `0.0016 [0.0047]` | `0.0049 -> 0.0003` | `0.0000 [0.0000]` | `0.0114 [0.0007]` | `1.644` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `70.5 [9.5]` (`p95=92.2`) | `771.5 [110.0]` | `166.0` | `52.0` | `39.5` | `5.30 / 16.79 / 5.0` | `38.5 / 0.0 / 2.5 / 132.0` | `2087.0 / 366.5 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2271.0 / 129.0 / 105.0` |
| `YokedMetabolism` | `70.5 [10.2]` (`p95=93.6`) | `774.5 [114.5]` | `162.0` | `43.0` | `38.5` | `5.17 / 14.55 / 4.0` | `38.5 / 0.0 / 2.5 / 113.0` | `2070.5 / 393.0 / 105.0` |
| `ActionShuffled` | `55.0 [2.0]` (`p95=61.4`) | `580.5 [41.0]` | `94.5` | `25.5` | `24.0` | `4.74 / 10.04 / 4.0` | `23.0 / 0.0 / 0.0 / 99.5` | `2117.5 / 318.5 / 105.0` |
| `UnlesionedPostLife` | `70.5 [9.5]` (`p95=92.2`) | `771.5 [110.0]` | `166.0` | `52.0` | `39.5` | `5.30 / 16.79 / 5.0` | `38.5 / 0.0 / 2.5 / 132.0` | `2609.0 / 453.5 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00292` | `2.0` (`142.0`) | `+0.9722` | `0.0010` | `0.0001` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00346` | `3.0` (`141.0`) | `+0.9583` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `0.7133` | `0.7052` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `42.0` (`102.0`) | `+0.4167` | `0.0509` | `0.0165` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1662` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00162` | `44.0` (`100.0`) | `+0.3889` | `0.1449` | `0.1124` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00293` | `27.0` (`117.0`) | `+0.6250` | `0.0110` | `0.0102` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `0.7453` | `0.7150` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `-0.00037` | `67.0` (`77.0`) | `+0.0694` | `0.7752` | `0.7770` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `67.0` (`77.0`) | `+0.0694` | `1.0000` | `0.6513` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00327` | `6.0` (`138.0`) | `+0.9167` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00329` | `3.0` (`141.0`) | `+0.9583` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `67.0` (`77.0`) | `+0.0694` | `0.7383` | `0.7569` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `65.0` (`79.0`) | `+0.0972` | `0.6853` | `0.6661` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `63.0` (`63.0`) | `-0.1250` | `0.4006` | `0.4896` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00200` | `47.0` (`97.0`) | `+0.3472` | `0.1808` | `0.1572` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00051` | `65.0` (`79.0`) | `+0.0972` | `0.7113` | `0.7075` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00186` | `4.0` (`140.0`) | `+0.9444` | `0.0010` | `0.0001` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `98.0% [4.1%]` (`p95=100.0%`) | `53.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `2.0% [4.1%]` (`p95=18.0%`) | `2.5` | `1.00 [0.25]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.1%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `97.8% [3.5%]` (`p95=100.0%`) | `55.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `1.7% [3.5%]` (`p95=8.0%`) | `3.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.4%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `99.3% [1.0%]` (`p95=100.0%`) | `55.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.3% [1.0%]` (`p95=2.1%`) | `0.5` | `0.50 [1.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.7%`) | `0.0` | `0.00 [0.25]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `98.0% [4.1%]` (`p95=100.0%`) | `53.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `2.0% [4.1%]` (`p95=18.0%`) | `2.5` | `1.00 [0.25]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.1%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `240` | `10%` | `0.0217` | `13.615` | `40.0` | `417.0` | `4.24` | `4.0` | `0.0000` | `0.7%` | `98.4%` / `1.00` | `0.5%` / `1.00` | `0.7%` / `1.00` |
| `600` | `25%` | `0.0000` | `7.101` | `55.0` | `595.5` | `5.04` | `4.0` | `0.0000` | `0.0%` | `99.2%` / `1.00` | `0.8%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0003` | `2.536` | `70.0` | `774.0` | `5.26` | `4.5` | `0.0000` | `0.0%` | `99.1%` / `1.00` | `0.9%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `1.383` | `70.0` | `764.5` | `5.29` | `4.5` | `0.0000` | `0.0%` | `99.7%` / `1.00` | `0.3%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `0.256` | `70.5` | `771.5` | `5.30` | `5.0` | `0.0000` | `0.0%` | `98.0%` / `1.00` | `2.0%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.6139` | `0.2781 / 0.5315` | `0.0345 / 0.0961` | `3.6436 / 0.9846` | `0.0252` | `0.2972` |
| `FrozenPlasticity` | `12` | `0.4017` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6953 / 0.9986` | `0.0005` | `0.2905` |
| `YokedMetabolism` | `12` | `1.1834` | `0.2583 / 0.4930` | `0.0198 / 0.0798` | `3.6160 / 0.9772` | `0.0288` | `0.3282` |
| `ActionShuffled` | `12` | `0.7985` | `0.2632 / 0.6680` | `0.0552 / 0.1636` | `3.6244 / 0.9795` | `0.0581` | `0.3151` |
| `UnlesionedPostLife` | `12` | `0.8326` | `0.2781 / 0.5315` | `0.1185 / 0.4870` | `3.4330 / 0.9277` | `0.1509` | `0.3340` |
| `FlaglessPostLifeLesion` | `12` | `0.6515` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6751 / 0.9931` | `0.0029` | `0.2229` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1111.0 [148.8]` | `341.137 [626.398]` | `0.950 [0.007]` | `2.0 [0.2]` | `4.0` | `231.11` | `0.0` |
| `FrozenPlasticity` | `1448.5 [51.0]` | `465.524 [189.532]` | `0.450 [0.000]` | `2.0 [1.2]` | `8.5` | `177.16` | `0.0` |
| `YokedMetabolism` | `1063.0 [245.8]` | `282.279 [171.081]` | `0.949 [0.004]` | `2.0 [1.2]` | `38.0` | `228.30` | `1.0` |
| `ActionShuffled` | `1158.5 [288.8]` | `556.314 [188.149]` | `0.950 [0.004]` | `1.0 [1.0]` | `3.0` | `180.81` | `0.5` |
| `UnlesionedPostLife` | `256.0 [210.2]` | `92.523 [135.363]` | `0.950 [0.007]` | `1.0 [0.2]` | `9.5` | `59.88` | `0.0` |
| `FlaglessPostLifeLesion` | `403.5 [41.2]` | `221.354 [94.597]` | `0.000 [0.000]` | `1.0 [0.2]` | `4.0` | `54.64` | `0.0` |

### Cross-Habitat Modality Comparison (`01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)` vs `01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `-255.00000` | `-0.8333` | `0.0010` |
| `echo_energy_received` | `-49.36514` | `+0.0972` | `0.7113` |
| `photonic_band_conductance_share` | `+0.98028` | `+1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `+0.01972` | `+0.0208` | `0.9441` |
