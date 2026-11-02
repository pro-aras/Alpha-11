# Observational Twin Study: 01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0018 [0.0007]` (`0.0013..0.0039`) | `0.0007 [0.0011]` | `0.0013 [0.0012]` | `0.0034 -> 0.0000` | `0.0000 [0.0000]` | `0.0130 [0.0014]` | `9.293` |
| `FrozenPlasticity` | `2400` | `0.0007 [0.0005]` (`0.0004..0.0014`) | `0.0006 [0.0012]` | `0.0001 [0.0012]` | `0.0012 -> 0.0001` | `0.0000 [0.0000]` | `0.0118 [0.0016]` | `10.217` |
| `YokedMetabolism` | `2400` | `0.0009 [0.0008]` (`0.0003..0.0028`) | `0.0008 [0.0014]` | `0.0003 [0.0017]` | `0.0015 -> 0.0001` | `0.0000 [0.0000]` | `0.0128 [0.0015]` | `9.361` |
| `ActionShuffled` | `2400` | `0.0005 [0.0002]` (`0.0003..0.0016`) | `0.0006 [0.0008]` | `-0.0002 [0.0010]` | `0.0009 -> 0.0001` | `0.0000 [0.0000]` | `0.0122 [0.0012]` | `6.070` |
| `UnlesionedPostLife` | `600` | `0.0014 [0.0009]` (`0.0006..0.0077`) | `0.0002 [0.0008]` | `0.0012 [0.0011]` | `0.0026 -> 0.0000` | `0.0000 [0.0000]` | `0.0116 [0.0010]` | `9.834` |
| `FlaglessPostLifeLesion` | `600` | `0.0019 [0.0028]` (`0.0007..0.0089`) | `0.0003 [0.0022]` | `0.0012 [0.0046]` | `0.0020 -> 0.0000` | `0.0000 [0.0000]` | `0.0114 [0.0007]` | `9.563` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `40.5 [2.2]` (`p95=43.5`) | `377.0 [55.0]` | `33.5` | `9.0` | `8.5` | `4.31 / 9.59 / 4.0` | `8.5 / 0.0 / 1.0 / 136.0` | `2246.5 / 158.5 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2271.5 / 128.5 / 105.0` |
| `YokedMetabolism` | `40.5 [3.0]` (`p95=44.0`) | `373.5 [34.5]` | `34.0` | `9.0` | `8.5` | `4.33 / 9.60 / 4.0` | `8.5 / 0.0 / 1.0 / 124.5` | `2250.5 / 154.0 / 105.0` |
| `ActionShuffled` | `40.5 [3.0]` (`p95=43.0`) | `385.5 [41.5]` | `34.0` | `8.5` | `8.5` | `4.47 / 11.43 / 4.0` | `8.5 / 0.0 / 0.0 / 98.5` | `2240.5 / 160.0 / 105.0` |
| `UnlesionedPostLife` | `40.5 [2.2]` (`p95=43.5`) | `377.0 [55.0]` | `33.5` | `9.0` | `8.5` | `4.31 / 9.59 / 4.0` | `8.5 / 0.0 / 1.0 / 136.0` | `2800.0 / 205.0 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `574.0 / 26.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00107` | `2.0` (`142.0`) | `+0.9722` | `0.0010` | `0.0001` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00112` | `29.0` (`115.0`) | `+0.5972` | `0.0180` | `0.0141` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00005` | `45.0` (`45.0`) | `-0.3750` | `0.1079` | `0.0954` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1662` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1658` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00084` | `18.0` (`126.0`) | `+0.7500` | `0.0040` | `0.0020` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00098` | `33.0` (`111.0`) | `+0.5417` | `0.0300` | `0.0262` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00006` | `42.0` (`42.0`) | `-0.4167` | `0.0709` | `0.0634` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `65.0` (`79.0`) | `+0.0972` | `0.4935` | `0.5139` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `70.0` (`74.0`) | `+0.0278` | `0.7552` | `0.8939` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00129` | `3.0` (`141.0`) | `+0.9583` | `0.0010` | `0.0001` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00141` | `24.0` (`120.0`) | `+0.6667` | `0.0060` | `0.0061` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `-0.00007` | `55.0` (`55.0`) | `-0.2361` | `0.3197` | `0.2734` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `65.0` (`79.0`) | `+0.0972` | `0.4935` | `0.5139` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.5797` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00050` | `56.0` (`56.0`) | `-0.2222` | `0.3916` | `0.3708` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00003` | `70.0` (`74.0`) | `+0.0278` | `0.9271` | `0.9310` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00118` | `23.0` (`121.0`) | `+0.6806` | `0.0050` | `0.0051` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [75.6%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=17.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=45.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `240` | `10%` | `0.0145` | `12.317` | `39.0` | `404.0` | `4.25` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `600` | `25%` | `0.0000` | `11.613` | `40.0` | `398.0` | `4.28` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `11.016` | `40.0` | `386.0` | `4.28` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `28.3%` / `0.50` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `10.156` | `40.0` | `382.5` | `4.31` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `9.293` | `40.5` | `377.0` | `4.31` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `0.0%` / `0.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.3757` | `0.8364 / 1.0000` | `0.0738 / 0.2207` | `3.6987 / 0.9995` | `0.0013` | `0.4208` |
| `FrozenPlasticity` | `12` | `0.4207` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6952 / 0.9986` | `0.0005` | `0.2928` |
| `YokedMetabolism` | `12` | `0.6388` | `0.5736 / 1.0000` | `0.0182 / 0.1241` | `3.6988 / 0.9996` | `0.0008` | `0.3704` |
| `ActionShuffled` | `12` | `0.6384` | `0.4545 / 1.0000` | `0.0375 / 0.1385` | `3.6992 / 0.9997` | `0.0007` | `0.3454` |
| `UnlesionedPostLife` | `12` | `1.0853` | `0.8364 / 1.0000` | `0.0912 / 0.2826` | `3.6953 / 0.9986` | `0.0031` | `0.4870` |
| `FlaglessPostLifeLesion` | `12` | `0.9045` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6855 / 0.9960` | `0.0024` | `0.2433` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1366.0 [25.8]` | `390.502 [115.942]` | `0.950 [0.001]` | `2.0 [1.2]` | `5.0` | `182.48` | `0.0` |
| `FrozenPlasticity` | `1457.5 [33.5]` | `526.976 [286.529]` | `0.450 [0.000]` | `2.0 [0.5]` | `9.0` | `175.88` | `0.0` |
| `YokedMetabolism` | `1373.5 [55.2]` | `465.477 [185.596]` | `0.950 [0.002]` | `2.0 [2.0]` | `10.5` | `180.64` | `0.0` |
| `ActionShuffled` | `1404.0 [18.8]` | `540.213 [187.396]` | `0.950 [0.000]` | `2.0 [0.5]` | `8.5` | `167.05` | `1.5` |
| `UnlesionedPostLife` | `343.0 [28.8]` | `108.643 [33.001]` | `0.950 [0.001]` | `1.0 [0.0]` | `0.5` | `43.55` | `0.0` |
| `FlaglessPostLifeLesion` | `367.0 [12.0]` | `151.884 [36.851]` | `0.000 [0.000]` | `1.0 [1.0]` | `0.5` | `45.26` | `0.0` |

### Cross-Habitat Modality Comparison (`01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)` vs `01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `+255.00000` | `+0.8333` | `0.0010` |
| `echo_energy_received` | `+49.36514` | `-0.0972` | `0.7113` |
| `photonic_band_conductance_share` | `-0.98028` | `-1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `-0.01972` | `-0.0208` | `0.9441` |
