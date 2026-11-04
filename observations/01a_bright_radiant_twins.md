# Observational Twin Study: 01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0034 [0.0018]` (`0.0022..0.0079`) | `0.0007 [0.0014]` | `0.0024 [0.0019]` | `0.0063 -> 0.0000` | `0.0006 [0.0285]` | `0.0426 [0.0396]` | `0.000` |
| `FrozenPlasticity` | `2400` | `0.0008 [0.0004]` (`0.0004..0.0017`) | `0.0007 [0.0015]` | `0.0000 [0.0016]` | `0.0016 -> 0.0000` | `0.0000 [0.0000]` | `0.0224 [0.0134]` | `10.421` |
| `YokedMetabolism` | `2400` | `0.0016 [0.0035]` (`0.0006..0.0053`) | `0.0051 [0.0044]` | `-0.0024 [0.0050]` | `0.0019 -> 0.0001` | `0.0011 [0.0018]` | `0.0339 [0.0144]` | `0.000` |
| `ActionShuffled` | `2400` | `0.0012 [0.0012]` (`0.0003..0.0036`) | `0.0009 [0.0020]` | `0.0003 [0.0015]` | `0.0022 -> 0.0000` | `0.0000 [0.0002]` | `0.0316 [0.0132]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0020 [0.0036]` (`0.0003..0.0175`) | `0.0042 [0.0115]` | `-0.0026 [0.0140]` | `0.0040 -> 0.0000` | `0.0000 [0.0003]` | `0.0469 [0.0216]` | `0.275` |
| `FlaglessPostLifeLesion` | `600` | `0.0025 [0.0046]` (`0.0006..0.0093`) | `0.0017 [0.0048]` | `0.0013 [0.0086]` | `0.0047 -> 0.0000` | `0.0000 [0.0000]` | `0.0122 [0.0008]` | `1.844` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `68.5 [8.5]` (`p95=86.3`) | `768.5 [92.5]` | `240.5` | `99.5` | `75.0` | `5.29 / 18.13 / 5.0` | `36.5 / 1.0 / 3.5 / 111.5` | `2169.5 / 402.0 / 105.5` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2271.0 / 129.0 / 105.0` |
| `YokedMetabolism` | `68.5 [9.2]` (`p95=85.8`) | `754.5 [144.2]` | `174.0` | `52.5` | `73.5` | `5.09 / 11.54 / 4.0` | `36.5 / 0.0 / 3.0 / 136.0` | `2127.0 / 376.0 / 105.0` |
| `ActionShuffled` | `57.5 [5.0]` (`p95=64.9`) | `608.0 [68.2]` | `160.0` | `79.0` | `51.0` | `4.82 / 8.78 / 4.0` | `25.5 / 0.0 / 1.0 / 140.5` | `2158.5 / 303.0 / 105.0` |
| `UnlesionedPostLife` | `68.5 [8.5]` (`p95=86.3`) | `768.5 [92.5]` | `240.5` | `99.5` | `75.0` | `5.29 / 18.13 / 5.0` | `36.5 / 1.0 / 3.5 / 111.5` | `2730.0 / 443.0 / 145.5` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00253` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00242` | `8.0` (`136.0`) | `+0.8889` | `0.0010` | `0.0002` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00001` | `70.0` (`74.0`) | `+0.0278` | `0.9201` | `0.9248` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00065` | `36.0` (`108.0`) | `+0.5000` | `0.0240` | `0.0071` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00173` | `36.0` (`108.0`) | `+0.5000` | `0.0460` | `0.0404` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00483` | `10.0` (`134.0`) | `+0.8611` | `0.0010` | `0.0004` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00011` | `59.5` (`59.5`) | `-0.1736` | `0.4885` | `0.4591` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `-0.00046` | `68.0` (`76.0`) | `+0.0556` | `0.8192` | `0.8339` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `60.0` (`60.0`) | `-0.1667` | `0.4935` | `0.1662` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00216` | `12.0` (`132.0`) | `+0.8333` | `0.0010` | `0.0006` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00219` | `17.0` (`127.0`) | `+0.7639` | `0.0030` | `0.0017` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `64.0` (`80.0`) | `+0.1111` | `0.6354` | `0.6186` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00065` | `51.0` (`93.0`) | `+0.2917` | `0.2028` | `0.1863` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `1.0000` | `0.3593` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00043` | `55.0` (`55.0`) | `-0.2361` | `0.3656` | `0.3408` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00393` | `53.0` (`53.0`) | `-0.2639` | `0.3177` | `0.2855` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00161` | `14.0` (`130.0`) | `+0.8056` | `0.0010` | `0.0009` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `86.8% [20.8%]` (`p95=99.5%`) | `49.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `13.2% [20.8%]` (`p95=35.7%`) | `10.5` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `92.6% [22.8%]` (`p95=98.8%`) | `49.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `7.4% [22.8%]` (`p95=50.6%`) | `11.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=1.1%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `85.6% [12.7%]` (`p95=97.0%`) | `46.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `14.4% [12.7%]` (`p95=27.8%`) | `13.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.1%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `86.8% [20.8%]` (`p95=99.5%`) | `49.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `13.2% [20.8%]` (`p95=35.7%`) | `10.5` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `240` | `10%` | `0.0239` | `14.136` | `40.0` | `416.5` | `4.30` | `4.0` | `0.0000` | `0.0%` | `89.5%` / `1.00` | `10.3%` / `1.00` | `0.0%` / `0.00` |
| `600` | `25%` | `0.0012` | `8.510` | `55.0` | `596.5` | `4.75` | `4.0` | `0.0010` | `0.0%` | `86.1%` / `1.00` | `13.9%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0008` | `2.501` | `68.0` | `765.0` | `5.19` | `4.0` | `0.0159` | `0.0%` | `74.5%` / `1.00` | `25.5%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `1.187` | `68.5` | `773.5` | `5.29` | `5.0` | `0.0099` | `0.0%` | `74.9%` / `1.00` | `24.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `0.000` | `68.5` | `768.5` | `5.29` | `5.0` | `0.0006` | `0.0%` | `86.8%` / `1.00` | `13.2%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.4379` | `0.4332 / 0.6538` | `0.0349 / 0.0995` | `3.5735 / 0.9657` | `0.1165` | `0.3179` |
| `FrozenPlasticity` | `12` | `0.4184` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6952 / 0.9986` | `0.0006` | `0.2926` |
| `YokedMetabolism` | `12` | `0.7842` | `0.3771 / 0.6656` | `0.0294 / 0.0964` | `3.5266 / 0.9530` | `0.1213` | `0.3349` |
| `ActionShuffled` | `12` | `0.7414` | `0.3723 / 0.5743` | `0.0503 / 0.2161` | `3.5949 / 0.9715` | `0.0908` | `0.3367` |
| `UnlesionedPostLife` | `12` | `1.3104` | `0.4332 / 0.6538` | `0.0770 / 0.1605` | `2.9507 / 0.7974` | `0.6334` | `0.3911` |
| `FlaglessPostLifeLesion` | `12` | `0.7661` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6743 / 0.9929` | `0.0029` | `0.2327` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `989.5 [780.5]` | `1049.940 [2287.353]` | `0.950 [0.003]` | `2.5 [1.5]` | `5.0` | `422.99` | `0.0` |
| `FrozenPlasticity` | `1452.0 [53.5]` | `494.092 [168.172]` | `0.450 [0.000]` | `2.0 [0.5]` | `6.5` | `176.99` | `0.0` |
| `YokedMetabolism` | `917.0 [424.8]` | `594.272 [916.060]` | `0.950 [0.009]` | `1.0 [1.0]` | `75.5` | `260.30` | `0.5` |
| `ActionShuffled` | `1047.5 [331.8]` | `948.275 [1139.405]` | `0.950 [0.000]` | `2.0 [0.2]` | `4.5` | `232.87` | `1.0` |
| `UnlesionedPostLife` | `219.5 [378.5]` | `45.460 [133.789]` | `0.950 [0.003]` | `0.0 [1.0]` | `3.5` | `45.16` | `0.0` |
| `FlaglessPostLifeLesion` | `409.0 [43.2]` | `239.488 [80.592]` | `0.000 [0.000]` | `1.0 [0.2]` | `4.5` | `54.59` | `0.0` |

### Cross-Habitat Modality Comparison (`01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)` vs `01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `+776.50000` | `+0.3889` | `0.1449` |
| `echo_energy_received` | `+793.12549` | `+0.6250` | `0.0110` |
| `photonic_band_conductance_share` | `+0.86843` | `+1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `+0.13157` | `+0.1181` | `0.6274` |
