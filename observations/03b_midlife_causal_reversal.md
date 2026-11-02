# Observational Twin Study: 03b Mid-Life Causal Reversal (Motor + Thermal Vent Polarity Inverted at Tick 1500)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `3000` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `3000` | `0.0030 [0.0027]` (`0.0010..0.0068`) | `0.0004 [0.0004]` | `0.0026 [0.0026]` | `0.0054 -> 0.0002` | `0.0001 [0.0007]` | `0.0246 [0.0225]` | `0.000` |
| `FrozenPlasticity` | `3000` | `0.0008 [0.0006]` (`0.0004..0.0030`) | `0.0003 [0.0005]` | `0.0005 [0.0008]` | `0.0013 -> 0.0002` | `0.0000 [0.0000]` | `0.0259 [0.0099]` | `10.132` |
| `YokedMetabolism` | `3000` | `0.0015 [0.0014]` (`0.0001..0.0041`) | `0.0022 [0.0045]` | `0.0002 [0.0024]` | `0.0029 -> 0.0003` | `0.0000 [0.0006]` | `0.0539 [0.0287]` | `0.000` |
| `ActionShuffled` | `3000` | `0.0005 [0.0005]` (`0.0002..0.0017`) | `0.0008 [0.0013]` | `0.0000 [0.0013]` | `0.0007 -> 0.0002` | `0.0000 [0.0002]` | `0.0452 [0.0401]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0009 [0.0033]` (`0.0000..0.0047`) | `0.0029 [0.0076]` | `-0.0024 [0.0075]` | `0.0005 -> 0.0000` | `0.0000 [0.0000]` | `0.0315 [0.0270]` | `0.000` |
| `FlaglessPostLifeLesion` | `600` | `0.0003 [0.0052]` (`0.0000..0.0149`) | `0.0031 [0.0034]` | `-0.0000 [0.0028]` | `0.0000 -> 0.0000` | `0.0000 [0.0000]` | `0.0114 [0.0007]` | `1.033` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `70.5 [15.0]` (`p95=89.5`) | `771.5 [175.0]` | `200.0` | `107.5` | `39.0` | `5.29 / 16.29 / 4.5` | `38.5 / 0.5 / 3.0 / 164.0` | `2617.5 / 494.5 / 147.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2826.0 / 174.0 / 145.5` |
| `YokedMetabolism` | `70.5 [15.8]` (`p95=90.5`) | `776.0 [195.5]` | `168.5` | `47.0` | `39.5` | `5.11 / 14.33 / 4.0` | `38.5 / 0.0 / 3.0 / 142.0` | `2603.0 / 473.5 / 147.0` |
| `ActionShuffled` | `55.0 [2.0]` (`p95=61.4`) | `581.5 [62.2]` | `139.0` | `69.5` | `23.5` | `4.74 / 10.04 / 4.0` | `23.0 / 0.0 / 0.0 / 130.0` | `2711.5 / 369.5 / 147.0` |
| `UnlesionedPostLife` | `70.5 [15.0]` (`p95=89.5`) | `771.5 [175.0]` | `200.0` | `107.5` | `39.0` | `5.29 / 16.29 / 4.5` | `38.5 / 0.5 / 3.0 / 164.0` | `3182.5 / 576.5 / 168.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `576.0 / 24.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00217` | `14.0` (`130.0`) | `+0.8056` | `0.0010` | `0.0009` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00201` | `15.0` (`129.0`) | `+0.7917` | `0.0020` | `0.0011` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00002` | `70.0` (`74.0`) | `+0.0278` | `0.9391` | `0.9301` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00010` | `36.0` (`108.0`) | `+0.5000` | `0.0240` | `0.0071` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1662` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00149` | `46.0` (`98.0`) | `+0.3611` | `0.1708` | `0.1410` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00237` | `11.0` (`133.0`) | `+0.8472` | `0.0010` | `0.0005` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00007` | `71.0` (`73.0`) | `+0.0139` | `0.9760` | `0.9767` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00010` | `62.0` (`82.0`) | `+0.1389` | `0.5744` | `0.5403` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `65.0` (`79.0`) | `+0.0972` | `0.4935` | `0.5139` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00249` | `6.0` (`138.0`) | `+0.9167` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00251` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00004` | `54.5` (`89.5`) | `+0.2431` | `0.3357` | `0.3173` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00010` | `60.0` (`84.0`) | `+0.1667` | `0.5005` | `0.4585` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00057` | `59.0` (`59.0`) | `-0.1806` | `0.4815` | `0.4650` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00238` | `51.5` (`51.5`) | `-0.2847` | `0.2687` | `0.2481` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `-0.00050` | `61.0` (`61.0`) | `-0.1528` | `0.5504` | `0.5440` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `93.4% [9.1%]` (`p95=99.5%`) | `52.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `6.6% [9.1%]` (`p95=18.9%`) | `6.5` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.6%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `96.9% [3.9%]` (`p95=99.4%`) | `55.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `3.1% [3.9%]` (`p95=6.8%`) | `5.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `99.6% [1.5%]` (`p95=100.0%`) | `58.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.4% [1.3%]` (`p95=3.7%`) | `1.0` | `1.00 [1.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.6%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `93.4% [9.1%]` (`p95=99.5%`) | `52.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `6.6% [9.1%]` (`p95=18.9%`) | `6.5` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.6%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `300` | `10%` | `0.0174` | `12.333` | `43.0` | `452.5` | `4.40` | `4.0` | `0.0000` | `0.0%` | `98.6%` / `1.00` | `1.1%` / `1.00` | `0.0%` / `0.00` |
| `750` | `25%` | `0.0000` | `4.022` | `62.0` | `678.0` | `5.13` | `4.5` | `0.0000` | `0.0%` | `99.7%` / `1.00` | `0.3%` / `1.00` | `0.0%` / `0.00` |
| `1500` | `50%` | `0.0001` | `1.893` | `70.0` | `770.0` | `5.29` | `4.5` | `0.0000` | `0.0%` | `98.4%` / `1.00` | `1.6%` / `1.00` | `0.0%` / `0.00` |
| `2250` | `75%` | `0.0002` | `0.416` | `70.5` | `771.0` | `5.29` | `4.5` | `0.0002` | `0.0%` | `96.6%` / `1.00` | `3.4%` / `1.00` | `0.0%` / `0.00` |
| `3000` | `100%` | `0.0000` | `0.000` | `70.5` | `771.5` | `5.29` | `4.5` | `0.0001` | `0.0%` | `93.4%` / `1.00` | `6.6%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.5673` | `0.3055 / 0.5723` | `0.0390 / 0.0927` | `3.6461 / 0.9853` | `0.0318` | `0.3009` |
| `FrozenPlasticity` | `12` | `0.6810` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6968 / 0.9990` | `0.0006` | `0.3202` |
| `YokedMetabolism` | `12` | `0.6045` | `0.2628 / 0.5192` | `0.0245 / 0.0711` | `3.6340 / 0.9820` | `0.0239` | `0.2896` |
| `ActionShuffled` | `12` | `0.7044` | `0.2232 / 0.3934` | `0.0662 / 0.1822` | `3.6519 / 0.9869` | `0.0420` | `0.3005` |
| `UnlesionedPostLife` | `12` | `1.1367` | `0.3055 / 0.5723` | `0.0680 / 0.1509` | `3.4328 / 0.9277` | `0.2405` | `0.3507` |
| `FlaglessPostLifeLesion` | `12` | `1.4354` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6704 / 0.9919` | `0.0025` | `0.2715` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1370.5 [106.0]` | `788.387 [562.799]` | `0.950 [0.000]` | `2.0 [2.2]` | `6.5` | `351.04` | `1.0` |
| `FrozenPlasticity` | `1791.0 [42.0]` | `548.055 [154.110]` | `0.450 [0.000]` | `2.0 [1.0]` | `10.0` | `222.18` | `0.0` |
| `YokedMetabolism` | `1297.0 [159.0]` | `449.353 [181.160]` | `0.950 [0.000]` | `2.0 [1.2]` | `30.0` | `301.77` | `1.0` |
| `ActionShuffled` | `1568.5 [456.5]` | `919.840 [714.549]` | `0.950 [0.001]` | `1.0 [1.0]` | `7.0` | `252.93` | `1.0` |
| `UnlesionedPostLife` | `237.0 [152.5]` | `56.108 [116.343]` | `0.950 [0.000]` | `0.0 [0.0]` | `0.0` | `53.61` | `0.0` |
| `FlaglessPostLifeLesion` | `431.5 [13.8]` | `276.501 [28.390]` | `0.000 [0.000]` | `0.5 [1.0]` | `1.5` | `65.90` | `0.0` |

## 8. Mid-Life Causal Reversal Shock, Contradiction Splitting & Re-Acquisition (`Natural`, `N = 12`)

- **Contradiction Splits (`1..=1500` vs `1501..=3000`)**: `0.0 [0.0]` vs `0.0 [2.5]`
- **Mean Temperature (`1201..=1500` -> `1501..=1800`, equal 300-tick windows)**: `0.4952 [0.0224]` -> `0.5623 [0.1145]` (`r_rb = +0.6667`, `p_perm = 0.0060`)
- **Initial Learning Windows (`1..=300` -> `301..=600` Influx/Tick `p50`)**: `0.0173` -> `0.0000`
- **Post-Reversal Windows (`1501..=1800` -> `1801..=2100` -> `2401..=3000` Influx/Tick `p50`)**: `0.0002` -> `0.0000` -> `0.0000`

| Reversal Hypothesis Test | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | ---: | ---: | ---: |
| `Post-Reversal Half (1501..=3000): Natural vs FrozenPlasticity` | `+0.00002` | `+0.0278` | `0.9391` |
| `Post-Reversal Half (1501..=3000): Natural vs YokedMetabolism` | `-0.00007` | `+0.0139` | `0.9760` |
| `Post-Reversal Half (1501..=3000): Natural vs ActionShuffled` | `+0.00004` | `+0.2431` | `0.3357` |
| `Post-Reversal Late (2401..=3000) vs Shock (1501..=1800)` | `-0.00018` | `-0.4167` | `0.0629` |
| `Re-Acquisition (1501..=1800) vs Initial (1..=300)` | `-0.01716` | `-0.8889` | `0.0010` |
| `Re-Acquisition (1801..=2100) vs Initial (301..=600)` | `+0.00000` | `-0.1111` | `0.4635` |
