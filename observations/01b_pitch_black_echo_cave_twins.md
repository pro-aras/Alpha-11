# Observational Twin Study: 01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0017 [0.0009]` (`0.0012..0.0031`) | `0.0005 [0.0008]` | `0.0012 [0.0012]` | `0.0033 -> 0.0000` | `0.0000 [0.0000]` | `0.0119 [0.0014]` | `0.479` |
| `FrozenPlasticity` | `2400` | `0.0008 [0.0006]` (`0.0004..0.0015`) | `0.0006 [0.0013]` | `0.0000 [0.0007]` | `0.0012 -> 0.0001` | `0.0000 [0.0000]` | `0.0115 [0.0009]` | `10.237` |
| `YokedMetabolism` | `2400` | `0.0012 [0.0011]` (`0.0002..0.0024`) | `0.0004 [0.0013]` | `0.0006 [0.0008]` | `0.0025 -> 0.0000` | `0.0000 [0.0000]` | `0.0121 [0.0015]` | `0.399` |
| `ActionShuffled` | `2400` | `0.0006 [0.0004]` (`0.0003..0.0011`) | `0.0010 [0.0014]` | `-0.0000 [0.0017]` | `0.0008 -> 0.0001` | `0.0000 [0.0000]` | `0.0129 [0.0024]` | `1.355` |
| `UnlesionedPostLife` | `600` | `0.0019 [0.0026]` (`0.0004..0.0056`) | `0.0001 [0.0010]` | `0.0015 [0.0022]` | `0.0029 -> 0.0000` | `0.0000 [0.0000]` | `0.0122 [0.0012]` | `1.094` |
| `FlaglessPostLifeLesion` | `600` | `0.0024 [0.0027]` (`0.0006..0.0074`) | `0.0007 [0.0037]` | `0.0016 [0.0063]` | `0.0043 -> 0.0000` | `0.0000 [0.0000]` | `0.0121 [0.0008]` | `1.252` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `58.5 [7.5]` (`p95=65.2`) | `616.0 [109.5]` | `106.5` | `27.5` | `53.0` | `5.04 / 16.59 / 4.5` | `26.5 / 0.0 / 1.0 / 174.0` | `2217.5 / 190.5 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2272.0 / 128.0 / 105.0` |
| `YokedMetabolism` | `60.0 [8.8]` (`p95=65.3`) | `626.0 [89.8]` | `112.0` | `28.0` | `56.0` | `5.16 / 15.78 / 4.5` | `28.0 / 0.0 / 1.0 / 138.0` | `2215.0 / 190.0 / 105.0` |
| `ActionShuffled` | `52.5 [2.2]` (`p95=55.5`) | `542.5 [29.5]` | `82.5` | `20.5` | `41.0` | `5.02 / 15.22 / 4.0` | `20.5 / 0.0 / 0.0 / 199.5` | `2240.0 / 160.5 / 105.0` |
| `UnlesionedPostLife` | `58.5 [7.5]` (`p95=65.2`) | `616.0 [109.5]` | `106.5` | `27.5` | `53.0` | `5.04 / 16.59 / 4.5` | `26.5 / 0.0 / 1.0 / 174.0` | `2765.0 / 243.0 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00089` | `8.0` (`136.0`) | `+0.8889` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00117` | `20.0` (`124.0`) | `+0.7222` | `0.0040` | `0.0029` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00010` | `46.5` (`46.5`) | `-0.3542` | `0.1568` | `0.1341` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00042` | `35.0` (`109.0`) | `+0.5139` | `0.0410` | `0.0351` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00057` | `37.0` (`107.0`) | `+0.4861` | `0.0500` | `0.0464` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00000` | `58.5` (`85.5`) | `+0.1875` | `0.4176` | `0.3712` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `0.7473` | `0.5807` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00111` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00122` | `13.0` (`131.0`) | `+0.8194` | `0.0010` | `0.0007` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `-0.00011` | `48.0` (`48.0`) | `-0.3333` | `0.1838` | `0.1537` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `71.5` (`71.5`) | `-0.0069` | `1.0000` | `1.0000` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `71.5` (`71.5`) | `-0.0069` | `1.0000` | `1.0000` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00057` | `53.0` (`53.0`) | `-0.2639` | `0.3177` | `0.2855` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00017` | `71.0` (`73.0`) | `+0.0139` | `0.9810` | `0.9770` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00167` | `18.0` (`126.0`) | `+0.7500` | `0.0040` | `0.0020` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=3.9%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `50.0% [100.0%]` (`p95=100.0%`) | `0.5` | `0.50 [1.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `7.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=45.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=3.9%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `240` | `10%` | `0.0132` | `11.570` | `40.0` | `416.0` | `4.29` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `600` | `25%` | `0.0002` | `7.107` | `52.0` | `547.0` | `4.78` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `2.441` | `58.5` | `623.5` | `5.04` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `1.354` | `58.5` | `617.0` | `5.04` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `0.479` | `58.5` | `616.0` | `5.04` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `0.0%` / `0.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.3357` | `0.6636 / 1.0000` | `0.0527 / 0.1381` | `3.6003 / 0.9729` | `0.0738` | `0.3661` |
| `FrozenPlasticity` | `12` | `0.4550` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6952 / 0.9986` | `0.0005` | `0.2970` |
| `YokedMetabolism` | `12` | `0.6082` | `0.7383 / 1.0000` | `0.0652 / 0.1760` | `3.6041 / 0.9740` | `0.0535` | `0.4190` |
| `ActionShuffled` | `12` | `0.4765` | `0.8049 / 1.0000` | `0.0190 / 0.0530` | `3.6611 / 0.9894` | `0.0345` | `0.4115` |
| `UnlesionedPostLife` | `12` | `0.8044` | `0.6636 / 1.0000` | `0.0729 / 0.1589` | `3.4591 / 0.9348` | `0.1885` | `0.4190` |
| `FlaglessPostLifeLesion` | `12` | `0.7552` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6761 / 0.9934` | `0.0030` | `0.2319` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `213.0 [1118.0]` | `256.814 [84.291]` | `0.948 [0.008]` | `2.0 [0.5]` | `4.0` | `179.22` | `0.0` |
| `FrozenPlasticity` | `1457.5 [28.8]` | `538.328 [246.847]` | `0.450 [0.000]` | `2.0 [0.0]` | `11.5` | `176.83` | `0.0` |
| `YokedMetabolism` | `257.0 [394.2]` | `260.294 [64.070]` | `0.949 [0.017]` | `1.5 [2.0]` | `2.5` | `182.44` | `0.0` |
| `ActionShuffled` | `1200.0 [80.2]` | `423.320 [145.591]` | `0.950 [0.000]` | `2.0 [0.0]` | `3.0` | `153.38` | `1.0` |
| `UnlesionedPostLife` | `7.5 [304.2]` | `2.598 [103.649]` | `0.948 [0.008]` | `1.0 [0.0]` | `1.0` | `41.85` | `0.0` |
| `FlaglessPostLifeLesion` | `405.0 [58.0]` | `218.630 [100.561]` | `0.000 [0.000]` | `1.0 [0.0]` | `1.5` | `52.64` | `0.0` |

### Cross-Habitat Modality Comparison (`01b PitchBlackEchoCave (128ch: Zero Light + Active Sonar + NoisyTV)` vs `01a BrightRadiantWorld (128ch: Photonic + Echo + NoisyTV)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `-776.50000` | `-0.3889` | `0.1449` |
| `echo_energy_received` | `-793.12549` | `-0.6250` | `0.0110` |
| `photonic_band_conductance_share` | `-0.86843` | `-1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `-0.13157` | `-0.1181` | `0.6274` |
