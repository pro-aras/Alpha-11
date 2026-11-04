# Study 2: Coordinate-Free Bijective Wire Permutation Invariance (`PermutedMedium`)

All `128` ambient sensory channels and all `6` motor actuator channels were bijectively scrambled via `PermutedMedium::new(world, seed ^ 0x5C8A_4B1E_9D2F_7103)` before entering `Alpha11Organism`. Channel identities were inverted **exclusively inside `a11-observe`** (`invert_sensor_channel` / `invert_motor_channel`) after each snapshot.

## Invariance Summary (`Permuted` vs `Unpermuted` `Natural` Cohort, `N = 12` Seeds)

| Habitat | Observable | Permuted `p50` | Unpermuted `p50` | `median(Perm) - median(Unperm)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `BrightRadiantWorld` | `achieved_influx_per_tick` | `0.0038` | `0.0034` | `+0.00040` | `-0.1389` | `0.5974` |
| `BrightRadiantWorld` | `decoded_photonic_conductance_share` | `0.8135` | `0.8684` | `-0.05496` | `-0.0208` | `0.9441` |
| `BrightRadiantWorld` | `decoded_sonar_conductance_share` | `0.1865` | `0.1316` | `+0.05496` | `-0.0139` | `0.9730` |
| `BrightRadiantWorld` | `decoded_noisy_tv_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `+0.1667` | `0.4935` |
| `BrightRadiantWorld` | `active_pings_emitted` | `789.0000` | `989.5000` | `-200.50000` | `-0.1667` | `0.5145` |
| `BrightRadiantWorld` | `decoded_emit_pulse_motor_conductance` | `0.9500` | `0.9499` | `+0.00011` | `+0.1250` | `0.6094` |
| `PitchBlackEchoCave` | `achieved_influx_per_tick` | `0.0012` | `0.0017` | `-0.00049` | `-0.3889` | `0.1449` |
| `PitchBlackEchoCave` | `decoded_photonic_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `+0.0000` | `1.0000` |
| `PitchBlackEchoCave` | `decoded_sonar_conductance_share` | `1.0000` | `0.0000` | `+1.00000` | `+0.2153` | `0.4256` |
| `PitchBlackEchoCave` | `decoded_noisy_tv_conductance_share` | `0.0000` | `0.0000` | `+0.00000` | `-0.0833` | `1.0000` |
| `PitchBlackEchoCave` | `active_pings_emitted` | `309.0000` | `213.0000` | `+96.00000` | `+0.1528` | `0.5534` |
| `PitchBlackEchoCave` | `decoded_emit_pulse_motor_conductance` | `0.9496` | `0.9483` | `+0.00126` | `+0.2361` | `0.3606` |

---

# Observational Twin Study: 02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0038 [0.0031]` (`0.0005..0.0136`) | `0.0008 [0.0014]` | `0.0025 [0.0019]` | `0.0048 -> 0.0021` | `0.0009 [0.0090]` | `0.0405 [0.0516]` | `0.824` |
| `FrozenPlasticity` | `2400` | `0.0009 [0.0009]` (`0.0003..0.0017`) | `0.0004 [0.0007]` | `0.0006 [0.0009]` | `0.0011 -> 0.0001` | `0.0000 [0.0000]` | `0.0364 [0.0092]` | `10.865` |
| `YokedMetabolism` | `2400` | `0.0031 [0.0039]` (`0.0001..0.0089`) | `0.0072 [0.0082]` | `-0.0032 [0.0087]` | `0.0030 -> 0.0015` | `0.0002 [0.0032]` | `0.0356 [0.0499]` | `0.988` |
| `ActionShuffled` | `2400` | `0.0006 [0.0009]` (`0.0002..0.0027`) | `0.0002 [0.0007]` | `0.0003 [0.0010]` | `0.0011 -> 0.0000` | `0.0000 [0.0002]` | `0.0423 [0.0333]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0019 [0.0057]` (`0.0002..0.0140`) | `0.0018 [0.0071]` | `-0.0003 [0.0042]` | `0.0019 -> 0.0000` | `0.0000 [0.0003]` | `0.0486 [0.0458]` | `0.770` |
| `FlaglessPostLifeLesion` | `600` | `0.0018 [0.0019]` (`0.0006..0.0115`) | `0.0006 [0.0019]` | `0.0012 [0.0018]` | `0.0024 -> 0.0003` | `0.0000 [0.0000]` | `0.0121 [0.0007]` | `1.873` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `67.0 [10.5]` (`p95=98.6`) | `715.0 [176.0]` | `180.0` | `101.0` | `71.5` | `5.08 / 14.57 / 4.0` | `35.0 / 1.0 / 4.0 / 139.0` | `2210.0 / 484.0 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2271.5 / 128.5 / 105.0` |
| `YokedMetabolism` | `67.0 [11.0]` (`p95=98.6`) | `728.5 [157.2]` | `192.5` | `87.5` | `70.0` | `5.12 / 15.55 / 4.0` | `35.0 / 0.0 / 4.0 / 138.5` | `2209.0 / 392.0 / 105.0` |
| `ActionShuffled` | `55.0 [1.2]` (`p95=60.5`) | `592.0 [23.2]` | `105.0` | `38.5` | `46.0` | `4.80 / 8.74 / 4.0` | `23.0 / 0.0 / 0.0 / 115.5` | `2138.5 / 313.0 / 105.0` |
| `UnlesionedPostLife` | `67.0 [10.5]` (`p95=98.6`) | `715.0 [176.0]` | `180.0` | `101.0` | `71.5` | `5.08 / 14.57 / 4.0` | `35.0 / 1.0 / 4.0 / 139.0` | `2744.5 / 530.5 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00282` | `21.0` (`123.0`) | `+0.7083` | `0.0040` | `0.0035` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00182` | `22.0` (`122.0`) | `+0.6944` | `0.0050` | `0.0043` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00207` | `57.0` (`87.0`) | `+0.2083` | `0.4086` | `0.3967` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00089` | `18.0` (`126.0`) | `+0.7500` | `0.0010` | `0.0004` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `-0.25000` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00064` | `59.0` (`85.0`) | `+0.1806` | `0.4845` | `0.4705` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00563` | `16.0` (`128.0`) | `+0.7778` | `0.0030` | `0.0014` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00067` | `63.0` (`63.0`) | `-0.1250` | `0.6284` | `0.6142` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00068` | `58.0` (`86.0`) | `+0.1944` | `0.4535` | `0.4234` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `62.0` (`62.0`) | `-0.1389` | `0.6094` | `0.4709` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00313` | `19.0` (`125.0`) | `+0.7361` | `0.0040` | `0.0024` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00214` | `22.0` (`122.0`) | `+0.6944` | `0.0050` | `0.0043` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00214` | `44.5` (`99.5`) | `+0.3819` | `0.1099` | `0.0957` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00089` | `40.5` (`103.5`) | `+0.4375` | `0.0609` | `0.0558` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `70.0` (`74.0`) | `+0.0278` | `0.7552` | `0.8939` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00009` | `71.0` (`73.0`) | `+0.0139` | `0.9810` | `0.9770` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00147` | `52.0` (`52.0`) | `-0.2778` | `0.2887` | `0.2602` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00088` | `30.0` (`114.0`) | `+0.5833` | `0.0220` | `0.0166` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `81.3% [22.1%]` (`p95=99.3%`) | `48.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `18.7% [24.0%]` (`p95=47.7%`) | `16.0` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=1.9%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `45.0% [21.3%]` (`p95=62.2%`) | `9.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `22.5% [15.0%]` (`p95=39.5%`) | `4.5` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `25.0% [11.2%]` (`p95=46.2%`) | `5.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `78.7% [20.2%]` (`p95=97.7%`) | `42.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `21.3% [20.7%]` (`p95=55.1%`) | `19.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.3%]` (`p95=1.6%`) | `0.0` | `0.00 [1.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `91.5% [9.4%]` (`p95=97.6%`) | `53.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `8.5% [8.8%]` (`p95=26.5%`) | `10.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.6%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `81.3% [22.1%]` (`p95=99.3%`) | `48.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `18.7% [24.0%]` (`p95=47.7%`) | `16.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=1.9%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `9.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `4.5` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `5.0` | `1.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `30.0%` | `45.0%` / `1.00` | `30.0%` / `1.00` | `30.0%` / `1.00` |
| `240` | `10%` | `0.0198` | `13.084` | `40.0` | `417.0` | `4.29` | `4.0` | `0.0000` | `0.5%` | `74.7%` / `1.00` | `24.5%` / `1.00` | `0.5%` / `1.00` |
| `600` | `25%` | `0.0009` | `7.407` | `55.0` | `597.0` | `4.70` | `4.0` | `0.0000` | `0.0%` | `68.8%` / `1.00` | `31.2%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `2.175` | `65.0` | `706.5` | `5.01` | `4.0` | `0.0028` | `0.0%` | `67.2%` / `1.00` | `32.8%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0034` | `2.144` | `66.5` | `716.5` | `5.08` | `4.0` | `0.0004` | `0.0%` | `83.0%` / `1.00` | `17.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `0.824` | `67.0` | `715.0` | `5.08` | `4.0` | `0.0009` | `0.0%` | `81.3%` / `1.00` | `18.7%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.8535` | `0.3889 / 0.7156` | `0.0407 / 0.0918` | `3.5572 / 0.9613` | `0.1168` | `0.3468` |
| `FrozenPlasticity` | `12` | `0.5206` | `0.8470 / 1.0000` | `0.0002 / 0.0007` | `3.6956 / 0.9987` | `0.0005` | `0.4222` |
| `YokedMetabolism` | `12` | `0.8702` | `0.4169 / 0.6944` | `0.0323 / 0.0862` | `3.6092 / 0.9754` | `0.0859` | `0.3535` |
| `ActionShuffled` | `12` | `0.8335` | `0.2987 / 0.4668` | `0.0192 / 0.1010` | `3.5672 / 0.9640` | `0.0932` | `0.3169` |
| `UnlesionedPostLife` | `12` | `1.1356` | `0.3889 / 0.7156` | `0.1229 / 0.3371` | `3.2844 / 0.8876` | `0.3584` | `0.3843` |
| `FlaglessPostLifeLesion` | `12` | `1.0645` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6756 / 0.9933` | `0.0027` | `0.2532` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `789.0 [978.8]` | `619.075 [1489.828]` | `0.950 [0.005]` | `2.5 [2.2]` | `8.0` | `304.54` | `1.0` |
| `FrozenPlasticity` | `1424.0 [23.5]` | `498.008 [144.844]` | `0.450 [0.000]` | `2.0 [1.0]` | `4.0` | `180.67` | `1.0` |
| `YokedMetabolism` | `843.5 [320.5]` | `888.398 [850.194]` | `0.950 [0.003]` | `1.0 [1.2]` | `123.0` | `346.95` | `1.0` |
| `ActionShuffled` | `876.0 [162.2]` | `555.086 [424.547]` | `0.950 [0.005]` | `2.0 [2.0]` | `4.5` | `168.02` | `1.0` |
| `UnlesionedPostLife` | `193.5 [319.0]` | `32.550 [364.795]` | `0.950 [0.005]` | `0.5 [1.0]` | `7.5` | `46.04` | `0.0` |
| `FlaglessPostLifeLesion` | `417.5 [37.2]` | `214.454 [75.849]` | `0.000 [0.000]` | `1.0 [0.0]` | `0.5` | `57.90` | `0.0` |

### Cross-Habitat Modality Comparison (`02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` vs `02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `+480.00000` | `+0.1528` | `0.5534` |
| `echo_energy_received` | `+264.13051` | `+0.6806` | `0.0050` |
| `photonic_band_conductance_share` | `+0.81347` | `+1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `-0.81347` | `-0.2361` | `0.3497` |

---

# Observational Twin Study: 02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `2400` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `2400` | `0.0012 [0.0012]` (`0.0007..0.0050`) | `0.0010 [0.0012]` | `0.0006 [0.0016]` | `0.0023 -> 0.0000` | `0.0000 [0.0000]` | `0.0126 [0.0028]` | `1.370` |
| `FrozenPlasticity` | `2400` | `0.0008 [0.0006]` (`0.0002..0.0026`) | `0.0006 [0.0008]` | `0.0002 [0.0011]` | `0.0013 -> 0.0002` | `0.0000 [0.0000]` | `0.0281 [0.0067]` | `10.427` |
| `YokedMetabolism` | `2400` | `0.0006 [0.0005]` (`0.0001..0.0021`) | `0.0003 [0.0015]` | `0.0002 [0.0005]` | `0.0011 -> 0.0000` | `0.0000 [0.0000]` | `0.0123 [0.0021]` | `1.756` |
| `ActionShuffled` | `2400` | `0.0005 [0.0002]` (`0.0002..0.0015`) | `0.0012 [0.0050]` | `-0.0008 [0.0047]` | `0.0011 -> 0.0000` | `0.0000 [0.0000]` | `0.0136 [0.0022]` | `0.518` |
| `UnlesionedPostLife` | `600` | `0.0017 [0.0020]` (`0.0005..0.0077`) | `0.0000 [0.0008]` | `0.0015 [0.0023]` | `0.0031 -> 0.0000` | `0.0000 [0.0000]` | `0.0118 [0.0005]` | `1.745` |
| `FlaglessPostLifeLesion` | `600` | `0.0022 [0.0022]` (`0.0007..0.0061`) | `0.0008 [0.0020]` | `0.0013 [0.0021]` | `0.0025 -> 0.0000` | `0.0000 [0.0000]` | `0.0121 [0.0007]` | `1.877` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `57.5 [4.0]` (`p95=68.5`) | `594.5 [64.2]` | `101.0` | `28.0` | `51.0` | `5.17 / 17.59 / 5.0` | `25.5 / 0.0 / 1.0 / 149.5` | `2203.5 / 203.5 / 105.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2270.0 / 130.0 / 105.0` |
| `YokedMetabolism` | `58.0 [7.0]` (`p95=69.6`) | `590.0 [108.2]` | `104.0` | `28.0` | `52.0` | `5.15 / 15.38 / 4.5` | `26.0 / 0.0 / 1.0 / 183.5` | `2224.5 / 185.5 / 105.0` |
| `ActionShuffled` | `53.0 [0.8]` (`p95=57.5`) | `548.5 [10.5]` | `84.0` | `21.0` | `42.0` | `5.13 / 16.84 / 5.0` | `21.0 / 0.0 / 0.0 / 225.5` | `2244.0 / 156.5 / 105.5` |
| `UnlesionedPostLife` | `57.5 [4.0]` (`p95=68.5`) | `594.5 [64.2]` | `101.0` | `28.0` | `51.0` | `5.17 / 17.59 / 5.0` | `25.5 / 0.0 / 1.0 / 149.5` | `2753.0 / 255.5 / 147.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00038` | `31.0` (`113.0`) | `+0.5694` | `0.0240` | `0.0194` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00048` | `48.0` (`96.0`) | `+0.3333` | `0.1998` | `0.1749` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `-0.00016` | `55.0` (`55.0`) | `-0.2361` | `0.3267` | `0.3087` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `54.0` (`90.0`) | `+0.2500` | `0.2318` | `0.0788` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `-0.25000` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00056` | `27.0` (`117.0`) | `+0.6250` | `0.0110` | `0.0102` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00044` | `43.0` (`101.0`) | `+0.4028` | `0.1309` | `0.0999` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00000` | `65.0` (`79.0`) | `+0.0972` | `0.6833` | `0.6548` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `54.0` (`90.0`) | `+0.2500` | `0.2318` | `0.0788` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `60.0` (`60.0`) | `-0.1667` | `0.4935` | `0.1658` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00064` | `16.0` (`128.0`) | `+0.7778` | `0.0030` | `0.0014` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00143` | `21.0` (`123.0`) | `+0.7083` | `0.0040` | `0.0035` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `0.7133` | `0.7052` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `59.5` (`84.5`) | `+0.1736` | `0.3526` | `0.2860` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `1.0000` | `0.3593` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00050` | `67.0` (`67.0`) | `-0.0694` | `0.7882` | `0.7950` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00018` | `65.0` (`79.0`) | `+0.0972` | `0.7113` | `0.7075` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00136` | `26.0` (`118.0`) | `+0.6389` | `0.0080` | `0.0086` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [100.0%]` (`p95=100.0%`) | `3.0` | `1.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `45.0% [21.3%]` (`p95=62.2%`) | `9.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `22.5% [15.0%]` (`p95=39.5%`) | `4.5` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `25.0% [11.2%]` (`p95=46.2%`) | `5.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `50.0% [100.0%]` (`p95=100.0%`) | `0.5` | `0.50 [1.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `7.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=3.3%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [100.0%]` (`p95=100.0%`) | `3.0` | `1.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `9.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `4.5` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `5.0` | `1.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `30.0%` | `45.0%` / `1.00` | `30.0%` / `1.00` | `30.0%` / `1.00` |
| `240` | `10%` | `0.0116` | `11.194` | `40.0` | `415.5` | `4.27` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `600` | `25%` | `0.0000` | `7.050` | `50.5` | `530.0` | `4.67` | `4.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1200` | `50%` | `0.0000` | `2.603` | `56.5` | `599.5` | `5.01` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `75%` | `0.0000` | `1.816` | `56.5` | `594.5` | `5.06` | `5.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `2400` | `100%` | `0.0000` | `1.370` | `57.5` | `594.5` | `5.17` | `5.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.7492` | `0.7685 / 1.0000` | `0.0300 / 0.1001` | `3.6036 / 0.9738` | `0.0838` | `0.4314` |
| `FrozenPlasticity` | `12` | `0.7072` | `0.8470 / 1.0000` | `0.0002 / 0.0007` | `3.6955 / 0.9987` | `0.0006` | `0.4402` |
| `YokedMetabolism` | `12` | `0.7861` | `0.8549 / 1.0000` | `0.0279 / 0.1071` | `3.6041 / 0.9740` | `0.0856` | `0.4555` |
| `ActionShuffled` | `12` | `0.6226` | `0.8317 / 1.0000` | `0.0322 / 0.1149` | `3.6589 / 0.9888` | `0.0382` | `0.4368` |
| `UnlesionedPostLife` | `12` | `0.8903` | `0.7685 / 1.0000` | `0.0638 / 0.1531` | `3.4604 / 0.9351` | `0.2126` | `0.4501` |
| `FlaglessPostLifeLesion` | `12` | `0.6870` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6774 / 0.9938` | `0.0027` | `0.2262` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `309.0 [1070.0]` | `354.944 [103.857]` | `0.950 [0.001]` | `2.0 [2.0]` | `5.5` | `177.56` | `0.0` |
| `FrozenPlasticity` | `1421.5 [17.8]` | `468.393 [90.905]` | `0.450 [0.000]` | `2.0 [0.2]` | `4.0` | `179.63` | `0.0` |
| `YokedMetabolism` | `255.5 [1184.5]` | `291.543 [106.203]` | `0.950 [0.001]` | `1.5 [1.0]` | `3.5` | `177.94` | `0.5` |
| `ActionShuffled` | `1181.0 [72.0]` | `386.399 [176.153]` | `0.950 [0.000]` | `2.0 [2.0]` | `4.5` | `153.23` | `1.0` |
| `UnlesionedPostLife` | `33.5 [298.5]` | `16.677 [76.097]` | `0.950 [0.001]` | `1.0 [0.0]` | `0.0` | `37.65` | `0.0` |
| `FlaglessPostLifeLesion` | `399.5 [27.8]` | `189.072 [86.445]` | `0.000 [0.000]` | `1.0 [0.0]` | `2.5` | `52.53` | `0.5` |

### Cross-Habitat Modality Comparison (`02b PermutedMedium<PitchBlackEchoCave> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` vs `02a PermutedMedium<BrightRadiantWorld> (128 Scrambled Sensor + 6 Scrambled Motor Wires)` — `Natural` Cohort)

| Cross-World Observable | `median(Self) - median(Peer)` | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) |
| :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `-480.00000` | `-0.1528` | `0.5534` |
| `echo_energy_received` | `-264.13051` | `-0.6806` | `0.0050` |
| `photonic_band_conductance_share` | `-0.81347` | `-1.0000` | `0.0010` |
| `sonar_echo_band_conductance_share` | `+0.81347` | `+0.2361` | `0.3497` |
