# Observational Twin Study: 03a Cavefish Day-to-Darkness Transition (Illumination 1.0 -> 0.0 at Tick 1500)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `3000` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `3000` | `0.0030 [0.0017]` (`0.0010..0.0065`) | `0.0004 [0.0005]` | `0.0023 [0.0024]` | `0.0054 -> 0.0001` | `0.0000 [0.0000]` | `0.0138 [0.0029]` | `0.000` |
| `FrozenPlasticity` | `3000` | `0.0007 [0.0004]` (`0.0004..0.0016`) | `0.0006 [0.0014]` | `0.0002 [0.0020]` | `0.0013 -> 0.0000` | `0.0000 [0.0000]` | `0.0114 [0.0010]` | `9.820` |
| `YokedMetabolism` | `3000` | `0.0015 [0.0011]` (`0.0006..0.0039`) | `0.0010 [0.0011]` | `0.0006 [0.0024]` | `0.0029 -> 0.0003` | `0.0000 [0.0000]` | `0.0137 [0.0022]` | `0.000` |
| `ActionShuffled` | `3000` | `0.0005 [0.0004]` (`0.0002..0.0016`) | `0.0005 [0.0012]` | `0.0001 [0.0011]` | `0.0007 -> 0.0000` | `0.0000 [0.0000]` | `0.0131 [0.0022]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0031 [0.0026]` (`0.0003..0.0073`) | `0.0004 [0.0012]` | `0.0020 [0.0021]` | `0.0062 -> 0.0000` | `0.0000 [0.0000]` | `0.0134 [0.0015]` | `0.797` |
| `FlaglessPostLifeLesion` | `600` | `0.0026 [0.0046]` (`0.0010..0.0093`) | `0.0010 [0.0037]` | `0.0019 [0.0057]` | `0.0050 -> 0.0001` | `0.0000 [0.0000]` | `0.0115 [0.0007]` | `0.649` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `70.0 [11.2]` (`p95=89.0`) | `751.0 [147.0]` | `159.0` | `42.0` | `38.5` | `5.29 / 16.29 / 4.5` | `38.0 / 0.0 / 2.5 / 225.0` | `2641.5 / 388.0 / 147.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2826.5 / 173.5 / 145.5` |
| `YokedMetabolism` | `70.0 [12.0]` (`p95=89.2`) | `756.5 [172.2]` | `162.0` | `41.0` | `38.0` | `5.10 / 14.33 / 4.0` | `38.0 / 0.0 / 2.5 / 223.5` | `2635.5 / 381.0 / 147.0` |
| `ActionShuffled` | `55.0 [2.0]` (`p95=61.4`) | `564.5 [21.0]` | `94.0` | `24.0` | `23.0` | `4.74 / 11.98 / 4.0` | `23.0 / 0.0 / 0.0 / 199.0` | `2689.0 / 335.5 / 147.0` |
| `UnlesionedPostLife` | `70.0 [11.2]` (`p95=89.0`) | `751.0 [147.0]` | `159.0` | `42.0` | `38.5` | `5.29 / 16.29 / 4.5` | `38.0 / 0.0 / 2.5 / 225.0` | `3217.5 / 412.0 / 168.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.5 / 24.5 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00227` | `6.0` (`138.0`) | `+0.9167` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00201` | `14.0` (`130.0`) | `+0.8056` | `0.0010` | `0.0009` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00007` | `60.0` (`84.0`) | `+0.1667` | `0.5195` | `0.4953` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00151` | `37.0` (`107.0`) | `+0.4861` | `0.0500` | `0.0464` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00160` | `33.0` (`111.0`) | `+0.5417` | `0.0300` | `0.0262` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00017` | `60.5` (`60.5`) | `-0.1597` | `0.5125` | `0.5222` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `71.5` (`72.5`) | `+0.0069` | `1.0000` | `1.0000` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `71.5` (`71.5`) | `-0.0069` | `1.0000` | `1.0000` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00251` | `9.0` (`135.0`) | `+0.8750` | `0.0010` | `0.0003` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00212` | `11.0` (`133.0`) | `+0.8472` | `0.0010` | `0.0005` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00008` | `55.0` (`89.0`) | `+0.2361` | `0.3457` | `0.3278` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `71.5` (`71.5`) | `-0.0069` | `1.0000` | `1.0000` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `65.0` (`65.0`) | `-0.0972` | `0.4935` | `0.5136` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00052` | `70.0` (`70.0`) | `-0.0278` | `0.9271` | `0.9310` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.00005` | `68.0` (`76.0`) | `+0.0556` | `0.8322` | `0.8399` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00186` | `4.0` (`140.0`) | `+0.9444` | `0.0010` | `0.0001` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [100.0%]` (`p95=100.0%`) | `2.0` | `1.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=14.2%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [80.0%]` (`p95=100.0%`) | `2.0` | `1.00 [0.25]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=33.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [100.0%]` (`p95=100.0%`) | `0.0` | `0.00 [1.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [100.0%]` (`p95=100.0%`) | `2.0` | `1.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=14.2%`) | `0.0` | `0.00 [0.00]` |
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
| `2250` | `75%` | `0.0000` | `0.408` | `70.0` | `751.5` | `5.29` | `4.5` | `0.0000` | `0.0%` | `5.4%` / `1.00` | `94.6%` / `1.00` | `0.0%` / `0.00` |
| `3000` | `100%` | `0.0000` | `0.000` | `70.0` | `751.0` | `5.29` | `4.5` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.5449` | `0.8799 / 1.0000` | `0.0382 / 0.0927` | `3.6808 / 0.9947` | `0.0062` | `0.4423` |
| `FrozenPlasticity` | `12` | `0.4696` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6968 / 0.9990` | `0.0005` | `0.2988` |
| `YokedMetabolism` | `12` | `0.5980` | `0.8992 / 1.0000` | `0.0246 / 0.0714` | `3.6772 / 0.9937` | `0.0083` | `0.4490` |
| `ActionShuffled` | `12` | `0.7139` | `0.8348 / 1.0000` | `0.0656 / 0.1822` | `3.6789 / 0.9942` | `0.0147` | `0.4540` |
| `UnlesionedPostLife` | `12` | `0.6023` | `0.8799 / 1.0000` | `0.0652 / 0.1509` | `3.6759 / 0.9934` | `0.0118` | `0.4548` |
| `FlaglessPostLifeLesion` | `12` | `0.7371` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6724 / 0.9924` | `0.0030` | `0.2303` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1643.5 [177.0]` | `537.130 [392.113]` | `0.950 [0.000]` | `2.5 [1.2]` | `7.0` | `291.45` | `0.0` |
| `FrozenPlasticity` | `1798.5 [40.8]` | `535.348 [259.758]` | `0.450 [0.000]` | `2.0 [1.0]` | `6.5` | `221.50` | `0.0` |
| `YokedMetabolism` | `1561.0 [156.2]` | `526.013 [213.392]` | `0.950 [0.000]` | `2.0 [1.2]` | `21.0` | `288.21` | `0.5` |
| `ActionShuffled` | `1688.0 [317.8]` | `793.018 [178.949]` | `0.950 [0.000]` | `2.0 [1.2]` | `6.5` | `239.90` | `1.0` |
| `UnlesionedPostLife` | `396.5 [40.8]` | `185.952 [45.161]` | `0.950 [0.000]` | `1.0 [0.0]` | `3.0` | `53.36` | `0.0` |
| `FlaglessPostLifeLesion` | `432.5 [49.5]` | `274.197 [95.677]` | `0.000 [0.000]` | `1.0 [0.0]` | `2.0` | `65.30` | `0.0` |

## 8. Pre-Darkness (`Ticks 1..=1500`) vs Post-Darkness (`Ticks 1501..=3000`) Shift Telemetry (`Natural`)

| Phase Metric | Day Half (`1..=1500`, `p50 [IQR]`) | Dark Half (`1501..=3000`, `p50 [IQR]`) | `median(Dark) - median(Day)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `665.0 [105.8]` | `1017.0 [118.5]` | `+352.00` | `+0.7222` | `0.0040` |
| `echo_energy_received` | `204.749 [254.402]` | `328.599 [117.209]` | `+123.850` | `+0.2083` | `0.4236` |
| `springs_depleted` | `2.0 [1.2]` | `0.0 [1.0]` | `-2.00` | `-0.7361` | `0.0040` |
