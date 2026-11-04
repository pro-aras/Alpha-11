# Observational Twin Study: 03b Mid-Life Causal Reversal (Motor + Thermal Vent Polarity Inverted at Tick 1500)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `3000` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `3000` | `0.0032 [0.0013]` (`0.0025..0.0063`) | `0.0010 [0.0011]` | `0.0023 [0.0022]` | `0.0052 -> 0.0015` | `0.0075 [0.0193]` | `0.0359 [0.0660]` | `0.442` |
| `FrozenPlasticity` | `3000` | `0.0007 [0.0004]` (`0.0002..0.0031`) | `0.0006 [0.0008]` | `0.0002 [0.0009]` | `0.0013 -> 0.0000` | `0.0000 [0.0000]` | `0.0266 [0.0098]` | `9.796` |
| `YokedMetabolism` | `3000` | `0.0012 [0.0027]` (`0.0004..0.0042`) | `0.0081 [0.0080]` | `-0.0048 [0.0071]` | `0.0016 -> 0.0004` | `0.0004 [0.0036]` | `0.0350 [0.0267]` | `0.776` |
| `ActionShuffled` | `3000` | `0.0016 [0.0009]` (`0.0003..0.0025`) | `0.0009 [0.0019]` | `0.0003 [0.0011]` | `0.0019 -> 0.0002` | `0.0000 [0.0000]` | `0.0405 [0.0200]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0002 [0.0009]` (`0.0000..0.0064`) | `0.0023 [0.0161]` | `-0.0010 [0.0159]` | `0.0005 -> 0.0000` | `0.0015 [0.0020]` | `0.0368 [0.0516]` | `0.016` |
| `FlaglessPostLifeLesion` | `600` | `0.0003 [0.0052]` (`0.0000..0.0151`) | `0.0030 [0.0034]` | `0.0000 [0.0030]` | `0.0000 -> 0.0000` | `0.0000 [0.0000]` | `0.0121 [0.0007]` | `0.995` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `68.5 [7.5]` (`p95=85.4`) | `778.5 [98.2]` | `277.5` | `154.0` | `78.5` | `5.33 / 17.65 / 5.0` | `37.0 / 13.0 / 3.5 / 149.5` | `2741.5 / 531.0 / 135.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2825.0 / 175.0 / 147.0` |
| `YokedMetabolism` | `69.0 [7.5]` (`p95=84.9`) | `748.0 [112.8]` | `196.0` | `88.0` | `74.0` | `5.07 / 9.79 / 4.0` | `37.0 / 0.5 / 3.5 / 184.5` | `2675.0 / 415.0 / 137.0` |
| `ActionShuffled` | `58.5 [5.0]` (`p95=62.9`) | `616.5 [58.5]` | `175.0` | `83.5` | `55.0` | `4.86 / 11.10 / 4.0` | `26.5 / 0.5 / 1.5 / 200.0` | `2725.0 / 431.0 / 145.5` |
| `UnlesionedPostLife` | `68.5 [7.5]` (`p95=85.4`) | `778.5 [98.2]` | `277.5` | `154.0` | `78.5` | `5.33 / 17.65 / 5.0` | `37.0 / 13.0 / 3.5 / 149.5` | `3303.5 / 567.5 / 170.5` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `576.0 / 24.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00246` | `6.0` (`138.0`) | `+0.9167` | `0.0010` | `0.0002` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00212` | `12.0` (`132.0`) | `+0.8333` | `0.0010` | `0.0006` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00142` | `37.0` (`107.0`) | `+0.4861` | `0.0490` | `0.0454` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00746` | `18.0` (`126.0`) | `+0.7500` | `0.0010` | `0.0004` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00199` | `30.0` (`114.0`) | `+0.5833` | `0.0220` | `0.0166` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00713` | `10.0` (`134.0`) | `+0.8611` | `0.0010` | `0.0004` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00101` | `48.5` (`95.5`) | `+0.3264` | `0.2058` | `0.1841` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00711` | `52.0` (`92.0`) | `+0.2778` | `0.2787` | `0.2475` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00155` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00207` | `13.0` (`131.0`) | `+0.8194` | `0.0010` | `0.0007` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00126` | `49.0` (`95.0`) | `+0.3194` | `0.2248` | `0.1935` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00746` | `23.0` (`121.0`) | `+0.6806` | `0.0040` | `0.0023` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00009` | `63.0` (`63.0`) | `-0.1250` | `0.6044` | `0.6171` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00097` | `47.0` (`47.0`) | `-0.3472` | `0.1798` | `0.1571` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `-0.00039` | `63.0` (`63.0`) | `-0.1250` | `0.6104` | `0.6228` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `82.1% [19.2%]` (`p95=96.9%`) | `50.0` | `1.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `17.9% [19.2%]` (`p95=34.5%`) | `14.0` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `91.1% [15.6%]` (`p95=99.8%`) | `47.5` | `1.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `8.9% [15.6%]` (`p95=36.3%`) | `10.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.5%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `86.1% [7.8%]` (`p95=97.3%`) | `52.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `13.9% [7.8%]` (`p95=21.2%`) | `11.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `82.1% [19.2%]` (`p95=96.9%`) | `50.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `17.9% [19.2%]` (`p95=34.5%`) | `14.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Photonic 2D (0..64)` Share / `\Delta` | `Wave-Echo Sonar (64..96)` Share / `\Delta` | `Noisy TV Static (96..128)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `20.0%` | `55.0%` / `1.00` | `27.5%` / `1.00` | `20.0%` / `1.00` |
| `300` | `10%` | `0.0191` | `12.815` | `43.0` | `452.5` | `4.41` | `4.0` | `0.0000` | `0.0%` | `91.8%` / `1.00` | `8.2%` / `1.00` | `0.0%` / `0.00` |
| `750` | `25%` | `0.0019` | `5.462` | `62.0` | `681.0` | `4.97` | `4.0` | `0.0017` | `0.0%` | `82.5%` / `1.00` | `17.5%` / `1.00` | `0.0%` / `0.00` |
| `1500` | `50%` | `0.0001` | `1.849` | `68.5` | `775.5` | `5.25` | `5.0` | `0.0077` | `0.0%` | `74.8%` / `1.00` | `25.2%` / `1.00` | `0.0%` / `0.00` |
| `2250` | `75%` | `0.0000` | `0.974` | `69.0` | `779.5` | `5.35` | `5.0` | `0.0067` | `0.0%` | `69.5%` / `1.00` | `30.5%` / `1.00` | `0.0%` / `0.00` |
| `3000` | `100%` | `0.0010` | `0.442` | `68.5` | `778.5` | `5.33` | `5.0` | `0.0075` | `0.0%` | `82.1%` / `1.00` | `17.9%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.3114` | `0.3500 / 0.5869` | `0.0400 / 0.1028` | `3.6251 / 0.9796` | `0.0700` | `0.2817` |
| `FrozenPlasticity` | `12` | `0.8183` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6974 / 0.9992` | `0.0005` | `0.3314` |
| `YokedMetabolism` | `12` | `0.7758` | `0.3739 / 0.7113` | `0.0367 / 0.0974` | `3.5681 / 0.9642` | `0.1119` | `0.3363` |
| `ActionShuffled` | `12` | `0.4960` | `0.3035 / 0.5021` | `0.0543 / 0.2148` | `3.5884 / 0.9697` | `0.1013` | `0.2971` |
| `UnlesionedPostLife` | `12` | `1.7928` | `0.3500 / 0.5869` | `0.0760 / 0.1595` | `3.1948 / 0.8634` | `0.4561` | `0.3908` |
| `FlaglessPostLifeLesion` | `12` | `1.4251` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6704 / 0.9919` | `0.0025` | `0.2710` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `1346.5 [510.8]` | `1709.004 [1033.987]` | `0.950 [0.000]` | `3.0 [2.0]` | `10.0` | `579.11` | `1.0` |
| `FrozenPlasticity` | `1782.5 [40.0]` | `516.881 [88.831]` | `0.450 [0.000]` | `2.5 [1.2]` | `11.5` | `220.87` | `0.0` |
| `YokedMetabolism` | `1126.5 [913.2]` | `1260.647 [1170.619]` | `0.950 [0.000]` | `1.0 [1.2]` | `140.5` | `363.47` | `0.5` |
| `ActionShuffled` | `1221.5 [559.5]` | `1132.186 [2826.457]` | `0.950 [0.000]` | `2.0 [1.2]` | `9.5` | `298.66` | `1.0` |
| `UnlesionedPostLife` | `413.0 [392.0]` | `100.090 [638.416]` | `0.950 [0.000]` | `0.0 [0.0]` | `3.0` | `42.61` | `0.0` |
| `FlaglessPostLifeLesion` | `429.5 [18.8]` | `273.664 [55.074]` | `0.000 [0.000]` | `0.5 [1.0]` | `1.5` | `64.27` | `0.0` |

## 8. Mid-Life Causal Reversal Shock, Contradiction Splitting & Re-Acquisition (`Natural`, `N = 12`)

- **Contradiction Splits (`1..=1500` vs `1501..=3000`)**: `1.0 [3.8]` vs `7.5 [12.5]`
- **Mean Temperature (`1201..=1500` -> `1501..=1800`, equal 300-tick windows)**: `0.5086 [0.0129]` -> `0.5360 [0.1941]` (`r_rb = +0.5972`, `p_perm = 0.0180`)
- **Initial Learning Windows (`1..=300` -> `301..=600` Influx/Tick `p50`)**: `0.0190` -> `0.0010`
- **Post-Reversal Windows (`1501..=1800` -> `1801..=2100` -> `2401..=3000` Influx/Tick `p50`)**: `0.0001` -> `0.0000` -> `0.0007`

| Reversal Hypothesis Test | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | ---: | ---: | ---: |
| `Post-Reversal Half (1501..=3000): Natural vs FrozenPlasticity` | `+0.00142` | `+0.4861` | `0.0490` |
| `Post-Reversal Half (1501..=3000): Natural vs YokedMetabolism` | `+0.00101` | `+0.3264` | `0.2058` |
| `Post-Reversal Half (1501..=3000): Natural vs ActionShuffled` | `+0.00126` | `+0.3194` | `0.2248` |
| `Post-Reversal Late (2401..=3000) vs Shock (1501..=1800)` | `+0.00058` | `+0.0417` | `0.8791` |
| `Re-Acquisition (1501..=1800) vs Initial (1..=300)` | `-0.01895` | `-0.9861` | `0.0010` |
| `Re-Acquisition (1801..=2100) vs Initial (301..=600)` | `-0.00105` | `-0.6667` | `0.0030` |
