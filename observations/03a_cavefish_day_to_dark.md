# Observational Twin Study: 03a Cavefish Day-to-Darkness Transition (Illumination 1.0 -> 0.0 at Tick 1500)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `3000` ticks | **Post-Life Lesion Window (`T_eval`)**: `600` ticks
- **Medium Boundary**: `128` unlabelled ambient channels, `6` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `25.00%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `3000` | `0.0027 [0.0018]` (`0.0018..0.0051`) | `0.0006 [0.0006]` | `0.0021 [0.0016]` | `0.0052 -> 0.0000` | `0.0000 [0.0000]` | `0.0129 [0.0020]` | `0.000` |
| `FrozenPlasticity` | `3000` | `0.0007 [0.0004]` (`0.0002..0.0013`) | `0.0006 [0.0008]` | `0.0001 [0.0007]` | `0.0013 -> 0.0000` | `0.0000 [0.0000]` | `0.0117 [0.0010]` | `9.552` |
| `YokedMetabolism` | `3000` | `0.0012 [0.0018]` (`0.0003..0.0042`) | `0.0047 [0.0031]` | `-0.0028 [0.0032]` | `0.0016 -> 0.0000` | `0.0000 [0.0000]` | `0.0139 [0.0030]` | `0.000` |
| `ActionShuffled` | `3000` | `0.0010 [0.0009]` (`0.0003..0.0021`) | `0.0005 [0.0011]` | `0.0002 [0.0010]` | `0.0019 -> 0.0000` | `0.0000 [0.0000]` | `0.0132 [0.0032]` | `0.000` |
| `UnlesionedPostLife` | `600` | `0.0014 [0.0038]` (`0.0005..0.0085`) | `0.0002 [0.0033]` | `0.0012 [0.0065]` | `0.0024 -> 0.0000` | `0.0000 [0.0000]` | `0.0141 [0.0025]` | `0.035` |
| `FlaglessPostLifeLesion` | `600` | `0.0026 [0.0046]` (`0.0010..0.0093`) | `0.0010 [0.0037]` | `0.0019 [0.0057]` | `0.0050 -> 0.0001` | `0.0000 [0.0000]` | `0.0121 [0.0008]` | `0.894` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `68.5 [9.2]` (`p95=83.9`) | `699.5 [117.8]` | `248.0` | `92.5` | `74.5` | `5.42 / 18.27 / 5.0` | `36.5 / 1.0 / 3.5 / 239.0` | `2696.5 / 486.0 / 144.5` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `2826.0 / 174.0 / 146.0` |
| `YokedMetabolism` | `68.5 [9.0]` (`p95=84.0`) | `717.5 [108.0]` | `175.0` | `55.0` | `74.0` | `5.09 / 11.54 / 4.0` | `36.5 / 0.0 / 3.0 / 270.0` | `2667.0 / 434.0 / 147.0` |
| `ActionShuffled` | `57.5 [5.0]` (`p95=62.9`) | `559.5 [52.2]` | `155.5` | `69.5` | `51.0` | `4.84 / 8.78 / 4.0` | `25.5 / 0.0 / 1.0 / 270.0` | `2704.0 / 408.5 / 147.0` |
| `UnlesionedPostLife` | `68.5 [9.2]` (`p95=83.9`) | `699.5 [117.8]` | `248.0` | `92.5` | `74.5` | `5.42 / 18.27 / 5.0` | `36.5 / 1.0 / 3.5 / 239.0` | `3214.0 / 540.0 / 168.5` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `575.0 / 25.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00201` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00202` | `2.0` (`142.0`) | `+0.9722` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.00000` | `68.0` (`76.0`) | `+0.0556` | `0.8442` | `0.8163` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.00000` | `54.0` (`90.0`) | `+0.2500` | `0.2318` | `0.0788` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1662` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.00148` | `27.0` (`117.0`) | `+0.6250` | `0.0110` | `0.0102` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.00485` | `11.0` (`133.0`) | `+0.8472` | `0.0010` | `0.0005` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `-0.00000` | `59.0` (`59.0`) | `-0.1806` | `0.4605` | `0.4315` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.00000` | `61.5` (`82.5`) | `+0.1458` | `0.6144` | `0.3740` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `70.0` (`70.0`) | `-0.0278` | `0.7552` | `0.8939` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00172` | `3.0` (`141.0`) | `+0.9583` | `0.0010` | `0.0001` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.00190` | `4.0` (`140.0`) | `+0.9444` | `0.0010` | `0.0001` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.00000` | `52.0` (`92.0`) | `+0.2778` | `0.1728` | `0.1610` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.00000` | `61.5` (`82.5`) | `+0.1458` | `0.6144` | `0.3740` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `60.0` (`84.0`) | `+0.1667` | `0.4935` | `0.1662` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00126` | `51.0` (`51.0`) | `-0.2917` | `0.2657` | `0.2366` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `-0.00076` | `70.0` (`70.0`) | `-0.0278` | `0.9271` | `0.9310` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `+0.00197` | `2.0` (`142.0`) | `+0.9722` | `0.0010` | `0.0001` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `Natural` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `96.2% [100.0%]` (`p95=100.0%`) | `16.5` | `1.00 [1.00]` |
| `Natural` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=9.5%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [100.0%]` (`p95=100.0%`) | `6.0` | `1.00 [1.00]` |
| `YokedMetabolism` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=100.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `12.5` | `1.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Photonic 2D (0..64)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Wave-Echo Sonar (64..96)` | `active_echo` | `25.0%` | `96.2% [100.0%]` (`p95=100.0%`) | `16.5` | `1.00 [1.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (96..128)` | `noisy_tv_distractor` | `25.0%` | `0.0% [0.0%]` (`p95=9.5%`) | `0.0` | `0.00 [0.00]` |
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
| `2250` | `75%` | `0.0000` | `0.354` | `68.5` | `740.0` | `5.42` | `5.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `100.0%` / `1.00` | `0.0%` / `0.00` |
| `3000` | `100%` | `0.0000` | `0.000` | `68.5` | `699.5` | `5.42` | `5.0` | `0.0000` | `0.0%` | `0.0%` / `0.00` | `96.2%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.3729` | `0.6618 / 1.0000` | `0.0402 / 0.1011` | `3.5987 / 0.9725` | `0.0745` | `0.3676` |
| `FrozenPlasticity` | `12` | `0.4710` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.6971 / 0.9991` | `0.0005` | `0.2989` |
| `YokedMetabolism` | `12` | `0.8008` | `0.7968 / 1.0000` | `0.0333 / 0.0901` | `3.5886 / 0.9698` | `0.0736` | `0.4425` |
| `ActionShuffled` | `12` | `0.6096` | `0.6152 / 1.0000` | `0.0556 / 0.2148` | `3.6194 / 0.9781` | `0.0585` | `0.3867` |
| `UnlesionedPostLife` | `12` | `0.9529` | `0.6618 / 1.0000` | `0.0762 / 0.1596` | `3.2268 / 0.8720` | `0.2509` | `0.4242` |
| `FlaglessPostLifeLesion` | `12` | `0.7048` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.6722 / 0.9924` | `0.0028` | `0.2275` |

## 7. External World `CaveMetrics` & Active Sonar Emission Telemetry

| Control Arm | Active Pings (`p50 [IQR]`) | Echo Energy (`p50 [IQR]`) | Emit-Pulse Motor `g` (`p50 [IQR]`) | Springs Depleted (`p50 [IQR]`) | Wall Impacts (`p50`) | Distance (`p50`) | Valve Activations (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Natural` | `812.0 [596.2]` | `897.984 [870.906]` | `0.950 [0.004]` | `2.5 [1.8]` | `5.0` | `359.39` | `0.0` |
| `FrozenPlasticity` | `1799.5 [26.8]` | `556.495 [198.578]` | `0.450 [0.000]` | `2.0 [0.5]` | `7.5` | `221.58` | `0.0` |
| `YokedMetabolism` | `997.5 [551.2]` | `513.053 [759.356]` | `0.950 [0.008]` | `1.0 [1.2]` | `103.0` | `278.51` | `0.0` |
| `ActionShuffled` | `1101.0 [237.8]` | `927.295 [718.623]` | `0.950 [0.001]` | `2.0 [1.0]` | `5.5` | `218.79` | `1.0` |
| `UnlesionedPostLife` | `37.0 [93.0]` | `6.613 [18.295]` | `0.950 [0.004]` | `1.0 [0.2]` | `0.5` | `30.54` | `0.0` |
| `FlaglessPostLifeLesion` | `432.0 [49.5]` | `273.450 [95.251]` | `0.000 [0.000]` | `1.0 [0.0]` | `2.0` | `65.30` | `0.0` |

## 8. Pre-Darkness (`Ticks 1..=1500`) vs Post-Darkness (`Ticks 1501..=3000`) Shift Telemetry (`Natural`)

| Phase Metric | Day Half (`1..=1500`, `p50 [IQR]`) | Dark Half (`1501..=3000`, `p50 [IQR]`) | `median(Dark) - median(Day)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `active_pings_emitted` | `539.5 [236.5]` | `305.5 [253.5]` | `-234.00` | `-0.5139` | `0.0410` |
| `echo_energy_received` | `702.101 [969.669]` | `151.282 [219.175]` | `-550.819` | `-0.5694` | `0.0240` |
| `springs_depleted` | `2.5 [1.2]` | `0.0 [0.0]` | `-2.50` | `-0.9375` | `0.0010` |
