# Observational Twin Study: 04a HarmonicWaveStream (96ch: 1D Traveling Waves + 3..8 Tick Delayed Resonance + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `1800` ticks | **Post-Life Lesion Window (`T_eval`)**: `450` ticks
- **Medium Boundary**: `96` unlabelled ambient channels, `4` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `33.33%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `1800` | `0.1910 [0.0238]` (`0.1729..0.2235`) | `0.0013 [0.0004]` | `0.1897 [0.0241]` | `0.1819 -> 0.2015` | `0.0723 [0.0537]` | `0.2028 [0.0418]` | `100.000` |
| `FrozenPlasticity` | `1800` | `0.0125 [0.0009]` (`0.0108..0.0137`) | `0.0063 [0.0005]` | `0.0062 [0.0016]` | `0.0134 -> 0.0114` | `0.0000 [0.0000]` | `0.0850 [0.0202]` | `31.489` |
| `YokedMetabolism` | `1800` | `0.0857 [0.0138]` (`0.0454..0.1040`) | `0.0389 [0.0094]` | `0.0467 [0.0209]` | `0.0878 -> 0.0918` | `0.0545 [0.0471]` | `0.2061 [0.0522]` | `100.000` |
| `ActionShuffled` | `1800` | `0.0714 [0.0036]` (`0.0672..0.0752`) | `0.0347 [0.0017]` | `0.0375 [0.0038]` | `0.0733 -> 0.0693` | `0.0004 [0.0036]` | `0.0950 [0.0095]` | `99.966` |
| `UnlesionedPostLife` | `450` | `0.1790 [0.0534]` (`0.1646..0.2417`) | `0.0015 [0.0023]` | `0.1775 [0.0560]` | `0.1828 -> 0.1777` | `0.0121 [0.0618]` | `0.2352 [0.0400]` | `100.000` |
| `FlaglessPostLifeLesion` | `450` | `0.0055 [0.0005]` (`0.0048..0.0063`) | `0.0027 [0.0001]` | `0.0028 [0.0007]` | `0.0059 -> 0.0052` | `0.0000 [0.0000]` | `0.0115 [0.0010]` | `99.996` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `105.0 [0.0]` (`p95=105.0`) | `1769.5 [84.0]` | `941.5` | `463.5` | `167.5` | `31.96 / 311.55 / 8.0` | `73.0 / 701.5 / 83.0 / 24.0` | `1796.0 / 222.0 / 63.5` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `1690.0 / 110.0 / 86.0` |
| `YokedMetabolism` | `105.0 [0.0]` (`p95=105.0`) | `1901.5 [112.8]` | `1032.0` | `501.5` | `166.0` | `31.37 / 321.52 / 8.0` | `73.0 / 1079.5 / 83.0 / 23.0` | `1795.5 / 337.0 / 62.5` |
| `ActionShuffled` | `98.0 [4.0]` (`p95=102.4`) | `1164.5 [56.8]` | `112.5` | `202.5` | `46.0` | `5.32 / 8.74 / 4.0` | `73.0 / 1.0 / 50.0 / 127.5` | `1791.0 / 130.5 / 63.0` |
| `UnlesionedPostLife` | `105.0 [0.0]` (`p95=105.0`) | `1769.5 [84.0]` | `941.5` | `463.5` | `167.5` | `31.96 / 311.55 / 8.0` | `73.0 / 701.5 / 83.0 / 24.0` | `2246.0 / 227.0 / 81.5` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `418.0 / 32.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.17845` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.18348` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.19009` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.07234` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.10533` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.14297` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.10973` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.01785` | `67.0` (`67.0`) | `-0.0694` | `0.7882` | `0.7950` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `66.0` (`78.0`) | `+0.0833` | `1.0000` | `0.3593` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.11957` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.15216` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.13219` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.07190` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `48.5` (`48.5`) | `-0.3264` | `0.1099` | `0.0809` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.17356` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.17471` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `-0.00708` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `80.6% [6.2%]` (`p95=89.5%`) | `48.0` | `1.00 [0.00]` |
| `Natural` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `19.4% [6.3%]` (`p95=23.2%`) | `16.0` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.2%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `79.1% [5.5%]` (`p95=90.0%`) | `48.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `20.9% [5.5%]` (`p95=23.3%`) | `16.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `100.0% [0.3%]` (`p95=100.0%`) | `48.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.3%]` (`p95=0.5%`) | `0.0` | `0.00 [1.00]` |
| `UnlesionedPostLife` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `80.6% [6.2%]` (`p95=89.5%`) | `48.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `19.4% [6.3%]` (`p95=23.2%`) | `16.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.2%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Traveling Wave (0..48)` Share / `\Delta` | `Delayed Resonance (48..64)` Share / `\Delta` | `Noisy TV Static (64..96)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `35.0%` | `45.0%` / `1.00` | `15.0%` / `1.00` | `35.0%` / `1.00` |
| `180` | `10%` | `0.1314` | `32.661` | `38.0` | `401.0` | `4.20` | `4.0` | `0.0516` | `2.2%` | `77.3%` / `1.00` | `20.8%` / `1.00` | `2.2%` / `1.00` |
| `450` | `25%` | `0.1962` | `79.600` | `49.0` | `628.0` | `5.03` | `5.0` | `0.1044` | `0.0%` | `86.0%` / `1.00` | `13.8%` / `1.00` | `0.0%` / `0.00` |
| `900` | `50%` | `0.1950` | `100.000` | `68.0` | `1003.0` | `9.51` | `7.0` | `0.0860` | `0.0%` | `83.8%` / `1.00` | `16.2%` / `1.00` | `0.0%` / `0.00` |
| `1350` | `75%` | `0.2047` | `100.000` | `87.0` | `1420.0` | `20.72` | `8.0` | `0.0851` | `0.0%` | `78.5%` / `1.00` | `21.5%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `100%` | `0.1950` | `100.000` | `105.0` | `1769.5` | `31.96` | `8.0` | `0.0723` | `0.0%` | `80.6%` / `1.00` | `19.4%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.0797` | `0.0170 / 0.0450` | `0.0286 / 0.1284` | `3.0049 / 0.9479` | `0.0008` | `0.1484` |
| `FrozenPlasticity` | `12` | `0.0656` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.1672 / 0.9991` | `0.0013` | `0.2343` |
| `YokedMetabolism` | `12` | `0.2044` | `0.0153 / 0.0498` | `0.0221 / 0.0566` | `3.0048 / 0.9479` | `0.0019` | `0.1703` |
| `ActionShuffled` | `12` | `0.0335` | `0.0003 / 0.0010` | `0.0269 / 0.1177` | `3.0110 / 0.9499` | `0.0007` | `0.1337` |
| `UnlesionedPostLife` | `12` | `0.1563` | `0.0170 / 0.0450` | `0.1645 / 0.6068` | `2.9969 / 0.9454` | `0.0029` | `0.1974` |
| `FlaglessPostLifeLesion` | `12` | `0.0752` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.1004 / 0.9781` | `0.0029` | `0.1399` |

## 7. `HarmonicWaveStream` Phase-Lock & Delayed-Resonance Metrics

| Control Arm | In-Phase Locks (`p50 [IQR]`) | Anti-Phase Collisions (`p50 [IQR]`) | Delayed Echo Energy (`p50 [IQR]`) |
| :--- | :--- | :--- | :--- |
| `Natural` | `1617.0 [70.0]` | `73.0 [26.0]` | `6658.180 [145.428]` |
| `FrozenPlasticity` | `749.0 [56.8]` | `704.5 [53.5]` | `218.135 [7.561]` |
| `YokedMetabolism` | `859.5 [102.5]` | `760.0 [126.2]` | `6568.578 [129.113]` |
| `ActionShuffled` | `816.5 [29.8]` | `766.0 [39.2]` | `5432.317 [142.157]` |
| `UnlesionedPostLife` | `393.0 [59.8]` | `16.0 [25.0]` | `1807.075 [73.570]` |
| `FlaglessPostLifeLesion` | `176.5 [22.2]` | `164.0 [12.0]` | `8.751 [0.374]` |

- **`in_phase_locks` `Natural` vs `FrozenPlasticity`**: `diff = +868.00`, `r_rb = +1.0000`, `p_perm = 0.0010`
- **`in_phase_locks` `Natural` vs `ActionShuffled`**: `diff = +800.50`, `r_rb = +1.0000`, `p_perm = 0.0010`

---

# Identical-Genome Constrained-Randomness Divergence (`genome_seed=777`)

- **Twins (`N`)**: `12` | **World Seed**: `2153408684502287112` | **Horizon**: `2000` ticks | **Initial Jitter `epsilon`**: `1.00e-4`
- **Influx / Tick (`p50 [IQR]`, `min..max`)**: `0.0030 [0.0001]` (`0.0029..0.0050`)
- **Outcome `CV`**: `0.1940`
- **Mean Pairwise Pore Cosine Distance**: `0.1379` (`max=0.3239`)
- **Mean Pairwise `tau` JSD**: `0.0284` bits (`max=0.0806`)
- **Mean Intra-Individual Trajectory Entropy**: `3.6090` bits (`norm=0.9753`)
- **Inter-Individual Trajectory JSD**: `0.0087` bits
- **Composite Individuality Index**: `0.2044`

### Zero-Perturbation Determinism Control (`epsilon = 0.0`, `N = 12` Identical Twins)
- **Outcome `CV`**: `0.000000` | **Pairwise Pore Cosine Dist**: `0.000000` | **Pairwise `tau` JSD**: `0.000000` | **Inter-Twin Trajectory JSD**: `0.000000` | **Composite Individuality**: `0.122219`
