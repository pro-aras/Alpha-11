# Observational Twin Study: 04a HarmonicWaveStream (96ch: 1D Traveling Waves + 3..8 Tick Delayed Resonance + NoisyTV)

- **Seeds (`N`)**: `12` (`[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]`)
- **Primary Life Horizon (`T`)**: `1800` ticks | **Post-Life Lesion Window (`T_eval`)**: `450` ticks
- **Medium Boundary**: `96` unlabelled ambient channels, `4` continuous actuators
- **Uniform Noisy-TV Channel Expectation**: `33.33%` | **Flagless Lesion Preserved `ControlMode::Natural`**: `true`

## 1. Metabolic, Homeostatic & Causal Distributions Across All Twin Controls

| Control Arm | Ticks | Influx / Tick (`p50 [IQR]`, `min..max`) | Stress / Tick (`p50 [IQR]`) | Net Energy / Tick (`p50 [IQR]`) | 1st->2nd Half Influx / Tick (`p50`) | Agency (`p50 [IQR]`) | Pred Error (`p50 [IQR]`) | Final ATP (`p50`) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | ---: |
| `Natural` | `1800` | `0.1956 [0.0067]` (`0.1830..0.2046`) | `0.0010 [0.0004]` | `0.1944 [0.0070]` | `0.1909 -> 0.2002` | `0.0828 [0.0227]` | `0.1563 [0.0386]` | `100.000` |
| `FrozenPlasticity` | `1800` | `0.0126 [0.0011]` (`0.0107..0.0137`) | `0.0063 [0.0004]` | `0.0063 [0.0018]` | `0.0134 -> 0.0114` | `0.0000 [0.0000]` | `0.0901 [0.0206]` | `31.563` |
| `YokedMetabolism` | `1800` | `0.0715 [0.0126]` (`0.0417..0.0917`) | `0.0394 [0.0061]` | `0.0328 [0.0182]` | `0.0676 -> 0.0707` | `0.0703 [0.0232]` | `0.1648 [0.0321]` | `100.000` |
| `ActionShuffled` | `1800` | `0.0696 [0.0043]` (`0.0603..0.0770`) | `0.0340 [0.0048]` | `0.0354 [0.0039]` | `0.0724 -> 0.0691` | `0.0065 [0.0066]` | `0.1093 [0.0068]` | `99.230` |
| `UnlesionedPostLife` | `450` | `0.2229 [0.0276]` (`0.1703..0.2401`) | `0.0004 [0.0008]` | `0.2224 [0.0280]` | `0.2200 -> 0.2275` | `0.0207 [0.0433]` | `0.1835 [0.0343]` | `100.000` |
| `FlaglessPostLifeLesion` | `450` | `0.0055 [0.0005]` (`0.0048..0.0063`) | `0.0027 [0.0001]` | `0.0028 [0.0007]` | `0.0059 -> 0.0052` | `0.0000 [0.0000]` | `0.0120 [0.0009]` | `99.996` |

## 2. Unbounded Morphogenesis, Time-Constants & Motor Clutch Dynamics

| Control Arm | Cells (`p50 [IQR]`, `p95`) | Tubes (`p50 [IQR]`) | Stiff Highways (`p50`) | Inhibitory (`p50`) | Echo-Delay Tubes (`p50`) | Mean `tau` / Max `tau` / Octaves (`p50`) | Mitosis / Splits / Myelin / Atrophy (`p50`) | Pulses / Rehearsals / Rest (`p50`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `Natural` | `105.0 [0.0]` (`p95=105.0`) | `1763.5 [69.2]` | `955.0` | `458.5` | `245.0` | `51.98 / 738.97 / 9.0` | `73.0 / 683.0 / 81.0 / 22.5` | `1796.5 / 72.0 / 61.0` |
| `FrozenPlasticity` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `13.5` | `12.5` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `1690.5 / 109.5 / 86.0` |
| `YokedMetabolism` | `105.0 [0.0]` (`p95=105.0`) | `1832.5 [121.2]` | `998.5` | `453.5` | `238.0` | `59.17 / 730.66 / 9.0` | `73.0 / 882.5 / 81.0 / 22.5` | `1793.0 / 331.5 / 60.0` |
| `ActionShuffled` | `105.0 [1.0]` (`p95=105.0`) | `1274.5 [12.8]` | `205.5` | `260.0` | `116.5` | `7.44 / 49.85 / 6.0` | `73.0 / 27.5 / 48.5 / 41.0` | `1796.5 / 71.0 / 63.0` |
| `UnlesionedPostLife` | `105.0 [0.0]` (`p95=105.0`) | `1763.5 [69.2]` | `955.0` | `458.5` | `245.0` | `51.98 / 738.97 / 9.0` | `73.0 / 683.0 / 81.0 / 22.5` | `2246.5 / 75.0 / 80.0` |
| `FlaglessPostLifeLesion` | `32.0 [0.0]` (`p95=32.0`) | `319.0 [2.0]` | `0.0` | `0.0` | `0.0` | `3.91 / 8.00 / 4.0` | `0.0 / 0.0 / 0.0 / 0.0` | `418.0 / 32.0 / 21.0` |

## 3. Non-Parametric Two-Sample Control Comparisons (Mann-Whitney U & Rank-Biserial)

| Metric | Arm A | Arm B | `median(A) - median(B)` | `U` (`U_A`) | Rank-Biserial `r_rb` | `p_perm` (`1000` perms) | `p_norm` |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| `achieved_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.18298` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `FrozenPlasticity` | `+0.18810` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `FrozenPlasticity` | `+0.18881` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `FrozenPlasticity` | `+0.08284` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `noisy_tv_conductance_share` | `Natural` | `FrozenPlasticity` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `achieved_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.12404` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `YokedMetabolism` | `+0.16158` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `YokedMetabolism` | `+0.12946` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `YokedMetabolism` | `+0.01253` | `43.0` (`101.0`) | `+0.4028` | `0.1309` | `0.0999` |
| `noisy_tv_conductance_share` | `Natural` | `YokedMetabolism` | `+0.00000` | `72.0` (`72.0`) | `+0.0000` | `1.0000` | `1.0000` |
| `achieved_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.12596` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `net_achieved_energy_per_tick` | `Natural` | `ActionShuffled` | `+0.15902` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `second_half_influx_per_tick` | `Natural` | `ActionShuffled` | `+0.13104` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `mean_interventional_agency` | `Natural` | `ActionShuffled` | `+0.07632` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `noisy_tv_conductance_share` | `Natural` | `ActionShuffled` | `+0.00000` | `66.0` (`66.0`) | `-0.0833` | `1.0000` | `0.3593` |
| `post_life_achieved_influx_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.21744` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `post_life_net_energy_per_tick` | `UnlesionedPostLife` | `FlaglessPostLifeLesion` | `+0.21963` | `0.0` (`144.0`) | `+1.0000` | `0.0010` | `0.0000` |
| `lesion_vs_frozen_influx_per_tick` | `FlaglessPostLifeLesion` | `FrozenPlasticity` | `-0.00713` | `0.0` (`0.0`) | `-1.0000` | `0.0010` | `0.0000` |

## 4. Sense-Organ Band Conductance Allocation & Delay Tuning

| Control Arm | Band Label | Role | Uniform Share | Observed Conductance Share (`p50 [IQR]`, `p95`) | Pore Count (`p50`) | Weighted Delay `\Delta` (`p50 [IQR]`) |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| `Natural` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `78.7% [5.0%]` (`p95=86.1%`) | `48.0` | `1.00 [0.00]` |
| `Natural` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `21.3% [5.0%]` (`p95=23.4%`) | `16.0` | `1.00 [0.00]` |
| `Natural` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `20.0` | `1.00 [0.00]` |
| `FrozenPlasticity` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FrozenPlasticity` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `YokedMetabolism` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `79.0% [3.8%]` (`p95=86.7%`) | `48.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `21.0% [3.8%]` (`p95=24.6%`) | `16.0` | `1.00 [0.00]` |
| `YokedMetabolism` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `100.0% [0.0%]` (`p95=100.0%`) | `48.0` | `1.00 [0.00]` |
| `ActionShuffled` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `ActionShuffled` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.1%`) | `0.0` | `0.00 [0.00]` |
| `UnlesionedPostLife` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `78.7% [5.0%]` (`p95=86.1%`) | `48.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `21.3% [5.0%]` (`p95=23.4%`) | `16.0` | `1.00 [0.00]` |
| `UnlesionedPostLife` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Traveling Wave (0..48)` | `structured_physical` | `50.0%` | `0.0% [0.0%]` (`p95=0.0%`) | `20.0` | `1.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Delayed Resonance (48..64)` | `active_echo` | `16.7%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |
| `FlaglessPostLifeLesion` | `Noisy TV Static (64..96)` | `noisy_tv_distractor` | `33.3%` | `0.0% [0.0%]` (`p95=0.0%`) | `0.0` | `0.00 [0.00]` |

## 5. `Natural` Developmental Time-Series Trajectory (`p50` Across Seeds)

| Tick | Stage | Window Influx/Tick | ATP | Cells | Tubes | Mean `tau` | Octaves | Agency | Noisy-TV Share | `Traveling Wave (0..48)` Share / `\Delta` | `Delayed Resonance (48..64)` Share / `\Delta` | `Noisy TV Static (64..96)` Share / `\Delta` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| `1` | `0%` | `0.0000` | `11.999` | `32.0` | `319.0` | `3.91` | `4.0` | `0.0000` | `35.0%` | `45.0%` / `1.00` | `15.0%` / `1.00` | `35.0%` / `1.00` |
| `180` | `10%` | `0.1480` | `35.629` | `38.0` | `418.0` | `4.17` | `4.0` | `0.0592` | `1.9%` | `77.1%` / `1.00` | `21.3%` / `1.00` | `1.9%` / `1.00` |
| `450` | `25%` | `0.2003` | `84.345` | `49.0` | `628.5` | `5.37` | `5.0` | `0.1165` | `0.0%` | `87.0%` / `1.00` | `12.9%` / `1.00` | `0.0%` / `0.00` |
| `900` | `50%` | `0.2003` | `100.000` | `68.0` | `1010.0` | `10.92` | `7.0` | `0.0881` | `0.0%` | `83.1%` / `1.00` | `16.9%` / `1.00` | `0.0%` / `0.00` |
| `1350` | `75%` | `0.1978` | `100.000` | `87.0` | `1396.5` | `27.32` | `8.0` | `0.0932` | `0.0%` | `82.8%` / `1.00` | `17.2%` / `1.00` | `0.0%` / `0.00` |
| `1800` | `100%` | `0.1993` | `100.000` | `105.0` | `1763.5` | `51.98` | `9.0` | `0.0828` | `0.0%` | `78.7%` / `1.00` | `21.3%` / `1.00` | `0.0%` / `0.00` |

## 6. Emergent Individuality & Constrained-Randomness Divergence

| Control Arm | Cohort `N` | Outcome `CV` | Pairwise Pore Cosine Dist (`mean` / `max`) | Pairwise `tau` JSD (`mean` / `max` bits) | Intra-Trajectory Entropy (`bits` / `norm`) | Inter-Individual Trajectory JSD (`bits`) | Composite Individuality (`[0..1]`) |
| :--- | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| `Natural` | `12` | `0.0316` | `0.0150 / 0.0390` | `0.0220 / 0.1050` | `3.0041 / 0.9477` | `0.0009` | `0.1354` |
| `FrozenPlasticity` | `12` | `0.0695` | `0.3758 / 0.5000` | `0.0002 / 0.0007` | `3.1671 / 0.9991` | `0.0013` | `0.2352` |
| `YokedMetabolism` | `12` | `0.1918` | `0.0106 / 0.0497` | `0.0197 / 0.0421` | `3.0053 / 0.9481` | `0.0021` | `0.1664` |
| `ActionShuffled` | `12` | `0.0614` | `0.0000 / 0.0001` | `0.0322 / 0.0830` | `2.9989 / 0.9460` | `0.0074` | `0.1411` |
| `UnlesionedPostLife` | `12` | `0.1100` | `0.0150 / 0.0390` | `0.0912 / 0.4803` | `2.9977 / 0.9457` | `0.0020` | `0.1696` |
| `FlaglessPostLifeLesion` | `12` | `0.0752` | `0.0000 / 0.0000` | `0.0002 / 0.0007` | `3.1004 / 0.9781` | `0.0029` | `0.1399` |

## 7. `HarmonicWaveStream` Phase-Lock & Delayed-Resonance Metrics

| Control Arm | In-Phase Locks (`p50 [IQR]`) | Anti-Phase Collisions (`p50 [IQR]`) | Delayed Echo Energy (`p50 [IQR]`) |
| :--- | :--- | :--- | :--- |
| `Natural` | `1645.0 [60.2]` | `59.5 [22.0]` | `6490.347 [153.409]` |
| `FrozenPlasticity` | `748.5 [65.0]` | `699.0 [52.8]` | `218.105 [8.287]` |
| `YokedMetabolism` | `784.0 [64.8]` | `821.0 [72.5]` | `6130.977 [269.224]` |
| `ActionShuffled` | `812.0 [46.8]` | `771.0 [27.5]` | `5019.293 [728.950]` |
| `UnlesionedPostLife` | `439.5 [13.8]` | `4.5 [8.0]` | `1817.749 [62.471]` |
| `FlaglessPostLifeLesion` | `176.5 [22.2]` | `164.0 [12.0]` | `8.751 [0.374]` |

- **`in_phase_locks` `Natural` vs `FrozenPlasticity`**: `diff = +896.50`, `r_rb = +1.0000`, `p_perm = 0.0010`
- **`in_phase_locks` `Natural` vs `ActionShuffled`**: `diff = +833.00`, `r_rb = +1.0000`, `p_perm = 0.0010`

---

# Identical-Genome Constrained-Randomness Divergence (`genome_seed=777`)

- **Twins (`N`)**: `12` | **World Seed**: `2153408684502287112` | **Horizon**: `2000` ticks | **Initial Jitter `epsilon`**: `1.00e-4`
- **Influx / Tick (`p50 [IQR]`, `min..max`)**: `0.0037 [0.0027]` (`0.0032..0.0208`)
- **Outcome `CV`**: `0.8753`
- **Mean Pairwise Pore Cosine Distance**: `0.3468` (`max=0.6172`)
- **Mean Pairwise `tau` JSD**: `0.0400` bits (`max=0.0863`)
- **Mean Intra-Individual Trajectory Entropy**: `3.6414` bits (`norm=0.9840`)
- **Inter-Individual Trajectory JSD**: `0.0396` bits
- **Composite Individuality Index**: `0.3378`

### Zero-Perturbation Determinism Control (`epsilon = 0.0`, `N = 12` Identical Twins)
- **Outcome `CV`**: `0.000000` | **Pairwise Pore Cosine Dist**: `0.000000` | **Pairwise `tau` JSD**: `0.000000` | **Inter-Twin Trajectory JSD**: `0.000000` | **Composite Individuality**: `0.123234`
