# Study 5: Multi-Generational Cultural Ratchet in `MultiGenTerrarium` (`N = 12` Seeds)

- **Elder Pre-Habitation (`1,500` ticks, `2` `Natural` Elders)**: Mark Coverage `p50 = 0.1250 [0.0742]`, Corridor Alignment `p50 = 0.5415 [0.0936]`
- **Newborn Evaluation Horizon**: `1,500` ticks (`32`-cell initial newborn seed matched across all 4 arms)

## 1. Newborn Metabolic, Thermal Stress & Cultural Niche Telemetry

| Newborn Cohort Arm | Influx/Tick (`p50 [IQR]`) | 1st->2nd Half Influx | Stress/Tick (`p50 [IQR]`) | Night Stress/Tick (`lux <= 0.10`, `p50 [IQR]`) | Final ATP (`p50`) | Cells (`p50`) | Agency (`p50`) | Noisy-TV Share (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `CohabitingNewborn` | `0.0059 [0.0030]` | `0.0045 -> 0.0056` | `0.0172 [0.0044]` | `0.0358 [0.0097]` | `3.000` | `65.0` | `0.0019` | `0.0%` |
| `SoloOrphan` | `0.0050 [0.0028]` | `0.0024 -> 0.0074` | `0.0161 [0.0034]` | `0.0307 [0.0063]` | `2.672` | `62.5` | `0.0004` | `0.0%` |
| `PhaseMatchedSoloOrphan` | `0.0058 [0.0030]` | `0.0053 -> 0.0055` | `0.0193 [0.0031]` | `0.0415 [0.0069]` | `2.708` | `66.5` | `0.0042` | `0.0%` |
| `CohabitingFrozenNewborn` | `0.0001 [0.0008]` | `0.0000 -> 0.0001` | `0.0118 [0.0090]` | `0.0251 [0.0218]` | `9.662` | `32.0` | `0.0000` | `0.0%` |

## 2. Persistent Surface Marks & Elder-Newborn Synchrony

| Newborn Cohort Arm | Mark Coverage (`p50 [IQR]`) | Total Mark Mass (`p50 [IQR]`) | Corridor Alignment (`p50 [IQR]`) | Elder-Newborn Acoustic Sync (`p50 [IQR]`) | Elder-Newborn Motor Sync (`p50 [IQR]`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CohabitingNewborn` | `0.2578 [0.0635]` | `11.668 [1.610]` | `0.4808 [0.0809]` | `0.4154 [0.3140]` | `0.0378 [0.3106]` |
| `SoloOrphan` | `0.0312 [0.0322]` | `1.914 [0.974]` | `0.5370 [0.1500]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` |
| `PhaseMatchedSoloOrphan` | `0.0625 [0.0303]` | `2.792 [1.191]` | `0.4478 [0.1173]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` |
| `CohabitingFrozenNewborn` | `0.1797 [0.0547]` | `8.173 [2.552]` | `0.4856 [0.0555]` | `0.5275 [0.2440]` | `0.0714 [0.3892]` |

## 3. Non-Parametric Mann-Whitney U Comparisons

| Metric | Arm A | Arm B | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `SoloOrphan` | `+0.00095` | `+0.1528` | `0.5534` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `SoloOrphan` | `+0.00104` | `+0.2639` | `0.3177` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `SoloOrphan` | `+0.00503` | `+0.5556` | `0.0280` |
| `newborn_final_atp` | `CohabitingNewborn` | `SoloOrphan` | `+0.32768` | `+0.2083` | `0.4236` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `SoloOrphan` | `-0.05615` | `-0.1250` | `0.6274` |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `+0.00012` | `-0.0139` | `0.9810` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00211` | `-0.3333` | `0.1998` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00570` | `-0.3333` | `0.1998` |
| `newborn_final_atp` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `+0.29208` | `+0.3194` | `0.2268` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `+0.03306` | `+0.3194` | `0.2268` |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.00581` | `+1.0000` | `0.0010` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.00538` | `+0.6528` | `0.0060` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.01064` | `+0.5833` | `0.0220` |
| `newborn_final_atp` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `-6.66167` | `-1.0000` | `0.0010` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `-0.00472` | `+0.0694` | `0.7882` |
