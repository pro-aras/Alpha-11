# Study 5: Multi-Generational Cultural Ratchet in `MultiGenTerrarium` (`N = 12` Seeds)

- **Elder Pre-Habitation (`1,500` ticks, `2` `Natural` Elders)**: Mark Coverage `p50 = 0.1484 [0.0527]`, Corridor Alignment `p50 = 0.4407 [0.1037]`
- **Newborn Evaluation Horizon**: `1,500` ticks (`32`-cell initial newborn seed matched across all 4 arms)

## 1. Newborn Metabolic, Thermal Stress & Cultural Niche Telemetry

| Newborn Cohort Arm | Influx/Tick (`p50 [IQR]`) | 1st->2nd Half Influx | Stress/Tick (`p50 [IQR]`) | Night Stress/Tick (`lux <= 0.10`, `p50 [IQR]`) | Final ATP (`p50`) | Cells (`p50`) | Agency (`p50`) | Noisy-TV Share (`p50`) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `CohabitingNewborn` | `0.0048 [0.0036]` | `0.0026 -> 0.0062` | `0.0177 [0.0015]` | `0.0360 [0.0038]` | `2.728` | `61.5` | `0.0022` | `0.0%` |
| `SoloOrphan` | `0.0051 [0.0024]` | `0.0036 -> 0.0065` | `0.0194 [0.0015]` | `0.0362 [0.0021]` | `2.622` | `62.5` | `0.0000` | `0.0%` |
| `PhaseMatchedSoloOrphan` | `0.0052 [0.0018]` | `0.0035 -> 0.0064` | `0.0185 [0.0048]` | `0.0382 [0.0126]` | `2.533` | `64.0` | `0.0033` | `0.0%` |
| `CohabitingFrozenNewborn` | `0.0001 [0.0008]` | `0.0000 -> 0.0000` | `0.0114 [0.0065]` | `0.0252 [0.0143]` | `9.282` | `32.0` | `0.0000` | `0.0%` |

## 2. Persistent Surface Marks & Elder-Newborn Synchrony

| Newborn Cohort Arm | Mark Coverage (`p50 [IQR]`) | Total Mark Mass (`p50 [IQR]`) | Corridor Alignment (`p50 [IQR]`) | Elder-Newborn Acoustic Sync (`p50 [IQR]`) | Elder-Newborn Motor Sync (`p50 [IQR]`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CohabitingNewborn` | `0.2793 [0.0205]` | `11.602 [1.340]` | `0.4313 [0.0309]` | `0.5324 [0.2231]` | `0.2304 [0.4881]` |
| `SoloOrphan` | `0.0625 [0.0400]` | `2.787 [1.086]` | `0.4761 [0.0674]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` |
| `PhaseMatchedSoloOrphan` | `0.0762 [0.0283]` | `3.371 [1.729]` | `0.4362 [0.1334]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` |
| `CohabitingFrozenNewborn` | `0.1973 [0.0527]` | `7.837 [1.498]` | `0.4189 [0.0532]` | `0.5413 [0.1507]` | `0.0304 [0.3429]` |

## 3. Non-Parametric Mann-Whitney U Comparisons

| Metric | Arm A | Arm B | `median(A) - median(B)` | Rank-Biserial `r_rb` | `p_perm` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `SoloOrphan` | `-0.00036` | `+0.0417` | `0.8881` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `SoloOrphan` | `-0.00171` | `-0.4444` | `0.0849` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `SoloOrphan` | `-0.00024` | `+0.0833` | `0.7473` |
| `newborn_final_atp` | `CohabitingNewborn` | `SoloOrphan` | `+0.10531` | `+0.1944` | `0.4476` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `SoloOrphan` | `-0.04485` | `-0.4722` | `0.0579` |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00043` | `-0.0417` | `0.8881` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00074` | `-0.1250` | `0.6274` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00217` | `-0.1528` | `0.5534` |
| `newborn_final_atp` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `+0.19439` | `+0.1806` | `0.4845` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `PhaseMatchedSoloOrphan` | `-0.00495` | `-0.0556` | `0.8322` |
| `newborn_influx_per_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.00467` | `+1.0000` | `0.0010` |
| `newborn_stress_per_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.00636` | `+0.6806` | `0.0050` |
| `newborn_night_stress_per_night_tick` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.01079` | `+0.5833` | `0.0220` |
| `newborn_final_atp` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `-6.55391` | `-1.0000` | `0.0010` |
| `mark_corridor_alignment` | `CohabitingNewborn` | `CohabitingFrozenNewborn` | `+0.01235` | `+0.1667` | `0.5145` |
