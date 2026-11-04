# Study 6: Unforced Human-Activity Portals (`N = 12` Seeds, `1500` Ticks, Zero Game-Score Rewards)

Standalone portals provide a strictly flat constant basal metabolic trickle (`0.008`/tick) with `0.0` bonus for coverage, matches, rhythm, or balance. `CompositePortalMedium<BrightRadiantWorld>` attaches all 4 portals onto `BrightRadiantWorld` with `0.0` portal metabolic reward (`752` ambient channels, `27` actuators).

## Portal: `DrawingScratchpad` (`192` Channels, `5` Actuators, Uniform Noisy-TV Share `16.67%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `active_stroke_ticks` | `278.0000 [86.5000]` | `493.0000 [12.0000]` | `450.5000 [22.0000]` | `-1.0000` (`p=0.0010`) | `-0.9931` (`p=0.0010`) |
| `canvas_coverage_fraction` | `0.0868 [0.0660]` | `0.8229 [0.0590]` | `0.5139 [0.2292]` | `-1.0000` (`p=0.0010`) | `-0.9861` (`p=0.0010`) |
| `spatial_autocorrelation` | `0.1962 [0.0928]` | `0.2959 [0.1440]` | `0.2851 [0.1398]` | `-0.5556` (`p=0.0280`) | `-0.4722` (`p=0.0579`) |
| `revisit_ratio` | `0.1100 [0.0364]` | `0.2520 [0.0155]` | `0.2300 [0.0324]` | `-1.0000` (`p=0.0010`) | `-1.0000` (`p=0.0010`) |
| `stylus_trajectory_entropy` | `0.9624 [0.0109]` | `0.9742 [0.0086]` | `0.9710 [0.0095]` | `-0.7778` (`p=0.0030`) | `-0.5278` (`p=0.0360`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0004]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.2500` (`p=0.2318`) | `+0.2500` (`p=0.2318`) |
| `mean_interventional_agency` | `0.0569 [0.0367]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |

## Portal: `PatternMemoryBoard` (`176` Channels, `4` Actuators, Uniform Noisy-TV Share `18.18%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiles_inspected` | `63.5000 [13.5000]` | `63.5000 [7.5000]` | `77.0000 [9.7500]` | `+0.1736` (`p=0.4915`) | `-0.7222` (`p=0.0040`) |
| `unique_tiles_visited` | `16.0000 [0.0000]` | `16.0000 [1.0000]` | `16.0000 [1.2500]` | `+0.4167` (`p=0.0400`) | `+0.3333` (`p=0.0969`) |
| `matched_resonances_locked` | `2.0000 [1.2500]` | `0.0000 [1.2500]` | `1.0000 [1.0000]` | `+0.6250` (`p=0.0120`) | `+0.5972` (`p=0.0200`) |
| `repeat_visit_interval_mean` | `83.1822 [23.6518]` | `66.2867 [10.8360]` | `60.9942 [17.4451]` | `+0.6389` (`p=0.0080`) | `+0.6806` (`p=0.0050`) |
| `cursor_dwell_stability` | `0.7870 [0.0257]` | `0.7453 [0.0173]` | `0.7500 [0.0177]` | `+0.9861` (`p=0.0010`) | `+0.8889` (`p=0.0010`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0833` (`p=1.0000`) | `-0.0069` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0630 [0.0565]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |

## Portal: `RhythmEchoPad` (`128` Channels, `5` Actuators, Uniform Noisy-TV Share `25.00%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `total_pad_strikes` | `627.5000 [46.5000]` | `727.5000 [19.5000]` | `755.0000 [34.2500]` | `-0.8333` (`p=0.0010`) | `-0.8681` (`p=0.0010`) |
| `listening_vs_playing_ratio` | `0.2735 [0.0291]` | `0.2524 [0.0142]` | `0.2635 [0.0265]` | `+0.4444` (`p=0.0849`) | `+0.2361` (`p=0.3656`) |
| `phase_locking_value` | `0.0215 [0.0183]` | `0.0220 [0.0129]` | `0.0227 [0.0222]` | `+0.1806` (`p=0.4845`) | `+0.0694` (`p=0.7882`) |
| `motif_continuation_similarity` | `0.1528 [0.0428]` | `0.1681 [0.0244]` | `0.1461 [0.0418]` | `-0.1528` (`p=0.5534`) | `+0.1250` (`p=0.6274`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0000` (`p=1.0000`) | `+0.0000` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0000 [0.0005]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.4167` (`p=0.0509`) | `+0.2778` (`p=0.1728`) |

## Portal: `BalanceCatchToy` (`128` Channels, `3` Actuators, Uniform Noisy-TV Share `25.00%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mean_upright_duration_ticks` | `17.7625 [4.5564]` | `24.4407 [1.3863]` | `23.8116 [0.6153]` | `-1.0000` (`p=0.0010`) | `-1.0000` (`p=0.0010`) |
| `max_upright_streak` | `45.5000 [10.7500]` | `60.0000 [13.0000]` | `57.0000 [10.5000]` | `-0.6736` (`p=0.0060`) | `-0.6250` (`p=0.0090`) |
| `sphere_falls` | `79.0000 [20.2500]` | `58.0000 [3.2500]` | `59.5000 [1.5000]` | `+1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |
| `torque_phase_lead_correlation` | `-0.0752 [0.1130]` | `-0.0000 [0.0568]` | `0.0017 [0.0474]` | `-0.6111` (`p=0.0140`) | `-0.6111` (`p=0.0140`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0833` (`p=1.0000`) | `+0.0833` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0127 [0.0083]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.9167` (`p=0.0010`) | `+0.9028` (`p=0.0010`) |

## Portal: `CompositePortalMedium<BrightRadiantWorld>` (`752` Channels, `27` Actuators, Uniform Noisy-TV Share `21.28%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `host_achieved_influx_per_tick` | `0.0025 [0.0047]` | `0.0008 [0.0004]` | `0.0009 [0.0008]` | `+0.5972` (`p=0.0180`) | `+0.5000` (`p=0.0460`) |
| `final_atp` | `1.0053 [1.1441]` | `11.0217 [0.6782]` | `0.9186 [0.4216]` | `-1.0000` (`p=0.0010`) | `+0.3472` (`p=0.1808`) |
| `host_springs_depleted` | `1.0000 [1.2500]` | `1.0000 [1.0000]` | `1.0000 [0.2500]` | `+0.0833` (`p=0.7233`) | `+0.1458` (`p=0.4795`) |
| `total_portal_engaged_ticks` | `336.5000 [244.7500]` | `0.5000 [1.2500]` | `13.0000 [26.2500]` | `+1.0000` (`p=0.0010`) | `+0.8056` (`p=0.0010`) |
| `sum_mean_portal_focus` | `0.2459 [0.0616]` | `0.2947 [0.0070]` | `0.2361 [0.0524]` | `-0.7917` (`p=0.0020`) | `+0.1250` (`p=0.6274`) |
| `atp_vs_portal_focus_pearson_r` | `0.0883 [0.1721]` | `0.0417 [0.0564]` | `-0.0007 [0.1887]` | `+0.2222` (`p=0.3916`) | `+0.3333` (`p=0.1998`) |
| `high_minus_low_atp_portal_focus` | `0.0714 [0.0673]` | `0.0134 [0.0127]` | `-0.0063 [0.0571]` | `+0.5556` (`p=0.0280`) | `+0.5000` (`p=0.0460`) |
| `portal_structured_conductance_share` | `0.1369 [0.0874]` | `0.0000 [0.0000]` | `0.1192 [0.0832]` | `+1.0000` (`p=0.0010`) | `+0.1806` (`p=0.4845`) |
| `noisy_tv_conductance_share` | `0.2983 [0.1647]` | `0.0000 [0.0000]` | `0.0719 [0.1291]` | `+1.0000` (`p=0.0010`) | `+0.7778` (`p=0.0030`) |
| `mean_interventional_agency` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.2500` (`p=0.2318`) | `+0.1875` (`p=0.2318`) |

