# Study 6: Unforced Human-Activity Portals (`N = 12` Seeds, `1500` Ticks, Zero Game-Score Rewards)

Standalone portals provide a strictly flat constant basal metabolic trickle (`0.008`/tick) with `0.0` bonus for coverage, matches, rhythm, or balance. `CompositePortalMedium<BrightRadiantWorld>` attaches all 4 portals onto `BrightRadiantWorld` with `0.0` portal metabolic reward (`752` ambient channels, `27` actuators).

## Portal: `DrawingScratchpad` (`192` Channels, `5` Actuators, Uniform Noisy-TV Share `16.67%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `active_stroke_ticks` | `494.0000 [32.0000]` | `490.5000 [15.0000]` | `510.5000 [20.2500]` | `+0.2083` (`p=0.4136`) | `-0.1806` (`p=0.4725`) |
| `canvas_coverage_fraction` | `0.8681 [0.0590]` | `0.8229 [0.0903]` | `0.8507 [0.0608]` | `+0.2431` (`p=0.3477`) | `+0.0903` (`p=0.7073`) |
| `spatial_autocorrelation` | `0.2990 [0.0955]` | `0.3193 [0.1213]` | `0.3828 [0.1531]` | `-0.1667` (`p=0.5145`) | `-0.4722` (`p=0.0579`) |
| `revisit_ratio` | `0.2665 [0.0107]` | `0.2550 [0.0155]` | `0.2825 [0.0223]` | `+0.5694` (`p=0.0240`) | `-0.4722` (`p=0.0579`) |
| `stylus_trajectory_entropy` | `0.9772 [0.0056]` | `0.9744 [0.0097]` | `0.9733 [0.0089]` | `+0.1389` (`p=0.5974`) | `+0.2222` (`p=0.3916`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0833` (`p=1.0000`) | `-0.0069` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0245 [0.0132]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |

## Portal: `PatternMemoryBoard` (`176` Channels, `4` Actuators, Uniform Noisy-TV Share `18.18%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiles_inspected` | `82.0000 [12.0000]` | `61.5000 [10.2500]` | `90.0000 [12.5000]` | `+0.9306` (`p=0.0010`) | `-0.3889` (`p=0.1379`) |
| `unique_tiles_visited` | `16.0000 [0.0000]` | `16.0000 [1.0000]` | `16.0000 [0.2500]` | `+0.4167` (`p=0.0400`) | `+0.2500` (`p=0.2108`) |
| `matched_resonances_locked` | `1.0000 [1.0000]` | `0.5000 [1.2500]` | `1.0000 [1.0000]` | `+0.2917` (`p=0.2108`) | `+0.2917` (`p=0.2458`) |
| `repeat_visit_interval_mean` | `76.0096 [6.3418]` | `66.6417 [10.3504]` | `61.0953 [6.2033]` | `+0.5833` (`p=0.0220`) | `+0.8889` (`p=0.0010`) |
| `cursor_dwell_stability` | `0.7449 [0.0162]` | `0.7441 [0.0179]` | `0.7344 [0.0234]` | `+0.0694` (`p=0.7882`) | `+0.3056` (`p=0.2438`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0000` (`p=1.0000`) | `-0.0833` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0173 [0.0195]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |

## Portal: `RhythmEchoPad` (`128` Channels, `5` Actuators, Uniform Noisy-TV Share `25.00%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `total_pad_strikes` | `826.5000 [17.0000]` | `727.5000 [23.7500]` | `887.5000 [33.2500]` | `+1.0000` (`p=0.0010`) | `-0.9583` (`p=0.0010`) |
| `listening_vs_playing_ratio` | `0.2275 [0.0082]` | `0.2513 [0.0129]` | `0.2251 [0.0109]` | `-0.7361` (`p=0.0040`) | `+0.1806` (`p=0.4845`) |
| `phase_locking_value` | `0.0205 [0.0158]` | `0.0198 [0.0168]` | `0.0160 [0.0167]` | `-0.0694` (`p=0.7882`) | `+0.2778` (`p=0.2887`) |
| `motif_continuation_similarity` | `0.1442 [0.0351]` | `0.1666 [0.0272]` | `0.1383 [0.0284]` | `-0.4444` (`p=0.0849`) | `+0.0972` (`p=0.7113`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0000` (`p=1.0000`) | `-0.0833` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0833` (`p=1.0000`) | `+0.0069` (`p=1.0000`) |

## Portal: `BalanceCatchToy` (`128` Channels, `3` Actuators, Uniform Noisy-TV Share `25.00%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `mean_upright_duration_ticks` | `23.2160 [1.7597]` | `24.0167 [0.9333]` | `23.8116 [1.3407]` | `-0.2847` (`p=0.2747`) | `-0.3472` (`p=0.1828`) |
| `max_upright_streak` | `56.5000 [9.0000]` | `58.5000 [9.0000]` | `60.5000 [23.7500]` | `-0.0139` (`p=0.9660`) | `-0.3125` (`p=0.2248`) |
| `sphere_falls` | `61.0000 [4.5000]` | `59.0000 [2.2500]` | `59.5000 [3.2500]` | `+0.2778` (`p=0.2827`) | `+0.3472` (`p=0.1798`) |
| `torque_phase_lead_correlation` | `-0.0371 [0.0823]` | `0.0030 [0.0682]` | `-0.0002 [0.0448]` | `-0.4583` (`p=0.0729`) | `-0.4444` (`p=0.0849`) |
| `noisy_tv_conductance_share` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0000` (`p=1.0000`) | `+0.0000` (`p=1.0000`) |
| `mean_interventional_agency` | `0.0015 [0.0061]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.7500` (`p=0.0010`) | `+0.7014` (`p=0.0030`) |

## Portal: `CompositePortalMedium<BrightRadiantWorld>` (`752` Channels, `27` Actuators, Uniform Noisy-TV Share `21.28%`)

| Behavioral / Structural Observable | `Natural` (`p50 [IQR]`) | `FrozenPlasticity` (`p50 [IQR]`) | `ActionShuffled` (`p50 [IQR]`) | `Nat vs Frozen` (`r_rb`, `p_perm`) | `Nat vs Shuffled` (`r_rb`, `p_perm`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `host_achieved_influx_per_tick` | `0.0048 [0.0031]` | `0.0008 [0.0005]` | `0.0008 [0.0012]` | `+0.9444` (`p=0.0010`) | `+0.9444` (`p=0.0010`) |
| `final_atp` | `1.8243 [1.0802]` | `11.0375 [0.7960]` | `0.8901 [0.1665]` | `-1.0000` (`p=0.0010`) | `+1.0000` (`p=0.0010`) |
| `host_springs_depleted` | `2.0000 [1.0000]` | `1.0000 [0.2500]` | `1.0000 [0.2500]` | `+0.7708` (`p=0.0020`) | `+0.7014` (`p=0.0030`) |
| `total_portal_engaged_ticks` | `32.5000 [42.0000]` | `0.0000 [1.0000]` | `6.5000 [10.2500]` | `+1.0000` (`p=0.0010`) | `+0.7431` (`p=0.0040`) |
| `sum_mean_portal_focus` | `0.2128 [0.0374]` | `0.2932 [0.0048]` | `0.2592 [0.0486]` | `-0.8611` (`p=0.0010`) | `-0.5000` (`p=0.0460`) |
| `atp_vs_portal_focus_pearson_r` | `0.0103 [0.2144]` | `0.0381 [0.0512]` | `-0.2372 [0.1730]` | `-0.1389` (`p=0.5974`) | `+0.7500` (`p=0.0040`) |
| `high_minus_low_atp_portal_focus` | `0.0086 [0.0458]` | `0.0108 [0.0127]` | `-0.0488 [0.0502]` | `-0.1389` (`p=0.5974`) | `+0.5417` (`p=0.0300`) |
| `portal_structured_conductance_share` | `0.1127 [0.0451]` | `0.0000 [0.0000]` | `0.1018 [0.0468]` | `+1.0000` (`p=0.0010`) | `+0.2361` (`p=0.3656`) |
| `noisy_tv_conductance_share` | `0.0230 [0.0205]` | `0.0000 [0.0000]` | `0.0000 [0.0023]` | `+0.8333` (`p=0.0010`) | `+0.6667` (`p=0.0040`) |
| `mean_interventional_agency` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `0.0000 [0.0000]` | `+0.0000` (`p=1.0000`) | `+0.0000` (`p=1.0000`) |

