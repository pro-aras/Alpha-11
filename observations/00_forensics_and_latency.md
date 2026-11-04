# Alpha 11 Forensic, Persistence, Plasticity & Latency Audit

## 1. Production Source Hygiene
- **Files Scanned**: `9` | **Clean**: `true`
- **String Literals**: `0` | **Forbidden Anthropomorphic Identifiers**: `0` | **`unsafe` Blocks**: `0` | **`unwrap(` Calls**: `0`

## 2. Transactional CRC32C Persistence & Fail-Closed Poisoning
- **All Checks Passed**: `true` | **Checkpoint Size @100**: `76853` bytes
- **`@100` Restore Snapshot/Hash Exact**: `true` / `true`
- **`@200` Continuation Bit-Identical (`Pulses` / `Snapshot` / `Hash`)**: `true` / `true` / `true`
- **Payload Bit-Flip Fail-Closed**: `true` | **Header/CRC Bit-Flips Passed**: `2/2` | **Truncations Passed**: `5/5` | **Oversized Fail-Closed**: `true`

## 3. `FrozenPlasticity` Invariance & Flagless Lesion Verification
- **All Checks Passed**: `true` (`500` frozen steps tested)
- **Max Changed Parameter Hash Bits (`FrozenPlasticity` / Post-Growth `FrozenPlasticity`)**: `0` / `0`
- **Frozen Topology Invariant**: `true` | **Flagless Lesion Preserved `Natural` Mode**: `true` | **Zeroed All Pore Conductances**: `true`

## 4. Step Latency & Logical / OS Memory Profile
- **Steps Profiled**: `2000` (`4096` ambient channels, `6` actuators)
- **`OrganismEngine::step` Latency (`us`)**: `p50=85.79`, `p95=98.49`, `p99=105.37`, `max=129.11`, `mean=79.78`
- **Logical State Bytes (`final` / `peak`)**: `270687` / `271099` bytes | **Checkpoint Bytes**: `271017` bytes
- **Linux `/proc/self/status` Memory**: `VmRSS=5128192` bytes (`4.89 MiB`), `VmHWM=5476352` bytes (`5.22 MiB`)

## 5. Multi-Crate Source Hygiene Breakdown (`a11-medium`, `a11-core`, `a11-worlds`, `a11-portals`)

| Crate | Files Scanned | String Literals | Forbidden Identifiers | `unsafe` Blocks | `unwrap(` Calls | Zero Forbidden/Unsafe/Unwrap |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `a11-medium` | `1` | `0` | `0` | `0` | `0` | `true` |
| `a11-core` | `9` | `0` | `0` | `0` | `0` | `true` |
| `a11-worlds` | `7` | `0` | `0` | `0` | `0` | `true` |
| `a11-portals` | `7` | `13` | `0` | `0` | `0` | `true` |
  - Finding in `live_host.rs:53` (`StringLiteral`): `"organism.ckpt"`
  - Finding in `live_host.rs:54` (`StringLiteral`): `"telemetry.jsonl"`
  - Finding in `live_host.rs:65` (`StringLiteral`): `"tmp"`
  - Finding in `live_host.rs:71` (`StringLiteral`): `"prev"`
  - Finding in `live_host.rs:77` (`StringLiteral`): `"prev.tmp"`
  - Finding in `live_host.rs:83` (`StringLiteral`): `"1"`
  - Finding in `live_host.rs:455` (`StringLiteral`): `"{{\"tick\":{},\"atp\":{},\"adeno"`
  - Finding in `live_host.rs:477` (`StringLiteral`): `"{v:.6}"`

## 6. Bounded `O(1)` Perceptual Scaling Comparison (`128` vs `4,096` Ambient Channels = `32x` Input Width)

| World Arena | Ambient Channels | Steps | `step()` `p50` (`us`) | `step()` `p95` (`us`) | `step()` `p99` (`us`) | `step()` `max` (`us`) | Full Tick `p50` (`us`) | Final Logical State (`bytes`) | Checkpoint (`bytes`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BrightRadiantWorld (128ch)` | `128` | `2000` | `94.08` | `107.25` | `112.99` | `137.87` | `103.85` | `115525` | `115855` |
| `DenseFieldWorld (4096ch)` | `4096` | `2000` | `85.79` | `98.49` | `105.37` | `129.11` | `368.34` | `270687` | `271017` |

- **`32x` Channel Width Scaling Ratio (`4096ch / 128ch`)**: `p50_ratio = 0.912x`, `p99_ratio = 0.933x` (vs `32.000x` linear `O(N)` scaling).
