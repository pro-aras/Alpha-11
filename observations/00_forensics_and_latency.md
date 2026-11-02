# Alpha 11 Forensic, Persistence, Plasticity & Latency Audit

## 1. Production Source Hygiene
- **Files Scanned**: `9` | **Clean**: `true`
- **String Literals**: `0` | **Forbidden Anthropomorphic Identifiers**: `0` | **`unsafe` Blocks**: `0` | **`unwrap(` Calls**: `0`

## 2. Transactional CRC32C Persistence & Fail-Closed Poisoning
- **All Checks Passed**: `true` | **Checkpoint Size @100**: `54418` bytes
- **`@100` Restore Snapshot/Hash Exact**: `true` / `true`
- **`@200` Continuation Bit-Identical (`Pulses` / `Snapshot` / `Hash`)**: `true` / `true` / `true`
- **Payload Bit-Flip Fail-Closed**: `true` | **Header/CRC Bit-Flips Passed**: `2/2` | **Truncations Passed**: `5/5` | **Oversized Fail-Closed**: `true`

## 3. `FrozenPlasticity` Invariance & Flagless Lesion Verification
- **All Checks Passed**: `true` (`500` frozen steps tested)
- **Max Changed Parameter Hash Bits (`FrozenPlasticity` / Post-Growth `FrozenPlasticity`)**: `0` / `0`
- **Frozen Topology Invariant**: `true` | **Flagless Lesion Preserved `Natural` Mode**: `true` | **Zeroed All Pore Conductances**: `true`

## 4. Step Latency & Logical / OS Memory Profile
- **Steps Profiled**: `2000` (`4096` ambient channels, `6` actuators)
- **`OrganismEngine::step` Latency (`us`)**: `p50=65.65`, `p95=72.41`, `p99=75.57`, `max=165.79`, `mean=59.85`
- **Logical State Bytes (`final` / `peak`)**: `243026` / `243046` bytes | **Checkpoint Bytes**: `243359` bytes
- **Linux `/proc/self/status` Memory**: `VmRSS=4812800` bytes (`4.59 MiB`), `VmHWM=5341184` bytes (`5.09 MiB`)

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
| `BrightRadiantWorld (128ch)` | `128` | `2000` | `71.44` | `79.36` | `87.32` | `231.80` | `81.48` | `88873` | `89206` |
| `DenseFieldWorld (4096ch)` | `4096` | `2000` | `65.65` | `72.41` | `75.57` | `165.79` | `355.73` | `243026` | `243359` |

- **`32x` Channel Width Scaling Ratio (`4096ch / 128ch`)**: `p50_ratio = 0.919x`, `p99_ratio = 0.865x` (vs `32.000x` linear `O(N)` scaling).
