# Parked workers for raw FFN packing

The optional fused raw GELU and SwiGLU packers reuse the existing parked CPU
workers when `--cpu-prefill-pool` is requested and its pool scope is active.
The caller handles one row partition and the three workers handle the others.
A stack work descriptor carries all raw projection and scale pointers through
the pool's four-pointer callback interface. The blocking dispatch finishes
before that descriptor goes out of scope.

The packer retains the same FP32 arithmetic, fragment layout, independent
split-K scales, inactive-row initialization and payload padding. Capacity
below 16 rows and inactive pool scopes retain the existing dispatch path.
The prefill-pool option remains false by default.

An isolated prototype passed eight CPU tests covering both activations,
multiple raw buffers, changed-input JIT replay, partial rows, exact payload
and scale bits, caller affinity and parked workers. Five CPU pack geometries
also passed 800 exact output checks in alternating paired timing blocks.
The partial-row case improved by about 2x; full-row gains were smaller and
variable. These primitive results do not establish a model speedup.

Repository checks and full NPU/native replay model measurements are recorded
separately. The model benchmark table retains its qualified measurements
until the new path has full-model evidence.

Integrated CPU verification passed 46 tests with two explicit hardware skips,
plus mypy on 280 files and repository Ruff. Both original serving models
were restored and answered fresh requests.
[CPU and primitive evidence](../../../oNPU/docs/benchmark-ffn-prefill-pool-cpu-verification-v1.json).

Full-model FFN pool measurements completed with four ABBA processes per model.
The complete SiLU/GELU NPU/native-replay test passed at 17/128 rows with seven
changed inputs. All four sets of 32 teacher vectors and all 24 generated
sequences per model matched exactly.

| Model | Prefill proxy tok/s, local threads / parked workers | Decode tok/s, local threads / parked workers |
|---|---:|---:|
| Qwen2/0.5B | 481.74 / 503.99 | 38.676 / 38.617 |
| TinyLLAMA/1.1B | 335.23 / 347.20 | 24.954 / 25.038 |
| InternLM2/1.8B | 252.74 / 252.51 | 15.149 / 14.746 |
| Gemma2/2B | 174.11 / 174.21 | 9.654 / 9.678 |

The gains vary by model and clocks remain adaptive. TinyLLAMA alone qualified
for the conservative preferred-row update; other preferred rows were retained.
Comparable HF/RKLLM quality and the all-model speed goal remain unqualified.
[Full-model evidence](../../../oNPU/docs/benchmark-ffn-prefill-pool-models-verification-v2.json).
