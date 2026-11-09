# SDK-derived K2048 prefill and erf-GELU packing

The optional SDK prefill path accepts full `(M,N,K)` shapes `(128,12288,2048)`, `(128,3072,2048)` and `(128,1024,2048)`. Captured per-core widths are 4096, 1024, 352 and 320. The ordinary path remains selected unless `ONPU_INT8_SDK_PREFILL=1`.

The SDK uses two 64-row tasks per core, processing rows 64–127 before rows 0–63. Each task has four data and eight weight CBUF banks, compared with eight data and four weight banks for the ordinary 128-row task. Input/output addresses advance by `64*16` bytes inside the A16/C4 surface; the full 128-row surface pitch remains unchanged. Command generation retains all 108 ordered register writes per task, including duplicate reset/configuration writes, and builds fresh PC links and buffer addresses.

For the captured erf-GELU gate/up geometry `(128,13312,2048)`, the raw FFN uses the 12288- and 1024-column SDK tiles. CPU packing retains the existing per-K-group quantization and arithmetic. `ONPU_SWIGLU_PREFILL_ROWS=4` also batches erf-GELU packing when every source tile has SDK C4 layout. Ordinary-layout erf-GELU retains one-row packing: its previous row-batching screen did not show a useful gain. Channel scratch is not enabled by this qualification.

CPU golden tests compare every ordered word in eight captured task bodies (864 words), verify three-core chain boundaries and reject unsupported shapes. Packet/scale tests cover active-row tails and ensure ordinary layout does not accidentally select the new batching path. The explicit hardware experiment passes 56 complete INT32 matrix checks and 64 mixed INT8/FP16 checks.

Spark-X2.5 1.7B's current backend candidate records about 285.80 prefill / 15.85 decode tok/s, versus the published 331.41 / 15.61. It does not establish prefill parity or consistent decode wins. All 32 full-vocabulary local regression vectors match; independent quality, model-maximum context and stock API qualification remain open. The app profile is unchanged.

Evidence is recorded in `/home/orangepi/oNPU/docs/benchmark-spark-sdk-prefill-20261031.json`. The evidence retains the earlier invalid exporter reports separately; their saved arrays/timings were audited without rewriting their failed completion fields. Current production runs use the corrected exporter and no prototype patches.
