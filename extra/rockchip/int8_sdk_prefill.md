# SDK-derived INT8 prefill layouts

The GDB-derived recipe needs 16-channel input blocks and four-channel output
blocks. Copying its registers into the existing row-major pipeline is incorrect.
The opt-in implementation writes those blocks directly during quantization and
reads them directly during FP32 dequantization, retaining per-token scales and
the existing integer product. Weight packing and quantization boundaries stay
unchanged.

`ONPU_INT8_SDK_PREFILL=1` selects only these previously qualified complete
matrix shapes: M/N/K = 64/2112/1024, 64/3072/2048 and 128/3072/1024. The default
is off. The low-level raw API requires an explicit layout flag; fused FFN raw
intermediates retain their existing layout. Unsupported shapes retain their
previous recipe, including the two known incorrect extrapolations.

Full projection measurements include CPU quantization, upload, blocking NPU
execution, readback, FP32 dequantization and native dispatch. Inputs are already
resident; numerical readback/checks are outside the timed region. There are
three warmups and 20 alternating ABBA/BAAB blocks, 40 timed calls per variant.

| M | N | K | Existing pipeline median ms | SDK pipeline median ms | Paired speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 2112 | 1024 | 1.332 | 1.352 | 0.997× |
| 64 | 3072 | 2048 | 1.498 | 1.411 | 1.069× |
| 128 | 3072 | 1024 | 2.644 | 2.697 | 0.994× |

The second case favored the SDK pipeline in 15 of 20 blocks. The other two
are near parity. This is substantially less than the earlier NPU-call-only
gains, which excluded packing and output conversion. Ratios are medians of
paired block ratios rather than quotients of separate latency medians.

Six additional M=128, K=1024 widths (2048, 2304, 2560, 4096, 6144 and 7168)
also produced correct outputs, but their SDK pipelines were slower in every
paired block. They remain outside the production whitelist. Tiling CPU packing
and conversion to reduce strided cache traffic is a hypothesis to test next.

All 126 outputs across nine cases and seven changed inputs per variant matched
an independent quantization/int32-matmul/FP32-dequantization reference bit for
bit. This includes native replay and alternating-sign inputs. Steady-state NPU
allocations were zero. CPU tests also cover blocked input padding, mixed output
layouts, partial columns and rejected descriptors.

The probe ran with a 2 GiB memory limit and no swap, while the owner monitored
a 2 GiB system reserve. The original GPU worker remained loaded. The NPU worker
was restored, and both original workers answered fresh requests afterward.

[Raw samples, source identities, CPU checks and restoration evidence](int8_sdk_prefill_results.json)
retain both the successful and slower cases. These are projection results;
the LLM benchmark table and comparable-quality qualification are unchanged.

The v2 repeat groups CPU conversion across four rows. Quantized bytes, scales,
nonfinite-row handling and FP32 dequantization remain unchanged. Eleven CPU
tests passed with six hardware skips; mypy and Ruff passed. Ten complete
projection cases produced 140 bitwise reference matches through native replay.

| M | N | K | Existing median ms | SDK median ms | Paired speedup | Faster blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2112 | 1024 | 1.141 | 1.172 | 0.989× | 9/20 |
| 64 | 3072 | 2048 | 1.237 | 1.142 | 1.097× | 18/20 |
| 128 | 3072 | 1024 | 2.225 | 2.216 | 0.986× | 9/20 |
| 128, split into two SDK calls | 3072 | 2048 | 1.777 | 1.965 | 0.903× | 0/20 |

The original 64/3072/2048 case is now 9.7% faster as a complete projection.
The six expanded widths still lose overall, with paired speedups between
0.815× and 0.914×. The split-row variant is also slower and is not enabled.
The repeat provides separate, single-request diagnostic profiles outside the
timing samples; those profiles do not isolate stable per-component speedups.
Historical v1/v2 differences also include clock and process-state variation.

The broader dispatch remains disabled. Further work returns to full-model
bottlenecks, starting with Gemma2's capped CPU attention. The NPU worker was
restored and both normal workers answered fresh requests; the GPU remained
loaded throughout the operator run. [V2 samples, profiles, CPU checks and
restoration evidence](int8_sdk_prefill_v2_results.json).
