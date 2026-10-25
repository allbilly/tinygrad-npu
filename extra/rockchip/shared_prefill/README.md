# Experimental shared GPU/NPU prefill projection

This package measures an imperative projection operator that shares NPU GEM buffers with OpenCL through DMA-BUF. The GPU packs FP32 activations to FP16 in the shared allocation, the NPU runs a projection with FP16 weights, and the GPU unpacks the output.

This is a standalone operator experiment. It is not installed in model serving and is not TinyJit-safe: TinyJit does not record these imperative GPU/NPU side effects. It has not been validated as a full language-model prefill path or end-to-end speed improvement.

## Precision and historical result

The benchmark compares three paths:

- `gpu_fp32`: FP32 activations, FP16 weights, FP32 outputs.
- `npu_copy_fp16`: activations rounded to FP16 and copied to the NPU, with FP16 weights and FP32 outputs.
- `npu_shared_fp16`: the same FP16-rounded activations and weights, passed through the shared-buffer operator.

Across the four historically measured shapes below, shared and copied NPU outputs were bit-identical. Shared-buffer NPU execution was **2.1–3.1× faster** than the copy path. The GPU FP32 path uses different activation precision, so its timing is not a same-precision comparison.

| Rows × columns × depth | Shared vs. copied NPU median |
|---|---:|
| 64 × 2,048 × 2,048 | 2.96× |
| 64 × 6,144 × 2,048 | 2.60× |
| 64 × 12,288 × 2,048 | 2.10× |
| 64 × 2,048 × 4,096 | 3.07× |

The measurements include per-call GPU packing, cache and queue synchronization, finite-value checks, NPU submissions, GPU unpacking, and completion. They exclude one-time data/weight setup, imported GEM creation, helper allocation, kernel compilation, warmups, and validation readback. These four projection shapes do not establish a full LLM win. The four historical reports and expected source identities are included in [recorded-results.json](recorded-results.json). The operator helper remains byte-identical to the measured version.

## Commands

Run these commands from the tinygrad checkout root using its Python environment.

Check all packaged historical results and the synthetic validator cases without devices:

```sh
python -m extra.rockchip.shared_prefill.validate_report --recorded
python -m extra.rockchip.shared_prefill.smoke_validator
```

List the available cases:

```sh
python -m extra.rockchip.shared_prefill.benchmark --list-cases
```

Preview a case without opening devices or writing files:

```sh
python -m extra.rockchip.shared_prefill.benchmark --case m64_n6144_k2048 --output /tmp/shared-prefill.json --dry-run
```

Run that case on the device:

```sh
python -m extra.rockchip.shared_prefill.benchmark --case m64_n6144_k2048 --output /tmp/shared-prefill.json
```

A completed run writes the report and a sibling `.manifest.json`. Validate it against that manifest:

```sh
python -m extra.rockchip.shared_prefill.validate_report --report /tmp/shared-prefill.json
```

Existing reports and manifests are never overwritten. The list, dry-run and validator commands have been checked without device imports. The portable wrapper has not been rerun on hardware; its packaged measurements come from the original managed run.

## Device requirements

Run measurements on an idle RK3588 with the GPU and NPU available exclusively to this experiment. The script does not stop serving workers or take ownership away from another process. It checks the 2 GiB RAM reserve before device initialization and between operator calls, with an additional 512 MiB of required startup headroom.

The historical measurements used a managed device lease. Source and historical results can be inspected on other systems, including Apple hardware, but this Rockchip GPU/NPU experiment cannot run there.
