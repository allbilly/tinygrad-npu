# Text input-channel calibration

`calibrate_text_int8.py` keeps FP16 OpenCL calibration as its default. Its optional `--device CPU` route observes the current INT8 CPU/NPU path without loading another full GPU model. That route currently accepts dense Transformer blocks, uses deferred file-backed CPU weights and bounded CPU preparation scratch, and keeps projections unfused while collecting their inputs.

For unquantized checkpoint inputs without another GPU model, use `--device CPU --cpu-reference`. This dense-Transformer mode reads only selected embedding rows and evaluates projections with chunked FP32 BLAS over checkpoint weights. It rounds each projection to its input/weight result dtype before bias. Weight tiles are bounded; activations and outputs remain model-sized. CPU scratch caching and both temporary hooks are restored on every exit. No accelerator is configured, and the report records the opened devices and arithmetic used.

```sh
DEV=CPU OPENBLAS_NUM_THREADS=4 python extra/rockchip/calibrate_text_int8.py --device CPU --cpu-reference \
  --checkpoint /path/to/model.gguf --dataset /path/to/calibration-prompts.json \
  --output /path/to/reference-input-scales.json --alpha .5 --no-skip-down
```

Use a separate calibration prompt set. Keep held-out quality data out of calibration and parameter selection. For example:

```sh
DEV=CPU python extra/rockchip/calibrate_text_int8.py --device CPU \
  --checkpoint /path/to/model.gguf --dataset /path/to/calibration-prompts.json \
  --output /path/to/input-scales.json --alpha .5 --skip-down
```

The JSON dataset is a list of objects with an `input` string. At most 128 tokens per prompt are observed. Channel maxima are collected from actual eager projection inputs; weight maxima are reduced in bounded CPU chunks. Q/K/V and gate/up share scale tensors. Leaving FFN down unscaled retains the prepared QKV and packed SwiGLU paths. The backend evaluates `X/s` and `W*s` with its existing NPU matmuls, but quantization changes, so a fresh independent quality and complete-request performance check is required.

CPU/NPU statistics come from an already quantized execution. The CPU reference mode skips that quantization. All activation statistics widen to FP32 before scale exponentiation, including FP16 GPU inputs. Reports from earlier source epochs retain their original statistics. Scales are calibration evidence, not proof of preserved model quality. The output binds the checkpoint, prompt dataset, source and harness hashes. Failed observations restore the temporary hooks; startup restores the CPU allocator policy on success or failure.

CPU checks cover wrapper discovery without opening the NPU, shared scale identity, bounded weight reductions, finite inputs, observer restoration, and mocked CLI startup/cleanup. The full Phi3 run observed 193 projections in 97 groups from eight separate 128-token prompts. Its fixed-alpha 0.5 scales passed the unchanged 1,024-target limited original-checkpoint screen and were verified through the oNPU API at context 512. This does not qualify other checkpoints, model maximum context, or general task accuracy.
