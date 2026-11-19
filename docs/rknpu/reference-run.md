# Running the reference (oNPU) on this board

The precise, verified way to obtain a reference prefill number.

## The recorded command — read it, do not retype it

`docs/rknpu/results/ref-recorded-command.txt` holds the argv that produced the **1,201.41 ms**
figure, copied from the archive's own receipt:

    /home/orangepi/oNPU/analysis/release-20261115/r2-lfm-20261116/candidate-retime-20261117-qwen35-4b/receipt.json

It is 54 argv entries, **36 of them feature flags**:

```
--prefill on --decode on --context 512 --tokens 64 --warmups 1 --runs 3
--fuse-projections --cpu-attention --native-replay --fused-ffn --split-down --fused-qkv
--cpu-norm --cpu-fast-softmax --cache-cleaning --wake-helper --cpu-pool --cpu-prefill-pool
--share-projection-weights --cpu-recurrent-scan --fused-swiglu-ffn --fused-swiglu-decode
--fused-qkv-without-norm --fused-recurrent-projections --cpu-q8-embedding --scalar-decode-loop
--trim-preparation-cache --file-backed-cpu-weights --defer-weight-realization
--chunk-size 128 --tile-columns 24384
```

Input: `.../staged/w8a8-sweep-v1/inputs/qwen35-4b.txt`.

## Why this matters — a measured trap

Driving `onpu.worker` over the JSON protocol instead (the earlier version of this document) applies
**none of those 36 flags**, and the reference then measures **2,251.42 ms** at 140 tokens, 80-84 C.
That is **1.87x** the archived 1,201.41 ms. Reporting it as "the reference today" was wrong: it was
the reference in its *default* configuration, not the configuration the archived number came from.

So: any reference figure must name the recorded command that produced it. There is no such thing as
"the reference number" independent of those flags.


## The backend tree — this is the part that was wrong for every re-run today

The archived 1,201.41 ms was **not** produced by `~/tinygrad`. Its receipt records the backend:

```
branch            codex/qwen35-pack-signature-fix
dirty_diff_sha256 7f87f2211b343c08398ddb466ffca24ca3b597ce2659ad4c2bba901504f12f77
dirty_paths       extra/rockchip/benchmark_inference.py, tinygrad/llm/kernels/rockchip_gelu_ffn.py,
                  test/models/test_cpu_gdn_benchmark.py, test/models/test_rockchip_gelu_ffn.py
```

That tree is still on disk at `~/tg-qwen35-pack-signature-fix`, verified:

    head        1b2a5db33
    dirty files 4
    diff sha    7f87f2211b343c08   (matches the receipt)

`~/tinygrad` has moved on, and running the recorded command against it fails at
`configure(model, ... decode_tile_columns ...)` -> `TypeError: unexpected keyword argument`.
So a re-run must use `~/tg-qwen35-pack-signature-fix` as the backend (its venv, its `extra/`), while
keeping `--report` and `ONPU_TIMING_TEMPERATURES_JSONL` pointed into this checkout.

Because `dirty_diff_sha256` is the real identity of the measured configuration, it must be re-checked
before any re-run: a clean checkout of `1b2a5db` is **not** the measured configuration.

## Protocol

- **2 warm-ups + 3 measured**, one worker process, one `generate` per trial (the worker accepts
  repeated requests, so no reload between trials).
- Prompt: the 128-token text `bench_llm` uses. It must be built by decoding
  `SimpleTokenizer.encode(PROMPT)[:128]`, because the worker's context is 512 and
  `prompt + 64 reply` must fit. The ×8 raw text is 612 tokens and is rejected:
  `Prompt (612 tokens) + reply limit (64) exceeds context (512)`.
- The reference tokenizes it to **140** tokens; quote that with the number.

## How to know the run is valid

| check | expected |
| --- | --- |
| `ready:` line appears | `Qwen3.5-4B context=512 device=CPU+ROCKCHIP after ~180s` |
| per-trial line prints `prompt=` | **140** (not 26, not 612) |
| per-trial line prints `temp=` | record it; the reference's own gate is 52-55 °C |
| final lines | `REFERENCE ttft [...] median N ms`, `REFERENCE decode [...] median N tok/s` |
| a `total_seconds` near 30 s with `prefill_tps` ~2.8 | **prefill is on the CPU** — the flag is wrong |

## Four defects that must stay fixed in `run_ref.py`

| # | defect | symptom it produced |
| --: | --- | --- |
| 1 | `stderr=subprocess.PIPE` never drained | worker blocks on a full 64 KiB pipe: "ready at 182.7 s, then silence" |
| 2 | stale NPU lock from an interrupted run | `NPU is owned by another worker` — was misread as a thermal gate |
| 3 | `ttft_ms` read at top level, but `onpu/worker.py:1030` emits `{"event":"done","metrics":{...}}` | completed runs discarded as "no TTFT" |
| 4 | `ONPU_NPU_PREFILL="off"` | prefill on CPU: 49,310.76 ms reported as a reference measurement |

## What a correct run has produced so far

| configuration | TTFT | prefill t/s | decode t/s |
| --- | --: | --: | --: |
| `NPU_PREFILL=off`, 26-token prompt, 1 cold request | 9,101.31 ms | 2.86 | 2.95 |
| `NPU_PREFILL=off`, 140-token prompt, 2+3 | 49,310.76 ms | 2.84 | 2.93 |
| archived `ref-profile` (this board, earlier session) | **1,201.41 ms** | ~101 | 6.48 |

The `on` configuration is the one that must be quoted, and it has not yet completed a measured
run. Until it does, the only reference figure available is the archived 1,201.41 ms, taken at
128 tokens under the 52-55 °C gate.
