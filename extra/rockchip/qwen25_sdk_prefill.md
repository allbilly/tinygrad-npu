# Qwen2.5 1.5B split prefill

The opt-in A16/C4 path supports M128/K1536 gate/up matrices with 12,288 and 5,632 output columns. Three NPU cores receive widths 4,096/4,096/4,096 or 1,888/1,888/1,856 respectively. Each core executes two linked tasks: the last 44 rows at offset 84, followed by the first 84 rows. Physical input/output row strides remain 128.

The 12 CBUF banks are 32 KiB each. The captured plan allocates three data/nine weight banks for 44 rows and four data/eight weight banks for 84 rows. `int8_sdk_bodies` derives these bounded fields and retains the SDK's ordered configuration/reset writes. The CPU golden fixture checks all 648 captured words across the three core widths, including duplicate writes and normalized addresses.

Each task has 108 body words. The first task's tail points to the second command block and supplies its register amount; the second terminates the core's chain. The CPU-only descriptor test verifies links, bounds, amount fields and termination for all three cores, plus unchanged single-task paths. **Changing task counts/subcore ranges without constructing PC links is insufficient.** The first experimental replay omitted those links and timed out; it was stopped and never promoted. Original sessions recovered and answered fresh requests before the corrected replay ran.

The shared CPU consumer reads C4 output in four-row batches. One packed input upload serves all down-projection K groups, retaining their independent scales and rounding. No arithmetic approximation was introduced.

Evidence workspace: `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/`.

| Evidence | Outcome |
|---|---|
| `qwen25-core-sdk-capture-v1/results/` | Passive SDK captures of all three exact core widths. |
| `qwen25-sdk-primitive-v2/results/single.json` | Corrected single-core replay, exact changed inputs; 2.4061 / 1.8547 ms ordinary / SDK. The summary's three-core scope string is corrected in `execution-audit.json`. |
| `qwen25-sdk-primitive-v2/results/full.json` | Three-core replay exact for both matrices; wide tile 3.8178 / 2.9616 ms, smaller tile 1.3511 / 1.4599 ms. Both gains and losses retained. |
| `qwen25-sdk-model-v2/results/sdk-upload.json` | Shared-upload candidate: 308.43 ms first-token median, exact 32-vector regression. |
| `qwen25-sdk-production-v1/results/qwen25.json` | Actual port: 310.85 ms backend first-token median / 17.61 decode tok/s, exact regression. |
| `qwen25-sdk-api-v1/results/qwen25-api.json` | Ten uncached 128-input/64-output API requests at context 512: 352.89 ms TTFT, 362.72 prefill-proxy / 17.61 decode tok/s. All ten beat both published targets; all 18 serving checks pass. |

The original OpenCL model remained resident. No governor, clock-limit, IRQ or idle settings were changed. API first-text TTFT and backend first-token time remain separate metrics. Independent RKLLM quality equivalence and model-maximum context are unqualified; the failed v1 and recovery records remain preserved.
