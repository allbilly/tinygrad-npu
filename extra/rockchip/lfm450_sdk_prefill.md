# LFM2.5-VL 450M text-decoder prefill

The qualified INT8 gate/up shape is M128, N9216, K1024, split into three N3072 cores. The A16 input and C4 output use the existing SDK-style command and packing path. For this bounded shape, the feature surface follows M: register 0x107c is M*4 and the output surface pitch is M*16. The 128-row, 1024-depth input fits the existing CBUF allocation. No new decode schedule is selected.

The private primitive retained identical INT8 weights, compared changed inputs against an independent INT32 oracle, and interleaved FP16 health calls. All 14 integer checks and 15 FP16 health checks passed. The device-call median changed from 2.429 to 1.002 ms; this excludes input packing and is not full-model throughput.

The actual backend port additionally materializes the ten small short-convolution coefficient tensors at startup with their original dtype. The CPU tests compare ordinary and A16/C4 activation packets and FP32 scales across active-row tails and nonfinite rows. All 32 x 65,536 regression values and all 1,024 natural full-vocabulary vectors retain their original byte hashes. The natural-text reference is the same Q8_0 checkpoint evaluated by llama.cpp CPU; its limited NLL screen does not qualify RKLLM quality equivalence or maximum context.

The oNPU profile is pinned to the exact checkpoint and context 512. It checks six attention blocks and ten short-convolution blocks, enables the measured shared workers and scalar decoder, and disables static uncached prefill because the convolution state is recurrent. The GGUF is the text decoder: these measurements do not include the vision encoder or connector.

Evidence lives under `/home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/`: `lfm450-sdk-primitive-20261101-v1`, `lfm450-sdk-model-20261101-v1`, and `lfm450-sdk-port-quality-20261101-v1`. API results and operating-condition differences must be reported separately from backend timing; the first app recipe missed prefill parity despite a raw-token backend lead.
