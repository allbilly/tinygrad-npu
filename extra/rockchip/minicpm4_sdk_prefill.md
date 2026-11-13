# MiniCPM4 A16/C4 gate/up prefill

The qualified shape is M128/K1024/N8192, with core widths2752/2752/2688. Both native108-word command bodies match captured SDK writes. Input channels use16-channel blocks; outputs use4-channel blocks. Runtime commands use fresh owned addresses. Existing per-core command/task construction remains unchanged.

The shared legacy direct-division SwiGLU path accepts this shape through `SDK_SILU_DIV_PREFILL_SHAPES`. Full/tail CPU tests preserve every payload/scale byte; independent integer oracles and32full-model vectors pass. Other dimensions continue through their existing qualified paths. The app pins the MiniCPM4 checksum/context512, enables `ONPU_INT8_SDK_PREFILL=1`, and checks descriptor plus activation capability before loading.

The same-worker screen improved, but two fresh HTTP process medians were971.90 and868.27prefill tok/s. The pooled result is889.60prefill /57.59decode versus published946.12/45.34. Prefill parity remains unproven. All36functional checks pass and original GPU/NPU sessions are restored with fresh replies. Independent quality and model-maximum context remain unqualified.

Evidence: `/home/orangepi/oNPU/docs/benchmark-minicpm4-sdk-api-20261031.json`. The ordinary feature-grain64 alternative was slower and was not ported.
