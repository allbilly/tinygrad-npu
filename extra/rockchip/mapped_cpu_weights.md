# File-backed CPU GGUF inputs

Deferred GGUF loading normally copies every packed raw tensor to CPU before
building lazy decoded weights. The opt-in `mapped_cpu=True` route instead
borrows contiguous file-byte ranges from one private copy-on-write mapping per
file. CPU buffer storage retains the mapping through views and JIT references.
It avoids the packed staging copy; it does not remove decoded tensors, NPU
packed weights, file-backed resident pages or all memory costs.

The default copied and streaming paths stay available. Mapped loading requires
one CPU device and deferred weight realization. `Transformer.from_gguf` and the
Gemma3n/Gemma4 loaders propagate the flag. Benchmark it with both
`--defer-weight-realization` and `--file-backed-cpu-weights`. Do not use it as a
GPU mapping or a replacement for the NPU's packed weight format.

Tests compare mapped/copied Q8 and native values, nonzero offsets, owner lifetime
after source release, private writes/file immutability, exact INT8 scales and
packed codes, and the ChatGLM projection-permutation/cached-decode reference.
Strided input is rejected before materialization. The last guard was added after
the timed model run; the recorded source cohorts preserve that difference.

ChatGLM3-6B completed three full warmups and three measured 128-input/64-output
requests at context 512: 55.69 prefill proxy tok/s and 4.32 decode tok/s. The
recorded RKLLM reference is 94.61 and 4.98, so this candidate still loses. The
supervisor's 250 ms samples record about 5.0 GiB minimum available RAM; it retains
only the running minimum, not the raw sample series. Comparable output quality
and complete learned-matrix INT8 coverage remain unqualified.

Raw model and verification records are in
`~/oNPU/docs/benchmark-chatglm-mapped-weights-v1.json` and
`~/oNPU/docs/benchmark-chatglm-mapped-verification-v1.json`.
