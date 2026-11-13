# Packed SwiGLU with down-projection bias

The packed CPU/NPU SwiGLU path supports a static CPU FP16 or FP32 down-projection bias vector of exactly the output width. Gate/up bias, a scaled FFN down projection, and biased internal K parts remain excluded. Other gated-activation families retain their existing selection guards.

The operator dequantizes each K part, adds the parts in their existing order, casts to the input dtype, shrinks to logical rows, then adds the parent bias once. This matches `RockchipInt8SplitLinear.npu`. The bias addition does not inherit a zero-tail marker; subsequent capacity conversion pads inactive rows normally.

CPU tests cover installation, prefill/decode selection, supported vector types, wrong shapes/types and biased split-child rejection. The serialized NPU test compares complete biased FFNs against the existing split path: seven changed bias/input cases at each of 1, 17 and 128 rows, plus seven dynamic row rebindings and zero-padding checks. All 28 byte comparisons and both cached FP16 health checks passed.

A TeleChat2 current-source candidate retains all 32×131,072 fixed-prefix logits and all 64 generated IDs against a fresh control. Its unprofiled backend measures 445.13 ms TTFT, 287.56 prefill-proxy and 10.588 decode tok/s at 128 input/64 output/context512. These are backend timings, not a qualified serving result. The unscaled candidate fails the separate original-precision quality screen, so channel calibration and actual API qualification remain required.

The original CPU reference uses an isolated llama.cpp graph adapter with unchanged weights, NeoX RoPE, RMSNorm, SiLU and both output biases. It widens F16 norm/bias vectors exactly and aliases metadata in memory, including lossless tokenizer-attribute type conversion. The graph was checked against a small original-precision Tinygrad model before scoring the full checkpoint. It is not native TeleChat2 loader support and does not cover dynamic NTK beyond 8,192 tokens.
