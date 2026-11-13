# Experimental packing copy aliases

`ONPU_REPLAY_EXTRA_PACK_ALIAS=1`, together with `ONPU_FFN_PACKED_UPLOAD=1`,
allows replay's complete-write classifier to recognize the internal
`rockchip_int8_erf_gelu_pack` output. The same opt-in also recognizes the
complete `rockchip_quantize_sdk` output, which does not require the FFN flag. This is a capture-time opt-in; build
fresh captures when comparing policies. The default remains disabled.

The producer fills all packed activation bytes, scales, inactive rows and
header padding. CPU tests initialize the output with two opposing poison
patterns and compare every byte against the independent Tensor activation
and quantization reference. They cover one-row decode, full and partial
128-row capacity, and nonfinite weights. The classifier still checks the
exact output extent and rejects input slots and larger buffer ranges.

An accepted alias from either recognized producer also records its exact
allocation identity, address and byte count. With the experiment enabled,
native TO-device cleaning may cover that proven view (at most 1 MiB) inside
an already clean arena of at most 16 MiB. Cache-line alignment, containment,
flags and clean-version checks still apply. This cleans only the producer's
view, including when other NPU outputs share the arena; unrelated buffers
keep the original allocation limit. CPU tests reject missing or mismatched
provenance, dirty metadata, oversized views/arenas and misaligned ranges.

This classification does not itself establish that an alias is safe. The
existing future-reader, lifetime, overlap and cache-ownership checks still
apply. Before promotion, compare native replay with the flag off/on using
identical physical weights and complete FP32 input/output work. Record the
actual copy aliases and cache-maintenance guard decisions; verify changed
inputs, invalid-input rejection and recovery, full model logits and API
behavior. No performance gain or hardware qualification is claimed by the
CPU complete-write tests.
