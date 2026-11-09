# Separate packed input aliases from large-allocation cache cleaning

`ONPU_REPLAY_EXTRA_PACK_ALIAS=1` admits SDK quantization and erf packing to the
existing full-write, ownership, overlap and future-reader proof. It does not
permit arbitrary CPU outputs to borrow NPU memory.

`ONPU_REPLAY_EXTRA_PACK_NATIVE_CLEAN` defaults to one, retaining the previous
large-allocation native-clean exception. Set it to zero to retain driver TO
synchronization for those views while keeping accepted input aliases. Ordinary
small-allocation native cleaning and all output-invalidation guards are unchanged.
Both original defaults and the original producer whitelist are preserved.

The exact LFM/context-512 app profile selects aliases=1/native-clean=0. Its
qualified static capture has 67 input aliases, zero prefill native-clean sites,
and 83 decode native-clean sites. It records 26 direct SDK provenance entries;
that count differs from total aliases because recognizing future full writes can
also make earlier uploads safe to redirect. Startup checks require the supported
policy and the actual warmed native capture, rather than assuming env settings
activated it.

The November 3 full native-clean experiment was numerically exact but lost the
paired API comparison: TTFT 102.26→119.53 ms. Its temporary extension of native
provenance to ordinary quantization/SwiGLU was not retained. The alias-only path
keeps driver synchronization and reduces paired API TTFT 122.70→112.48 ms. Each
timed request proves its expected alias count and increases the selected native
execution counter. The first timed control call was its first native execution
after three ordinary warmup requests; independent default workers are reported
separately.

A post-numerical export assertion expected 42 provenance records instead of 26.
The original failed exit is preserved. An independent receipt verifies saved
32×65536 teacher-logit bytes, all 1,024 natural-target hashes, repeated data and
0/67/83 execution counts; no inference was repeated to repair that metadata error.

See `~/oNPU/docs/benchmark-lfm450-upload-policy-20261103.json` for both slower and
selected cohorts, source epochs and restoration. These tests cover text/context
512. They do not establish independent RKLLM quality, model maximum, image
processing or concurrent generation. No board settings change is needed.
