# Avoid materializing unused function arguments

Gemma's precompiled block functions receive the block object explicitly.
Function argument collection includes its NPU wrappers' original lazy CPU
weights. Previously, precompilation made every captured argument contiguous,
even if the function never read it. Native profiles showed complete Q8 weight
matrices being unpacked to FP16 on every generated token.

Precompiled functions now remove unused pure arguments and remap retained
parameter slots. Arguments containing stores, calls or ordering dependencies
remain, custom-gradient calls retain their positional interface, and outputs
without their own device retain the original placement inputs. This changes
argument preparation, not model arithmetic or quantization.

Renaming must use a single-pass substitution. An early prototype used the
rewriter's fixed-point mode, merging same-shaped inputs along slot-renaming
chains. Hardware token checks caught the error and that attempt was stopped
and rejected. Regression tests now cover distinct same-shaped inputs, symbolic
and tuple outputs, nested calls, output-device inference, side effects and the
custom-gradient interface.

The corrected code passed 153 runtime/model tests and 63 subtests, with four
skips and one expected failure; eight NULL-device function tests, mypy and Ruff
also passed. Its hardware stock-token check precedes any complete timing run.
The complete Gemma4 and Gemma3n candidates each retained all six prior 64-token
sequences. The rejected prototype's timings do not enter the comparison table.

The measured improvements combine this argument fix, Gemma's CPU-attention
integration and mapped weight loading. They do not establish comparable RKLLM
quality or a speed win. Whole-model evidence is published separately in the
[oNPU benchmark records](../../../oNPU/docs/benchmark-gemma-pruned-verification-v3.json).
