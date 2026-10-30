# Prepared causal NPU prefill attention

`--cpu-attention --npu-prefill-attention` opts into a combined attention
operator for CPU Q/K/V preparation with NPU projections. It currently supports
16 query heads, 8 KV heads, head width 128, context 64/128/256/512, and a query
capacity of at most 128. Unsupported geometries use the existing CPU hook.
The option remains experimental and false by default. Model performance and
comparable RKLLM quality have not yet been qualified.

With INT8 projections, use column tiles no larger than 24,384 (`--tile-columns 24384`).
Mixed INT8/FP16 replay at K=1024 passed 20 changed-input cycles with 8,128 columns per
core, but 8,192-column INT8 tasks corrupted the following FP16 scores, including
with one core. The configuration guard rejects larger tiles when NPU prefill
attention is enabled with INT8 projections. INT8-only and FP16-only configurations
retain their existing bounds.

The operator receives CPU query and live KV-cache pointers. It prepares row-major FP16
queries and native packed K/V, executes prepared three-core QK commands, applies causal FP32 CPU softmax,
packs FP16 probabilities, executes prepared PV commands, and returns CPU FP32
output. It selects a context bucket at runtime rather than always computing the
allocated maximum context. Query rows beyond the active count are zeroed, and
future cache entries are never consumed. Changed input, cached prefixes, and
partial query tiles use the same graph.
Paired query heads share eight packed K/V matrices; query group ordering and
output unpacking retain the model's original head order. Host softmax follows
the selected `--cpu-fast-softmax` option.
Matrix outputs and probability inputs use row-major storage. The returned CPU
buffer has token-major physical storage and a head-major logical view, allowing
the following model transpose to remain a view. The captured 128/64/128 FP16
command was replayed with known operands and matched the row-major oracle;
the generalized geometry range still requires hardware and model qualification.

The CPU runtime recognizes an explicit attention descriptor. Ordinary execution
and native replay both acquire existing NPU ownership, retain prepared resources,
account for both submissions, check errors, and propagate device failures.
Packed buffers and command descriptors survive calls and are released with
the device. The optional parked CPU workers run only the host phases.
Integer lengths and static query counts use CPU integer buffers for the
descriptor's pointer arguments. Symbolic lengths and counts retain their dynamic
values; both forms can be scheduled before the NPU is opened.

This mixed-precision attention changes FP32 queries and probabilities to FP16
at the matrix boundaries. Its numerical oracle models those casts explicitly;
model quality must be assessed separately against the original checkpoint and
RKLLM. Correct repeated IDs alone do not qualify a speed or quality win.

Grouped FP16 matrix ABI 7 separately marks a dynamic right-hand operand; ABI 8
adds the explicit row-major matrix layout. Their
ordinary runtime and native replay validate both matrices on every execution.
ABI 1 and all existing immutable projection-weight guards retain their contract.
The combined attention path validates live cache operands during packing and
does not use immutable-weight assumptions for K/V.

Validation includes CPU packing/layout/masking oracles, invalid and poisoned
input checks, changed-cache hardware execution, and native replay. A complete
model comparison is required before selecting this path for API serving.
