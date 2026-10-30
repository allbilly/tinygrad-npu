# RK3588 NPU memory domains

`ROCKCHIP` and `ROCKCHIP:1` address different IOMMU memory domains on the same
NPU. They share one DRM descriptor, submission lock, health state and counters;
they do not add compute cores. Each matrix keeps its input, weights, output and
prepared commands in one domain. The current driver UAPI uses a 48-byte
MEM_CREATE and a 104-byte SUBMIT with its domain ID at offset 32.

INT8 packing can opt into sequential domain placement with
`ONPU_NPU_DOMAINS=2`. `ONPU_NPU_DOMAIN_BYTES` defaults to 3 GiB per domain,
leaving address space for commands and activations. The default domain count is
one. Native replay records each prepared plan's domain. Quantization boundaries,
scales, layouts and integer arithmetic do not change.

The capacity probe retained 4.5 GiB of mapped buffers across two domains while
24 small INT8 matrix checks passed before and after allocation. Distinct backing
objects used identical virtual DMA addresses in the two domains. Grouping calls
by domain was faster than alternating domains: roughly 29 microseconds per
additional switch in that small-operation test. This does not qualify every
larger model or prove every mapped byte was read by the NPU.

Qwen3.5-4B now completes three warmups and three measured 128-input/64-output
requests at context 512, using `--defer-weight-realization` to avoid retaining
all eager CPU weights. It records 32.20 prefill proxy tok/s and 4.84 decode tok/s.
The recorded RKLLM reference is 61.87 and 6.22; this is a timing candidate, with
output quality and complete INT8 model coverage still unqualified. The raw
record is in `~/oNPU/docs/benchmark-npu-domains-qwen35-4b-v3.json` and the related
verification is `~/oNPU/docs/benchmark-npu-domains-verification-v1.json`.

A later gated CPU-attention follow-up records 41.74 prefill proxy tok/s and
5.23 decode tok/s for Qwen3.5-4B. Those sequences differ from the earlier path;
quality remains unqualified. The current comparison and exact source cohorts
are in `~/oNPU/benchmark.md` and
`~/oNPU/docs/benchmark-gated-attention-verification-v1.json`.

The model timing precedes allocation/finalization changes in `ops_rockchip.py`;
separate hardware tests cover those changes. One later pre-copy health guard is
covered by a CPU regression and the full CPU checks. The report preserves these
source cohorts. Device finalization drains all live
mappings before releasing ownership. Deferred allocator frees become no-ops,
and the shared DRM descriptor closes only after the last domain releases it.
The ownership probe keeps raw buffers alive across finalization, checks that the
other domain retains its lock, then frees the old buffers while a replacement
process owns the NPU and verifies that process's exact matrix output.

With the normal NPU worker unloaded, repeat the ownership check using:

```sh
DEV=CPU python -m extra.rockchip.probe_domains --report /tmp/npu-domains.json
```

Native replay also tracks overwritten byte intervals before removing a CPU pack
copy, and compares actual ranges for adjacent views of one NPU allocation. Its
regression cases retain copies when stale bytes can be read, writes are partial,
or ranges overlap. Multi-domain and ordinary matrix replay hardware tests pass.
The server's UI does not expose the new domain configuration yet.
