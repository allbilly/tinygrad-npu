# Native NPU output synchronization

`ONPU_REPLAY_NATIVE_FROM=1` opts native CPU/NPU replay into direct ARM cache
maintenance for eligible NPU outputs. The default is zero. It changes output
synchronization only; matrix commands, packing, quantization and arithmetic stay
the same. Ordinary replay and unsupported mappings keep driver synchronization.

A qualifying copy must immediately follow its blocking NPU producer and use the
same allocation identity, address and size. The allocation must have exactly the
ordinary cacheable/noncontiguous/IOVA-aligned flags, clean CPU version metadata,
a page-aligned allocation base, a 64-byte-aligned view, and a total allocation
size no larger than 4 MiB. Kernel-mapped buffers and unmatched views use the driver.
The existing replay lifetime and output-cleanliness guards remain required.

The helper uses `DC CIVAC` followed by `DSB SY`. A child process checks instruction
access and a 64-byte cache line before activation. Failure leaves the driver
path available. The raw DMA qualification additionally checks new NPU values
against independent INT32 matrix products before any driver invalidation for
that input. Instruction access by itself does not establish DMA visibility.

`NativeReplay.invalidate_sites` and `driver_from_sites` record which generated
copy sites select each path. These are site counts; native replay runs must also
be observed before claiming that the direct path executed.

Run the CPU guard tests:

```sh
DEV=CPU python -m pytest test/runtime/test_rockchip_native_from.py -q -n12 --dist=loadfile
```

With exclusive NPU ownership, qualify changed outputs and input rebinding through
the actual generated replay:

```sh
ROCKCHIP_TEST=1 DEV=CPU python -m pytest test/runtime/test_rockchip_native_from.py \
  -k changed_integer_outputs_survive_native_invalidation_and_rebinding -q -n12 --dist=loadfile
```

The hardware test covers single-row decode, a partial prefill, and a 3 MiB
prefill result, with eight input patterns under both synchronization settings.
Model measurements and fixed-token numerical comparisons are recorded separately
in oNPU's benchmark documents. Transfer-only gains do not establish a model win.
