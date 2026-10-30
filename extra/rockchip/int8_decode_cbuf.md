# INT8 decode buffer allocation

A read-only GDB capture of the existing Qwen3-0.6B RKLLM W8A8 model showed
one input bank and eleven weight banks for single-row reductions. The earlier
oNPU recipe allocated input banks as though at least 64 rows were present.
Both recipes programmed weight banks; this change corrects the allocation for
qualified K=4096 operations with per-core N=512 or N>=1024. All other register words,
packing, integer arithmetic, activation scales and prepared-command reuse stay
identical. Prefill keeps the earlier allocation.

Measured on this RK3588S with both original serving sessions restored afterward.
The timings include the blocking prepared runtime call, Python guards and ioctl;
input uploads and output copies are excluded. Each comparison uses 3 warmups
and 40 alternating ABBA/BAAB blocks, 80 timed calls per variant. Ratios below are
medians of paired block ratios, which suppress drift visible in separate medians.

| Full output width N | K | Paired speedup |
|---:|---:|---:|
| 3,072 | 4,096 | 1.119× |
| 4,096 | 4,096 | 1.141× |
| 6,144 | 4,096 | 1.141× |
| 12,288 | 4,096 | 1.152× |
| 24,576 | 4,096 | 1.160× |

A later shape-specific repeat measured 1.087× for total N=1536 (per-core
N=512), K=4096, with exact independent integer outputs. That narrow case also
passed seven changed signed-int8 inputs through TinyJit and native replay,
including alternating -128/127 activations. Its guard does not include the
untested per-core N=704 shape. Full-model effects are measured separately.

Every paired block in these five cases favored the new allocation. The two
widest cases also reproduced the first sweep's paired ratios. The unchanged
64-row control was within 0.2% of parity in the repeat. Smaller K cases did not
show a consistent benefit and retain the earlier recipe.

Across the first sweep, bank repeat, and experimental full-depth sweep, all
204 integer-output checks matched the independent int32 matrix product exactly.
This covers random activations, all-127 activations, and alternating -128/127.
Steady-state NPU allocations were zero.

A separate **experimental** full-depth probe tested K=4608,6144,7168,8192 at
N=2048,6144. One full-depth matrix submission gave 1.171–1.360× speedups against two existing
K<=4096 matrix submissions (three core tasks per submission). Baseline CPU partial-output copying/addition was excluded. The
full-depth model path is not enabled: changing projection quantization boundaries
needs separate quality testing and prefill layout qualification.

[Raw timing samples, numerical hashes and restoration evidence](int8_decode_cbuf_results.json)
record the measured source cohort and original external raw report hashes.
These primitive results do not change the model comparison table or establish
an RKLLM model win. The hardware regression test passed 20 further exact-output checks with changed
inputs through ordinary TinyJit capture and replay. A separate tiny operation
ran correctly in memory domains 0, 1, and 0, restoring the original domain.
That probe establishes domain selection only; simultaneous mappings, larger
capacity, and model performance remain unqualified.

To repeat the regression test run it with exclusive NPU ownership:

```sh
ROCKCHIP_TEST=1 DEV=CPU python -m pytest test/runtime/test_rockchip_int8.py \
  -k wide_decode_integer_outputs_survive_replay -q -n12 --dist=loadfile
```
