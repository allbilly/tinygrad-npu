# Reference Qwen3.5 4B on RK3588 — organised record

What this file is: everything this checkout knows about **how the reference runs Qwen3.5 4B
at 1.2 s prefill**, gathered in one place — the measured target, the exact command, the
measured decomposition, the mechanism, and **the kernel C source** with our port status.

It exists because the same facts were spread across `results/split/FINDINGS.md` (2,726
lines), five `results/ref-*` directories, and the archive trees. Numbers here carry their
source; where our side is quoted it is labelled ours.

Scope: Qwen3.5 4B only. Other models' reference rows are in `npu-gemm-baseline.md`.

---

## 1. The artifacts — they are not interchangeable

| path                              | what it is                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| `~/oNPU`                          | the harness **and the recipe**: `onpu/worker.py`, `runtime.py`, `app.py`, `rockchip_options.py`  |
| `~/tg-qwen35-pack-signature-fix`  | the tree the **Qwen3.5 4B row came from**: `1b2a5db` + a 4-file dirty patch, diff sha `7f87f221…` |
| `~/tg-commit`                     | clean `1b2a5db`, the Qwen3-VL 2B row                                                           |
| `~/tinygrad`                      | the tree most *other* published rows were measured from                                        |
| `~/tg-decode-all`                 | default `TREE` of `retime_with_profile.py`; **failed** for these models (`cycle detected`)      |

`1b2a5db` alone is **not** the measured configuration for 4B. A clean checkout of that
commit does not reproduce the row; the dirty patch is part of the identity. Treat all five
as read-only evidence.

---

## 2. The measured target

The reference's native prefill is **1,201.4 ms**, reproduced in this checkout in two runs.

| source                              | value                                    | what it measures                                      |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------------- |
| `results/ref-profile/report.json`   | 1200.39 / 1201.41 / 1201.21 / 1203.12 ms | `first_token_ms`, 4 consecutive stable trials         |
| `results/ref-phase/report.json`     | 1200.09 / 1198.62 ms                     | trials 3-4 of the second run                          |
| `results/ref-phase2/report.json`    | 1205.70 ms                               | one native trial, same run as the phase split         |
| `results/ref-phase2` decode         | 6.476 tok/s (trial 3: 6.4665, trial 4: 6.4857) | median of trials 3-4                           |
| `results/ref-profile` decode        | 6.52-6.54 tok/s                          | trials 4-7                                            |

All seven `ref-profile` trials, so the stabilisation is visible rather than asserted:

    17219.07  9236.56  26274.37  1200.39  1201.41  1201.21  1203.12  ms

**Pre-registered rule: quote trial 4 onward.** The first three calls are 9-26 s in *both*
trees — the same pathology, not a property of either implementation. Spread over the four
stable trials is 0.23%. The 1,201.4 figure is the report's `median_first_token_ms` over all
five measured trials including the 26,274 ms one; the four stable trials average 1,201.3, so
the median is not flattered by excluding the first-trial artifact.

The published archive row is **1,263.09 ms / 6.48 tok/s**, a **median over 20 requests**
(`~/oNPU/docs/benchmark-qwen-latest-audit-20261103.json`), measured under a different protocol
(64 generated tokens, 1 warm-up + 3 runs). It is not the target here: `1,201.4` is, because it
is the same 128-token / 8-token / 2+5 protocol this checkout can reproduce.

### Protocol and conditions

| item               | value                                                          |
| ------------------ | -------------------------------------------------------------- |
| prompt             | 128 tokens, `staged/w8a8-sweep-v1/inputs/qwen35-4b.txt`         |
| generated          | 8 tokens                                                       |
| warm-ups / runs    | 2 / 5                                                          |
| chunk size         | 128                                                            |
| context            | 512                                                            |
| CPU affinity       | `taskset -c 4-7`                                               |
| clocks recorded    | NPU 1 GHz, DDR 2112 MHz, CPU4/6 2256000 (`initial_clocks`)       |
| device             | `CPU` with all Linears on `ROCKCHIP` (`--device CPU --precision int8`) |

The published archive rows ran at adaptive governors; the reproduction above recorded
maximum on all three domains. **A reproduction at pinned clocks that matches the published
row means the published row was not flattered by its clocks.**

---

## 3. Reproduce it

Do not retype this. The env and argv below are the ones `extra/rockchip/ref_kernel_profile.py`
**executed** (it reproduced 1,263.35 / 1,205.67 ms and the reference's full per-kernel profile),
taken from `results/ref-profile/run.sh`, which is itself the recorded configuration. Five flags
were once dropped and made prefill run eager — 35,200 submits instead of 475 (`mistake.md` M22) —
so a hand-built line is a different configuration, not a reproduction.

```bash
export ONPU_REPLAY_PROFILE=1
export DEV=CPU HALF=1 JIT=1 CAPTURING=1 PARALLEL=0 PROFILE=0 VIZ=0 DEBUG=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/orangepi/tg-qwen35-pack-signature-fix
export ONPU_BENCHMARK_CHECKPOINT_SHA256=10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1
export ONPU_CPU_DECODE_POOL=0 ONPU_CPU_GQA_PREFILL=0 ONPU_CPU_PREFILL_LAYOUT=1
export ONPU_CPU_TOKEN_INPUTS=1 ONPU_EXP2_FAST_PATH=1 ONPU_FFN_PACKED_UPLOAD=1
export ONPU_GDN_DECODE_PREP_POOL=1 ONPU_GDN_DECODE_SPIN=1 ONPU_GDN_DIRECT_DECODE=1
export ONPU_GDN_PREP_POOL=1 ONPU_HOST_POOL_WFE=1 ONPU_INT8_DECODE_NARROW_CBUF=1
export ONPU_INT8_EXPERT_DEPTHS=0 ONPU_INT8_NATURAL2304=1 ONPU_INT8_SDK_PREFILL=0
export ONPU_NPU_DOMAINS=3 ONPU_NPU_DOMAIN_BYTES=3221225472 ONPU_PREFILL_CPU_TUNING=1
export ONPU_REPLAY_BORROW_BATCH=1 ONPU_REPLAY_BORROW_OUTPUTS=1 ONPU_REPLAY_DEDUP_GUARDS=1
export ONPU_REPLAY_NATIVE_FROM=1 ONPU_REPLAY_NATIVE_FROM_LARGE=1 ONPU_REPLAY_SCALAR_COPY=1
export ONPU_SHARED_HOST_POOL=1 ONPU_SWIGLU_PACK_STATIC_LAYOUT=1 ONPU_SWIGLU_PREFILL_ROWS=4
export ONPU_ZERO_PAD_VIEWS=1

cd /home/orangepi/tg-qwen35-pack-signature-fix
taskset -c 4-7 /home/orangepi/tinygrad/.venv/bin/python -u -m extra.rockchip.benchmark_inference \
  --checkpoint /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  --input /home/orangepi/vlm-bench/baseline-20261021/main-task-20261023/staged/w8a8-sweep-v1/inputs/qwen35-4b.txt \
  --device CPU --precision int8 --prefill on --decode on --context 512 --tokens 8 \
  --warmups 2 --runs 3 --chunk-size 128 --tile-columns 24384 \
  --fuse-projections --cpu-attention --native-replay --fused-ffn --split-down --fused-qkv \
  --cpu-norm --cpu-fast-softmax --cache-cleaning --wake-helper --cpu-pool --cpu-prefill-pool \
  --share-projection-weights --cpu-recurrent-scan --fused-swiglu-ffn --fused-swiglu-decode \
  --fused-qkv-without-norm --fused-recurrent-projections --cpu-q8-embedding --scalar-decode-loop \
  --trim-preparation-cache --file-backed-cpu-weights --defer-weight-realization \
  --report /home/orangepi/pr/tinygrad/docs/rknpu/results/ref-kernels/report.json
```

`--cpu-pool` and `--cpu-prefill-pool` are **CLI flags**, not `ONPU_CPU_POOL` / `ONPU_CPU_PREFILL_POOL`
environment variables; the latter two do not exist in either tree.

Run it from the reference tree, or the import resolves to this checkout and fails. The interpreter
is `/home/orangepi/tinygrad/.venv/bin/python` — the archive's venv, not `~/pr/tinygrad/.venv`.

For the per-kernel split, use the probe instead of the raw module: it attaches to the harness's own
`observe()` prefill boundary and to `CPUProgram.__call__`, and writes `kernel-profile.json` plus
every compiled kernel's C source:

```bash
cd /home/orangepi/tg-qwen35-pack-signature-fix
/home/orangepi/tinygrad/.venv/bin/python \
  /home/orangepi/pr/tinygrad/extra/rockchip/ref_kernel_profile.py
```

---

## 4. Where 1,206 ms goes — measured per kernel

`results/ref-phase2/phases.jsonl` splits every window into host CPU time and time blocked in
`rknpu_submit_ioctl`:

| phase   | wall ms | main CPU ms | process CPU ms | cores busy | off-CPU ms |
| ------- | ------: | ----------: | -------------: | ---------: | ---------: |
| prefill |   1,206 |     **720** |          5,140 |    ~4 ± 0.3 |        486 |
| decode  |   1,090 |         180 |          4,590 |       4.21 |        910 |

Read the last column with its instrument: the sampler polls at 0.1 s over a 1.2 s window, so
±0.1 s × 4 cores is ±0.33 cores, and the run is `taskset -c 4-7` — a **four-CPU mask**. The
process was therefore **saturated at its four cores**, not measured above them.

`5,140 ms` is **thread-seconds**, not work. The same run records per-kernel wall time, and the
CPU programs total **736.6 ms** over 1,597 calls; 736.6 ms at four threads is ~2,950 ms of thread
time, and the pool's spin-wait (`ONPU_HOST_POOL_WFE`) supplies the rest. Both figures are
consistent, and neither implies more than four cores.

### The per-kernel decomposition (the measurement that was missing)

From `native_execution.prefill.profile` in the same report — the reference's own per-kernel wall
clock for one native prefill (1,205.70 ms). Full table with our column:
`results/ref-kernels/compare.md`; machine-readable: `results/ref-kernels/kernel-profile.json`.

| item                    | ms     | calls | note |
| ----------------------- | -----: | ----: | --- |
| CPU programs            |  736.6 | 1,597 | wall, all four threads running |
| NPU programs            |  428.0 |   475 | two NPU domains (`ROCKCHIP`, `ROCKCHIP:1`) |
| device→host copies      |   35.4 |   758 | |
| blocked in submit ioctl |    486 |     — | phase probe |

Their CPU top-5: `rockchip_cpu_gated_delta_scan` 177.4 ms / 24 = 7.39 ms/call,
`E_2_16_32_32_4_4` 99.3 / 48 = 2.07, `rockchip_int8_swiglu_pack` 53.7 / 32 = 1.68,
`E_32_32_32_4_4` 51.3 / 24 = 2.14, `E_64_16_64_4` 51.0 / 16 = 3.19.

### Three kernels exist under the same name and shape in both trees

| kernel            | ref ms/call | our ms/call | our/ref |
| ----------------- | ----------: | ----------: | ------: |
| `E_2_16_32_32_4_4` |        2.07 |        6.46 | **3.12x** |
| `r_32_16_4_32_4`   |        0.95 |        2.93 | **3.09x** |
| `E_32_32_128_4`    |        1.98 |        5.83 | **2.95x** |
| all CPU programs   | 736.6 total | 2,417.7 total | **3.28x** |

A uniform ~3.1x on identical generated code is one mechanism applied to all of them — four
threads inside one call — not a better kernel. It matches the only clean measurement in this
tree (3.70x on `E_393216_3`, `results/threadsplit/README.md`).

Programs and submits in the same run:

|                                  | prefill            | decode                  |
| -------------------------------- | ------------------ | ----------------------- |
| CPU programs                     | **1,597**          | 1,086                   |
| NPU programs                     | 475                | 203                     |
| NPU submits                      | **475** (1:1)      | 1,421 (203/step, ×7)    |
| CPU programs per NPU program     | 3.4                | 5.3                     |

Two things this rules out as explanations:

- **Not fewer programs.** They run **more** CPU programs than we do (1,597 vs our ~1,006)
  and are still 2.1x faster. Program count is not the discriminator.
- **Not prefill submit count.** 475 vs our 355 is 1.34x in *our* favour, yet they win.

Our side, for comparison (from `results/split/FINDINGS.md` and the in-model probe):

| metric                | ours              | reference         |
| --------------------- | ----------------: | ----------------: |
| prefill               | 2,532 ms          | 1,201.4 ms        |
| decode                | 3.22 tok/s        | 6.52 tok/s        |
| CPU programs/prefill  | ~1,006            | 1,597             |
| CPU kernel time       | 2,391.3 ms        | 736.6 ms          |
| outside the kernels   | 154.2 ms          | not measured      |
| decode submits/step   | 307               | 203               |

---

## 5. The mechanism, in code

Three things produce the reference's four-wide execution. All three are visible in the sources.

### 5.1 Persistent worker pool, with an in-kernel serial fallback

Some fused CPU kernels (`rockchip_norm_quantize`, the scan) take the **address of the pool
dispatch function** as an integer and
calls through it, falling back to serial if the pool declines
(`rockchip_norm_quant.py:77-79`):

```c
void rockchip_norm_quantize(unsigned char *raw,const float *input,const float *weight,float *scale_out) {
  for(int i=128*4;i<header;i++)raw[i]=0;
  Work work={raw,input,weight,scale_out};
  void *args[4]={&work,0,0,0};
  if(((int(*)(void(*)(void**,int,int),void**,int))ADDRESS_ULL)(part,args,0))
    for(int i=0;i<4;i++) part(args,0,i);
}
```

The workers are created once by the pool (`--cpu-pool --cpu-prefill-pool`,
`ONPU_CPU_POOL=1`, `ONPU_HOST_POOL_WFE=1`), so a dispatch is a wake, not a
`pthread_create`. This is the piece we do not have.

### 5.2 Threshold-based self-threading inside a single kernel

Independent of the pool, the kernels spawn four threads themselves above a size threshold:

| kernel            | condition                 | at 4B prefill (rows=128)          |
| ----------------- | ------------------------- | --------------------------------- |
| `rockchip_cpu_rmsnorm` | `rows*width >= 131072` | 128 x 2560 = 327,680 → **4 threads** |
| `rockchip_swiglu_pack` | `rows*depth >= 32768`  | 4 rows x 9216 = 36,864 → **4 threads** (1.13x the threshold, so marginal) |

This one *is* transferable to our tree, because it is entirely inside the kernel — it does
not depend on the reference's scheduler. Note that it is per-kernel, not per-graph: it
cannot saturate four cores on its own; it is the per-kernel split repeated across kernels
that does.

### 5.3 Native replay

`--native-replay`, `ONPU_REPLAY_*`: the captured graph replays without re-entering the
Python executor per program. This is why 1,597 programs cost them 720 ms of main-thread
time, while our 1,006 programs measured 2,391.3 ms inside the kernels and 154.2 ms outside
them (`results/progcost/report.json`).

---

## 6. Kernel code

Quoted from `~/tg-qwen35-pack-signature-fix/tinygrad/llm/kernels/` (the tree the row came
from). The `rockchip_cpu.py` text is byte-identical to
`~/oNPU/patches/tg-decode-all-20261109/rockchip_cpu.py` apart from one decode-only fusion
block — verified with `diff -U0` (2 hunks, both that block).

Our `E_*` kernel inventory at the same shapes is in `results/progcost/report.json`; the
labels (`E_2_16_32_32_4_4`, `r_32_16_4_32_4`, …) are tinygrad codegen names.

### 6.1 `exp2` polynomial — the arithmetic every fast path shares

`rockchip_cpu.py:588-590`. A minimax degree-6 fit for `2^s` on `[-0.5, 0.5]`, applied after
range reduction, with the integer part added to the exponent field:

```c
static __attribute__((always_inline)) float4 rockchip_exp2_unchecked(float4 x) {
  int4 q=__builtin_convertvector(x+(x<0?(float4)-.5f:(float4).5f),int4);
  float4 s=x-__builtin_convertvector(q,float4);
  float4 y=((((((0.00015359208919107914f*s+0.0013392627006396651f)*s+0.009618384763598442f)*s+
        0.055503472685813904f)*s+0.24022644758224487f)*s+0.6931471824645996f)*s+1.0f);
  return __builtin_bit_cast(float4,__builtin_bit_cast(int4,y)+q*(int4)0x800000);
}
```

Guarded variant `rockchip_exp2_vector` (`:615`) keeps this path only while
`max|x| < 0x42fa0000` (`|x| < 125`), else calls a checked path that handles inf/nan and
`x >= 128` / `x < -150`. Sigmoid is then one multiply and one reciprocal
(`:655-658`):

```c
static __attribute__((always_inline)) float4 rockchip_sigmoid_vector(float4 value) {
  float4 x=value*-1.4426950216293335f;
  return 1.0f/(rockchip_exp2_vector(x)+1.0f);
}
```

There is also `_sigmoid_source()` (`:56`), which is not hand-written: it calls
`to_program()` on a tinygrad graph, reads the `Ops.SOURCE` child, and marks the result
`always_inline`. **The reference generates some of its C with tinygrad too** — same
renderer, same ABI.

### 6.2 `rockchip_cpu_rmsnorm` — with an `exact` variant that matches tinygrad bit for bit

`rockchip_cpu.py:500-565`. The comment states why two arithmetic variants exist, and it is
the most useful thing in the file:

```
#   1. the variance reduces through ONE scalar accumulator with the square multiply
#      contracted into an FMA (acc = fma(v,v,acc)). A float4-lane accumulator rounds
#      the square separately and drifts 1 ulp on ~25% of rows.
#   2. .rsqrt() lowers to a DIVISION (x/sqrt(t)), not x*(1/sqrt(t)). The reciprocal
#      rounds twice and differs on ~28% of elements.
```

```c
// exact=true: bit-identical to tinygrad's generic elementwise decomposition
float acc=0.0f;
for(int c=0;c<2560;c+=4) {float4 v=*(const float4*)(src+r*2560+c);
  acc=__builtin_fmaf(v[0],v[0],acc); acc=__builtin_fmaf(v[1],v[1],acc);
  acc=__builtin_fmaf(v[2],v[2],acc); acc=__builtin_fmaf(v[3],v[3],acc);}
float denominator=__builtin_sqrtf(acc*0.000390625f+9.999999974752427e-07f);
for(int c=0;c<2560;c+=4) {float4 value=*(const float4*)(src+r*2560+c),gamma=*(const float4*)(weight+c);
  *(float4*)(out+r*2560+c)=((value/(float4)denominator))*gamma;}

// exact=false: lane accumulator + reciprocal, faster, 1 ulp off
float4 acc=(float4)0; ... float inverse=1.0f/__builtin_sqrtf(sum*0.000390625f+eps);
... (value*(float4)inverse)*gamma;
```

**Port status: not ported.** We run generated `r_*` code, single-core. The `exact` variant
is the one to port first, because it is provably bit-identical to what we already compute.

### 6.3 `rockchip_norm_quantize` — RMSNorm + INT8 quantisation + NPU packing in one kernel

`rockchip_norm_quant.py:46-83`. Three ops that we run as three programs, fused into one
four-partition kernel with a NaN guard:

```c
static void part(void **args,int unused,int worker) {
  Work *w=args[0];unsigned char *raw=w->raw;const float *input=w->input;const float *weight=w->weight;
  float *scales=(float*)raw;signed char *output=(signed char*)(raw+header);
  int begin=128*worker/4,end=128*(worker+1)/4;
  for(int first=begin;first<end;first+=4) {
   int count=end-first<4?end-first:4;float values[4][2560] __attribute__((aligned(16))),inverse[4];int invalid[4];
   for(int r=0;r<count;r++) {
    const float *x=input+(first+r)*2560;float4 acc=(float4)0;
    for(int c=0;c<2560;c+=4) {float4 v=*(const float4*)(x+c);acc=acc+v*v;}
    float sum=((acc[0]+acc[1])+acc[2])+acc[3];
    float inv=1.0f/__builtin_sqrtf(sum*0.000390625f+9.999999974752427e-07f);
    float4 maximum={0};uint4 bad={0};
    for(int c=0;c<2560;c+=4) {float4 v=*(const float4*)(x+c),g=*(const float4*)(weight+c);v=(v*(float4)inv)*g;
     *(float4*)(values[r]+c)=v;uint4 bits=__builtin_bit_cast(uint4,v);
     bad|=(bits&(uint4)0x7f800000)==(uint4)0x7f800000;float4 a=v>=(float4)0?v:-v;maximum=a>maximum?a:maximum;
    }
    float largest=maximum[0];for(int j=1;j<4;j++)if(maximum[j]>largest)largest=maximum[j];
    invalid[r]=__builtin_reduce_or(bad)!=0;float scale=largest/127.0f;if(scale<1e-8f)scale=1e-8f;
    scales[first+r]=w->scales[first+r]=invalid[r]?__builtin_nanf(""):scale;inverse[r]=1.0f/scale;
   }
   for(int c=0;c<padded;c+=16)for(int r=0;r<count;r++)for(int j=0;j<16;j+=4) {
    float4 v=(c+j<2560&&!invalid[r])?*(const float4*)(values[r]+c+j):(float4)0;
    *(char4*)(output+(first+r)*padded+c+j)=rockchip_quantize_vector(v,inverse[r]);
   }
  }
}
```

Three details worth copying even without the pool:

- the **NaN/inf guard** writes NaN into the scale and zeros the row, instead of emitting a
  poisoned int8 row that silently ruins a matmul;
- `scale<1e-8 -> 1e-8` avoids a division blow-up on an all-zero row;
- the output loop writes **padded** int8 in the NPU's 16-wide tile order, so no separate
  layout pass exists.

**Port status: not ported.** We norm in one program, quantise in `qz_row` in another.

### 6.4 `rockchip_swiglu_pack` — activation + fp16 cast + NPU layout, with 4 threads

`rockchip_cpu.py:677-750`. This is the direct replacement for our worst elementwise group.
It computes `gate*sigmoid(gate)*up`, converts to `__fp16`, and writes the NPU's
`(c/8, row, 8)` packed geometry in one pass:

```c
// emitted body when input_rows != 0 (vectorised path)
for(int c=begin;c<end;c+=4)for(int row=0;row<128;row++) {
  const int start=row/4*4,m=128-start<4?128-start:4;
  const int g=row*18432+c,u=g+9216;              // gate and up interleaved in the combined buffer
  float4 gate=*(const float4*)(src+g),up=*(const float4*)(src+u);
  float4 reciprocal=rockchip_sigmoid_vector(gate);
  float4 activated=gate*reciprocal,value=activated*up;
  *(half4*)(out+start*9216+(c/8*m+row-start)*8+c%8)=__builtin_convertvector(value,half4);
}

// dispatch: 4 pthreads over columns when rows*depth >= 32768
for(int i=0;i<4;i++)work[i]=(Work){out,src,9216*i/4,9216*(i+1)/4};
for(int i=0;i<3;i++){created[i]=pthread_create(&threads[i],(void*)0,pack_worker,&work[i])==0;
  if(!created[i])pack_part(out,src,work[i].begin,work[i].end);}
pack_part(out,src,work[3].begin,work[3].end);
for(int i=0;i<3;i++)if(created[i])pthread_join(threads[i],(void**)0);
```

`tile_rows=4` comes from `ONPU_SWIGLU_PREFILL_ROWS=4`. The `input_rows` variant walks
projection tiles (`swiglu_pack_tiles`), which is how the FFN avoids materialising gate and
up separately.

**Port status: not ported.** Ours is `gated_activation` + a separate cast/store, measured
at 0.31 s in-model (`E_2_16_32_32_4_4`, 48 calls) plus more for the transpose. A standalone
float4 SwiGLU as a standalone kernel lost 52 ms and a standalone vectorised sigmoid lost
75 ms in-model — the win here is the *fusion of the
store layout*, not the polynomial alone, which is exactly what this kernel does and the
standalone attempt did not.

### 6.5 `rockchip_softmax_vector`

`rockchip_cpu.py:662-675`, used with `--cpu-fast-softmax`:

```c
static void rockchip_softmax_vector(float *values,int count,float maximum) {
  float sum=0;int t=0;
  for(;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);
    float4 y=rockchip_exp2_vector((x-(float4)maximum)*(float4)1.4426950216293335f);
    __builtin_memcpy(values+t,&y,16);
    sum+=y[0];sum+=y[1];sum+=y[2];sum+=y[3];}
  for(;t<count;t++){values[t]=expf(values[t]-maximum);sum+=values[t];}
  for(t=0;t+3<count;t+=4){float4 x;__builtin_memcpy(&x,values+t,16);x=x/(float4)sum;__builtin_memcpy(values+t,&x,16);}
  for(;t<count;t++)values[t]/=sum;
}
```

**Port status: ported** as the attention kernel (`extra/rockchip/attn.c`, `attn_native.py`).
In-model effect: **±0** over 24-48 calls — the attention route, not its arithmetic, is what
costs us (`FINDINGS.md` "the fix was the route, not a new GEMM").

### 6.6 `rockchip_quantize_vector` — round-to-even, not round-half-away

`rockchip_int8.py:22-35`. Uses the ARM instruction directly:

```c
static __attribute__((always_inline)) float4 rockchip_round_even(float4 x) {
  #if defined(__aarch64__)
  float4 result;__asm__("frintn %0.4s, %1.4s":"=w"(result):"w"(x));return result;
  #else
  return (float4){__builtin_nearbyintf(x[0]),__builtin_nearbyintf(x[1]),__builtin_nearbyintf(x[2]),__builtin_nearbyintf(x[3])};
  #endif
}
static __attribute__((always_inline)) char4 rockchip_quantize_vector(float4 x,float inverse) {
  float4 q=rockchip_round_even(x*(float4)inverse);
  q=q>(float4)127?(float4)127:q;q=q<(float4)-127?(float4)-127:q;
  return __builtin_convertvector(q,char4);
}
```

Note `-127`, not `-128`. **Port status: equivalent exists** in our `qz_row`, but this is the
authority for the rounding mode; a 1-ULP difference here is how a faster, wrong config once
looked convincing (`mistake.md` M47).

### 6.7 GDN scan

`extra/rockchip/recurrent.c` in **our** tree is the reference kernel, including
`cpu_gated_delta_scan_pooled(..., scan_parallel_fn parallel, int *dispatch_status)`. We call
`cpu_gated_delta_scan(..., NULL)`, so every scan runs on one core. The reference invokes it
**24x per prefill**, identical to ours (`ref-profile` reports 96 because it runs four
replays; swiglu 128 vs 32 shows the same 4x). Chunking was measured and refuted as a
standalone result (see §8), not as a reference comparison.

---

## 7. Exporting tinygrad kernels to C — the shared ABI

Both trees compile CPU kernels to the same shape, so a port is a **source swap at one ABI**,
and a comparison is a `diff` of arithmetic. The compiler invocation is
`runtime/support/compiler_cpu.py:21`:

```
clang -c -x c -O2 -fPIC -ffreestanding -fno-math-errno -nostdlib -fno-ident \
      --target=arm64-none-unknown-elf -ffixed-x18 -mcpu=native - -o -
```

and the entry point is always

```c
void <kernel_name>(void *args);   // args = packed buffer addresses, one 8-byte slot each
```

Three ways to get our C out:

| method                       | how                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------- |
| environment                  | `DEBUG=4` prints the generated source at `codegen/__init__.py:477` per kernel            |
| programmatic                 | `to_program(sink, ClangRenderer(Target('CPU', arch='arm64,native', renderer='CLANG')))`, then read the `Ops.SOURCE` child — exactly what `_sigmoid_source()` does |
| **one file per kernel**      | `extra/rockchip/dump_ckernels.py` (this checkout) hooks `ClangCompiler.compile` and writes `<name>.c` |

```bash
# enumerate our kernels at Qwen3.5 4B shapes (quick: 16-token prompt, no warm-up)
.venv/bin/python extra/rockchip/dump_ckernels.py --out docs/rknpu/results/cdump -- --quick
# full 128-token prefill, NPU-enabled
.venv/bin/python extra/rockchip/dump_ckernels.py --out docs/rknpu/results/cdump -- \
  --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf \
  --name qwen35-4b --out /tmp/ignore --npu --warmups 0 --runs 1
```

Then compare, for example, `cdump/E_2_16_32_32_4_4.c` against §6.4: the difference is that
ours materialises a float tensor and the store/transpose is a *separate* program, while the
reference folds activation, fp16 cast, packed geometry and four-thread split into one.

---

## 8. What transfers, and what was measured *not* to

Everything below was measured in this checkout, in-model unless stated. Full context and
retractions are in `results/split/FINDINGS.md`.

| item                                        | measurement                                            | verdict        |
| ------------------------------------------- | ------------------------------------------------------ | -------------- |
| m-tile merge of submits                     | 603 → 355 submits, 2,703 → 2,614 ms, bitwise identical | **landed**     |
| scan buffer reuse                           | 2,614 → 2,532 ms, parity `4.209e-07`                   | **landed**     |
| scan chunking (1x128 vs 2x64 vs 4x32)       | 1x128 = 42.04 ms; 2x64 = 56.13; 4x32 = 74.76 — and not numerically equivalent | **refuted** |
| float4 SwiGLU as a standalone kernel        | bit-identical, **+52 ms** in-model                      | refuted        |
| vectorised sigmoid as a standalone kernel   | bit-identical, **+75 ms** in-model                      | refuted        |
| attention kernel replacement                | **±0** over 24-48 calls                                 | refuted        |
| 4 Python threads over head-slices           | **0.24x** — CPU device serialises                      | refuted        |
| pthreads inside the scan kernel             | correct, **+100 ms** in-model                           | refuted        |
| pure-C 4-thread scan (no tinygrad)          | **1.21x** ceiling — memory-bound                        | ceiling        |
| descriptor route for the merge              | ~66 ms/Linear vs 2.04 ms/call fused (`rockchip_linear_64_2560_4096`, 196 ms / 96) | refuted |
| m=128 at k>2048                             | `banks → 0x1b`; the geometry that wedged the NPU        | closed         |

Two conclusions follow directly:

1. **A kernel port only wins when it also removes a store boundary.** All three
   single-expression ports lost; the two ports that won (scan, `qz_row`) both changed the
   data layout or the call count. §6.3 and §6.4 are both of that kind — each replaces
   *three* of our programs, not one.
2. **The remaining gap is execution width, not arithmetic and not the scheduler.** The same
   generated kernel runs 3.0-3.1x faster in the reference (§4), and the graph itself has
   almost no inter-kernel parallelism to exploit (4 workers = 1.23x on our trace), so the
   lever is four threads *inside* each kernel. That is §5.2, applied to every kernel; §5.1's
   pool is one way to dispatch them, not the mechanism by itself.

---

## 9. Provenance

| claim                                        | file                                                       |
| -------------------------------------------- | ---------------------------------------------------------- |
| 1,201.4 ms, 7 trials, 2,072 programs         | `results/ref-profile/report.json`                           |
| CPU/NPU phase split, ~4 cores saturated, 486 ms off-CPU | `results/ref-phase2/phases.jsonl`, `report.json` |
| per-kernel decomposition of the 1,201 ms     | `results/ref-kernels/kernel-profile.json`, `compare.md`     |
| exact reproduce command + env                | `results/ref-profile/run.sh`                                |
| kernel C                                     | `~/tg-qwen35-pack-signature-fix/tinygrad/llm/kernels/rockchip_{cpu,norm_quant,int8}.py` |
| scan kernel                                  | `extra/rockchip/recurrent.c` (this tree, ported)            |
| our per-kernel in-model times                | `results/progcost/report.json`                              |
| our full decomposition and retractions       | `results/split/FINDINGS.md`                                 |
| advisor numbers (TTFT 1,263.09, decode 6.48) | `~/oNPU/benchmark.md:1179`, `plan.md:35,221`                |
| RKLLM comparison                             | `~/oNPU/benchmark.md` (61.87 tok/s prefill, 6.22 tok/s decode) |
