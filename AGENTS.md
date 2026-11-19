# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
- tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules.

## oNPU RK3588 work (this checkout)

`~/pr/tinygrad` is the only worktree on main. Do not create extra worktrees or
experiment subfolders; new code lives under `extra/rockchip/`.

`~/oNPU`, `~/tinygrad` and **`~/tg-decode-all`** are **archives**: read them for
evidence and design. Never write to them, never import their trees into this one.
Results and logs belong in this checkout, under `docs/rknpu/results/`.

**Which archive is which — they are not interchangeable, and there is no single
"reference tree":**

| path | what it is |
| --- | --- |
| `~/oNPU` | the harness **and the recipe**: `onpu/worker.py`, `runtime.py`, `app.py`, and `onpu/rockchip_options.py` (the per-checksum option registry) |
| `~/tinygrad` | the tree **most published rows were measured from**, via `staged/<cohort>/probe.py` → `python -m onpu --backend ~/tinygrad` → `onpu.worker`, protocol **3 warm-ups + 10 measured** |
| `~/tg-qwen35-pack-signature-fix` | the tree the **Qwen3.5 4B/2B** rows came from, at `1b2a5db` **plus a 4-file dirty patch whose diff sha256 is the real identifier** (`7f87f221…`). A clean checkout of that commit is *not* the measured configuration |
| `~/tg-commit` | the tree the **Qwen3-VL 2B** row came from, `1b2a5db`, clean |
| `~/tg-decode-all` | the default `TREE` of `retime_with_profile.py` — **a different harness that failed for these models** (`qwen2-0.5b/report.json`: `complete: false`, `trials: 0`, `error: "cycle detected in assign graph"`). Do not assume it is the measured tree |
| `~/vlm-bench` | **data only** — GGUF checkpoints under `baseline-20261021/` and run logs. No code |

**Do not assume a single reproduce path.** Per-model detail is in
`docs/rknpu/results/repro-paths-{a,b,c}.md`. Three facts that bite:

- **Use the recorded command, verbatim.** Every archive run writes its exact argv to
  `receipt.json` (`.command`). Read it from the run being reproduced; never retype a
  flag list. A hand-built line that dropped five GDN flags made prefill run eager
  (35,200 submits instead of 475) for five runs (`mistake.md` M22). **A submit count
  that differs from the reference by more than a few percent means the configuration
  is different, not the machine.**

- **Source epochs are partly gone.** Each probe asserts `source_before ==
  manifest.source_sha256`; recomputing today, 26-33 pins per cohort differ. The
  command is exact; the tree it pinned is no longer on disk.
- **Protocols differ by cohort**: 3+10 for the `staged/*/probe.py` rows, 1+3 for the
  Qwen3.5 retimes, 3+3 for the corrected rule. Quote the protocol with the number.

Running a tree and calling the result "the reference" is a different implementation,
not a retime. `retime_with_profile.py`'s own docstring says it: *"a retime that invents
its own env measures a different configuration, which is not a retime."* The same is
true of inventing your own tree.

### Running

- Interpreter: `~/pr/tinygrad/.venv/bin/python` (3.12, numpy, tinygrad installed
  editable). Do not use any other venv or system python.
- No `PYTHONPATH` needed: the editable install resolves `tinygrad` to this tree.
- Baseline device: the model runs on the CPU host with every `nn.Linear`
  offloaded to the NPU GEMM (`bench_llm.py --npu`). `DEV=ROCKCHIP` is the NPU
  device itself; `DEV=CPU` alone is a reference only, not the baseline.

### NPU device

- `tinygrad/runtime/ops_rockchip.py` and `tinygrad/runtime/autogen/rockchip.py`
  are the ported RK3588 device. Only the `Compiled`/`Program` signature changes
  were needed: `renderers` is a class attribute and `Program` takes `links`.
- Single owner: it takes `/tmp/rk3588_npu_submit.lock` and opens
  `/dev/dri/card1`. Run one NPU process at a time.
- Smoke test: `~/pr/tinygrad/.venv/bin/python extra/rockchip/smoke_matmul.py`
  must print `max_abs_err=0.0000 rel=0.00000`.
- Matmul layouts: lhs `(m,k) -> (k/8,m,8)`, weight
  `(n,k) -> (n/16,k/32,16,32)`, out `(n/4,m,4) -> (m,n)`.
- Limits (FP16 recipe as ported): `m<=64` (`m<=128` in `row_major`), `k<=4096`,
  `n<=4096` per core (`n<=8192` per core when `m==1`; with `cores=3` the call
  limit is 12288 / 24576), `k%32==n%32==0`. These are recipe bounds, not silicon
  maxima. The ported code also enforces `m*k <= 65536` through
  `banks = ceil(m*k*2/32768) <= 4` (written into `REG_CNA_CBUF_CON0`), so the
  wrapper uses `row_tile = min(64, 65536//k)` and splits K at 4096.
- Reference evidence (`~/oNPU/CHATGLM_DECODE_ANALYSIS.md`): the reference reports the
  measured CBUF wall at `k <= 11264` (22 groups of 512) for INT8 and `m=64, k=5632`
  runs; `m=64, n=4608` needs 3 cores because 4096 is the per-core prefill width.
  **That 11264 figure does not transfer to this recipe.** Do not raise
  `_int8_supported_depth` above 4096.
- **k=9216 wedges the NPU. Never test it again.** Measured: `m=1, n=2560, k=9216,
  cores=3` makes `DRM_IOCTL_RKNPU_SUBMIT` return `TimeoutError [Errno 110]`, the
  driver logs `RKNPU: soft reset, num: 6`, and the board can reboot. The cause is
  the `banks` formula collapsing to its `0x1b` floor at 18 groups
  (`177 - 15*(18-1) = -78`, clamped to 27). So 4096 is a **hard wall for this
  register recipe**, not a conservative cap, and `ffn_down` (k=9216) must stay
  split into three K tiles. 323 submits per forward is the floor.
  Do not add a smoke shape above k=4096 and do not run one by hand.
- **Reboots are caused by a bad NPU submit, not by heat.** The two reboots on
  2026-11-18 both followed an invalid task: the k=9216 test produced a bad
  register program, the NPU wedged, the driver logged `RKNPU: soft reset, num: 6`
  and the board restarted. Temperature reached 85 °C during normal runs without a
  reboot. So the protection is **submit correctness** — verify geometry against
  the working path before running a new shape — not waiting for a cool board.
  Cooling still matters for *timing validity*, not stability.
- **The Qwen3.5 reference gate is 52-55 °C and it fails by being TOO COLD.** The
  board idles near **51.8 °C**, just under the band, so a run started from idle
  never enters it: the archive's own 0.8B attempt timed out after 7,202.6 s with
  zero in-band samples and a maximum of 51.769 °C. Being too cold is not a
  correctness problem — it is a gate artefact. Start the run **hot** (60-80 °C)
  and let it cool through the band, or warm the board into 52-55 °C before
  starting. Do not conclude the run is impossible because the gate says so.
- **After any NPU wedge or reboot, check for orphaned benchmark processes before
  starting another run.** A wedged NPU leaves the submit lock held by a dead
  process, and a second run then fails with "Rockchip NPU is owned by another
  worker" or, worse, competes for the device. `fuser /tmp/rk3588_npu_submit.lock`
  and `ps -eo pid,cmd | grep bench_llm` first, and kill by explicit PID.
- **An invalid submit is what wedges the board, and a wrong pointer is enough.**
  `probe_kdepth.py` passed a device DMA address where the fused C function
  dereferences a CPU pointer (`w_scale` is read by the CPU; the weight is read by
  the NPU). The driver logged `RKNPU: soft reset, num: 6` eight times; the board
  recovered and `smoke_matmul.py` still printed `max_abs_err=0.0000`. So: for every
  buffer, decide CPU address (`buf.addr`) vs device address (`buf.dma`) before the
  call, and run the smoke test after any crash.

### The `NPU programs` row is NOT device time — do not quote it as such

This mis-attribution cost a day and produced a wrong conclusion ("the only route is a
kernel fence"). The fused INT8 Linear is a **CPU PROGRAM**: one C function that does
quantize -> `sync_mem(TO_DEVICE)` -> blocking ioctl -> `sync_mem(FROM_DEVICE)` ->
dequantize. Its wall duration therefore contains CPU work, and the 603.2 ms "NPU
programs" row in `results/compare-ours-vs-ref.md` is that fused wall, not the device.

Measured split (`docs/rknpu/results/threadsplit/ioctlprobe/run.log`, 355 calls):

| component | ms | nature |
| --- | --: | --- |
| `submit` (blocking ioctl) | 423.5 | **device** — `submit_seconds`, also printed as `0.4s in npu` |
| `dequantize` | 106.7 | CPU, single-threaded, inside the fused call |
| `sync_from` | 17.2 | CPU cache maintenance |
| `sync_to` | 6.6 | CPU cache maintenance |
| quantize + remainder | ~49 | CPU, NEON, single-threaded |

`bench_llm` already prints the device part: `prefill 1.4s 355 submits 0.4s in npu`.
**That `0.4s` is the NPU row. Compare it to the reference, not the 603 ms.** At 423.5
against the reference's 428.0 ms the NPU is at parity, so the NPU is not the gap and
no fence or async submit is required to match 1.2 s. RKLLM and the reference both hit
their numbers with the same blocking submit and `fence_fd=-1`.

The addressable gap is therefore the **host half of the fused call** (~155 ms of
quantize + dequantize + syncs, single-threaded) plus the ~155 ms outside every kernel.
Both are CPU-side and parallelisable; neither needs a driver change.

### Reporting — every measurement carries a comparison table

After any measured number, in the same turn, report it against the reference tree
and RKLLM. A number with no comparison is a reading, not a result.

**During optimization, the comparison is a class-level decomposition table, not a running
commentary.** Keep it as a file, update it in the same turn a number changes, and drive the work
from its largest delta. It has exactly these columns:

| class | ours | reference | delta | provenance (which protocol produced each column) |

Rules, learned the hard way (`mistake.md` M74, M76):

- **Same classes, same units, a delta column, provenance per column.** Two rows (device/host) is not
  a decomposition; a per-kernel dump is evidence behind it, never the answer — this tree and the
  reference share **one kernel name out of 31 and 36**, so per-kernel joins are impossible.
- **When two protocols give two totals for the same configuration, the table carries both and says
  why.** Here: one call per replay segment gives the class shares (958 ctypes calls, reads higher),
  the shipped 64-call segments give the headline. Quoting one in place of the other is what makes
  successive answers look like different stories.
- **A row is "open" until its delta is closed by a paired measurement**, not by an isolation
  benchmark: kernel time in isolation has repeatedly failed to reach the wall (fills 19.3 -> 4.2 ms
  moved the wall not at all; a 27%-faster attention kernel made decode 9.4% slower).

| metric | ours | oNPU ref | RKLLM | vs ref | vs RKLLM |
| --- | --: | --: | --: | --: | --: |

Qwen3.5 4B canonical values, so the table does not have to be re-derived each
time. Prefill proxy is `128000 / TTFT(ms)`.

| metric | oNPU ref | RKLLM | source |
| --- | --: | --: | --- |
| TTFT, 128 tokens | 1263.09 ms | 2068.88 ms | `~/oNPU/benchmark.md:1179` |
| prefill proxy | 101.34 tok/s | 61.87 tok/s | `~/oNPU/benchmark.md:978` |
| decode | 6.48 tok/s | 6.22 tok/s | `~/oNPU/benchmark.md:1179` |
| decode step | 154.22 ms | — | 138.66 + 14.95 + 0.61 |
| — device | 138.66 ms | — | `~/oNPU/benchmark.md:1392` |
| — CPU kernels | 14.95 ms | — | `~/oNPU/benchmark.md:1392` |
| — copies / sync | 0.61 ms | — | `~/oNPU/benchmark.md:1392` |

State the clock basis every time, because it decides which way the gap moves:

| | clocks |
| --- | --- |
| ours | pinned: NPU 1 GHz, DDR 2112 MHz, CPU6 2256000 |
| oNPU ref | adaptive governors, no sudo tuning (`benchmark.md:1329`) |
| RKLLM | published at maximum clocks |

The reference running at adaptive clocks and still winning means our gap is
**understated**, not flattered.

**Split decode into device and host.** The device has been within 8% of the
reference since the INT8 port, so a blended tok/s figure hides the only thing
that matters. Report `decode_npu_ms_per_token` alongside the rate, and say which
part is host.

### Method — copy `metal_kernels`, do not invent one

`https://github.com/allbilly/metal_kernels` is the working method for this class of
problem, and this project broke every rule in it for a day. Its README states the
sequence:

1. **Profile the reference engines first.** Its §1.3/§1.4 are kernel breakdowns of
   `mlx_lm` and `oMLX` taken *before* any change. Here that means running
   `~/oNPU` on the model under study and diffing its call path against ours —
   not searching for wins by local measurement.
2. **Compute speed-of-light from measured bandwidth, and quote the percentage.**
   Its decode SOL is `1.505 GB/token / 58.2 GB/s = 25.86 ms -> 38.7 tok/s`, so
   36.9 tok/s is 95% of SOL. Ours is `4.2 GB/token / 33.79 GB/s = 124 ms ->
   8.04 tok/s`; we measure 1.51, i.e. **19% of SOL**. That single number says the
   gap is not the matmul, and it says so before any kernel work.
3. **Paired A/B in ABBA rounds.** `mlx_lm, oMLX, oMLX, mlx_lm` per round; compare
   the two cross-engine pairs as ratios, because *"the ratio cancels thermal/DVFS
   drift"*. Five rounds gives ten pairs. Never compare two numbers taken at
   different times on this board.
4. **Report wins as pair counts**, not medians: "9/10", "10/10". A median hides a
   configuration that wins half the time.
5. **Ablate to find the cause, then fix the route, not the kernel.** Its oMLX
   prefill was 4.4% behind and the cause was the attention *route*, not the GEMM —
   gating it off measured +5.6%, 10/10. It also tried a new GEMM and rejected it:
   *"0.581 TFLOP/s vs 1.640 for mx.matmul (0.35x), so the fix was the route, not a
   new GEMM."*
6. **Retract invalid measurements explicitly**, in the document, rather than
   quietly superseding them.
7. **Before building anything, name the existing entry point and say why it cannot
   be used.** Four of this project's mistakes were building a thing that already
   existed: a fused kernel (the reference had it), a target table (the docs had
   it), a worker driver (a shell pipe was enough), a scan harness (five lines
   settled it). If you cannot name the existing entry point, you have not looked.
   This rule comes *before* the decision to build, not after it.

The failure this prevents: writing a kernel to fix a problem whose cause was never
located. That is what happened here — a fused per-Linear kernel was built before
ablation showed the scan was 85% of prefill.

### Measurement discipline — a number is only valid with its conditions

Every one of the wasted hours in this project came from the same error: taking a
number measured under one set of conditions and using it under another. Two
concrete cases, both mine:

- **"the scan is 0.88 s"** — that was an isolated `_attention` A/B. The in-model
  cost is **24.5 s** (85% of prefill). An isolated repro of the same loop runs
  4x faster than in-model, so the isolated figure was not merely imprecise, it
  inverted the priority.
- **"~15 ms per Linear launch"** — that was the **eager** path
  (`packed_matmul_int8(...).realize()` in isolation). The JIT-captured path pays
  **0.79 ms** of host work per call. The 15 ms was never paid by the thing being
  optimised.

Rules that follow:

1. **Never quote an isolated measurement as an in-model number.** Isolation
   excludes the executor, the cache state and the surrounding graph. If the
   in-model number is what matters, measure in-model.
1b. **Never state a count, a size or a set from memory — derive it with a command
   and paste the output.** Four wrong numbers in one session came from this and
   nothing else: "22 recipe vars" (28), "scan 0.88 s" (24.5 s), "15 ms per Linear"
   (0.79 ms), "ran the reference" (launched, failed twice). When the claim is
   "all of X are applied", prove it with a set difference (`comm -23`) and show
   that it is empty.
2. **Attribute by ablation, never by multiplication.** To find where time goes,
   stub one component and diff the total. Do not multiply a per-call cost by a
   call count — that assumes the per-call cost is unchanged in-model, which is
   exactly the assumption that keeps failing.
3. **State the conditions with the number**: eager or JIT-captured, isolated or
   in-model, clocks pinned or not, `DEBUG` or clean, warm or cold. A number
   without its conditions cannot be compared to anything, including a later
   number from the same session.
4. **Check the clocks before and after every measurement.**
   `cat /sys/class/devfreq/dmc/cur_freq` must read `2112000000` and
   `/sys/devices/system/cpu/cpufreq/policy6/scaling_governor` must be
   `performance`. DDR at 528 MHz is 8.45 GB/s instead of 33.79 — a 4x error that
   looks like a regression. A reboot silently resets all of this.
5. **If two runs of the same thing differ by more than ~20%, stop and find out
   why before using either.** Do not average them, do not pick the convenient one.
   The 31.9 s vs 40.9 s prefill pair was one outlier trial, and reporting the
   median as a regression was wrong.
6. **`DEBUG=2` inflates CPU-bound kernels by up to 10x** because it prints per
   kernel. Use it for names and ordering only, never for timing. `PROFILE=1`
   collects viz events and prints no table in this tree.

### An A/B needs an in-process ABBA — `bench_llm` cannot do one

`bench_llm`/`bench_threaded` record `clocks_start`/`clocks_end` but **no per-trial temperature**,
and each invocation is a fresh process. Comparing two of them wall-to-wall attributes a thermal ramp
to whichever arm ran later. Measured cost of getting this wrong: `par-dq-3plus3` reported **+27.7 ms**
and `par-dq-big-3plus3` **+11.57 ms** against the baseline, both of which the fair method then showed
to be **neutral (0.9998x, 0.8% spread)**.

Rules that follow:

- An A/B of a code variant must run **both arms inside one process**, interleaved `A,B,B,A`, with
  temperature read from `/sys/class/thermal/thermal_zone0/temp` before every trial.
- If the variant is baked into generated source (a kernel emitter, a codegen flag), **drop the
  PROGRAM/PLAN caches between arms** or the second arm silently reuses the first arm's binary.
- Report **paired ratios**, the per-arm spread, and the temperature range. A single-arm spread above
  ~20% invalidates the pair.
- Reference implementation: `extra/rockchip/ab_fused_host.py`.

A 3+3 is still the right gate for an absolute number; it is the wrong tool for a comparison.

### `generated_ids` is sampled — it is not a parity gate

The baseline's own three runs in `threadsplit/scanpool-3plus3` diverge at **token 0**:
run0 `[141274, 166375, …]`, run1 `[182662, 201060, …]`, run2 `[185053, 51777, …]`. The model
samples, so identical ids are not expected from identical code and a mismatch proves nothing.

Use these instead:

| gate | artifact | deterministic |
| --- | --- | --- |
| model logits | `logits_sha256_16` per trial (`inmodel_program_cost.py`, `results/parity/nosplit`) | yes |
| per-kernel boundary | `extra/rockchip/parity_fused_linear.py` (serial emitter vs pooled emitter, bitwise) | yes |
| generated tokens | `results/*/report.json` `generated_ids` | **no — smoke only** |

`extra/rockchip/check_parity.py` now says so in its own output; it cannot award or deny parity.

### `--quick` is not comparable to a 128-token baseline — ever

`bench_llm.py --quick` sets `prompt=16, gen=8, runs=1, warmups=0` (the flag's own help text
says *"Never quote a --quick number as a measurement"*). Three consequences, all measured:

| effect | consequence |
| --- | --- |
| 16-token prompt instead of 128 | different prefill shapes, so different kernel names: `thread_split` reports **`transforms applied: NONE`**, meaning the four-way kernels are not even exercised |
| 0 warm-ups | the JIT capture and compile land inside the measured trial: 16.1 s where the warmed prefill is 1.4 s |
| different token count | `generated_ids` cannot equal a 128-token run's, so a parity check against a 128-token baseline always "fails" at token 0 |

So `--quick` is only good for "does it crash". It cannot A/B a change, cannot check parity, and
cannot show a regression either — a 16 s reading is compile time, not prefill. For parity use
`--prompt 128 --gen 64 --warmups 1 --runs 1` at minimum, and for a number use the full 3+3.

**Confirmed by control, not just by reading the flag.** The unmodified baseline source, run with
`--quick`, also "fails" a 128-token parity check at token 0 (`results/threadsplit/serial-quick-control`:
`transforms applied: NONE`, ids `[205565, 243740, …]` against the baseline's `[141274, 166375, …]`,
prefill 10.2 s). So a `--quick` parity failure says nothing about the change under test — run the
control before believing the failure.**

### Profiling the host half — py-spy, not REA flameox

Profiling host time on this board is possible **without root**:

- `.venv/bin/py-spy` (0.4.3, `uv pip install py-spy`) records a real sampling flame graph.
  `record -f speedscope -o out.json -- <command>` launches the child itself, so it never needs to
  attach to a running process.
- The `ptrace_scope` file is **absent** on this kernel, which is why same-user tracing works.
- `perf` is not installed and `perf_event_paranoid` is `2`, so hardware sampling is unavailable and
  would need root — py-spy covers the same question.
- REA's `flameox` is **not** in `rea-agents` 6.7.0 (the newest that runs on Node 24.12.0), and
  6.8+ needs Node ≥ 24.22.1. Do not upgrade Node to chase it: `nvm alias default` is the major
  `24`, so installing 24.22.1 would repoint PATH away from `v24.12.0/bin`, where `dsh` and `dst`
  live. Pin `nvm alias default 24.12.0` first if a second Node is ever needed.

Cheap alternatives when a full flame graph is overkill: `cProfile` for proportions only (its
absolute times are inflated), and direct timers for exact per-phase milliseconds.

### Timing protocol

- **3 warm-ups + 3 measured, not 1+3.** The reference confirmed this on MiniCPM4
  (`~/oNPU/plan.md:230`): *"measured TTFT 103.39 / 104.01 / 115.94 ms (median
  104.01 ms), decode median 57.10 tok/s, `timing_validity=PASS`"*. With 1+3 the
  first *measured* trial is still warming, and `plan.md:206-230` documents that
  across nine retimed models the last trial is **6.7-15.7x faster than the median
  of three** — the median of a 1+3 run is a first-trial artifact, not a prefill
  estimate.
- **Quote the stabilized (last-trial) number and say so.** The reference's own 4B
  row is median 9,335.8 ms but **last-trial 1,210.6 ms**, and only the latter is a
  real prefill figure. Our runs show the same shape: 67,246 / 31,933 / 31,749 ms,
  so our stabilized TTFT is 31,749 ms.
- 128-token prompt, 64 greedy decode tokens.
- Report every trial and the median. The oNPU reference tree saw a 6-15x TTFT
  spread from trial order; in this tree the CPU trials were within 6%. Never
  pick a trial post hoc without a pre-registered stabilization rule.
- `extra/rockchip/bench_llm.py` implements this and writes `report.json`.
- revalidate timing with a 52 to 55 oC cold start, depends on idle room temperature

### No config flags

- Derive config or use a module constant with the measurement in a comment; never hide it behind an env var.

### Commits

- Keep changes as an uncommitted working-tree diff. Do not commit unless asked.
- If commits are requested: small, focused, no amend commits (upstream rule).

### Markdown

- All markdown table shd fix spacing to plan text view readability
- **Never use a date in a filename under `docs/rknpu/`.** Name a document by what it contains
  (`reference-qwen35-4b.md`, `compare-ours-vs-ref.md`, `decode-rea.md`), not by when it was
  written. A dated name goes stale in the name itself, and two dated copies of the same subject
  become two answers. Put chronology in the file's text if it matters.

### Opencode

- In a interactive session, just plan dont run tasks without approval
- In a goal session, do all tasks with ur suggested actions do not ask user questions except remove non cacahe files need explict user approval 
- In opencode, u can create subagents teammates with roles and offload tasks to them, all doc review task shd do in subagents