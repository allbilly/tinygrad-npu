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
`docs/rknpu/results/repro-paths-{a,b,c}.md`. Two facts that bite:

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

### Reporting — every measurement carries a comparison table

After any measured number, in the same turn, report it against the reference tree
and RKLLM. A number with no comparison is a reading, not a result.

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

### Opencode

- In a interactive session, just plan dont run tasks without approval
- In a goal session, do all tasks with ur suggested actions do not ask user questions except remove non cacahe files need explict user approval 
- In opencode, u can create subagents teammates with roles and offload tasks to them, all doc review task shd do in subagents