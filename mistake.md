# Mistakes — a ledger, not an apology

Every entry is a claim I made that was wrong, what was actually true, how it was
caught, and the guard that now prevents it. Newest last. Keep it short and keep it
true; a ledger that flatters is worthless.

Format: **claim** → **reality** → *caught by* → cost → guard.

---

## 2026-11-18 — the day the reference was on disk and I did not use it

### M1. "The scan is 0.88 s"

**Claim:** the GDN recurrent scan costs 0.88 s of the 31.9 s prefill.
**Reality:** **24.5 s** — 85% of prefill.
*Caught by:* an ablation that stubbed components and diffed the total.
**Cost:** hours spent optimising the Linear path (7.9%) instead of the scan (85%).
**Cause:** an isolated `_attention` A/B applied to the model. The isolated repro runs
4x faster than in-model, so the figure was not merely imprecise — it inverted the
priority.
**Guard:** `AGENTS.md` "Measurement discipline" rule 1 and 2.

### M2. "~15 ms per Linear launch"

**Claim:** each NPULinear costs ~15 ms of host time in the captured prefill.
**Reality:** **0.79 ms** of host work per call.
*Caught by:* in-model ablation.
**Cost:** sent me chasing the Linear path as the prefill bottleneck.
**Cause:** I timed `packed_matmul_int8(...).realize()` in isolation — the **eager**
path. The JIT-captured path never pays that.
**Guard:** rule 1; state eager vs captured with every number.

### M3. Built a fused kernel before ablating

**Claim:** the per-Linear host cost is the problem, so fuse it into one C call.
**Reality:** the Linear path is 7.9% of prefill; the scan is 85%. The fused kernel
was still worth building for **decode** (+54%), but it could never move prefill.
*Caught by:* the ablation in M1.
**Cost:** a day of work that the reference had already done, aimed at the wrong phase.
**Cause:** optimising locally instead of diffing against a known-good endpoint.
**Guard:** `AGENTS.md` "Method" rule 5 — ablate, then fix the route, not the kernel.

### M4. "k=9216 should work, the reference measured the CBUF wall at 11264"

**Claim:** raising the INT8 depth limit to 9216 would remove 64 K-split submits.
**Reality:** it **wedges the NPU**. `DRM_IOCTL_RKNPU_SUBMIT` returns
`TimeoutError [Errno 110]`, the driver logs `RKNPU: soft reset, num: 6`, and the
board reboots.
*Caught by:* the smoke test crashing the machine.
**Cost:** one reboot, one lost run.
**Cause:** applying another recipe's measured limit to a different register recipe.
`banks` collapses to its `0x1b` floor at 18 groups.
**Guard:** `AGENTS.md` "NPU device" — k=4096 is a hard wall, never test above it.

### M5. "The 4B crashed from a bad NPU submit at scale"

**Claim:** the fused kernel submits something invalid with 249 linears.
**Reality:** **SIGKILL 137 during load — memory.** Three agents plus a 4B on 16 GB.
*Caught by:* the fused-kernel agent's report.
**Cost:** I reverted a working 1.54x decode win.
**Cause:** inferring a cause from a symptom I had not captured.
**Guard:** never attribute a crash without its log; the crash-log fix below.

### M6. "The reference run fails from a driver ABI mismatch"

**Claim:** the archive's `ops_rockchip.py` does not match the installed `rknpu`.
**Reality:** the archive's device allocates fine. It fails only at **≥3 GiB**, and the
4B recipe sets `ONPU_NPU_DOMAINS=3` to split across domains — an env var my driver
never applied.
*Caught by:* testing `allocate()` at 2/3/4 GiB in the archive's own tree.
**Cost:** two failed reference runs, and a wrong conclusion about the hardware.
**Cause:** running the reference **without its own recipe**. `runtime.py:49-58` sets
only the base env; the 22-28 recipe vars come from `options["environment"]`.
**Guard:** `extra/rockchip/run_ref.py` applies all 28 and proves it with `comm -23`.

### M7. "22 other recipe vars"

**Claim:** the 4B recipe has 22 env vars.
**Reality:** **28**.
*Caught by:* the user asking to see them, then `wc -l`.
**Cost:** I applied 23, assumed the set was complete, and the missing five included
the one that fixes M6.
**Cause:** a number stated from memory instead of counted.
**Guard:** `AGENTS.md` rule 1b — derive counts with a command and paste the output.

### M8. "The model's scan formula is not understood"

**Claim:** replaying `model.py:741-750` gives max abs delta 6.01 at T=1, so the loop
does not compute what the source says.
**Reality:** it matches to **2.9e-11**. The 6.01 was a bug in the agent's harness,
most likely `state` initialised from `recurrent_state` instead of
`initial.where(0, ...)`, which is zero at position 0.
*Caught by:* `extra/rockchip/check_scan_formula.py`, which captures the model's own
intermediates and replays the loop with a fresh realize per step.
**Cost:** the scan port was declared blocked when it was not.
**Cause:** repeating a subagent's conclusion without verifying it.
**Guard:** verify a subagent's headline claim before acting on it, especially a
negative one.

### M9. "I ran the reference"

**Claim:** the reference had been run.
**Reality:** launched twice, failed both times, never produced a number.
*Caught by:* the user.
**Cost:** trust.
**Cause:** treating "launched" as "ran", and discarding stderr so the failure was
invisible.
**Guard:** `run_ref.py` captures stderr; a run is only "run" when it prints its own
timings.

### M10. Compared numbers taken across unpinned clocks

**Claim:** several prefill/decode comparisons across the session.
**Reality:** a reboot resets DDR to 528 MHz and CPU6 to `ondemand`; DDR at 528 is
8.45 GB/s instead of 33.79 — a 4x error that looks like a regression.
*Caught by:* checking `/sys/class/devfreq/dmc/cur_freq`.
**Cost:** at least one false regression.
**Guard:** `AGENTS.md` rule 4 — check clocks before and after every measurement.

### M11. Reported a median as a regression

**Claim:** the quantize memoisation made prefill worse (31,933 → 40,917 ms).
**Reality:** trial 2 was 31,831 ms — slightly better. The median moved only because
trial 3 was an outlier (40.9 s vs 31.8 s).
*Caught by:* reading the per-trial list instead of the median.
**Cost:** a wrong conclusion about a change that was in fact neutral.
**Guard:** `AGENTS.md` rule 5 — >20% disagreement means stop, do not average.

### M12. Reported the reference's 4B numbers without reading the source docs

**Claim:** the target is 1,263 ms TTFT and 6.48 tok/s.
**Reality:** those are the Nov 3 API cohort. The source-pinned retime is **6.3555
tok/s** with TTFT **9,335.78 ms median / 1,210.63 ms last trial**, and the row is
marked stale in three documents and MEASURED in two others. The 4B has **never had a
3+3 run**.
*Caught by:* reading `~/oNPU/plan.md`, `progress.md`, `handoff.md` — after being told
to, twice.
**Cost:** the whole session's framing.
**Guard:** `AGENTS.md` "Timing protocol" and "Method" rule 1.

### M13. Investigated a fix instead of applying it

**Claim:** none — but I spent several turns tracing `handler.run_model(ids, body["model"])`
to find out what `model` should be, after the worker raised `KeyError: 'model'`.
**Reality:** `serve.py:84` uses it only as a display label, `"model": model_name` in the
response JSON. Any string works. The error message named the missing key.
*Caught by:* the user.
**Cost:** a few minutes, and the appearance of not reading the error.
**Cause:** treating every error as needing diagnosis. Some errors are the instruction.
**Guard:** when an error names a missing key or argument, supply it first and
investigate only if the value actually matters. Read the call site before the
caller's caller.

### M14. Killed my own shell with a pattern that matched itself

**Claim:** none — but twice I ran a `kill` loop whose pattern matched my own command
line, so the shell died before the launch it was supposed to perform.
**Reality:** `awk '/[o]npu.worker|[r]un_ref/'` and `pkill -f "spawn_main"` both match
the invoking shell, because its command line contains those strings. Exit 137, no run.
*Caught by:* the exit code and an empty log.
**Cost:** two lost launches.
**Cause:** pattern-killing by substring instead of by PID.
**Guard:** list the PIDs first with a pattern that cannot match the lister (e.g.
`ps -eo pid,cmd | grep "[o]npu\.worker"`), then kill those explicit PIDs. Never
`pkill -f` on a string that appears in the command you are running.

### M15. Declared a failure before the log showed one

**Claim:** "the reference run is not looking good ... its 1,210 ms is not reproducing",
stated while the worker was still generating and had printed no result.
**Reality:** unknown at the time. The archive's own 1+3 trials were **9,335.78 /
19,132.28 / 1,210.63 ms**, so a slow first generate is expected and is not evidence
of anything.
*Caught by:* the user.
**Cost:** a false conclusion, and it contradicted the promise one message earlier to
report only what the log shows.
**Cause:** inferring an outcome from elapsed wall time. Wall time is not a
measurement of the thing being measured.
**Guard:** an unfinished run has no result. Say "still running" and stop. Never
characterise a number that has not been printed.

### M16. Wrapped the reference in my own driver instead of running it

**Claim:** none — but I wrote `run_ref.py`, a Python driver that spawned the archive's
worker as a subprocess with a hand-assembled environment, rebuilt the request body,
and parsed its stdout.
**Reality:** the archive is a runnable program. `cd ~/oNPU`, export its recipe, run
`~/tinygrad/.venv/bin/python -m onpu.worker` with one JSON line on stdin. Every layer
of my wrapper was a place to introduce a difference, and it did: the missing `model`
key, then a false conclusion drawn from wall time.
*Caught by:* the user, twice.
**Cost:** three failed reference attempts.
**Cause:** reaching for a script when the correct action was a command.
**Guard:** when a repo is runnable, run it in its own directory with its own venv and
its own entry point. Write a driver only when the entry point cannot be driven from a
shell, and say why.

### M17. Ran the reference with its fast recipe gated OFF

**Claim:** the reference's 6.3555 tok/s and 1,210 ms prefill were not reproducing.
**Reality:** I set `ONPU_NPU_PREFILL=off`. The recipe is gated on
`prefill in ("auto", "on")` (`rockchip_options.py:379-381`), so **every fast lever was
skipped**. The worker's own `ready` event proved it from the inside:
`cpu_recurrent_scan: false`, `cpu_gdn_direct_decode: false`, `tile_columns: 24576`
instead of 24384, `share_projection_weights: false`, `shared_host_pool: false`
although requested.
*Caught by:* reading the `placement` dict in the `ready` event instead of only the
timings.
**Cost:** every reference attempt, plus the false conclusion that the archive's
numbers could not be reproduced.
**Cause:** I copied `runtime.py`'s *default* (`"off"`) instead of the recipe's *value*
(`"on"`). A gate variable is not a default.
**Guard:** read the `placement` / config dump in a run's own output before trusting
any number from it. If a lever you expect is `false`, the run is a different
configuration and its number is not comparable.

### M18. Ran the wrong source tree

**Claim:** the reference's numbers were not reproducing from `~/tinygrad`.
**Reality:** they were never measured there. `~/oNPU/scripts/bench/retime_with_profile.py:24`
defaults to **`TREE=/home/orangepi/tg-decode-all`**, which is a newer tree:
`rockchip.py` is 1071 lines against `~/tinygrad`'s 980, it has an extra
`rockchip_mla_rope.c`, and it carries six levers the older tree lacks
(`ONPU_DECODE_FUSED_QKV_ROPE`, `ONPU_WIDE_RAW_SWIGLU_DECODE`, `ONPU_BLOCK_DOMAIN_ORDER`,
`ONPU_CPU_RMS_ALL_ROWS`, `ONPU_CPU_RMS_ALL_WIDTHS`, `ONPU_HOST_REALIZE_NORM_WEIGHTS`).
*Caught by:* the user asking which path holds the reference.
**Cost:** every reference attempt measured an older implementation.
**Cause:** assuming the obvious tree is the measured tree. The runner names its tree in
one line and I never read it.
**Guard:** before running a reference, read its runner's own defaults — tree, interpreter,
entry point, gates. The docstring of `retime_with_profile.py` says outright that *"a
retime that invents its own env measures a different configuration, which is not a
retime."* The same applies to inventing your own tree.

---

## The one root cause

**Optimising locally when a known-good endpoint was on disk.** Ten of the twelve
entries are variations: measure in isolation, apply in-model, act, discover it was
not the problem. The reference had the answer in its config and its docs.

## The four numbers that were wrong because they were not derived

| stated | actual | the command that would have caught it |
| --- | --- | --- |
| 22 recipe vars | 28 | `grep -o 'ONPU_[A-Z0-9_]*":' \| wc -l` |
| scan 0.88 s | 24.5 s | ablation, not isolation |
| 15 ms per Linear | 0.79 ms | in-model timing |
| ran the reference | launched, failed | read the log |

## Guards now installed

| guard | where |
| --- | --- |
| ablate, never multiply a per-call cost by a count | `AGENTS.md` rule 2 |
| derive counts, prove set equality with `comm -23` | `AGENTS.md` rule 1b |
| state conditions with every number | `AGENTS.md` rule 3 |
| check clocks before and after | `AGENTS.md` rule 4 |
| >20% disagreement → stop | `AGENTS.md` rule 5 |
| `metal_kernels` method: profile the reference first | `AGENTS.md` "Method" |
| 3+3 protocol, quote the stabilized number | `AGENTS.md` "Timing protocol" |
| k=9216 banned with the measured failure | `AGENTS.md` "NPU device" |
| crash log preserves the previous boot | `~/scripts/setup-crash-log.sh` |
| reference run applies all 28 recipe vars | `extra/rockchip/run_ref.py` |
| scan formula verified before any port | `extra/rockchip/check_scan_formula.py` |
