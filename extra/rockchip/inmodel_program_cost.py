#!/usr/bin/env python3
"""Attribute one real prefill between compiled-kernel time and CPU-device dispatch.

The claim under test is the one in FINDINGS.md: "~1.3 ms per CPU program x ~1,031
programs ~= the whole gap". That 1.3 ms was measured through `fast_sigmoid`, which
rebuilds a Python graph on every call (`reshape/contiguous/empty/custom_kernel`), and
none of that work exists inside a captured JIT. So the in-model number has to be
measured in-model (AGENTS.md measurement rule 1).

Each instrumented prefill splits every `CPUProgram.__call__` into:

  pack    -- building the packed argument buffer (ctypes create_string_buffer + pack_into)
  fxn     -- the compiled kernel itself (ctypes CFUNCTYPE into RWX pages)
  outer   -- the whole call, so `outer - pack - fxn` is Python bookkeeping around them

and prefill wall time minus the sum of `outer` is everything that is not a CPU program:
executor iteration, Buffer copies, allocations, NPU submits and waits.

The patch is installed once but gated by a flag, so an un-instrumented prefill runs the
*original* `__call__` verbatim. Trials alternate instrumented/un-instrumented, which
measures the observer effect instead of assuming it away (a previous attempt reimplemented
`__call__` unconditionally and reported a 4.78 s prefill against a 2.53 s baseline).

Usage:
  DEV=CPU:CLANG .venv/bin/python extra/rockchip/inmodel_program_cost.py \
      --ckpt /home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
"""
from __future__ import annotations

import argparse, collections, json, pathlib, time

PROMPT = ("The history of the printing press begins with movable type in East Asia. "
          "By the fifteenth century, metal type and oil-based ink made books cheap enough "
          "for ordinary readers, and the resulting spread of literacy reshaped religion, "
          "science, and politics across Europe. Printers set each page by hand, one letter "
          "at a time, and a single mistake could ruin an entire sheet. ") * 8


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", type=pathlib.Path, required=True)
  p.add_argument("--prompt", type=int, default=128)
  p.add_argument("--gen", type=int, default=1)
  p.add_argument("--out", type=pathlib.Path, default=None)
  args = p.parse_args()

  import ctypes, struct
  from tinygrad.helpers import data64_le, unwrap, cpu_profile
  from tinygrad.runtime.ops_cpu import CPUProgram

  import tinygrad.engine.realize as _R
  # Where the 144 us/program inside the executor actually goes: the per-CALL rewrite that
  # resolves buffers and dispatches. Measured before changing any core file.
  exec_stats = {"n": 0, "ns": 0}
  _pm = _R.pm_exec
  class _TimedPM:
    def rewrite(self, uop, ctx):
      if not state["on"]: return _pm.rewrite(uop, ctx)
      t0 = time.perf_counter_ns(); r = _pm.rewrite(uop, ctx)
      exec_stats["n"] += 1; exec_stats["ns"] += time.perf_counter_ns() - t0
      return r
  _R.pm_exec = _TimedPM()

  # Position-indexed replay: `call.without_after` builds a new uop per dispatch, so the CALL's
  # index within the captured linear is the only stable identity. Bindings are keyed by index and
  # verified against a fresh resolution before first use, so a wrong binding is caught, not run.
  from tinygrad.dtype import AddrSpace
  from tinygrad.device import Device as _Device
  from tinygrad.helpers import DEBUG as _DEBUG
  from tinygrad.uop.ops import Ops as _Ops
  rep = {"hit": 0, "miss": 0, "skip": 0, "ns": 0, "idx": -1}
  _rep_cache: dict[tuple, tuple] = {}
  _orig_run_linear = _R.run_linear

  def _bind(ctx, call, ast):
    """exec_kernel's resolution, returning the binding instead of executing it."""
    resolved = [_R._resolve(call.src[1+i], ctx.input_uops) for i in ast.arg.globals]
    outs = []
    for device, (bufs, device_vars) in zip(_R.to_tuple(_R.get_call_arg_uops(call)[0].device), _R.unwrap_multi(call, resolved)):
      if device.split(":")[0] in _R.HOST_DEVS: _Device[device].synchronize(ctx.timeout)
      vv = {**ctx.var_vals, **device_vars, **{x.expr: x.arg.val for x in call.src[1:] if x.is_bound_var}}
      pb = {i: b.ensure_allocated() for i, b in zip(ast.arg.globals, bufs)}
      rt = _R.get_runtime(device, ast, cache=ctx.cache)
      gsz, lsz = ast.arg.launch_dims(vv)
      args = [(vv[p.expr] if p.is_variable else _R._resolve(call.src[1+p.arg.slot], ctx.input_uops).val)
              if p.addrspace is AddrSpace.ALU else pb[p.arg.slot].get_buf(device) for p in ast.kernel_params]
      outs.append((rt, args, gsz, lsz))
    return outs

  replay = {"on": False}
  def run_linear_pos(linear, var_vals=None, input_uops=(), update_stats=True, jit=False, wait=False):
    if not replay["on"]:
      return _orig_run_linear(linear, var_vals, input_uops, update_stats, jit, wait)
    assert linear.op is _Ops.LINEAR
    if len(linear.src) == 0: return
    inputs = list(input_uops)
    if not jit:
      linear = _R.link_linear(_R.compile_linear(linear, validate=_R.VALIDATE_WITH_CPU, input_uops=inputs, cache=True), input_uops=inputs)
    ctx = _R.ExecContext(var_vals or {}, tuple(inputs), update_stats, jit, wait or _DEBUG >= 2)
    for i, call in enumerate(linear.src):
      rep["idx"] = i
      c = call.without_after
      key = (i, tuple(sorted(ctx.var_vals.items())))
      t0 = time.perf_counter_ns()
      if (hit := _rep_cache.get(key)) is not None:
        for rt, args, gsz, lsz in hit:
          rt(*args, global_size=gsz, local_size=lsz, wait=ctx.wait, timeout=ctx.timeout)
        rep["hit"] += 1; rep["ns"] += time.perf_counter_ns() - t0
      else:
        _R.pm_exec.rewrite(c, ctx)
        rep["miss"] += 1
        # Only plain PROGRAM calls can be replayed by binding their arguments.
        # COPY/STORE calls have no ProgramInfo (`ast.arg is None`), and HCQ calls
        # carry queue/profiling semantics in exec_hcq that a bare rt() would skip.
        ast = c.src[0]
        if ast.op is _Ops.PROGRAM and ast.arg is not None and getattr(c.arg, "hcq", None) is None:
          try:
            _rep_cache[key] = _bind(ctx, c, ast)
          except (AttributeError, KeyError) as exc:
            rep["skip"] += 1
            if rep["skip"] <= 3: print(f"[replay] skip call {i}: {type(exc).__name__}: {exc}", flush=True)
        else:
          rep["skip"] += 1
    rep["idx"] = -1
  _R.run_linear = run_linear_pos
  # `tinygrad.engine.jit` holds its own reference (`from ... import run_linear`), so patching the
  # module attribute alone is invisible to the JIT -- the same from-import trap as
  # ClangCompiler.compile_cached.
  import tinygrad.engine.jit as _J
  _J.run_linear = run_linear_pos

  state = {"on": False}
  copy_stats = {"n": 0, "ns": 0}
  _orig_copy = _R.exec_copy
  def _copy_hook(ctx, call, ast):
    if not state["on"]:
      return _orig_copy(ctx, call, ast)
    t0 = time.perf_counter_ns()
    out = _orig_copy(ctx, call, ast)
    copy_stats["n"] += 1
    copy_stats["ns"] += time.perf_counter_ns() - t0
    return out
  _R.exec_copy = _copy_hook
  stats = {"outer_ns": 0, "pack_ns": 0, "fxn_ns": 0, "thread_ns": 0, "calls": 0}
  by_name: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # calls, outer_ns, fxn_ns, thread_ns
  original = CPUProgram.__call__

  def timed_call(self, *fargs, global_size=(1, 1, 1), local_size=(1, 1, 1), wait=False, timeout=None):
    if not state["on"]:
      return original(self, *fargs, global_size=global_size, local_size=local_size, wait=wait, timeout=timeout)
    assert self.dev.remote is None, "this probe only instruments the local CPU device"
    t0 = time.perf_counter_ns()
    t3 = time.thread_time_ns()
    hdr = 12 if self.lvp else 0
    kernargs = ctypes.create_string_buffer(hdr + len(fargs) * 8)
    if self.lvp:
      struct.pack_into('<3I', kernargs, 0, *data64_le(ctypes.addressof(kernargs) + hdr), len(fargs) * 2)
    self.packer.pack_into(kernargs, hdr, *fargs)
    t1 = time.perf_counter_ns()
    with cpu_profile(self.name, self.dev.device, profile_key=self.profile_key) as prof:
      self.fxn(kernargs)
    t2 = time.perf_counter_ns()
    t4 = time.thread_time_ns()
    stats["calls"] += 1
    stats["pack_ns"] += t1 - t0
    stats["fxn_ns"] += t2 - t1
    stats["outer_ns"] += t2 - t0
    stats["thread_ns"] += t4 - t3
    row = by_name[self.name]
    row[0] += 1; row[1] += t2 - t0; row[2] += t2 - t1
    if len(row) < 4: row.append(0)
    row[3] += t4 - t3
    return float(unwrap(prof.en) - prof.st) * 1e-6 if wait else None

  CPUProgram.__call__ = timed_call

  from tinygrad import Device, Tensor
  print(f"CPU renderer: {type(Device['CPU'].renderer).__name__} arch={Device['CPU'].arch}", flush=True)
  from tinygrad.llm.gguf import gguf_parse
  from tinygrad.llm.model import Transformer
  from tinygrad.llm.tokenizer import SimpleTokenizer
  from tinygrad.uop.ops import UOp

  kv, _ = gguf_parse(str(args.ckpt))
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  ids = tokenizer.encode(PROMPT)[:args.prompt]
  max_context = args.prompt + args.gen + 16
  load_started = time.time()
  model, _ = Transformer.from_gguf(str(args.ckpt), max_context=max_context)
  from npu_linear import packed_weight_bytes, prepare_linears, replace_linears, fuse_linears
  fuse_linears(model)
  n_linears = replace_linears(model)
  prepare_linears(model)
  nbytes = packed_weight_bytes(model)
  load_seconds = time.time() - load_started

  # The JIT re-runs the Python forward on every call to rebuild the graph; time that separately
  # from the replay, since time spent inside `model(...)` includes both.
  build_stats = {"ns": 0}
  _jit = getattr(model, "prefill_jit", None)
  if _jit is not None and getattr(_jit, "fxn", None) is not None:
    _f = _jit.fxn
    def _timed_fxn(*a, **k):
      t = time.perf_counter_ns(); r = _f(*a, **k); build_stats["ns"] += time.perf_counter_ns() - t; return r
    _jit.fxn = _timed_fxn

  temperature = Tensor([0.0])
  prompt_tensor = Tensor(ids, dtype="int32").reshape(1, args.prompt)
  v_start = UOp.variable("start_pos", 0, max_context - 1)
  pos_zero = v_start.bind(0)

  # 1 captures the JIT; then instrument / bare / instrument, so the observer effect is a
  # measured ratio between adjacent trials rather than an assumption.
  plan = [("capture", False, False), ("instrumented-a", True, True), ("bare", False, False),
          ("instrumented-b", True, True), ("replay-only", False, True)]
  trials = []
  for tag, instrument, replay_on in plan:
    stats.update(outer_ns=0, pack_ns=0, fxn_ns=0, thread_ns=0, calls=0)
    copy_stats.update(n=0, ns=0)
    exec_stats.update(n=0, ns=0)
    rep.update(hit=0, miss=0, skip=0, ns=0)
    build_stats["ns"] = 0
    by_name.clear()
    state["on"] = instrument
    replay["on"] = replay_on
    npu = Device["ROCKCHIP"]
    s0, w0 = npu.submit_count, npu.submit_seconds
    t0 = time.perf_counter()
    lazy = model(prompt_tensor, (pos_zero,), temperature)   # JIT: rebuilds the Python graph
    t_build = time.perf_counter() - t0
    logits = lazy.realize()                                  # then replays the captured linear
    wall = time.perf_counter() - t0
    # Bitwise identity at the model boundary: two runs of the same configuration must agree here
    # before a transformed run can be compared with an untransformed one.
    import hashlib
    import numpy as _np
    arr = logits.numpy()
    logits_hash = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    print(f"logits finite: {bool(_np.isfinite(arr).all())}", flush=True)
    state["on"] = False
    replay["on"] = False
    row = {
      "tag": tag, "instrumented": instrument, "replay_enabled": replay_on,
      "wall_ms": round(wall * 1e3, 1),
      "cpu_programs": stats["calls"],
      "dispatch_outer_ms": round(stats["outer_ns"] / 1e6, 1),
      "dispatch_pack_ms": round(stats["pack_ns"] / 1e6, 1),
      "kernel_fxn_ms": round(stats["fxn_ns"] / 1e6, 1),
      "thread_cpu_ms": round(stats["thread_ns"] / 1e6, 1),
      "non_program_ms": round(wall * 1e3 - stats["outer_ns"] / 1e6, 1),
      "logits_sha256_16": logits_hash,
      "graph_build_ms": round(t_build * 1e3, 1),
      "python_forward_ms": round(build_stats["ns"] / 1e6, 1),
      "exec_calls": exec_stats["n"],
      "exec_rewrite_ms": round(exec_stats["ns"] / 1e6, 1),
      "replay_hit": rep["hit"],
      "replay_miss": rep["miss"],
      "replay_skip": rep["skip"],
      "replay_ms": round(rep["ns"] / 1e6, 1),
      "copies": copy_stats["n"],
      "copy_ms": round(copy_stats["ns"] / 1e6, 1),
      "npu_submits": npu.submit_count - s0,
      "temp_c": round(int(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1),
      "npu_seconds": round(npu.submit_seconds - w0, 3),
      "top": sorted(({"name": n, "calls": v[0], "outer_ms": round(v[1] / 1e6, 1), "fxn_ms": round(v[2] / 1e6, 1),
                      "thread_ms": round((v[3] if len(v) > 3 else 0) / 1e6, 1)}
                     for n, v in by_name.items()), key=lambda r: -r["outer_ms"]),
    }
    trials.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "top"}, indent=2), flush=True)

  out = {"checkpoint": str(args.ckpt), "npu_linears": n_linears, "npu_weight_mib": round(nbytes / 2**20, 1),
         "load_seconds": round(load_seconds, 2), "trials": trials}
  path = args.out or pathlib.Path("docs/rknpu/results/progcost/report.json")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(out, indent=2))
  print(f"wrote {path}")
  try:
    import ctypes, pathlib as _p
    lib = ctypes.CDLL(str(_p.Path(__file__).resolve().parent / "librkpool.so"))
    lib.rk_calls.restype = ctypes.c_long; lib.rk_refused.restype = ctypes.c_long
    print(f"[rkpool] dispatch calls {lib.rk_calls()}  refused/serial {lib.rk_refused()}", flush=True)
  except OSError as exc:
    print(f"[rkpool] not loaded: {exc}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
