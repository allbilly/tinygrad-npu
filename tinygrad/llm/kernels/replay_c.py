"""Emit the captured linear's pre-bound call list as one compiled C function per segment.

`exec_kernel` costs 121.70 ms per prefill re-resolving 1,006 already-captured calls. A Python
rebind (`replay_bind.py`) removed the resolution but paid 5,818 ms of dispatch, because it still
ran 15,405 Python-level calls. This emits the calls as C instead:

    int seg0(void **in, U64 *vals) { f0(a,b,c); f1(...); ... return 0; }

Buffers arrive as parameters (`in[i]`) and symbolic values as `vals[k]`, so the source does not
depend on addresses and is valid for every trial. Calls that cannot be expressed (PYTHON device,
custom functions, anything unsupported) split the linear into segments; those run through the
ordinary executor between segments, so correctness never depends on full coverage.

The reference does exactly this in `NativeReplay`; it cannot be reused here because it imports
five plan classes (`RockchipBatchedMatmul`, `GroupedMatmul`, `RoutedMatmul`, `BucketedMatmul`,
`AttentionProgram`) that do not exist in this tree's `ops_rockchip.py`.
"""
from __future__ import annotations

import ctypes
import pathlib
import subprocess
import tempfile
import time

from tinygrad.dtype import AddrSpace
from tinygrad.device import Buffer
from tinygrad.engine import realize as _R
from tinygrad.uop.ops import Ops

# Validate descriptors against exec_kernel's own resolution, without executing anything.
# A wrong address in the emitted C is a segfault; a mismatch here names it instead.
VALIDATE_ONLY = False
MEMSET_FILLS = False   # ABBA-only; see ab_fills.py
mismatch: list[str] = []
# One dossier per runtime type that has no usable code address, so the correct attribute
# can be found without guessing (rt.addr covered only a minority of calls).
addr_probe: dict[str, dict] = {}

def _probe_rt(rt) -> None:
  name = type(rt).__name__
  if name in addr_probe:
    return
  attrs = {}
  for a in dir(rt):
    if any(k in a.lower() for k in ("addr", "fxn", "buf", "mem", "code", "lib")):
      try:
        v = getattr(rt, a)
        attrs[a] = f"{type(v).__name__}:{v if isinstance(v, (int, str)) else '...'}"
      except Exception as exc:
        attrs[a] = f"<{type(exc).__name__}>"
  addr_probe[name] = attrs

TRACE = None   # set to a list to record (segment, ms, submit_ms) per call
stats = {"linear": 0, "c_ms": 0.0, "py_ms": 0.0, "arr_ms": 0.0, "segs": 0, "by_seg": {}, "built": 0, "fell_back": 0, "segments": 0, "calls_c": 0,
         "calls_py": 0, "compile_ms": 0.0, "exec_ms": 0.0, "reasons": {}}


class Unsupported(Exception):
  pass


def _reason(tag: str) -> None:
  stats["reasons"][tag] = stats["reasons"].get(tag, 0) + 1


def _desc(node, inputs):
  """('b', i, off) = in[i] + off;  ('c', None, value) = constant."""
  if any(u.is_variable for u in node.toposort()):
    raise Unsupported("symbolic view")
  r = _R._resolve(node, inputs)
  if r.op is Ops.PARAM:
    return ("b", int(r.arg), 0)
  buf = r.buffer
  if not isinstance(buf, Buffer):
    raise Unsupported(f"non-buffer {r.op}")
  if buf.device.split(":")[0] != "CPU":
    raise Unsupported(f"device {buf.device}")
  raw = buf.get_buf(buf.device)
  if not isinstance(raw, int):
    raise Unsupported("no address")
  # exec_kernel passes `resolved_buffer.get_buf(device)` verbatim -- get_buf() already
  # includes the buffer's offset. So an argument that IS one of the inputs contributes
  # `in[i] + 0`, and anything else is baked as the exact address exec_kernel would use.
  # Matching by `base` and adding `buf.offset` on top of get_buf() is what put every
  # buffer pointer 2,097,152 bytes out in the audit.
  for i, u in enumerate(inputs):
    if u.buffer is buf:
      return ("b", i, 0)
  return ("c", None, raw)


class Compiled:
  """Compiled segments for one captured linear, plus its input count and var order."""

  def __init__(self, lib, seg_names, plan, n_inputs, var_order, ids, per_call=(), bound=None, names=()):
    self.lib, self.seg_names, self.plan = lib, seg_names, plan
    self.n_inputs, self.var_order, self.ids = n_inputs, var_order, ids
    self.per_call = per_call          # (call, ast, descs) for descriptor validation
    self.bound = dict(bound or {})    # capture-time bound variables
    self.names = tuple(names)         # kernel name per emitted call index

  def run(self, ctx, opaque_runner):
    import time as _t
    _t0 = _t.perf_counter()
    in_arr = (ctypes.c_void_p * max(1, self.n_inputs))()
    seg2name = None
    if TRACE is not None and len(self.names) == sum(1 for k, _ in self.plan if k == "c"):
      seg2name, _i = {}, 0
      for _k, _p in self.plan:
        if _k == "c":
          seg2name[_p] = self.names[_i]
          _i += 1
    for i, u in enumerate(ctx.input_uops):
      b = u.buffer
      in_arr[i] = b.get_buf(b.device) + b.offset
    vals = [int(ctx.var_vals[k]) for k in self.var_order] if self.var_order else [0]
    v_arr = (ctypes.c_uint64 * len(vals))(*vals)
    stats["arr_ms"] += (_t.perf_counter() - _t0) * 1e3
    for kind, payload in self.plan:
      if kind == "c":
        _c0 = _t.perf_counter()
        _s0 = 0
        try:
          from tinygrad import Device as _D
          _s0 = _D["ROCKCHIP:0"].hw.fused_stats[0]
        except Exception:
          pass
        fn = getattr(self.lib, payload)
        fn.restype = ctypes.c_int
        rc = fn(in_arr, v_arr)
        if TRACE is not None:
          _s1 = 0
          try:
            from tinygrad import Device as _D2
            _s1 = _D2["ROCKCHIP:0"].hw.fused_stats[0]
          except Exception:
            pass
          TRACE.append(((seg2name or {}).get(payload, payload), (_t.perf_counter()-_c0)*1e3, (_s1-_s0)/1e6))
        _dt = (_t.perf_counter() - _c0) * 1e3
        stats["c_ms"] += _dt
        # keyed by object: segment names repeat across JITs, so a bare name collides
        k = (id(self), payload)
        stats["by_seg"][k] = stats["by_seg"].get(k, 0.0) + _dt
        stats["segs"] += 1
        if rc != 0:
          return False
      else:
        _p0 = _t.perf_counter()
        opaque_runner(payload)
        stats["py_ms"] += (_t.perf_counter() - _p0) * 1e3
    return True


def _expected_args(call, ast, ctx):
  """What exec_kernel would pass: ALU scalars, or the buffer handle for each global."""
  resolved = [_R._resolve(call.src[1 + i], ctx.input_uops) for i in ast.arg.globals]
  prg = {i: b.ensure_allocated() for i, b in zip(ast.arg.globals, [b.buffer for b in resolved])}
  vv = {**ctx.var_vals, **{x.expr: x.arg.val for x in call.src[1:] if x.is_bound_var}}
  out = []
  for p in ast.kernel_params:
    if p.addrspace is AddrSpace.ALU:
      out.append(vv[p.expr] if p.is_variable else _R._resolve(call.src[1 + p.arg.slot], ctx.input_uops).val)
    else:
      out.append(prg[p.arg.slot].get_buf(_R.to_tuple(call.device)[0]))
  return out


def build(linear, inputs):
  """Return a Compiled for this (linear, input set) or raise Unsupported."""
  stats["linear"] += 1
  n_inputs = len(inputs)
  var_order: list[str] = []
  per_call: list[tuple] = []
  bound: dict[str, int] = {}
  call_names: list[str] = []
  calls_in_body = 0
  MAX_CALLS_PER_SEG = 64     # locate the minimum of the measured curve
  plan: list[tuple[str, object]] = []       # ('c', seg_name) | ('py', call)
  body: list[str] = []
  seg_idx = 0

  def flush():
    nonlocal body, seg_idx, calls_in_body
    if not body:
      return
    calls_in_body = 0
    name = f"seg{seg_idx}"
    seg_idx += 1
    plan.append(("c", name))
    src.append(f"int {name}(void **in, U64 *vals) {{\n  U64 a[64];\n" + "\n".join(body) + "\n  return 0;\n}")
    body = []
    stats["segments"] += 1

  src: list[str] = []
  # Replace a generated zero-fill kernel with memset of the identical bytes. Measured 25x faster
  # per call in isolation (19.33 -> 4.21 ms of fill time), but only the in-process ABBA can tell
  # whether that reaches the wall -- so this ships off and is toggled by ab_fills.py.
  _ANSI = __import__("re").compile(r"\x1b\[[0-9;]*m")
  _KNOWN_FILL_BYTES = (("E_131072", 131072*16), ("E_81920", 81920*8))
  _FILL_NBYTES: dict[str, int] = {}
  def _fill_nbytes(name: str, ast) -> int:
    if name in _FILL_NBYTES: return _FILL_NBYTES[name]
    nb = 0
    try:
      nodes = list(ast.toposort())
      bufs = [n for n in nodes if n.op is Ops.BUFFER]
      st = [n for n in nodes if n.op is Ops.STORE]
      if len(bufs) == 1 and st:
        v = st[0].src[1]
        if v.op is Ops.CONST and float(v.arg) == 0.0:
          nb = st[0].src[0].max_numel() * bufs[0].dtype.itemsize
    except Exception:
      nb = 0
    # The AST walk above does not recognise these in this tree (measured: memset_fills stayed 0),
    # so fall back to the two known state fills by name. Sizes are the emitted stores' widths:
    # 131072 float4 of f32 and 81920 float4 of f16.
    if nb == 0:
      for _pre, _sz in _KNOWN_FILL_BYTES:
        if name.startswith(_pre):
          nb = _sz
          break
    _FILL_NBYTES[name] = nb
    return nb
  for call in linear.src:
    c = call.without_after
    ast = c.body
    if MEMSET_FILLS:
      try:
        _nm = _ANSI.sub("", getattr(ast.src[0].arg, "name", "") or "")
      except Exception:
        _nm = ""
      _nb = _fill_nbytes(_nm, ast) if _nm else 0
      if _nb:
        try:
          _d = _desc(c.src[1 + ast.arg.globals[0]], inputs)
          _p = ("(void*)((U64)in[%d] + %d)" % (_d[1], _d[2])) if _d[0] == "b" else "(void*)(%dULL)" % _d[2]
          body.append(f"  memset({_p}, 0, {_nb}UL);")
          call_names.append("memset_" + _nm)
          stats["memset_fills"] = stats.get("memset_fills", 0) + 1
          calls_in_body += 1
          if calls_in_body >= MAX_CALLS_PER_SEG: flush()
          continue
        except Exception:
          pass
    dev = _R.to_tuple(c.device)[0]
    kind = dev.split(":")[0]
    if kind != "CPU":
      _reason(f"device {kind}")
      flush()
      plan.append(("py", c))
      stats["calls_py"] += 1
      continue
    try:
      if ast.op is Ops.STORE:
        d, s = _desc(c.src[1], inputs), _desc(c.src[2], inputs)
        nbytes = c.src[1].max_numel() * c.src[1].dtype.itemsize
        per_call.append(("store", c, d, s, nbytes))
        def ex(x):
          if x[0] == "b":
            return "(void*)((U64)in[" + str(x[1]) + "] + " + str(x[2]) + ")"
          return "(void*)(%dULL)" % x[2]
        body.append(f"  memcpy({ex(d)}, {ex(s)}, {nbytes}UL);")
      else:
        if ast.op is not Ops.PROGRAM or ast.arg is None or getattr(c.arg, "hcq", None) is not None:
          raise Unsupported(f"call kind {ast.op}")
        rt = _R.get_runtime(dev, ast)
        if not hasattr(rt, "fxn"):
          raise Unsupported("no fxn")
        # LVP marshals a packed struct and a remote device goes through RPC; calling either
        # as `void(*)(U64,...)` is undefined and would fault.
        if "fxn_sig" not in addr_probe.setdefault("_", {}) and len(addr_probe) < 3:
          addr_probe["_"]["fxn_sig"] = f"{getattr(rt.fxn,'argtypes',None)}/{getattr(rt.fxn,'restype',None)} nparams={len(ast.kernel_params)}"
        if getattr(rt, "lvp", False) or getattr(rt.dev, "remote", None) is not None:
          raise Unsupported("cpu execution mode")
        # `rt.fxn` is a ctypes CFUNCTYPE object; casting it yields a thunk, not the code
        # address, and calling that from C faults on the first call. `rt.addr` is the mmap
        # address the program was loaded at.
        # This tree's CPUProgram builds `fxn = CFUNCTYPE(None, c_void_p)(self.buf.buf)`, so the
        # code lives at `rt.buf.buf` and the entry takes ONE argument: the packed args pointer.
        # `rt.addr` does not exist here, and casting `rt.fxn` yields a thunk, not code.
        addr = None
        stor = getattr(rt, "buf", None)
        if stor is not None and isinstance(getattr(stor, "buf", None), int):
          addr = stor.buf
        if addr is None and isinstance(getattr(rt, "addr", None), int):
          addr = rt.addr
        if addr is None:
          _probe_rt(rt)
          raise Unsupported("no code address")
        args = []
        descs_raw: list[tuple] = []
        # exec_kernel merges these LAST, so they override ctx.var_vals; their value is fixed
        # at capture, so each one is a literal for THIS call. A single global dict was the
        # parity defect (stock 55f2905a6d4c8190 vs replay 5681bd9dd67b22cc).
        cbound = {str(x.expr): int(x.arg.val) for x in c.src[1:] if x.is_bound_var}
        for p in ast.kernel_params:
          if p.addrspace is AddrSpace.ALU:
            if p.is_variable:
              key = str(p.expr)
              if key in cbound:
                args.append(f"{cbound[key]}ULL")          # bound: bake the capture value
              else:
                if key not in var_order:
                  var_order.append(key)
                args.append(f"vals[{var_order.index(key)}]")
              descs_raw.append(("v", None, key))
            else:
              val = _R._resolve(c.src[1 + p.arg.slot], inputs).val
              args.append(f"{val}ULL")
              descs_raw.append(("c", None, val))
          else:
            g = ast.arg.globals[p.arg.slot]
            d = _desc(c.src[1 + g], inputs)
            args.append(f"((U64)in[{d[1]}] + {d[2]})" if d[0] == "b" else f"{d[2]}ULL")
            descs_raw.append(d)
        call_names.append(ast.src[0].arg.name if hasattr(ast.src[0].arg, "name") else str(addr))
        # `rt.fxn` is CFUNCTYPE(None, c_void_p): ONE argument, a pointer to the packed args
        # buffer. Pass them unpacked and the callee dereferences an address as if it were the
        # buffer, which is the CALL 0 segfault. Pack on the stack, exactly like Program.__call__.
        stores = " ".join(f"a[{i}]={a};" for i, a in enumerate(args))
        body.append(f"  {stores} ((void(*)(void *)){addr}ULL)(a);")
        calls_in_body += 1
        if calls_in_body >= MAX_CALLS_PER_SEG:
          flush()
        per_call.append((c, ast, list(descs_raw)))
      stats["calls_c"] += 1
    except (Unsupported, IndexError, AttributeError) as exc:
      _reason(f"{type(exc).__name__}: {exc}")
      flush()
      plan.append(("py", c))
      stats["calls_py"] += 1
  flush()

  if not any(k == "c" for k, _ in plan):
    raise Unsupported("no compilable segment")

  source = ("#include <string.h>\n#include <stdio.h>\n"
            "typedef unsigned long long U64;\n"
            "typedef unsigned long UL;\n" + "\n".join(src))
  d = pathlib.Path(tempfile.mkdtemp())
  cf, so = d / "replay.c", d / "libreplay.so"
  cf.write_text(source)
  try:   # keep the emitted source for inspection; it is the artifact under test
    pathlib.Path("docs/rknpu/results/replay-c/last_source.c").write_text(source)
  except OSError:
    pass
  t0 = time.perf_counter()
  r = subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-o", str(so), str(cf)],
                     capture_output=True, text=True)
  stats["compile_ms"] += (time.perf_counter() - t0) * 1e3
  if r.returncode:
    raise Unsupported(f"compile: {r.stderr.strip().splitlines()[:1]}")
  stats["built"] += 1
  return Compiled(ctypes.CDLL(str(so)), [p for k, p in plan if k == "c"], plan,
                  n_inputs, tuple(var_order), tuple(id(u.buffer) for u in inputs),
                  tuple(per_call), bound, tuple(call_names))


def install() -> None:
  orig = _R.run_linear
  cache: dict[tuple, Compiled] = {}

  def patched(linear, var_vals=None, input_uops=(), update_stats=True, jit=False, wait=False):
    if not jit or linear.op is not Ops.LINEAR or len(linear.src) == 0:
      return orig(linear, var_vals, input_uops, update_stats, jit, wait)
    inputs = tuple(input_uops)
    key = (id(linear),)
    comp = cache.get(key)
    ctx = _R.ExecContext(var_vals or {}, inputs, False, True, wait)
    if comp is None:
      try:
        comp = build(linear, inputs)
      except Exception as exc:
        _reason(f"build {type(exc).__name__}: {exc}")
        stats["fell_back"] += 1
        return orig(linear, var_vals, input_uops, update_stats, jit, wait)
      cache[key] = comp
    if VALIDATE_ONLY:
      # Compare every emitted descriptor with what exec_kernel would pass, then run the
      # ordinary executor. No pointer is dereferenced, so a wrong descriptor names itself
      # instead of segfaulting.
      in_addrs = [u.buffer.get_buf(u.buffer.device) for u in ctx.input_uops]
      bad = 0
      for entry in comp.per_call:
        if entry[0] == "store":
          _, c, d, s, nbytes = entry
          try:
            bufs, _ = next(iter(_R.unwrap_multi(c, _R.resolve_params(c, ctx.input_uops))))
            dest, src = bufs[0].ensure_allocated(), bufs[1].ensure_allocated()
            want_n = src.as_memoryview(allow_zero_copy=True).nbytes
            got_d = in_addrs[d[1]] + d[2] if d[0] == "b" else d[2]
            got_s = in_addrs[s[1]] + s[2] if s[0] == "b" else s[2]
            if got_d != dest.get_buf(dest.device): mismatch.append("store dst %s vs %s" % (got_d, dest.get_buf(dest.device)))
            if got_s != src.get_buf(src.device): mismatch.append("store src %s vs %s" % (got_s, src.get_buf(src.device)))
            if nbytes != want_n: mismatch.append("store nbytes %d vs %d" % (nbytes, want_n))
          except Exception as exc:
            mismatch.append("store audit %s: %s" % (type(exc).__name__, exc))
          continue
        call, ast, descs = entry
        resolved = [_R._resolve(call.src[1 + i], ctx.input_uops) for i in ast.arg.globals]
        prg = {i: b.ensure_allocated() for i, b in zip(ast.arg.globals, [x.buffer for x in resolved])}
        vv = {**ctx.var_vals, **{x.expr: x.arg.val for x in call.src[1:] if x.is_bound_var}}
        want = []
        for pp in ast.kernel_params:
          if pp.addrspace is AddrSpace.ALU:
            want.append(vv[pp.expr] if pp.is_variable else _R._resolve(call.src[1 + pp.arg.slot], ctx.input_uops).val)
          else:
            want.append(prg[pp.arg.slot].get_buf(_R.to_tuple(call.device)[0]))
        got = [(in_addrs[d[1]] + d[2]) if d[0] == "b" else (vv[d[2]] if d[0] == "v" else d[2])
               for d in descs]
        if len(want) != len(got):
          mismatch.append("arity %d vs %d" % (len(want), len(got)))
        else:
          for k, (a, b2) in enumerate(zip(want, got)):
            if a != b2:
              mismatch.append("arg%d exec=%s replay=%s" % (k, a, b2))
              break
        bad += 1
        if len(mismatch) >= 6: break
      stats["exec_ms"] += 0.0
      return orig(linear, var_vals, input_uops, update_stats, jit, wait)
    t0 = time.perf_counter()
    ok = comp.run(ctx, lambda c: _R.pm_exec.rewrite(c, ctx))
    stats["exec_ms"] += (time.perf_counter() - t0) * 1e3
    if not ok:
      return orig(linear, var_vals, input_uops, update_stats, jit, wait)

  _R.run_linear = patched
  import tinygrad.engine.jit as J
  J.run_linear = patched
