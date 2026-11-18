#!/usr/bin/env python3
"""Attribute each captured graph's execution: how much is kernel time vs bookkeeping.

`exec_kernel` calls `rt(*args)` inline, so timing around `pm_exec.rewrite` measures kernel
execution plus bookkeeping, not dispatch alone. And whole-run aggregates are meaningless
here because the first execution is dominated by compilation.

So this records **one row per graph** and, for each program, splits `rt(...)` from the rest.
The last prefill-sized row is the steady-state answer.

Usage (from ~/pr/tinygrad):
  DEV=CPU .venv/bin/python extra/rockchip/profile_graph_dispatch.py -- <bench_llm args>
"""
from __future__ import annotations

import collections
import json
import re
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PREFILL_MIN_PROGRAMS = 100  # distinguishes the prefill graph from small decode graphs


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--"]
    if not argv:
        print("no bench_llm args given", file=sys.stderr)
        return 2

    sys.path.insert(0, HERE)
    import tinygrad.engine.jit as jitmod
    import tinygrad.engine.realize as R
    from tinygrad.helpers import DEBUG
    from tinygrad.uop.ops import Ops


    def node_name(node) -> str:
        """ProgramInfo drops KernelInfo.name, so identify a kernel by what it carries."""
        arg = getattr(node, "arg", None)
        name = getattr(arg, "name", None)
        if isinstance(name, str) and name:
            return name
        # A CALL wraps a PROGRAM, and the PROGRAM carries the kernel SOURCE, so search
        # both the node's children and their children.
        candidates = list(getattr(node, "src", ()))
        candidates += [c for parent in getattr(node, "src", ()) for c in getattr(parent, "src", ())]
        for item in candidates:
            if getattr(item, "op", None) is Ops.SOURCE and isinstance(item.arg, str):
                match = re.search(r"\b(?:void|int|float|half)\s+([A-Za-z_]\w*)\s*\(", item.arg)
                if match:
                    SOURCES.setdefault(match.group(1), item.arg)
                    return match.group(1)
                return item.arg.splitlines()[0][:48] if item.arg else "empty-source"
        return f"{node.op}:{type(arg).__name__}"

    rows: list[dict] = []
    DUMPED = [False]
    SOURCES: dict[str, str] = {}
    current = {"kernel_s": 0.0, "kernel_calls": 0, "by_name": collections.Counter()}

    def timed_run_linear(linear, var_vals=None, input_uops=(), update_stats=True, jit=False, wait=False):
        if len(linear.src) == 0:
            return
        inputs = list(input_uops)
        if not jit:
            linear = R.link_linear(R.compile_linear(linear, validate=R.VALIDATE_WITH_CPU,
                                                    input_uops=inputs, cache=True), input_uops=inputs)
        ctx = R.ExecContext(var_vals or {}, tuple(inputs), update_stats, jit, wait or DEBUG >= 2)
        current["kernel_s"] = 0.0
        current["kernel_calls"] = 0
        current["by_name"] = collections.Counter()
        by_device = collections.Counter()
        by_name = collections.Counter()
        by_count = collections.Counter()
        loop_start = time.perf_counter()
        for call in linear.src:
            node = call.without_after
            start = time.perf_counter()
            R.pm_exec.rewrite(node, ctx)
            elapsed = time.perf_counter() - start
            by_device[str(node.device)] += elapsed
            _nm = node_name(node)
            by_name[_nm] += elapsed
            by_count[_nm] += 1
            if not DUMPED[0] and len(linear.src) > 5000:
                DUMPED[0] = True
                print(json.dumps({"event": "node_shape", "op": str(node.op),
                                  "arg": type(node.arg).__name__,
                                  "src_ops": [str(getattr(c, "op", None)) for c in getattr(node, "src", ())],
                                  "src0_src_ops": [str(getattr(c, "op", None)) for c in getattr(getattr(node, "src", (None,))[0], "src", ())]}),
                      flush=True)
        loop_s = time.perf_counter() - loop_start
        if len(linear.src) >= PREFILL_MIN_PROGRAMS:
            rows.append({"programs": len(linear.src), "loop_s": round(loop_s, 2),
                         "kernel_s": round(current["kernel_s"], 2),
                         "bookkeeping_s": round(loop_s - current["kernel_s"], 2),
                         "by_device": {k: round(v, 2) for k, v in by_device.most_common()},
                         "top_kernels": [[round(t, 2), n, by_count[n]] for n, t in by_name.most_common(14)]})
        return None

    original_get_runtime = R.get_runtime

    def timed_get_runtime(*args, **kwargs):
        runtime = original_get_runtime(*args, **kwargs)
        ast = args[1] if len(args) > 1 else kwargs.get("ast")
        name = getattr(getattr(ast, "arg", None), "name", None) or "?"
        if getattr(runtime, "_timed", False):
            return runtime

        def timed_runtime(*call_args, **call_kwargs):
            start = time.perf_counter()
            result = runtime(*call_args, **call_kwargs)
            elapsed = time.perf_counter() - start
            current["kernel_s"] += elapsed
            current["kernel_calls"] += 1
            current["by_name"][name] += elapsed
            return result

        timed_runtime._timed = True
        return timed_runtime

    R.get_runtime = timed_get_runtime
    R.run_linear = timed_run_linear
    jitmod.run_linear = timed_run_linear
    print(json.dumps({"event": "patch", "bindings": ["realize.run_linear", "jit.run_linear", "realize.get_runtime"]}),
          flush=True)

    sys.argv = ["bench_llm"] + argv
    try:
        runpy.run_path(os.path.join(HERE, "bench_llm.py"), run_name="__main__")
    finally:
        print(json.dumps({"event": "graph_rows", "rows": rows}, indent=None), flush=True)
        # The PREFILL row is the one with the largest loop time. `rows[-1]` (the old bug) is the
        # last decode step, whose kernel set is different -- that is why the first attempt dumped
        # decode kernels and missed E_2_16_32_32_4_4 entirely.
        prefill = [r for r in rows if r["programs"] > 100]
        if prefill:
            # 44 lines was not enough to see an elementwise inner loop, and top-2 skipped the
            # generated CPU kernels entirely (they are 70% of prefill). Dump the top 8 in full.
            for _, name, _calls in max(prefill, key=lambda r: r["loop_s"])["top_kernels"][:8]:
                src = SOURCES.get(name, "(source not captured)")
                print(json.dumps({"event": "kernel_source", "name": name,
                                  "head": src}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
