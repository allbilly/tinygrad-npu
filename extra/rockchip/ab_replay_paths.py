#!/usr/bin/env python3
"""Time the SAME captured linear two ways in one process, per path, with counters.

The compiled-C replay removed ~132 ms of resolution yet measured 346 ms slower. Marshalling,
cpu_profile and synchronize are all far too small to explain that. This executes each linear
through both paths back to back, timing each, so the loss is localised to the C entry or to the
surrounding Python instead of inferred.

One warm-up forward, then one timed forward. Each linear runs twice per forward (stock, then
replay), so this measures cost, not end-to-end time.
"""
import pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad import Tensor
from tinygrad.engine import realize as R
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp, Ops
from tinygrad.llm.kernels import replay_c as RC
import npu_linear as NL
CKPT="/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT=("The history of the printing press begins with movable type in East Asia. "*12)
kv,_=gguf_parse(CKPT); tok=SimpleTokenizer.from_gguf_kv(kv); ids=tok.encode(PROMPT)[:128]
model,_=Transformer.from_gguf(CKPT,max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt=Tensor(ids,dtype="int32").reshape(1,128); tp=Tensor([0.0]); vs=UOp.variable("start_pos",0,150).bind(0)
orig=R.run_linear
acc={"stock_ms":0.0,"replay_ms":0.0,"calls":0,"linears":0}
cache={}; on=[False]
def patched(linear, var_vals=None, input_uops=(), update_stats=True, jit=False, wait=False):
    if not on[0] or not jit or linear.op is not Ops.LINEAR or len(linear.src)==0:
        return orig(linear, var_vals, input_uops, update_stats, jit, wait)
    inputs=tuple(input_uops); key=id(linear)
    comp=cache.get(key)
    if comp is None:
        try: comp=RC.build(linear, inputs)
        except Exception: return orig(linear, var_vals, input_uops, update_stats, jit, wait)
        cache[key]=comp
    t0=time.perf_counter(); r1=orig(linear, var_vals, input_uops, update_stats, jit, wait); t1=time.perf_counter()
    ctx=R.ExecContext(var_vals or {}, inputs, False, True, wait)
    t2=time.perf_counter(); comp.run(ctx, lambda c: R.pm_exec.rewrite(c, ctx)); t3=time.perf_counter()
    acc["stock_ms"]+=(t1-t0)*1e3; acc["replay_ms"]+=(t3-t2)*1e3
    acc["calls"]+=len(linear.src); acc["linears"]+=1
    return r1
R.run_linear=patched
import tinygrad.engine.jit as J; J.run_linear=patched
for _ in range(2): model(pt,(vs,),tp).realize()
on[0]=True
t0=time.perf_counter(); model(pt,(vs,),tp).realize(); wall=(time.perf_counter()-t0)*1e3
on[0]=False
print(f"linears {acc['linears']}  calls {acc['calls']}")
print(f"stock  total {acc['stock_ms']:8.1f} ms   per call {acc['stock_ms']/max(1,acc['calls'])*1000:6.1f} us")
print(f"replay total {acc['replay_ms']:8.1f} ms   per call {acc['replay_ms']/max(1,acc['calls'])*1000:6.1f} us")
print(f"replay vs stock {acc['replay_ms']-acc['stock_ms']:+8.1f} ms  ({(acc['replay_ms']/acc['stock_ms']-1)*100:+.1f}%)   wall {wall:.1f}")
