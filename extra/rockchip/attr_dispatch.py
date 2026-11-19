#!/usr/bin/env python3
"""Dispatch cost = time inside run_linear MINUS time inside CPUProgram.__call__.

The previous attempt timed the segment call and absorbed the kernel work, so it could not separate
the two. Here both sides are timed independently at their own boundaries:

    run_linear_total  <- around realize.run_linear
    kernel_total      <- around CPUProgram.__call__ (every CPU kernel, fused ones included)
    dispatch          <- run_linear_total - kernel_total

Run with `replay` to measure the compiled-C path, or without for the stock executor.
"""
import pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
if len(sys.argv) > 1 and sys.argv[1] == "replay":
    from tinygrad.llm.kernels import replay_c as RC
    RC.install()
from tinygrad import Tensor
from tinygrad.engine import realize as R
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL
CKPT="/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT=("The history of the printing press begins with movable type in East Asia. "*12)
kv,_=gguf_parse(CKPT); tok=SimpleTokenizer.from_gguf_kv(kv); ids=tok.encode(PROMPT)[:128]
model,_=Transformer.from_gguf(CKPT,max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt=Tensor(ids,dtype="int32").reshape(1,128); tp=Tensor([0.0]); vs=UOp.variable("start_pos",0,150).bind(0)
st={"rl":0.0,"k":0.0,"n":0}; on=[False]
o_rl=R.run_linear; o_k=CPUProgram.__call__
def rl(linear,*a,**k):
    if not on[0]: return o_rl(linear,*a,**k)
    t0=time.perf_counter(); r=o_rl(linear,*a,**k); st["rl"]+=(time.perf_counter()-t0)*1e3; return r
def kk(self,*a,**k):
    if not on[0]: return o_k(self,*a,**k)
    t0=time.perf_counter(); r=o_k(self,*a,**k); st["k"]+=(time.perf_counter()-t0)*1e3; st["n"]+=1; return r
R.run_linear=rl; CPUProgram.__call__=kk
import tinygrad.engine.jit as J; J.run_linear=rl
for _ in range(3): model(pt,(vs,),tp).realize()
for x in st: st[x]=0
on[0]=True
t0=time.perf_counter(); model(pt,(vs,),tp).realize(); wall=(time.perf_counter()-t0)*1e3
on[0]=False
print(f"mode {'replay' if 'RC' in dir() else 'stock'}")
print(f"wall                {wall:8.2f} ms")
print(f"run_linear total    {st['rl']:8.2f} ms")
print(f"kernel total        {st['k']:8.2f} ms   ({st['n']} CPUProgram calls)")
print(f"DISPATCH            {st['rl']-st['k']:8.2f} ms   ({100*(st['rl']-st['k'])/st['rl']:.1f}% of run_linear)")
