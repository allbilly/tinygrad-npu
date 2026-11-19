#!/usr/bin/env python3
"""Wall minus time inside run_linear = the host time no counter currently attributes.

The profile arithmetic leaves 83.13 ms of the 1,259.09 ms outside every measured row (device,
quantise, dequantise, syncs, non-fused kernels). If that is model-level Python -- the forward body
and the Tensor/UOp construction between kernel launches -- then wall - sum(run_linear) recovers it
and nothing inside run_linear can explain it.
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
orig=R.run_linear; acc={"ms":0.0,"n":0,"calls":0}; on=[False]
def patched(linear,*a,**k):
    if not on[0]: return orig(linear,*a,**k)
    t0=time.perf_counter(); r=orig(linear,*a,**k); acc["ms"]+=(time.perf_counter()-t0)*1e3
    acc["n"]+=1; acc["calls"]+=len(linear.src)
    return r
R.run_linear=patched
import tinygrad.engine.jit as J; J.run_linear=patched
for _ in range(2): model(pt,(vs,),tp).realize()
on[0]=True
t0=time.perf_counter(); model(pt,(vs,),tp).realize(); wall=(time.perf_counter()-t0)*1e3
on[0]=False
print(f"mode {'replay' if 'RC' in dir() else 'stock'}")
print(f"wall                 {wall:8.2f} ms")
print(f"inside run_linear    {acc['ms']:8.2f} ms   ({acc['n']} linears, {acc['calls']} calls)")
print(f"OUTSIDE run_linear   {wall-acc['ms']:8.2f} ms   ({100*(wall-acc['ms'])/wall:.1f}% of wall)")
