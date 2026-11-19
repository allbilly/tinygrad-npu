#!/usr/bin/env python3
"""Attribute the replay dispatch loop: segment ctypes calls vs arg prep vs opaque PYTHON."""
import pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Tensor
import tinygrad.engine.jit as J
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
for _ in range(3): model(pt,(vs,),tp).realize()
for k in ("c_ms","py_ms","arr_ms","segs"): RC.stats[k]=0
J.JIT_TIMES.update(prep_ms=0.0, run_linear_ms=0.0, symbolic_ret_ms=0.0)
J.JIT_PROFILE=True
t0=time.perf_counter(); model(pt,(vs,),tp).realize(); wall=(time.perf_counter()-t0)*1e3
J.JIT_PROFILE=False
s=RC.stats
print(f"wall                 {wall:8.2f} ms")
print(f"run_linear_ms        {J.JIT_TIMES['run_linear_ms']:8.2f} ms")
print(f"  segments ctypes    {s['c_ms']:8.2f} ms  ({s['segs']} segment calls)")
print(f"  arg prep (in/vals) {s['arr_ms']:8.2f} ms")
print(f"  opaque PYTHON      {s['py_ms']:8.2f} ms")
print(f"  unattributed in rl {J.JIT_TIMES['run_linear_ms']-s['c_ms']-s['arr_ms']-s['py_ms']:8.2f} ms")
