#!/usr/bin/env python3
"""One ablation arm in a fresh process: SKIP_SYNCS is chosen before TinyJit captures.

In-process ABBA cannot toggle generated source -- capture pins the kernels -- so each arm has to
be its own process. Prints one line: ARM <name> <first_token_ms> <temp_c>.
"""
import argparse, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split; thread_split.install()
from tinygrad import Tensor
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL
CKPT="/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
PROMPT=("The history of the printing press begins with movable type in East Asia. "*12)
def temp():
  try: return float(pathlib.Path("/sys/class/thermal/thermal_zone0/temp").read_text())/1000
  except OSError: return float("nan")
ap=argparse.ArgumentParser(); ap.add_argument("--skip-syncs",action="store_true"); ap.add_argument("--skip-submit",action="store_true")
ap.add_argument("--warmups",type=int,default=1); ap.add_argument("--runs",type=int,default=1); a=ap.parse_args()
NL.SKIP_SYNCS=a.skip_syncs; NL.SKIP_SUBMIT=a.skip_submit
NL._FUSED_PROGRAMS.clear(); NL._FUSED_PLANS.clear()
kv,_=gguf_parse(CKPT); tok=SimpleTokenizer.from_gguf_kv(kv); ids=tok.encode(PROMPT)[:128]
model,_=Transformer.from_gguf(CKPT,max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt=Tensor(ids,dtype="int32").reshape(1,128); tp=Tensor([0.0]); vs=UOp.variable("start_pos",0,150).bind(0)
for _ in range(a.warmups): model(pt,(vs,),tp).realize()
ts=[]
for _ in range(a.runs):
    t=temp(); t0=time.perf_counter(); model(pt,(vs,),tp).realize(); ts.append((time.perf_counter()-t0)*1e3)
import statistics as st
ms=st.median(ts); t=temp()
arm = "no-submit" if a.skip_submit else ("no-sync" if a.skip_syncs else "with-sync")
print(f"ARM {arm} {ms:.2f} {t:.1f}  trials={[round(x,1) for x in ts]}", flush=True)
