#!/usr/bin/env python3
"""Temperature-fair ABBA: fused call with the two cache-sync ioctls vs without.

Timing only. The no-sync arm produces garbage (NPU and CPU disagree about cache state), so this
is an ablation to bound the syncs' cost, never a candidate configuration.
"""
import argparse, pathlib, statistics as st, sys, time
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
ap=argparse.ArgumentParser(); ap.add_argument("--rounds",type=int,default=3); a=ap.parse_args()
kv,_=gguf_parse(CKPT); tok=SimpleTokenizer.from_gguf_kv(kv); ids=tok.encode(PROMPT)[:128]
model,_=Transformer.from_gguf(CKPT,max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
pt=Tensor(ids,dtype="int32").reshape(1,128); tp=Tensor([0.0]); vs=UOp.variable("start_pos",0,150).bind(0)
def run(skip):
  NL.SKIP_SUBMIT=skip; NL.SKIP_SYNCS=False; NL._FUSED_PROGRAMS.clear(); NL._FUSED_PLANS.clear()
  t=temp(); t0=time.perf_counter(); model(pt,(vs,),tp).realize()
  return (time.perf_counter()-t0)*1e3, t
for s in (False,True,False,True): run(s)
rows=[]
for r in range(a.rounds):
  for arm in (False,True,True,False):
    ms,t=run(arm); rows.append((r,arm,ms,t)); print(f"round {r} {'no-submit' if arm else 'with-submit'} {ms:8.2f} ms  {t:5.1f} C", flush=True)
print()
for r in range(a.rounds):
  w=[m for rr,f,m,_ in rows if rr==r and not f]; n=[m for rr,f,m,_ in rows if rr==r and f]
  print(f"round {r}: with {st.mean(w):8.2f}  without {st.mean(n):8.2f}  delta {st.mean(n)-st.mean(w):+7.2f} ms")
W=[m for _,f,m,_ in rows if not f]; N=[m for _,f,m,_ in rows if f]
print(f"\nwith-sync median {st.median(W):.2f} ms   no-sync median {st.median(N):.2f} ms   delta {st.median(N)-st.median(W):+.2f} ms")
print(f"spread: with {100*(max(W)-min(W))/st.median(W):.1f}%  without {100*(max(N)-min(N))/st.median(N):.1f}%  temp {min(t for *_,t in rows):.1f}-{max(t for *_,t in rows):.1f} C")
