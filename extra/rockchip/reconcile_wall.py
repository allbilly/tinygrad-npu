#!/usr/bin/env python3
"""Reconcile one decode step: driver device time + in-call CPU + standalone CPU vs wall.

Every earlier decomposition inferred device time from a kernel MANIFEST produced by a different
protocol, and the last ~8 ms never reconciled. This measures the four quantities on the *same* steps:

  wall      = perf_counter around model(...).realize()
  thread_cpu= thread_time_ns delta: every CPU instruction spent on this thread, whatever the
              dispatch path (the fused Linear binds its CALL uop directly and never enters
              CPUProgram.__call__, so a class hook measures zero -- see the first attempt)
  device    = delta of ROCKCHIP `npu.submit_seconds`, i.e. time blocked in the submit ioctl
  residual  = wall - thread_cpu - device

`residual` is the part that is neither CPU work nor the submit wait: syncs, copies, allocator.

  python extra/rockchip/reconcile_wall.py [steps]
"""
import collections, pathlib, statistics as st, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thread_split as TS
TS.install()
from tinygrad.llm.kernels import replay_c as RC
RC.install()
from tinygrad import Device, Tensor
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad.llm.gguf import gguf_parse
from tinygrad.llm.model import Transformer
from tinygrad.llm.tokenizer import SimpleTokenizer
from tinygrad.uop.ops import UOp
import npu_linear as NL

CKPT = "/home/orangepi/vlm-bench/baseline-20261021/checkpoints/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

kv, _ = gguf_parse(CKPT)
ids = SimpleTokenizer.from_gguf_kv(kv).encode("x "*128)[:128]
model, _ = Transformer.from_gguf(CKPT, max_context=160)
NL.fuse_linears(model); NL.replace_linears(model); NL.prepare_linears(model)
tp = Tensor([0.0])
model(Tensor(ids, dtype="int32").reshape(1, len(ids)), (UOp.variable("start_pos", 0, 150).bind(0),), tp).realize()

npu = Device["ROCKCHIP"]
pos = 128
def step():
    global pos
    model(Tensor([100 + (pos % 20)], dtype="int32").reshape(1, 1),
          (UOp.variable("start_pos", 0, 150).bind(pos),), tp).realize()
    pos += 1
step(); step()                                     # capture + warm
rows = []
for _ in range(STEPS):
    s0, w0 = npu.submit_count, npu.submit_seconds
    c0 = time.thread_time_ns()
    a = time.perf_counter(); step(); wall = (time.perf_counter()-a)*1e3
    cpu = (time.thread_time_ns()-c0)/1e6
    rows.append(dict(wall=wall, cpu=cpu, submits=npu.submit_count-s0, device=(npu.submit_seconds-w0)*1e3))
    print(f"  step {len(rows)}: wall {wall:6.1f}  cpu {cpu:6.1f}  device {rows[-1]['device']:6.1f}  "
          f"npu {int(open('/sys/class/devfreq/fdab0000.npu/cur_freq').read())/1e6:.0f} MHz  "
          f"temp {int(open('/sys/class/thermal/thermal_zone0/temp').read())/1000:.1f} C", flush=True)
med = {k: st.median([r[k] for r in rows]) for k in rows[0]}
resid = med["wall"] - med["cpu"] - med["device"]
print(f"steps {STEPS}  submits/token {med['submits']:.0f}")
print(f"  thread CPU (all kernels + python) {med['cpu']:8.2f} ms  {med['cpu']/med['wall']*100:5.1f}%")
print(f"  device wait (submit_seconds)      {med['device']:8.2f} ms  {med['device']/med['wall']*100:5.1f}%")
print(f"  residual (syncs/copies/alloc)     {resid:8.2f} ms  {resid/med['wall']*100:5.1f}%")
print(f"  ---------------------------------------------------")
print(f"  WALL                              {med['wall']:8.2f} ms")
