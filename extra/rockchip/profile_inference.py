"""Attribute inclusive host costs in benchmark_inference; timings are diagnostic."""
import functools, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ['DEV'] = sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'CPU'
import benchmark_inference as benchmark  # noqa: E402
from tinygrad.runtime.ops_cpu import CPUProgram  # noqa: E402
from tinygrad.runtime.ops_rockchip import RockchipAllocator, RockchipDevice, RockchipProgram  # noqa: E402
from tinygrad.runtime.autogen import rockchip as rk  # noqa: E402

stats = defaultdict(lambda:[0,0.0])
requests = []
snapshots = {}


def timed(label,fn,named=False):
  @functools.wraps(fn)
  def call(*args,**kwargs):
    name = getattr(args[0],'name',getattr(getattr(args[0],'obj',None),'name','')) if named else ''
    key = label+(':'+name if name else '')
    start = time.perf_counter()
    try: return fn(*args,**kwargs)
    finally:
      stats[key][0] += 1
      stats[key][1] += time.perf_counter()-start
  return call


def snapshot(): return {key:tuple(value) for key,value in stats.items()}


def difference(after,before):
  return sorted([{'name':key,'calls':value[0]-before.get(key,(0,0))[0],
                  'inclusive_ms':(value[1]-before.get(key,(0,0))[1])*1000}
                 for key,value in after.items() if value[0]!=before.get(key,(0,0))[0]],
                key=lambda row:row['inclusive_ms'],reverse=True)


def observe(event,index):
  now = snapshot()
  if event=='begin': snapshots[index] = {'begin':now}
  elif event=='first': snapshots[index]['first'] = now
  elif event=='end':
    requests.append({'index':index,'prefill':difference(snapshots[index]['first'],snapshots[index]['begin']),
                     'decode':difference(now,snapshots[index]['first'])})


def save():
  target = Path(sys.argv[sys.argv.index('--report')+1]).with_suffix('.profile.json')
  target.write_text(json.dumps({'inclusive_costs_overlap':True,'requests':requests,
    'profiler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')


for cls,method,label,named in [(CPUProgram,'__call__','CPU',True),(RockchipProgram,'__call__','NPU',True),
    (RockchipAllocator,'_copyin','NPU copyin',False),(RockchipAllocator,'_copyout','NPU copyout',False),
    (RockchipDevice,'sync','NPU sync',False),(RockchipDevice,'reset','NPU reset',False)]:
  setattr(cls,method,timed(label,getattr(cls,method),named))
rk.DRM_IOCTL_RKNPU_SUBMIT = timed('NPU submit',rk.DRM_IOCTL_RKNPU_SUBMIT)
benchmark.observe = observe

if __name__=='__main__':
  try:
    model,jits = benchmark.main()
    from tinygrad.uop.ops import Ops
    directory = Path(sys.argv[sys.argv.index('--report')+1]).with_suffix('.sources')
    directory.mkdir(exist_ok=True)
    for phase,jit in jits.items():
      if jit.captured is None: continue
      for call in jit.captured.linear.src:
        body = call.without_after.body
        if body.op is not Ops.PROGRAM or call.device!='CPU': continue
        source = next(u.arg for u in body.src if u.op is Ops.SOURCE)
        (directory/f'{phase}-{body.src[0].arg.function_name}.c').write_text(source)
  finally: save()
