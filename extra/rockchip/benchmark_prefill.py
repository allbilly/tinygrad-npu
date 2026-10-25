"""Isolate the SmolVLM text-decoder prefill phase on the shared embedding input."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--runs',type=int,default=7)
  parser.add_argument('--profile',action='store_true')
  args = parser.parse_args()
  os.environ['DEV']='CPU'
  if args.profile:os.environ['ONPU_REPLAY_PROFILE']='1'
  from tinygrad import Tensor
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  repo=Path(__file__).resolve().parents[2]
  paths=[p for p in (repo/'tinygrad').rglob('*') if p.suffix in ('.py','.c')]+[Path(__file__)]
  hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
  ids=json.loads((args.root/'demo/input_ids.json').read_text())
  values=np.fromfile(args.root/'demo/input_embeddings.f32',dtype=np.float32).reshape(1,len(ids),576)
  report={'complete':False,'profile':args.profile,'source_sha256':hashes,'rows':len(ids),'trials':[]}
  def save():args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  model=SmolVLM.from_hf(args.root/'hf',512).text
  jit=model.embeddings_prefill_jit
  configure(model,prefill='on',decode='on',precision='int8',tile_columns=24576,fuse_projections=True,
            cpu_attention=True,native_replay=True,fused_ffn=True,cpu_norm=True,cpu_pool=True,cpu_fast_softmax=True,
            cache_cleaning=True,wake_helper=True)
  inputs=Tensor(values,device='CPU').realize()
  print(json.dumps({'event':'loaded','rows':len(ids)}),flush=True)
  for index in range(5+args.runs):
    start=time.perf_counter()
    output=model.embeddings_prefill_jit(inputs)
    result=output.numpy()
    elapsed=(time.perf_counter()-start)*1000
    row={'warmup':index<5,'ms':elapsed,'tok_s':len(ids)*1000/elapsed,'argmax':int(result.argmax())}
    assert np.isfinite(result).all()
    if index==4:
      captured=jit.captured
      assert captured.native is not None,captured.reason
      captured.native.profile_ns.clear()
      captured.native.profile_counts.clear()
      native_start=captured.native_runs
    report['trials'].append(row)
    print(json.dumps(row),flush=True)
  report['median_ms']=statistics.median(r['ms'] for r in report['trials'] if not r['warmup'])
  report['tok_s']=len(ids)*1000/report['median_ms']
  native=jit.captured.native
  report['native_runs']=jit.captured.native_runs-native_start
  report['kernels']=sorted([{'name':name,'ms':ns/report['native_runs']/1e6,
                            'calls':native.profile_counts[name]/report['native_runs']}
                           for name,ns in native.profile_ns.items()],key=lambda r:r['ms'],reverse=True)
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
  report['complete']=True
  save()
  print(json.dumps({'median_ms':report['median_ms'],'tok_s':report['tok_s'],'kernels':report['kernels'][:15]}),flush=True)


if __name__=='__main__':main()
