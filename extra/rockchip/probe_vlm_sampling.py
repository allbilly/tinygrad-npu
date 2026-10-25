"""Compare external and captured Gumbel sampling on the qualified NPU decoder."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--runs',type=int,default=10)
  args = parser.parse_args()
  os.environ['DEV']='CPU'
  from tinygrad import Tensor
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  repo = Path(__file__).resolve().parents[2]
  paths = [p for p in (repo/'tinygrad').rglob('*') if p.suffix in ('.py','.c')]+[Path(__file__)]
  hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
  ids = json.loads((args.root/'demo/input_ids.json').read_text())
  values = np.fromfile(args.root/'demo/input_embeddings.f32',dtype=np.float32).reshape(1,len(ids),576)
  report = {'complete':False,'source_sha256':hashes,'positions':len(ids),'trials':[]}
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  model = SmolVLM.from_hf(args.root/'hf',512).text
  sampled = model.sampled_embeddings_prefill_jit
  configure(model,prefill='on',decode='on',precision='int8',tile_columns=24576,fuse_projections=True,
            cpu_attention=True,native_replay=True,fused_ffn=True,cpu_norm=True,cpu_pool=True,cpu_fast_softmax=True,
            cache_cleaning=True,wake_helper=True)
  inputs = Tensor(values,device='CPU').realize()
  def run(mode,temperature):
    if mode=='captured':return int(model.sampled_embeddings_prefill_jit(inputs,temperature).item())
    logits = model.embeddings_prefill_jit(inputs)
    return int((logits/temperature.maximum(1e-12)-Tensor.rand_like(logits).maximum(1e-12).log().neg().log()).argmax(-1).item())
  for mode in ('external','captured'):
    for _ in range(5):run(mode,Tensor([.6]).realize())
  assert sampled.captured.native is not None,sampled.captured.reason
  counter = Tensor._device_rng_counters['CPU']
  native_start = sampled.captured.native_runs
  for iteration in range(args.runs):
    temperature = (.6,2.,.05,.6,1.2)[iteration%5]
    temp = Tensor([temperature]).realize()
    before = counter.numpy().copy()
    outputs,states = [],[]
    for mode in (('external','captured') if iteration%2==0 else ('captured','external')):
      counter.assign(Tensor(before)).realize()
      start = time.perf_counter()
      token = run(mode,temp)
      elapsed = (time.perf_counter()-start)*1000
      outputs.append(token)
      states.append(counter.numpy().copy())
      row = {'iteration':iteration,'mode':mode,'temperature':temperature,'ms':elapsed,'tok_s':len(ids)*1000/elapsed,'token':token}
      report['trials'].append(row)
      print(json.dumps(row),flush=True)
    assert outputs[0]==outputs[1],outputs
    np.testing.assert_array_equal(states[0],states[1])
    assert not np.array_equal(before,states[0]),'RNG state did not advance'
  report['native_runs'] = sampled.captured.native_runs-native_start
  assert report['native_runs']==args.runs,report['native_runs']
  report['medians'] = {mode:{'ms':statistics.median(r['ms'] for r in report['trials'] if r['mode']==mode),
                             'tok_s':statistics.median(r['tok_s'] for r in report['trials'] if r['mode']==mode)}
                       for mode in ('external','captured')}
  assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
  report.update(complete=True,matched_tokens=True,matched_rng=True)
  args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'medians':report['medians'],'native_runs':report['native_runs']}),flush=True)


if __name__=='__main__':main()
