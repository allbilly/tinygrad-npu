"""Attribute an 8K VLM prefill to native CPU/NPU kernels; diagnostic, not a speed gate.

Unload the app's NPU owner first. The Rockchip device takes its own exclusive
lock. Short warmups compile the same bounded 128-row graph used by serving.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root',type=Path,required=True)
  parser.add_argument('--fixture',type=Path,required=True)
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  os.environ['DEV'],os.environ['ONPU_REPLAY_PROFILE'] = 'CPU','1'
  cores = os.sched_getaffinity(0).intersection({4,5,6,7})
  if cores: os.sched_setaffinity(0,cores)
  from tinygrad import Tensor, Device
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision
  from tinygrad.llm.kernels.rockchip_replay import NativeReplay
  from onpu.vlm import VisionSession
  repo = Path(__file__).resolve().parents[2]
  paths = [p for p in (repo/'tinygrad').rglob('*') if p.suffix in ('.py','.c')]+[Path(__file__)]
  hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
  report = {'complete':False,'scope':'diagnostic native-kernel attribution, not accepted comparative timing',
            'source_sha256':hashes,'cpu_affinity':sorted(os.sched_getaffinity(0))}
  def save(): args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  instances = []
  original = NativeReplay.__init__
  def track(self,*a,**kw):
    original(self,*a,**kw)
    instances.append(self)
  NativeReplay.__init__ = track
  try:
    ids = json.loads((args.fixture/'input_ids.json').read_text())
    assert 7900<=len(ids)<=8191
    vlm = SmolVLM.from_hf(args.root/'hf',8192)
    session = VisionSession(vlm,None)
    session.prepare_prefill()
    report['placement'] = configure(vlm.text,prefill='auto',decode='on',tile_columns=24576,prefill_rows=128,
      fuse_projections=True,cpu_attention=True,native_replay=True,fused_ffn=True,fused_qkv=True,cpu_norm=True,
      split_down=True,cpu_fast_softmax=True,cache_cleaning=True,wake_helper=True,precision='int8',cpu_pool=True,
      cpu_packed_attention=True)
    assert report['placement']['prefill_layers']==211
    configure(vlm.vision,prefill='on',decode='off')
    encoder = StagedVision(vlm,'ROCKCHIP',npu_connector=True,fused_mlp=True)
    Device['ROCKCHIP'].reset()
    image = encoder(Tensor(np.load(args.fixture/'pixels.npy'),device='CPU').realize()).realize()
    embeddings = vlm.embed_prompt(ids,image).realize()
    for _ in range(3): session.prefill(embeddings[:,:128].contiguous().realize(),greedy=True).item()
    for instance in instances:
      instance.profile_ns.clear()
      instance.profile_counts.clear()
    print(json.dumps({'event':'prepared','tokens':len(ids)}),flush=True)
    begin = time.perf_counter()
    token = session.prefill(embeddings,greedy=True,tokens=ids).item()
    elapsed = time.perf_counter()-begin
    timings,counts = {},{}
    for instance in instances:
      for label,ns in instance.profile_ns.items(): timings[label] = timings.get(label,0)+ns
      for label,count in instance.profile_counts.items(): counts[label] = counts.get(label,0)+count
    kernels = sorted([{'name':name,'ms':ns/1e6,'calls':counts[name]} for name,ns in timings.items()],
                     key=lambda row:row['ms'],reverse=True)
    assert kernels and any('rockchip_packed_attention' in k['name'] for k in kernels)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report.update(complete=True,tokens=len(ids),seconds=elapsed,tok_s=len(ids)/elapsed,first_token=token,kernels=kernels)
    save()
    print(json.dumps({k:report[k] for k in ('complete','seconds','tok_s','kernels')}),flush=True)
  except BaseException as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise
  finally: NativeReplay.__init__ = original


if __name__=='__main__': main()
