"""Compare first-token selection inside and outside the prepared decoder graph."""
import json, os, statistics, sys, time
from pathlib import Path
import numpy as np
os.environ['DEV']='CPU'
from tinygrad import Tensor, TinyJit  # noqa: E402
from tinygrad.llm.vision import SmolVLM  # noqa: E402
from tinygrad.llm.kernels.rockchip import configure  # noqa: E402
from tinygrad.llm.kernels.rockchip_greedy import greedy_argmax  # noqa: E402


def main():
  root=Path(sys.argv[1])
  ids=json.loads((root/'demo/input_ids.json').read_text())
  data=np.fromfile(root/'demo/input_embeddings.f32',dtype=np.float32).reshape(1,len(ids),576)
  model=SmolVLM.from_hf(root/'hf',512).text
  model.greedy_prefill_jit=TinyJit(lambda x:greedy_argmax(model.logits_from_embeddings(x,0)).realize())
  configure(model,prefill='on',decode='on',precision='int8',tile_columns=24576,fuse_projections=True,
            cpu_attention=True,native_replay=True,fused_ffn=True,cpu_norm=True,cpu_pool=True,cpu_fast_softmax=True,
            cache_cleaning=True,wake_helper=True)
  inputs=Tensor(data,device='CPU').realize()
  print(json.dumps({'event':'loaded'}),flush=True)
  expected=None
  for mode in ('logits','generic','native','captured'):
    samples=[]
    for index in range(12):
      start=time.perf_counter()
      if mode=='captured':token=int(model.greedy_prefill_jit(inputs).item())
      else:
        logits=model.embeddings_prefill_jit(inputs)
        token=int(logits.numpy().argmax()) if mode=='logits' else int((logits.argmax(-1) if mode=='generic' else greedy_argmax(logits)).item())
      elapsed=(time.perf_counter()-start)*1000
      if expected is None:expected=token
      assert token==expected
      if index>=5:samples.append(elapsed)
    print(json.dumps({'mode':mode,'median_ms':statistics.median(samples),'tok_s':len(ids)*1000/statistics.median(samples),
                      'all_ms':samples,'token':token}),flush=True)


if __name__=='__main__':main()
