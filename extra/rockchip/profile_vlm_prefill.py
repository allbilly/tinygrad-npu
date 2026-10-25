"""Split inclusive VLM prefill time into embedding, decoder and first-token costs."""
import json, os, sys, time
from pathlib import Path
os.environ['ONPU_REPLAY_PROFILE']='1'
os.environ['DEV']=sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'CPU'
import benchmark_vlm as benchmark  # noqa: E402
from tinygrad import Tensor  # noqa: E402
from tinygrad.llm.vision import SmolVLM  # noqa: E402
from tinygrad.llm.kernels import rockchip  # noqa: E402

records=[]
current={}
factory=rockchip.configure
embed=SmolVLM.embed_prompt
item=Tensor.item


def configure(model,*args,**kwargs):
  report=factory(model,*args,**kwargs)
  def wrap(jit):
    def run(*args,**kwargs):
      start=time.perf_counter()
      result=jit(*args,**kwargs)
      current['decoder_ms']=(time.perf_counter()-start)*1000
      current['await_item']=True
      return result
    return run
  for name in ('embeddings_prefill_jit','greedy_embeddings_prefill_jit'):
    if hasattr(model,name):setattr(model,name,wrap(getattr(model,name)))
  return report


def embed_prompt(self,*args,**kwargs):
  current.clear()
  start=time.perf_counter()
  result=embed(self,*args,**kwargs).realize()
  current['embedding_ms']=(time.perf_counter()-start)*1000
  return result


def first_item(self):
  if not current.get('await_item'):return item(self)
  start=time.perf_counter()
  result=item(self)
  current['item_ms']=(time.perf_counter()-start)*1000
  current.pop('await_item')
  records.append(current.copy())
  return result


rockchip.configure=configure
SmolVLM.embed_prompt=embed_prompt
Tensor.item=first_item
if __name__=='__main__':
  try:benchmark.main()
  finally:
    target=Path(sys.argv[sys.argv.index('--report')+1]).with_suffix('.stages.json')
    target.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'prefill_stages':records}),flush=True)
