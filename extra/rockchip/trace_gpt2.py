"""Locate GPT-2 numerical differences using identical checkpoints and per-layer observations."""
import argparse, json, os
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--report',type=Path,required=True)
  parser.add_argument('--length',type=int,default=32)
  parser.add_argument('--stock-attention',action='store_true')
  args = parser.parse_args()
  os.environ.update(DEV='CPU',HALF='1',MAX_CONTEXT='1024',JIT_BATCH_SIZE='0')
  import numpy as np
  from unittest.mock import patch
  from tinygrad import Tensor, Variable, nn
  from examples.gpt2 import GPT2, TransformerBlock
  from tinygrad.llm.kernels.rockchip_gpt2 import configure_gpt2

  gpt = GPT2.build('gpt2')
  model = gpt.model
  prompt = gpt.tokenizer.encode('The quick brown fox jumps over the lazy dog. The capital of France is Paris. '
                                'A computer processes information using arithmetic. '*10)[:args.length]
  names = {id(block):f'block-{i}' for i,block in enumerate(model.h)}
  states:dict[str,np.ndarray] = {}
  original_block = TransformerBlock.__call__
  def block_call(block,*inputs):
    result = original_block(block,*inputs)
    states[names[id(block)]] = result.numpy().astype(np.float64)
    return result
  def cpu_linear(layer,x):
    out = x.half().float().linear(layer.weight.half().float().T).cast(x.dtype)
    return out if layer.bias is None else out+layer.bias
  def request():
    for block in model.h:
      if hasattr(block.attn,'cache_kv'): block.attn.cache_kv.assign(Tensor.zeros_like(block.attn.cache_kv)).realize()
    inp = Tensor([prompt],device='CPU').realize()
    position = Variable('start_pos',0,1023).bind(0)
    result = model.logits(inp,position).numpy().astype(np.float64)
    states['logits'] = result
    return {k:v.copy() for k,v in states.items()}
  with patch.object(TransformerBlock,'__call__',block_call),patch.object(nn.Linear,'__call__',cpu_linear): reference = request()
  configure_gpt2(model,fast_attention=not args.stock_attention,wake_helper=False)
  with patch.object(TransformerBlock,'__call__',block_call): actual = request()
  report = {'input_ids':prompt,'stock_attention':args.stock_attention,'layers':[]}
  for name,b in reference.items():
    a = actual[name]
    error = a-b
    row = {'name':name,'shape':list(a.shape),'max_abs':float(np.abs(error).max()),'rmse':float(np.sqrt(np.mean(error**2))),
           'reference_rms':float(np.sqrt(np.mean(b**2))),'mean_offset':float(error.mean()),'centered_rmse':float(np.std(error))}
    report['layers'].append(row)
    print(json.dumps(row),flush=True)
  args.report.parent.mkdir(parents=True,exist_ok=True)
  args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
