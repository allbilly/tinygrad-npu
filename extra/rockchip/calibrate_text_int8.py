"""Calibrate existing INT8 input-channel scaling on separate text prompts.

X/s and W*s preserve the real-valued projection. Calibration observes the FP16
GPU path directly; it never reads the held-out quality corpus or logits.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from extra.rockchip.quality_inference import digest, sources, check_sources


def load_scales(path:Path, checkpoint_hash:str):
  from tinygrad import Tensor, dtypes
  data = json.loads(path.read_text())
  if data.get('schema_version')!=1 or data.get('checkpoint_sha256')!=checkpoint_hash:
    raise ValueError('Text input calibration does not match the checkpoint')
  result = {}
  for group in data['groups'].values():
    values = np.array(group['scales'],np.float32)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all() or np.any(values<=0):
      raise ValueError('Input channel scales must be finite positive vectors')
    scale = Tensor(values,device='CPU',dtype=dtypes.float)
    for name in group['members']:
      if name in result:raise ValueError('Duplicate calibrated projection')
      result[name] = scale
  return result


def collect_modules(model, skip_down:bool=True) -> dict:
  """Use the backend's dotted attribute/list/dict paths for every prepared Linear."""
  from tinygrad import Tensor
  from tinygrad.llm.kernels.mali import MaliLinear
  modules,seen = {},set()
  def visit(obj, prefix=''):
    if id(obj) in seen or isinstance(obj,(Tensor,type)):return
    seen.add(id(obj))
    if isinstance(obj,MaliLinear):
      if not skip_down or 'ffn_down' not in prefix.split('.'):modules[prefix] = obj
      return
    items = enumerate(obj) if isinstance(obj,list) else obj.items() if isinstance(obj,dict) else (
      vars(obj).items() if hasattr(obj,'__dict__') else ())
    for key,value in items:visit(value,f'{prefix}.{key}' if prefix else str(key))
  visit(model)
  return modules


def calibration_groups(modules:dict) -> dict[str,list[str]]:
  """Share scales for compatible separate QKV and gate/up inputs; retain other layers."""
  groups,assigned = {},set()
  for prefix in sorted({name.rpartition('.')[0] for name in modules}):
    for group,parts in (('attn_qkv',('attn_q','attn_k','attn_v')),('gate_up',('ffn_gate','ffn_up'))):
      label = f'{prefix}.{group}' if prefix else group
      members = [f'{prefix}.{part}' if prefix else part for part in parts]
      if label not in modules and all(name in modules for name in members) and len({modules[name].weight.shape[1] for name in members})==1:
        groups[label] = members
        assigned.update(members)
  for name in sorted(modules.keys()-assigned):groups[name] = [name]
  return groups


def weight_column_max(weight, max_chunk_bytes:int=8*1024**2):
  """Reduce rows in bounded host chunks, preserving FP32 column maxima exactly."""
  if weight.ndim!=2 or not all(type(size) is int and size>0 for size in weight.shape):
    raise ValueError('Expected a nonempty static weight matrix')
  if type(max_chunk_bytes) is not int or max_chunk_bytes<1:raise ValueError('Weight chunk budget must be positive')
  rows,depth = weight.shape
  chunk_rows = max(1,max_chunk_bytes//(depth*(weight.dtype.itemsize+4)))
  maximum = np.zeros(depth,np.float32)
  for start in range(0,rows,chunk_rows):
    values = weight[start:min(start+chunk_rows,rows)].numpy().astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Calibration weights must be finite')
    np.abs(values,out=values)
    np.maximum(maximum,values.max(axis=0),out=maximum)
  return maximum


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--dataset',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--alpha',type=float,default=.5)
  parser.add_argument('--skip-down',action=argparse.BooleanOptionalAction,default=True)
  args = parser.parse_args()
  if not 0<=args.alpha<=1:parser.error('Alpha must be between0 and1')
  if args.output.exists():raise FileExistsError(args.output)
  os.environ['DEV'] = 'CL'
  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.kernels.mali import MaliLinear, configure
  import tinygrad.llm.model as model_module
  model,kv = Transformer.from_gguf(args.checkpoint,max_context=512,realize=True)
  recurrent_blocks = sum(isinstance(block,GatedDeltaNetBlock) for block in model.blk)
  gpu_kernels = configure(model,recurrent_scan=bool(recurrent_blocks))
  if gpu_kernels.get('fused_gated_delta_blocks',0)!=recurrent_blocks:
    raise ValueError('Calibration requires the qualified GPU scan for every GatedDeltaNet block')
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  modules = collect_modules(model,skip_down=args.skip_down)
  if not modules:raise ValueError('No supported calibration projections')
  names = {id(module):name for name,module in modules.items()}
  activations = {}
  original = MaliLinear.__call__
  def observe(self,x):
    if (name:=names.get(id(self))) is not None:
      values = x.numpy().reshape(-1,x.shape[-1])
      if not np.isfinite(values).all():raise ValueError('Calibration activations must be finite')
      maximum = np.abs(values).max(axis=0)
      activations[name] = np.maximum(activations[name],maximum) if name in activations else maximum
    return original(self,x)
  hashes = sources()
  data = json.loads(args.dataset.read_text())
  original_function = model_module.function
  # Eager blocks expose real tensors to the observer rather than compiler PARAMs.
  model_module.function = lambda fxn=None,**kwargs:fxn if fxn is not None else lambda actual:actual
  MaliLinear.__call__ = observe
  try:
    for index,row in enumerate(data):
      ids = tokenizer.encode(row['input'])[:128]
      if not ids:raise ValueError('Empty calibration prompt')
      result = model.logits(Tensor([ids],device='CL',dtype='int32'),0).realize().numpy()
      if not np.isfinite(result).all():raise ValueError('Invalid calibration logits')
      print(json.dumps({'calibration_prompt':index,'tokens':len(ids)}),flush=True)
  finally:
    MaliLinear.__call__ = original
    model_module.function = original_function
  if missing:=modules.keys()-activations.keys():raise ValueError(f'Unobserved calibration projections: {sorted(missing)}')
  groups = calibration_groups(modules)
  report = {'schema_version':1,'checkpoint_sha256':digest(args.checkpoint),'dataset_sha256':digest(args.dataset),
            'calibration_prompts':len(data),'alpha':args.alpha,'skip_down':args.skip_down,'source_sha256':hashes,
            'capture_mode':'eager FP16 GPU blocks outside compiler function boundaries',
            'gpu_kernels':gpu_kernels,'observed_projections':sorted(activations),
            'harness_sha256':digest(Path(__file__)),'groups':{}}
  for name,members in groups.items():
    amax = np.maximum.reduce([activations[member] for member in members])
    wmax = np.maximum.reduce([weight_column_max(modules[member].weight) for member in members])
    scale = (np.maximum(amax,1e-5)**args.alpha/np.maximum(wmax,1e-5)**(1-args.alpha)).clip(1e-4,1e4)
    scale = np.where((amax==0)|(wmax==0),1,scale).astype(np.float32)
    report['groups'][name] = {'members':members,'scales':scale.tolist()}
  check_sources(hashes)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'complete':True,'groups':len(groups),'alpha':args.alpha,'skip_down':args.skip_down}),flush=True)


if __name__=='__main__':main()
