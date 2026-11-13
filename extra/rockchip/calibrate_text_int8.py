"""Calibrate existing INT8 input-channel scaling on separate text prompts.

X/s and W*s preserve the real-valued projection. Observe the checkpoint CPU/GPU
path or the existing INT8 CPU/NPU path; never use held-out quality data.
"""
import argparse, contextlib, json, os
from pathlib import Path
import numpy as np
from extra.rockchip.quality_inference import digest, sources, check_sources


def load_scales(path:Path, checkpoint_hash:str):
  from tinygrad.llm.kernels.rockchip_calibration import load_input_scales
  return load_input_scales(path,checkpoint_hash)


def collect_modules(model, skip_down:bool=True, linear_type:type|None=None) -> dict:
  """Use the backend's dotted attribute/list/dict paths for every prepared Linear."""
  from tinygrad import Tensor
  if linear_type is None:
    from tinygrad.llm.kernels.mali import MaliLinear
    linear_type = MaliLinear
  modules,seen = {},set()
  def visit(obj, prefix=''):
    if id(obj) in seen or isinstance(obj,(Tensor,type)):return
    seen.add(id(obj))
    if isinstance(obj,linear_type):
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


@contextlib.contextmanager
def cpu_reference_execution(linear_type:type, max_chunk_bytes:int=8*1024**2):
  """Evaluate CPU row lookups and checkpoint projections without vocabulary reductions."""
  from tinygrad import Device, Tensor, dtypes, nn
  if type(max_chunk_bytes) is not int or max_chunk_bytes<1:raise ValueError('Reference chunk budget must be positive')
  original = linear_type.__call__
  original_embedding = nn.Embedding.__call__
  def embedding(self,indices):
    weight = self.weight
    if (indices.device!='CPU' or not dtypes.is_int(indices.dtype) or weight.device!='CPU'
        or weight.ndim!=2 or weight.dtype not in (dtypes.half,dtypes.float)):
      raise ValueError('CPU reference embedding requires CPU integer indices and floating weights')
    ids = indices.numpy()
    result = np.zeros((*ids.shape,weight.shape[1]),np.float16 if weight.dtype is dtypes.half else np.float32)
    output = result.reshape(-1,weight.shape[1])
    for row,token in enumerate(ids.reshape(-1)):
      if 0<=token<weight.shape[0]:
        output[row] = weight[int(token):int(token)+1].numpy()[0]
    return Tensor(result,device='CPU')
  def project(self,x):
    weight = self.weight
    if (x.device!='CPU' or weight.device!='CPU' or x.dtype not in (dtypes.half,dtypes.float)
        or weight.dtype not in (dtypes.half,dtypes.float) or x.ndim<2 or weight.ndim!=2 or x.shape[-1]!=weight.shape[1]):
      raise ValueError('CPU reference GEMM requires compatible FP16/FP32 CPU operands')
    rows,depth = weight.shape
    values = x.numpy().astype(np.float32).reshape(-1,depth)
    half = x.dtype is dtypes.half and weight.dtype is dtypes.half
    output = np.empty((values.shape[0],rows),np.float16 if half else np.float32)
    chunk_rows = max(1,max_chunk_bytes//(depth*(weight.dtype.itemsize+4)))
    for start in range(0,rows,chunk_rows):
      matrix = weight[start:min(start+chunk_rows,rows)].numpy().astype(np.float32)
      output[:,start:start+matrix.shape[0]] = np.matmul(values,matrix.T)
    result = Tensor(output,device='CPU').reshape(*x.shape[:-1],rows)
    return result if self.bias is None else result+self.bias
  allocator = Device['CPU'].allocator
  previous_lru = allocator.lru
  try:
    allocator.lru = False
    allocator.free_cache()
    linear_type.__call__ = project
    nn.Embedding.__call__ = embedding
    yield
  finally:
    linear_type.__call__ = original
    nn.Embedding.__call__ = original_embedding
    try:allocator.free_cache()
    finally:allocator.lru = previous_lru


def capture_activations(model, modules:dict, linear_type:type, data:list, tokenizer, device:str) -> dict:
  """Observe eager projection inputs and restore both hooks on every exit."""
  from tinygrad import Tensor
  import tinygrad.llm.model as model_module
  names = {id(module):name for name,module in modules.items()}
  activations:dict[str,np.ndarray] = {}
  original,original_function = linear_type.__call__,model_module.function
  def observe(self,x):
    if (name:=names.get(id(self))) is not None:
      if name not in activations and len(activations)%16==0:
        print(json.dumps({'calibration_projection':name,'observed_before':len(activations)}),flush=True)
      values = x.numpy().astype(np.float32).reshape(-1,x.shape[-1])
      if not np.isfinite(values).all():raise ValueError('Calibration activations must be finite')
      maximum = np.abs(values).max(axis=0)
      activations[name] = np.maximum(activations[name],maximum) if name in activations else maximum
    return original(self,x)
  # Eager blocks expose tensors rather than compiler PARAMs. This CLI owns its process.
  model_module.function = lambda fxn=None,**kwargs:fxn if fxn is not None else lambda actual:actual
  linear_type.__call__ = observe
  try:
    for index,row in enumerate(data):
      ids = tokenizer.encode(row['input'])[:128]
      if not ids:raise ValueError('Empty calibration prompt')
      result = model.logits(Tensor([ids],device=device,dtype='int32'),0).realize().numpy()
      if not np.isfinite(result).all():raise ValueError('Invalid calibration logits')
      print(json.dumps({'calibration_prompt':index,'tokens':len(ids)}),flush=True)
  finally:
    linear_type.__call__ = original
    model_module.function = original_function
  if missing:=modules.keys()-activations.keys():raise ValueError(f'Unobserved calibration projections: {sorted(missing)}')
  return activations


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--dataset',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--alpha',type=float,default=.5)
  parser.add_argument('--skip-down',action=argparse.BooleanOptionalAction,default=True)
  parser.add_argument('--device',choices=('CL','CPU'),default='CL',help='FP16 GPU or eager INT8 CPU/NPU calibration')
  parser.add_argument('--cpu-reference',action='store_true',
                      help='Use unquantized checkpoint CPU projections instead of initializing the NPU')
  parser.add_argument('--npu-domains',type=int,choices=(1,2,3),default=3)
  args = parser.parse_args()
  if not 0<=args.alpha<=1:parser.error('Alpha must be between0 and1')
  if args.cpu_reference and args.device!='CPU':parser.error('--cpu-reference requires --device CPU')
  if args.output.exists():raise FileExistsError(args.output)
  os.environ['DEV'] = args.device
  if args.device=='CPU' and not args.cpu_reference:
    os.environ['ONPU_NPU_DOMAINS'] = str(args.npu_domains)
    os.environ['ONPU_NPU_DOMAIN_BYTES'] = str(3*1024**3)
  from tinygrad import Device
  from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, TransformerBlock
  from tinygrad.llm.cli import SimpleTokenizer
  model,kv = Transformer.from_gguf(args.checkpoint,max_context=512,realize=args.device=='CL',mapped_cpu=args.device=='CPU')
  recurrent_blocks = sum(isinstance(block,GatedDeltaNetBlock) for block in model.blk)
  if args.device=='CL':
    from tinygrad.llm.kernels.mali import MaliLinear, configure
    linear_type:type = MaliLinear
    placement = configure(model,recurrent_scan=bool(recurrent_blocks))
    if placement.get('fused_gated_delta_blocks',0)!=recurrent_blocks:
      raise ValueError('Calibration requires the qualified GPU scan for every GatedDeltaNet block')
  elif args.cpu_reference:
    from tinygrad import nn
    if any(type(block) is not TransformerBlock for block in model.blk) or model.config.num_experts:
      raise ValueError('CPU reference calibration currently requires dense Transformer blocks')
    linear_type = nn.Linear
    placement = None
  else:
    if any(type(block) is not TransformerBlock for block in model.blk) or model.config.num_experts:
      raise ValueError('CPU/NPU calibration currently requires dense Transformer blocks')
    from tinygrad.llm.kernels.rockchip import RockchipLinear, configure as configure_npu
    linear_type = RockchipLinear
    allocator = Device['CPU'].allocator
    previous = allocator.lru
    try:
      allocator.lru = False
      allocator.free_cache()
      placement = configure_npu(model,prefill='on',decode='on',precision='int8',prefill_rows=128,tile_columns=24384,
                                cpu_attention=True,cpu_norm=True,cpu_fast_softmax=True,split_down=True)
    finally:
      try:allocator.free_cache()
      finally:allocator.lru = previous
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  modules = collect_modules(model,skip_down=args.skip_down,linear_type=linear_type)
  if not modules:raise ValueError('No supported calibration projections')
  hashes = sources()
  data = json.loads(args.dataset.read_text())
  with cpu_reference_execution(linear_type) if args.cpu_reference else contextlib.nullcontext():
    activations = capture_activations(model,modules,linear_type,data,tokenizer,args.device)
  groups = calibration_groups(modules)
  report = {'schema_version':1,'checkpoint_sha256':digest(args.checkpoint),'dataset_sha256':digest(args.dataset),
            'calibration_prompts':len(data),'alpha':args.alpha,'skip_down':args.skip_down,'source_sha256':hashes,
            'capture_mode':('eager checkpoint CPU blocks' if args.cpu_reference else
                            'eager FP16 GPU blocks' if args.device=='CL' else 'eager INT8 CPU/NPU blocks') +
                           ' outside compiler function boundaries',
            'gpu_kernels':placement if args.device=='CL' else None,
            'cpu_npu_placement':placement if args.device=='CPU' and not args.cpu_reference else None,
            'cpu_reference':args.cpu_reference,'capture_weight_dtypes':sorted({str(module.weight.dtype) for module in modules.values()}),
            'cpu_reference_gemm':('bounded FP32 BLAS; projection dtype rounding before bias' if args.cpu_reference else None),
            'cpu_reference_embedding':('direct selected rows; invalid indices zero-filled' if args.cpu_reference else None),
            'npu_submissions':sum(Device[name].submit_count for name in Device._opened_devices if name.split(':')[0]=='ROCKCHIP'),
            'observed_projections':sorted(activations),
            'harness_sha256':digest(Path(__file__)),'groups':{}}
  for name,members in groups.items():
    amax = np.maximum.reduce([activations[member] for member in members])
    wmax = np.maximum.reduce([weight_column_max(modules[member].weight) for member in members])
    scale = (np.maximum(amax,1e-5)**args.alpha/np.maximum(wmax,1e-5)**(1-args.alpha)).clip(1e-4,1e4)
    scale = np.where((amax==0)|(wmax==0),1,scale).astype(np.float32)
    report['groups'][name] = {'members':members,'scales':scale.tolist()}
  check_sources(hashes)
  if args.cpu_reference and any(name.split(':')[0] not in ('CPU','NPY','PYTHON','DISK') for name in Device._opened_devices):
    raise RuntimeError('CPU reference calibration opened an accelerator device')
  report['opened_devices'] = sorted(Device._opened_devices)
  report['activation_statistics_dtype'] = 'float32'
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'complete':True,'groups':len(groups),'alpha':args.alpha,'skip_down':args.skip_down}),flush=True)


if __name__=='__main__':main()
