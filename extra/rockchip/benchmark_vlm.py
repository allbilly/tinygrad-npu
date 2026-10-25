"""Measure image encoding + image-conditioned generation against saved FP32 inputs.

When using the NPU, unload its app session first. The driver holds the exclusive
NPU lock. Do not run this alongside an RKNN/RKLLM process.
"""
import argparse, faulthandler, hashlib, json, os, signal, statistics, time
from pathlib import Path
import numpy as np

def errors(actual, expected):
  a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
  return {'finite':bool(np.isfinite(a).all()),'max_abs':float(np.max(abs(a-b))),
          'normalized_rmse':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)),
          'cosine':float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))}

def main():
  faulthandler.register(signal.SIGUSR1)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--fixture',type=Path,required=True)
  parser.add_argument('--device',choices=['CPU','CL'],default='CPU')
  parser.add_argument('--vision-device',choices=['CPU','CL'])
  parser.add_argument('--vision-npu',choices=['off','auto','on'],default='off')
  parser.add_argument('--vision-precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--text-npu',choices=['off','auto','on'],default='off')
  parser.add_argument('--precision',choices=['fp16','int8'],default='fp16')
  parser.add_argument('--float32',action='store_true',help='FP32 stored weights for source parity')
  parser.add_argument('--cpu-vision-attention',action='store_true')
  parser.add_argument('--staged-vision',action='store_true')
  parser.add_argument('--staged-attention',choices=['CL','ROCKCHIP'],default='CL')
  parser.add_argument('--reset-before-vision',action='store_true',help='Include an exclusive NPU reset at the image boundary')
  parser.add_argument('--vision-calibration',type=Path)
  parser.add_argument('--npu-connector',action='store_true')
  parser.add_argument('--fused-vision-mlp',action='store_true')
  parser.add_argument('--shared-gpu-softmax',action='store_true')
  parser.add_argument('--prepared-vision-mlp',action='store_true')
  parser.add_argument('--batch-vision-prefill',action='store_true')
  parser.add_argument('--native-generation-loop',action='store_true')
  parser.add_argument('--context',type=int,default=512)
  parser.add_argument('--tokens',type=int,default=32)
  parser.add_argument('--warmups',type=int,default=3)
  parser.add_argument('--runs',type=int,default=3)
  parser.add_argument('--verify-only',action='store_true')
  parser.add_argument('--report',type=Path,required=True)
  args = parser.parse_args()
  if args.warmups<3 or args.runs<1 or args.tokens<2: parser.error('Need >=3 warmups, >=1 run, >=2 tokens')
  if args.staged_vision and (args.device!='CPU' or args.vision_device not in (None,'CPU')):
    parser.error('Staged vision requires CPU projection sources')
  os.environ['DEV'] = args.device
  if args.vision_npu!='off': os.environ.setdefault('ROCKCHIP_BINDING_CACHE','4096')
  from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
  from tinygrad.llm.vision import SmolVLM
  from tinygrad.llm.kernels.rockchip import configure
  root = Path(__file__).resolve().parents[2]
  sources = [*sorted(p for p in (root/'tinygrad').rglob('*') if p.suffix in ('.py','.c')),Path(__file__)]
  source_hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
  report = {'complete':False,'configuration':vars(args).copy(),'source_sha256':source_hashes,'trials':[],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'binding_cache':os.environ.get('ROCKCHIP_BINDING_CACHE','256'),
            'memory_planner_disabled':os.environ.get('NO_MEMORY_PLANNER')=='1',
            'checkpoint_sha256':hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest()}
  report['configuration'] = {k:str(v) if isinstance(v,Path) else v for k,v in report['configuration'].items()}
  def save(): args.report.write_text(json.dumps(report,indent=2)+'\n')
  save()
  reference = json.loads((args.fixture/'reference.json').read_text())
  ids = reference['input_ids']
  if len(ids)+args.tokens>args.context: parser.error('Prompt plus generated tokens exceed context')
  start = time.perf_counter()
  model = SmolVLM.from_hf(args.checkpoint,args.context,dtypes.float if args.float32 else dtypes.half,args.vision_device)
  report['load_seconds'] = time.perf_counter()-start
  jits = {'vision':model.encode_jit,'prefill':model.text.greedy_embeddings_prefill_jit,
          'prefill_logits':model.text.embeddings_prefill_jit,'decode':model.text.greedy_rollout_jit}
  scalar_jit = None
  if args.native_generation_loop:
    from tinygrad.llm.kernels.rockchip_decode import prepare_scalar_decode
    scalar_jit = prepare_scalar_decode(model.text)
    jits['scalar_decode'] = scalar_jit
  print(json.dumps({'event':'loaded','seconds':report['load_seconds']}),flush=True)
  if args.cpu_vision_attention:
    from tinygrad.llm.kernels.rockchip_vision import vision_attention
    for block in model.vision.layers: block.self_attn.attention = vision_attention
  if args.vision_npu!='off':
    scales = None
    if args.vision_calibration:
      from tinygrad.llm.kernels.rockchip_vision_pipeline import load_input_scales
      scales = load_input_scales(args.vision_calibration,report['checkpoint_sha256'])
    model.vision.prefill_jit = model.encode_jit
    report['vision_placement'] = configure(model.vision,prefill=args.vision_npu,decode='off',precision=args.vision_precision,
      native_replay=(args.vision_device or args.device)=='CPU',cache_cleaning=True,wake_helper=True,input_scales=scales,
      batch_prefill=args.batch_vision_prefill)
    model.encode_jit = model.vision.prefill_jit
  if args.text_npu!='off':
    report['text_placement'] = configure(model.text,prefill=args.text_npu,decode=args.text_npu,precision=args.precision,
      tile_columns=24576,fuse_projections=True,cpu_attention=args.device=='CPU',native_replay=args.device=='CPU',
      fused_ffn=True,cpu_norm=True,cpu_pool=True,cpu_fast_softmax=True,cache_cleaning=True,wake_helper=True)
  scalar_decode = None
  if scalar_jit is not None:
    from tinygrad.llm.kernels.rockchip_decode import ScalarDecoder
    from tinygrad.llm.kernels.rockchip_idle import idle_helper
    from tinygrad.llm.kernels.rockchip_pool import attention_pool
    placement = report.get('text_placement',{})
    scalar_decode = ScalarDecoder(model.text,scalar_jit,pool=attention_pool() if placement.get('cpu_pool') else None,
                                  helper=idle_helper() if placement.get('wake_helper') else None)
  staged = None
  if args.staged_vision:
    from tinygrad.llm.kernels.rockchip_vision_pipeline import StagedVision
    staged = StagedVision(model,args.staged_attention,npu_connector=args.npu_connector,fused_mlp=args.fused_vision_mlp,
                         gpu_softmax=args.shared_gpu_softmax,prepared_mlp=args.prepared_vision_mlp)
    model.encode_jit = staged
  pixels = Tensor(np.load(args.fixture/'pixels.npy'),device=args.vision_device or args.device).realize()
  npu = Device['ROCKCHIP'] if args.vision_npu!='off' or args.text_npu!='off' else None
  def encode_image():
    if args.reset_before_vision:
      if npu is None: raise ValueError('Reset requires an enabled NPU path')
      npu.reset()
    return model.encode_jit(pixels)
  print(json.dumps({'event':'prepared'}),flush=True)
  vision = TinyJit(lambda p:model.vision(p).realize())
  actual_vision = vision(pixels).numpy()
  image = encode_image()
  actual_image = image.numpy()
  embeds = model.embed_prompt(ids,image).realize()
  actual_logits = model.text.embeddings_prefill_jit(embeds).numpy()
  report['parity'] = {name:errors(actual,np.load(args.fixture/file)) for name,actual,file in [
    ('vision',actual_vision,'vision.npy'),('image_embeddings',actual_image,'image_embeddings.npy'),('logits',actual_logits,'logits.npy')]}
  report['parity']['first_token'] = int(actual_logits.argmax())
  report['parity']['greedy_first_token'] = int(model.text.greedy_embeddings_prefill_jit(embeds).item())
  if report['parity']['greedy_first_token']!=report['parity']['first_token']:raise RuntimeError('Greedy prefill token mismatch')
  print(json.dumps({'event':'parity',**report['parity']}),flush=True)
  save()
  if not all(value['finite'] for key,value in report['parity'].items() if isinstance(value,dict)):
    raise RuntimeError('Nonfinite VLM output')
  if args.verify_only:
    report['complete'] = True
    save()
    return
  for index in range(args.warmups+args.runs):
    # All prompt positions are overwritten. Decoder only reads [0, start+T).
    start = time.perf_counter()
    submits = npu.submit_count if npu else 0
    image = encode_image().realize()
    Device[args.vision_device or args.device].synchronize()
    encoded = time.perf_counter()
    encoded_submits = npu.submit_count if npu else 0
    embeds = model.embed_prompt(ids,image).realize()
    generated = [int(model.text.greedy_embeddings_prefill_jit(embeds).item())]
    first = time.perf_counter()
    first_submits = npu.submit_count if npu else 0
    if scalar_decode is not None:
      generated.extend(scalar_decode(generated[-1],len(ids),args.tokens-1))
    else:
      for step in range(args.tokens-1):
        pos = len(ids)+step
        position = UOp.variable('position',1,args.context-1).bind(pos)
        generated.append(int(model.text.greedy_rollout_jit(Tensor([[generated[-1]]]).contiguous(),position).item()))
    end = time.perf_counter()
    row = {'warmup':index<args.warmups,'vision_ms':(encoded-start)*1000,'first_token_ms':(first-start)*1000,
           'prefill_tps':len(ids)/(first-encoded),'decode_tps':(args.tokens-1)/(end-first),
           'image_to_answer_ms':(end-start)*1000,'generated_ids':generated}
    row.update(vision_submits=encoded_submits-submits,prefill_submits=first_submits-encoded_submits,
               decode_submits=(npu.submit_count if npu else 0)-first_submits)
    report['trials'].append(row)
    print(json.dumps({'event':'trial','index':index,**row}),flush=True)
    save()
  measured = [row for row in report['trials'] if not row['warmup']]
  if any(row['generated_ids']!=measured[0]['generated_ids'] for row in measured): raise RuntimeError('Nondeterministic greedy IDs')
  if any(hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest for p,digest in source_hashes.items()):
    raise RuntimeError('Sources changed during measurement')
  report['medians'] = {name:statistics.median(row[name] for row in measured) for name in
                       ('vision_ms','first_token_ms','prefill_tps','decode_tps','image_to_answer_ms')}
  report['native_execution'] = {}
  for name,jit in jits.items():
    captured=jit.captured
    native=getattr(captured,'native',None)
    runs=getattr(captured,'native_runs',0)
    report['native_execution'][name]={'runs':runs,'reason':getattr(captured,'reason',None),
      'kernels_ms_per_run':{label:ns/max(runs,1)/1e6 for label,ns in getattr(native,'profile_ns',{}).items()}}
  if staged is not None: report['staged_vision'] = staged.statistics()
  if scalar_decode is not None:
    report['native_loop_tokens'] = scalar_decode.loop.tokens if scalar_decode.loop is not None else 0
  report['complete'] = True
  save()
  print(json.dumps(report['medians']),flush=True)

if __name__ == '__main__': main()
